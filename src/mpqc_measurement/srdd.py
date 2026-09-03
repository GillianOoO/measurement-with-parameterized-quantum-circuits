"""Shallow Rotated-Density Decomposition and its universal residual wrapper."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from ._version import __version__
from .allocation import evaluate_error, require_positive_integer
from .collector import exact_shallow_collector
from .electronic import (
    fock_orbital_unitary,
    hartree_fock_state,
    one_body_dense,
    spatial_occupations,
    spin_free_excitations,
)
from .models import DecompositionResult, Fragment, HamiltonianData
from .pauli import universal_completion_frontier
from .selection import FrontierPoint, sector_indices, select_frontier
from .srdd_tensor import optimize_srdd_tensor


DEPTH_SELECTION_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class SRDDConfig:
    rank_grid: tuple[int, ...] = tuple(range(1, 11))
    depth_grid: tuple[int, ...] = (1, 2, 3)
    rho: float = 1.0e-6
    cycles: int = 4
    angle_steps: int = 20
    gradient_tolerance: float = 1.0e-7
    seed: int = 20260826
    calibrate: bool = True
    calibration_seed: int = 918273
    f3_allocation_iterations: int = 6
    f3_spectral_evaluations: int = 50
    collector_angle_steps: int = 400
    pauli_tolerance: float = 1.0e-12

    def validate(self, shots: int) -> None:
        if int(shots) < 1:
            raise ValueError("shots must be positive")
        if not self.rank_grid or any(int(value) < 1 for value in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")
        if not self.depth_grid or any(int(value) not in (1, 2, 3) for value in self.depth_grid):
            raise ValueError("depth_grid values must be 1, 2, or 3")
        if (
            self.rho < 0.0
            or self.cycles < 1
            or self.angle_steps < 1
            or self.collector_angle_steps < 1
        ):
            raise ValueError(
                "rho must be nonnegative; cycles, angle_steps, and "
                "collector_angle_steps positive"
            )


def _centered_action(matrix: np.ndarray, state: np.ndarray) -> np.ndarray:
    action = matrix @ state
    mean = np.vdot(state, action)
    return action - mean * state


def _centered_full_fock_half_range(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
    return 0.5 * float(np.max(eigenvalues) - np.min(eigenvalues))


def _f3_r2(
    base_one_body: np.ndarray,
    rotations: np.ndarray,
    leaf_fragments: list[Fragment],
    q_fragments: list[Fragment],
    q_spatial: list[np.ndarray],
    proxy_state: np.ndarray,
    config: SRDDConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    collector0 = one_body_dense(base_one_body)
    leaves = [fragment.matrix for fragment in leaf_fragments]
    directions = [fragment.matrix for fragment in q_fragments]
    v0 = _centered_action(collector0, proxy_state)
    leaf_actions = np.column_stack([_centered_action(value, proxy_state) for value in leaves])
    direction_actions = np.column_stack(
        [_centered_action(value, proxy_state) for value in directions]
    )
    variances = np.asarray(
        [float(np.vdot(v0, v0).real)]
        + [
            float(np.vdot(leaf_actions[:, index], leaf_actions[:, index]).real)
            for index in range(len(leaves))
        ]
    )
    allocation = np.sqrt(np.maximum(variances, 1.0e-16))
    allocation /= float(np.sum(allocation))
    gram = np.real(direction_actions.conj().T @ direction_actions)
    rhs0 = np.real(direction_actions.conj().T @ v0)
    alpha = np.zeros(len(leaves), dtype=float)
    for _ in range(config.f3_allocation_iterations):
        normal = gram / allocation[0]
        rhs = -rhs0 / allocation[0]
        for leaf in range(len(leaves)):
            normal[leaf, leaf] += gram[leaf, leaf] / allocation[leaf + 1]
            rhs[leaf] += float(
                np.real(np.vdot(direction_actions[:, leaf], leaf_actions[:, leaf]))
            ) / allocation[leaf + 1]
        normal.flat[:: len(normal) + 1] += 1.0e-11 * max(
            1.0, float(np.trace(normal)) / max(len(normal), 1)
        )
        alpha = np.linalg.lstsq(normal, rhs, rcond=1.0e-12)[0]
        collector_action = v0 + direction_actions @ alpha
        adjusted_actions = leaf_actions - direction_actions * alpha[None, :]
        variances = np.asarray(
            [float(np.vdot(collector_action, collector_action).real)]
            + [
                float(np.vdot(adjusted_actions[:, index], adjusted_actions[:, index]).real)
                for index in range(len(leaves))
            ]
        )
        allocation = np.sqrt(np.maximum(variances, 1.0e-16))
        allocation /= float(np.sum(allocation))
    proxy_alpha = alpha.copy()

    def spectral(values: np.ndarray) -> float:
        collector_spatial = base_one_body.copy()
        for coefficient, direction in zip(values, q_spatial):
            collector_spatial += float(coefficient) * direction
        total = _centered_full_fock_half_range(one_body_dense(collector_spatial))
        for coefficient, leaf, direction in zip(values, leaves, directions):
            total += _centered_full_fock_half_range(
                leaf - float(coefficient) * direction
            )
        return float(total)

    zeros = np.zeros_like(proxy_alpha)
    optimization = minimize(
        lambda values: spectral(np.asarray(values, dtype=float)),
        proxy_alpha,
        method="Powell",
        options={
            "maxiter": 3,
            "maxfev": config.f3_spectral_evaluations,
            "xtol": 2.0e-3,
            "ftol": 2.0e-5,
        },
    )
    candidates = [
        (spectral(zeros), zeros, "zero_safeguard"),
        (spectral(proxy_alpha), proxy_alpha, "hartree_fock_proxy"),
        (spectral(np.asarray(optimization.x)), np.asarray(optimization.x), "centered_range_refinement"),
    ]
    selected_value, alpha, selected = min(candidates, key=lambda item: item[0])
    collector_spatial = base_one_body.copy()
    for coefficient, direction in zip(alpha, q_spatial):
        collector_spatial += float(coefficient) * direction
    return np.asarray(alpha), collector_spatial, {
        "selected": selected,
        "range_domain": "full_fock",
        "selected_centered_range_sum": float(selected_value),
        "zero_centered_range_sum": float(candidates[0][0]),
        "hf_proxy_centered_range_sum": float(candidates[1][0]),
        "refined_centered_range_sum": float(candidates[2][0]),
        "powell_evaluations": int(optimization.nfev),
        "transfer_coefficients": np.asarray(alpha).tolist(),
    }


def _fit_electronic_source(
    data: HamiltonianData,
    rank: int,
    depth: int,
    config: SRDDConfig,
) -> dict[str, object]:
    assert data.electronic is not None
    electronic = data.electronic
    m = electronic.spatial_orbitals
    if m < 2:
        raise ValueError("the SRDD electronic backend requires at least two spatial orbitals")
    if electronic.n_electrons % 2:
        raise ValueError("the SRDD electronic backend requires n_alpha = n_beta")
    rotations, z_tensors, fitted, fit_audit = optimize_srdd_tensor(
        electronic.two_body_chemist,
        int(rank),
        int(depth),
        config.rho,
        config.cycles,
        config.angle_steps,
        config.gradient_tolerance,
        f"{config.seed}|K{rank}|d{depth}",
    )
    occupations = spatial_occupations(m)
    leaf_fragments, q_fragments, q_spatial = [], [], []
    measurement_unitaries = []
    for index, (rotation, z_tensor) in enumerate(zip(rotations, z_tensors)):
        fock = fock_orbital_unitary(rotation)
        measurement = fock.conj().T
        measurement_unitaries.append(measurement)
        diagonal = (
            0.5 * np.einsum("bi,ij,bj->b", occupations, z_tensor, occupations)
            - 0.5 * occupations @ np.diag(z_tensor)
        )
        direction = np.sum(z_tensor, axis=1) - 0.5 * np.diag(z_tensor)
        q_diagonal = occupations @ direction
        leaf_fragments.append(
            Fragment(measurement, diagonal, f"srdd-two-body-{index:03d}")
        )
        q_fragments.append(
            Fragment(measurement, q_diagonal, f"srdd-r2-direction-{index:03d}")
        )
        q_spatial.append(rotation @ np.diag(direction) @ rotation.T)
    approximate = float(electronic.constant) * np.eye(
        data.dimension, dtype=np.complex128
    )
    approximate += one_body_dense(electronic.one_body)
    for fragment in leaf_fragments:
        approximate += fragment.matrix
    approximate = (approximate + approximate.conj().T) / 2.0
    residual = (data.matrix - approximate + (data.matrix - approximate).conj().T) / 2.0
    return {
        "rank": int(rank),
        "depth": int(depth),
        "rotations": rotations,
        "z_tensors": z_tensors,
        "fitted_tensor": fitted,
        "fit_audit": fit_audit,
        "occupations": occupations,
        "leaf_fragments": leaf_fragments,
        "q_fragments": q_fragments,
        "q_spatial": q_spatial,
        "measurement_unitaries": measurement_unitaries,
        "source_approximate": approximate,
        "source_residual": residual,
    }


def _materialize_electronic_point(
    data: HamiltonianData,
    source: dict[str, object],
    config: SRDDConfig,
) -> FrontierPoint:
    assert data.electronic is not None
    electronic = data.electronic
    rank = int(source["rank"])
    depth = int(source["depth"])
    rotations = np.asarray(source["rotations"], dtype=float)
    z_tensors = np.asarray(source["z_tensors"], dtype=float)
    fitted = np.asarray(source["fitted_tensor"], dtype=float)
    fit_audit = source["fit_audit"]
    occupations = np.asarray(source["occupations"], dtype=float)
    leaf_fragments = list(source["leaf_fragments"])
    q_fragments = list(source["q_fragments"])
    q_spatial = list(source["q_spatial"])
    measurement_unitaries = list(source["measurement_unitaries"])
    m = electronic.spatial_orbitals
    proxy = hartree_fock_state(m, electronic.n_electrons)
    if rank == 0:
        alpha = np.zeros(0, dtype=float)
        collector_spatial = np.array(electronic.one_body, dtype=float, copy=True)
        f3_audit = {
            "selected": "rank_zero_no_transfer_coordinate",
            "range_domain": "full_fock",
            "transfer_coefficients": [],
        }
    else:
        alpha, collector_spatial, f3_audit = _f3_r2(
            electronic.one_body,
            rotations,
            leaf_fragments,
            q_fragments,
            q_spatial,
            proxy,
            config,
        )
    cover_rotations, cover_coefficients, collector_audit = exact_shallow_collector(
        rotations,
        collector_spatial,
        depth,
        seed_key=f"{config.seed}|K{rank}|d{depth}",
        angle_steps=config.collector_angle_steps,
    )
    constant = float(electronic.constant)
    fragments: list[Fragment] = []
    for index in range(rank):
        diagonal = (
            leaf_fragments[index].diagonal
            - float(alpha[index]) * q_fragments[index].diagonal
            + occupations @ cover_coefficients[index]
        )
        midpoint = 0.5 * float(np.max(diagonal) + np.min(diagonal))
        constant += midpoint
        centered_diagonal = diagonal - midpoint
        if float(np.max(centered_diagonal) - np.min(centered_diagonal)) == 0.0:
            continue
        fragments.append(
            Fragment(
                measurement_unitaries[index],
                centered_diagonal,
                f"srdd-setting-{index:03d}",
                metadata={
                    "kind": "fitted two-body leaf plus materialized one-body share",
                    "spatial_rotation": rotations[index].tolist(),
                    "z_tensor": z_tensors[index].tolist(),
                    "r2_alpha": float(alpha[index]),
                    "depth": depth,
                },
            )
        )
    for index in range(rank, len(cover_rotations)):
        fock = fock_orbital_unitary(cover_rotations[index])
        diagonal = occupations @ cover_coefficients[index]
        midpoint = 0.5 * float(np.max(diagonal) + np.min(diagonal))
        constant += midpoint
        centered_diagonal = diagonal - midpoint
        if float(np.max(centered_diagonal) - np.min(centered_diagonal)) == 0.0:
            continue
        fragments.append(
            Fragment(
                fock.conj().T,
                centered_diagonal,
                f"srdd-one-body-extra-{index-rank:03d}",
                metadata={
                    "kind": "depth-matched shallow one-body completion",
                    "spatial_rotation": cover_rotations[index].tolist(),
                    "depth": depth,
                },
            )
        )
    approximate = constant * np.eye(data.dimension, dtype=np.complex128)
    for fragment in fragments:
        approximate += fragment.matrix
    approximate = (approximate + approximate.conj().T) / 2.0
    residual = (data.matrix - approximate + (data.matrix - approximate).conj().T) / 2.0
    return FrontierPoint(
        size=int(rank),
        constant=constant,
        fragments=fragments,
        approximate=approximate,
        residual=residual,
        metadata={
            "rank": int(rank),
            "depth": int(depth),
            "fit": fit_audit,
            "f3_r2": f3_audit,
            "collector": collector_audit,
            "eri_fit_residual_frobenius": float(
                np.linalg.norm(electronic.two_body_chemist - fitted)
            ),
        },
    )


def _electronic_point(
    data: HamiltonianData,
    rank: int,
    depth: int,
    config: SRDDConfig,
) -> FrontierPoint:
    """Build one fully postprocessed point; retained for focused diagnostics."""
    source = _fit_electronic_source(data, rank, depth, config)
    return _materialize_electronic_point(data, source, config)


def build_electronic_frontier(
    data: HamiltonianData, shots: int, config: SRDDConfig
) -> list[FrontierPoint]:
    if data.electronic is None:
        raise ValueError("the electronic SRDD frontier requires electronic tensors")
    indices = sector_indices(data.n_qubits, data.electronic.n_electrons)
    points = []
    m = data.electronic.spatial_orbitals
    rank_zero_failures = []
    for depth in sorted(set(int(value) for value in config.depth_grid)):
        zero_source: dict[str, object] = {
            "rank": 0,
            "depth": depth,
            "rotations": np.empty((0, m, m), dtype=float),
            "z_tensors": np.empty((0, m, m), dtype=float),
            "fitted_tensor": np.zeros_like(data.electronic.two_body_chemist),
            "fit_audit": {"rank_zero_candidate": True},
            "occupations": spatial_occupations(m),
            "leaf_fragments": [],
            "q_fragments": [],
            "q_spatial": [],
            "measurement_unitaries": [],
            "source_approximate": (
                float(data.electronic.constant)
                * np.eye(data.dimension, dtype=np.complex128)
                + one_body_dense(data.electronic.one_body)
            ),
            "source_residual": (
                data.matrix
                - float(data.electronic.constant)
                * np.eye(data.dimension, dtype=np.complex128)
                - one_body_dense(data.electronic.one_body)
            ),
        }
        try:
            zero_point = _materialize_electronic_point(data, zero_source, config)
        except RuntimeError as error:
            rank_zero_failures.append({"depth": depth, "reason": str(error)})
            continue
        zero_point.metadata["depth_selection"] = {
            "rule": "first feasible depth for the rank-zero one-body cover",
            "selected_depth": depth,
            "failed_shallower_depths": rank_zero_failures,
        }
        points.append(zero_point)
        break
    if not points:
        raise RuntimeError(
            "the SRDD rank-zero one-body cover is infeasible on the supplied depth grid"
        )
    for rank in sorted(set(int(value) for value in config.rank_grid)):
        candidates = []
        for depth in sorted(set(int(value) for value in config.depth_grid)):
            source = _fit_electronic_source(data, rank, depth, config)
            residual = np.asarray(source["source_residual"])
            sector_block = residual[np.ix_(indices, indices)]
            sector_frobenius = float(np.linalg.norm(sector_block, ord="fro"))
            full_frobenius = float(np.linalg.norm(residual, ord="fro"))
            source["physical_sector_frobenius_residual"] = sector_frobenius
            source["full_fock_frobenius_residual"] = full_frobenius
            candidates.append(source)
        minimum = min(
            float(source["physical_sector_frobenius_residual"])
            for source in candidates
        )
        tied = [
            source
            for source in candidates
            if float(source["physical_sector_frobenius_residual"])
            <= minimum + DEPTH_SELECTION_TOLERANCE
        ]
        selected_source = min(tied, key=lambda source: int(source["depth"]))
        point = _materialize_electronic_point(data, selected_source, config)
        point.metadata["physical_sector_frobenius_residual"] = float(
            selected_source["physical_sector_frobenius_residual"]
        )
        point.metadata["full_fock_frobenius_residual"] = float(
            selected_source["full_fock_frobenius_residual"]
        )
        point.metadata["depth_selection"] = {
            "rule": "minimum physical-sector Frobenius residual",
            "tie_tolerance": DEPTH_SELECTION_TOLERANCE,
            "tie_break": "shallower depth",
            "selected_depth": int(selected_source["depth"]),
            "trials": [
                {
                    "depth": int(source["depth"]),
                    "physical_sector_frobenius_residual": float(
                        source["physical_sector_frobenius_residual"]
                    ),
                    "full_fock_frobenius_residual": float(
                        source["full_fock_frobenius_residual"]
                    ),
                }
                for source in candidates
            ],
        }
        points.append(point)
    return points


def build_universal_frontier(
    data: HamiltonianData, shots: int, config: SRDDConfig
) -> list[FrontierPoint]:
    constant, raw_fragments = universal_completion_frontier(
        data.matrix, coefficient_tolerance=config.pauli_tolerance
    )
    if not raw_fragments:
        return []
    fragments: list[Fragment] = []
    scalar = constant * np.eye(data.dimension, dtype=np.complex128)
    points: list[FrontierPoint] = [
        FrontierPoint(
            size=0,
            constant=constant,
            fragments=[],
            approximate=scalar,
            residual=(data.matrix - scalar + (data.matrix - scalar).conj().T) / 2.0,
            metadata={
                "retained_product_pauli_groups": 0,
                "available_product_pauli_groups": len(raw_fragments),
                "rank_zero_candidate": True,
            },
        )
    ]
    running_constant = constant
    for size, raw in enumerate(raw_fragments, start=1):
        midpoint = raw.midpoint
        running_constant += midpoint
        fragments.append(
            Fragment(
                raw.unitary,
                raw.diagonal - midpoint,
                raw.label,
                metadata={**raw.metadata, "universal_residual_completion": True},
            )
        )
        approximate = running_constant * np.eye(data.dimension, dtype=np.complex128)
        for fragment in fragments:
            approximate += fragment.matrix
        approximate = (approximate + approximate.conj().T) / 2.0
        residual = (data.matrix - approximate + (data.matrix - approximate).conj().T) / 2.0
        points.append(
            FrontierPoint(
                size=size,
                constant=running_constant,
                fragments=list(fragments),
                approximate=approximate,
                residual=residual,
                metadata={
                    "retained_product_pauli_groups": size,
                    "available_product_pauli_groups": len(raw_fragments),
                    "completion_exact": size == len(raw_fragments),
                    "tail_pauli_l1_spectral_certificate": float(
                        sum(abs(value) for fragment in raw_fragments[size:] for _, value in fragment.metadata["pauli_terms"])
                    ),
                },
            )
        )
    return points


def fit_srdd(
    data: HamiltonianData,
    shots: int,
    config: SRDDConfig | None = None,
) -> DecompositionResult:
    settings = config or SRDDConfig()
    total_shots = require_positive_integer(shots, name="shots")
    settings.validate(total_shots)
    backend = "electronic SRDD core" if data.electronic is not None else "universal Pauli residual completion"
    frontier = (
        build_electronic_frontier(data, total_shots, settings)
        if data.electronic is not None
        else build_universal_frontier(data, total_shots, settings)
    )
    if not frontier:
        constant = float(np.trace(data.matrix).real / data.dimension)
        approximate = constant * np.eye(data.dimension, dtype=np.complex128)
        residual = data.matrix - approximate
        error = evaluate_error(data, approximate, [], np.zeros(0, dtype=int))
        return DecompositionResult(
            method="SRDD",
            shots=total_shots,
            constant=constant,
            fragments=[],
            shot_allocation=np.zeros(0, dtype=int),
            approximate_hamiltonian=approximate,
            residual=residual,
            error=error,
            selected_size=0,
            metadata={
                "backend": backend,
                "package_version": __version__,
                "input_hamiltonian_sha256": data.hamiltonian_sha256,
                "constant_hamiltonian": True,
                "unused_shots": total_shots,
            },
        )
    point, allocation, proxy, selection = select_frontier(
        data,
        frontier,
        total_shots,
        calibrate=settings.calibrate,
        calibration_seed=settings.calibration_seed,
    )
    error = evaluate_error(
        data,
        point.approximate,
        point.fragments,
        allocation,
        calibrated_proxy_rmse=proxy if settings.calibrate else None,
    )
    selected_at_maximum = point.size == max(item.size for item in frontier)
    return DecompositionResult(
        method="SRDD",
        shots=total_shots,
        constant=point.constant,
        fragments=point.fragments,
        shot_allocation=allocation,
        approximate_hamiltonian=point.approximate,
        residual=point.residual,
        error=error,
        selected_size=point.size,
        metadata={
            "config": asdict(settings),
            "package_version": __version__,
            "input_hamiltonian_sha256": data.hamiltonian_sha256,
            "unused_shots": total_shots - int(np.sum(allocation)),
            "backend": backend,
            "selected_point": point.metadata,
            "selection": selection,
            "selected_at_maximum_rank_or_group": selected_at_maximum,
            "selection_is_provisional": bool(
                selected_at_maximum
                and data.electronic is not None
                and point.size > 0
            ),
            "dense_materialization_is_exponential": True,
            "target_state_used_for_construction": False,
        },
    )
