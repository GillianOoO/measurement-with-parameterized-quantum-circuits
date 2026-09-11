#!/usr/bin/env python3
"""Add independently calibrated shallow-RCDF to the audited H4 Figure 3(a).

Every bond length uses the archived 8-qubit Hamiltonian and exact ground state
from the H4 all-bond experiment.  For each geometry and every rank K=1,...,10,
all shallow-RCDF leaves are jointly refitted at depths d_R=1,2,3.  The depth
with the smallest four-electron-sector Frobenius residual is retained.  F3-R2's
one-body collector is then exactly redistributed over the native leaves plus a
minimum-size bank of collector-tailored rotations at the same depth; no
independently diagonalized collector remains.  Geometry-specific weights are fitted
from 500 grouped probes at K=1,...,10.  At T=2038, the frozen loss selects K;
only then is the exact ground state used for the 50-repeat sampling benchmark.

The historical Derand/OGM/SG/TND/GPD columns are read verbatim from the audited
Figure 3 data export.  This runner never regenerates or modifies those values.
Rank artifacts are checkpointed, so an interrupted 21-geometry run can resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent.parent
CORE = ARCHIVE / "code" / "methods" / "srdd"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from srdd_release_paths import DATA, RUNS

from adaptive_f3_pool import (  # noqa: E402
    DIMENSION,
    N_QUBITS,
    OCCUPATIONS,
    SPATIAL_OCCUPATIONS,
    SHALLOW_RCDF_DEPTHS,
    Fragment,
    LABELS,
    MoleculeCase,
    hermitian,
    interaction_dense,
    make_shallow_rcdf_bank,
)
from five_molecule_cases import (  # noqa: E402
    _extract_pair_coefficients,
    _restricted_spatial_coefficients,
    _symmetrized_chemist,
    _target_sha256,
)
from stabilizer_calibration import (  # noqa: E402
    CALIBRATION_K_VALUES,
    SHOT_GRID,
    PrefixDecomposition,
    build_minimum_depth_zero_rank_shallow_collector,
    build_shallow_srcdf_prefix_decomposition,
    calibrate_weights,
    generate_stabilizer_probes,
    integer_range_allocation,
    monte_carlo_estimates,
    outcome_models,
)


VERSION = "h4-all-bond-figure3a-srcdf-shallow-collector-v2"
BONDS = tuple(round(value, 1) for value in np.arange(0.4, 4.4 + 0.1, 0.2))
TARGET_SHOTS = 2038
MAX_RANK = 10
DEFAULT_REPEATS = 50
SELECTION_TOLERANCE = 1.0e-12

HAMILTONIAN_ROOT = DATA / "H4" / "inputs" / "bond_scan"
HISTORICAL_CSV = (
    DATA / "H4" / "results" / "srdd_bond_scan_selected"
    / "figure3a_H4_bond_scan_2038_with_srcdf.csv"
)
DEFAULT_OUTPUT = RUNS / "h4_srdd"
DEFAULT_FROZEN_RANK_SOURCE = (
    DATA / "H4" / "results" / "srdd_bond_scan_rank_fits"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=json_default) + "\n", encoding="utf-8"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hamiltonian_path(bond: float) -> Path:
    return HAMILTONIAN_ROOT / f"H4_R{bond:.1f}.npz"


def sampling_seed(bond: float, k: int) -> int:
    payload = f"{VERSION}|R{bond:.1f}|T{TARGET_SHOTS}|K{k}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def load_case(bond: float):
    source = hamiltonian_path(bond)
    with np.load(source, allow_pickle=False) as data:
        target = hermitian(np.asarray(data["H"], dtype=np.complex128))
        exact_state = np.asarray(data["state"], dtype=np.complex128)
        exact_energy = float(data["energy"])
        metadata = json.loads(str(data["metadata_json"][0]))

    recovered = _extract_pair_coefficients(target, N_QUBITS)
    constant, one_spatial, two_openfermion, recovery = (
        _restricted_spatial_coefficients(*recovered)
    )
    reconstructed = interaction_dense(constant, one_spatial, two_openfermion)
    reconstruction_fro = float(np.linalg.norm(target - reconstructed, "fro"))
    reconstruction_max = float(np.max(np.abs(target - reconstructed)))
    if reconstruction_fro > 1.0e-8:
        raise RuntimeError(
            f"R={bond:.1f}: dense-to-integral reconstruction failed: "
            f"{reconstruction_fro:.3e}"
        )

    base_collector = interaction_dense(
        constant, one_spatial, np.zeros_like(two_openfermion)
    )
    sector_indices = np.flatnonzero(np.sum(OCCUPATIONS, axis=1) == 4)
    proxy_state = np.zeros(DIMENSION, dtype=np.complex128)
    proxy_state[int(sector_indices[0])] = 1.0
    exact_state = exact_state / np.linalg.norm(exact_state)
    outside = np.setdiff1d(np.arange(DIMENSION), sector_indices)
    state_leakage = float(np.sum(np.abs(exact_state[outside]) ** 2))
    eigen_residual = float(np.linalg.norm(target @ exact_state - exact_energy * exact_state))
    sector_ground = float(
        np.linalg.eigvalsh(target[np.ix_(sector_indices, sector_indices)])[0]
    )
    if state_leakage > 1.0e-10 or eigen_residual > 1.0e-8:
        raise RuntimeError(f"R={bond:.1f}: archived ground-state audit failed")

    case = MoleculeCase(
        name=f"H4_R{bond:.1f}A",
        label=f"linear H4, R={bond:.1f} A",
        geometry=f"linear H4, uniform bond R={bond:.1f} Angstrom, STO-3G",
        source=source,
        frozen_orbitals=(),
        active_orbitals=(0, 1, 2, 3),
        active_electrons=4,
        nuclear_repulsion=float(metadata["nuclear_repulsion"]),
        active_constant=constant,
        one_spatial=one_spatial,
        two_openfermion=two_openfermion,
        eri_chemist=_symmetrized_chemist(two_openfermion),
        target=target,
        base_collector=base_collector,
        proxy_state=proxy_state,
        sector_indices=sector_indices,
        metadata={
            **metadata,
            "bond_length_angstrom": bond,
            "target_sha256": _target_sha256(target),
        },
    )
    audit = {
        "source": str(source),
        "source_sha256": sha256(source),
        "target_sha256": _target_sha256(target),
        "dense_reconstruction_frobenius": reconstruction_fro,
        "dense_reconstruction_maximum_absolute": reconstruction_max,
        "state_sector_leakage": state_leakage,
        "state_eigenpair_residual": eigen_residual,
        "archived_energy": exact_energy,
        "sector_diagonalization_energy": sector_ground,
        "energy_difference": abs(sector_ground - exact_energy),
        "integral_recovery": recovery,
    }
    return case, exact_state, exact_energy, audit


def approximation_matrix(case: MoleculeCase, fragments) -> np.ndarray:
    return hermitian(
        case.base_collector
        + sum((fragment.matrix for fragment in fragments), start=np.zeros_like(case.target))
    )


def residual_metrics(case: MoleculeCase, approximation: np.ndarray) -> dict[str, float]:
    residual = hermitian(case.target - approximation)
    sector = residual[np.ix_(case.sector_indices, case.sector_indices)]
    return {
        "full_Frobenius_distance": float(np.linalg.norm(residual, "fro")),
        "sector_Frobenius_distance": float(np.linalg.norm(sector, "fro")),
        "sector_spectral_distance": float(np.max(np.abs(np.linalg.eigvalsh(sector)))),
    }


def collector_kwargs(args) -> dict[str, Any]:
    return {
        "collector_mode": args.collector_mode,
        "collector_min_extra_leaves": args.collector_min_extra_leaves,
        "collector_max_extra_leaves": args.collector_max_extra_leaves,
        "collector_objective": args.collector_objective,
        "collector_proxy_state": "hf",
        "collector_reference_shots": args.collector_reference_shots,
        "collector_proxy_cycles": args.collector_proxy_cycles,
        "collector_equality_tolerance": args.collector_equality_tolerance,
        "collector_extra_angle_steps": args.collector_extra_angle_steps,
    }


def checkpoint_fragments(data, k: int) -> list[Fragment]:
    fragments = []
    for index in range(k):
        rotation = np.asarray(data["spatial_rotations"][index], dtype=float)
        z_tensor = np.asarray(data["z_tensors"][index], dtype=float)
        if "source_diagonals" in data.files:
            diagonal = np.asarray(data["source_diagonals"][index])
            one_matrix = np.asarray(data["source_one_matrices"][index])
            one_diagonal = np.asarray(data["source_one_diagonals"][index])
            q_direction = np.asarray(data["source_q_directions"][index])
            q_diagonal = np.asarray(data["source_q_diagonals"][index])
        else:
            diagonal = (
                0.5
                * np.einsum(
                    "bi,ij,bj->b",
                    SPATIAL_OCCUPATIONS,
                    z_tensor,
                    SPATIAL_OCCUPATIONS,
                )
                - 0.5 * SPATIAL_OCCUPATIONS @ np.diag(z_tensor)
            )
            one_matrix = np.zeros((DIMENSION, DIMENSION), dtype=np.complex128)
            one_diagonal = np.zeros(DIMENSION)
            direction = np.sum(z_tensor, axis=1) - 0.5 * np.diag(z_tensor)
            q_direction = interaction_dense(
                0.0,
                rotation @ np.diag(direction) @ rotation.T,
                np.zeros((SPATIAL_OCCUPATIONS.shape[1],) * 4),
            )
            q_diagonal = SPATIAL_OCCUPATIONS @ direction
        fragments.append(
            Fragment(
                family="shallow_rcdf",
                family_label=LABELS["shallow_rcdf"],
                matrix=np.asarray(data["source_matrices"][index]),
                diagonal=diagonal,
                one_matrix=one_matrix,
                one_diagonal=one_diagonal,
                remainder_diagonal=np.asarray(diagonal - one_diagonal),
                q_direction=q_direction,
                q_diagonal=q_diagonal,
                f3_interface="standard_shallow_tensor_native_f3_r2",
                source_term=index,
                candidate_metadata={
                    "spatial_rotation": rotation.tolist(),
                    "z_tensor": z_tensor.tolist(),
                    "shallow_rcdf_depth": int(data["selected_depth"]),
                },
                rcdf_bank_index=index,
            )
        )
    return fragments


def save_selected_rank(
    path: Path,
    case: MoleculeCase,
    fragments,
    prefix: PrefixDecomposition,
    depth: int,
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(VERSION),
        target_sha256=np.asarray(_target_sha256(case.target)),
        rho=np.asarray(args.rho),
        cycles=np.asarray(args.cycles),
        angle_steps=np.asarray(args.angle_steps),
        spectral_max_evaluations=np.asarray(args.spectral_max_evaluations),
        collector_mode=np.asarray(args.collector_mode),
        collector_min_extra_leaves=np.asarray(args.collector_min_extra_leaves),
        collector_max_extra_leaves=np.asarray(args.collector_max_extra_leaves),
        collector_objective=np.asarray(args.collector_objective),
        collector_reference_shots=np.asarray(args.collector_reference_shots),
        collector_proxy_cycles=np.asarray(args.collector_proxy_cycles),
        collector_equality_tolerance=np.asarray(args.collector_equality_tolerance),
        collector_extra_angle_steps=np.asarray(args.collector_extra_angle_steps),
        selected_depth=np.asarray(depth),
        source_matrices=np.asarray([fragment.matrix for fragment in fragments]),
        source_diagonals=np.asarray([fragment.diagonal for fragment in fragments]),
        source_one_matrices=np.asarray(
            [fragment.one_matrix for fragment in fragments]
        ),
        source_one_diagonals=np.asarray(
            [fragment.one_diagonal for fragment in fragments]
        ),
        source_q_directions=np.asarray(
            [fragment.q_direction for fragment in fragments]
        ),
        source_q_diagonals=np.asarray(
            [fragment.q_diagonal for fragment in fragments]
        ),
        spatial_rotations=np.asarray(
            [fragment.candidate_metadata["spatial_rotation"] for fragment in fragments]
        ),
        z_tensors=np.asarray(
            [fragment.candidate_metadata["z_tensor"] for fragment in fragments]
        ),
        f3_alpha=prefix.alpha_by_source,
        setting_matrices=np.asarray([setting.matrix for setting in prefix.settings]),
        setting_centered_ranges=np.asarray(
            [setting.centered_range for setting in prefix.settings]
        ),
        setting_rotations=np.asarray(
            [setting.rotation for setting in prefix.settings], dtype=float
        ),
        setting_native_diagonals=np.asarray(
            [setting.native_diagonal for setting in prefix.settings], dtype=float
        ),
        setting_kinds=np.asarray(
            [setting.setting_kind for setting in prefix.settings]
        ),
        setting_collector_linear_coefficients=np.asarray(
            [setting.collector_linear_coefficients for setting in prefix.settings],
            dtype=float,
        ),
        collector_audit_json=np.asarray(json.dumps(prefix.collector_audit)),
    )


def load_selected_rank(path: Path, case: MoleculeCase, k: int, args):
    with np.load(path, allow_pickle=False) as data:
        if str(data["version"]) != VERSION:
            raise ValueError("checkpoint version changed")
        if str(data["target_sha256"]) != _target_sha256(case.target):
            raise ValueError("checkpoint target changed")
        if not math.isclose(float(data["rho"]), args.rho, rel_tol=0.0, abs_tol=1e-18):
            raise ValueError("checkpoint rho changed")
        if int(data["cycles"]) != args.cycles or int(data["angle_steps"]) != args.angle_steps:
            raise ValueError("checkpoint optimizer budget changed")
        if int(data["spectral_max_evaluations"]) != args.spectral_max_evaluations:
            raise ValueError("checkpoint F3 budget changed")
        controls = {
            "collector_mode": str(data["collector_mode"]),
            "collector_min_extra_leaves": int(data["collector_min_extra_leaves"]),
            "collector_max_extra_leaves": int(data["collector_max_extra_leaves"]),
            "collector_objective": str(data["collector_objective"]),
            "collector_reference_shots": int(data["collector_reference_shots"]),
            "collector_proxy_cycles": int(data["collector_proxy_cycles"]),
            "collector_equality_tolerance": float(
                data["collector_equality_tolerance"]
            ),
            "collector_extra_angle_steps": int(data["collector_extra_angle_steps"]),
        }
        expected_controls = {
            key: value for key, value in collector_kwargs(args).items()
            if key != "collector_proxy_state"
        }
        if controls != expected_controls:
            raise ValueError("checkpoint collector controls changed")
        depth = int(data["selected_depth"])
        sources = np.asarray(data["source_matrices"], dtype=np.complex128)
        saved_matrices = np.asarray(data["setting_matrices"], dtype=np.complex128)
        fragments = checkpoint_fragments(data, k)
    if len(sources) != k:
        raise RuntimeError(f"K={k}: malformed checkpoint")
    prefix = build_shallow_srcdf_prefix_decomposition(
        case,
        fragments,
        depth,
        args.spectral_max_evaluations,
        **collector_kwargs(args),
    )
    matrices = np.asarray([setting.matrix for setting in prefix.settings])
    if matrices.shape != saved_matrices.shape:
        raise RuntimeError(f"K={k}: checkpoint setting count changed")
    replay = float(np.linalg.norm(matrices - saved_matrices))
    if replay > 1.0e-8:
        raise RuntimeError(f"K={k}: checkpoint shallow setting replay {replay:.3e}")
    reconstruction = float(prefix.reconstruction_error)
    if reconstruction > 1.0e-6:
        raise RuntimeError(f"K={k}: checkpoint F3 reconstruction {reconstruction:.3e}")
    approximation = hermitian(np.sum(matrices, axis=0))
    return {
        "K": k,
        "depth": depth,
        "source_matrices": sources,
        "prefix": prefix,
        "approximation": approximation,
        "F3_reconstruction_error": reconstruction,
        **residual_metrics(case, approximation),
    }


def migrate_frozen_rank(path: Path, case: MoleculeCase, k: int, args):
    """Reuse only the old native rank-K leaves, never its exact collector setting."""
    with np.load(path, allow_pickle=False) as data:
        if str(data["target_sha256"]) != _target_sha256(case.target):
            raise ValueError("frozen checkpoint target changed")
        if not math.isclose(float(data["rho"]), args.rho, rel_tol=0.0, abs_tol=1e-18):
            raise ValueError("frozen checkpoint rho changed")
        if int(data["cycles"]) != args.cycles or int(data["angle_steps"]) != args.angle_steps:
            raise ValueError("frozen checkpoint optimizer budget changed")
        depth = int(data["selected_depth"])
        sources = np.asarray(data["source_matrices"], dtype=np.complex128)
        fragments = checkpoint_fragments(data, k)
    if len(sources) != k:
        raise RuntimeError(f"K={k}: malformed frozen checkpoint")
    prefix = build_shallow_srcdf_prefix_decomposition(
        case,
        fragments,
        depth,
        args.spectral_max_evaluations,
        **collector_kwargs(args),
    )
    approximation = hermitian(
        sum((setting.matrix for setting in prefix.settings), start=np.zeros_like(case.target))
    )
    return {
        "K": k,
        "depth": depth,
        "fragments": fragments,
        "source_matrices": sources,
        "prefix": prefix,
        "approximation": approximation,
        "F3_reconstruction_error": float(prefix.reconstruction_error),
        "frozen_rank_source": str(path),
        "frozen_rank_source_sha256": sha256(path),
        **residual_metrics(case, approximation),
    }


def fit_rank(case: MoleculeCase, k: int, args, case_output: Path):
    checkpoint = case_output / f"selected_rank_K{k:02d}.npz"
    if checkpoint.exists():
        try:
            loaded = load_selected_rank(checkpoint, case, k, args)
            print(
                f"[resume] {case.name} K={k:02d} dR={loaded['depth']} "
                f"sectorF={loaded['sector_Frobenius_distance']:.6g}",
                flush=True,
            )
            return loaded, []
        except (ValueError, RuntimeError) as exc:
            print(f"[checkpoint rejected] {checkpoint.name}: {exc}", flush=True)

    frozen_checkpoint = (
        args.frozen_rank_source
        / f"R{float(case.metadata['bond_length_angstrom']):.1f}"
        / f"selected_rank_K{k:02d}.npz"
    )
    if frozen_checkpoint.exists():
        try:
            migrated = migrate_frozen_rank(
                frozen_checkpoint, case, k, args
            )
            save_selected_rank(
                checkpoint,
                case,
                migrated["fragments"],
                migrated["prefix"],
                migrated["depth"],
                args,
            )
            print(
                f"[migrated native leaves] {case.name} K={k:02d} "
                f"dR={migrated['depth']}",
                flush=True,
            )
            migrated.pop("fragments")
            return migrated, []
        except (ValueError, RuntimeError) as exc:
            print(
                f"[frozen checkpoint rejected] {frozen_checkpoint.name}: {exc}",
                flush=True,
            )

    candidates = []
    trials = []
    for depth in SHALLOW_RCDF_DEPTHS:
        started = time.perf_counter()
        bank, bank_audit = make_shallow_rcdf_bank(
            case, k, depth, args.rho, args.cycles, args.angle_steps
        )
        approximation = approximation_matrix(case, bank)
        row = {
            "bond_length_angstrom": float(case.metadata["bond_length_angstrom"]),
            "K_source_terms": k,
            "shallow_rotation_depth": depth,
            **residual_metrics(case, approximation),
            "eri_residual_frobenius": float(bank_audit["eri_residual_frobenius"]),
            "full_fock_operator_residual_frobenius": float(
                bank_audit["full_fock_operator_residual_frobenius"]
            ),
            "native_diagonal_spectral_audit": float(
                bank_audit["maximum_native_diagonal_spectral_audit"]
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "selected_depth": False,
            "fit_semantics": "independent_complete_joint_rank_K_refit",
        }
        trials.append(row)
        candidates.append((row, bank))
        print(
            f"[fit] {case.name} K={k:02d} dR={depth} "
            f"sectorF={row['sector_Frobenius_distance']:.6g}",
            flush=True,
        )
    minimum = min(float(item[0]["sector_Frobenius_distance"]) for item in candidates)
    tied = [
        item
        for item in candidates
        if float(item[0]["sector_Frobenius_distance"])
        <= minimum + SELECTION_TOLERANCE
    ]
    selected_row, bank = min(tied, key=lambda item: int(item[0]["shallow_rotation_depth"]))
    selected_row["selected_depth"] = True
    prefix = build_shallow_srcdf_prefix_decomposition(
        case,
        bank,
        int(selected_row["shallow_rotation_depth"]),
        args.spectral_max_evaluations,
        **collector_kwargs(args),
    )
    approximation = hermitian(sum((setting.matrix for setting in prefix.settings)))
    metrics = residual_metrics(case, approximation)
    selected = {
        "K": k,
        "depth": int(selected_row["shallow_rotation_depth"]),
        "source_matrices": np.asarray([fragment.matrix for fragment in bank]),
        "prefix": prefix,
        "approximation": approximation,
        "F3_reconstruction_error": float(prefix.reconstruction_error),
        **metrics,
    }
    save_selected_rank(checkpoint, case, bank, prefix, selected["depth"], args)
    return selected, trials


def run_bond(bond: float, args, output: Path):
    case, ground, exact_energy, source_audit = load_case(bond)
    case_output = output / f"R{bond:.1f}"
    case_output.mkdir(parents=True, exist_ok=True)
    base_prefix, base_depth = build_minimum_depth_zero_rank_shallow_collector(
        case,
        args.collector_zero_rank_depths,
        args.spectral_max_evaluations,
        **collector_kwargs(args),
    )
    base_approximation = hermitian(
        sum((setting.matrix for setting in base_prefix.settings))
    )
    rank_data = [
        {
            "K": 0,
            "depth": base_depth,
            "source_matrices": np.empty((0, *case.target.shape), dtype=np.complex128),
            "prefix": base_prefix,
            "approximation": base_approximation,
            "F3_reconstruction_error": float(base_prefix.reconstruction_error),
            **residual_metrics(case, base_approximation),
        }
    ]
    depth_rows: list[dict[str, Any]] = []
    for k in range(1, MAX_RANK + 1):
        selected, trials = fit_rank(case, k, args, case_output)
        rank_data.append(selected)
        depth_rows.extend(trials)
        if trials:
            previous = []
            trial_path = case_output / "rank_depth_trials.csv"
            if trial_path.exists() and trial_path.stat().st_size:
                previous = [
                    row
                    for row in read_csv(trial_path)
                    if int(row["K_source_terms"]) != k
                ]
            write_csv(trial_path, [*previous, *trials])

    states, probe_records = generate_stabilizer_probes(case.sector_indices)
    prefixes = [row["prefix"] for row in rank_data]
    approximations = [row["approximation"] for row in rank_data]
    weights, approximation_rows, sampling_rows, allocation_rows = calibrate_weights(
        case,
        [],
        prefixes,
        states,
        probe_records,
        SHOT_GRID,
        CALIBRATION_K_VALUES,
        independent_approximations=approximations,
    )
    weights.update(
        {
            "calibration_source_family": "standalone_shallow_rcdf",
            "calibration_source_semantics": (
                "independent complete joint rank-K shallow-RCDF fits; K=1,...,10; "
                "collector-tailored greedy exact-fill shallow augmentation"
            ),
            "bond_length_angstrom": bond,
            "target_sha256": case.metadata["target_sha256"],
            "exact_ground_state_used_for_calibration": False,
        }
    )
    np.savez_compressed(case_output / "stabilizer_probes.npz", states=states)
    write_csv(case_output / "stabilizer_probe_manifest.csv", probe_records)
    write_csv(case_output / "calibration_approximation_components.csv", approximation_rows)
    write_csv(case_output / "calibration_sampling_components.csv", sampling_rows)
    write_csv(case_output / "calibration_shot_allocations.csv", allocation_rows)
    write_json(case_output / "stabilizer_weights.json", weights)

    weight_a = float(weights["weight_approximation"])
    weight_s = float(weights["weight_sampling"])
    candidates = []
    for row in rank_data:
        prefix = row["prefix"]
        lambdas = np.asarray([item.centered_range for item in prefix.settings], dtype=float)
        shots = integer_range_allocation(TARGET_SHOTS, lambdas)
        active = shots > 0
        x_sampling = math.sqrt(
            max(float(np.sum(lambdas[active] ** 2 / shots[active])), 0.0)
        )
        eps_a = weight_a * float(row["sector_Frobenius_distance"])
        eps_s = weight_s * x_sampling
        candidates.append(
            {
                "bond_length_angstrom": bond,
                "T_total_shots": TARGET_SHOTS,
                "K_source_terms": int(row["K"]),
                "shallow_rotation_depth": int(row["depth"]),
                "sector_Frobenius_distance": float(row["sector_Frobenius_distance"]),
                "sector_spectral_distance": float(row["sector_spectral_distance"]),
                "full_Frobenius_distance": float(row["full_Frobenius_distance"]),
                "x_sampling": x_sampling,
                "epsilon_approximation": eps_a,
                "epsilon_sampling": eps_s,
                "stabilizer_loss": math.hypot(eps_a, eps_s),
                "total_settings": len(prefix.settings),
                "collector_mode": prefix.collector_audit["mode"],
                "collector_extra_settings": prefix.collector_audit[
                    "extra_rotation_count"
                ],
                "collector_dictionary_rank": prefix.collector_audit[
                    "dictionary_rank"
                ],
                "collector_reconstruction_residual": prefix.collector_audit[
                    "matrix_reconstruction_residual_frobenius"
                ],
                "independent_collector_setting_present": prefix.collector_audit[
                    "independent_collector_setting_present"
                ],
                "shot_vector": " ".join(str(int(value)) for value in shots),
                "allocated_total_shots": int(np.sum(shots)),
                "right_censored_at_rank_10": int(row["K"]) == MAX_RANK,
                "_rank": row,
                "_shots": shots,
            }
        )
    selected = min(candidates, key=lambda row: (row["stabilizer_loss"], row["K_source_terms"]))
    if selected["K_source_terms"] == MAX_RANK:
        raise RuntimeError(
            f"H4 R={bond:.2f} T={TARGET_SHOTS}: the selected s-RCDF rank lies "
            "at the fitted endpoint; extend MAX_RANK and rerun before publishing."
        )
    selected_rank = selected.pop("_rank")
    selected_shots = selected.pop("_shots")
    public_candidates = []
    for row in candidates:
        record = {key: value for key, value in row.items() if not key.startswith("_")}
        record["selected_K"] = int(record["K_source_terms"] == selected["K_source_terms"])
        public_candidates.append(record)
    write_csv(case_output / "candidate_by_T_K.csv", public_candidates)

    models = outcome_models(ground, selected_rank["prefix"])
    approximate_mean = float(sum(model[2] for model in models))
    sampling_variance = float(
        sum(
            model[3] / int(count)
            for model, count in zip(models, selected_shots)
            if count > 0
        )
    )
    seed = sampling_seed(bond, int(selected["K_source_terms"]))
    estimates = monte_carlo_estimates(models, selected_shots, args.repeats, seed)
    errors = estimates - exact_energy
    summary = {
        **{key: value for key, value in selected.items() if not key.startswith("_")},
        "weight_approximation": weight_a,
        "weight_sampling": weight_s,
        "exact_ground_energy": exact_energy,
        "approximate_ground_state_mean": approximate_mean,
        "signed_approximation_bias": approximate_mean - exact_energy,
        "analytic_sampling_SE": math.sqrt(max(sampling_variance, 0.0)),
        "analytic_total_RMSE": math.hypot(
            approximate_mean - exact_energy, math.sqrt(max(sampling_variance, 0.0))
        ),
        "empirical_total_RMSE": float(np.sqrt(np.mean(errors**2))),
        "empirical_total_MAE": float(np.mean(np.abs(errors))),
        "sampling_repeats": args.repeats,
        "sampling_seed": seed,
        "target_sha256": case.metadata["target_sha256"],
    }
    replicate_rows = [
        {
            "bond_length_angstrom": bond,
            "T_total_shots": TARGET_SHOTS,
            "K_source_terms": int(selected["K_source_terms"]),
            "shallow_rotation_depth": int(selected["shallow_rotation_depth"]),
            "repeat": repeat,
            "energy_estimate": float(value),
            "signed_total_error": float(value - exact_energy),
        }
        for repeat, value in enumerate(estimates)
    ]
    write_csv(case_output / "sampling_replicates.csv", replicate_rows)
    write_csv(case_output / "rank_selected.csv", [
        {
            "bond_length_angstrom": bond,
            "K_source_terms": int(row["K"]),
            "shallow_rotation_depth": int(row["depth"]),
            "sector_Frobenius_distance": float(row["sector_Frobenius_distance"]),
            "sector_spectral_distance": float(row["sector_spectral_distance"]),
            "full_Frobenius_distance": float(row["full_Frobenius_distance"]),
            "F3_reconstruction_error": float(row["F3_reconstruction_error"]),
            "centered_spectral_norm_sum": float(row["prefix"].spectral_sum),
            "measurement_settings": len(row["prefix"].settings),
            "collector_mode": row["prefix"].collector_audit["mode"],
            "collector_extra_settings": row["prefix"].collector_audit[
                "extra_rotation_count"
            ],
            "collector_dictionary_rank": row["prefix"].collector_audit[
                "dictionary_rank"
            ],
            "collector_reconstruction_residual": row[
                "prefix"
            ].collector_audit["matrix_reconstruction_residual_frobenius"],
            "independent_collector_setting_present": row[
                "prefix"
            ].collector_audit["independent_collector_setting_present"],
        }
        for row in rank_data
    ])
    maximum_f3 = max(float(row["F3_reconstruction_error"]) for row in rank_data)
    maximum_collector_relative = max(
        float(
            row["prefix"].collector_audit.get(
                "matrix_relative_reconstruction_residual", 0.0
            )
        )
        for row in rank_data
    )
    all_depth_limited = all(
        setting.rotation_depth is not None
        for row in rank_data
        for setting in row["prefix"].settings
    )
    independent_collector_present = any(
        bool(
            row["prefix"].collector_audit.get(
                "independent_collector_setting_present", False
            )
        )
        for row in rank_data
    )
    production_contract = (
        args.collector_mode != "augment"
        or (
            all_depth_limited
            and not independent_collector_present
            and maximum_collector_relative
            <= 5.0 * args.collector_equality_tolerance
            and all(
                int(row["prefix"].collector_audit["extra_rotation_count"])
                >= args.collector_min_extra_leaves
                for row in rank_data
            )
        )
    )
    audit = {
        "status": (
            "PASS"
            if maximum_f3 <= 1.0e-6 and production_contract
            else "FAIL"
        ),
        "bond_length_angstrom": bond,
        "source": source_audit,
        "all_K_independently_jointly_refitted": True,
        "native_rank_leaves_migrated_without_refitting": any(
            "frozen_rank_source" in row for row in rank_data
        ),
        "historical_exact_collector_settings_reused": False,
        "frozen_native_rank_sources": [
            row["frozen_rank_source"]
            for row in rank_data
            if "frozen_rank_source" in row
        ],
        "fit_K": list(CALIBRATION_K_VALUES),
        "fit_T": list(SHOT_GRID),
        "stabilizer_probe_count": int(states.shape[1]),
        "stabilizer_weight_family": "shallow_rcdf",
        "selected_K_is_loss_argmin": True,
        "selected_K": int(summary["K_source_terms"]),
        "selected_depth": int(summary["shallow_rotation_depth"]),
        "allocated_shots": int(np.sum(selected_shots)),
        "maximum_F3_reconstruction_error": maximum_f3,
        "collector_mode": args.collector_mode,
        "collector_objective": args.collector_objective,
        "collector_proxy_state": "hf",
        "collector_extra_angle_steps": args.collector_extra_angle_steps,
        "minimum_extra_settings_over_rank_grid": min(
            int(row["prefix"].collector_audit["extra_rotation_count"])
            for row in rank_data
        ),
        "maximum_extra_settings_over_rank_grid": max(
            int(row["prefix"].collector_audit["extra_rotation_count"])
            for row in rank_data
        ),
        "maximum_collector_relative_reconstruction_residual": (
            maximum_collector_relative
        ),
        "all_measurement_settings_depth_limited": all_depth_limited,
        "independent_collector_setting_present": independent_collector_present,
        "production_collector_contract_passed": production_contract,
        "exact_ground_state_used_for_selection": False,
        "exact_ground_state_used_only_after_weight_and_K_freeze": True,
    }
    write_json(case_output / "audit.json", audit)
    print(
        f"[selected] R={bond:.1f} K={summary['K_source_terms']} "
        f"dR={summary['shallow_rotation_depth']} "
        f"loss={summary['stabilizer_loss']:.6g} "
        f"RMSE={summary['empirical_total_RMSE']:.6g}",
        flush=True,
    )
    return summary, audit


def make_figure(rows: list[dict[str, Any]], output: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bonds = np.asarray([float(row["bond_length_angstrom"]) for row in rows])
    styles = {
        "Derand": dict(color="#4c78a8", marker="x", linewidth=1.5, markersize=5),
        "OGM": dict(color="#f58518", marker="^", linewidth=1.5, markersize=5),
        "SG": dict(color="#7f7f7f", marker="v", linewidth=1.5, markersize=5),
        "TND": dict(color="#54a24b", marker="s", linewidth=1.7, markersize=5),
        "GPD": dict(color="#8f63b8", marker="P", linewidth=1.7, markersize=5),
    }
    fig = plt.figure(figsize=(10.6, 7.6))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.15, 4.0),
        left=0.09,
        right=0.98,
        bottom=0.09,
        top=0.79,
        hspace=0.10,
    )
    energy_ax = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=energy_ax)
    for method, style in styles.items():
        ax.plot(bonds, [float(row[method]) for row in rows], label=method, **style)
    ax.plot(
        bonds,
        [float(row["sRCDF_empirical_RMSE"]) for row in rows],
        color="#c62828",
        marker="o",
        linewidth=2.5,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.7,
        label=r"s-RCDF-F (loss-selected $K,d_R$)",
        zorder=10,
    )
    ax.set_yscale("log")
    ax.set_xlim(0.3, 4.5)
    ax.set_xticks(np.arange(0.4, 4.5, 0.4))
    ax.set_xlabel(r"H--H bond length $R$ ($\AA$)")
    ax.set_ylabel("Energy-estimation RMSE (Ha)")
    ax.grid(True, which="both", alpha=0.24, linewidth=0.7)
    handles, labels = ax.get_legend_handles_labels()

    energy_ax.plot(
        bonds,
        [float(row["ground_truth_energy_hartree"]) for row in rows],
        color="black",
        linewidth=1.7,
    )
    energy_ax.set_ylabel(r"Exact $E_0$ (Ha)")
    energy_ax.grid(alpha=0.22)
    energy_ax.tick_params(axis="x", labelbottom=False)
    fig.suptitle(
        r"Linear H$_4$ bond scan: all methods at $T=2038$ shots",
        fontsize=15,
        y=0.975,
    )
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.925),
        framealpha=0.96,
        fontsize=9.5,
    )

    pdf = output / "H4_Figure3a_with_sRCDF_2038shots.pdf"
    png = output / "H4_Figure3a_with_sRCDF_2038shots.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def audit_selected_leaf_spectra(output: Path) -> list[dict[str, Any]]:
    """Recompute and persist every selected leaf/native spectral comparison."""
    rows: list[dict[str, Any]] = []
    for bond in BONDS:
        for k in range(1, MAX_RANK + 1):
            path = output / f"R{bond:.1f}" / f"selected_rank_K{k:02d}.npz"
            with np.load(path, allow_pickle=False) as data:
                matrices = np.asarray(data["source_matrices"], dtype=np.complex128)
                tensors = np.asarray(data["z_tensors"], dtype=float)
                depth = int(data["selected_depth"])
            for leaf, (matrix, z_tensor) in enumerate(zip(matrices, tensors), start=1):
                native_diagonal = (
                    0.5
                    * np.einsum(
                        "bi,ij,bj->b",
                        SPATIAL_OCCUPATIONS,
                        z_tensor,
                        SPATIAL_OCCUPATIONS,
                    )
                    - 0.5 * SPATIAL_OCCUPATIONS @ np.diag(z_tensor)
                )
                matrix_spectrum = np.linalg.eigvalsh(matrix)
                absolute = float(
                    np.max(
                        np.abs(
                            np.sort(matrix_spectrum)
                            - np.sort(np.real(native_diagonal))
                        )
                    )
                )
                scale = max(
                    1.0,
                    float(np.max(np.abs(matrix_spectrum))),
                    float(np.max(np.abs(native_diagonal))),
                )
                tolerance = max(1.0e-7, 2.0e-7 * scale)
                rows.append(
                    {
                        "bond_length_angstrom": bond,
                        "K_source_terms": k,
                        "shallow_rotation_depth": depth,
                        "leaf_index": leaf,
                        "absolute_spectral_audit": absolute,
                        "spectral_scale": scale,
                        "relative_spectral_audit": absolute / scale,
                        "audit_tolerance": tolerance,
                        "within_tolerance": absolute <= tolerance,
                    }
                )
    if not all(bool(row["within_tolerance"]) for row in rows):
        raise RuntimeError("One or more selected shallow-RCDF leaves fail spectral audit")
    write_csv(output / "selected_leaf_spectral_audits.csv", rows)
    return rows


def aggregate(output: Path, args) -> None:
    historical = read_csv(HISTORICAL_CSV)
    if len(historical) != len(BONDS):
        raise RuntimeError("Historical Figure 3(a) CSV is not the expected 21-row table")
    historical_by_bond = {round(float(row["bond_length_angstrom"]), 1): row for row in historical}
    merged = []
    summaries = []
    audits = []
    maximum_energy_mismatch = 0.0
    for bond in BONDS:
        case_output = output / f"R{bond:.1f}"
        summary_path = case_output / "sampling_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing completed bond output: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audit = json.loads((case_output / "audit.json").read_text(encoding="utf-8"))
        old = historical_by_bond[bond]
        mismatch = abs(float(old["ground_truth_energy_hartree"]) - float(summary["exact_ground_energy"]))
        maximum_energy_mismatch = max(maximum_energy_mismatch, mismatch)
        merged.append(
            {
                **old,
                "sRCDF": summary["empirical_total_RMSE"],
                "sRCDF_empirical_RMSE": summary["empirical_total_RMSE"],
                "sRCDF_analytic_total_RMSE": summary["analytic_total_RMSE"],
                "sRCDF_analytic_sampling_SE": summary["analytic_sampling_SE"],
                "sRCDF_signed_approximation_bias": summary["signed_approximation_bias"],
                "sRCDF_selected_K": summary["K_source_terms"],
                "sRCDF_selected_depth": summary["shallow_rotation_depth"],
                "sRCDF_stabilizer_loss": summary["stabilizer_loss"],
                "sRCDF_weight_approximation": summary["weight_approximation"],
                "sRCDF_weight_sampling": summary["weight_sampling"],
                "sRCDF_sampling_repeats": summary["sampling_repeats"],
                "sRCDF_sampling_seed": summary["sampling_seed"],
                "sRCDF_target_sha256": summary["target_sha256"],
            }
        )
        summaries.append(summary)
        audits.append(audit)
    if maximum_energy_mismatch > 2.0e-8:
        raise RuntimeError(
            f"Historical/new target energy mismatch: {maximum_energy_mismatch:.3e}"
        )
    write_csv(output / "figure3a_H4_bond_scan_2038_with_srcdf.csv", merged)
    write_csv(output / "srcdf_sampling_summary.csv", summaries)
    leaf_spectral_rows = audit_selected_leaf_spectra(output)
    pdf, png = make_figure(merged, output)
    manifest = {
        "version": VERSION,
        "status": "PASS" if all(item["status"] == "PASS" for item in audits) else "FAIL",
        "protocol": {
            "bonds_angstrom": list(BONDS),
            "shots": TARGET_SHOTS,
            "sampling_repeats": args.repeats,
            "fit_K": list(CALIBRATION_K_VALUES),
            "fit_T": list(SHOT_GRID),
            "depths": list(SHALLOW_RCDF_DEPTHS),
            "rank_semantics": "independent complete joint rank-K fits",
            "frozen_native_rank_source": str(args.frozen_rank_source),
            "historical_exact_collector_settings_reused": False,
            "depth_selection": "minimum four-electron-sector Frobenius residual",
            "K_selection": "minimum frozen s-RCDF-specific stabilizer RSS loss",
            "collector_redistribution": {
                "mode": args.collector_mode,
                "minimum_extra_leaves": args.collector_min_extra_leaves,
                "maximum_extra_leaves": args.collector_max_extra_leaves,
                "formal_objective": args.collector_objective,
                "proxy_state": "hf",
                "reference_shots": args.collector_reference_shots,
                "proxy_cycles": args.collector_proxy_cycles,
                "equality_tolerance": args.collector_equality_tolerance,
                "extra_angle_steps": args.collector_extra_angle_steps,
                "zero_rank_depth_grid": list(args.collector_zero_rank_depths),
                "exact_and_absorb_are_diagnostic_only": True,
            },
        },
        "historical_baselines": {
            "methods": ["Derand", "OGM", "SG", "TND", "GPD"],
            "csv": str(HISTORICAL_CSV),
            "csv_sha256": sha256(HISTORICAL_CSV),
            "source_note": "Retained historical baseline columns in released SRDD comparison table",
            "values_modified": False,
        },
        "audit": {
            "completed_bonds": len(audits),
            "maximum_historical_vs_archived_energy_difference": maximum_energy_mismatch,
            "maximum_dense_reconstruction_frobenius": max(
                float(item["source"]["dense_reconstruction_frobenius"]) for item in audits
            ),
            "maximum_F3_reconstruction_error": max(
                float(item["maximum_F3_reconstruction_error"]) for item in audits
            ),
            "selected_leaf_spectral_audit_rows": len(leaf_spectral_rows),
            "maximum_selected_leaf_absolute_spectral_audit": max(
                float(item["absolute_spectral_audit"])
                for item in leaf_spectral_rows
            ),
            "maximum_selected_leaf_relative_spectral_audit": max(
                float(item["relative_spectral_audit"])
                for item in leaf_spectral_rows
            ),
            "all_selected_leaf_spectral_audits_within_tolerance": all(
                bool(item["within_tolerance"]) for item in leaf_spectral_rows
            ),
            "all_selected_shot_totals_equal_2038": all(
                int(item["allocated_shots"]) == TARGET_SHOTS for item in audits
            ),
            "all_exact_ground_states_withheld_until_post_selection": all(
                bool(item["exact_ground_state_used_only_after_weight_and_K_freeze"])
                for item in audits
            ),
        },
        "code": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "adaptive_core_sha256": sha256(CORE / "adaptive_f3_pool.py"),
            "shallow_rcdf_sha256": sha256(CORE / "shallow_rcdf.py"),
            "calibration_sha256": sha256(CORE / "stabilizer_calibration.py"),
            "collector_core_sha256": sha256(
                CORE / "shallow_collector_redistribution.py"
            ),
        },
        "outputs": {
            "pdf": str(pdf),
            "pdf_sha256": sha256(pdf),
            "png": str(png),
            "png_sha256": sha256(png),
        },
    }
    write_json(output / "manifest.json", manifest)
    if manifest["status"] != "PASS":
        raise RuntimeError("One or more per-bond audits failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bonds", nargs="+", type=float, default=BONDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frozen-rank-source",
        type=Path,
        default=DEFAULT_FROZEN_RANK_SOURCE,
        help=(
            "Optional v1 result root whose native fitted leaves are migrated; "
            "its historical exact collector settings are never read."
        ),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--rho", type=float, default=1.0e-6)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--angle-steps", type=int, default=40)
    parser.add_argument("--spectral-max-evaluations", type=int, default=8)
    parser.add_argument(
        "--collector-mode",
        choices=("augment", "absorb", "exact"),
        default="augment",
    )
    parser.add_argument("--collector-min-extra-leaves", type=int, default=1)
    parser.add_argument("--collector-max-extra-leaves", type=int, default=8)
    parser.add_argument(
        "--collector-objective",
        choices=("greedy", "variance", "range", "hybrid"),
        default="greedy",
    )
    parser.add_argument("--collector-reference-shots", type=int, default=3000)
    parser.add_argument("--collector-proxy-cycles", type=int, default=6)
    parser.add_argument("--collector-equality-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--collector-extra-angle-steps", type=int, default=400)
    parser.add_argument(
        "--collector-zero-rank-depths",
        nargs="+",
        type=int,
        choices=SHALLOW_RCDF_DEPTHS,
        default=SHALLOW_RCDF_DEPTHS,
    )
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.collector_mode == "augment" and (
        args.collector_min_extra_leaves < 1
        or args.collector_max_extra_leaves < args.collector_min_extra_leaves
    ):
        parser.error(
            "augment requires 1 <= collector-min-extra-leaves <= "
            "collector-max-extra-leaves"
        )
    if (
        args.collector_reference_shots < 1
        or args.collector_proxy_cycles < 1
        or args.collector_equality_tolerance <= 0.0
        or args.collector_extra_angle_steps < 1
    ):
        parser.error("invalid collector controls")
    return args


def main() -> None:
    args = parse_args()
    requested = tuple(round(float(value), 1) for value in args.bonds)
    if any(value not in BONDS for value in requested):
        raise ValueError(f"--bonds must be selected from {BONDS}")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    args.frozen_rank_source = args.frozen_rank_source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for bond in requested:
        started = time.perf_counter()
        summary, _ = run_bond(bond, args, output)
        write_json(output / f"R{bond:.1f}" / "sampling_summary.json", summary)
        print(f"[completed] R={bond:.1f} elapsed={time.perf_counter()-started:.2f}s", flush=True)
    completed = all((output / f"R{bond:.1f}" / "sampling_summary.json").exists() for bond in BONDS)
    if args.aggregate or completed:
        aggregate(output, args)
        print(f"[PASS] final Figure 3(a)+s-RCDF artifacts: {output}", flush=True)
    else:
        print(
            f"[checkpoint] completed requested bonds {requested}; run remaining bonds "
            "or pass --aggregate after all 21 are present",
            flush=True,
        )


if __name__ == "__main__":
    main()
