#!/usr/bin/env python3
"""Adaptive four-circuit-family Hamiltonian decomposition primitives.

This module is the executable companion to ``algorithm_update.md`` for the
new A-F portfolio requested after that document was written.  It deliberately
keeps the existing, audited implementations untouched.

The common benchmark backend is exact and eight-qubit.  GFRO, Operator-pool,
NNkUCCGSDI, and shallow iSWAP+SU(2) use the circuit definitions from the
audited circuit screen.  The executable stabilizer runner controls the
family-boundary size through ``eta=block_size`` (currently one); at each
boundary all four circuit families are trialled on private copies of the same
residual.  Tensor-native shallow-RC-DF is exposed only as an independent
complete-rank factorization utility; it is not a pool member.

The current publication runner keeps every accepted circuit fragment as a
complete native shallow setting and sends no source direction to a collector.
Its fixed molecular one-body reference is represented by the depth-bounded
cover in ``stabilizer_calibration.py::build_prefix_decomposition``.  The
older generalized dense F3 adapter remains below only for archived diagnostic
replay; it is not the production AGPD measurement implementation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


from srdd_release_paths import BUNDLE, DATA, METHODS

try:
    import scipy.linalg as la
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover - exercised by CLI environment check
    raise RuntimeError(
        "SciPy is required. Install the dependencies documented in the release README."
    ) from exc

from shallow_rcdf import optimize_shallow_rcdf


N_QUBITS = 8
SPATIAL_ORBITALS = 4
DIMENSION = 1 << N_QUBITS
BLOCK_SIZE = 4
FAMILIES = (
    "gfro",
    "operator_pool",
    "nnk_uccgsdi",
    "shallow_iswap_su2",
)
LABELS = {
    "gfro": "GFRO-F",
    "operator_pool": "Ope-pool-F",
    "nnk_uccgsdi": "NNkUCCGSDI-F",
    "shallow_iswap_su2": "iSWAP-F",
    "shallow_rcdf": "shallow-RC-DF-F",
    "rcdf": "RCDF-F (historical)",
    "collector": "F3 collector",
}
STANDARD_F3_FAMILIES = frozenset(("gfro", "shallow_rcdf", "rcdf"))
GENERALIZED_F3_FAMILIES = frozenset(FAMILIES) - STANDARD_F3_FAMILIES
F3_TRANSFER_FAMILIES = frozenset((*FAMILIES, "shallow_rcdf", "rcdf"))
VERSION = "adaptive-af-four-circuit-pool-sector-range-shallow-collector-v9"
LEGACY_DENSE_PORTFOLIO_VERSION = "adaptive-af-four-circuit-pool-generalized-f3-r2-v6"
CIRCUIT_SEED_NAMESPACE = "adaptive-af-pool-dd-shallow-rcdf-generalized-f3-r2-v5"
SHALLOW_RCDF_DEPTHS = (1, 2, 3)

OCCUPATIONS = (
    (np.arange(DIMENSION)[:, None] >> (N_QUBITS - 1 - np.arange(N_QUBITS))) & 1
).astype(float)
DD_FEATURES = np.column_stack(
    [np.ones(DIMENSION)]
    + [OCCUPATIONS[:, p] for p in range(N_QUBITS)]
    + [
        OCCUPATIONS[:, p] * OCCUPATIONS[:, q]
        for p in range(N_QUBITS)
        for q in range(p + 1, N_QUBITS)
    ]
)
DD_PINV = np.linalg.pinv(DD_FEATURES, rcond=1.0e-12)
SPATIAL_OCCUPATIONS = (
    OCCUPATIONS[:, :SPATIAL_ORBITALS]
    + OCCUPATIONS[:, SPATIAL_ORBITALS:]
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_circuit_module():
    path = METHODS / "gpd" / "run_exact_seven_ansatz_screen.py"
    return _load_module("adaptive_af_circuits", path)


def load_rcdf_module():
    raise NotImplementedError(
        "Historical unrestricted RCDF is outside this release; use make_shallow_rcdf_bank."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hermitian(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (array + array.conj().T)


def centered_action(matrix: np.ndarray, state: np.ndarray) -> np.ndarray:
    action = matrix @ state
    mean = np.vdot(state, action)
    return action - mean * state


def matrix_centered_norm(
    matrix: np.ndarray, sector_indices: np.ndarray | None = None
) -> float:
    value = hermitian(matrix)
    if sector_indices is not None:
        value = value[np.ix_(sector_indices, sector_indices)]
    values = np.linalg.eigvalsh(value)
    return float(0.5 * (values[-1] - values[0]))


def diagonal_centered_norm(values: np.ndarray) -> float:
    values = np.real(np.asarray(values))
    return float(0.5 * (np.max(values) - np.min(values)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class MoleculeCase:
    name: str
    label: str
    geometry: str
    source: Path
    frozen_orbitals: tuple[int, ...]
    active_orbitals: tuple[int, ...]
    active_electrons: int
    nuclear_repulsion: float
    active_constant: float
    one_spatial: np.ndarray
    two_openfermion: np.ndarray
    eri_chemist: np.ndarray
    target: np.ndarray
    base_collector: np.ndarray
    proxy_state: np.ndarray
    sector_indices: np.ndarray
    metadata: dict[str, Any]


def _blocked_spin_coefficients(one_spatial: np.ndarray, two_openfermion: np.ndarray):
    from openfermion.chem.molecular_data import spinorb_from_spatial

    one_interleaved, two_interleaved = spinorb_from_spatial(
        one_spatial, two_openfermion
    )
    m = one_spatial.shape[0]
    permutation = np.asarray(
        [2 * p for p in range(m)] + [2 * p + 1 for p in range(m)], dtype=int
    )
    one = one_interleaved[np.ix_(permutation, permutation)]
    two = two_interleaved[np.ix_(permutation, permutation, permutation, permutation)]
    return one, two


def interaction_dense(
    constant: float, one_spatial: np.ndarray, two_openfermion: np.ndarray
) -> np.ndarray:
    from openfermion import InteractionOperator, get_sparse_operator

    one, two = _blocked_spin_coefficients(one_spatial, two_openfermion)
    # ``spinorb_from_spatial`` returns the raw molecular two-body tensor;
    # OpenFermion's MolecularData.get_molecular_hamiltonian applies this same
    # 1/2 before constructing the InteractionOperator.
    operator = InteractionOperator(float(constant), one, 0.5 * two)
    dense = get_sparse_operator(operator, n_qubits=2 * len(one_spatial)).toarray()
    return hermitian(dense)


def load_active_case(name: str) -> MoleculeCase:
    """Build a controlled four-spatial-orbital active-space benchmark."""
    from openfermion.ops.representations import get_active_space_integrals

    definitions = {
        "LiH": {
            "directory": "01_LiH_R1.50A",
            "label": "LiH CAS(2e,4o)",
            "geometry": "R=1.50 Angstrom, STO-3G",
            "frozen": (0,),
            "active": (1, 2, 3, 4),
            "electrons": 2,
        },
        "H2O": {
            "directory": "02_H2O_R0.80A",
            "label": "H2O CAS(4e,4o)",
            "geometry": "O-H=0.80 Angstrom, angle=104.5 deg, STO-3G",
            "frozen": (0, 1, 2),
            "active": (3, 4, 5, 6),
            "electrons": 4,
        },
    }
    if name not in definitions:
        raise ValueError(f"Unknown molecule {name!r}")
    definition = definitions[name]
    source = DATA / name / "inputs" / "integrals_mo.npz"
    with np.load(source, allow_pickle=False) as saved:
        one_full = np.asarray(saved["one_body_integrals"], dtype=float)
        two_full = np.asarray(
            saved["two_body_integrals_openfermion_order"], dtype=float
        )
        nuclear = float(saved["nuclear_repulsion_hartree"])
    core_shift, one_active, two_active = get_active_space_integrals(
        one_full,
        two_full,
        occupied_indices=list(definition["frozen"]),
        active_indices=list(definition["active"]),
    )
    active_constant = nuclear + float(core_shift)
    target = interaction_dense(active_constant, one_active, two_active)
    base_collector = interaction_dense(
        active_constant, one_active, np.zeros_like(two_active)
    )
    eri = two_active.transpose(0, 3, 1, 2).copy()
    eri = sum(
        np.transpose(eri, axes)
        for axes in (
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (0, 1, 3, 2),
            (1, 0, 3, 2),
            (2, 3, 0, 1),
            (3, 2, 0, 1),
            (2, 3, 1, 0),
            (3, 2, 1, 0),
        )
    ) / 8.0
    n_alpha = int(definition["electrons"]) // 2
    occupied = list(range(n_alpha)) + [SPATIAL_ORBITALS + p for p in range(n_alpha)]
    proxy_index = sum(1 << (N_QUBITS - 1 - q) for q in occupied)
    proxy = np.zeros(DIMENSION, dtype=np.complex128)
    proxy[proxy_index] = 1.0
    number = OCCUPATIONS.sum(axis=1)
    sector = np.flatnonzero(number == int(definition["electrons"]))
    sector_ground = float(np.linalg.eigvalsh(target[np.ix_(sector, sector)])[0])
    metadata = {
        "molecule": name,
        "label": definition["label"],
        "geometry": definition["geometry"],
        "source_integrals": str(source),
        "source_sha256": sha256(source),
        "source_spatial_orbitals": int(one_full.shape[0]),
        "frozen_doubly_occupied_orbitals": list(definition["frozen"]),
        "active_orbitals": list(definition["active"]),
        "discarded_virtual_orbitals": sorted(
            set(range(one_full.shape[0]))
            - set(definition["frozen"])
            - set(definition["active"])
        ),
        "active_electrons": int(definition["electrons"]),
        "active_spatial_orbitals": SPATIAL_ORBITALS,
        "active_qubits": N_QUBITS,
        "active_constant_hartree": active_constant,
        "full_fock_frobenius_norm": float(np.linalg.norm(target, "fro")),
        "active_sector_ground_energy_hartree": sector_ground,
        "number_conservation_commutator_frobenius": float(
            np.linalg.norm(target @ np.diag(number) - np.diag(number) @ target)
        ),
    }
    return MoleculeCase(
        name=name,
        label=definition["label"],
        geometry=definition["geometry"],
        source=source,
        frozen_orbitals=tuple(definition["frozen"]),
        active_orbitals=tuple(definition["active"]),
        active_electrons=int(definition["electrons"]),
        nuclear_repulsion=nuclear,
        active_constant=active_constant,
        one_spatial=np.asarray(one_active, dtype=float),
        two_openfermion=np.asarray(two_active, dtype=float),
        eri_chemist=np.asarray(eri, dtype=float),
        target=target,
        base_collector=base_collector,
        proxy_state=proxy,
        sector_indices=sector,
        metadata=metadata,
    )


@dataclass
class Fragment:
    family: str
    family_label: str
    matrix: np.ndarray
    diagonal: np.ndarray
    one_matrix: np.ndarray
    one_diagonal: np.ndarray
    remainder_diagonal: np.ndarray
    q_direction: np.ndarray
    q_diagonal: np.ndarray
    f3_interface: str
    source_term: int
    candidate_metadata: dict[str, Any] = field(default_factory=dict)
    rcdf_bank_index: int | None = None

    @property
    def remainder_matrix(self) -> np.ndarray:
        return hermitian(self.matrix - self.one_matrix)

    def scaled(self, scale: float, source_term: int | None = None) -> "Fragment":
        return Fragment(
            family=self.family,
            family_label=self.family_label,
            matrix=scale * self.matrix,
            diagonal=scale * self.diagonal,
            one_matrix=scale * self.one_matrix,
            one_diagonal=scale * self.one_diagonal,
            remainder_diagonal=scale * self.remainder_diagonal,
            q_direction=scale * self.q_direction,
            q_diagonal=scale * self.q_diagonal,
            f3_interface=self.f3_interface,
            source_term=self.source_term if source_term is None else source_term,
            candidate_metadata={**self.candidate_metadata, "residual_scale": scale},
            rcdf_bank_index=self.rcdf_bank_index,
        )


def polynomial_coefficients(diagonal: np.ndarray):
    """Return degree 0/1/2 multilinear coefficients of an arbitrary diagonal."""
    values = np.real_if_close(np.asarray(diagonal)).real
    constant = float(values[0])
    one = np.empty(N_QUBITS, dtype=float)
    for p in range(N_QUBITS):
        index = 1 << (N_QUBITS - 1 - p)
        one[p] = values[index] - constant
    two = np.zeros((N_QUBITS, N_QUBITS), dtype=float)
    for p in range(N_QUBITS):
        ip = 1 << (N_QUBITS - 1 - p)
        for q in range(p + 1, N_QUBITS):
            iq = 1 << (N_QUBITS - 1 - q)
            coefficient = values[ip | iq] - constant - one[p] - one[q]
            two[p, q] = two[q, p] = coefficient
    return constant, one, two


def density_density_projection(raw_diagonal: np.ndarray) -> np.ndarray:
    """Orthogonally project a full computational diagonal onto D_dd."""
    raw = np.real_if_close(np.asarray(raw_diagonal)).real
    return DD_FEATURES @ (DD_PINV @ raw)


def fragment_from_circuit(
    circuits,
    residual: np.ndarray,
    family: str,
    candidate,
    term: int,
    winner: dict[str, Any],
) -> tuple[float, Fragment]:
    operations = circuits.build_operations(family, candidate)
    rotated = circuits.conjugate(residual, operations)
    raw_diagonal = np.real_if_close(np.diag(rotated)).real
    diagonal = density_density_projection(raw_diagonal)
    reduction = float(
        2.0 * np.dot(raw_diagonal, diagonal) - np.dot(diagonal, diagonal)
    )
    matrix = circuits.conjugate(
        np.diag(diagonal.astype(np.complex128)), operations, inverse=True
    )
    matrix = hermitian(matrix)
    constant, one, two = polynomial_coefficients(diagonal)
    one_diagonal = constant + OCCUPATIONS @ one
    remainder_diagonal = diagonal - one_diagonal
    # GFRO convention from algorithm_update.md: d_p=1/2 sum_{q != p} g_pq.
    row_direction = 0.5 * np.sum(two, axis=1)
    q_diagonal = OCCUPATIONS @ row_direction
    one_matrix = circuits.conjugate(
        np.diag(one_diagonal.astype(np.complex128)), operations, inverse=True
    )
    q_direction = circuits.conjugate(
        np.diag(q_diagonal.astype(np.complex128)), operations, inverse=True
    )
    interface = (
        "standard_one_body_f3_r2"
        if family == "gfro"
        else "generalized_dense_collector_f3_r2"
    )
    fragment = Fragment(
        family=family,
        family_label=LABELS[family],
        matrix=matrix,
        diagonal=diagonal,
        one_matrix=hermitian(one_matrix),
        one_diagonal=one_diagonal,
        remainder_diagonal=remainder_diagonal,
        q_direction=hermitian(q_direction),
        q_diagonal=q_diagonal,
        f3_interface=interface,
        source_term=term,
        candidate_metadata=winner,
    )
    return reduction, fragment


def _serializable_candidate(circuits, family: str, candidate):
    return circuits.serializable_candidate(family, candidate)


def _copy_candidate(circuits, family: str, candidate):
    return circuits.copy_candidate(family, candidate)


def stable_seed(case: str, family: str, depth: int, term: int, start: int) -> int:
    # Removing a pool member must not silently replace every circuit random
    # start.  The namespace is frozen to the audited v5 circuit ledger so the
    # v6 delta isolates the removal of shallow-RC-DF from family selection.
    payload = (
        f"{CIRCUIT_SEED_NAMESPACE}|{case}|{family}|d{depth}|K{term}|S{start}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def optimize_circuit_fragment(
    circuits,
    case_name: str,
    residual: np.ndarray,
    family: str,
    depth: int,
    term: int,
    starts: int,
    iterations: int,
) -> tuple[float, Fragment]:
    circuits.DEPTH = depth
    circuits.ITERATIONS = iterations
    best_score = -math.inf
    best_candidate = None
    best_record = None
    start_trials: list[dict[str, Any]] = []
    for start in range(starts):
        seed = stable_seed(case_name, family, depth, term, start)
        rng = np.random.default_rng(seed)
        candidate = circuits.initial_candidate(family, rng)

        def score(value) -> float:
            operations = circuits.build_operations(family, value)
            rotated = circuits.conjugate(residual, operations)
            raw_diagonal = np.real_if_close(np.diag(rotated)).real
            diagonal = density_density_projection(raw_diagonal)
            return float(
                2.0 * np.dot(raw_diagonal, diagonal)
                - np.dot(diagonal, diagonal)
            )

        current_score = score(candidate)
        initial_score = current_score
        accepted = 0
        evaluations = 1
        for iteration in range(iterations):
            plus, minus, _ = circuits.proposals(family, candidate, rng, iteration)
            plus_score = score(plus)
            minus_score = score(minus)
            evaluations += 2
            if plus_score > current_score + 1.0e-14 or minus_score > current_score + 1.0e-14:
                if plus_score >= minus_score:
                    candidate, current_score = plus, plus_score
                else:
                    candidate, current_score = minus, minus_score
                accepted += 1
        start_trials.append(
            {
                "start": start,
                "seed": seed,
                "initial_reduction": initial_score,
                "final_reduction": current_score,
                "accepted_iterations": accepted,
                "objective_evaluations": evaluations,
            }
        )
        if current_score > best_score + 1.0e-14:
            best_score = current_score
            best_candidate = _copy_candidate(circuits, family, candidate)
            best_record = {
                "winner_start": start,
                "winner_seed": seed,
                "initial_reduction": initial_score,
                "final_reduction": current_score,
                "accepted_iterations": accepted,
                "objective_evaluations": evaluations,
            }
    if best_candidate is None or best_record is None:
        raise RuntimeError(f"No candidate generated for {family}")
    best_record["random_start_count"] = starts
    best_record["start_trials"] = start_trials
    best_record["ansatz_depth"] = int(depth)
    best_record["candidate"] = _serializable_candidate(
        circuits, family, best_candidate
    )
    return fragment_from_circuit(
        circuits, residual, family, best_candidate, term, best_record
    )


def _chemist_to_openfermion(eri: np.ndarray) -> np.ndarray:
    return np.asarray(eri).transpose(0, 2, 3, 1).copy()


def make_rcdf_bank(
    case: MoleculeCase,
    leaves: int,
    rho: float,
    cycles: int,
    u_steps: int,
) -> tuple[list[Fragment], dict[str, Any]]:
    """Fit one global RCDF bank and expose its collector plus two-body leaves."""
    module = load_rcdf_module()
    maximum_leaves = SPATIAL_ORBITALS * (SPATIAL_ORBITALS + 1) // 2
    if not 1 <= leaves <= maximum_leaves:
        raise ValueError(f"RCDF leaves must be in [1,{maximum_leaves}], got {leaves}")
    rotations, tensors, fitted, optimization = module.optimize_rcdf(
        case.eri_chemist,
        leaves,
        rho,
        cycles,
        u_steps,
        1.0e-7,
    )
    bank: list[Fragment] = []
    for index, (rotation, z) in enumerate(zip(rotations, tensors)):
        eri_leaf = np.einsum(
            "pk,qk,kl,rl,sl->pqrs",
            rotation,
            rotation,
            z,
            rotation,
            rotation,
            optimize=True,
        )
        two_leaf = _chemist_to_openfermion(eri_leaf)
        matrix = interaction_dense(
            0.0, np.zeros((SPATIAL_ORBITALS, SPATIAL_ORBITALS)), two_leaf
        )
        diagonal = (
            0.5
            * np.einsum(
                "bi,ij,bj->b", SPATIAL_OCCUPATIONS, z, SPATIAL_OCCUPATIONS
            )
            - 0.5 * SPATIAL_OCCUPATIONS @ np.diag(z)
        )
        # Unlike GFRO post-processing, the standard RCDF-F3 interface starts
        # from the complete normal-ordered pair leaf.  Its -diag(Z)/2 linear
        # contribution is therefore not pre-collected: only alpha*d is moved
        # by R2, exactly as in algorithm_update.md sections 14.5 and 15.
        one_matrix = np.zeros_like(matrix)
        one_diagonal = np.zeros(DIMENSION, dtype=float)
        direction = np.sum(z, axis=1) - 0.5 * np.diag(z)
        q_spatial = rotation @ np.diag(direction) @ rotation.T
        q_matrix = interaction_dense(0.0, q_spatial, np.zeros_like(two_leaf))
        q_diagonal = SPATIAL_OCCUPATIONS @ direction
        spectral_audit = float(
            np.max(
                np.abs(
                    np.sort(np.linalg.eigvalsh(matrix))
                    - np.sort(np.real(diagonal))
                )
            )
        )
        bank.append(
            Fragment(
                family="rcdf",
                family_label=LABELS["rcdf"],
                matrix=matrix,
                diagonal=np.real(diagonal),
                one_matrix=one_matrix,
                one_diagonal=np.real(one_diagonal),
                remainder_diagonal=np.real(diagonal),
                q_direction=q_matrix,
                q_diagonal=np.real(q_diagonal),
                f3_interface="standard_tensor_native_f3_r2",
                source_term=index,
                candidate_metadata={
                    "rcdf_component": "two_body_leaf",
                    "native_diagonal_spectral_audit": spectral_audit,
                },
                rcdf_bank_index=index,
            )
        )
    fitted_operator = interaction_dense(
        case.active_constant,
        case.one_spatial,
        _chemist_to_openfermion(fitted),
    )
    audit = {
        "rho": rho,
        "leaves": leaves,
        "bank_two_body_terms": len(bank),
        "eri_residual_frobenius": float(np.linalg.norm(case.eri_chemist - fitted)),
        "full_fock_operator_residual_frobenius": float(
            np.linalg.norm(case.target - fitted_operator, "fro")
        ),
        "maximum_native_diagonal_spectral_audit": max(
            float(item.candidate_metadata.get("native_diagonal_spectral_audit", 0.0))
            for item in bank
        ),
        "optimization": optimization,
    }
    if audit["maximum_native_diagonal_spectral_audit"] > 1.0e-7:
        raise RuntimeError(
            "RCDF leaf/native-diagonal convention mismatch: "
            f"{audit['maximum_native_diagonal_spectral_audit']:.6g}"
        )
    return bank, audit


def make_shallow_rcdf_bank(
    case: MoleculeCase,
    leaves: int,
    depth: int,
    rho: float,
    cycles: int,
    angle_steps: int,
) -> tuple[list[Fragment], dict[str, Any]]:
    """Jointly fit and materialize one depth-constrained shallow-RC-DF bank."""
    maximum_leaves = SPATIAL_ORBITALS * (SPATIAL_ORBITALS + 1) // 2
    if not 1 <= leaves <= maximum_leaves:
        raise ValueError(f"shallow-RC-DF leaves must be in [1,{maximum_leaves}]")
    if depth not in SHALLOW_RCDF_DEPTHS:
        raise ValueError(f"shallow-RC-DF depth must be one of {SHALLOW_RCDF_DEPTHS}")
    rotations, tensors, fitted, optimization = optimize_shallow_rcdf(
        case.eri_chemist,
        leaves,
        depth,
        rho,
        cycles,
        angle_steps,
        1.0e-7,
        case.name,
    )
    bank: list[Fragment] = []
    angle_records = optimization["angle_records"]
    for index, (rotation, z) in enumerate(zip(rotations, tensors)):
        eri_leaf = np.einsum(
            "pk,qk,kl,rl,sl->pqrs",
            rotation,
            rotation,
            z,
            rotation,
            rotation,
            optimize=True,
        )
        two_leaf = _chemist_to_openfermion(eri_leaf)
        matrix = interaction_dense(
            0.0, np.zeros((SPATIAL_ORBITALS, SPATIAL_ORBITALS)), two_leaf
        )
        diagonal = (
            0.5
            * np.einsum(
                "bi,ij,bj->b", SPATIAL_OCCUPATIONS, z, SPATIAL_OCCUPATIONS
            )
            - 0.5 * SPATIAL_OCCUPATIONS @ np.diag(z)
        )
        direction = np.sum(z, axis=1) - 0.5 * np.diag(z)
        q_spatial = rotation @ np.diag(direction) @ rotation.T
        q_matrix = interaction_dense(0.0, q_spatial, np.zeros_like(two_leaf))
        q_diagonal = SPATIAL_OCCUPATIONS @ direction
        matrix_spectrum = np.linalg.eigvalsh(matrix)
        diagonal_spectrum = np.real(diagonal)
        spectral_audit = float(
            np.max(
                np.abs(
                    np.sort(matrix_spectrum)
                    - np.sort(diagonal_spectrum)
                )
            )
        )
        # Persist both absolute and relative diagnostics.  Near-degenerate
        # occupation spectra can acquire O(1e-7) symmetric splittings in the
        # dense JW realization while rotations remain orthogonal to O(1e-16).
        # Retain the historical 1e-7 floor and require a strict 2e-7 relative
        # agreement on the realized leaf spectral scale.
        spectral_scale = max(
            1.0,
            float(np.max(np.abs(matrix_spectrum))),
            float(np.max(np.abs(diagonal_spectrum))),
        )
        spectral_tolerance = max(1.0e-7, 2.0e-7 * spectral_scale)
        bank.append(
            Fragment(
                family="shallow_rcdf",
                family_label=LABELS["shallow_rcdf"],
                matrix=matrix,
                diagonal=np.real(diagonal),
                one_matrix=np.zeros_like(matrix),
                one_diagonal=np.zeros(DIMENSION, dtype=float),
                remainder_diagonal=np.real(diagonal),
                q_direction=q_matrix,
                q_diagonal=np.real(q_diagonal),
                f3_interface="standard_shallow_tensor_native_f3_r2",
                source_term=index,
                candidate_metadata={
                    "rcdf_component": "two_body_leaf",
                    "shallow_rcdf_depth": depth,
                    "spatial_rotation": np.asarray(rotation).tolist(),
                    "z_tensor": np.asarray(z).tolist(),
                    "spatial_angles_per_leaf": depth * (SPATIAL_ORBITALS - 1),
                    "logical_spin_givens_per_leaf": 2
                    * depth
                    * (SPATIAL_ORBITALS - 1),
                    "native_givens_depth_per_leaf": 2 * depth,
                    "logical_cnot_count_per_leaf": 4
                    * depth
                    * (SPATIAL_ORBITALS - 1),
                    "logical_cnot_depth_per_leaf": 4 * depth,
                    # Historical compatibility key; the value now follows the
                    # exact two-CNOT fermionic-Givens decomposition.
                    "unreduced_cnot_upper_bound_per_leaf": 4
                    * depth
                    * (SPATIAL_ORBITALS - 1),
                    "even_odd_angle_record": angle_records[index],
                    "native_diagonal_spectral_audit": spectral_audit,
                    "native_diagonal_spectral_scale": spectral_scale,
                    "native_diagonal_spectral_relative_audit": (
                        spectral_audit / spectral_scale
                    ),
                    "native_diagonal_spectral_audit_tolerance": spectral_tolerance,
                },
                rcdf_bank_index=index,
            )
        )
    fitted_operator = interaction_dense(
        case.active_constant,
        case.one_spatial,
        _chemist_to_openfermion(fitted),
    )
    audit = {
        "family": "shallow_rcdf",
        "depth": depth,
        "rho": rho,
        "leaves": leaves,
        "bank_two_body_terms": len(bank),
        "eri_residual_frobenius": float(np.linalg.norm(case.eri_chemist - fitted)),
        "full_fock_operator_residual_frobenius": float(
            np.linalg.norm(case.target - fitted_operator, "fro")
        ),
        "maximum_native_diagonal_spectral_audit": max(
            float(item.candidate_metadata["native_diagonal_spectral_audit"])
            for item in bank
        ),
        "maximum_native_diagonal_spectral_relative_audit": max(
            float(item.candidate_metadata["native_diagonal_spectral_relative_audit"])
            for item in bank
        ),
        "all_native_diagonal_spectral_audits_within_tolerance": all(
            float(item.candidate_metadata["native_diagonal_spectral_audit"])
            <= float(item.candidate_metadata["native_diagonal_spectral_audit_tolerance"])
            for item in bank
        ),
        "optimization": optimization,
    }
    if not audit["all_native_diagonal_spectral_audits_within_tolerance"]:
        raise RuntimeError(
            "shallow-RC-DF leaf/native-diagonal convention mismatch: "
            f"absolute={audit['maximum_native_diagonal_spectral_audit']:.6g}, "
            "one or more leaves exceed max(1e-7, 2e-7*spectral_scale)"
        )
    return bank, audit


def best_rcdf_fragment(
    residual: np.ndarray,
    bank: list[Fragment],
    used: set[int],
    term: int,
) -> tuple[float, Fragment]:
    best = None
    for candidate in bank:
        index = int(candidate.rcdf_bank_index)
        if index in used:
            continue
        norm_sq = float(np.vdot(candidate.matrix, candidate.matrix).real)
        if norm_sq <= 1.0e-20:
            continue
        overlap = float(np.vdot(candidate.matrix, residual).real)
        scale = overlap / norm_sq
        reduction = max(0.0, overlap * overlap / norm_sq)
        scaled = candidate.scaled(scale, source_term=term)
        scaled.candidate_metadata.update(
            {
                "selected_bank_index": index,
                "unscaled_overlap": overlap,
                "unscaled_norm_sq": norm_sq,
                "predicted_Frobenius_sq_reduction": reduction,
            }
        )
        key = (reduction, -index)
        if best is None or key > best[0]:
            best = (key, reduction, scaled)
    if best is None:
        raise RuntimeError("RCDF bank is exhausted")
    return best[1], best[2]


@dataclass
class F3Result:
    alpha_proxy: np.ndarray
    alpha: np.ndarray
    allocations: np.ndarray
    variances: np.ndarray
    collector: np.ndarray
    collector_norm: float
    leaf_norms: np.ndarray
    spectral_norm_sum: float
    spectral_norm_sum_before_r2: float
    raw_norm_sum_before_centering: float
    proxy_variance_factor: float
    reconstruction_error: float
    optimizer: dict[str, Any]


def f3_r2(
    fragments: list[Fragment],
    proxy_state: np.ndarray,
    sector_indices: np.ndarray | None = None,
    base_collector: np.ndarray | None = None,
    allocation_iterations: int = 6,
    spectral_max_evaluations: int = 50,
) -> F3Result:
    """Globally repartition all accepted leaves and apply spectral safeguard."""
    base = (
        np.zeros((DIMENSION, DIMENSION), dtype=np.complex128)
        if base_collector is None
        else hermitian(base_collector)
    )
    if not fragments:
        variance = float(np.vdot(centered_action(base, proxy_state), centered_action(base, proxy_state)).real)
        centered = matrix_centered_norm(base, sector_indices)
        block = base if sector_indices is None else base[np.ix_(sector_indices, sector_indices)]
        raw = float(np.max(np.abs(np.linalg.eigvalsh(block))))
        return F3Result(
            alpha_proxy=np.zeros(0),
            alpha=np.zeros(0),
            allocations=np.ones(1),
            variances=np.asarray([variance]),
            collector=base,
            collector_norm=centered,
            leaf_norms=np.zeros(0),
            spectral_norm_sum=centered,
            spectral_norm_sum_before_r2=centered,
            raw_norm_sum_before_centering=raw,
            proxy_variance_factor=variance,
            reconstruction_error=0.0,
            optimizer={"status": "empty"},
        )
    collector0 = hermitian(
        base
        + sum(
            (f.one_matrix for f in fragments),
            start=np.zeros_like(fragments[0].matrix),
        )
    )
    leaves = [f.remainder_matrix for f in fragments]
    directions = [f.q_direction for f in fragments]
    v0 = centered_action(collector0, proxy_state)
    vleaves = np.column_stack([centered_action(value, proxy_state) for value in leaves])
    basis = np.column_stack([centered_action(value, proxy_state) for value in directions])
    variances = np.asarray(
        [float(np.vdot(v0, v0).real)]
        + [float(np.vdot(vleaves[:, i], vleaves[:, i]).real) for i in range(len(fragments))]
    )
    allocations = np.sqrt(np.maximum(variances, 1.0e-16))
    allocations /= np.sum(allocations)
    alpha = np.zeros(len(fragments), dtype=float)
    gram = np.real(basis.conj().T @ basis)
    rhs0 = np.real(basis.conj().T @ v0)
    for _ in range(allocation_iterations):
        normal = gram / allocations[0]
        rhs = -rhs0 / allocations[0]
        for leaf in range(len(fragments)):
            normal[leaf, leaf] += gram[leaf, leaf] / allocations[leaf + 1]
            rhs[leaf] += float(
                np.real(np.vdot(basis[:, leaf], vleaves[:, leaf]))
            ) / allocations[leaf + 1]
        ridge = 1.0e-11 * max(1.0, float(np.trace(normal)) / len(normal))
        normal.flat[:: len(normal) + 1] += ridge
        try:
            alpha = la.solve(normal, rhs, assume_a="pos", check_finite=False)
        except la.LinAlgError:
            alpha = np.linalg.lstsq(normal, rhs, rcond=1.0e-12)[0]
        collector_action = v0 + basis @ alpha
        adjusted_actions = vleaves - basis * alpha[None, :]
        variances = np.asarray(
            [float(np.vdot(collector_action, collector_action).real)]
            + [
                float(np.vdot(adjusted_actions[:, i], adjusted_actions[:, i]).real)
                for i in range(len(fragments))
            ]
        )
        allocations = np.sqrt(np.maximum(variances, 1.0e-16))
        allocations /= np.sum(allocations)
    alpha_proxy = alpha.copy()

    def spectral(alpha_values: np.ndarray, materialize: bool = False):
        collector = hermitian(
            collector0
            + sum(
                (float(value) * direction for value, direction in zip(alpha_values, directions)),
                start=np.zeros_like(collector0),
            )
        )
        collector_norm = matrix_centered_norm(collector, sector_indices)
        leaf_norms = np.asarray(
            [
                matrix_centered_norm(
                    fragment.remainder_matrix
                    - float(value) * fragment.q_direction,
                    sector_indices,
                )
                for value, fragment in zip(alpha_values, fragments)
            ]
        )
        total = float(collector_norm + np.sum(leaf_norms))
        if materialize:
            return total, collector, collector_norm, leaf_norms
        return total

    zero_alpha = np.zeros_like(alpha_proxy)
    before = float(spectral(zero_alpha))
    proxy_value = float(spectral(alpha_proxy))
    # R2 constrains one scalar per leaf.  A direct convex/nonsmooth spectral
    # refinement is permitted inside that same parametrization.  Zero is kept
    # as an explicit fallback, so this stage cannot increase the centered sum.
    optimization = minimize(
        lambda values: spectral(np.asarray(values, dtype=float)),
        alpha_proxy,
        method="Powell",
        options={
            "maxiter": 3,
            "maxfev": spectral_max_evaluations,
            "xtol": 2.0e-3,
            "ftol": 2.0e-5,
        },
    )
    candidates = [
        (before, zero_alpha, "zero_safeguard"),
        (proxy_value, alpha_proxy, "hf_proxy"),
        (float(optimization.fun), np.asarray(optimization.x), "spectral_refinement"),
    ]
    _, alpha, selected = min(candidates, key=lambda item: item[0])
    total, collector, collector_norm, leaf_norms = spectral(alpha, materialize=True)
    # ``OptimizeResult.fun`` can differ slightly from a fresh eigensolve at
    # the returned point (notably near a multiple eigenvalue).  Enforce the
    # zero-transfer safeguard against the freshly materialized value as well;
    # this is the value that is reported and used for shot allocation.
    post_materialization_safeguard_triggered = bool(total > before)
    if post_materialization_safeguard_triggered:
        alpha = zero_alpha.copy()
        selected = "zero_safeguard_after_materialization"
        total, collector, collector_norm, leaf_norms = spectral(
            alpha, materialize=True
        )
    collector_block = collector
    if sector_indices is not None:
        collector_block = collector[np.ix_(sector_indices, sector_indices)]
    raw_norm_sum = float(np.max(np.abs(np.linalg.eigvalsh(collector_block))))
    for value, fragment in zip(alpha, fragments):
        adjusted = hermitian(
            fragment.remainder_matrix - float(value) * fragment.q_direction
        )
        if sector_indices is not None:
            adjusted = adjusted[np.ix_(sector_indices, sector_indices)]
        raw_norm_sum += float(np.max(np.abs(np.linalg.eigvalsh(adjusted))))
    adjusted_leaves = [
        fragment.remainder_matrix - float(value) * fragment.q_direction
        for value, fragment in zip(alpha, fragments)
    ]
    reconstructed = collector + sum(adjusted_leaves, start=np.zeros_like(collector))
    original = base + sum(
        (fragment.matrix for fragment in fragments), start=np.zeros_like(collector)
    )
    reconstruction_error = float(np.linalg.norm(reconstructed - original, "fro"))
    # Report proxy variance for the actually selected coefficients.
    collector_action = v0 + basis @ alpha
    adjusted_actions = vleaves - basis * alpha[None, :]
    selected_variances = np.asarray(
        [float(np.vdot(collector_action, collector_action).real)]
        + [
            float(np.vdot(adjusted_actions[:, i], adjusted_actions[:, i]).real)
            for i in range(len(fragments))
        ]
    )
    selected_allocations = np.sqrt(np.maximum(selected_variances, 1.0e-16))
    selected_allocations /= np.sum(selected_allocations)
    return F3Result(
        alpha_proxy=alpha_proxy,
        alpha=np.asarray(alpha),
        allocations=selected_allocations,
        variances=selected_variances,
        collector=collector,
        collector_norm=collector_norm,
        leaf_norms=leaf_norms,
        spectral_norm_sum=total,
        spectral_norm_sum_before_r2=before,
        raw_norm_sum_before_centering=raw_norm_sum,
        proxy_variance_factor=float(np.sum(np.sqrt(np.maximum(selected_variances, 0.0))) ** 2),
        reconstruction_error=reconstruction_error,
        optimizer={
            "selected": selected,
            "powell_success": bool(optimization.success),
            "powell_status": int(optimization.status),
            "powell_message": str(optimization.message),
            "powell_evaluations": int(optimization.nfev),
            "zero_safeguard_value": before,
            "hf_proxy_value": proxy_value,
            "spectral_refinement_value": float(optimization.fun),
            "post_materialization_safeguard_triggered": (
                post_materialization_safeguard_triggered
            ),
        },
    )


def production_f3_r2(
    fragments: list[Fragment],
    proxy_state: np.ndarray,
    sector_indices: np.ndarray | None = None,
    base_collector: np.ndarray | None = None,
    spectral_max_evaluations: int = 50,
) -> F3Result:
    """Replay the deprecated exact-dense F3 profile for legacy diagnostics.

    GFRO/shallow-RC-DF directions remain the scalable one-body standard adapter.  The
    other three families use the exact generalized adapter: their transferred
    directions are accumulated in one dense Hermitian collector, which is
    diagonalized as an additional measurement setting.  This is exact for the
    present eight-qubit implementation, but its compilation cost is not
    claimed scalable.  Current AGPD publication experiments must use
    ``stabilizer_calibration.build_prefix_decomposition`` instead.
    """
    result = f3_r2(
        fragments,
        proxy_state,
        sector_indices,
        base_collector,
        spectral_max_evaluations=spectral_max_evaluations,
    )
    result.optimizer = {
        **result.optimizer,
        "legacy_diagnostic_policy": (
            "standard one-body F3-R2 for GFRO/shallow-RC-DF; exact generalized dense-"
            "collector F3-R2 for Operator-pool/NNkUCCGSDI/iSWAP"
        ),
        "standard_source_slots": [
            index
            for index, fragment in enumerate(fragments)
            if fragment.family in STANDARD_F3_FAMILIES
        ],
        "generalized_dense_source_slots": [
            index
            for index, fragment in enumerate(fragments)
            if fragment.family in GENERALIZED_F3_FAMILIES
        ],
        "spectral_zero_safeguard_nonincrease": bool(
            result.spectral_norm_sum
            <= result.spectral_norm_sum_before_r2 + 1.0e-10
        ),
    }
    return result


@dataclass
class PortfolioState:
    residual: np.ndarray
    fragments: list[Fragment]
    used_rcdf: set[int]
    rows: list[dict[str, Any]]
    f3: F3Result


def clone_state(state: PortfolioState) -> PortfolioState:
    return PortfolioState(
        residual=state.residual.copy(),
        fragments=list(state.fragments),
        used_rcdf=set(state.used_rcdf),
        rows=[dict(row) for row in state.rows],
        f3=state.f3,
    )


def normalized_loss(
    frobenius: float,
    spectral_sum: float,
    frobenius_reference: float,
    target_centered_norm: float,
    frobenius_weight: float,
) -> float:
    # Molecular Hamiltonians contain a large arbitrary identity offset.  Using
    # ||H||_F as the scale would make a catastrophically inaccurate candidate
    # look cheap once its centered spectral sum is small.  The user supplied
    # stopping accuracy is the stable, physically meaningful F scale.
    f_component = frobenius / max(frobenius_reference, 1.0e-15)
    s_component = spectral_sum / max(target_centered_norm, 1.0e-15)
    return float(frobenius_weight * f_component + (1.0 - frobenius_weight) * s_component)


def append_family_block(
    state: PortfolioState,
    family: str,
    case: MoleculeCase,
    circuits,
    rcdf_bank: list[Fragment],
    depth: int,
    starts: int,
    iterations: int,
    block_size: int,
    frobenius_reference: float,
    target_centered_norm: float,
    frobenius_weight: float,
    spectral_max_evaluations: int,
) -> PortfolioState:
    trial = clone_state(state)
    for _ in range(block_size):
        term = len(trial.fragments) + 1
        old_sq = float(np.vdot(trial.residual, trial.residual).real)
        if family == "rcdf":
            reduction, fragment = best_rcdf_fragment(
                trial.residual, rcdf_bank, trial.used_rcdf, term
            )
            trial.used_rcdf.add(int(fragment.rcdf_bank_index))
        else:
            reduction, fragment = optimize_circuit_fragment(
                circuits,
                case.name,
                trial.residual,
                family,
                depth,
                term,
                starts,
                iterations,
            )
        trial.residual = hermitian(trial.residual - fragment.matrix)
        actual = old_sq - float(np.vdot(trial.residual, trial.residual).real)
        if actual < -1.0e-8:
            raise RuntimeError(f"{family} increased F-distance at K={term}")
        if abs(actual - reduction) > 2.0e-7 * max(1.0, abs(reduction)):
            raise RuntimeError(
                f"Reduction identity failed for {family} K={term}: {actual} vs {reduction}"
            )
        trial.fragments.append(fragment)
        # Curves report the reoptimized complete-prefix F3, not stale per-leaf
        # coefficients.  Non-Gaussian directions enter the dense collector.
        prefix_f3 = production_f3_r2(
            trial.fragments,
            case.proxy_state,
            None,
            case.base_collector,
            spectral_max_evaluations=spectral_max_evaluations,
        )
        frobenius = float(np.linalg.norm(trial.residual, "fro"))
        loss = normalized_loss(
            frobenius,
            prefix_f3.spectral_norm_sum,
            frobenius_reference,
            target_centered_norm,
            frobenius_weight,
        )
        trial.rows.append(
            {
                "K": term,
                "K_source_terms": term,
                "total_measurement_settings": term + 1,
                "family": family,
                "family_label": LABELS[family],
                "f3_interface": fragment.f3_interface,
                "Frobenius_distance": frobenius,
                "Frobenius_sq_reduction": actual,
                "centered_spectral_norm_sum": prefix_f3.spectral_norm_sum,
                "spectral_norm_domain": "full_active_fock_space",
                "centered_spectral_norm_sum_before_R2": prefix_f3.spectral_norm_sum_before_r2,
                "raw_spectral_norm_sum_before_centering": prefix_f3.raw_norm_sum_before_centering,
                "total_loss": loss,
                "f3_reconstruction_error": prefix_f3.reconstruction_error,
                "hf_proxy_variance_factor": prefix_f3.proxy_variance_factor,
            }
        )
        trial.f3 = prefix_f3
    return trial


def run_portfolio(
    case: MoleculeCase,
    output: Path,
    max_terms: int = 16,
    depth: int = 1,
    starts: int = 3,
    iterations: int = 3,
    frobenius_weight: float = 0.8,
    f_threshold: float = 1.0,
    plateau_relative_tolerance: float = 2.0e-3,
    plateau_patience: int = 2,
    rcdf_rho: float = 1.0e-6,
    rcdf_cycles: int = 4,
    rcdf_u_steps: int = 2,
    spectral_max_evaluations: int = 40,
) -> dict[str, Any]:
    if max_terms < BLOCK_SIZE:
        raise ValueError(f"max_terms must be at least {BLOCK_SIZE}")
    if depth < 1 or starts < 1 or iterations < 0:
        raise ValueError("depth and starts must be positive; iterations must be nonnegative")
    if not 0.0 <= frobenius_weight <= 1.0:
        raise ValueError("frobenius_weight must lie in [0,1]")
    if f_threshold <= 0.0:
        raise ValueError("f_threshold must be positive")
    if plateau_relative_tolerance < 0.0 or plateau_patience < 1:
        raise ValueError("plateau tolerance must be nonnegative and patience positive")
    if rcdf_cycles < 1 or rcdf_u_steps < 1 or spectral_max_evaluations < 1:
        raise ValueError("RCDF and spectral optimization budgets must be positive")
    circuits = load_circuit_module()
    circuits.DEPTH = depth
    circuits.ITERATIONS = iterations
    circuits.validate_operator_pool_definition()
    initial_residual = hermitian(case.target - case.base_collector)
    initial_frobenius = float(np.linalg.norm(initial_residual, "fro"))
    target_centered_norm = matrix_centered_norm(case.target)
    empty_f3 = production_f3_r2(
        [], case.proxy_state, None, case.base_collector
    )
    state = PortfolioState(
        residual=initial_residual,
        fragments=[],
        used_rcdf=set(),
        rows=[],
        f3=empty_f3,
    )
    selections = []
    rcdf_audits = []
    plateau_count = 0
    previous_boundary_loss = normalized_loss(
        initial_frobenius,
        empty_f3.spectral_norm_sum,
        f_threshold,
        target_centered_norm,
        frobenius_weight,
    )
    stop_reason = "maximum_terms"
    started = time.perf_counter()
    while len(state.fragments) < max_terms:
        block_size = min(BLOCK_SIZE, max_terms - len(state.fragments))
        boundary = len(state.fragments)
        rcdf_leaves = min(
            SPATIAL_ORBITALS * (SPATIAL_ORBITALS + 1) // 2,
            boundary + block_size,
        )
        rcdf_bank, rcdf_audit = make_rcdf_bank(
            case, rcdf_leaves, rcdf_rho, rcdf_cycles, rcdf_u_steps
        )
        rcdf_audits.append({"boundary_after_K": boundary, **rcdf_audit})
        candidates = []
        for family in FAMILIES:
            print(
                f"[{case.name}] boundary K={boundary:02d}: trial {LABELS[family]}",
                flush=True,
            )
            trial_started = time.perf_counter()
            source_state = clone_state(state)
            if family == "rcdf":
                # The bank is jointly fitted at the current endpoint budget;
                # indices only prevent duplicate use inside this trial block.
                source_state.used_rcdf = set()
            candidate = append_family_block(
                source_state,
                family,
                case,
                circuits,
                rcdf_bank,
                depth,
                starts,
                iterations,
                block_size,
                f_threshold,
                target_centered_norm,
                frobenius_weight,
                spectral_max_evaluations,
            )
            endpoint = candidate.rows[-1]
            candidates.append((family, candidate, time.perf_counter() - trial_started))
            print(
                f"[{case.name}] {LABELS[family]} -> "
                f"F={endpoint['Frobenius_distance']:.6g}, "
                f"sum||D||2={endpoint['centered_spectral_norm_sum']:.6g}, "
                f"loss={endpoint['total_loss']:.6g}",
                flush=True,
            )
        if not candidates:
            stop_reason = "no_eligible_candidate"
            break
        family_order = {family: index for index, family in enumerate(FAMILIES)}
        chosen_family, chosen, chosen_elapsed = min(
            candidates,
            key=lambda item: (
                float(item[1].rows[-1]["total_loss"]),
                family_order[item[0]],
            ),
        )
        chosen_loss = float(chosen.rows[-1]["total_loss"])
        relative_improvement = (previous_boundary_loss - chosen_loss) / max(
            abs(previous_boundary_loss), 1.0e-15
        )
        selection_record = {
                "boundary_after_K": boundary,
                "block_size": block_size,
                "candidate_end_K": boundary + block_size,
                "candidates": [
                    {
                        "family": family,
                        "family_label": LABELS[family],
                        "Frobenius_distance": float(candidate.rows[-1]["Frobenius_distance"]),
                        "centered_spectral_norm_sum": float(candidate.rows[-1]["centered_spectral_norm_sum"]),
                        "total_loss": float(candidate.rows[-1]["total_loss"]),
                        "elapsed_seconds": elapsed,
                    }
                    for family, candidate, elapsed in candidates
                ],
                "selected_family": chosen_family,
                "selected_family_label": LABELS[chosen_family],
                "selected_loss": chosen_loss,
                "relative_loss_improvement": relative_improvement,
                "selected_trial_elapsed_seconds": chosen_elapsed,
            }
        candidate_has_threshold = any(
            float(row["Frobenius_distance"]) < f_threshold
            for row in chosen.rows[boundary:]
        )
        if relative_improvement < 0.0 and not candidate_has_threshold:
            selection_record["accepted"] = False
            selection_record["rejection_reason"] = "total_loss_did_not_decrease"
            selections.append(selection_record)
            stop_reason = "total_loss_not_improved"
            break
        selection_record["accepted"] = True
        selection_record["rejection_reason"] = None
        selection_record["accepted_end_K"] = boundary + block_size
        selection_record["truncated_at_threshold"] = False
        selection_record["accepted_prefix_loss"] = chosen_loss
        selections.append(selection_record)
        state = chosen
        print(
            f"[{case.name}] accepted K={boundary + 1}-{len(state.fragments)}: "
            f"{LABELS[chosen_family]}",
            flush=True,
        )
        first_below = next(
            (
                int(row["K"])
                for row in state.rows[boundary:]
                if float(row["Frobenius_distance"]) < f_threshold
            ),
            None,
        )
        if first_below is not None:
            # The family was selected by the complete four-term comparison;
            # retain only the physically required prefix and globally redo F3.
            keep = first_below
            state.fragments = state.fragments[:keep]
            approximation = sum(
                (fragment.matrix for fragment in state.fragments),
                start=case.base_collector.copy(),
            )
            state.residual = hermitian(case.target - approximation)
            state.rows = state.rows[:keep]
            state.used_rcdf = {
                int(fragment.rcdf_bank_index)
                for fragment in state.fragments
                if fragment.rcdf_bank_index is not None
            }
            state.f3 = production_f3_r2(
                state.fragments,
                case.proxy_state,
                None,
                case.base_collector,
                spectral_max_evaluations=spectral_max_evaluations,
            )
            terminal_row = state.rows[-1]
            terminal_row["centered_spectral_norm_sum"] = state.f3.spectral_norm_sum
            terminal_row["centered_spectral_norm_sum_before_R2"] = (
                state.f3.spectral_norm_sum_before_r2
            )
            terminal_row["raw_spectral_norm_sum_before_centering"] = (
                state.f3.raw_norm_sum_before_centering
            )
            terminal_row["total_loss"] = normalized_loss(
                float(terminal_row["Frobenius_distance"]),
                state.f3.spectral_norm_sum,
                f_threshold,
                target_centered_norm,
                frobenius_weight,
            )
            terminal_row["f3_reconstruction_error"] = state.f3.reconstruction_error
            terminal_row["hf_proxy_variance_factor"] = state.f3.proxy_variance_factor
            selection_record["accepted_end_K"] = keep
            selection_record["truncated_at_threshold"] = keep < boundary + block_size
            selection_record["accepted_prefix_loss"] = float(terminal_row["total_loss"])
            stop_reason = "Frobenius_distance_below_threshold"
            break
        if relative_improvement <= plateau_relative_tolerance:
            plateau_count += 1
        else:
            plateau_count = 0
        if plateau_count >= plateau_patience:
            stop_reason = "total_loss_plateau"
            break
        previous_boundary_loss = chosen_loss

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics_by_k.csv", state.rows)
    setting_rows = []
    cumulative_norm = 0.0
    final_settings = [
        (
            0,
            "collector",
            LABELS["collector"],
            "collector",
            "generalized_dense_collector",
            state.f3.collector,
            state.f3.collector_norm,
        )
    ]
    for index, fragment in enumerate(state.fragments):
        adjusted = hermitian(
            fragment.remainder_matrix
            - state.f3.alpha[index] * fragment.q_direction
        )
        final_settings.append(
            (
                index + 1,
                fragment.family,
                fragment.family_label,
                "source_leaf",
                fragment.f3_interface,
                adjusted,
                float(state.f3.leaf_norms[index]),
            )
        )
    for setting_index, family, family_label, kind, interface, matrix, norm in final_settings:
        eigenvalues = np.linalg.eigvalsh(matrix)
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        cumulative_norm += norm
        setting_rows.append(
            {
                "setting_index": setting_index,
                "source_term_K": max(0, setting_index),
                "family": family,
                "family_label": family_label,
                "setting_kind": kind,
                "f3_interface": interface,
                "lambda_min": minimum,
                "lambda_max": maximum,
                "raw_spectral_norm": float(np.max(np.abs(eigenvalues))),
                "optimal_identity_shift": 0.5 * (minimum + maximum),
                "centered_spectral_norm": norm,
                "cumulative_centered_spectral_norm": cumulative_norm,
                "spectral_norm_domain": "full_active_fock_space",
            }
        )
    write_csv(output / "terminal_fragment_spectral_norms.csv", setting_rows)
    np.savez_compressed(
        output / "selected_fragments.npz",
        matrices=np.asarray([fragment.matrix for fragment in state.fragments]),
        families=np.asarray([fragment.family for fragment in state.fragments]),
        f3_interfaces=np.asarray(
            [fragment.f3_interface for fragment in state.fragments]
        ),
        diagonals=np.asarray([fragment.diagonal for fragment in state.fragments]),
        one_diagonals=np.asarray([fragment.one_diagonal for fragment in state.fragments]),
        q_diagonals=np.asarray([fragment.q_diagonal for fragment in state.fragments]),
        f3_alpha=state.f3.alpha,
        f3_allocations=state.f3.allocations,
        f3_leaf_centered_spectral_norms=state.f3.leaf_norms,
        f3_collector=state.f3.collector,
        exact_base_collector=case.base_collector,
    )
    (output / "selection_log.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8"
    )
    maximum_reconstruction = max(
        [float(row["f3_reconstruction_error"]) for row in state.rows] or [0.0]
    )
    final_frobenius = float(np.linalg.norm(state.residual, "fro"))
    manifest = {
        "definition_version": LEGACY_DENSE_PORTFOLIO_VERSION,
        "current_publication_source_core_version": VERSION,
        "legacy_dense_diagnostic_profile": True,
        "algorithm": "hybrid-ansatz-F algorithm",
        "implementation_alias": "adaptive A-F pool with block size four",
        "ansatz_pool": [LABELS[family] for family in FAMILIES],
        "molecule": case.metadata,
        "block_size": BLOCK_SIZE,
        "ansatz_depth": depth,
        "starts_per_circuit_fragment": starts,
        "iterations_per_start": iterations,
        "objective_evaluations_per_start": 1 + 2 * iterations,
        "loss": {
            "formula": "w_F*(F/F_stop)+(1-w_F)*(sum_centered_spectral/target_centered_spectral)",
            "frobenius_weight": frobenius_weight,
            "Frobenius_reference_F_stop": f_threshold,
            "target_centered_spectral_norm": target_centered_norm,
            "spectral_norm_domain": "full_active_fock_space",
        },
        "stopping": {
            "Frobenius_threshold": f_threshold,
            "plateau_relative_tolerance": plateau_relative_tolerance,
            "plateau_patience_blocks": plateau_patience,
            "maximum_source_terms": max_terms,
            "stop_reason": stop_reason,
        },
        "F3_R2": {
            "allocation_iterations": 6,
            "spectral_refinement_max_function_evaluations": spectral_max_evaluations,
            "standard_one_body_families": [
                LABELS[family] for family in STANDARD_F3_FAMILIES
            ],
            "generalized_dense_collector_families": [
                LABELS[family] for family in GENERALIZED_F3_FAMILIES
            ],
            "non_gaussian_policy": (
                "Use an exact dense Hermitian collector for transferred "
                "non-Gaussian directions; diagonalize and count it as the "
                "collector setting, while explicitly reporting exponential "
                "generic-unitary compilation risk."
            ),
            "final_collector_interface": "generalized_dense_collector",
            "final_optimizer": state.f3.optimizer,
        },
        "RCDF": {
            "mode": "budget-matched residual-local dictionary extension",
            "note": (
                "At each four-leaf boundary, a fresh tensor-native L2-RCDF bank "
                "is jointly fitted with the current endpoint leaf budget; four "
                "native leaves are then scaled by exact residual projection. This "
                "makes RCDF composable with non-Gaussian residuals and is not the "
                "unchanged global-prefix L2-RCDF algorithm of algorithm_update.md."
            ),
            "boundary_fits": rcdf_audits,
        },
        "completed_source_terms_K": len(state.fragments),
        "completed_measurement_settings_including_collector": len(state.fragments) + 1,
        "final_Frobenius_distance": final_frobenius,
        "final_centered_spectral_norm_sum": state.f3.spectral_norm_sum,
        "maximum_F3_reconstruction_error": maximum_reconstruction,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    audit = audit_result(
        case,
        state,
        max_reconstruction=maximum_reconstruction,
        manifest=manifest,
        setting_rows=setting_rows,
    )
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"Audit failed: {audit}")
    return {
        "case": case,
        "state": state,
        "selections": selections,
        "manifest": manifest,
        "audit": audit,
        "output": output,
    }


def audit_result(
    case: MoleculeCase,
    state: PortfolioState,
    max_reconstruction: float | None = None,
    manifest: dict[str, Any] | None = None,
    setting_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    approximation = sum(
        (fragment.matrix for fragment in state.fragments),
        start=case.base_collector.copy(),
    )
    residual = hermitian(case.target - approximation)
    row_frobenius_error = 0.0
    replay = hermitian(case.target - case.base_collector)
    monotone = True
    previous = float(np.linalg.norm(replay, "fro"))
    lengths_match = len(state.rows) == len(state.fragments)
    k_sequence = [int(row.get("K", -1)) for row in state.rows]
    k_is_contiguous = k_sequence == list(range(1, len(state.rows) + 1))
    for row, fragment in zip(state.rows, state.fragments):
        replay = hermitian(replay - fragment.matrix)
        value = float(np.linalg.norm(replay, "fro"))
        row_frobenius_error = max(
            row_frobenius_error, abs(value - float(row["Frobenius_distance"]))
        )
        monotone &= value <= previous + 1.0e-9
        previous = value
    final_residual_error = float(np.linalg.norm(residual - state.residual, "fro"))
    fragment_hermiticity = max(
        [float(np.linalg.norm(f.matrix - f.matrix.conj().T, "fro")) for f in state.fragments]
        or [0.0]
    )
    reported_f3_reconstruction = (
        state.f3.reconstruction_error if max_reconstruction is None else max_reconstruction
    )
    adjusted_leaves = [
        fragment.remainder_matrix - float(alpha) * fragment.q_direction
        for fragment, alpha in zip(state.fragments, state.f3.alpha)
    ]
    f3_original = case.base_collector + sum(
        (fragment.matrix for fragment in state.fragments),
        start=np.zeros_like(case.target),
    )
    f3_reconstructed = state.f3.collector + sum(
        adjusted_leaves, start=np.zeros_like(case.target)
    )
    independent_f3_reconstruction = float(
        np.linalg.norm(f3_reconstructed - f3_original, "fro")
    )
    final_frobenius = float(np.linalg.norm(state.residual, "fro"))
    terminal_row_frobenius_error = (
        abs(float(state.rows[-1]["Frobenius_distance"]) - final_frobenius)
        if state.rows else 0.0
    )
    terminal_row_spectral_error = (
        abs(
            float(state.rows[-1]["centered_spectral_norm_sum"])
            - state.f3.spectral_norm_sum
        )
        if state.rows else 0.0
    )
    setting_sum_error = (
        abs(
            float(setting_rows[-1]["cumulative_centered_spectral_norm"])
            - state.f3.spectral_norm_sum
        )
        if setting_rows else 0.0
    )
    manifest_frobenius_error = (
        abs(float(manifest["final_Frobenius_distance"]) - final_frobenius)
        if manifest is not None else 0.0
    )
    manifest_spectral_error = (
        abs(
            float(manifest["final_centered_spectral_norm_sum"])
            - state.f3.spectral_norm_sum
        )
        if manifest is not None else 0.0
    )
    status = "PASS" if (
        lengths_match
        and k_is_contiguous
        and final_residual_error < 1.0e-9
        and row_frobenius_error < 1.0e-8
        and fragment_hermiticity < 1.0e-9
        and reported_f3_reconstruction < 1.0e-8
        and independent_f3_reconstruction < 1.0e-8
        and terminal_row_frobenius_error < 1.0e-10
        and terminal_row_spectral_error < 1.0e-10
        and setting_sum_error < 1.0e-10
        and manifest_frobenius_error < 1.0e-10
        and manifest_spectral_error < 1.0e-10
        and monotone
    ) else "FAIL"
    return {
        "status": status,
        "completed_K": len(state.fragments),
        "rows_equal_fragments": lengths_match,
        "K_sequence_is_contiguous": k_is_contiguous,
        "Frobenius_monotone_nonincreasing": bool(monotone),
        "maximum_row_Frobenius_replay_error": row_frobenius_error,
        "final_residual_replay_error": final_residual_error,
        "maximum_fragment_hermiticity_error": fragment_hermiticity,
        "maximum_reported_F3_Hamiltonian_reconstruction_error": reported_f3_reconstruction,
        "independent_terminal_F3_Hamiltonian_reconstruction_error": independent_f3_reconstruction,
        "terminal_row_Frobenius_error": terminal_row_frobenius_error,
        "terminal_row_spectral_sum_error": terminal_row_spectral_error,
        "terminal_setting_cumulative_sum_error": setting_sum_error,
        "manifest_final_Frobenius_error": manifest_frobenius_error,
        "manifest_final_spectral_sum_error": manifest_spectral_error,
        "F3_does_not_change_Frobenius_residual": True,
    }
