#!/usr/bin/env python3
"""Exact local-depolarizing audit for Nature-2023 overlapping FC-IMA.

This script is deliberately isolated from all manuscript and plotting sources.
It replays the coefficient-sorted overlapping fully-commuting groups, the
iterative measurement allocation (IMA), the Yen--Verteletskyi--Izmaylov
measurement Clifford, Qiskit's logical all-to-all compilation, and the frozen
T=3000 FC-IMA shot vector produced by
``error_eval/traditional_cm_error_eval.py``.

The estimator is the pooled estimator of Eqs. (14) and (17) of
Yen--Ganeshram--Izmaylov, npj Quantum Information 9, 14 (2023).  If Pauli k is
present in multiple bases, its total number of observations is

    M_k = sum_{alpha containing k} m_alpha.

The noisy mean is accumulated basis by basis because the same Pauli can acquire
different attenuation in different diagonalizing circuits.  Independent shots
and independent bases give the exact noisy extension of Eq. (17):

    Var(Hbar) = sum_alpha m_alpha Var[sum_{k in alpha} c_k P_k / M_k].

Noise model
-----------
After every explicitly compiled CNOT on support S, apply

    E_{p,S}(rho) = (1-p) rho + (p/4) I_S tensor Tr_S(rho).

State preparation, one-qubit gates, and readout are ideal.  In the Heisenberg
picture a Pauli is unchanged if it is identity on both qubits in S and is
otherwise multiplied by (1-p).  Since the compiled measurement circuit is a
Clifford, every propagated observable remains a signed Pauli.  Consequently
the first and second noisy moments of every FC-IMA basis are evaluated exactly,
without MPO truncation, Gaussian replay, or Monte Carlo noise trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import qiskit
import scipy
import scipy.sparse as sp
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, Pauli


VERSION = "nature-2023-overlapping-fc-ima-exact-per-cnot-local-depolarizing-v4"
P_GRID = tuple(float(value) for value in np.linspace(0.0, 0.003, 10))
TOTAL_SHOTS = 3000
ALLOCATION = "fc_ima_primary"
MOMENT_TOLERANCE = 2.0e-8
ANGLE_TOLERANCE = 2.0e-8

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pauli_product"))
from release_paths import DATA_ROOT, OUTPUT_ROOT
BENCHMARK_ROOT = DATA_ROOT / "fully_commuting" / "BeH2_N2_FC_IMA"
ERROR_ROOT = BENCHMARK_ROOT / "error_eval"
ERROR_RESULTS = ERROR_ROOT / "results"
RESULTS_ROOT = OUTPUT_ROOT / "fully_commuting" / "noise_eval" / "results"
DEPTH_MODULE_PATH = HERE / "traditional_cm_depth.py"
ERROR_MODULE_PATH = HERE / "traditional_cm_error_eval.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


depth = load_module("traditional_cm_depth_dependency", DEPTH_MODULE_PATH)
error = load_module("traditional_cm_error_dependency", ERROR_MODULE_PATH)


@dataclass(frozen=True)
class CompiledGroup:
    group_index: int
    optimized: QuantumCircuit
    output_z_masks: np.ndarray
    output_signs: np.ndarray
    term_exponents: np.ndarray
    pair_exponents: np.ndarray
    operation_names: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qiskit_mask(pauli: Pauli, component: str) -> int:
    values = getattr(pauli, component)
    result = 0
    for qubit, value in enumerate(values):
        if bool(value):
            result |= 1 << qubit
    return result


def pauli_sign(pauli: Pauli) -> int:
    label = pauli.to_label()
    if label.startswith("-i") or label.startswith("i"):
        raise AssertionError(f"Hermitian Pauli acquired imaginary phase: {label}")
    return -1 if label.startswith("-") else 1


def compiled_operations(circuit: QuantumCircuit) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
    result: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = tuple(circuit.find_bit(bit).index for bit in instruction.qubits)
        params = tuple(float(value) for value in instruction.operation.params)
        result.append((name, qubits, params))
    return result


def validate_cx_only_two_qubit_operations(
    operations: Sequence[tuple[str, tuple[int, ...], tuple[float, ...]]],
) -> None:
    """Fail closed unless every FC two-qubit element is already an explicit CX."""
    illegal = [
        name for name, qubits, _ in operations if len(qubits) == 2 and name != "cx"
    ]
    if illegal:
        raise RuntimeError(f"Unexpected non-CX two-qubit gates: {illegal}")
    oversized = [name for name, qubits, _ in operations if len(qubits) > 2]
    if oversized:
        raise RuntimeError(f"Unexpected >2q compiled gates: {oversized}")


def attenuation_exponents(
    output_z_masks: np.ndarray,
    operations: Sequence[tuple[str, tuple[int, ...], tuple[float, ...]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reverse-propagate output Z Paulis and count active 2q channels.

    ``output_z_masks`` uses Qiskit's little-endian qubit bit order.  The
    returned X/Z masks describe U^dagger Z U.  Clifford phases are irrelevant
    to attenuation and are independently fixed by the verified output signs.
    """
    z = np.asarray(output_z_masks, dtype=np.uint64).copy()
    x = np.zeros_like(z)
    counts = np.zeros(len(z), dtype=np.int32)
    for name, qubits, params in reversed(operations):
        if name == "cx":
            if len(qubits) != 2:
                raise AssertionError("CX does not have two qubits")
            control, target = qubits
            support = np.uint64((1 << control) | (1 << target))
            counts += (((x | z) & support) != 0).astype(np.int32)
            # CX is self-adjoint: x_t <- x_t xor x_c, z_c <- z_c xor z_t.
            x ^= ((x >> np.uint64(control)) & np.uint64(1)) << np.uint64(target)
            z ^= ((z >> np.uint64(target)) & np.uint64(1)) << np.uint64(control)
        elif name == "rz":
            if len(qubits) != 1 or len(params) != 1:
                raise AssertionError("Unexpected RZ signature")
            quarter_turns = int(round(params[0] / (math.pi / 2.0)))
            residual = params[0] - quarter_turns * (math.pi / 2.0)
            if abs(residual) > ANGLE_TOLERANCE:
                raise RuntimeError(
                    f"Non-Clifford RZ angle {params[0]:.17g}; residual {residual:.3e}"
                )
            if quarter_turns & 1:
                qubit = qubits[0]
                z ^= ((x >> np.uint64(qubit)) & np.uint64(1)) << np.uint64(qubit)
        elif name in {"sx", "sxdg"}:
            if len(qubits) != 1:
                raise AssertionError(f"Unexpected {name} signature")
            qubit = qubits[0]
            x ^= ((z >> np.uint64(qubit)) & np.uint64(1)) << np.uint64(qubit)
        elif name in {"x", "y", "z", "id", "barrier"}:
            # Pauli gates only change phase.  Barriers and identity do nothing.
            continue
        elif name == "h":
            qubit = qubits[0]
            xb = (x >> np.uint64(qubit)) & np.uint64(1)
            zb = (z >> np.uint64(qubit)) & np.uint64(1)
            changed = xb ^ zb
            x ^= changed << np.uint64(qubit)
            z ^= changed << np.uint64(qubit)
        elif name in {"s", "sdg"}:
            qubit = qubits[0]
            z ^= ((x >> np.uint64(qubit)) & np.uint64(1)) << np.uint64(qubit)
        else:
            raise RuntimeError(
                f"Unsupported compiled operation {name!r}; exact Clifford audit stopped"
            )
    return counts, x, z


def build_compiled_group(
    group_index: int,
    group: Sequence[Any],
    n_qubits: int,
    seed: int,
) -> CompiledGroup:
    complete_basis = depth.complete_commuting_basis([term.label for term in group])
    expanded_core = depth.build_expanded_yen_circuit(complete_basis)
    core_clifford = Clifford(expanded_core)
    local = depth.local_qwc_diagonalizer(core_clifford, complete_basis)
    expanded = expanded_core.copy()
    expanded.compose(local, inplace=True)
    final_clifford = Clifford(expanded)
    raw = core_clifford.to_circuit()
    raw.compose(local, inplace=True)
    if Clifford(raw) != final_clifford:
        raise AssertionError("Clifford resynthesis changed the FC unitary")
    optimized = transpile(
        raw,
        basis_gates=list(depth.BASIS_GATES),
        optimization_level=1,
        seed_transpiler=seed,
    )
    if Clifford(optimized) != final_clifford:
        raise AssertionError("Qiskit transpilation changed the FC Clifford")

    output_masks: list[int] = []
    output_signs: list[int] = []
    original_x: list[int] = []
    original_z: list[int] = []
    for term in group:
        transformed = Pauli(term.label).evolve(final_clifford, frame="s")
        if np.any(transformed.x):
            raise AssertionError(
                f"Group {group_index}: non-diagonal image {transformed.to_label()}"
            )
        output_masks.append(qiskit_mask(transformed, "z"))
        output_signs.append(pauli_sign(transformed))
        # Depth module label bits run left-to-right; convert to Qiskit order.
        error_x, error_z, _ = error.label_to_masks(term.label)
        original_x.append(error_x)
        original_z.append(error_z)

    operations = compiled_operations(optimized)
    names = tuple(name for name, _, _ in operations)
    validate_cx_only_two_qubit_operations(operations)
    output_array = np.asarray(output_masks, dtype=np.uint64)
    term_counts, traced_x, traced_z = attenuation_exponents(output_array, operations)
    if not np.array_equal(traced_x, np.asarray(original_x, dtype=np.uint64)):
        raise AssertionError(f"Group {group_index}: reverse X masks do not match inputs")
    if not np.array_equal(traced_z, np.asarray(original_z, dtype=np.uint64)):
        raise AssertionError(f"Group {group_index}: reverse Z masks do not match inputs")

    pair_masks = np.bitwise_xor(output_array[:, None], output_array[None, :])
    unique_masks, inverse = np.unique(pair_masks, return_inverse=True)
    pair_unique_counts, _, _ = attenuation_exponents(unique_masks, operations)
    pair_counts = pair_unique_counts[inverse].reshape(pair_masks.shape)
    return CompiledGroup(
        group_index=group_index,
        optimized=optimized,
        output_z_masks=output_array,
        output_signs=np.asarray(output_signs, dtype=np.int8),
        term_exponents=term_counts,
        pair_exponents=pair_counts,
        operation_names=names,
    )


def action_gram_matrix(
    group: Sequence[Any],
    state: np.ndarray,
    support: np.ndarray,
    parity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return <P_i> and <P_i P_j> using sparse Pauli-action columns."""
    amplitudes = state[support]
    column_size = len(support)
    term_count = len(group)
    indices = np.empty(column_size * term_count, dtype=np.int32)
    values = np.empty(column_size * term_count, dtype=np.complex128)
    means = np.empty(term_count, dtype=float)
    for index, term in enumerate(group):
        start = index * column_size
        stop = start + column_size
        targets = np.bitwise_xor(support, np.uint32(term.xmask))
        signs = 1 - 2 * parity[np.bitwise_and(support, np.uint32(term.zmask))]
        acted = (1j ** term.y_count) * signs * amplitudes
        indices[start:stop] = targets.astype(np.int32, copy=False)
        values[start:stop] = acted
        mean = np.vdot(state[targets], acted)
        if abs(mean.imag) > MOMENT_TOLERANCE:
            raise RuntimeError(f"Non-real Pauli mean {mean} for {term.label}")
        means[index] = float(mean.real)
    indptr = np.arange(term_count + 1, dtype=np.int64) * column_size
    actions = sp.csc_matrix(
        (values, indices, indptr), shape=(len(state), term_count), copy=False
    )
    actions.sort_indices()
    gram_complex = (actions.conjugate().T @ actions).toarray()
    max_imaginary = float(np.max(np.abs(gram_complex.imag)))
    if max_imaginary > MOMENT_TOLERANCE:
        raise RuntimeError(f"Non-real commuting Pauli Gram matrix: {max_imaginary:.3e}")
    gram = np.asarray(gram_complex.real, dtype=float)
    asymmetry = float(np.max(np.abs(gram - gram.T)))
    if asymmetry > MOMENT_TOLERANCE:
        raise RuntimeError(f"Asymmetric Pauli Gram matrix: {asymmetry:.3e}")
    diagonal_error = float(np.max(np.abs(np.diag(gram) - 1.0)))
    if diagonal_error > MOMENT_TOLERANCE:
        raise RuntimeError(f"Pauli action norm error: {diagonal_error:.3e}")
    return means, gram


def load_ima_groups(
    molecule: str, terms: Sequence[Any]
) -> tuple[list[list[Any]], dict[int, int]]:
    """Load and fail-closed validate the Nature-2023 overlapping FC groups."""
    path = ERROR_RESULTS / molecule / "group_membership.csv"
    terms_by_source_row = {int(term.source_row): term for term in terms}
    term_index_by_source_row = {
        int(term.source_row): index for index, term in enumerate(terms)
    }
    grouped: dict[int, list[Any]] = defaultdict(list)
    seen_in_group: dict[int, set[int]] = defaultdict(set)
    declared_occurrences: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            group_index = int(row["group_index"])
            source_row = int(row["source_csv_row_1based_including_header"])
            source_index = int(row["source_index_zero_based_nonidentity"])
            if source_row not in terms_by_source_row:
                raise RuntimeError(
                    f"{molecule}: membership references unknown source row {source_row}"
                )
            term = terms_by_source_row[source_row]
            if source_index != term_index_by_source_row[source_row]:
                raise RuntimeError(
                    f"{molecule}: source-index mismatch for source row {source_row}: "
                    f"artifact={source_index}, loader={term_index_by_source_row[source_row]}"
                )
            artifact_label = row["pauli_label_site0_to_siteNminus1"]
            if artifact_label != term.label:
                raise RuntimeError(
                    f"{molecule}: label mismatch at source row {source_row}: "
                    f"artifact={artifact_label}, loader={term.label}"
                )
            if abs(float(row["coefficient_hartree"]) - term.coefficient) > 1.0e-14:
                raise RuntimeError(
                    f"{molecule}: coefficient mismatch at source row {source_row}"
                )
            if source_row in seen_in_group[group_index]:
                raise RuntimeError(
                    f"{molecule}: duplicate Pauli source row {source_row} in group {group_index}"
                )
            seen_in_group[group_index].add(source_row)
            grouped[group_index].append(term)
            declared = int(row["occurrence_count"])
            previous = declared_occurrences.setdefault(source_row, declared)
            if previous != declared:
                raise RuntimeError(
                    f"{molecule}: inconsistent occurrence count for source row {source_row}"
                )

    if not grouped:
        raise RuntimeError(f"{molecule}: empty FC-IMA membership artifact {path}")
    expected_group_indices = list(range(max(grouped) + 1))
    if sorted(grouped) != expected_group_indices:
        raise RuntimeError(
            f"{molecule}: non-contiguous FC-IMA group indices {sorted(grouped)}"
        )
    groups = [grouped[index] for index in expected_group_indices]
    actual_occurrences: dict[int, int] = defaultdict(int)
    for group_index, group in enumerate(groups):
        for left_index, left in enumerate(group):
            actual_occurrences[int(left.source_row)] += 1
            for right in group[left_index + 1 :]:
                if not error.fully_commutes(left, right):
                    raise RuntimeError(
                        f"{molecule}: noncommuting membership in FC-IMA group "
                        f"{group_index}: {left.label}, {right.label}"
                    )
    if set(actual_occurrences) != set(terms_by_source_row):
        missing = sorted(set(terms_by_source_row) - set(actual_occurrences))
        extra = sorted(set(actual_occurrences) - set(terms_by_source_row))
        raise RuntimeError(
            f"{molecule}: FC-IMA membership does not cover Hamiltonian; "
            f"missing={missing}, extra={extra}"
        )
    if dict(actual_occurrences) != declared_occurrences:
        raise RuntimeError(
            f"{molecule}: declared and reconstructed occurrence counts differ"
        )
    return groups, dict(actual_occurrences)


def load_frozen_shots(molecule: str, group_count: int) -> np.ndarray:
    path = ERROR_RESULTS / molecule / "shot_allocations.csv"
    selected: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["allocation"] == ALLOCATION
                and int(row["T_total_shots"]) == TOTAL_SHOTS
            ):
                selected[int(row["group_index"])] = int(row["allocated_shots"])
    if set(selected) != set(range(group_count)):
        raise RuntimeError(
            f"{molecule}: frozen shot vector has groups {sorted(selected)}, "
            f"expected 0..{group_count - 1}"
        )
    shots = np.asarray([selected[index] for index in range(group_count)], dtype=np.int64)
    if int(np.sum(shots)) != TOTAL_SHOTS or np.any(shots < 1):
        raise RuntimeError(f"{molecule}: invalid frozen shot vector {shots}")
    return shots


def pooled_shot_counts(
    groups: Sequence[Sequence[Any]], shots: Sequence[int]
) -> dict[int, int]:
    """Return M_k = sum_{alpha containing k} m_alpha from Eq. (14)."""
    if len(groups) != len(shots):
        raise ValueError("Group and shot-vector lengths differ")
    result: dict[int, int] = defaultdict(int)
    for group, shot_count in zip(groups, shots):
        if int(shot_count) <= 0:
            raise ValueError("FC-IMA requires positive m_alpha")
        for term in group:
            result[int(term.source_row)] += int(shot_count)
    return dict(result)


def load_and_validate_pooled_shots(
    molecule: str,
    terms: Sequence[Any],
    reconstructed: dict[int, int],
) -> None:
    """Cross-check the independent Eq. (14) M_k artifact from error_eval."""
    path = ERROR_RESULTS / molecule / "pooled_term_allocations.csv"
    selected: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["allocation"] == ALLOCATION
                and int(row["T_total_shots"]) == TOTAL_SHOTS
            ):
                source_row = int(row["source_csv_row_1based_including_header"])
                selected[source_row] = int(row["total_pooled_shots_Mj"])
    expected_rows = {int(term.source_row) for term in terms}
    if set(selected) != expected_rows:
        raise RuntimeError(
            f"{molecule}: pooled-shot artifact term coverage differs from Hamiltonian"
        )
    if selected != reconstructed:
        differences = {
            source_row: (selected.get(source_row), reconstructed.get(source_row))
            for source_row in sorted(expected_rows)
            if selected.get(source_row) != reconstructed.get(source_row)
        }
        raise RuntimeError(
            f"{molecule}: Eq. (14) pooled-shot cross-check failed: {differences}"
        )


def pooled_ima_group_contribution(
    shot_count: int,
    coefficients: np.ndarray,
    pooled_counts: np.ndarray,
    pauli_means: np.ndarray,
    pauli_pair_moments: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return one basis's exact Eq. (14)/(17) mean and variance contribution.

    The per-shot random variable for basis alpha is
    X_alpha = sum_k c_k P_k / M_k.  ``shot_count`` independent realizations
    contribute m_alpha E[X_alpha] to the Hamiltonian mean and
    m_alpha Var[X_alpha] to its estimator variance.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    pooled_counts = np.asarray(pooled_counts, dtype=float)
    pauli_means = np.asarray(pauli_means, dtype=float)
    pauli_pair_moments = np.asarray(pauli_pair_moments, dtype=float)
    if int(shot_count) <= 0 or np.any(pooled_counts <= 0.0):
        raise ValueError("FC-IMA m_alpha and M_k must be positive")
    size = len(coefficients)
    if pauli_means.shape != (size,) or pauli_pair_moments.shape != (size, size):
        raise ValueError("Inconsistent FC-IMA moment shapes")
    weights = coefficients / pooled_counts
    per_shot_mean = float(weights @ pauli_means)
    per_shot_second = float(weights @ pauli_pair_moments @ weights)
    per_shot_variance = per_shot_second - per_shot_mean * per_shot_mean
    if per_shot_variance < -MOMENT_TOLERANCE:
        raise RuntimeError(f"Negative FC-IMA per-shot variance {per_shot_variance}")
    per_shot_variance = max(per_shot_variance, 0.0)
    return (
        int(shot_count) * per_shot_mean,
        int(shot_count) * per_shot_variance,
        per_shot_mean,
        per_shot_variance,
    )


def case_spec(molecule: str):
    return next(spec for spec in error.CASES if spec.molecule == molecule)


def evaluate_molecule(
    molecule: str,
    p_grid: Sequence[float],
    seed: int,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    spec = case_spec(molecule)
    case_dir = error.SELECTED_ROOT / spec.directory
    pauli_path = case_dir / "hamiltonian_pauli_blocked_spin.csv"
    mps_path = case_dir / "ground_state_mps_blocked_spin.npz"
    metadata_path = case_dir / "metadata.json"
    exact_energy = float(json.loads(metadata_path.read_text(encoding="utf-8"))["fci_ground_energy_hartree"])

    error_terms, identity, pauli_audit = error.load_pauli_terms(pauli_path)
    if not error_terms:
        raise RuntimeError(f"{molecule}: Hamiltonian has no retained nonidentity terms")
    n_qubits = len(error_terms[0].label)
    ima_groups, occurrence_counts = load_ima_groups(molecule, error_terms)
    shots = load_frozen_shots(molecule, len(ima_groups))
    pooled_shots = pooled_shot_counts(ima_groups, shots)
    load_and_validate_pooled_shots(molecule, error_terms, pooled_shots)
    reference_summary_path = ERROR_RESULTS / molecule / "summary.json"
    reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))

    state, state_audit = error.load_dense_mps(mps_path)
    support = np.flatnonzero(np.abs(state) > 0.0).astype(np.uint32)
    parity = error.parity_table(n_qubits)
    group_rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []
    totals: dict[float, dict[str, float]] = {
        float(p): {"mean": identity, "sampling_variance": 0.0} for p in p_grid
    }
    for group_index, moment_group in enumerate(ima_groups):
        group_started = time.perf_counter()
        compiled = build_compiled_group(group_index, moment_group, n_qubits, seed)
        compiled_metrics = {
            "gate_count": int(compiled.optimized.size()),
            "total_depth": int(compiled.optimized.depth() or 0),
            "two_qubit_depth": depth.two_qubit_depth(compiled.optimized),
            "cx_count": compiled.operation_names.count("cx"),
        }
        means, gram = action_gram_matrix(moment_group, state, support, parity)
        coefficients = np.asarray([term.coefficient for term in moment_group], dtype=float)
        group_pooled_counts = np.asarray(
            [pooled_shots[int(term.source_row)] for term in moment_group], dtype=float
        )
        ideal_mean = float(coefficients @ means)
        ideal_second = float(coefficients @ gram @ coefficients)
        ideal_variance = max(ideal_second - ideal_mean * ideal_mean, 0.0)

        coefficient_outer = coefficients[:, None] * coefficients[None, :]
        second_kernel = coefficient_outer * gram
        for p in p_grid:
            p = float(p)
            term_factors = np.power(1.0 - p, compiled.term_exponents)
            pair_factors = np.power(1.0 - p, compiled.pair_exponents)
            noisy_pauli_means = term_factors * means
            noisy_pair_moments = pair_factors * gram
            noisy_mean = float(np.dot(coefficients, noisy_pauli_means))
            noisy_second = float(np.sum(second_kernel * pair_factors))
            raw_variance = noisy_second - noisy_mean * noisy_mean
            if raw_variance < -MOMENT_TOLERANCE:
                raise RuntimeError(
                    f"{molecule} group {group_index} p={p}: negative raw-group "
                    f"variance {raw_variance}"
                )
            noisy_variance = max(raw_variance, 0.0)
            (
                estimator_mean_contribution,
                estimator_variance_contribution,
                estimator_per_shot_mean,
                estimator_per_shot_variance,
            ) = pooled_ima_group_contribution(
                int(shots[group_index]),
                coefficients,
                group_pooled_counts,
                noisy_pauli_means,
                noisy_pair_moments,
            )
            totals[p]["mean"] += estimator_mean_contribution
            totals[p]["sampling_variance"] += estimator_variance_contribution
            group_rows.append(
                {
                    "molecule": molecule,
                    "group_index": group_index,
                    "depolarizing_rate": p,
                    "allocated_shots_T3000": int(shots[group_index]),
                    "allocation": ALLOCATION,
                    "estimator": "Nature-2023 pooled FC-IMA Eqs. (14) and (17)",
                    "term_count": len(moment_group),
                    "unique_hamiltonian_term_count": len(set(term.source_row for term in moment_group)),
                    "pooled_Mj_min": int(np.min(group_pooled_counts)),
                    "pooled_Mj_max": int(np.max(group_pooled_counts)),
                    "compiled_cx_count": compiled.operation_names.count("cx"),
                    "depolarizing_channel_count": compiled.operation_names.count("cx"),
                    "channels_per_compiled_cx": 1,
                    "raw_group_ideal_mean_hartree": ideal_mean,
                    "raw_group_noisy_mean_hartree": noisy_mean,
                    "raw_group_ideal_second_moment_hartree2": ideal_second,
                    "raw_group_noisy_second_moment_hartree2": noisy_second,
                    "raw_group_ideal_variance_hartree2": ideal_variance,
                    "raw_group_noisy_variance_hartree2": noisy_variance,
                    "eq14_per_circuit_estimator_mean_hartree": estimator_per_shot_mean,
                    "eq17_per_circuit_estimator_variance_hartree2": estimator_per_shot_variance,
                    "eq14_hamiltonian_mean_contribution_hartree": estimator_mean_contribution,
                    "eq17_sampling_variance_contribution_hartree2": estimator_variance_contribution,
                    "mean_attenuation_exponent_min": int(np.min(compiled.term_exponents)),
                    "mean_attenuation_exponent_max": int(np.max(compiled.term_exponents)),
                    "second_attenuation_exponent_min": int(np.min(compiled.pair_exponents)),
                    "second_attenuation_exponent_max": int(np.max(compiled.pair_exponents)),
                }
            )
        circuit_rows.append(
            {
                "molecule": molecule,
                "group_index": group_index,
                "term_count": len(moment_group),
                "allocated_shots_T3000": int(shots[group_index]),
                "pooled_Mj_min": int(np.min(group_pooled_counts)),
                "pooled_Mj_max": int(np.max(group_pooled_counts)),
                "compiled_gate_count": compiled_metrics["gate_count"],
                "compiled_total_depth": compiled_metrics["total_depth"],
                "compiled_two_qubit_depth": compiled_metrics["two_qubit_depth"],
                "compiled_cx_count": compiled_metrics["cx_count"],
                "depolarizing_channel_count": compiled_metrics["cx_count"],
                "channels_per_compiled_cx": 1,
                "all_two_qubit_operations_are_explicit_cx": True,
                "unique_operation_names": " ".join(sorted(set(compiled.operation_names))),
                "all_input_symplectic_masks_recovered": True,
                "unique_output_term_masks": len(set(map(int, compiled.output_z_masks))),
                "unique_output_pair_masks": len(
                    set(
                        map(
                            int,
                            np.bitwise_xor(
                                compiled.output_z_masks[:, None],
                                compiled.output_z_masks[None, :],
                            ).reshape(-1),
                        )
                    )
                ),
                "elapsed_seconds": time.perf_counter() - group_started,
            }
        )
        if (group_index + 1) % 10 == 0 or group_index + 1 == len(ima_groups):
            print(
                f"[{molecule}] exact FC-IMA noise moments "
                f"{group_index + 1}/{len(ima_groups)}; "
                f"latest CX={circuit_rows[-1]['compiled_cx_count']}",
                flush=True,
            )

    curve_rows: list[dict[str, Any]] = []
    for p in p_grid:
        p = float(p)
        noisy_mean = totals[p]["mean"]
        bias = noisy_mean - exact_energy
        sampling_variance = totals[p]["sampling_variance"]
        sampling_std = math.sqrt(max(sampling_variance, 0.0))
        rmse = math.sqrt(bias * bias + max(sampling_variance, 0.0))
        curve_rows.append(
            {
                "molecule": molecule,
                "method": "FC-IMA",
                "fc_variant": "overlapping FC-IMA (Nature 2023)",
                "depolarizing_rate": p,
                "T_total_shots": TOTAL_SHOTS,
                "T_actual_shots": int(np.sum(shots)),
                "group_count": len(ima_groups),
                "overlapping_membership_count": int(sum(map(len, ima_groups))),
                "unique_pauli_term_count": len(error_terms),
                "total_pooled_pauli_observations": int(sum(pooled_shots.values())),
                "noisy_energy_mean_hartree": noisy_mean,
                "exact_energy_hartree": exact_energy,
                "noise_bias_hartree": bias,
                "sampling_variance_hartree2": sampling_variance,
                "sampling_std_hartree": sampling_std,
                "total_rmse_hartree": rmse,
                "error_hartree": rmse,
                "noise_model": "two-qubit local depolarizing after every compiled CX",
                "channel": "E_p(rho)=(1-p)rho+(p/4)I_S tensor Tr_S(rho)",
                "state_preparation_noise_included": False,
                "one_qubit_gate_noise_included": False,
                "readout_noise_included": False,
                "decomposition_and_shot_vector_frozen": True,
                "allocation": ALLOCATION,
                "sampling_estimator": "Eq. (14) pooled Pauli means; Eq. (17) covariance",
                "moment_evaluation": "exact noisy Born first/second moments; no sampled trajectories",
            }
        )

    p0 = next(row for row in curve_rows if row["depolarizing_rate"] == 0.0)
    reference_rmse = float(reference_summary["fc_primary_analytic_rmse_hartree"])
    p0_rmse_difference = float(p0["total_rmse_hartree"] - reference_rmse)
    p0_energy_difference = float(p0["noisy_energy_mean_hartree"] - exact_energy)
    regression = {
        "pooled_estimator_mean_minus_exact_energy_hartree": p0_energy_difference,
        "total_rmse_hartree": float(p0["total_rmse_hartree"]),
        "reference_total_rmse_hartree": reference_rmse,
        "total_rmse_difference_hartree": p0_rmse_difference,
        "reference": "error_eval integer FC-IMA Eq. (17) analytic RMSE",
    }
    if (
        abs(p0_energy_difference) > MOMENT_TOLERANCE
        or abs(p0_rmse_difference) > MOMENT_TOLERANCE
    ):
        raise RuntimeError(f"{molecule}: p=0 regression failed: {regression}")

    case_output = output_root / molecule
    write_csv(case_output / "group_noise_moments.csv", group_rows)
    write_csv(case_output / "circuit_audit.csv", circuit_rows)
    summary = {
        "molecule": molecule,
        "method": "overlapping full commuting iterative measurement allocation (FC-IMA; Nature 2023)",
        "method_reference": (
            "Tzu-Ching Yen, Aadithya Ganeshram, and Artur F. Izmaylov, "
            "npj Quantum Information 9, article 14 (2023)"
        ),
        "estimator_equations": "Yen--Ganeshram--Izmaylov 2023 Eqs. (14) and (17)--(22)",
        "number_qubits": n_qubits,
        "group_count": len(ima_groups),
        "overlapping_membership_count": int(sum(map(len, ima_groups))),
        "retained_pauli_terms": len(error_terms),
        "occurrence_count_min": min(occurrence_counts.values()),
        "occurrence_count_max": max(occurrence_counts.values()),
        "identity_offset_hartree": identity,
        "exact_energy_hartree": exact_energy,
        "total_shots": TOTAL_SHOTS,
        "allocation": ALLOCATION,
        "frozen_shot_vector": shots,
        "pooled_term_shots_Mj": pooled_shots,
        "total_pooled_pauli_observations": int(sum(pooled_shots.values())),
        "p_grid": list(map(float, p_grid)),
        "sampling_process": {
            "ideal_protocol": (
                "For each basis alpha, execute m_alpha shots; every compatible Pauli "
                "outcome is pooled across its bases according to Nature-2023 Eq. (14)."
            ),
            "reported_curve": (
                "Exact first and second noisy Born moments of that estimator, aggregated "
                "with Eq. (17); no Gaussian replay and no finite-repeat Monte Carlo."
            ),
            "full_noisy_bitstring_trajectories_generated": False,
            "reason": (
                "An exact noisy Born distribution would require a 4^n density matrix "
                "or an exponentially branching Pauli-channel mixture.  Shot-by-shot "
                "Pauli trajectories are approximate and would require a fresh non-"
                "stabilizer state/circuit evolution for every shot, basis, and p, while "
                "adding seed-dependent Monte Carlo error.  Clifford-Pauli propagation "
                "evaluates the requested estimator mean and RMSE exactly."
            ),
            "exact_density_matrix_complex_entries_4^n": int(4**n_qubits),
            "exact_density_matrix_complex128_bytes": int(16 * 4**n_qubits),
        },
        "noise_model": {
            "channel": "E_{p,S}(rho)=(1-p)rho+(p/4)I_S tensor Tr_S(rho)",
            "location": "after every explicitly compiled CNOT (one channel per CX)",
            "state_preparation": "ideal",
            "one_qubit_gates": "ideal",
            "readout": "ideal",
            "heisenberg_rule": "identity on S: factor 1; otherwise factor (1-p)",
            "moment_evaluation": "exact Clifford-Pauli propagation for lambda and lambda^2 observables",
            "relation_to_nature_2023": (
                "Present-work hardware-noise extension; Nature 2023 defines the ideal "
                "FC-IMA estimator but does not prescribe this gate-noise channel."
            ),
        },
        "p0_regression": regression,
        "curve": curve_rows,
        "input_audit": {
            "files": {
                "pauli_csv": str(pauli_path),
                "pauli_csv_sha256": sha256_file(pauli_path),
                "ground_state_mps": str(mps_path),
                "ground_state_mps_sha256": sha256_file(mps_path),
                "metadata_json": str(metadata_path),
                "metadata_json_sha256": sha256_file(metadata_path),
                "frozen_shot_csv": str(ERROR_RESULTS / molecule / "shot_allocations.csv"),
                "frozen_shot_csv_sha256": sha256_file(
                    ERROR_RESULTS / molecule / "shot_allocations.csv"
                ),
                "group_membership_csv": str(ERROR_RESULTS / molecule / "group_membership.csv"),
                "group_membership_csv_sha256": sha256_file(
                    ERROR_RESULTS / molecule / "group_membership.csv"
                ),
                "pooled_term_allocations_csv": str(
                    ERROR_RESULTS / molecule / "pooled_term_allocations.csv"
                ),
                "pooled_term_allocations_csv_sha256": sha256_file(
                    ERROR_RESULTS / molecule / "pooled_term_allocations.csv"
                ),
            },
            "pauli": pauli_audit,
            "state": state_audit,
            "overlapping_group_membership_validated": True,
            "all_hamiltonian_terms_covered": True,
            "declared_occurrence_counts_reconstructed": True,
            "pooled_Mj_reconstructed_from_group_shots": True,
            "all_compiled_term_symplectic_masks_recovered": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(case_output / "summary.json", summary)
    return curve_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=("BeH2", "N2", "all"), default="all")
    parser.add_argument("--seed", type=int, default=depth.DEFAULT_SEED)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--p-grid",
        type=float,
        nargs="+",
        default=list(P_GRID),
        help="Depolarizing probabilities (default: 10-point linspace from 0 to 0.003).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.p_grid or any(p < 0.0 or p > 1.0 for p in args.p_grid):
        raise ValueError("Every p must lie in [0,1]")
    if 0.0 not in args.p_grid:
        raise ValueError("p-grid must include 0 for the mandatory regression")
    selected = ("BeH2", "N2") if args.molecule == "all" else (args.molecule,)
    all_curve_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for molecule in selected:
        rows, summary = evaluate_molecule(
            molecule, args.p_grid, args.seed, args.output_root
        )
        all_curve_rows.extend(rows)
        summaries.append(summary)
    write_csv(args.output_root / "fc_local_depolarizing_curve.csv", all_curve_rows)
    # Preserve the historical artifact path while keeping the public method
    # value and channel description uniformly FC-IMA.
    write_csv(args.output_root / "cm_local_depolarizing_curve.csv", all_curve_rows)
    manifest = {
        "version": VERSION,
        "method": "FC-IMA",
        "fc_variant": "overlapping FC-IMA (Nature 2023)",
        "method_reference": (
            "Tzu-Ching Yen, Aadithya Ganeshram, and Artur F. Izmaylov, "
            "npj Quantum Information 9, article 14 (2023)"
        ),
        "legacy_directory_note": (
            "The traditional_cm_benchmarks directory name is retained only for "
            "artifact-path compatibility; all public method labels are FC-IMA."
        ),
        "grouping": (
            "Nature-2023 overlapping extension of absolute-coefficient-descending "
            "fully-commuting sorted insertion"
        ),
        "circuit": "Yen sigma-tau-sigma construction; Clifford resynthesis; Qiskit optimization level 1",
        "basis_gates": list(depth.BASIS_GATES),
        "coupling": "logical all-to-all; no routing",
        "total_shots": TOTAL_SHOTS,
        "allocation": ALLOCATION,
        "allocation_method": (
            "Nature-2023 iterative measurement allocation, Eqs. (19)--(22); "
            "positive integer T=3000 vector frozen from error_eval"
        ),
        "estimator": (
            "Pauli outcomes pooled across every compatible basis with Eq. (14); "
            "independent-basis covariance aggregated with Eq. (17)"
        ),
        "p_grid": list(map(float, args.p_grid)),
        "exact_method": (
            "For each output Z Pauli, reverse the compiled Clifford.  Each local "
            "depolarizing channel contributes one factor (1-p) iff that Heisenberg "
            "Pauli is nonidentity on the gate support.  Apply this independently "
            "to each Pauli and Pauli-pair observable, then aggregate the overlapping "
            "FC-IMA estimator exactly with Eqs. (14) and (17)."
        ),
        "trajectory_sampling": (
            "No finite noisy bitstring trajectories: the output is the exact expected "
            "RMSE of the Nature-2023 estimator under the stated present-work channel."
        ),
        "dependencies": {
            "depth_script": str(DEPTH_MODULE_PATH),
            "depth_script_sha256": sha256_file(DEPTH_MODULE_PATH),
            "error_script": str(ERROR_MODULE_PATH),
            "error_script_sha256": sha256_file(ERROR_MODULE_PATH),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "qiskit": qiskit.__version__,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "molecules": [
            {
                "molecule": summary["molecule"],
                "group_count": summary["group_count"],
                "p0_regression": summary["p0_regression"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
            for summary in summaries
        ],
        "total_elapsed_seconds": time.perf_counter() - started,
        "legacy_aliases_refreshed_with_fc_content": [
            str(args.output_root / "cm_local_depolarizing_curve.csv")
        ],
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(f"Wrote exact FC noise audit to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
