#!/usr/bin/env python3
"""Overlapping full-commuting IMA (FC-IMA) depth benchmark.

The measurement settings follow the extended sorted-insertion construction in
Supplementary Note 1 of Yen, Ganeshram, and Izmaylov, npj Quantum
Information 9, 14 (2023):

* coefficient-sorted, overlapping fully-commuting groups;
* the Yen--Verteletskyi--Izmaylov sigma--tau--sigma Clifford construction;
* Qiskit Clifford resynthesis followed by logical all-to-all transpilation;
* verification that every Hamiltonian Pauli occurrence maps to signed Z.

Depth is a property of the overlapping settings and is independent of the
subsequent IMA shot vector.  The reported resource is the maximum logical CNOT
layer depth over all settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, Pauli


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "pauli_product"))
from release_paths import DATA_ROOT, OUTPUT_ROOT
SELECTED_ROOT = DATA_ROOT

MOLECULES = {
    "BeH2": SELECTED_ROOT / "BeH2" / "inputs",
    "N2": SELECTED_ROOT / "N2" / "inputs",
}

CSV_NAME = "hamiltonian_pauli_blocked_spin.csv"
BASIS_GATES = ("rz", "sx", "x", "cx")
DEFAULT_SEED = 20260828
COEFFICIENT_IMAG_TOL = 1.0e-12
COEFFICIENT_ZERO_TOL = 1.0e-15

# Exact private-source blobs used as the reproduction target.
PRIVATE_SOURCE = {
    "repository": "Fragecity/CommuteMeasurement",
    "branch": "depth",
    "cm.py_blob": "4de498010e4a7309b189bbf7978c17f86d718d10",
    "grouping.py_blob": "35b43eb4c2fd2ad0a7cf2fa9f9c5398da4fafd14",
    "find_basis.py_blob": "ca8769252651dade74f7deaa1b7fdcdfdfc1f859",
}

OVERLAP_SOURCE = {
    "article": "Yen, Ganeshram, and Izmaylov, npj Quantum Information 9, 14 (2023)",
    "doi": "10.1038/s41534-023-00683-y",
    "algorithm": "Supplementary Note 1 extended sorted insertion, order=pi",
    "reference_code": "RMeasurementAnsatz/ToBW/utils/grouping_utils.py",
}


@dataclass(frozen=True)
class HamiltonianTerm:
    source_index: int
    label: str
    coefficient: float
    x_mask: int
    z_mask: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_to_masks(label: str) -> tuple[int, int]:
    """Return X/Z masks; bit i corresponds to label character i."""
    x_mask = 0
    z_mask = 0
    for i, char in enumerate(label):
        bit = 1 << i
        if char in "XY":
            x_mask |= bit
        if char in "YZ":
            z_mask |= bit
    return x_mask, z_mask


def label_to_vec(label: str) -> np.ndarray:
    """Private-repository convention: [x(label order), z(label order)]."""
    x_mask, z_mask = label_to_masks(label)
    n_qubits = len(label)
    x = np.fromiter(((x_mask >> i) & 1 for i in range(n_qubits)), np.uint8)
    z = np.fromiter(((z_mask >> i) & 1 for i in range(n_qubits)), np.uint8)
    return np.concatenate((x, z))


def vec_to_label(vec: np.ndarray) -> str:
    n_qubits = len(vec) // 2
    chars: list[str] = []
    for x, z in zip(vec[:n_qubits], vec[n_qubits:]):
        chars.append({(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}[(int(x), int(z))])
    return "".join(chars)


def vec_to_int(vec: np.ndarray) -> int:
    result = 0
    for i, value in enumerate(vec):
        if int(value):
            result |= 1 << i
    return result


def symplectic_product_masks(
    x1: int, z1: int, x2: int, z2: int
) -> int:
    return ((x1 & z2) ^ (z1 & x2)).bit_count() & 1


def symplectic_product_vec(vec1: np.ndarray, vec2: np.ndarray) -> int:
    n_qubits = len(vec1) // 2
    x1, z1 = vec1[:n_qubits], vec1[n_qubits:]
    x2, z2 = vec2[:n_qubits], vec2[n_qubits:]
    return int((np.dot(x1, z2) + np.dot(z1, x2)) & 1)


def terms_commute(first: HamiltonianTerm, second: HamiltonianTerm) -> bool:
    return symplectic_product_masks(
        first.x_mask, first.z_mask, second.x_mask, second.z_mask
    ) == 0


def read_hamiltonian(
    path: Path,
) -> tuple[list[HamiltonianTerm], float, int, dict[str, float | int]]:
    """Read the Hermitian (real-Pauli) Hamiltonian from the exported CSV.

    The N2 export also contains 90 purely imaginary, O(1e-8) numerical
    artefacts.  A Hermitian operator has real coefficients in the Hermitian
    Pauli basis, so these zero-real rows are excluded and documented.
    """
    terms: list[HamiltonianTerm] = []
    identity_offset = 0.0
    n_qubits: int | None = None
    dropped_pure_imaginary = 0
    max_dropped_imaginary = 0.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "full_label_site0_to_siteNminus1",
            "coefficient_real_hartree",
            "coefficient_imag_hartree",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for source_index, row in enumerate(reader):
            label = row["full_label_site0_to_siteNminus1"].strip().upper()
            real = float(row["coefficient_real_hartree"])
            imag = float(row["coefficient_imag_hartree"])
            if n_qubits is None:
                n_qubits = len(label)
            if len(label) != n_qubits or any(char not in "IXYZ" for char in label):
                raise ValueError(f"Invalid Pauli label at row {source_index}: {label!r}")
            if abs(imag) > COEFFICIENT_IMAG_TOL:
                if abs(real) > COEFFICIENT_ZERO_TOL:
                    raise ValueError(
                        "Pauli row has both nonzero real and imaginary coefficients "
                        f"at row {source_index}: {real} + {imag}i"
                    )
                dropped_pure_imaginary += 1
                max_dropped_imaginary = max(max_dropped_imaginary, abs(imag))
                continue
            if abs(real) <= COEFFICIENT_ZERO_TOL:
                continue
            if set(label) == {"I"}:
                identity_offset += real
                continue
            x_mask, z_mask = label_to_masks(label)
            terms.append(
                HamiltonianTerm(source_index, label, real, x_mask, z_mask)
            )
    if n_qubits is None:
        raise ValueError(f"Hamiltonian CSV is empty: {path}")
    projection = {
        "dropped_pure_imaginary_term_count": dropped_pure_imaginary,
        "max_abs_dropped_imaginary_coefficient_hartree": max_dropped_imaginary,
        "imaginary_detection_tolerance_hartree": COEFFICIENT_IMAG_TOL,
    }
    return terms, identity_offset, n_qubits, projection


def extended_sorted_insertion_fc_groups(
    terms: Sequence[HamiltonianTerm],
) -> tuple[list[list[HamiltonianTerm]], list[set[int]]]:
    """Build the paper's overlapping FC groups with ``order='pi'``.

    At each iteration the unassigned terms form the ordinary SI seed group.
    Previously assigned terms are then scanned in their original insertion
    order and copied into the new group whenever they commute with every term
    already present.  ``seed_rows`` proves that the non-overlap seeds partition
    the Hamiltonian even though the final groups contain repeated occurrences.
    """
    remaining = sorted(
        terms, key=lambda term: (-abs(term.coefficient), term.source_index)
    )
    previously_grouped: list[HamiltonianTerm] = []
    groups: list[list[HamiltonianTerm]] = []
    seed_rows: list[set[int]] = []
    while remaining:
        group: list[HamiltonianTerm] = []
        selected_indices: list[int] = []
        for index, term in enumerate(remaining):
            if all(terms_commute(term, existing) for existing in group):
                group.append(term)
                selected_indices.append(index)
        seeds = {remaining[index].source_index for index in selected_indices}
        for term in previously_grouped:
            if all(terms_commute(term, existing) for existing in group):
                group.append(term)
        groups.append(group)
        seed_rows.append(seeds)

        # The paper appends newly assigned terms to P* in the same order in
        # which they entered the current group.  Do not pop in reverse order:
        # that historical ToBW implementation detail changes later overlaps.
        selected_terms = [remaining[index] for index in selected_indices]
        selected_set = set(selected_indices)
        remaining = [
            term for index, term in enumerate(remaining) if index not in selected_set
        ]
        previously_grouped.extend(selected_terms)
        if len(groups) % 25 == 0 or not remaining:
            print(
                f"  grouping: {len(groups)} groups formed; "
                f"{len(remaining)} terms remain",
                flush=True,
            )

    expected = {term.source_index for term in terms}
    seeded = set().union(*seed_rows)
    if seeded != expected or sum(map(len, seed_rows)) != len(terms):
        raise AssertionError("Extended-SI seeds did not partition the Hamiltonian")
    for group_id, group in enumerate(groups):
        if len({term.source_index for term in group}) != len(group):
            raise AssertionError(f"Group {group_id} contains a duplicate occurrence")
        for i, first in enumerate(group):
            for second in group[i + 1 :]:
                if not terms_commute(first, second):
                    raise AssertionError(f"Group {group_id} is not fully commuting")
    return groups, seed_rows


def column_echelon(matrix: np.ndarray) -> np.ndarray:
    """Exact port of the private branch's GF(2) column-echelon helper."""
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.uint8)
    work = matrix.copy().astype(np.uint8)
    n_rows, n_cols = work.shape
    pivot_rows: list[int] = []
    independent_cols: list[int] = []
    for col in range(n_cols):
        pivot = next(
            (row for row in range(n_rows) if row not in pivot_rows and work[row, col]),
            None,
        )
        if pivot is None:
            continue
        for other_col in range(n_cols):
            if other_col != col and work[pivot, other_col]:
                work[:, other_col] ^= work[:, col]
        pivot_rows.append(pivot)
        independent_cols.append(col)
    if not independent_cols:
        return np.zeros((n_rows, 0), dtype=np.uint8)
    return work[:, independent_cols]


def kernel_basis(matrix: np.ndarray) -> np.ndarray:
    """Exact port of the private branch's GF(2) kernel helper."""
    m, n = matrix.shape
    augmented = np.vstack((matrix, np.eye(n, dtype=np.uint8)))
    pivots: set[int] = set()
    for row in range(m):
        pivot_col = next(
            (col for col in range(n) if col not in pivots and augmented[row, col]),
            None,
        )
        if pivot_col is not None:
            pivots.add(pivot_col)
            for col in range(n):
                if col != pivot_col and augmented[row, col]:
                    augmented[:, col] ^= augmented[:, pivot_col]
    kernel_cols = [col for col in range(n) if col not in pivots]
    return augmented[m:, kernel_cols]


def complete_commuting_basis(labels: Sequence[str]) -> list[str]:
    """Port of private ``find_basis``: retain inputs then complete to n."""
    if not labels:
        return []
    n_qubits = len(labels[0])
    input_vecs = [label_to_vec(label) for label in labels]
    current: list[np.ndarray] = []
    for vec in input_vecs:
        if len(current) >= n_qubits:
            break
        if all(symplectic_product_vec(vec, basis_vec) == 0 for basis_vec in current):
            if not current:
                if np.any(vec):
                    current.append(vec)
            else:
                test = np.column_stack(current + [vec])
                if column_echelon(test).shape[1] > len(current):
                    current.append(vec)
    actual_rank = len(current)
    while len(current) < n_qubits:
        constraint = np.zeros((len(current), 2 * n_qubits), dtype=np.uint8)
        for row, basis_vec in enumerate(current):
            constraint[row, :n_qubits] = basis_vec[n_qubits:]
            constraint[row, n_qubits:] = basis_vec[:n_qubits]
        candidates = kernel_basis(constraint)
        found = False
        for col in range(candidates.shape[1]):
            candidate = candidates[:, col]
            test = np.column_stack(current + [candidate])
            if column_echelon(test).shape[1] > len(current):
                current.append(candidate)
                found = True
                break
        if not found:
            raise RuntimeError(
                f"Could not complete commuting basis (rank {actual_rank}, size {len(current)})"
            )
    for i, first in enumerate(current):
        for second in current[i + 1 :]:
            if symplectic_product_vec(first, second):
                raise AssertionError("Completed measurement basis is not commuting")
    if column_echelon(np.column_stack(current)).shape[1] != n_qubits:
        raise AssertionError("Completed measurement basis is not full rank")
    return [vec_to_label(vec) for vec in current]


class GeneratorSolver:
    """GF(2) span solver that also returns generator-exponent masks."""

    def __init__(self) -> None:
        self.entries: dict[int, tuple[int, int]] = {}
        self.generator_count = 0

    def reduce(self, vector: int, combination: int = 0) -> tuple[int, int]:
        for pivot in sorted(self.entries, reverse=True):
            if (vector >> pivot) & 1:
                basis_vector, basis_combination = self.entries[pivot]
                vector ^= basis_vector
                combination ^= basis_combination
        return vector, combination

    def add_generator(self, vector: int) -> bool:
        reduced, combination = self.reduce(vector, 1 << self.generator_count)
        if reduced == 0:
            return False
        pivot = reduced.bit_length() - 1
        self.entries[pivot] = (reduced, combination)
        self.generator_count += 1
        return True

    def solve(self, vector: int) -> int:
        reduced, combination = self.reduce(vector)
        if reduced:
            raise ValueError("Target Pauli is not in the independent-generator span")
        return combination


def independent_generators_and_exponents(
    group: Sequence[HamiltonianTerm], n_qubits: int
) -> tuple[list[str], list[int]]:
    solver = GeneratorSolver()
    generators: list[str] = []
    for term in group:
        vec_int = term.x_mask | (term.z_mask << n_qubits)
        if solver.add_generator(vec_int):
            generators.append(term.label)
    exponents = [
        solver.solve(term.x_mask | (term.z_mask << n_qubits)) for term in group
    ]
    return generators, exponents


def tau_sigma(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Port of private ``TauSigma``; returns adjusted Tau and Sigma."""
    tau = matrix.copy().astype(np.uint8)
    sigma = np.zeros_like(tau)
    n_qubits = tau.shape[0] // 2
    n_cols = tau.shape[1]
    for target in range(n_cols):
        found = False
        for qubit in range(n_qubits):
            x = int(tau[qubit, target])
            z = int(tau[qubit + n_qubits, target])
            if not (x or z):
                continue
            if sigma[qubit, :target].any() or sigma[qubit + n_qubits, :target].any():
                continue
            if (x, z) == (1, 0):
                sigma[qubit + n_qubits, target] = 1
            else:  # Z or Y is paired with X.
                sigma[qubit, target] = 1
            found = True
            break
        if not found:
            raise RuntimeError(f"TauSigma could not choose sigma column {target}")
        for previous in range(target):
            if symplectic_product_vec(sigma[:, target], tau[:, previous]):
                sigma[:, target] ^= sigma[:, previous]
        for later in range(target + 1, n_cols):
            if symplectic_product_vec(sigma[:, target], tau[:, later]):
                tau[:, later] ^= tau[:, target]
    for i in range(n_cols):
        for j in range(n_cols):
            expected = int(i == j)
            actual = symplectic_product_vec(sigma[:, i], tau[:, j])
            if actual != expected:
                raise AssertionError(
                    f"TauSigma duality failure at ({i}, {j}): {actual} != {expected}"
                )
            if symplectic_product_vec(tau[:, i], tau[:, j]):
                raise AssertionError("Tau columns do not commute")
            if symplectic_product_vec(sigma[:, i], sigma[:, j]):
                raise AssertionError("Sigma columns do not commute")
    return tau, sigma


def append_pauli_exponential(circuit: QuantumCircuit, label: str) -> None:
    """Append exp(+i*pi*P/4), matching private ``constructVi``."""
    n_qubits = len(label)
    active: list[int] = []
    for index, char in enumerate(label):
        qubit = n_qubits - 1 - index
        if char == "I":
            continue
        active.append(qubit)
        if char == "X":
            circuit.h(qubit)
        elif char == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)
    if not active:
        return
    target = active[-1]
    controls = active[:-1]
    for control in controls:
        circuit.cx(control, target)
    circuit.sdg(target)
    for control in reversed(controls):
        circuit.cx(control, target)
    for index, char in enumerate(label):
        qubit = n_qubits - 1 - index
        if char == "X":
            circuit.h(qubit)
        elif char == "Y":
            circuit.h(qubit)
            circuit.s(qubit)


def build_expanded_yen_circuit(complete_basis: Sequence[str]) -> QuantumCircuit:
    n_qubits = len(complete_basis[0])
    matrix = np.column_stack([label_to_vec(label) for label in complete_basis])
    tau, sigma = tau_sigma(matrix)
    circuit = QuantumCircuit(n_qubits)
    for column in range(matrix.shape[1]):
        tau_label = vec_to_label(tau[:, column])
        sigma_label = vec_to_label(sigma[:, column])
        # Right-to-left operator order: sigma, tau, sigma in circuit time.
        append_pauli_exponential(circuit, sigma_label)
        append_pauli_exponential(circuit, tau_label)
        append_pauli_exponential(circuit, sigma_label)
    return circuit


def pauli_axis(pauli: Pauli, qubit: int) -> str:
    x = bool(pauli.x[qubit])
    z = bool(pauli.z[qubit])
    return {(False, False): "I", (True, False): "X", (True, True): "Y", (False, True): "Z"}[(x, z)]


def local_qwc_diagonalizer(
    core_clifford: Clifford, complete_basis: Sequence[str]
) -> QuantumCircuit:
    """Rotate the QWC Sigma image to Z; this adds no two-qubit gates."""
    n_qubits = len(complete_basis[0])
    images = [Pauli(label).evolve(core_clifford, frame="s") for label in complete_basis]
    axes: list[str] = []
    for qubit in range(n_qubits):
        nonidentity = {pauli_axis(image, qubit) for image in images} - {"I"}
        if len(nonidentity) > 1:
            raise AssertionError(
                f"TauSigma image is not qubit-wise commuting at qubit {qubit}: {nonidentity}"
            )
        axes.append(next(iter(nonidentity), "I"))
    local = QuantumCircuit(n_qubits)
    for qubit, axis in enumerate(axes):
        if axis == "X":
            local.h(qubit)
        elif axis == "Y":
            local.sdg(qubit)
            local.h(qubit)
    return local


def two_qubit_depth(circuit: QuantumCircuit) -> int:
    value = circuit.depth(
        filter_function=lambda instruction: instruction.operation.num_qubits == 2
    )
    return int(value or 0)


def two_qubit_count(circuit: QuantumCircuit) -> int:
    return sum(
        1 for instruction in circuit.data if instruction.operation.num_qubits == 2
    )


def strip_pauli_prefix(label: str) -> tuple[int, str]:
    if label.startswith("-i") or label.startswith("i"):
        raise AssertionError(f"Hermitian Pauli acquired an imaginary phase: {label}")
    if label.startswith("-"):
        return -1, label[1:]
    return 1, label


def qiskit_z_mask(pauli: Pauli) -> int:
    mask = 0
    for qubit, value in enumerate(pauli.z):
        if bool(value):
            mask |= 1 << qubit
    return mask


def circuit_metrics(circuit: QuantumCircuit) -> dict[str, int]:
    return {
        "total_depth": int(circuit.depth() or 0),
        "two_qubit_depth": two_qubit_depth(circuit),
        "two_qubit_count": two_qubit_count(circuit),
        "gate_count": int(circuit.size()),
    }


def synthesize_group(
    group: Sequence[HamiltonianTerm],
    n_qubits: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    actual_generators, exponents = independent_generators_and_exponents(group, n_qubits)
    complete_basis = complete_commuting_basis([term.label for term in group])
    expanded_core = build_expanded_yen_circuit(complete_basis)
    core_clifford = Clifford(expanded_core)
    local = local_qwc_diagonalizer(core_clifford, complete_basis)

    expanded = expanded_core.copy()
    expanded.compose(local, inplace=True)
    final_clifford = Clifford(expanded)

    # This mirrors the private depth branch: ``cm(...).to_circuit()`` first,
    # then Qiskit's optimizer.  The local suffix is one-qubit only.
    raw_clifford = core_clifford.to_circuit()
    raw_clifford.compose(local, inplace=True)
    if Clifford(raw_clifford) != final_clifford:
        raise AssertionError("Clifford resynthesis changed the FC unitary")
    optimized = transpile(
        raw_clifford,
        basis_gates=list(BASIS_GATES),
        optimization_level=1,
        seed_transpiler=seed,
    )
    if Clifford(optimized) != final_clifford:
        raise AssertionError("Qiskit transpilation changed the FC Clifford")

    transformed_rows: list[dict[str, object]] = []
    for term, exponent_mask in zip(group, exponents):
        transformed = Pauli(term.label).evolve(final_clifford, frame="s")
        if np.any(transformed.x):
            raise AssertionError(
                f"FC failed to diagonalize {term.label}: {transformed.to_label()}"
            )
        sign, unsigned_label = strip_pauli_prefix(transformed.to_label())
        label_order_mask = label_to_masks(unsigned_label)[1]
        transformed_rows.append(
            {
                "source_index": term.source_index,
                "pauli": term.label,
                "coefficient_hartree": term.coefficient,
                "generator_exponents_hex": hex(exponent_mask),
                "generator_exponents_bits_lsb_generator0": format(
                    exponent_mask, f"0{len(actual_generators)}b"
                )[::-1],
                "transformed_sign": sign,
                "transformed_z_label": unsigned_label,
                "transformed_z_mask_label_order_hex": hex(label_order_mask),
                "transformed_z_mask_qiskit_order_hex": hex(qiskit_z_mask(transformed)),
            }
        )

    expanded_metrics = circuit_metrics(expanded)
    raw_metrics = circuit_metrics(raw_clifford)
    optimized_metrics = circuit_metrics(optimized)
    group_row: dict[str, object] = {
        "group_size": len(group),
        "independent_rank": len(actual_generators),
        "independent_generators": " ".join(actual_generators),
        "complete_basis": " ".join(complete_basis),
        "expanded_total_depth": expanded_metrics["total_depth"],
        "expanded_two_qubit_depth": expanded_metrics["two_qubit_depth"],
        "expanded_two_qubit_count": expanded_metrics["two_qubit_count"],
        "expanded_gate_count": expanded_metrics["gate_count"],
        "raw_clifford_total_depth": raw_metrics["total_depth"],
        "raw_clifford_two_qubit_depth": raw_metrics["two_qubit_depth"],
        "raw_clifford_two_qubit_count": raw_metrics["two_qubit_count"],
        "raw_clifford_gate_count": raw_metrics["gate_count"],
        "optimized_total_depth": optimized_metrics["total_depth"],
        # Because ``cx`` is the only two-qubit member of BASIS_GATES, these
        # aliases are the publication CNOT metrics.  The historical
        # ``optimized_two_qubit_*`` fields are retained for artifact replay.
        "optimized_cnot_depth": optimized_metrics["two_qubit_depth"],
        "optimized_cnot_count": optimized_metrics["two_qubit_count"],
        "optimized_two_qubit_depth": optimized_metrics["two_qubit_depth"],
        "optimized_two_qubit_count": optimized_metrics["two_qubit_count"],
        "optimized_gate_count": optimized_metrics["gate_count"],
        "all_terms_z_only": True,
    }
    return group_row, transformed_rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark_molecule(
    molecule: str,
    source_dir: Path,
    output_dir: Path,
    seed: int,
    max_groups: int | None,
) -> dict[str, object]:
    source_csv = source_dir / CSV_NAME
    print(f"[{molecule}] reading {source_csv}", flush=True)
    terms, identity_offset, n_qubits, hermitian_projection = read_hamiltonian(source_csv)
    started = time.perf_counter()
    groups, seed_rows = extended_sorted_insertion_fc_groups(terms)
    grouping_seconds = time.perf_counter() - started
    if max_groups is not None:
        groups_to_synthesize = groups[:max_groups]
        partial = len(groups_to_synthesize) != len(groups)
    else:
        groups_to_synthesize = groups
        partial = False

    group_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    synthesis_started = time.perf_counter()
    for group_id, group in enumerate(groups_to_synthesize):
        group_started = time.perf_counter()
        group_row, transformed_rows = synthesize_group(group, n_qubits, seed)
        group_row = {
            "molecule": molecule,
            "group_id": group_id,
            **group_row,
            "synthesis_seconds": time.perf_counter() - group_started,
        }
        group_rows.append(group_row)
        term_rows.extend(
            {
                "molecule": molecule,
                "group_id": group_id,
                "is_seed_occurrence": int(
                    int(row["source_index"]) in seed_rows[group_id]
                ),
                **row,
            }
            for row in transformed_rows
        )
        if (group_id + 1) % 10 == 0 or group_id + 1 == len(groups_to_synthesize):
            print(
                f"  synthesis: {group_id + 1}/{len(groups_to_synthesize)} groups; "
                f"latest optimized CNOT depth={group_row['optimized_cnot_depth']}",
                flush=True,
            )
    synthesis_seconds = time.perf_counter() - synthesis_started

    suffix = "_partial" if partial else ""
    groups_path = output_dir / f"{molecule}_cm_groups{suffix}.csv"
    terms_path = output_dir / f"{molecule}_cm_terms{suffix}.csv"
    write_csv(groups_path, group_rows)
    write_csv(terms_path, term_rows)

    optimized_depths = [int(row["optimized_two_qubit_depth"]) for row in group_rows]
    raw_depths = [int(row["raw_clifford_two_qubit_depth"]) for row in group_rows]
    expanded_depths = [int(row["expanded_two_qubit_depth"]) for row in group_rows]
    memberships: dict[int, int] = {term.source_index: 0 for term in terms}
    for group in groups:
        for term in group:
            memberships[term.source_index] += 1
    occurrence_count = sum(map(len, groups))
    summary: dict[str, object] = {
        "molecule": molecule,
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "n_qubits": n_qubits,
        "identity_offset_hartree": identity_offset,
        "hermitian_projection": hermitian_projection,
        "nonidentity_term_count": len(terms),
        "group_count": len(groups),
        "overlapping_term_occurrence_count": occurrence_count,
        "overlap_extra_occurrence_count": occurrence_count - len(terms),
        "minimum_groups_per_pauli": min(memberships.values()),
        "maximum_groups_per_pauli": max(memberships.values()),
        "mean_groups_per_pauli": float(np.mean(list(memberships.values()))),
        "synthesized_group_count": len(groups_to_synthesize),
        "partial": partial,
        "max_optimized_two_qubit_depth": max(optimized_depths),
        "max_optimized_cnot_depth": max(optimized_depths),
        "argmax_optimized_cnot_depth_group": int(np.argmax(optimized_depths)),
        "mean_optimized_cnot_depth": float(np.mean(optimized_depths)),
        "argmax_optimized_two_qubit_depth_group": int(np.argmax(optimized_depths)),
        "mean_optimized_two_qubit_depth": float(np.mean(optimized_depths)),
        "max_raw_clifford_two_qubit_depth": max(raw_depths),
        "argmax_raw_clifford_two_qubit_depth_group": int(np.argmax(raw_depths)),
        "mean_raw_clifford_two_qubit_depth": float(np.mean(raw_depths)),
        "max_expanded_two_qubit_depth": max(expanded_depths),
        "argmax_expanded_two_qubit_depth_group": int(np.argmax(expanded_depths)),
        "mean_expanded_two_qubit_depth": float(np.mean(expanded_depths)),
        "all_groups_fully_commuting": True,
        "overlapping_groups": True,
        "all_terms_covered": set(memberships) == {term.source_index for term in terms},
        "ordinary_si_seed_occurrences_partition_terms_once": (
            sum(map(len, seed_rows)) == len(terms)
        ),
        "all_synthesized_terms_z_only": all(
            bool(row["all_terms_z_only"]) for row in group_rows
        ),
        "grouping_seconds": grouping_seconds,
        "synthesis_seconds": synthesis_seconds,
        "transpiler": {
            "optimization_level": 1,
            "seed_transpiler": seed,
            "basis_gates": list(BASIS_GATES),
            "coupling": "logical all-to-all (no coupling_map; no routing)",
            "reported_depth": (
                "maximum logical CNOT/CX layer depth across overlapping fully "
                "commuting settings; cx is the only compiled two-qubit basis gate"
            ),
        },
        "private_source": PRIVATE_SOURCE,
        "overlap_source": OVERLAP_SOURCE,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": {"groups_csv": str(groups_path), "terms_csv": str(terms_path)},
    }
    summary_path = output_dir / f"{molecule}_cm_depth_summary{suffix}.json"
    summary["outputs"]["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{molecule}] {len(groups)} groups; max optimized logical CNOT depth "
        f"{summary['max_optimized_cnot_depth']} "
        f"(group {summary['argmax_optimized_cnot_depth_group']})",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecule",
        choices=("BeH2", "N2", "all"),
        default="all",
        help="Molecule to benchmark (default: all).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / "fully_commuting" / "depth", help="Output directory."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Smoke-test only: synthesize the first N groups and mark outputs partial.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_groups is not None and args.max_groups <= 0:
        raise ValueError("--max-groups must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = MOLECULES if args.molecule == "all" else {args.molecule: MOLECULES[args.molecule]}
    summaries = [
        benchmark_molecule(
            molecule,
            source_dir,
            args.output_dir,
            args.seed,
            args.max_groups,
        )
        for molecule, source_dir in selected.items()
    ]
    combined_path = args.output_dir / (
        "cm_depth_summary_partial.json" if args.max_groups is not None else "cm_depth_summary.json"
    )
    combined = {
        "method": (
            "overlapping full commuting with iterative measurement allocation "
            "(FC-IMA; Yen, Ganeshram, and Izmaylov, npj Quantum Information "
            "2023), with Yen et al. "
            "JCTC 2020 diagonalizing unitaries"
        ),
        "grouping": (
            "absolute-coefficient descending extended sorted insertion with "
            "overlap, Supplementary Note 1 order=pi"
        ),
        "private_source": PRIVATE_SOURCE,
        "overlap_source": OVERLAP_SOURCE,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "molecules": summaries,
    }
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Wrote combined summary: {combined_path}", flush=True)


if __name__ == "__main__":
    main()
