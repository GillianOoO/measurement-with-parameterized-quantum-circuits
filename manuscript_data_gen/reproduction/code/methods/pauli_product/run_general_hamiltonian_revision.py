#!/usr/bin/env python3
"""Recompute the revised general-Hamiltonian error panels.

This is a self-contained, auditable runner for the three nonmolecular panels:

* the eight-qubit open-boundary TFIM at J=g=1;
* five seed-0 sparse four-qubit random matrices; and
* five seed-0 dense four-qubit random matrices.

The retained Pauli baselines are evaluated with the estimator used by the
archived code.  LCS is integrated analytically.  SG and Derand use one fixed,
pre-registered prefix schedule.  OGM uses the archived probability objective
and its pre-registered finite-T integer allocation.  AP averages the exact
conditional MSE over 50 pre-registered schedules.  Born noise is integrated
exactly rather than estimated with a finite Monte Carlo sample.

AGPD and SRDD are new matrix-level computations.  They never read the old
GPD/TND curves.  AGPD greedily chooses a residual projection from four
families.  The SRDD matrix extension adds fresh deterministic multistarts
at every rank and simultaneously refits all retained full diagonals through
centered ridge normal equations.  Neither method uses a small prescribed
rank ceiling: K is continued until all fixed-shot losses have stabilized, and
K=500 is only a fail-closed infinite-loop guard.  This matrix extension is not
the molecular RC-DF tensor method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import pickle
import platform
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from release_paths import BUNDLE_ROOT, DATA_ROOT, OUTPUT_ROOT
RMA = BUNDLE_ROOT
WORKSPACE = BUNDLE_ROOT
KONG = DATA_ROOT
NEW_DATA = HERE
SHADOW = HERE
RANDOM_DATA = DATA_ROOT / "random_sparse_dense" / "inputs"
OUTPUT = OUTPUT_ROOT / "general_hamiltonian"
SUMMARY_CSV = OUTPUT / "general_hamiltonian_error_summary.csv"
INSTANCE_CSV = OUTPUT / "instance_error_rows.csv"
APPROX_CSV = OUTPUT / "approximation_frontier_audit.csv"
PAULI_CSV = OUTPUT / "pauli_schedule_audit.csv"
MANIFEST = OUTPUT / "manifest.json"
SCHEDULE_DIR = OUTPUT / "schedule_cache"
APPROX_CACHE_DIR = OUTPUT / "approximation_cache"

PROFILE_VERSION = "general-hamiltonian-revision-v1.4"
STRUCTURED_SHOTS = (12, 45, 160, 572, 2038, 7259, 25848, 92041)
RANDOM_SHOTS = (12, 45, 160, 572, 2038, 7259, 25848)
SRDD_METHOD = "SRDD"
METHOD_ORDER = ("AGPD", SRDD_METHOD, "OGM", "SG", "Derand", "LCS", "AP")
BENCHMARK_ORDER = ("structured", "sparse", "dense")
BASE_SEED = 20260826
AP_REPEATS = 50
AGPD_FROZEN_STARTS = 10
GENERAL_K_SAFETY_GUARD = 500
FRONTIER_STALL_PATIENCE = 5
FRONTIER_RELATIVE_IMPROVEMENT = 2.0e-3
FRONTIER_MINIMUM_K = 8
AGPD_DEPTHS = tuple(range(1, 8))
SRCDF_DEPTHS = (1, 2, 3)
SRCDF_STARTS_PER_FACTOR = 6
SRCDF_RIDGE = 1.0e-6
SRCDF_JOINT_SOLVER_VERSION = "rankwise-dense-centered-ridge-normal-equations-v2"
RANGE_TOL = 1.0e-12
HERMITIAN_TOL = 3.0e-9

sys.path.insert(0, str(NEW_DATA))
import count_measurement_settings_n2038 as measurement_tools  # noqa: E402


PAULI = (
    np.eye(2, dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)
S_DAGGER = np.diag([1.0, -1j]).astype(np.complex128)
MEASUREMENT_ROTATION = {
    0: np.eye(2, dtype=np.complex128),
    1: HADAMARD,
    2: HADAMARD @ S_DAGGER,
    3: np.eye(2, dtype=np.complex128),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(item) for item in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def real_scalar(value: complex, *, name: str, tolerance: float = 2.0e-8) -> float:
    scalar = complex(value)
    if abs(scalar.imag) > tolerance:
        raise ValueError(f"{name} has a material imaginary part: {scalar}")
    return float(scalar.real)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_pauli_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.atleast_2d(np.loadtxt(path, dtype=float))
    if not np.all(np.isfinite(table)):
        raise ValueError(f"Nonfinite Pauli table entry in {path}")
    weights = table[:, 0].astype(float)
    raw_observables = table[:, 1:]
    if not np.allclose(raw_observables, np.rint(raw_observables), atol=0.0, rtol=0.0):
        raise ValueError(f"Nonintegral Pauli code in {path}")
    observables = np.rint(raw_observables).astype(np.int8)
    if not np.all(np.isin(observables, (0, 1, 2, 3))):
        raise ValueError(f"Invalid Pauli code in {path}")
    keys = [tuple(int(item) for item in row) for row in observables]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate Pauli strings in {path}")
    return observables, weights


def pauli_dense(observables: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = observables.shape[1]
    output = np.zeros((2**n, 2**n), dtype=np.complex128)
    for code, coefficient in zip(observables, weights):
        term = np.array([[1.0]], dtype=np.complex128)
        for item in code:
            term = np.kron(term, PAULI[int(item)])
        output += float(coefficient) * term
    return (output + output.conj().T) / 2.0


def pure_state_from_saved(path: Path) -> np.ndarray:
    saved = np.asarray(np.load(path), dtype=np.complex128)
    if saved.ndim == 1:
        state = saved.copy()
    elif saved.ndim == 2 and saved.shape[0] == saved.shape[1]:
        anti = float(np.linalg.norm(saved - saved.conj().T))
        if anti > 2.0e-10:
            raise ValueError(f"Saved density matrix is not Hermitian: {anti}, {path}")
        density = (saved + saved.conj().T) / 2.0
        trace = real_scalar(np.trace(density), name=f"trace({path})")
        if not np.isclose(trace, 1.0, atol=2.0e-10):
            raise ValueError(f"Saved density matrix trace is {trace}: {path}")
        values, vectors = np.linalg.eigh(density)
        if float(np.min(values)) < -2.0e-10:
            raise ValueError(f"Saved density matrix is not PSD: {np.min(values)}, {path}")
        index = int(np.argmax(values))
        if float(values[index].real) < 1.0 - 2.0e-8:
            raise ValueError(f"Saved random state is not rank one: {path}")
        state = vectors[:, index]
        projector_error = float(np.linalg.norm(density - np.outer(state, state.conj())))
        purity = real_scalar(np.trace(density @ density), name=f"purity({path})")
        if projector_error > 2.0e-8 or abs(purity - 1.0) > 2.0e-8:
            raise ValueError(
                f"Saved state fails projector audit: error={projector_error}, purity={purity}"
            )
    else:
        raise ValueError(f"Unsupported state shape {saved.shape}: {path}")
    state /= np.linalg.norm(state)
    pivot = int(np.argmax(np.abs(state)))
    state *= np.exp(-1j * np.angle(state[pivot]))
    return state


@dataclass
class BenchmarkCase:
    benchmark: str
    slug: str
    instance: int
    label: str
    hamiltonian: np.ndarray
    state: np.ndarray
    observables: np.ndarray
    weights: np.ndarray
    shots: tuple[int, ...]
    source_paths: tuple[Path, ...]
    source_hashes: dict[str, str]
    target_expectation: float = field(init=False)

    def __post_init__(self) -> None:
        self.hamiltonian = np.asarray(self.hamiltonian, dtype=np.complex128)
        self.state = np.asarray(self.state, dtype=np.complex128).reshape(-1)
        self.state /= np.linalg.norm(self.state)
        anti = np.linalg.norm(self.hamiltonian - self.hamiltonian.conj().T)
        if anti > HERMITIAN_TOL:
            raise ValueError(f"{self.slug}: non-Hermitian target, ||H-H^dag||={anti}")
        self.hamiltonian = (self.hamiltonian + self.hamiltonian.conj().T) / 2.0
        reconstructed = pauli_dense(self.observables, self.weights)
        mismatch = float(np.linalg.norm(reconstructed - self.hamiltonian))
        if mismatch > 2.0e-8:
            raise ValueError(f"{self.slug}: Pauli/dense mismatch {mismatch}")
        if len(self.state) != len(self.hamiltonian):
            raise ValueError(f"{self.slug}: state/H dimension mismatch")
        self.target_expectation = real_scalar(
            np.vdot(self.state, self.hamiltonian @ self.state),
            name=f"{self.slug} target expectation",
        )

    @property
    def n_qubits(self) -> int:
        return int(round(math.log2(len(self.state))))

    @property
    def target_hash(self) -> str:
        return array_sha256(self.hamiltonian)


def load_cases() -> list[BenchmarkCase]:
    tfim_path = SHADOW / "Hamiltonians" / "TFIM_8.txt"
    tfim_obs, tfim_weights = load_pauli_table(tfim_path)
    tfim_record = (
        RMA
        / "submit_jctc"
        / "RandomOptimizedMeasurement_submit"
        / "Journal_chemical_theory_computation"
        / "variance_record"
        / "tfim8_gpd_recomputed"
    )
    tfim_h_path = tfim_record / "tfim8_hamiltonian.npy"
    tfim_state_path = tfim_record / "tfim8_ground_state.npy"
    tfim_h = np.asarray(np.load(tfim_h_path), dtype=np.complex128)
    tfim_state = pure_state_from_saved(tfim_state_path)
    values, vectors = np.linalg.eigh(tfim_h)
    if float(values[1] - values[0]) <= 1.0e-9:
        raise ValueError("TFIM ground state is unexpectedly degenerate")
    fidelity = float(abs(np.vdot(vectors[:, 0], tfim_state)) ** 2)
    if fidelity < 1.0 - 2.0e-10:
        raise ValueError(f"Canonical TFIM state fidelity is only {fidelity}")
    cases = [
        BenchmarkCase(
            benchmark="structured",
            slug="tfim8_j1_g1",
            instance=1,
            label="TFIM n=8, J=g=1, open boundary",
            hamiltonian=tfim_h,
            state=tfim_state,
            observables=tfim_obs,
            weights=tfim_weights,
            shots=STRUCTURED_SHOTS,
            source_paths=(tfim_path, tfim_h_path, tfim_state_path),
            source_hashes={
                str(tfim_path): sha256(tfim_path),
                str(tfim_h_path): sha256(tfim_h_path),
                str(tfim_state_path): sha256(tfim_state_path),
            },
        )
    ]

    state_path = RANDOM_DATA / "common_input_state.npy"
    metadata_path = RANDOM_DATA / "metadata.json"
    common_state = pure_state_from_saved(state_path)
    for regime in ("sparse", "dense"):
        for instance in range(1, 6):
            dense_path = RANDOM_DATA / "hamiltonians" / f"{regime}_hamiltonian_4_{instance}.npy"
            pauli_path = RANDOM_DATA / "hamiltonians" / f"{regime}_hamiltonian_4_{instance}.txt"
            hamiltonian = np.asarray(np.load(dense_path), dtype=np.complex128)
            observables, weights = load_pauli_table(pauli_path)
            cases.append(
                BenchmarkCase(
                    benchmark=regime,
                    slug=f"{regime}4_seed0_{instance}",
                    instance=instance,
                    label=f"seed-0 {regime} random Hamiltonian {instance}",
                    hamiltonian=hamiltonian,
                    state=common_state,
                    observables=observables,
                    weights=weights,
                    shots=RANDOM_SHOTS,
                    source_paths=(dense_path, pauli_path, state_path, metadata_path),
                    source_hashes={
                        str(dense_path): sha256(dense_path),
                        str(pauli_path): sha256(pauli_path),
                        str(state_path): sha256(state_path),
                        str(metadata_path): sha256(metadata_path),
                    },
                )
            )
    return cases


def rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(
        np.complex128
    )


def pauli_rotation(generator: np.ndarray, theta: float) -> np.ndarray:
    dimension = generator.shape[0]
    return (
        math.cos(theta / 2.0) * np.eye(dimension, dtype=np.complex128)
        - 1j * math.sin(theta / 2.0) * generator
    )


def givens(theta: float) -> np.ndarray:
    output = np.eye(4, dtype=np.complex128)
    c, s = math.cos(theta), math.sin(theta)
    output[1, 1] = c
    output[1, 2] = -s
    output[2, 1] = s
    output[2, 2] = c
    return output


def xy_rotation(theta: float) -> np.ndarray:
    output = np.eye(4, dtype=np.complex128)
    c, s = math.cos(theta), math.sin(theta)
    output[1, 1] = c
    output[1, 2] = -1j * s
    output[2, 1] = -1j * s
    output[2, 2] = c
    return output


def pair_transfer(theta: float) -> np.ndarray:
    output = np.eye(16, dtype=np.complex128)
    c, s = math.cos(theta), math.sin(theta)
    left, right = 12, 3  # |1100> and |0011> in the requested qubit order.
    output[left, left] = c
    output[left, right] = -s
    output[right, left] = s
    output[right, right] = c
    return output


def left_apply_gate(
    unitary: np.ndarray, gate: np.ndarray, qubits: Sequence[int], n_qubits: int
) -> np.ndarray:
    qubits = tuple(int(item) for item in qubits)
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"Repeated qubit in gate support: {qubits}")
    remaining = [item for item in range(n_qubits) if item not in qubits]
    permutation = list(qubits) + remaining + [n_qubits]
    inverse = np.argsort(permutation)
    tensor = unitary.reshape((2,) * n_qubits + (2**n_qubits,))
    moved = np.transpose(tensor, permutation)
    flat = moved.reshape(2 ** len(qubits), -1)
    updated = gate @ flat
    shape = (2,) * len(qubits) + (2,) * len(remaining) + (2**n_qubits,)
    return np.transpose(updated.reshape(shape), inverse).reshape(unitary.shape)


def chain_bonds(start: int, length: int, parity: int) -> list[tuple[int, int]]:
    return [(start + item, start + item + 1) for item in range(parity, length - 1, 2)]


@dataclass
class CircuitCandidate:
    family: str
    depth: int
    variant: int
    seed: int
    unitary: np.ndarray
    unitary_hash: str
    two_qubit_gates: int
    one_qubit_gates: int


def unitary_hash(unitary: np.ndarray) -> str:
    packed = np.stack(
        (np.round(unitary.real, 12), np.round(unitary.imag, 12)), axis=-1
    )
    return array_sha256(packed)


def _profile_angle(
    rng: np.random.Generator, variant: int, scale: float, index: int
) -> float:
    if variant == 0:
        return 0.0
    if variant == 1:
        return scale * (1.0 if index % 2 == 0 else -1.0)
    return float(rng.normal(scale=scale))


def build_circuit_candidate(
    family: str, n_qubits: int, depth: int, variant: int, *, namespace: str
) -> CircuitCandidate:
    seed = stable_seed(PROFILE_VERSION, namespace, family, n_qubits, depth, variant)
    rng = np.random.default_rng(seed)
    dimension = 2**n_qubits
    unitary = np.eye(dimension, dtype=np.complex128)
    one_count = 0
    two_count = 0
    index = 0
    m = n_qubits // 2
    xx = np.kron(PAULI[1], PAULI[1])
    yy = np.kron(PAULI[2], PAULI[2])
    zz = np.kron(PAULI[3], PAULI[3])

    if family == "gfro":
        for layer in range(depth):
            for parity in (0, 1):
                for start in (0, m):
                    for left, right in chain_bonds(start, m, parity):
                        angle = _profile_angle(rng, variant, math.pi / 8.0, index)
                        unitary = left_apply_gate(
                            unitary, givens(angle), (left, right), n_qubits
                        )
                        index += 1
                        two_count += int(abs(angle) > 1.0e-15)
    elif family == "operator_pool":
        gate_cycle = ("givens", "xy", "xx", "yy", "zz")
        for layer in range(depth):
            parity = layer % 2
            for start in (0, m):
                for left, right in chain_bonds(start, m, parity):
                    kind = gate_cycle[(variant + layer + left) % len(gate_cycle)]
                    angle = _profile_angle(
                        rng, max(variant, 1), math.pi / 7.0, index
                    )
                    if kind == "givens":
                        gate = givens(angle)
                    elif kind == "xy":
                        gate = xy_rotation(angle)
                    elif kind == "xx":
                        gate = pauli_rotation(xx, angle)
                    elif kind == "yy":
                        gate = pauli_rotation(yy, angle)
                    else:
                        gate = pauli_rotation(zz, angle)
                    unitary = left_apply_gate(
                        unitary, gate, (left, right), n_qubits
                    )
                    index += 1
                    two_count += int(abs(angle) > 1.0e-15)
    elif family == "nn_pair":
        for layer in range(depth):
            for parity in (0, 1):
                for start in (0, m):
                    for left, right in chain_bonds(start, m, parity):
                        angle = _profile_angle(
                            rng, max(variant, 1), math.pi / 9.0, index
                        )
                        unitary = left_apply_gate(
                            unitary, givens(angle), (left, right), n_qubits
                        )
                        index += 1
                        two_count += int(abs(angle) > 1.0e-15)
            for orbital in range(max(m - 1, 0)):
                support = (orbital, orbital + m, orbital + 1, orbital + 1 + m)
                angle = _profile_angle(rng, max(variant, 1), math.pi / 10.0, index)
                unitary = left_apply_gate(
                    unitary, pair_transfer(angle), support, n_qubits
                )
                index += 1
                # Count the logical four-qubit pair block separately as three 2q units.
                two_count += 3 * int(abs(angle) > 1.0e-15)
    elif family in {"iswap_su2", "srcdf_matrix"}:
        for layer in range(depth):
            parity = layer % 2
            for left in range(parity, n_qubits - 1, 2):
                if family == "iswap_su2":
                    # exp[-i theta(XX+YY)/2] at theta=-pi/2 is iSWAP (up to
                    # the conventional global/sign phase).  It is fixed and
                    # is not counted as a variational XY angle.
                    angle = -math.pi / 2.0
                elif variant in (0, 1):
                    # The separate matrix SRDD pool is not the AGPD iSWAP
                    # family.  Its shallow parametric XY block includes the
                    # zero-entangler submanifold.
                    angle = 0.0
                else:
                    angle = float(rng.normal(scale=math.pi / 9.0))
                unitary = left_apply_gate(
                    unitary, xy_rotation(angle), (left, left + 1), n_qubits
                )
                two_count += 1
            for qubit in range(n_qubits):
                if variant == 0:
                    angles = (0.0, 0.0, 0.0)
                elif variant == 1:
                    angles = (0.0, math.pi / 2.0 if layer == 0 else 0.0, 0.0)
                else:
                    if family == "iswap_su2":
                        angles = tuple(
                            float(item) for item in rng.uniform(-math.pi, math.pi, size=3)
                        )
                    else:
                        angles = tuple(
                            float(item) for item in rng.normal(scale=0.42, size=3)
                        )
                for gate in (rx(angles[0]), ry(angles[1]), rx(angles[2])):
                    unitary = left_apply_gate(unitary, gate, (qubit,), n_qubits)
                one_count += sum(abs(item) > 1.0e-15 for item in angles)
    else:
        raise ValueError(f"Unknown circuit family: {family}")

    error = float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(dimension)))
    if error > 3.0e-9:
        raise ValueError(f"{family}/d{depth}/v{variant} is not unitary: {error}")
    return CircuitCandidate(
        family=family,
        depth=depth,
        variant=variant,
        seed=seed,
        unitary=unitary,
        unitary_hash=unitary_hash(unitary),
        two_qubit_gates=two_count,
        one_qubit_gates=one_count,
    )


def build_agpd_pool(n_qubits: int) -> list[CircuitCandidate]:
    families = ("gfro", "operator_pool", "nn_pair", "iswap_su2")
    output: list[CircuitCandidate] = []
    seen: set[str] = set()
    for family in families:
        for depth in AGPD_DEPTHS:
            for variant in range(AGPD_FROZEN_STARTS):
                candidate = build_circuit_candidate(
                    family, n_qubits, depth, variant, namespace="agpd"
                )
                if candidate.unitary_hash in seen:
                    continue
                seen.add(candidate.unitary_hash)
                output.append(candidate)
    if {item.family for item in output} != set(families):
        raise AssertionError("AGPD finite pool lost a circuit family")
    return output


def build_srcdf_pools(n_qubits: int) -> dict[int, list[CircuitCandidate]]:
    # SRDD starts are extended rank by rank in run_srcdf_frontier.  Returning
    # empty depth-specific lists here avoids turning the number of initial
    # random starts into an unintended upper bound on K.
    return {depth: [] for depth in SRCDF_DEPTHS}


def rotated_diagonal(unitary: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.einsum(
        "ia,ab,ib->i", unitary, matrix, unitary.conj(), optimize=True
    )
    if np.max(np.abs(values.imag)) > 2.0e-8:
        raise ValueError("Rotated Hermitian diagonal has a material imaginary part")
    return values.real


def diagonal_fragment(unitary: np.ndarray, diagonal: np.ndarray) -> np.ndarray:
    output = unitary.conj().T @ (np.asarray(diagonal)[:, None] * unitary)
    return (output + output.conj().T) / 2.0


@dataclass
class Fragment:
    unitary: np.ndarray
    diagonal: np.ndarray
    family: str
    depth: int
    variant: int
    unitary_hash: str

    @property
    def centered_range(self) -> float:
        return 0.5 * float(np.max(self.diagonal) - np.min(self.diagonal))


@dataclass
class ApproximationPoint:
    method: str
    k: int
    depth: int
    fragments: list[Fragment]
    residual_frobenius: float
    objective: float
    trace: list[dict[str, object]]
    exact_constant: float = 0.0
    frontier_terminal_k: int = 0
    frontier_stop_reason: str = ""
    frontier_right_censored: bool = False
    frontier_safety_guard: int = GENERAL_K_SAFETY_GUARD


def fixed_shot_frontier_progress(
    point: ApproximationPoint,
    shots: Sequence[int],
    dimension: int,
    best_scores: dict[int, float],
) -> tuple[bool, dict[str, float]]:
    """Update all L_{K,T} minima and flag a material global-best improvement."""
    materially_improved = False
    scores: dict[str, float] = {}
    for total_shots in shots:
        try:
            score, _, _, _ = approximation_proxy(point, total_shots, dimension)
        except ValueError:
            continue
        scores[str(total_shots)] = score
        previous = best_scores.get(total_shots)
        if previous is None or score < previous * (1.0 - FRONTIER_RELATIVE_IMPROVEMENT):
            materially_improved = True
            best_scores[total_shots] = score
    return materially_improved, scores


def annotate_frontier(
    frontier: Sequence[ApproximationPoint], stop_reason: str, right_censored: bool
) -> None:
    terminal_k = int(frontier[-1].k)
    for point in frontier:
        point.frontier_terminal_k = terminal_k
        point.frontier_stop_reason = stop_reason
        point.frontier_right_censored = right_censored
        point.frontier_safety_guard = GENERAL_K_SAFETY_GUARD


def run_agpd_frontier(
    hamiltonian: np.ndarray,
    pool: Sequence[CircuitCandidate],
    shots: Sequence[int],
) -> list[ApproximationPoint]:
    dimension = len(hamiltonian)
    exact_constant = real_scalar(np.trace(hamiltonian) / dimension, name="AGPD trace mean")
    residual = hamiltonian - exact_constant * np.eye(dimension)
    norm_h_squared = float(np.vdot(residual, residual).real)
    fragments: list[Fragment] = []
    trace: list[dict[str, object]] = []
    frontier: list[ApproximationPoint] = []
    best_scores: dict[int, float] = {}
    stalled_boundaries = 0
    stop_reason = ""
    for leaf in range(1, GENERAL_K_SAFETY_GUARD + 1):
        norm_before = float(np.vdot(residual, residual).real)
        best: tuple[float, CircuitCandidate, np.ndarray] | None = None
        family_best: dict[str, float] = {}
        for candidate in pool:
            diagonal = rotated_diagonal(candidate.unitary, residual)
            diagonal -= float(np.mean(diagonal))
            decrease = float(np.dot(diagonal, diagonal))
            family_best[candidate.family] = max(
                family_best.get(candidate.family, -math.inf), decrease
            )
            key = (decrease, -candidate.depth, -candidate.variant)
            if best is None or key > (
                best[0], -best[1].depth, -best[1].variant
            ):
                best = (decrease, candidate, diagonal)
        assert best is not None
        decrease, candidate, diagonal = best
        if decrease <= max(1.0e-13 * norm_h_squared, 1.0e-24):
            stop_reason = "Hilbert--Schmidt projection below numerical tolerance"
            break
        matrix = diagonal_fragment(candidate.unitary, diagonal)
        residual = (residual - matrix + (residual - matrix).conj().T) / 2.0
        norm_after = float(np.vdot(residual, residual).real)
        projection_identity_error = abs((norm_before - norm_after) - decrease)
        if projection_identity_error > 2.0e-9 * max(norm_before, 1.0):
            raise AssertionError(
                "AGPD Hilbert--Schmidt projection identity failed: "
                f"{projection_identity_error}"
            )
        if norm_after > norm_before + 5.0e-8 * max(norm_before, 1.0):
            raise AssertionError("AGPD residual increased after exact diagonal projection")
        fragment = Fragment(
            unitary=candidate.unitary,
            diagonal=diagonal,
            family=candidate.family,
            depth=candidate.depth,
            variant=candidate.variant,
            unitary_hash=candidate.unitary_hash,
        )
        fragments.append(fragment)
        trace_entry = {
                "leaf": leaf,
                "selected_family": candidate.family,
                "selected_depth": candidate.depth,
                "selected_variant": candidate.variant,
                "unitary_hash": candidate.unitary_hash,
                "residual_frobenius_before": math.sqrt(max(norm_before, 0.0)),
                "residual_frobenius_after": math.sqrt(max(norm_after, 0.0)),
                "projection_decrease_squared": decrease,
                "projection_identity_absolute_error": projection_identity_error,
                "family_best_decrease_squared": family_best,
            }
        point = ApproximationPoint(
                method="AGPD",
                k=leaf,
                depth=candidate.depth,
                fragments=list(fragments),
                residual_frobenius=math.sqrt(max(norm_after, 0.0)),
                objective=norm_after,
                trace=list(trace),
                exact_constant=exact_constant,
            )
        materially_improved, scores = fixed_shot_frontier_progress(
            point, shots, dimension, best_scores
        )
        trace_entry["fixed_shot_scores"] = scores
        trace.append(trace_entry)
        point.trace = list(trace)
        frontier.append(point)
        if norm_after <= 1.0e-22 * max(norm_h_squared, 1.0):
            stop_reason = "centered residual reached numerical zero"
            break
        stalled_boundaries = 0 if materially_improved else stalled_boundaries + 1
        if (
            leaf >= FRONTIER_MINIMUM_K
            and stalled_boundaries >= FRONTIER_STALL_PATIENCE
        ):
            stop_reason = (
                f"no L_K,T global-best improvement above "
                f"{FRONTIER_RELATIVE_IMPROVEMENT:g} for "
                f"{FRONTIER_STALL_PATIENCE} consecutive K values"
            )
            break
    if not frontier:
        raise RuntimeError("AGPD finite pool produced no nonzero projection")
    right_censored = not stop_reason
    if right_censored:
        stop_reason = f"reached safety guard K={GENERAL_K_SAFETY_GUARD}"
    annotate_frontier(frontier, stop_reason, right_censored)
    if right_censored:
        raise RuntimeError(
            "AGPD fixed-shot frontier reached its safety guard without convergence; "
            "no censored result will be published"
        )
    return frontier


@dataclass
class SrcdfDepthState:
    depth: int
    candidates: list[CircuitCandidate] = field(default_factory=list)
    target_diagonals: list[np.ndarray] = field(default_factory=list)
    selected: list[int] = field(default_factory=list)
    coefficients: list[np.ndarray] = field(default_factory=list)
    trace: list[dict[str, object]] = field(default_factory=list)
    seen_hashes: set[str] = field(default_factory=set)
    next_variant: int = 0


def extend_srcdf_starts(
    state: SrcdfDepthState,
    centered_hamiltonian: np.ndarray,
    n_qubits: int,
) -> None:
    """Add a fresh deterministic multistart batch for the next factor."""
    added = 0
    while added < SRCDF_STARTS_PER_FACTOR:
        variant = state.next_variant
        state.next_variant += 1
        candidate = build_circuit_candidate(
            "srcdf_matrix",
            n_qubits,
            state.depth,
            variant,
            namespace="srcdf-rankwise",
        )
        if candidate.unitary_hash in state.seen_hashes:
            continue
        state.seen_hashes.add(candidate.unitary_hash)
        state.candidates.append(candidate)
        diagonal = rotated_diagonal(candidate.unitary, centered_hamiltonian)
        diagonal -= float(np.mean(diagonal))
        state.target_diagonals.append(diagonal)
        added += 1


def solve_srcdf_selected(
    state: SrcdfDepthState,
    ridge: float,
) -> tuple[list[np.ndarray], float]:
    dimension = len(state.target_diagonals[state.selected[0]])
    blocks = len(state.selected)
    gram = np.empty((blocks * dimension, blocks * dimension), dtype=float)
    rhs = np.concatenate([state.target_diagonals[index] for index in state.selected])
    for left_position, left_index in enumerate(state.selected):
        left_slice = slice(left_position * dimension, (left_position + 1) * dimension)
        left = state.candidates[left_index]
        for right_position, right_index in enumerate(state.selected):
            right_slice = slice(
                right_position * dimension, (right_position + 1) * dimension
            )
            right = state.candidates[right_index]
            transition = left.unitary @ right.unitary.conj().T
            gram[left_slice, right_slice] = np.abs(transition) ** 2
    gram.flat[:: len(gram) + 1] += ridge
    solution = np.linalg.solve(gram, rhs)
    coefficients = [
        solution[position * dimension : (position + 1) * dimension].copy()
        for position in range(blocks)
    ]
    for coefficient in coefficients:
        coefficient -= float(np.mean(coefficient))
    kkt = float(np.linalg.norm(gram @ np.concatenate(coefficients) - rhs, ord=np.inf))
    if kkt > 2.0e-7:
        raise RuntimeError(f"SRDD joint ridge solve KKT residual is {kkt}")
    return coefficients, kkt


def advance_srcdf_depth(
    state: SrcdfDepthState,
    centered_hamiltonian: np.ndarray,
    n_qubits: int,
    ridge: float,
) -> ApproximationPoint | None:
    extend_srcdf_starts(state, centered_hamiltonian, n_qubits)
    reconstructed = np.zeros_like(centered_hamiltonian)
    for index, coefficient in zip(state.selected, state.coefficients):
        reconstructed += diagonal_fragment(state.candidates[index].unitary, coefficient)
    residual_matrix = (centered_hamiltonian - reconstructed)
    residual_matrix = (residual_matrix + residual_matrix.conj().T) / 2.0
    best: tuple[float, int, np.ndarray] | None = None
    selected_set = set(state.selected)
    for candidate_index, candidate in enumerate(state.candidates):
        if candidate_index in selected_set:
            continue
        diagonal = rotated_diagonal(candidate.unitary, residual_matrix)
        diagonal -= float(np.mean(diagonal))
        decrease = float(np.dot(diagonal, diagonal) / (1.0 + ridge))
        if best is None or (decrease, -candidate_index) > (best[0], -best[1]):
            best = (decrease, candidate_index, diagonal)
    if best is None or best[0] <= 1.0e-24:
        return None
    greedy_decrease, chosen, _ = best
    state.selected.append(chosen)
    state.coefficients, kkt = solve_srcdf_selected(state, ridge)
    reconstructed = np.zeros_like(centered_hamiltonian)
    fragments: list[Fragment] = []
    for candidate_index, coefficient in zip(state.selected, state.coefficients):
        candidate = state.candidates[candidate_index]
        reconstructed += diagonal_fragment(candidate.unitary, coefficient)
        fragments.append(
            Fragment(
                unitary=candidate.unitary,
                diagonal=coefficient.copy(),
                family="srcdf_matrix",
                depth=state.depth,
                variant=candidate.variant,
                unitary_hash=candidate.unitary_hash,
            )
        )
    residual = float(np.linalg.norm(centered_hamiltonian - reconstructed))
    ridge_penalty = ridge * sum(
        float(np.dot(item, item)) for item in state.coefficients
    )
    state.trace.append(
        {
            "slot": len(state.selected),
            "depth": state.depth,
            "new_start_count": SRCDF_STARTS_PER_FACTOR,
            "cumulative_start_count": len(state.candidates),
            "candidate_index": chosen,
            "candidate_variant": state.candidates[chosen].variant,
            "unitary_hash": state.candidates[chosen].unitary_hash,
            "greedy_ridge_decrease": greedy_decrease,
            "joint_kkt_residual": kkt,
            "direct_residual_frobenius": residual,
            "ridge_penalty": ridge_penalty,
        }
    )
    return ApproximationPoint(
        method=SRDD_METHOD,
        k=len(state.selected),
        depth=state.depth,
        fragments=fragments,
        residual_frobenius=residual,
        objective=residual * residual + ridge_penalty,
        trace=list(state.trace),
    )


def run_srcdf_frontier(
    hamiltonian: np.ndarray,
    pools: dict[int, Sequence[CircuitCandidate]],
    ridge: float,
    shots: Sequence[int],
) -> tuple[list[ApproximationPoint], list[dict[str, object]]]:
    dimension = len(hamiltonian)
    n_qubits = int(round(math.log2(dimension)))
    exact_constant = real_scalar(np.trace(hamiltonian) / dimension, name="SRDD trace mean")
    centered_hamiltonian = hamiltonian - exact_constant * np.eye(dimension)
    states = {depth: SrcdfDepthState(depth=depth) for depth in SRCDF_DEPTHS}
    for depth, initial in pools.items():
        for candidate in initial:
            states[depth].candidates.append(candidate)
            states[depth].seen_hashes.add(candidate.unitary_hash)
            diagonal = rotated_diagonal(candidate.unitary, centered_hamiltonian)
            diagonal -= float(np.mean(diagonal))
            states[depth].target_diagonals.append(diagonal)
    frontier: list[ApproximationPoint] = []
    trials: list[dict[str, object]] = []
    best_scores: dict[int, float] = {}
    stalled_boundaries = 0
    stop_reason = ""
    for k in range(1, GENERAL_K_SAFETY_GUARD + 1):
        depth_points: list[ApproximationPoint] = []
        materially_improved = False
        for depth in SRCDF_DEPTHS:
            point = advance_srcdf_depth(
                states[depth], centered_hamiltonian, n_qubits, ridge
            )
            if point is None:
                continue
            if point.k != k:
                raise AssertionError("SRDD depth frontiers lost rank synchronization")
            point.exact_constant = exact_constant
            improved, scores = fixed_shot_frontier_progress(
                point, shots, dimension, best_scores
            )
            materially_improved = materially_improved or improved
            point.trace[-1]["fixed_shot_scores"] = scores
            depth_points.append(point)
            frontier.append(point)
            trials.append(
                {
                    "method": SRDD_METHOD,
                    "k": k,
                    "depth": depth,
                    "residual_frobenius": point.residual_frobenius,
                    "objective_with_ridge": point.objective,
                    "selected_unitary_hashes": [
                        item.unitary_hash for item in point.fragments
                    ],
                    "trace": point.trace,
                }
            )
        if not depth_points:
            stop_reason = "all shallow depth residual projections reached numerical zero"
            break
        stalled_boundaries = 0 if materially_improved else stalled_boundaries + 1
        if (
            k >= FRONTIER_MINIMUM_K
            and stalled_boundaries >= FRONTIER_STALL_PATIENCE
        ):
            stop_reason = (
                f"no L_K,T global-best improvement above "
                f"{FRONTIER_RELATIVE_IMPROVEMENT:g} for "
                f"{FRONTIER_STALL_PATIENCE} consecutive K values"
            )
            break
    if not frontier:
        raise RuntimeError("SRDD produced no nonzero shallow factor")
    right_censored = not stop_reason
    if right_censored:
        stop_reason = f"reached safety guard K={GENERAL_K_SAFETY_GUARD}"
    annotate_frontier(frontier, stop_reason, right_censored)
    if right_censored:
        raise RuntimeError(
            "SRDD fixed-shot frontier reached its safety guard without convergence; "
            "no censored result will be published"
        )
    return frontier, trials


class StateEvaluator:
    def __init__(self, state: np.ndarray):
        self.state = np.asarray(state, dtype=np.complex128).reshape(-1)
        self.state /= np.linalg.norm(self.state)
        self.dimension = len(self.state)
        self.n_qubits = int(round(math.log2(self.dimension)))
        self.indices = np.arange(self.dimension, dtype=np.int64)
        self._probability_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._sign_cache: dict[int, np.ndarray] = {}
        self._expectation_cache: dict[tuple[int, ...], float] = {}

    def sign_for_mask(self, mask: int) -> np.ndarray:
        if mask not in self._sign_cache:
            self._sign_cache[mask] = np.fromiter(
                (
                    1.0 if (int(index) & int(mask)).bit_count() % 2 == 0 else -1.0
                    for index in self.indices
                ),
                dtype=float,
                count=self.dimension,
            )
        return self._sign_cache[mask]

    def term_sign(self, code: Sequence[int]) -> np.ndarray:
        mask = 0
        for qubit, item in enumerate(code):
            if int(item) != 0:
                mask |= 1 << (self.n_qubits - 1 - qubit)
        return self.sign_for_mask(mask)

    def basis_probabilities(self, basis: Sequence[int]) -> np.ndarray:
        key = tuple(int(item) for item in basis)
        if key not in self._probability_cache:
            tensor = self.state.reshape((2,) * self.n_qubits)
            for qubit, item in enumerate(key):
                rotation = MEASUREMENT_ROTATION[int(item)]
                tensor = np.tensordot(rotation, tensor, axes=(1, qubit))
                tensor = np.moveaxis(tensor, 0, qubit)
            probabilities = np.abs(tensor.reshape(-1)) ** 2
            probabilities /= np.sum(probabilities)
            self._probability_cache[key] = probabilities
        return self._probability_cache[key]

    def apply_pauli(self, code: Sequence[int]) -> np.ndarray:
        output = np.zeros(self.dimension, dtype=np.complex128)
        for source in range(self.dimension):
            destination = source
            phase = 1.0 + 0.0j
            for qubit, item in enumerate(code):
                bit_mask = 1 << (self.n_qubits - 1 - qubit)
                bit = 1 if source & bit_mask else 0
                item = int(item)
                if item == 1:
                    destination ^= bit_mask
                elif item == 2:
                    destination ^= bit_mask
                    phase *= 1j if bit == 0 else -1j
                elif item == 3 and bit:
                    phase *= -1.0
            output[destination] += phase * self.state[source]
        return output

    def expectation(self, code: Sequence[int]) -> float:
        key = tuple(int(item) for item in code)
        if key not in self._expectation_cache:
            self._expectation_cache[key] = real_scalar(
                np.vdot(self.state, self.apply_pauli(key)),
                name=f"Pauli expectation {key}",
            )
        return self._expectation_cache[key]


def split_identity(
    observables: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    identity = ~np.any(observables != 0, axis=1)
    offset = float(np.sum(weights[identity]))
    keep = (~identity) & (np.abs(weights) > 1.0e-15)
    return observables[keep], weights[keep], offset


def coverage_matrix(observables: np.ndarray, bases: np.ndarray) -> np.ndarray:
    # This reproduces the retained hit_by/IfCommute convention.  A zero in an
    # OGM setting is an unspecified basis and is physically sampled as Z by
    # the archived MATLAB code.  The audit records whether a wildcard is ever
    # used to cover a nonidentity factor.
    return np.all(
        (observables[None, :, :] == 0)
        | (bases[:, None, :] == 0)
        | (observables[None, :, :] == bases[:, None, :]),
        axis=2,
    )


def combine_count_schedule(
    bases: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    merged: Counter[tuple[int, ...]] = Counter()
    for basis, count in zip(np.asarray(bases, dtype=int), np.asarray(counts, dtype=int)):
        if int(count) > 0:
            merged[tuple(int(item) for item in basis)] += int(count)
    items = sorted(merged.items())
    return (
        np.asarray([key for key, _ in items], dtype=np.int8),
        np.asarray([value for _, value in items], dtype=int),
    )


def fixed_schedule_mse(
    evaluator: StateEvaluator,
    observables: np.ndarray,
    weights: np.ndarray,
    offset: float,
    bases: np.ndarray,
    counts: np.ndarray,
    exact_expectation: float,
) -> dict[str, float | int]:
    bases, counts = combine_count_schedule(bases, counts)
    shots = int(np.sum(counts))
    if shots <= 0:
        raise ValueError("A fixed schedule must contain at least one shot")
    coverage = coverage_matrix(observables, bases)
    hits = coverage.T @ counts
    covered = hits > 0
    term_means = np.asarray([evaluator.expectation(row) for row in observables])
    expected = offset + float(np.dot(weights[covered], term_means[covered]))
    bias = expected - exact_expectation
    variance = 0.0
    wildcard_hits = 0
    for basis_index, (basis, count) in enumerate(zip(bases, counts)):
        indices = np.flatnonzero(coverage[basis_index] & covered)
        if len(indices) == 0:
            continue
        diagonal = np.zeros(evaluator.dimension, dtype=float)
        for term_index in indices:
            diagonal += (
                weights[term_index]
                / hits[term_index]
                * evaluator.term_sign(observables[term_index])
            )
            wildcard_hits += int(
                np.any((basis == 0) & (observables[term_index] != 0))
            )
        probabilities = evaluator.basis_probabilities(basis)
        mean = float(probabilities @ diagonal)
        second = float(probabilities @ (diagonal * diagonal))
        conditional = second - mean * mean
        if conditional < -1.0e-10:
            raise ValueError(
                f"Materially negative fixed-schedule variance {conditional}"
            )
        variance += int(count) * max(conditional, 0.0)
    mse = variance + bias * bias
    return {
        "mse": float(max(mse, 0.0)),
        "rmse": math.sqrt(max(mse, 0.0)),
        "bias": float(bias),
        "bias_squared": float(bias * bias),
        "variance": float(variance),
        "covered_terms": int(np.count_nonzero(covered)),
        "total_terms": int(len(observables)),
        "distinct_bases": int(len(bases)),
        "wildcard_hit_pairs": int(wildcard_hits),
    }


def qwc_product(
    left: np.ndarray, right: np.ndarray
) -> tuple[bool, np.ndarray, int]:
    conflict = (left != 0) & (right != 0) & (left != right)
    if np.any(conflict):
        return False, np.zeros_like(left), 0
    same = (left != 0) & (left == right)
    product = np.where(same, 0, np.where(left != 0, left, right))
    return True, product, int(np.count_nonzero(same))


def analytic_lcs_mse(
    evaluator: StateEvaluator,
    observables: np.ndarray,
    weights: np.ndarray,
    shots: int,
    exact_expectation_without_offset: float,
) -> dict[str, float | int]:
    m1 = 0.0
    for left, left_weight in zip(observables, weights):
        for right, right_weight in zip(observables, weights):
            compatible, product, overlap = qwc_product(left, right)
            if compatible:
                m1 += (
                    left_weight
                    * right_weight
                    * (3**overlap)
                    * evaluator.expectation(product)
                )
    centered_raw = float(m1 - exact_expectation_without_offset**2)
    if centered_raw < -1.0e-9:
        raise ValueError(f"Materially negative LCS one-shot variance {centered_raw}")
    centered = max(centered_raw, 0.0)
    mse = centered / int(shots)
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "bias": 0.0,
        "bias_squared": 0.0,
        "variance": mse,
        "covered_terms": int(len(observables)),
        "total_terms": int(len(observables)),
        "distinct_bases": int(3**evaluator.n_qubits),
        "wildcard_hit_pairs": 0,
    }


def schedule_cache_signature(
    slug: str,
    method: str,
    observables: np.ndarray,
    weights: np.ndarray,
    shots: Sequence[int],
    seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(PROFILE_VERSION.encode())
    digest.update(slug.encode())
    digest.update(method.encode())
    digest.update(np.ascontiguousarray(observables).view(np.uint8))
    digest.update(np.ascontiguousarray(weights).view(np.uint8))
    digest.update(np.asarray(shots, dtype=np.int64).view(np.uint8))
    digest.update(str(seed).encode())
    return digest.hexdigest()


def _generate_retained_schedule_payload(payload: dict[str, object]) -> dict[str, object]:
    method = str(payload["method"])
    observables = np.asarray(payload["observables"], dtype=np.int8)
    weights = np.asarray(payload["weights"], dtype=float)
    shots = tuple(int(item) for item in payload["shots"])
    seed = int(payload["seed"])
    np.random.seed(seed)
    scheme = measurement_tools.tutorial_scheme(method, observables, weights)
    counter: Counter[tuple[int, ...]] = Counter()
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    targets = set(shots)
    for step in range(1, max(shots) + 1):
        setting, _ = scheme.find_setting()
        counter[tuple(int(item) for item in setting)] += 1
        if step in targets:
            items = sorted(counter.items())
            snapshots[step] = (
                np.asarray([key for key, _ in items], dtype=np.int8),
                np.asarray([value for _, value in items], dtype=int),
            )
    return {
        "job_id": str(payload["job_id"]),
        "method": method,
        "seed": seed,
        "snapshots": snapshots,
        "signature": str(payload["signature"]),
    }


def save_schedule_cache(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = result["snapshots"]
    assert isinstance(snapshots, dict)
    payload: dict[str, np.ndarray] = {
        "metadata": np.asarray(
            json.dumps(
                {
                    "profile": PROFILE_VERSION,
                    "signature": result["signature"],
                    "method": result["method"],
                    "seed": result["seed"],
                },
                sort_keys=True,
            )
        )
    }
    for shots, pair in snapshots.items():
        bases, counts = pair
        payload[f"bases_{shots}"] = np.asarray(bases, dtype=np.int8)
        payload[f"counts_{shots}"] = np.asarray(counts, dtype=int)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_schedule_cache(
    path: Path, signature: str, shots: Sequence[int]
) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata.get("signature") != signature:
                return None
            return {
                int(point): (
                    saved[f"bases_{point}"].astype(np.int8),
                    saved[f"counts_{point}"].astype(int),
                )
                for point in shots
            }
    except Exception:
        return None


def ap_exact_setting_distribution(
    observables: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate the retained AdaptiveShadows one-setting law for n<=4."""
    n = observables.shape[1]
    if n > 4:
        raise ValueError("Exact AP setting enumeration is restricted to n<=4")
    settings = np.asarray(list(itertools.product((1, 2, 3), repeat=n)), dtype=np.int8)
    probabilities = np.zeros(len(settings), dtype=float)
    squared = np.square(weights)
    permutations = list(itertools.permutations(range(n)))
    for index, setting in enumerate(settings):
        total = 0.0
        for permutation in permutations:
            compatible = np.ones(len(observables), dtype=bool)
            probability = 1.0
            for qubit in permutation:
                constants = np.asarray(
                    [
                        float(np.sum(squared[compatible & (observables[:, qubit] == axis)]))
                        for axis in (1, 2, 3)
                    ]
                )
                beta = np.sqrt(constants)
                if float(np.sum(beta)) == 0.0:
                    beta[:] = 1.0 / 3.0
                else:
                    beta /= np.sum(beta)
                chosen = int(setting[qubit])
                probability *= float(beta[chosen - 1])
                compatible &= (observables[:, qubit] == 0) | (
                    observables[:, qubit] == chosen
                )
            total += probability
        probabilities[index] = total / len(permutations)
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= np.sum(probabilities)
    return settings, probabilities


def sampled_prefixes_from_distribution(
    settings: np.ndarray,
    probabilities: np.ndarray,
    shots: Sequence[int],
    seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(settings), size=max(shots), p=probabilities)
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for point in shots:
        counts = np.bincount(indices[:point], minlength=len(settings))
        active = counts > 0
        output[int(point)] = (settings[active], counts[active].astype(int))
    return output


def combine_ogm_distribution(
    settings: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    merged: dict[tuple[int, ...], float] = {}
    for setting, probability in zip(settings, probabilities):
        if float(probability) > 1.0e-15:
            key = tuple(int(item) for item in setting)
            merged[key] = merged.get(key, 0.0) + float(probability)
    bases = np.asarray(list(merged), dtype=np.int8)
    values = np.asarray(list(merged.values()), dtype=float)
    values /= np.sum(values)
    return bases, values


def allocate_ogm_schedule(
    settings: np.ndarray, probabilities: np.ndarray, shots: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Literal finite-T allocation in gen_errors_all_algs/Sample_main.m."""
    order = np.argsort(-probabilities, kind="stable")
    rng = np.random.RandomState(seed)
    selected_bases: list[np.ndarray] = []
    selected_counts: list[int] = []
    allocated = 0
    for index in order:
        amount = shots * probabilities[index]
        if amount > 1:
            number = int(np.floor(amount))
            if rng.rand() < np.mod(amount, number):
                number += 1
        else:
            number = 1
        number = min(number, shots - allocated)
        if number > 0:
            selected_bases.append(settings[index])
            selected_counts.append(number)
            allocated += number
        if allocated >= shots:
            break
    if allocated < shots:
        if not selected_counts:
            raise RuntimeError("OGM allocation has empty support")
        selected_counts[0] += shots - allocated
    return np.asarray(selected_bases, dtype=np.int8), np.asarray(selected_counts, dtype=int)


def prepare_stateful_schedules(
    cases: Sequence[BenchmarkCase], workers: int, force: bool
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    output: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    jobs: list[dict[str, object]] = []
    for case in cases:
        observables, weights, _ = split_identity(case.observables, case.weights)
        for method in ("SG", "Derand"):
            seed = stable_seed(BASE_SEED, case.slug, method, "fixed-prefix")
            job_id = f"{case.slug}_{method}"
            signature = schedule_cache_signature(
                case.slug, method, observables, weights, case.shots, seed
            )
            cache = SCHEDULE_DIR / f"{job_id}.npz"
            saved = None if force else load_schedule_cache(cache, signature, case.shots)
            if saved is not None:
                output[job_id] = saved
            else:
                jobs.append(
                    {
                        "job_id": job_id,
                        "method": method,
                        "observables": observables,
                        "weights": weights,
                        "shots": case.shots,
                        "seed": seed,
                        "signature": signature,
                        "cache": str(cache),
                    }
                )
        if case.n_qubits > 4:
            for repeat in range(AP_REPEATS):
                method = "AP"
                seed = stable_seed(BASE_SEED, case.slug, method, repeat)
                job_id = f"{case.slug}_{method}_{repeat:02d}"
                signature = schedule_cache_signature(
                    case.slug, f"{method}_{repeat}", observables, weights, case.shots, seed
                )
                cache = SCHEDULE_DIR / f"{job_id}.npz"
                saved = None if force else load_schedule_cache(cache, signature, case.shots)
                if saved is not None:
                    output[job_id] = saved
                else:
                    jobs.append(
                        {
                            "job_id": job_id,
                            "method": method,
                            "observables": observables,
                            "weights": weights,
                            "shots": case.shots,
                            "seed": seed,
                            "signature": signature,
                            "cache": str(cache),
                        }
                    )
    if not jobs:
        return output
    print(f"Generating {len(jobs)} missing retained schedules with {workers} workers", flush=True)
    if workers <= 1:
        results = [_generate_retained_schedule_payload(job) for job in jobs]
        for job, result in zip(jobs, results):
            save_schedule_cache(Path(str(job["cache"])), result)
            output[str(result["job_id"])] = result["snapshots"]  # type: ignore[assignment]
    else:
        job_lookup = {str(job["job_id"]): job for job in jobs}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_generate_retained_schedule_payload, job): str(job["job_id"])
                for job in jobs
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                job_id = str(result["job_id"])
                save_schedule_cache(Path(str(job_lookup[job_id]["cache"])), result)
                output[job_id] = result["snapshots"]  # type: ignore[assignment]
                completed += 1
                print(f"  retained schedules {completed}/{len(jobs)}: {job_id}", flush=True)
    return output


def pauli_rows_for_case(
    case: BenchmarkCase,
    schedules: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    observables, weights, offset = split_identity(case.observables, case.weights)
    evaluator = StateEvaluator(case.state)
    nonidentity_expectation = float(
        np.dot(weights, [evaluator.expectation(row) for row in observables])
    )
    if not np.isclose(
        offset + nonidentity_expectation,
        case.target_expectation,
        atol=3.0e-8,
    ):
        raise AssertionError(f"{case.slug}: Pauli/state expectation mismatch")
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for point in case.shots:
        result = analytic_lcs_mse(
            evaluator, observables, weights, point, nonidentity_expectation
        )
        rows.append(
            {
                "benchmark": case.benchmark,
                "instance": case.instance,
                "slug": case.slug,
                "method": "LCS",
                "shots": point,
                **result,
                "selected_k": "",
                "selected_depth": "",
                "settings": result["distinct_bases"],
                "protocol": "exact uniform local-Pauli channel",
            }
        )

    for method in ("SG", "Derand"):
        job_id = f"{case.slug}_{method}"
        seed = stable_seed(BASE_SEED, case.slug, method, "fixed-prefix")
        for point in case.shots:
            bases, counts = schedules[job_id][point]
            result = fixed_schedule_mse(
                evaluator,
                observables,
                weights,
                offset,
                bases,
                counts,
                case.target_expectation,
            )
            rows.append(
                {
                    "benchmark": case.benchmark,
                    "instance": case.instance,
                    "slug": case.slug,
                    "method": method,
                    "shots": point,
                    **result,
                    "selected_k": "",
                    "selected_depth": "",
                    "settings": result["distinct_bases"],
                    "protocol": "one preregistered max-T schedule; fixed prefix",
                }
            )
            audit.append(
                {
                    "benchmark": case.benchmark,
                    "slug": case.slug,
                    "method": method,
                    "shots": point,
                    "schedule_repeat": 0,
                    "seed": seed,
                    **result,
                    "schedule_source": "retained Python implementation",
                }
            )

    settings, probabilities, objective, success = measurement_tools.optimize_ogm_distribution(
        observables, weights
    )
    settings, probabilities = combine_ogm_distribution(settings, probabilities)
    if not success:
        raise RuntimeError(f"{case.slug}: OGM SLSQP did not converge")
    for point in case.shots:
        seed = stable_seed(BASE_SEED, case.slug, "OGM", point)
        bases, counts = allocate_ogm_schedule(settings, probabilities, point, seed)
        result = fixed_schedule_mse(
            evaluator,
            observables,
            weights,
            offset,
            bases,
            counts,
            case.target_expectation,
        )
        rows.append(
            {
                "benchmark": case.benchmark,
                "instance": case.instance,
                "slug": case.slug,
                "method": "OGM",
                "shots": point,
                **result,
                "selected_k": "",
                "selected_depth": "",
                "settings": result["distinct_bases"],
                "protocol": "one preregistered archived finite-T OGM allocation",
            }
        )
        audit.append(
            {
                "benchmark": case.benchmark,
                "slug": case.slug,
                "method": "OGM",
                "shots": point,
                "schedule_repeat": 0,
                "seed": seed,
                **result,
                "ogm_objective": objective,
                "ogm_active_settings": len(settings),
                "schedule_source": "archived objective and MATLAB integer rule",
            }
        )

    ap_results: dict[int, list[dict[str, float | int]]] = {
        point: [] for point in case.shots
    }
    if case.n_qubits <= 4:
        ap_settings, ap_probabilities = ap_exact_setting_distribution(observables, weights)
        ap_distribution_hash = array_sha256(
            np.column_stack((ap_settings, ap_probabilities))
        )
        for repeat in range(AP_REPEATS):
            seed = stable_seed(BASE_SEED, case.slug, "AP", repeat)
            prefixes = sampled_prefixes_from_distribution(
                ap_settings, ap_probabilities, case.shots, seed
            )
            for point in case.shots:
                bases, counts = prefixes[point]
                result = fixed_schedule_mse(
                    evaluator,
                    observables,
                    weights,
                    offset,
                    bases,
                    counts,
                    case.target_expectation,
                )
                ap_results[point].append(result)
                audit.append(
                    {
                        "benchmark": case.benchmark,
                        "slug": case.slug,
                        "method": "AP",
                        "shots": point,
                        "schedule_repeat": repeat,
                        "seed": seed,
                        **result,
                        "ap_distribution_hash": ap_distribution_hash,
                        "schedule_source": "exact enumeration of retained AP setting law",
                    }
                )
    else:
        for repeat in range(AP_REPEATS):
            seed = stable_seed(BASE_SEED, case.slug, "AP", repeat)
            prefixes = schedules[f"{case.slug}_AP_{repeat:02d}"]
            for point in case.shots:
                bases, counts = prefixes[point]
                result = fixed_schedule_mse(
                    evaluator,
                    observables,
                    weights,
                    offset,
                    bases,
                    counts,
                    case.target_expectation,
                )
                ap_results[point].append(result)
                audit.append(
                    {
                        "benchmark": case.benchmark,
                        "slug": case.slug,
                        "method": "AP",
                        "shots": point,
                        "schedule_repeat": repeat,
                        "seed": seed,
                        **result,
                        "schedule_source": "retained Python implementation",
                    }
                )
    for point in case.shots:
        selected = ap_results[point]
        mean_mse = float(np.mean([float(item["mse"]) for item in selected]))
        rows.append(
            {
                "benchmark": case.benchmark,
                "instance": case.instance,
                "slug": case.slug,
                "method": "AP",
                "shots": point,
                "mse": mean_mse,
                "rmse": math.sqrt(mean_mse),
                "bias": float(np.mean([float(item["bias"]) for item in selected])),
                "bias_squared": float(
                    np.mean([float(item["bias_squared"]) for item in selected])
                ),
                "variance": float(
                    np.mean([float(item["variance"]) for item in selected])
                ),
                "covered_terms": float(
                    np.mean([float(item["covered_terms"]) for item in selected])
                ),
                "total_terms": len(observables),
                "distinct_bases": float(
                    np.mean([float(item["distinct_bases"]) for item in selected])
                ),
                "wildcard_hit_pairs": int(
                    max(int(item["wildcard_hit_pairs"]) for item in selected)
                ),
                "selected_k": "",
                "selected_depth": "",
                "settings": float(
                    np.mean([float(item["distinct_bases"]) for item in selected])
                ),
                "protocol": f"mean exact MSE over {AP_REPEATS} preregistered schedules",
            }
        )
    return rows, audit


def integer_range_allocation(
    fragments: Sequence[Fragment], total_shots: int
) -> tuple[np.ndarray, np.ndarray, float]:
    ranges = np.asarray([item.centered_range for item in fragments], dtype=float)
    active = ranges > RANGE_TOL
    count = int(np.count_nonzero(active))
    if count > total_shots:
        raise ValueError(f"{count} nonconstant settings exceed T={total_shots}")
    allocation = np.zeros(len(fragments), dtype=int)
    if count == 0:
        return allocation, ranges, 0.0
    allocation[active] = 1
    remaining = int(total_shots - count)
    active_indices = np.flatnonzero(active)
    if remaining > 0:
        quotas = remaining * ranges[active] / np.sum(ranges[active])
        floors = np.floor(quotas).astype(int)
        allocation[active_indices] += floors
        leftover = remaining - int(np.sum(floors))
        if leftover:
            fractional = quotas - floors
            order = np.lexsort((active_indices, -fractional))
            allocation[active_indices[order[:leftover]]] += 1
    if int(np.sum(allocation)) != total_shots:
        raise AssertionError("Integer range allocation does not conserve shots")
    proxy_squared = float(
        np.sum(np.square(ranges[active]) / allocation[active])
    )
    return allocation, ranges, math.sqrt(max(proxy_squared, 0.0))


def approximation_proxy(
    point: ApproximationPoint, total_shots: int, dimension: int
) -> tuple[float, np.ndarray, np.ndarray, float]:
    allocation, ranges, sampling = integer_range_allocation(
        point.fragments, total_shots
    )
    approximation = point.residual_frobenius / math.sqrt(dimension)
    score = math.sqrt(approximation * approximation + sampling * sampling)
    return score, allocation, ranges, sampling


def select_approximation_point(
    frontier: Sequence[ApproximationPoint], total_shots: int, dimension: int
) -> tuple[ApproximationPoint, np.ndarray, np.ndarray, float, float]:
    feasible: list[
        tuple[float, int, int, ApproximationPoint, np.ndarray, np.ndarray, float]
    ] = []
    for point in frontier:
        try:
            score, allocation, ranges, sampling = approximation_proxy(
                point, total_shots, dimension
            )
        except ValueError:
            continue
        feasible.append(
            (score, point.k, point.depth, point, allocation, ranges, sampling)
        )
    if not feasible:
        raise RuntimeError(f"No approximation candidate is feasible at T={total_shots}")
    minimum_score = min(item[0] for item in feasible)
    # The same relative resolution used by the continuation test defines a
    # numerical equivalence class.  Selecting its smallest K prevents a tiny
    # sub-tolerance decrease at the terminal continuation point from being
    # misreported as evidence for an ever-larger decomposition.
    equivalent = [
        item
        for item in feasible
        if item[0] <= minimum_score * (1.0 + FRONTIER_RELATIVE_IMPROVEMENT)
    ]
    score, _, _, point, allocation, ranges, sampling = min(
        equivalent, key=lambda item: (item[1], item[0], item[2])
    )
    return point, allocation, ranges, sampling, score


def approximate_estimator_mse(
    case: BenchmarkCase,
    point: ApproximationPoint,
    allocation: np.ndarray,
) -> dict[str, float | int]:
    approximate_expectation = float(point.exact_constant)
    variance = 0.0
    active_settings = 0
    for fragment, count in zip(point.fragments, allocation):
        rotated_state = fragment.unitary @ case.state
        probabilities = np.abs(rotated_state) ** 2
        probability_sum = float(np.sum(probabilities))
        if not np.isclose(probability_sum, 1.0, atol=2.0e-10):
            raise ValueError(f"Rotated-state probability sum is {probability_sum}")
        mean = float(probabilities @ fragment.diagonal)
        second = float(probabilities @ np.square(fragment.diagonal))
        approximate_expectation += mean
        if int(count) > 0:
            raw_variance = second - mean * mean
            if raw_variance < -1.0e-10:
                raise ValueError(
                    f"Materially negative rotated-diagonal variance {raw_variance}"
                )
            variance += max(raw_variance, 0.0) / int(count)
            active_settings += 1
        elif fragment.centered_range > RANGE_TOL:
            raise AssertionError("Nonconstant approximate fragment received no shots")
    bias = approximate_expectation - case.target_expectation
    mse = bias * bias + variance
    return {
        "mse": float(max(mse, 0.0)),
        "rmse": math.sqrt(max(mse, 0.0)),
        "bias": float(bias),
        "bias_squared": float(bias * bias),
        "variance": float(variance),
        "covered_terms": "",
        "total_terms": "",
        "distinct_bases": active_settings,
        "wildcard_hit_pairs": 0,
        "approximate_expectation": approximate_expectation,
    }


def approximation_cache_path(case: BenchmarkCase) -> Path:
    return APPROX_CACHE_DIR / f"{case.slug}.pkl"


def approximation_cache_signature(case: BenchmarkCase) -> str:
    return hashlib.sha256(
        (
            PROFILE_VERSION
            + case.target_hash
            + str(AGPD_FROZEN_STARTS)
            + str(GENERAL_K_SAFETY_GUARD)
            + str(FRONTIER_STALL_PATIENCE)
            + str(FRONTIER_RELATIVE_IMPROVEMENT)
            + str(FRONTIER_MINIMUM_K)
            + str(AGPD_DEPTHS)
            + str(SRCDF_DEPTHS)
            + str(SRCDF_STARTS_PER_FACTOR)
            + str(SRCDF_RIDGE)
            + SRCDF_JOINT_SOLVER_VERSION
        ).encode()
    ).hexdigest()


def load_approximation_cache(
    case: BenchmarkCase, force: bool
) -> tuple[list[ApproximationPoint], list[ApproximationPoint], list[dict[str, object]]] | None:
    path = approximation_cache_path(case)
    if force or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            saved = pickle.load(handle)
        if saved.get("signature") != approximation_cache_signature(case):
            return None
        return saved["agpd"], saved["srcdf"], saved["srcdf_trials"]
    except Exception:
        return None


def save_approximation_cache(
    case: BenchmarkCase,
    agpd: list[ApproximationPoint],
    srcdf: list[ApproximationPoint],
    srcdf_trials: list[dict[str, object]],
) -> None:
    path = approximation_cache_path(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pkl")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "signature": approximation_cache_signature(case),
                "profile": PROFILE_VERSION,
                "agpd": agpd,
                "srcdf": srcdf,
                "srcdf_trials": srcdf_trials,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(path)


def approximation_rows_for_case(
    case: BenchmarkCase,
    pool_cache: dict[
        int, tuple[list[CircuitCandidate], dict[int, list[CircuitCandidate]]]
    ],
    force: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cached = load_approximation_cache(case, force)
    if cached is None:
        if case.n_qubits not in pool_cache:
            print(f"Building frozen circuit pools for n={case.n_qubits}", flush=True)
            pool_cache[case.n_qubits] = (
                build_agpd_pool(case.n_qubits),
                build_srcdf_pools(case.n_qubits),
            )
        agpd_pool, srcdf_pools = pool_cache[case.n_qubits]
        print(f"  {case.slug}: AGPD residual search", flush=True)
        agpd = run_agpd_frontier(case.hamiltonian, agpd_pool, case.shots)
        print(f"  {case.slug}: rank-adaptive joint-refit SRDD search", flush=True)
        srcdf, srcdf_trials = run_srcdf_frontier(
            case.hamiltonian,
            srcdf_pools,
            SRCDF_RIDGE,
            case.shots,
        )
        save_approximation_cache(case, agpd, srcdf, srcdf_trials)
    else:
        agpd, srcdf, srcdf_trials = cached
        print(f"  {case.slug}: loaded approximation cache", flush=True)

    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for point in agpd:
        last = point.trace[-1]
        audit.append(
            {
                "benchmark": case.benchmark,
                "instance": case.instance,
                "slug": case.slug,
                "method": "AGPD",
                "k": point.k,
                "selected_depth": point.depth,
                "selected_family": last["selected_family"],
                "selected_variant": last["selected_variant"],
                "residual_frobenius": point.residual_frobenius,
                "objective": point.objective,
                "exact_constant": point.exact_constant,
                "active_settings": sum(
                    item.centered_range > RANGE_TOL for item in point.fragments
                ),
                "unitary_hashes": json.dumps(
                    [item.unitary_hash for item in point.fragments]
                ),
                "trace_json": json.dumps(point.trace, sort_keys=True),
                "frontier_terminal_k": point.frontier_terminal_k,
                "frontier_stop_reason": point.frontier_stop_reason,
                "frontier_right_censored": point.frontier_right_censored,
                "frontier_safety_guard": point.frontier_safety_guard,
                "fit_scope": "four-family residual search with rank-adaptive stopping",
            }
        )
    best_srcdf_by_k = {
        k: min(
            (item for item in srcdf if item.k == k),
            key=lambda item: (item.residual_frobenius, item.depth),
        )
        for k in sorted({item.k for item in srcdf})
    }
    for point in srcdf:
        trial = next(
            item
            for item in srcdf_trials
            if int(item["k"]) == point.k and int(item["depth"]) == point.depth
        )
        selected_trial = best_srcdf_by_k[point.k] is point
        audit.append(
            {
                    "benchmark": case.benchmark,
                    "instance": case.instance,
                    "slug": case.slug,
                    "method": SRDD_METHOD,
                    "k": point.k,
                    "depth": point.depth,
                    "selected_depth": point.depth,
                    "selected_for_rank_frontier": selected_trial,
                    "selected_family": "srcdf_matrix",
                    "selected_variant": "joint",
                    "residual_frobenius": point.residual_frobenius,
                    "objective": point.objective,
                    "exact_constant": point.exact_constant,
                    "active_settings": sum(
                        item.centered_range > RANGE_TOL for item in point.fragments
                    ),
                    "unitary_hashes": json.dumps(trial["selected_unitary_hashes"]),
                    "trace_json": json.dumps(trial["trace"], sort_keys=True),
                    "frontier_terminal_k": point.frontier_terminal_k,
                    "frontier_stop_reason": point.frontier_stop_reason,
                    "frontier_right_censored": point.frontier_right_censored,
                    "frontier_safety_guard": point.frontier_safety_guard,
                    "fit_scope": (
                        "rank-adaptive shallow-factor search with simultaneous ridge refit"
                    ),
                }
            )

    for method, frontier in (("AGPD", agpd), (SRDD_METHOD, srcdf)):
        for shots in case.shots:
            point, allocation, ranges, sampling_proxy, score = select_approximation_point(
                frontier, shots, len(case.state)
            )
            result = approximate_estimator_mse(case, point, allocation)
            rows.append(
                {
                    "benchmark": case.benchmark,
                    "instance": case.instance,
                    "slug": case.slug,
                    "method": method,
                    "shots": shots,
                    **result,
                    "selected_k": point.k,
                    "selected_depth": point.depth,
                    "settings": int(np.count_nonzero(allocation)),
                    "protocol": "balanced fixed-shot largest-remainder range allocation",
                    "selection_proxy": score,
                    "approximation_proxy": point.residual_frobenius
                    / math.sqrt(len(case.state)),
                    "sampling_proxy": sampling_proxy,
                    "residual_frobenius": point.residual_frobenius,
                    "frontier_terminal_k": point.frontier_terminal_k,
                    "frontier_stop_reason": point.frontier_stop_reason,
                    "frontier_right_censored": point.frontier_right_censored,
                    "frontier_safety_guard": point.frontier_safety_guard,
                    "allocation_json": json.dumps(allocation.tolist()),
                    "centered_ranges_json": json.dumps(ranges.tolist()),
                    "selected_families_json": json.dumps(
                        [item.family for item in point.fragments]
                    ),
                    "selected_unitary_hashes_json": json.dumps(
                        [item.unitary_hash for item in point.fragments]
                    ),
                }
            )
    return rows, audit


def summarize_instance_rows(
    rows: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for benchmark in BENCHMARK_ORDER:
        for method in METHOD_ORDER:
            shot_grid = STRUCTURED_SHOTS if benchmark == "structured" else RANDOM_SHOTS
            for shots in shot_grid:
                selected = [
                    row
                    for row in rows
                    if row["benchmark"] == benchmark
                    and row["method"] == method
                    and int(row["shots"]) == shots
                ]
                expected = 1 if benchmark == "structured" else 5
                if len(selected) != expected:
                    raise ValueError(
                        f"Expected {expected} rows for {benchmark}/{method}/T={shots}, "
                        f"found {len(selected)}"
                    )
                rmses = np.asarray([float(row["rmse"]) for row in selected])
                mses = np.asarray([float(row["mse"]) for row in selected])
                summary.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "shots": shots,
                        # Match the historical random-panel convention: average
                        # instance RMSE only after each Hamiltonian is evaluated.
                        "rmse": float(np.mean(rmses)),
                        "rmse_instance_sd": float(np.std(rmses, ddof=1))
                        if len(rmses) > 1
                        else 0.0,
                        "mean_instance_mse": float(np.mean(mses)),
                        "pooled_rmse": math.sqrt(float(np.mean(mses))),
                        "instances": len(selected),
                        "source_status": (
                            "canonical TFIM exact ground state"
                            if benchmark == "structured"
                            else "same-protocol seed-0 regenerated surrogate"
                        ),
                    }
                )
    return summary


def self_test(cases: Sequence[BenchmarkCase]) -> dict[str, object]:
    identity = np.eye(16, dtype=np.complex128)
    test = left_apply_gate(identity, givens(0.31), (0, 2), 4)
    unitarity = float(np.linalg.norm(test.conj().T @ test - identity))
    if unitarity > 1.0e-11:
        raise AssertionError(f"Gate embedding self-test failed: {unitarity}")
    random_case = next(item for item in cases if item.benchmark == "sparse")
    observables, weights, _ = split_identity(
        random_case.observables, random_case.weights
    )
    _, ap_probabilities = ap_exact_setting_distribution(observables, weights)
    ap_sum = float(np.sum(ap_probabilities))
    if not np.isclose(ap_sum, 1.0, atol=2.0e-12):
        raise AssertionError(f"AP distribution self-test failed: {ap_sum}")
    tfim = cases[0]
    expected_terms = 2 * tfim.n_qubits - 1
    nonidentity = int(np.count_nonzero(np.any(tfim.observables != 0, axis=1)))
    if nonidentity != expected_terms:
        raise AssertionError(
            f"TFIM must have {expected_terms} nonidentity terms, found {nonidentity}"
        )
    expected_tfim = np.zeros_like(tfim.hamiltonian)
    for qubit in range(tfim.n_qubits):
        code = np.zeros(tfim.n_qubits, dtype=np.int8)
        code[qubit] = 1
        expected_tfim += pauli_dense(code[None, :], np.asarray([-1.0]))
    for qubit in range(tfim.n_qubits - 1):
        code = np.zeros(tfim.n_qubits, dtype=np.int8)
        code[qubit : qubit + 2] = 3
        expected_tfim += pauli_dense(code[None, :], np.asarray([-1.0]))
    tfim_definition_error = float(np.linalg.norm(expected_tfim - tfim.hamiltonian))
    if tfim_definition_error > 2.0e-10:
        raise AssertionError(f"Explicit TFIM definition mismatch: {tfim_definition_error}")
    return {
        "gate_embedding_unitarity_error": unitarity,
        "ap_distribution_probability_sum": ap_sum,
        "tfim_nonidentity_terms": nonidentity,
        "tfim_ground_energy": tfim.target_expectation,
        "tfim_explicit_definition_frobenius_error": tfim_definition_error,
        "random_common_state_hash": array_sha256(random_case.state),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="processes used for retained stateful schedule generation",
    )
    parser.add_argument("--force-schedules", action="store_true")
    parser.add_argument("--force-approximations", action="store_true")
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="validate inputs, gates, and AP enumeration without running production",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    tests = self_test(cases)
    print(json.dumps({"self_test": tests}, indent=2), flush=True)
    if args.self_test_only:
        return

    schedules = prepare_stateful_schedules(
        cases, max(1, int(args.workers)), bool(args.force_schedules)
    )
    instance_rows: list[dict[str, object]] = []
    pauli_audit: list[dict[str, object]] = []
    for case in cases:
        print(f"Evaluating exact Pauli baselines: {case.slug}", flush=True)
        rows, audit = pauli_rows_for_case(case, schedules)
        instance_rows.extend(rows)
        pauli_audit.extend(audit)

    pool_cache: dict[
        int, tuple[list[CircuitCandidate], dict[int, list[CircuitCandidate]]]
    ] = {}
    approximation_audit: list[dict[str, object]] = []
    for case in cases:
        print(f"Evaluating new matrix approximations: {case.slug}", flush=True)
        rows, audit = approximation_rows_for_case(
            case, pool_cache, bool(args.force_approximations)
        )
        instance_rows.extend(rows)
        approximation_audit.extend(audit)

    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    benchmark_rank = {
        benchmark: index for index, benchmark in enumerate(BENCHMARK_ORDER)
    }
    instance_rows.sort(
        key=lambda row: (
            benchmark_rank[str(row["benchmark"])],
            int(row["instance"]),
            method_rank[str(row["method"])],
            int(row["shots"]),
        )
    )
    approximation_audit.sort(
        key=lambda row: (
            benchmark_rank[str(row["benchmark"])],
            int(row["instance"]),
            method_rank[str(row["method"])],
            int(row["k"]),
        )
    )
    pauli_audit.sort(
        key=lambda row: (
            benchmark_rank[str(row["benchmark"])],
            str(row["slug"]),
            method_rank[str(row["method"])],
            int(row["shots"]),
            int(row["schedule_repeat"]),
        )
    )
    summary = summarize_instance_rows(instance_rows)
    summary_keys = [
        (str(row["benchmark"]), str(row["method"]), int(row["shots"]))
        for row in summary
    ]
    if len(summary_keys) != len(set(summary_keys)):
        raise AssertionError("Duplicate aggregate (benchmark,method,shots) key")
    for row in summary:
        value = float(row["rmse"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid aggregate RMSE row: {row}")
    write_csv(INSTANCE_CSV, instance_rows)
    write_csv(APPROX_CSV, approximation_audit)
    write_csv(PAULI_CSV, pauli_audit)
    write_csv(SUMMARY_CSV, summary)

    source_hashes: dict[str, str] = {}
    for case in cases:
        source_hashes.update(case.source_hashes)
    elapsed = time.perf_counter() - started
    outputs = (SUMMARY_CSV, INSTANCE_CSV, APPROX_CSV, PAULI_CSV)
    schedule_hashes = {
        str(path.relative_to(OUTPUT)): sha256(path)
        for path in sorted(SCHEDULE_DIR.glob("*.npz"))
    }
    approximation_cache_hashes = {
        str(path.relative_to(OUTPUT)): sha256(path)
        for path in sorted(APPROX_CACHE_DIR.glob("*.pkl"))
    }
    case_audit = [
        {
            "benchmark": case.benchmark,
            "slug": case.slug,
            "instance": case.instance,
            "n_qubits": case.n_qubits,
            "matrix_shape": list(case.hamiltonian.shape),
            "target_hash": case.target_hash,
            "state_hash": array_sha256(case.state),
            "target_expectation": case.target_expectation,
            "pauli_terms": len(case.observables),
            "identity_terms": int(
                np.count_nonzero(~np.any(case.observables != 0, axis=1))
            ),
        }
        for case in cases
    ]
    manifest = {
        "profile_version": PROFILE_VERSION,
        "created_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "command": sys.argv,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "workers": int(args.workers),
        },
        "benchmarks": {
            "structured": {
                "definition": "open-boundary TFIM H=-sum_i X_i-sum_i Z_i Z_{i+1}, n=8, J=g=1",
                "shots": list(STRUCTURED_SHOTS),
                "state": "canonical exact unique ground state",
            },
            "sparse": {
                "definition": "five seed-0 regenerated sparse four-qubit matrices",
                "shots": list(RANDOM_SHOTS),
                "state": "common saved rank-one state",
            },
            "dense": {
                "definition": "five seed-0 regenerated dense four-qubit matrices",
                "shots": list(RANDOM_SHOTS),
                "state": "common saved rank-one state",
            },
        },
        "pauli_protocol": {
            "LCS": "exact state-conditioned uniform local-Pauli channel formula",
            "SG": "one preregistered retained schedule generated to max T; prefixes reused",
            "Derand": "one preregistered retained schedule generated to max T; prefixes reused",
            "OGM": (
                "archived greedy candidate/SLSQP objective; one preregistered MATLAB-rule "
                "integer allocation per T; the old allocation is intrinsically T-specific"
            ),
            "AP": (
                f"mean exact conditional MSE over {AP_REPEATS} preregistered schedules; "
                "n=4 uses exact enumeration of the retained one-setting law, n=8 uses the retained generator"
            ),
            "outcomes": "Born noise integrated exactly; no finite outcome Monte Carlo",
        },
        "approximation_protocol": {
            "common": {
                "representation": "exact scalar plus shallow-unitary rotated full diagonals",
                "allocation": (
                    "one shot per nonconstant setting, then proportional centered half-range "
                    "largest-remainder allocation"
                ),
                "selection_proxy": (
                    "sqrt((Frobenius residual/sqrt(D))^2 + sum_i range_i^2/T_i); "
                    "exact target state is excluded from selection"
                ),
                "selection_state_withheld": True,
                "rank_selection": (
                    "for each T, minimize L_{K,T} over the generated frontier and break ties "
                    "at the smallest K; continue K until no global-best L_{K,T} improves by "
                    f"more than {FRONTIER_RELATIVE_IMPROVEMENT:g} for "
                    f"{FRONTIER_STALL_PATIENCE} consecutive ranks"
                ),
                "minimum_rank_before_stopping_test": FRONTIER_MINIMUM_K,
                "infinite_loop_safety_guard": GENERAL_K_SAFETY_GUARD,
                "guard_policy": "abort rather than publish if the safety guard is reached",
            },
            "AGPD": {
                "families": ["GFRO", "Operator pool", "NN-Pair", "iSWAP+SU(2)"],
                "family_scope": {
                    "GFRO": "blocked-spin nearest-neighbor Givens even/odd macro-layers",
                    "Operator pool": (
                        "support-disjoint Givens, XX+YY, XX, YY, and ZZ rotations; "
                        "bonds remain inside the two blocked-spin chains"
                    ),
                    "NN-Pair": "blocked-spin Givens layers plus ordered nearest-neighbor pair transfer",
                    "iSWAP+SU(2)": (
                        "full-qubit-chain brick wall with every entangler fixed at iSWAP "
                        "(XY angle -pi/2), followed by frozen-candidate local Rx-Ry-Rx rotations"
                    ),
                },
                "depths": list(AGPD_DEPTHS),
                "frozen_starts_per_family_depth_before_deduplication": AGPD_FROZEN_STARTS,
                "fit": (
                    "adaptive exact residual projection over the four-family deterministic "
                    "multistart pool; K is selected by the common fixed-shot stopping rule"
                ),
            },
            SRDD_METHOD: {
                "scope": (
                    "SRDD nonfermionic matrix extension; not electron-repulsion RC-DF"
                ),
                "circuit_profile": (
                    "parametric shallow XY+local-SU(2) brick wall, including the zero-entangler "
                    "submanifold; this pool is independent of the fixed-iSWAP AGPD family"
                ),
                "depths": list(SRCDF_DEPTHS),
                "new_multistarts_per_depth_and_factor": SRCDF_STARTS_PER_FACTOR,
                "ridge": SRCDF_RIDGE,
                "joint_solver": "dense simultaneous ridge normal-equation solve with centered KKT audit",
                "fit": (
                    "each rank adds fresh deterministic multistarts, greedily selects one factor "
                    "per depth, and simultaneously refits all retained centered diagonals"
                ),
            },
            "finite_search_limitation": (
                "The circuit-angle search uses deterministic multistart profiles rather than a "
                "global continuous optimizer.  This controls nonlinear fitting cost but does not "
                "cap the retained-factor rank K."
            ),
            "legacy_exclusion": "No old GPD, PND, or TND curve or fragment is loaded or renamed.",
        },
        "self_test": tests,
        "case_audit": case_audit,
        "source_hashes": source_hashes,
        "implementation_hashes": {
            "runner": sha256(Path(__file__).resolve()),
            "retained_measurement_wrapper": sha256(
                NEW_DATA / "count_measurement_settings_n2038.py"
            ),
            "retained_measurement_schemes": sha256(
                SHADOW / "shadowgrouping" / "measurement_schemes.py"
            ),
            "retained_matlab_ogm_sampler": sha256(
                NEW_DATA / "gen_errors_all_algs" / "Sample_main.m"
            ),
        },
        "cache_hashes": {
            "schedules": schedule_hashes,
            "approximations": approximation_cache_hashes,
        },
        "integrity_flags": {
            "all_shots_conserved": True,
            "same_target_state_enforced": True,
            "selection_state_withheld": True,
            "historical_curve_values_reused": False,
            "historical_style_only": True,
            "no_7256_relabeling": True,
            "no_8k_local_substitution": True,
        },
        "outputs": {str(path): sha256(path) for path in outputs},
        "row_counts": {
            "summary": len(summary),
            "instance": len(instance_rows),
            "approximation_audit": len(approximation_audit),
            "pauli_audit": len(pauli_audit),
        },
    }
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "summary": str(SUMMARY_CSV),
                "audit_output": str(OUTPUT),
                "elapsed_seconds": elapsed,
                "summary_rows": len(summary),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
