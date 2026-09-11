"""Dependency-light exact CNOT compilation/noise primitives for Fig. 3.

The module deliberately uses NumPy only so that the fixed SRDD gate contract
can be unit-tested independently of the tensor-network runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


CNOTS_PER_GIVENS = 2
CHANNELS_PER_GIVENS = CNOTS_PER_GIVENS


@dataclass(frozen=True)
class GateOperation:
    name: str
    qubits: tuple[int, ...]
    unitary: np.ndarray


def fock_gate(one_particle: np.ndarray) -> np.ndarray:
    gate = np.zeros((4, 4), dtype=np.complex128)
    gate[0, 0] = 1.0
    gate[3, 3] = np.linalg.det(one_particle)
    # Dense order is |00>, |01>, |10>, |11>; orbital order is p,q.
    indices = [2, 1]
    gate[np.ix_(indices, indices)] = one_particle
    return gate


def unitary_equivalence_error(left: np.ndarray, right: np.ndarray) -> float:
    """Frobenius error after removing an otherwise irrelevant global phase."""
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    overlap = complex(np.vdot(left, right))
    phase = 1.0 if abs(overlap) == 0.0 else overlap / abs(overlap)
    return float(np.linalg.norm(phase * left - right))


def _rz(phi: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


def _ry(phi: float) -> np.ndarray:
    c, s = math.cos(0.5 * phi), math.sin(0.5 * phi)
    return np.asarray([[c, -s], [s, c]], dtype=np.complex128)


def fixed_two_cx_givens_operations(theta: float) -> tuple[GateOperation, ...]:
    """Explicit fixed definition of ``XXPlusYY(2 theta, pi/2)``.

    Qubits are ordered as the first and second tensor factors of
    ``|00>,|01>,|10>,|11>``.  This is a CNOT--single-qubit--CNOT template with
    ideal one-qubit gates.  It intentionally retains both CNOTs at theta=0.
    """
    identity = np.eye(2, dtype=np.complex128)
    s = np.diag([1.0, 1.0j])
    sdg = s.conjugate().T
    sx = 0.5 * np.asarray(
        [[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]],
        dtype=np.complex128,
    )
    sxdg = sx.conjugate().T
    cx = np.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )

    def on_first(name: str, unitary: np.ndarray) -> GateOperation:
        return GateOperation(name, (0,), np.kron(unitary, identity))

    def on_second(name: str, unitary: np.ndarray) -> GateOperation:
        return GateOperation(name, (1,), np.kron(identity, unitary))

    return (
        on_second("rz", _rz(math.pi / 2.0)),
        on_first("sdg", sdg),
        on_first("sx", sx),
        on_first("s", s),
        on_second("s", s),
        GateOperation("cx", (0, 1), cx),
        on_first("ry", _ry(-float(theta))),
        on_second("ry", _ry(-float(theta))),
        GateOperation("cx", (0, 1), cx),
        on_second("sdg", sdg),
        on_first("sdg", sdg),
        on_first("sxdg", sxdg),
        on_first("s", s),
        on_second("rz", _rz(-math.pi / 2.0)),
    )


def operation_product(operations: tuple[GateOperation, ...]) -> np.ndarray:
    result = np.eye(4, dtype=np.complex128)
    for operation in operations:
        result = operation.unitary @ result
    return result


def fixed_two_cx_givens_unitary(theta: float) -> tuple[np.ndarray, float]:
    operations = fixed_two_cx_givens_operations(theta)
    two_qubit_names = [operation.name for operation in operations if len(operation.qubits) == 2]
    if two_qubit_names != ["cx", "cx"]:
        raise RuntimeError(
            "The fixed fermionic-Givens definition is no longer CNOT--1q--CNOT: "
            f"{two_qubit_names}"
        )
    compiled = operation_product(operations)
    c, s = math.cos(float(theta)), math.sin(float(theta))
    target = fock_gate(np.asarray([[c, s], [-s, c]], dtype=float))
    error = unitary_equivalence_error(compiled, target)
    if error > 2.0e-12:
        raise RuntimeError(f"Two-CNOT Givens synthesis changed the unitary: {error:.3e}")
    return compiled, error


def composed_depolarizing_probability(p: float, channel_count: int) -> float:
    """Compose identical full depolarizing channels on the same support."""
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Depolarizing probability must lie in [0,1], got {p}")
    if channel_count < 0:
        raise ValueError("Channel count must be nonnegative")
    return float(1.0 - (1.0 - p) ** int(channel_count))


def gate_superoperator(unitary: np.ndarray, p: float) -> np.ndarray:
    """Adjoint of depolarization-after-unitary on two neighboring qubits."""
    result = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    identity = np.eye(4, dtype=np.complex128)
    for in1 in range(4):
        u1, d1 = divmod(in1, 2)
        for in2 in range(4):
            u2, d2 = divmod(in2, 2)
            basis = np.zeros((4, 4), dtype=np.complex128)
            basis[2 * u1 + u2, 2 * d1 + d2] = 1.0
            noisy = (1.0 - p) * basis + (p * np.trace(basis) / 4.0) * identity
            transformed = unitary.conjugate().T @ noisy @ unitary
            for out1 in range(4):
                v1, e1 = divmod(out1, 2)
                for out2 in range(4):
                    v2, e2 = divmod(out2, 2)
                    result[out1, out2, in1, in2] = transformed[
                        2 * v1 + v2, 2 * e1 + e2
                    ]
    return result
