"""Pauli input conversion and the SRDD universal residual completion."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence

import numpy as np

from .models import Fragment


I2 = np.eye(2, dtype=np.complex128)
X = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
HADAMARD = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)
S_DAGGER = np.diag([1.0, -1j]).astype(np.complex128)
PAULI_MATRIX = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_all(factors: Sequence[np.ndarray]) -> np.ndarray:
    output = np.asarray([[1.0 + 0.0j]])
    for factor in factors:
        output = np.kron(output, factor)
    return output


def pauli_matrix(word: str) -> np.ndarray:
    normalized = str(word).upper()
    if not normalized or any(symbol not in PAULI_MATRIX for symbol in normalized):
        raise ValueError(f"invalid Pauli word {word!r}")
    return kron_all([PAULI_MATRIX[symbol] for symbol in normalized])


def pauli_terms_to_dense(
    n_qubits: int, terms: Iterable[tuple[str, complex]]
) -> np.ndarray:
    if int(n_qubits) < 1:
        raise ValueError("n_qubits must be positive")
    dimension = 2 ** int(n_qubits)
    output = np.zeros((dimension, dimension), dtype=np.complex128)
    for word, coefficient in terms:
        normalized = str(word).upper()
        if len(normalized) != int(n_qubits):
            raise ValueError(
                f"Pauli word {word!r} has length {len(normalized)}, expected {n_qubits}"
            )
        value = complex(coefficient)
        if abs(value.imag) > 1.0e-10:
            raise ValueError("Hermitian Pauli coefficients must be real")
        output += value.real * pauli_matrix(normalized)
    return (output + output.conj().T) / 2.0


def dense_to_pauli(
    matrix: np.ndarray, *, coefficient_tolerance: float = 1.0e-12
) -> list[tuple[str, float]]:
    values = np.asarray(matrix, dtype=np.complex128)
    dimension = values.shape[0]
    n_qubits = int(round(math.log2(dimension)))
    if values.shape != (dimension, dimension) or 2**n_qubits != dimension:
        raise ValueError("matrix must have dimension 2**n")
    terms: list[tuple[str, float]] = []
    for symbols in itertools.product("IXYZ", repeat=n_qubits):
        word = "".join(symbols)
        coefficient = np.trace(pauli_matrix(word) @ values) / dimension
        if abs(coefficient.imag) > 2.0e-9:
            raise ValueError("dense Hermitian matrix produced a complex Pauli coefficient")
        if abs(coefficient.real) > coefficient_tolerance:
            terms.append((word, float(coefficient.real)))
    return terms


def _compatible(pattern: list[str], word: str) -> bool:
    return all(left == "I" or right == "I" or left == right for left, right in zip(pattern, word))


def _merge_pattern(pattern: list[str], word: str) -> list[str]:
    return [right if left == "I" else left for left, right in zip(pattern, word)]


def group_product_paulis(
    terms: Sequence[tuple[str, float]],
) -> list[tuple[str, list[tuple[str, float]]]]:
    """Deterministic coefficient-sorted qubit-wise commuting grouping."""
    ordered = sorted(
        ((str(word).upper(), float(coefficient)) for word, coefficient in terms),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    groups: list[tuple[list[str], list[tuple[str, float]]]] = []
    for word, coefficient in ordered:
        if set(word) == {"I"}:
            continue
        placed = False
        for index, (pattern, members) in enumerate(groups):
            if _compatible(pattern, word):
                groups[index] = (_merge_pattern(pattern, word), [*members, (word, coefficient)])
                placed = True
                break
        if not placed:
            groups.append((list(word), [(word, coefficient)]))
    output = []
    for pattern, members in groups:
        basis = "".join("Z" if symbol == "I" else symbol for symbol in pattern)
        output.append((basis, members))
    return output


def basis_change_unitary(basis: str) -> np.ndarray:
    local = {
        "X": HADAMARD,
        "Y": HADAMARD @ S_DAGGER,
        "Z": I2,
    }
    normalized = str(basis).upper()
    if not normalized or any(symbol not in local for symbol in normalized):
        raise ValueError(f"invalid product-Pauli basis {basis!r}")
    return kron_all([local[symbol] for symbol in normalized])


def group_fragment(
    basis: str, members: Sequence[tuple[str, float]], *, label: str
) -> Fragment:
    n_qubits = len(basis)
    dimension = 2**n_qubits
    diagonal = np.zeros(dimension, dtype=float)
    for index in range(dimension):
        bits = f"{index:0{n_qubits}b}"
        value = 0.0
        for word, coefficient in members:
            eigenvalue = 1.0
            for bit, symbol in zip(bits, word):
                if symbol != "I" and bit == "1":
                    eigenvalue *= -1.0
            value += float(coefficient) * eigenvalue
        diagonal[index] = value
    return Fragment(
        unitary=basis_change_unitary(basis),
        diagonal=diagonal,
        label=label,
        metadata={"basis": basis, "pauli_terms": list(members)},
    )


def universal_completion_frontier(
    matrix: np.ndarray,
    *,
    coefficient_tolerance: float = 1.0e-12,
) -> tuple[float, list[Fragment]]:
    """Return identity coefficient and ordered product-Pauli residual groups."""
    terms = dense_to_pauli(matrix, coefficient_tolerance=coefficient_tolerance)
    identity_word = "I" * int(round(math.log2(len(matrix))))
    constant = sum(coefficient for word, coefficient in terms if word == identity_word)
    nonidentity = [(word, coefficient) for word, coefficient in terms if word != identity_word]
    groups = group_product_paulis(nonidentity)
    fragments = [
        group_fragment(basis, members, label=f"residual-group-{index:03d}")
        for index, (basis, members) in enumerate(groups, start=1)
    ]
    return float(constant), fragments
