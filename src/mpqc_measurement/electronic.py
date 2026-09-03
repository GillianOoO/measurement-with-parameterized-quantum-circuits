"""Small-system dense helpers for real spin-restricted electronic inputs."""

from __future__ import annotations

import itertools

import numpy as np

from .models import ElectronicIntegrals


def annihilation_operator(index: int, n_modes: int) -> np.ndarray:
    """Jordan--Wigner annihilation matrix with q0 as the most-significant bit."""
    dimension = 2**n_modes
    output = np.zeros((dimension, dimension), dtype=np.complex128)
    shift = n_modes - 1 - int(index)
    for column in range(dimension):
        if ((column >> shift) & 1) == 0:
            continue
        preceding = sum((column >> (n_modes - 1 - q)) & 1 for q in range(index))
        row = column & ~(1 << shift)
        output[row, column] = -1.0 if preceding % 2 else 1.0
    return output


def spin_free_excitations(spatial_orbitals: int) -> np.ndarray:
    """Return E[p,q] for blocked alpha-then-beta spin-orbital ordering."""
    m = int(spatial_orbitals)
    n_modes = 2 * m
    annihilators = [annihilation_operator(index, n_modes) for index in range(n_modes)]
    creators = [value.conj().T for value in annihilators]
    dimension = 2**n_modes
    output = np.empty((m, m, dimension, dimension), dtype=np.complex128)
    for p in range(m):
        for q in range(m):
            output[p, q] = creators[p] @ annihilators[q] + creators[m + p] @ annihilators[m + q]
    return output


def one_body_dense(matrix: np.ndarray, excitations: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    m = values.shape[0]
    e = spin_free_excitations(m) if excitations is None else excitations
    output = np.einsum("pq,pqab->ab", values, e, optimize=True)
    return (output + output.conj().T) / 2.0


def electronic_hamiltonian_dense(integrals: ElectronicIntegrals) -> np.ndarray:
    """Materialize the spin-free chemist-order convention documented by the package."""
    m = integrals.spatial_orbitals
    excitations = spin_free_excitations(m)
    dimension = excitations.shape[-1]
    output = float(integrals.constant) * np.eye(dimension, dtype=np.complex128)
    output += np.einsum("pq,pqab->ab", integrals.one_body, excitations, optimize=True)
    for p, q, r, s in itertools.product(range(m), repeat=4):
        coefficient = 0.5 * float(integrals.two_body_chemist[p, q, r, s])
        if coefficient == 0.0:
            continue
        term = excitations[p, q] @ excitations[r, s]
        if q == r:
            term = term - excitations[p, s]
        output += coefficient * term
    return (output + output.conj().T) / 2.0


def spatial_occupations(spatial_orbitals: int) -> np.ndarray:
    m = int(spatial_orbitals)
    dimension = 2 ** (2 * m)
    output = np.zeros((dimension, m), dtype=float)
    for index in range(dimension):
        bits = f"{index:0{2 * m}b}"
        for orbital in range(m):
            output[index, orbital] = int(bits[orbital]) + int(bits[m + orbital])
    return output


def _one_spin_fock_unitary(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    m = matrix.shape[0]
    dimension = 2**m
    output = np.zeros((dimension, dimension), dtype=np.complex128)
    subsets = {
        number: [
            tuple(index for index, bit in enumerate(f"{value:0{m}b}") if bit == "1")
            for value in range(dimension)
            if value.bit_count() == number
        ]
        for number in range(m + 1)
    }
    subset_to_index = {
        tuple(index for index, bit in enumerate(f"{value:0{m}b}") if bit == "1"): value
        for value in range(dimension)
    }
    for number, sector in subsets.items():
        if number == 0:
            output[0, 0] = 1.0
            continue
        for occupied_out in sector:
            row = subset_to_index[occupied_out]
            for occupied_in in sector:
                column = subset_to_index[occupied_in]
                output[row, column] = np.linalg.det(
                    matrix[np.ix_(occupied_out, occupied_in)]
                )
    return output


def fock_orbital_unitary(rotation: np.ndarray) -> np.ndarray:
    """Return the blocked-spin Fock unitary induced by a spatial rotation."""
    one_spin = _one_spin_fock_unitary(rotation)
    output = np.kron(one_spin, one_spin)
    error = float(np.linalg.norm(output.conj().T @ output - np.eye(len(output))))
    if error > 2.0e-8 * max(1.0, np.sqrt(len(output))):
        raise ValueError(f"spatial rotation did not induce a unitary: {error:.3e}")
    return output


def hartree_fock_state(spatial_orbitals: int, n_electrons: int) -> np.ndarray:
    if n_electrons % 2:
        raise ValueError("the SRDD electronic backend requires a closed-shell even electron count")
    m = int(spatial_orbitals)
    n_alpha = n_electrons // 2
    if n_alpha > m:
        raise ValueError("too many electrons for the spatial orbital count")
    occupied = list(range(n_alpha)) + [m + index for index in range(n_alpha)]
    bits = [0] * (2 * m)
    for index in occupied:
        bits[index] = 1
    basis_index = int("".join(str(value) for value in bits), 2)
    state = np.zeros(2 ** (2 * m), dtype=np.complex128)
    state[basis_index] = 1.0
    return state
