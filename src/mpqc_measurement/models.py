"""Shared validated data models for GPD and SRDD."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


HERMITIAN_TOLERANCE = 1.0e-9
UNITARY_TOLERANCE = 5.0e-6


def _strict_integer(value: Any, *, name: str) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer")
    return int(raw)


def _finite_real_scalar(value: Any, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _finite_real_array(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numeric values")
    values = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _as_hermitian(matrix: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(matrix)
    if raw.dtype.kind not in "biufc":
        raise ValueError(f"{name} must contain numeric values")
    values = np.asarray(raw, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square matrix; received {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    error = float(np.linalg.norm(values - values.conj().T, ord="fro"))
    scale = max(1.0, float(np.linalg.norm(values, ord="fro")))
    if error > HERMITIAN_TOLERANCE * scale:
        raise ValueError(f"{name} is not Hermitian: relative error {error / scale:.3e}")
    return (values + values.conj().T) / 2.0


@dataclass(frozen=True)
class ElectronicIntegrals:
    """Real spin-restricted electronic input in chemist tensor order.

    ``two_body_chemist[p,q,r,s]`` multiplies
    ``(E_pq E_rs - delta_qr E_ps)/2``.
    """

    constant: float
    one_body: np.ndarray
    two_body_chemist: np.ndarray
    n_electrons: int

    def __post_init__(self) -> None:
        constant = _finite_real_scalar(self.constant, name="constant")
        one = _finite_real_array(self.one_body, name="one_body")
        two = _finite_real_array(
            self.two_body_chemist, name="two_body_chemist"
        )
        if one.ndim != 2 or one.shape[0] != one.shape[1]:
            raise ValueError("one_body must be a real square matrix")
        m = one.shape[0]
        if two.shape != (m, m, m, m):
            raise ValueError(
                f"two_body_chemist must have shape {(m, m, m, m)}; received {two.shape}"
            )
        n_electrons = _strict_integer(self.n_electrons, name="n_electrons")
        if not 0 <= n_electrons <= 2 * m:
            raise ValueError("n_electrons must lie between 0 and 2*m")
        if n_electrons % 2:
            raise ValueError("n_electrons must be even for a closed-shell input")
        if not np.allclose(one, one.T, atol=1.0e-10, rtol=0.0):
            raise ValueError("one_body must be symmetric")
        if not np.allclose(two, two.swapaxes(0, 1), atol=1.0e-10, rtol=0.0):
            raise ValueError("two_body_chemist must be symmetric under p <-> q")
        if not np.allclose(two, two.swapaxes(2, 3), atol=1.0e-10, rtol=0.0):
            raise ValueError("two_body_chemist must be symmetric under r <-> s")
        if not np.allclose(two, two.transpose(2, 3, 0, 1), atol=1.0e-10, rtol=0.0):
            raise ValueError("two_body_chemist must be symmetric under (pq) <-> (rs)")
        object.__setattr__(self, "one_body", one)
        object.__setattr__(self, "two_body_chemist", two)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "n_electrons", n_electrons)

    @property
    def spatial_orbitals(self) -> int:
        return int(self.one_body.shape[0])


@dataclass(frozen=True)
class HamiltonianData:
    matrix: np.ndarray
    state: np.ndarray | None = None
    particle_number: int | None = None
    electronic: ElectronicIntegrals | None = None
    source_format: str = "array"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = _as_hermitian(self.matrix, name="Hamiltonian")
        dimension = matrix.shape[0]
        n_qubits = int(round(np.log2(dimension))) if dimension else -1
        if dimension < 2 or 2**n_qubits != dimension:
            raise ValueError("Hamiltonian dimension must be 2**n with n >= 1")
        state = self.state
        if state is not None:
            raw_state = np.asarray(state)
            if raw_state.dtype.kind not in "biufc":
                raise ValueError("state must contain numeric values")
            state = np.asarray(raw_state, dtype=np.complex128).reshape(-1)
            if state.shape != (dimension,):
                raise ValueError(
                    f"state must have shape {(dimension,)}; received {state.shape}"
                )
            if not np.all(np.isfinite(state)):
                raise ValueError("state must contain only finite values")
            norm = float(np.linalg.norm(state))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("state must have finite nonzero norm")
            state = state / norm
        particle_number = self.particle_number
        if particle_number is not None:
            particle_number = _strict_integer(
                particle_number, name="particle_number"
            )
            if not 0 <= particle_number <= n_qubits:
                raise ValueError("particle_number must lie between 0 and n_qubits")
        electronic = self.electronic
        if electronic is not None:
            if not isinstance(electronic, ElectronicIntegrals):
                raise ValueError("electronic must be an ElectronicIntegrals instance")
            expected_qubits = 2 * electronic.spatial_orbitals
            if n_qubits != expected_qubits:
                raise ValueError(
                    "Hamiltonian dimension is inconsistent with the electronic "
                    f"integrals: expected {expected_qubits} qubits"
                )
            if particle_number is None:
                particle_number = electronic.n_electrons
            elif particle_number != electronic.n_electrons:
                raise ValueError(
                    "particle_number must equal electronic.n_electrons"
                )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "particle_number",
            particle_number,
        )

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(round(np.log2(self.dimension)))

    @property
    def hamiltonian_sha256(self) -> str:
        array = np.ascontiguousarray(self.matrix)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.view(np.uint8))
        return digest.hexdigest()


@dataclass
class Fragment:
    """One executable setting ``U^dagger diag(diagonal) U``."""

    unitary: np.ndarray
    diagonal: np.ndarray
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_unitary = np.asarray(self.unitary)
        if raw_unitary.dtype.kind not in "biufc":
            raise ValueError("fragment unitary must contain numeric values")
        unitary = np.asarray(raw_unitary, dtype=np.complex128)
        if not np.all(np.isfinite(unitary)):
            raise ValueError("fragment unitary must contain only finite values")
        diagonal = _finite_real_array(
            self.diagonal, name="fragment diagonal"
        ).reshape(-1)
        if unitary.shape != (len(diagonal), len(diagonal)):
            raise ValueError("fragment unitary and diagonal dimensions do not match")
        error = float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(len(diagonal))))
        # GPD's documented float32 circuit path accumulates roundoff above a
        # float64-scale tolerance.  The Frobenius threshold therefore scales as
        # O(eps_float32 * sqrt(D)) while still rejecting material non-unitarity.
        if error > UNITARY_TOLERANCE * max(1.0, np.sqrt(len(diagonal))):
            raise ValueError(f"fragment unitary is not unitary: {error:.3e}")
        self.unitary = unitary
        self.diagonal = diagonal

    @property
    def matrix(self) -> np.ndarray:
        value = self.unitary.conj().T @ (self.diagonal[:, None] * self.unitary)
        return (value + value.conj().T) / 2.0

    @property
    def midpoint(self) -> float:
        return 0.5 * float(np.max(self.diagonal) + np.min(self.diagonal))

    @property
    def half_range(self) -> float:
        return 0.5 * float(np.max(self.diagonal) - np.min(self.diagonal))


@dataclass(frozen=True)
class ErrorEstimate:
    approximation_spectral_norm: float
    approximation_frobenius_norm: float
    sampling_variance_bound: float
    state_independent_rmse_bound: float
    calibrated_proxy_rmse: float | None = None
    state_bias: float | None = None
    state_sampling_variance: float | None = None
    state_rmse: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "approximation_spectral_norm",
            "approximation_frobenius_norm",
            "sampling_variance_bound",
            "state_independent_rmse_bound",
            "calibrated_proxy_rmse",
            "state_bias",
            "state_sampling_variance",
            "state_rmse",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            finite = _finite_real_scalar(value, name=name)
            object.__setattr__(self, name, finite)


@dataclass
class DecompositionResult:
    method: str
    shots: int
    constant: float
    fragments: list[Fragment]
    shot_allocation: np.ndarray
    approximate_hamiltonian: np.ndarray
    residual: np.ndarray
    error: ErrorEstimate
    selected_size: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shots = _strict_integer(self.shots, name="shots")
        if shots <= 0:
            raise ValueError("shots must be a positive integer")
        selected_size = _strict_integer(self.selected_size, name="selected_size")
        if selected_size < 0:
            raise ValueError("selected_size must be nonnegative")
        constant = _finite_real_scalar(self.constant, name="constant")
        raw_allocation = np.asarray(self.shot_allocation)
        if raw_allocation.dtype.kind not in "iu":
            raise ValueError("shot allocation must contain integers")
        allocation = np.asarray(raw_allocation, dtype=np.int64)
        if allocation.shape != (len(self.fragments),):
            raise ValueError("shot allocation length does not match fragment count")
        expected = shots if self.fragments else 0
        if np.any(allocation < 0) or int(np.sum(allocation)) != expected:
            raise ValueError(
                "shot allocation must be nonnegative and sum to T whenever nonconstant settings exist"
            )
        self.shots = shots
        self.selected_size = selected_size
        self.constant = constant
        self.shot_allocation = allocation
        self.approximate_hamiltonian = _as_hermitian(
            self.approximate_hamiltonian, name="approximate Hamiltonian"
        )
        self.residual = _as_hermitian(self.residual, name="residual")
