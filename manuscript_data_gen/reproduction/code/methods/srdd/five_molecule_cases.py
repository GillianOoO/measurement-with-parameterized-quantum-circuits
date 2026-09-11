#!/usr/bin/env python3
"""Controlled eight-qubit cases for the five-molecule hybrid-ansatz-F study.

LiH and H2O reuse :func:`adaptive_f3_pool.load_active_case`.  H4, H6, and F2
are recovered from the exact dense Hamiltonians already archived in this
workspace.  The recovery is not a fit: the vacuum, one-particle, and
two-particle blocks uniquely determine a number-conserving two-body fermionic
Hamiltonian.  The opposite-spin pair block then recovers the restricted
spatial-orbital tensor used by the tensor-native RCDF implementation.

The 12-qubit H6 and F2 sources are reduced to controlled four-spatial-orbital
active spaces so that all five ansatz families can use the same exact 8-qubit
backend.  The CLI independently reconstructs each original dense source in
row chunks, avoiding a second 4096 x 4096 dense allocation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterator

import numpy as np


HERE = Path(__file__).resolve().parent
from srdd_release_paths import BUNDLE as ARCHIVE, DATA
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adaptive_f3_pool import (  # noqa: E402
    DIMENSION,
    N_QUBITS,
    OCCUPATIONS,
    SPATIAL_ORBITALS,
    MoleculeCase,
    interaction_dense,
    make_rcdf_bank,
)


VERSION = "five-molecule-cases-exact-inversion-v1"


@dataclass(frozen=True)
class DenseSourceSpec:
    name: str
    label: str
    geometry: str
    path: Path
    slice_index: int | None
    source_qubits: int
    source_electrons: int
    frozen_orbitals: tuple[int, ...]
    active_orbitals: tuple[int, ...]


SOURCE_SPECS = {
    "H4": DenseSourceSpec(
        name="H4",
        label="H4 CAS(4e,4o)",
        geometry="linear H4, R=1.20 Angstrom, STO-3G",
        path=DATA / "H4" / "inputs" / "fixed_R1p2" / "h4_r1p2_dense_ground.npz",
        slice_index=None,
        source_qubits=8,
        source_electrons=4,
        frozen_orbitals=(),
        active_orbitals=(0, 1, 2, 3),
    ),
    "H6": DenseSourceSpec(
        name="H6",
        label="H6 CAS(4e,4o)",
        geometry="linear H6, R=3.40 Angstrom, STO-3G",
        path=DATA / "H6" / "inputs" / "H6_R3p4_STO3G.npy",
        slice_index=None,
        source_qubits=12,
        source_electrons=6,
        frozen_orbitals=(0,),
        active_orbitals=(1, 2, 3, 4),
    ),
    "F2": DenseSourceSpec(
        name="F2",
        label="F2 CAS(6e,4o)",
        geometry="F2, R=1.50 Angstrom, legacy STO-3G active Hamiltonian",
        path=DATA / "F2" / "inputs" / "12Q.npy",
        slice_index=10,
        source_qubits=12,
        source_electrons=10,
        frozen_orbitals=(0, 1),
        active_orbitals=(2, 3, 4, 5),
    ),
}


def _canonical_name(name: str) -> str:
    aliases = {"h4": "H4", "h6": "H6"}
    try:
        return aliases[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unavailable molecule {name!r}; this released dense driver supports H4 and H6") from exc


def _target_sha256(target: np.ndarray) -> str:
    """Hash a canonical little-endian complex128 active Hamiltonian."""
    canonical = np.ascontiguousarray(target, dtype=np.dtype("<c16"))
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.view(np.uint8))
    return digest.hexdigest()


def _dense_index(mask: int, n_qubits: int) -> int:
    result = 0
    for orbital in range(n_qubits):
        if mask & (1 << orbital):
            result |= 1 << (n_qubits - 1 - orbital)
    return result


def _annihilate(mask: int, orbital: int):
    bit = 1 << orbital
    if not mask & bit:
        return None
    sign = -1.0 if (mask & (bit - 1)).bit_count() % 2 else 1.0
    return mask ^ bit, sign


def _create(mask: int, orbital: int):
    bit = 1 << orbital
    if mask & bit:
        return None
    sign = -1.0 if (mask & (bit - 1)).bit_count() % 2 else 1.0
    return mask | bit, sign


def _apply_one(mask: int, p: int, q: int):
    first = _annihilate(mask, q)
    if first is None:
        return None
    second = _create(first[0], p)
    if second is None:
        return None
    return second[0], first[1] * second[1]


@contextmanager
def _open_dense_source(spec: DenseSourceSpec) -> Iterator[np.ndarray]:
    if spec.path.suffix.lower() == ".npz":
        with np.load(spec.path, allow_pickle=False) as saved:
            source = saved["H"]
            if source.shape != (1 << spec.source_qubits,) * 2:
                raise ValueError(f"Unexpected {spec.name} source shape {source.shape}")
            yield source
        return
    container = np.load(spec.path, mmap_mode="r", allow_pickle=False)
    source = container if spec.slice_index is None else container[spec.slice_index]
    if source.shape != (1 << spec.source_qubits,) * 2:
        raise ValueError(f"Unexpected {spec.name} source shape {source.shape}")
    yield source


def _extract_pair_coefficients(source: np.ndarray, n_qubits: int):
    """Recover constant, one-body matrix, and antisymmetric pair matrix."""
    constant = complex(source[0, 0])
    one_masks = [1 << q for q in range(n_qubits)]
    one_indices = [_dense_index(mask, n_qubits) for mask in one_masks]
    one_body = np.asarray(source[np.ix_(one_indices, one_indices)], dtype=np.complex128)
    one_body -= constant * np.eye(n_qubits, dtype=np.complex128)

    pairs = tuple((p, q) for p in range(n_qubits) for q in range(p + 1, n_qubits))
    pair_masks = [(1 << p) | (1 << q) for p, q in pairs]
    pair_indices = [_dense_index(mask, n_qubits) for mask in pair_masks]
    pair_block = np.asarray(source[np.ix_(pair_indices, pair_indices)], dtype=np.complex128)

    pair_lookup = {mask: index for index, mask in enumerate(pair_masks)}
    one_lift = np.zeros_like(pair_block)
    active_one = np.argwhere(np.abs(one_body) > 1.0e-14)
    for column, mask in enumerate(pair_masks):
        for p, q in active_one:
            applied = _apply_one(mask, int(p), int(q))
            if applied is not None:
                one_lift[pair_lookup[applied[0]], column] += one_body[p, q] * applied[1]
    pair_body = pair_block - constant * np.eye(len(pairs)) - one_lift
    one_body = 0.5 * (one_body + one_body.conj().T)
    pair_body = 0.5 * (pair_body + pair_body.conj().T)
    return constant, one_body, pair_body, pairs


def _restricted_spatial_coefficients(
    constant: complex,
    one_body: np.ndarray,
    pair_body: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
):
    n_qubits = one_body.shape[0]
    if n_qubits % 2:
        raise ValueError("Blocked-spin recovery requires an even number of qubits")
    m = n_qubits // 2
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    one_alpha = one_body[:m, :m]
    one_beta = one_body[m:, m:]
    one_spatial = 0.5 * (one_alpha + one_beta)
    two_openfermion = np.empty((m, m, m, m), dtype=np.complex128)
    for p in range(m):
        for q in range(m):
            creation = pair_index[(p, q + m)]
            for r in range(m):
                for s in range(m):
                    annihilation = pair_index[(s, r + m)]
                    two_openfermion[p, q, r, s] = pair_body[creation, annihilation]

    imaginary_norm = float(
        np.linalg.norm(one_spatial.imag) + np.linalg.norm(two_openfermion.imag)
    )
    if imaginary_norm > 1.0e-9:
        raise ValueError(f"Recovered restricted integrals have imaginary norm {imaginary_norm}")
    diagnostics = {
        "constant_imaginary_abs": abs(float(constant.imag)),
        "one_alpha_beta_frobenius": float(np.linalg.norm(one_alpha - one_beta)),
        "one_spin_mixing_frobenius": float(
            np.hypot(np.linalg.norm(one_body[:m, m:]), np.linalg.norm(one_body[m:, :m]))
        ),
        "spatial_integral_imaginary_frobenius": imaginary_norm,
    }
    return (
        float(constant.real),
        np.asarray(one_spatial.real, dtype=float),
        np.asarray(two_openfermion.real, dtype=float),
        diagnostics,
    )


@lru_cache(maxsize=None)
def _recover_source_integrals(name: str):
    spec = SOURCE_SPECS[name]
    with _open_dense_source(spec) as source:
        recovered = _extract_pair_coefficients(source, spec.source_qubits)
    return _restricted_spatial_coefficients(*recovered)


def _symmetrized_chemist(two_openfermion: np.ndarray) -> np.ndarray:
    eri = two_openfermion.transpose(0, 3, 1, 2).copy()
    return sum(
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


@lru_cache(maxsize=None)
def load_five_molecule_case(name: str) -> MoleculeCase:
    """Return one of five controlled, exact eight-qubit benchmark cases."""
    canonical = _canonical_name(name)
    if canonical in {"LiH", "H2O"}:
        from adaptive_f3_pool import load_active_case

        case = load_active_case(canonical)
        case.metadata = {
            **case.metadata,
            "five_molecule_case_version": VERSION,
            "active_target_sha256": _target_sha256(case.target),
        }
        return case

    from openfermion.ops.representations import get_active_space_integrals

    spec = SOURCE_SPECS[canonical]
    source_constant, one_full, two_full, recovery = _recover_source_integrals(canonical)
    core_shift, one_active, two_active = get_active_space_integrals(
        one_full,
        two_full,
        occupied_indices=list(spec.frozen_orbitals),
        active_indices=list(spec.active_orbitals),
    )
    active_constant = source_constant + float(core_shift)
    target = interaction_dense(active_constant, one_active, two_active)
    base_collector = interaction_dense(
        active_constant, one_active, np.zeros_like(two_active)
    )
    active_electrons = spec.source_electrons - 2 * len(spec.frozen_orbitals)
    n_alpha = active_electrons // 2
    occupied = list(range(n_alpha)) + [SPATIAL_ORBITALS + p for p in range(n_alpha)]
    proxy_index = sum(1 << (N_QUBITS - 1 - q) for q in occupied)
    proxy = np.zeros(DIMENSION, dtype=np.complex128)
    proxy[proxy_index] = 1.0
    number = OCCUPATIONS.sum(axis=1)
    sector = np.flatnonzero(number == active_electrons)
    sector_ground = float(np.linalg.eigvalsh(target[np.ix_(sector, sector)])[0])
    discarded = sorted(
        set(range(one_full.shape[0]))
        - set(spec.frozen_orbitals)
        - set(spec.active_orbitals)
    )
    metadata = {
        "molecule": canonical,
        "label": spec.label,
        "geometry": spec.geometry,
        "five_molecule_case_version": VERSION,
        "source_dense": str(spec.path),
        "source_dense_slice_index": spec.slice_index,
        "source_qubits": spec.source_qubits,
        "source_spatial_orbitals": int(one_full.shape[0]),
        "source_active_electrons": spec.source_electrons,
        "source_effective_constant_hartree": source_constant,
        "integral_recovery": (
            "exact vacuum/one-particle/two-particle inversion; opposite-spin pair block"
        ),
        "integral_recovery_diagnostics": recovery,
        "frozen_doubly_occupied_orbitals": list(spec.frozen_orbitals),
        "active_orbitals": list(spec.active_orbitals),
        "discarded_virtual_orbitals": discarded,
        "active_electrons": active_electrons,
        "active_spatial_orbitals": SPATIAL_ORBITALS,
        "active_qubits": N_QUBITS,
        "active_target_sha256": _target_sha256(target),
        "active_constant_hartree": active_constant,
        "full_fock_frobenius_norm": float(np.linalg.norm(target, "fro")),
        "active_sector_ground_energy_hartree": sector_ground,
        "number_conservation_commutator_frobenius": float(
            np.linalg.norm(target @ np.diag(number) - np.diag(number) @ target)
        ),
    }
    return MoleculeCase(
        name=canonical,
        label=spec.label,
        geometry=spec.geometry,
        source=spec.path,
        frozen_orbitals=spec.frozen_orbitals,
        active_orbitals=spec.active_orbitals,
        active_electrons=active_electrons,
        nuclear_repulsion=source_constant,
        active_constant=active_constant,
        one_spatial=np.asarray(one_active, dtype=float),
        two_openfermion=np.asarray(two_active, dtype=float),
        eri_chemist=np.asarray(_symmetrized_chemist(two_active), dtype=float),
        target=target,
        base_collector=base_collector,
        proxy_state=proxy,
        sector_indices=sector,
        metadata=metadata,
    )


def _source_sparse(
    constant: float, one_spatial: np.ndarray, two_openfermion: np.ndarray
):
    from openfermion import InteractionOperator, get_sparse_operator
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
    operator = InteractionOperator(float(constant), one, 0.5 * two)
    sparse = get_sparse_operator(operator, n_qubits=2 * m).tocsr()
    sparse.sum_duplicates()
    sparse.eliminate_zeros()
    return sparse


def _active_embedding_indices(spec: DenseSourceSpec) -> np.ndarray:
    """Map the controlled 8-qubit basis into the archived blocked-spin basis."""
    source_spatial = spec.source_qubits // 2
    frozen_mask = sum(
        (1 << orbital) | (1 << (orbital + source_spatial))
        for orbital in spec.frozen_orbitals
    )
    indices = np.empty(DIMENSION, dtype=np.int64)
    for active_index in range(DIMENSION):
        mask = frozen_mask
        for active_orbital, source_orbital in enumerate(spec.active_orbitals):
            alpha_bit = 1 << (N_QUBITS - 1 - active_orbital)
            beta_bit = 1 << (
                N_QUBITS - 1 - (SPATIAL_ORBITALS + active_orbital)
            )
            if active_index & alpha_bit:
                mask |= 1 << source_orbital
            if active_index & beta_bit:
                mask |= 1 << (source_orbital + source_spatial)
        indices[active_index] = _dense_index(mask, spec.source_qubits)
    return indices


def reconstruction_audit(name: str, chunk_rows: int = 64) -> dict:
    """Independently audit full-source and projected active-space reconstruction."""
    canonical = _canonical_name(name)
    case = load_five_molecule_case(canonical)
    active_reconstructed = interaction_dense(
        case.active_constant, case.one_spatial, case.two_openfermion
    )
    active_error = float(np.linalg.norm(active_reconstructed - case.target, "fro"))
    report = {
        "molecule": canonical,
        "case_label": case.label,
        "active_electrons": case.active_electrons,
        "active_orbitals": list(case.active_orbitals),
        "frozen_orbitals": list(case.frozen_orbitals),
        "active_dense_reconstruction_frobenius": active_error,
        "active_number_conservation_commutator_frobenius": case.metadata[
            "number_conservation_commutator_frobenius"
        ],
        "active_target_sha256": case.metadata["active_target_sha256"],
    }
    if canonical in {"LiH", "H2O"}:
        report.update(
            {
                "source_dense": case.metadata["source_integrals"],
                "source_dense_slice_index": None,
                "full_source_reconstruction_frobenius": "N/A (integral-native source)",
                "full_source_reconstruction_relative_frobenius": "N/A (integral-native source)",
                "status": "PASS" if active_error <= 1.0e-10 else "FAIL",
            }
        )
        return report

    spec = SOURCE_SPECS[canonical]
    constant, one_full, two_full, recovery = _recover_source_integrals(canonical)
    sparse = _source_sparse(constant, one_full, two_full)
    error_sq = 0.0
    source_sq = 0.0
    maximum = 0.0
    with _open_dense_source(spec) as source:
        for start in range(0, source.shape[0], chunk_rows):
            stop = min(start + chunk_rows, source.shape[0])
            original = np.asarray(source[start:stop], dtype=np.complex128)
            difference = original - sparse[start:stop].toarray()
            error_sq += float(np.vdot(difference, difference).real)
            source_sq += float(np.vdot(original, original).real)
            maximum = max(maximum, float(np.max(np.abs(difference))))
    full_error = float(np.sqrt(max(error_sq, 0.0)))
    source_norm = float(np.sqrt(max(source_sq, 0.0)))
    relative = full_error / max(source_norm, 1.0e-300)
    embedded_indices = _active_embedding_indices(spec)
    with _open_dense_source(spec) as source:
        source_active_block = np.asarray(
            source[np.ix_(embedded_indices, embedded_indices)],
            dtype=np.complex128,
        )
    projection_difference = source_active_block - case.target
    projection_error = float(np.linalg.norm(projection_difference, "fro"))
    projection_norm = float(np.linalg.norm(source_active_block, "fro"))
    report.update(
        {
            "source_dense": str(spec.path),
            "source_dense_slice_index": spec.slice_index,
            "source_qubits": spec.source_qubits,
            "source_electrons": spec.source_electrons,
            "full_source_frobenius_norm": source_norm,
            "full_source_reconstruction_frobenius": full_error,
            "full_source_reconstruction_relative_frobenius": relative,
            "full_source_reconstruction_max_abs": maximum,
            "active_projection_from_source_frobenius": projection_error,
            "active_projection_from_source_relative_frobenius": (
                projection_error / max(projection_norm, 1.0e-300)
            ),
            "active_projection_from_source_max_abs": float(
                np.max(np.abs(projection_difference))
            ),
            "restricted_recovery_diagnostics": recovery,
            "status": (
                "PASS"
                if (
                    active_error <= 1.0e-10
                    and relative <= 1.0e-10
                    and projection_error
                    <= 1.0e-10 * max(projection_norm, 1.0)
                )
                else "FAIL"
            ),
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecules",
        nargs="+",
        choices=("H4", "H6"),
        default=("H4", "H6"),
    )
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument(
        "--rcdf-smoke",
        action="store_true",
        help="also fit a two-leaf, one-cycle tensor-native RCDF bank",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    reports = []
    for name in args.molecules:
        report = reconstruction_audit(name, chunk_rows=args.chunk_rows)
        if args.rcdf_smoke:
            case = load_five_molecule_case(name)
            bank, rcdf_audit = make_rcdf_bank(case, 2, 1.0e-6, 1, 1)
            report["tensor_native_RCDF_smoke"] = {
                "leaf_count": len(bank),
                "eri_residual_frobenius": rcdf_audit["eri_residual_frobenius"],
                "full_fock_operator_residual_frobenius": rcdf_audit[
                    "full_fock_operator_residual_frobenius"
                ],
                "maximum_native_diagonal_spectral_audit": rcdf_audit[
                    "maximum_native_diagonal_spectral_audit"
                ],
            }
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    if any(report["status"] != "PASS" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
