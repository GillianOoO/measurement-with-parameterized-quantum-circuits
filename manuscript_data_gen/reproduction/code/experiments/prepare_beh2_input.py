#!/usr/bin/env python3
"""Convert the archived BeH2 JW benchmark into the tensor-native case schema.

This is an exact algebraic conversion of the archived 14-qubit QubitOperator;
it does not rerun an electronic-structure package.  Mixed-spin normal-ordered
fermionic coefficients uniquely recover the spatial two-electron tensor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import quimb.tensor as qtn
from scipy.sparse.linalg import eigsh
from openfermion import (
    FermionOperator,
    InteractionOperator,
    QubitOperator,
    get_fermion_operator,
    get_interaction_operator,
    get_sparse_operator,
    jordan_wigner,
    normal_ordered,
    reverse_jordan_wigner,
)
from openfermion.chem.molecular_data import spinorb_from_spatial


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "methods" / "srdd"))
from srdd_release_paths import DATA, RUNS

ARCHIVE = DATA / "BeH2" / "inputs" / "hamiltonian_pauli_blocked_spin.csv"
OUTPUT = RUNS / "beh2_input"
N_SPATIAL = 7
N_QUBITS = 14
N_ALPHA = 3
N_BETA = 3
ARCHIVED_EXACT_ENERGY = -19.045049602807797
VERSION = "beh2-archived-jw-exact-inverse-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_archive() -> tuple[QubitOperator, list[tuple[str, complex]]]:
    path = ARCHIVE
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            entries = [
                (row["full_label_site0_to_siteNminus1"], complex(
                    float(row["coefficient_real_hartree"]), float(row["coefficient_imag_hartree"])
                )) for row in csv.DictReader(handle)
            ]
    else:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) % 2:
            raise ValueError("Archived JW file does not contain label/coefficient pairs")
        entries = list(zip(lines[0::2], lines[1::2]))
    rows: list[tuple[str, complex]] = []
    operator = QubitOperator()
    for label, coefficient_text in entries:
        if len(label) != N_QUBITS:
            raise ValueError(f"Unexpected label length: {label}")
        coefficient = complex(coefficient_text)
        rows.append((label, coefficient))
        term = tuple((site, letter) for site, letter in enumerate(label) if letter != "I")
        operator += QubitOperator(term, coefficient)
    return operator, rows


def max_qubit_error(left: QubitOperator, right: QubitOperator) -> float:
    keys = set(left.terms) | set(right.terms)
    return float(max(abs(left.terms.get(key, 0.0) - right.terms.get(key, 0.0)) for key in keys))


def recover_spatial_tensors(qubit: QubitOperator):
    fermion = normal_ordered(reverse_jordan_wigner(qubit, n_qubits=N_QUBITS))
    interaction = get_interaction_operator(fermion, n_qubits=N_QUBITS)
    one_spin = np.asarray(interaction.one_body_tensor)
    alpha = np.arange(N_SPATIAL)
    beta = np.arange(N_SPATIAL, N_QUBITS)
    one = 0.5 * (one_spin[np.ix_(alpha, alpha)] + one_spin[np.ix_(beta, beta)])
    two_open = np.empty((N_SPATIAL,) * 4, dtype=np.complex128)
    for p in range(N_SPATIAL):
        for q in range(N_SPATIAL):
            for r in range(N_SPATIAL):
                for s in range(N_SPATIAL):
                    raw = FermionOperator(((p, 1), (N_SPATIAL + q, 1), (N_SPATIAL + r, 0), (s, 0)))
                    canonical = normal_ordered(raw)
                    key, sign = next(iter(canonical.terms.items()))
                    two_open[p, q, r, s] = fermion.terms.get(key, 0.0) / sign
    if np.max(np.abs(one.imag)) > 1.0e-12 or np.max(np.abs(two_open.imag)) > 1.0e-12:
        raise ValueError("Recovered spatial tensors are unexpectedly complex")
    one = one.real
    two_open = two_open.real
    one_interleaved, two_interleaved = spinorb_from_spatial(one, two_open)
    permutation = np.asarray([2 * p for p in range(N_SPATIAL)] + [2 * p + 1 for p in range(N_SPATIAL)])
    one_blocked = one_interleaved[np.ix_(permutation, permutation)]
    two_blocked = two_interleaved[np.ix_(permutation, permutation, permutation, permutation)]
    replay = jordan_wigner(
        get_fermion_operator(
            InteractionOperator(float(np.real(interaction.constant)), one_blocked, 0.5 * two_blocked)
        )
    )
    return float(np.real(interaction.constant)), one, two_open, max_qubit_error(qubit, replay)


def sector_indices() -> np.ndarray:
    alpha_configs = list(__import__("itertools").combinations(range(N_SPATIAL), N_ALPHA))
    beta_configs = list(__import__("itertools").combinations(range(N_SPATIAL), N_BETA))
    indices = []
    for a in alpha_configs:
        for b in beta_configs:
            occupied = set(a) | {N_SPATIAL + q for q in b}
            index = sum(1 << (N_QUBITS - 1 - q) for q in occupied)
            indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def exact_ground_state(qubit: QubitOperator):
    full = get_sparse_operator(qubit, n_qubits=N_QUBITS).tocsr()
    indices = sector_indices()
    sector = full[indices][:, indices]
    eigenvalues, eigenvectors = eigsh(sector, k=1, which="SA", tol=1.0e-12, maxiter=20000)
    vector = np.zeros(1 << N_QUBITS, dtype=np.complex128)
    vector[indices] = eigenvectors[:, 0]
    phase_index = int(np.argmax(np.abs(vector)))
    vector *= np.exp(-1j * np.angle(vector[phase_index]))
    vector /= np.linalg.norm(vector)
    residual = np.linalg.norm(full @ vector - eigenvalues[0] * vector)
    return float(eigenvalues[0]), vector, float(residual), int(sector.shape[0])


def save_mps(path: Path, vector: np.ndarray) -> int:
    mps = qtn.MatrixProductState.from_dense(vector, dims=[2] * N_QUBITS, cutoff=1.0e-14)
    arrays: dict[str, np.ndarray] = {}
    for site, tensor in enumerate(mps.tensors):
        array = np.asarray(tensor.data)
        if site == 0:
            array = array[None, :, :]
        elif site == N_QUBITS - 1:
            array = array[:, :, None]
        arrays[f"A{site:03d}"] = array
    np.savez_compressed(path, **arrays)
    return max(max(array.shape[0], array.shape[2]) for array in arrays.values())


def main() -> None:
    global ARCHIVE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ARCHIVE, OUTPUT = args.input.resolve(), args.output.resolve()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    qubit, rows = read_archive()
    constant, one, two_open, replay_error = recover_spatial_tensors(qubit)
    eri = two_open.transpose(0, 3, 1, 2)
    symmetry_axes = (
        (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
        (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
    )
    symmetry_error = max(float(np.linalg.norm(eri - eri.transpose(axes))) for axes in symmetry_axes)
    energy, vector, residual, sector_dimension = exact_ground_state(qubit)
    if abs(energy - ARCHIVED_EXACT_ENERGY) > 2.0e-10:
        raise RuntimeError(f"Ground energy mismatch: {energy} versus {ARCHIVED_EXACT_ENERGY}")
    if replay_error > 1.0e-10 or symmetry_error > 1.0e-10 or residual > 1.0e-9:
        raise RuntimeError("Algebraic BeH2 reconstruction audit failed")
    np.savez_compressed(
        OUTPUT / "integrals_mo.npz",
        one_body_integrals=one,
        two_body_integrals_openfermion_order=two_open,
        nuclear_repulsion_hartree=np.asarray(constant),
    )
    with (OUTPUT / "hamiltonian_pauli_blocked_spin.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("full_label_site0_to_siteNminus1", "coefficient_real_hartree", "coefficient_imag_hartree"),
        )
        writer.writeheader()
        for label, coefficient in rows:
            writer.writerow({
                "full_label_site0_to_siteNminus1": label,
                "coefficient_real_hartree": float(coefficient.real),
                "coefficient_imag_hartree": float(coefficient.imag),
            })
    max_bond = save_mps(OUTPUT / "ground_state_mps_blocked_spin.npz", vector)
    (OUTPUT / "geometry.xyz").write_text(
        "3\nBeH2 linear archived benchmark, Angstrom\nBe 0 0 0\nH 0 0 1.33376\nH 0 0 -1.33376\n",
        encoding="utf-8",
    )
    metadata = {
        "version": VERSION,
        "molecule": "BeH2",
        "geometry": "linear H-Be-H; R(Be-H)=1.33376 A",
        "basis": "STO-3G",
        "charge": 0,
        "spin_multiplicity": 1,
        "number_spatial_orbitals": N_SPATIAL,
        "number_qubits": N_QUBITS,
        "number_electrons": N_ALPHA + N_BETA,
        "site_order": "blocked spin alpha_0..alpha_6,beta_0..beta_6",
        "mapper": "Jordan-Wigner",
        "fci_ground_energy_hartree": energy,
        "energy_convention": "archived electronic Hamiltonian; reverse-JW vacuum constant is zero",
        "archived_exact_energy_hartree": ARCHIVED_EXACT_ENERGY,
        "fixed_spin_sector_dimension": sector_dimension,
        "ground_state_mps_max_bond": max_bond,
        "ground_eigen_residual_2norm": residual,
        "qubit_reconstruction_max_coefficient_error": replay_error,
        "chemist_eri_eightfold_max_frobenius_error": symmetry_error,
        "derivation": "exact reverse Jordan-Wigner plus mixed-spin spatial-tensor recovery; no new SCF calculation",
        "archive_jw_path": str(ARCHIVE.resolve()),
        "archive_jw_sha256": sha256_file(ARCHIVE),
        "preparation_script": str(Path(__file__).resolve()),
        "preparation_script_sha256": sha256_file(Path(__file__)),
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    hashes = {
        path.name: sha256_file(path)
        for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "sha256.json"
    }
    (OUTPUT / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"[complete] {OUTPUT}")


if __name__ == "__main__":
    main()
