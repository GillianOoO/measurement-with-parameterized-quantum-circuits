#!/usr/bin/env python3
"""Extend the released four-qubit random ensembles without changing their prefix.

The recovered seed-zero stream first draws one complex Gaussian state, five
dense matrices, and five sparse matrices.  Extra instances append one dense
matrix and one sparse matrix at a time.  The original state and Hamiltonian
files are copied byte for byte after checking this reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("MEASUREMENT_DATA_ROOT", BUNDLE_ROOT / "data"))
DIMENSION = 16
ORIGINAL_INSTANCES = 5
REGIMES = ("dense", "sparse")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Use the dtype/shape/data convention of the released GPD backend."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def draw_matrix(rng: np.random.RandomState, regime: str) -> np.ndarray:
    if regime == "dense":
        return rng.uniform(size=(DIMENSION, DIMENSION))
    matrix = np.zeros((DIMENSION, DIMENSION), dtype=np.float64)
    positions = rng.choice(DIMENSION * DIMENSION, size=4, replace=False)
    matrix.flat[positions] = rng.uniform(size=4)
    if np.count_nonzero(matrix) != 4:
        raise RuntimeError("Sparse generator did not produce four nonzero entries")
    return matrix


def reconstruct_stream(instances: int) -> tuple[np.ndarray, dict[tuple[str, int], np.ndarray]]:
    rng = np.random.RandomState(0)
    state = rng.standard_normal(DIMENSION) + 1j * rng.standard_normal(DIMENSION)
    state /= np.linalg.norm(state)
    matrices: dict[tuple[str, int], np.ndarray] = {}
    # Preserve these original RNG calls even when only a short output is requested.
    for regime in REGIMES:
        for instance in range(1, ORIGINAL_INSTANCES + 1):
            matrices[regime, instance] = draw_matrix(rng, regime)
    for instance in range(ORIGINAL_INSTANCES + 1, instances + 1):
        for regime in REGIMES:
            matrices[regime, instance] = draw_matrix(rng, regime)
    return state, matrices


def ensure_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or file_sha256(destination) != file_sha256(source):
            raise FileExistsError(f"Existing input differs from the released file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if file_sha256(destination) != file_sha256(source):
        raise RuntimeError(f"Copy verification failed: {destination}")


def ensure_array(destination: Path, expected: np.ndarray) -> None:
    if destination.exists():
        actual = np.load(destination, allow_pickle=False)
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise FileExistsError(f"Existing generated input differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.save(handle, expected, allow_pickle=False)


def ensure_json(destination: Path, expected: dict[str, Any]) -> None:
    if destination.exists():
        actual = json.loads(destination.read_text(encoding="utf-8"))
        if actual != expected:
            raise FileExistsError(f"Existing input manifest differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True)
        handle.write("\n")


def generate_inputs(data_root: Path, output: Path, instances: int) -> dict[str, Any]:
    if instances < ORIGINAL_INSTANCES:
        raise ValueError("--instances must be at least 5 to retain the released ensemble")
    source = data_root.resolve() / "random_sparse_dense" / "inputs"
    output = output.resolve()
    state_source = source / "common_input_state.npy"
    saved_density = np.asarray(np.load(state_source, allow_pickle=False), dtype=np.complex128)
    recovered_state, raw_matrices = reconstruct_stream(instances)
    recovered_density = np.outer(recovered_state, recovered_state.conj())
    if saved_density.shape != (DIMENSION, DIMENSION):
        raise ValueError("The released common input must be a 16 by 16 density matrix")
    state_error = float(np.max(np.abs(saved_density - recovered_density)))
    if state_error > 1e-14:
        raise ValueError(f"Released state does not match recovered seed-zero stream: {state_error}")
    if not np.allclose(saved_density, saved_density.conj().T, atol=1e-14, rtol=0):
        raise ValueError("Released state is not Hermitian")
    if not np.isclose(np.trace(saved_density), 1.0, atol=1e-14, rtol=0):
        raise ValueError("Released state is not normalized")
    if not np.allclose(saved_density @ saved_density, saved_density, atol=1e-14, rtol=0):
        raise ValueError("Released state is not pure")

    # Audit every released Hamiltonian before writing any generated inputs.
    source_records: list[dict[str, Any]] = []
    for regime in REGIMES:
        for instance in range(1, ORIGINAL_INSTANCES + 1):
            raw = raw_matrices[regime, instance]
            recovered = (raw + raw.T) / 2.0
            path = source / "hamiltonians" / f"{regime}_hamiltonian_4_{instance}.npy"
            saved = np.load(path, allow_pickle=False)
            if saved.dtype != np.dtype("float64") or not np.array_equal(saved, recovered):
                raise ValueError(f"Released Hamiltonian failed exact generator reconstruction: {path}")
            source_records.append({"path": str(path), "file_sha256": file_sha256(path),
                                   "maximum_absolute_reconstruction_error": 0.0})

    state_path = output / "common_input_state.npy"
    ensure_copy(state_source, state_path)
    density_state_hash = array_sha256(saved_density)
    cases: list[dict[str, Any]] = []
    checksums: dict[str, str] = {"common_input_state.npy": file_sha256(state_path)}
    seen_hashes: set[str] = set()
    for regime in ("sparse", "dense"):
        for instance in range(1, instances + 1):
            raw = raw_matrices[regime, instance]
            matrix = (raw + raw.T) / 2.0
            if matrix.dtype != np.dtype("float64") or not np.array_equal(matrix, matrix.T):
                raise RuntimeError("Generated Hamiltonian has the wrong dtype or is not Hermitian")
            target_hash = array_sha256(matrix.astype(np.complex128))
            if target_hash in seen_hashes:
                raise RuntimeError(f"Duplicate Hamiltonian: {regime} instance {instance}")
            seen_hashes.add(target_hash)
            filename = f"{regime}_hamiltonian_4_{instance}.npy"
            matrix_path = output / "hamiltonians" / filename
            if instance <= ORIGINAL_INSTANCES:
                ensure_copy(source / "hamiltonians" / filename, matrix_path)
            else:
                ensure_array(matrix_path, matrix)
            raw_path = output / "generation_matrices" / filename
            ensure_array(raw_path, raw)
            input_file_hash = file_sha256(matrix_path)
            checksums[str(matrix_path.relative_to(output)).replace("\\", "/")] = input_file_hash
            checksums[str(raw_path.relative_to(output)).replace("\\", "/")] = file_sha256(raw_path)
            cases.append({
                "benchmark": regime, "instance": instance,
                "slug": f"{regime}4_seed0_{instance}", "n_qubits": 4,
                "input_path": str(matrix_path), "file_sha256": input_file_hash,
                "target_hash": target_hash, "float64_array_sha256": array_sha256(matrix),
                "state_path": str(state_path), "state_file_sha256": file_sha256(state_path),
                "density_state_hash": density_state_hash,
                "generation_matrix_path": str(raw_path),
                "generation_matrix_nonzero_entries": int(np.count_nonzero(raw)),
                "target_expectation": float(np.trace(saved_density @ matrix).real),
                "released_original": instance <= ORIGINAL_INSTANCES,
            })

    manifest = {
        "schema_version": 1, "n_qubits": 4, "dimension": DIMENSION,
        "instances_per_regime": instances, "case_count": len(cases),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "data_root": str(data_root.resolve()), "output": str(output),
        "generation": {
            "rng": "numpy.random.RandomState(MT19937)", "ensemble_seed": 0,
            "state_draw": "16 normal real amplitudes, then 16 normal imaginary amplitudes; normalize",
            "original_call_order": "state, dense instances 1..5, sparse instances 1..5",
            "extension_call_order": "for i=6..N: dense instance i, then sparse instance i",
            "dense_A": "256 independent Uniform[0,1] entries",
            "sparse_A": "four distinct flattened positions chosen without replacement, then four Uniform[0,1] values",
            "symmetrization": "H=(A+A.T)/2", "dtype": "float64",
            "target_hash_convention": "SHA256(str(dtype)+str(shape)+contiguous bytes), after complex128 cast",
        },
        "verification": {
            "hermitian_all": True, "unique_hamiltonians": len(seen_hashes),
            "shared_normalized_pure_state": True,
            "state_stream_maximum_absolute_error": state_error,
            "original_hamiltonians_exactly_reconstructed": 10,
            "original_inputs_copied_bytewise": True,
        },
        "source_provenance": {
            "state": {"path": str(state_source), "file_sha256": file_sha256(state_source)},
            "hamiltonians": source_records,
        },
        "cases": cases,
    }
    ensure_json(output / "input_checksums.json", checksums)
    ensure_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Released data/ root containing random_sparse_dense/inputs")
    parser.add_argument("--output", type=Path, required=True, help="New run's inputs directory")
    parser.add_argument("--instances", type=int, default=50, help="Independent Hamiltonians per regime")
    args = parser.parse_args()
    manifest = generate_inputs(args.data_root, args.output, args.instances)
    print(json.dumps({"status": "PASS", "instances_per_regime": args.instances,
                      "case_count": manifest["case_count"], "output": manifest["output"],
                      "original_inputs_verified": 10, "duplicate_hamiltonians": 0}, indent=2))


if __name__ == "__main__":
    main()
