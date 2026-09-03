"""Versioned Hamiltonian input and decomposition output formats."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import DecompositionResult, ElectronicIntegrals, HamiltonianData
from .pauli import pauli_terms_to_dense


def _coefficient(value: Any) -> complex:
    if isinstance(value, bool):
        raise ValueError("coefficient must be a finite real number or [real, imaginary]")
    if isinstance(value, (int, float)):
        result = complex(float(value))
        if not np.isfinite(result.real):
            raise ValueError("coefficient must be finite")
        return result
    if isinstance(value, list) and len(value) == 2:
        try:
            result = complex(float(value[0]), float(value[1]))
        except (TypeError, ValueError) as error:
            raise ValueError("coefficient components must be real numbers") from error
        if not np.isfinite(result.real) or not np.isfinite(result.imag):
            raise ValueError("coefficient must be finite")
        return result
    raise ValueError("coefficient must be a real number or [real, imaginary]")


def load_state(path: str | Path) -> np.ndarray:
    """Load a state vector from the same safe formats accepted by the API."""
    source = Path(path).resolve()
    if source.suffix.lower() == ".npy":
        values = np.asarray(np.load(source, allow_pickle=False), dtype=np.complex128)
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError("state NPZ must contain at least one array")
            key = "state" if "state" in archive else list(archive.keys())[0]
            values = np.asarray(archive[key], dtype=np.complex128)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        encoded = payload.get("state") if isinstance(payload, dict) else payload
        if encoded is None:
            raise ValueError("state JSON object requires a state field")
        values = np.asarray(
            [
                complex(*item) if isinstance(item, list) else complex(item)
                for item in encoded
            ],
            dtype=np.complex128,
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("state must contain only finite values")
    return values


def load_hamiltonian(
    path: str | Path,
    *,
    state_path: str | Path | None = None,
    particle_number: int | None = None,
) -> HamiltonianData:
    source = Path(path).resolve()
    suffix = source.suffix.lower()
    external_state = load_state(state_path) if state_path else None
    electronic = None
    metadata: dict[str, Any] = {"source_name": source.name}
    if suffix == ".npy":
        matrix = np.asarray(np.load(source, allow_pickle=False), dtype=np.complex128)
        state = external_state
        source_format = "dense-npy"
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            keys = set(archive.keys())
            if {"one_body", "two_body_chemist"}.issubset(keys) or {
                "one_body",
                "eri_chemist",
            }.issubset(keys):
                two_key = "two_body_chemist" if "two_body_chemist" in archive else "eri_chemist"
                if "n_electrons" not in archive:
                    raise ValueError("electronic-integral NPZ requires n_electrons")
                n_electrons = np.asarray(archive["n_electrons"])
                electronic = ElectronicIntegrals(
                    constant=np.asarray(archive.get("constant", 0.0)),
                    one_body=np.asarray(archive["one_body"]),
                    two_body_chemist=np.asarray(archive[two_key]),
                    n_electrons=n_electrons,
                )
                from .electronic import electronic_hamiltonian_dense

                matrix = electronic_hamiltonian_dense(electronic)
                particle_number = n_electrons if particle_number is None else particle_number
                source_format = "electronic-integrals-npz"
            else:
                key = "hamiltonian" if "hamiltonian" in archive else "H"
                if key not in archive:
                    raise ValueError("NPZ requires hamiltonian/H or electronic integral keys")
                matrix = np.asarray(archive[key], dtype=np.complex128)
                source_format = "dense-npz"
            state = external_state
            if state is None and "state" in archive:
                state = np.asarray(archive["state"], dtype=np.complex128)
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Pauli JSON must be an object")
        schema_version = payload.get("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError("Pauli JSON schema_version must be the integer 1")
        if payload.get("qubit_order") != "q0-most-significant":
            raise ValueError(
                "Pauli JSON qubit_order must be 'q0-most-significant'"
            )
        n_qubits_value = payload.get("n_qubits")
        if (
            type(n_qubits_value) is not int
            or isinstance(n_qubits_value, bool)
            or n_qubits_value < 1
        ):
            raise ValueError("Pauli JSON n_qubits must be a positive integer")
        n_qubits = n_qubits_value
        if not isinstance(payload.get("terms"), list):
            raise ValueError("Pauli JSON terms must be a list")
        terms = [
            (str(row["pauli"]), _coefficient(row["coefficient"]))
            for row in payload["terms"]
        ]
        matrix = pauli_terms_to_dense(n_qubits, terms)
        state = external_state
        if state is None and "state" in payload:
            state = np.asarray(
                [
                    complex(*item) if isinstance(item, list) else complex(item)
                    for item in payload["state"]
                ],
                dtype=np.complex128,
            )
        if particle_number is None and payload.get("particle_number") is not None:
            particle_number = payload["particle_number"]
        source_format = "pauli-json"
        metadata["pauli_terms"] = len(terms)
    else:
        raise ValueError("supported inputs are .npy, .npz, and Pauli .json")
    return HamiltonianData(
        matrix=matrix,
        state=state,
        particle_number=particle_number,
        electronic=electronic,
        source_format=source_format,
        metadata=metadata,
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_result(result: DecompositionResult, output: str | Path) -> Path:
    directory = Path(output).resolve()
    if directory.exists():
        if not directory.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {directory}")
        if next(directory.iterdir(), None) is not None:
            raise FileExistsError(
                f"output directory already exists and is not empty: {directory}"
            )
    else:
        directory.mkdir(parents=True, exist_ok=False)
    fragment_rows = []
    for index, (fragment, shots) in enumerate(zip(result.fragments, result.shot_allocation)):
        path = directory / f"fragment_{index:03d}.npz"
        np.savez_compressed(path, unitary=fragment.unitary, diagonal=fragment.diagonal)
        fragment_rows.append(
            {
                "index": index,
                "label": fragment.label,
                "shots": int(shots),
                "half_range": fragment.half_range,
                "file": path.name,
                "metadata": fragment.metadata,
            }
        )
    np.savez_compressed(
        directory / "approximation.npz",
        approximate_hamiltonian=result.approximate_hamiltonian,
        residual=result.residual,
        shot_allocation=result.shot_allocation,
    )
    summary = {
        "format_version": "1.0",
        "method": result.method,
        "shots": result.shots,
        "selected_size": result.selected_size,
        "constant": result.constant,
        "error": result.error,
        "fragments": fragment_rows,
        "metadata": result.metadata,
        "approximation_file": "approximation.npz",
    }
    path = directory / "result.json"
    path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
