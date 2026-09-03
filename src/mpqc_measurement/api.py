"""Stable public convenience API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .allocation import require_positive_integer
from .gpd import GPDConfig, fit_gpd
from .io import load_hamiltonian, load_state, save_result
from .models import DecompositionResult, HamiltonianData
from .srdd import SRDDConfig, fit_srdd


InputHamiltonian = HamiltonianData | np.ndarray | str | Path


def _coerce(
    source: InputHamiltonian,
    *,
    state: np.ndarray | str | Path | None = None,
    particle_number: int | None = None,
) -> HamiltonianData:
    if isinstance(source, HamiltonianData):
        if state is not None or particle_number is not None:
            raise ValueError("state/particle_number must be embedded when source is HamiltonianData")
        return source
    if isinstance(source, (str, Path)):
        state_path = state if isinstance(state, (str, Path)) else None
        data = load_hamiltonian(
            source, state_path=state_path, particle_number=particle_number
        )
        if isinstance(state, np.ndarray):
            data = HamiltonianData(
                matrix=data.matrix,
                state=state,
                particle_number=data.particle_number,
                electronic=data.electronic,
                source_format=data.source_format,
                metadata=data.metadata,
            )
        return data
    if isinstance(state, (str, Path)):
        state_value = load_state(state)
    elif isinstance(state, np.ndarray):
        state_value = state
    elif state is None:
        state_value = None
    else:
        raise ValueError("state must be an array or a path to a supported state file")
    return HamiltonianData(
        matrix=np.asarray(source),
        state=state_value,
        particle_number=particle_number,
    )


def run_gpd(
    source: InputHamiltonian,
    shots: int,
    *,
    state: np.ndarray | str | Path | None = None,
    particle_number: int | None = None,
    config: GPDConfig | None = None,
    output: str | Path | None = None,
) -> DecompositionResult:
    total_shots = require_positive_integer(shots, name="shots")
    data = _coerce(source, state=state, particle_number=particle_number)
    result = fit_gpd(data, total_shots, config)
    if output is not None:
        save_result(result, output)
    return result


def run_srdd(
    source: InputHamiltonian,
    shots: int,
    *,
    state: np.ndarray | str | Path | None = None,
    particle_number: int | None = None,
    config: SRDDConfig | None = None,
    output: str | Path | None = None,
) -> DecompositionResult:
    total_shots = require_positive_integer(shots, name="shots")
    data = _coerce(source, state=state, particle_number=particle_number)
    result = fit_srdd(data, total_shots, config)
    if output is not None:
        save_result(result, output)
    return result


def decompose(
    source: InputHamiltonian,
    algorithm: str,
    shots: int,
    **kwargs,
) -> DecompositionResult:
    normalized = str(algorithm).strip().lower()
    if normalized == "gpd":
        return run_gpd(source, shots, **kwargs)
    if normalized == "srdd":
        return run_srdd(source, shots, **kwargs)
    raise ValueError("algorithm must be 'gpd' or 'srdd'")
