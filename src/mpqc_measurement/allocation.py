"""Exact integer shot allocation and error evaluation."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import numpy as np

from .models import ErrorEstimate, Fragment, HamiltonianData


def require_positive_integer(value: object, *, name: str = "total_shots") -> int:
    """Return an integer value without accepting booleans or lossy coercions."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def integer_range_allocation(ranges: Sequence[float], total_shots: int) -> np.ndarray:
    """Minimum-one, proportional-range, largest-remainder allocation."""
    raw_values = np.asarray(ranges)
    if raw_values.dtype.kind not in "iuf" or np.iscomplexobj(raw_values):
        raise ValueError("ranges must be a finite nonnegative real vector")
    values = np.asarray(raw_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("ranges must be a finite nonnegative vector")
    shots = require_positive_integer(total_shots)
    active = np.flatnonzero(values > 0.0)
    if len(active) > shots:
        raise ValueError(
            f"{len(active)} nonconstant settings cannot receive at least one of T={shots} shots"
        )
    allocation = np.zeros(len(values), dtype=np.int64)
    if not len(active):
        return allocation
    allocation[active] = 1
    remaining = shots - len(active)
    if remaining:
        quotas = remaining * values[active] / float(np.sum(values[active]))
        floors = np.floor(quotas).astype(np.int64)
        allocation[active] += floors
        leftover = remaining - int(np.sum(floors))
        if leftover:
            fractions = quotas - floors
            order = np.lexsort((active, -fractions))
            allocation[active[order[:leftover]]] += 1
    if int(np.sum(allocation)) != shots:
        raise AssertionError("integer allocation failed exact shot conservation")
    return allocation


def centered_fragments(
    constant: float, fragments: Sequence[Fragment]
) -> tuple[float, list[Fragment]]:
    """Move every diagonal midpoint into the classically exact constant."""
    updated_constant = float(constant)
    output: list[Fragment] = []
    for fragment in fragments:
        midpoint = fragment.midpoint
        updated_constant += midpoint
        diagonal = fragment.diagonal - midpoint
        if float(np.max(diagonal) - np.min(diagonal)) <= 0.0:
            continue
        output.append(
            Fragment(
                unitary=fragment.unitary,
                diagonal=diagonal,
                label=fragment.label,
                metadata={**fragment.metadata, "removed_midpoint": midpoint},
            )
        )
    return updated_constant, output


def materialize(constant: float, fragments: Sequence[Fragment]) -> np.ndarray:
    if not fragments:
        raise ValueError("at least one fragment is required to infer the dimension")
    dimension = len(fragments[0].diagonal)
    output = float(constant) * np.eye(dimension, dtype=np.complex128)
    for fragment in fragments:
        output += fragment.matrix
    return (output + output.conj().T) / 2.0


def evaluate_error(
    data: HamiltonianData,
    approximate: np.ndarray,
    fragments: Sequence[Fragment],
    allocation: np.ndarray,
    *,
    calibrated_proxy_rmse: float | None = None,
) -> ErrorEstimate:
    residual = (data.matrix - approximate + (data.matrix - approximate).conj().T) / 2.0
    spectral = float(np.max(np.abs(np.linalg.eigvalsh(residual))))
    frobenius = float(np.linalg.norm(residual, ord="fro"))
    sampling_bound = 0.0
    for fragment, count in zip(fragments, allocation):
        if int(count) > 0:
            sampling_bound += fragment.half_range**2 / int(count)
    independent = math.sqrt(max(spectral * spectral + sampling_bound, 0.0))
    bias = variance = rmse = None
    if data.state is not None:
        state = data.state
        exact_mean = float(np.real(np.vdot(state, data.matrix @ state)))
        approximate_mean = float(np.real(np.vdot(state, approximate @ state)))
        bias = approximate_mean - exact_mean
        variance = 0.0
        for fragment, count in zip(fragments, allocation):
            if int(count) <= 0:
                continue
            operator = fragment.matrix
            acted = operator @ state
            mean = float(np.real(np.vdot(state, acted)))
            second = float(np.real(np.vdot(acted, acted)))
            variance += max(second - mean * mean, 0.0) / int(count)
        rmse = math.sqrt(max(bias * bias + variance, 0.0))
    return ErrorEstimate(
        approximation_spectral_norm=spectral,
        approximation_frobenius_norm=frobenius,
        sampling_variance_bound=float(sampling_bound),
        state_independent_rmse_bound=independent,
        calibrated_proxy_rmse=calibrated_proxy_rmse,
        state_bias=bias,
        state_sampling_variance=variance,
        state_rmse=rmse,
    )
