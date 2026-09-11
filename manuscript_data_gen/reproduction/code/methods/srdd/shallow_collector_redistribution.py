#!/usr/bin/env python3
"""Shared linear algebra for exact shallow one-body collector redistribution.

The module is intentionally independent of a particular many-body state
representation.  Callers provide their own occupation moments and setting
objects, while this core owns the projector dictionary, exact affine solve,
and collector-tailored depth-constrained rotation construction.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from shallow_rcdf import rotation_and_derivatives


VERSION = "shared-shallow-collector-redistribution-v1"


def _stable_seed(seed_namespace: str, seed_key: str, *parts: Any) -> int:
    raw = "|".join([seed_namespace, seed_key, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32)


def symmetric_vector(matrix: np.ndarray) -> np.ndarray:
    """Pack the upper triangle of a real symmetric matrix without rescaling."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Expected a square matrix")
    return values[np.triu_indices(values.shape[0])]


def collector_dictionary(rotations: np.ndarray) -> np.ndarray:
    """Return A with columns svec(R[:,j] R[:,j]^T)."""
    bank = np.asarray(rotations, dtype=float)
    if bank.ndim != 3 or bank.shape[1] != bank.shape[2]:
        raise ValueError("Rotations must have shape (settings, m, m)")
    columns = []
    for rotation in bank:
        for orbital in range(rotation.shape[1]):
            vector = rotation[:, orbital]
            columns.append(symmetric_vector(np.outer(vector, vector)))
    if not columns:
        dimension = bank.shape[1] * (bank.shape[1] + 1) // 2
        return np.empty((dimension, 0))
    return np.column_stack(columns)


def collector_matrix(rotations: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Materialize sum_t R_t diag(eta_t) R_t^T for audit purposes."""
    bank = np.asarray(rotations, dtype=float)
    eta = np.asarray(coefficients, dtype=float)
    if bank.ndim != 3 or eta.shape != (len(bank), bank.shape[1]):
        raise ValueError("Collector coefficients do not match the rotation bank")
    return np.einsum("tpi,ti,tqi->pq", bank, eta, bank, optimize=True)


def dictionary_diagnostics(
    design: np.ndarray, rcond: float
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Return numerical rank, retained condition number, singular values, null(A)."""
    if rcond <= 0.0:
        raise ValueError("rcond must be positive")
    _, singular_values, vh = la.svd(design, full_matrices=True, check_finite=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0, float("inf"), singular_values, vh.T
    threshold = rcond * singular_values[0]
    rank = int(np.sum(singular_values > threshold))
    condition = (
        float(singular_values[0] / singular_values[rank - 1])
        if rank
        else float("inf")
    )
    return rank, condition, singular_values, vh[rank:].T


def equality_project(
    vector: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    rcond: float = 1.0e-13,
) -> np.ndarray:
    """Project a coefficient vector onto A eta = target with minimum correction."""
    values = np.asarray(vector, dtype=float)
    residual = np.asarray(target, dtype=float) - design @ values
    if not np.any(residual):
        return values
    correction = design.T @ np.linalg.lstsq(
        design @ design.T, residual, rcond=rcond
    )[0]
    return np.asarray(values + correction)


def collector_tailored_extra_shallow_rotations(
    existing: np.ndarray,
    collector: np.ndarray,
    depth: int,
    count: int,
    seed_key: str,
    angle_steps: int,
    seed_namespace: str = VERSION,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Greedily shallow-diagonalize the current collector residual.

    Every added depth-d_R rotation maximizes
    ``||diag(R.T @ C_res @ R)||_F^2`` with an analytic angle gradient.  The
    corresponding diagonal coefficients form a seed; callers then use
    :func:`greedy_exact_fill` to restore the exact equality with the existing
    projector bank.
    """
    base = np.asarray(existing, dtype=float)
    target = np.asarray(collector, dtype=float)
    if base.ndim != 3 or base.shape[1] != base.shape[2]:
        raise ValueError("Existing rotations must have shape (K,m,m)")
    if target.shape != base.shape[1:]:
        raise ValueError("Collector dimension does not match the rotation bank")
    if count < 0 or angle_steps < 1:
        raise ValueError("count must be nonnegative and angle_steps positive")
    if count == 0:
        return (
            np.empty((0, base.shape[1], base.shape[2])),
            np.empty((0, base.shape[1])),
            {
                "method": "collector_tailored_greedy_shallow_diagonalization",
                "selected_flat_angles": [],
                "greedy_diagonal_coefficients": [],
                "steps": [],
                "greedy_residual_frobenius": float(np.linalg.norm(target)),
            },
        )
    m = base.shape[1]
    angle_count = depth * (m - 1)
    residual = 0.5 * (target + target.T)
    rotations = []
    coefficients = []
    records = []
    for leaf in range(count):
        residual_before = float(np.linalg.norm(residual))

        def diagonal_capture(values: np.ndarray):
            rotation, derivatives = rotation_and_derivatives(values, m, depth)
            rotated = rotation.T @ residual @ rotation
            diagonal = np.diag(rotated)
            value = -0.5 * float(diagonal @ diagonal)
            gradient = np.empty(len(derivatives))
            for index, derivative in enumerate(derivatives):
                diagonal_derivative = 2.0 * np.diag(
                    derivative.T @ residual @ rotation
                )
                gradient[index] = -float(diagonal @ diagonal_derivative)
            return value, gradient

        starts = [np.zeros(angle_count)]
        for start in range(1, 4):
            rng = np.random.default_rng(
                _stable_seed(
                    seed_namespace,
                    seed_key,
                    "collector-tailored",
                    depth,
                    leaf,
                    start,
                )
            )
            starts.append(rng.uniform(-math.pi, math.pi, size=angle_count))
        winner = None
        trials = []
        for start_index, initial in enumerate(starts):
            result = minimize(
                diagonal_capture,
                initial,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-math.pi, math.pi)] * angle_count,
                options={
                    "maxiter": angle_steps,
                    "ftol": 1.0e-13,
                    "gtol": 1.0e-9,
                },
            )
            trials.append({
                "start": start_index,
                "captured_diagonal_frobenius_sq": float(-2.0 * result.fun),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "success": bool(result.success),
                "message": result.message,
            })
            key = (float(result.fun), start_index)
            if winner is None or key < winner[0]:
                winner = (key, np.asarray(result.x), result, start_index)
        if winner is None:
            raise RuntimeError("No collector-tailored shallow rotation was produced")
        _, angles, result, winner_start = winner
        rotation = rotation_and_derivatives(angles, m, depth)[0]
        diagonal = np.diag(rotation.T @ residual @ rotation).copy()
        captured = rotation @ np.diag(diagonal) @ rotation.T
        residual = 0.5 * ((residual - captured) + (residual - captured).T)
        rotations.append(rotation)
        coefficients.append(diagonal)
        records.append({
            "leaf": leaf,
            "winner_start": winner_start,
            "flat_angles": angles.tolist(),
            "greedy_diagonal_coefficients": diagonal.tolist(),
            "collector_residual_frobenius_before": residual_before,
            "collector_residual_frobenius_after": float(np.linalg.norm(residual)),
            "captured_diagonal_frobenius_sq": float(np.vdot(captured, captured).real),
            "optimizer_success": bool(result.success),
            "trials": trials,
        })
    return np.asarray(rotations), np.asarray(coefficients), {
        "method": "collector_tailored_greedy_shallow_diagonalization",
        "selected_flat_angles": [row["flat_angles"] for row in records],
        "greedy_diagonal_coefficients": [
            row["greedy_diagonal_coefficients"] for row in records
        ],
        "steps": records,
        "greedy_residual_frobenius": float(np.linalg.norm(residual)),
    }


def greedy_exact_fill(
    design: np.ndarray,
    target: np.ndarray,
    source_columns: int,
    greedy_extra_coefficients: np.ndarray,
    equality_tolerance: float,
) -> tuple[np.ndarray, float]:
    """Keep tailored extra coefficients and exactly fill with the source bank."""
    vector = np.zeros(design.shape[1])
    if source_columns < 0 or source_columns > design.shape[1]:
        raise ValueError("Invalid source-column boundary")
    extra = np.asarray(greedy_extra_coefficients, dtype=float).reshape(-1)
    if len(extra) != design.shape[1] - source_columns:
        raise ValueError("Greedy extra coefficients do not match dictionary columns")
    vector[source_columns:] = extra
    remaining = target - design[:, source_columns:] @ extra
    source_fill = np.linalg.lstsq(
        design[:, :source_columns], remaining, rcond=equality_tolerance
    )[0]
    source_residual = float(
        np.linalg.norm(design[:, :source_columns] @ source_fill - remaining)
        / max(np.linalg.norm(target), 1.0)
    )
    vector[:source_columns] = source_fill
    if source_residual > equality_tolerance:
        vector = equality_project(vector, design, target)
    return vector, source_residual
