"""Validated greedy exact-fill shallow one-body completion used by SRDD."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .srdd_tensor import rotation_and_derivatives


COLLECTOR_SEED_NAMESPACE = "shared-shallow-collector-redistribution-v1"


def symmetric_vector(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return values[np.triu_indices(values.shape[0])]


def collector_dictionary(rotations: np.ndarray) -> np.ndarray:
    bank = np.asarray(rotations, dtype=float)
    columns = []
    for rotation in bank:
        for orbital in range(rotation.shape[1]):
            vector = rotation[:, orbital]
            columns.append(symmetric_vector(np.outer(vector, vector)))
    dimension = bank.shape[1] * (bank.shape[1] + 1) // 2
    return np.column_stack(columns) if columns else np.empty((dimension, 0))


def collector_matrix(rotations: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Materialize ``sum_t R_t diag(eta_t) R_t.T`` for equality audits."""
    bank = np.asarray(rotations, dtype=float)
    eta = np.asarray(coefficients, dtype=float)
    if bank.ndim != 3 or eta.shape != (len(bank), bank.shape[1]):
        raise ValueError("collector coefficients do not match the rotation bank")
    return np.einsum("tpi,ti,tqi->pq", bank, eta, bank, optimize=True)


def dictionary_diagnostics(
    design: np.ndarray, rcond: float
) -> tuple[int, float, np.ndarray]:
    """Return numerical rank, retained condition number, and singular values."""
    if rcond <= 0.0:
        raise ValueError("rcond must be positive")
    singular_values = np.linalg.svd(
        np.asarray(design, dtype=float), compute_uv=False
    )
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0, float("inf"), singular_values
    threshold = float(rcond) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    condition = (
        float(singular_values[0] / singular_values[rank - 1])
        if rank
        else float("inf")
    )
    return rank, condition, singular_values


def equality_project(
    vector: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    rcond: float = 1.0e-13,
) -> np.ndarray:
    """Project coefficients onto ``design @ vector = target``."""
    values = np.asarray(vector, dtype=float)
    residual = np.asarray(target, dtype=float) - design @ values
    if not np.any(residual):
        return values.copy()
    correction = design.T @ np.linalg.lstsq(
        design @ design.T, residual, rcond=rcond
    )[0]
    return np.asarray(values + correction)


def greedy_exact_fill(
    design: np.ndarray,
    target: np.ndarray,
    source_columns: int,
    greedy_extra_coefficients: np.ndarray,
    equality_tolerance: float,
) -> tuple[np.ndarray, float]:
    """Keep greedy extra coefficients and fill the remainder with source columns."""
    if source_columns < 0 or source_columns > design.shape[1]:
        raise ValueError("invalid source-column boundary")
    vector = np.zeros(design.shape[1], dtype=float)
    extra = np.asarray(greedy_extra_coefficients, dtype=float).reshape(-1)
    if len(extra) != design.shape[1] - source_columns:
        raise ValueError("greedy extra coefficients do not match dictionary columns")
    vector[source_columns:] = extra
    remaining = target - design[:, source_columns:] @ extra
    source_fill = np.linalg.lstsq(
        design[:, :source_columns], remaining, rcond=equality_tolerance
    )[0]
    source_residual = float(
        np.linalg.norm(design[:, :source_columns] @ source_fill - remaining)
        / max(float(np.linalg.norm(target)), 1.0)
    )
    vector[:source_columns] = source_fill
    if source_residual > equality_tolerance:
        vector = equality_project(vector, design, target, equality_tolerance)
    return vector, source_residual


def _seed(seed_key: str, depth: int, leaf: int, start: int) -> int:
    raw = (
        f"{COLLECTOR_SEED_NAMESPACE}|{seed_key}|collector-tailored|"
        f"{depth}|{leaf}|{start}"
    ).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32)


def tailored_rotations(
    collector: np.ndarray,
    depth: int,
    count: int,
    seed_key: str,
    angle_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target = np.asarray(collector, dtype=float)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("collector must be a real square matrix")
    if depth < 1 or count < 0 or angle_steps < 1:
        raise ValueError("depth and angle_steps must be positive; count nonnegative")
    m = target.shape[0]
    residual = 0.5 * (target + target.T)
    rotations, coefficients, records = [], [], []
    for leaf in range(count):
        angle_count = depth * (m - 1)

        def objective(values: np.ndarray):
            rotation, derivatives = rotation_and_derivatives(values, m, depth)
            rotated = rotation.T @ residual @ rotation
            diagonal = np.diag(rotated)
            value = -0.5 * float(diagonal @ diagonal)
            gradient = []
            for derivative in derivatives:
                diagonal_derivative = 2.0 * np.diag(derivative.T @ residual @ rotation)
                gradient.append(-float(diagonal @ diagonal_derivative))
            return value, np.asarray(gradient)

        starts = [np.zeros(angle_count)]
        for start in range(1, 4):
            starts.append(
                np.random.default_rng(_seed(seed_key, depth, leaf, start)).uniform(
                    -math.pi, math.pi, angle_count
                )
            )
        winner = None
        trials = []
        for start_index, initial in enumerate(starts):
            result = minimize(
                objective,
                initial,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-math.pi, math.pi)] * angle_count,
                options={"maxiter": angle_steps, "ftol": 1.0e-13, "gtol": 1.0e-9},
            )
            key = (float(result.fun), start_index)
            trials.append(
                {
                    "start": start_index,
                    "captured_diagonal_frobenius_sq": float(-2.0 * result.fun),
                    "iterations": int(result.nit),
                    "evaluations": int(result.nfev),
                    "success": bool(result.success),
                    "message": str(result.message),
                }
            )
            if winner is None or key < winner[0]:
                winner = (key, np.asarray(result.x), result, start_index)
        assert winner is not None
        _, angles, result, winner_start = winner
        rotation = rotation_and_derivatives(angles, m, depth)[0]
        diagonal = np.diag(rotation.T @ residual @ rotation)
        captured = rotation @ np.diag(diagonal) @ rotation.T
        residual = 0.5 * ((residual - captured) + (residual - captured).T)
        rotations.append(rotation)
        coefficients.append(diagonal)
        records.append(
            {
                "leaf": leaf,
                "winner_start": winner_start,
                "flat_angles": angles.tolist(),
                "greedy_diagonal_coefficients": diagonal.tolist(),
                "captured_frobenius_squared": float(np.vdot(captured, captured).real),
                "residual_frobenius": float(np.linalg.norm(residual)),
                "optimizer_success": bool(result.success),
                "trials": trials,
            }
        )
    return np.asarray(rotations), np.asarray(coefficients), {
        "method": "collector_tailored_greedy_shallow_diagonalization",
        "selected_flat_angles": [row["flat_angles"] for row in records],
        "greedy_diagonal_coefficients": [
            row["greedy_diagonal_coefficients"] for row in records
        ],
        "steps": records,
        "greedy_residual_frobenius": float(np.linalg.norm(residual)),
    }


def exact_shallow_collector(
    source_rotations: np.ndarray,
    collector: np.ndarray,
    depth: int,
    *,
    seed_key: str,
    angle_steps: int,
    tolerance: float = 1.0e-10,
    maximum_extra: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = np.asarray(source_rotations, dtype=float)
    target_matrix = np.asarray(collector, dtype=float)
    if source.ndim != 3 or source.shape[1] != source.shape[2]:
        raise ValueError("source_rotations must have shape (K,m,m)")
    if target_matrix.shape != source.shape[1:]:
        raise ValueError("collector dimension does not match source rotations")
    if depth < 1 or angle_steps < 1 or tolerance <= 0.0:
        raise ValueError("depth, angle_steps, and tolerance must be positive")
    m = target_matrix.shape[0]
    maximum = int(maximum_extra if maximum_extra is not None else 8)
    if maximum < 1:
        raise ValueError("maximum_extra must be at least one")
    target_matrix = 0.5 * (target_matrix + target_matrix.T)
    target = symmetric_vector(target_matrix)
    target_dimension = len(target)
    source_columns = len(source) * m
    attempts = []
    for count in range(1, maximum + 1):
        extra, greedy_extra, extra_audit = tailored_rotations(
            target_matrix, depth, count, seed_key, angle_steps
        )
        bank = np.concatenate((source, extra), axis=0)
        design = collector_dictionary(bank)
        rank, condition, singular_values = dictionary_diagnostics(design, tolerance)
        particular = np.linalg.lstsq(design, target, rcond=tolerance)[0]
        particular = equality_project(particular, design, target, tolerance)
        particular_relative = float(
            np.linalg.norm(design @ particular - target)
            / max(float(np.linalg.norm(target)), 1.0)
        )
        greedy_vector, source_fill_residual = greedy_exact_fill(
            design,
            target,
            source_columns,
            greedy_extra,
            tolerance,
        )
        # The formal greedy policy keeps the tailored seed, fills from the
        # source bank, and finally restores the exact affine equality.
        greedy_vector = equality_project(greedy_vector, design, target, tolerance)
        greedy_relative = float(
            np.linalg.norm(design @ greedy_vector - target)
            / max(float(np.linalg.norm(target)), 1.0)
        )
        attempt = {
            "extra_rotation_count": count,
            "dictionary_rank": rank,
            "dictionary_columns": int(design.shape[1]),
            "symmetric_target_dimension": target_dimension,
            "dictionary_condition_on_numerical_range": condition,
            "smallest_retained_singular_value": (
                float(singular_values[rank - 1]) if rank else 0.0
            ),
            "relative_reconstruction_residual": particular_relative,
            "greedy_exact_fill_relative_reconstruction_residual": greedy_relative,
            "full_symmetric_dictionary_rank": rank == target_dimension,
        }
        attempts.append(attempt)
        if not (
            rank == target_dimension
            and particular_relative <= tolerance
            and greedy_relative <= tolerance
        ):
            continue
        coefficients = greedy_vector.reshape(len(bank), m)
        reconstructed = collector_matrix(bank, coefficients)
        matrix_residual = float(np.linalg.norm(reconstructed - target_matrix, ord="fro"))
        matrix_relative = matrix_residual / max(
            float(np.linalg.norm(target_matrix, ord="fro")), 1.0
        )
        if matrix_relative > 5.0 * tolerance:
            continue
        return bank, coefficients, {
            "status": "PASS",
            "mode": "augment",
            "objective": "greedy",
            "selected_candidate": "collector_tailored_greedy_exact_fill",
            "source_rotation_count": len(source),
            "extra_rotation_count": count,
            "extra_rotations": count,
            "measurement_setting_count": len(bank),
            "dictionary_rank": rank,
            "dictionary_columns": int(design.shape[1]),
            "dictionary_nullity": int(design.shape[1] - rank),
            "symmetric_target_dimension": target_dimension,
            "dictionary_condition_on_numerical_range": condition,
            "smallest_retained_singular_value": (
                float(singular_values[rank - 1]) if rank else 0.0
            ),
            "attempted_extra_counts": attempts,
            "extra_rotation_construction": extra_audit,
            "tailored_rotation_audit": extra_audit,
            "equality_tolerance": tolerance,
            "relative_equality_residual": matrix_relative,
            "relative_reconstruction_residual": matrix_relative,
            "matrix_reconstruction_residual_frobenius": matrix_residual,
            "matrix_relative_reconstruction_residual": matrix_relative,
            "greedy_existing_bank_fill_relative_residual": source_fill_residual,
            "coefficients": coefficients.tolist(),
            "depth": depth,
            "all_measurement_settings_depth_limited": True,
            "independent_collector_setting_present": False,
        }
    raise RuntimeError(
        "depth-matched greedy exact-fill completion did not obtain a full-rank "
        f"exact dictionary with 1..{maximum} extra rotations; attempts={attempts}"
    )
