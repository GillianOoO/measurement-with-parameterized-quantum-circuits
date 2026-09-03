"""Tensor-native shallow rotated-density variable-projection fit."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize


VERSION = "srdd-even-odd-variable-projection-v2"


def givens_layout(m: int, depth: int) -> list[tuple[int, str, int, int]]:
    if m < 2 or depth < 1:
        raise ValueError("m >= 2 and depth >= 1 are required")
    layout: list[tuple[int, str, int, int]] = []
    for layer in range(depth):
        layout.extend((layer, "even", p, p + 1) for p in range(0, m - 1, 2))
        layout.extend((layer, "odd", p, p + 1) for p in range(1, m - 1, 2))
    return layout


def _givens(m: int, p: int, q: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    cosine, sine = math.cos(theta), math.sin(theta)
    gate = np.eye(m)
    derivative = np.zeros((m, m))
    gate[p, p], gate[p, q], gate[q, p], gate[q, q] = cosine, -sine, sine, cosine
    derivative[p, p], derivative[p, q] = -sine, -cosine
    derivative[q, p], derivative[q, q] = cosine, -sine
    return gate, derivative


def rotation_and_derivatives(
    angles: np.ndarray, m: int, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(angles, dtype=float).reshape(-1)
    layout = givens_layout(m, depth)
    if len(values) != len(layout):
        raise ValueError(f"expected {len(layout)} angles, received {len(values)}")
    gates, gate_derivatives = [], []
    for theta, (_, _, p, q) in zip(values, layout):
        gate, derivative = _givens(m, p, q, float(theta))
        gates.append(gate)
        gate_derivatives.append(derivative)
    prefixes = [np.eye(m)]
    for gate in gates:
        prefixes.append(gate @ prefixes[-1])
    rotation = prefixes[-1]
    derivatives = [np.empty_like(rotation) for _ in gates]
    left = np.eye(m)
    for index in range(len(gates) - 1, -1, -1):
        derivatives[index] = left @ gate_derivatives[index] @ prefixes[index]
        left = left @ gates[index]
    return rotation, np.asarray(derivatives)


def _pairs(m: int) -> list[tuple[int, int]]:
    return [(k, l) for k in range(m) for l in range(k, m)]


def _design_matrix(rotations: np.ndarray) -> np.ndarray:
    leaves, m, _ = rotations.shape
    design = np.empty((m**4, leaves * len(_pairs(m))), dtype=float)
    column = 0
    for rotation in rotations:
        projectors = np.einsum("pk,qk->pqk", rotation, rotation).reshape(m * m, m)
        for k, l in _pairs(m):
            feature = np.outer(projectors[:, k], projectors[:, l])
            if k != l:
                feature += np.outer(projectors[:, l], projectors[:, k])
            design[:, column] = feature.reshape(-1)
            column += 1
    return design


def _unpack_z(vector: np.ndarray, leaves: int, m: int) -> np.ndarray:
    tensors = np.zeros((leaves, m, m), dtype=float)
    offset = 0
    for leaf in range(leaves):
        for k, l in _pairs(m):
            tensors[leaf, k, l] = tensors[leaf, l, k] = vector[offset]
            offset += 1
    return tensors


def _solve_z(rotations: np.ndarray, eri: np.ndarray, rho: float):
    leaves, m, _ = rotations.shape
    pairs = _pairs(m)
    design = _design_matrix(rotations)
    weights = np.tile(np.asarray([1.0 if k == l else 2.0 for k, l in pairs]), leaves)
    normal = design.T @ design
    normal.flat[:: normal.shape[0] + 1] += 2.0 * rho * weights
    rhs = design.T @ eri.reshape(-1)
    try:
        packed = la.solve(normal, rhs, assume_a="pos", check_finite=False)
        solver = "dense_cholesky_normal_equations"
    except la.LinAlgError:
        packed = la.lstsq(normal, rhs, check_finite=False)[0]
        solver = "dense_symmetric_lstsq_fallback"
    tensors = _unpack_z(packed, leaves, m)
    fitted = (design @ packed).reshape(eri.shape)
    residual = fitted - eri
    residual_squared = float(np.vdot(residual, residual).real)
    regularizer = float(rho * np.vdot(tensors, tensors).real)
    return tensors, fitted, 0.5 * residual_squared + regularizer, residual_squared, regularizer, solver


def _xdf_initial(eri: np.ndarray, leaves: int, seed_key: str) -> np.ndarray:
    m = eri.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(eri.reshape(m * m, m * m))
    order = list(np.argsort(np.abs(eigenvalues))[::-1])
    rotations = []
    for index in order[: min(leaves, len(order))]:
        matrix = eigenvectors[:, int(index)].reshape(m, m)
        matrix = 0.5 * (matrix + matrix.T)
        _, rotation = np.linalg.eigh(matrix)
        for column in range(m):
            pivot = int(np.argmax(np.abs(rotation[:, column])))
            if rotation[pivot, column] < 0.0:
                rotation[:, column] *= -1.0
        if np.linalg.det(rotation) < 0.0:
            rotation[:, -1] *= -1.0
        rotations.append(rotation)
    while len(rotations) < leaves:
        payload = f"{VERSION}|{seed_key}|overcomplete|{len(rotations)}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        rotation, triangular = np.linalg.qr(rng.normal(size=(m, m)))
        signs = np.sign(np.diag(triangular))
        signs[signs == 0.0] = 1.0
        rotation *= signs
        if np.linalg.det(rotation) < 0.0:
            rotation[:, -1] *= -1.0
        rotations.append(rotation)
    return np.asarray(rotations)


def _seed(seed_key: str, depth: int, leaf: int, start: int) -> int:
    raw = f"{VERSION}|{seed_key}|d{depth}|leaf{leaf}|start{start}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32)


def _fit_target_rotation(
    target: np.ndarray, depth: int, seed_key: str, leaf: int, maximum_iterations: int
) -> tuple[np.ndarray, dict[str, Any]]:
    m = target.shape[0]
    count = depth * (m - 1)

    def objective(values: np.ndarray):
        rotation, derivatives = rotation_and_derivatives(values, m, depth)
        delta = rotation - target
        return 0.5 * float(np.vdot(delta, delta).real), np.asarray(
            [float(np.vdot(delta, derivative).real) for derivative in derivatives]
        )

    starts = [np.zeros(count)]
    starts.append(np.random.default_rng(_seed(seed_key, depth, leaf, 1)).uniform(-math.pi, math.pi, count))
    trials, winner = [], None
    for start_index, initial in enumerate(starts):
        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-math.pi, math.pi)] * count,
            options={"maxiter": maximum_iterations, "ftol": 1.0e-13, "gtol": 1.0e-9},
        )
        trials.append(
            {
                "start": start_index,
                "objective": float(result.fun),
                "iterations": int(result.nit),
                "success": bool(result.success),
            }
        )
        key = (float(result.fun), start_index)
        if winner is None or key < winner[0]:
            winner = (key, np.asarray(result.x), start_index)
    assert winner is not None
    return winner[1], {"winner_start": winner[2], "trials": trials}


def optimize_srdd_tensor(
    eri: np.ndarray,
    leaves: int,
    depth: int,
    rho: float,
    cycles: int,
    angle_steps: int,
    gradient_tolerance: float,
    seed_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    tensor = np.asarray(eri, dtype=float)
    if tensor.ndim != 4 or len(set(tensor.shape)) != 1:
        raise ValueError("ERI must be a real m x m x m x m tensor")
    if leaves < 1 or depth not in (1, 2, 3):
        raise ValueError("leaves must be positive and depth must be 1, 2, or 3")
    if rho < 0.0 or cycles < 1 or angle_steps < 1:
        raise ValueError("rho must be nonnegative; cycles and angle_steps positive")
    m = tensor.shape[0]
    initial_rotations = _xdf_initial(tensor, leaves, seed_key)
    angle_rows, initializer = [], []
    for leaf, target in enumerate(initial_rotations):
        angles, audit = _fit_target_rotation(
            target, depth, seed_key, leaf, max(20, 8 * angle_steps)
        )
        angle_rows.append(angles)
        initializer.append(audit)
    angles = np.asarray(angle_rows, dtype=float)

    def evaluate(flat: np.ndarray):
        matrix = np.asarray(flat, dtype=float).reshape(leaves, depth * (m - 1))
        rotations, derivatives = [], []
        for row in matrix:
            rotation, bank = rotation_and_derivatives(row, m, depth)
            rotations.append(rotation)
            derivatives.append(bank)
        rotations = np.asarray(rotations)
        tensors, fitted, objective, residual_squared, regularizer, solver = _solve_z(
            rotations, tensor, rho
        )
        residual = fitted - tensor
        gradient = []
        for rotation, z_tensor, derivative_bank in zip(rotations, tensors, derivatives):
            euclidean = 4.0 * np.einsum(
                "pqrs,qk,kl,rl,sl->pk",
                residual,
                rotation,
                z_tensor,
                rotation,
                rotation,
                optimize=True,
            )
            gradient.append(
                [float(np.vdot(euclidean, derivative).real) for derivative in derivative_bank]
            )
        return (
            objective,
            np.asarray(gradient).reshape(-1),
            rotations,
            tensors,
            fitted,
            residual_squared,
            regularizer,
            solver,
        )

    history = []
    for cycle in range(cycles):
        def value_gradient(flat: np.ndarray):
            evaluated = evaluate(flat)
            return float(evaluated[0]), np.asarray(evaluated[1])

        result = minimize(
            value_gradient,
            angles.reshape(-1),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-math.pi, math.pi)] * angles.size,
            options={
                "maxiter": angle_steps,
                "ftol": 1.0e-13,
                "gtol": gradient_tolerance,
                "maxls": 20,
            },
        )
        angles = np.asarray(result.x).reshape(angles.shape)
        current = evaluate(angles.reshape(-1))
        history.append(
            {
                "cycle": cycle + 1,
                "objective": float(current[0]),
                "eri_residual_frobenius": math.sqrt(max(float(current[5]), 0.0)),
                "gradient_norm": float(np.linalg.norm(current[1])),
                "iterations": int(result.nit),
                "success": bool(result.success),
            }
        )
        if float(np.linalg.norm(current[1])) <= gradient_tolerance:
            break
    final = evaluate(angles.reshape(-1))
    return final[2], final[3], final[4], {
        "version": VERSION,
        "depth": depth,
        "leaves": leaves,
        "rho": rho,
        "final_objective": float(final[0]),
        "final_eri_residual_frobenius": math.sqrt(max(float(final[5]), 0.0)),
        "final_gradient_norm": float(np.linalg.norm(final[1])),
        "z_solver": final[7],
        "initializer": initializer,
        "angles": angles.tolist(),
        "history": history,
    }
