#!/usr/bin/env python3
"""Tensor-native optimizer for the standalone shallow-RC-DF algorithm.

Each spatial orbital rotation is parameterized directly as ``depth`` nearest-
neighbour Givens macro-layers.  A macro-layer applies its even bonds first and
its odd bonds second.  For every angle objective evaluation the complete set
of symmetric ``Z_t`` tensors is ridge-refitted exactly.  This variable-
projection implementation therefore optimizes the constrained shallow
objective itself; it never fits unrestricted RC-DF and truncates its circuit.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize


VERSION = "shallow-rcdf-even-odd-variable-projection-v2-overcomplete"


def givens_layout(m: int, depth: int) -> list[tuple[int, str, int, int]]:
    if m < 2 or depth < 1:
        raise ValueError("m >= 2 and depth >= 1 are required")
    layout: list[tuple[int, str, int, int]] = []
    for layer in range(depth):
        layout.extend((layer, "even", p, p + 1) for p in range(0, m - 1, 2))
        layout.extend((layer, "odd", p, p + 1) for p in range(1, m - 1, 2))
    if len(layout) != depth * (m - 1):
        raise RuntimeError("Nearest-neighbour even/odd layout has wrong size")
    return layout


def _givens(m: int, p: int, q: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    cosine = math.cos(theta)
    sine = math.sin(theta)
    gate = np.eye(m)
    derivative = np.zeros((m, m))
    gate[p, p] = cosine
    gate[p, q] = -sine
    gate[q, p] = sine
    gate[q, q] = cosine
    derivative[p, p] = -sine
    derivative[p, q] = -cosine
    derivative[q, p] = cosine
    derivative[q, q] = -sine
    return gate, derivative


def rotation_and_derivatives(
    angles: np.ndarray, m: int, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return U and dU/dtheta for U=U_O^d U_E^d ... U_O^1 U_E^1."""
    values = np.asarray(angles, dtype=float).reshape(-1)
    layout = givens_layout(m, depth)
    if len(values) != len(layout):
        raise ValueError(f"Expected {len(layout)} angles, received {len(values)}")
    gates: list[np.ndarray] = []
    gate_derivatives: list[np.ndarray] = []
    for theta, (_, _, p, q) in zip(values, layout):
        gate, derivative = _givens(m, p, q, float(theta))
        gates.append(gate)
        gate_derivatives.append(derivative)

    prefixes = [np.eye(m)]
    for gate in gates:
        prefixes.append(gate @ prefixes[-1])
    rotation = prefixes[-1]

    derivatives: list[np.ndarray] = [np.empty_like(rotation) for _ in gates]
    left = np.eye(m)
    for index in range(len(gates) - 1, -1, -1):
        derivatives[index] = left @ gate_derivatives[index] @ prefixes[index]
        left = left @ gates[index]
    return rotation, np.asarray(derivatives)


def _symmetric_pairs(m: int) -> list[tuple[int, int]]:
    return [(k, l) for k in range(m) for l in range(k, m)]


def _design_matrix(rotations: np.ndarray) -> np.ndarray:
    leaves, m, _ = rotations.shape
    pairs = _symmetric_pairs(m)
    design = np.empty((m**4, leaves * len(pairs)), dtype=float)
    column = 0
    for unitary in rotations:
        projectors = np.einsum("pk,qk->pqk", unitary, unitary).reshape(m * m, m)
        for k, l in pairs:
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
        for k, l in _symmetric_pairs(m):
            tensors[leaf, k, l] = tensors[leaf, l, k] = vector[offset]
            offset += 1
    return tensors


def _solve_z(rotations: np.ndarray, eri: np.ndarray, rho: float):
    leaves, m, _ = rotations.shape
    pairs = _symmetric_pairs(m)
    design = _design_matrix(rotations)
    weights = np.tile(
        np.asarray([1.0 if k == l else 2.0 for k, l in pairs]), leaves
    )
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
    residual_sq = float(np.vdot(residual, residual).real)
    regularizer = float(rho * np.vdot(tensors, tensors).real)
    return tensors, fitted, 0.5 * residual_sq + regularizer, residual_sq, regularizer, solver


def _xdf_initial(
    eri: np.ndarray, leaves: int, seed_key: str
) -> tuple[np.ndarray, np.ndarray]:
    m = eri.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(eri.reshape(m * m, m * m))
    nonzero_order = [
        int(index)
        for index in np.argsort(np.abs(eigenvalues))[::-1]
        if abs(eigenvalues[index]) > 1.0e-12
    ]
    # X-DF is only a starting point for the joint shallow variable-projection
    # fit.  A geometry can have fewer than ten numerically nonzero X-DF modes
    # even though the four-orbital shallow-RCDF parameterization still permits
    # K=10.  Fill any requested extra initial leaves with the remaining
    # zero/near-zero eigen-directions; their Z tensors and shallow rotations
    # are then optimized jointly with every other leaf.  No zero direction is
    # reported as a fitted result without that subsequent optimization.
    remaining_order = [
        int(index)
        for index in np.argsort(np.abs(eigenvalues))[::-1]
        if int(index) not in set(nonzero_order)
    ]
    order = nonzero_order + remaining_order
    rotations = []
    outer_values = []
    for index in order[: min(leaves, len(order))]:
        matrix = eigenvectors[:, index].reshape(m, m)
        matrix = 0.5 * (matrix + matrix.T)
        _, unitary = np.linalg.eigh(matrix)
        for column in range(m):
            pivot = int(np.argmax(np.abs(unitary[:, column])))
            if unitary[pivot, column] < 0.0:
                unitary[:, column] *= -1.0
        if np.linalg.det(unitary) < 0.0:
            unitary[:, -1] *= -1.0
        rotations.append(unitary)
        outer_values.append(float(eigenvalues[index]))

    # RC-DF itself is an overcomplete nonlinear factorization and does not
    # require K <= m^2.  X-DF supplies at most m^2 spectral initializers, so
    # requested extra leaves receive deterministic orthogonal targets.  They
    # are subsequently projected into the declared shallow Givens ansatz and
    # optimized jointly with every other leaf; no random target is reported
    # as a fitted result without that optimization.  This branch is first
    # needed by the 7-orbital H2O K_max=50 experiment (m^2=49).
    for leaf in range(len(order), leaves):
        payload = f"{VERSION}|{seed_key}|overcomplete|leaf{leaf}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        unitary, triangular = np.linalg.qr(rng.normal(size=(m, m)))
        signs = np.sign(np.diag(triangular))
        signs[signs == 0.0] = 1.0
        unitary *= signs
        if np.linalg.det(unitary) < 0.0:
            unitary[:, -1] *= -1.0
        rotations.append(unitary)
        outer_values.append(0.0)
    return np.asarray(rotations), np.asarray(outer_values)


def _stable_seed(seed_key: str, depth: int, leaf: int, start: int) -> int:
    payload = f"{VERSION}|{seed_key}|d{depth}|leaf{leaf}|start{start}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _fit_target_rotation(
    target: np.ndarray,
    depth: int,
    seed_key: str,
    leaf: int,
    maximum_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    m = target.shape[0]
    count = depth * (m - 1)

    def objective(values: np.ndarray):
        rotation, derivatives = rotation_and_derivatives(values, m, depth)
        delta = rotation - target
        value = 0.5 * float(np.vdot(delta, delta).real)
        gradient = np.asarray(
            [float(np.vdot(delta, derivative).real) for derivative in derivatives]
        )
        return value, gradient

    starts = [np.zeros(count)]
    rng = np.random.default_rng(_stable_seed(seed_key, depth, leaf, 1))
    starts.append(rng.uniform(-math.pi, math.pi, size=count))
    trials = []
    best = None
    for start_index, initial in enumerate(starts):
        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-math.pi, math.pi)] * count,
            options={"maxiter": maximum_iterations, "ftol": 1.0e-13, "gtol": 1.0e-9},
        )
        record = {
            "start": start_index,
            "initial_objective": float(objective(initial)[0]),
            "final_objective": float(result.fun),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "success": bool(result.success),
        }
        trials.append(record)
        key = (float(result.fun), start_index)
        if best is None or key < best[0]:
            best = (key, np.asarray(result.x), start_index)
    if best is None:
        raise RuntimeError("No shallow rotation initializer was produced")
    return best[1], {"winner_start": best[2], "trials": trials}


def optimize_shallow_rcdf(
    eri: np.ndarray,
    leaves: int,
    depth: int,
    rho: float,
    cycles: int,
    angle_steps: int,
    gradient_tolerance: float,
    seed_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a depth-constrained shallow-RC-DF bank by variable projection."""
    tensor = np.asarray(eri, dtype=float)
    if tensor.ndim != 4 or len(set(tensor.shape)) != 1:
        raise ValueError("ERI must be a real m x m x m x m tensor")
    if leaves < 1 or depth not in (1, 2, 3):
        raise ValueError("leaves must be positive and depth must be 1, 2, or 3")
    if rho < 0.0 or cycles < 1 or angle_steps < 1:
        raise ValueError("rho must be nonnegative; cycles and angle_steps positive")
    m = tensor.shape[0]
    xdf_rotations, outer_values = _xdf_initial(tensor, leaves, seed_key)
    initializer_records = []
    angle_rows = []
    for leaf, target in enumerate(xdf_rotations):
        angles, record = _fit_target_rotation(
            target,
            depth,
            seed_key,
            leaf,
            maximum_iterations=max(20, 8 * angle_steps),
        )
        angle_rows.append(angles)
        initializer_records.append(record)
    angles = np.asarray(angle_rows, dtype=float)

    def evaluate(flat: np.ndarray):
        matrix = np.asarray(flat, dtype=float).reshape(leaves, depth * (m - 1))
        rotations = []
        derivative_banks = []
        for row in matrix:
            rotation, derivatives = rotation_and_derivatives(row, m, depth)
            rotations.append(rotation)
            derivative_banks.append(derivatives)
        rotations_array = np.asarray(rotations)
        tensors, fitted, objective, residual_sq, regularizer, solver = _solve_z(
            rotations_array, tensor, rho
        )
        residual = fitted - tensor
        gradient_rows = []
        for rotation, z, derivatives in zip(
            rotations_array, tensors, derivative_banks
        ):
            euclidean = 4.0 * np.einsum(
                "pqrs,qk,kl,rl,sl->pk",
                residual,
                rotation,
                z,
                rotation,
                rotation,
                optimize=True,
            )
            gradient_rows.append(
                [float(np.vdot(euclidean, derivative).real) for derivative in derivatives]
            )
        return (
            objective,
            np.asarray(gradient_rows).reshape(-1),
            rotations_array,
            tensors,
            fitted,
            residual_sq,
            regularizer,
            solver,
        )

    initial = evaluate(angles.reshape(-1))
    initial_objective = float(initial[0])
    history = []
    for cycle in range(cycles):
        def value_gradient(flat: np.ndarray):
            result = evaluate(flat)
            return float(result[0]), np.asarray(result[1])

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
                "regularizer": float(current[6]),
                "angle_gradient_norm": float(np.linalg.norm(current[1])),
                "optimizer_iterations": int(result.nit),
                "objective_evaluations": int(result.nfev),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        if float(np.linalg.norm(current[1])) <= gradient_tolerance:
            break

    final = evaluate(angles.reshape(-1))
    rotations, tensors, fitted = final[2], final[3], final[4]
    layout = givens_layout(m, depth)
    angle_records = []
    for leaf, row in enumerate(angles):
        angle_records.append(
            {
                "leaf": leaf,
                "flat_angles": row.tolist(),
                "gates": [
                    {
                        "layer": layer + 1,
                        "sublayer": sublayer,
                        "p": p,
                        "q": q,
                        "theta": float(theta),
                    }
                    for theta, (layer, sublayer, p, q) in zip(row, layout)
                ],
            }
        )
    audit = {
        "version": VERSION,
        "method": "joint shallow-angle/Z variable projection",
        "depth": depth,
        "leaves": leaves,
        "rho": rho,
        "cycles_requested": cycles,
        "angle_steps_per_cycle": angle_steps,
        "cycles_completed": len(history),
        "initial_objective": initial_objective,
        "final_objective": float(final[0]),
        "final_eri_residual_frobenius": math.sqrt(max(float(final[5]), 0.0)),
        "final_regularizer": float(final[6]),
        "final_angle_gradient_norm": float(np.linalg.norm(final[1])),
        "z_solver": final[7],
        "gradient_tolerance": gradient_tolerance,
        "spatial_angles_per_leaf": depth * (m - 1),
        "logical_spin_givens_per_leaf": 2 * depth * (m - 1),
        "native_givens_depth_per_leaf": 2 * depth,
        # A real fermionic Givens rotation is exactly an XX+YY rotation up to
        # one-qubit phases and has a two-CNOT decomposition.  The alpha and
        # beta spin chains therefore contain 4*d_R*(m-1) CNOTs in total, while
        # their disjoint gates run in parallel at CNOT depth 4*d_R.
        "logical_cnot_count_per_leaf": 4 * depth * (m - 1),
        "logical_cnot_depth_per_leaf": 4 * depth,
        # Retain the historical key for schema compatibility, but correct its
        # former four-CNOT-per-Givens overestimate to the exact logical count.
        "unreduced_cnot_upper_bound_per_leaf": 4 * depth * (m - 1),
        "outer_xdf_eigenvalues": outer_values.tolist(),
        "xdf_used_only_for_shallow_angle_initialization": True,
        "unrestricted_rotation_truncation_used": False,
        "initializer_records": initializer_records,
        "angle_records": angle_records,
        "history": history,
    }
    return rotations, tensors, fitted, audit
