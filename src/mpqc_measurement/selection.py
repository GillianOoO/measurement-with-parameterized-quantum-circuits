"""Stabilizer-calibrated balanced fixed-shot selection shared by GPD and SRDD."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .allocation import integer_range_allocation
from .models import Fragment, HamiltonianData


CALIBRATION_SIZES = tuple(range(1, 11))
CALIBRATION_SHOTS = (100, 200, 300, 500, 800, 1200, 2000, 3000)
SAMPLING_RATIO_RELATIVE_THRESHOLD = 1.0e-12
SAMPLING_RATIO_ABSOLUTE_THRESHOLD = 1.0e-30


@dataclass
class FrontierPoint:
    size: int
    constant: float
    fragments: list[Fragment]
    approximate: np.ndarray
    residual: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialStallTracker:
    """Single-target-budget replay of the calibrated GPD stall rule."""

    relative_improvement: float = 2.0e-3
    patience: int = 5
    minimum_k: int = 10
    calibration_cutoff: int = 10
    anchor: float = math.inf
    stalled: int = 0
    stop_k: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def update(self, k: int, score: float | None) -> bool:
        if self.stop_k is not None:
            raise RuntimeError("the material-stall tracker has already stopped")
        materially_improved = (
            score is not None
            and score
            < self.anchor * (1.0 - float(self.relative_improvement))
        )
        if materially_improved:
            self.anchor = float(score)
        counted = int(k) > int(self.calibration_cutoff)
        if counted:
            self.stalled = 0 if materially_improved else self.stalled + 1
        else:
            self.stalled = 0
        should_stop = (
            int(k) >= max(int(self.minimum_k), int(self.calibration_cutoff) + 1)
            and self.stalled >= int(self.patience)
        )
        self.events.append(
            {
                "K": int(k),
                "score": None if score is None else float(score),
                "materially_improved": bool(materially_improved),
                "stall_counted": bool(counted),
                "consecutive_nonmaterial_boundaries": int(self.stalled),
            }
        )
        if should_stop:
            self.stop_k = int(k)
        return bool(should_stop)


def sector_indices(n_qubits: int, particle_number: int | None) -> np.ndarray:
    if particle_number is None:
        return np.arange(2**n_qubits, dtype=np.int64)
    return np.asarray(
        [index for index in range(2**n_qubits) if index.bit_count() == particle_number],
        dtype=np.int64,
    )


def _largest_remainder_counts(total: int) -> tuple[int, int, int]:
    raw = np.asarray((0.6, 0.2, 0.2)) * int(total)
    counts = np.floor(raw).astype(int)
    order = np.lexsort((np.arange(3), -(raw - counts)))
    counts[order[: int(total - np.sum(counts))]] += 1
    return tuple(int(value) for value in counts)


def _split_map(
    order: np.ndarray, counts: tuple[int, int, int] | None = None
) -> dict[int, int]:
    train, validation, test = counts or _largest_remainder_counts(len(order))
    if train + validation + test != len(order):
        raise ValueError("split counts do not match the group order")
    output: dict[int, int] = {}
    for position, slot in enumerate(order):
        output[int(slot)] = 0 if position < train else 1 if position < train + validation else 2
    return output


def generate_grouped_stabilizer_probes(
    indices: np.ndarray, dimension: int, *, seed: int = 918273
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate the frozen grouped determinant/two-determinant probe design.

    Sectors large enough for the molecular catalog use 28 determinants and
    118 four-phase pair groups (500 states, split 300/100/100).  Smaller
    sectors use the complete nonduplicated catalog.  In particular, the
    four-qubit full space gives 496 states split 298/99/99.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("stabilizer calibration requires at least two accessible basis states")
    if np.any(indices < 0) or np.any(indices >= int(dimension)):
        raise ValueError("probe-sector indices lie outside the Hilbert space")
    rng = np.random.default_rng(int(seed))
    pairs = [
        (int(a), int(b))
        for position, a in enumerate(indices)
        for b in indices[position + 1 :]
    ]
    if len(indices) >= 28 and len(pairs) >= 118:
        determinant_slots = rng.permutation(len(indices))[:28]
        pair_slots = rng.choice(len(pairs), size=118, replace=False)
        determinant_splits = _split_map(np.arange(28), (20, 4, 4))
        pair_splits = _split_map(rng.permutation(118), (70, 24, 24))
        design = "frozen-500-determinant-plus-four-phase-pair"
    else:
        determinant_slots = rng.permutation(len(indices))
        pair_slots = rng.permutation(len(pairs))
        determinant_splits = _split_map(np.arange(len(determinant_slots)))
        pair_splits = _split_map(np.arange(len(pair_slots)))
        design = "complete-nonduplicated-determinant-plus-four-phase-pair"
    states: list[np.ndarray] = []
    splits: list[int] = []
    for group_slot, sector_slot in enumerate(determinant_slots):
        basis_index = int(indices[int(sector_slot)])
        state = np.zeros(dimension, dtype=np.complex128)
        state[int(basis_index)] = 1.0
        states.append(state)
        splits.append(determinant_splits[group_slot])
    for group_slot, pair_slot in enumerate(pair_slots):
        left, right = pairs[int(pair_slot)]
        for phase in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            state = np.zeros(dimension, dtype=np.complex128)
            state[left] = 1.0 / math.sqrt(2.0)
            state[right] = np.exp(1j * phase) / math.sqrt(2.0)
            states.append(state)
            splits.append(pair_splits[group_slot])
    state_array = np.column_stack(states)
    split_array = np.asarray(splits, dtype=np.int8)
    counts = {
        "train": int(np.count_nonzero(split_array == 0)),
        "validation": int(np.count_nonzero(split_array == 1)),
        "test": int(np.count_nonzero(split_array == 2)),
    }
    if design.startswith("frozen-500") and counts != {
        "train": 300,
        "validation": 100,
        "test": 100,
    }:
        raise RuntimeError(f"frozen 500-probe split mismatch: {counts}")
    contiguous = np.ascontiguousarray(state_array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.view(np.uint8))
    return state_array, split_array, {
        "design": design,
        "seed": int(seed),
        "probe_count": int(state_array.shape[1]),
        "determinant_probe_count": int(len(determinant_slots)),
        "pair_group_count": int(len(pair_slots)),
        "state_split_counts": counts,
        "group_disjoint_splits": True,
        "probe_array_sha256": digest.hexdigest(),
    }


def _expectation(states: np.ndarray, operator: np.ndarray) -> np.ndarray:
    values = np.sum(np.conj(states) * (operator @ states), axis=0)
    if np.max(np.abs(values.imag)) > 2.0e-8:
        raise FloatingPointError("Hermitian expectation has a material imaginary part")
    return values.real


def _fragment_variances(states: np.ndarray, fragment: Fragment) -> np.ndarray:
    operator = fragment.matrix
    acted = operator @ states
    means = np.sum(np.conj(states) * acted, axis=0)
    seconds = np.sum(np.square(np.abs(acted)), axis=0)
    if float(np.max(np.abs(means.imag))) > 2.0e-8:
        raise FloatingPointError("fragment expectation has a material imaginary part")
    return np.maximum(seconds.real - np.square(means.real), 0.0)


def _feature(point: FrontierPoint, indices: np.ndarray) -> float:
    block = point.residual[np.ix_(indices, indices)]
    return float(np.linalg.norm(block, ord="fro"))


def _sampling_feature(point: FrontierPoint, total_shots: int) -> tuple[float, np.ndarray]:
    ranges = [fragment.half_range for fragment in point.fragments]
    allocation = integer_range_allocation(ranges, total_shots)
    value = sum(
        half_range * half_range / int(count)
        for half_range, count in zip(ranges, allocation)
        if int(count) > 0
    )
    return math.sqrt(max(float(value), 0.0)), allocation


def calibrate_weights(
    data: HamiltonianData,
    frontier: list[FrontierPoint],
    *,
    seed: int = 918273,
) -> tuple[float, float, dict[str, Any]]:
    indices = sector_indices(data.n_qubits, data.particle_number)
    probes, splits, probe_audit = generate_grouped_stabilizer_probes(
        indices, data.dimension, seed=seed
    )
    exact = _expectation(probes, data.matrix)
    train = splits == 0
    by_size: dict[int, FrontierPoint] = {}
    for point in frontier:
        if point.size in by_size:
            raise ValueError(f"calibration frontier has duplicate K={point.size}")
        by_size[point.size] = point
    missing = [size for size in CALIBRATION_SIZES if size not in by_size]
    if missing:
        raise ValueError(
            "stabilizer calibration requires independently optimized candidates "
            f"K=1,...,10; missing {missing}. Disable calibration only when an "
            "uncalibrated proxy is explicitly intended."
        )
    calibration_points = [by_size[size] for size in CALIBRATION_SIZES]
    approximation_strata: list[tuple[int, float]] = []
    sampling_candidates: list[tuple[int, int, float, float]] = []
    heldout: dict[int, dict[str, list[float]]] = {
        1: {"a_actual": [], "a_feature": [], "s_actual": [], "s_feature": []},
        2: {"a_actual": [], "a_feature": [], "s_actual": [], "s_feature": []},
    }
    for point in calibration_points:
        x_a = _feature(point, indices)
        if x_a > 0.0:
            predicted = _expectation(probes, point.approximate)
            bias = predicted - exact
            approximation_strata.append(
                (point.size, float(np.mean(np.square(bias[train]))) / (x_a * x_a))
            )
            for split_id in (1, 2):
                mask = splits == split_id
                heldout[split_id]["a_actual"].extend(np.abs(bias[mask]).tolist())
                heldout[split_id]["a_feature"].extend(
                    np.full(int(np.count_nonzero(mask)), x_a).tolist()
                )
        fragment_variances = np.asarray(
            [_fragment_variances(probes, fragment) for fragment in point.fragments]
        )
        for shots in CALIBRATION_SHOTS:
            try:
                x_s, allocation = _sampling_feature(point, shots)
            except ValueError:
                continue
            q_value = x_s * x_s
            if q_value <= 0.0:
                continue
            actual = np.zeros(probes.shape[1], dtype=float)
            for row, count in zip(fragment_variances, allocation):
                if int(count) > 0:
                    actual += row / int(count)
            sampling_candidates.append(
                (point.size, int(shots), q_value, float(np.mean(actual[train])))
            )
            for split_id in (1, 2):
                mask = splits == split_id
                heldout[split_id]["s_actual"].extend(
                    np.sqrt(np.maximum(actual[mask], 0.0)).tolist()
                )
                heldout[split_id]["s_feature"].extend(
                    np.full(int(np.count_nonzero(mask)), x_s).tolist()
                )
    if not approximation_strata or not sampling_candidates:
        raise ValueError("the fixed stabilizer calibration has no positive fit strata")
    q_reference = float(np.median([row[2] for row in sampling_candidates]))
    q_threshold = max(
        SAMPLING_RATIO_ABSOLUTE_THRESHOLD,
        SAMPLING_RATIO_RELATIVE_THRESHOLD * q_reference,
    )
    sampling_strata = [
        (size, shots, actual / q_value)
        for size, shots, q_value, actual in sampling_candidates
        if q_value > q_threshold
    ]
    if not sampling_strata:
        raise ValueError("the fixed stabilizer calibration has no eligible sampling strata")
    theta_a = max(float(np.mean([row[1] for row in approximation_strata])), 0.0)
    theta_s = max(float(np.mean([row[2] for row in sampling_strata])), 0.0)
    weight_a, weight_s = math.sqrt(theta_a), math.sqrt(theta_s)

    def r_squared(actual: list[float], prediction: list[float]) -> float | None:
        values = np.asarray(actual, dtype=float)
        predictions = np.asarray(prediction, dtype=float)
        if not len(values):
            return None
        denominator = float(np.sum(np.square(values - float(np.mean(values)))))
        if denominator <= 1.0e-30:
            return None
        return 1.0 - float(np.sum(np.square(values - predictions))) / denominator

    diagnostics: dict[str, Any] = {}
    for split_id, name in ((1, "validation"), (2, "test")):
        row = heldout[split_id]
        diagnostics[name] = {
            "approximation_amplitude_R2": r_squared(
                row["a_actual"], [weight_a * value for value in row["a_feature"]]
            ),
            "sampling_standard_error_R2": r_squared(
                row["s_actual"], [weight_s * value for value in row["s_feature"]]
            ),
        }
    return math.sqrt(theta_a), math.sqrt(theta_s), {
        "probe_design": probe_audit,
        "calibration_sizes": list(CALIBRATION_SIZES),
        "calibration_rank_zero_audit_only": True,
        "calibration_shot_grid": list(CALIBRATION_SHOTS),
        "weight_approximation": weight_a,
        "weight_sampling": weight_s,
        "theta_approximation": theta_a,
        "theta_sampling": theta_s,
        "sampling_ratio_filter": {
            "q_reference_median_positive": q_reference,
            "relative_threshold": SAMPLING_RATIO_RELATIVE_THRESHOLD,
            "absolute_threshold": SAMPLING_RATIO_ABSOLUTE_THRESHOLD,
            "q_threshold": q_threshold,
        },
        "heldout_diagnostics": diagnostics,
        "target_state_used": False,
    }


def select_frontier(
    data: HamiltonianData,
    frontier: list[FrontierPoint],
    total_shots: int,
    *,
    calibrate: bool,
    calibration_seed: int = 918273,
) -> tuple[FrontierPoint, np.ndarray, float, dict[str, Any]]:
    if not frontier:
        raise ValueError("frontier is empty")
    indices = sector_indices(data.n_qubits, data.particle_number)
    if calibrate:
        weight_a, weight_s, audit = calibrate_weights(data, frontier, seed=calibration_seed)
        mode = "stabilizer-calibrated empirical proxy"
    else:
        weight_a = weight_s = 1.0
        audit = {"target_state_used": False}
        mode = "unweighted Frobenius-plus-range proxy"
    feasible = []
    for point in frontier:
        try:
            x_s, allocation = _sampling_feature(point, total_shots)
        except ValueError:
            continue
        x_a = _feature(point, indices)
        score = math.hypot(weight_a * x_a, weight_s * x_s)
        feasible.append((score, point.size, point, allocation, x_a, x_s))
    if not feasible:
        raise ValueError("no frontier point is feasible for the supplied shot budget")
    score, _, point, allocation, x_a, x_s = min(
        feasible, key=lambda item: (item[0], item[1])
    )
    return point, allocation, float(score), {
        **audit,
        "mode": mode,
        "selected_approximation_feature": float(x_a),
        "selected_sampling_feature": float(x_s),
        "selected_proxy": float(score),
        "frontier_points": len(frontier),
    }
