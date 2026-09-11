#!/usr/bin/env python3
"""Non-destructive stabilizer-calibrated selection for saved iSWAP-GPD prefixes.

The saved nonlinear fragments are treated as immutable inputs.  This program
validates their hashes, reconstructs the prefix Hamiltonians, fits GPD-specific
weights on grouped stabilizer probes, and writes a new calibrated frontier:

    x_a(K)   = ||Pi (H - H_hat_K) Pi||_F,
    x_s(K,T) = sqrt(sum_i lambda_i(K)^2 / T_i),
    L(K,T)   = hypot(w_a x_a(K), w_s x_s(K,T)).

Neither the benchmark state nor its energy is read until the calibration,
frontier construction, material-stall replay, and K selection are complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pauli_product"))
from release_paths import BUNDLE_ROOT, DATA_ROOT, OUTPUT_ROOT, archived_path

WORKSPACE = BUNDLE_ROOT
OLD_ROOT = DATA_ROOT / "random_sparse_dense" / "results" / "gpd_source_prefixes"
OLD_RUNNER = HERE / "run_iswap_gpd_main_loss.py"
DEFAULT_SOURCE = OLD_ROOT / "sparse4_seed0_1"
DEFAULT_OUTPUT = OUTPUT_ROOT / "gpd_balanced" / "sparse4_seed0_1"

VERSION = "periodic-iswap-gpd-stabilizer-balanced-v2"
PROBE_SEED = 918273
CALIBRATION_K = tuple(range(1, 11))
CALIBRATION_SHOTS = (100, 200, 300, 500, 800, 1200, 2000, 3000)
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SAMPLING_RATIO_RELATIVE_THRESHOLD = 1.0e-12
SAMPLING_RATIO_ABSOLUTE_THRESHOLD = 1.0e-30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_old_runner():
    spec = importlib.util.spec_from_file_location("immutable_iswap_gpd_v1", OLD_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load immutable GPD runner: {OLD_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def hermitian(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.complex128)
    return (matrix + matrix.conj().T) / 2.0


def sector_indices(n_qubits: int, particle_number: int | None) -> np.ndarray:
    if particle_number is None:
        return np.arange(2**n_qubits, dtype=np.int64)
    return np.asarray(
        [index for index in range(2**n_qubits) if index.bit_count() == particle_number],
        dtype=np.int64,
    )


def _largest_remainder_counts(total: int) -> tuple[int, int, int]:
    raw = np.asarray(SPLIT_FRACTIONS) * int(total)
    counts = np.floor(raw).astype(int)
    order = np.lexsort((np.arange(3), -(raw - counts)))
    counts[order[: int(total - np.sum(counts))]] += 1
    return tuple(int(value) for value in counts)


def _split_map(
    order: np.ndarray, counts: tuple[int, int, int] | None = None
) -> dict[int, str]:
    train, validation, test = counts or _largest_remainder_counts(len(order))
    if train + validation + test != len(order):
        raise ValueError("Split counts do not match the group order")
    boundaries = (train, train + validation, train + validation + test)
    result: dict[int, str] = {}
    for position, slot in enumerate(order):
        split = (
            "train"
            if position < boundaries[0]
            else "validation"
            if position < boundaries[1]
            else "test"
        )
        result[int(slot)] = split
    return result


def generate_grouped_stabilizer_probes(
    indices: np.ndarray, dimension: int, seed: int = PROBE_SEED
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Generate the SRDD-style determinant/two-determinant grouped design.

    Physical sectors of dimension at least 28 use the frozen 500-state design:
    28 determinants and 118 distinct determinant pairs with four Clifford
    phases.  Smaller spaces use the complete nonduplicated catalog of the same
    two probe families.  Whole groups, rather than individual phase variants,
    are assigned to train/validation/test.
    """

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("At least two accessible basis states are required")
    if np.any(indices < 0) or np.any(indices >= dimension):
        raise ValueError("Probe-sector indices are outside the Hilbert space")
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    pair_candidates = [
        (int(a), int(b))
        for position, a in enumerate(indices)
        for b in indices[position + 1 :]
    ]

    if len(indices) >= 28 and len(pair_candidates) >= 118:
        # Match stabilizer_calibration.generate_stabilizer_probes exactly,
        # including its RNG-call order and group-level split assignment.
        determinant_slots = rng.permutation(len(indices))[:28]
        chosen_pair_slots = rng.choice(len(pair_candidates), size=118, replace=False)
        determinant_splits = _split_map(np.arange(28), (20, 4, 4))
        pair_splits = _split_map(rng.permutation(118), (70, 24, 24))
        design = "frozen-500-determinant-plus-four-phase-pair"
    else:
        determinant_slots = rng.permutation(len(indices))
        chosen_pair_slots = rng.permutation(len(pair_candidates))
        determinant_splits = _split_map(np.arange(len(determinant_slots)))
        pair_splits = _split_map(np.arange(len(chosen_pair_slots)))
        design = "complete-nonduplicated-determinant-plus-four-phase-pair"

    for group_slot, sector_slot in enumerate(determinant_slots):
        basis_index = int(indices[int(sector_slot)])
        vector = np.zeros(dimension, dtype=np.complex128)
        vector[basis_index] = 1.0
        vectors.append(vector)
        records.append(
            {
                "probe_id": len(records),
                "group_id": f"determinant_{basis_index:0{len(str(dimension - 1))}d}",
                "group_size": 1,
                "probe_type": "determinant",
                "basis_a": basis_index,
                "basis_b": "",
                "relative_phase_radians": 0.0,
                "split": determinant_splits[group_slot],
                "probe_seed": int(seed),
            }
        )

    for group_slot, pair_slot in enumerate(chosen_pair_slots):
        a, b = pair_candidates[int(pair_slot)]
        group_id = f"pair_{a:0{len(str(dimension - 1))}d}_{b:0{len(str(dimension - 1))}d}"
        for phase in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            vector = np.zeros(dimension, dtype=np.complex128)
            vector[a] = 1.0 / math.sqrt(2.0)
            vector[b] = np.exp(1j * phase) / math.sqrt(2.0)
            vectors.append(vector)
            records.append(
                {
                    "probe_id": len(records),
                    "group_id": group_id,
                    "group_size": 4,
                    "probe_type": "two_determinant_equal_amplitude",
                    "basis_a": a,
                    "basis_b": b,
                    "relative_phase_radians": phase,
                    "split": pair_splits[group_slot],
                    "probe_seed": int(seed),
                }
            )

    states = np.column_stack(vectors)
    norms = np.sum(np.square(np.abs(states)), axis=0)
    outside = np.setdiff1d(np.arange(dimension), indices)
    leakage = (
        float(np.max(np.sum(np.square(np.abs(states[outside])), axis=0)))
        if len(outside)
        else 0.0
    )
    if float(np.max(np.abs(norms - 1.0))) > 1.0e-12 or leakage > 1.0e-12:
        raise RuntimeError("Generated stabilizer probes failed sector/normalization audit")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        group_splits[str(row["group_id"])].add(str(row["split"]))
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise RuntimeError("A stabilizer group crosses data splits")
    counts = {
        split: sum(str(row["split"]) == split for row in records)
        for split in SPLIT_NAMES
    }
    metadata = {
        "design": design,
        "seed": int(seed),
        "probe_count": int(states.shape[1]),
        "group_count": len(group_splits),
        "determinant_probe_count": len(determinant_slots),
        "pair_group_count": len(chosen_pair_slots),
        "state_split_counts": counts,
        "group_disjoint_splits": True,
        "probe_array_sha256": array_sha256(states),
    }
    if design.startswith("frozen-500") and counts != {
        "train": 300,
        "validation": 100,
        "test": 100,
    }:
        raise RuntimeError(f"Frozen 500-probe split mismatch: {counts}")
    return states, records, metadata


def expectation(states: np.ndarray, operator: np.ndarray) -> np.ndarray:
    acted = operator @ states
    values = np.sum(np.conj(states) * acted, axis=0)
    if float(np.max(np.abs(values.imag))) > 1.0e-8:
        raise FloatingPointError("Hermitian expectation has a material imaginary part")
    return values.real


def expectation_and_variance(
    states: np.ndarray, operator: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    acted = operator @ states
    means = np.sum(np.conj(states) * acted, axis=0)
    seconds = np.sum(np.square(np.abs(acted)), axis=0)
    if float(np.max(np.abs(means.imag))) > 1.0e-8:
        raise FloatingPointError("Hermitian expectation has a material imaginary part")
    variances = np.maximum(seconds.real - np.square(means.real), 0.0)
    return means.real, variances


def integer_range_allocation(
    total_shots: int, ranges: Sequence[float]
) -> tuple[np.ndarray, float]:
    values = np.asarray(ranges, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Allocation ranges must be finite and nonnegative")
    active_indices = np.flatnonzero(values > 0.0)
    if int(total_shots) < len(active_indices):
        raise ValueError(
            f"T={total_shots} is smaller than {len(active_indices)} active settings"
        )
    shots = np.zeros(len(values), dtype=np.int64)
    if not len(active_indices):
        return shots, 0.0
    shots[active_indices] = 1
    remaining = int(total_shots) - len(active_indices)
    if remaining:
        raw = remaining * values[active_indices] / float(np.sum(values[active_indices]))
        extra = np.floor(raw).astype(np.int64)
        shots[active_indices] += extra
        remainder = remaining - int(np.sum(extra))
        order = np.lexsort((active_indices, -(raw - extra)))
        shots[active_indices[order[:remainder]]] += 1
    if int(np.sum(shots)) != int(total_shots):
        raise AssertionError("Integer allocation failed exact shot conservation")
    q_value = float(
        np.sum(np.square(values[active_indices]) / shots[active_indices])
    )
    return shots, math.sqrt(max(q_value, 0.0))


def equal_stratum_ratio_fit(
    rows: Iterable[dict[str, Any]],
    strata: tuple[str, ...],
    numerator: str,
    denominator: str,
) -> tuple[float, float, list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        denominator_value = float(row[denominator])
        if denominator_value <= 0.0:
            raise ValueError("Ratio fit received a nonpositive denominator")
        key = tuple(row[field] for field in strata)
        grouped[key].append(float(row[numerator]) / denominator_value)
    if not grouped:
        raise RuntimeError("No eligible calibration strata")
    stratum_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ratios = np.asarray(grouped[key], dtype=float)
        theta = max(float(np.mean(ratios)), 0.0)
        record = {field: key[index] for index, field in enumerate(strata)}
        record.update(
            {
                "rows": len(ratios),
                "theta_ratio_mean": theta,
                "ratio_minimum": float(np.min(ratios)),
                "ratio_median": float(np.median(ratios)),
                "ratio_maximum": float(np.max(ratios)),
            }
        )
        stratum_rows.append(record)
    theta_global = max(
        float(np.mean([float(row["theta_ratio_mean"]) for row in stratum_rows])),
        0.0,
    )
    return theta_global, math.sqrt(theta_global), stratum_rows


def regression_metrics(target: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    residual = target_array - prediction_array
    ss_res = float(np.dot(residual, residual))
    centered = target_array - float(np.mean(target_array))
    ss_total = float(np.dot(centered, centered))
    return {
        "rows": len(target_array),
        "RMSE": math.sqrt(ss_res / max(len(target_array), 1)),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": 1.0 - ss_res / ss_total if ss_total > 1.0e-30 else None,
    }


def load_hamiltonian(case_manifest: dict[str, Any]) -> np.ndarray:
    input_path = archived_path(case_manifest["input_path"])
    if input_path.suffix.lower() == ".npz":
        with np.load(input_path) as archive:
            matrix = np.asarray(archive["H"], dtype=np.complex128)
    else:
        matrix = np.asarray(np.load(input_path), dtype=np.complex128)
    matrix = hermitian(matrix)
    if array_sha256(matrix) != case_manifest["target_hash"]:
        raise RuntimeError("Source target hash does not match the loaded Hamiltonian")
    return matrix


def load_benchmark_state(case_manifest: dict[str, Any]) -> np.ndarray:
    state_path = archived_path(case_manifest["state_path"])
    if state_path.suffix.lower() == ".npz":
        with np.load(state_path) as archive:
            saved = np.asarray(archive["state"], dtype=np.complex128)
    else:
        saved = np.asarray(np.load(state_path), dtype=np.complex128)
    if saved.ndim == 2:
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian(saved))
        state = eigenvectors[:, int(np.argmax(eigenvalues))]
    elif saved.ndim == 1:
        state = saved.copy()
    else:
        raise ValueError(f"Unsupported benchmark-state shape: {saved.shape}")
    state /= float(np.linalg.norm(state))
    pivot = int(np.argmax(np.abs(state)))
    state *= np.exp(-1j * np.angle(state[pivot]))
    if array_sha256(state) != case_manifest["state_hash"]:
        raise RuntimeError("Source state hash does not match the loaded benchmark state")
    return state


def validate_source(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Source GPD frontier is not marked complete")
    algorithm = manifest.get("algorithm", {})
    if algorithm.get("source_fit") != "sequential full-diagonal GPD":
        raise RuntimeError("Source is not the expected sequential full-diagonal GPD")
    fragments = list(manifest.get("fragments", []))
    if [int(row["k"]) for row in fragments] != list(range(1, len(fragments) + 1)):
        raise RuntimeError("Source fragments are not a contiguous K=1,... sequence")
    if len(fragments) < max(CALIBRATION_K):
        raise RuntimeError("At least K=0,...,10 prefixes are required for calibration")
    for row in fragments:
        # Probability arrays are benchmark-state-derived and are deliberately
        # not touched until K selection has completed.
        for kind in ("parameter", "diagonal"):
            path = source / str(row[f"{kind}_file"])
            if not path.exists() or sha256(path) != row[f"{kind}_sha256"]:
                raise RuntimeError(f"Source {kind} hash mismatch at K={row['k']}")
    return manifest, fragments


def reconstruct_fragments(
    source: Path,
    source_manifest: dict[str, Any],
    fragment_records: Sequence[dict[str, Any]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    old = load_old_runner()
    case = source_manifest["case"]
    n_qubits = int(case["n_qubits"])
    internal_layers = int(source_manifest["algorithm"]["internal_layers"])
    use_x64 = bool(source_manifest["algorithm"].get("x64", False))
    jax.config.update("jax_enable_x64", use_x64)
    real_dtype = jnp.float64 if use_x64 else jnp.float32
    complex_dtype = jnp.complex128 if use_x64 else jnp.complex64
    actions = old.canonical.CircuitActions(n_qubits, internal_layers)
    inverse = jax.jit(actions.conjugate_inverse)
    matrices: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    for record in fragment_records:
        k = int(record["k"])
        parameter_path = source / str(record["parameter_file"])
        diagonal_path = source / str(record["diagonal_file"])
        parameters = jnp.asarray(np.load(parameter_path), dtype=real_dtype)
        diagonal = np.asarray(np.load(diagonal_path), dtype=np.float64)
        fragment = inverse(
            jnp.diag(jnp.asarray(diagonal, dtype=complex_dtype)), parameters
        )
        fragment.block_until_ready()
        matrix = hermitian(np.asarray(fragment, dtype=np.complex128))
        matrices.append(matrix)
        audit_rows.append(
            {
                "K": k,
                "parameter_path": str(parameter_path.resolve()),
                "parameter_sha256": sha256(parameter_path),
                "diagonal_path": str(diagonal_path.resolve()),
                "diagonal_sha256": sha256(diagonal_path),
                "probability_path": str((source / str(record["probability_file"])).resolve()),
                "probability_sha256": str(record["probability_sha256"]),
                "reconstructed_fragment_array_sha256": array_sha256(matrix),
                "reconstructed_fragment_frobenius": float(np.linalg.norm(matrix, "fro")),
                "allocation_centered_half_range": float(record["centered_half_range"]),
                "allocation_range_branch": str(record["allocation_range_branch"]),
            }
        )
    return matrices, audit_rows


def prefix_data(
    hamiltonian: np.ndarray,
    fragments: Sequence[np.ndarray],
    indices: np.ndarray,
) -> list[dict[str, Any]]:
    dimension = len(hamiltonian)
    exact_constant = float(np.trace(hamiltonian).real / dimension)
    approximation = exact_constant * np.eye(dimension, dtype=np.complex128)
    rows: list[dict[str, Any]] = []
    for k in range(len(fragments) + 1):
        if k:
            approximation = hermitian(approximation + fragments[k - 1])
        residual = hermitian(hamiltonian - approximation)
        projected = residual[np.ix_(indices, indices)]
        rows.append(
            {
                "K": k,
                "approximation": approximation.copy(),
                "residual": residual,
                "approximation_feature_xa": float(np.linalg.norm(projected, "fro")),
                "full_residual_frobenius": float(np.linalg.norm(residual, "fro")),
                "residual_array_sha256": array_sha256(residual),
            }
        )
    return rows


def calibrate_weights(
    hamiltonian: np.ndarray,
    prefix_rows: Sequence[dict[str, Any]],
    fragment_matrices: Sequence[np.ndarray],
    ranges: Sequence[float],
    probes: np.ndarray,
    probe_records: Sequence[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if tuple(int(row["K"]) for row in prefix_rows[:11]) != tuple(range(11)):
        raise ValueError("Calibration requires contiguous K=0,...,10 prefix data")
    split = np.asarray([str(row["split"]) for row in probe_records])
    exact_energies = expectation(probes, hamiltonian)
    fragment_variances = [
        expectation_and_variance(probes, matrix)[1]
        for matrix in fragment_matrices[: max(CALIBRATION_K)]
    ]
    approximation_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []

    for k in range(max(CALIBRATION_K) + 1):
        row = prefix_rows[k]
        approximate_energies = expectation(probes, row["approximation"])
        biases = approximate_energies - exact_energies
        x_a = float(row["approximation_feature_xa"])
        for probe_id, bias in enumerate(biases):
            approximation_rows.append(
                {
                    "probe_id": probe_id,
                    "group_id": str(probe_records[probe_id]["group_id"]),
                    "split": str(split[probe_id]),
                    "K": k,
                    "fit_eligible": k in CALIBRATION_K and x_a > 0.0,
                    "approximation_feature_xa": x_a,
                    "approximation_ratio_denominator_xa_squared": x_a * x_a,
                    "exact_probe_energy": float(exact_energies[probe_id]),
                    "approximate_probe_energy": float(approximate_energies[probe_id]),
                    "signed_approximation_error": float(bias),
                    "absolute_approximation_error": abs(float(bias)),
                    "squared_approximation_error": float(bias * bias),
                }
            )
        for total_shots in CALIBRATION_SHOTS:
            shots, x_s = integer_range_allocation(total_shots, ranges[:k])
            q_value = x_s * x_s
            allocation_key = f"calibration_K{k}_T{total_shots}"
            for setting, (half_range, count) in enumerate(
                zip(ranges[:k], shots), start=1
            ):
                allocation_rows.append(
                    {
                        "allocation_key": allocation_key,
                        "K": k,
                        "T_total_shots": total_shots,
                        "setting_index": setting,
                        "centered_half_range": float(half_range),
                        "allocated_shots": int(count),
                    }
                )
            if k:
                active = shots > 0
                actual_variance = np.sum(
                    np.asarray(fragment_variances[:k])[active]
                    / shots[active, None],
                    axis=0,
                )
            else:
                actual_variance = np.zeros(probes.shape[1], dtype=float)
            for probe_id, variance in enumerate(actual_variance):
                sampling_rows.append(
                    {
                        "probe_id": probe_id,
                        "group_id": str(probe_records[probe_id]["group_id"]),
                        "split": str(split[probe_id]),
                        "K": k,
                        "T_total_shots": total_shots,
                        "fit_eligible_K": k in CALIBRATION_K,
                        "allocation_key": allocation_key,
                        "allocated_total_shots": int(np.sum(shots)),
                        "active_settings": int(np.count_nonzero(shots)),
                        "sampling_feature_xs": x_s,
                        "range_variance_proxy_q": q_value,
                        "actual_sampling_variance": float(variance),
                        "actual_sampling_standard_error": math.sqrt(max(float(variance), 0.0)),
                    }
                )

    train_approximation = [
        row
        for row in approximation_rows
        if row["split"] == "train" and bool(row["fit_eligible"])
    ]
    theta_a, weight_a, approximation_strata = equal_stratum_ratio_fit(
        train_approximation,
        ("K",),
        "squared_approximation_error",
        "approximation_ratio_denominator_xa_squared",
    )
    positive_q = np.asarray(
        [
            float(row["range_variance_proxy_q"])
            for row in sampling_rows
            if bool(row["fit_eligible_K"])
            and float(row["range_variance_proxy_q"]) > 0.0
        ],
        dtype=float,
    )
    if not len(positive_q):
        raise RuntimeError("No positive sampling proxy in calibration grid")
    q_reference = float(np.median(positive_q))
    q_threshold = max(
        SAMPLING_RATIO_ABSOLUTE_THRESHOLD,
        SAMPLING_RATIO_RELATIVE_THRESHOLD * q_reference,
    )
    for row in sampling_rows:
        q_value = float(row["range_variance_proxy_q"])
        eligible = bool(row["fit_eligible_K"]) and q_value > q_threshold
        row["sampling_ratio_q_reference"] = q_reference
        row["sampling_ratio_q_threshold"] = q_threshold
        row["fit_eligible"] = eligible
        row["sampling_variance_ratio"] = (
            float(row["actual_sampling_variance"]) / q_value if eligible else "N/A"
        )
    train_sampling = [
        row
        for row in sampling_rows
        if row["split"] == "train" and bool(row["fit_eligible"])
    ]
    theta_s, weight_s, sampling_strata = equal_stratum_ratio_fit(
        train_sampling,
        ("K", "T_total_shots"),
        "actual_sampling_variance",
        "range_variance_proxy_q",
    )

    for row in approximation_rows:
        row["predicted_approximation_component"] = weight_a * float(
            row["approximation_feature_xa"]
        )
        row["predicted_approximation_squared"] = theta_a * float(
            row["approximation_feature_xa"]
        ) ** 2
    for row in sampling_rows:
        row["predicted_sampling_component"] = weight_s * float(
            row["sampling_feature_xs"]
        )
        row["predicted_sampling_variance"] = theta_s * float(
            row["range_variance_proxy_q"]
        )

    diagnostics: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        a_rows = [
            row
            for row in approximation_rows
            if row["split"] == split_name and bool(row["fit_eligible"])
        ]
        s_rows = [
            row
            for row in sampling_rows
            if row["split"] == split_name and bool(row["fit_eligible"])
        ]
        diagnostics[split_name] = {
            "approximation_amplitude": regression_metrics(
                [float(row["absolute_approximation_error"]) for row in a_rows],
                [float(row["predicted_approximation_component"]) for row in a_rows],
            ),
            "approximation_squared": regression_metrics(
                [float(row["squared_approximation_error"]) for row in a_rows],
                [float(row["predicted_approximation_squared"]) for row in a_rows],
            ),
            "sampling_standard_error": regression_metrics(
                [float(row["actual_sampling_standard_error"]) for row in s_rows],
                [float(row["predicted_sampling_component"]) for row in s_rows],
            ),
            "sampling_variance": regression_metrics(
                [float(row["actual_sampling_variance"]) for row in s_rows],
                [float(row["predicted_sampling_variance"]) for row in s_rows],
            ),
        }
    weights = {
        "weight_approximation_wa": weight_a,
        "weight_sampling_ws": weight_s,
        "theta_approximation": theta_a,
        "theta_sampling": theta_s,
        "calibration_K_fit": list(CALIBRATION_K),
        "calibration_K_audit_only": [0],
        "calibration_shot_grid": list(CALIBRATION_SHOTS),
        "fit_definition": {
            "approximation": (
                "For each K=1,...,10, average probe bias squared / x_a,K squared "
                "over train probes; average the ten K-stratum slopes equally."
            ),
            "sampling": (
                "For each eligible (K,T_cal), average actual sampling variance / "
                "x_s,K,T squared over train probes; average K,T strata equally."
            ),
            "weight_mapping": "w_a=sqrt(theta_a), w_s=sqrt(theta_s)",
            "loss": "hypot(w_a*x_a, w_s*x_s)",
        },
        "sampling_ratio_filter": {
            "q_reference_median_positive": q_reference,
            "relative_threshold": SAMPLING_RATIO_RELATIVE_THRESHOLD,
            "absolute_threshold": SAMPLING_RATIO_ABSOLUTE_THRESHOLD,
            "q_threshold": q_threshold,
        },
        "equal_weight_strata": {
            "approximation_by_K": approximation_strata,
            "sampling_by_K_T": sampling_strata,
        },
        "four_track_diagnostics": diagnostics,
        "ground_state_used_for_fit": False,
    }
    return weights, approximation_rows, sampling_rows, allocation_rows


def build_frontier(
    source_manifest: dict[str, Any],
    prefix_rows: Sequence[dict[str, Any]],
    ranges: Sequence[float],
    weights: dict[str, Any],
) -> list[dict[str, Any]]:
    case = source_manifest["case"]
    weight_a = float(weights["weight_approximation_wa"])
    weight_s = float(weights["weight_sampling_ws"])
    rows: list[dict[str, Any]] = []
    for prefix in prefix_rows:
        k = int(prefix["K"])
        for total_shots in sorted(int(value) for value in case["shots"]):
            try:
                shots, x_s = integer_range_allocation(total_shots, ranges[:k])
            except ValueError:
                continue
            x_a = float(prefix["approximation_feature_xa"])
            approximation_component = weight_a * x_a
            sampling_component = weight_s * x_s
            rows.append(
                {
                    "benchmark": str(case["benchmark"]),
                    "slug": str(case["slug"]),
                    "k": k,
                    "shots": total_shots,
                    "active_settings": int(np.count_nonzero(shots)),
                    "allocated_total_shots": int(np.sum(shots)),
                    "unused_shots": total_shots - int(np.sum(shots)),
                    "approximation_feature_xa": x_a,
                    "sampling_feature_xs": x_s,
                    "weight_approximation_wa": weight_a,
                    "weight_sampling_ws": weight_s,
                    "calibrated_approximation_component": approximation_component,
                    "calibrated_sampling_component": sampling_component,
                    "loss": math.hypot(approximation_component, sampling_component),
                    "predicted_total_rmse": math.hypot(
                        approximation_component, sampling_component
                    ),
                    "full_residual_frobenius": float(prefix["full_residual_frobenius"]),
                    "residual_array_sha256": str(prefix["residual_array_sha256"]),
                    "allocation_json": json.dumps(shots.tolist()),
                    "centered_half_ranges_json": json.dumps(
                        [float(value) for value in ranges[:k]]
                    ),
                    "selection_uses_benchmark_state": False,
                }
            )
    return rows


def replay_material_stall(
    frontier: Sequence[dict[str, Any]],
    source_terminal_k: int,
    relative_improvement: float,
    stall_patience: int,
    minimum_k: int,
) -> dict[str, Any]:
    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in frontier:
        by_k[int(row["k"])].append(row)
    anchors: dict[int, float] = {}
    events: list[dict[str, Any]] = []
    stalled = 0
    stop_k: int | None = None
    for k in range(source_terminal_k + 1):
        materially_improved = False
        updates: dict[str, float] = {}
        for row in by_k.get(k, []):
            total = int(row["shots"])
            score = float(row["loss"])
            previous = anchors.get(total, math.inf)
            if score < previous * (1.0 - relative_improvement):
                anchors[total] = score
                updates[str(total)] = score
                materially_improved = True
        if k <= max(CALIBRATION_K):
            stalled = 0
            counted = False
        else:
            stalled = 0 if materially_improved else stalled + 1
            counted = True
        events.append(
            {
                "K": k,
                "materially_improved_any_budget": materially_improved,
                "anchor_updates": updates,
                "stall_counted": counted,
                "consecutive_nonmaterial_boundaries": stalled,
            }
        )
        if k >= max(minimum_k, max(CALIBRATION_K) + 1) and stalled >= stall_patience:
            stop_k = k
            break
    effective_terminal = source_terminal_k if stop_k is None else stop_k
    return {
        "effective_terminal_k": effective_terminal,
        "source_terminal_k": source_terminal_k,
        "stop_observed": stop_k is not None,
        "right_censored": stop_k is None,
        "stop_reason": (
            f"no global-best calibrated-loss improvement above {relative_improvement:g} "
            f"for {stall_patience} consecutive post-calibration K boundaries"
            if stop_k is not None
            else "source prefix bank ended before the calibrated material-stall rule fired"
        ),
        "material_anchor_after_replay": {str(key): value for key, value in anchors.items()},
        "events": events,
    }


def select_rows(
    frontier: Sequence[dict[str, Any]], effective_terminal_k: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    shot_values = sorted({int(row["shots"]) for row in frontier})
    for total_shots in shot_values:
        candidates = [
            row
            for row in frontier
            if int(row["shots"]) == total_shots
            and int(row["k"]) <= effective_terminal_k
        ]
        if not candidates:
            continue
        winner = min(candidates, key=lambda row: (float(row["loss"]), int(row["k"])))
        selected.append(dict(winner))
    return selected


def add_post_selection_diagnostics(
    selected: list[dict[str, Any]],
    source: Path,
    source_manifest: dict[str, Any],
    prefix_rows: Sequence[dict[str, Any]],
    fragment_records: Sequence[dict[str, Any]],
    repeats: int,
    empirical_base_seed: int,
    output: Path,
) -> None:
    # This is deliberately the first point at which the benchmark state is read.
    state = load_benchmark_state(source_manifest["case"])
    for record in fragment_records:
        probability_path = source / str(record["probability_file"])
        if (
            not probability_path.exists()
            or sha256(probability_path) != record["probability_sha256"]
        ):
            raise RuntimeError(
                f"Source probability hash mismatch at K={record['k']}"
            )
    target = float(np.real(np.vdot(state, prefix_rows[0]["residual"] @ state)))
    exact_constant = float(
        np.trace(prefix_rows[0]["approximation"]).real / len(prefix_rows[0]["approximation"])
    )
    for row in selected:
        k = int(row["k"])
        row_seed = stable_seed(
            VERSION,
            source_manifest["case"]["slug"],
            int(row["shots"]),
            int(empirical_base_seed),
        )
        rng = np.random.default_rng(row_seed)
        shots = np.asarray(json.loads(str(row["allocation_json"])), dtype=np.int64)
        approximation = prefix_rows[k]["approximation"]
        approximate_expectation = float(np.real(np.vdot(state, approximation @ state)))
        exact_expectation = float(np.real(np.vdot(state, (approximation + prefix_rows[k]["residual"]) @ state)))
        bias = approximate_expectation - exact_expectation
        variance = 0.0
        for record, count in zip(fragment_records[:k], shots):
            if int(count) > 0:
                variance += float(record["state_single_shot_variance"]) / int(count)
        mse = bias * bias + variance
        row.update(
            {
                "approximate_expectation": approximate_expectation,
                "target_expectation": exact_expectation,
                "bias": bias,
                "bias_squared": bias * bias,
                "variance": variance,
                "mse": mse,
                "analytic_rmse": math.sqrt(max(mse, 0.0)),
                "benchmark_state_used_only_after_selection": True,
            }
        )
        estimates = np.full(int(repeats), exact_constant, dtype=float)
        for record, count in zip(fragment_records[:k], shots):
            if int(count) <= 0:
                estimates += float(record["allocation_midpoint"])
                continue
            diagonal = np.asarray(
                np.load(source / str(record["diagonal_file"])), dtype=float
            )
            probabilities = np.asarray(
                np.load(source / str(record["probability_file"])), dtype=float
            )
            probabilities = np.maximum(probabilities, 0.0)
            probabilities /= float(np.sum(probabilities))
            counts = rng.multinomial(int(count), probabilities, size=int(repeats))
            estimates += counts @ diagonal / int(count)
        errors = estimates - exact_expectation
        row["empirical_repeats"] = int(repeats)
        row["empirical_seed"] = row_seed
        row["empirical_rmse"] = float(np.sqrt(np.mean(np.square(errors))))
        write_csv(
            output / f"empirical_replicates_T{int(row['shots'])}.csv",
            [
                {
                    "repeat": repeat,
                    "estimate": float(estimate),
                    "error": float(error),
                    "selected_K": k,
                    "T_total_shots": int(row["shots"]),
                    "empirical_seed": row_seed,
                }
                for repeat, (estimate, error) in enumerate(zip(estimates, errors))
            ],
        )
    if abs(target - (float(source_manifest["case"]["target_expectation"]) - exact_constant)) > 5.0e-7:
        raise RuntimeError("Post-selection benchmark expectation audit failed")


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Calibrated output must not overwrite or nest inside the source")
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"Output is nonempty; pass --force to refresh v2 files: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_manifest, fragment_records = validate_source(source)
    hamiltonian = load_hamiltonian(source_manifest["case"])
    dimension = len(hamiltonian)
    n_qubits = int(round(math.log2(dimension)))
    indices = sector_indices(n_qubits, source_manifest["case"].get("particle_number"))
    fragment_matrices, fragment_audit = reconstruct_fragments(
        source, source_manifest, fragment_records
    )
    prefixes = prefix_data(hamiltonian, fragment_matrices, indices)
    ranges = [float(row["centered_half_range"]) for row in fragment_records]
    probes, probe_records, probe_metadata = generate_grouped_stabilizer_probes(
        indices, dimension, int(args.probe_seed)
    )
    weights, approximation_rows, sampling_rows, calibration_allocations = calibrate_weights(
        hamiltonian,
        prefixes,
        fragment_matrices,
        ranges,
        probes,
        probe_records,
    )

    probe_path = output / "stabilizer_probes.npz"
    probe_manifest_path = output / "stabilizer_probe_manifest.csv"
    weights_path = output / "stabilizer_weights.json"
    write_npz(probe_path, states=probes, sector_indices=indices)
    write_csv(probe_manifest_path, probe_records)
    weights.update(
        {
            "version": VERSION,
            "probe_design": probe_metadata,
            "probe_npz_sha256": sha256(probe_path),
            "probe_manifest_sha256": sha256(probe_manifest_path),
            "source_target_hash": source_manifest["case"]["target_hash"],
            "source_case_slug": source_manifest["case"]["slug"],
            "ground_state_used_for_calibration_or_selection": False,
        }
    )
    write_json(weights_path, weights)
    write_csv(output / "calibration_approximation_components.csv", approximation_rows)
    write_csv(output / "calibration_sampling_components.csv", sampling_rows)
    write_csv(output / "calibration_shot_allocations.csv", calibration_allocations)
    write_csv(output / "source_fragment_hash_audit.csv", fragment_audit)

    frontier = build_frontier(source_manifest, prefixes, ranges, weights)
    source_terminal = int(source_manifest["frontier"]["terminal_k"])
    replay = replay_material_stall(
        frontier,
        source_terminal,
        float(args.relative_improvement),
        int(args.stall_patience),
        int(args.minimum_k),
    )
    effective_terminal = int(replay["effective_terminal_k"])
    retained_frontier = [row for row in frontier if int(row["k"]) <= effective_terminal]
    selected = select_rows(retained_frontier, effective_terminal)
    for row in selected:
        row["frontier_effective_terminal_k"] = effective_terminal
        row["source_terminal_k"] = source_terminal
        row["frontier_right_censored"] = bool(replay["right_censored"])
        row["selected_at_effective_terminal_k"] = int(row["k"]) == effective_terminal
        row["selection_is_provisional"] = bool(replay["right_censored"])
        row["source_case_dir"] = str(source)

    # The selector is now frozen.  Only post-selection diagnostics below may
    # touch the benchmark state or its saved measurement probabilities.
    add_post_selection_diagnostics(
        selected,
        source,
        source_manifest,
        prefixes,
        fragment_records,
        int(args.empirical_repeats),
        int(args.empirical_base_seed),
        output,
    )
    write_csv(output / "frontier_rows.csv", retained_frontier)
    write_csv(output / "selected_results.csv", selected)
    write_json(output / "material_stall_replay.json", replay)

    endpoint_budgets = [
        int(row["shots"])
        for row in selected
        if bool(row["selected_at_effective_terminal_k"])
    ]
    manifest = {
        "version": VERSION,
        "status": "complete" if not replay["right_censored"] else "right-censored",
        "completed_utc": utc_now(),
        "scope": "non-destructive stabilizer-calibrated reselection of immutable GPD prefixes",
        "source": {
            "case_dir": str(source),
            "manifest_path": str((source / "manifest.json").resolve()),
            "manifest_sha256": sha256(source / "manifest.json"),
            "input_path": source_manifest["case"]["input_path"],
            "input_sha256": sha256(archived_path(source_manifest["case"]["input_path"])),
            "state_path": source_manifest["case"]["state_path"],
            "state_file_sha256_post_selection_audit_only": sha256(
                archived_path(source_manifest["case"]["state_path"])
            ),
            "old_runner_path": str(OLD_RUNNER.resolve()),
            "old_runner_sha256": sha256(OLD_RUNNER),
            "source_profile": source_manifest.get("profile"),
            "target_hash": source_manifest["case"]["target_hash"],
            "state_hash_post_selection_audit_only": source_manifest["case"]["state_hash"],
            "source_terminal_k": source_terminal,
            "fragment_count": len(fragment_records),
        },
        "case": {
            **source_manifest["case"],
            "active_sector_dimension": len(indices),
        },
        "fragment_optimization": {
            "reused_without_refitting": True,
            "objective": "offdiagonal Frobenius norm squared",
            "source_fit": source_manifest["algorithm"]["source_fit"],
            "ansatz": source_manifest["algorithm"]["ansatz"],
            "coefficient_update": source_manifest["algorithm"]["coefficient_update"],
        },
        "calibration": {
            "GPD_specific": True,
            "ground_state_used_for_calibration_or_selection": False,
            "probe_seed": int(args.probe_seed),
            "probe_design": probe_metadata,
            "K_fit": list(CALIBRATION_K),
            "K0_audit_only": True,
            "T_calibration": list(CALIBRATION_SHOTS),
            "weights": {
                "w_a": weights["weight_approximation_wa"],
                "w_s": weights["weight_sampling_ws"],
                "theta_a": weights["theta_approximation"],
                "theta_s": weights["theta_sampling"],
            },
            "weights_path": str(weights_path.resolve()),
            "weights_sha256": sha256(weights_path),
            "probe_path": str(probe_path.resolve()),
            "probe_sha256": sha256(probe_path),
            "probe_manifest_path": str(probe_manifest_path.resolve()),
            "probe_manifest_sha256": sha256(probe_manifest_path),
        },
        "selection": {
            "loss": "hypot(w_a*x_a, w_s*x_s)",
            "x_a": "||Pi(H-H_hat_K)Pi||_F",
            "x_s": "sqrt(sum_i lambda_i,K^2/T_i) after exact integer allocation",
            "shot_allocation": "minimum-one proportional centered-half-range largest remainder",
            "relative_material_improvement": float(args.relative_improvement),
            "stall_patience": int(args.stall_patience),
            "minimum_k": int(args.minimum_k),
            "stall_counting_begins_after_K": max(CALIBRATION_K),
            "effective_terminal_k": effective_terminal,
            "source_terminal_k": source_terminal,
            "right_censored": bool(replay["right_censored"]),
            "stop_reason": replay["stop_reason"],
            "selected_k_by_shots": {
                str(row["shots"]): int(row["k"]) for row in selected
            },
            "selected_at_endpoint_budgets": endpoint_budgets,
            "ground_state_used_for_selection": False,
        },
        "post_selection_evaluation": {
            "benchmark_state_loaded_only_after_selection": True,
            "empirical_repeats": int(args.empirical_repeats),
            "empirical_base_seed": int(args.empirical_base_seed),
            "seed_rule": "stable_seed(version, slug, T, empirical_base_seed)",
        },
        "artifacts": {
            path.name: sha256(path)
            for path in (
                weights_path,
                probe_path,
                probe_manifest_path,
                output / "calibration_approximation_components.csv",
                output / "calibration_sampling_components.csv",
                output / "calibration_shot_allocations.csv",
                output / "source_fragment_hash_audit.csv",
                output / "frontier_rows.csv",
                output / "selected_results.csv",
                output / "material_stall_replay.json",
            )
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "elapsed_seconds": time.perf_counter() - started,
            "numpy": np.__version__,
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": manifest["status"],
                "w_a": weights["weight_approximation_wa"],
                "w_s": weights["weight_sampling_ws"],
                "effective_terminal_k": effective_terminal,
                "selected_k_by_shots": manifest["selection"]["selected_k_by_shots"],
                "elapsed_seconds": manifest["runner"]["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-seed", type=int, default=PROBE_SEED)
    parser.add_argument("--relative-improvement", type=float, default=2.0e-3)
    parser.add_argument("--stall-patience", type=int, default=5)
    parser.add_argument("--minimum-k", type=int, default=10)
    parser.add_argument("--empirical-repeats", type=int, default=50)
    parser.add_argument(
        "--empirical-base-seed",
        "--empirical-seed",
        dest="empirical_base_seed",
        type=int,
        default=20260902,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.empirical_repeats < 2:
        raise ValueError("--empirical-repeats must be at least two")
    if args.stall_patience < 1 or args.minimum_k < max(CALIBRATION_K):
        raise ValueError("Invalid material-stall parameters")
    if not 0.0 <= args.relative_improvement < 1.0:
        raise ValueError("--relative-improvement must lie in [0,1)")
    run(args)


if __name__ == "__main__":
    main()
