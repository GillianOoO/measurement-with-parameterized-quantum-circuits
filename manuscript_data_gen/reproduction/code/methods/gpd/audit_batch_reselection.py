#!/usr/bin/env python3
"""Audit every calibrated GPD case without importing the selector implementation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pauli_product"))
from release_paths import DATA_ROOT, OUTPUT_ROOT, archived_path
VERSION = "periodic-iswap-gpd-stabilizer-balanced-v2"
BASE_SEED = 20260902


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


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, message: str) -> None:
    require(
        abs(actual - expected) <= 3.0e-12 * max(1.0, abs(actual), abs(expected)),
        f"{message}: {actual} != {expected}",
    )


def equal_stratum_theta(
    source: list[dict[str, str]],
    strata: tuple[str, ...],
    numerator: str,
    denominator: str,
) -> float:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in source:
        if row["split"] != "train" or row["fit_eligible"].lower() != "true":
            continue
        grouped[tuple(row[key] for key in strata)].append(
            float(row[numerator]) / float(row[denominator])
        )
    require(bool(grouped), "No eligible train strata")
    return float(np.mean([np.mean(value) for value in grouped.values()]))


def audit_case(case_dir: Path) -> dict[str, object]:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    weights = json.loads((case_dir / "stabilizer_weights.json").read_text(encoding="utf-8"))
    frontier = rows(case_dir / "frontier_rows.csv")
    selected = rows(case_dir / "selected_results.csv")
    require(manifest["version"] == VERSION, f"{case_dir.name}: version")
    require(manifest["calibration"]["ground_state_used_for_calibration_or_selection"] is False, f"{case_dir.name}: calibration state flag")
    require(manifest["selection"]["ground_state_used_for_selection"] is False, f"{case_dir.name}: selection state flag")
    require(manifest["post_selection_evaluation"]["benchmark_state_loaded_only_after_selection"] is True, f"{case_dir.name}: state boundary")
    require(manifest["selection"]["minimum_k"] == 10, f"{case_dir.name}: minimum K")
    require(manifest["selection"]["stall_counting_begins_after_K"] == 10, f"{case_dir.name}: stall floor")
    require(manifest["calibration"]["K_fit"] == list(range(1, 11)), f"{case_dir.name}: Kcal")

    with np.load(case_dir / "stabilizer_probes.npz") as archive:
        probes = np.asarray(archive["states"])
        indices = np.asarray(archive["sector_indices"], dtype=int)
    require(array_sha256(probes) == weights["probe_design"]["probe_array_sha256"], f"{case_dir.name}: probe array hash")
    require(probes.shape[1] == int(weights["probe_design"]["probe_count"]), f"{case_dir.name}: probe count")
    require(float(np.max(np.abs(np.sum(np.abs(probes) ** 2, axis=0) - 1.0))) < 1.0e-12, f"{case_dir.name}: probe normalization")
    outside = np.setdiff1d(np.arange(probes.shape[0]), indices)
    if len(outside):
        require(float(np.max(np.sum(np.abs(probes[outside]) ** 2, axis=0))) < 1.0e-12, f"{case_dir.name}: probe sector leakage")
    expected_probe_count = 500 if manifest["case"]["benchmark"] == "h4" else 496
    require(probes.shape[1] == expected_probe_count, f"{case_dir.name}: design size")

    approximation = rows(case_dir / "calibration_approximation_components.csv")
    sampling = rows(case_dir / "calibration_sampling_components.csv")
    theta_a = equal_stratum_theta(
        approximation,
        ("K",),
        "squared_approximation_error",
        "approximation_ratio_denominator_xa_squared",
    )
    theta_s = equal_stratum_theta(
        sampling,
        ("K", "T_total_shots"),
        "actual_sampling_variance",
        "range_variance_proxy_q",
    )
    close(theta_a, float(weights["theta_approximation"]), f"{case_dir.name}: theta_a")
    close(theta_s, float(weights["theta_sampling"]), f"{case_dir.name}: theta_s")

    for row in frontier:
        require("approximation_proxy_x2" not in row, f"{case_dir.name}: legacy x2 column")
        expected_loss = math.hypot(
            float(row["weight_approximation_wa"]) * float(row["approximation_feature_xa"]),
            float(row["weight_sampling_ws"]) * float(row["sampling_feature_xs"]),
        )
        close(float(row["loss"]), expected_loss, f"{case_dir.name}: loss")
        allocation = np.asarray(json.loads(row["allocation_json"]), dtype=int)
        require(int(np.sum(allocation)) == int(row["allocated_total_shots"]), f"{case_dir.name}: allocation sum")
        require(int(row["allocated_total_shots"]) + int(row["unused_shots"]) == int(row["shots"]), f"{case_dir.name}: budget accounting")

    effective_terminal = int(manifest["selection"]["effective_terminal_k"])
    require(max(int(row["k"]) for row in frontier) == effective_terminal, f"{case_dir.name}: terminal")
    endpoint_budgets: list[int] = []
    seen_seeds: set[int] = set()
    for row in selected:
        candidates = [item for item in frontier if item["shots"] == row["shots"]]
        winner = min(candidates, key=lambda item: (float(item["loss"]), int(item["k"])))
        require(int(row["k"]) == int(winner["k"]), f"{case_dir.name}: selected K at T={row['shots']}")
        close(float(row["mse"]), float(row["bias_squared"]) + float(row["variance"]), f"{case_dir.name}: MSE")
        expected_seed = stable_seed(VERSION, case_dir.name, int(row["shots"]), BASE_SEED)
        require(int(row["empirical_seed"]) == expected_seed, f"{case_dir.name}: empirical seed")
        require(expected_seed not in seen_seeds, f"{case_dir.name}: duplicate per-T seed")
        seen_seeds.add(expected_seed)
        require(int(row["empirical_repeats"]) == 50, f"{case_dir.name}: repeats")
        if int(row["k"]) == effective_terminal:
            endpoint_budgets.append(int(row["shots"]))
    require(endpoint_budgets == manifest["selection"]["selected_at_endpoint_budgets"], f"{case_dir.name}: endpoint list")
    if bool(manifest["selection"]["right_censored"]):
        require(manifest["status"] == "right-censored", f"{case_dir.name}: censor status")
        require(effective_terminal == int(manifest["selection"]["source_terminal_k"]), f"{case_dir.name}: censor terminal")
    else:
        require(manifest["status"] == "complete", f"{case_dir.name}: complete status")

    for name, expected_hash in manifest["artifacts"].items():
        require(sha256(case_dir / name) == expected_hash, f"{case_dir.name}: artifact {name}")
    for row in rows(case_dir / "source_fragment_hash_audit.csv"):
        require(sha256(archived_path(row["parameter_path"], row["parameter_sha256"])) == row["parameter_sha256"], f"{case_dir.name}: parameter source hash")
        require(sha256(archived_path(row["diagonal_path"], row["diagonal_sha256"])) == row["diagonal_sha256"], f"{case_dir.name}: diagonal source hash")
        require(sha256(archived_path(row["probability_path"], row["probability_sha256"])) == row["probability_sha256"], f"{case_dir.name}: probability source hash")
    return {
        "slug": case_dir.name,
        "status": manifest["status"],
        "right_censored": manifest["selection"]["right_censored"],
        "effective_terminal_k": effective_terminal,
        "selected_k_by_shots": manifest["selection"]["selected_k_by_shots"],
        "w_a": weights["weight_approximation_wa"],
        "w_s": weights["weight_sampling_ws"],
    }


def main() -> None:
    case_root = DATA_ROOT / "random_sparse_dense" / "results" / "gpd_balanced"
    case_dirs = sorted(case_root.glob("*/manifest.json"))
    require(len(case_dirs) == 10, f"Expected 10 released random cases, found {len(case_dirs)}")
    results = [audit_case(path.parent) for path in case_dirs]
    censored = [row["slug"] for row in results if bool(row["right_censored"])]
    payload = {
        "status": "pass",
        "case_count": len(results),
        "right_censored_cases": censored,
        "checks": {
            "ground_state_exclusion_flags": True,
            "probe_normalization_sector_and_hash": True,
            "equal_stratum_weights_recomputed": True,
            "balanced_loss_and_argmin_recomputed": True,
            "per_case_per_T_empirical_seeds": True,
            "MSE_identity": True,
            "source_and_output_hashes": True,
        },
        "cases": results,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "gpd_batch_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("status", "case_count", "right_censored_cases", "checks")}, indent=2))


if __name__ == "__main__":
    main()
