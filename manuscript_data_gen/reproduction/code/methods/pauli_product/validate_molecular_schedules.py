#!/usr/bin/env python3
"""Validate corrected Huang-2021 schedules on BeH2 and N2 Hamiltonians."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from release_paths import DATA_ROOT
ROOT = DATA_ROOT
SELECTED_ROOT = DATA_ROOT
CORE_DIR = HERE
CORE_PATH = CORE_DIR / "derand_huang2021.py"
sys.path.insert(0, str(CORE_DIR))

from derand_huang2021 import (  # noqa: E402
    appendix_c_schedule,
    github_target_schedule,
    schedule_hit_counts,
)


PAULI_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
CASES = {
    "BeH2": "BeH2/inputs",
    "N2": "N2/inputs",
}
OFFICIAL_CPP_TOY_GOLDEN = np.asarray(
    [
        [1, 2, 3], [2, 3, 1], [3, 1, 2], [2, 3, 3],
        [3, 1, 1], [1, 2, 3], [2, 3, 1], [3, 1, 2],
        [1, 2, 2], [2, 3, 3], [3, 1, 1], [2, 3, 1],
        [3, 1, 1], [2, 1, 3], [1, 2, 3], [3, 1, 2],
    ],
    dtype=np.int8,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(name: str) -> tuple[np.ndarray, np.ndarray, Path]:
    path = SELECTED_ROOT / CASES[name] / "hamiltonian_pauli_blocked_spin.csv"
    # Match the production runner's pandas parser exactly.  Symmetry-related
    # coefficients can differ by one ulp under Python's scalar float parser,
    # changing stable sort/tie paths in both the cover and Derand schedule.
    frame = pd.read_csv(path)
    all_observables = np.asarray(
        [
            [PAULI_CODE[item] for item in label]
            for label in frame["full_label_site0_to_siteNminus1"].astype(str)
        ],
        dtype=np.int8,
    )
    all_coefficients = frame["coefficient_real_hartree"].to_numpy(dtype=float)
    keep = np.any(all_observables != 0, axis=1) & (np.abs(all_coefficients) > 1.0e-14)
    return (
        all_observables[keep],
        all_coefficients[keep],
        path,
    )


def official_cpp_toy_check() -> dict[str, object]:
    path = HERE / "official_cpp_toy_observables.txt"
    rows: list[list[int]] = []
    weights: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if line_number == 0:
            parts = parts[1:]
        locality = int(parts[0])
        row = [0, 0, 0]
        for index in range(locality):
            row[int(parts[2 + 2 * index])] = PAULI_CODE[parts[1 + 2 * index]]
        rows.append(row)
        weights.append(float(parts[1 + 2 * locality]))
    actual = github_target_schedule(
        np.asarray(rows, dtype=np.int8), np.asarray(weights), 3
    )
    return {
        "official_cpp_rows": int(len(OFFICIAL_CPP_TOY_GOLDEN)),
        "exact_setting_by_setting_match": bool(
            np.array_equal(actual, OFFICIAL_CPP_TOY_GOLDEN)
        ),
        "fixture_path": str(path),
    }


def legacy_schedule(
    observables: np.ndarray, coefficients: np.ndarray, shots: int
) -> np.ndarray:
    """Historical local full-sum/last-qubit-rollover implementation."""

    half_eta = 0.9 / 2.0
    nu = 1.0 - math.exp(-half_eta)
    weights = np.abs(coefficients) / np.max(np.abs(coefficients))
    sites = observables.shape[1]
    localities = np.zeros((sites + 1, len(coefficients)), dtype=np.int16)
    for site in range(sites):
        localities[site] = np.sum(observables[:, site:] != 0, axis=1)
    hits = np.zeros(len(coefficients), dtype=np.int64)
    rows = np.empty((shots, sites), dtype=np.int8)
    for shot in range(shots):
        prefix = np.ones(len(coefficients), dtype=bool)
        for site in range(sites):
            candidates: list[np.ndarray] = []
            bounds: list[float] = []
            for action in (1, 2, 3):
                compatible = prefix & (
                    (observables[:, site] == 0)
                    | (observables[:, site] == action)
                )
                candidates.append(compatible)
                if site + 1 < sites:
                    logs = np.log1p(
                        -nu * compatible / np.power(3.0, localities[site + 1])
                    )
                    logs -= half_eta * hits
                else:
                    next_hits = hits + compatible
                    logs = np.log1p(-nu / np.power(3.0, localities[0]))
                    logs -= half_eta * next_hits
                bounds.append(float(np.sum(np.exp(logs / weights))))
            selected = int(np.argmin(bounds))
            rows[shot, site] = selected + 1
            prefix = candidates[selected]
        hits += prefix
    return rows


def sorted_insertion_cover(
    observables: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    groups: list[np.ndarray] = []
    for index in np.argsort(-np.abs(coefficients), kind="stable"):
        term = observables[index]
        for setting in groups:
            if np.all((term == 0) | (setting == 0) | (term == setting)):
                fill = (setting == 0) & (term != 0)
                setting[fill] = term[fill]
                break
        else:
            groups.append(term.copy())
    result = np.asarray(groups, dtype=np.int8)
    result[result == 0] = 3
    return result


def schedule_metrics(
    observables: np.ndarray,
    schedule: np.ndarray,
    *,
    initial_hits: np.ndarray | None = None,
) -> dict[str, object]:
    hits = schedule_hit_counts(observables, schedule)
    if initial_hits is not None:
        hits = hits + np.asarray(initial_hits, dtype=np.int64)
    seen: set[tuple[int, ...]] = set()
    last_new = 0
    for index, row in enumerate(schedule, 1):
        key = tuple(map(int, row))
        if key not in seen:
            seen.add(key)
            last_new = index
    terminal = 0
    if len(schedule):
        for row in schedule[::-1]:
            if np.array_equal(row, schedule[-1]):
                terminal += 1
            else:
                break
    quantiles = np.quantile(hits, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "shots": int(len(schedule)),
        "distinct_settings": int(len(seen)),
        "last_new_setting": int(last_new),
        "terminal_repeat_length": int(terminal),
        "zero_hit_observables": int(np.count_nonzero(hits == 0)),
        "hit_quantiles_min_q25_median_q75_max": [float(value) for value in quantiles],
    }


def equal_prefix_length(left: np.ndarray, right: np.ndarray) -> int:
    count = min(len(left), len(right))
    unequal = np.flatnonzero(np.any(left[:count] != right[:count], axis=1))
    return int(unequal[0]) if unequal.size else count


def validate_case(name: str, shots: int, legacy_prefix: int) -> dict[str, object]:
    started = time.time()
    observables, coefficients, path = load_case(name)
    corrected, corrected_audit = appendix_c_schedule(
        observables, coefficients, shots, return_audit=True
    )
    corrected_short = appendix_c_schedule(observables, coefficients, legacy_prefix)
    historical_short = legacy_schedule(observables, coefficients, legacy_prefix)

    hits_before_tail = schedule_hit_counts(observables, corrected[: max(0, shots - 600)])
    hits_full = schedule_hit_counts(observables, corrected)

    cover = sorted_insertion_cover(observables, coefficients)
    cover_hits = schedule_hit_counts(observables, cover)
    seeded_raw, seeded_audit = appendix_c_schedule(
        observables,
        coefficients,
        shots,
        initial_hits=cover_hits,
        return_audit=True,
    )
    if len(cover) > shots:
        raise RuntimeError(f"{name}: cover is larger than the requested total budget")
    combined = np.concatenate([cover, seeded_raw[: shots - len(cover)]], axis=0)

    return {
        "molecule": name,
        "hamiltonian_path": str(path),
        "hamiltonian_sha256": sha256(path),
        "qubits": int(observables.shape[1]),
        "nonzero_nonidentity_observables": int(len(observables)),
        "small_prefix_settings_checked": int(legacy_prefix),
        "corrected_equals_historical_first_24": bool(
            np.array_equal(corrected_short[:24], historical_short[:24])
        ),
        "historical_equal_prefix_length": equal_prefix_length(
            corrected_short, historical_short
        ),
        "historical_comparison_role": (
            "diagnostic only: the local legacy scorer has a known final-qubit "
            "rollover and complete-sum cancellation error"
        ),
        "paper_fixed_budget": {
            **corrected_audit.as_dict(),
            "terms_gaining_hits_in_last_600_shots": int(
                np.count_nonzero(hits_full > hits_before_tail)
            ),
            "prefix_replay_exact": bool(
                np.array_equal(corrected_short, corrected[:legacy_prefix])
            ),
        },
        "qwc_seeded_production": {
            "cover_settings": int(len(cover)),
            "cover_minimum_hits": int(np.min(cover_hits)),
            "raw_generator": seeded_audit.as_dict(),
            "combined_T_metrics": schedule_metrics(observables, combined),
        },
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=3000)
    parser.add_argument("--legacy-prefix", type=int, default=64)
    parser.add_argument("--output", type=Path, default=HERE / "validation_report.json")
    args = parser.parse_args()

    payload = {
        "status": "PASS",
        "algorithm": "Huang-2021 Appendix C C8/C11 stable affected-term delta",
        "official_repository": "https://github.com/hsinyuan-huang/predicting-quantum-properties",
        "official_master_commit_checked": "dce179128f7fd91a8f9128f29be5492e8a1461bb",
        "official_cpp_measurement_commit": "d63667fc7e6a5c8d7960c678d1f462864ca51ad3",
        "implementation_path": str(CORE_PATH),
        "implementation_sha256": sha256(CORE_PATH),
        "authoritative_small_scale_check": official_cpp_toy_check(),
        "shots": args.shots,
        "cases": [
            validate_case(name, args.shots, args.legacy_prefix)
            for name in ("BeH2", "N2")
        ],
    }
    if (
        not payload["authoritative_small_scale_check"]["exact_setting_by_setting_match"]
        or not all(
        case["paper_fixed_budget"]["prefix_replay_exact"]
        and case["paper_fixed_budget"]["zero_hit_observables"] == 0
        and case["paper_fixed_budget"]["terminal_repeat_length"] <= 10
        and case["qwc_seeded_production"]["combined_T_metrics"]["zero_hit_observables"] == 0
        for case in payload["cases"]
        )
    ):
        payload["status"] = "FAIL"
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
