#!/usr/bin/env python3
"""Unit and regression tests for the corrected Huang-2021 implementation."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent
CORE_DIR = ROOT
sys.path.insert(0, str(CORE_DIR))

from derand_huang2021 import (  # noqa: E402
    appendix_c_schedule,
    github_target_schedule,
    schedule_hit_counts,
)


OFFICIAL_CPP_TOY_GOLDEN = np.asarray(
    [
        [1, 2, 3],
        [2, 3, 1],
        [3, 1, 2],
        [2, 3, 3],
        [3, 1, 1],
        [1, 2, 3],
        [2, 3, 1],
        [3, 1, 2],
        [1, 2, 2],
        [2, 3, 3],
        [3, 1, 1],
        [2, 3, 1],
        [3, 1, 1],
        [2, 1, 3],
        [1, 2, 3],
        [3, 1, 2],
    ],
    dtype=np.int8,
)


def load_official_toy() -> tuple[np.ndarray, np.ndarray]:
    path = Path(__file__).with_name("official_cpp_toy_observables.txt")
    rows: list[list[int]] = []
    weights: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if line_number == 0:
            parts = parts[1:]
        locality = int(parts[0])
        row = [0, 0, 0]
        for index in range(locality):
            pauli = {"X": 1, "Y": 2, "Z": 3}[parts[1 + 2 * index]]
            qubit = int(parts[2 + 2 * index])
            row[qubit] = pauli
        rows.append(row)
        weights.append(float(parts[1 + 2 * locality]))
    return np.asarray(rows, dtype=np.int8), np.asarray(weights, dtype=float)


def scalar_c8_schedule(
    observables: np.ndarray, coefficients: np.ndarray, shots: int, eta: float = 0.9
) -> np.ndarray:
    """Small, direct full-cost implementation of paper Eqs. C8/C11."""

    weights = np.abs(coefficients) / np.max(np.abs(coefficients))
    hits = np.zeros(len(weights), dtype=int)
    nu = 1.0 - math.exp(-eta / 2.0)
    output = np.empty((shots, observables.shape[1]), dtype=np.int8)
    for shot in range(shots):
        prefix = np.ones(len(weights), dtype=bool)
        for site in range(observables.shape[1]):
            values = []
            candidate_prefixes = []
            remaining = np.sum(observables[:, site + 1 :] != 0, axis=1)
            for action in (1, 2, 3):
                compatible = prefix & (
                    (observables[:, site] == 0)
                    | (observables[:, site] == action)
                )
                log_factor = np.zeros(len(weights), dtype=float)
                log_factor[compatible] = np.log1p(
                    -nu * np.power(1.0 / 3.0, remaining[compatible])
                )
                log_cost = (-eta * hits / 2.0 + log_factor) / weights
                values.append(math.fsum(math.exp(value) for value in log_cost))
                candidate_prefixes.append(compatible)
            selected = int(np.argmin(values))
            output[shot, site] = selected + 1
            prefix = candidate_prefixes[selected]
        hits += prefix
    return output


class HuangDerandomizationTests(unittest.TestCase):
    def test_official_cpp_target_golden(self) -> None:
        observables, weights = load_official_toy()
        actual = github_target_schedule(observables, weights, 3)
        np.testing.assert_array_equal(actual, OFFICIAL_CPP_TOY_GOLDEN)
        hits = schedule_hit_counts(observables, actual)
        self.assertTrue(np.all(hits >= np.floor(3 * weights)))

    def test_one_qubit_equal_weight_cycle_and_tie_order(self) -> None:
        observables = np.asarray([[1], [2], [3]], dtype=np.int8)
        actual = appendix_c_schedule(observables, np.ones(3), 9)
        np.testing.assert_array_equal(actual[:, 0], np.asarray([1, 2, 3] * 3))

    def test_stable_vectorization_matches_direct_c8(self) -> None:
        observables, _ = load_official_toy()
        coefficients = np.asarray(
            [1.0, 0.8, 0.6, 0.9, 0.7, 0.5, 0.95, 0.75, 0.55,
             0.65, 0.45, 0.35, 0.85, 0.4, 0.3]
        )
        expected = scalar_c8_schedule(observables, coefficients, 50)
        actual = appendix_c_schedule(observables, coefficients, 50)
        np.testing.assert_array_equal(actual, expected)

    def test_prefix_and_global_coefficient_scale_invariance(self) -> None:
        observables, _ = load_official_toy()
        coefficients = np.linspace(0.2, 1.6, len(observables))
        short = appendix_c_schedule(observables, coefficients, 31)
        long = appendix_c_schedule(observables, coefficients, 100)
        scaled = appendix_c_schedule(observables, -7.25 * coefficients, 100)
        np.testing.assert_array_equal(short, long[: len(short)])
        np.testing.assert_array_equal(long, scaled)

    def test_seeded_continuation_is_exact(self) -> None:
        observables, _ = load_official_toy()
        coefficients = np.linspace(0.3, 1.7, len(observables))
        full = appendix_c_schedule(observables, coefficients, 80)
        prefix = full[:17]
        initial_hits = schedule_hit_counts(observables, prefix)
        tail = appendix_c_schedule(
            observables, coefficients, 63, initial_hits=initial_hits
        )
        np.testing.assert_array_equal(tail, full[17:])

    def test_large_initial_hits_remain_finite_and_unlocked(self) -> None:
        observables, _ = load_official_toy()
        coefficients = np.geomspace(1.0e-8, 1.0, len(observables))
        initial_hits = np.arange(len(observables), dtype=np.int64) * 100_000
        schedule, audit = appendix_c_schedule(
            observables,
            coefficients,
            200,
            initial_hits=initial_hits,
            return_audit=True,
        )
        self.assertTrue(np.all((schedule >= 1) & (schedule <= 3)))
        self.assertGreater(audit.distinct_settings, 1)
        self.assertLess(audit.terminal_repeat_length, len(schedule))


if __name__ == "__main__":
    unittest.main(verbosity=2)
