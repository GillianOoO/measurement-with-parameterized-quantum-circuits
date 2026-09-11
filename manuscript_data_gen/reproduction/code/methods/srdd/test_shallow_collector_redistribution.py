#!/usr/bin/env python3
"""Focused tests for exact shallow-collector redistribution."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

sys.path.insert(0, str(HERE.parents[1] / "experiments"))
import large_molecule_srdd_pauli as engine
import shallow_collector_redistribution as collector_core
from shallow_rcdf import rotation_and_derivatives


def shallow_bank(m: int, depth: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray([
        rotation_and_derivatives(
            rng.uniform(-np.pi, np.pi, size=depth * (m - 1)), m, depth
        )[0]
        for _ in range(count)
    ])


def synthetic_problem(source_count: int):
    m = 4
    depth = 3
    rotations = shallow_bank(m, depth, source_count, 8231)
    rng = np.random.default_rng(771)
    raw = rng.normal(size=(m, m))
    collector = 0.5 * (raw + raw.T)
    configs = engine.determinant_configs(m, 2)
    occupations = engine.spatial_occupations(configs, m)
    all_occupations = engine.all_spatial_occupations(m)
    sector_shape = (len(configs), len(configs))
    base_sector = np.zeros((source_count, *sector_shape))
    base_full = np.zeros((source_count, len(all_occupations)))
    ci = rng.normal(size=sector_shape)
    ci /= np.linalg.norm(ci)
    return (
        collector,
        rotations,
        base_sector,
        base_full,
        configs,
        occupations,
        all_occupations,
        ci,
        depth,
    )


class ShallowCollectorRedistributionTests(unittest.TestCase):
    def run_solution(self, source_count: int, extra_leaves: int, objective: str = "hybrid"):
        problem = synthetic_problem(source_count)
        rotations, eta, _, audit = engine.shallow_collector_redistribution(
            *problem[:-1],
            depth=problem[-1],
            extra_leaves=extra_leaves,
            objective=objective,
            proxy_state="ground",
            reference_shots=3000,
            proxy_cycles=3,
            equality_tolerance=1.0e-10,
            extra_angle_steps=30,
            seed_key=f"unit-test|K{source_count}|L{extra_leaves}",
        )
        reconstructed = engine.collector_matrix(rotations, eta)
        np.testing.assert_allclose(reconstructed, problem[0], atol=2.0e-9, rtol=2.0e-9)
        self.assertEqual(audit["status"], "PASS")
        self.assertLess(audit["matrix_relative_reconstruction_residual"], 5.0e-10)
        self.assertEqual(sum(audit["selected_reference_shot_vector"]), 3000)
        return rotations, eta, audit

    def test_absorb_is_exact_and_conserves_reference_shots(self):
        rotations, eta, audit = self.run_solution(5, 0)
        self.assertEqual(rotations.shape, (5, 4, 4))
        self.assertEqual(eta.shape, (5, 4))
        self.assertEqual(audit["mode"], "absorb")
        self.assertEqual(audit["dictionary_rank"], 10)
        np.testing.assert_allclose(
            collector_core.collector_matrix(rotations, eta),
            engine.collector_matrix(rotations, eta),
            atol=1.0e-13,
        )

    def test_augment_adds_only_depth_limited_rotations_and_is_exact(self):
        rotations, _, audit = self.run_solution(1, 4)
        self.assertEqual(rotations.shape, (5, 4, 4))
        self.assertEqual(audit["mode"], "augment")
        self.assertEqual(audit["extra_rotation_count"], 4)
        for rotation in rotations:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(4), atol=2.0e-12)
        extra_audit = audit["extra_rotation_construction"]
        for index, (angles, step) in enumerate(
            zip(extra_audit["selected_flat_angles"], extra_audit["steps"])
        ):
            reconstructed = rotation_and_derivatives(np.asarray(angles), 4, 3)[0]
            np.testing.assert_allclose(rotations[1 + index], reconstructed, atol=2.0e-12)
            self.assertLessEqual(
                step["collector_residual_frobenius_after"],
                step["collector_residual_frobenius_before"] + 1.0e-12,
            )

    def test_infeasible_absorption_fails_closed(self):
        problem = synthetic_problem(1)
        with self.assertRaises(engine.CollectorRepresentationError):
            engine.shallow_collector_redistribution(
                *problem[:-1],
                depth=problem[-1],
                extra_leaves=0,
                objective="variance",
                proxy_state="hf",
                reference_shots=3000,
                proxy_cycles=2,
                equality_tolerance=1.0e-10,
                extra_angle_steps=20,
                seed_key="unit-test-infeasible",
            )

    def test_range_and_variance_objectives_remain_exact(self):
        for objective in ("greedy", "variance", "range"):
            with self.subTest(objective=objective):
                _, _, audit = self.run_solution(5, 0, objective)
                self.assertEqual(audit["objective"], objective)


if __name__ == "__main__":
    unittest.main()
