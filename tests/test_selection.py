import hashlib
import unittest

import numpy as np

from mpqc_measurement.models import Fragment, HamiltonianData
from mpqc_measurement.selection import (
    CALIBRATION_SHOTS,
    FrontierPoint,
    MaterialStallTracker,
    calibrate_weights,
    generate_grouped_stabilizer_probes,
    select_frontier,
)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


class SelectionTests(unittest.TestCase):
    def test_four_qubit_probe_catalog_is_frozen(self):
        probes, splits, audit = generate_grouped_stabilizer_probes(
            np.arange(16), 16, seed=918273
        )
        self.assertEqual(probes.shape, (16, 496))
        self.assertEqual(audit["state_split_counts"], {
            "train": 298,
            "validation": 99,
            "test": 99,
        })
        self.assertEqual(
            array_sha256(probes),
            "99d5eee207c84f6fd075b275ce26916cd270be7a95332e5b14148e850907f47a",
        )
        np.testing.assert_allclose(np.sum(np.abs(probes) ** 2, axis=0), 1.0)
        self.assertEqual(len(splits), 496)

    def test_four_particle_eight_qubit_probe_catalog_is_frozen(self):
        indices = np.asarray([index for index in range(256) if index.bit_count() == 4])
        probes, _, audit = generate_grouped_stabilizer_probes(
            indices, 256, seed=918273
        )
        self.assertEqual(probes.shape, (256, 500))
        self.assertEqual(audit["state_split_counts"], {
            "train": 300,
            "validation": 100,
            "test": 100,
        })
        self.assertEqual(
            array_sha256(probes),
            "8b2e38ba5f4cef458ef8fe193bdeb9dbb7ccbcc52173bdf2532c6781eba9726e",
        )

    def test_calibration_uses_fixed_sizes_and_shot_grid(self):
        diagonal = np.asarray([-1.0, -0.2, 0.3, 1.1])
        matrix = np.diag(diagonal)
        data = HamiltonianData(matrix)
        identity = np.eye(4)
        constant = float(np.trace(matrix).real / 4)
        centered = diagonal - constant
        frontier = [
            FrontierPoint(
                size=0,
                constant=constant,
                fragments=[],
                approximate=constant * identity,
                residual=matrix - constant * identity,
            )
        ]
        fragments = []
        for size in range(1, 11):
            fragments.append(Fragment(identity, centered / 10.0, f"part-{size}"))
            approximate = constant * identity + np.diag(size * centered / 10.0)
            frontier.append(
                FrontierPoint(
                    size=size,
                    constant=constant,
                    fragments=list(fragments),
                    approximate=approximate,
                    residual=matrix - approximate,
                )
            )
        first = calibrate_weights(data, frontier, seed=918273)
        second = calibrate_weights(data, frontier, seed=918273)
        self.assertEqual(first[:2], second[:2])
        self.assertEqual(first[2]["calibration_sizes"], list(range(1, 11)))
        self.assertEqual(first[2]["calibration_shot_grid"], list(CALIBRATION_SHOTS))
        self.assertGreater(first[0], 0.0)
        self.assertGreater(first[1], 0.0)

    def test_calibration_rejects_an_incomplete_frontier(self):
        matrix = np.diag([-1.0, 1.0])
        point = FrontierPoint(0, 0.0, [], np.zeros((2, 2)), matrix)
        with self.assertRaisesRegex(ValueError, "missing"):
            calibrate_weights(HamiltonianData(matrix), [point])

    def test_rank_zero_candidate_can_win_a_small_budget(self):
        matrix = np.diag([-0.5, 0.5])
        data = HamiltonianData(matrix)
        zero = FrontierPoint(0, 0.0, [], np.zeros((2, 2)), matrix)
        high_range = Fragment(np.eye(2), np.asarray([-10.0, 10.0]), "high-range")
        one = FrontierPoint(1, 0.0, [high_range], matrix, np.zeros((2, 2)))
        point, allocation, _, _ = select_frontier(
            data, [zero, one], 1, calibrate=False
        )
        self.assertEqual(point.size, 0)
        self.assertEqual(len(allocation), 0)

    def test_material_stall_replay_and_strict_boundary(self):
        tracker = MaterialStallTracker()
        for size in range(11):
            self.assertFalse(tracker.update(size, 2.0 - 0.1 * size))
        self.assertFalse(tracker.update(11, 1.0))
        self.assertFalse(tracker.update(12, 1.0))
        self.assertFalse(tracker.update(13, 0.9))
        for size in range(14, 18):
            self.assertFalse(tracker.update(size, 0.9))
        self.assertTrue(tracker.update(18, 0.9))
        self.assertEqual(tracker.stop_k, 18)

        boundary = MaterialStallTracker(relative_improvement=2.0e-3)
        boundary.update(10, 1.0)
        boundary.update(11, 0.998)
        self.assertFalse(boundary.events[-1]["materially_improved"])
        self.assertEqual(boundary.stalled, 1)


if __name__ == "__main__":
    unittest.main()
