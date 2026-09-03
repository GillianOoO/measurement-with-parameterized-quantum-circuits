import math
import unittest

import numpy as np

from mpqc_measurement.allocation import evaluate_error, integer_range_allocation
from mpqc_measurement.models import Fragment, HamiltonianData


class AllocationAndErrorTests(unittest.TestCase):
    def test_largest_remainder_is_exact_and_minimum_one(self):
        allocation = integer_range_allocation([1.0, 2.0, 1.0], 10)
        np.testing.assert_array_equal(allocation, [3, 4, 3])
        self.assertEqual(int(np.sum(allocation)), 10)
        self.assertTrue(np.all(allocation >= 1))

    def test_infeasible_budget_fails(self):
        with self.assertRaises(ValueError):
            integer_range_allocation([1.0, 1.0, 1.0], 2)

    def test_every_strictly_positive_range_is_active(self):
        allocation = integer_range_allocation([1.0e-30, 1.0], 2)
        np.testing.assert_array_equal(allocation, [1, 1])

    def test_shot_budget_requires_a_positive_integer(self):
        for invalid in (0, -1, 2.0, 2.7, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                integer_range_allocation([1.0], invalid)

    def test_nonfinite_numeric_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            HamiltonianData(np.asarray([[np.nan, 0.0], [0.0, 1.0]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            Fragment(np.eye(2), np.asarray([np.inf, 0.0]), "bad")
        with self.assertRaisesRegex(ValueError, "finite"):
            integer_range_allocation([np.inf], 10)

    def test_float32_scale_unitarity_tolerance(self):
        nearly_unitary = np.eye(2, dtype=np.complex128)
        nearly_unitary[0, 0] += 1.0e-7
        Fragment(nearly_unitary, np.asarray([1.0, -1.0]), "float32-roundoff")

        materially_nonunitary = np.eye(2, dtype=np.complex128)
        materially_nonunitary[0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "not unitary"):
            Fragment(
                materially_nonunitary,
                np.asarray([1.0, -1.0]),
                "materially-nonunitary",
            )

    def test_exact_state_rmse(self):
        z = np.diag([1.0, -1.0])
        plus = np.asarray([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
        fragment = Fragment(np.eye(2), np.asarray([1.0, -1.0]), "Z")
        data = HamiltonianData(z, state=plus)
        estimate = evaluate_error(data, z, [fragment], np.asarray([100]))
        self.assertAlmostEqual(estimate.state_bias, 0.0, places=12)
        self.assertAlmostEqual(estimate.state_sampling_variance, 0.01, places=12)
        self.assertAlmostEqual(estimate.state_rmse, 0.1, places=12)


if __name__ == "__main__":
    unittest.main()
