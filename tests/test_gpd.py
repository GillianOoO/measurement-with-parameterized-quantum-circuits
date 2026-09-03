import unittest
import warnings

import numpy as np

from mpqc_measurement.gpd import GPDConfig, build_gpd_frontier, fit_gpd
from mpqc_measurement.models import HamiltonianData


class GPDTests(unittest.TestCase):
    def test_odd_periodic_schedule_fails_closed(self):
        matrix = np.diag(np.arange(8, dtype=float))
        with self.assertRaisesRegex(ValueError, "even qubit count"):
            fit_gpd(
                HamiltonianData(matrix),
                10,
                GPDConfig(max_terms=1, steps=1, calibrate=False),
            )

    def test_seeded_dense_smoke_and_allocation(self):
        x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
        z = np.diag([1.0, -1.0])
        matrix = np.kron(z, z) + 0.2 * np.kron(x, np.eye(2))
        config = GPDConfig(
            paper_depth=0,
            max_terms=1,
            starts=1,
            steps=2,
            calibrate=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_gpd(HamiltonianData(matrix), 10, config)
        self.assertEqual(result.selected_size, 1)
        self.assertEqual(
            result.metadata["numeric_profile"]["optimizer"], "float32/complex64"
        )
        self.assertEqual(int(np.sum(result.shot_allocation)), 10)
        np.testing.assert_allclose(
            matrix,
            result.approximate_hamiltonian + result.residual,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(
            result.approximate_hamiltonian,
            result.approximate_hamiltonian.conj().T,
            atol=2.0e-10,
        )

    def test_nontrivial_frontier_contains_scalar_candidate(self):
        matrix = np.diag([-1.0, -0.2, 0.4, 1.3])
        frontier = build_gpd_frontier(
            HamiltonianData(matrix),
            10,
            GPDConfig(
                paper_depth=0,
                max_terms=1,
                steps=1,
                calibrate=False,
            ),
        )
        self.assertEqual([point.size for point in frontier], [0, 1])
        self.assertEqual(frontier[0].fragments, [])
        np.testing.assert_allclose(
            frontier[0].approximate,
            np.trace(matrix).real / 4 * np.eye(4),
        )

    def test_constant_hamiltonian_uses_no_quantum_setting(self):
        matrix = 1.25 * np.eye(4)
        result = fit_gpd(
            HamiltonianData(matrix),
            10,
            GPDConfig(max_terms=1, steps=1, calibrate=False),
        )
        self.assertEqual(result.selected_size, 0)
        self.assertEqual(len(result.fragments), 0)
        self.assertEqual(result.metadata["unused_shots"], 10)


if __name__ == "__main__":
    unittest.main()
