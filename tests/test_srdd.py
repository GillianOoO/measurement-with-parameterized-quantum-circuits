import unittest
from unittest.mock import patch

import numpy as np

from mpqc_measurement.collector import collector_matrix, exact_shallow_collector
from mpqc_measurement.electronic import (
    electronic_hamiltonian_dense,
    fock_orbital_unitary,
    one_body_dense,
    spatial_occupations,
)
from mpqc_measurement.models import ElectronicIntegrals, Fragment, HamiltonianData
from mpqc_measurement.selection import FrontierPoint
from mpqc_measurement.srdd import (
    SRDDConfig,
    build_electronic_frontier,
    fit_srdd,
)


def symmetric_two_body_tensor() -> np.ndarray:
    tensor = np.zeros((2, 2, 2, 2), dtype=float)
    seeds = [
        ((0, 0, 0, 0), 0.70),
        ((1, 1, 1, 1), 0.55),
        ((0, 0, 1, 1), 0.22),
        ((0, 1, 0, 1), 0.13),
    ]
    for (p, q, r, s), value in seeds:
        for index in {
            (p, q, r, s),
            (q, p, r, s),
            (p, q, s, r),
            (q, p, s, r),
            (r, s, p, q),
            (s, r, p, q),
            (r, s, q, p),
            (s, r, q, p),
        }:
            tensor[index] = value
    return tensor


class SRDDTests(unittest.TestCase):
    def test_fock_rotation_matches_one_body_rotation(self):
        theta = 0.31
        rotation = np.asarray(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        coefficients = np.asarray([-0.8, 0.35])
        spatial = rotation @ np.diag(coefficients) @ rotation.T
        fock = fock_orbital_unitary(rotation)
        diagonal = spatial_occupations(2) @ coefficients
        fragment = Fragment(fock.conj().T, diagonal, "one-body")
        np.testing.assert_allclose(fragment.matrix, one_body_dense(spatial), atol=2.0e-12)

    def test_universal_completion_can_be_exact(self):
        x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
        y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
        z = np.diag([1.0, -1.0])
        identity = np.eye(2)
        matrix = 0.7 * np.kron(x, x) - 0.2 * np.kron(y, identity) + 0.3 * np.kron(identity, z)
        result = fit_srdd(HamiltonianData(matrix), 1000, SRDDConfig(calibrate=False))
        np.testing.assert_allclose(result.approximate_hamiltonian, matrix, atol=2.0e-12)
        self.assertAlmostEqual(result.error.approximation_spectral_norm, 0.0, places=11)

    def test_small_electronic_core(self):
        electronic = ElectronicIntegrals(
            constant=0.2,
            one_body=np.asarray([[-1.0, 0.1], [0.1, 0.5]]),
            two_body_chemist=np.zeros((2, 2, 2, 2)),
            n_electrons=2,
        )
        matrix = electronic_hamiltonian_dense(electronic)
        data = HamiltonianData(
            matrix,
            particle_number=2,
            electronic=electronic,
        )
        result = fit_srdd(
            data,
            20,
            SRDDConfig(
                rank_grid=(1,),
                depth_grid=(1,),
                cycles=1,
                angle_steps=1,
                collector_angle_steps=10,
                calibrate=False,
            ),
        )
        np.testing.assert_allclose(result.approximate_hamiltonian, matrix, atol=1.0e-10)
        self.assertEqual(int(np.sum(result.shot_allocation)), 20)
        self.assertEqual(result.metadata["backend"], "electronic SRDD core")

    def test_nonzero_symmetric_eri_uses_formal_srdd_postprocessing(self):
        electronic = ElectronicIntegrals(
            constant=0.15,
            one_body=np.asarray([[-1.10, 0.17], [0.17, 0.42]]),
            two_body_chemist=symmetric_two_body_tensor(),
            n_electrons=2,
        )
        self.assertGreater(float(np.linalg.norm(electronic.two_body_chemist)), 0.0)
        matrix = electronic_hamiltonian_dense(electronic)
        result = fit_srdd(
            HamiltonianData(
                matrix,
                particle_number=2,
                electronic=electronic,
            ),
            30,
            SRDDConfig(
                rank_grid=(1,),
                depth_grid=(1,),
                cycles=1,
                angle_steps=3,
                f3_allocation_iterations=2,
                f3_spectral_evaluations=8,
                collector_angle_steps=30,
                calibrate=False,
            ),
        )
        np.testing.assert_allclose(
            result.approximate_hamiltonian + result.residual,
            matrix,
            atol=2.0e-10,
        )
        self.assertEqual(int(np.sum(result.shot_allocation)), 30)
        selected = result.metadata["selected_point"]
        self.assertEqual(selected["f3_r2"]["range_domain"], "full_fock")
        collector = selected["collector"]
        self.assertEqual(collector["objective"], "greedy")
        self.assertEqual(
            collector["selected_candidate"],
            "collector_tailored_greedy_exact_fill",
        )
        self.assertGreaterEqual(collector["extra_rotation_count"], 1)
        self.assertEqual(
            collector["dictionary_rank"],
            collector["symmetric_target_dimension"],
        )
        self.assertLess(collector["matrix_relative_reconstruction_residual"], 5.0e-10)

    def test_depth_is_selected_within_each_rank_before_fixed_shot_selection(self):
        electronic = ElectronicIntegrals(
            constant=0.0,
            one_body=np.diag([-1.0, 0.4]),
            two_body_chemist=np.zeros((2, 2, 2, 2)),
            n_electrons=2,
        )
        matrix = electronic_hamiltonian_dense(electronic)
        data = HamiltonianData(
            matrix,
            particle_number=2,
            electronic=electronic,
        )
        sector_index = next(
            index for index in range(data.dimension) if index.bit_count() == 2
        )
        metrics = {
            (1, 1): 1.0,
            (1, 2): 0.5 + 5.0e-13,
            (1, 3): 0.5,
            (2, 1): 0.8,
            (2, 2): 0.4,
            (2, 3): 0.6,
        }

        def fake_source(_data, rank, depth, _config):
            residual = np.zeros_like(matrix)
            residual[sector_index, sector_index] = metrics[(rank, depth)]
            return {
                "rank": rank,
                "depth": depth,
                "source_residual": residual,
            }

        def fake_materialization(_data, source, _config):
            residual = np.asarray(source["source_residual"])
            return FrontierPoint(
                size=int(source["rank"]),
                constant=0.0,
                fragments=[],
                approximate=matrix - residual,
                residual=residual,
                metadata={
                    "rank": int(source["rank"]),
                    "depth": int(source["depth"]),
                },
            )

        config = SRDDConfig(
            rank_grid=(1, 2),
            depth_grid=(1, 2, 3),
            calibrate=False,
        )
        with patch(
            "mpqc_measurement.srdd._fit_electronic_source",
            side_effect=fake_source,
        ) as source_fit, patch(
            "mpqc_measurement.srdd._materialize_electronic_point",
            side_effect=fake_materialization,
        ) as materialize:
            frontier = build_electronic_frontier(data, 10, config)
        self.assertEqual(source_fit.call_count, 6)
        self.assertEqual(materialize.call_count, 3)
        self.assertEqual([point.size for point in frontier], [0, 1, 2])
        self.assertEqual([point.metadata["depth"] for point in frontier], [1, 2, 2])
        self.assertTrue(
            all(
                len(point.metadata["depth_selection"]["trials"]) == 3
                for point in frontier
                if point.size > 0
            )
        )
        self.assertEqual(
            frontier[1].metadata["depth_selection"]["tie_break"],
            "shallower depth",
        )

    def test_collector_uses_full_rank_greedy_exact_fill(self):
        source = np.eye(2, dtype=float)[None, :, :]
        target = np.asarray([[1.1, 0.28], [0.28, -0.35]])
        rotations, coefficients, audit = exact_shallow_collector(
            source,
            target,
            1,
            seed_key="collector-regression",
            angle_steps=40,
        )
        np.testing.assert_allclose(
            collector_matrix(rotations, coefficients),
            target,
            atol=5.0e-10,
        )
        self.assertEqual(SRDDConfig().collector_angle_steps, 400)
        self.assertGreaterEqual(audit["extra_rotation_count"], 1)
        self.assertEqual(audit["dictionary_rank"], 3)
        self.assertEqual(audit["symmetric_target_dimension"], 3)
        self.assertEqual(audit["objective"], "greedy")
        self.assertEqual(
            audit["selected_candidate"],
            "collector_tailored_greedy_exact_fill",
        )
        self.assertLess(audit["matrix_relative_reconstruction_residual"], 5.0e-10)


if __name__ == "__main__":
    unittest.main()
