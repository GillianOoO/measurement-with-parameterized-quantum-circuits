import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("traditional_cm_error_eval.py")
SPEC = importlib.util.spec_from_file_location("traditional_cm_error_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def term(label: str, coefficient: float = 1.0, row: int = 2):
    xmask, zmask, y_count = MODULE.label_to_masks(label)
    return MODULE.PauliTerm(row, label, coefficient, xmask, zmask, y_count)


class TraditionalCMErrorEvalTests(unittest.TestCase):
    def test_pauli_action_matches_site_ordered_kronecker_matrix(self):
        rng = np.random.default_rng(12891)
        state = rng.normal(size=8) + 1j * rng.normal(size=8)
        state /= np.linalg.norm(state)
        matrices = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.diag([1, -1]).astype(complex),
        }
        parity = MODULE.parity_table(3)
        for label in ("XYZ", "YII", "IZY", "XXX", "ZZY"):
            dense = matrices[label[0]]
            for letter in label[1:]:
                dense = np.kron(dense, matrices[letter])
            actual = MODULE.apply_pauli_dense(state, term(label), parity)
            np.testing.assert_allclose(actual, dense @ state, atol=2.0e-14)

    def test_mps_contraction_uses_site_zero_as_most_significant_index(self):
        # Product state |0,1,0> in site0,site1,site2 order.
        arrays = []
        for occupied in (0, 1, 0):
            tensor = np.zeros((1, 2, 1), dtype=float)
            tensor[0, occupied, 0] = 1.0
            arrays.append(tensor)
        value = arrays[0][0, :, :]
        for array in arrays[1:]:
            value = np.tensordot(value, array, axes=(-1, 0))
        state = value[..., 0].reshape(-1)
        expected = np.zeros(8)
        expected[2] = 1.0  # binary 010
        np.testing.assert_array_equal(state, expected)

    def test_generator_sign_and_exact_range_for_xx_yy_zz(self):
        group = [term("XX", row=2), term("YY", row=3), term("ZZ", row=4)]
        minimum, maximum, half_range, rank, coordinates, signs = (
            MODULE.exact_group_range(group, 2)
        )
        self.assertEqual(rank, 2)
        self.assertEqual((minimum, maximum, half_range), (-3.0, 1.0, 2.0))
        self.assertEqual(len(coordinates), 3)
        self.assertIn(-1, signs)

    def test_sorted_insertion_is_disjoint_and_fully_commuting(self):
        terms = [
            term("XX", 4.0, 2),
            term("ZZ", 3.0, 3),
            term("XI", 2.0, 4),
            term("IZ", 1.0, 5),
        ]
        groups = MODULE.sorted_insertion_fc(terms)
        self.assertEqual(sum(map(len, groups)), len(terms))
        self.assertEqual(
            len({item.source_row for group in groups for item in group}), len(terms)
        )
        for group in groups:
            for i, left in enumerate(group):
                self.assertTrue(
                    all(MODULE.fully_commutes(left, right) for right in group[i + 1 :])
                )

    def test_extended_si_preserves_paper_pstar_order_and_seed_partition(self):
        labels = ("IIX", "IIY", "IIZ", "IXI", "IXX", "IXY", "XII")
        terms = [term(label, 10.0 - index, index + 2) for index, label in enumerate(labels)]
        groups, seed_rows = MODULE.extended_sorted_insertion_fc(terms)
        self.assertEqual(
            [[item.label for item in group] for group in groups],
            [
                ["IIX", "IXI", "IXX", "XII"],
                ["IIY", "IXY", "IXI", "XII"],
                ["IIZ", "IXI", "XII"],
            ],
        )
        self.assertEqual(set().union(*seed_rows), {item.source_row for item in terms})
        self.assertEqual(sum(map(len, seed_rows)), len(terms))

    def test_pooled_estimator_variance_matches_explicit_eq17_fragments(self):
        rng = np.random.default_rng(7701)
        state = rng.normal(size=4) + 1j * rng.normal(size=4)
        state /= np.linalg.norm(state)
        support = np.arange(4, dtype=np.uint32)
        parity = MODULE.parity_table(2)
        xx = term("XX", 0.7, 2)
        zz = term("ZZ", -0.2, 3)
        zi = term("ZI", 0.4, 4)
        groups = [[xx, zz], [zz, zi]]
        memberships = MODULE.pauli_memberships(groups)
        measurement = np.array([3.0, 5.0])
        actual, fragment_variances, pooled, reconstructed_mean = (
            MODULE.pooled_estimator_variance(
                groups, measurement, memberships, state, support, parity
            )
        )
        fragments, _ = MODULE.redistributed_fragments(
            groups, measurement, memberships
        )
        expected_variances = np.array(
            [
                MODULE.group_moments_sparse(group, state, support, parity)[1]
                for group in fragments
            ]
        )
        self.assertAlmostEqual(actual, float(np.sum(expected_variances / measurement)))
        np.testing.assert_allclose(fragment_variances, expected_variances)
        self.assertEqual(pooled, {2: 3.0, 3: 8.0, 4: 5.0})
        original_mean = sum(
            MODULE.group_moments_sparse([item], state, support, parity)[0]
            for item in (xx, zz, zi)
        )
        self.assertAlmostEqual(reconstructed_mean, original_mean, places=13)

    def test_yen_diagonalizer_born_distribution_reproduces_group_moments(self):
        state = np.zeros(4, dtype=complex)
        state[0] = state[3] = 1 / np.sqrt(2)
        parity = MODULE.parity_table(2)
        support = np.flatnonzero(np.abs(state) > 0).astype(np.uint32)
        group = [term("XX", 0.31, 2), term("ZZ", -0.17, 3)]
        distribution = MODULE.exact_group_born_distribution(group, state)
        values = np.zeros(len(distribution.basis_indices))
        for item, mask, sign in zip(
            group, distribution.term_z_masks, distribution.term_signs
        ):
            eigenvalues = 1 - 2 * parity[
                np.bitwise_and(distribution.basis_indices, mask)
            ].astype(np.int8)
            values += item.coefficient * int(sign) * eigenvalues
        expected_mean = float(np.sum(distribution.probabilities * values))
        expected_variance = float(
            np.sum(distribution.probabilities * values**2) - expected_mean**2
        )
        actual_mean, actual_variance, _ = MODULE.group_moments_sparse(
            group, state, support, parity
        )
        self.assertAlmostEqual(float(np.sum(distribution.probabilities)), 1.0, places=13)
        self.assertAlmostEqual(expected_mean, actual_mean, places=12)
        self.assertAlmostEqual(expected_variance, actual_variance, places=12)

    def test_largest_remainder_matches_manuscript_rule(self):
        actual = MODULE.largest_remainder_allocation(11, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(actual, np.array([2, 4, 5]))
        self.assertEqual(int(actual.sum()), 11)
        self.assertTrue(np.all(actual >= 1))

    def test_ima_initial_allocation_has_no_covariance_dictionary_floor(self):
        initial = MODULE.variance_proportions([1.0e-16, 1.0], variance_floor=0.0)
        update = MODULE.variance_proportions([1.0e-16, 1.0])
        self.assertLess(initial[0], 2.0e-8)
        self.assertGreater(update[0], 9.0e-5)

    def test_sparse_group_moments_match_dense_matrix(self):
        rng = np.random.default_rng(311)
        state = np.zeros(8, dtype=complex)
        support = np.array([1, 2, 5, 6], dtype=np.uint32)
        state[support] = rng.normal(size=4) + 1j * rng.normal(size=4)
        state /= np.linalg.norm(state)
        group = [term("XYZ", 0.31, 2), term("ZYX", -0.17, 3)]
        self.assertTrue(MODULE.fully_commutes(group[0], group[1]))
        parity = MODULE.parity_table(3)
        mean, variance, norm_squared = MODULE.group_moments_sparse(
            group, state, support, parity
        )
        acted = sum(
            item.coefficient * MODULE.apply_pauli_dense(state, item, parity)
            for item in group
        )
        expected_mean = float(np.vdot(state, acted).real)
        expected_norm = float(np.vdot(acted, acted).real)
        self.assertAlmostEqual(mean, expected_mean, places=13)
        self.assertAlmostEqual(norm_squared, expected_norm, places=13)
        self.assertAlmostEqual(variance, expected_norm - expected_mean**2, places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
