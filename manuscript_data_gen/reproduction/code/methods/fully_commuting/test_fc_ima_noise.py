"""Focused exactness tests for overlapping FC-IMA noisy moments."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import traditional_cm_noise_eval as target


class DefaultGridTests(unittest.TestCase):
    def test_default_grid_is_ten_point_inclusive_linspace(self) -> None:
        grid = np.asarray(target.P_GRID, dtype=float)
        self.assertEqual(len(grid), 10)
        self.assertEqual(float(grid[0]), 0.0)
        self.assertEqual(float(grid[-1]), 0.003)
        np.testing.assert_allclose(np.diff(grid), np.full(9, 0.003 / 9.0))


class AttenuationTests(unittest.TestCase):
    def test_fc_rejects_uncompiled_two_qubit_gate(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        with self.assertRaisesRegex(RuntimeError, "non-CX"):
            target.validate_cx_only_two_qubit_operations(
                target.compiled_operations(circuit)
            )

    def test_fc_accepts_explicit_cx_and_one_qubit_gates(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.s(1)
        target.validate_cx_only_two_qubit_operations(
            target.compiled_operations(circuit)
        )

    def test_identity_never_attenuates(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        counts, x, z = target.attenuation_exponents(
            np.asarray([0], dtype=np.uint64), target.compiled_operations(circuit)
        )
        np.testing.assert_array_equal(counts, [0])
        np.testing.assert_array_equal(x, [0])
        np.testing.assert_array_equal(z, [0])

    def test_cx_support_and_reverse_map(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        # Output Z on target maps to Z_control Z_target and is attenuated once.
        counts, x, z = target.attenuation_exponents(
            np.asarray([0b10], dtype=np.uint64), target.compiled_operations(circuit)
        )
        np.testing.assert_array_equal(counts, [1])
        np.testing.assert_array_equal(x, [0])
        np.testing.assert_array_equal(z, [0b11])

    def test_single_qubit_cliffords_do_not_add_noise(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.sx(0)
        circuit.rz(math.pi / 2.0, 0)
        counts, _, _ = target.attenuation_exponents(
            np.asarray([1], dtype=np.uint64), target.compiled_operations(circuit)
        )
        np.testing.assert_array_equal(counts, [0])

    def test_channel_formula(self) -> None:
        p = 0.003
        k = np.asarray([0, 1, 7, 23])
        np.testing.assert_allclose(np.power(1.0 - p, k), [1.0, 0.997, 0.997**7, 0.997**23])

    def test_first_and_second_moments_against_dense_channel(self) -> None:
        """A full two-qubit density-matrix check of both exact moments."""
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        operations = target.compiled_operations(circuit)
        output_masks = np.asarray([0b01, 0b10, 0b11], dtype=np.uint64)
        coefficients = np.asarray([0.37, -0.21, 0.13])
        term_k, _, _ = target.attenuation_exponents(output_masks, operations)
        pair_masks = np.bitwise_xor(output_masks[:, None], output_masks[None, :])
        unique, inverse = np.unique(pair_masks, return_inverse=True)
        unique_k, _, _ = target.attenuation_exponents(unique, operations)
        pair_k = unique_k[inverse].reshape(pair_masks.shape)

        raw = np.asarray([1.0, 0.2 + 0.1j, -0.3j, 0.4], dtype=np.complex128)
        psi = raw / np.linalg.norm(raw)
        rho = np.outer(psi, psi.conjugate())
        unitary = Operator(circuit).data
        ideal_output = unitary @ rho @ unitary.conjugate().T
        identity = np.eye(2, dtype=np.complex128)
        z = np.diag([1.0, -1.0]).astype(np.complex128)
        # Qiskit dense order is |q1 q0>.
        output_paulis = [np.kron(identity, z), np.kron(z, identity), np.kron(z, z)]
        observable = sum(c * pauli for c, pauli in zip(coefficients, output_paulis))
        ideal_term_means = np.asarray(
            [np.trace(pauli @ ideal_output).real for pauli in output_paulis]
        )
        ideal_pair_means = np.asarray(
            [
                [np.trace(left @ right @ ideal_output).real for right in output_paulis]
                for left in output_paulis
            ]
        )
        p = 0.17
        formula_mean = float(np.sum(coefficients * (1.0 - p) ** term_k * ideal_term_means))
        formula_second = float(
            np.sum(
                coefficients[:, None]
                * coefficients[None, :]
                * (1.0 - p) ** pair_k
                * ideal_pair_means
            )
        )
        noisy_output = (1.0 - p) * ideal_output + (p / 4.0) * np.eye(4)
        dense_mean = float(np.trace(observable @ noisy_output).real)
        dense_second = float(np.trace(observable @ observable @ noisy_output).real)
        self.assertAlmostEqual(formula_mean, dense_mean, places=13)
        self.assertAlmostEqual(formula_second, dense_second, places=13)


class PooledImaEstimatorTests(unittest.TestCase):
    def test_pooled_shot_counts_implements_eq14(self) -> None:
        a = SimpleNamespace(source_row=2)
        b = SimpleNamespace(source_row=3)
        groups = [[a, b], [a]]
        self.assertEqual(target.pooled_shot_counts(groups, [2, 3]), {2: 5, 3: 2})

    def test_eq14_mean_and_eq17_variance_for_overlapping_groups(self) -> None:
        # P_a occurs in both bases; P_b occurs only in basis 0.
        # m=(2,3), hence M_a=5 and M_b=2.
        coefficients0 = np.asarray([1.2, -0.7])
        pooled0 = np.asarray([5.0, 2.0])
        means0 = np.asarray([0.25, -0.4])
        pairs0 = np.asarray([[1.0, 0.1], [0.1, 1.0]])
        mean0, variance0, _, _ = target.pooled_ima_group_contribution(
            2, coefficients0, pooled0, means0, pairs0
        )
        mean1, variance1, _, _ = target.pooled_ima_group_contribution(
            3,
            np.asarray([1.2]),
            np.asarray([5.0]),
            np.asarray([0.25]),
            np.asarray([[1.0]]),
        )

        # At p=0, pooling exactly reconstructs sum_k c_k <P_k>.
        self.assertAlmostEqual(mean0 + mean1, 1.2 * 0.25 - 0.7 * -0.4)

        w_a = 1.2 / 5.0
        w_b = -0.7 / 2.0
        expected0 = 2.0 * (
            w_a**2 * (1.0 - 0.25**2)
            + w_b**2 * (1.0 - (-0.4) ** 2)
            + 2.0 * w_a * w_b * (0.1 - 0.25 * -0.4)
        )
        expected1 = 3.0 * w_a**2 * (1.0 - 0.25**2)
        self.assertAlmostEqual(variance0, expected0)
        self.assertAlmostEqual(variance1, expected1)

    def test_basis_dependent_noisy_means_are_weighted_by_basis_shots(self) -> None:
        # The same Pauli has a different noisy mean in two diagonalizers.
        # Eq. (14) pools all five outcomes rather than selecting one basis.
        first, _, _, _ = target.pooled_ima_group_contribution(
            2, np.asarray([1.0]), np.asarray([5.0]), np.asarray([0.8]), np.asarray([[1.0]])
        )
        second, _, _, _ = target.pooled_ima_group_contribution(
            3, np.asarray([1.0]), np.asarray([5.0]), np.asarray([0.5]), np.asarray([[1.0]])
        )
        self.assertAlmostEqual(first + second, (2.0 * 0.8 + 3.0 * 0.5) / 5.0)


if __name__ == "__main__":
    unittest.main()
