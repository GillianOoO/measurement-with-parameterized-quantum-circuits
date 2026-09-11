"""Fast gate-level regressions for the Fig. 3 SRDD noise model."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import cnot_noise_semantics as target  # noqa: E402


def encode_operator(matrix: np.ndarray) -> np.ndarray:
    """Use the paired bra/ket physical-index layout of gate_superoperator."""
    result = np.empty((4, 4), dtype=np.complex128)
    for first in range(4):
        up_first, down_first = divmod(first, 2)
        for second in range(4):
            up_second, down_second = divmod(second, 2)
            result[first, second] = matrix[
                2 * up_first + up_second, 2 * down_first + down_second
            ]
    return result


def decode_operator(vector: np.ndarray) -> np.ndarray:
    result = np.empty((4, 4), dtype=np.complex128)
    for first in range(4):
        up_first, down_first = divmod(first, 2)
        for second in range(4):
            up_second, down_second = divmod(second, 2)
            result[2 * up_first + up_second, 2 * down_first + down_second] = (
                vector[first, second]
            )
    return result


class FixedTwoCNOTGivensTests(unittest.TestCase):
    def test_fixed_definition_is_two_cx_and_exact_even_at_zero(self) -> None:
        for theta in (0.0, 0.37, -0.91, math.pi):
            with self.subTest(theta=theta):
                operations = target.fixed_two_cx_givens_operations(theta)
                self.assertEqual(
                    [operation.name for operation in operations].count("cx"), 2
                )
                self.assertTrue(
                    all(
                        len(operation.qubits) == 1 or operation.name == "cx"
                        for operation in operations
                    )
                )
                c, s = math.cos(theta), math.sin(theta)
                # shallow_forward_gates transposes the single-particle factor;
                # fock_gate's [|10>,|01>] insertion restores G(theta) in the
                # standard [|01>,|10>] block.
                expected = target.fock_gate(np.asarray([[c, s], [-s, c]]))
                compiled = target.operation_product(operations)
                self.assertLess(
                    target.unitary_equivalence_error(expected, compiled), 2.0e-12
                )
                # Independent Qiskit-library check of both the matrix identity
                # and its fixed two-CX definition.
                library_gate = XXPlusYYGate(2.0 * theta, math.pi / 2.0)
                self.assertEqual(library_gate.definition.count_ops().get("cx", 0), 2)
                self.assertLess(
                    target.unitary_equivalence_error(
                        compiled, np.asarray(Operator(library_gate).data)
                    ),
                    2.0e-12,
                )

    def test_two_per_cx_channels_equal_covariant_composition_for_moments(self) -> None:
        theta = 0.413
        p = 0.037
        operations = target.fixed_two_cx_givens_operations(theta)
        raw = np.asarray([1.0, 0.2 + 0.3j, -0.4j, 0.7 - 0.1j])
        psi = raw / np.linalg.norm(raw)
        initial = np.outer(psi, psi.conjugate())

        explicit = initial.copy()
        cx_seen = 0
        for operation in operations:
            explicit = operation.unitary @ explicit @ operation.unitary.conjugate().T
            if operation.name == "cx":
                cx_seen += 1
                explicit = (1.0 - p) * explicit + (p / 4.0) * np.eye(4)
        self.assertEqual(cx_seen, target.CHANNELS_PER_GIVENS)

        unitary = target.operation_product(operations)
        ideal_output = unitary @ initial @ unitary.conjugate().T
        effective_p = target.composed_depolarizing_probability(
            p, target.CHANNELS_PER_GIVENS
        )
        collapsed = (1.0 - effective_p) * ideal_output + (
            effective_p / 4.0
        ) * np.eye(4)
        np.testing.assert_allclose(explicit, collapsed, rtol=2.0e-13, atol=2.0e-13)

        observable = np.asarray(
            [
                [0.3, 0.1j, 0.2, 0.0],
                [-0.1j, -0.5, 0.07j, -0.1],
                [0.2, -0.07j, 0.8, 0.15],
                [0.0, -0.1, 0.15, -0.2],
            ],
            dtype=np.complex128,
        )
        for moment_operator in (observable, observable @ observable):
            with self.subTest(moment=1 if moment_operator is observable else 2):
                superoperator = target.gate_superoperator(unitary, effective_p)
                propagated_vector = np.einsum(
                    "abij,ij->ab", superoperator, encode_operator(moment_operator)
                )
                propagated = decode_operator(propagated_vector)
                direct_adjoint = unitary.conjugate().T @ (
                    (1.0 - effective_p) * moment_operator
                    + (effective_p * np.trace(moment_operator) / 4.0) * np.eye(4)
                ) @ unitary
                np.testing.assert_allclose(
                    propagated, direct_adjoint, rtol=2.0e-13, atol=2.0e-13
                )
                explicit_moment = np.trace(moment_operator @ explicit)
                collapsed_moment = np.trace(propagated @ initial)
                self.assertAlmostEqual(explicit_moment.real, collapsed_moment.real, places=13)
                self.assertAlmostEqual(explicit_moment.imag, collapsed_moment.imag, places=13)

    def test_probability_composition(self) -> None:
        for p in (0.0, 0.003, 0.4, 1.0):
            self.assertAlmostEqual(
                target.composed_depolarizing_probability(p, 2),
                1.0 - (1.0 - p) ** 2,
                places=15,
            )


if __name__ == "__main__":
    unittest.main()
