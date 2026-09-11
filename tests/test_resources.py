"""Regression checks for native gate counts and executed measurement resources."""

import unittest

from mpqc_measurement.resources import (
    GateResources, gpd_gate_resources, source_gate_resources, summarize_gate_resources,
)


class NativeGateResourcesTests(unittest.TestCase):
    def test_periodic_four_qubit_depth_four(self):
        resource = gpd_gate_resources(4, 4)
        self.assertEqual(resource.iswap_count, 8)
        self.assertEqual(resource.cnot_count, 0)
        self.assertEqual(resource.two_qubit_depth, 4)
        self.assertEqual(resource.two_qubit_count, 8)

    def test_legacy_open_chain_keeps_actual_pairs(self):
        resource = source_gate_resources("shallow_iswap_su2", 4, 4)
        self.assertEqual(resource.iswap_count, 6)  # 2 + 1 + 2 + 1
        self.assertEqual(resource.cnot_count, 0)
        self.assertEqual(resource.two_qubit_depth, 4)
        self.assertEqual(source_gate_resources("shallow_iswap_su2", 4, 8).iswap_count, 14)

    def test_empty_entangling_layers(self):
        self.assertEqual(gpd_gate_resources(4, 0), GateResources())
        self.assertEqual(gpd_gate_resources(1, 4), GateResources())
        # On two open-chain qubits, the odd matching is empty.
        resource = source_gate_resources("shallow_iswap_su2", 4, 2)
        self.assertEqual((resource.iswap_count, resource.two_qubit_depth), (2, 2))

    def test_explicit_odd_qubit_extension(self):
        with self.assertRaisesRegex(ValueError, "even qubit"):
            gpd_gate_resources(3, 4)
        self.assertEqual(gpd_gate_resources(3, 4, "open-chain").iswap_count, 4)

    def test_existing_cnot_compilers_are_preserved(self):
        expected = {"gfro": (24, 8, False), "operator_pool": (24, 6, True),
                    "nnk_uccgsdi": (624, 608, True)}
        for family, values in expected.items():
            resource = source_gate_resources(family, 2, 8)
            self.assertEqual((resource.cnot_count, resource.two_qubit_depth,
                              resource.is_upper_bound), values)
            self.assertEqual(resource.iswap_count, 0)

    def test_mixed_gate_budget_counts_every_setting_shot(self):
        summary = summarize_gate_resources(
            [GateResources(cnot_count=4, two_qubit_depth=2), gpd_gate_resources(4, 4)],
            [3, 5],
        )
        self.assertEqual(summary["cnot_gate_budget"], 12)
        self.assertEqual(summary["iswap_gate_budget"], 40)
        self.assertEqual(summary["two_qubit_gate_budget"], 52)
        self.assertEqual(summary["max_two_qubit_depth"], 4)

    def test_unreachable_target_is_not_zero(self):
        summary = summarize_gate_resources([gpd_gate_resources(4, 4)], None)
        for key in ("cnot_gate_budget", "iswap_gate_budget", "two_qubit_gate_budget"):
            self.assertIsNone(summary[key])
        self.assertEqual(summary["max_two_qubit_depth"], 4)

    def test_no_executed_settings(self):
        summary = summarize_gate_resources([], [])
        self.assertEqual(summary["max_two_qubit_depth"], 0)
        self.assertEqual(summary["two_qubit_gate_budget"], 0)

    def test_invalid_allocation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one entry per setting"):
            summarize_gate_resources([gpd_gate_resources(4, 4)], [])
        for bad in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                summarize_gate_resources([gpd_gate_resources(4, 4)], [bad])

    def test_unknown_family_is_rejected(self):
        with self.assertRaises(KeyError):
            source_gate_resources("unknown", 4, 4)


if __name__ == "__main__":
    unittest.main()

