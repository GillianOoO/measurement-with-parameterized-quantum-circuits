"""Regression checks for native gate counts and executed measurement resources."""

import ast
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from native_gate_resources import (
    GATE_RESOURCE_CONVENTION, GateResources, gpd_gate_resources,
    source_gate_resources, summarize_gate_resources,
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


class ResourceSummaryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load reporting functions without importing the legacy archive runner,
        # creating caches or requiring its historical data/dependencies.
        source = Path(__file__).with_name("build_resource_statistics.py")
        parsed = ast.parse(source.read_text(encoding="utf-8"))
        names = {"add_gate_type_fields", "chart_summary", "write_gate_budget_table",
                 "format_integer"}
        nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in names]
        cls.reporting = {
            "Any": Any, "Path": Path, "HERE": source.parent,
            "GATE_RESOURCE_CONVENTION": GATE_RESOURCE_CONVENTION,
            "METHODS": ("AGPD", "SRDD"),
            "publication_method_target_is_omitted": lambda *_: False,
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
             cls.reporting)

    def native_row(self):
        return {
            "benchmark": "H4, R=1.20 A", "method": "AGPD",
            "reference_total_shots": 3000, "decomposition_items": 2,
            "shots_to_accuracy": 8, "status": "reported",
            **summarize_gate_resources(
                [GateResources(cnot_count=4, two_qubit_depth=2),
                 gpd_gate_resources(4, 4)], [3, 5]),
        }

    def test_chart_preserves_both_gate_types(self):
        chart = self.reporting["chart_summary"]([self.native_row()])[0]
        self.assertEqual(chart["two_qubit_gate_budget_to_error_0p01"], 52)
        self.assertEqual(chart["cnot_gate_budget_to_error_0p01"], 12)
        self.assertEqual(chart["iswap_gate_budget_to_error_0p01"], 40)
        self.assertEqual(chart["max_two_qubit_depth"], 4)

    def test_legacy_untyped_gpd_totals_are_rejected(self):
        row = {"method": "AGPD", "two_qubit_gate_budget": 16}
        with self.assertRaisesRegex(ValueError, "rebuilt with native iSWAP"):
            self.reporting["add_gate_type_fields"]([row])

    def test_cnot_only_geometry_rows_keep_their_counts(self):
        row = {"method": "SRDD", "two_qubit_gate_budget_to_error_0p01": 123}
        self.reporting["add_gate_type_fields"]([row], suffix="_to_error_0p01")
        self.assertEqual(row["cnot_gate_budget_to_error_0p01"], 123)
        self.assertEqual(row["iswap_gate_budget_to_error_0p01"], 0)

    def test_table_labels_mixed_gates_without_writing_tex(self):
        with patch.object(Path, "write_text") as write:
            self.reporting["write_gate_budget_table"]([self.native_row()])
        payload = write.call_args.args[0]
        self.assertIn("12 CNOT + 40 iSWAP", payload)


if __name__ == "__main__":
    unittest.main()
