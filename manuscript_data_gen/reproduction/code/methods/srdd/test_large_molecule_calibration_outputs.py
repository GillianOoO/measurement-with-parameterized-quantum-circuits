"""Check the large-molecule calibration output contract without a tensor refit.

Execute the production function bodies with deterministic response fixtures to
avoid importing the molecular/MPS stack or writing any benchmark results.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "experiments" / "large_molecule_srdd_pauli.py"


def load_calibration_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"integer_allocation", "fit_weights"}
    ]
    if len(selected) != 2:
        raise AssertionError("Production calibration functions not found")
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    namespace["wedge_rotation"] = lambda rotation, configs: rotation
    namespace["state_expectation"] = lambda residual, supports: residual * supports
    namespace["setting_moments_for_state"] = lambda settings, wedges, record: (
        np.zeros(len(settings)),
        np.array([s["lambda"] ** 2 * record["variance_ratio"] for s in settings]),
    )
    return namespace


class LargeMoleculeCalibrationOutputTests(unittest.TestCase):
    def setUp(self):
        self.engine = load_calibration_functions()
        self.case = {"spec": SimpleNamespace(name="fixture"), "constant": 0.0}
        self.fits = [
            {
                "K": rank,
                "residual_terms": residual,
                "sector_F": residual,
                "settings": [{"lambda": scale, "rotation": np.eye(2)} for scale in scales],
                "collector_mode": "augment",
                "collector_extra_leaves": 1,
            }
            for rank, residual, scales in ((6, 2.0, [2.0, 3.0]), (40, 0.5, [4.0, 5.0]))
        ]
        self.catalogs = [
            {"group": i, "split": split, "supports": 0.25, "variance_ratio": 0.36}
            for i, split in enumerate(("train", "train", "validation", "test"))
        ]
        self.budgets = [10, 30, 100]

    def calibrate(self, catalogs):
        return self.engine["fit_weights"](self.case, self.fits, catalogs, None, self.budgets)

    def test_weights_labels_and_output_keys(self):
        weights, rows = self.calibrate(self.catalogs)
        self.assertEqual(set(weights), {
            "weight_approximation", "weight_sampling", "train_rows", "validation_rows", "test_rows",
        })
        self.assertAlmostEqual(weights["weight_approximation"], 0.25)
        self.assertAlmostEqual(weights["weight_sampling"], 0.6)
        self.assertEqual(len(rows), 24)
        self.assertEqual([weights[f"{split}_rows"] for split in ("train", "validation", "test")], [12, 6, 6])
        self.assertEqual({row["K"] for row in rows}, {6, 40})
        self.assertEqual({row["T"] for row in rows}, set(self.budgets))

    def test_heldout_responses_do_not_change_weights_or_candidate_losses(self):
        before, before_rows = self.calibrate(self.catalogs)
        changed = [dict(record) for record in self.catalogs]
        for record in changed:
            if record["split"] != "train":
                record.update(supports=0.95, variance_ratio=0.9)
        after, after_rows = self.calibrate(changed)
        self.assertEqual(before, after)
        self.assertNotEqual(before_rows, after_rows)
        for total in self.budgets:
            losses = []
            for weights in (before, after):
                candidates = []
                for fit in self.fits:
                    scales = np.array([s["lambda"] for s in fit["settings"]])
                    allocation = self.engine["integer_allocation"](total, scales)
                    self.assertEqual(int(allocation.sum()), total)
                    self.assertTrue(np.all(allocation >= 1))
                    sampling = math.sqrt(float(np.sum(scales ** 2 / allocation)))
                    candidates.append((math.hypot(
                        weights["weight_approximation"] * fit["sector_F"],
                        weights["weight_sampling"] * sampling,
                    ), fit["K"]))
                losses.append(candidates)
            self.assertEqual(losses[0], losses[1])
            self.assertEqual(min(losses[0]), min(losses[1]))


if __name__ == "__main__":
    unittest.main()
