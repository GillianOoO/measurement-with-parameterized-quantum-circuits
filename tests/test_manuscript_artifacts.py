"""Fast contracts for the current figure configurations and bundled provenance."""

import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "manuscript_data_gen"


class ManuscriptArtifactsTests(unittest.TestCase):
    def test_bundle_hashes(self):
        spec = importlib.util.spec_from_file_location("artifact_build_all", ARTIFACTS / "build_all.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertGreater(module.verify_bundle(check_inputs=False), 3)

    def test_all_current_figures_use_canonical_renderers(self):
        configs = [json.loads(path.read_text()) for path in ARTIFACTS.glob("main/fig*/artifact.json")]
        configs += [json.loads((ARTIFACTS / "supp/fig_s1_state_dependent_variance/artifact.json").read_text())]
        self.assertEqual({row["figure_id"] for row in configs}, {"2", "3", "4", "5", "s1"})
        self.assertTrue(all(row["kind"] == "canonical" for row in configs))

    def test_si_display_policy(self):
        config = json.loads((ARTIFACTS / "supp/fig_s1_state_dependent_variance/artifact.json").read_text())
        self.assertEqual(config["displayed_bias_rows"], 38)
        self.assertEqual(config["random_display_measurements"], [45, 160, 572, 2038, 7259, 25848])
        self.assertEqual(config["bias_axis"], "left, linear, zero at top; nonnegative magnitude labels")


if __name__ == "__main__":
    unittest.main()
