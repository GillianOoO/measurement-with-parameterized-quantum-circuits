"""Fast contracts for the current figure configurations and bundled provenance."""

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "manuscript_data_gen"


class ManuscriptArtifactsTests(unittest.TestCase):
    def framework_module(self):
        path = ARTIFACTS / "reproduction/code/plotting/export_framework.py"
        spec = importlib.util.spec_from_file_location("framework_export", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_framework_source_and_reviewed_export(self):
        module = self.framework_module()
        config, record = module.verified_source()
        self.assertEqual(config["slide"], 1)
        self.assertEqual(record["slide"], 1)
        self.assertEqual(len(record["files"]), 3)
        left, bottom, right, top = config["crop_box_points"]
        width, height = config["slide_size_points"]
        self.assertTrue(0 <= left < right <= width)
        self.assertTrue(0 <= bottom < top <= height)

    def test_framework_copy_requires_no_numerical_inputs(self):
        module = self.framework_module()
        config, record = module.verified_source()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "figure1.pdf"
            module.copy_reviewed(output, output.with_suffix(".png"))
            self.assertEqual(module.sha256(output), record["files"][config["pdf"]])
            self.assertEqual(module.sha256(output.with_suffix(".png")), record["files"][config["png"]])

    def test_framework_rejects_stale_pptx(self):
        module = self.framework_module()
        config, record = module.verified_source()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in record["files"]:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(module.ROOT / relative, target)
            with (root / config["pptx"]).open("ab") as handle:
                handle.write(b"changed-source")
            with patch.object(module, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "source/export mismatch"):
                    module.copy_reviewed(root / "invalid.pdf")

    def test_bundle_hashes(self):
        spec = importlib.util.spec_from_file_location("artifact_build_all", ARTIFACTS / "build_all.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertGreater(module.verify_bundle(check_inputs=False), 3)

    def test_all_current_figures_use_canonical_renderers(self):
        configs = [json.loads(path.read_text()) for path in ARTIFACTS.glob("main/fig*/artifact.json")]
        configs += [json.loads((ARTIFACTS / "supp/fig_s1_state_dependent_variance/artifact.json").read_text())]
        self.assertEqual({row["figure_id"] for row in configs}, {"1", "2", "3", "4", "5", "s1"})
        self.assertTrue(all(row["kind"] == "canonical" for row in configs))

    def test_si_display_policy(self):
        config = json.loads((ARTIFACTS / "supp/fig_s1_state_dependent_variance/artifact.json").read_text())
        self.assertEqual(config["displayed_bias_rows"], 38)
        self.assertEqual(config["random_display_measurements"], [45, 160, 572, 2038, 7259, 25848])
        self.assertEqual(config["bias_axis"], "left, linear, zero at top; nonnegative magnitude labels")


if __name__ == "__main__":
    unittest.main()
