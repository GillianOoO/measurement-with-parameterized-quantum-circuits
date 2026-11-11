"""Fast contracts for the current figure configurations and bundled provenance."""

import importlib.util
import csv
import json
from pathlib import Path
import tempfile
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
        self.assertEqual(config["instances_per_regime"], 50)
        self.assertEqual(config["random_data_root"], "reproduction/data/random_ensemble_50")

    def test_current_random_dependencies(self):
        config = json.loads((ARTIFACTS / "main/fig05_random_hamiltonian_errors/artifact.json").read_text())
        self.assertEqual(config["instances_per_regime"], 50)
        self.assertEqual(config["data_root"], "reproduction/data/random_ensemble_50")
        inputs = json.loads((ARTIFACTS / "reproduction/required_inputs.json").read_text())["files"]
        self.assertIn("data/random_ensemble_50/results/instance_results.csv", inputs)
        self.assertFalse(any(path.startswith("data/random_sparse_dense/") for path in inputs))
        for regime in ("sparse", "dense"):
            for instance in range(1, 51):
                case = f"data/random_ensemble_50/cases/{regime}4_seed0_{instance}"
                self.assertIn(f"{case}/gpd_selected/selected_results.csv", inputs)
                self.assertIn(f"{case}/gpd_source/frontier_rows.csv", inputs)
                self.assertIn(f"{case}/pauli/AP_schedules.npz", inputs)

    def test_saved_setting_counts_and_fixed_tolerance(self):
        archive = ARTIFACTS / "reproduction"
        data = archive / "data"
        if not (data / "random_ensemble_50/results/manifest.json").is_file():
            self.skipTest("Run materialize_data.py for numerical table checks")
        spec = importlib.util.spec_from_file_location("setting_counts", archive / "code/analysis/rebuild_setting_counts.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        temporary_root = ROOT / "build"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            output = Path(directory)
            _, report = module.rebuild(data, output, [])
            with (output / "si_measurement_settings.csv").open(newline="") as stream:
                rows = {(r["benchmark"], r["method"]): float(r["mean_distinct_settings"]) for r in csv.DictReader(stream)}
            expected = {"sparse": (8.22, 9.50, 9.54, 14.24, 81.0, 39.4712),
                        "dense": (15.14, 42.74, 42.96, 50.94, 81.0, 80.9952)}
            for regime, values in expected.items():
                for method, value in zip(module.METHODS, values):
                    self.assertAlmostEqual(rows[regime, method], value, places=10)
            with (output / "fixed_frobenius_tolerance_summary.csv").open(newline="") as stream:
                thresholds = {r["benchmark"]: float(r["mean_measurement_settings"]) for r in csv.DictReader(stream)}
            self.assertEqual(thresholds, dict(H4=5.0, H6=4.0, BeH2=21.0, N2=37.0, sparse=4.9, dense=16.98))
            self.assertEqual(report["fixed_tolerance_instances"], 104)


if __name__ == "__main__":
    unittest.main()
