"""Contracts for the self-contained, lossless manuscript numerical release."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1] / "manuscript_data_gen/reproduction"


class NumericalReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "datasets/manifest.json").read_text())

    def test_every_member_is_safe_scientific_data(self):
        self.assertGreater(len(self.manifest["files"]), 2000)
        forbidden = {"before", "build", "qa", "logs", "__pycache__"}
        for path in self.manifest["files"]:
            parsed = PurePosixPath(path)
            self.assertTrue(path.startswith("data/"), path)
            self.assertFalse(parsed.is_absolute(), path)
            self.assertFalse(forbidden.intersection(parsed.parts), path)
            self.assertNotIn("..", parsed.parts, path)
            self.assertNotIn("\\", path)
            self.assertNotIn(parsed.suffix, {".tex", ".pdf", ".docx", ".pptx"}, path)

    def test_archives_have_exact_manifest_members_and_hashes(self):
        for name, record in self.manifest["archives"].items():
            path = ROOT / "datasets" / name
            self.assertLess(path.stat().st_size, 95 * 1024**2)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            members = {p: info for p, info in self.manifest["files"].items() if info["archive"] == name}
            with zipfile.ZipFile(path) as z:
                self.assertEqual(set(z.namelist()), set(members))
                for relative, item in members.items():
                    self.assertEqual(z.getinfo(relative).file_size, item["size"])

    def test_every_plot_data_dependency_is_released(self):
        required = json.loads((ROOT / "required_inputs.json").read_text())["files"]
        for relative, sha in required.items():
            if relative.startswith("data/"):
                self.assertEqual(self.manifest["files"][relative]["sha256"], sha)
            else:
                self.assertTrue((ROOT / relative).is_file(), relative)
        files = self.manifest["files"]
        self.assertIn("data/random_sparse_dense/inputs/hamiltonians/sparse_hamiltonian_4_1.txt", files)
        self.assertIn("data/H6/results/pauli_replay/source_sampling_error_replicates.csv", files)
        self.assertTrue(any("gpd_extended_sources" in p for p in files))


if __name__ == "__main__":
    unittest.main()
