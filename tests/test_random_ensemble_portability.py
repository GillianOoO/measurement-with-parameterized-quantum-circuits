"""Input-stream identity and portable process boundaries for the 100-case release."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "manuscript_data_gen/reproduction/code/experiments"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RandomEnsemblePortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load("random_ensemble_inputs")
        cls.driver = load("run_random_ensemble")

    def test_generator_matches_all_published_hamiltonians_and_raw_matrices(self):
        data = ROOT / "manuscript_data_gen/reproduction/data/random_ensemble_50/inputs"
        if not (data / "manifest.json").is_file():
            self.skipTest("Run manuscript_data_gen/materialize_data.py first")
        state, matrices = self.inputs.reconstruct_stream(50)
        np.testing.assert_allclose(
            np.outer(state, state.conj()), np.load(data / "common_input_state.npy"),
            rtol=0, atol=1e-14,
        )
        self.assertEqual(len(matrices), 100)
        for (regime, instance), raw in matrices.items():
            filename = f"{regime}_hamiltonian_4_{instance}.npy"
            np.testing.assert_array_equal(raw, np.load(data / "generation_matrices" / filename))
            np.testing.assert_array_equal((raw + raw.T) / 2, np.load(data / "hamiltonians" / filename))
            if regime == "sparse":
                self.assertEqual(np.count_nonzero(raw), 4)

    def test_relative_interpreters_survive_subprocess_working_directory_change(self):
        # No executable needs to exist: parsing resolves paths before the child
        # process switches to the reproduction working directory.
        folder = EXPERIMENTS
        with contextlib.chdir(folder):
            args = self.driver.parse_args([
                "pilot", "--gpd-python", "gpd/bin/python", "--pauli-python",
                "pauli/bin/python", "--gpd-runtime", "packages",
            ])
            self.assertEqual(args.gpd_python, Path(folder) / "gpd/bin/python")
            self.assertEqual(args.pauli_python, Path(folder) / "pauli/bin/python")
            self.assertEqual(args.gpd_runtime, [Path(folder) / "packages"])

    def test_virtual_environments_do_not_inherit_unrelated_pythonpath(self):
        args = self.driver.parse_args(["pilot"])
        self.assertEqual(args.gpd_python, Path(sys.executable).resolve())
        self.assertEqual(args.gpd_runtime, [])
        with patch.dict(os.environ, {"PYTHONPATH": "unrelated-packages", "OMP_NUM_THREADS": "7"}):
            gpd, pauli = self.driver.environments(args)
        self.assertNotIn("PYTHONPATH", gpd)
        self.assertNotIn("PYTHONPATH", pauli)
        self.assertEqual(gpd["OMP_NUM_THREADS"], "1")
        self.assertNotIn("OMP_NUM_THREADS", pauli)

    def test_driver_rejects_writing_into_packaged_data(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.driver.parse_args([
                "aggregate", "--output",
                str(self.driver.ROOT / "data/random_ensemble_50"),
            ])


if __name__ == "__main__":
    unittest.main()
