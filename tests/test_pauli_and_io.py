import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mpqc_measurement.api import run_srdd
from mpqc_measurement.io import load_hamiltonian, save_result
from mpqc_measurement.models import ElectronicIntegrals, HamiltonianData
from mpqc_measurement.pauli import dense_to_pauli, pauli_terms_to_dense
from mpqc_measurement.srdd import SRDDConfig, fit_srdd


class PauliAndIOTests(unittest.TestCase):
    def test_pauli_round_trip(self):
        terms = [("II", -0.2), ("XX", 0.3), ("YI", -0.1), ("IZ", 0.4)]
        matrix = pauli_terms_to_dense(2, terms)
        reconstructed = pauli_terms_to_dense(2, dense_to_pauli(matrix))
        np.testing.assert_allclose(reconstructed, matrix, atol=1.0e-12)

    def test_json_input_and_safe_output(self):
        payload = {
            "schema_version": 1,
            "n_qubits": 2,
            "qubit_order": "q0-most-significant",
            "terms": [
                {"pauli": "II", "coefficient": -0.2},
                {"pauli": "XX", "coefficient": 0.3},
                {"pauli": "IZ", "coefficient": 0.4},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "hamiltonian.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            data = load_hamiltonian(source)
            result = fit_srdd(data, 20, SRDDConfig(calibrate=False))
            result_path = save_result(result, root / "result")
            self.assertTrue(result_path.exists())
            stored = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["method"], "SRDD")
            self.assertEqual(sum(row["shots"] for row in stored["fragments"]), 20)
            with np.load(root / "result" / "approximation.npz", allow_pickle=False) as archive:
                np.testing.assert_allclose(
                    archive["approximate_hamiltonian"], result.approximate_hamiltonian
                )

    def test_pauli_json_version_order_and_coefficients_fail_closed(self):
        valid = {
            "schema_version": 1,
            "n_qubits": 1,
            "qubit_order": "q0-most-significant",
            "terms": [{"pauli": "Z", "coefficient": 1.0}],
        }
        invalid_payloads = []
        for key in ("schema_version", "qubit_order"):
            payload = dict(valid)
            del payload[key]
            invalid_payloads.append(payload)
        wrong_version = dict(valid)
        wrong_version["schema_version"] = 2
        invalid_payloads.append(wrong_version)
        wrong_order = dict(valid)
        wrong_order["qubit_order"] = "q0-least-significant"
        invalid_payloads.append(wrong_order)
        nonfinite = dict(valid)
        nonfinite["terms"] = [{"pauli": "Z", "coefficient": float("nan")}]
        invalid_payloads.append(nonfinite)
        fractional_particle_number = dict(valid)
        fractional_particle_number["particle_number"] = 0.5
        invalid_payloads.append(fractional_particle_number)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "hamiltonian.json"
            for index, payload in enumerate(invalid_payloads):
                with self.subTest(index=index):
                    source.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_hamiltonian(source)

    def test_electronic_integral_contract_and_particle_consistency(self):
        one = np.asarray([[-1.0, 0.1], [0.1, 0.5]])
        two = np.zeros((2, 2, 2, 2))
        electronic = ElectronicIntegrals(0.0, one, two, 2)
        data = HamiltonianData(np.eye(16), electronic=electronic)
        self.assertEqual(data.particle_number, 2)

        invalid_cases = [
            (0.0, one.astype(np.complex128), two, 2),
            (float("nan"), one, two, 2),
            (0.0, np.asarray([[1.0, 0.2], [0.0, 1.0]]), two, 2),
            (0.0, one, np.full_like(two, np.inf), 2),
            (0.0, one, two, 1),
        ]
        asymmetric_two = two.copy()
        asymmetric_two[0, 1, 0, 0] = 0.2
        invalid_cases.append((0.0, one, asymmetric_two, 2))
        for index, arguments in enumerate(invalid_cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                ElectronicIntegrals(*arguments)

        with self.assertRaisesRegex(ValueError, "must equal"):
            HamiltonianData(
                np.eye(16),
                particle_number=0,
                electronic=electronic,
            )

    def test_array_input_loads_state_path_and_rejects_fractional_shots(self):
        matrix = np.diag([1.0, -1.0])
        state = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "state.npy"
            np.save(state_path, state)
            result = run_srdd(
                matrix,
                20,
                state=state_path,
                config=SRDDConfig(calibrate=False),
            )
            np.save(state_path, np.asarray([np.nan, 1.0]))
            with self.assertRaisesRegex(ValueError, "finite"):
                run_srdd(
                    matrix,
                    20,
                    state=state_path,
                    config=SRDDConfig(calibrate=False),
                )
        self.assertIsNotNone(result.error.state_rmse)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_srdd(matrix, 1.9, config=SRDDConfig(calibrate=False))

    def test_save_result_refuses_a_nonempty_directory(self):
        result = fit_srdd(
            HamiltonianData(np.diag([1.0, -1.0])),
            10,
            SRDDConfig(calibrate=False),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_result(result, output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not overwrite")
            self.assertEqual([path.name for path in output.iterdir()], ["keep.txt"])

    def test_electronic_npz_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "electronic.npz"
            np.savez_compressed(
                source,
                constant=0.1,
                one_body=np.asarray([[-1.0, 0.08], [0.08, 0.45]]),
                two_body_chemist=np.zeros((2, 2, 2, 2)),
                n_electrons=2,
            )
            data = load_hamiltonian(source)
            self.assertEqual(data.source_format, "electronic-integrals-npz")
            self.assertEqual(data.n_qubits, 4)
            self.assertEqual(data.particle_number, 2)
            self.assertIsNotNone(data.electronic)
            self.assertAlmostEqual(data.electronic.constant, 0.1)


if __name__ == "__main__":
    unittest.main()
