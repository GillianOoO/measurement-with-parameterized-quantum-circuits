"""Rebuild current main/SI numerical tables from released numerical records.

This verifies saved exact variances, biases and compiled circuits; it does not
rerun the molecular electronic-structure calculation or fit a new decomposition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[2]
MOLECULES = ("H4", "H6", "BeH2", "N2")
METHODS = ("SRDD", "FC-IMA", "OGM", "SG", "Derand")
LABELS = {"H4": r"H$_4$", "H6": r"H$_6$", "BeH2": r"BeH$_2$", "N2": r"N$_2$"}


def main(data_root: Path, output: Path) -> dict:
    data_root, output = data_root.resolve(), output.resolve()
    if output == data_root or output.is_relative_to(data_root):
        raise ValueError("Output must be outside the released source data")
    output.mkdir(parents=True, exist_ok=True)
    inputs: set[Path] = set()

    def read_csv(relative):
        path = data_root / relative
        inputs.add(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def read_json(relative):
        path = data_root / relative
        inputs.add(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    variances = read_csv("shared_processed/variance/state_dependent_variance_T3000.csv")
    resources = read_csv("shared_processed/resources/resource_chart_summary.csv")
    reference = {(row["plot_label"], row["method"]): row for row in resources}
    variance_map = {(row["molecule"], row["method"]): row for row in variances}
    fcroot = Path("fully_commuting/BeH2_N2_FC_IMA")
    rows = []
    for molecule in MOLECULES:
        for method in METHODS:
            if method == "FC-IMA" and molecule in ("H4", "H6"):
                continue
            resource = reference[(molecule, method)]
            settings = int(float(resource["decomposition_items"]))
            if method == "FC-IMA":
                summary = read_json(fcroot / "error_eval/results" / molecule / "summary.json")
                coefficient = float(summary["inverse_error_frozen_variance_coefficient_T0_times_variance_hartree2_shots"])
                variance = float(summary["inverse_error_frozen_T0_variance_hartree2"])
                bias = 0.0
                assert settings == int(summary["fc_overlapping_group_count"])
            else:
                source = variance_map[(molecule, method)]
                coefficient = float(source["T_times_fixed_variance_hartree2"])
                variance = float(source["estimator_variance_hartree2"])
                bias = float(source["signed_approximation_bias_hartree"])
                assert int(source["T_total_shots"]) == 3000
            assert math.isclose(coefficient, 3000 * variance, rel_tol=1e-12, abs_tol=1e-12)
            assert abs(bias) < 0.01
            required = math.ceil(max(settings, coefficient / (0.01**2 - bias**2)))
            assert required == int(resource["shots_to_error_0p01"]), (molecule, method, required)
            rows.append(dict(molecule=molecule, method=method, reference_measurements=3000,
                             sampling_variance=variance, signed_bias=bias,
                             variance_coefficient=coefficient, distinct_settings=settings,
                             projected_measurements_RMSE_0p01=required))

    circuit_rows = []
    for molecule in ("BeH2", "N2"):
        frozen = data_root / "shared_processed/figure3/empirical50" / f"{molecule}_settings.npz"
        inputs.add(frozen)
        with np.load(frozen, allow_pickle=False) as payload:
            m = int(payload["m"])
            layout = payload["layout"]
            shots = payload["shots"]
            assert np.all(shots > 0) and int(shots.sum()) == 3000
            assert payload["angles"].shape == (len(shots), len(layout))
            assert len(layout) % (m - 1) == 0
            orbital_depth = len(layout) // (m - 1)
            # Two spin registers, two compiled CNOTs per spatial Givens.
            srdd_count, srdd_depth = 4 * len(layout), 4 * orbital_depth
        assert len(shots) == int(float(reference[(molecule, "SRDD")]["decomposition_items"]))
        assert srdd_depth == int(float(reference[(molecule, "SRDD")]["max_two_qubit_depth"]))
        circuit_rows.append(dict(molecule=molecule, method="SRDD", distinct_settings=len(shots),
                                 max_compiled_CNOT_count=srdd_count, max_compiled_CNOT_depth=srdd_depth))
        audit = read_csv(fcroot / "noise_eval/results" / molecule / "circuit_audit.csv")
        assert sum(int(row["allocated_shots_T3000"]) for row in audit) == 3000
        active = [row for row in audit if int(row["allocated_shots_T3000"]) > 0]
        fc_count = max(int(row["compiled_cx_count"]) for row in active)
        fc_depth = max(int(row["compiled_two_qubit_depth"]) for row in active)
        assert len(active) == int(float(reference[(molecule, "FC-IMA")]["decomposition_items"]))
        assert fc_depth == int(float(reference[(molecule, "FC-IMA")]["max_two_qubit_depth"]))
        circuit_rows.append(dict(molecule=molecule, method="FC-IMA", distinct_settings=len(active),
                                 max_compiled_CNOT_count=fc_count, max_compiled_CNOT_depth=fc_depth))

    def save_csv(name, values):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    save_csv("main_resource_and_si_accuracy.csv", rows)
    save_csv("main_circuit_resources.csv", circuit_rows)
    table = {(r["molecule"], r["method"]): r for r in rows}
    # The figure uses continuous projections; the SI table above rounds up.
    extended = read_csv("shared_processed/figure3/fig3_extended_curves.csv")
    inverse_rows = []
    for molecule in ("BeH2", "N2"):
        for method in METHODS:
            source = sorted((row for row in extended if row["molecule"] == molecule
                             and row["method"] == method
                             and row["curve"] == "estimated_measurements_vs_error"),
                            key=lambda row: float(row["x"]))
            grid = np.geomspace(0.01, 0.5, 120 if method == "FC-IMA" else 121)
            assert len(source) == len(grid)
            record = table[(molecule, method)]
            for target, expected in zip(grid, source):
                assert math.isclose(float(target), float(expected["x"]), rel_tol=1e-12)
                continuous = max(record["distinct_settings"], record["variance_coefficient"] /
                                 (float(target)**2 - record["signed_bias"]**2))
                plotted = min(continuous, 1e8)
                assert math.isclose(plotted, float(expected["y"]), rel_tol=1e-11, abs_tol=1e-10)
                inverse_rows.append(dict(molecule=molecule, method=method,
                                         target_RMSE=float(target), projected_measurements=plotted))
    save_csv("fig3_inverse_accuracy.csv", inverse_rows)
    lines = [r"\begin{tabular}{lccccc}", r"\hline",
             r"Benchmark & SRDD & FC-IMA & OGM & SG & Derand\\", r"\hline"]
    for molecule in MOLECULES:
        values = [f"{table[(molecule, method)]['projected_measurements_RMSE_0p01']:,}"
                  if (molecule, method) in table else "n.a." for method in METHODS]
        lines.append(LABELS[molecule] + " & " + " & ".join(values) + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    (output / "si_required_measurements.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    circuits = {(r["molecule"], r["method"]): r for r in circuit_rows}
    lines = [r"\begin{tabular}{llcc}", r"\hline", r"Quantity & Method & BeH$_2$ & N$_2$\\", r"\hline"]
    for quantity, key in (("Distinct measurement settings", "distinct_settings"),
                          ("CNOT count", "max_compiled_CNOT_count"),
                          ("CNOT depth", "max_compiled_CNOT_depth")):
        for method in ("SRDD", "FC-IMA"):
            values = [str(circuits[(molecule, method)][key]) for molecule in ("BeH2", "N2")]
            lines.append(quantity + " & " + method + " & " + " & ".join(values) + r"\\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (output / "main_circuit_resources.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = dict(status="PASS", scope="Rebuilt from released exact variance, bias, frozen circuit and allocation records",
                  projected_measurement_entries=len(rows), circuit_resource_rows=len(circuit_rows),
                  inverse_accuracy_points=len(inverse_rows),
                  simulation_refitted=False,
                  inputs={path.relative_to(data_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(inputs)},
                  outputs={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(output.iterdir()) if path.suffix in (".csv", ".tex")})
    (output / "table_reproduction_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "projected_measurement_entries", "circuit_resource_rows")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ARCHIVE / "data")
    parser.add_argument("--output", type=Path, default=ARCHIVE / "build" / "manuscript_tables")
    args = parser.parse_args()
    main(args.data_root, args.output)
