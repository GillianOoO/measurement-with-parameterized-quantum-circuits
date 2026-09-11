"""Recompute Figure 2 summaries from the released molecular repeated estimates.

This verifies recorded outcomes, not a fresh optimizer or a new random draw.
It covers SRDD H4/H6 fixed geometries and the H4 scan, plus H4/H6 Pauli rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[2]


def main(data_root: Path, output: Path):
    data_root, output = data_root.resolve(), output.resolve()
    if output == data_root or output.is_relative_to(data_root):
        raise ValueError("Output must be outside the released source data")
    output.mkdir(parents=True, exist_ok=True)
    inputs = set()
    checks = []

    def read(relative):
        path = data_root / relative
        inputs.add(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def check(case, method, shots, repetitions, error_column, expected_count,
              summary_rmse, plot_rmse, source):
        assert len(repetitions) == expected_count, (case, method, shots, len(repetitions))
        assert sorted(int(row["repeat"]) for row in repetitions) == list(range(expected_count))
        errors = [float(row[error_column]) for row in repetitions]
        assert all(math.isfinite(error) for error in errors)
        rmse = math.sqrt(math.fsum(error*error for error in errors)/expected_count)
        summary_delta = abs(rmse-summary_rmse)
        plot_delta = abs(rmse-plot_rmse)
        if max(summary_delta, plot_delta) > 1e-12:
            raise RuntimeError(f"Recorded repetitions do not reproduce {case}/{method}/T={shots}")
        checks.append(dict(case=case, method=method, measurements=shots,
                           repetitions=expected_count, recomputed_RMSE=rmse,
                           summary_RMSE=summary_rmse, figure_input_RMSE=plot_rmse,
                           summary_absolute_difference=summary_delta,
                           figure_absolute_difference=plot_delta,
                           source_replicates=source))

    for molecule, subdirectory, plot_file in (
        ("H4", "srdd_fixed_selected", "fixed_geometry_method_curves.csv"),
        ("H6", "srdd_selected", "method_curves.csv"),
    ):
        directory = Path(molecule) / "results" / subdirectory
        source = directory / "sampling_replicates.csv"
        repetitions = read(source)
        summary = read(directory / "sampling_summary.csv")
        plotted = {int(row["T_total_shots"]): row for row in read(Path(molecule)/"processed"/plot_file)
                   if row["method"] == "standalone s-RCDF-F"}
        assert len(summary) == len(plotted) == 8
        for row in summary:
            shots = int(row["T_total_shots"])
            selected = [r for r in repetitions if int(r["T_total_shots"]) == shots]
            assert int(row["sampling_repeats"]) == 200
            check(molecule+" fixed", "SRDD", shots, selected, "signed_total_error", 200,
                  float(row["empirical_total_RMSE"]), float(plotted[shots]["empirical_total_RMSE"]), source.as_posix())

    directory = Path("H4/results/srdd_bond_scan_selected")
    summary = read(directory/"srcdf_sampling_summary.csv")
    plotted = {float(row["bond_length_angstrom"]): row for row in read("H4/processed/bond_scan_plot_data.csv")}
    assert len(summary) == len(plotted) == 21
    for row in summary:
        bond = float(row["bond_length_angstrom"])
        source = directory/f"R{bond:.1f}"/"sampling_replicates.csv"
        repetitions = read(source)
        assert int(row["sampling_repeats"]) == 50
        check(f"H4 R={bond:.1f}", "SRDD", 2038, repetitions, "signed_total_error", 50,
              float(row["empirical_total_RMSE"]), float(plotted[bond]["SRDD_empirical_RMSE"]), source.as_posix())

    # The current renderer reads these audited same-Hamiltonian Pauli summaries
    # directly. Older Pauli columns in the processed H4 scan table are unused.
    directory = Path("H4/results/pauli_bond_scan")
    for prefix, expected_count, expected_rows in (("scan", 50, 63), ("fixed", 200, 24)):
        source = directory/f"{prefix}_sampling_error_replicates.csv"
        repetitions = read(source)
        summary = read(directory/f"{prefix}_sampling_error_summary.csv")
        assert len(summary) == expected_rows
        for row in summary:
            shots, method, case = int(row["T_total_shots"]), row["method"], row["case"]
            selected = [r for r in repetitions if r["case"] == case and r["method"] == method
                        and int(r["T_total_shots"]) == shots]
            assert int(row["repeat_count"]) == expected_count
            target = float(row["empirical_total_RMSE_hartree"])
            check("H4 "+case, method, shots, selected, "total_error_about_exact_energy_hartree",
                  expected_count, target, target, source.as_posix())

    # Recover the H6 portion of the original v4 multi-molecule source without
    # rewriting its other rows; original committed file hashes remain valid.
    directory = Path("H6/results/pauli_replay")
    source = directory/"source_sampling_error_replicates.csv"
    summary_path = directory/"source_sampling_error_summary.csv"
    original_hashes = {
        source: "ce72d5e27773e1f32e3a768da24d36f4abf47984b172c98322672aa0a0aa1bba",
        summary_path: "978cb3193484c96e3072f38ad44c2acf2a4f921ec70dce292cfa248832714b12",
    }
    for relative, expected_hash in original_hashes.items():
        if hashlib.sha256((data_root/relative).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Original H6 aggregate source hash changed: {relative}")
    repetitions = [r for r in read(source) if r["molecule"] == "H6" and r["method"] in ("OGM", "SG", "Derand")]
    summary = [r for r in read(summary_path) if r["molecule"] == "H6" and r["method"] in ("OGM", "SG", "Derand")]
    plotted = {(r["method"], int(r["T_total_shots"])): r for r in read("H6/processed/method_curves.csv")
               if r["method"] in ("OGM", "SG", "Derand")}
    assert len(summary) == len(plotted) == 24 and len(repetitions) == 4800
    for row in summary:
        method, shots = row["method"], int(row["T_total_shots"])
        selected = [r for r in repetitions if r["method"] == method and int(r["T_total_shots"]) == shots]
        assert int(row["repeat_count"]) == 200
        check("H6 fixed", method, shots, selected, "total_error_about_exact_energy_hartree", 200,
              float(row["empirical_total_RMSE_hartree"]), float(plotted[(method, shots)]["empirical_total_RMSE"]), source.as_posix())

    csv_path = output/"figure2_saved_replicate_checks.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    srdd = [row for row in checks if row["method"] == "SRDD"]
    report = dict(status="PASS", scope="Reaggregation of released estimates; no new optimizer or random draws",
                  checked_points=len(checks), checked_repetitions=sum(row["repetitions"] for row in checks),
                  SRDD_points=len(srdd), SRDD_repetitions=sum(row["repetitions"] for row in srdd),
                  H4_Pauli_points=sum(row["method"] != "SRDD" and row["case"].startswith("H4") for row in checks),
                  H6_Pauli_points=sum(row["method"] != "SRDD" and row["case"].startswith("H6") for row in checks),
                  maximum_absolute_RMSE_difference=max(max(row["summary_absolute_difference"],
                                                            row["figure_absolute_difference"]) for row in checks),
                  not_checked=[],
                  inputs={path.relative_to(data_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(inputs)},
                  outputs={csv_path.name: hashlib.sha256(csv_path.read_bytes()).hexdigest()})
    (output/"molecular_replicate_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status","checked_points","checked_repetitions","SRDD_repetitions","maximum_absolute_RMSE_difference")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ARCHIVE/"data")
    parser.add_argument("--output", type=Path, default=ARCHIVE/"build/molecular_replicates")
    args = parser.parse_args()
    main(args.data_root, args.output)
