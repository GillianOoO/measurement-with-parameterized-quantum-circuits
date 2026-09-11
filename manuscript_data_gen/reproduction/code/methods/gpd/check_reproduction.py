"""Recompute the ten published calibrated GPD cases and compare Fig. 5 inputs."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pauli_product"))
from release_paths import DATA_ROOT, OUTPUT_ROOT, archived_path


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "gpd_validation")
    args = parser.parse_args()
    cases_root = DATA_ROOT / "random_sparse_dense" / "results" / "gpd_balanced"
    figure_rows = read_csv(DATA_ROOT / "random_sparse_dense" / "processed" / "review_figures" / "random_calibrated_gpd_empirical50_instances.csv")
    report = {"numpy": np.__version__, "cases": [], "max_nonempirical_delta": 0.0, "max_empirical_delta": 0.0, "max_figure5_delta": 0.0}
    for case_dir in sorted(cases_root.iterdir()):
        if not (case_dir / "manifest.json").is_file():
            continue
        manifest = json.loads((case_dir / "manifest.json").read_text())
        source = archived_path(manifest["source"]["manifest_path"], manifest["source"]["manifest_sha256"]).parent
        output = args.output / case_dir.name
        subprocess.run([sys.executable, str(HERE / "reselect_stabilizer_balanced.py"), "--source", str(source), "--output", str(output), "--empirical-repeats", "50", "--force"], check=True, stdout=subprocess.DEVNULL)
        old_rows = {int(row["shots"]): row for row in read_csv(case_dir / "selected_results.csv")}
        new_rows = {int(row["shots"]): row for row in read_csv(output / "selected_results.csv")}
        if old_rows.keys() != new_rows.keys():
            raise AssertionError(f"Shot grid mismatch: {case_dir.name}")
        nonempirical = empirical = figure_delta = 0.0
        for shots, old in old_rows.items():
            new = new_rows[shots]
            for key, value in old.items():
                if key in ("k", "allocation_json", "empirical_seed") and value != new[key]:
                    raise AssertionError(f"Selection/allocation/seed mismatch: {case_dir.name}, T={shots}, {key}")
                try:
                    delta = abs(float(value) - float(new[key]))
                except (ValueError, KeyError):
                    continue
                if key == "empirical_rmse":
                    empirical = max(empirical, delta)
                else:
                    nonempirical = max(nonempirical, delta)
            expected = next(row for row in figure_rows if row["slug"] == case_dir.name and int(row["shots"]) == shots)
            figure_delta = max(figure_delta, abs(float(new["empirical_rmse"]) - float(expected["empirical_rmse"])))
        report["cases"].append({"slug": case_dir.name, "rows": len(old_rows), "max_nonempirical_delta": nonempirical, "max_empirical_delta": empirical, "max_figure5_delta": figure_delta})
        report["max_nonempirical_delta"] = max(report["max_nonempirical_delta"], nonempirical)
        report["max_empirical_delta"] = max(report["max_empirical_delta"], empirical)
        report["max_figure5_delta"] = max(report["max_figure5_delta"], figure_delta)
        print(case_dir.name, report["cases"][-1], flush=True)
    if len(report["cases"]) != 10:
        raise AssertionError(f"Expected ten random cases, found {len(report['cases'])}")
    report["status"] = "pass" if max(report["max_nonempirical_delta"], report["max_figure5_delta"]) < 1e-12 else "numeric_mismatch"
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
