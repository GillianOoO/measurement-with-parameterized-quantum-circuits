#!/usr/bin/env python3
"""Audit the released ten-instance prefix of a fresh expanded random ensemble.

Current calibrated GPD is compared with gpd_balanced, not the obsolete GPD
rows in random_all_methods_empirical50_instance.csv.  The latter file remains
the approved source for the five product-Pauli methods.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MEASUREMENT_DATA_ROOT", ROOT / "data"))
DEFAULT_RUN = Path(os.environ.get("PAPER_REPRO_RANDOM_ENSEMBLE_DIR", DATA_ROOT / "random_ensemble_50"))
METHODS = ("GPD", "OGM", "SG", "Derand", "LCS", "AP")
SHOTS = (12, 45, 160, 572, 2038, 7259, 25848)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_index(rows, fields):
    indexed = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in indexed:
            raise ValueError(f"Duplicate record key {key}")
        indexed[key] = row
    return indexed


def audit(data_root, run_root, tolerance):
    archive = data_root.resolve() / "random_sparse_dense"
    run_root = run_root.resolve()
    source_files = set()
    def read(path):
        source_files.add(path)
        return read_csv(path)

    old_pauli = unique_index(
        [row for row in read(archive / "results/random_all_methods_empirical50_instance.csv")
         if row["method"] != "GPD"], ("slug", "method", "shots"))
    old_pauli_reps = unique_index(
        [row for row in read(archive / "results/random_all_methods_empirical50_replicates.csv")
         if row["method"] != "GPD"], ("slug", "method", "shots", "repeat"))
    approved_summary = unique_index(
        read(archive / "processed/review_figures/random_calibrated_gpd_empirical50_summary.csv"),
        ("benchmark", "method", "shots"))
    metrics = ("empirical_mse", "empirical_rmse", "empirical_mean_error",
               "empirical_error_sample_sd", "analytic_mse", "analytic_rmse",
               "analytic_bias", "analytic_variance", "target_expectation")
    maxima = {method: {metric: 0.0 for metric in metrics} for method in METHODS}
    maxima["replicates"] = {"estimate": 0.0, "error": 0.0}
    rows_out, reps_out, identities, schedule_checks = [], [], [], []
    for regime in ("sparse", "dense"):
        for instance in range(1, 6):
            slug = f"{regime}4_seed0_{instance}"
            fresh_case = run_root / "cases" / slug
            frozen_gpd = archive / "results/gpd_balanced" / slug
            fresh_gpd = fresh_case / "gpd_selected"
            frozen_selected = unique_index(read(frozen_gpd / "selected_results.csv"), ("shots",))
            fresh_selected = unique_index(read(fresh_gpd / "selected_results.csv"), ("shots",))
            fresh_pauli = unique_index(read(fresh_case / "pauli/instance_results.csv"), ("method", "shots"))
            fresh_pauli_reps = unique_index(read(fresh_case / "pauli/replicates.csv"),
                                           ("method", "shots", "repeat"))
            old_h = archive / "inputs/hamiltonians" / f"{regime}_hamiltonian_4_{instance}.npy"
            new_h = run_root / "inputs/hamiltonians" / old_h.name
            old_state = archive / "inputs/common_input_state.npy"
            new_state = run_root / "inputs/common_input_state.npy"
            if digest(old_h) != digest(new_h) or digest(old_state) != digest(new_state):
                raise ValueError(f"Input bytes differ for {slug}")
            identities.append({"slug": slug, "hamiltonian_file_sha256": digest(new_h),
                               "state_file_sha256": digest(new_state), "identical_input_bytes": True})
            for method in ("OGM", "SG", "Derand", "AP"):
                archive_subdir = ("results/pauli_schedules" if method in ("OGM", "AP")
                                  else "provenance/original_schedule_cache")
                old_schedule = archive / archive_subdir / f"{slug}_{method}.npz"
                new_schedule = fresh_case / "pauli" / f"{method}_schedules.npz"
                source_files.update((old_schedule, new_schedule))
                with np.load(old_schedule, allow_pickle=False) as previous, np.load(new_schedule, allow_pickle=False) as current:
                    fields = [field for field in previous.files if field.startswith(("bases_", "counts_"))]
                    differing_fields = [field for field in fields if field not in current.files or
                                        not np.array_equal(previous[field], current[field])]
                    if not fields:
                        raise ValueError(f"Missing archived schedule arrays for {slug}/{method}")
                schedule_checks.append({"slug": slug, "method": method,
                                        "basis_and_count_arrays_exact": not differing_fields,
                                        "differing_fields": differing_fields,
                                        "array_count": len(fields), "released_file_sha256": digest(old_schedule),
                                        "fresh_file_sha256": digest(new_schedule)})
            for total in SHOTS:
                gpd_old_reps = unique_index(read(frozen_gpd / f"empirical_replicates_T{total}.csv"), ("repeat",))
                gpd_new_reps = unique_index(read(fresh_gpd / f"empirical_replicates_T{total}.csv"), ("repeat",))
                for method in METHODS:
                    if method == "GPD":
                        old = dict(frozen_selected[str(total),])
                        new = dict(fresh_selected[str(total),])
                        for item, records in ((old, gpd_old_reps), (new, gpd_new_reps)):
                            errors = np.asarray([float(records[(str(i),)]["error"]) for i in range(50)])
                            item.update(empirical_mse=float(np.mean(errors ** 2)),
                                        empirical_mean_error=float(np.mean(errors)),
                                        empirical_error_sample_sd=float(np.std(errors, ddof=1)),
                                        analytic_mse=float(item["mse"]),
                                        analytic_bias=float(item["bias"]),
                                        analytic_variance=float(item["variance"]))
                        if old["k"] != new["k"] or old["allocation_json"] != new["allocation_json"]:
                            raise ValueError(f"Calibrated selected design differs for {slug}/T={total}")
                        if new["frontier_right_censored"].lower() != "false":
                            raise ValueError(f"Calibrated frontier is censored for {slug}")
                    else:
                        old = old_pauli[slug, method, str(total)]
                        new = fresh_pauli[method, str(total)]
                        if old["target_hash"] != new["target_hash"] or old["state_hash"] != new["state_hash"]:
                            raise ValueError(f"Input array identity differs for {slug}/{method}")
                    differences = {metric: abs(float(old[metric]) - float(new[metric])) for metric in metrics}
                    for metric, value in differences.items():
                        maxima[method][metric] = max(maxima[method][metric], value)
                    rows_out.append({"benchmark": regime, "instance": instance, "slug": slug,
                                     "method": method, "shots": total,
                                     "fresh_empirical_rmse": float(new["empirical_rmse"]),
                                     "released_empirical_rmse": float(old["empirical_rmse"]),
                                     "maximum_absolute_difference": max(differences.values()),
                                     "absolute_differences": differences,
                                     "selected_k": int(new["k"]) if method == "GPD" else None})
                    for repeat in range(50):
                        if method == "GPD":
                            previous = gpd_old_reps[str(repeat),]
                            current = gpd_new_reps[str(repeat),]
                            seed_field = "empirical_seed"
                        else:
                            previous = old_pauli_reps[slug, method, str(total), str(repeat)]
                            current = fresh_pauli_reps[method, str(total), str(repeat)]
                            seed_field = "outcome_seed"
                            # Fresh NPZs add seed/optimizer metadata, so compare
                            # executed basis/count arrays above, not ZIP hashes.
                            for field in ("schedule_repeat",):
                                if str(previous[field]) != str(current[field]):
                                    raise ValueError(f"{field} differs at {slug}/{method}/{total}/{repeat}")
                        if str(previous[seed_field]) != str(current[seed_field]):
                            raise ValueError(f"Outcome seed differs at {slug}/{method}/{total}/{repeat}")
                        for field in ("estimate", "error"):
                            value = abs(float(previous[field]) - float(current[field]))
                            maxima["replicates"][field] = max(maxima["replicates"][field], value)
                        reps_out.append((slug, method, total, repeat))

    summary_rows = []
    for regime in ("sparse", "dense"):
        for method in METHODS:
            for total in SHOTS:
                selected = [row for row in rows_out if row["benchmark"] == regime and
                            row["method"] == method and row["shots"] == total]
                if len(selected) != 5:
                    raise ValueError("Incomplete legacy ensemble mean")
                current = float(np.mean([row["fresh_empirical_rmse"] for row in selected]))
                previous = float(approved_summary[regime, method, str(total)]["mean_instance_empirical_rmse"])
                summary_rows.append({"benchmark": regime, "method": method, "shots": total,
                                     "fresh_prefix_mean_rmse": current, "approved_figure_mean_rmse": previous,
                                     "absolute_difference": abs(current - previous)})
    numeric_maximum = max(value for values in maxima.values() for value in values.values())
    summary_maximum = max(row["absolute_difference"] for row in summary_rows)
    schedules_identical = all(row["basis_and_count_arrays_exact"] for row in schedule_checks)
    passed = numeric_maximum <= tolerance and summary_maximum <= tolerance and schedules_identical
    if len(rows_out) != 420 or len(reps_out) != 21000 or len(summary_rows) != 84:
        raise ValueError("Legacy-prefix audit has the wrong record count")
    return {"status": "PASS" if passed else "FAIL", "absolute_tolerance": tolerance,
            "scope": "Original five sparse and five dense Hamiltonians; six methods; seven measurement counts; 50 repetitions",
            "approved_GPD_source": "data/random_sparse_dense/results/gpd_balanced (calibrated GPD)",
            "approved_Pauli_source": "data/random_sparse_dense/results/random_all_methods_empirical50_instance.csv excluding historical GPD",
            "case_count": 10, "method_count": 6, "measurement_count_values": list(SHOTS),
            "instance_method_budget_rows": len(rows_out), "complete_estimates_compared": len(reps_out),
            "approved_figure_points_compared": len(summary_rows),
            "maximum_absolute_difference_by_method_metric": maxima,
            "maximum_approved_figure_mean_absolute_difference": summary_maximum,
            "empirical_rmse_exact_for_all_420_rows": all(row["absolute_differences"]["empirical_rmse"] == 0 for row in rows_out),
            "individual_estimates_exact_for_all_21000_repetitions": maxima["replicates"]["estimate"] == 0,
            "all_selected_GPD_K_and_allocations_identical": True,
            "all_Pauli_basis_and_count_arrays_identical": schedules_identical,
            "all_measurement_seeds_identical": True,
            "schedule_file_note": "Fresh NPZs contain additional seed/optimizer metadata; executed bases and counts are compared directly",
            "fresh_run_root": str(run_root), "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "generator": str(Path(__file__).resolve()), "generator_sha256": digest(Path(__file__).resolve()),
            "identical_input_audit": identities, "schedule_array_audit": schedule_checks,
            "rows": rows_out, "figure_prefix_means": summary_rows,
            "source_files_sha256": {str(path): digest(path) for path in sorted(source_files)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    result = audit(args.data_root, args.run_root, args.absolute_tolerance)
    output = (args.output or ROOT.parents[1] / "build/random_ensemble_50/legacy_prefix_reproduction.json").resolve()
    frozen = args.data_root.resolve()
    if output == frozen or output.is_relative_to(frozen):
        parser.error("Audit outputs must not overwrite frozen release data")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "instance_method_budget_rows",
          "complete_estimates_compared", "empirical_rmse_exact_for_all_420_rows",
          "individual_estimates_exact_for_all_21000_repetitions",
          "maximum_absolute_difference_by_method_metric",
          "maximum_approved_figure_mean_absolute_difference")}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
