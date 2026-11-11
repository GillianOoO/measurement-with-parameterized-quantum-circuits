"""Rebuild SI setting counts and fixed-Frobenius-tolerance diagnostics.

Only saved candidates and frozen schedules are read. Fixed-tolerance minima
are over the recorded candidate grids, not claims of global optimality.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

METHODS = ("GPD", "OGM", "SG", "Derand", "LCS", "AP")
TOLERANCE = 0.01


def rebuild(data_root: Path, output: Path, molecular_rows: list[dict]):
    inputs: set[Path] = set()

    def csv_rows(relative):
        path = data_root / relative
        inputs.add(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def save(name, rows):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    settings = [dict(benchmark=row["molecule"], method=row["method"], instances=1,
                     measurements=3000, mean_distinct_settings=row["distinct_settings"],
                     counting_rule="recorded molecular reference configuration")
                for row in molecular_rows]
    run = Path("random_ensemble_50")
    manifest_path = data_root / run / "results/manifest.json"
    inputs.add(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "COMPLETE" or int(manifest["instances_per_regime"]) != 50:
        raise ValueError("Current SI tables require the complete 50-instance ensembles")
    random_settings, tolerance_rows = [], []
    for benchmark in ("sparse", "dense"):
        for instance in range(1, 51):
            case = run / "cases" / f"{benchmark}4_seed0_{instance}"
            selected = [r for r in csv_rows(case / "gpd_selected/selected_results.csv")
                        if int(r["shots"]) == 25848]
            if len(selected) != 1:
                raise ValueError(f"Missing unique GPD reference setting: {case}")
            for method in METHODS:
                if method == "GPD":
                    count = int(selected[0]["k"])
                    rule = "selected GPD terms at T=25848"
                else:
                    path = data_root / case / "pauli" / f"{method}_schedules.npz"
                    inputs.add(path)
                    with np.load(path, allow_pickle=False) as payload:
                        if method in ("OGM", "LCS"):
                            count = len(np.unique(payload["settings"], axis=0))
                            rule = "constructed OGM groups" if method == "OGM" else "full LCS support"
                        elif method == "AP":
                            count = float(np.mean([len(np.unique(payload[f"bases_r{j:02d}_T25848"], axis=0))
                                                   for j in range(50)]))
                            rule = "mean distinct bases across 50 saved AP schedules at T=25848"
                        else:
                            count = len(np.unique(payload["bases_25848"], axis=0))
                            rule = "distinct bases in the saved T=25848 schedule"
                random_settings.append(dict(benchmark=benchmark, instance=instance, method=method,
                                            measurements=25848, distinct_settings=count, counting_rule=rule))
            frontier = csv_rows(case / "gpd_source/frontier_rows.csv")
            residuals = {}
            for row in frontier:
                k, residual = int(row["k"]), float(row["residual_frobenius"])
                if k in residuals and not math.isclose(residuals[k], residual, rel_tol=1e-12, abs_tol=1e-14):
                    raise ValueError(f"Budget-dependent operator residual: {case}/K{k}")
                residuals[k] = residual
            eligible = [k for k, residual in residuals.items() if residual <= TOLERANCE]
            if not eligible:
                raise ValueError(f"No recorded candidate satisfies the fixed tolerance: {case}")
            k = min(eligible)
            tolerance_rows.append(dict(benchmark=benchmark, instance=instance, source_terms=k,
                                       measurement_settings=k, source_full_frobenius=residuals[k],
                                       completion_full_frobenius_bound=0.0,
                                       total_full_frobenius_upper_bound=residuals[k], tolerance=TOLERANCE))
        for method in METHODS:
            rows = [r for r in random_settings if r["benchmark"] == benchmark and r["method"] == method]
            settings.append(dict(benchmark=benchmark, method=method, instances=50, measurements=25848,
                                 mean_distinct_settings=float(np.mean([r["distinct_settings"] for r in rows])),
                                 counting_rule=rows[0]["counting_rule"]))

    for molecule, folder in (("H4", "H4/results/srdd_fixed_selected"),
                             ("H6", "H6/results/srdd_selected")):
        candidates = []
        for row in csv_rows(Path(folder) / "rank_selected.csv"):
            residual = float(row["full_Frobenius_distance"])
            completion = float(row["F3_reconstruction_error"])
            if residual + completion <= TOLERANCE:
                candidates.append(dict(benchmark=molecule, instance=1, source_terms=int(row["K_source_terms"]),
                                       measurement_settings=int(row["measurement_settings"]),
                                       source_full_frobenius=residual, completion_full_frobenius_bound=completion,
                                       total_full_frobenius_upper_bound=residual+completion, tolerance=TOLERANCE))
        if not candidates:
            raise ValueError(f"No fixed-tolerance candidate for {molecule}")
        tolerance_rows.append(min(candidates, key=lambda r: (r["measurement_settings"], r["source_terms"])))

    for molecule, folder in (("BeH2", "BeH2/results/srdd_and_pauli/BeH2"),
                             ("N2", "N2/results/srdd_and_pauli")):
        ranks = csv_rows(Path(folder) / "srcdf_rank_depth_trials.csv")
        collectors = csv_rows(Path(folder) / "collector_redistribution_trials.csv")
        integral_path = data_root / molecule / "inputs/integrals_mo.npz"
        inputs.add(integral_path)
        with np.load(integral_path, allow_pickle=False) as payload:
            m = payload["one_body_integrals"].shape[0]
        candidates = []
        for row in ranks:
            if row["selected_depth"].lower() != "true":
                continue
            for collector in collectors:
                if (collector["feasible"].lower() != "true" or
                    int(collector["K"]) != int(row["K"]) or int(collector["depth"]) != int(row["depth"])):
                    continue
                residual = float(row["full_Frobenius_distance"])
                delta = float(collector["matrix_reconstruction_residual_frobenius"])
                # For two spin registers: ||dGamma(delta C)||_F^2 =
                # 4^m (||delta C||_F^2/2 + (tr delta C)^2).
                # Cauchy-Schwarz bounds |tr delta C| <= sqrt(m)||delta C||_F.
                completion = math.sqrt(4**m * (m + 0.5)) * delta
                if residual + completion <= TOLERANCE:
                    candidates.append(dict(benchmark=molecule, instance=1, source_terms=int(row["K"]),
                                           measurement_settings=int(collector["measurement_setting_count"]),
                                           source_full_frobenius=residual, completion_full_frobenius_bound=completion,
                                           total_full_frobenius_upper_bound=residual+completion, tolerance=TOLERANCE))
        if not candidates:
            raise ValueError(f"No fixed-tolerance candidate for {molecule}")
        tolerance_rows.append(min(candidates, key=lambda r: (r["measurement_settings"], r["source_terms"])))
    tolerance_summary = []
    for benchmark in ("H4", "H6", "BeH2", "N2", "sparse", "dense"):
        rows = [row for row in tolerance_rows if row["benchmark"] == benchmark]
        tolerance_summary.append(dict(benchmark=benchmark, instances=len(rows), tolerance=TOLERANCE,
                                      mean_measurement_settings=float(np.mean([r["measurement_settings"] for r in rows])),
                                      maximum_full_frobenius_upper_bound=max(r["total_full_frobenius_upper_bound"] for r in rows)))
    save("si_measurement_settings.csv", settings)
    save("random_measurement_setting_instances.csv", random_settings)
    save("fixed_frobenius_tolerance_instances.csv", tolerance_rows)
    save("fixed_frobenius_tolerance_summary.csv", tolerance_summary)
    methods = ("SRDD", "GPD", "FC-IMA", "OGM", "SG", "Derand", "LCS", "AP")
    table = {(r["benchmark"], r["method"]): r for r in settings}
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\hline", "Benchmark & " + " & ".join(methods) + r"\\", r"\hline"]
    for benchmark in ("H4", "H6", "BeH2", "N2", "sparse", "dense"):
        values = []
        for method in methods:
            row = table.get((benchmark, method))
            values.append((f"{row['mean_distinct_settings']:.2f}" if row["instances"] == 50
                           else str(int(row["mean_distinct_settings"]))) if row else "--")
        lines.append(benchmark + " & " + " & ".join(values) + r"\\")
    lines += [r"\hline", r"\end{tabular}"]
    (output / "si_measurement_settings.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inputs, dict(setting_count_entries=len(settings), fixed_tolerance_instances=len(tolerance_rows),
                        fixed_tolerance_scope="minimum over recorded source candidates plus completion residual bounds")
