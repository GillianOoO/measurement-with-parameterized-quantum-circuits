#!/usr/bin/env python3
"""Independently audit a completed random ensemble and render its mean errors.

The main and individual sparse/dense panels use the manuscript plotting
backend.  Error curves are arithmetic means of instance-level empirical
RMSEs.  Standard errors remain in the accompanying CSV, without error bands.
Exact variance and GPD approximation-bias tables are also exported for the
Supporting Information; this script does not alter shared plotters or TeX.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MEASUREMENT_DATA_ROOT", ROOT / "data"))
DEFAULT_RUN = Path(os.environ.get("PAPER_REPRO_RANDOM_ENSEMBLE_DIR", DATA_ROOT / "random_ensemble_50"))
sys.path.insert(0, str(ROOT / "code/plotting"))
import plotting_backend as base

np, plt = base.np, base.plt
METHODS = ("GPD", "OGM", "SG", "Derand", "LCS", "AP")
REGIMES = ("sparse", "dense")
SHOTS = (12, 45, 160, 572, 2038, 7259, 25848)
REPEATS = 50


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def close(observed: float, expected: float, context: str, *, atol: float = 3e-12) -> None:
    if not math.isfinite(observed) or not math.isfinite(expected):
        raise RuntimeError(f"Nonfinite value: {context}")
    if not math.isclose(observed, expected, rel_tol=1e-11, abs_tol=atol):
        raise RuntimeError(f"Independent check failed: {context}; {observed} != {expected}")


def check_hashes(folder: Path, records: dict[str, str]) -> None:
    if not records:
        raise RuntimeError(f"No recorded hashes: {folder}")
    for name, expected in records.items():
        path = folder / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Hash mismatch: {path}")


def point_key(row: dict) -> tuple[str, int, str, int]:
    return row["benchmark"], int(row["instance"]), row["method"], int(row["shots"])


def independently_check_stop(folder: Path, selected: list[dict], terminal: int) -> None:
    """Reconstruct the calibrated material-stall rule without selector imports."""
    manifest = read_json(folder / "manifest.json")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Incomplete calibrated GPD: {folder}")
    policy = manifest["selection"]
    if policy["right_censored"]:
        raise RuntimeError("Right-censored GPD trajectory")
    frontier = read_csv(folder / "frontier_rows.csv")
    by_k = defaultdict(list)
    by_t = defaultdict(list)
    for row in frontier:
        k, total = int(row["k"]), int(row["shots"])
        xa, xs = float(row["approximation_feature_xa"]), float(row["sampling_feature_xs"])
        wa, ws = float(row["weight_approximation_wa"]), float(row["weight_sampling_ws"])
        close(float(row["loss"]), math.hypot(wa*xa, ws*xs), "calibrated frontier loss")
        by_k[k].append(row)
        if k <= terminal:
            by_t[total].append(row)
    anchors, stalled, observed_stop = {}, 0, None
    relative = float(policy["relative_material_improvement"])
    patience = int(policy["stall_patience"])
    minimum = int(policy["minimum_k"])
    calibration_end = max(int(k) for k in manifest["calibration"]["K_fit"])
    for k in sorted(by_k):
        improved = False
        for row in by_k[k]:
            total, score = int(row["shots"]), float(row["loss"])
            if score < anchors.get(total, math.inf)*(1-relative):
                anchors[total] = score
                improved = True
        if k <= calibration_end:
            stalled = 0
        else:
            stalled = 0 if improved else stalled+1
        if k >= max(minimum, calibration_end+1) and stalled >= patience:
            observed_stop = k
            break
    if observed_stop != terminal:
        raise RuntimeError(f"Independent calibrated stopping mismatch: {folder}")
    if tuple(sorted(int(r["shots"]) for r in selected)) != SHOTS:
        raise RuntimeError("Incomplete selected GPD budget grid")
    for row in selected:
        best = min(by_t[int(row["shots"])], key=lambda r:(float(r["loss"]), int(r["k"])))
        if int(row["k"]) != int(best["k"]):
            raise RuntimeError("Selected GPD K is not the smallest loss minimizer")
        counts = json.loads(row["allocation_json"])
        k = int(row["k"])
        if k and (len(counts) != k or min(counts) < 1 or sum(counts) != int(row["shots"])):
            raise RuntimeError("Invalid selected GPD integer allocation")


def audit(run: Path) -> tuple[list[dict], list[dict], dict]:
    results = run / "results"
    result_manifest = read_json(results / "manifest.json")
    if result_manifest.get("status") != "COMPLETE":
        raise RuntimeError("The ensemble is not complete")
    count = int(result_manifest["instances_per_regime"])
    if count not in (20, 50):
        raise RuntimeError("Expected the recorded 50-or-20-instance timing decision")
    if tuple(result_manifest["methods"]) != METHODS or tuple(result_manifest["shots"]) != SHOTS:
        raise RuntimeError("The method/budget grid differs from the manuscript")
    if int(result_manifest["repeats_per_instance"]) != REPEATS:
        raise RuntimeError("The number of energy estimates per Hamiltonian changed")
    check_hashes(results, result_manifest["output_sha256"])
    for relative, key in (("inputs/manifest.json", "input_manifest_sha256"),
                          ("timing_decision.json", "timing_decision_sha256")):
        if sha256(run / relative) != result_manifest[key]:
            raise RuntimeError(f"Manifest input binding changed: {relative}")
    if int(read_json(run / "timing_decision.json")["selected_instances_per_regime"]) != count:
        raise RuntimeError("Timing decision and completed instance count disagree")
    inputs = read_json(run / "inputs/manifest.json")
    input_cases = {(r["benchmark"], int(r["instance"])):r for r in inputs["cases"]}
    instances = read_csv(results / "instance_results.csv")
    repetitions = read_csv(results / "replicates.csv")
    summaries = read_csv(results / "ensemble_summary.csv")
    expected_points = {(regime, i, method, total) for regime in REGIMES
                       for i in range(1, count+1) for method in METHODS for total in SHOTS}
    indexed = {point_key(row):row for row in instances}
    if len(indexed) != len(instances) or set(indexed) != expected_points:
        raise RuntimeError("Missing/duplicate instance-method-budget result")
    by_point = defaultdict(list)
    for row in repetitions:
        key = point_key(row)
        if key not in expected_points:
            raise RuntimeError("Unexpected replicate case")
        by_point[key].append(row)
    if set(by_point) != expected_points:
        raise RuntimeError("Incomplete replicate grid")
    target_spreads, stop_records = [], []
    for regime in REGIMES:
        for instance in range(1, count+1):
            source = input_cases[regime, instance]
            slug = source["slug"]
            h_path = run / "inputs/hamiltonians" / f"{regime}_hamiltonian_4_{instance}.npy"
            state_path = run / "inputs/common_input_state.npy"
            input_hash, state_hash = sha256(h_path), sha256(state_path)
            if input_hash != source["file_sha256"] or state_hash != source["state_file_sha256"]:
                raise RuntimeError(f"Input generation binding changed: {slug}")
            hamiltonian = np.asarray(np.load(h_path, allow_pickle=False), dtype=complex)
            density = np.asarray(np.load(state_path, allow_pickle=False), dtype=complex)
            if density.shape != hamiltonian.shape:
                raise RuntimeError("The common input must remain the saved density matrix")
            target = float(np.trace(density @ hamiltonian).real)
            folder = run / "cases" / slug
            completion = read_json(folder / "complete.json")
            if completion.get("status") != "COMPLETE":
                raise RuntimeError(f"Incomplete case: {slug}")
            check_hashes(folder, completion["result_sha256"])
            if completion["binding"]["input_file_sha256"] != input_hash or completion["binding"]["state_file_sha256"] != state_hash:
                raise RuntimeError("Completion input binding mismatch")
            selected = read_csv(folder / "gpd_selected/selected_results.csv")
            independently_check_stop(folder / "gpd_selected", selected, int(completion["effective_terminal_k"]))
            stop_records.append(dict(slug=slug, effective_terminal_k=int(completion["effective_terminal_k"]), status="PASS"))
            targets = []
            for method in METHODS:
                for total in SHOTS:
                    key = regime, instance, method, total
                    row, reps = indexed[key], by_point[key]
                    if row["input_file_sha256"] != input_hash or row["state_file_sha256"] != state_hash:
                        raise RuntimeError("Cross-method Hamiltonian/state pairing failed")
                    if len(reps) != REPEATS or sorted(int(r["repeat"]) for r in reps) != list(range(REPEATS)):
                        raise RuntimeError(f"Missing/duplicate estimators: {key}")
                    targets.append(float(row["target_expectation"]))
                    close(float(row["target_expectation"]), target, "exact input-state expectation")
                    errors = []
                    for rep in reps:
                        close(float(rep["error"]), float(rep["estimate"])-target, "estimator error")
                        close(float(rep["squared_error"]), float(rep["error"])**2, "squared estimator error")
                        if rep["input_file_sha256"] != input_hash or rep["state_file_sha256"] != state_hash:
                            raise RuntimeError("Replicate pairing check failed")
                        errors.append(float(rep["error"]))
                    mse = float(np.mean(np.square(errors)))
                    close(float(row["empirical_mse"]), mse, "instance empirical MSE")
                    close(float(row["empirical_rmse"]), math.sqrt(mse), "instance empirical RMSE")
                    bias = float(row["analytic_bias"])
                    bias_squared = float(row.get("analytic_bias_squared") or bias*bias)
                    variance = float(row["analytic_variance"])
                    if variance < 0 or bias_squared < 0:
                        raise RuntimeError("Negative analytic variance or squared bias")
                    close(float(row["analytic_mse"]), variance+bias_squared, "analytic MSE decomposition")
            target_spreads.append(max(targets)-min(targets))
    summary_index = {(r["benchmark"], r["method"], int(r["shots"])):r for r in summaries}
    if len(summaries) != len(REGIMES)*len(METHODS)*len(SHOTS) or len(summary_index) != len(summaries):
        raise RuntimeError("Missing/duplicate ensemble summaries")
    for regime in REGIMES:
        for method in METHODS:
            for total in SHOTS:
                row = summary_index[regime, method, total]
                members = [indexed[regime, i, method, total] for i in range(1, count+1)]
                values = np.asarray([float(r["empirical_rmse"]) for r in members])
                sd = float(np.std(values, ddof=1))
                if int(row["instances"]) != count or int(row["empirical_repeats_per_instance"]) != REPEATS:
                    raise RuntimeError("Incorrect summary sample sizes")
                close(float(row["mean_instance_empirical_rmse"]), float(np.mean(values)), "ensemble mean of instance RMSE")
                close(float(row["sd_instance_empirical_rmse"]), sd, "instance sample SD")
                close(float(row["se_instance_empirical_rmse"]), sd/math.sqrt(count), "instance standard error")
                close(float(row["mean_instance_analytic_rmse"]), float(np.mean([float(r["analytic_rmse"]) for r in members])), "mean analytic RMSE")
                if float(row["mean_instance_empirical_rmse"]) <= 0:
                    raise RuntimeError("Logarithmic error plot requires positive mean errors")
    report = dict(status="PASS", instances_per_regime=count,
                  independent_checks=["input and completed-output hashes", "complete six-method seven-budget grid",
                                      "50 unique estimators per point", "same H and density matrix for all methods",
                                      "estimator errors and empirical RMSE from raw estimates", "analytic variance-bias MSE",
                                      "arithmetic mean, sample SD and SE over Hamiltonian instances",
                                      "calibrated material-stall rule and smallest minimizing GPD K"],
                  instance_rows=len(instances), replicate_rows=len(repetitions), summary_rows=len(summaries),
                  maximum_cross_method_target_difference=max(target_spreads), calibrated_stops=stop_records)
    return summaries, instances, report


def panel(ax, rows: list[dict], benchmark: str, label: str) -> None:
    for method in METHODS:
        selected = sorted((r for r in rows if r["benchmark"] == benchmark and r["method"] == method),
                          key=lambda r:int(r["shots"]))
        base.line(ax, [int(r["shots"]) for r in selected],
                  [float(r["mean_instance_empirical_rmse"]) for r in selected], method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("The number of measurements")
    ax.set_title(f"{benchmark.capitalize()} four-qubit ensemble")
    ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.38)
    for spine in ax.spines.values():
        spine.set_visible(True)
    base.panel_label(ax, label)


def render(rows: list[dict], output: Path) -> list[Path]:
    base.configure()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), sharey=False)
    for ax, regime, label in zip(axes, REGIMES, ("(a)", "(b)")):
        panel(ax, rows, regime, label)
    axes[0].set_ylabel("Error")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.01),
               columnspacing=0.9, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.18, top=0.78, wspace=0.28)
    paths = base.save(fig, output, "general_hamiltonian_gpd_average_errors")
    plt.close(fig)
    for regime, label in zip(REGIMES, ("(a)", "(b)")):
        fig, ax = plt.subplots(figsize=(3.65, 2.75))
        panel(ax, rows, regime, label)
        ax.set_ylabel("Error")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.02),
                   columnspacing=0.9, frameon=False)
        fig.subplots_adjust(left=0.20, right=0.99, bottom=0.18, top=0.73)
        paths.extend(base.save(fig, output, f"general_hamiltonian_gpd_average_errors_{regime}"))
        plt.close(fig)
    return paths


def variance_exports(instances: list[dict], count: int) -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    for source in instances:
        bias = float(source["analytic_bias"])
        is_gpd = source["method"] == "GPD"
        rows.append(dict(benchmark=source["benchmark"], instance=int(source["instance"]),
                         slug=source["slug"], method=source["method"], shots=int(source["shots"]),
                         state_dependent_variance_hartree2=float(source["analytic_variance"]),
                         absolute_approximation_bias=abs(bias) if is_gpd else "",
                         squared_approximation_bias=bias*bias if is_gpd else "",
                         analytic_mean_estimator_bias=bias,
                         analytic_mean_squared_estimator_bias=float(source.get("analytic_bias_squared") or bias*bias),
                         selected_k=source.get("selected_k", ""),
                         input_file_sha256=source["input_file_sha256"],
                         state_file_sha256=source["state_file_sha256"]))
    for regime in REGIMES:
        for method in METHODS:
            for total in SHOTS:
                selected = [r for r in rows if r["benchmark"] == regime and r["method"] == method and r["shots"] == total]
                values = np.asarray([r["state_dependent_variance_hartree2"] for r in selected])
                row = dict(benchmark=regime, method=method, shots=total, instances=count,
                           mean_instance_state_dependent_variance_hartree2=float(np.mean(values)),
                           instance_variance_sample_sd_hartree2=float(np.std(values, ddof=1)),
                           zero_variance_instances=int(np.count_nonzero(values == 0)),
                           averaging_rule="arithmetic mean of exact instance-level conditional variances",
                           mean_instance_estimator_bias=float(np.mean([r["analytic_mean_estimator_bias"] for r in selected])),
                           mean_instance_squared_estimator_bias=float(np.mean([r["analytic_mean_squared_estimator_bias"] for r in selected])))
                if method == "GPD":
                    row.update(mean_instance_absolute_approximation_bias=float(np.mean([r["absolute_approximation_bias"] for r in selected])),
                               mean_instance_squared_approximation_bias=float(np.mean([r["squared_approximation_bias"] for r in selected])))
                summaries.append(row)
    return rows, summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN,
                        help="Materialized random_ensemble_50 directory or a completed fresh run")
    parser.add_argument("--output", type=Path,
                        default=ROOT.parents[1] / "build/random_ensemble_50",
                        help="New figures, data summaries and audit report")
    parser.add_argument("--audit-only", action="store_true",
                        help="Verify saved inputs and statistics without rendering figures")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    for frozen in ((ROOT / "data").resolve(), DATA_ROOT.resolve(), run):
        if output == frozen or output.is_relative_to(frozen):
            raise ValueError("Plot outputs must be outside input data and the frozen release tree")
    if not (run / "results/manifest.json").is_file():
        raise FileNotFoundError(
            f"Missing completed ensemble at {run}; run manuscript_data_gen/materialize_data.py "
            "or pass --run for a completed fresh experiment"
        )
    summaries, instances, report = audit(run)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "independent_audit.json", report)
    source_csv = output / "random_stabilizer_gpd_comparison.csv"
    write_csv(source_csv, summaries)
    variance_rows, variance_summary = variance_exports(instances, report["instances_per_regime"])
    variance_path = output / "random_state_variance_instances.csv"
    variance_summary_path = output / "random_state_variance_summary.csv"
    write_csv(variance_path, variance_rows)
    write_csv(variance_summary_path, variance_summary)
    files = [source_csv, variance_path, variance_summary_path, output / "independent_audit.json"]
    if not args.audit_only:
        files.extend(render(summaries, output))
    manifest = dict(status="COMPLETE", audit_only=args.audit_only, run=str(run), output=str(output),
                    instances_per_regime=report["instances_per_regime"], methods=METHODS,
                    shots=SHOTS, estimators_per_instance_budget=REPEATS,
                    line_statistic="arithmetic mean of instance-level empirical RMSE",
                    uncertainty="instance-level sample SD and SE in CSV; no error bands drawn",
                    style_source=str(Path(base.__file__).resolve()),
                    environment=dict(python=platform.python_version(), numpy=np.__version__,
                                     matplotlib=base.matplotlib.__version__),
                    sources_sha256={str(Path(__file__).resolve()):sha256(Path(__file__)),
                                    str(Path(base.__file__).resolve()):sha256(Path(base.__file__)),
                                    str(run / "results/manifest.json"):sha256(run / "results/manifest.json")},
                    output_sha256={path.name:sha256(path) for path in files})
    write_json(output / "plot_manifest.json", manifest)
    print(json.dumps(dict(status="COMPLETE", output=str(output), files=[path.name for path in files],
                          independent_audit=report["status"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
