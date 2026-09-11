"""Fresh Born replay of the five current random-Hamiltonian Pauli baselines.

The published input states, Hamiltonians and fixed schedules are retained.
GPD is verified separately by methods/gpd/check_reproduction.py. This entrypoint
does not run the historical all-method main or its obsolete structured case.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

ARCHIVE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARCHIVE / "code/methods/pauli_product"))
import run_general_hamiltonian_revision as general
import replay_random_all_methods_empirical50 as sampling

METHODS = ("OGM", "SG", "Derand", "LCS", "AP")
CASES = tuple(f"{regime}4_seed0_{instance}" for regime in ("sparse", "dense") for instance in range(1, 6))


def main(data_root: Path, output: Path, cases: list[str]):
    data_root, output = data_root.resolve(), output.resolve()
    if output == data_root or output.is_relative_to(data_root):
        raise ValueError("Output must be outside the released source data")
    output.mkdir(parents=True, exist_ok=True)
    root = data_root / "random_sparse_dense"
    inputs = set()

    def read(relative):
        path = root / relative
        inputs.add(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    source_rows = read("results/random_all_methods_empirical50_replicates.csv")
    expected = {(r["slug"], r["method"], int(r["shots"]), int(r["repeat"])): r for r in source_rows}
    expected_instances = {(r["slug"], r["method"], int(r["shots"])): r
                          for r in read("results/random_all_methods_empirical50_instance.csv")}
    figure_rows = read("processed/review_figures/random_calibrated_gpd_empirical50_summary.csv")
    state_path = root / "inputs/common_input_state.npy"
    inputs.add(state_path)
    common_state = general.pure_state_from_saved(state_path)
    replicated, instances, checks = [], [], []
    started = time.perf_counter()
    for slug in cases:
        regime, instance = slug.split("4_seed0_")
        h_path = root / "inputs/hamiltonians" / f"{regime}_hamiltonian_4_{instance}.npy"
        p_path = h_path.with_suffix(".txt")
        inputs.update((h_path, p_path))
        hamiltonian = np.asarray(np.load(h_path, allow_pickle=False), dtype=complex)
        observables, weights = general.load_pauli_table(p_path)
        case = general.BenchmarkCase(regime, slug, int(instance), slug, hamiltonian,
                                     common_state.copy(), observables, weights,
                                     sampling.SHOT_BUDGETS, (h_path, p_path, state_path), {})
        observables, weights, offset = general.split_identity(case.observables, case.weights)
        evaluator = general.StateEvaluator(case.state)
        _, joint, scores = sampling.lcs_joint_law(general, evaluator, observables, weights)
        lcs_mean = float(offset + joint @ scores)
        lcs_coefficient = float(joint @ np.square(scores) - (joint @ scores)**2)
        if not math.isclose(lcs_mean, case.target_expectation, rel_tol=0., abs_tol=3e-12):
            raise RuntimeError(f"{slug}: LCS target mismatch")
        schedules = {}
        for method in ("OGM", "SG", "Derand", "AP"):
            subfolder = "results/pauli_schedules" if method in ("OGM", "AP") else "provenance/original_schedule_cache"
            path = root / subfolder / f"{slug}_{method}.npz"
            inputs.add(path)
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected[(slug, method, 12, 0)]["schedule_sha256"]:
                raise RuntimeError(f"{slug}/{method}: frozen schedule hash mismatch")
            with np.load(path, allow_pickle=False) as payload:
                schedules[method] = {key: payload[key].copy() for key in payload.files if key.startswith(("bases_", "counts_"))}
        max_estimate_delta, max_variance_delta = 0., 0.
        for method in METHODS:
            for shots in sampling.SHOT_BUDGETS:
                values, variances, biases = [], [], []
                if method not in ("LCS", "AP"):
                    bases = schedules[method][f"bases_{shots}"]
                    counts = schedules[method][f"counts_{shots}"]
                    exact = general.fixed_schedule_mse(evaluator, observables, weights, offset,
                                                       bases, counts, case.target_expectation)
                for repeat in range(50):
                    seed = sampling.stable_seed(sampling.PROFILE, sampling.BASE_SEED, slug, method, shots, repeat)
                    if method == "LCS":
                        value = sampling.lcs_replay(joint, scores, offset, shots, seed)
                        variance, bias = lcs_coefficient / shots, 0.
                    else:
                        if method == "AP":
                            bases = schedules[method][f"bases_r{repeat:02d}_T{shots}"]
                            counts = schedules[method][f"counts_r{repeat:02d}_T{shots}"]
                            exact = general.fixed_schedule_mse(evaluator, observables, weights, offset,
                                                               bases, counts, case.target_expectation)
                        value = sampling.fixed_schedule_replay(general, evaluator, observables, weights,
                                                               offset, bases, counts, seed)
                        variance, bias = float(exact["variance"]), float(exact["bias"])
                    reference = expected[(slug, method, shots, repeat)]
                    if seed != int(reference["outcome_seed"]):
                        raise RuntimeError("Published outcome seed changed")
                    delta = abs(value - float(reference["estimate"]))
                    max_estimate_delta = max(max_estimate_delta, delta)
                    values.append(value)
                    variances.append(variance)
                    biases.append(bias)
                    replicated.append(dict(benchmark=regime, instance=int(instance), slug=slug, method=method,
                                           shots=shots, repeat=repeat, outcome_seed=seed, estimate=value,
                                           error=value-case.target_expectation,
                                           reference_estimate=float(reference["estimate"]), absolute_difference=delta))
                errors = np.asarray(values) - case.target_expectation
                rmse = float(np.sqrt(np.mean(errors**2)))
                variance = float(np.mean(variances))
                reference = expected_instances[(slug, method, shots)]
                delta = abs(variance - float(reference["analytic_variance"]))
                max_variance_delta = max(max_variance_delta, delta)
                if not math.isclose(rmse, float(reference["empirical_rmse"]), rel_tol=0., abs_tol=1e-11):
                    raise RuntimeError(f"{slug}/{method}/T={shots}: empirical RMSE changed")
                instances.append(dict(benchmark=regime, instance=int(instance), slug=slug, method=method,
                                      shots=shots, empirical_repeats=50, empirical_rmse=rmse,
                                      analytic_variance=variance, analytic_mean_bias=float(np.mean(biases))))
        checks.append(dict(slug=slug, maximum_estimate_difference=max_estimate_delta,
                           maximum_analytic_variance_difference=max_variance_delta,
                           matches_within_tolerance=max_estimate_delta<=1e-11 and max_variance_delta<=1e-11))
        print(json.dumps(checks[-1]), flush=True)
    summaries = []
    for regime in ("sparse", "dense"):
        if not all(f"{regime}4_seed0_{i}" in cases for i in range(1, 6)):
            continue
        for method in METHODS:
            for shots in sampling.SHOT_BUDGETS:
                values = [row["empirical_rmse"] for row in instances
                          if row["benchmark"] == regime and row["method"] == method and row["shots"] == shots]
                mean = float(np.mean(values))
                reference = next(r for r in figure_rows if r["benchmark"] == regime and r["method"] == method
                                 and int(r["shots"]) == shots)
                delta = abs(mean-float(reference["mean_instance_empirical_rmse"]))
                if delta > 1e-11:
                    raise RuntimeError(f"Figure 5 mismatch: {regime}/{method}/T={shots}")
                summaries.append(dict(benchmark=regime, method=method, shots=shots, instances=5,
                                      empirical_repeats_per_instance=50, mean_instance_empirical_rmse=mean,
                                      empirical_rmse_instance_sample_sd=float(np.std(values, ddof=1)),
                                      reference_absolute_difference=delta))
    def save(name, rows):
        if rows:
            with (output/name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    save("pauli_replicates.csv", replicated)
    save("pauli_instance_results.csv", instances)
    save("pauli_figure5_summary.csv", summaries)
    result = dict(status="PASS" if all(row["matches_within_tolerance"] for row in checks) else "DIFFERENT",
                  scope="Fresh Born outcomes from published frozen Pauli schedules; GPD checked separately",
                  cases=len(cases), repetitions=len(replicated), instance_budget_method_rows=len(instances),
                  figure5_pauli_points=len(summaries), seconds=time.perf_counter()-started,
                  environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                  checks=checks,
                  inputs={p.relative_to(data_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)},
                  outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.csv"))})
    (output/"random_pauli_reproduction.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status","cases","repetitions","figure5_pauli_points","seconds")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=ARCHIVE/"data")
    parser.add_argument("--output",type=Path,default=ARCHIVE/"build/random_pauli_replay")
    parser.add_argument("--case",action="append",choices=CASES,help="Repeat to select cases; default all ten")
    args=parser.parse_args()
    result=main(args.data_root,args.output,list(dict.fromkeys(args.case or CASES)))
    raise SystemExit(0 if result["status"]=="PASS" else 1)
