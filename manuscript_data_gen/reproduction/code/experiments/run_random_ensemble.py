#!/usr/bin/env python3
"""Time and rerun the manuscript's matched sparse/dense Hamiltonian ensemble.

Only the count of independent Hamiltonians changes.  Native GPD fitting,
stabilizer calibration, Pauli schedules, measurement counts, and 50 complete
energy estimates per Hamiltonian follow the released experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("GPD", "OGM", "SG", "Derand", "LCS", "AP")
SHOTS = (12, 45, 160, 572, 2038, 7259, 25848)
HERE = Path(__file__).resolve().parent
GPD = ROOT / "code/methods/gpd"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def environments(args):
    gpd_env = os.environ.copy()
    # A normal virtual environment needs no machine-specific package paths.
    # Explicit runtime overrides remain available for the recorded environment.
    if args.gpd_runtime:
        gpd_env["PYTHONPATH"] = os.pathsep.join(str(p.resolve()) for p in args.gpd_runtime)
    else:
        gpd_env.pop("PYTHONPATH", None)
    gpd_env["JAX_ENABLE_X64"] = "false"
    gpd_env["JAX_PLATFORM_NAME"] = "cpu"
    gpd_env["PYTHONUNBUFFERED"] = "1"
    pauli_env = os.environ.copy()
    pauli_env.pop("PYTHONPATH", None)
    pauli_env["PYTHONUNBUFFERED"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        gpd_env[key] = "1"
    # Retain the original numerical environment for SLSQP and deterministic
    # schedule tie decisions. Thread restrictions change near-tied OGM
    # probabilities at roundoff and can change finite-T allocations.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        pauli_env.pop(key, None)
    return gpd_env, pauli_env


def command(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n" + json.dumps({"utc": utc_now(), "command": [str(x) for x in command]}) + "\n")
        handle.flush()
        result = subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-18:])
        raise RuntimeError(f"Exit {result.returncode}: {log}\n{tail}")
    return time.perf_counter() - start


def input_case(args, benchmark, instance):
    inputs = args.output / "inputs"
    return dict(benchmark=benchmark, instance=instance,
                slug=f"{benchmark}4_seed0_{instance}",
                input=inputs / "hamiltonians" / f"{benchmark}_hamiltonian_4_{instance}.npy",
                state=inputs / "common_input_state.npy")


def run_case(args, benchmark, instance):
    case = input_case(args, benchmark, instance)
    slug = case["slug"]
    output = args.output / "cases" / slug
    gpd_env, pauli_env = environments(args)
    complete_path = output / "complete.json"
    binding = dict(input_file_sha256=sha256(case["input"]),
                   state_file_sha256=sha256(case["state"]), repeats=50,
                   gpd_runtime_paths=[str(p.resolve()) for p in args.gpd_runtime],
                   pauli_thread_policy="original-default",
                   source_fit_sha256=sha256(GPD / "generate_frobenius_prefixes.py"),
                   selector_sha256=sha256(GPD / "reselect_stabilizer_balanced.py"),
                   pauli_runner_sha256=sha256(HERE / "random_ensemble_pauli.py"))
    pending_path = output / "pauli_refresh_pending.json"
    if not complete_path.exists() and pending_path.exists() and args.refresh_pauli:
        pending = read_json(pending_path)
        if pending["binding"] != binding:
            raise RuntimeError("The interrupted Pauli refresh used different code or inputs")
        return refresh_pauli_case(args, case, output, pending["old_complete"], binding, pauli_env)
    if complete_path.exists():
        old = read_json(complete_path)
        if old["binding"] != binding:
            ignored = {"pauli_runner_sha256", "pauli_thread_policy"}
            old_core = {k:v for k,v in old["binding"].items() if k not in ignored}
            new_core = {k:v for k,v in binding.items() if k not in ignored}
            if args.refresh_pauli and old_core == new_core:
                return refresh_pauli_case(args, case, output, old, binding, pauli_env)
            raise RuntimeError(f"Completed case binding changed: {slug}")
        for name, digest in old["result_sha256"].items():
            if sha256(output / name) != digest:
                raise RuntimeError(f"Modified completed case: {slug}/{name}")
        return old
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print(json.dumps(dict(event="case_started", slug=slug, utc=utc_now())), flush=True)
    source = output / "gpd_source"
    selected = output / "gpd_selected"
    gpd_seconds = 0.0
    for bank_size in range(30, 101, 10):
        source_command = [args.gpd_python, "-u", GPD / "generate_frobenius_prefixes.py", "case",
                          "--benchmark", benchmark, "--slug", slug,
                          "--input", case["input"], "--state", case["state"], "--output", source,
                          "--paper-depth", "4", "--starts", "1", "--steps", "600",
                          "--learning-rate", "0.01", "--base-seed", "20260808",
                          "--shots", *map(str, SHOTS),
                          "--minimum-k", "101", "--max-k", str(bank_size),
                          "--stall-patience", "5", "--relative-improvement", "0.002",
                          "--progress-every", "600", "--resume"]
        # Only the calibrated post-K10 rule defines completion.  Suppress the
        # source generator's historical uncalibrated stop; extend its bank as
        # needed and fail closed at the numerical nontermination guard.
        gpd_seconds += command(source_command, output / "gpd_source.log", gpd_env)
        gpd_seconds += command([args.gpd_python, "-u", GPD / "reselect_stabilizer_balanced.py",
                                "--source", source, "--output", selected,
                                "--probe-seed", "918273", "--minimum-k", "10",
                                "--stall-patience", "5", "--relative-improvement", "0.002",
                                "--empirical-repeats", "50", "--empirical-base-seed", "20260902",
                                "--force"], output / "gpd_selection.log", gpd_env)
        stopping = read_json(selected / "material_stall_replay.json")
        if stopping["stop_observed"] and not stopping["right_censored"]:
            break
        print(json.dumps(dict(event="extend_source_bank", slug=slug,
                              previous_bank_size=bank_size)), flush=True)
    else:
        raise RuntimeError(f"{slug}: calibrated termination not observed before K_guard=100")
    pauli_seconds = command([args.pauli_python, "-u", HERE / "random_ensemble_pauli.py",
                             "--input", case["input"], "--state", case["state"],
                             "--benchmark", benchmark, "--instance", str(instance), "--slug", slug,
                             "--output", output / "pauli", "--repeats", "50"],
                            output / "pauli.log", pauli_env)
    pauli = read_json(output / "pauli/manifest.json")
    if pauli["status"] != "COMPLETE":
        raise RuntimeError(f"{slug}: incomplete Pauli result")
    gpd_rows = read_csv(selected / "selected_results.csv")
    if len(gpd_rows) != 7 or any(r["frontier_right_censored"].lower() != "false" for r in gpd_rows):
        raise RuntimeError(f"{slug}: invalid or incomplete GPD frontier")
    # The exact same immutable input files are passed to all six methods.
    if pauli["configuration"]["input_file_sha256"] != binding["input_file_sha256"]:
        raise RuntimeError("Cross-method Hamiltonian file mismatch")
    if pauli["configuration"]["state_file_sha256"] != binding["state_file_sha256"]:
        raise RuntimeError("Cross-method input-state file mismatch")
    if sha256(case["input"]) != binding["input_file_sha256"] or sha256(case["state"]) != binding["state_file_sha256"]:
        raise RuntimeError("Inputs changed during computation")
    result_files = ["gpd_selected/selected_results.csv", "gpd_selected/manifest.json",
                    "gpd_selected/material_stall_replay.json", "pauli/instance_results.csv",
                    "pauli/replicates.csv", "pauli/manifest.json"]
    result_files += [f"gpd_selected/empirical_replicates_T{total}.csv" for total in SHOTS]
    result = dict(status="COMPLETE", benchmark=benchmark, instance=instance, slug=slug,
                  completed_utc=utc_now(), binding=binding,
                  gpd_seconds=gpd_seconds, pauli_seconds=pauli_seconds,
                  elapsed_seconds=time.perf_counter()-start,
                  effective_terminal_k=stopping["effective_terminal_k"], source_bank_size=bank_size,
                  result_sha256={name:sha256(output/name) for name in result_files})
    write_json(complete_path, result)
    print(json.dumps(dict(event="case_complete", slug=slug,
                          elapsed_seconds=result["elapsed_seconds"],
                          effective_terminal_k=result["effective_terminal_k"])), flush=True)
    return result


def refresh_pauli_case(args, case, output, old, binding, pauli_env):
    """Preserve verified GPD fits and replace only audited Pauli numerics."""
    history = output / "before_original_pauli_arithmetic"
    resuming = (output / "pauli_refresh_pending.json").exists() and history.is_dir()
    for name, digest in old["result_sha256"].items():
        previous_path = history/name if resuming and name.startswith("pauli/") else output/name
        if sha256(previous_path) != digest:
            raise RuntimeError(f"Modified prior case output: {case['slug']}/{name}")
    if history.exists() and not resuming:
        raise RuntimeError(f"Refusing to overwrite prior Pauli correction history: {history}")
    if not resuming:
        history.mkdir()
        (output / "pauli").rename(history / "pauli")
        (output / "complete.json").rename(history / "complete.json")
        old_log = output / "pauli.log"
        if old_log.exists():
            old_log.rename(history / "pauli.log")
    # If this invocation is interrupted, the archived complete marker restores
    # the verified GPD binding for the next --refresh-pauli invocation.
    write_json(output / "pauli_refresh_pending.json", dict(old_complete=old, binding=binding))
    print(json.dumps(dict(event="pauli_refresh_started", slug=case["slug"])), flush=True)
    elapsed = command([args.pauli_python, "-u", HERE / "random_ensemble_pauli.py",
                       "--input", case["input"], "--state", case["state"],
                       "--benchmark", case["benchmark"], "--instance", str(case["instance"]),
                       "--slug", case["slug"], "--output", output / "pauli", "--repeats", "50"],
                      output / "pauli.log", pauli_env)
    manifest = read_json(output / "pauli/manifest.json")
    if manifest["status"] != "COMPLETE" or any(manifest["configuration"][key] != binding[key]
           for key in ("input_file_sha256", "state_file_sha256")):
        raise RuntimeError("Refreshed Pauli result is incomplete or unpaired")
    result = dict(old)
    result.update(binding=binding, completed_utc=utc_now(), pauli_seconds=elapsed,
                  elapsed_seconds=float(old["gpd_seconds"])+elapsed,
                  gpd_fit_reused_after_exact_legacy_reproduction=True,
                  pauli_correction_history=str(history))
    result["result_sha256"] = {name:sha256(output/name) for name in old["result_sha256"]}
    write_json(output / "complete.json", result)
    (output / "pauli_refresh_pending.json").unlink()
    print(json.dumps(dict(event="pauli_refresh_complete", slug=case["slug"],
                          elapsed_seconds=elapsed)), flush=True)
    return result


def pilot(args):
    start = time.perf_counter()
    cases = [run_case(args, benchmark, 1) for benchmark in ("sparse", "dense")]
    # Include every method, calibration, full max-T schedules, and all 50
    # energy estimates.  Use a conservative serialized projection, independent
    # of any later parallel throughput, with a 50% variability margin.
    mean_pair = sum(float(c["elapsed_seconds"]) for c in cases)
    raw_50 = 50 * mean_pair
    conservative_50 = 1.5 * raw_50
    count = 50 if conservative_50 <= 24*3600 else 20
    result = dict(status="COMPLETE", completed_utc=utc_now(), pilot_cases=cases,
                  observed_sparse_dense_pair_seconds=mean_pair,
                  projected_50_per_regime_serial_seconds=raw_50,
                  variability_margin=1.5,
                  estimated_50_per_regime_seconds=conservative_50,
                  threshold_seconds=24*3600, selected_instances_per_regime=count,
                  decision_rule="50 if conservative serialized projection <=24h, otherwise20",
                  pilot_driver_elapsed_seconds=time.perf_counter()-start,
                  pilot_results_reused_in_production=True)
    write_json(args.output / "timing_decision.json", result)
    print(json.dumps(result, indent=2), flush=True)


def run(args):
    decision = read_json(args.output / "timing_decision.json")
    count = int(decision["selected_instances_per_regime"])
    if count not in (20, 50):
        raise RuntimeError("Invalid recorded timing decision")
    pairs = [(regime, i) for i in range(1, count+1) for regime in ("sparse", "dense")]
    started = time.perf_counter()
    records, failures = [], []
    status_path = args.output / "progress.json"
    def progress(status):
        write_json(status_path, dict(status=status, updated_utc=utc_now(),
                    instances_per_regime=count, cases_expected=2*count,
                    cases_complete=len(records), elapsed_seconds=time.perf_counter()-started,
                    workers=args.workers, completed_cases=[r["slug"] for r in records], failures=failures))
    progress("RUNNING")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, args, regime, instance):(regime, instance) for regime, instance in pairs}
        for future in as_completed(futures):
            regime, instance = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(dict(benchmark=regime, instance=instance, error=str(exc)))
                print(json.dumps(dict(event="case_failed", **failures[-1])), flush=True)
            progress("RUNNING")
    if failures:
        progress("FAILED")
        raise RuntimeError(f"{len(failures)} cases failed; no ensemble figure accepted")
    progress("COMPUTATIONS_COMPLETE")
    aggregate(args, count)
    progress("COMPLETE")


def aggregate(args, count=None):
    import numpy as np
    if count is None:
        count = int(read_json(args.output / "timing_decision.json")["selected_instances_per_regime"])
    instance_rows, replicate_rows, audit = [], [], []
    for regime in ("sparse", "dense"):
        for instance in range(1, count+1):
            case = input_case(args, regime, instance)
            folder = args.output / "cases" / case["slug"]
            complete = read_json(folder / "complete.json")
            if complete["status"] != "COMPLETE":
                raise RuntimeError(f"Missing case {case['slug']}")
            for name, digest in complete["result_sha256"].items():
                if sha256(folder / name) != digest:
                    raise RuntimeError(f"Modified completed output {case['slug']}/{name}")
            common = dict(benchmark=regime, instance=instance, slug=case["slug"],
                          input_file_sha256=sha256(case["input"]), state_file_sha256=sha256(case["state"]))
            pauli_rows = read_csv(folder / "pauli/instance_results.csv")
            pauli_reps = read_csv(folder / "pauli/replicates.csv")
            for row in pauli_rows:
                if row["input_file_sha256"] != common["input_file_sha256"] or row["state_file_sha256"] != common["state_file_sha256"]:
                    raise RuntimeError("Pairing check failed")
                instance_rows.append(row)
            replicate_rows.extend(pauli_reps)
            for row in read_csv(folder / "gpd_selected/selected_results.csv"):
                total = int(row["shots"])
                reps = read_csv(folder / "gpd_selected" / f"empirical_replicates_T{total}.csv")
                errors = np.asarray([float(r["error"]) for r in reps])
                if len(reps) != 50 or not math.isclose(float(np.sqrt(np.mean(errors**2))), float(row["empirical_rmse"]), abs_tol=2e-13):
                    raise RuntimeError("GPD empirical RMSE check failed")
                values = dict(common, method="GPD", shots=total, empirical_repeats=50,
                              empirical_mse=float(np.mean(errors**2)), empirical_rmse=float(np.sqrt(np.mean(errors**2))),
                              analytic_mse=float(row["mse"]), analytic_rmse=float(row["analytic_rmse"]),
                              analytic_bias=float(row["bias"]), analytic_variance=float(row["variance"]),
                              selected_k=int(row["k"]), effective_terminal_k=complete["effective_terminal_k"],
                              target_expectation=float(row["target_expectation"]))
                instance_rows.append(values)
                for repeat in reps:
                    replicate_rows.append(dict(common, method="GPD", shots=total,
                        repeat=int(repeat["repeat"]), estimate=float(repeat["estimate"]),
                        error=float(repeat["error"]), squared_error=float(repeat["error"])**2,
                        target_expectation=float(row["target_expectation"])))
            for row in pauli_rows:
                subset = [r for r in pauli_reps if r["method"] == row["method"] and r["shots"] == row["shots"]]
                expected = math.sqrt(sum(float(r["error"])**2 for r in subset)/50)
                if len(subset) != 50 or not math.isclose(expected,float(row["empirical_rmse"]),abs_tol=2e-13):
                    raise RuntimeError("Pauli empirical RMSE check failed")
            target_values = [float(r["target_expectation"]) for r in instance_rows if r["slug"] == case["slug"]]
            if np.ptp(target_values) > 2e-12:
                raise RuntimeError("Cross-method target expectation mismatch")
            audit.append(dict(common, status="PASS", target_expectation_max_difference=float(np.ptp(target_values)),
                              effective_terminal_k=complete["effective_terminal_k"],
                              all_methods_same_input_files=True, right_censored=False))
    summaries = []
    for regime in ("sparse", "dense"):
        for method in METHODS:
            for total in SHOTS:
                rows = [r for r in instance_rows if r["benchmark"] == regime and r["method"] == method and int(r["shots"]) == total]
                if len(rows) != count or len({int(r["instance"]) for r in rows}) != count:
                    raise RuntimeError(f"Incomplete method/measurement grid {regime}/{method}/{total}")
                values = np.asarray([float(r["empirical_rmse"]) for r in rows])
                sd = float(np.std(values, ddof=1))
                summaries.append(dict(benchmark=regime, method=method, shots=total, instances=count,
                    empirical_repeats_per_instance=50, mean_instance_empirical_rmse=float(np.mean(values)),
                    sd_instance_empirical_rmse=sd, se_instance_empirical_rmse=sd/math.sqrt(count),
                    mean_instance_analytic_rmse=float(np.mean([float(r["analytic_rmse"]) for r in rows])),
                    pooled_empirical_rmse=float(np.sqrt(np.mean([float(r["empirical_mse"]) for r in rows])))))
    if len(instance_rows) != count*2*6*7 or len(replicate_rows) != count*2*6*7*50:
        raise RuntimeError("Incomplete total data grid")
    results = args.output / "results"
    write_csv(results / "instance_results.csv", instance_rows)
    write_csv(results / "replicates.csv", replicate_rows)
    write_csv(results / "ensemble_summary.csv", summaries)
    write_csv(results / "paired_input_audit.csv", audit)
    names = ("instance_results.csv", "replicates.csv", "ensemble_summary.csv", "paired_input_audit.csv")
    write_json(results / "manifest.json", dict(status="COMPLETE", completed_utc=utc_now(),
               instances_per_regime=count, methods=METHODS, shots=SHOTS, repeats_per_instance=50,
               aggregation="arithmetic mean of instance empirical RMSE over 50 complete estimates per instance",
               instance_rows=len(instance_rows), replicate_rows=len(replicate_rows), summary_rows=len(summaries),
               all_pairs_validated=True, all_gpd_calibrated_stops_observed=True,
               output_sha256={name:sha256(results/name) for name in names},
               input_manifest_sha256=sha256(args.output / "inputs/manifest.json"),
               timing_decision_sha256=sha256(args.output / "timing_decision.json")))
    print(json.dumps(dict(event="aggregation_complete", instances=count,
                          instance_rows=len(instance_rows), replicate_rows=len(replicate_rows))), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "run", "aggregate"))
    parser.add_argument("--output", type=Path, default=ROOT / "regenerated/random_ensemble_new",
                        help="New experiment directory, separate from the frozen production data")
    parser.add_argument("--gpd-python", type=Path, default=Path(sys.executable),
                        help="Python with requirements-random-ensemble-gpd.txt installed")
    parser.add_argument("--pauli-python", type=Path, default=Path(sys.executable),
                        help="Python with requirements-random-pauli.txt installed")
    parser.add_argument("--gpd-runtime", type=Path, nargs="+", default=[],
                        help="Optional package directories prepended for the GPD subprocess")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--refresh-pauli", action="store_true",
                        help="Reuse verified GPD fits when only the Pauli runner or thread policy changed")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    frozen = (ROOT / "data").resolve()
    if args.output == frozen or args.output.is_relative_to(frozen):
        parser.error("New computations must be outside the frozen data directory")
    # The subprocess working directory is ROOT, not necessarily the caller's cwd.
    args.gpd_python = args.gpd_python.resolve()
    args.pauli_python = args.pauli_python.resolve()
    args.gpd_runtime = [path.resolve() for path in args.gpd_runtime]
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    {"pilot":pilot, "run":run, "aggregate":aggregate}[arguments.stage](arguments)
