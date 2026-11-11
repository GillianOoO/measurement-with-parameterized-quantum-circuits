#!/usr/bin/env python3
"""Fresh matched-Hamiltonian Figure 5 product-Pauli computations.

Run once per independently generated four-qubit Hamiltonian.  The measurement
rules, budgets and seeds follow the published random-Hamiltonian experiment;
the number of Hamiltonian instances is controlled by the external ensemble
driver.  ``--repeats`` controls independent energy estimators for ONE H and
does not set the number of independently generated Hamiltonians.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "code/methods/pauli_product"
sys.path.insert(0, str(KERNELS))
import run_general_hamiltonian_revision as general
import replay_random_all_methods_empirical50 as sampling

PROFILE = "fresh-random-ensemble-product-pauli-v1"
METHODS = ("OGM", "SG", "Derand", "LCS", "AP")
SHOTS = tuple(sampling.SHOT_BUDGETS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Empty output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_npz(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def exact_pauli_decomposition(hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate the lexicographically ordered Hermitian Pauli basis."""
    dimension = len(hamiltonian)
    n = int(round(math.log2(dimension)))
    observables, weights = [], []
    for code in itertools.product(range(4), repeat=n):
        matrix = np.asarray([[1.0]], dtype=complex)
        for item in code:
            matrix = np.kron(matrix, general.PAULI[item])
        # Preserve the original generator's arithmetic order.  Different
        # floating-point reductions can change deterministic schedule ties.
        value = np.trace(matrix @ hamiltonian) / dimension
        if abs(value.imag) > 1e-12:
            raise ValueError("Pauli coefficient is not real")
        if abs(value.real) > 1e-12:
            observables.append(code)
            weights.append(float(value.real))
    if not observables:
        raise ValueError("The benchmark Hamiltonian has no nonzero Pauli terms")
    return np.asarray(observables, dtype=np.int8), np.asarray(weights, dtype=float)


def validate_schedule(bases: np.ndarray, counts: np.ndarray, shots: int) -> None:
    if len(bases) != len(counts) or np.any(counts <= 0) or int(np.sum(counts)) != shots:
        raise RuntimeError(f"Invalid schedule at T={shots}")
    if not np.all(np.isin(bases, (1, 2, 3))):
        raise RuntimeError("A schedule contains an unexecutable local-Pauli basis")


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    output = args.output.resolve()
    frozen = (ROOT / "data").resolve()
    if output == frozen or output.is_relative_to(frozen):
        raise ValueError("New computations must be outside the frozen data directory")
    output.mkdir(parents=True, exist_ok=True)
    h_path, state_path = args.input.resolve(), args.state.resolve()
    hamiltonian = np.asarray(np.load(h_path, allow_pickle=False), dtype=np.complex128)
    if hamiltonian.shape != (16, 16):
        raise ValueError("The published sparse/dense protocol uses four qubits")
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("Nonfinite Hamiltonian")
    state = general.pure_state_from_saved(state_path)
    observables, weights = exact_pauli_decomposition(hamiltonian)
    case = general.BenchmarkCase(args.benchmark, args.slug, args.instance, args.slug,
                                 hamiltonian, state, observables, weights, SHOTS,
                                 (h_path, state_path), {})
    state_hash = general.array_sha256(case.state)
    target_hash = case.target_hash
    input_file_hash = sha256(h_path)
    state_file_hash = sha256(state_path)
    configuration = dict(profile=PROFILE, slug=case.slug, benchmark=case.benchmark,
                         instance=case.instance, repeats=args.repeats, shots=list(SHOTS),
                         target_hash=target_hash, state_hash=state_hash,
                         input_file_sha256=input_file_hash, state_file_sha256=state_file_hash,
                         generator_sha256=sha256(Path(__file__)))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration") != configuration:
            raise RuntimeError("Output contains a different case/configuration; use a new directory")
        if previous.get("status") == "COMPLETE":
            for name, digest in previous["outputs_sha256"].items():
                if sha256(output / name) != digest:
                    raise RuntimeError(f"Completed output was modified: {name}")
            print(json.dumps(dict(status="CACHED_COMPLETE", slug=case.slug)), flush=True)
            return previous
    write_json(manifest_path, dict(status="RUNNING", configuration=configuration))
    table = np.column_stack((weights, observables))
    np.savetxt(output / "pauli_coefficients.txt", table, fmt=["%.18e", "%d", "%d", "%d", "%d"])
    reconstruction_error = float(np.linalg.norm(general.pauli_dense(observables, weights) - case.hamiltonian))
    obs, coefficients, offset = general.split_identity(observables, weights)
    evaluator = general.StateEvaluator(case.state.copy())
    replicates, instance_rows, timings = [], [], []
    schedule_paths = []

    def stage(method: str, message: str, **extra) -> None:
        print(json.dumps(dict(slug=case.slug, method=method, stage=message,
                              elapsed_seconds=time.perf_counter()-started, **extra)), flush=True)

    def record(method: str, shots: int, values: list[float], seeds: list[int],
               analytic: list[dict], schedule_repeats: list[int], schedule_hash: str) -> None:
        if len(values) != args.repeats or len(analytic) != args.repeats:
            raise RuntimeError("Incomplete estimator point")
        errors = np.asarray(values) - case.target_expectation
        for repeat, (value, error, seed, schedule_repeat, exact) in enumerate(
                zip(values, errors, seeds, schedule_repeats, analytic)):
            replicates.append(dict(benchmark=case.benchmark, instance=case.instance, slug=case.slug,
                                   method=method, shots=shots, repeat=repeat, schedule_repeat=schedule_repeat,
                                   outcome_seed=seed, estimate=value, error=float(error),
                                   squared_error=float(error*error), target_expectation=case.target_expectation,
                                   target_hash=target_hash, state_hash=state_hash,
                                   input_file_sha256=input_file_hash, state_file_sha256=state_file_hash,
                                   analytic_bias=float(exact["bias"]), analytic_variance=float(exact["variance"]),
                                   schedule_sha256=schedule_hash))
        mse = float(np.mean([float(item["mse"]) for item in analytic]))
        instance_rows.append(dict(benchmark=case.benchmark, instance=case.instance, slug=case.slug,
                                  method=method, shots=shots, empirical_repeats=args.repeats,
                                  empirical_mse=float(np.mean(errors**2)),
                                  empirical_rmse=float(np.sqrt(np.mean(errors**2))),
                                  empirical_mean_error=float(np.mean(errors)),
                                  empirical_error_sample_sd=float(np.std(errors, ddof=1)),
                                  analytic_mse=mse, analytic_rmse=math.sqrt(max(0.0, mse)),
                                  analytic_bias=float(np.mean([float(item["bias"]) for item in analytic])),
                                  analytic_bias_squared=float(np.mean([float(item["bias_squared"]) for item in analytic])),
                                  analytic_variance=float(np.mean([float(item["variance"]) for item in analytic])),
                                  target_expectation=case.target_expectation, target_hash=target_hash,
                                  state_hash=state_hash, input_file_sha256=input_file_hash,
                                  state_file_sha256=state_file_hash, schedule_sha256=schedule_hash))

    for method in METHODS:
        method_started = time.perf_counter()
        stage(method, "schedule_construction_started")
        schedule_payload, schedules = {}, {}
        if method in ("SG", "Derand"):
            seed = general.stable_seed(general.BASE_SEED, case.slug, method, "fixed-prefix")
            signature = general.schedule_cache_signature(case.slug, method, obs, coefficients, SHOTS, seed)
            payload = dict(job_id=f"{case.slug}_{method}", method=method, observables=obs,
                           weights=coefficients, shots=SHOTS, seed=seed, signature=signature)
            generated = general._generate_retained_schedule_payload(payload)
            schedules = generated["snapshots"]
            for shots, (bases, counts) in schedules.items():
                validate_schedule(bases, counts, shots)
                schedule_payload[f"bases_{shots}"] = bases
                schedule_payload[f"counts_{shots}"] = counts
            schedule_payload["schedule_seed"] = np.asarray(seed, dtype=np.uint64)
        elif method == "OGM":
            settings, probabilities, objective, success = general.measurement_tools.optimize_ogm_distribution(obs, coefficients)
            if not success:
                raise RuntimeError(f"{case.slug}: OGM SLSQP did not converge")
            settings, probabilities = general.combine_ogm_distribution(settings, probabilities)
            schedule_payload.update(settings=settings, probabilities=probabilities,
                                    objective=np.asarray(objective), optimizer_success=np.asarray(success))
            for shots in SHOTS:
                seed = general.stable_seed(general.BASE_SEED, case.slug, method, shots)
                bases, counts = general.allocate_ogm_schedule(settings, probabilities, shots, seed)
                validate_schedule(bases, counts, shots)
                schedules[shots] = (bases, counts)
                schedule_payload[f"bases_{shots}"] = bases
                schedule_payload[f"counts_{shots}"] = counts
                schedule_payload[f"allocation_seed_{shots}"] = np.asarray(seed, dtype=np.uint64)
        elif method == "AP":
            settings, probabilities = general.ap_exact_setting_distribution(obs, coefficients)
            schedule_payload.update(settings=settings, probabilities=probabilities)
            for repeat in range(args.repeats):
                seed = general.stable_seed(general.BASE_SEED, case.slug, method, repeat)
                prefixes = general.sampled_prefixes_from_distribution(settings, probabilities, SHOTS, seed)
                schedule_payload[f"schedule_seed_r{repeat:02d}"] = np.asarray(seed, dtype=np.uint64)
                for shots in SHOTS:
                    bases, counts = prefixes[shots]
                    validate_schedule(bases, counts, shots)
                    schedules[(repeat, shots)] = (bases, counts)
                    schedule_payload[f"bases_r{repeat:02d}_T{shots}"] = bases
                    schedule_payload[f"counts_r{repeat:02d}_T{shots}"] = counts
        else:
            settings, joint, scores = sampling.lcs_joint_law(general, evaluator, obs, coefficients)
            exact_mean = offset + float(joint @ scores)
            if not math.isclose(exact_mean, case.target_expectation, rel_tol=0.0, abs_tol=3e-12):
                raise RuntimeError("LCS mean does not equal target expectation")
            variance_coefficient = max(0.0, float(joint @ (scores**2) - (joint @ scores)**2))
            schedule_payload.update(settings=settings, joint_probabilities=joint, scores=scores,
                                    variance_coefficient=np.asarray(variance_coefficient))
        construction_seconds = time.perf_counter() - method_started
        schedule_payload["metadata"] = np.asarray(json.dumps(dict(profile=PROFILE, method=method,
                                      slug=case.slug, target_hash=target_hash, state_hash=state_hash,
                                      shots=list(SHOTS), repeats=args.repeats), sort_keys=True))
        schedule_path = output / f"{method}_schedules.npz"
        write_npz(schedule_path, schedule_payload)
        schedule_paths.append(schedule_path)
        schedule_hash = sha256(schedule_path)
        stage(method, "schedule_construction_complete", construction_seconds=construction_seconds)
        replay_started = time.perf_counter()
        for shots in SHOTS:
            values, seeds, exacts, schedule_repeats = [], [], [], []
            if method not in ("AP", "LCS"):
                bases, counts = schedules[shots]
                exact = general.fixed_schedule_mse(evaluator, obs, coefficients, offset,
                                                   bases, counts, case.target_expectation)
            for repeat in range(args.repeats):
                seed = sampling.stable_seed(sampling.PROFILE, sampling.BASE_SEED,
                                            case.slug, method, shots, repeat)
                if method == "LCS":
                    value = sampling.lcs_replay(joint, scores, offset, shots, seed)
                    variance = variance_coefficient / shots
                    exact = dict(mse=variance, bias=0.0, bias_squared=0.0, variance=variance)
                else:
                    if method == "AP":
                        bases, counts = schedules[(repeat, shots)]
                        exact = general.fixed_schedule_mse(evaluator, obs, coefficients, offset,
                                                           bases, counts, case.target_expectation)
                    value = sampling.fixed_schedule_replay(general, evaluator, obs, coefficients,
                                                           offset, bases, counts, seed)
                values.append(value)
                seeds.append(seed)
                exacts.append(dict(exact))
                schedule_repeats.append(repeat if method in ("AP", "LCS") else 0)
            record(method, shots, values, seeds, exacts, schedule_repeats, schedule_hash)
        replay_seconds = time.perf_counter() - replay_started
        timings.append(dict(method=method, schedule_construction_seconds=construction_seconds,
                            replay_seconds=replay_seconds, total_seconds=time.perf_counter()-method_started))
        write_csv(output / "replicates.csv", replicates)
        write_csv(output / "instance_results.csv", instance_rows)
        write_csv(output / "method_timings.csv", timings)
        stage(method, "complete", timing=timings[-1])
    expected = len(METHODS) * len(SHOTS) * args.repeats
    if len(replicates) != expected or len(instance_rows) != len(METHODS)*len(SHOTS):
        raise RuntimeError("Unexpected output row count")
    if not all(np.isfinite(float(r["empirical_rmse"])) for r in instance_rows):
        raise RuntimeError("Nonfinite result")
    outputs = [output / name for name in ("pauli_coefficients.txt", "replicates.csv", "instance_results.csv", "method_timings.csv")]
    outputs.extend(schedule_paths)
    sources = [Path(__file__), Path(general.__file__), Path(sampling.__file__),
               Path(general.measurement_tools.__file__)]
    sources.extend(sorted((KERNELS / "shadowgrouping").glob("*.py")))
    result = dict(status="COMPLETE", configuration=configuration,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  inputs={str(h_path):sha256(h_path), str(state_path):sha256(state_path)},
                  source_sha256={str(path):sha256(path) for path in sources},
                  outputs_sha256={path.name:sha256(path) for path in outputs},
                  checks=dict(pauli_reconstruction_frobenius_error=reconstruction_error,
                              replicate_rows=len(replicates), instance_rows=len(instance_rows),
                              identical_target_and_state_hashes_for_all_methods=True),
                  method_timings=timings, elapsed_seconds=time.perf_counter()-started,
                  semantics=dict(ensemble_aggregation="arithmetic mean of instance-level empirical RMSE",
                                 finite_measurement_repeats=args.repeats,
                                 OGM="one freshly optimized distribution and archived-rule T-specific allocations",
                                 SG="one freshly generated max-T schedule, prefixes at seven budgets",
                                 Derand="one freshly generated max-T schedule, prefixes at seven budgets",
                                 LCS="independent uniform local-Pauli setting and Born-outcome trials",
                                 AP="one independent Born replay for each freshly generated AP prefix schedule"))
    write_json(manifest_path, result)
    print(json.dumps(dict(status="COMPLETE", slug=case.slug,
                          elapsed_seconds=result["elapsed_seconds"], rows=len(replicates))), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--benchmark", choices=("sparse", "dense"), required=True)
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 2 or args.instance < 1:
        parser.error("--repeats must be at least two and --instance must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
