#!/usr/bin/env python3
"""Create a matched 50-estimator sparse/dense empirical replay table.

The six exported methods are GPD, OGM, SG, Derand, LCS, and AP.  GPD uses
the first 50 estimators from the already audited 200-replay artifact.  The
Pauli methods use the exact common rank-one state and genuine local-Pauli
Born outcomes.  OGM and AP schedules are materialized before sampling so
that later audits do not depend on an optimizer or NumPy/SciPy RNG drift.

This generator must be run in the environment that produced the frozen
general-Hamiltonian baselines: Python 3.14.7, NumPy 2.5.2, SciPy 1.18.0.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
GENERAL_RUNNER = (
    WORKSPACE
    / "RMeasurementAnsatz"
    / "manuscript"
    / "Journal_chemical_theory_computation"
    / "plots_figures"
    / "run_general_hamiltonian_revision.py"
)
GENERAL_OUTPUT = (
    WORKSPACE / "outputs" / "adaptive_f3_ansatz_pool" / "general_hamiltonian_revision_v1"
)
GENERAL_MANIFEST = GENERAL_OUTPUT / "manifest.json"
GENERAL_INSTANCE = GENERAL_OUTPUT / "instance_error_rows.csv"
GENERAL_PAULI_AUDIT = GENERAL_OUTPUT / "pauli_schedule_audit.csv"
GENERAL_SCHEDULES = GENERAL_OUTPUT / "schedule_cache"

GPD_REPLICATES = HERE / "random_gpd_empirical_replicates.csv"
GPD_INSTANCES = HERE / "random_gpd_empirical_instance.csv"
GPD_MANIFEST = HERE / "random_gpd_empirical_manifest.json"
GPD_AUDIT = HERE / "random_gpd_empirical_audit.json"
GPD_COMMIT = HERE / "random_gpd_empirical_audit.commit.json"
GPD_CASES = HERE / "cases"

SCHEDULE_OUTPUT = HERE / "random_pauli_empirical50_schedules"
REPLICATE_OUTPUT = HERE / "random_all_methods_empirical50_replicates.csv"
INSTANCE_OUTPUT = HERE / "random_all_methods_empirical50_instance.csv"
SUMMARY_OUTPUT = HERE / "random_all_methods_empirical50_summary.csv"
MANIFEST_OUTPUT = HERE / "random_all_methods_empirical50_manifest.json"

PROFILE = "random-sparse-dense-all-methods-empirical50-v1"
METHODS = ("GPD", "OGM", "SG", "Derand", "LCS", "AP")
PAULI_METHODS = ("OGM", "SG", "Derand", "LCS", "AP")
BENCHMARKS = ("sparse", "dense")
INSTANCES = tuple(range(1, 6))
SHOT_BUDGETS = (12, 45, 160, 572, 2038, 7259, 25848)
REPEATS = 50
BASE_SEED = 20260902


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(item) for item in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def load_general_module():
    spec = importlib.util.spec_from_file_location("general_hamiltonian_revision", GENERAL_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(GENERAL_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_frozen_environment() -> None:
    observed = (platform.python_version(), np.__version__, scipy.__version__)
    expected = ("3.14.7", "2.5.2", "1.18.0")
    if observed != expected:
        raise RuntimeError(
            "OGM must be materialized in the frozen baseline environment; "
            f"expected Python/NumPy/SciPy={expected}, observed={observed}"
        )


def validate_source_bindings(general) -> tuple[dict[str, object], dict[str, object]]:
    manifest = read_json(GENERAL_MANIFEST)
    if manifest.get("profile_version") != "general-hamiltonian-revision-v1.4":
        raise RuntimeError("Unexpected general-Hamiltonian source profile")
    if sha256(GENERAL_RUNNER) != manifest["implementation_hashes"]["runner"]:
        raise RuntimeError("General-Hamiltonian runner hash mismatch")
    for name, expected in dict(manifest["source_hashes"]).items():
        path = Path(name)
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"General-Hamiltonian source hash mismatch: {path}")
    for name, expected in dict(manifest["outputs"]).items():
        path = Path(name)
        if not path.exists() or sha256(path) != expected:
            raise RuntimeError(f"General-Hamiltonian output hash mismatch: {path}")

    gpd_manifest = read_json(GPD_MANIFEST)
    gpd_audit = read_json(GPD_AUDIT)
    gpd_commit = read_json(GPD_COMMIT)
    if gpd_manifest.get("status") != "COMPLETE" or gpd_audit.get("status") != "PASS":
        raise RuntimeError("GPD empirical source is not complete and audited")
    if gpd_commit.get("status") != "COMMITTED":
        raise RuntimeError("GPD empirical audit is not committed")
    if sha256(GPD_MANIFEST) != gpd_commit["replay_manifest_sha256"]:
        raise RuntimeError("GPD replay manifest commit mismatch")
    if sha256(GPD_AUDIT) != gpd_commit["audit_json_sha256"]:
        raise RuntimeError("GPD replay audit commit mismatch")
    for name, expected in dict(gpd_manifest["outputs_sha256"]).items():
        if sha256(HERE / name) != expected:
            raise RuntimeError(f"GPD empirical source output mismatch: {name}")
    return manifest, gpd_manifest


def fixed_schedule_replay(
    general,
    evaluator,
    observables: np.ndarray,
    weights: np.ndarray,
    offset: float,
    bases: np.ndarray,
    counts: np.ndarray,
    seed: int,
) -> float:
    bases, counts = general.combine_count_schedule(bases, counts)
    coverage = general.coverage_matrix(observables, bases)
    hits = coverage.T @ counts
    covered = hits > 0
    rng = np.random.default_rng(int(seed))
    estimate = float(offset)
    for basis_index, (basis, count) in enumerate(zip(bases, counts)):
        indices = np.flatnonzero(coverage[basis_index] & covered)
        if len(indices) == 0:
            continue
        diagonal = np.zeros(evaluator.dimension, dtype=float)
        for term_index in indices:
            diagonal += (
                weights[term_index]
                / hits[term_index]
                * evaluator.term_sign(observables[term_index])
            )
        probabilities = evaluator.basis_probabilities(basis)
        draws = rng.multinomial(int(count), probabilities)
        estimate += float(draws @ diagonal)
    return estimate


def lcs_joint_law(general, evaluator, observables: np.ndarray, weights: np.ndarray):
    settings = np.asarray(
        list(itertools.product((1, 2, 3), repeat=evaluator.n_qubits)), dtype=np.int8
    )
    probabilities: list[float] = []
    scores: list[float] = []
    setting_probability = 1.0 / len(settings)
    supports = np.count_nonzero(observables, axis=1)
    for basis in settings:
        covered = np.all(
            (observables == 0) | (observables == basis[None, :]), axis=1
        )
        diagonal = np.zeros(evaluator.dimension, dtype=float)
        for index in np.flatnonzero(covered):
            diagonal += (
                weights[index]
                * (3.0 ** int(supports[index]))
                * evaluator.term_sign(observables[index])
            )
        born = evaluator.basis_probabilities(basis)
        probabilities.extend((setting_probability * born).tolist())
        scores.extend(diagonal.tolist())
    joint = np.asarray(probabilities, dtype=float)
    joint /= float(np.sum(joint))
    return settings, joint, np.asarray(scores, dtype=float)


def lcs_replay(joint: np.ndarray, scores: np.ndarray, offset: float, shots: int, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(int(shots), joint)
    return float(offset + counts @ scores / int(shots))


def close_metrics(result: dict[str, float | int], source: dict[str, str], label: str) -> None:
    for key in ("mse", "rmse", "bias", "bias_squared", "variance"):
        if not math.isclose(float(result[key]), float(source[key]), rel_tol=0.0, abs_tol=3.0e-13):
            raise RuntimeError(f"{label}: analytic {key} does not match frozen source")


def freeze_ogm(
    general,
    case,
    observables: np.ndarray,
    weights: np.ndarray,
    offset: float,
    evaluator,
    audit_rows: dict[tuple[str, str, int, int], dict[str, str]],
) -> tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]]:
    settings, probabilities, objective, success = general.measurement_tools.optimize_ogm_distribution(
        observables, weights
    )
    settings, probabilities = general.combine_ogm_distribution(settings, probabilities)
    if not success:
        raise RuntimeError(f"{case.slug}: OGM SLSQP did not converge in the frozen environment")
    payload: dict[str, np.ndarray] = {
        "settings": settings.astype(np.int8),
        "probabilities": probabilities.astype(np.float64),
        "objective": np.asarray(objective, dtype=np.float64),
    }
    schedules: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    seeds: dict[str, int] = {}
    for shots in SHOT_BUDGETS:
        seed = general.stable_seed(general.BASE_SEED, case.slug, "OGM", shots)
        bases, counts = general.allocate_ogm_schedule(settings, probabilities, shots, seed)
        result = general.fixed_schedule_mse(
            evaluator, observables, weights, offset, bases, counts, case.target_expectation
        )
        source = audit_rows[(case.slug, "OGM", shots, 0)]
        close_metrics(result, source, f"{case.slug}/OGM/T={shots}")
        if not math.isclose(objective, float(source["ogm_objective"]), rel_tol=0.0, abs_tol=3.0e-13):
            raise RuntimeError(f"{case.slug}/OGM objective mismatch")
        payload[f"bases_{shots}"] = bases.astype(np.int8)
        payload[f"counts_{shots}"] = counts.astype(np.int64)
        schedules[shots] = (bases, counts)
        seeds[str(shots)] = int(seed)
    metadata = {
        "profile": PROFILE,
        "method": "OGM",
        "slug": case.slug,
        "source_allocation_seeds": seeds,
        "settings_probabilities_sha256": array_sha256(
            np.column_stack((settings, probabilities))
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path = SCHEDULE_OUTPUT / f"{case.slug}_OGM.npz"
    write_npz(path, payload)
    return path, schedules


def freeze_ap(
    general,
    case,
    observables: np.ndarray,
    weights: np.ndarray,
    offset: float,
    evaluator,
    audit_rows: dict[tuple[str, str, int, int], dict[str, str]],
) -> tuple[Path, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    settings, probabilities = general.ap_exact_setting_distribution(observables, weights)
    distribution_hash = array_sha256(np.column_stack((settings, probabilities)))
    payload: dict[str, np.ndarray] = {
        "settings": settings.astype(np.int8),
        "probabilities": probabilities.astype(np.float64),
    }
    schedules: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    seeds: dict[str, int] = {}
    for repeat in range(REPEATS):
        seed = general.stable_seed(general.BASE_SEED, case.slug, "AP", repeat)
        prefixes = general.sampled_prefixes_from_distribution(
            settings, probabilities, SHOT_BUDGETS, seed
        )
        seeds[str(repeat)] = int(seed)
        for shots in SHOT_BUDGETS:
            bases, counts = prefixes[shots]
            result = general.fixed_schedule_mse(
                evaluator,
                observables,
                weights,
                offset,
                bases,
                counts,
                case.target_expectation,
            )
            source = audit_rows[(case.slug, "AP", shots, repeat)]
            close_metrics(result, source, f"{case.slug}/AP/r={repeat}/T={shots}")
            if source["ap_distribution_hash"] != distribution_hash:
                raise RuntimeError(f"{case.slug}/AP distribution hash mismatch")
            payload[f"bases_r{repeat:02d}_T{shots}"] = bases.astype(np.int8)
            payload[f"counts_r{repeat:02d}_T{shots}"] = counts.astype(np.int64)
            schedules[(repeat, shots)] = (bases, counts)
    metadata = {
        "profile": PROFILE,
        "method": "AP",
        "slug": case.slug,
        "source_schedule_seeds": seeds,
        "distribution_sha256": distribution_hash,
        "schedule_repeats": REPEATS,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path = SCHEDULE_OUTPUT / f"{case.slug}_AP.npz"
    write_npz(path, payload)
    return path, schedules


def append_replicates(
    target: list[dict[str, object]],
    *,
    case,
    method: str,
    shots: int,
    estimates: Sequence[float],
    seeds: Sequence[int | str],
    schedule_repeats: Sequence[int | str],
    schedule_sha256: str,
    protocol: str,
) -> None:
    if len(estimates) != REPEATS or len(seeds) != REPEATS or len(schedule_repeats) != REPEATS:
        raise ValueError("A replay point must contain exactly 50 estimators")
    for repeat, (estimate, seed, schedule_repeat) in enumerate(
        zip(estimates, seeds, schedule_repeats)
    ):
        error = float(estimate) - float(case.target_expectation)
        target.append(
            {
                "benchmark": case.benchmark,
                "instance": case.instance,
                "slug": case.slug,
                "method": method,
                "shots": int(shots),
                "repeat": repeat,
                "schedule_repeat": schedule_repeat,
                "outcome_seed": seed,
                "estimate": float(estimate),
                "error": error,
                "squared_error": error * error,
                "target_expectation": case.target_expectation,
                "target_hash": case.target_hash,
                "state_hash": array_sha256(case.state),
                "schedule_sha256": schedule_sha256,
                "protocol": protocol,
            }
        )


def summarize_instances(
    replicates: Sequence[dict[str, object]],
    analytic_rows: dict[tuple[str, str, int], dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for benchmark in BENCHMARKS:
        for instance in INSTANCES:
            slug = f"{benchmark}4_seed0_{instance}"
            for method in METHODS:
                for shots in SHOT_BUDGETS:
                    selected = [
                        row
                        for row in replicates
                        if row["slug"] == slug
                        and row["method"] == method
                        and int(row["shots"]) == shots
                    ]
                    if len(selected) != REPEATS:
                        raise RuntimeError(f"Incomplete replay point: {slug}/{method}/T={shots}")
                    errors = np.asarray([float(row["error"]) for row in selected])
                    source = analytic_rows[(slug, method, shots)]
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "instance": instance,
                            "slug": slug,
                            "method": method,
                            "shots": shots,
                            "empirical_repeats": REPEATS,
                            "empirical_mse": float(np.mean(np.square(errors))),
                            "empirical_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                            "empirical_mean_error": float(np.mean(errors)),
                            "empirical_error_sample_sd": float(np.std(errors, ddof=1)),
                            "analytic_mse": float(source["mse"]),
                            "analytic_rmse": float(source["rmse"]),
                            "analytic_bias": float(source["bias"]),
                            "analytic_variance": float(source["variance"]),
                            "target_expectation": float(selected[0]["target_expectation"]),
                            "target_hash": selected[0]["target_hash"],
                            "state_hash": selected[0]["state_hash"],
                            "protocol": selected[0]["protocol"],
                        }
                    )
    return rows


def summarize_ensembles(instance_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            for shots in SHOT_BUDGETS:
                selected = [
                    row
                    for row in instance_rows
                    if row["benchmark"] == benchmark
                    and row["method"] == method
                    and int(row["shots"]) == shots
                ]
                if len(selected) != len(INSTANCES):
                    raise RuntimeError(f"Incomplete ensemble point: {benchmark}/{method}/T={shots}")
                empirical_rmse = np.asarray([float(row["empirical_rmse"]) for row in selected])
                empirical_mse = np.asarray([float(row["empirical_mse"]) for row in selected])
                analytic_rmse = np.asarray([float(row["analytic_rmse"]) for row in selected])
                rows.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "shots": shots,
                        "instances": len(INSTANCES),
                        "empirical_repeats_per_instance": REPEATS,
                        "total_empirical_estimators": REPEATS * len(INSTANCES),
                        "mean_instance_empirical_rmse": float(np.mean(empirical_rmse)),
                        "empirical_rmse_instance_sample_sd": float(np.std(empirical_rmse, ddof=1)),
                        "mean_instance_empirical_mse": float(np.mean(empirical_mse)),
                        "pooled_empirical_rmse": float(np.sqrt(np.mean(empirical_mse))),
                        "mean_instance_analytic_rmse": float(np.mean(analytic_rmse)),
                        "analytic_rmse_instance_sample_sd": float(np.std(analytic_rmse, ddof=1)),
                        "rmse_semantics": "arithmetic mean and sample SD of five instance-level 50-estimator empirical RMSE values",
                    }
                )
    return rows


def main() -> None:
    require_frozen_environment()
    general = load_general_module()
    general_manifest, gpd_manifest = validate_source_bindings(general)
    SCHEDULE_OUTPUT.mkdir(parents=True, exist_ok=True)

    source_instances = {
        (row["slug"], row["method"], int(row["shots"])): row
        for row in read_csv(GENERAL_INSTANCE)
        if row["benchmark"] in BENCHMARKS and row["method"] in PAULI_METHODS
    }
    pauli_audit = {
        (row["slug"], row["method"], int(row["shots"]), int(row["schedule_repeat"])): row
        for row in read_csv(GENERAL_PAULI_AUDIT)
        if row["benchmark"] in BENCHMARKS and row["method"] in {"OGM", "SG", "Derand", "AP"}
    }
    gpd_instance_source: dict[tuple[str, str, int], dict[str, str]] = {}
    for benchmark in BENCHMARKS:
        for instance in INSTANCES:
            slug = f"{benchmark}4_seed0_{instance}"
            for row in read_csv(GPD_CASES / slug / "selected_results.csv"):
                gpd_instance_source[(slug, "GPD", int(row["shots"]))] = {
                    "mse": row["mse"],
                    "rmse": row["analytic_rmse"],
                    "bias": row["bias"],
                    "variance": row["variance"],
                }
    analytic_rows = dict(source_instances)
    analytic_rows.update(gpd_instance_source)

    source_gpd_replicates = read_csv(GPD_REPLICATES)
    gpd_lookup: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in source_gpd_replicates:
        if int(row["repeat"]) < REPEATS:
            gpd_lookup.setdefault((row["slug"], int(row["shots"])), []).append(row)

    cases = [case for case in general.load_cases() if case.benchmark in BENCHMARKS]
    replicates: list[dict[str, object]] = []
    schedule_files: list[Path] = []
    schedule_sources: dict[str, str] = {}
    case_records: list[dict[str, object]] = []
    gpd_replicate_hash = sha256(GPD_REPLICATES)

    for case in cases:
        observables, weights, offset = general.split_identity(case.observables, case.weights)
        evaluator = general.StateEvaluator(case.state)
        case_records.append(
            {
                "benchmark": case.benchmark,
                "instance": case.instance,
                "slug": case.slug,
                "target_hash": case.target_hash,
                "state_hash": array_sha256(case.state),
                "target_expectation": case.target_expectation,
            }
        )

        ogm_path, ogm_schedules = freeze_ogm(
            general, case, observables, weights, offset, evaluator, pauli_audit
        )
        ap_path, ap_schedules = freeze_ap(
            general, case, observables, weights, offset, evaluator, pauli_audit
        )
        schedule_files.extend((ogm_path, ap_path))

        fixed_sources: dict[str, tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]]] = {
            "OGM": (ogm_path, ogm_schedules)
        }
        for method in ("SG", "Derand"):
            path = GENERAL_SCHEDULES / f"{case.slug}_{method}.npz"
            expected = general_manifest["cache_hashes"]["schedules"][str(path.relative_to(GENERAL_OUTPUT))]
            if sha256(path) != expected:
                raise RuntimeError(f"Frozen {method} schedule hash mismatch: {path}")
            schedules: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            with np.load(path, allow_pickle=False) as saved:
                for shots in SHOT_BUDGETS:
                    schedules[shots] = (saved[f"bases_{shots}"], saved[f"counts_{shots}"])
            fixed_sources[method] = (path, schedules)
            schedule_sources[str(path.resolve())] = expected

        for shots in SHOT_BUDGETS:
            source = sorted(gpd_lookup[(case.slug, shots)], key=lambda row: int(row["repeat"]))
            if len(source) != REPEATS or [int(row["repeat"]) for row in source] != list(range(REPEATS)):
                raise RuntimeError(f"Incomplete GPD 50-row subset: {case.slug}/T={shots}")
            append_replicates(
                replicates,
                case=case,
                method="GPD",
                shots=shots,
                estimates=[float(row["estimate"]) for row in source],
                seeds=[row["empirical_seed"] for row in source],
                schedule_repeats=["frozen-prefix"] * REPEATS,
                schedule_sha256=gpd_replicate_hash,
                protocol="first 50 estimators from the audited 200-replay frozen-prefix GPD artifact",
            )

            for method in ("OGM", "SG", "Derand"):
                path, schedules = fixed_sources[method]
                bases, counts = schedules[shots]
                source_row = pauli_audit[(case.slug, method, shots, 0)]
                exact = general.fixed_schedule_mse(
                    evaluator,
                    observables,
                    weights,
                    offset,
                    bases,
                    counts,
                    case.target_expectation,
                )
                close_metrics(exact, source_row, f"{case.slug}/{method}/T={shots}")
                seeds = [
                    stable_seed(PROFILE, BASE_SEED, case.slug, method, shots, repeat)
                    for repeat in range(REPEATS)
                ]
                estimates = [
                    fixed_schedule_replay(
                        general, evaluator, observables, weights, offset, bases, counts, seed
                    )
                    for seed in seeds
                ]
                append_replicates(
                    replicates,
                    case=case,
                    method=method,
                    shots=shots,
                    estimates=estimates,
                    seeds=seeds,
                    schedule_repeats=[0] * REPEATS,
                    schedule_sha256=sha256(path),
                    protocol="50 independent ideal Born replays of one frozen integer schedule",
                )

            lcs_source = source_instances[(case.slug, "LCS", shots)]
            lcs_settings, lcs_joint, lcs_scores = lcs_joint_law(
                general, evaluator, observables, weights
            )
            exact_mean = offset + float(lcs_joint @ lcs_scores)
            exact_variance = float(
                lcs_joint @ np.square(lcs_scores) - (lcs_joint @ lcs_scores) ** 2
            ) / shots
            if not math.isclose(exact_mean, case.target_expectation, rel_tol=0.0, abs_tol=3.0e-12):
                raise RuntimeError(f"{case.slug}/LCS expectation mismatch")
            if not math.isclose(exact_variance, float(lcs_source["variance"]), rel_tol=0.0, abs_tol=3.0e-12):
                raise RuntimeError(f"{case.slug}/LCS variance mismatch")
            lcs_law_hash = array_sha256(np.column_stack((lcs_joint, lcs_scores)))
            seeds = [
                stable_seed(PROFILE, BASE_SEED, case.slug, "LCS", shots, repeat)
                for repeat in range(REPEATS)
            ]
            estimates = [
                lcs_replay(lcs_joint, lcs_scores, offset, shots, seed) for seed in seeds
            ]
            append_replicates(
                replicates,
                case=case,
                method="LCS",
                shots=shots,
                estimates=estimates,
                seeds=seeds,
                schedule_repeats=list(range(REPEATS)),
                schedule_sha256=lcs_law_hash,
                protocol="50 independent uniform-local-Pauli setting and ideal Born-outcome trials",
            )

            ap_estimates: list[float] = []
            ap_seeds: list[int] = []
            for repeat in range(REPEATS):
                bases, counts = ap_schedules[(repeat, shots)]
                seed = stable_seed(PROFILE, BASE_SEED, case.slug, "AP", shots, repeat)
                ap_seeds.append(seed)
                ap_estimates.append(
                    fixed_schedule_replay(
                        general, evaluator, observables, weights, offset, bases, counts, seed
                    )
                )
            append_replicates(
                replicates,
                case=case,
                method="AP",
                shots=shots,
                estimates=ap_estimates,
                seeds=ap_seeds,
                schedule_repeats=list(range(REPEATS)),
                schedule_sha256=sha256(ap_path),
                protocol="one independent ideal Born replay for each of 50 frozen AP prefix schedules",
            )

    expected_replicates = len(BENCHMARKS) * len(INSTANCES) * len(METHODS) * len(SHOT_BUDGETS) * REPEATS
    if len(replicates) != expected_replicates:
        raise RuntimeError(f"Expected {expected_replicates} replicates, got {len(replicates)}")
    instance_rows = summarize_instances(replicates, analytic_rows)
    summary_rows = summarize_ensembles(instance_rows)
    write_csv(REPLICATE_OUTPUT, replicates)
    write_csv(INSTANCE_OUTPUT, instance_rows)
    write_csv(SUMMARY_OUTPUT, summary_rows)

    manifest = {
        "schema": 1,
        "status": "COMPLETE",
        "profile": PROFILE,
        "methods": list(METHODS),
        "benchmarks": list(BENCHMARKS),
        "instances_per_benchmark": len(INSTANCES),
        "shot_budgets": list(SHOT_BUDGETS),
        "empirical_repeats_per_instance_budget": REPEATS,
        "base_seed": BASE_SEED,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "semantics": {
            "GPD": "first 50 estimators from the committed 200-replay artifact",
            "OGM": "50 Born replays of the frozen T-specific archived-rule allocation",
            "SG": "50 Born replays of the frozen max-T schedule prefix",
            "Derand": "50 Born replays of the frozen max-T schedule prefix",
            "LCS": "50 independent uniform setting plus Born-outcome trials",
            "AP": "one Born replay for each of the 50 frozen AP schedule prefixes",
        },
        "case_records": case_records,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "source_sha256": {
            str(GENERAL_MANIFEST.resolve()): sha256(GENERAL_MANIFEST),
            str(GENERAL_INSTANCE.resolve()): sha256(GENERAL_INSTANCE),
            str(GENERAL_PAULI_AUDIT.resolve()): sha256(GENERAL_PAULI_AUDIT),
            str(GPD_MANIFEST.resolve()): sha256(GPD_MANIFEST),
            str(GPD_AUDIT.resolve()): sha256(GPD_AUDIT),
            str(GPD_COMMIT.resolve()): sha256(GPD_COMMIT),
            str(GPD_REPLICATES.resolve()): sha256(GPD_REPLICATES),
            str(GPD_INSTANCES.resolve()): sha256(GPD_INSTANCES),
            **schedule_sources,
        },
        "materialized_schedule_sha256": {
            str(path.resolve()): sha256(path) for path in schedule_files
        },
        "outputs_sha256": {
            path.name: sha256(path)
            for path in (REPLICATE_OUTPUT, INSTANCE_OUTPUT, SUMMARY_OUTPUT)
        },
        "row_counts": {
            REPLICATE_OUTPUT.name: len(replicates),
            INSTANCE_OUTPUT.name: len(instance_rows),
            SUMMARY_OUTPUT.name: len(summary_rows),
        },
    }
    write_json(MANIFEST_OUTPUT, manifest)
    print(json.dumps({"status": "COMPLETE", "row_counts": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
