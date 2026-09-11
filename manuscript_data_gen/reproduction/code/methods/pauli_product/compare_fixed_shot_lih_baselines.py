#!/usr/bin/env python3
"""Compare sampling the selected Hybrid-F Hamiltonian with OGM and SG.

The comparison uses the same LiH CAS(2e,4o) target, the same exact
fixed-particle ground state, the same total number of state preparations, and
independent ideal Born sampling.  Hybrid-F reuses the audited K(T) prefixes and
400-repetition samples.  OGM and Shadow Grouping measure the exact Pauli
Hamiltonian with the repository's retained measurement-design and estimator
rules; OGM probabilities use the same minimum-one plus largest-remainder
fixed-T integerization as Hybrid-F.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from release_paths import BUNDLE_ROOT, DATA_ROOT, OUTPUT_ROOT
WORKSPACE = BUNDLE_ROOT
DEFAULT_HYBRID = DATA_ROOT / "LiH" / "results" / "stabilizer_calibration"
DEFAULT_OUTPUT = OUTPUT_ROOT / "lih_baseline_comparison"
LEGACY_DRIVER = HERE / "run_lih_r1p50_sampling_comparison.py"
DEFAULT_BUDGETS = (100, 300, 1000, 3000, 10_000, 30_000, 100_000)
METHOD_ORDER = ("Hybrid-F", "OGM", "SG")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_legacy_driver():
    spec = importlib.util.spec_from_file_location("lih_pauli_baselines", LEGACY_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {LEGACY_DRIVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dense_to_pauli(dense: np.ndarray, threshold: float):
    dense = np.asarray(dense, dtype=np.complex128)
    hermiticity_error = float(np.linalg.norm(dense - dense.conj().T, "fro"))
    if hermiticity_error > 1.0e-10:
        raise RuntimeError(
            f"Dense Hamiltonian is not Hermitian: Frobenius error {hermiticity_error}"
        )
    dense = 0.5 * (dense + dense.conj().T)
    num_qubits = int(round(math.log2(dense.shape[0])))
    if dense.shape != (1 << num_qubits, 1 << num_qubits):
        raise ValueError(f"Invalid dense Hamiltonian shape {dense.shape}")
    local_paulis = np.asarray(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=np.complex128,
    )
    local_transform = np.transpose(local_paulis, (0, 2, 1)) / 2.0
    dense_tensor = dense.reshape((2,) * (2 * num_qubits))
    dense_subscripts = list(range(2 * num_qubits))
    arguments: list[Any] = [dense_tensor, dense_subscripts]
    output_subscripts = []
    for site in range(num_qubits):
        pauli_axis = 2 * num_qubits + site
        arguments.extend(
            [local_transform, [pauli_axis, site, num_qubits + site]]
        )
        output_subscripts.append(pauli_axis)
    coefficients = np.einsum(*arguments, output_subscripts, optimize="greedy")
    max_imaginary = float(np.max(np.abs(coefficients.imag)))
    if max_imaginary > 1.0e-9:
        raise RuntimeError(f"Pauli coefficients are complex: {max_imaginary}")
    coefficients = coefficients.real
    full_pauli_norm_sq = float((1 << num_qubits) * np.sum(coefficients**2))
    dense_norm_sq = float(np.linalg.norm(dense, "fro") ** 2)
    active_indices = np.argwhere(np.abs(coefficients) > threshold)
    observables = np.asarray(active_indices, dtype=np.int8)
    weights = np.asarray(
        [coefficients[tuple(index)] for index in active_indices], dtype=float
    )
    identity = np.all(observables == 0, axis=1)
    if np.count_nonzero(identity) != 1:
        raise RuntimeError("Expected exactly one retained identity Pauli term")
    reconstructed = np.zeros_like(dense)
    for observable, weight in zip(observables, weights):
        matrix = local_paulis[int(observable[0])]
        for pauli in observable[1:]:
            matrix = np.kron(matrix, local_paulis[int(pauli)])
        reconstructed += weight * matrix
    reconstruction_error = float(np.linalg.norm(dense - reconstructed, "fro"))
    audit = {
        "number_qubits": num_qubits,
        "threshold_hartree": threshold,
        "retained_terms_including_identity": int(len(observables)),
        "maximum_imaginary_coefficient": max_imaginary,
        "input_hermiticity_frobenius_error_hartree": hermiticity_error,
        "retained_pauli_dense_reconstruction_frobenius_error_hartree": reconstruction_error,
        "dense_frobenius_norm_squared": dense_norm_sq,
        "complete_pauli_frobenius_norm_squared": full_pauli_norm_sq,
        "norm_squared_difference_before_threshold": full_pauli_norm_sq
        - dense_norm_sq,
    }
    return observables[~identity], weights[~identity], float(weights[identity][0]), audit


def minimum_one_probability_schedule(
    settings: np.ndarray, probabilities: np.ndarray, shots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Use every positive-probability OGM basis once, then largest remainder."""
    active = np.flatnonzero(probabilities > 1.0e-15)
    if shots < len(active):
        raise ValueError(
            f"OGM needs at least {len(active)} shots for minimum-one allocation, got {shots}"
        )
    normalized = probabilities[active] / np.sum(probabilities[active])
    remaining = shots - len(active)
    raw = remaining * normalized
    extra = np.floor(raw).astype(np.int64)
    deficit = int(remaining - np.sum(extra))
    if deficit:
        order = np.argsort(-(raw - extra), kind="stable")
        extra[order[:deficit]] += 1
    counts = np.zeros(len(settings), dtype=np.int64)
    counts[active] = 1 + extra
    schedule = np.repeat(settings, counts, axis=0).astype(np.int8, copy=False)
    if len(schedule) != shots:
        raise RuntimeError(f"OGM allocation produced {len(schedule)} of {shots} shots")
    schedule[schedule == 0] = 3
    return schedule, counts


def exact_sector_ground(
    target: np.ndarray, sector_indices: np.ndarray
) -> tuple[np.ndarray, float]:
    block = target[np.ix_(sector_indices, sector_indices)]
    eigenvalues, eigenvectors = np.linalg.eigh(block)
    state = np.zeros(target.shape[0], dtype=np.complex128)
    state[sector_indices] = eigenvectors[:, 0]
    return state, float(eigenvalues[0])


def analytic_fixed_schedule(
    legacy,
    schedule: np.ndarray,
    observables: np.ndarray,
    weights: np.ndarray,
    offset: float,
    cache,
) -> dict[str, Any]:
    pattern_counts = legacy.packed_hit_patterns(schedule, observables)
    hit_counts = np.zeros(len(observables), dtype=np.int64)
    selected_by_pattern: dict[bytes, np.ndarray] = {}
    for key, count in pattern_counts.items():
        selected = legacy.unpack_pattern(key, len(observables))
        selected_by_pattern[key] = selected
        hit_counts[selected] += count

    mean = float(offset)
    variance = 0.0
    for key, count in pattern_counts.items():
        if not np.any(selected_by_pattern[key]):
            continue
        indices, probabilities, signs = cache.get(key)
        coefficients = weights[indices] / hit_counts[indices]
        category_energy = signs @ coefficients
        pattern_mean = float(probabilities @ category_energy)
        pattern_second = float(probabilities @ (category_energy**2))
        mean += count * pattern_mean
        variance += count * max(pattern_second - pattern_mean**2, 0.0)
    return {
        "mean_energy": mean,
        "sampling_variance": float(variance),
        "hit_counts": hit_counts,
        "distinct_hit_patterns": int(len(pattern_counts)),
    }


def bootstrap_rmse(
    errors: np.ndarray, seed: int, resamples: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(errors), size=(resamples, len(errors)))
    values = np.sqrt(np.mean(errors[draws] ** 2, axis=1))
    lower, upper = np.quantile(values, (0.025, 0.975))
    return float(lower), float(upper)


def empirical_fields(
    sampling_errors: np.ndarray,
    total_errors: np.ndarray,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, float]:
    lower, upper = bootstrap_rmse(total_errors, seed, bootstrap_resamples)
    return {
        "empirical_sampling_bias_hartree": float(np.mean(sampling_errors)),
        "empirical_sampling_RMSE_hartree": float(
            np.sqrt(np.mean(sampling_errors**2))
        ),
        "empirical_total_bias_hartree": float(np.mean(total_errors)),
        "empirical_total_MAE_hartree": float(np.mean(np.abs(total_errors))),
        "empirical_total_RMSE_hartree": float(np.sqrt(np.mean(total_errors**2))),
        "empirical_total_RMSE_bootstrap95_low_hartree": lower,
        "empirical_total_RMSE_bootstrap95_high_hartree": upper,
    }


def load_hybrid_rows(
    hybrid_dir: Path,
    budgets: list[int],
    repeats: int,
    bootstrap_resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    summary_path = hybrid_dir / "LiH_K_and_ground_sampling_error_by_T.csv"
    repeat_path = hybrid_dir / "LiH_ground_sampling_replicates.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summaries = {int(row["T_total_shots"]): row for row in csv.DictReader(handle)}
    repeat_groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    with repeat_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            repeat_groups[int(row["T_total_shots"])].append(row)

    rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    ground_energy: float | None = None
    for point, shots in enumerate(budgets):
        if shots not in summaries or shots not in repeat_groups:
            raise ValueError(f"Hybrid artifact has no saved T={shots}")
        source = summaries[shots]
        if str(source["right_censored_at_Kmax"]).lower() == "true":
            raise RuntimeError(
                f"Hybrid T={shots} is right-censored at its K guard; extend the "
                "trajectory before constructing comparison data"
            )
        available = sorted(repeat_groups[shots], key=lambda row: int(row["repeat"]))
        if len(available) < repeats:
            raise ValueError(
                f"Hybrid T={shots} has {len(available)} repeats, requested {repeats}"
            )
        selected = available[:repeats]
        estimates = np.asarray([float(row["energy_estimate"]) for row in selected])
        sampling_errors = np.asarray(
            [float(row["sampling_error_about_approximation"]) for row in selected]
        )
        total_errors = np.asarray(
            [float(row["total_error_about_exact_ground_energy"]) for row in selected]
        )
        current_ground = float(source["exact_ground_energy"])
        if ground_energy is None:
            ground_energy = current_ground
        elif abs(current_ground - ground_energy) > 1.0e-12:
            raise RuntimeError("Hybrid reference energy changes across shot budgets")
        bias = float(source["signed_ground_state_approximation_error"])
        analytic_sampling = float(source["ground_state_sampling_standard_error"])
        row = {
            "method": "Hybrid-F",
            "T_total_shots": shots,
            "T_actual_shots": shots,
            "selected_K": int(source["K"]),
            "right_censored": False,
            "deterministic_bias_hartree": bias,
            "deterministic_bias_type": "approximate-H expectation minus exact-H expectation",
            "analytic_sampling_SE_hartree": analytic_sampling,
            "analytic_total_RMSE_hartree": math.hypot(bias, analytic_sampling),
            "empirical_mean_energy_hartree": float(np.mean(estimates)),
            **empirical_fields(
                sampling_errors,
                total_errors,
                891_000 + point,
                bootstrap_resamples,
            ),
            "repeat_count": repeats,
            "measurement_settings_or_distinct_bases": int(
                source["total_measurement_settings"]
            ),
            "minimum_basis_or_setting_shots": int(
                source["allocation_minimum_nonzero_shots"]
            ),
            "maximum_basis_or_setting_shots": int(
                source["allocation_maximum_shots"]
            ),
            "covered_pauli_terms": "",
            "unhit_pauli_terms": "",
            "minimum_pauli_term_hits": "",
            "maximum_pauli_term_hits": "",
            "distinct_hit_patterns": "",
            "allocation_conserved": source["allocation_conserved"].lower()
            == "true",
        }
        rows.append(row)
        for repeat, (estimate, sample_error, total_error) in enumerate(
            zip(estimates, sampling_errors, total_errors)
        ):
            repeat_rows.append(
                {
                    "method": "Hybrid-F",
                    "T_total_shots": shots,
                    "selected_K": int(source["K"]),
                    "repeat": repeat,
                    "energy_estimate_hartree": float(estimate),
                    "sampling_error_about_method_mean_hartree": float(sample_error),
                    "total_error_about_exact_energy_hartree": float(total_error),
                }
            )
    assert ground_energy is not None
    return rows, repeat_rows, ground_energy


def plot_comparison(output: Path, rows: list[dict[str, Any]]) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    colors = {"Hybrid-F": "#2563eb", "OGM": "#d97706", "SG": "#16a34a"}
    markers = {"Hybrid-F": "D", "OGM": "o", "SG": "s"}
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        shots = np.asarray([row["T_total_shots"] for row in method_rows], dtype=float)
        empirical = np.asarray(
            [row["empirical_total_RMSE_hartree"] for row in method_rows]
        )
        lower = np.asarray(
            [row["empirical_total_RMSE_bootstrap95_low_hartree"] for row in method_rows]
        )
        upper = np.asarray(
            [row["empirical_total_RMSE_bootstrap95_high_hartree"] for row in method_rows]
        )
        analytic = np.asarray(
            [row["analytic_total_RMSE_hartree"] for row in method_rows]
        )
        axis.plot(
            shots,
            empirical,
            color=colors[method],
            marker=markers[method],
            linewidth=2.0,
            markersize=5,
            label=f"{method} empirical",
        )
        axis.fill_between(shots, lower, upper, color=colors[method], alpha=0.12)
        axis.plot(
            shots,
            analytic,
            color=colors[method],
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            label=f"{method} analytic",
        )
    hybrid_rows = [row for row in rows if row["method"] == "Hybrid-F"]
    previous_k = None
    for row in hybrid_rows:
        k = int(row["selected_K"])
        if k != previous_k:
            suffix = "+" if str(row["right_censored"]).lower() == "true" else ""
            axis.annotate(
                f"K={k}{suffix}",
                (row["T_total_shots"], row["empirical_total_RMSE_hartree"]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=8,
            )
        previous_k = k
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Total state preparations / shots T")
    axis.set_ylabel("Total energy RMSE (Ha)")
    axis.set_title("LiH fixed-shot sampling: Hybrid-F vs OGM and SG")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(figure_dir / "LiH_fixed_shot_total_RMSE_comparison.png", dpi=240)
    figure.savefig(figure_dir / "LiH_fixed_shot_total_RMSE_comparison.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharex=True)
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        shots = np.asarray([row["T_total_shots"] for row in method_rows], dtype=float)
        sampling = np.asarray(
            [row["analytic_sampling_SE_hartree"] for row in method_rows]
        )
        axes[0].plot(
            shots,
            sampling,
            color=colors[method],
            marker=markers[method],
            label=method,
        )
        if method == "Hybrid-F":
            axes[1].plot(
                shots,
                np.abs(
                    [row["deterministic_bias_hartree"] for row in method_rows]
                ),
                color=colors[method],
                marker=markers[method],
                label=method,
            )
    axes[1].text(
        0.96,
        0.92,
        "OGM / SG: zero coverage bias",
        transform=axes[1].transAxes,
        fontsize=8,
        ha="right",
        va="top",
    )
    for axis, ylabel, title in (
        (axes[0], "Analytic sampling SE (Ha)", "Sampling component"),
        (axes[1], "Absolute deterministic bias (Ha)", "Approximation / coverage bias"),
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Total shots T")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "LiH_fixed_shot_error_components.png", dpi=240)
    figure.savefig(figure_dir / "LiH_fixed_shot_error_components.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-dir", type=Path, default=DEFAULT_HYBRID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--pauli-threshold", type=float, default=1.0e-10)
    args = parser.parse_args()

    budgets = sorted(set(args.budgets))
    if not budgets or min(budgets) <= 0:
        raise ValueError("Shot budgets must be positive")
    if args.repeats <= 1 or args.bootstrap_resamples <= 0:
        raise ValueError("Need at least two repeats and positive bootstrap resamples")
    hybrid_dir = args.hybrid_dir.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix with nonempty output directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "schedules").mkdir(exist_ok=True)
    started = time.time()

    legacy = load_legacy_driver()
    operator_path = hybrid_dir / "LiH_active_case_operators.npz"
    with np.load(operator_path) as data:
        target = np.asarray(data["target"], dtype=np.complex128)
        sector_indices = np.asarray(data["sector_indices"], dtype=np.int64)
    ground, exact_energy = exact_sector_ground(target, sector_indices)
    observables, weights, offset, pauli_audit = dense_to_pauli(
        target, args.pauli_threshold
    )
    pauli_energy, term_expectations = legacy.audit_energy(
        observables, weights, offset, ground
    )
    pauli_energy_error = pauli_energy - exact_energy
    if abs(pauli_energy_error) > 1.0e-9:
        raise RuntimeError(f"Pauli energy audit failed: {pauli_energy_error}")
    np.savez_compressed(
        output / "LiH_pauli_ground_input.npz",
        observables=observables,
        weights=weights,
        offset=np.asarray(offset),
        ground_state=ground,
        exact_energy=np.asarray(exact_energy),
        sector_indices=sector_indices,
        term_expectations=term_expectations,
    )

    summary_rows, repeat_rows, hybrid_energy = load_hybrid_rows(
        hybrid_dir, budgets, args.repeats, args.bootstrap_resamples
    )
    if abs(hybrid_energy - exact_energy) > 1.0e-11:
        raise RuntimeError(
            f"Hybrid and baseline reference energies differ by {hybrid_energy-exact_energy}"
        )

    print(
        f"LiH target: {len(observables)+1} Pauli terms, E0={exact_energy:.15f}",
        flush=True,
    )
    print("Optimizing OGM distribution...", flush=True)
    ogm_settings, ogm_probabilities, ogm_info = legacy.optimize_ogm(
        observables, weights
    )
    np.savez_compressed(
        output / "ogm_distribution.npz",
        settings=ogm_settings,
        probabilities=ogm_probabilities,
    )
    print(f"Generating SG schedule through T={max(budgets)}...", flush=True)
    sg_schedule = legacy.shadow_grouping_schedule(observables, weights, max(budgets))
    np.savez_compressed(output / "schedules" / "SG_max.npz", settings=sg_schedule)

    cache = legacy.JointPatternCache(observables, ground)
    schedule_lengths_ok = True
    analytic_nonnegative = True
    all_pauli_terms_covered = True
    for method_index, method in enumerate(("OGM", "SG"), start=1):
        for point, shots in enumerate(budgets):
            point_start = time.time()
            if method == "OGM":
                schedule, _ = minimum_one_probability_schedule(
                    ogm_settings, ogm_probabilities, shots
                )
                np.savez_compressed(
                    output / "schedules" / f"OGM_T{shots}.npz", settings=schedule
                )
            else:
                schedule = sg_schedule[:shots]
            schedule_lengths_ok &= len(schedule) == shots
            analytic = analytic_fixed_schedule(
                legacy, schedule, observables, weights, offset, cache
            )
            energies, hit_counts, pattern_count = legacy.sample_fixed_schedule(
                schedule,
                observables,
                weights,
                offset,
                args.repeats,
                args.seed + method_index * 1_000_000 + shots,
                cache,
            )
            if not np.array_equal(hit_counts, analytic["hit_counts"]):
                raise RuntimeError(f"{method} T={shots}: hit-count replay mismatch")
            if pattern_count != analytic["distinct_hit_patterns"]:
                raise RuntimeError(f"{method} T={shots}: pattern replay mismatch")
            deterministic_bias = float(analytic["mean_energy"] - exact_energy)
            sampling_variance = float(analytic["sampling_variance"])
            analytic_nonnegative &= sampling_variance >= -1.0e-14
            sampling_variance = max(sampling_variance, 0.0)
            sampling_errors = energies - analytic["mean_energy"]
            total_errors = energies - exact_energy
            row = {
                "method": method,
                "T_total_shots": shots,
                "T_actual_shots": int(len(schedule)),
                "selected_K": "",
                "right_censored": False,
                "deterministic_bias_hartree": deterministic_bias,
                "deterministic_bias_type": "covered retained-Pauli estimator minus exact target energy; numerical zero after complete coverage",
                "analytic_sampling_SE_hartree": math.sqrt(sampling_variance),
                "analytic_total_RMSE_hartree": math.hypot(
                    deterministic_bias, math.sqrt(sampling_variance)
                ),
                "empirical_mean_energy_hartree": float(np.mean(energies)),
                **empirical_fields(
                    sampling_errors,
                    total_errors,
                    args.seed + 8_000_000 + method_index * 1000 + point,
                    args.bootstrap_resamples,
                ),
                "repeat_count": args.repeats,
                "measurement_settings_or_distinct_bases": legacy.distinct_rows(
                    schedule
                ),
                "minimum_basis_or_setting_shots": int(
                    np.unique(schedule, axis=0, return_counts=True)[1].min()
                ),
                "maximum_basis_or_setting_shots": int(
                    np.unique(schedule, axis=0, return_counts=True)[1].max()
                ),
                "covered_pauli_terms": int(np.count_nonzero(hit_counts)),
                "unhit_pauli_terms": int(np.count_nonzero(hit_counts == 0)),
                "minimum_pauli_term_hits": int(hit_counts.min()),
                "maximum_pauli_term_hits": int(hit_counts.max()),
                "distinct_hit_patterns": int(pattern_count),
                "allocation_conserved": int(len(schedule)) == shots,
            }
            all_pauli_terms_covered &= row["unhit_pauli_terms"] == 0
            summary_rows.append(row)
            for repeat, (estimate, sample_error, total_error) in enumerate(
                zip(energies, sampling_errors, total_errors)
            ):
                repeat_rows.append(
                    {
                        "method": method,
                        "T_total_shots": shots,
                        "selected_K": "",
                        "repeat": repeat,
                        "energy_estimate_hartree": float(estimate),
                        "sampling_error_about_method_mean_hartree": float(sample_error),
                        "total_error_about_exact_energy_hartree": float(total_error),
                    }
                )
            print(
                f"{method:3s} T={shots:7d} analytic={row['analytic_total_RMSE_hartree']:.7g} "
                f"empirical={row['empirical_total_RMSE_hartree']:.7g} "
                f"unhit={row['unhit_pauli_terms']} "
                f"({time.time()-point_start:.2f}s)",
                flush=True,
            )

    summary_rows.sort(key=lambda row: (int(row["T_total_shots"]), METHOD_ORDER.index(row["method"])))
    hybrid_by_shots = {
        int(row["T_total_shots"]): row
        for row in summary_rows
        if row["method"] == "Hybrid-F"
    }
    for row in summary_rows:
        reference = hybrid_by_shots[int(row["T_total_shots"])]
        row["analytic_RMSE_ratio_to_Hybrid_F"] = float(
            row["analytic_total_RMSE_hartree"]
            / reference["analytic_total_RMSE_hartree"]
        )
        row["empirical_RMSE_ratio_to_Hybrid_F"] = float(
            row["empirical_total_RMSE_hartree"]
            / reference["empirical_total_RMSE_hartree"]
        )
    repeat_rows.sort(
        key=lambda row: (
            int(row["T_total_shots"]),
            METHOD_ORDER.index(row["method"]),
            int(row["repeat"]),
        )
    )
    winner_rows = []
    for shots in budgets:
        candidates = [row for row in summary_rows if row["T_total_shots"] == shots]
        empirical_winner = min(candidates, key=lambda row: row["empirical_total_RMSE_hartree"])
        analytic_winner = min(candidates, key=lambda row: row["analytic_total_RMSE_hartree"])
        winner_rows.append(
            {
                "T_total_shots": shots,
                "empirical_RMSE_winner": empirical_winner["method"],
                "empirical_winner_RMSE_hartree": empirical_winner[
                    "empirical_total_RMSE_hartree"
                ],
                "analytic_RMSE_winner": analytic_winner["method"],
                "analytic_winner_RMSE_hartree": analytic_winner[
                    "analytic_total_RMSE_hartree"
                ],
                "winner_scope": "minimum among evaluated rows",
                "Hybrid_F_right_censored": str(
                    hybrid_by_shots[shots]["right_censored"]
                ).lower()
                == "true",
            }
        )

    write_csv(output / "fixed_shot_error_summary.csv", summary_rows)
    write_csv(output / "fixed_shot_sampling_replicates.csv", repeat_rows)
    write_csv(output / "fixed_shot_winners.csv", winner_rows)
    plot_comparison(output, summary_rows)
    output_files = [
        output / "fixed_shot_error_summary.csv",
        output / "fixed_shot_sampling_replicates.csv",
        output / "fixed_shot_winners.csv",
        output / "LiH_pauli_ground_input.npz",
        output / "ogm_distribution.npz",
        output / "figures" / "LiH_fixed_shot_total_RMSE_comparison.png",
        output / "figures" / "LiH_fixed_shot_total_RMSE_comparison.pdf",
        output / "figures" / "LiH_fixed_shot_error_components.png",
        output / "figures" / "LiH_fixed_shot_error_components.pdf",
    ] + sorted((output / "schedules").glob("*.npz"))
    all_allocations_conserved = all(
        bool(row["allocation_conserved"]) for row in summary_rows
    )
    with (hybrid_dir / "LiH_K_by_T_candidate_grid.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        grid_rows = list(csv.DictReader(handle))
    grid_argmin = {}
    grid_max_k = {}
    for shots in budgets:
        candidates = [
            row for row in grid_rows if int(row["T_total_shots"]) == shots
        ]
        if not candidates:
            raise RuntimeError(f"No Hybrid candidate-grid rows for T={shots}")
        grid_argmin[shots] = int(
            min(candidates, key=lambda row: float(row["predicted_total_RMSE"]))["K"]
        )
        grid_max_k[shots] = max(int(row["K"]) for row in candidates)
    hybrid_k_argmin_ok = all(
        int(hybrid_by_shots[shots]["selected_K"]) == grid_argmin[shots]
        for shots in budgets
    )
    hybrid_censor_flags_ok = all(
        (str(hybrid_by_shots[shots]["right_censored"]).lower() == "true")
        == (grid_argmin[shots] == grid_max_k[shots])
        for shots in budgets
    )
    audit = {
        "status": "PASS"
        if (
            abs(pauli_energy_error) <= 1.0e-9
            and schedule_lengths_ok
            and analytic_nonnegative
            and all_pauli_terms_covered
            and all_allocations_conserved
            and pauli_audit[
                "retained_pauli_dense_reconstruction_frobenius_error_hartree"
            ]
            <= 1.0e-9
            and hybrid_k_argmin_ok
            and hybrid_censor_flags_ok
            and len(summary_rows) == len(METHOD_ORDER) * len(budgets)
            and len(repeat_rows)
            == len(METHOD_ORDER) * len(budgets) * args.repeats
        )
        else "FAIL",
        "pauli_energy_audit_error_hartree": pauli_energy_error,
        "all_schedule_lengths_equal_requested_shots": bool(schedule_lengths_ok),
        "all_analytic_sampling_variances_nonnegative": bool(analytic_nonnegative),
        "all_retained_nonidentity_pauli_terms_covered": bool(
            all_pauli_terms_covered
        ),
        "all_allocations_conserved": bool(all_allocations_conserved),
        "hybrid_K_equals_predicted_loss_argmin": bool(hybrid_k_argmin_ok),
        "hybrid_right_censor_flags_match_saved_frontier": bool(
            hybrid_censor_flags_ok
        ),
        "summary_row_count": len(summary_rows),
        "expected_summary_row_count": len(METHOD_ORDER) * len(budgets),
        "repeat_row_count": len(repeat_rows),
        "expected_repeat_row_count": len(METHOD_ORDER)
        * len(budgets)
        * args.repeats,
    }
    write_json(output / "audit.json", audit)
    output_files.append(output / "audit.json")
    hybrid_provenance = [
        hybrid_dir / "manifest.json",
        hybrid_dir / "audit.json",
        hybrid_dir / "stabilizer_weights.json",
        hybrid_dir / "LiH_K_by_T_candidate_grid.csv",
        hybrid_dir / "LiH_dynamic_shot_allocations_by_T_K.csv",
        hybrid_dir / "dynamic_depth_fragments.npz",
        hybrid_dir / "dynamic_prefix_setting_ranges.csv",
        hybrid_dir.parent / "stabilizer_calibration_lih.py",
    ]
    manifest = {
        "definition_version": "lih-cas-fixed-shot-hybrid-f-vs-ogm-sg-v5",
        "created_unix_time": time.time(),
        "wall_seconds": time.time() - started,
        "molecule": "LiH",
        "bond_length_angstrom": 1.5,
        "active_space": "CAS(2e,4o), 8 qubits",
        "reference_state": "exact ground state in the saved two-electron sector",
        "reference_energy_hartree": exact_energy,
        "shot_budgets": budgets,
        "repeat_count_per_method_and_budget": args.repeats,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "command_line": [str(Path(__file__).resolve()), *sys.argv[1:]],
        },
        "comparison": (
            "Same target, state, and inference-time state preparations. Offline "
            "training and measurement-design costs are excluded. Hybrid-F samples its "
            "stabilizer-selected approximate Hamiltonian and is evaluated relative to "
            "the exact-H energy. OGM and SG sample the exact Pauli Hamiltonian with the "
            "repository's retained fixed-schedule estimators."
        ),
        "method_definitions": {
            "Hybrid-F": "isolated F-dominant pilot; every selected endpoint is GFRO-F; saved K(T)<=24, F3-R2 settings, minimum-one proportional-range shots",
            "OGM": "OGM-style fixed pooled-hit estimator: main_CutOGM greedy QWC candidates, SLSQP coverage objective, minimum-one plus largest-remainder fixed-T schedule; not iid/HT OGM",
            "SG": "Shadow Grouping fixed-prefix pooled-hit estimator with Bernstein weight and epsilon=0.1",
        },
        "error_definition": {
            "sampling_only": "estimate minus the exact expectation of the method's measured operator/covered estimator",
            "deterministic_bias": "method mean minus exact target-H ground energy",
            "total": "estimate minus exact target-H ground energy",
            "RMSE": "root mean squared total error over independent ideal Born repetitions",
        },
        "pauli_decomposition_audit": pauli_audit,
        "pauli_energy_audit_error_hartree": pauli_energy_error,
        "ogm_optimizer": ogm_info,
        "inputs": {
            str(operator_path): sha256(operator_path),
            str(hybrid_dir / "LiH_K_and_ground_sampling_error_by_T.csv"): sha256(
                hybrid_dir / "LiH_K_and_ground_sampling_error_by_T.csv"
            ),
            str(hybrid_dir / "LiH_ground_sampling_replicates.csv"): sha256(
                hybrid_dir / "LiH_ground_sampling_replicates.csv"
            ),
            str(LEGACY_DRIVER): sha256(LEGACY_DRIVER),
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            **{str(path): sha256(path) for path in hybrid_provenance},
        },
        "output_hashes": {
            str(path.relative_to(output)): sha256(path) for path in output_files
        },
        "audit": audit,
    }
    write_json(output / "manifest.json", manifest)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Comparison audit failed: {audit}")
    print(f"PASS: results written to {output}", flush=True)


if __name__ == "__main__":
    main()
