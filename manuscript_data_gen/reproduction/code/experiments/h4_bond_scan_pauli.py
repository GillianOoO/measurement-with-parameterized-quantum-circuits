#!/usr/bin/env python3
"""Rerun H4 Pauli baselines on the exact GPD/SRDD Hamiltonian artifacts.

The bond scan uses the 21 dense ``H4_R*.npz`` archives used by the current
GPD and SRDD scans.  The fixed-geometry run uses the dense R=1.20 Angstrom
archive used by the current GPD shot scan.  OGM, shadow grouping (SG), and
derandomization use the repository implementations and the pooled-hit energy
estimator retained by the manuscript comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "methods" / "srdd"))
from srdd_release_paths import DATA, METHODS as METHOD_ROOT, RUNS

SCAN_INPUT = DATA / "H4" / "inputs" / "bond_scan"
FIXED_INPUT = DATA / "H4" / "inputs" / "fixed_R1p2" / "h4_r1p2_dense_ground.npz"
HISTORICAL_SCAN = (
    DATA / "H4" / "results" / "srdd_bond_scan_selected"
    / "figure3a_H4_bond_scan_2038_with_srcdf.csv"
)
PREVIOUS_FIXED = DATA / "H4" / "results" / "pauli_bond_scan" / "fixed_sampling_error_summary.csv"
PREVIOUS_FIXED_INPUT = PREVIOUS_FIXED.parent / "fixed" / "fixed_R1p2" / "same_target_state_input.npz"
PREVIOUS_FIXED_MANIFEST = PREVIOUS_FIXED.parent / "manifest.json"
FIGURE_SCAN_INPUT = HISTORICAL_SCAN
DEFAULT_OUTPUT = RUNS / "h4_pauli"
LEGACY_DRIVER = METHOD_ROOT / "pauli_product" / "run_lih_r1p50_sampling_comparison.py"
PAULI_HELPERS = METHOD_ROOT / "pauli_product" / "compare_fixed_shot_lih_baselines.py"
HUANG_DERAND_IMPLEMENTATION = HERE.parent / "methods" / "pauli_product" / "derand_huang2021.py"

METHODS = ("OGM", "SG", "Derand")
FIXED_BUDGETS = (100, 200, 300, 500, 800, 1200, 2000, 3000)
SCAN_SHOTS = 2038
SCAN_REPEATS = 50
FIXED_REPEATS = 200
BOOTSTRAP_RESAMPLES = 2000
PAULI_THRESHOLD = 1.0e-10
VERSION = "h4-gpd-srdd-identical-input-pauli-rerun-v1"
PREVIOUS_FIXED_SEED_VERSION = "five-molecule-same-target-state-v7-hybrid-f-vs-pauli-v4"


def load_module(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_module(LEGACY_DRIVER, "h4_same_input_legacy_pauli")
HELPERS = load_module(PAULI_HELPERS, "h4_same_input_pauli_helpers")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray, dtype: str | None = None) -> str:
    values = np.asarray(array)
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    canonical = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(canonical.view(np.uint8))
    return digest.hexdigest()


def stable_seed(*parts: Any, version: str = VERSION) -> int:
    payload = "|".join([version, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_dense_case(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        hamiltonian = np.asarray(archive["H"], dtype=np.complex128)
        state = np.asarray(archive["state"], dtype=np.complex128)
        energy = float(np.asarray(archive["energy"]).reshape(()))
    if hamiltonian.shape != (256, 256) or state.shape != (256,):
        raise RuntimeError(f"Unexpected H4 dimensions in {path}")
    hermiticity_error = float(np.linalg.norm(hamiltonian - hamiltonian.conj().T, "fro"))
    norm_error = abs(float(np.vdot(state, state).real) - 1.0)
    eigen_residual = float(np.linalg.norm(hamiltonian @ state - energy * state))
    if hermiticity_error > 1.0e-10 or norm_error > 1.0e-10 or eigen_residual > 1.0e-9:
        raise RuntimeError(
            f"Invalid H4 input {path}: herm={hermiticity_error}, "
            f"norm={norm_error}, eigen={eigen_residual}"
        )
    return {
        "path": path.resolve(),
        "file_sha256": sha256_file(path),
        "hamiltonian": hamiltonian,
        "hamiltonian_sha256": sha256_array(hamiltonian, "<c16"),
        "state": state,
        "state_sha256": sha256_array(state, "<c16"),
        "state_projector_sha256": sha256_array(np.outer(state, state.conj()), "<c16"),
        "energy": energy,
        "hermiticity_error": hermiticity_error,
        "state_norm_error": norm_error,
        "eigen_residual": eigen_residual,
    }


def make_designs(observables: np.ndarray, weights: np.ndarray, budgets: list[int]):
    ogm_settings, ogm_probabilities, ogm_audit = LEGACY.optimize_ogm(
        observables, weights
    )
    if not bool(ogm_audit["success"]):
        raise RuntimeError(f"OGM optimization failed: {ogm_audit}")
    maximum = max(budgets)
    fixed_maximum = {
        "SG": LEGACY.shadow_grouping_schedule(observables, weights, maximum),
        "Derand": LEGACY.derandomized_schedule(observables, weights, maximum),
    }
    return ogm_settings, ogm_probabilities, ogm_audit, fixed_maximum


def evaluate_case(
    *,
    label: str,
    case: dict[str, Any],
    budgets: list[int],
    repeats: int,
    output: Path,
    fixed_seed_compatibility: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    observables, weights, offset, pauli_audit = HELPERS.dense_to_pauli(
        case["hamiltonian"], PAULI_THRESHOLD
    )
    pauli_energy, term_expectations = LEGACY.audit_energy(
        observables, weights, offset, case["state"]
    )
    pauli_energy_error = float(pauli_energy - case["energy"])
    if abs(pauli_energy_error) > 1.0e-9:
        raise RuntimeError(f"{label}: Pauli energy audit failed: {pauli_energy_error}")

    case_dir = output / label
    schedules_dir = case_dir / "schedules"
    schedules_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        case_dir / "same_target_state_input.npz",
        target=case["hamiltonian"],
        ground_state=case["state"],
        exact_energy=np.asarray(case["energy"]),
        pauli_observables=observables,
        pauli_weights=weights,
        pauli_offset=np.asarray(offset),
        pauli_term_expectations=term_expectations,
    )

    ogm_settings, ogm_probabilities, ogm_audit, fixed_maximum = make_designs(
        observables, weights, budgets
    )
    np.savez_compressed(
        case_dir / "ogm_distribution.npz",
        settings=ogm_settings,
        probabilities=ogm_probabilities,
    )
    for method, schedule in fixed_maximum.items():
        np.savez_compressed(schedules_dir / f"{method}_max.npz", settings=schedule)

    cache = LEGACY.JointPatternCache(observables, case["state"])
    summary_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for point, total_shots in enumerate(budgets):
            if method == "OGM":
                schedule, allocation = HELPERS.minimum_one_probability_schedule(
                    ogm_settings, ogm_probabilities, total_shots
                )
                np.savez_compressed(
                    schedules_dir / f"OGM_T{total_shots}.npz",
                    settings=schedule,
                    allocation=allocation,
                )
            else:
                schedule = fixed_maximum[method][:total_shots]
            analytic = HELPERS.analytic_fixed_schedule(
                LEGACY, schedule, observables, weights, offset, cache
            )
            if fixed_seed_compatibility:
                sample_seed = stable_seed(
                    "H4",
                    method,
                    total_shots,
                    version=PREVIOUS_FIXED_SEED_VERSION,
                )
                bootstrap_seed = stable_seed(
                    "H4",
                    method,
                    "bootstrap",
                    point,
                    version=PREVIOUS_FIXED_SEED_VERSION,
                )
            else:
                sample_seed = stable_seed(label, method, total_shots)
                bootstrap_seed = stable_seed(label, method, "bootstrap", point)
            estimates, hit_counts, pattern_count = LEGACY.sample_fixed_schedule(
                schedule,
                observables,
                weights,
                offset,
                repeats,
                sample_seed,
                cache,
            )
            if not np.array_equal(hit_counts, analytic["hit_counts"]):
                raise RuntimeError(f"{label}/{method}/T={total_shots}: hit-count mismatch")
            if int(pattern_count) != int(analytic["distinct_hit_patterns"]):
                raise RuntimeError(f"{label}/{method}/T={total_shots}: pattern mismatch")
            analytic_mean = float(analytic["mean_energy"])
            analytic_variance = max(float(analytic["sampling_variance"]), 0.0)
            signed_bias = analytic_mean - case["energy"]
            sampling_errors = estimates - analytic_mean
            total_errors = estimates - case["energy"]
            empirical = HELPERS.empirical_fields(
                sampling_errors,
                total_errors,
                bootstrap_seed,
                BOOTSTRAP_RESAMPLES,
            )
            row = {
                "case": label,
                "method": method,
                "T_total_shots": total_shots,
                "T_actual_shots": int(len(schedule)),
                "exact_energy_hartree": case["energy"],
                "analytic_mean_energy_hartree": analytic_mean,
                "signed_coverage_bias_hartree": signed_bias,
                "analytic_sampling_variance_hartree2": analytic_variance,
                "analytic_sampling_SE_hartree": math.sqrt(analytic_variance),
                "analytic_total_RMSE_hartree": math.hypot(
                    signed_bias, math.sqrt(analytic_variance)
                ),
                "empirical_mean_energy_hartree": float(np.mean(estimates)),
                **empirical,
                "repeat_count": repeats,
                "measurement_settings_or_distinct_bases": LEGACY.distinct_rows(
                    schedule
                ),
                "covered_pauli_terms": int(np.count_nonzero(hit_counts)),
                "unhit_pauli_terms": int(np.count_nonzero(hit_counts == 0)),
                "distinct_hit_patterns": int(pattern_count),
                "schedule_sha256": sha256_array(schedule, "|i1"),
                "sample_seed": sample_seed,
                "hamiltonian_file_sha256": case["file_sha256"],
                "hamiltonian_array_sha256": case["hamiltonian_sha256"],
                "state_projector_sha256": case["state_projector_sha256"],
                "pauli_terms_including_identity": int(len(observables) + 1),
                "pauli_reconstruction_frobenius_error_hartree": float(
                    pauli_audit[
                        "retained_pauli_dense_reconstruction_frobenius_error_hartree"
                    ]
                ),
                "pauli_energy_audit_error_hartree": pauli_energy_error,
            }
            summary_rows.append(row)
            for repeat, estimate in enumerate(estimates):
                replicate_rows.append(
                    {
                        "case": label,
                        "method": method,
                        "T_total_shots": total_shots,
                        "repeat": repeat,
                        "energy_estimate_hartree": float(estimate),
                        "sampling_error_about_analytic_mean_hartree": float(
                            sampling_errors[repeat]
                        ),
                        "total_error_about_exact_energy_hartree": float(
                            total_errors[repeat]
                        ),
                    }
                )
            print(
                f"[{label}] {method:6s} T={total_shots:4d} "
                f"analytic={row['analytic_total_RMSE_hartree']:.8f} "
                f"empirical={row['empirical_total_RMSE_hartree']:.8f} "
                f"settings={row['measurement_settings_or_distinct_bases']}",
                flush=True,
            )
    audit = {
        "input_path": str(case["path"]),
        "input_file_sha256": case["file_sha256"],
        "hamiltonian_array_sha256": case["hamiltonian_sha256"],
        "state_vector_sha256": case["state_sha256"],
        "state_projector_sha256": case["state_projector_sha256"],
        "exact_energy_hartree": case["energy"],
        "input_hermiticity_frobenius_error_hartree": case["hermiticity_error"],
        "state_norm_error": case["state_norm_error"],
        "ground_state_eigen_residual_hartree": case["eigen_residual"],
        "pauli_terms_including_identity": int(len(observables) + 1),
        "pauli_reconstruction_frobenius_error_hartree": float(
            pauli_audit["retained_pauli_dense_reconstruction_frobenius_error_hartree"]
        ),
        "pauli_energy_audit_error_hartree": pauli_energy_error,
        "ogm_optimizer": ogm_audit,
        "all_shot_counts_conserved": all(
            int(row["T_total_shots"]) == int(row["T_actual_shots"])
            for row in summary_rows
        ),
        "all_pauli_terms_covered": all(
            int(row["unhit_pauli_terms"]) == 0 for row in summary_rows
        ),
    }
    return summary_rows, replicate_rows, audit


def bond_from_path(path: Path) -> float:
    match = re.fullmatch(r"H4_R([0-9]+(?:\.[0-9]+)?)\.npz", path.name)
    if match is None:
        raise ValueError(f"Cannot parse bond length from {path.name}")
    return float(match.group(1))


def comparison_rows(
    scan_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    historical = {
        (float(row["bond_length_angstrom"]), method): float(row[method])
        for row in read_csv(HISTORICAL_SCAN)
        for method in METHODS
    }
    scan_comparison: list[dict[str, Any]] = []
    for row in scan_rows:
        bond = float(str(row["case"]).split("R", 1)[1].replace("p", "."))
        old = historical[(bond, str(row["method"]))]
        new = float(row["empirical_total_RMSE_hartree"])
        analytic = float(row["analytic_total_RMSE_hartree"])
        scan_comparison.append(
            {
                "bond_length_angstrom": bond,
                "method": row["method"],
                "historical_empirical_RMSE_hartree": old,
                "same_input_empirical_RMSE_hartree": new,
                "same_input_analytic_RMSE_hartree": analytic,
                "empirical_delta_same_input_minus_historical_hartree": new - old,
                "empirical_ratio_same_input_over_historical": new / old,
                "historical_minus_analytic_hartree": old - analytic,
                "same_input_empirical_minus_analytic_hartree": new - analytic,
            }
        )

    previous = {
        (row["method"], int(row["T_total_shots"])): row
        for row in read_csv(PREVIOUS_FIXED)
        if row.get("molecule", "H4") == "H4" and row["method"] in METHODS
    }
    fixed_comparison: list[dict[str, Any]] = []
    for row in fixed_rows:
        key = (str(row["method"]), int(row["T_total_shots"]))
        old = previous[key]
        fixed_comparison.append(
            {
                "method": key[0],
                "T_total_shots": key[1],
                "previous_empirical_RMSE_hartree": float(
                    old["empirical_total_RMSE_hartree"]
                ),
                "rerun_empirical_RMSE_hartree": float(
                    row["empirical_total_RMSE_hartree"]
                ),
                "empirical_absolute_delta_hartree": abs(
                    float(row["empirical_total_RMSE_hartree"])
                    - float(old["empirical_total_RMSE_hartree"])
                ),
                "previous_analytic_RMSE_hartree": float(
                    old["analytic_total_RMSE_hartree"]
                ),
                "rerun_analytic_RMSE_hartree": float(
                    row["analytic_total_RMSE_hartree"]
                ),
                "analytic_absolute_delta_hartree": abs(
                    float(row["analytic_total_RMSE_hartree"])
                    - float(old["analytic_total_RMSE_hartree"])
                ),
                "previous_distinct_settings": int(
                    old["measurement_settings_or_distinct_bases"]
                ),
                "rerun_distinct_settings": int(
                    row["measurement_settings_or_distinct_bases"]
                ),
                "schedule_and_seed_protocol_reproduced": True,
            }
        )
    return scan_comparison, fixed_comparison


def update_figure_scan_input(scan_rows: list[dict[str, Any]], output: Path) -> None:
    old_rows = read_csv(FIGURE_SCAN_INPUT)
    lookup = {
        (
            float(str(row["case"]).split("R", 1)[1].replace("p", ".")),
            str(row["method"]),
        ): row
        for row in scan_rows
    }
    updated: list[dict[str, Any]] = []
    for row in old_rows:
        bond = float(row["bond_length_angstrom"])
        mutable: dict[str, Any] = dict(row)
        for method in METHODS:
            source = lookup[(bond, method)]
            mutable[f"{method}_empirical_rmse"] = source[
                "empirical_total_RMSE_hartree"
            ]
            mutable[f"{method}_analytic_rmse"] = source[
                "analytic_total_RMSE_hartree"
            ]
            mutable[f"{method}_measurement_settings"] = source[
                "measurement_settings_or_distinct_bases"
            ]
        mutable["pauli_rerun_input_sha256"] = lookup[(bond, "OGM")][
            "hamiltonian_file_sha256"
        ]
        updated.append(mutable)
    write_csv(output / "h4_bond_scan_comparison.csv", updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--update-figure-input",
        action="store_true",
        help="Write updated Pauli columns to the new output directory after all audits pass.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix with nonempty output directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    scan_paths = sorted(SCAN_INPUT.glob("H4_R*.npz"), key=bond_from_path)
    if len(scan_paths) != 21:
        raise RuntimeError(f"Expected 21 H4 scan inputs, found {len(scan_paths)}")
    scan_rows: list[dict[str, Any]] = []
    scan_replicates: list[dict[str, Any]] = []
    case_audits: dict[str, Any] = {}
    for path in scan_paths:
        bond = bond_from_path(path)
        label = f"scan_R{bond:.1f}".replace(".", "p")
        rows, replicates, audit = evaluate_case(
            label=label,
            case=load_dense_case(path),
            budgets=[SCAN_SHOTS],
            repeats=SCAN_REPEATS,
            output=output / "scan",
            fixed_seed_compatibility=False,
        )
        scan_rows.extend(rows)
        scan_replicates.extend(replicates)
        case_audits[label] = audit

    fixed_rows, fixed_replicates, fixed_audit = evaluate_case(
        label="fixed_R1p2",
        case=load_dense_case(FIXED_INPUT),
        budgets=list(FIXED_BUDGETS),
        repeats=FIXED_REPEATS,
        output=output / "fixed",
        fixed_seed_compatibility=True,
    )
    case_audits["fixed_R1p2"] = fixed_audit
    scan_rows.sort(key=lambda row: (float(str(row["case"]).split("R", 1)[1].replace("p", ".")), METHODS.index(str(row["method"]))))
    fixed_rows.sort(key=lambda row: (int(row["T_total_shots"]), METHODS.index(str(row["method"]))))
    scan_replicates.sort(key=lambda row: (str(row["case"]), METHODS.index(str(row["method"])), int(row["repeat"])))
    fixed_replicates.sort(key=lambda row: (int(row["T_total_shots"]), METHODS.index(str(row["method"])), int(row["repeat"])))

    write_csv(output / "scan_sampling_error_summary.csv", scan_rows)
    write_csv(output / "scan_sampling_error_replicates.csv", scan_replicates)
    write_csv(output / "fixed_sampling_error_summary.csv", fixed_rows)
    write_csv(output / "fixed_sampling_error_replicates.csv", fixed_replicates)
    scan_comparison, fixed_comparison = comparison_rows(scan_rows, fixed_rows)
    write_csv(output / "scan_comparison_to_historical.csv", scan_comparison)
    write_csv(output / "fixed_comparison_to_previous.csv", fixed_comparison)

    audit = {
        "status": "PASS",
        "definition_version": VERSION,
        "same_dense_Hamiltonian_and_state_used_by_all_Pauli_methods": True,
        "historical_error_curves_reused": False,
        "scan_case_count": len(scan_paths),
        "scan_shot_budget": SCAN_SHOTS,
        "scan_repeat_count": SCAN_REPEATS,
        "fixed_shot_budgets": list(FIXED_BUDGETS),
        "fixed_repeat_count": FIXED_REPEATS,
        "methods": list(METHODS),
        "case_audits": case_audits,
        "maximum_pauli_reconstruction_frobenius_error_hartree": max(
            float(item["pauli_reconstruction_frobenius_error_hartree"])
            for item in case_audits.values()
        ),
        "maximum_pauli_energy_audit_error_hartree": max(
            abs(float(item["pauli_energy_audit_error_hartree"]))
            for item in case_audits.values()
        ),
        "maximum_ground_state_eigen_residual_hartree": max(
            float(item["ground_state_eigen_residual_hartree"])
            for item in case_audits.values()
        ),
        "all_shot_counts_conserved": all(
            bool(item["all_shot_counts_conserved"]) for item in case_audits.values()
        ),
        "all_pauli_terms_covered": all(
            bool(item["all_pauli_terms_covered"]) for item in case_audits.values()
        ),
        "maximum_fixed_empirical_reproduction_delta_hartree": max(
            float(row["empirical_absolute_delta_hartree"])
            for row in fixed_comparison
        ),
        "maximum_fixed_analytic_reproduction_delta_hartree": max(
            float(row["analytic_absolute_delta_hartree"])
            for row in fixed_comparison
        ),
        "fixed_distinct_settings_reproduced": all(
            int(row["previous_distinct_settings"])
            == int(row["rerun_distinct_settings"])
            for row in fixed_comparison
        ),
    }
    with np.load(PREVIOUS_FIXED_INPUT, allow_pickle=False) as previous_archive:
        previous_fixed_target = np.asarray(previous_archive["target"], dtype=np.complex128)
        previous_fixed_state = np.asarray(
            previous_archive["ground_state"], dtype=np.complex128
        )
    current_fixed = load_dense_case(FIXED_INPUT)
    audit["previous_fixed_target_max_abs_delta_hartree"] = float(
        np.max(np.abs(previous_fixed_target - current_fixed["hamiltonian"]))
    )
    audit["previous_fixed_state_projector_max_abs_delta"] = float(
        np.max(
            np.abs(
                np.outer(previous_fixed_state, previous_fixed_state.conj())
                - np.outer(current_fixed["state"], current_fixed["state"].conj())
            )
        )
    )
    previous_manifest = json.loads(PREVIOUS_FIXED_MANIFEST.read_text(encoding="utf-8"))
    previous_driver_hash = previous_manifest.get("method_provenance", {}).get(
        "legacy_pauli_driver_sha256"
    )
    audit["previous_fixed_driver_sha256"] = previous_driver_hash
    audit["current_driver_sha256"] = sha256_file(LEGACY_DRIVER)
    audit["previous_fixed_driver_matches_current"] = (
        previous_driver_hash == audit["current_driver_sha256"]
    )
    audit["fixed_results_changed_from_previous"] = bool(
        audit["maximum_fixed_empirical_reproduction_delta_hartree"] > 1.0e-14
        or audit["maximum_fixed_analytic_reproduction_delta_hartree"] > 1.0e-14
        or not audit["fixed_distinct_settings_reproduced"]
    )
    audit["status"] = "PASS" if (
        audit["scan_case_count"] == 21
        and len(scan_rows) == 21 * len(METHODS)
        and len(fixed_rows) == len(FIXED_BUDGETS) * len(METHODS)
        and audit["all_shot_counts_conserved"]
        and audit["all_pauli_terms_covered"]
        and audit["maximum_pauli_reconstruction_frobenius_error_hartree"] <= 1.0e-8
        and audit["maximum_pauli_energy_audit_error_hartree"] <= 1.0e-9
        and audit["maximum_ground_state_eigen_residual_hartree"] <= 1.0e-9
        and audit["previous_fixed_target_max_abs_delta_hartree"] <= 1.0e-12
        and audit["previous_fixed_state_projector_max_abs_delta"] <= 1.0e-12
    ) else "FAIL"
    write_json(output / "audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"H4 Pauli rerun audit failed: {audit}")
    if args.update_figure_input:
        update_figure_scan_input(scan_rows, output)

    output_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "definition_version": VERSION,
        "created_unix_time": time.time(),
        "wall_seconds": time.time() - started,
        "comparison_scope": (
            "Each Pauli method uses the exact dense Hamiltonian and exact ground-state "
            "vector stored in the corresponding current GPD/SRDD input archive."
        ),
        "scan": {
            "input_directory": str(SCAN_INPUT.resolve()),
            "shot_budget": SCAN_SHOTS,
            "repeat_count": SCAN_REPEATS,
        },
        "fixed": {
            "input": str(FIXED_INPUT.resolve()),
            "shot_budgets": list(FIXED_BUDGETS),
            "repeat_count": FIXED_REPEATS,
        },
        "methods": list(METHODS),
        "error_definition": (
            "RMSE of the pooled-hit energy estimate relative to the exact ground-state "
            "energy across independent ideal Born-outcome replays."
        ),
        "numerical_protocol": {
            "pauli_coefficient_threshold": PAULI_THRESHOLD,
            "identity_treatment": (
                "The single retained identity Pauli term is evaluated as an exact "
                "classical offset; coverage counts refer to the 184 nonidentity terms."
            ),
            "array_sha256_canonicalization": (
                "SHA256 of ASCII shape, NumPy dtype.str, and C-contiguous array bytes, "
                "concatenated in that order."
            ),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed_rule": (
                "SHA256('|'.join([seed_version, *parts]))[:8], interpreted as a "
                "little-endian unsigned integer modulo 2**32. Scan seeds use "
                f"{VERSION}; fixed-geometry replay-compatible seeds use "
                f"{PREVIOUS_FIXED_SEED_VERSION}."
            ),
        },
        "algorithm_parameters": {
            "OGM": {
                "candidate_construction": "greedy QWC cover in stable descending-|coefficient| order",
                "optimizer": "SLSQP",
                "optimizer_maxiter": 2000,
                "optimizer_ftol": 1.0e-12,
                "active_probability_cutoff": 1.0e-15,
                "integer_allocation": (
                    "Every active basis receives one shot; remaining shots are assigned "
                    "by normalized probabilities and stable largest-remainder rounding."
                ),
            },
            "SG": {
                "score": (
                    "Shadow-Grouping Bernstein score with "
                    "alpha=max(abs(c))/min(abs(c))+min(abs(c)); deterministic greedy QWC basis."
                )
            },
            "Derand": {
                "algorithm": "Huang-2021 Appendix-C C8/C11 fixed-budget schedule",
                "eta": 0.9,
                "epsilon": math.sqrt(0.9),
                "numerical_evaluation": "stable affected-term gain",
            },
            "estimator": (
                "Unbiased pooled-hit estimator retaining the joint Born covariance of "
                "all commuting Pauli terms hit by each measurement setting."
            ),
        },
        "method_provenance": {
            "legacy_pauli_driver": str(LEGACY_DRIVER.resolve()),
            "legacy_pauli_driver_sha256": sha256_file(LEGACY_DRIVER),
            "huang_derandomization_implementation": str(
                HUANG_DERAND_IMPLEMENTATION.resolve()
            ),
            "huang_derandomization_implementation_sha256": sha256_file(
                HUANG_DERAND_IMPLEMENTATION
            ),
            "dense_pauli_and_estimator_helpers": str(PAULI_HELPERS.resolve()),
            "dense_pauli_and_estimator_helpers_sha256": sha256_file(PAULI_HELPERS),
            "this_runner": str(Path(__file__).resolve()),
            "this_runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "audit": audit,
        "output_hashes": {
            str(path.relative_to(output)): sha256_file(path) for path in output_files
        },
    }
    write_json(output / "manifest.json", manifest)
    print(f"PASS: wrote audited H4 Pauli rerun to {output}", flush=True)


if __name__ == "__main__":
    main()
