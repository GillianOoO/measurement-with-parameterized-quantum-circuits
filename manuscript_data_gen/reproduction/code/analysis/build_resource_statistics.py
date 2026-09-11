#!/usr/bin/env python3
"""Build resource summaries for the molecular benchmarks retained in the main text.

The script reads the same machine-readable summaries used by the manuscript
figures.  It never digitizes plotted curves.  For every method, the estimator
selected or realized at the reference budget T0=3000 is frozen.  Its exact
state-dependent sampling variance and approximation bias define
RMSE(T)**2 = b**2 + V/T, where V=T0*Var(T0).  The exact-representation Pauli
methods and FC-IMA have b=0.  Every AGPD row contains a minimum-feasible
depth-d_A shallow one-body cover followed
by all K native source settings, for a total of L_A+K settings.  No source is
transferred into the cover, and both the reported maximum two-qubit depth and gate budget
include every cover and source setting.  Every SRDD resource row represents
K+L depth-d_R shallow settings after exact one-body-collector redistribution;
an independently diagonalized SRDD collector is rejected, and every
allocated SRDD setting shot contributes to the gate budget.  For the
two full-space targets, Nature-2023 overlapping full-commuting iterative
measurement allocation (FC-IMA) is loaded from its independent pooled-estimator
and compiled-Clifford audit outputs.

Fixed-geometry H4 and the 21-bond H4 scan use ground-state-blind
fully-corrective dense-sector AGPD v10 while retaining the same
leakage-certified sector ranges with full-Fock fallback.  H4 is the only
molecule with a publication-facing AGPD result; H6 retains SRDD and the
Pauli-product baselines only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

from native_gate_resources import (
    CX_GIVENS,
    GATE_RESOURCE_CONVENTION,
    GateResources,
    source_gate_resources,
    summarize_gate_resources,
)

MPL_CACHE = Path(__file__).resolve().parents[2] / "tmp" / "matplotlib" / "resource_statistics"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EPSILON_TARGET = 1.0e-2
REFERENCE_SHOTS = 3000
REPORTING_LIMIT = 100_000_000
FIXED_ESTIMATOR_MODEL = "RMSE(T)^2=b^2+V/T"
FIXED_ESTIMATOR_SCALING_EXPONENT = 0.5
PLOTS_DIR = (
    ROOT
    / "RMeasurementAnsatz"
    / "manuscript"
    / "Journal_chemical_theory_computation"
    / "plots_figures"
)

FC_METHOD = "FC-IMA"
FC_ALLOCATION = "fc_ima_primary"
SRDD_METHOD = "SRDD"
LEGACY_SRCDF_METHOD = "s-RCDF"
LEGACY_SRCDF_STORED_METHOD = "standalone s-RCDF-F"
METHODS = ("AGPD", SRDD_METHOD, FC_METHOD, "OGM", "SG", "Derand")
RESOURCE_FIGURE_METHODS = (SRDD_METHOD, FC_METHOD, "OGM", "SG", "Derand")
PUBLICATION_OMITTED_METHOD_TARGETS = (("H6", "AGPD"),)
METHOD_COLORS = {
    "AGPD": "#3F6F98",
    SRDD_METHOD: "#4F8A70",
    FC_METHOD: "#B45A45",
    "OGM": "#777472",
    "SG": "#4D4D4D",
    "Derand": "#A48974",
}

COMMON_V8 = (
    ROOT
    / "outputs"
    / "adaptive_f3_ansatz_pool"
    / "standalone_shallow_rcdf_vs_four_family_agpd_v8_shallow_collectors"
    / "sampling_error_comparison.csv"
)
H4_COMMON_V10 = (
    ROOT
    / "outputs"
    / "adaptive_f3_ansatz_pool"
    / "fully_corrective_dense_sector_agpd_vs_srcdf_and_pauli_v10_h4"
    / "sampling_error_comparison.csv"
)
AGPD_ROOT_V8 = (
    ROOT
    / "outputs"
    / "adaptive_f3_ansatz_pool"
    / "five_molecule_v8_shallow_agpd_collector"
)
AGPD_ROOT_V10 = (
    ROOT
    / "outputs"
    / "adaptive_f3_ansatz_pool"
    / "five_molecule_v10_fully_corrective_dense_sector_agpd"
)
AGPD_ROOTS = {
    "H4": AGPD_ROOT_V10,
}
AGPD_CONTRACTS = {
    "H4": "v10_fully_corrective_dense_sector",
}
SRCDF_ROOT = (
    ROOT
    / "outputs"
    / "adaptive_f3_ansatz_pool"
    / "standalone_shallow_rcdf_five_molecule_v5_shallow_collector"
)
SRCDF_SUMMARY = SRCDF_ROOT / "sampling_summary_all.csv"
BEH2 = (
    ROOT
    / "outputs"
    / "adaptive_f3_ansatz_pool"
    / "beh2_srcdf_pauli_fullspace_k50_v2"
    / "sampling_summary_all.csv"
)
BEH2_CANDIDATES = BEH2.parent / "BeH2" / "srcdf_candidate_by_T_K.csv"
N2 = (
    ROOT
    / "outputs"
    / "adaptive_f3_ansatz_pool"
    / "h2o_n2_srcdf_pauli_fullspace_k50_v3"
    / "N2"
    / "sampling_summary.csv"
)
N2_CANDIDATES = N2.parent / "srcdf_candidate_by_T_K.csv"
FC_LEGACY_ROOT = ROOT / "outputs" / "traditional_cm_benchmarks"
FC_ERROR_SUMMARY = FC_LEGACY_ROOT / "error_eval" / "results" / "summary.csv"
FC_NOISE = FC_LEGACY_ROOT / "noise_eval" / "results" / "fc_local_depolarizing_curve.csv"
FC_CIRCUIT_AUDITS = {
    molecule: FC_LEGACY_ROOT / "noise_eval" / "results" / molecule / "circuit_audit.csv"
    for molecule in ("BeH2", "N2")
}
FC_ALLOCATIONS = {
    molecule: FC_LEGACY_ROOT / "error_eval" / "results" / molecule / "shot_allocations.csv"
    for molecule in ("BeH2", "N2")
}
SRCDF_NOISE = (
    ROOT / "outputs" / "fig3_extended_benchmarks" / "srcdf_local_depolarizing_moments.csv"
)
STATE_VARIANCE = (
    ROOT
    / "outputs"
    / "state_dependent_variance_main_benchmarks"
    / "state_dependent_variance_T3000.csv"
)
H4_PAULI_RERUN_ROOT = (
    ROOT
    / "outputs"
    / "iswap_gpd_main_loss_v1"
    / "h4_pauli_same_input_rerun"
)
H4_PAULI_FIXED_SUMMARY = H4_PAULI_RERUN_ROOT / "fixed_sampling_error_summary.csv"
H4_PAULI_RERUN_MANIFEST = H4_PAULI_RERUN_ROOT / "manifest.json"
H4_SCREEN = (
    ROOT
    / "RMeasurementAnsatz"
    / "submit_jctc"
    / "RandomOptimizedMeasurement_submit"
    / "Journal_chemical_theory_computation"
    / "variance_record"
    / "h4_eight_ansatz_screen"
)
H4_SCAN_SUMMARY = (
    H4_SCREEN
    / "h4_all_bonds_srcdf_2038_shallow_collector_v2"
    / "figure3a_H4_bond_scan_2038_with_srcdf.csv"
)
H4_SCAN_HAMILTONIANS = H4_SCREEN / "h4_bond_scan_gfro_d4" / "hamiltonians"
H4_SCAN_AGPD = (
    ROOT
    / "outputs"
    / "adaptive_f3_ansatz_pool"
    / "h4_all_bonds_agpd_v10_fully_corrective_dense_sector_T2038"
)
H4_SCAN_AGPD_MANIFEST = PLOTS_DIR / "h4_bond_agpd_manifest.json"
LEGACY_PAULI = (
    ROOT
    / "RMeasurementAnsatz"
    / "Decompose-by-tensornetwork-feature-KongGit"
    / "existing_codes"
    / "new_datas_afterJun18"
    / "run_lih_r1p50_sampling_comparison.py"
)

# iSWAP is native. Other families retain their existing CNOT compiler rules
# in native_gate_resources; totals preserve CNOT and iSWAP counts separately.

AGPD_V8_RUNNER_VERSION = "hybrid-ansatz-F-shallow-one-body-cover-four-family-v8"
AGPD_V8_CORE_VERSION = "adaptive-af-four-circuit-pool-shallow-collector-v8"
AGPD_V8_RESULT_CLASS = "four_family_v8_shallow_collector_benchmark"
AGPD_V9_RUNNER_VERSION = "hybrid-ansatz-F-sector-range-shallow-cover-four-family-v9"
AGPD_V9_CORE_VERSION = "adaptive-af-four-circuit-pool-sector-range-shallow-collector-v9"
AGPD_V9_RESULT_CLASS = "four_family_v9_sector_range_shallow_cover_benchmark"
AGPD_V10_RUNNER_VERSION = (
    "hybrid-ansatz-F-fully-corrective-dense-sector-shallow-cover-v10"
)
AGPD_V10_CORE_VERSION = (
    "adaptive-af-four-circuit-pool-fully-corrective-dense-sector-"
    "shallow-collector-v10"
)
AGPD_V10_RESULT_CLASS = "fully_corrective_dense_sector_agpd_v10"
AGPD_V10_SELECTION_RULE = (
    "hypot(sector_residual_operator_norm,range_sampling_proxy)"
)
AGPD_V10_RCOND = 1.0e-12
SECTOR_RANGE_DOMAIN = "fixed_particle_number_sector"
FALLBACK_RANGE_DOMAIN = "full_fock_fallback_due_to_sector_leakage"
MIXED_RANGE_DOMAIN = "mixed_sector_and_full_fock_fallback"
SECTOR_LEAKAGE_TOLERANCE = 1.0e-10
SECTOR_RANGE_TOLERANCE = 1.0e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analytic_shots_to_target(total: int, sampling_se: float, bias: float) -> dict[str, Any]:
    if abs(bias) >= EPSILON_TARGET:
        return {
            "shots_to_target": None,
            "status": "bias_floor",
            "variance_coefficient": float(total * sampling_se**2),
        }
    variance_coefficient = float(total * sampling_se**2)
    denominator = EPSILON_TARGET**2 - bias**2
    estimate = variance_coefficient / denominator
    return {
        "shots_to_target": int(math.ceil(estimate)),
        "shots_to_target_unrounded": float(estimate),
        "status": "reported" if estimate <= REPORTING_LIMIT else "exceeds_1e8",
        "variance_coefficient": variance_coefficient,
    }


def fixed_pauli_variance(
    variance: pd.DataFrame,
    molecule: str,
    method: str,
    total: int,
) -> tuple[float, float]:
    """Return and validate the exact frozen-schedule variance and coefficient."""
    selected = variance.loc[
        (variance["molecule"] == molecule)
        & (variance["method"] == method)
        & (variance["T_total_shots"] == total)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one exact variance row for {molecule}/{method} at T={total}"
        )
    row = selected.iloc[0]
    if str(row.get("availability", "available")) != "available":
        raise RuntimeError(f"Exact variance is unavailable for {molecule}/{method}")
    estimator_variance = float(row["estimator_variance_hartree2"])
    variance_coefficient = float(row["T_times_fixed_variance_hartree2"])
    bias = float(row["signed_approximation_bias_hartree"])
    if estimator_variance <= 0.0 or variance_coefficient <= 0.0:
        raise RuntimeError(f"Nonpositive exact variance for {molecule}/{method}")
    if not math.isclose(
        variance_coefficient,
        total * estimator_variance,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(f"Inconsistent T*Var coefficient for {molecule}/{method}")
    if not math.isclose(bias, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"Exact Pauli estimator has nonzero bias for {molecule}/{method}")
    return estimator_variance, variance_coefficient


def validate_fixed_estimator_rows(
    rows: list[dict[str, Any]], variance: pd.DataFrame
) -> None:
    """Cross-check all available T0 analytic rows against the variance audit."""
    by_key = {
        (str(row["molecule"]), str(row["method"]), int(row["T_total_shots"])): row
        for _, row in variance.iterrows()
        if str(row.get("availability", "available")) == "available"
    }
    for record in rows:
        benchmark = str(record.get("benchmark", ""))
        if benchmark.startswith("H4 scan"):
            continue
        if record.get("fixed_estimator_model") != FIXED_ESTIMATOR_MODEL:
            raise RuntimeError(f"Missing frozen-estimator model metadata for {benchmark}")
        if not math.isclose(
            float(record.get("fixed_estimator_scaling_exponent", math.nan)),
            FIXED_ESTIMATOR_SCALING_EXPONENT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"Invalid frozen-estimator exponent metadata for {benchmark}")
        if "variance_coefficient" not in record:
            raise RuntimeError(f"Missing frozen variance coefficient for {benchmark}")
        method = str(record.get("method", ""))
        if method == FC_METHOD:
            continue
        molecule = benchmark.split(",", 1)[0]
        total = int(record["reference_total_shots"])
        audited = by_key.get((molecule, method, total))
        if audited is None:
            continue
        expected_variance = float(audited["estimator_variance_hartree2"])
        expected_coefficient = float(audited["T_times_fixed_variance_hartree2"])
        expected_bias = float(audited["signed_approximation_bias_hartree"])
        actual_coefficient = float(record["variance_coefficient"])
        actual_bias = float(record["signed_bias"])
        actual_se = float(record["analytic_sampling_se"])
        if not math.isclose(
            actual_coefficient, expected_coefficient, rel_tol=1.0e-11, abs_tol=1.0e-12
        ):
            raise RuntimeError(f"Frozen variance coefficient mismatch for {molecule}/{method}")
        if not math.isclose(actual_se**2, expected_variance, rel_tol=1.0e-11, abs_tol=1.0e-12):
            raise RuntimeError(f"Frozen T0 variance mismatch for {molecule}/{method}")
        if not math.isclose(actual_bias, expected_bias, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(f"Frozen bias mismatch for {molecule}/{method}")


def largest_remainder(total: int, reference: list[int]) -> np.ndarray:
    weights = np.asarray(reference, dtype=float)
    quotas = total * weights / weights.sum()
    counts = np.floor(quotas).astype(int)
    missing = total - int(counts.sum())
    if missing:
        order = np.argsort(-(quotas - counts), kind="stable")
        counts[order[:missing]] += 1
    if int(counts.sum()) != total:
        raise RuntimeError("Largest-remainder allocation failed shot conservation")
    return counts


def complete_setting_largest_remainder(total: int, reference: list[int]) -> np.ndarray:
    """Rescale a positive shot vector while retaining at least one shot per setting."""
    reference_counts = np.asarray(reference, dtype=int)
    if reference_counts.ndim != 1 or len(reference_counts) == 0:
        raise ValueError("Complete-setting allocation needs a nonempty reference vector")
    if np.any(reference_counts < 1) or total < len(reference_counts):
        raise ValueError("Complete-setting allocation needs positive shots and T >= settings")
    counts = np.ones(len(reference_counts), dtype=int)
    remaining = total - len(reference_counts)
    if remaining:
        weights = reference_counts.astype(float) - 1.0
        if float(weights.sum()) == 0.0:
            weights = np.ones(len(reference_counts), dtype=float)
        quotas = remaining * weights / float(weights.sum())
        floors = np.floor(quotas).astype(int)
        counts += floors
        leftover = remaining - int(floors.sum())
        order = np.argsort(-(quotas - floors), kind="stable")
        counts[order[:leftover]] += 1
    if int(counts.sum()) != total or np.any(counts < 1):
        raise RuntimeError("Complete-setting largest-remainder allocation failed")
    return counts


def fc_ima_largest_remainder(total: int, fractions: np.ndarray) -> np.ndarray:
    """Replay the executable FC-IMA allocation with one shot per setting."""
    weights = np.asarray(fractions, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("FC-IMA allocation fractions must be a nonempty vector")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Every overlapping FC-IMA setting needs positive weight")
    if total < len(weights):
        raise ValueError(
            f"T={total} is smaller than the {len(weights)} FC-IMA settings"
        )
    counts = np.ones(len(weights), dtype=int)
    remaining = total - len(weights)
    if remaining:
        quotas = remaining * weights / float(weights.sum())
        floors = np.floor(quotas).astype(int)
        counts += floors
        leftover = remaining - int(floors.sum())
        order = np.lexsort((np.arange(len(weights)), -(quotas - floors)))
        counts[order[:leftover]] += 1
    if int(counts.sum()) != total or np.any(counts < 1):
        raise RuntimeError("FC-IMA largest-remainder allocation failed")
    return counts


def parse_shot_vector(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item) for item in str(value).split()]


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def agpd_root_and_contract(molecule: str) -> tuple[Path, str]:
    try:
        return AGPD_ROOTS[molecule], AGPD_CONTRACTS[molecule]
    except KeyError as exc:
        raise KeyError(f"No fixed-geometry AGPD artifact route for {molecule}") from exc


def validate_agpd_manifest(path: Path, contract: str, context: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "v8_full_fock_range": (
            AGPD_V8_RUNNER_VERSION,
            AGPD_V8_CORE_VERSION,
            AGPD_V8_RESULT_CLASS,
        ),
        "v9_sector_range": (
            AGPD_V9_RUNNER_VERSION,
            AGPD_V9_CORE_VERSION,
            AGPD_V9_RESULT_CLASS,
        ),
        "v10_fully_corrective_dense_sector": (
            AGPD_V10_RUNNER_VERSION,
            AGPD_V10_CORE_VERSION,
            AGPD_V10_RESULT_CLASS,
        ),
    }
    if contract not in expected:
        raise RuntimeError(f"{context}: unknown AGPD artifact contract {contract!r}")
    observed = (
        payload.get("version"),
        payload.get("adaptive_core_definition_version"),
        payload.get("result_class"),
    )
    if observed != expected[contract]:
        raise RuntimeError(
            f"{context}: AGPD manifest contract mismatch; "
            f"expected {expected[contract]!r}, observed {observed!r}"
        )
    if contract == "v10_fully_corrective_dense_sector" and (
        float(payload.get("fully_corrective_rcond", -1.0)) != AGPD_V10_RCOND
        or payload.get("selection_rule") != AGPD_V10_SELECTION_RULE
        or payload.get("selection_uses_exact_ground_state") is not False
    ):
        raise RuntimeError(
            f"{context}: v10 manifest lacks the ground-state-blind "
            "fully-corrective selection contract"
        )


def validate_v10_fully_corrective_audit(path: Path, context: str) -> None:
    """Fail closed on the case-level dense-sector refit audit."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or payload.get("fully_corrective_dense_sector_contract") is not True
        or payload.get("fully_corrective_solver")
        != "real-coefficient SVD least squares"
        or float(payload.get("fully_corrective_rcond", -1.0))
        != AGPD_V10_RCOND
        or payload.get("selection_rule") != AGPD_V10_SELECTION_RULE
        or payload.get("exact_ground_state_used_for_selection") is not False
        or payload.get(
            "probe_calibration_retained_for_paired_audit_but_not_used_for_selection"
        )
        is not True
    ):
        raise RuntimeError(
            f"{context}: invalid fully-corrective dense-sector v10 audit"
        )


def validate_h4_scan_agpd_manifest(path: Path) -> None:
    """Validate the producer manifest for the 21-geometry H4 v10 scan."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_bonds = [round(0.4 + 0.2 * index, 1) for index in range(21)]
    observed_root = Path(str(payload.get("full_artifact_root", ""))).resolve()
    summary_path = Path(str(payload.get("summary", "")))
    checks = {
        "status": payload.get("status") == "PASS",
        "machine_algorithm": payload.get("machine_algorithm") == AGPD_V10_RUNNER_VERSION,
        "selection_rule": payload.get("selection_rule") == AGPD_V10_SELECTION_RULE,
        "fully_corrective_rcond": float(
            payload.get("fully_corrective_rcond", -1.0)
        )
        == AGPD_V10_RCOND,
        "ground_state_blind_selection": payload.get(
            "selection_uses_exact_ground_state"
        )
        is False,
        "selected_prefix_reconstruction": float(
            payload.get("maximum_selected_prefix_reconstruction_error", math.inf)
        )
        <= 2.0e-8,
        "shot_budget": int(payload.get("shot_budget", -1)) == 2038,
        "range_contract": payload.get("range_contract")
        == "certified_sector_range_or_full_fock_fallback",
        "sector_leakage_tolerance": math.isclose(
            float(payload.get("sector_leakage_tolerance", math.nan)),
            SECTOR_LEAKAGE_TOLERANCE,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "bond_grid": [round(float(value), 1) for value in payload.get("bond_lengths_angstrom", [])]
        == expected_bonds,
        "artifact_root": observed_root == H4_SCAN_AGPD.resolve(),
        "summary_exists": summary_path.is_file(),
        "summary_hash": summary_path.is_file()
        and sha256(summary_path) == payload.get("summary_sha256"),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "H4 21-bond AGPD producer manifest failed checks: " + ", ".join(failed)
        )


def load_fixed_comparison_overlay() -> pd.DataFrame:
    """Load H4 v10 plus the non-AGPD H6 baseline rows used for publication."""
    v8_manifest = json.loads(
        (COMMON_V8.parent / "manifest.json").read_text(encoding="utf-8")
    )
    v10_manifest = json.loads(
        (H4_COMMON_V10.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        v8_manifest.get("version")
        != "standalone-srcdf-vs-agpd-both-shallow-collectors-pauli-v8"
        or v8_manifest.get("status") != "PASS"
    ):
        raise RuntimeError("H6 comparison source is not the audited v8 comparison")
    if (
        v10_manifest.get("version")
        != "fully-corrective-dense-sector-agpd-vs-srcdf-and-pauli-v10"
        or v10_manifest.get("status") != "PASS"
        or v10_manifest.get("output_sha256", {}).get(
            "sampling_error_comparison.csv"
        )
        != sha256(H4_COMMON_V10)
        or v10_manifest.get("input_manifest_sha256", {}).get(
            str((AGPD_ROOT_V10 / "manifest.json").resolve())
        )
        != sha256(AGPD_ROOT_V10 / "manifest.json")
    ):
        raise RuntimeError(
            "H4 comparison overlay is not the audited fully-corrective "
            "dense-sector v10 comparison"
        )

    v8 = pd.read_csv(COMMON_V8)
    v10 = pd.read_csv(H4_COMMON_V10)
    if set(v10["molecule"].astype(str)) != {"H4"}:
        raise RuntimeError("The v10 fixed-comparison overlay must contain H4 only")
    expected_budgets = {100, 200, 300, 500, 800, 1200, 2000, 3000}
    expected_methods = {
        "four-family Hybrid-F", LEGACY_SRCDF_STORED_METHOD, "OGM", "SG", "Derand"
    }
    h4 = v10.loc[v10["molecule"] == "H4"].copy()
    rerun_manifest = json.loads(H4_PAULI_RERUN_MANIFEST.read_text(encoding="utf-8"))
    rerun_audit = rerun_manifest.get("audit", {})
    if (
        rerun_manifest.get("definition_version")
        != "h4-gpd-srdd-identical-input-pauli-rerun-v1"
        or not rerun_audit.get("all_pauli_terms_covered")
        or not rerun_audit.get("all_shot_counts_conserved")
        or rerun_manifest.get("output_hashes", {}).get(
            "fixed_sampling_error_summary.csv"
        )
        != sha256(H4_PAULI_FIXED_SUMMARY)
    ):
        raise RuntimeError("H4 identical-input Pauli rerun failed provenance checks")
    rerun = pd.read_csv(H4_PAULI_FIXED_SUMMARY)
    rerun = rerun.loc[rerun["case"] == "fixed_R1p2"].copy()
    if (
        set(rerun["method"].astype(str)) != {"OGM", "SG", "Derand"}
        or set(rerun["T_total_shots"].astype(int)) != expected_budgets
        or len(rerun) != 3 * len(expected_budgets)
    ):
        raise RuntimeError("H4 identical-input Pauli rerun has the wrong method grid")
    rerun_fields = {
        "analytic_sampling_SE_hartree": "analytic_sampling_SE",
        "signed_coverage_bias_hartree": "signed_approximation_bias",
        "analytic_total_RMSE_hartree": "analytic_total_RMSE",
        "empirical_total_RMSE_hartree": "empirical_total_RMSE",
        "repeat_count": "sampling_repeats",
        "empirical_total_RMSE_bootstrap95_low_hartree": (
            "empirical_total_RMSE_bootstrap95_low"
        ),
        "empirical_total_RMSE_bootstrap95_high_hartree": (
            "empirical_total_RMSE_bootstrap95_high"
        ),
        "measurement_settings_or_distinct_bases": (
            "measurement_settings_or_distinct_bases"
        ),
        "covered_pauli_terms": "covered_pauli_terms",
        "unhit_pauli_terms": "unhit_pauli_terms",
    }
    for source_row in rerun.itertuples(index=False):
        mask = (
            (h4["method"] == source_row.method)
            & (h4["T_total_shots"].astype(int) == int(source_row.T_total_shots))
        )
        if int(mask.sum()) != 1:
            raise RuntimeError("H4 comparison overlay row is not unique")
        for source_field, target_field in rerun_fields.items():
            h4.loc[mask, target_field] = getattr(source_row, source_field)
        h4.loc[mask, "provenance"] = "h4_identical_input_pauli_rerun_v1"
    h6_source = v8.loc[v8["molecule"] == "H6"].copy()
    for molecule, frame, contract in (
        ("H4", h4, "v10"),
        ("H6", h6_source, "baseline source"),
    ):
        if set(frame["T_total_shots"].astype(int)) != expected_budgets:
            raise RuntimeError(f"{molecule} {contract} comparison has the wrong shot grid")
        if set(frame["method"].astype(str)) != expected_methods:
            raise RuntimeError(f"{molecule} {contract} comparison has the wrong method set")
        counts = frame.groupby(["method", "T_total_shots"], dropna=False).size()
        if len(counts) != len(expected_methods) * len(expected_budgets) or not (counts == 1).all():
            raise RuntimeError(f"{molecule} {contract} comparison rows are not unique")

    h4_agpd = h4.loc[h4["method"] == "four-family Hybrid-F"]
    required_v10_columns = {
        "centered_range_domain", "full_fock_centered_range_sum",
        "sector_centered_range_sum", "sector_range_sum_reduction_fraction",
        "maximum_sector_leakage_relative_frobenius",
        "sector_operator_norm", "fully_corrective_design_rank",
        "fully_corrective_design_nullity", "fully_corrective_rcond",
        "selection_rule", "selection_uses_exact_ground_state",
    }
    missing = sorted(required_v10_columns - set(h4_agpd.columns))
    if missing or h4_agpd[list(required_v10_columns)].isna().any().any():
        raise RuntimeError(
            f"H4 v10 comparison lacks fully-corrective/range fields: {missing}"
        )
    if not (
        h4_agpd["provenance"]
        .astype(str)
        .str.contains("v10_fully_corrective_dense_sector")
    ).all():
        raise RuntimeError("H4 comparison overlay provenance is not dense-sector v10")
    if (
        not (h4_agpd["fully_corrective_rcond"].astype(float) == AGPD_V10_RCOND).all()
        or not (h4_agpd["selection_rule"].astype(str) == AGPD_V10_SELECTION_RULE).all()
        or h4_agpd["selection_uses_exact_ground_state"].map(parse_bool).any()
    ):
        raise RuntimeError("H4 v10 comparison violates fully-corrective selection contract")
    if (h6_source.loc[h6_source["method"] == "four-family Hybrid-F", "provenance"]
            .astype(str).str.contains("v9|v10", regex=True).any()):
        raise RuntimeError("H6 comparison rows must remain v8 and not be relabelled")
    h6 = h6_source.loc[
        h6_source["method"].isin(("OGM", "SG", "Derand"))
    ].copy()
    return pd.concat([h4, h6], ignore_index=True, sort=False)


def finite_nonnegative_range(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"{context}: expected a finite nonnegative value, got {value!r}")
    return result


def sector_range_close(left: float, right: float) -> bool:
    return abs(left - right) <= SECTOR_RANGE_TOLERANCE * max(
        1.0, abs(left), abs(right)
    )


def replay_integer_range_allocation(total_shots: int, ranges: list[float]) -> list[int]:
    # Match the executable AGPD allocator exactly: only a mathematically exact
    # zero may be omitted.  Fully-corrective refits can leave a tiny, positive
    # leakage guard (for example 2.9e-15 at H4 R=0.8); that guard still denotes
    # nonzero variance and therefore must retain its mandatory one shot.  Using
    # a numerical activity cutoff here would silently change both the saved
    # allocation and its leakage-certified sector/full-Fock fallback ledger.
    active = [index for index, value in enumerate(ranges) if value > 0.0]
    if total_shots < len(active):
        raise RuntimeError("Sector-range replay has fewer shots than active settings")
    shots = [0] * len(ranges)
    for index in active:
        shots[index] = 1
    remaining = total_shots - len(active)
    if remaining:
        normalization = sum(ranges[index] for index in active)
        if normalization <= 0.0:
            raise RuntimeError("Sector-range replay has zero active normalization")
        raw = [remaining * ranges[index] / normalization for index in active]
        extra = [int(math.floor(value)) for value in raw]
        for slot, index in enumerate(active):
            shots[index] += extra[slot]
        remainder = remaining - sum(extra)
        order = sorted(
            range(len(active)),
            key=lambda slot: (-(raw[slot] - extra[slot]), active[slot]),
        )
        for slot in order[:remainder]:
            shots[active[slot]] += 1
    if sum(shots) != total_shots:
        raise RuntimeError("Sector-range integer-allocation replay did not conserve shots")
    return shots


def validate_sector_range_aggregate_contract(
    record: dict[str, Any], *, total_settings: int, context: str
) -> dict[str, Any]:
    required = {
        "centered_range_domain", "full_fock_centered_range_sum",
        "sector_centered_range_sum", "sector_spectral_centered_range_sum",
        "sector_leakage_guard_increment_sum", "sector_range_sum_reduction_fraction",
        "maximum_sector_leakage_relative_frobenius",
        "all_settings_pass_sector_leakage_check", "sector_range_setting_count",
        "full_fock_fallback_setting_count",
        "all_settings_sector_safe_or_full_fock_fallback",
    }
    missing = sorted(required - set(record))
    if missing:
        raise RuntimeError(f"{context}: missing sector-range aggregate fields {missing}")
    sector_count = int(record["sector_range_setting_count"])
    fallback_count = int(record["full_fock_fallback_setting_count"])
    if sector_count < 0 or fallback_count < 0 or sector_count + fallback_count != total_settings:
        raise RuntimeError(f"{context}: sector/fallback counts do not cover all settings")
    expected_domain = SECTOR_RANGE_DOMAIN if fallback_count == 0 else MIXED_RANGE_DOMAIN
    if str(record["centered_range_domain"]) != expected_domain:
        raise RuntimeError(f"{context}: aggregate range domain/count mismatch")
    if not parse_bool(record["all_settings_sector_safe_or_full_fock_fallback"]):
        raise RuntimeError(f"{context}: an allocation setting is neither certified nor fallback")

    full_sum = finite_nonnegative_range(
        record["full_fock_centered_range_sum"], f"{context} full-Fock sum"
    )
    allocation_sum = finite_nonnegative_range(
        record["sector_centered_range_sum"], f"{context} allocation sum"
    )
    spectral_sum = finite_nonnegative_range(
        record["sector_spectral_centered_range_sum"], f"{context} sector spectral sum"
    )
    leakage = finite_nonnegative_range(
        record["maximum_sector_leakage_relative_frobenius"], f"{context} leakage"
    )
    guard = float(record["sector_leakage_guard_increment_sum"])
    if not math.isfinite(guard):
        raise RuntimeError(f"{context}: non-finite leakage-guard sum")
    if allocation_sum > full_sum + SECTOR_RANGE_TOLERANCE * max(1.0, full_sum):
        raise RuntimeError(f"{context}: allocation range exceeds full-Fock range")
    if spectral_sum > allocation_sum + SECTOR_RANGE_TOLERANCE * max(1.0, allocation_sum):
        raise RuntimeError(f"{context}: sector spectral range exceeds allocation range")
    if not sector_range_close(guard, allocation_sum - spectral_sum):
        raise RuntimeError(f"{context}: leakage-guard sum does not replay")
    all_pass = parse_bool(record["all_settings_pass_sector_leakage_check"])
    if fallback_count == 0:
        if not all_pass or leakage > SECTOR_LEAKAGE_TOLERANCE:
            raise RuntimeError(f"{context}: sector eligibility/leakage mismatch")
    elif all_pass or leakage <= SECTOR_LEAKAGE_TOLERANCE:
        raise RuntimeError(f"{context}: fallback domain lacks above-threshold leakage")
    expected_reduction = 0.0 if full_sum <= 0.0 else 1.0 - allocation_sum / full_sum
    if not sector_range_close(
        float(record["sector_range_sum_reduction_fraction"]), expected_reduction
    ):
        raise RuntimeError(f"{context}: sector-range reduction fraction does not replay")
    return {
        "sector_count": sector_count,
        "fallback_count": fallback_count,
        "domain": expected_domain,
        "full_sum": full_sum,
        "allocation_sum": allocation_sum,
        "spectral_sum": spectral_sum,
        "maximum_leakage": leakage,
    }


def validate_sector_range_selected_setting_ranges(
    best: dict[str, Any],
    sampling: dict[str, Any],
    cover_audit: dict[str, Any],
    *,
    total_shots: int,
    reference_shots: list[int],
    context: str,
) -> dict[str, Any]:
    required_arrays = {
        "setting_centered_ranges", "setting_full_fock_centered_ranges",
        "setting_sector_spectral_centered_ranges", "setting_allocation_centered_ranges",
        "setting_range_domains", "setting_sector_leakage_relative_frobenius",
        "setting_sector_range_eligible",
    }
    missing = sorted(required_arrays - set(sampling))
    if missing:
        raise RuntimeError(f"{context}: missing sector-range selected-setting arrays {missing}")
    total_settings = int(best["total_settings"])
    arrays = {key: list(sampling[key]) for key in required_arrays}
    if any(len(values) != total_settings for values in arrays.values()):
        raise RuntimeError(
            f"{context}: sector-range selected-setting arrays are not K+L_A complete"
        )
    effective = [
        finite_nonnegative_range(value, f"{context} effective range")
        for value in arrays["setting_centered_ranges"]
    ]
    allocation = [
        finite_nonnegative_range(value, f"{context} allocation range")
        for value in arrays["setting_allocation_centered_ranges"]
    ]
    full = [
        finite_nonnegative_range(value, f"{context} full-Fock range")
        for value in arrays["setting_full_fock_centered_ranges"]
    ]
    spectral = [
        finite_nonnegative_range(value, f"{context} sector spectral range")
        for value in arrays["setting_sector_spectral_centered_ranges"]
    ]
    leakages = [
        finite_nonnegative_range(value, f"{context} sector leakage")
        for value in arrays["setting_sector_leakage_relative_frobenius"]
    ]
    domains = [str(value) for value in arrays["setting_range_domains"]]
    eligibility = [parse_bool(value) for value in arrays["setting_sector_range_eligible"]]
    for index, values in enumerate(
        zip(effective, allocation, full, spectral, leakages, domains, eligibility)
    ):
        eff, alloc, full_value, spec, leakage, domain, eligible = values
        expected_eligible = leakage <= SECTOR_LEAKAGE_TOLERANCE
        expected_domain = SECTOR_RANGE_DOMAIN if expected_eligible else FALLBACK_RANGE_DOMAIN
        if eligible != expected_eligible or domain != expected_domain:
            raise RuntimeError(f"{context} setting {index}: domain/leakage/eligibility mismatch")
        if not sector_range_close(eff, alloc):
            raise RuntimeError(f"{context} setting {index}: effective/allocation mismatch")
        if alloc > full_value + SECTOR_RANGE_TOLERANCE * max(1.0, full_value):
            raise RuntimeError(f"{context} setting {index}: allocation exceeds full-Fock range")
        if spec > alloc + SECTOR_RANGE_TOLERANCE * max(1.0, alloc):
            raise RuntimeError(f"{context} setting {index}: spectral range exceeds allocation")
        if not eligible and not sector_range_close(alloc, full_value):
            raise RuntimeError(f"{context} setting {index}: fallback did not use full-Fock range")

    aggregate = validate_sector_range_aggregate_contract(
        best, total_settings=total_settings, context=f"{context} best"
    )
    cover_aggregate = validate_sector_range_aggregate_contract(
        cover_audit, total_settings=total_settings, context=f"{context} cover audit"
    )
    expected_sector_count = sum(eligibility)
    expected_fallback_count = total_settings - expected_sector_count
    if (
        aggregate["sector_count"] != expected_sector_count
        or aggregate["fallback_count"] != expected_fallback_count
        or not sector_range_close(aggregate["full_sum"], sum(full))
        or not sector_range_close(aggregate["allocation_sum"], sum(allocation))
        or not sector_range_close(aggregate["spectral_sum"], sum(spectral))
        or not sector_range_close(aggregate["maximum_leakage"], max(leakages, default=0.0))
    ):
        raise RuntimeError(
            f"{context}: best sector-range aggregate does not replay per-setting arrays"
        )
    for key in aggregate:
        left, right = aggregate[key], cover_aggregate[key]
        if isinstance(left, float):
            if not sector_range_close(left, float(right)):
                raise RuntimeError(
                    f"{context}: best/cover sector-range mismatch for {key}"
                )
        elif left != right:
            raise RuntimeError(
                f"{context}: best/cover sector-range mismatch for {key}"
            )
    if int(cover_audit.get("measurement_setting_count", -1)) != total_settings:
        raise RuntimeError(f"{context}: sector-range cover-audit setting count mismatch")
    if replay_integer_range_allocation(total_shots, allocation) != reference_shots:
        raise RuntimeError(
            f"{context}: saved shot vector does not replay sector-range allocation"
        )
    # The production allocator treats every strictly positive effective range
    # as active, including sub-1e-14 leakage guards from dense-sector refits.
    active = [index for index, value in enumerate(allocation) if value > 0.0]
    proxy = sum(allocation[index] ** 2 / reference_shots[index] for index in active)
    if int(best["nonconstant_settings"]) != len(active):
        raise RuntimeError(f"{context}: sector-range nonconstant-setting count mismatch")
    if (
        not sector_range_close(proxy, float(best["range_variance_proxy"]))
        or not sector_range_close(math.sqrt(max(proxy, 0.0)), float(best["x_sampling"]))
        or not sector_range_close(
            float(best["centered_spectral_norm_sum"]), aggregate["allocation_sum"]
        )
        or not sector_range_close(
            float(best["centered_spectral_norm_sum_before_R2"]),
            aggregate["allocation_sum"],
        )
    ):
        raise RuntimeError(f"{context}: sector-range proxy/sum replay failed")
    return aggregate


def first_field(row: pd.Series, names: tuple[str, ...], *, required: bool = True) -> Any:
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    if required:
        raise RuntimeError(f"None of the required fields are present: {names}")
    return None


def collector_metadata(row: pd.Series, k: int) -> dict[str, Any]:
    """Read either full-space or four-orbital shallow-collector schema."""
    mode = str(first_field(row, ("collector_mode",)))
    if mode != "augment":
        raise RuntimeError(
            f"Resource publication forbids diagnostic collector mode {mode!r}"
        )
    extra = int(first_field(
        row, ("collector_extra_leaves", "collector_extra_settings")
    ))
    settings = int(first_field(
        row, ("measurement_settings", "total_settings")
    ))
    if extra < 1:
        raise RuntimeError("Formal shallow augmentation requires L >= 1")
    if settings != k + extra:
        raise RuntimeError(f"Shallow setting count {settings} is not K+L={k + extra}")
    independent = first_field(
        row,
        ("independent_collector_setting_present", "independent_collector_setting"),
        required=False,
    )
    if independent is not None and parse_bool(independent):
        raise RuntimeError("Independent collector setting found in publication data")
    residual = first_field(
        row,
        (
            "collector_matrix_relative_reconstruction_residual",
            "collector_reconstruction_residual",
        ),
    )
    if not np.isfinite(float(residual)) or float(residual) > 1.0e-8:
        raise RuntimeError(f"Collector reconstruction residual is too large: {residual}")
    return {
        "mode": mode,
        "extra": extra,
        "settings": settings,
        "independent": False,
        "reconstruction_residual": float(residual),
        "row_independent_evidence_present": independent is not None,
    }


def validate_srcdf_audit(
    candidate_path: Path,
    collector: dict[str, Any],
) -> Path:
    audit_path = candidate_path.parent / "audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing shallow-collector audit: {audit_path}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError("Shallow-collector audit is not PASS")
    configuration = payload.get("collector_configuration")
    if isinstance(configuration, dict):
        required = {
            "mode", "extra_leaf_grid", "objective", "proxy_state",
            "equality_tolerance",
        }
        if not required.issubset(configuration):
            raise RuntimeError("Full-space collector audit configuration is incomplete")
        if str(configuration.get("mode")) != "augment":
            raise RuntimeError("Candidate and audit collector modes disagree")
        if str(configuration.get("objective")) != "greedy":
            raise RuntimeError("Publication resource rows require the greedy collector")
        if str(configuration.get("proxy_state")) != "hf":
            raise RuntimeError("Publication resource rows require the HF collector proxy")
        extra_grid = [int(value) for value in configuration.get("extra_leaf_grid", [])]
        if collector["extra"] not in extra_grid:
            raise RuntimeError("Selected L is absent from the audited collector grid")
        if payload.get("all_feasible_collector_matrix_reconstructions_pass") is not True:
            raise RuntimeError("Full-space collector audit reports a failed reconstruction")
        if payload.get("sRCDF_ground_state_used_to_optimize_collector") is not False:
            raise RuntimeError("Full-space collector audit used the exact state in selection")
    else:
        required = {
            "collector_mode",
            "collector_objective",
            "collector_proxy_state",
            "minimum_extra_settings_over_rank_grid",
            "maximum_collector_relative_reconstruction_residual",
            "all_measurement_settings_depth_limited",
            "independent_collector_setting_present",
        }
        if not required.issubset(payload):
            missing = sorted(required - set(payload))
            raise RuntimeError(f"Active-space collector audit is incomplete: {missing}")
        if (
            payload["collector_mode"] != "augment"
            or payload["collector_objective"] != "greedy"
            or payload["collector_proxy_state"] != "hf"
            or int(payload["minimum_extra_settings_over_rank_grid"]) < 1
            or payload["all_measurement_settings_depth_limited"] is not True
            or payload["independent_collector_setting_present"] is not False
            or float(payload["maximum_collector_relative_reconstruction_residual"])
            > 5.0e-10
        ):
            raise RuntimeError("Active-space collector audit violates the formal contract")
        if (
            "historical_exact_collector_settings_reused" in payload
            and payload["historical_exact_collector_settings_reused"] is not False
        ):
            raise RuntimeError("Historical exact collector settings were reused")
        if (
            "production_collector_contract_passed" in payload
            and payload["production_collector_contract_passed"] is not True
        ):
            raise RuntimeError("Active-space production collector contract did not pass")
        post_selection = payload.get(
            "exact_ground_state_used_only_after_selection",
            payload.get("exact_ground_state_used_only_after_weight_and_K_freeze"),
        )
        if post_selection is not True:
            raise RuntimeError("Active-space audit does not withhold the exact state")
    return audit_path


def validate_srcdf_noise(frame: pd.DataFrame) -> None:
    """Fail closed on stale independent-collector noise tables."""
    required = {
        "molecule", "p", "T_total_shots", "selected_K", "selected_L",
        "selected_depth", "collector_mode", "measurement_settings",
        "maximum_logical_cx_depth", "independent_collector_setting_present",
        "deterministic_constant_added_once", "shot_vector",
        "collector_matrix_relative_reconstruction_residual", "error_hartree",
    }
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise RuntimeError(f"SRDD noise table lacks shallow-collector fields: {missing}")
    for molecule in ("BeH2", "N2"):
        rows = frame.loc[frame["molecule"] == molecule].sort_values("p")
        if len(rows) != 10 or rows["p"].nunique() != 10:
            raise RuntimeError(f"Expected ten unique noise rates for {molecule}")
        if not np.allclose(
            rows["p"].to_numpy(dtype=float), np.linspace(0.0, 0.003, 10),
            rtol=0.0, atol=1.0e-15,
        ):
            raise RuntimeError(f"Noise grid is not the audited [0,0.003] grid for {molecule}")
        for row in rows.itertuples(index=False):
            k = int(row.selected_K)
            extra = int(row.selected_L)
            settings = int(row.measurement_settings)
            depth = int(row.selected_depth)
            shots = parse_shot_vector(row.shot_vector)
            mode = str(row.collector_mode)
            if mode != "augment":
                raise RuntimeError(f"Invalid collector mode in noise table for {molecule}")
            if extra < 1:
                raise RuntimeError(f"Collector mode/L mismatch in noise table for {molecule}")
            if (
                settings != k + extra
                or len(shots) != settings
                or int(row.T_total_shots) != 3000
                or sum(shots) != 3000
                or int(row.maximum_logical_cx_depth) != 4 * depth
            ):
                raise RuntimeError(f"Noise setting/shot/depth audit failed for {molecule}")
            if parse_bool(row.independent_collector_setting_present):
                raise RuntimeError(f"Independent collector in noise table for {molecule}")
            if not parse_bool(row.deterministic_constant_added_once):
                raise RuntimeError(f"Constant-offset audit failed for {molecule}")
            if float(row.collector_matrix_relative_reconstruction_residual) > 5.0e-10:
                raise RuntimeError(f"Collector reconstruction failed for {molecule}")


def srcdf_cx_resources(depth: int, spatial_orbitals: int) -> tuple[int, int]:
    count = 2 * depth * (spatial_orbitals - 1) * CX_GIVENS
    cx_depth = 2 * depth * CX_GIVENS
    return count, cx_depth


def agpd_shallow_setting_metadata(
    payload: dict[str, Any],
    *,
    expected_k: int,
    expected_total_shots: int,
    context: str,
    artifact_contract: str,
    n_qubits: int = 8,
) -> dict[str, Any]:
    """Validate and compile one complete AGPD cover-plus-source setting list."""
    supported_contracts = {
        "v8_full_fock_range",
        "v9_sector_range",
        "v10_fully_corrective_dense_sector",
    }
    if artifact_contract not in supported_contracts:
        raise RuntimeError(f"{context}: unsupported AGPD contract {artifact_contract!r}")
    sector_range_contract = artifact_contract in {
        "v9_sector_range",
        "v10_fully_corrective_dense_sector",
    }
    fully_corrective_contract = artifact_contract == "v10_fully_corrective_dense_sector"
    best = payload.get("best")
    sampling = payload.get("selected_sampling")
    if not isinstance(best, dict) or not isinstance(sampling, dict):
        raise RuntimeError(f"{context}: missing AGPD best/selected_sampling metadata")

    required_best = {
        "K_source_terms", "shot_vector", "allocated_total_shots",
        "nonconstant_settings", "total_settings",
        "standard_one_body_F3_source_count",
        "generalized_dense_F3_source_count", "F3_transfer_source_count",
        "source_to_collector_transfer_count",
        "complete_native_source_setting_count",
        "shallow_one_body_cover_setting_count", "shallow_one_body_cover_depth",
        "independent_collector_setting_present",
    }
    required_sampling = {
        "selected_K_source_terms", "allocated_total_shots",
        "shot_vector_exactly_reused", "setting_families", "setting_interfaces",
        "setting_kinds", "setting_rotation_depths", "setting_centered_ranges",
        "shallow_one_body_cover_setting_count", "shallow_one_body_cover_depth",
        "source_to_collector_transfer_count",
        "independent_collector_setting_present", "shallow_one_body_cover_audit",
    }
    if sector_range_contract:
        required_best.update({
            "centered_range_domain", "full_fock_centered_range_sum",
            "sector_centered_range_sum", "sector_spectral_centered_range_sum",
            "sector_leakage_guard_increment_sum", "sector_range_sum_reduction_fraction",
            "maximum_sector_leakage_relative_frobenius",
            "all_settings_pass_sector_leakage_check", "sector_range_setting_count",
            "full_fock_fallback_setting_count",
            "all_settings_sector_safe_or_full_fock_fallback",
        })
        required_sampling.update({
            "setting_full_fock_centered_ranges",
            "setting_sector_spectral_centered_ranges",
            "setting_allocation_centered_ranges", "setting_range_domains",
            "setting_sector_leakage_relative_frobenius",
            "setting_sector_range_eligible",
        })
    if fully_corrective_contract:
        required_best.update({
            "x_sector_operator_norm", "selection_rule",
            "selection_uses_calibrated_probe_mean_weights",
            "fully_corrective_dense_sector_audit",
        })
        required_sampling.update({
            "selection_sector_operator_norm", "selection_rule",
            "fully_corrective_dense_sector_audit",
        })
    missing_best = sorted(required_best - set(best))
    missing_sampling = sorted(required_sampling - set(sampling))
    if missing_best or missing_sampling:
        raise RuntimeError(
            f"{context}: incomplete AGPD {artifact_contract} metadata; "
            f"best missing {missing_best}, "
            f"selected_sampling missing {missing_sampling}"
        )

    k = int(best["K_source_terms"])
    cover_settings = int(best["shallow_one_body_cover_setting_count"])
    cover_depth = int(best["shallow_one_body_cover_depth"])
    total_settings = int(best["total_settings"])
    if (
        k != expected_k
        or int(sampling["selected_K_source_terms"]) != k
        or int(best["complete_native_source_setting_count"]) != k
    ):
        raise RuntimeError(f"{context}: AGPD native-source count does not equal K={expected_k}")

    fully_corrective_metadata: dict[str, Any] = {
        "selection_rule": None,
        "sector_operator_norm": None,
        "fully_corrective_rcond": None,
        "fully_corrective_design_rank": None,
        "fully_corrective_design_nullity": None,
    }
    if fully_corrective_contract:
        correction = best["fully_corrective_dense_sector_audit"]
        sampling_correction = sampling["fully_corrective_dense_sector_audit"]
        if not isinstance(correction, dict) or sampling_correction != correction:
            raise RuntimeError(f"{context}: selected fully-corrective audit does not replay")
        operator_norm = finite_nonnegative_range(
            best["x_sector_operator_norm"], f"{context} sector operator norm"
        )
        design_rank = int(correction.get("design_rank", -1))
        design_nullity = int(correction.get("design_nullity", -1))
        coefficient_shape = correction.get("coefficient_shape")
        if (
            best["selection_rule"] != AGPD_V10_SELECTION_RULE
            or sampling["selection_rule"] != AGPD_V10_SELECTION_RULE
            or parse_bool(best["selection_uses_calibrated_probe_mean_weights"])
            or not sector_range_close(
                float(sampling["selection_sector_operator_norm"]), operator_norm
            )
            or correction.get("method")
            != "fully_corrective_dense_sector_block_dd"
            or correction.get("objective")
            != "sector_frobenius_fixed_rotations_joint_dd_coefficients"
            or correction.get("solver") != "numpy.linalg.lstsq_svd_minimum_norm"
            or correction.get("uses_ground_state") is not False
            or correction.get("fixed_rotations") is not True
            or float(correction.get("svd_rcond", -1.0)) != AGPD_V10_RCOND
            or int(correction.get("source_count", -1)) != k
            or int(correction.get("dd_feature_count_per_source", -1)) != 37
            or coefficient_shape != [k, 37]
            or design_rank <= 0
            or design_nullity < 0
            or design_rank + design_nullity != 37 * k
            or float(correction.get("sector_frobenius_after", math.inf))
            > float(correction.get("sector_frobenius_before", -math.inf)) + 1.0e-10
            or not sector_range_close(
                float(correction.get("sector_operator_norm_after", math.nan)),
                operator_norm,
            )
        ):
            raise RuntimeError(
                f"{context}: invalid fully-corrective dense-sector selection/audit"
            )
        fully_corrective_metadata.update({
            "selection_rule": AGPD_V10_SELECTION_RULE,
            "sector_operator_norm": operator_norm,
            "fully_corrective_rcond": AGPD_V10_RCOND,
            "fully_corrective_design_rank": design_rank,
            "fully_corrective_design_nullity": design_nullity,
        })
    if cover_settings < 1 or cover_depth < 1 or total_settings != cover_settings + k:
        raise RuntimeError(f"{context}: AGPD setting count is not L_A+K={cover_settings + k}")
    if (
        int(best["nonconstant_settings"]) != total_settings
        or int(sampling["shallow_one_body_cover_setting_count"]) != cover_settings
        or int(sampling["shallow_one_body_cover_depth"]) != cover_depth
    ):
        raise RuntimeError(f"{context}: AGPD cover/settings metadata disagree")

    reference_shots = parse_shot_vector(best["shot_vector"])
    if (
        int(payload.get("total_shots", -1)) != expected_total_shots
        or int(best["allocated_total_shots"]) != expected_total_shots
        or int(sampling["allocated_total_shots"]) != expected_total_shots
        or len(reference_shots) != total_settings
        or sum(reference_shots) != expected_total_shots
        or any(value < 1 for value in reference_shots)
        or sampling["shot_vector_exactly_reused"] is not True
    ):
        raise RuntimeError(f"{context}: AGPD complete shot-vector audit failed")

    zero_count_fields = (
        "standard_one_body_F3_source_count",
        "generalized_dense_F3_source_count",
        "F3_transfer_source_count",
        "source_to_collector_transfer_count",
    )
    if any(int(best[field]) != 0 for field in zero_count_fields):
        raise RuntimeError(f"{context}: AGPD forbids F3/source-to-cover transfer")
    if int(sampling["source_to_collector_transfer_count"]) != 0:
        raise RuntimeError(f"{context}: selected sampling contains source-to-cover transfer")
    if (
        parse_bool(best["independent_collector_setting_present"])
        or parse_bool(sampling["independent_collector_setting_present"])
    ):
        raise RuntimeError(f"{context}: independent AGPD collector setting detected")

    families = [str(value) for value in sampling["setting_families"]]
    interfaces = [str(value) for value in sampling["setting_interfaces"]]
    kinds = [str(value) for value in sampling["setting_kinds"]]
    depths = [int(value) for value in sampling["setting_rotation_depths"]]
    centered_ranges = list(sampling["setting_centered_ranges"])
    if not all(
        len(values) == total_settings
        for values in (families, interfaces, kinds, depths, centered_ranges)
    ):
        raise RuntimeError(f"{context}: AGPD per-setting arrays are incomplete")
    cover_slice = slice(0, cover_settings)
    source_slice = slice(cover_settings, total_settings)
    if (
        any(value != "agpd_shallow_one_body" for value in families[cover_slice])
        or any(
            value != "agpd_depth_bounded_one_body_cover"
            for value in interfaces[cover_slice]
        )
        or any(value != "agpd_shallow_one_body_leaf" for value in kinds[cover_slice])
        or any(value != cover_depth for value in depths[cover_slice])
        or any(
            value != "native_shallow_source_no_collector_transfer"
            for value in interfaces[source_slice]
        )
        or any(value != "source_leaf" for value in kinds[source_slice])
        or any(value <= 0 for value in depths[source_slice])
    ):
        raise RuntimeError(
            f"{context}: settings must be L_A cover leaves first, then K native sources"
        )

    cover_audit = sampling["shallow_one_body_cover_audit"]
    if not isinstance(cover_audit, dict):
        raise RuntimeError(f"{context}: missing shallow one-body-cover audit")
    expected_audit = {
        "status": "PASS", "mode": "augment", "objective": "greedy",
        "proxy_state": "hf",
        "zero_rank_depth_selection": "smallest_strictly_feasible_depth",
        "selected_candidate": "collector_tailored_greedy_exact_fill",
        "method": "AGPD",
        "policy": "fixed_shallow_one_body_cover_and_complete_native_sources",
        "source_transfer_policy": "disabled_for_all_families",
    }
    if any(cover_audit.get(key) != value for key, value in expected_audit.items()):
        raise RuntimeError(f"{context}: AGPD shallow-cover policy audit failed")
    if (
        int(cover_audit.get("selected_zero_rank_depth", -1)) != cover_depth
        or int(cover_audit.get("collector_depth", -1)) != cover_depth
        or int(cover_audit.get("extra_rotation_count", -1)) != cover_settings
        or int(cover_audit.get("cover_setting_count", -1)) != cover_settings
        or int(cover_audit.get("measurement_setting_count", -1)) != total_settings
        or int(cover_audit.get("source_rotation_count", -1)) != 0
        or int(cover_audit.get("source_setting_count", -1)) != k
        or int(cover_audit.get("source_transfer_count", -1)) != 0
        or cover_audit.get("complete_source_matrices_retained") is not True
        or cover_audit.get("all_measurement_settings_depth_limited") is not True
        or parse_bool(cover_audit.get("independent_collector_setting_present", True))
    ):
        raise RuntimeError(
            f"{context}: AGPD cover/source audit violates the {artifact_contract} contract"
        )
    depth_attempts = cover_audit.get("zero_rank_depth_attempts")
    if not isinstance(depth_attempts, list) or not depth_attempts:
        raise RuntimeError(f"{context}: missing AGPD minimum-feasible-depth attempts")
    selected_attempts = [
        item for item in depth_attempts
        if int(item.get("rotation_depth", -1)) == cover_depth
    ]
    if (
        len(selected_attempts) != 1
        or selected_attempts[0].get("status") != "PASS"
        or any(
            int(item.get("rotation_depth", -1)) < cover_depth
            and item.get("status") == "PASS"
            for item in depth_attempts
        )
    ):
        raise RuntimeError(f"{context}: AGPD cover depth is not the first feasible depth")
    reconstruction_error = float(
        cover_audit.get("matrix_reconstruction_residual_frobenius", math.inf)
    )
    reconstruction_tolerance = float(
        cover_audit.get("matrix_reconstruction_tolerance_frobenius", -math.inf)
    )
    if (
        not np.isfinite(reconstruction_error)
        or not np.isfinite(reconstruction_tolerance)
        or reconstruction_tolerance < 0.0
        or reconstruction_error > reconstruction_tolerance
    ):
        raise RuntimeError(f"{context}: AGPD shallow-cover reconstruction audit failed")

    range_metadata: dict[str, Any] = {
        "artifact_contract": artifact_contract,
        "centered_range_domain": "full_fock" if artifact_contract == "v8_full_fock_range" else None,
        "full_fock_centered_range_sum": float(best["centered_spectral_norm_sum"]),
        "sector_centered_range_sum": None,
        "sector_range_setting_count": None,
        "full_fock_fallback_setting_count": None,
        "maximum_sector_leakage_relative_frobenius": None,
    }
    if sector_range_contract:
        aggregate = validate_sector_range_selected_setting_ranges(
            best,
            sampling,
            cover_audit,
            total_shots=expected_total_shots,
            reference_shots=reference_shots,
            context=context,
        )
        range_metadata.update({
            "centered_range_domain": aggregate["domain"],
            "full_fock_centered_range_sum": aggregate["full_sum"],
            "sector_centered_range_sum": aggregate["allocation_sum"],
            "sector_range_setting_count": aggregate["sector_count"],
            "full_fock_fallback_setting_count": aggregate["fallback_count"],
            "maximum_sector_leakage_relative_frobenius": aggregate["maximum_leakage"],
        })

    spatial_orbitals = n_qubits // 2
    cover_count = 4 * cover_depth * (spatial_orbitals - 1)
    cover_cx_depth = 4 * cover_depth
    resources = [GateResources(cnot_count=cover_count, two_qubit_depth=cover_cx_depth)
                 for _ in range(cover_settings)]
    resources.extend(
        source_gate_resources(family, depth, n_qubits)
        for family, depth in zip(families[source_slice], depths[source_slice])
    )
    if len(resources) != total_settings:
        raise RuntimeError(f"{context}: compiled resource vector is not L_A+K")
    return {
        "k": k, "cover_settings": cover_settings, "cover_depth": cover_depth,
        "total_settings": total_settings, "reference_shots": reference_shots,
        "resources": resources, "cover_cx_count": cover_count,
        "cover_cx_depth": cover_cx_depth,
        **fully_corrective_metadata,
        **range_metadata,
    }


def agpd_record(molecule: str, summary_row: pd.Series) -> dict[str, Any]:
    total = int(summary_row["T_total_shots"])
    root, artifact_contract = agpd_root_and_contract(molecule)
    path = root / molecule / f"budget_T{total}.json"
    validate_agpd_manifest(
        path.parent / "manifest.json", artifact_contract, f"{molecule} fixed geometry"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    k = int(summary_row["K_selected"])
    metadata = agpd_shallow_setting_metadata(
        payload,
        expected_k=k,
        expected_total_shots=total,
        context=f"{molecule} T={total}",
        artifact_contract=artifact_contract,
    )
    if (
        int(summary_row["measurement_settings"]) != metadata["total_settings"]
        or str(summary_row["collector_mode"]) != "fixed_shallow_one_body_cover"
        or int(summary_row["collector_extra_leaves"]) != metadata["cover_settings"]
        or int(summary_row["collector_depth"]) != metadata["cover_depth"]
        or int(summary_row["source_to_collector_transfer_count"]) != 0
        or parse_bool(summary_row["independent_collector_setting_present"])
    ):
        raise RuntimeError(
            f"{molecule} T={total}: comparison row is not AGPD {artifact_contract}"
        )
    if artifact_contract in {
        "v9_sector_range",
        "v10_fully_corrective_dense_sector",
    }:
        comparison_range_fields = {
            "centered_range_domain": metadata["centered_range_domain"],
            "full_fock_centered_range_sum": metadata["full_fock_centered_range_sum"],
            "sector_centered_range_sum": metadata["sector_centered_range_sum"],
            "maximum_sector_leakage_relative_frobenius": metadata[
                "maximum_sector_leakage_relative_frobenius"
            ],
        }
        for field, expected in comparison_range_fields.items():
            if field not in summary_row.index or pd.isna(summary_row[field]):
                raise RuntimeError(
                    f"{molecule} T={total}: comparison lacks range field {field}"
                )
            observed = summary_row[field]
            if isinstance(expected, float):
                if not sector_range_close(float(observed), expected):
                    raise RuntimeError(
                        f"{molecule} T={total}: comparison/artifact mismatch for {field}"
                    )
            elif str(observed) != str(expected):
                raise RuntimeError(
                    f"{molecule} T={total}: comparison/artifact mismatch for {field}"
                )
        expected_provenance = (
            "v10_fully_corrective_dense_sector"
            if artifact_contract == "v10_fully_corrective_dense_sector"
            else "v9_sector_range"
        )
        if expected_provenance not in str(summary_row.get("provenance", "")):
            raise RuntimeError(
                f"{molecule} T={total}: comparison provenance mismatch"
            )
        if artifact_contract == "v10_fully_corrective_dense_sector":
            fully_corrective_fields = {
                "sector_operator_norm": metadata["sector_operator_norm"],
                "fully_corrective_design_rank": metadata[
                    "fully_corrective_design_rank"
                ],
                "fully_corrective_design_nullity": metadata[
                    "fully_corrective_design_nullity"
                ],
                "fully_corrective_rcond": AGPD_V10_RCOND,
                "selection_rule": AGPD_V10_SELECTION_RULE,
                "selection_uses_exact_ground_state": False,
            }
            for field, expected in fully_corrective_fields.items():
                if field not in summary_row.index or pd.isna(summary_row[field]):
                    raise RuntimeError(
                        f"{molecule} T={total}: comparison lacks v10 field {field}"
                    )
                observed = summary_row[field]
                if isinstance(expected, bool):
                    matches = parse_bool(observed) is expected
                elif isinstance(expected, float):
                    matches = sector_range_close(float(observed), expected)
                elif isinstance(expected, int):
                    matches = int(observed) == expected
                else:
                    matches = str(observed) == str(expected)
                if not matches:
                    raise RuntimeError(
                        f"{molecule} T={total}: comparison/artifact mismatch "
                        f"for {field}"
                    )
    elif any(
        tag in str(summary_row.get("provenance", "")).lower()
        for tag in ("v9", "v10")
    ):
        raise RuntimeError(f"{molecule} T={total}: H6 v8 row was relabelled")
    plan = analytic_shots_to_target(
        total,
        float(summary_row["analytic_sampling_SE"]),
        float(summary_row["signed_approximation_bias"]),
    )
    if plan["shots_to_target"] is not None:
        plan["unfloored_shots_to_target"] = int(plan["shots_to_target"])
        plan["shots_to_target"] = max(
            int(plan["shots_to_target"]), metadata["total_settings"]
        )
    allocation = None
    if plan["shots_to_target"] is not None:
        allocation = complete_setting_largest_remainder(
            int(plan["shots_to_target"]), metadata["reference_shots"]
        )
        if len(allocation) != metadata["total_settings"]:
            raise RuntimeError("AGPD target allocation does not cover all L_A+K settings")
    gate_summary = summarize_gate_resources(metadata["resources"], allocation)
    return {
        "decomposition_items": metadata["total_settings"],
        "source_leaves_K": k,
        "extra_one_body_leaves_L": metadata["cover_settings"],
        "shallow_one_body_cover_leaves_L_A": metadata["cover_settings"],
        "collector_mode": "fixed_shallow_one_body_cover",
        "shallow_one_body_cover_depth_d_A": metadata["cover_depth"],
        "source_to_collector_transfer_count": 0,
        "independent_collector_setting_present": False,
        **gate_summary,
        "cover_two_qubit_depth": metadata["cover_cx_depth"],
        "cover_two_qubit_gate_count_per_setting": metadata["cover_cx_count"],
        "shots_to_accuracy": plan["shots_to_target"],
        "unfloored_shots_to_accuracy": plan.get(
            "unfloored_shots_to_target", plan["shots_to_target"]
        ),
        "status": plan["status"],
        "reference_total_shots": total,
        "source": str(path.resolve()),
        "analytic_sampling_se": float(summary_row["analytic_sampling_SE"]),
        "signed_bias": float(summary_row["signed_approximation_bias"]),
        "variance_coefficient": plan["variance_coefficient"],
        "fixed_estimator_scaling_exponent": FIXED_ESTIMATOR_SCALING_EXPONENT,
        "fixed_estimator_model": FIXED_ESTIMATOR_MODEL,
        "all_setting_shots_counted_in_gate_budget": True,
        "setting_order": "shallow_one_body_cover_then_complete_native_sources",
        "agpd_artifact_contract": artifact_contract,
        "centered_range_domain": metadata["centered_range_domain"],
        "full_fock_centered_range_sum": metadata["full_fock_centered_range_sum"],
        "sector_centered_range_sum": metadata["sector_centered_range_sum"],
        "sector_range_setting_count": metadata["sector_range_setting_count"],
        "full_fock_fallback_setting_count": metadata[
            "full_fock_fallback_setting_count"
        ],
        "maximum_sector_leakage_relative_frobenius": metadata[
            "maximum_sector_leakage_relative_frobenius"
        ],
        "sector_operator_norm": metadata["sector_operator_norm"],
        "fully_corrective_design_rank": metadata[
            "fully_corrective_design_rank"
        ],
        "fully_corrective_design_nullity": metadata[
            "fully_corrective_design_nullity"
        ],
        "fully_corrective_rcond": metadata["fully_corrective_rcond"],
        "selection_rule": metadata["selection_rule"],
        "selection_uses_exact_ground_state": (
            False
            if artifact_contract == "v10_fully_corrective_dense_sector"
            else None
        ),
    }


def selected_srcdf_candidate(
    molecule: str,
    k: int,
    depth: int,
    total: int,
    summary_row: pd.Series,
) -> tuple[pd.Series, Path]:
    path = {"BeH2": BEH2_CANDIDATES, "N2": N2_CANDIDATES}.get(
        molecule, SRCDF_ROOT / molecule / "candidate_by_T_K.csv"
    )
    frame = pd.read_csv(path)
    k_key = "K" if "K" in frame.columns else "K_source_terms"
    depth_key = "depth" if "depth" in frame.columns else "shallow_rotation_depth"
    summary_collector = collector_metadata(summary_row, k)
    rows = frame.loc[
        (frame["T_total_shots"] == total)
        & (frame[k_key] == k)
        & (frame[depth_key] == depth)
    ]
    if "collector_mode" in frame:
        rows = rows.loc[rows["collector_mode"] == summary_collector["mode"]]
    extra_key = next(
        (name for name in ("collector_extra_leaves", "collector_extra_settings") if name in frame),
        None,
    )
    if extra_key is not None:
        rows = rows.loc[rows[extra_key] == summary_collector["extra"]]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one SRDD candidate for {molecule}, K={k}, "
            f"L={summary_collector['extra']}, depth={depth}, T={total}"
        )
    candidate = rows.iloc[0]
    candidate_collector = collector_metadata(candidate, k)
    if any(
        candidate_collector[key] != summary_collector[key]
        for key in ("mode", "extra", "settings", "independent")
    ):
        raise RuntimeError("Summary and candidate shallow-collector metadata disagree")
    return candidate, path


def srcdf_record(
    molecule: str,
    summary_row: pd.Series,
    *,
    spatial_orbitals: int,
    full_space_schema: bool,
) -> dict[str, Any]:
    total = int(summary_row["T_total_shots"])
    k = int(first_field(
        summary_row,
        ("selected_K",) if full_space_schema else ("K_selected", "K_source_terms"),
    ))
    depth = int(first_field(
        summary_row,
        ("selected_depth", "shallow_rotation_depth"),
    ))
    se_value = float(first_field(
        summary_row,
        ("analytic_sampling_SE_hartree",)
        if full_space_schema
        else ("analytic_sampling_SE",),
    ))
    bias_value = float(first_field(
        summary_row,
        ("deterministic_bias_hartree",)
        if full_space_schema
        else ("signed_approximation_bias",),
    ))
    candidate, candidate_path = selected_srcdf_candidate(
        molecule, k, depth, total, summary_row
    )
    collector = collector_metadata(candidate, k)
    audit_path = validate_srcdf_audit(candidate_path, collector)
    reference_shots = parse_shot_vector(candidate["shot_vector"])
    if (
        len(reference_shots) != collector["settings"]
        or sum(reference_shots) != total
        or any(value < 0 for value in reference_shots)
    ):
        raise RuntimeError("Reference shallow setting allocation violates shot conservation")
    plan = analytic_shots_to_target(total, se_value, bias_value)
    if plan["shots_to_target"] is not None:
        plan["unfloored_shots_to_target"] = int(plan["shots_to_target"])
        plan["shots_to_target"] = max(
            int(plan["shots_to_target"]), collector["settings"]
        )
    count, cx_depth = srcdf_cx_resources(depth, spatial_orbitals)
    if cx_depth != 4 * depth:
        raise RuntimeError("SRDD maximum logical-CX depth is not 4*d_R")
    gate_budget = None
    if plan["shots_to_target"] is not None:
        allocation = largest_remainder(
            int(plan["shots_to_target"]), reference_shots
        )
        if len(allocation) != collector["settings"]:
            raise RuntimeError("Target allocation does not cover every K+L setting")
        gate_budget = int(np.sum(allocation) * count)
    return {
        "decomposition_items": collector["settings"],
        "source_leaves_K": k,
        "extra_one_body_leaves_L": collector["extra"],
        "collector_mode": collector["mode"],
        "independent_collector_setting_present": False,
        "max_two_qubit_depth": cx_depth,
        "depth_is_upper_bound": False,
        "shots_to_accuracy": plan["shots_to_target"],
        "unfloored_shots_to_accuracy": plan.get(
            "unfloored_shots_to_target", plan["shots_to_target"]
        ),
        "two_qubit_gate_budget": gate_budget,
        "gate_budget_is_upper_bound": False,
        "status": plan["status"],
        "reference_total_shots": total,
        "source": str(candidate_path.resolve()),
        "collector_audit_source": str(audit_path.resolve()),
        "analytic_sampling_se": se_value,
        "signed_bias": bias_value,
        "variance_coefficient": plan["variance_coefficient"],
        "fixed_estimator_scaling_exponent": FIXED_ESTIMATOR_SCALING_EXPONENT,
        "fixed_estimator_model": FIXED_ESTIMATOR_MODEL,
        "all_setting_shots_counted_in_gate_budget": True,
        "collector_reconstruction_residual": collector["reconstruction_residual"],
    }


def fc_record(molecule: str) -> dict[str, Any]:
    """Build the pooled overlapping FC-IMA resource row for a full-space target."""
    summary = pd.read_csv(FC_ERROR_SUMMARY)
    selected = summary.loc[summary["molecule"] == molecule]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one FC-IMA summary row for {molecule}")
    selected = selected.iloc[0]
    required = {
        "fc_overlapping_group_count",
        "grouping",
        "estimator",
        "allocation_optimization",
        "fc_primary_analytic_rmse_hartree",
        "inverse_error_frozen_T0_variance_hartree2",
        "inverse_error_frozen_variance_coefficient_T0_times_variance_hartree2_shots",
        "T_gate",
    }
    if not required.issubset(selected.index):
        missing = sorted(required - set(selected.index))
        raise RuntimeError(f"FC-IMA summary lacks pooled-estimator fields: {missing}")
    if "extended" not in str(selected["grouping"]).lower() or "overlap" not in str(
        selected["grouping"]
    ).lower():
        raise RuntimeError(f"FC-IMA grouping is not extended/overlapping for {molecule}")
    if "pooled" not in str(selected["estimator"]).lower() or "eq. (14)" not in str(
        selected["estimator"]
    ).lower():
        raise RuntimeError(f"FC-IMA summary lacks the Eq. (14) pooled estimator for {molecule}")
    if "ima" not in str(selected["allocation_optimization"]).lower():
        raise RuntimeError(f"FC-IMA summary lacks iterative allocation for {molecule}")
    total = int(selected["T_gate"])
    if total != 3000:
        raise RuntimeError(f"Unexpected FC-IMA reference budget for {molecule}: {total}")
    group_count = int(selected["fc_overlapping_group_count"])
    if group_count <= 0:
        raise RuntimeError(f"Invalid FC-IMA overlapping group count for {molecule}")
    sampling_se = float(selected["fc_primary_analytic_rmse_hartree"])
    plan = analytic_shots_to_target(total, sampling_se, 0.0)
    stored_variance = float(selected["inverse_error_frozen_T0_variance_hartree2"])
    stored_coefficient = float(
        selected[
            "inverse_error_frozen_variance_coefficient_T0_times_variance_hartree2_shots"
        ]
    )
    if not math.isclose(sampling_se**2, stored_variance, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RuntimeError(f"FC-IMA T=3000 RMSE/variance mismatch for {molecule}")
    if not math.isclose(
        plan["variance_coefficient"],
        stored_coefficient,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(f"FC-IMA frozen variance coefficient mismatch for {molecule}")
    if plan["shots_to_target"] is not None:
        plan["unfloored_shots_to_target"] = int(plan["shots_to_target"])
        plan["shots_to_target"] = max(int(plan["shots_to_target"]), group_count)

    groups = pd.read_csv(FC_CIRCUIT_AUDITS[molecule]).sort_values("group_index")
    circuit_required = {
        "group_index",
        "compiled_two_qubit_depth",
        "compiled_cx_count",
        "all_two_qubit_operations_are_explicit_cx",
        "all_input_symplectic_masks_recovered",
    }
    if not circuit_required.issubset(groups.columns):
        missing = sorted(circuit_required - set(groups.columns))
        raise RuntimeError(f"FC-IMA circuit audit lacks fields for {molecule}: {missing}")
    if len(groups) != group_count or not np.array_equal(
        groups["group_index"].to_numpy(dtype=int), np.arange(group_count)
    ):
        raise RuntimeError(f"FC-IMA compiled-group table is inconsistent for {molecule}")
    for column in (
        "all_two_qubit_operations_are_explicit_cx",
        "all_input_symplectic_masks_recovered",
    ):
        if not groups[column].map(parse_bool).all():
            raise RuntimeError(f"FC-IMA circuit audit failed {column} for {molecule}")

    allocations = pd.read_csv(FC_ALLOCATIONS[molecule])
    reference = allocations.loc[
        (allocations["allocation"] == FC_ALLOCATION)
        & (allocations["T_total_shots"] == total)
    ].sort_values("group_index")
    if len(reference) != group_count or not np.array_equal(
        reference["group_index"].to_numpy(dtype=int), np.arange(group_count)
    ):
        raise RuntimeError(f"FC-IMA T=3000 allocation is inconsistent for {molecule}")
    reference_shots = reference["allocated_shots"].to_numpy(dtype=int)
    if int(reference_shots.sum()) != total or np.any(reference_shots < 1):
        raise RuntimeError(f"FC-IMA T=3000 allocation lacks complete coverage for {molecule}")
    if "continuous_ima_fraction" not in reference.columns:
        raise RuntimeError(f"FC-IMA allocation lacks continuous IMA fractions for {molecule}")
    fractions = reference["continuous_ima_fraction"].to_numpy(dtype=float)
    if np.any(~np.isfinite(fractions)) or np.any(fractions <= 0.0) or not math.isclose(
        float(fractions.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise RuntimeError(f"FC-IMA continuous allocation is invalid for {molecule}")
    replay_shots = fc_ima_largest_remainder(total, fractions)
    if not np.array_equal(reference_shots, replay_shots):
        raise RuntimeError(f"FC-IMA T=3000 integer allocation replay failed for {molecule}")

    gate_budget = None
    if plan["shots_to_target"] is not None:
        target_allocation = fc_ima_largest_remainder(
            int(plan["shots_to_target"]), fractions
        )
        gate_budget = int(
            np.dot(
                target_allocation,
                groups["compiled_cx_count"].to_numpy(dtype=int),
            )
        )

    max_depth = int(groups["compiled_two_qubit_depth"].max())

    return {
        "decomposition_items": group_count,
        "max_two_qubit_depth": max_depth,
        "depth_is_upper_bound": False,
        "shots_to_accuracy": plan["shots_to_target"],
        "unfloored_shots_to_accuracy": plan.get(
            "unfloored_shots_to_target", plan["shots_to_target"]
        ),
        "two_qubit_gate_budget": gate_budget,
        "gate_budget_is_upper_bound": False,
        "status": plan["status"],
        "reference_total_shots": total,
        "source": str(FC_ERROR_SUMMARY.resolve()),
        "analytic_sampling_se": sampling_se,
        "signed_bias": 0.0,
        "variance_coefficient": plan["variance_coefficient"],
        "fixed_estimator_scaling_exponent": FIXED_ESTIMATOR_SCALING_EXPONENT,
        "fixed_estimator_model": FIXED_ESTIMATOR_MODEL,
        "allocation": FC_ALLOCATION,
        "estimator": str(selected["estimator"]),
        "grouping": str(selected["grouping"]),
    }


def pauli_record(
    molecule: str,
    method: str,
    frame: pd.DataFrame,
    variance: pd.DataFrame,
    *,
    full_space_schema: bool,
) -> dict[str, Any]:
    shots_key = "T_total_shots"
    if full_space_schema:
        settings_key = "measurement_settings"
    else:
        settings_key = "measurement_settings_or_distinct_bases"
    reference = frame.loc[frame[shots_key].idxmax()]
    total = int(reference[shots_key])
    if total != REFERENCE_SHOTS:
        raise RuntimeError(
            f"Unexpected frozen-estimator budget for {molecule}/{method}: T={total}"
        )
    estimator_variance, variance_coefficient = fixed_pauli_variance(
        variance, molecule, method, total
    )
    plan = analytic_shots_to_target(total, math.sqrt(estimator_variance), 0.0)
    setting_count = int(reference[settings_key])
    if plan["shots_to_target"] is not None:
        plan["unfloored_shots_to_target"] = int(plan["shots_to_target"])
        plan["shots_to_target"] = max(int(plan["shots_to_target"]), setting_count)
    return {
        "decomposition_items": setting_count,
        "max_two_qubit_depth": 0,
        "depth_is_upper_bound": False,
        "shots_to_accuracy": plan["shots_to_target"],
        "unfloored_shots_to_accuracy": plan.get(
            "unfloored_shots_to_target", plan["shots_to_target"]
        ),
        "two_qubit_gate_budget": 0,
        "gate_budget_is_upper_bound": False,
        "status": plan["status"],
        "reference_total_shots": total,
        "source": str(STATE_VARIANCE.resolve()),
        "schedule_source": "figure-error summary",
        "analytic_sampling_se": math.sqrt(estimator_variance),
        "signed_bias": 0.0,
        "variance_coefficient": variance_coefficient,
        "fixed_estimator_scaling_exponent": FIXED_ESTIMATOR_SCALING_EXPONENT,
        "fixed_estimator_model": FIXED_ESTIMATOR_MODEL,
    }


def h4_scan_pauli_counts(recompute: bool) -> pd.DataFrame:
    cache = HERE / "h4_scan_pauli_setting_counts_T2038.csv"
    if cache.exists() and not recompute:
        return pd.read_csv(cache)
    try:
        from qiskit.quantum_info import Operator, SparsePauliOp
    except ImportError as exc:
        raise RuntimeError("Qiskit is required only to rebuild the H4 scan Pauli-count cache") from exc
    legacy = load_module(LEGACY_PAULI, "resource_statistics_legacy_pauli")
    pauli_code = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    rows = []
    for index in range(21):
        bond = 0.4 + 0.2 * index
        archive = H4_SCAN_HAMILTONIANS / f"H4_R{bond:.1f}.npz"
        hamiltonian = np.load(archive, allow_pickle=False)["H"]
        operator = SparsePauliOp.from_operator(Operator(hamiltonian), atol=1e-10, rtol=1e-10)
        observables = np.asarray(
            [[pauli_code[symbol] for symbol in label] for label in operator.paulis.to_labels()],
            dtype=np.int8,
        )
        weights = np.real_if_close(operator.coeffs).real.astype(float)
        nonidentity = np.any(observables != 0, axis=1)
        observables, weights = observables[nonidentity], weights[nonidentity]
        sg = legacy.shadow_grouping_schedule(observables, weights, 2038)
        derand = legacy.derandomized_schedule(observables, weights, 2038)
        _, probabilities, audit = legacy.optimize_ogm(observables, weights)
        if not audit["success"]:
            raise RuntimeError(f"OGM did not converge for H4 R={bond:.1f}")
        rows.append(
            {
                "bond_length_angstrom": f"{bond:.1f}",
                "OGM_groups": int(np.count_nonzero(probabilities > 1e-12)),
                "SG_groups": int(len(np.unique(sg, axis=0))),
                "Derand_unique_bases": int(len(np.unique(derand, axis=0))),
                "nonidentity_pauli_terms": int(len(weights)),
                "hamiltonian_sha256": sha256(archive),
            }
        )
    pd.DataFrame(rows).to_csv(cache, index=False)
    return pd.DataFrame(rows)


def h4_scan_records(
    recompute_counts: bool,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(H4_SCAN_SUMMARY)
    counts = h4_scan_pauli_counts(recompute_counts)
    rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []

    agpd_values: list[dict[str, Any]] = []
    srcdf_values: list[dict[str, Any]] = []
    for record in summary.itertuples(index=False):
        record_fields = record._asdict()
        bond = float(record.bond_length_angstrom)
        agpd_path = H4_SCAN_AGPD / f"H4_R{bond:.1f}".replace(".", "p") / "budget_T2038.json"
        validate_agpd_manifest(
            agpd_path.parent / "manifest.json",
            "v10_fully_corrective_dense_sector",
            f"H4 scan R={bond:.1f}",
        )
        validate_v10_fully_corrective_audit(
            agpd_path.parent / "audit.json", f"H4 scan R={bond:.1f}"
        )
        payload = json.loads(agpd_path.read_text(encoding="utf-8"))
        k = int(payload["best"]["K_source_terms"])
        metadata = agpd_shallow_setting_metadata(
            payload,
            expected_k=k,
            expected_total_shots=2038,
            context=f"H4 scan R={bond:.1f}",
            artifact_contract="v10_fully_corrective_dense_sector",
        )
        selected_sampling = payload["selected_sampling"]
        required_fixed_error_fields = {
            "analytic_sampling_SE",
            "signed_approximation_bias",
            "empirical_total_RMSE",
        }
        missing_fixed_error_fields = sorted(
            required_fixed_error_fields - set(selected_sampling)
        )
        if missing_fixed_error_fields:
            raise RuntimeError(
                f"H4 scan R={bond:.1f}: selected_sampling lacks fixed-error "
                f"fields {missing_fixed_error_fields}"
            )
        empirical_error = float(selected_sampling["empirical_total_RMSE"])
        analytic_sampling_se = float(selected_sampling["analytic_sampling_SE"])
        signed_bias = float(selected_sampling["signed_approximation_bias"])
        if (
            not math.isfinite(empirical_error)
            or not math.isfinite(analytic_sampling_se)
            or analytic_sampling_se < 0.0
            or not math.isfinite(signed_bias)
        ):
            raise RuntimeError(
                f"H4 scan R={bond:.1f}: invalid AGPD fixed-error inputs"
            )
        plan = analytic_shots_to_target(
            2038, analytic_sampling_se, signed_bias
        )
        unfloored_target_shots = plan["shots_to_target"]
        target_shots = None
        allocation = None
        if unfloored_target_shots is not None:
            target_shots = max(
                int(unfloored_target_shots), metadata["total_settings"]
            )
            allocation = complete_setting_largest_remainder(
                target_shots, metadata["reference_shots"]
            )
            if len(allocation) != metadata["total_settings"]:
                raise RuntimeError(
                    "H4 AGPD target allocation does not cover all L_A+K settings"
                )
        gate_summary = summarize_gate_resources(metadata["resources"], allocation)
        gate_budget = gate_summary["two_qubit_gate_budget"]
        max_depth = gate_summary["max_two_qubit_depth"]
        is_upper_bound = gate_summary["depth_is_upper_bound"]
        agpd_values.append(
            {
                "settings": metadata["total_settings"],
                "max_depth": max_depth,
                "shots_to_target": target_shots,
                "unfloored_shots_to_target": unfloored_target_shots,
                "gate_budget": gate_budget,
                "cnot_gate_budget": gate_summary["cnot_gate_budget"],
                "iswap_gate_budget": gate_summary["iswap_gate_budget"],
                "is_upper_bound": is_upper_bound,
                "status": plan["status"],
                "variance_coefficient": plan["variance_coefficient"],
                "signed_bias": signed_bias,
            }
        )
        geometry_rows.append(
            {
                "bond_length_angstrom": bond,
                "method": "AGPD",
                "decomposition_items": metadata["total_settings"],
                "source_leaves_K": k,
                "extra_one_body_leaves_L": metadata["cover_settings"],
                "shallow_one_body_cover_leaves_L_A": metadata["cover_settings"],
                "collector_mode": "fixed_shallow_one_body_cover",
                "shallow_one_body_cover_depth_d_A": metadata["cover_depth"],
                "source_to_collector_transfer_count": 0,
                "independent_collector_setting_present": False,
                "max_two_qubit_depth": max_depth,
                "cover_two_qubit_depth": metadata["cover_cx_depth"],
                "cover_two_qubit_gate_count_per_setting": metadata["cover_cx_count"],
                "empirical_RMSE_at_T2038": empirical_error,
                "analytic_sampling_SE_at_T2038": analytic_sampling_se,
                "signed_approximation_bias": signed_bias,
                "variance_coefficient": plan["variance_coefficient"],
                "fixed_estimator_scaling_exponent": FIXED_ESTIMATOR_SCALING_EXPONENT,
                "fixed_estimator_model": FIXED_ESTIMATOR_MODEL,
                "fixed_error_status": plan["status"],
                "status": plan["status"],
                "unfloored_shots_to_error_0p01": unfloored_target_shots,
                "shots_to_error_0p01": target_shots,
                "two_qubit_gate_budget_to_error_0p01": gate_budget,
                "cnot_gate_budget_to_error_0p01": gate_summary["cnot_gate_budget"],
                "iswap_gate_budget_to_error_0p01": gate_summary["iswap_gate_budget"],
                "gate_resource_convention": GATE_RESOURCE_CONVENTION,
                "depth_is_upper_bound": is_upper_bound,
                "gate_budget_is_upper_bound": is_upper_bound,
                "all_setting_shots_counted_in_gate_budget": True,
                "setting_order": "shallow_one_body_cover_then_complete_native_sources",
                "agpd_artifact_contract": metadata["artifact_contract"],
                "centered_range_domain": metadata["centered_range_domain"],
                "full_fock_centered_range_sum": metadata[
                    "full_fock_centered_range_sum"
                ],
                "sector_centered_range_sum": metadata["sector_centered_range_sum"],
                "sector_range_setting_count": metadata["sector_range_setting_count"],
                "full_fock_fallback_setting_count": metadata[
                    "full_fock_fallback_setting_count"
                ],
                "maximum_sector_leakage_relative_frobenius": metadata[
                    "maximum_sector_leakage_relative_frobenius"
                ],
                "selection_rule": metadata["selection_rule"],
                "sector_operator_norm": metadata["sector_operator_norm"],
                "fully_corrective_rcond": metadata["fully_corrective_rcond"],
                "fully_corrective_design_rank": metadata[
                    "fully_corrective_design_rank"
                ],
                "fully_corrective_design_nullity": metadata[
                    "fully_corrective_design_nullity"
                ],
                "selection_uses_exact_ground_state": False,
                "source": str(agpd_path.resolve()),
            }
        )

        selected_k = int(record.sRCDF_selected_K)
        selected_depth = int(record.sRCDF_selected_depth)
        candidate_path = H4_SCAN_SUMMARY.parent / f"R{bond:.1f}" / "candidate_by_T_K.csv"
        candidates = pd.read_csv(candidate_path)
        selected = candidates.loc[
            (candidates["T_total_shots"] == 2038)
            & (candidates["K_source_terms"] == selected_k)
            & (candidates["shallow_rotation_depth"] == selected_depth)
        ]
        summary_mode = record_fields.get("sRCDF_collector_mode")
        if summary_mode is not None and "collector_mode" in selected:
            selected = selected.loc[selected["collector_mode"] == summary_mode]
        summary_extra = record_fields.get(
            "sRCDF_collector_extra_settings",
            record_fields.get("sRCDF_collector_extra_leaves"),
        )
        extra_key = next(
            (
                name
                for name in ("collector_extra_settings", "collector_extra_leaves")
                if name in selected
            ),
            None,
        )
        if summary_extra is not None and extra_key is not None:
            selected = selected.loc[selected[extra_key] == int(summary_extra)]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one H4 scan SRDD candidate at R={bond:.1f}")
        selected = selected.iloc[0]
        collector = collector_metadata(selected, selected_k)
        validate_srcdf_audit(candidate_path, collector)
        empirical_error = float(record.sRCDF_empirical_RMSE)
        target_shots = max(
            collector["settings"],
            int(math.ceil(2038 * (empirical_error / EPSILON_TARGET) ** 2)),
        )
        reference_shots = parse_shot_vector(selected["shot_vector"])
        if (
            len(reference_shots) != collector["settings"]
            or sum(reference_shots) != 2038
            or any(value < 0 for value in reference_shots)
        ):
            raise RuntimeError(f"H4 scan shot conservation failed at R={bond:.1f}")
        allocation = largest_remainder(
            target_shots, reference_shots
        )
        count, cx_depth = srcdf_cx_resources(selected_depth, 4)
        if cx_depth != 4 * selected_depth or len(allocation) != collector["settings"]:
            raise RuntimeError(f"H4 scan shallow-depth audit failed at R={bond:.1f}")
        gate_budget = int(np.sum(allocation) * count)
        srcdf_values.append(
            {
                "settings": collector["settings"],
                "max_depth": cx_depth,
                "shots_to_target": target_shots,
                "gate_budget": gate_budget,
                "cnot_gate_budget": gate_budget,
                "iswap_gate_budget": 0,
                "is_upper_bound": False,
                "status": "single_point_shot_noise",
            }
        )
        geometry_rows.append(
            {
                "bond_length_angstrom": bond,
                "method": SRDD_METHOD,
                "decomposition_items": collector["settings"],
                "source_leaves_K": selected_k,
                "extra_one_body_leaves_L": collector["extra"],
                "collector_mode": collector["mode"],
                "independent_collector_setting_present": False,
                "max_two_qubit_depth": cx_depth,
                "empirical_RMSE_at_T2038": empirical_error,
                "shots_to_error_0p01": target_shots,
                "two_qubit_gate_budget_to_error_0p01": gate_budget,
                "depth_is_upper_bound": False,
                "gate_budget_is_upper_bound": False,
                "all_setting_shots_counted_in_gate_budget": True,
            }
        )

        count_row = counts.loc[
            np.isclose(counts["bond_length_angstrom"].astype(float), bond)
        ].iloc[0]
        for method, error_key, count_key in (
            ("OGM", "OGM", "OGM_groups"),
            ("SG", "SG", "SG_groups"),
            ("Derand", "Derand", "Derand_unique_bases"),
        ):
            empirical_error = float(getattr(record, error_key))
            target_shots = int(math.ceil(2038 * (empirical_error / EPSILON_TARGET) ** 2))
            geometry_rows.append(
                {
                    "bond_length_angstrom": bond,
                    "method": method,
                    "decomposition_items": int(count_row[count_key]),
                    "max_two_qubit_depth": 0,
                    "empirical_RMSE_at_T2038": empirical_error,
                    "shots_to_error_0p01": target_shots,
                    "two_qubit_gate_budget_to_error_0p01": 0,
                    "depth_is_upper_bound": False,
                    "gate_budget_is_upper_bound": False,
                }
            )

    def range_text(values: list[int]) -> str:
        return str(min(values)) if min(values) == max(values) else f"{min(values)}--{max(values)}"

    for method, values in (
        ("AGPD", agpd_values),
        (SRDD_METHOD, srcdf_values),
    ):
        upper = any(item["is_upper_bound"] for item in values)
        finite_values = [
            item
            for item in values
            if item["shots_to_target"] is not None and item["gate_budget"] is not None
        ]
        bias_floor_count = sum(item["status"] == "bias_floor" for item in values)
        exceeds_limit_count = sum(
            item["status"] == "exceeds_1e8" for item in values
        )
        if method == "AGPD":
            # A mean over only the reachable geometries would incorrectly make the
            # full scan appear to reach the target.  Preserve finite-only means as
            # diagnostics, while the aggregate itself is n.r. if any geometry is
            # blocked by the frozen approximation-bias floor.
            if bias_floor_count:
                aggregate_shots = None
                aggregate_gate_budget = None
                aggregate_status = "bias_floor"
            else:
                aggregate_shots = float(
                    np.mean([item["shots_to_target"] for item in finite_values])
                )
                aggregate_gate_budget = float(
                    np.mean([item["gate_budget"] for item in finite_values])
                )
                aggregate_status = (
                    "exceeds_1e8" if exceeds_limit_count else "reported"
                )
            aggregate_source = str(H4_SCAN_AGPD.resolve())
        else:
            aggregate_shots = float(
                np.mean([item["shots_to_target"] for item in finite_values])
            )
            aggregate_gate_budget = float(
                np.mean([item["gate_budget"] for item in finite_values])
            )
            aggregate_status = "single_point_shot_noise_mean"
            aggregate_source = str(H4_SCAN_SUMMARY.resolve())
        component_budgets = {
            name: (None if aggregate_gate_budget is None else
                   float(np.mean([item[name] for item in finite_values])))
            for name in ("cnot_gate_budget", "iswap_gate_budget")
        }
        rows.append(
            {
                "benchmark": "H4 scan, R=0.4--4.4 A",
                "method": method,
                "decomposition_items": range_text(
                    [item["settings"] for item in values]
                ),
                "decomposition_items_mean": float(
                    np.mean([item["settings"] for item in values])
                ),
                "max_two_qubit_depth": range_text(
                    [item["max_depth"] for item in values]
                ),
                "max_two_qubit_depth_mean": float(
                    np.mean([item["max_depth"] for item in values])
                ),
                "depth_is_upper_bound": upper,
                "shots_to_accuracy": aggregate_shots,
                "two_qubit_gate_budget": aggregate_gate_budget,
                **component_budgets,
                "gate_resource_convention": GATE_RESOURCE_CONVENTION,
                "gate_budget_is_upper_bound": upper,
                "status": aggregate_status,
                "geometry_count": len(values),
                "finite_target_geometry_count": len(finite_values),
                "bias_floor_geometry_count": bias_floor_count,
                "exceeds_reporting_limit_geometry_count": exceeds_limit_count,
                "finite_shots_to_accuracy_mean": (
                    float(np.mean([item["shots_to_target"] for item in finite_values]))
                    if finite_values
                    else None
                ),
                "finite_two_qubit_gate_budget_mean": (
                    float(np.mean([item["gate_budget"] for item in finite_values]))
                    if finite_values
                    else None
                ),
                **{
                    "finite_" + name + "_mean": (
                        float(np.mean([item[name] for item in finite_values]))
                        if finite_values else None
                    ) for name in ("cnot_gate_budget", "iswap_gate_budget")
                },
                "fixed_estimator_model": (
                    FIXED_ESTIMATOR_MODEL if method == "AGPD" else None
                ),
                "fixed_estimator_scaling_exponent": (
                    FIXED_ESTIMATOR_SCALING_EXPONENT if method == "AGPD" else None
                ),
                "reference_total_shots": 2038,
                "source": aggregate_source,
            }
        )
    for method, column in (
        ("OGM", "OGM_groups"),
        ("SG", "SG_groups"),
        ("Derand", "Derand_unique_bases"),
    ):
        values = counts[column].astype(int).tolist()
        method_geometry = [item for item in geometry_rows if item["method"] == method]
        rows.append(
            {
                "benchmark": "H4 scan, R=0.4--4.4 A",
                "method": method,
                "decomposition_items": range_text(values),
                "decomposition_items_mean": float(np.mean(values)),
                "max_two_qubit_depth": 0,
                "max_two_qubit_depth_mean": 0.0,
                "depth_is_upper_bound": False,
                "shots_to_accuracy": float(
                    np.mean([item["shots_to_error_0p01"] for item in method_geometry])
                ),
                "two_qubit_gate_budget": 0,
                "gate_budget_is_upper_bound": False,
                "status": "single_point_shot_noise_mean",
                "reference_total_shots": 2038,
                "source": str((HERE / "h4_scan_pauli_setting_counts_T2038.csv").resolve()),
            }
        )
    return rows, counts, pd.DataFrame(geometry_rows)


def format_integer(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n.r."
    if isinstance(value, str):
        return value
    return f"{int(round(float(value))):,}"


def format_accuracy_shots(row: dict[str, Any] | None) -> str:
    if row is None:
        return "n.a."
    if row["status"] == "bias_floor":
        return "n.r."
    if row["status"] == "exceeds_1e8":
        return r"$>10^8$"
    return format_integer(row["shots_to_accuracy"])


def publication_method_target_is_omitted(molecule: str, method: str) -> bool:
    return (molecule, method) in PUBLICATION_OMITTED_METHOD_TARGETS


def write_accuracy_table(rows: list[dict[str, Any]]) -> Path:
    path = HERE / "resource_accuracy_table.tex"
    by_key = {(row["benchmark"], row["method"]): row for row in rows}
    benchmarks = [
        ("H4, R=1.20 A", r"H$_4$, $R=1.20$~\AA"),
        ("H6, R=3.40 A", r"H$_6$, $R=3.40$~\AA"),
        ("BeH2, R=1.33376 A", r"BeH$_2$, $R=1.33376$~\AA"),
        ("N2, R=2.25 A", r"N$_2$, $R=2.25$~\AA"),
    ]
    methods = METHODS
    lines = [
        r"\begin{table*}[t]",
        r"\color{BLUE}",
        r"\captionsetup{labelfont={color=BLUE},textfont={color=BLUE}}",
        r"\centering",
        r"\caption{Measurement estimates $T_{0.01}$ for total energy error $0.01$~Ha. For every method, the decomposition or realized measurement design at $T_0=3000$ is frozen and the estimate follows $\epsilon_m^2(T)=b_m^2+V_m/T$, with $V_m=T_0\operatorname{Var}(\widehat E_{m,T_0})$. FC-IMA, OGM, SG, and Derand represent the Hamiltonian exactly and therefore use $b_m=0$; approximation bias is retained for AGPD and SRDD. Each value also obeys its method-specific executable-setting floor. The AGPD floor is its complete ordered $L_A+K$ setting set, comprising the minimum-feasible depth-$d_A$ shallow one-body cover followed by all native source settings. The SRDD floor is its complete $K+L$ local shallow-setting set. FC-IMA uses Nature-2023 overlapping full-commuting groups, pooled Pauli means, and iterative measurement allocation, and is evaluated only for the full-space BeH$_2$ and N$_2$ targets. ``n.r.'' means that the target is at or below the frozen approximation-bias floor; ``n.a.'' denotes a method outside the evaluated implementation.}",
        r"\label{tab:si_resource_accuracy}",
        r"\small",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\hline",
        r"Benchmark & AGPD & SRDD & FC-IMA & OGM & SG & Derand\\",
        r"\hline",
    ]
    for key, label in benchmarks:
        molecule = key.split(",", 1)[0]
        cells = [
            "n.a."
            if publication_method_target_is_omitted(molecule, method)
            else format_accuracy_shots(by_key.get((key, method)))
            for method in methods
        ]
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def chart_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    add_gate_type_fields(rows)
    by_key = {(row["benchmark"], row["method"]): row for row in rows}
    benchmarks = (
        ("H4, R=1.20 A", "H4"),
        ("H6, R=3.40 A", "H6"),
        ("BeH2, R=1.33376 A", "BeH2"),
        ("N2, R=2.25 A", "N2"),
    )
    output: list[dict[str, Any]] = []
    for benchmark, label in benchmarks:
        for method in METHODS:
            if publication_method_target_is_omitted(label, method):
                continue
            row = by_key.get((benchmark, method))
            if row is None:
                continue
            is_scan = benchmark.startswith("H4 scan")
            output.append(
                {
                    "benchmark": benchmark,
                    "plot_label": label,
                    "method": method,
                    "reference_total_shots": row["reference_total_shots"],
                    "decomposition_items": (
                        row["decomposition_items_mean"]
                        if is_scan
                        else float(row["decomposition_items"])
                    ),
                    "max_two_qubit_depth": (
                        row["max_two_qubit_depth_mean"]
                        if is_scan
                        else float(row["max_two_qubit_depth"])
                    ),
                    "depth_is_upper_bound": row["depth_is_upper_bound"],
                    "shots_to_error_0p01": row["shots_to_accuracy"],
                    "two_qubit_gate_budget_to_error_0p01": row["two_qubit_gate_budget"],
                    "cnot_gate_budget_to_error_0p01": row["cnot_gate_budget"],
                    "iswap_gate_budget_to_error_0p01": row["iswap_gate_budget"],
                    "gate_resource_convention": row["gate_resource_convention"],
                    "gate_budget_is_upper_bound": row["gate_budget_is_upper_bound"],
                    "status": row["status"],
                }
            )
    return output


def compact_value(value: float) -> str:
    if value < 10 and not float(value).is_integer():
        return f"{value:.1f}"
    if value < 1000:
        return f"{value:.0f}"
    if value < 1_000_000:
        return f"{value / 1000:.1f}k"
    return f"{value / 1_000_000:.1f}M"


def resource_method_slots(plot_label: str) -> tuple[str, ...]:
    """Return the displayed method slots for one molecular x-coordinate.

    FC-IMA was not evaluated for H4 or H6, so those two groups omit its slot
    entirely.  This keeps OGM, SG, and Derand adjacent to SRDD instead of
    leaving a visible hole.  The BeH2 and N2 groups retain the common five-slot
    ordering used for the full-space comparison.
    """
    if plot_label == "H4":
        return tuple(method for method in RESOURCE_FIGURE_METHODS if method != FC_METHOD)
    if plot_label == "H6":
        return tuple(method for method in RESOURCE_FIGURE_METHODS if method != FC_METHOD)
    return RESOURCE_FIGURE_METHODS


def draw_grouped_bars(
    ax: plt.Axes,
    summary: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    panel_label: str,
    *,
    annotate_values: bool = True,
    inside_panel_label: bool = False,
    annotation_fontsize: float = 6.6,
    tick_labelsize: float = 9.0,
    panel_fontsize: float = 10.0,
    outside_panel_offset: tuple[float, float] = (-54.0, 0.0),
) -> None:
    labels = ("H4", "H6", "BeH2", "N2")
    x = np.arange(len(labels), dtype=float)
    width = 0.13
    lookup = {(row["plot_label"], row["method"]): row for row in summary}
    maximum = 0.0
    for label_index, label in enumerate(labels):
        slot_methods = resource_method_slots(label)
        for method_index, method in enumerate(slot_methods):
            offset = (method_index - (len(slot_methods) - 1) / 2) * width
            row = lookup.get((label, method))
            if row is None or row[metric] is None:
                continue
            raw_value = float(row[metric])
            capped = metric == "shots_to_error_0p01" and row["status"] == "exceeds_1e8"
            value = float(REPORTING_LIMIT) if capped else raw_value
            maximum = max(maximum, value)
            bar = ax.bar(
                x[label_index] + offset,
                value,
                width=width * 0.92,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.45,
                zorder=3,
            )[0]
            if annotate_values:
                ax.annotate(
                    (">" if capped else "") + compact_value(value),
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=annotation_fontsize,
                    clip_on=False,
                )
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8, top=maximum * 7.5)
    ax.set_xticks(x, [r"H$_4$", r"H$_6$", r"BeH$_2$", r"N$_2$"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.55, alpha=0.55, zorder=0)
    ax.tick_params(direction="in", top=True, right=True, labelsize=tick_labelsize)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if inside_panel_label:
        ax.annotate(
            panel_label,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-34.0, 0.0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=panel_fontsize,
            clip_on=False,
            annotation_clip=False,
        )
    else:
        ax.annotate(
            panel_label,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=outside_panel_offset,
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=panel_fontsize,
            clip_on=False,
            annotation_clip=False,
        )


def draw_variance_bars(
    ax: plt.Axes,
    variance: pd.DataFrame,
    panel_label: str,
    *,
    inside_panel_label: bool = False,
) -> None:
    labels = ("H4", "H6", "BeH2", "N2")
    x = np.arange(len(labels), dtype=float)
    width = 0.13
    selected = variance.loc[
        variance["molecule"].isin(labels)
        & variance["method"].isin(RESOURCE_FIGURE_METHODS)
        & (variance["T_total_shots"] == 3000)
    ].copy()
    lookup = {
        (str(row.molecule), str(row.method)): float(row.estimator_variance_hartree2)
        for row in selected.itertuples(index=False)
        if pd.notna(row.estimator_variance_hartree2)
    }
    positive = [value for value in lookup.values() if value > 0.0]
    if not positive:
        raise RuntimeError("No positive T=3000 state-dependent variances were found")
    for label_index, label in enumerate(labels):
        slot_methods = resource_method_slots(label)
        for method_index, method in enumerate(slot_methods):
            value = lookup.get((label, method), np.nan)
            if not np.isfinite(value):
                continue
            offset = (method_index - (len(slot_methods) - 1) / 2) * width
            ax.bar(
                x[label_index] + offset,
                float(value),
                width=width * 0.92,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.45,
                zorder=3,
            )
    ax.set_yscale("log")
    ax.set_ylim(min(positive) / 2.5, max(positive) * 3.5)
    ax.set_xticks(x, [r"H$_4$", r"H$_6$", r"BeH$_2$", r"N$_2$"])
    ax.set_ylabel("Exact variance")
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.55, alpha=0.55, zorder=0)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if inside_panel_label:
        ax.annotate(
            panel_label,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-34.0, 0.0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=plt.rcParams["font.size"],
            clip_on=False,
            annotation_clip=False,
        )
    else:
        ax.annotate(
            panel_label,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-54.0, 0.0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=10,
            clip_on=False,
            annotation_clip=False,
        )


def draw_noise_bars(
    ax: plt.Axes,
    srcdf_noise: pd.DataFrame,
    fc_noise: pd.DataFrame,
    panel_label: str,
) -> None:
    """Draw the common p=0.003 measurement-circuit noise comparison."""
    labels = ("BeH2", "N2")
    x = np.arange(len(labels), dtype=float)
    width = 0.28
    lookup: dict[tuple[str, str], float] = {}
    for molecule in labels:
        srcdf = srcdf_noise.loc[
            (srcdf_noise["molecule"] == molecule)
            & np.isclose(srcdf_noise["p"].astype(float), 0.003)
            & (srcdf_noise["T_total_shots"].astype(int) == 3000)
        ]
        fc = fc_noise.loc[
            (fc_noise["molecule"] == molecule)
            & (fc_noise["method"] == FC_METHOD)
            & np.isclose(fc_noise["depolarizing_rate"].astype(float), 0.003)
            & (fc_noise["T_total_shots"].astype(int) == 3000)
        ]
        if len(srcdf) != 1 or len(fc) != 1:
            raise RuntimeError(f"Expected one p=0.003 noise row per method for {molecule}")
        lookup[(molecule, SRDD_METHOD)] = float(srcdf.iloc[0]["error_hartree"])
        lookup[(molecule, FC_METHOD)] = float(fc.iloc[0]["error_hartree"])
    positive = list(lookup.values())
    for method_index, method in enumerate((SRDD_METHOD, FC_METHOD)):
        values = [lookup[(molecule, method)] for molecule in labels]
        bars = ax.bar(
            x + (method_index - 0.5) * width,
            values,
            width=width * 0.92,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.0,
                clip_on=False,
            )
    ax.set_yscale("log")
    ax.set_ylim(min(positive) / 1.8, max(positive) * 2.8)
    ax.set_xticks(x, [r"BeH$_2$", r"N$_2$"])
    ax.set_ylabel(r"Error at $p=0.003$ (Ha)")
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.55, alpha=0.55, zorder=0)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.annotate(
        panel_label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-18.0, 12.0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=10,
        clip_on=False,
        annotation_clip=False,
    )


def draw_target_error_curves(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    panel_label: str,
) -> None:
    """Draw the multi-target SRDD--FC-IMA measurement comparison."""
    epsilon = np.geomspace(0.01, 0.5, 121)
    configurations = (
        ("BeH2, R=1.33376 A", r"BeH$_2$ SRDD", SRDD_METHOD, "-", "D"),
        ("BeH2, R=1.33376 A", r"BeH$_2$ FC-IMA", FC_METHOD, "-", "o"),
        ("N2, R=2.25 A", r"N$_2$ SRDD", SRDD_METHOD, "--", "D"),
        ("N2, R=2.25 A", r"N$_2$ FC-IMA", FC_METHOD, "--", "o"),
    )
    by_key = {(str(row["benchmark"]), str(row["method"])): row for row in rows}
    all_values: list[float] = []
    for benchmark, label, method, linestyle, marker in configurations:
        row = by_key.get((benchmark, method))
        if row is None:
            raise RuntimeError(f"Missing target-error resource row for {benchmark}, {method}")
        bias = float(row["signed_bias"])
        variance_coefficient = float(row["variance_coefficient"])
        denominator = epsilon**2 - bias**2
        if np.any(denominator <= 0.0):
            raise RuntimeError(f"Bias floor enters the plotted error range for {benchmark}, {method}")
        setting_floor = int(row["decomposition_items"])
        values = np.maximum(variance_coefficient / denominator, setting_floor)
        expected_at_001 = float(row["shots_to_accuracy"])
        if abs(math.ceil(values[0]) - expected_at_001) > 0.0:
            raise RuntimeError(f"T_0.01 curve endpoint mismatch for {benchmark}, {method}")
        all_values.extend(values.tolist())
        ax.plot(
            epsilon,
            values,
            color=METHOD_COLORS[method],
            linestyle=linestyle,
            linewidth=1.25,
            marker=marker,
            markevery=20,
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=label,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(epsilon[0]), float(epsilon[-1]))
    ax.set_ylim(min(all_values) / 1.7, max(all_values) * 1.7)
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 0.5])
    ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "0.5"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Target error (Ha)")
    ax.set_ylabel("Required samples")
    ax.grid(which="major", linestyle=":", linewidth=0.55, alpha=0.55, zorder=0)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.legend(ncol=2, loc="upper right", frameon=False, fontsize=6.5, columnspacing=0.8)
    ax.annotate(
        panel_label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-18.0, 0.0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=10,
        clip_on=False,
        annotation_clip=False,
    )


def save_resource_plots(
    summary: list[dict[str, Any]],
    variance: pd.DataFrame,
    srcdf_noise: pd.DataFrame,
    fc_noise: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> list[Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    outputs: list[Path] = []
    specifications = (
        (
            "decomposition_items",
            r"Distinct measurement settings",
            "molecular_resource_decomposition_counts",
            "(b)",
        ),
        (
            "shots_to_error_0p01",
            "Required samples",
            "molecular_resource_required_measurements",
            "(c)",
        ),
    )
    # These two standalone panels are stacked at 0.96\linewidth in the response
    # letter.  The 13 pt source labels are therefore rendered at approximately
    # 12 pt, matching the surrounding response text without crowding the bar
    # annotations or legend.
    response_panel_style = {
        "font.size": 13.0,
        "axes.labelsize": 13.0,
        "legend.fontsize": 12.0,
    }
    for metric, ylabel, stem, panel in specifications:
        with plt.rc_context(response_panel_style):
            fig, ax = plt.subplots(figsize=(7.1, 3.35))
            draw_grouped_bars(
                ax,
                summary,
                metric,
                ylabel,
                panel,
                annotation_fontsize=12.0,
                tick_labelsize=12.5,
                panel_fontsize=13.0,
                outside_panel_offset=(-54.0, 0.0),
            )
            handles = [
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=METHOD_COLORS[m],
                    edgecolor="black",
                    linewidth=0.45,
                )
                for m in RESOURCE_FIGURE_METHODS
            ]
            ax.legend(
                handles,
                RESOURCE_FIGURE_METHODS,
                ncol=5,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.17),
                frameon=False,
                columnspacing=1.0,
                handletextpad=0.45,
                handlelength=1.45,
            )
            fig.tight_layout(pad=0.55)
            for suffix in ("pdf", "png"):
                path = PLOTS_DIR / f"{stem}.{suffix}"
                fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
                outputs.append(path)
            plt.close(fig)

    # Standalone response-letter copy of the current Fig. 4(a) method matrix.
    # Use the same draw routine so display omissions and style cannot drift from
    # the composite main-text figure.
    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    draw_variance_bars(ax, variance, "(a)")
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=METHOD_COLORS[m],
            edgecolor="black",
            linewidth=0.45,
        )
        for m in RESOURCE_FIGURE_METHODS
    ]
    ax.legend(
        handles,
        RESOURCE_FIGURE_METHODS,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        frameon=False,
    )
    fig.tight_layout(pad=0.55)
    for suffix in ("pdf", "png"):
        path = PLOTS_DIR / f"state_dependent_variance_T3000.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        outputs.append(path)
    plt.close(fig)

    fig, composite_axes = plt.subplots(1, 3, figsize=(7.25, 2.72))
    draw_variance_bars(
        composite_axes[0], variance, "(a)", inside_panel_label=True
    )
    for ax, (metric, ylabel, _stem, panel) in zip(
        composite_axes[1:3],
        (
            (*specifications[0][:3], "(b)"),
            (*specifications[1][:3], "(c)"),
        ),
    ):
        draw_grouped_bars(
            ax,
            summary,
            metric,
            ylabel,
            panel,
            annotate_values=False,
            inside_panel_label=True,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[m], edgecolor="black", linewidth=0.45)
        for m in RESOURCE_FIGURE_METHODS
    ]
    fig.legend(
        handles,
        RESOURCE_FIGURE_METHODS,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
    )
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.19, top=0.79, wspace=0.62)
    for suffix in ("pdf", "png"):
        path = PLOTS_DIR / f"molecular_resource_summary.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_depth_lists(rows: list[dict[str, Any]]) -> Path:
    path = HERE / "resource_depth_lists.tex"
    by_key = {(row["benchmark"], row["method"]): row for row in rows}
    agpd_h4 = by_key[("H4, R=1.20 A", "AGPD")]
    srcdf_h4 = by_key[("H4, R=1.20 A", SRDD_METHOD)]
    srcdf_h6 = by_key[("H6, R=3.40 A", SRDD_METHOD)]
    srcdf_beh2 = by_key[("BeH2, R=1.33376 A", SRDD_METHOD)]
    srcdf_n2 = by_key[("N2, R=2.25 A", SRDD_METHOD)]
    lines = [
        r"\begin{itemize}",
        r"\setlength{\itemsep}{1pt}",
        r"\setlength{\parskip}{0pt}",
        (
            r"\item \emph{AGPD}: H$_4$ at $R=1.20$~\AA, $d_{2q}^{\max}\leq"
            + format_integer(agpd_h4["max_two_qubit_depth"])
            + r"$. The maximum includes every native source setting and every "
            + r"depth-bounded one-body-cover leaf."
        ),
        (
            r"\item \emph{SRDD}: H$_4$ at $R=1.20$~\AA, $d_{2q}^{\max}="
            + format_integer(srcdf_h4["max_two_qubit_depth"])
            + r"$; H$_6$, $d_{2q}^{\max}="
            + format_integer(srcdf_h6["max_two_qubit_depth"])
            + r"$; BeH$_2$, $d_{2q}^{\max}="
            + format_integer(srcdf_beh2["max_two_qubit_depth"])
            + r"$; N$_2$, $d_{2q}^{\max}="
            + format_integer(srcdf_n2["max_two_qubit_depth"])
            + r"$. All values include every redistributed shallow setting and obey "
            + r"$d_{2q}^{\max}=4d_R$."
        ),
        r"\end{itemize}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_fc_srcdf_depth_table(rows: list[dict[str, Any]]) -> Path:
    """Write the T=3000 BeH2/N2 SRDD--FC-IMA CNOT-depth comparison."""
    path = HERE / "fc_srcdf_cnot_depth_table.tex"
    by_key = {(row["benchmark"], row["method"]): row for row in rows}
    specifications = (
        ("BeH2", "BeH2, R=1.33376 A", BEH2, r"BeH$_2$"),
        ("N2", "N2, R=2.25 A", N2, r"N$_2$"),
    )
    table_values: dict[str, dict[str, int]] = {}
    for molecule, benchmark, summary_path, label in specifications:
        summary = pd.read_csv(summary_path)
        selected = summary.loc[
            (summary["molecule"] == molecule)
            & (summary["method"] == LEGACY_SRCDF_METHOD)
            & (summary["T_total_shots"] == 3000)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one T=3000 SRDD summary for {molecule}")
        selected = selected.iloc[0]
        srcdf = by_key[(benchmark, SRDD_METHOD)]
        fc = by_key[(benchmark, FC_METHOD)]
        k = int(selected["selected_K"])
        depth = int(selected["selected_depth"])
        extra = int(selected["collector_extra_leaves"])
        settings = int(selected["measurement_settings"])
        if (
            str(selected["collector_mode"]) != "augment"
            or extra < 1
            or settings != k + extra
            or int(srcdf["decomposition_items"]) != settings
            or int(srcdf["max_two_qubit_depth"]) != 4 * depth
        ):
            raise RuntimeError(f"SRDD resource mismatch for {molecule}")
        table_values[molecule] = {
            "srcdf_settings": settings,
            "fc_settings": int(fc["decomposition_items"]),
            "srcdf_depth": int(srcdf["max_two_qubit_depth"]),
            "fc_depth": int(fc["max_two_qubit_depth"]),
        }
    beh2 = table_values["BeH2"]
    n2 = table_values["N2"]
    lines = [
        r"\begingroup",
        r"\color{BLUE}",
        r"\captionsetup{labelfont={color=BLUE},textfont={color=BLUE}}",
        r"\noindent\begin{minipage}{\columnwidth}",
        r"\centering",
        r"\captionof{table}{Distinct measurement settings and maximum compiled CNOT depths for full-space BeH$_2$ and N$_2$; state preparation, one-qubit gates, routing, and error correction are excluded.}",
        r"\label{tab:fc_srcdf_cnot_depth}",
        r"\normalsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\hline",
        r"Quantity & Method & BeH$_2$ & N$_2$\\",
        r"\hline",
        (
            r"\multirow{2}{*}{Distinct settings} & SRDD & "
            + r"\textbf{" + str(beh2["srcdf_settings"]) + "}"
            + " & "
            + r"\textbf{" + str(n2["srcdf_settings"]) + "}"
            + r"\\"
        ),
        (
            r"& FC-IMA~\cite{yen2023deterministic} & "
            + str(beh2["fc_settings"])
            + " & "
            + str(n2["fc_settings"])
            + r"\\"
        ),
        r"\cline{1-4}",
        (
            r"\multirow{2}{*}{CNOT depth, $d_{\rm CNOT}^{\max}$} & SRDD & "
            + r"\textbf{" + str(beh2["srcdf_depth"]) + "}"
            + " & "
            + r"\textbf{" + str(n2["srcdf_depth"]) + "}"
            + r"\\"
        ),
        (
            r"& FC-IMA~\cite{yen2023deterministic} & "
            + str(beh2["fc_depth"])
            + " & "
            + str(n2["fc_depth"])
            + r"\\"
        ),
        r"\hline",
        r"\end{tabular}",
        r"\end{minipage}",
        r"\endgroup",
    ]
    payload = "\n".join(lines) + "\n"
    path.write_text(payload, encoding="utf-8")
    # Retain the former filename only as a compatibility alias.  The table
    # content, caption, label, and method name are all the current FC-IMA version.
    (HERE / "cm_srcdf_depth_table.tex").write_text(payload, encoding="utf-8")
    return path


def add_gate_type_fields(rows: list[dict[str, Any]], suffix: str = "") -> None:
    """Label CNOT-only rows; reject legacy GPD totals without gate types."""
    for row in rows:
        if row["method"] in {"AGPD", "GPD"}:
            if row.get("gate_resource_convention") != GATE_RESOURCE_CONVENTION:
                raise ValueError("GPD gate resources must be rebuilt with native iSWAP accounting")
            for gate in ("cnot", "iswap"):
                if gate + "_gate_budget" + suffix not in row:
                    raise ValueError("GPD gate resources lack separate CNOT/iSWAP budgets")
        else:
            total = row["two_qubit_gate_budget" + suffix]
            row["cnot_gate_budget" + suffix] = total
            row["iswap_gate_budget" + suffix] = 0 if total is not None else None
            row["gate_resource_convention"] = GATE_RESOURCE_CONVENTION


def write_gate_budget_table(rows: list[dict[str, Any]]) -> Path:
    add_gate_type_fields(rows)
    path = HERE / "resource_gate_budget_table.tex"
    by_key = {(row["benchmark"], row["method"]): row for row in rows}
    benchmarks = [
        ("H4, R=1.20 A", r"H$_4$, $R=1.20$~\AA"),
        ("H6, R=3.40 A", r"H$_6$, $R=3.40$~\AA"),
        ("BeH2, R=1.33376 A", r"BeH$_2$, $R=1.33376$~\AA"),
        ("N2, R=2.25 A", r"N$_2$, $R=2.25$~\AA"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\color{BLUE}",
        r"\captionsetup{labelfont={color=BLUE},textfont={color=BLUE}}",
        r"\centering",
        r"\caption{Executed two-qubit gate counts at total energy error $0.01$~Ha for the executable designs fixed at $T_0=3000$. Counts are CNOTs unless separate CNOT and native iSWAP contributions are shown. iSWAP is not decomposed into CNOTs. Every AGPD shot over its complete ordered $L_A+K$ setting vector is counted: each depth-$d_A$ cover leaf has CNOT count $4d_A(m-1)$ and depth $4d_A$, and each retained source uses its family-specific native gate resources. Every SRDD shot over its $K+L$ local depth-$d_R$ shallow settings is also counted. AGPD values are conservative upper bounds whenever the selected decomposition contains compiler-dependent blocks. FC-IMA uses its compiled CNOT counts and iterative allocation and is evaluated only for BeH$_2$ and N$_2$. Pauli-product measurement circuits use no two-qubit gates. ``n.r.'' means that the target is at or below the frozen approximation-bias floor; ``n.a.'' denotes a method outside the evaluated implementation.}",
        r"\label{tab:si_resource_gate_budget}",
        r"\small",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\hline",
        r"Benchmark & AGPD & SRDD & FC-IMA & OGM & SG & Derand\\",
        r"\hline",
    ]
    for key, label in benchmarks:
        molecule = key.split(",", 1)[0]
        cells = []
        for method in METHODS:
            if publication_method_target_is_omitted(molecule, method):
                cells.append("n.a.")
                continue
            row = by_key.get((key, method))
            if row is None:
                cells.append("n.a.")
                continue
            value = format_integer(row["cnot_gate_budget"])
            if row["iswap_gate_budget"]:
                value += r" CNOT + " + format_integer(row["iswap_gate_budget"]) + r" iSWAP"
            if row["gate_budget_is_upper_bound"] and value != "0":
                value = r"$\leq$" + value
            cells.append(value)
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute-h4-scan-counts",
        action="store_true",
        help="Rebuild the 21-geometry Pauli setting-count cache with Qiskit",
    )
    parser.add_argument(
        "--statistics-only",
        action="store_true",
        help=(
            "Bootstrap resource_statistics.csv and TeX tables before the new "
            "Fig.3 noise table exists; skip all resource plots."
        ),
    )
    args = parser.parse_args()

    required = [
        COMMON_V8,
        COMMON_V8.parent / "manifest.json",
        H4_COMMON_V10,
        H4_COMMON_V10.parent / "manifest.json",
        H4_PAULI_FIXED_SUMMARY,
        H4_PAULI_RERUN_MANIFEST,
        SRCDF_SUMMARY,
        BEH2,
        BEH2_CANDIDATES,
        N2,
        N2_CANDIDATES,
        STATE_VARIANCE,
        FC_ERROR_SUMMARY,
        *FC_CIRCUIT_AUDITS.values(),
        *FC_ALLOCATIONS.values(),
        H4_SCAN_SUMMARY,
        AGPD_ROOT_V10 / "manifest.json",
        AGPD_ROOT_V10 / "H4" / "budget_T3000.json",
        AGPD_ROOT_V10 / "H4" / "manifest.json",
        AGPD_ROOT_V10 / "H4" / "audit.json",
        *(SRCDF_ROOT / molecule / "candidate_by_T_K.csv" for molecule in ("H4", "H6")),
        *(SRCDF_ROOT / molecule / "audit.json" for molecule in ("H4", "H6")),
        BEH2.parent / "BeH2" / "audit.json",
        N2.parent / "audit.json",
        H4_SCAN_AGPD_MANIFEST,
        *(
            H4_SCAN_AGPD / f"H4_R{0.4 + 0.2 * index:.1f}".replace(".", "p") / "budget_T2038.json"
            for index in range(21)
        ),
        *(
            H4_SCAN_AGPD / f"H4_R{0.4 + 0.2 * index:.1f}".replace(".", "p") / "manifest.json"
            for index in range(21)
        ),
        *(
            H4_SCAN_AGPD / f"H4_R{0.4 + 0.2 * index:.1f}".replace(".", "p") / "audit.json"
            for index in range(21)
        ),
        *(
            H4_SCAN_SUMMARY.parent / f"R{0.4 + 0.2 * index:.1f}" / "candidate_by_T_K.csv"
            for index in range(21)
        ),
        *(
            H4_SCAN_SUMMARY.parent / f"R{0.4 + 0.2 * index:.1f}" / "audit.json"
            for index in range(21)
        ),
        *(
            H4_SCAN_HAMILTONIANS / f"H4_R{0.4 + 0.2 * index:.1f}.npz"
            for index in range(21)
        ),
    ]
    if not args.statistics_only:
        required.extend((FC_NOISE, SRCDF_NOISE))
    if args.recompute_h4_scan_counts:
        required.append(LEGACY_PAULI)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required resource inputs:\n" + "\n".join(missing))

    validate_agpd_manifest(
        AGPD_ROOT_V10 / "manifest.json",
        "v10_fully_corrective_dense_sector",
        "H4 fixed root",
    )
    validate_v10_fully_corrective_audit(
        AGPD_ROOT_V10 / "H4" / "audit.json", "H4 fixed geometry"
    )
    validate_h4_scan_agpd_manifest(H4_SCAN_AGPD_MANIFEST)
    common = load_fixed_comparison_overlay()
    state_variance = pd.read_csv(STATE_VARIANCE)
    standalone_srcdf = pd.read_csv(SRCDF_SUMMARY)
    rows, h4_counts, h4_geometry_resources = h4_scan_records(args.recompute_h4_scan_counts)
    diagnostics = []

    for molecule, benchmark in (
        ("H4", "H4, R=1.20 A"),
        ("H6", "H6, R=3.40 A"),
    ):
        selected = common.loc[common["molecule"] == molecule]
        tmax = int(selected["T_total_shots"].max())
        if tmax != 3000:
            raise RuntimeError(f"Unexpected maximum plotted budget for {molecule}: T={tmax}")
        srcdf_rows = standalone_srcdf.loc[
            (standalone_srcdf["molecule"] == molecule)
            & (standalone_srcdf["T_total_shots"] == tmax)
        ]
        if len(srcdf_rows) != 1:
            raise RuntimeError(
                f"Expected one redistributed SRDD summary for {molecule}, T={tmax}"
            )
        srcdf_summary = srcdf_rows.iloc[0]
        if molecule == "H4":
            agpd_rows = selected.loc[
                (selected["method"] == "four-family Hybrid-F")
                & (selected["T_total_shots"] == tmax)
            ]
            if len(agpd_rows) != 1:
                raise RuntimeError("Expected one H4 AGPD v10 row at T=3000")
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": "AGPD",
                    **agpd_record(molecule, agpd_rows.iloc[0]),
                }
            )
        rows.append(
            {
                "benchmark": benchmark,
                "method": SRDD_METHOD,
                **srcdf_record(molecule, srcdf_summary, spatial_orbitals=4, full_space_schema=False),
            }
        )
        for method in ("OGM", "SG", "Derand"):
            method_frame = selected.loc[selected["method"] == method]
            record = pauli_record(
                molecule,
                method,
                method_frame,
                state_variance,
                full_space_schema=False,
            )
            rows.append({"benchmark": benchmark, "method": method, **record})
            diagnostics.append({"benchmark": benchmark, "method": method, **record})

    for molecule, benchmark, source, spatial_orbitals in (
        ("BeH2", "BeH2, R=1.33376 A", BEH2, 7),
        ("N2", "N2, R=2.25 A", N2, 10),
    ):
        frame = pd.read_csv(source)
        tmax = int(frame["T_total_shots"].max())
        if tmax != 3000:
            raise RuntimeError(f"Unexpected maximum plotted budget for {molecule}: T={tmax}")
        srcdf_summary = frame.loc[
            (frame["method"] == LEGACY_SRCDF_METHOD) & (frame["T_total_shots"] == tmax)
        ].iloc[0]
        rows.append(
            {
                "benchmark": benchmark,
                "method": SRDD_METHOD,
                **srcdf_record(
                    molecule,
                    srcdf_summary,
                    spatial_orbitals=spatial_orbitals,
                    full_space_schema=True,
                ),
            }
        )
        rows.append({"benchmark": benchmark, "method": FC_METHOD, **fc_record(molecule)})
        for method in ("OGM", "SG", "Derand"):
            method_frame = frame.loc[frame["method"] == method]
            record = pauli_record(
                molecule,
                method,
                method_frame,
                state_variance,
                full_space_schema=True,
            )
            rows.append({"benchmark": benchmark, "method": method, **record})
            diagnostics.append({"benchmark": benchmark, "method": method, **record})

    validate_fixed_estimator_rows(rows, state_variance)

    add_gate_type_fields(rows)
    geometry_records = h4_geometry_resources.to_dict("records")
    add_gate_type_fields(geometry_records, suffix="_to_error_0p01")
    h4_geometry_resources = pd.DataFrame(geometry_records)

    columns = sorted({key for row in rows for key in row})
    with (HERE / "resource_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(diagnostics).to_csv(
        HERE / "pauli_fixed_estimator_diagnostics.csv", index=False
    )
    h4_geometry_resources.to_csv(HERE / "h4_scan_geometry_resources.csv", index=False)
    summary = chart_summary(rows)
    pd.DataFrame(summary).to_csv(HERE / "resource_chart_summary.csv", index=False)
    accuracy_latex_path = write_accuracy_table(rows)
    gate_budget_latex_path = write_gate_budget_table(rows)
    depth_lists_path = write_depth_lists(rows)
    fc_depth_table_path = write_fc_srcdf_depth_table(rows)
    fc_depth_legacy_path = HERE / "cm_srcdf_depth_table.tex"
    variance = state_variance.copy()
    fc_summary = pd.read_csv(FC_ERROR_SUMMARY)
    fc_variance = pd.DataFrame(
        {
            "molecule": fc_summary["molecule"],
            "method": FC_METHOD,
            "T_total_shots": 3000,
            "estimator_variance_hartree2": fc_summary[
                "inverse_error_frozen_T0_variance_hartree2"
            ],
        }
    )
    if set(fc_variance["molecule"]) != {"BeH2", "N2"} or len(fc_variance) != 2:
        raise RuntimeError("FC-IMA variance summary must contain exactly BeH2 and N2")
    variance = pd.concat([variance, fc_variance], ignore_index=True, sort=False)
    if args.statistics_only:
        plot_paths = []
    else:
        srcdf_noise = pd.read_csv(SRCDF_NOISE)
        validate_srcdf_noise(srcdf_noise)
        fc_noise = pd.read_csv(FC_NOISE)
        plot_paths = save_resource_plots(summary, variance, srcdf_noise, fc_noise, rows)

    generated = [
        HERE / "resource_statistics.csv",
        HERE / "pauli_fixed_estimator_diagnostics.csv",
        HERE / "h4_scan_pauli_setting_counts_T2038.csv",
        HERE / "h4_scan_geometry_resources.csv",
        HERE / "resource_chart_summary.csv",
        accuracy_latex_path,
        gate_budget_latex_path,
        depth_lists_path,
        fc_depth_table_path,
        fc_depth_legacy_path,
        *plot_paths,
    ]
    manifest = {
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "target_total_error_hartree": EPSILON_TARGET,
        "fixed_estimator_reference_shots": REFERENCE_SHOTS,
        "fixed_estimator_model": "RMSE(T)^2=b^2+V/T; V=T0*Var(T0)",
        "reporting_limit_measurements": REPORTING_LIMIT,
        "statistics_only_bootstrap": bool(args.statistics_only),
        "srcdf_collector_policy": (
            "exactly redistributed over K+L depth-d_R settings; "
            "independent collector forbidden"
        ),
        "srcdf_gate_budget_counts_all_setting_shots": True,
        "srcdf_maximum_logical_cx_depth_formula": "4*d_R",
        "srcdf_deterministic_constant_is_classical_offset": True,
        "agpd_artifact_contracts_by_scope": {
            "H4_fixed_R1p20": {
                "contract": "v10_fully_corrective_dense_sector",
                "runner_version": AGPD_V10_RUNNER_VERSION,
                "selection_rule": AGPD_V10_SELECTION_RULE,
                "fully_corrective_rcond": AGPD_V10_RCOND,
                "selection_uses_exact_ground_state": False,
                "root": str(AGPD_ROOT_V10.resolve()),
                "comparison": str(H4_COMMON_V10.resolve()),
            },
            "H4_21_bond_scan_T2038": {
                "contract": "v10_fully_corrective_dense_sector",
                "runner_version": AGPD_V10_RUNNER_VERSION,
                "selection_rule": AGPD_V10_SELECTION_RULE,
                "fully_corrective_rcond": AGPD_V10_RCOND,
                "selection_uses_exact_ground_state": False,
                "root": str(H4_SCAN_AGPD.resolve()),
                "producer_manifest": str(H4_SCAN_AGPD_MANIFEST.resolve()),
                "producer_manifest_sha256": sha256(H4_SCAN_AGPD_MANIFEST),
            },
        },
        "fixed_comparison_overlay_policy": (
            "H4 rows come only from the audited fully-corrective dense-sector "
            "v10 comparison; H6 contributes only SRDD and Pauli-product "
            "baselines, with no publication-facing AGPD row"
        ),
        "publication_omitted_method_targets": [
            {
                "molecule": molecule,
                "method": method,
                "reason": (
                    "fully-corrective dense-sector AGPD v10 has not been run "
                    "for H6, so the pair is excluded from every publication "
                    "CSV, table, and figure panel"
                ),
            }
            for molecule, method in PUBLICATION_OMITTED_METHOD_TARGETS
        ],
        "agpd_one_body_cover_policy": (
            "minimum-feasible depth-d_A shallow one-body cover first, followed "
            "by all K native source settings; total settings L_A+K"
        ),
        "agpd_setting_order": "L_A_cover_leaves_then_K_complete_native_sources",
        "agpd_source_to_cover_transfer_count": 0,
        "agpd_independent_collector_setting_present": False,
        "agpd_gate_budget_counts_all_setting_shots": True,
        "agpd_shot_rescaling": (
            "largest remainder after reserving one shot for every L_A+K setting"
        ),
        "agpd_cover_logical_cx_count_formula": "4*d_A*(m-1)",
        "agpd_cover_maximum_logical_cx_depth_formula": "4*d_A",
        "gate_resource_convention": GATE_RESOURCE_CONVENTION,
        "agpd_native_source_resources": "native_iSWAP_and_family_specific_CNOT_rules",
        "iswap_is_native": True,
        "gate_budgets_reported_separately": ["cnot", "iswap"],
        "two_qubit_gate_budget_definition": "cnot_gate_budget + iswap_gate_budget",
        "agpd_strict_metadata_validation": {
            "H4_fixed_v10_fully_corrective_dense_sector": True,
            "H4_21_bond_v10_fully_corrective_dense_sector": True,
        },
        "agpd_v10_fully_corrective_dense_sector_policy": (
            "all selected rotations are jointly refit over their 37 density-density "
            "coefficients by rcond=1e-12 dense-sector SVD; finite-shot prefix selection "
            "uses hypot(sector operator norm, range sampling proxy) and never the exact "
            "ground state"
        ),
        "agpd_v10_fully_corrective_dense_sector_validation": (
            "producer and case manifests, case FC audit, selected sector operator norm, "
            "selection rule, SVD rcond/rank/nullity, held-out-ground-state flag, and "
            "selected-sampling audit replay"
        ),
        "agpd_sector_range_policy": (
            "each setting uses a fixed-particle-number-sector spectral range only "
            "when leakage is at most 1e-10; otherwise it uses the full-Fock range"
        ),
        "agpd_sector_range_validation": (
            "setting-resolved ranges, leakage, eligibility, fallback domains, aggregate "
            "sums/counts, reduction fraction, cover audit, allocation replay, and proxy replay"
        ),
        "fc_group_unitary_is_yen_2020_full_commuting_measurement": True,
        "fc_estimator_and_allocation": (
            "Nature-2023 overlapping FC-IMA, Eq. (14) pooled estimator and "
            "Eqs. (19)-(22) iterative measurement allocation"
        ),
        "legacy_fc_artifact_root": str(FC_LEGACY_ROOT.resolve()),
        "noise_comparison_rate": 0.003,
        "inputs": {str(path.resolve()): sha256(path) for path in required},
        "outputs": {str(path.resolve()): sha256(path) for path in generated},
        "h4_scan_count_rows": int(len(h4_counts)),
        "h4_scan_resource_rows": int(len(h4_geometry_resources)),
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(rows)} resource rows and {len(diagnostics)} "
        f"Pauli fixed-estimator audits to {HERE}"
    )


if __name__ == "__main__":
    main()
