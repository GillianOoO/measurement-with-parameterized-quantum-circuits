#!/usr/bin/env python3
"""Build exact state-conditioned variance tables for the main molecular targets.

The Pauli baselines use the *fixed-schedule pooled-hit estimator* that produced
the manuscript curves.  Its variance is conditioned on the archived or exactly
regenerated list of local Pauli bases.  For reference, the script also evaluates
the OGM-paper iid variance obtained by reinterpreting those basis frequencies as
an iid distribution; the two numbers differ by a nonnegative between-basis term.

AGPD and SRDD use the exact post-selection, fixed-stratum variance
``sum_i Var_rho(Y_i) / T_i``.  Every active-space AGPD row is cross-checked
against its selected-sampling artifact so that all K complete native sources
and all L_A shallow one-body-cover settings are present in the saved shot vector
and in the reported variance.  H4 uses the v10 fully-corrective dense-sector
fit with leakage-certified particle-sector range allocation; H6 and LiH retain
their audited v8 artifacts.  BeH2 and N2
have no full-space AGPD artifact and are therefore emitted as explicitly
unavailable rather than substituted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable


for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADAPTIVE = ROOT / "outputs" / "adaptive_f3_ansatz_pool"
PAULI_ROOT = ADAPTIVE / "five_molecule_pauli_sampling_comparison_v4_equal_stratum"
H4_PAULI_RERUN_ROOT = (
    ROOT
    / "outputs"
    / "iswap_gpd_main_loss_v1"
    / "h4_pauli_same_input_rerun"
)
H4_PAULI_FIXED_ROOT = H4_PAULI_RERUN_ROOT / "fixed" / "fixed_R1p2"
H4_PAULI_FIXED_SUMMARY = H4_PAULI_RERUN_ROOT / "fixed_sampling_error_summary.csv"
H4_PAULI_RERUN_MANIFEST = H4_PAULI_RERUN_ROOT / "manifest.json"
COMBINED_V8 = (
    ADAPTIVE
    / "standalone_shallow_rcdf_vs_four_family_agpd_v8_shallow_collectors"
    / "sampling_error_comparison.csv"
)
COMBINED_V8_MANIFEST = COMBINED_V8.parent / "manifest.json"
COMBINED_H4_V10 = (
    ADAPTIVE
    / "fully_corrective_dense_sector_agpd_vs_srcdf_and_pauli_v10_h4"
    / "sampling_error_comparison.csv"
)
COMBINED_H4_V10_MANIFEST = COMBINED_H4_V10.parent / "manifest.json"
AGPD_ROOT_V8 = ADAPTIVE / "five_molecule_v8_shallow_agpd_collector"
AGPD_ROOT_H4_V10 = ADAPTIVE / "five_molecule_v10_fully_corrective_dense_sector_agpd"
AGPD_V8_MOLECULES = ("H6", "LiH")
AGPD_V10_MOLECULES = ("H4",)
AGPD_V8_VERSION = "hybrid-ansatz-F-shallow-one-body-cover-four-family-v8"
AGPD_V10_VERSION = (
    "hybrid-ansatz-F-fully-corrective-dense-sector-shallow-cover-v10"
)
AGPD_V10_CORE_VERSION = (
    "adaptive-af-four-circuit-pool-fully-corrective-dense-sector-"
    "shallow-collector-v10"
)
AGPD_V10_RESULT_CLASS = "fully_corrective_dense_sector_agpd_v10"
AGPD_V10_COMPARISON_VERSION = (
    "fully-corrective-dense-sector-agpd-vs-srcdf-and-pauli-v10"
)
AGPD_V10_SELECTION_RULE = (
    "hypot(sector_residual_operator_norm,range_sampling_proxy)"
)
AGPD_V10_RCOND = 1.0e-12
AGPD_SECTOR_LEAKAGE_TOLERANCE = 1.0e-10
BEH2_ROOT = ADAPTIVE / "beh2_srcdf_pauli_fullspace_k50_v2"
N2_ROOT = ADAPTIVE / "h2o_n2_srcdf_pauli_fullspace_k50_v3" / "N2"
SELECTED_ROOT = (
    ROOT
    / "RMeasurementAnsatz"
    / "submit_jctc"
    / "LiH_H2O_N2_potential_curves"
    / "mps_bond_dimension_analysis"
    / "selected_hamiltonians"
)
LEGACY_PAULI_DRIVER = (
    ROOT
    / "RMeasurementAnsatz"
    / "submit_jctc"
    / "LiH_H2O_N2_potential_curves"
    / "tnd_gpd_depth_scan"
    / "LiH_R1.50A_TND_chi64_corrected_sampling"
    / "run_same_hamiltonian_pauli_methods.py"
)
SAMPLING_HELPERS = LEGACY_PAULI_DRIVER.parents[1] / "sampling_error_estimation"
SHADOW_GROUPING = (
    ROOT
    / "RMeasurementAnsatz"
    / "Decompose-by-tensornetwork-feature-KongGit"
    / "existing_codes"
    / "new_datas_afterJun18"
    / "shadowgrouping-master"
)

DETAIL_CSV = HERE / "state_dependent_variance_by_budget.csv"
SUMMARY_CSV = HERE / "state_dependent_variance_T3000.csv"
MANIFEST_JSON = HERE / "manifest.json"
README_MD = HERE / "README.md"

ACTIVE_BUDGETS = (100, 200, 300, 500, 800, 1200, 2000, 3000)
FULLSPACE_BUDGETS = (1300, 1600, 2000, 2400, 3000)
MOLECULE_ORDER = ("H4", "H6", "LiH", "BeH2", "N2")
SRDD_METHOD = "SRDD"
LEGACY_SRCDF_METHOD = "s-RCDF"
LEGACY_SRCDF_STORED_METHOD = "standalone s-RCDF-F"
METHOD_ORDER = ("Derand", "OGM", "SG", SRDD_METHOD, "AGPD")
GEOMETRIES = {
    "H4": "linear R=1.20 A, STO-3G CAS(4e,4o), 8 qubits",
    "H6": "linear R=3.40 A, STO-3G CAS(4e,4o), 8 qubits",
    "LiH": "R=1.50 A, STO-3G CAS(2e,4o), 8 qubits",
    "BeH2": "linear Be-H=1.33376 A, full STO-3G (6e,7o), 14 qubits",
    "N2": "R=2.25 A, full STO-3G (14e,10o), 20 qubits",
}
FULLSPACE_CASES = {
    "BeH2": {
        "directory": "05_BeH2_R1.33376A_ArchivedJW",
        "electrons": 6,
        "summary": BEH2_ROOT / "sampling_summary_all.csv",
        "replicates": BEH2_ROOT / "sampling_replicates_all.csv",
        "audit": BEH2_ROOT / "BeH2" / "audit.json",
    },
    "N2": {
        "directory": "03_N2_R2.25A",
        "electrons": 14,
        "summary": N2_ROOT / "sampling_summary.csv",
        "replicates": N2_ROOT / "sampling_replicates.csv",
        "audit": N2_ROOT / "audit.json",
    },
}
PAULI_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
OGM_ITERATION_GUARD = 100_000
TOL = 2.0e-9
DERAND_ALGORITHM_ID = "qwc-seeded-huang2021-appendix-c-c8-c11-stable-delta-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def agpd_root_for(molecule: str) -> Path:
    if molecule in AGPD_V10_MOLECULES:
        return AGPD_ROOT_H4_V10
    if molecule in AGPD_V8_MOLECULES:
        return AGPD_ROOT_V8
    raise KeyError(f"No active-space AGPD artifact is registered for {molecule}")


def active_combined_rows() -> list[dict[str, str]]:
    """Overlay the H4 v10 comparison on the unchanged H6/LiH v8 rows."""
    sources = {
        "H4": COMBINED_H4_V10,
        "H6": COMBINED_V8,
        "LiH": COMBINED_V8,
    }
    output: list[dict[str, str]] = []
    for molecule, path in sources.items():
        selected = [row for row in read_csv(path) if row.get("molecule") == molecule]
        keys = [
            (row.get("method", ""), int(float(row["T_total_shots"])))
            for row in selected
        ]
        expected_methods = {
            "four-family Hybrid-F",
            LEGACY_SRCDF_STORED_METHOD,
            "OGM",
            "SG",
            "Derand",
        }
        methods = {method for method, _ in keys}
        budgets_by_method = {
            method: {total for stored_method, total in keys if stored_method == method}
            for method in methods
        }
        if (
            len(keys) != len(set(keys))
            or methods != expected_methods
            or any(set(ACTIVE_BUDGETS) != budgets for budgets in budgets_by_method.values())
        ):
            raise RuntimeError(
                f"{molecule}: active comparison overlay is incomplete or duplicated in {path}"
            )
        output.extend(selected)
    return output


def validate_combined_manifest() -> dict[str, Any]:
    v8_payload = json.loads(COMBINED_V8_MANIFEST.read_text(encoding="utf-8"))
    v10_payload = json.loads(COMBINED_H4_V10_MANIFEST.read_text(encoding="utf-8"))
    v8_srcdf_contract = v8_payload.get("standalone_collector_contract")
    v10_srcdf_contract = v10_payload.get("standalone_collector_contract")
    v8_agpd_contract = v8_payload.get("agpd_shallow_collector_contract")
    v10_agpd_contract = v10_payload.get("agpd_shallow_collector_contract")

    v8_agpd_manifest_path = (AGPD_ROOT_V8 / "manifest.json").resolve()
    v10_agpd_manifest_path = (AGPD_ROOT_H4_V10 / "manifest.json").resolve()
    v8_agpd_manifest = json.loads(
        v8_agpd_manifest_path.read_text(encoding="utf-8")
    )
    v10_agpd_manifest = json.loads(
        v10_agpd_manifest_path.read_text(encoding="utf-8")
    )

    def valid_srcdf_contract(contract: Any) -> bool:
        return (
            isinstance(contract, dict)
            and contract.get("status") == "PASS"
            and contract.get("mode") == "augment"
            and contract.get("formal_objective") == "greedy"
            and contract.get("proxy_state") == "hf"
        )

    def valid_cover_contract(contract: Any) -> bool:
        return (
            isinstance(contract, dict)
            and contract.get("status") == "PASS"
            and contract.get("mode") == "fixed_shallow_one_body_cover"
            and int(contract.get("source_to_collector_transfer_count", -1)) == 0
            and not parse_bool(
                contract.get("independent_collector_setting_present", True)
            )
            and contract.get("setting_count_rule") == "K+L_A"
        )

    v8_input_hashes = v8_payload.get("input_manifest_sha256", {})
    v10_input_hashes = v10_payload.get("input_manifest_sha256", {})
    v8_recorded_agpd_manifest_hash = v8_input_hashes.get(
        str(v8_agpd_manifest_path)
    )
    v10_recorded_agpd_manifest_hash = v10_input_hashes.get(
        str(v10_agpd_manifest_path)
    )
    if (
        v8_payload.get("status") != "PASS"
        or v8_payload.get("version")
        != "standalone-srcdf-vs-agpd-both-shallow-collectors-pauli-v8"
        or not valid_srcdf_contract(v8_srcdf_contract)
        or not valid_cover_contract(v8_agpd_contract)
        or v8_payload.get("output_sha256", {}).get(
            "sampling_error_comparison.csv"
        )
        != sha256_file(COMBINED_V8)
        or v8_recorded_agpd_manifest_hash
        != sha256_file(v8_agpd_manifest_path)
        or v8_agpd_manifest.get("version") != AGPD_V8_VERSION
        or v8_agpd_manifest.get("all_passed") is not True
        or not set(AGPD_V8_MOLECULES).issubset(
            set(v8_agpd_manifest.get("molecules", []))
        )
        or not set(ACTIVE_BUDGETS).issubset(
            {int(value) for value in v8_agpd_manifest.get("shot_budgets", [])}
        )
    ):
        raise RuntimeError(
            "H6/LiH comparison lacks the audited AGPD v8 shallow-cover contract"
        )

    v10_range_policy = (
        "fixed-particle-number sector when leakage-certified; "
        "otherwise full-Fock fallback"
    )
    if (
        v10_payload.get("status") != "PASS"
        or v10_payload.get("version") != AGPD_V10_COMPARISON_VERSION
        or v10_payload.get("fresh_rerun_molecules") != ["H4"]
        or v10_payload.get("path_certified_reuse_molecules") != []
        or not valid_srcdf_contract(v10_srcdf_contract)
        or not valid_cover_contract(v10_agpd_contract)
        or v10_agpd_contract.get("centered_range_policy") != v10_range_policy
        or v10_agpd_contract.get("fully_corrective_solver")
        != "real-coefficient SVD least squares"
        or float(v10_agpd_contract.get("fully_corrective_rcond", -1.0))
        != AGPD_V10_RCOND
        or v10_agpd_contract.get("selection_rule") != AGPD_V10_SELECTION_RULE
        or v10_agpd_contract.get("selection_uses_exact_ground_state") is not False
        or not np.isclose(
            float(
                v10_agpd_contract.get(
                    "maximum_allowed_sector_leakage_relative_frobenius",
                    math.inf,
                )
            ),
            AGPD_SECTOR_LEAKAGE_TOLERANCE,
            rtol=0.0,
            atol=0.0,
        )
        or v10_agpd_contract.get("molecules") != ["H4"]
        or v10_payload.get("output_sha256", {}).get(
            "sampling_error_comparison.csv"
        )
        != sha256_file(COMBINED_H4_V10)
        or v10_recorded_agpd_manifest_hash
        != sha256_file(v10_agpd_manifest_path)
        or v10_agpd_manifest.get("version") != AGPD_V10_VERSION
        or v10_agpd_manifest.get("adaptive_core_definition_version")
        != AGPD_V10_CORE_VERSION
        or v10_agpd_manifest.get("result_class") != AGPD_V10_RESULT_CLASS
        or float(v10_agpd_manifest.get("fully_corrective_rcond", -1.0))
        != AGPD_V10_RCOND
        or v10_agpd_manifest.get("selection_rule") != AGPD_V10_SELECTION_RULE
        or v10_agpd_manifest.get("selection_uses_exact_ground_state") is not False
        or v10_agpd_manifest.get("all_passed") is not True
        or set(v10_agpd_manifest.get("molecules", [])) != {"H4"}
        or set(int(value) for value in v10_agpd_manifest.get("shot_budgets", []))
        != set(ACTIVE_BUDGETS)
    ):
        raise RuntimeError(
            "H4 comparison lacks the fresh AGPD v10 fully-corrective "
            "dense-sector/range contract"
        )

    required_v8_contracts = (
        "v8_calibration_contract_500_probes_K1to10_8T_equal_strata",
        "four_family_pool_covered_at_every_boundary",
        "every_shot_budget_rerun_independently",
        "all_saved_shot_vectors_conserved",
        "circuit_depth_contract_1_le_d_lt_8",
        "eta_matches_configured_block_size_and_pool_boundaries_are_aligned",
        "all_four_families_use_complete_native_sources_plus_shallow_one_body_cover",
        "every_circuit_term_uses_ten_random_starts_and_keeps_the_best",
    )
    case_audit_hashes: dict[str, str] = {}
    case_effective_status: dict[str, str] = {}
    supplemental_audit_hashes: dict[str, str] = {}
    for molecule in AGPD_V8_MOLECULES:
        audit_path = AGPD_ROOT_V8 / molecule / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        required_contracts_pass = all(
            audit.get(contract) is True for contract in required_v8_contracts
        )
        common_contracts_pass = (
            required_contracts_pass
            and audit.get("source_to_collector_transfer_policy")
            == "disabled for all four families"
            and audit.get("exact_ground_state_used_for_selection") is False
        )
        base_audit_passes = (
            audit.get("status") == "PASS"
            and common_contracts_pass
            and audit.get("sampling_uses_loss_selected_approximate_Hamiltonian")
            is True
        )
        effective_status = "PASS" if base_audit_passes else "FAIL"

        # H4's decimal-grid hashes can fail at the producer's 2e-10 matrix
        # threshold even though an independent 2e-8 numerical replay closes.
        # Admit only that explicitly audited failure mode; every structural v8
        # contract and every saved budget row must still pass independently.
        if not base_audit_passes:
            supplemental_path = (
                AGPD_ROOT_V8 / molecule / "sampling_equivalence_audit.json"
            )
            supplemental = (
                json.loads(supplemental_path.read_text(encoding="utf-8"))
                if supplemental_path.is_file()
                else None
            )
            supplemental_rows = (
                supplemental.get("rows") if isinstance(supplemental, dict) else None
            )
            try:
                matrix_tolerance = float(
                    supplemental["matrix_frobenius_tolerance"]
                )
                loss_tolerance = float(supplemental["loss_tolerance"])
                maximum_matrix_error = float(
                    supplemental[
                        "maximum_setting_sum_reconstruction_error_frobenius"
                    ]
                )
                maximum_loss_error = float(
                    supplemental["maximum_loss_replay_error"]
                )
            except (KeyError, TypeError, ValueError):
                matrix_tolerance = loss_tolerance = math.nan
                maximum_matrix_error = maximum_loss_error = math.nan

            rows_by_budget: dict[int, dict[str, Any]] = {}
            rows_are_valid = isinstance(supplemental_rows, list)
            if rows_are_valid:
                for row in supplemental_rows:
                    try:
                        total_shots = int(row["T_total_shots"])
                        row_matrix_error = float(row["matrix_frobenius_error"])
                        row_loss_error = float(row["loss_replay_error"])
                    except (KeyError, TypeError, ValueError):
                        rows_are_valid = False
                        break
                    if (
                        total_shots in rows_by_budget
                        or not math.isfinite(row_matrix_error)
                        or not math.isfinite(row_loss_error)
                        or row_matrix_error < 0.0
                        or row_loss_error < 0.0
                        or row_matrix_error > matrix_tolerance
                        or row_loss_error > loss_tolerance
                        or row.get("status") != "PASS"
                        or row.get("matrix_equivalent_within_tolerance") is not True
                        or row.get("loss_replayed_within_tolerance") is not True
                        or row.get("shot_vector_exactly_reused_and_conserved")
                        is not True
                        or row.get("loss_selected_K_exactly_reused") is not True
                    ):
                        rows_are_valid = False
                        break
                    rows_by_budget[total_shots] = row

            supplemental_closure_passes = (
                audit.get("status") == "FAIL"
                and common_contracts_pass
                and audit.get(
                    "sampling_uses_loss_selected_approximate_Hamiltonian"
                )
                is False
                and isinstance(supplemental, dict)
                and supplemental.get("status") == "PASS"
                and supplemental.get("case") == molecule
                and all(
                    math.isfinite(value) and value >= 0.0
                    for value in (
                        matrix_tolerance,
                        loss_tolerance,
                        maximum_matrix_error,
                        maximum_loss_error,
                    )
                )
                and maximum_matrix_error <= matrix_tolerance
                and maximum_loss_error <= loss_tolerance
                and rows_are_valid
                and set(rows_by_budget) == set(ACTIVE_BUDGETS)
            )
            if supplemental_closure_passes:
                effective_status = "PASS_WITH_SAMPLING_EQUIVALENCE_AUDIT"
                supplemental_audit_hashes[molecule] = sha256_file(
                    supplemental_path
                )

        if effective_status == "FAIL":
            raise RuntimeError(
                f"{molecule}: AGPD v8 case audit lacks an accepted numerical closure"
            )
        case_audit_hashes[molecule] = sha256_file(audit_path)
        case_effective_status[molecule] = effective_status

    h4_audit_path = AGPD_ROOT_H4_V10 / "H4" / "audit.json"
    h4_audit = json.loads(h4_audit_path.read_text(encoding="utf-8"))
    required_v9_contracts = (
        "v9_calibration_contract_500_probes_K1to10_8T_equal_strata",
        "four_family_pool_covered_at_every_boundary",
        "every_shot_budget_rerun_independently",
        "all_saved_shot_vectors_conserved",
        "circuit_depth_contract_1_le_d_lt_8",
        "eta_matches_configured_block_size_and_pool_boundaries_are_aligned",
        "all_four_families_use_complete_native_sources_plus_shallow_one_body_cover",
        "all_AGPD_allocations_use_certified_sector_ranges_or_full_fock_fallback",
        "every_circuit_term_uses_ten_random_starts_and_keeps_the_best",
    )
    if (
        h4_audit.get("status") != "PASS"
        or h4_audit.get("molecule") != "H4"
        or not all(h4_audit.get(key) is True for key in required_v9_contracts)
        or h4_audit.get("sampling_uses_loss_selected_approximate_Hamiltonian")
        is not True
        or h4_audit.get("source_to_collector_transfer_policy")
        != "disabled for all four families"
        or h4_audit.get("exact_ground_state_used_for_selection") is not False
        or h4_audit.get("fully_corrective_dense_sector_contract") is not True
        or h4_audit.get("fully_corrective_solver")
        != "real-coefficient SVD least squares"
        or float(h4_audit.get("fully_corrective_rcond", -1.0))
        != AGPD_V10_RCOND
        or h4_audit.get("selection_rule") != AGPD_V10_SELECTION_RULE
        or h4_audit.get(
            "probe_calibration_retained_for_paired_audit_but_not_used_for_selection"
        )
        is not True
    ):
        raise RuntimeError(
            "H4 AGPD v10 case audit lacks a strict fully-corrective "
            "dense-sector/range numerical closure"
        )
    case_audit_hashes["H4"] = sha256_file(h4_audit_path)
    case_effective_status["H4"] = "PASS_V10_FULLY_CORRECTIVE_DENSE_SECTOR"

    # This also rejects a partial or duplicated molecule-level overlay before
    # any variance row is trusted.
    active_combined_rows()
    return {
        "status": "PASS",
        "srcdf_mode": "augment",
        "agpd_mode": "fixed_shallow_one_body_cover",
        "agpd_setting_count_rule": "K+L_A",
        "agpd_source_to_collector_transfer_count": 0,
        "agpd_independent_collector_setting_present": False,
        "scope": (
            "SRDD uses K+L_K shallow settings; H4 AGPD uses the fresh v10 "
            "fully-corrective dense-sector fit and leakage-certified range "
            "allocation, while H6/LiH "
            "retain their audited v8 cover-plus-native-source estimators"
        ),
        "comparison_manifest_by_molecule": {
            "H4": str(COMBINED_H4_V10_MANIFEST.resolve()),
            "H6": str(COMBINED_V8_MANIFEST.resolve()),
            "LiH": str(COMBINED_V8_MANIFEST.resolve()),
        },
        "comparison_summary_sha256_by_molecule": {
            "H4": sha256_file(COMBINED_H4_V10),
            "H6": sha256_file(COMBINED_V8),
            "LiH": sha256_file(COMBINED_V8),
        },
        "agpd_manifest_by_molecule": {
            "H4": str(v10_agpd_manifest_path),
            "H6": str(v8_agpd_manifest_path),
            "LiH": str(v8_agpd_manifest_path),
        },
        "agpd_manifest_sha256_by_molecule": {
            "H4": v10_recorded_agpd_manifest_hash,
            "H6": v8_recorded_agpd_manifest_hash,
            "LiH": v8_recorded_agpd_manifest_hash,
        },
        "agpd_version_by_molecule": {
            "H4": AGPD_V10_VERSION,
            "H6": AGPD_V8_VERSION,
            "LiH": AGPD_V8_VERSION,
        },
        "agpd_centered_range_policy_by_molecule": {
            "H4": v10_range_policy,
            "H6": "audited v8 full-Fock centered half-range",
            "LiH": "audited v8 full-Fock centered half-range",
        },
        "h4_sector_leakage_tolerance": AGPD_SECTOR_LEAKAGE_TOLERANCE,
        "agpd_case_audit_sha256": case_audit_hashes,
        "agpd_case_effective_status": case_effective_status,
        "agpd_sampling_equivalence_audit_sha256": supplemental_audit_hashes,
    }


def validate_combined_agpd_row(row: dict[str, str]) -> dict[str, Any]:
    """Fail closed unless one AGPD row is its registered complete estimator."""
    molecule = row.get("molecule", "")
    is_h4_v10 = molecule in AGPD_V10_MOLECULES
    required = {
        "molecule",
        "T_total_shots",
        "K_selected",
        "measurement_settings",
        "collector_mode",
        "collector_extra_leaves",
        "collector_depth",
        "source_to_collector_transfer_count",
        "independent_collector_setting_present",
        "analytic_sampling_SE",
        "signed_approximation_bias",
        "provenance",
    }
    if is_h4_v10:
        required.update(
            {
                "centered_range_domain",
                "full_fock_centered_range_sum",
                "sector_centered_range_sum",
                "sector_range_sum_reduction_fraction",
                "maximum_sector_leakage_relative_frobenius",
                "sector_operator_norm",
                "fully_corrective_design_rank",
                "fully_corrective_design_nullity",
                "fully_corrective_rcond",
                "selection_rule",
                "selection_uses_exact_ground_state",
            }
        )
    missing = sorted(required - set(row))
    if missing:
        version = "v10 fully-corrective dense-sector" if is_h4_v10 else "v8"
        raise RuntimeError(f"Active-space AGPD row lacks {version} fields: {missing}")

    total = int(float(row["T_total_shots"]))
    k = int(float(row["K_selected"]))
    extra = int(float(row["collector_extra_leaves"]))
    settings = int(float(row["measurement_settings"]))
    cover_depth = int(float(row["collector_depth"]))
    budget_path = agpd_root_for(molecule) / molecule / f"budget_T{total}.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    best = budget["best"]
    selected = budget["selected_sampling"]
    cover_audit = selected.get("shallow_one_body_cover_audit", {})

    if (
        row["collector_mode"] != "fixed_shallow_one_body_cover"
        or extra < 1
        or settings != k + extra
        or not 1 <= cover_depth <= 3
        or int(float(row["source_to_collector_transfer_count"])) != 0
        or parse_bool(row["independent_collector_setting_present"])
    ):
        raise RuntimeError(
            f"{molecule} T={total}: combined AGPD row violates the v8 K+L_A contract"
        )

    shots = [int(value) for value in best["shot_vector"]]
    families = list(selected["setting_families"])
    interfaces = list(selected["setting_interfaces"])
    kinds = list(selected["setting_kinds"])
    depths = list(selected["setting_rotation_depths"])
    ranges = [float(value) for value in selected["setting_centered_ranges"]]
    if any(
        len(values) != settings
        for values in (shots, families, interfaces, kinds, depths, ranges)
    ):
        raise RuntimeError(
            f"{molecule} T={total}: AGPD cover/source setting arrays are not K+L_A complete"
        )
    if (
        sum(shots) != total
        or any(value < 0 for value in shots)
        or any(shot == 0 and width > 1.0e-14 for shot, width in zip(shots, ranges))
        or any(not np.isfinite(width) or width < 0.0 for width in ranges)
    ):
        raise RuntimeError(f"{molecule} T={total}: invalid complete AGPD shot vector")

    range_metadata: dict[str, Any]
    if is_h4_v10:
        full_ranges = [
            float(value) for value in selected["setting_full_fock_centered_ranges"]
        ]
        sector_ranges = [
            float(value)
            for value in selected["setting_sector_spectral_centered_ranges"]
        ]
        allocation_ranges = [
            float(value) for value in selected["setting_allocation_centered_ranges"]
        ]
        range_domains = [str(value) for value in selected["setting_range_domains"]]
        leakage_values = [
            float(value)
            for value in selected["setting_sector_leakage_relative_frobenius"]
        ]
        sector_eligible = [
            parse_bool(value) for value in selected["setting_sector_range_eligible"]
        ]
        if any(
            len(values) != settings
            for values in (
                full_ranges,
                sector_ranges,
                allocation_ranges,
                range_domains,
                leakage_values,
                sector_eligible,
            )
        ):
            raise RuntimeError(
                f"H4 T={total}: v10 range/leakage arrays are not K+L_A complete"
            )
        allowed_domains = {
            "fixed_particle_number_sector",
            "full_fock_fallback_due_to_sector_leakage",
        }
        for index, (
            width,
            full_width,
            sector_width,
            allocation_width,
            domain,
            leakage,
            eligible,
        ) in enumerate(
            zip(
                ranges,
                full_ranges,
                sector_ranges,
                allocation_ranges,
                range_domains,
                leakage_values,
                sector_eligible,
            )
        ):
            if (
                domain not in allowed_domains
                or not all(
                    math.isfinite(value) and value >= 0.0
                    for value in (
                        width,
                        full_width,
                        sector_width,
                        allocation_width,
                        leakage,
                    )
                )
                or not np.isclose(width, allocation_width, rtol=1.0e-12, atol=1.0e-14)
            ):
                raise RuntimeError(f"H4 T={total}: invalid v10 range row at setting {index}")
            if eligible:
                valid_domain = (
                    domain == "fixed_particle_number_sector"
                    and leakage <= AGPD_SECTOR_LEAKAGE_TOLERANCE
                    and allocation_width <= full_width + 1.0e-12
                    and allocation_width + 1.0e-12 >= sector_width
                )
            else:
                valid_domain = (
                    domain == "full_fock_fallback_due_to_sector_leakage"
                    and leakage > AGPD_SECTOR_LEAKAGE_TOLERANCE
                    and np.isclose(
                        allocation_width,
                        full_width,
                        rtol=1.0e-12,
                        atol=1.0e-14,
                    )
                )
            if not valid_domain:
                raise RuntimeError(
                    f"H4 T={total}: sector eligibility/domain mismatch at setting {index}"
                )

        sector_count = sum(
            domain == "fixed_particle_number_sector" for domain in range_domains
        )
        fallback_count = settings - sector_count
        aggregate_domain = (
            "fixed_particle_number_sector"
            if fallback_count == 0
            else "mixed_sector_and_full_fock_fallback"
        )
        full_sum = float(sum(full_ranges))
        allocation_sum = float(sum(allocation_ranges))
        reduction = 0.0 if full_sum <= 0.0 else 1.0 - allocation_sum / full_sum
        maximum_leakage = max(leakage_values, default=0.0)
        if (
            row["provenance"]
            != "fresh_four_family_v10_fully_corrective_dense_sector_rerun"
            or best.get("centered_range_domain") != aggregate_domain
            or int(best.get("sector_range_setting_count", -1)) != sector_count
            or int(best.get("full_fock_fallback_setting_count", -1))
            != fallback_count
            or best.get("all_settings_sector_safe_or_full_fock_fallback") is not True
            or not np.isclose(
                float(best.get("full_fock_centered_range_sum", math.nan)),
                full_sum,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or not np.isclose(
                float(best.get("sector_centered_range_sum", math.nan)),
                allocation_sum,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or not np.isclose(
                float(best.get("sector_range_sum_reduction_fraction", math.nan)),
                reduction,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or not np.isclose(
                float(
                    best.get(
                        "maximum_sector_leakage_relative_frobenius", math.nan
                    )
                ),
                maximum_leakage,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or cover_audit.get("centered_range_domain") != aggregate_domain
            or int(cover_audit.get("sector_range_setting_count", -1))
            != sector_count
            or int(cover_audit.get("full_fock_fallback_setting_count", -1))
            != fallback_count
            or not np.isclose(
                float(row["full_fock_centered_range_sum"]),
                full_sum,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or not np.isclose(
                float(row["sector_centered_range_sum"]),
                allocation_sum,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or row["centered_range_domain"] != aggregate_domain
            or not np.isclose(
                float(row["sector_range_sum_reduction_fraction"]),
                reduction,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or not np.isclose(
                float(row["maximum_sector_leakage_relative_frobenius"]),
                maximum_leakage,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
        ):
            raise RuntimeError(
                f"H4 T={total}: v10 aggregate range/leakage provenance failed audit"
            )
        correction = best.get("fully_corrective_dense_sector_audit")
        sampled_correction = selected.get("fully_corrective_dense_sector_audit")
        operator_norm = float(best.get("x_sector_operator_norm", math.inf))
        design_rank = int(correction.get("design_rank", -1)) if isinstance(correction, dict) else -1
        design_nullity = (
            int(correction.get("design_nullity", -1))
            if isinstance(correction, dict)
            else -1
        )
        if (
            not isinstance(correction, dict)
            or sampled_correction != correction
            or best.get("selection_rule") != AGPD_V10_SELECTION_RULE
            or selected.get("selection_rule") != AGPD_V10_SELECTION_RULE
            or parse_bool(best.get("selection_uses_calibrated_probe_mean_weights", True))
            or not np.isclose(
                float(selected.get("selection_sector_operator_norm", math.nan)),
                operator_norm,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or correction.get("method") != "fully_corrective_dense_sector_block_dd"
            or correction.get("objective")
            != "sector_frobenius_fixed_rotations_joint_dd_coefficients"
            or correction.get("solver") != "numpy.linalg.lstsq_svd_minimum_norm"
            or correction.get("uses_ground_state") is not False
            or correction.get("fixed_rotations") is not True
            or float(correction.get("svd_rcond", -1.0)) != AGPD_V10_RCOND
            or int(correction.get("source_count", -1)) != k
            or int(correction.get("dd_feature_count_per_source", -1)) != 37
            or correction.get("coefficient_shape") != [k, 37]
            or design_rank <= 0
            or design_nullity < 0
            or design_rank + design_nullity != 37 * k
            or float(correction.get("sector_frobenius_after", math.inf))
            > float(correction.get("sector_frobenius_before", -math.inf)) + 1.0e-10
            or not np.isclose(
                float(row["sector_operator_norm"]),
                operator_norm,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or int(float(row["fully_corrective_design_rank"])) != design_rank
            or int(float(row["fully_corrective_design_nullity"])) != design_nullity
            or float(row["fully_corrective_rcond"]) != AGPD_V10_RCOND
            or row["selection_rule"] != AGPD_V10_SELECTION_RULE
            or parse_bool(row["selection_uses_exact_ground_state"])
        ):
            raise RuntimeError(
                f"H4 T={total}: invalid v10 fully-corrective selected-prefix audit"
            )
        range_metadata = {
            "artifact_version": AGPD_V10_VERSION,
            "centered_range_policy": (
                "fixed-particle-number sector when leakage-certified; "
                "otherwise full-Fock fallback"
            ),
            "centered_range_domain": aggregate_domain,
            "full_fock_centered_range_sum": full_sum,
            "sector_allocation_centered_range_sum": allocation_sum,
            "sector_range_sum_reduction_fraction": reduction,
            "maximum_sector_leakage_relative_frobenius": maximum_leakage,
            "sector_leakage_tolerance": AGPD_SECTOR_LEAKAGE_TOLERANCE,
            "sector_range_setting_count": sector_count,
            "full_fock_fallback_setting_count": fallback_count,
            "sector_operator_norm": operator_norm,
            "fully_corrective_design_rank": design_rank,
            "fully_corrective_design_nullity": design_nullity,
            "fully_corrective_rcond": AGPD_V10_RCOND,
            "selection_rule": AGPD_V10_SELECTION_RULE,
            "selection_uses_exact_ground_state": False,
        }
    else:
        if row["provenance"] != "fresh_four_family_v8_shallow_collector_rerun":
            raise RuntimeError(
                f"{molecule} T={total}: v8 row has incorrect provenance label"
            )
        range_metadata = {
            "artifact_version": AGPD_V8_VERSION,
            "centered_range_policy": "audited v8 full-Fock centered half-range",
            "centered_range_domain": "full_fock",
            "full_fock_centered_range_sum": float(sum(ranges)),
            "sector_allocation_centered_range_sum": "",
            "sector_range_sum_reduction_fraction": "",
            "maximum_sector_leakage_relative_frobenius": "",
            "sector_leakage_tolerance": "",
            "sector_range_setting_count": 0,
            "full_fock_fallback_setting_count": settings,
            "sector_operator_norm": "",
            "fully_corrective_design_rank": "",
            "fully_corrective_design_nullity": "",
            "fully_corrective_rcond": "",
            "selection_rule": "",
            "selection_uses_exact_ground_state": "",
        }

    cover_slice = slice(0, extra)
    source_slice = slice(extra, settings)
    if (
        any(value != "agpd_shallow_one_body" for value in families[cover_slice])
        or any(
            value != "agpd_depth_bounded_one_body_cover"
            for value in interfaces[cover_slice]
        )
        or any(value != "agpd_shallow_one_body_leaf" for value in kinds[cover_slice])
        or any(int(value) != cover_depth for value in depths[cover_slice])
        or any(value == "agpd_shallow_one_body" for value in families[source_slice])
        or any(
            value != "native_shallow_source_no_collector_transfer"
            for value in interfaces[source_slice]
        )
        or any(value != "source_leaf" for value in kinds[source_slice])
        or any(value is None or not 1 <= int(value) < 8 for value in depths[source_slice])
    ):
        raise RuntimeError(
            f"{molecule} T={total}: AGPD setting order/kind/depth is not "
            "L_A shallow-cover leaves followed by K complete native sources"
        )

    if (
        int(budget["total_shots"]) != total
        or bool(budget.get("right_censored", True))
        or int(best["K_source_terms"]) != k
        or int(best["complete_native_source_setting_count"]) != k
        or int(best["shallow_one_body_cover_setting_count"]) != extra
        or int(best["shallow_one_body_cover_depth"]) != cover_depth
        or int(best["total_settings"]) != settings
        or int(best.get("F3_transfer_source_count", -1)) != 0
        or int(best.get("source_to_collector_transfer_count", -1)) != 0
        or bool(best.get("independent_collector_setting_present", True))
        or int(selected["selected_K_source_terms"]) != k
        or int(selected["shallow_one_body_cover_setting_count"]) != extra
        or int(selected["shallow_one_body_cover_depth"]) != cover_depth
        or int(selected.get("source_to_collector_transfer_count", -1)) != 0
        or bool(selected.get("independent_collector_setting_present", True))
        or selected.get("shot_vector_exactly_reused") is not True
        or int(selected["allocated_total_shots"]) != total
        or cover_audit.get("policy")
        != "fixed_shallow_one_body_cover_and_complete_native_sources"
        or int(cover_audit.get("source_transfer_count", -1)) != 0
        or int(cover_audit.get("cover_setting_count", -1)) != extra
        or int(cover_audit.get("source_setting_count", -1)) != k
        or int(cover_audit.get("measurement_setting_count", -1)) != settings
        or int(cover_audit.get("collector_depth", -1)) != cover_depth
        or cover_audit.get("all_measurement_settings_depth_limited") is not True
        or cover_audit.get("complete_source_matrices_retained") is not True
        or bool(cover_audit.get("independent_collector_setting_present", True))
    ):
        raise RuntimeError(f"{molecule} T={total}: saved AGPD selected prefix failed audit")

    source_se = float(selected["analytic_sampling_SE"])
    source_bias = float(selected["signed_approximation_bias"])
    if (
        not np.isclose(
            float(row["analytic_sampling_SE"]), source_se, rtol=1.0e-12, atol=1.0e-14
        )
        or not np.isclose(
            float(row["signed_approximation_bias"]),
            source_bias,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
    ):
        raise RuntimeError(
            f"{molecule} T={total}: combined AGPD variance/bias differs from selected_sampling"
        )
    return {
        "settings": settings,
        "source_leaves_K": k,
        "extra_one_body_leaves_L": extra,
        "cover_depth": cover_depth,
        "shot_vector": shots,
        "analytic_sampling_SE": source_se,
        "signed_bias": source_bias,
        "source": str(budget_path.resolve()),
        **range_metadata,
    }


def validate_combined_srcdf_row(row: dict[str, str]) -> int:
    required = {
        "K_selected",
        "measurement_settings",
        "collector_mode",
        "collector_extra_leaves",
        "collector_reconstruction_residual",
        "independent_collector_setting_present",
    }
    missing = sorted(required - set(row))
    if missing:
        raise RuntimeError(f"Active-space SRDD row lacks collector fields: {missing}")
    k = int(float(row["K_selected"]))
    extra = int(float(row["collector_extra_leaves"]))
    settings = int(float(row["measurement_settings"]))
    if row["collector_mode"] != "augment" or extra < 1 or settings != k + extra:
        raise RuntimeError("Active-space SRDD row is not a formal K+L augmentation")
    if parse_bool(row["independent_collector_setting_present"]):
        raise RuntimeError("Active-space SRDD row contains an independent collector")
    residual = float(row["collector_reconstruction_residual"])
    if not np.isfinite(residual) or residual > 1.0e-8:
        raise RuntimeError("Active-space SRDD collector reconstruction failed")
    return settings


def validate_fullspace_srcdf_row(
    molecule: str, row: dict[str, str], audit: dict[str, Any]
) -> int:
    configuration = audit.get("collector_configuration")
    if (
        audit.get("status") != "PASS"
        or not isinstance(configuration, dict)
        or configuration.get("mode") != "augment"
        or configuration.get("objective") != "greedy"
        or configuration.get("proxy_state") != "hf"
        or audit.get("all_feasible_collector_matrix_reconstructions_pass") is not True
        or audit.get("sRCDF_ground_state_used_to_optimize_collector") is not False
    ):
        raise RuntimeError(f"{molecule}: invalid full-space shallow-collector audit")
    required = {
        "selected_K",
        "collector_mode",
        "collector_extra_leaves",
        "measurement_settings",
        "collector_matrix_relative_reconstruction_residual",
        "shot_vector",
        "T_total_shots",
    }
    missing = sorted(required - set(row))
    if missing:
        raise RuntimeError(f"{molecule}: full-space SRDD row lacks {missing}")
    k = int(float(row["selected_K"]))
    extra = int(float(row["collector_extra_leaves"]))
    settings = int(float(row["measurement_settings"]))
    extra_grid = [int(value) for value in configuration.get("extra_leaf_grid", [])]
    if (
        row["collector_mode"] != "augment"
        or extra < 1
        or extra not in extra_grid
        or settings != k + extra
    ):
        raise RuntimeError(f"{molecule}: full-space setting count is not formal K+L")
    residual = float(row["collector_matrix_relative_reconstruction_residual"])
    if not np.isfinite(residual) or residual > 5.0e-10:
        raise RuntimeError(f"{molecule}: full-space collector reconstruction failed")
    shots = [int(value) for value in row["shot_vector"].split()]
    if (
        len(shots) != settings
        or sum(shots) != int(row["T_total_shots"])
        or any(value < 1 for value in shots)
    ):
        raise RuntimeError(f"{molecule}: full-space K+L shot audit failed")
    return settings


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy_pauli_module():
    sys.path[:0] = [str(SAMPLING_HELPERS), str(SHADOW_GROUPING)]
    return load_module(LEGACY_PAULI_DRIVER, "state_variance_legacy_pauli")


def dense_mps(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        arrays = [np.asarray(payload[key]) for key in sorted(payload.files)]
    value = arrays[0][0, :, :]
    for array in arrays[1:]:
        value = np.tensordot(value, array, axes=(-1, 0))
    state = np.asarray(value[..., 0], dtype=np.complex128).reshape(-1)
    state /= np.linalg.norm(state)
    pivot = state[int(np.argmax(np.abs(state)))]
    if abs(pivot):
        state *= np.exp(-1.0j * np.angle(pivot))
    return state


def particle_sector_indices(n: int, electrons: int) -> np.ndarray:
    if n % 2 or electrons % 2:
        raise ValueError("Blocked-spin sector requires even n and electron count")
    m = n // 2
    nalpha = electrons // 2
    low_mask = (1 << m) - 1
    return np.fromiter(
        (
            index
            for index in range(1 << n)
            if (index >> m).bit_count() == nalpha
            and (index & low_mask).bit_count() == nalpha
        ),
        dtype=np.int64,
    )


def project_state(state: np.ndarray, sector_indices: np.ndarray) -> tuple[np.ndarray, float]:
    outside = np.ones(len(state), dtype=bool)
    outside[sector_indices] = False
    leakage_norm = float(np.linalg.norm(state[outside]))
    projected = np.zeros_like(state)
    projected[sector_indices] = state[sector_indices]
    retained = float(np.vdot(projected, projected).real)
    if retained <= 0:
        raise ValueError("State has zero norm in the declared physical sector")
    projected /= math.sqrt(retained)
    return projected, leakage_norm


def load_active_case(molecule: str) -> dict[str, Any]:
    path = (
        H4_PAULI_FIXED_ROOT / "same_target_state_input.npz"
        if molecule == "H4"
        else PAULI_ROOT / molecule / "same_target_state_input.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        state = np.asarray(payload["ground_state"], dtype=np.complex128)
        observables = np.asarray(payload["pauli_observables"], dtype=np.int8)
        weights = np.asarray(payload["pauli_weights"], dtype=float)
        offset = float(payload["pauli_offset"])
        exact_energy = float(payload["exact_energy"])
        saved_means = np.asarray(payload["pauli_term_expectations"], dtype=float)
        sector = (
            particle_sector_indices(observables.shape[1], 4)
            if molecule == "H4"
            else np.asarray(payload["sector_indices"], dtype=np.int64)
        )
    state, leakage = project_state(state / np.linalg.norm(state), sector)
    return {
        "name": molecule,
        "n": observables.shape[1],
        "observables": observables,
        "weights": weights,
        "offset": offset,
        "state": state,
        "sector_indices": sector,
        "sector_leakage_norm": leakage,
        "exact_energy": exact_energy,
        "saved_term_means": saved_means,
        "input_paths": [path],
    }


def load_fullspace_case(molecule: str) -> dict[str, Any]:
    spec = FULLSPACE_CASES[molecule]
    directory = SELECTED_ROOT / str(spec["directory"])
    pauli_path = directory / "hamiltonian_pauli_blocked_spin.csv"
    metadata_path = directory / "metadata.json"
    state_path = directory / "ground_state_mps_blocked_spin.npz"
    # Match the archived full-space runner byte-for-byte here.  Its schedules
    # are sensitive to floating-point tie breaking in SG/Derand, and pandas'
    # CSV parser can differ by one ulp from Python's scalar ``float`` parser.
    # Reusing the original parser is therefore part of exact schedule replay.
    frame = pd.read_csv(pauli_path)
    labels = frame["full_label_site0_to_siteNminus1"].astype(str).tolist()
    observables_all = np.asarray(
        [[PAULI_CODE[item] for item in label] for label in labels], dtype=np.int8
    )
    weights_all = frame["coefficient_real_hartree"].to_numpy(dtype=float)
    identity = np.all(observables_all == 0, axis=1)
    if int(identity.sum()) != 1:
        raise ValueError(f"{molecule}: expected exactly one identity term")
    keep = (~identity) & (np.abs(weights_all) > 1.0e-14)
    observables = observables_all[keep]
    weights = weights_all[keep]
    offset = float(weights_all[identity][0])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state = dense_mps(state_path)
    sector = particle_sector_indices(observables.shape[1], int(spec["electrons"]))
    state, leakage = project_state(state, sector)
    return {
        "name": molecule,
        "n": observables.shape[1],
        "observables": observables,
        "weights": weights,
        "offset": offset,
        "state": state,
        "sector_indices": sector,
        "sector_leakage_norm": leakage,
        "exact_energy": float(metadata["fci_ground_energy_hartree"]),
        "saved_term_means": None,
        "input_paths": [pauli_path, metadata_path, state_path],
    }


def counter_from_rows(rows: np.ndarray) -> Counter[tuple[int, ...]]:
    return Counter(tuple(map(int, row)) for row in np.asarray(rows, dtype=np.int8))


def load_active_schedules(molecule: str) -> dict[str, dict[int, Counter]]:
    directory = (
        H4_PAULI_FIXED_ROOT / "schedules"
        if molecule == "H4"
        else PAULI_ROOT / molecule / "schedules"
    )
    output: dict[str, dict[int, Counter]] = {"OGM": {}, "SG": {}, "Derand": {}}
    with np.load(directory / "SG_max.npz", allow_pickle=False) as payload:
        sg = np.asarray(payload["settings"], dtype=np.int8)
    with np.load(directory / "Derand_max.npz", allow_pickle=False) as payload:
        derand = np.asarray(payload["settings"], dtype=np.int8)
    for total in ACTIVE_BUDGETS:
        output["SG"][total] = counter_from_rows(sg[:total])
        output["Derand"][total] = counter_from_rows(derand[:total])
        with np.load(directory / f"OGM_T{total}.npz", allow_pickle=False) as payload:
            output["OGM"][total] = counter_from_rows(payload["settings"])
    return output


def sorted_insertion_cover(
    observables: np.ndarray, weights: np.ndarray, legacy
) -> np.ndarray:
    groups: list[np.ndarray] = []
    for index in np.argsort(-np.abs(weights), kind="stable"):
        term = observables[index]
        for setting in groups:
            if np.all((term == 0) | (setting == 0) | (term == setting)):
                mask = (setting == 0) & (term != 0)
                setting[mask] = term[mask]
                break
        else:
            groups.append(term.copy())
    result = np.asarray(groups, dtype=np.int8)
    result[result == 0] = 3
    hits = legacy.hit_matrix(observables, result)
    if np.any(hits.sum(axis=1) == 0):
        raise RuntimeError("Sorted-insertion QWC cover is incomplete")
    return result


def unique_settings(rows: Iterable[np.ndarray]) -> np.ndarray:
    seen: dict[tuple[int, ...], np.ndarray] = {}
    for row in rows:
        key = tuple(map(int, row))
        seen.setdefault(key, np.asarray(row, dtype=np.int8))
    return np.asarray(list(seen.values()), dtype=np.int8)


def sparse_hit_matrix(observables: np.ndarray, settings: np.ndarray, legacy) -> sp.csc_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    for column, setting in enumerate(settings):
        hit = np.flatnonzero(legacy.setting_hits(observables, setting))
        row_indices.extend(hit.tolist())
        column_indices.extend([column] * len(hit))
    matrix = sp.csc_matrix(
        (np.ones(len(row_indices)), (row_indices, column_indices)),
        shape=(len(observables), len(settings)),
    )
    if np.any(np.asarray(matrix.sum(axis=1)).reshape(-1) == 0):
        raise RuntimeError("OGM candidate bank is incomplete")
    return matrix


def optimize_ogm(weights: np.ndarray, coverage: sp.csc_matrix) -> tuple[np.ndarray, dict[str, Any]]:
    weights_sq = weights**2
    initial = np.asarray(np.abs(weights) @ coverage).reshape(-1)
    initial = np.maximum(initial, 1.0e-12)
    initial /= initial.sum()

    def value_gradient(logits: np.ndarray):
        shifted = logits - np.max(logits)
        exponential = np.exp(shifted)
        probabilities = exponential / exponential.sum()
        chi = np.maximum(np.asarray(coverage @ probabilities).reshape(-1), 1.0e-14)
        value = float(np.sum(weights_sq / chi))
        gradient_probability = -np.asarray(
            coverage.T @ (weights_sq / chi**2)
        ).reshape(-1)
        gradient = probabilities * (
            gradient_probability - float(gradient_probability @ probabilities)
        )
        return value, gradient

    result = minimize(
        value_gradient,
        np.log(initial),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": OGM_ITERATION_GUARD,
            "maxfun": 500_000,
            "ftol": 1.0e-13,
            "gtol": 1.0e-8,
            "maxls": 100,
        },
    )
    shifted = np.asarray(result.x) - np.max(result.x)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    audit = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "candidate_count": int(coverage.shape[1]),
        "nonzero_probability_count": int(np.count_nonzero(probabilities > 1.0e-12)),
    }
    if not result.success:
        raise RuntimeError(f"OGM optimization failed: {audit}")
    return probabilities, audit


def largest_remainder_counts(total: int, probabilities: np.ndarray) -> np.ndarray:
    quotas = total * probabilities
    counts = np.floor(quotas).astype(np.int64)
    missing = total - int(counts.sum())
    if missing:
        order = np.argsort(-(quotas - counts), kind="stable")
        counts[order[:missing]] += 1
    return counts


def generate_fullspace_schedules(
    case: dict[str, Any], legacy, archived_audit: dict[str, Any]
) -> tuple[dict[str, dict[int, Counter]], dict[str, Any]]:
    observables = case["observables"]
    weights = case["weights"]
    maximum = max(FULLSPACE_BUDGETS)
    core = sorted_insertion_cover(observables, weights, legacy)
    sg_raw = legacy.vectorized_shadow_grouping_schedule(observables, weights, maximum)
    core_hits = legacy.hit_matrix(observables, core).sum(axis=1).astype(np.int64)
    derand_raw, derand_generator_audit = legacy.vectorized_derandomization_schedule(
        observables,
        weights,
        maximum,
        initial_hits=core_hits,
        return_audit=True,
    )
    sg_raw[sg_raw == 0] = 3
    derand_raw[derand_raw == 0] = 3
    bank = unique_settings(itertools.chain(core, sg_raw, derand_raw))
    coverage = sparse_hit_matrix(observables, bank, legacy)
    probabilities, optimizer_audit = optimize_ogm(weights, coverage)
    index = {tuple(map(int, row)): i for i, row in enumerate(bank)}
    core_indices = np.asarray(
        [index[tuple(map(int, row))] for row in core], dtype=np.int64
    )
    output: dict[str, dict[int, Counter]] = {"OGM": {}, "SG": {}, "Derand": {}}
    for total in FULLSPACE_BUDGETS:
        remainder = total - len(core)
        if remainder < 0:
            raise ValueError(f"T={total} is smaller than the common QWC cover")
        for method, raw in (("SG", sg_raw), ("Derand", derand_raw)):
            counter = counter_from_rows(core)
            counter.update(tuple(map(int, row)) for row in raw[:remainder])
            output[method][total] = counter
        counts = largest_remainder_counts(remainder, probabilities)
        counts[core_indices] += 1
        output["OGM"][total] = Counter(
            {
                tuple(map(int, bank[i])): int(count)
                for i, count in enumerate(counts)
                if count
            }
        )
    audit = {
        "common_QWC_cover_size": len(core),
        "SG_raw_unique_at_max_T": len(np.unique(sg_raw, axis=0)),
        "Derand_raw_unique_at_max_T": len(np.unique(derand_raw, axis=0)),
        "Derand_algorithm_id": DERAND_ALGORITHM_ID,
        "Derand_cover_seed_min_hits": int(np.min(core_hits)),
        "Derand_cover_seed_max_hits": int(np.max(core_hits)),
        "Derand_generator_audit": derand_generator_audit.as_dict(),
        "OGM_candidate_bank_size": len(bank),
        "OGM_optimization": optimizer_audit,
    }
    # The cover and SG generator are unchanged and must still replay.  Old
    # Derand/OGM counts are intentionally not accepted after the Huang-2021
    # correction because Derand settings also contribute to the OGM bank.
    for field in ("common_QWC_cover_size", "SG_raw_unique_at_max_T"):
        if int(audit[field]) != int(archived_audit[field]):
            raise RuntimeError(
                f"{case['name']} regenerated {field}={audit[field]} does not match "
                f"the archived value {archived_audit[field]}"
            )
    if archived_audit.get("Derand_algorithm_id") == DERAND_ALGORITHM_ID:
        for field in ("Derand_raw_unique_at_max_T", "OGM_candidate_bank_size"):
            if int(audit[field]) != int(archived_audit[field]):
                raise RuntimeError(
                    f"{case['name']} regenerated {field}={audit[field]} does not match "
                    f"the corrected archived value {archived_audit[field]}"
                )
        archived_optimizer = archived_audit["OGM_optimization"]
        if abs(optimizer_audit["objective"] - float(archived_optimizer["objective"])) > 2.0e-9:
            raise RuntimeError(f"{case['name']} corrected OGM objective does not reproduce")
    return output, audit


def encode_paulis(observables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = observables.shape[1]
    powers = np.asarray([1 << (n - 1 - site) for site in range(n)], dtype=np.uint64)
    x_masks = np.sum(
        (((observables == 1) | (observables == 2)).astype(np.uint64) * powers),
        axis=1,
        dtype=np.uint64,
    )
    z_masks = np.sum(
        (((observables == 2) | (observables == 3)).astype(np.uint64) * powers),
        axis=1,
        dtype=np.uint64,
    )
    return x_masks, z_masks


def all_unique_settings(schedules: dict[str, dict[int, Counter]]) -> list[tuple[int, ...]]:
    return sorted(
        set().union(
            *(set(counter) for method in schedules.values() for counter in method.values())
        )
    )


def setting_hit_cache(
    observables: np.ndarray,
    settings: Iterable[tuple[int, ...]],
    legacy,
) -> dict[tuple[int, ...], np.ndarray]:
    return {
        setting: np.flatnonzero(
            legacy.setting_hits(observables, np.asarray(setting, dtype=np.int8))
        )
        for setting in settings
    }


def collect_moment_keys(
    x_masks: np.ndarray,
    z_masks: np.ndarray,
    hit_cache: dict[tuple[int, ...], np.ndarray],
    n: int,
) -> np.ndarray:
    keys = set(
        map(
            int,
            (x_masks << np.uint64(n)) | z_masks,
        )
    )
    for hit in hit_cache.values():
        for position, left in enumerate(hit):
            right = hit[position:]
            products = (
                ((x_masks[left] ^ x_masks[right]) << np.uint64(n))
                | (z_masks[left] ^ z_masks[right])
            )
            keys.update(map(int, products))
    output = np.fromiter(keys, dtype=np.uint64, count=len(keys))
    output.sort()
    return output


def fwht(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values).copy()
    width = 1
    while width < output.size:
        view = output.reshape(-1, 2, width)
        left = view[:, 0, :].copy()
        right = view[:, 1, :].copy()
        view[:, 0, :] = left + right
        view[:, 1, :] = left - right
        width *= 2
    return output


def evaluate_moments(
    state: np.ndarray,
    sector_indices: np.ndarray,
    keys: np.ndarray,
    n: int,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    start_time = time.time()
    dimension = 1 << n
    if state.size != dimension:
        raise ValueError(f"{label}: state dimension mismatch")
    state = np.asarray(state, dtype=np.complex128)
    state /= np.linalg.norm(state)
    use_real = float(np.max(np.abs(state.imag))) < 2.0e-11
    working_state = state.real if use_real else state
    parity = np.fromiter(
        ((index.bit_count() & 1) for index in range(dimension)),
        dtype=np.int8,
        count=dimension,
    )
    x_values = keys >> np.uint64(n)
    z_values = keys & np.uint64(dimension - 1)
    unique_x, starts, counts = np.unique(
        x_values, return_index=True, return_counts=True
    )
    moments = np.empty(len(keys), dtype=float)
    max_direct_elements = 4_000_000
    fwht_threshold = 16 * dimension
    fwht_groups = 0
    direct_groups = 0
    maximum_imaginary = 0.0
    support = np.asarray(sector_indices, dtype=np.uint64)
    for group_number, (x_value, begin, count) in enumerate(
        zip(unique_x, starts, counts), 1
    ):
        end = int(begin + count)
        z_group = z_values[begin:end]
        targets = support ^ x_value
        base = np.conjugate(working_state[targets]) * working_state[support]
        valid = np.abs(base) > 1.0e-30
        indices = support[valid]
        base = base[valid]
        if len(indices) == 0:
            raw = np.zeros(len(z_group), dtype=np.complex128)
        elif len(indices) * len(z_group) > fwht_threshold:
            vector = np.zeros(dimension, dtype=base.dtype)
            vector[indices] = base
            raw = fwht(vector)[z_group.astype(np.int64)]
            fwht_groups += 1
        else:
            raw = np.empty(len(z_group), dtype=np.result_type(base.dtype, np.float64))
            block = max(1, min(len(z_group), max_direct_elements // len(indices)))
            for first in range(0, len(z_group), block):
                selected = z_group[first : first + block]
                masked = np.bitwise_and(indices[:, None], selected[None, :])
                signs = 1.0 - 2.0 * parity[masked]
                raw[first : first + len(selected)] = base @ signs
            direct_groups += 1
        phases = np.fromiter(
            (1.0j ** (int(x_value & z).bit_count()) for z in z_group),
            dtype=np.complex128,
            count=len(z_group),
        )
        complex_moments = np.asarray(raw, dtype=np.complex128) * phases
        if len(complex_moments):
            maximum_imaginary = max(
                maximum_imaginary, float(np.max(np.abs(complex_moments.imag)))
            )
        moments[begin:end] = complex_moments.real
        if group_number % 500 == 0 or group_number == len(unique_x):
            print(
                f"[{label}] state moments {group_number}/{len(unique_x)} x-masks "
                f"({time.time() - start_time:.1f}s)",
                flush=True,
            )
    if maximum_imaginary > 2.0e-7:
        raise RuntimeError(
            f"{label}: material imaginary Pauli expectation {maximum_imaginary}"
        )
    return moments, {
        "requested_Pauli_moments": len(keys),
        "unique_x_masks": len(unique_x),
        "FWHT_groups": fwht_groups,
        "direct_groups": direct_groups,
        "maximum_discarded_imaginary_expectation": maximum_imaginary,
        "state_treated_as_real_after_global_phase": use_real,
        "elapsed_seconds": time.time() - start_time,
    }


def lookup_moments(
    table_keys: np.ndarray, table_values: np.ndarray, query: np.ndarray
) -> np.ndarray:
    positions = np.searchsorted(table_keys, query)
    in_range = positions < len(table_keys)
    if not np.all(in_range):
        raise KeyError("Moment table is missing a requested Pauli product")
    if np.any(table_keys[positions] != query):
        raise KeyError("Moment table is missing a requested Pauli product")
    return table_values[positions]


def build_covariance(
    case: dict[str, Any],
    x_masks: np.ndarray,
    z_masks: np.ndarray,
    hit_cache: dict[tuple[int, ...], np.ndarray],
    moment_keys: np.ndarray,
    moment_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start_time = time.time()
    n = case["n"]
    observable_keys = (x_masks << np.uint64(n)) | z_masks
    means = lookup_moments(moment_keys, moment_values, observable_keys)
    number_terms = len(means)
    covariance = np.zeros((number_terms, number_terms), dtype=float)
    covariance[np.diag_indices(number_terms)] = np.maximum(1.0 - means**2, 0.0)
    settings = list(hit_cache.items())
    for setting_number, (_, hit) in enumerate(settings, 1):
        for position, left in enumerate(hit[:-1]):
            right = hit[position + 1 :]
            product_keys = (
                ((x_masks[left] ^ x_masks[right]) << np.uint64(n))
                | (z_masks[left] ^ z_masks[right])
            )
            products = lookup_moments(moment_keys, moment_values, product_keys)
            values = products - means[left] * means[right]
            covariance[left, right] = values
            covariance[right, left] = values
        if setting_number % 500 == 0 or setting_number == len(settings):
            print(
                f"[{case['name']}] covariance blocks {setting_number}/{len(settings)} "
                f"({time.time() - start_time:.1f}s)",
                flush=True,
            )
    saved = case.get("saved_term_means")
    saved_error = None
    if saved is not None:
        saved_error = float(np.max(np.abs(means - saved)))
        if saved_error > 5.0e-10:
            raise RuntimeError(
                f"{case['name']}: Pauli means differ from saved means by {saved_error}"
            )
    energy = float(case["offset"] + np.dot(case["weights"], means))
    energy_error = energy - float(case["exact_energy"])
    if abs(energy_error) > 8.0e-7:
        raise RuntimeError(
            f"{case['name']}: Pauli/state energy mismatch {energy_error} Ha"
        )
    return means, covariance, {
        "maximum_saved_term_mean_error": saved_error,
        "reconstructed_energy_hartree": energy,
        "energy_error_hartree": energy_error,
        "elapsed_seconds": time.time() - start_time,
    }


def analyze_fixed_schedule(
    case: dict[str, Any],
    counter: Counter,
    hit_cache: dict[tuple[int, ...], np.ndarray],
    means: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, Any]:
    total = int(sum(counter.values()))
    hit_counts = np.zeros(len(means), dtype=np.int64)
    for setting, count in counter.items():
        hit_counts[hit_cache[setting]] += int(count)
    if np.any(hit_counts <= 0):
        raise RuntimeError(f"{case['name']}: fixed schedule has uncovered terms")
    coefficients = case["weights"] / hit_counts
    fixed_variance = 0.0
    conditional_mean_second = 0.0
    for setting, count in counter.items():
        hit = hit_cache[setting]
        local = coefficients[hit]
        block = covariance[np.ix_(hit, hit)]
        local_variance = float(local @ block @ local)
        if local_variance < -2.0e-9:
            raise RuntimeError(
                f"{case['name']}: negative conditional variance {local_variance}"
            )
        fixed_variance += int(count) * max(local_variance, 0.0)
        iid_conditional_mean = total * float(local @ means[hit])
        conditional_mean_second += int(count) / total * iid_conditional_mean**2
    target_nonidentity_mean = float(np.dot(case["weights"], means))
    between_basis = conditional_mean_second - target_nonidentity_mean**2
    if between_basis < -2.0e-7:
        raise RuntimeError(
            f"{case['name']}: negative iid between-basis variance {between_basis}"
        )
    between_basis = max(between_basis, 0.0)
    normalized_fixed = total * fixed_variance
    return {
        "estimator_variance_hartree2": fixed_variance,
        "T_times_fixed_variance_hartree2": normalized_fixed,
        "iid_between_basis_variance_hartree2": between_basis,
        "iid_empirical_distribution_one_shot_variance_hartree2": (
            normalized_fixed + between_basis
        ),
        "analytic_sampling_SE_hartree": math.sqrt(max(fixed_variance, 0.0)),
        "minimum_term_hits": int(hit_counts.min()),
        "covered_pauli_terms": int(np.count_nonzero(hit_counts)),
        "measurement_settings": len(counter),
    }


def official_pauli_rows() -> dict[tuple[str, str, int], dict[str, str]]:
    rows = read_csv(PAULI_ROOT / "sampling_error_summary.csv")
    output = {
        (row["molecule"], row["method"], int(row["T_total_shots"])): row
        for row in rows
        if row["molecule"] != "H4" and row["method"] in {"OGM", "SG", "Derand"}
    }
    for row in read_csv(H4_PAULI_FIXED_SUMMARY):
        if row["method"] in {"OGM", "SG", "Derand"}:
            output[("H4", row["method"], int(row["T_total_shots"]))] = row
    return output


def validate_h4_pauli_rerun_manifest() -> None:
    payload = json.loads(H4_PAULI_RERUN_MANIFEST.read_text(encoding="utf-8"))
    audit = payload.get("audit", {})
    expected_hash = payload.get("output_hashes", {}).get(
        "fixed_sampling_error_summary.csv"
    )
    if (
        payload.get("definition_version")
        != "h4-gpd-srdd-identical-input-pauli-rerun-v1"
        or not audit.get("all_pauli_terms_covered")
        or not audit.get("all_shot_counts_conserved")
        or expected_hash != sha256_file(H4_PAULI_FIXED_SUMMARY)
    ):
        raise RuntimeError("H4 identical-input Pauli rerun failed provenance checks")


def official_fullspace_rows(molecule: str) -> dict[tuple[str, int], dict[str, str]]:
    rows = read_csv(Path(FULLSPACE_CASES[molecule]["summary"]))
    return {
        (row["method"], int(row["T_total_shots"])): row
        for row in rows
    }


def analyze_pauli_case(
    case: dict[str, Any],
    schedules: dict[str, dict[int, Counter]],
    legacy,
    official_rows: dict[tuple[str, str, int], dict[str, str]] | None,
    fullspace_rows: dict[tuple[str, int], dict[str, str]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.time()
    settings = all_unique_settings(schedules)
    hit_cache = setting_hit_cache(case["observables"], settings, legacy)
    x_masks, z_masks = encode_paulis(case["observables"])
    moment_keys = collect_moment_keys(x_masks, z_masks, hit_cache, case["n"])
    print(
        f"[{case['name']}] {len(settings)} bases, {len(moment_keys)} unique "
        "Pauli moments",
        flush=True,
    )
    moment_values, moment_audit = evaluate_moments(
        case["state"],
        case["sector_indices"],
        moment_keys,
        case["n"],
        case["name"],
    )
    means, covariance, covariance_audit = build_covariance(
        case, x_masks, z_masks, hit_cache, moment_keys, moment_values
    )
    rows: list[dict[str, Any]] = []
    maximum_archived_variance_error = 0.0
    archived_variance_rows_checked = 0
    for method in ("Derand", "OGM", "SG"):
        for total, counter in sorted(schedules[method].items()):
            result = analyze_fixed_schedule(case, counter, hit_cache, means, covariance)
            official = None
            if official_rows is not None:
                official = official_rows[(case["name"], method, total)]
            elif fullspace_rows is not None:
                official = fullspace_rows[(method, total)]
            if official is not None:
                if int(official["T_actual_shots"]) != total:
                    raise RuntimeError(f"{case['name']} {method} T={total}: archived shot mismatch")
                settings_field = (
                    official.get("measurement_settings_or_distinct_bases")
                    or official.get("measurement_settings")
                )
                if settings_field and int(float(settings_field)) != result["measurement_settings"]:
                    raise RuntimeError(
                        f"{case['name']} {method} T={total}: basis-count mismatch"
                    )
                archived_se = (
                    official.get("analytic_sampling_SE_hartree")
                    or official.get("analytic_sampling_SE")
                )
                if archived_se:
                    archived_variance_rows_checked += 1
                    error = abs(result["estimator_variance_hartree2"] - float(archived_se) ** 2)
                    maximum_archived_variance_error = max(
                        maximum_archived_variance_error, error
                    )
                    if error > 2.0e-10:
                        raise RuntimeError(
                            f"{case['name']} {method} T={total}: variance replay "
                            f"error {error}"
                        )
            rows.append(
                {
                    "molecule": case["name"],
                    "geometry_and_representation": GEOMETRIES[case["name"]],
                    "qubits": case["n"],
                    "method": method,
                    "T_total_shots": total,
                    **result,
                    "signed_approximation_bias_hartree": 0.0,
                    "analytic_MSE_hartree2": result["estimator_variance_hartree2"],
                    "variance_protocol": "fixed Pauli schedule, pooled-hit estimator, conditioned on basis counts",
                    "source_status": "exact state-conditioned replay",
                }
            )
    return rows, {
        "molecule": case["name"],
        "qubits": case["n"],
        "nonidentity_Pauli_terms": len(case["weights"]),
        "unique_measurement_bases_across_methods_and_budgets": len(settings),
        "sector_state_leakage_norm_before_projection": case["sector_leakage_norm"],
        "moment_evaluation": moment_audit,
        "covariance_evaluation": covariance_audit,
        "archived_analytic_variance_rows_checked": archived_variance_rows_checked,
        "maximum_archived_variance_replay_error_hartree2": (
            maximum_archived_variance_error
            if archived_variance_rows_checked
            else None
        ),
        "elapsed_seconds": time.time() - started,
    }


def rotated_diagonal_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in active_combined_rows():
        method = {
            "four-family Hybrid-F": "AGPD",
            LEGACY_SRCDF_STORED_METHOD: SRDD_METHOD,
        }.get(row["method"])
        if method is None or row["molecule"] not in {"H4", "H6", "LiH"}:
            continue
        total = int(row["T_total_shots"])
        if not row.get("measurement_settings"):
            raise RuntimeError(
                f"{row['molecule']} {row['method']}: explicit measurement_settings is required"
            )
        settings = int(float(row["measurement_settings"]))
        agpd_metadata: dict[str, Any] | None = None
        if method == SRDD_METHOD:
            settings = validate_combined_srcdf_row(row)
            standard_error = float(row["analytic_sampling_SE"])
            bias = float(row["signed_approximation_bias"])
        else:
            agpd_metadata = validate_combined_agpd_row(row)
            settings = int(agpd_metadata["settings"])
            standard_error = float(agpd_metadata["analytic_sampling_SE"])
            bias = float(agpd_metadata["signed_bias"])
        variance = standard_error**2
        collector_fields = (
            {
                "source_leaves_K": agpd_metadata["source_leaves_K"],
                "extra_one_body_leaves_L": agpd_metadata[
                    "extra_one_body_leaves_L"
                ],
                "collector_mode": "fixed_shallow_one_body_cover",
                "collector_depth": agpd_metadata["cover_depth"],
                "source_to_collector_transfer_count": 0,
                "independent_collector_setting_present": False,
                "complete_setting_shot_vector": " ".join(
                    str(value) for value in agpd_metadata["shot_vector"]
                ),
                "all_K_plus_L_A_settings_included_in_variance": True,
                "selected_sampling_source": agpd_metadata["source"],
                "agpd_artifact_version": agpd_metadata["artifact_version"],
                "centered_range_policy": agpd_metadata["centered_range_policy"],
                "centered_range_domain": agpd_metadata["centered_range_domain"],
                "full_fock_centered_range_sum": agpd_metadata[
                    "full_fock_centered_range_sum"
                ],
                "sector_allocation_centered_range_sum": agpd_metadata[
                    "sector_allocation_centered_range_sum"
                ],
                "sector_range_sum_reduction_fraction": agpd_metadata[
                    "sector_range_sum_reduction_fraction"
                ],
                "maximum_sector_leakage_relative_frobenius": agpd_metadata[
                    "maximum_sector_leakage_relative_frobenius"
                ],
                "sector_leakage_tolerance": agpd_metadata[
                    "sector_leakage_tolerance"
                ],
                "sector_range_setting_count": agpd_metadata[
                    "sector_range_setting_count"
                ],
                "full_fock_fallback_setting_count": agpd_metadata[
                    "full_fock_fallback_setting_count"
                ],
                "sector_operator_norm": agpd_metadata["sector_operator_norm"],
                "fully_corrective_design_rank": agpd_metadata[
                    "fully_corrective_design_rank"
                ],
                "fully_corrective_design_nullity": agpd_metadata[
                    "fully_corrective_design_nullity"
                ],
                "fully_corrective_rcond": agpd_metadata[
                    "fully_corrective_rcond"
                ],
                "selection_rule": agpd_metadata["selection_rule"],
                "selection_uses_exact_ground_state": agpd_metadata[
                    "selection_uses_exact_ground_state"
                ],
            }
            if agpd_metadata is not None
            else {}
        )
        output.append(
            {
                "molecule": row["molecule"],
                "geometry_and_representation": GEOMETRIES[row["molecule"]],
                "qubits": 8,
                "method": method,
                "T_total_shots": total,
                "estimator_variance_hartree2": variance,
                "T_times_fixed_variance_hartree2": total * variance,
                "iid_between_basis_variance_hartree2": "",
                "iid_empirical_distribution_one_shot_variance_hartree2": "",
                "analytic_sampling_SE_hartree": standard_error,
                "minimum_term_hits": "",
                "covered_pauli_terms": "",
                "measurement_settings": settings,
                "signed_approximation_bias_hartree": bias,
                "analytic_MSE_hartree2": variance + bias**2,
                "variance_protocol": (
                    (
                        "independent rotated-diagonal strata over the complete K+L_A "
                        "H4 fully-corrective dense-sector AGPD v10 setting list; "
                        "selection is ground-state-blind and allocation uses the certified "
                        "fixed-particle-number-sector centered range and fails closed "
                        "to the full-Fock range above relative leakage 1e-10"
                    )
                    if method == "AGPD" and row["molecule"] == "H4"
                    else (
                        "independent rotated-diagonal strata over the complete K+L_A "
                        "AGPD v8 setting list with its selected integer T_i"
                    )
                    if method == "AGPD"
                    else "independent rotated-diagonal strata with archived integer T_i"
                ),
                "source_status": row["provenance"],
                **collector_fields,
            }
        )
    for molecule in ("BeH2", "N2"):
        fullspace_audit = json.loads(
            Path(FULLSPACE_CASES[molecule]["audit"]).read_text(encoding="utf-8")
        )
        for row in read_csv(Path(FULLSPACE_CASES[molecule]["summary"])):
            if row["method"] != LEGACY_SRCDF_METHOD:
                continue
            settings = validate_fullspace_srcdf_row(molecule, row, fullspace_audit)
            total = int(row["T_total_shots"])
            standard_error = float(row["analytic_sampling_SE_hartree"])
            variance = standard_error**2
            bias = float(row["deterministic_bias_hartree"])
            output.append(
                {
                    "molecule": molecule,
                    "geometry_and_representation": GEOMETRIES[molecule],
                    "qubits": 14 if molecule == "BeH2" else 20,
                    "method": SRDD_METHOD,
                    "T_total_shots": total,
                    "estimator_variance_hartree2": variance,
                    "T_times_fixed_variance_hartree2": total * variance,
                    "iid_between_basis_variance_hartree2": "",
                    "iid_empirical_distribution_one_shot_variance_hartree2": "",
                    "analytic_sampling_SE_hartree": standard_error,
                    "minimum_term_hits": "",
                    "covered_pauli_terms": "",
                    "measurement_settings": settings,
                    "signed_approximation_bias_hartree": bias,
                    "analytic_MSE_hartree2": variance + bias**2,
                    "variance_protocol": "independent rotated-diagonal strata with archived integer T_i",
                    "source_status": "audited full-space SRDD post-selection analytic variance",
                }
            )
    return output


def t3000_summary(detail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["molecule"], row["method"]): row
        for row in detail
        if int(row["T_total_shots"]) == 3000
    }
    output: list[dict[str, Any]] = []
    for molecule in MOLECULE_ORDER:
        for method in METHOD_ORDER:
            row = lookup.get((molecule, method))
            if row is None:
                if method != "AGPD" or molecule not in {"BeH2", "N2"}:
                    raise RuntimeError(f"Missing T=3000 result for {molecule}/{method}")
                output.append(
                    {
                        "molecule": molecule,
                        "geometry_and_representation": GEOMETRIES[molecule],
                        "method": method,
                        "T_total_shots": 3000,
                        "estimator_variance_hartree2": "",
                        "T_times_fixed_variance_hartree2": "",
                        "iid_empirical_distribution_one_shot_variance_hartree2": "",
                        "signed_approximation_bias_hartree": "",
                        "analytic_MSE_hartree2": "",
                        "availability": "n.a. — full-space AGPD was not evaluated",
                    }
                )
                continue
            output.append(
                {
                    "molecule": molecule,
                    "geometry_and_representation": row["geometry_and_representation"],
                    "method": method,
                    "T_total_shots": 3000,
                    "estimator_variance_hartree2": row["estimator_variance_hartree2"],
                    "T_times_fixed_variance_hartree2": row[
                        "T_times_fixed_variance_hartree2"
                    ],
                    "iid_empirical_distribution_one_shot_variance_hartree2": row[
                        "iid_empirical_distribution_one_shot_variance_hartree2"
                    ],
                    "signed_approximation_bias_hartree": row[
                        "signed_approximation_bias_hartree"
                    ],
                    "analytic_MSE_hartree2": row["analytic_MSE_hartree2"],
                    "availability": "available",
                }
            )
    return output


def format_number(value: Any) -> str:
    if value in (None, ""):
        return "--"
    return f"{float(value):.9g}"


def write_readme(summary: list[dict[str, Any]]) -> None:
    table_rows = []
    lookup = {(row["molecule"], row["method"]): row for row in summary}
    for molecule in MOLECULE_ORDER:
        cells = [molecule]
        for method in METHOD_ORDER:
            cells.append(format_number(lookup[(molecule, method)]["estimator_variance_hartree2"]))
        table_rows.append("| " + " | ".join(cells) + " |")
    normalized_rows = []
    for molecule in MOLECULE_ORDER:
        cells = [molecule]
        for method in METHOD_ORDER:
            cells.append(
                format_number(
                    lookup[(molecule, method)]["T_times_fixed_variance_hartree2"]
                )
            )
        normalized_rows.append("| " + " | ".join(cells) + " |")
    text = f"""# 正文固定构型的态相关测量方差

本目录只使用当前修订稿的五个固定构型：H4 1.20 A、H6 3.40 A、
LiH 1.50 A、BeH2 1.33376 A 和 N2 2.25 A。H4/H6/LiH 是正文的
8-qubit active-space target；BeH2/N2 是 14/20-qubit full-space target。

## 数值口径

下表是总预算 `T=3000` 时，正文实际固定调度估计器的解析态相关方差
`Var_rho(E_hat_T)`，单位 Ha^2。它只含 sampling variance；近似方法的 MSE
还需加保存的 bias squared。

| molecule | Derand | OGM | SG | SRDD | AGPD |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

为便于和随 `1/T` 缩放的 variance coefficient 比较，下面同时给出
`T Var_rho(E_hat_T)`。这是固定分层/固定 schedule 的归一化方差，不应与
OGM iid 单次方差混为一谈。

| molecule | Derand | OGM | SG | SRDD | AGPD |
|---|---:|---:|---:|---:|---:|
{chr(10).join(normalized_rows)}

BeH2/N2 的 full-space AGPD 在当前指数存储 dense 实现中未计算，因此严格写
`n.a.`，没有使用小活性空间结果替代。

## Derand、OGM、SG：正文固定 schedule 公式

写

```math
H=c_0I+\\sum_i a_iQ_i,\\qquad \\mu_i=\\operatorname{{Tr}}(\\rho Q_i).
```

局域 Pauli 基 `P_s` 覆盖 `Q_i` 记作 `f_i(P_s)=1`，并定义

```math
N_i=\\sum_{{s=1}}^T f_i(P_s),\\qquad
N_{{ij}}=\\sum_{{s=1}}^T f_i(P_s)f_j(P_s).
```

正文的 pooled-hit estimator 条件方差为

```math
\\operatorname{{Var}}_\\rho(\\widehat E_T)=
\\sum_{{i,j}}a_i a_j\\frac{{N_{{ij}}}}{{N_iN_j}}
\\left[\\operatorname{{Tr}}(\\rho Q_iQ_j)-\\mu_i\\mu_j\\right].
```

三种方法使用同一方差公式，只是生成的固定 basis schedule 不同。若有未覆盖项，
再加 `|Tr[rho(H_hat-H)]|^2` 才得到 MSE；本表所有 Pauli schedule 均强制完整覆盖。

## OGM 论文的 iid 公式

若每个 basis 是从分布 `p_P` 独立抽取，而不是预先固定每个 basis 的次数，定义

```math
\\chi_i=\\sum_Pp_Pf_i(P),\\qquad
\\chi_{{ij}}=\\sum_Pp_Pf_i(P)f_j(P).
```

则 OGM 单次估计器的态相关方差为

```math
V_{{\\rho,\\mathrm{{iid}}}}=
\\sum_{{i,j}}a_i a_j\\frac{{\\chi_{{ij}}}}{{\\chi_i\\chi_j}}
\\operatorname{{Tr}}(\\rho Q_iQ_j)
-\\left(\\sum_i a_i\\mu_i\\right)^2,
```

`T` 次 iid 平均的方差是 `V_rho,iid/T`。CSV 还给出把实际 basis 频率
`p_P=N_P/T` 代入该式所得的 `iid_empirical_distribution_one_shot_variance`。
它等于 `T Var_fixed` 加一个非负的 between-basis 项；正文固定调度消除了这项，
所以二者不能互换。

## AGPD 与 SRDD

两者都不是 Pauli-product schedule。令选中的 rotated-diagonal settings 为

```math
M_i=U_i^\\dagger\\operatorname{{diag}}[d_i(z)]U_i,
```

在状态 `rho` 上

```math
p_i(z)=\\langle z|U_i\\rho U_i^\\dagger|z\\rangle,\\quad
\\mu_i=\\sum_zp_i(z)d_i(z),\\quad
\\sigma_i^2=\\sum_zp_i(z)d_i(z)^2-\\mu_i^2.
```

固定整数分配 `T_i` 后

```math
\\operatorname{{Var}}_\\rho(\\widehat E_{{\\kappa,T}})
=\\sum_i\\frac{{\\sigma_i^2}}{{T_i}},\\qquad
\\operatorname{{MSE}}=|\\operatorname{{Tr}}[\\rho(\\widehat H_\\kappa-H)]|^2
+\\sum_i\\frac{{\\sigma_i^2}}{{T_i}}.
```

AGPD 与 SRDD 的区别不仅在 `M_i` 的构造，也在本次归档的 range provenance。
H4 AGPD 使用 ground-state-blind fully-corrective dense-sector v10：固定已选
rotation 后，在物理 sector 内以 `rcond=1e-12` 联合重拟合每个 source 的 37 个
density-density 系数；prefix 选择使用 sector residual operator norm 与 range
sampling proxy 的合成目标，不使用 exact ground state。其 range 规则为：只有当
`||(I-Pi) M_i Pi||_F / max(1, ||M_i||_F) <= 1e-10` 时才使用固定粒子数
sector 内的中心半谱宽；否则该 setting fail closed 到 full-Fock 中心半谱宽。
H6/LiH AGPD 保留已审计的 v8 full-Fock range，不能标成 sector-range 数据。
SRDD 使用各自归档的中心半谱宽。所有方法都对其实际 range vector 做
minimum-one largest-remainder 整数分配；量子态只用于事后解析评估，不参与
分解选择或 shot allocation。

## 文件

- `state_dependent_variance_by_budget.csv`：所有正式预算的长表；
- `state_dependent_variance_T3000.csv`：共同最大预算的紧凑表；
- `manifest.json`：输入哈希、schedule 重建审计和数值交叉检查；
- `build_state_dependent_variance.py`：完整重建程序。
"""
    README_MD.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=MOLECULE_ORDER,
        default=list(MOLECULE_ORDER),
        help="Molecular cases to recompute (default: all five)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if set(args.cases) != set(MOLECULE_ORDER):
        raise ValueError("Publication tables require all five --cases")
    HERE.mkdir(parents=True, exist_ok=True)
    legacy = load_legacy_pauli_module()
    validate_h4_pauli_rerun_manifest()
    pauli_official = official_pauli_rows()
    combined_collector_contract = validate_combined_manifest()
    detail_rows = rotated_diagonal_rows()
    case_audits: dict[str, Any] = {}
    schedule_audits: dict[str, Any] = {}
    input_paths: set[Path] = {
        COMBINED_V8,
        COMBINED_V8_MANIFEST,
        COMBINED_H4_V10,
        COMBINED_H4_V10_MANIFEST,
        AGPD_ROOT_V8 / "manifest.json",
        AGPD_ROOT_H4_V10 / "manifest.json",
        PAULI_ROOT / "sampling_error_summary.csv",
        H4_PAULI_FIXED_SUMMARY,
        H4_PAULI_RERUN_MANIFEST,
        LEGACY_PAULI_DRIVER,
    }
    for molecule in ("H4", "H6", "LiH"):
        agpd_root = agpd_root_for(molecule)
        input_paths.add(agpd_root / molecule / "audit.json")
        supplemental_path = (
            agpd_root / molecule / "sampling_equivalence_audit.json"
        )
        if supplemental_path.is_file():
            input_paths.add(supplemental_path)
        input_paths.update(
            agpd_root / molecule / f"budget_T{total}.json"
            for total in ACTIVE_BUDGETS
        )

    for molecule in ("H4", "H6", "LiH"):
        print(f"[{molecule}] loading archived schedules", flush=True)
        case = load_active_case(molecule)
        schedules = load_active_schedules(molecule)
        rows, audit = analyze_pauli_case(
            case, schedules, legacy, pauli_official, None
        )
        detail_rows.extend(rows)
        case_audits[molecule] = audit
        input_paths.update(case["input_paths"])
        schedule_directory = (
            H4_PAULI_FIXED_ROOT / "schedules"
            if molecule == "H4"
            else PAULI_ROOT / molecule / "schedules"
        )
        input_paths.update(schedule_directory.glob("*.npz"))

    for molecule in ("BeH2", "N2"):
        print(f"[{molecule}] regenerating coverage-enforced Pauli schedules", flush=True)
        case = load_fullspace_case(molecule)
        archived_payload = json.loads(
            Path(FULLSPACE_CASES[molecule]["audit"]).read_text(encoding="utf-8")
        )
        schedules, schedule_audit = generate_fullspace_schedules(
            case, legacy, archived_payload["schedule"]
        )
        rows, audit = analyze_pauli_case(
            case,
            schedules,
            legacy,
            None,
            # The pre-correction full-space rows are not a valid replay oracle
            # for the corrected, cover-seeded Huang-2021 schedule.
            None,
        )
        detail_rows.extend(rows)
        case_audits[molecule] = audit
        schedule_audits[molecule] = schedule_audit
        input_paths.update(case["input_paths"])
        input_paths.add(Path(FULLSPACE_CASES[molecule]["summary"]))
        input_paths.add(Path(FULLSPACE_CASES[molecule]["replicates"]))
        input_paths.add(Path(FULLSPACE_CASES[molecule]["audit"]))

    detail_rows.sort(
        key=lambda row: (
            MOLECULE_ORDER.index(row["molecule"]),
            int(row["T_total_shots"]),
            METHOD_ORDER.index(row["method"]),
        )
    )
    summary_rows = t3000_summary(detail_rows)
    write_csv(DETAIL_CSV, detail_rows)
    write_csv(SUMMARY_CSV, summary_rows)
    write_readme(summary_rows)
    output_paths = [DETAIL_CSV, SUMMARY_CSV, README_MD]
    manifest = {
        "status": "PASS",
        "variance_scope": {
            "reported_estimator_variance": (
                "Exact state-conditioned sampling variance of the actual fixed-stratum "
                "or fixed-schedule manuscript estimator"
            ),
            "Pauli_fixed_schedule_formula": (
                "sum_ij ai aj Nij/(Ni Nj) [Tr(rho QiQj)-Tr(rho Qi)Tr(rho Qj)]"
            ),
            "Pauli_iid_reference_formula": (
                "sum_ij ai aj chiij/(chi_i chi_j) Tr(rho QiQj)"
                "-(sum_i ai Tr(rho Qi))^2; i,j exclude the deterministic identity"
            ),
            "rotated_diagonal_formula": "sum_i Var_rho(Y_i)/T_i",
            "MSE_formula": "sampling_variance + signed_approximation_bias^2",
        },
        "molecular_targets": GEOMETRIES,
        "AGPD_availability": {
            "H4": "available",
            "H6": "available",
            "LiH": "available",
            "BeH2": "n.a. — full-space exact dense AGPD not evaluated",
            "N2": "n.a. — full-space exact dense AGPD not evaluated",
        },
        "case_audits": case_audits,
        "fullspace_schedule_regeneration_audits": schedule_audits,
        "combined_shallow_cover_contract": combined_collector_contract,
        "srcdf_shallow_collector_contract": {
            "status": combined_collector_contract["status"],
            "mode": combined_collector_contract["srcdf_mode"],
        },
        "agpd_shallow_collector_contract": {
            "status": combined_collector_contract["status"],
            "mode": combined_collector_contract["agpd_mode"],
            "setting_count_rule": combined_collector_contract[
                "agpd_setting_count_rule"
            ],
            "source_to_collector_transfer_count": 0,
            "independent_collector_setting_present": False,
            "all_cover_and_source_shot_vectors_validated": True,
            "all_K_plus_L_A_settings_included_in_variance": True,
            "artifact_version_by_molecule": combined_collector_contract[
                "agpd_version_by_molecule"
            ],
            "centered_range_policy_by_molecule": combined_collector_contract[
                "agpd_centered_range_policy_by_molecule"
            ],
            "h4_sector_leakage_tolerance": combined_collector_contract[
                "h4_sector_leakage_tolerance"
            ],
            "h4_sector_range_or_full_fock_fallback_validated_per_setting": True,
            "h4_v10_fully_corrective_dense_sector_validated": True,
            "h4_fully_corrective_rcond": AGPD_V10_RCOND,
            "h4_selection_rule": AGPD_V10_SELECTION_RULE,
            "h4_selection_uses_exact_ground_state": False,
            "h6_lih_not_relabelled_as_sector_range": True,
            "case_effective_status": combined_collector_contract[
                "agpd_case_effective_status"
            ],
            "sampling_equivalence_audit_sha256": combined_collector_contract[
                "agpd_sampling_equivalence_audit_sha256"
            ],
        },
        "input_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in sorted(input_paths)
            if path.is_file()
        },
        "output_sha256": {
            str(path.name): sha256_file(path) for path in output_paths
        },
    }
    MANIFEST_JSON.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if manifest["status"] != "PASS":
        raise RuntimeError("Variance audit did not pass")
    print(f"[PASS] wrote {DETAIL_CSV}", flush=True)
    print(f"[PASS] wrote {SUMMARY_CSV}", flush=True)
    print(f"[PASS] wrote {MANIFEST_JSON}", flush=True)


if __name__ == "__main__":
    main()
