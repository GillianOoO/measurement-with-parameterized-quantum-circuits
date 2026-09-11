#!/usr/bin/env python3
"""Build inverse-accuracy and local-depolarizing data for main-text Fig. 3.

The two full-space molecular targets are BeH2 and N2.  The inverse-accuracy
curves freeze every method's T0=3000 estimator and use the common analytic
model RMSE(T)**2=b**2+V/T, where V=T0*Var(T0).  The exact Pauli and FC-IMA
estimators have b=0, whereas SRDD retains its approximation bias.  The noise
calculation fixes the T=3000 selected
   SRDD decomposition and integer shot vector.  Its one-body collector is
   exactly redistributed over K source leaves and, when selected, L additional
   depth-d_R one-body leaves; there is no independently diagonalized collector
   circuit.  Every fermionic Givens element is compiled with the fixed exact
   ``CNOT--single-qubit--CNOT`` definition of an ``XXPlusYY`` gate.  A local
   two-qubit depolarizing channel is inserted after each of those two CNOTs in
   every one of the K+L measurement circuits; state preparation is ideal.
   Pauli-product measurement circuits have no such gates and therefore furnish
   p-independent reference curves.  The overlapping full-commuting iterative
   measurement-allocation (FC-IMA) curves are read from an independent audited
   pipeline.  Its budget curve uses the Nature-2023 pooled estimator of
   Eqs. (14) and (17), evaluated from exact ground-state moments (not a Gaussian
   replay), and its channel is inserted after every compiled logical CX.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ``--assemble-only`` only validates and merges existing machine-readable
# moment tables.  Keep the optional tensor-network dependency out of that
# lightweight path so figure assembly remains usable in a clean analysis
# environment; the full noise recomputation still fails closed without quimb.
ASSEMBLE_ONLY_REQUESTED = "--assemble-only" in sys.argv
if not ASSEMBLE_ONLY_REQUESTED:
    import quimb.tensor as qtn

from cnot_noise_semantics import (
    CHANNELS_PER_GIVENS,
    CNOTS_PER_GIVENS,
    composed_depolarizing_probability,
    fixed_two_cx_givens_unitary,
    fock_gate,
    gate_superoperator,
    unitary_equivalence_error,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MPL_CACHE = ROOT / "tmp" / "matplotlib" / "fig3_extended"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
ADAPTIVE = ROOT / "outputs" / "adaptive_f3_ansatz_pool"
RESOURCE = ROOT / "outputs" / "resource_statistics_main_benchmarks"
VARIANCE = ROOT / "outputs" / "state_dependent_variance_main_benchmarks"
ENGINE_PATH = ADAPTIVE / "compare_h2o_n2_srcdf_pauli_fullspace.py"
SHALLOW_PATH = ADAPTIVE / "shallow_rcdf.py"
COLLECTOR_PATH = ADAPTIVE / "shallow_collector_redistribution.py"
BEH2_ROOT = ADAPTIVE / "beh2_srcdf_pauli_fullspace_k50_v2"
N2_ROOT = ADAPTIVE / "h2o_n2_srcdf_pauli_fullspace_k50_v3" / "N2"
BEH2_SUMMARY = BEH2_ROOT / "sampling_summary_all.csv"
N2_SUMMARY = N2_ROOT / "sampling_summary.csv"
BEH2_CANDIDATES = BEH2_ROOT / "BeH2" / "srcdf_candidate_by_T_K.csv"
N2_CANDIDATES = N2_ROOT / "srcdf_candidate_by_T_K.csv"
BEH2_AUDIT = BEH2_ROOT / "BeH2" / "audit.json"
N2_AUDIT = N2_ROOT / "audit.json"
VARIANCE_T3000 = VARIANCE / "state_dependent_variance_T3000.csv"
RESOURCE_STATS = RESOURCE / "resource_statistics.csv"
OUTPUT_CSV = HERE / "fig3_extended_curves.csv"
MODEL_AUDIT_CSV = HERE / "inverse_accuracy_model_audit.csv"
NOISE_CSV = HERE / "srcdf_local_depolarizing_moments.csv"
AUDIT_JSON = HERE / "audit.json"
ANGLE_NPZ = HERE / "recovered_measurement_circuits.npz"
FC_ROOT = ROOT / "outputs" / "traditional_cm_benchmarks"  # legacy directory name
FC_ERROR_CURVES = FC_ROOT / "error_eval" / "results" / "fc_fig3_ready_curves.csv"
FC_ERROR_SUMMARY = FC_ROOT / "error_eval" / "results" / "summary.csv"
FC_NOISE_CURVES = (
    FC_ROOT / "noise_eval" / "results" / "fc_local_depolarizing_curve.csv"
)

SRDD_METHOD = "SRDD"
LEGACY_SRCDF_METHOD = "s-RCDF"
METHODS = (SRDD_METHOD, "OGM", "SG", "Derand")
MOLECULES = ("BeH2", "N2")
ERROR_GRID = np.geomspace(0.01, 0.5, 121)
# The independent FC-IMA benchmark retains legacy ``traditional_cm_*`` paths
# for artifact compatibility.  It intentionally writes 120 target-error points
# (see traditional_cm_error_eval.py).  Keep its source grid intact rather than
# silently interpolating it onto the 121-point grid used by the other methods.
FC_ERROR_GRID = np.geomspace(0.01, 0.5, 120)
NOISE_GRID = np.linspace(0.0, 0.003, 10, dtype=float)
REPORTING_LIMIT = 1.0e8
T_FIXED = 3000
FC_BUDGETS = np.array([1300, 1600, 2000, 2400, 3000], dtype=int)
FC_METHOD = "FC-IMA"
FC_ALLOCATION = "fc_ima_primary"
@dataclass(frozen=True)
class CompiledGivens:
    """One fixed-template two-CNOT fermionic Givens measurement element.

    The saved ``unitary`` is the exact aggregate of the surrounding ideal
    single-qubit gates and the two CNOTs.  ``propagate_adjoint`` applies the two
    post-CNOT depolarizing channels through their exact covariance composition;
    no arbitrary-two-qubit-gate noise event is used.
    """

    first: int
    second: int
    unitary: np.ndarray
    cnot_count: int
    decomposition_error: float


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(ADAPTIVE) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE))
if ASSEMBLE_ONLY_REQUESTED:
    engine = None
    shallow = None
else:
    engine = load_module(ENGINE_PATH, "fig3_fullspace_engine")
    shallow = load_module(SHALLOW_PATH, "fig3_shallow_rcdf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_summary(path: Path, molecule: str) -> pd.Series:
    frame = pd.read_csv(path)
    selected = frame.loc[
        (frame["molecule"] == molecule)
        & (frame["method"] == LEGACY_SRCDF_METHOD)
        & (frame["T_total_shots"] == T_FIXED)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {molecule} SRDD row at T={T_FIXED}")
    return selected.iloc[0]


def selected_shots(path: Path, summary: pd.Series) -> tuple[np.ndarray, pd.Series]:
    required_summary = {
        "selected_K", "selected_depth", "collector_mode",
        "collector_extra_leaves", "measurement_settings",
        "collector_matrix_relative_reconstruction_residual",
    }
    if not required_summary.issubset(summary.index):
        missing = sorted(required_summary - set(summary.index))
        raise RuntimeError(f"Saved SRDD summary lacks shallow-collector fields: {missing}")
    selected_k = int(summary["selected_K"])
    selected_depth = int(summary["selected_depth"])
    collector_mode = str(summary["collector_mode"])
    extra_leaves = int(summary["collector_extra_leaves"])
    setting_count = int(summary["measurement_settings"])
    if collector_mode != "augment":
        raise RuntimeError(
            "Publication replay requires the selected local shallow-collector "
            f"augmentation, got {collector_mode!r}"
        )
    if extra_leaves < 1:
        raise RuntimeError("Publication shallow-collector augmentation requires L>=1")
    if setting_count != selected_k + extra_leaves:
        raise RuntimeError("Saved SRDD setting count is not K+L")

    frame = pd.read_csv(path)
    required_candidate = {
        "T_total_shots", "K", "depth", "collector_mode",
        "collector_extra_leaves", "measurement_settings", "shot_vector",
        "collector_matrix_relative_reconstruction_residual",
    }
    if not required_candidate.issubset(frame.columns):
        missing = sorted(required_candidate - set(frame.columns))
        raise RuntimeError(f"Saved candidate table lacks shallow-collector fields: {missing}")
    rows = frame.loc[
        (frame["T_total_shots"] == T_FIXED)
        & (frame["K"] == selected_k)
        & (frame["depth"] == selected_depth)
        & (frame["collector_mode"] == collector_mode)
        & (frame["collector_extra_leaves"] == extra_leaves)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one candidate for T={T_FIXED}, K={selected_k}, "
            f"dR={selected_depth}, mode={collector_mode}, L={extra_leaves}"
        )
    row = rows.iloc[0]
    if int(row["measurement_settings"]) != setting_count:
        raise RuntimeError("Summary/candidate shallow setting counts disagree")
    for label, residual in (
        ("summary", summary["collector_matrix_relative_reconstruction_residual"]),
        ("candidate", row["collector_matrix_relative_reconstruction_residual"]),
    ):
        if float(residual) > 5.0e-10:
            raise RuntimeError(f"Stored {label} collector reconstruction failed")
    shots = np.fromstring(str(row["shot_vector"]), dtype=int, sep=" ")
    if int(shots.sum()) != T_FIXED or len(shots) != setting_count or np.any(shots < 0):
        raise RuntimeError("Stored SRDD shot vector is inconsistent")
    return shots, row


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def fc_error_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and fail-closed validate the Nature-2023 pooled FC-IMA curves."""
    frame = pd.read_csv(FC_ERROR_CURVES)
    required = {
        "molecule", "curve", "method", "x", "y", "raw_y",
        "right_censored", "model", "T_fixed",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"FC curve table has unexpected columns: {list(frame.columns)}")
    if set(frame["molecule"]) != set(MOLECULES) or set(frame["method"]) != {FC_METHOD}:
        raise RuntimeError(
            "FC-IMA curve table must contain only BeH2/N2 overlapping "
            "full-commuting rows"
        )
    if frame["model"].astype(str).str.contains("Gaussian", case=False).any():
        raise RuntimeError(
            "Gaussian replay rows are not admissible in the main-text FC-IMA curves"
        )

    summary = pd.read_csv(FC_ERROR_SUMMARY)
    if set(summary["molecule"]) != set(MOLECULES) or len(summary) != len(MOLECULES):
        raise RuntimeError("FC-IMA error summary must contain exactly one row per molecule")
    summary_required = {
        "molecule",
        "fc_overlapping_group_count",
        "grouping",
        "estimator",
        "allocation_optimization",
        "fc_primary_analytic_rmse_hartree",
        "inverse_error_frozen_reference_T0_shots",
        "inverse_error_frozen_T0_variance_hartree2",
        "inverse_error_frozen_variance_coefficient_T0_times_variance_hartree2_shots",
    }
    if not summary_required.issubset(summary.columns):
        missing = sorted(summary_required - set(summary.columns))
        raise RuntimeError(f"FC-IMA summary lacks pooled-estimator fields: {missing}")

    budget_rows: list[dict[str, Any]] = []
    inverse_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for molecule in MOLECULES:
        record = summary.loc[summary["molecule"] == molecule]
        if len(record) != 1:
            raise RuntimeError(f"Expected one FC-IMA error summary row for {molecule}")
        record = record.iloc[0]
        group_count = int(record["fc_overlapping_group_count"])
        if group_count <= 0:
            raise RuntimeError(f"Invalid overlapping FC-IMA group count for {molecule}")
        if "extended" not in str(record["grouping"]).lower() or "overlap" not in str(
            record["grouping"]
        ).lower():
            raise RuntimeError(f"FC-IMA grouping is not extended/overlapping for {molecule}")
        if "pooled" not in str(record["estimator"]).lower() or "eq. (14)" not in str(
            record["estimator"]
        ).lower():
            raise RuntimeError(f"FC-IMA summary lacks the Eq. (14) pooled estimator for {molecule}")
        if "ima" not in str(record["allocation_optimization"]).lower():
            raise RuntimeError(f"FC-IMA summary lacks iterative allocation for {molecule}")

        selected_budget = frame.loc[
            (frame["molecule"] == molecule)
            & (frame["curve"] == "error_vs_measurements")
        ].sort_values("x")
        if not np.array_equal(selected_budget["x"].to_numpy(dtype=int), FC_BUDGETS):
            raise RuntimeError(f"Unexpected FC-IMA budget grid for {molecule}")
        budget_models = selected_budget["model"].astype(str)
        if not (
            budget_models.str.contains("Nature-2023", case=False).all()
            and budget_models.str.contains("IMA", case=False).all()
            and budget_models.str.contains(r"Eq\. \(17\)", case=False, regex=True).all()
        ):
            raise RuntimeError(f"FC-IMA budget curve is not the pooled Eq. (17) model for {molecule}")
        expected_t0 = float(record["fc_primary_analytic_rmse_hartree"])
        actual_t0 = float(selected_budget.loc[selected_budget["x"] == T_FIXED, "y"].iloc[0])
        if not math.isclose(actual_t0, expected_t0, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(f"FC-IMA T=3000 RMSE mismatch for {molecule}")

        selected_inverse = frame.loc[
            (frame["molecule"] == molecule)
            & (frame["curve"] == "estimated_measurements_vs_error")
        ].sort_values("x")
        if len(selected_inverse) != len(FC_ERROR_GRID) or not np.allclose(
            selected_inverse["x"].to_numpy(dtype=float), FC_ERROR_GRID, rtol=1.0e-12, atol=0.0
        ):
            raise RuntimeError(f"Unexpected FC-IMA inverse-error grid for {molecule}")
        raw_inverse = selected_inverse["raw_y"].to_numpy(dtype=float)
        plotted_inverse = selected_inverse["y"].to_numpy(dtype=float)
        reference_t0 = int(record["inverse_error_frozen_reference_T0_shots"])
        reference_variance = float(record["inverse_error_frozen_T0_variance_hartree2"])
        variance_coefficient = float(
            record[
                "inverse_error_frozen_variance_coefficient_T0_times_variance_hartree2_shots"
            ]
        )
        if reference_t0 != T_FIXED:
            raise RuntimeError(f"Unexpected FC-IMA inverse reference budget for {molecule}")
        if not math.isclose(expected_t0**2, reference_variance, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise RuntimeError(f"FC-IMA T0 RMSE/variance mismatch for {molecule}")
        if not math.isclose(
            variance_coefficient,
            reference_t0 * reference_variance,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(f"FC-IMA frozen variance coefficient mismatch for {molecule}")
        expected_raw_inverse = variance_coefficient / FC_ERROR_GRID**2
        if not np.allclose(
            raw_inverse, expected_raw_inverse, rtol=1.0e-12, atol=1.0e-12
        ):
            raise RuntimeError(f"FC-IMA inverse curve is not V/epsilon^2 for {molecule}")
        expected_inverse = np.maximum(raw_inverse, group_count)
        if np.any(raw_inverse <= 0.0) or not np.allclose(
            plotted_inverse, expected_inverse, rtol=1.0e-12, atol=0.0
        ):
            raise RuntimeError(
                f"FC-IMA inverse curve violates its executable overlapping-group "
                f"floor for {molecule}"
            )

        selected_inverse = selected_inverse.copy()
        selected_inverse.loc[:, "model"] = (
            "frozen T0=3000 analytic bias-variance: RMSE^2=b^2+V/T"
        )
        selected_inverse.loc[:, "T_fixed"] = T_FIXED

        budget_rows.extend(selected_budget.to_dict(orient="records"))
        inverse_rows.extend(selected_inverse.to_dict(orient="records"))
        audits.append({
            "molecule": molecule,
            "method": FC_METHOD,
            "model": "frozen T0=3000 analytic bias-variance: RMSE^2=b^2+V/T",
            "reference_T0_shots": reference_t0,
            "reference_variance_hartree2": reference_variance,
            "variance_coefficient": variance_coefficient,
            "signed_bias_hartree": 0.0,
            "sampling_scaling_exponent": 0.5,
            "minimum_executable_measurements": group_count,
            "grouping": str(record["grouping"]),
            "estimator": str(record["estimator"]),
            "allocation_optimization": str(record["allocation_optimization"]),
        })
    return budget_rows, inverse_rows, audits


def fc_noise_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the compiled-CX pooled FC-IMA noise calculation, with strict checks."""
    frame = pd.read_csv(FC_NOISE_CURVES)
    required = {
        "molecule", "method", "depolarizing_rate", "T_total_shots",
        "T_actual_shots", "group_count", "error_hartree", "noise_model",
        "channel", "state_preparation_noise_included",
        "one_qubit_gate_noise_included", "readout_noise_included",
        "decomposition_and_shot_vector_frozen", "allocation",
        "sampling_estimator",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"FC-IMA noise table has unexpected columns: {list(frame.columns)}")
    if len(frame) != len(MOLECULES) * len(NOISE_GRID):
        expected = len(MOLECULES) * len(NOISE_GRID)
        raise RuntimeError(
            f"FC-IMA noise table must contain exactly {expected} molecule/rate rows"
        )

    curves: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    error_summary = pd.read_csv(FC_ERROR_SUMMARY)
    expected_t0 = {
        row["molecule"]: float(row["fc_primary_analytic_rmse_hartree"])
        for row in error_summary.to_dict(orient="records")
    }
    expected_group_count = {
        row["molecule"]: int(row["fc_overlapping_group_count"])
        for row in error_summary.to_dict(orient="records")
    }
    for molecule in MOLECULES:
        selected = frame.loc[frame["molecule"] == molecule].sort_values("depolarizing_rate")
        if len(selected) != len(NOISE_GRID) or set(selected["method"]) != {FC_METHOD}:
            raise RuntimeError(f"Incomplete FC-IMA noise rows for {molecule}")
        if not np.allclose(
            selected["depolarizing_rate"].to_numpy(dtype=float), NOISE_GRID,
            rtol=0.0, atol=1.0e-15,
        ):
            raise RuntimeError(f"Unexpected FC-IMA noise grid for {molecule}")
        if set(selected["T_total_shots"].astype(int)) != {T_FIXED} or set(
            selected["T_actual_shots"].astype(int)
        ) != {T_FIXED}:
            raise RuntimeError(f"FC-IMA noise shot count is not frozen at T=3000 for {molecule}")
        if set(selected["group_count"].astype(int)) != {expected_group_count[molecule]}:
            raise RuntimeError(f"FC-IMA overlapping group count mismatch for {molecule}")
        if set(selected["allocation"].astype(str)) != {FC_ALLOCATION}:
            raise RuntimeError(f"FC-IMA noise allocation mismatch for {molecule}")
        estimator_labels = selected["sampling_estimator"].astype(str)
        if not (
            estimator_labels.str.contains(r"Eq\. \(14\)", regex=True).all()
            and estimator_labels.str.contains(r"Eq\. \(17\)", regex=True).all()
            and estimator_labels.str.contains("pooled", case=False).all()
        ):
            raise RuntimeError(f"FC-IMA noise rows do not use the pooled estimator for {molecule}")
        if not all(parse_bool(value) for value in selected["decomposition_and_shot_vector_frozen"]):
            raise RuntimeError(f"FC-IMA decomposition/shot vector is not frozen for {molecule}")
        for column in (
            "state_preparation_noise_included",
            "one_qubit_gate_noise_included",
            "readout_noise_included",
        ):
            if any(parse_bool(value) for value in selected[column]):
                raise RuntimeError(f"Unexpected {column} in FC-IMA noise table")
        p0 = float(selected.loc[selected["depolarizing_rate"] == 0.0, "error_hartree"].iloc[0])
        if not math.isclose(p0, expected_t0[molecule], rel_tol=0.0, abs_tol=1.0e-11):
            raise RuntimeError(f"FC-IMA p=0 noise regression failed for {molecule}")
        for row in selected.to_dict(orient="records"):
            curves.append({
                "molecule": molecule,
                "curve": "error_vs_depolarizing_rate",
                "method": FC_METHOD,
                "x": float(row["depolarizing_rate"]),
                "y": float(row["error_hartree"]),
                "raw_y": float(row["error_hartree"]),
                "right_censored": False,
                "model": (
                    "Nature-2023 Eq. (14) pooled FC-IMA under a local two-qubit "
                    "depolarizing channel after every compiled logical CX"
                ),
                "T_fixed": T_FIXED,
            })
        audits.append({
            "molecule": molecule,
            "method": FC_METHOD,
            "allocation": FC_ALLOCATION,
            "group_count": expected_group_count[molecule],
            "p0_error_hartree": p0,
            "pmax_error_hartree": float(selected.iloc[-1]["error_hartree"]),
            "noise_model": str(selected.iloc[0]["noise_model"]),
            "channel": str(selected.iloc[0]["channel"]),
        })
    return curves, audits


def validated_srcdf_noise_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Fail closed unless the saved SRDD table uses the publication grid."""
    required = {
        "molecule", "p", "T_total_shots", "signed_bias_hartree",
        "sampling_variance_hartree2", "error_hartree", "selected_K",
        "selected_L", "selected_depth", "collector_mode", "measurement_settings",
        "maximum_logical_cx_depth", "independent_collector_setting_present",
        "deterministic_constant_added_once", "shot_vector",
        "collector_matrix_relative_reconstruction_residual",
        "two_qubit_givens_gates_each", "compiled_cnot_gates_each",
        "depolarizing_channels_each", "channels_per_givens",
        "same_support_channel_composition_exact",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(
            f"SRDD noise table has unexpected columns: {list(frame.columns)}"
        )
    expected = len(MOLECULES) * len(NOISE_GRID)
    if len(frame) != expected or frame.duplicated(["molecule", "p"]).any():
        raise RuntimeError(
            f"SRDD noise table must contain exactly {expected} unique molecule/rate rows"
        )
    ordered: list[pd.DataFrame] = []
    for molecule in MOLECULES:
        selected = frame.loc[frame["molecule"] == molecule].sort_values("p")
        if len(selected) != len(NOISE_GRID):
            raise RuntimeError(f"Incomplete SRDD noise rows for {molecule}")
        if not np.allclose(
            selected["p"].to_numpy(dtype=float), NOISE_GRID,
            rtol=0.0, atol=1.0e-15,
        ):
            raise RuntimeError(f"Unexpected SRDD noise grid for {molecule}")
        if set(selected["T_total_shots"].astype(int)) != {T_FIXED}:
            raise RuntimeError(
                f"SRDD noise shot count is not frozen at T={T_FIXED} for {molecule}"
            )
        modes = set(selected["collector_mode"].astype(str))
        if modes != {"augment"}:
            raise RuntimeError(
                f"Publication Fig. 3 requires local augment-only collector rows for {molecule}"
            )
        for row in selected.itertuples(index=False):
            k = int(row.selected_K)
            extra = int(row.selected_L)
            mode = str(row.collector_mode)
            settings = int(row.measurement_settings)
            depth = int(row.selected_depth)
            shots = np.fromstring(str(row.shot_vector), dtype=int, sep=" ")
            if settings != k + extra or len(shots) != settings or int(shots.sum()) != T_FIXED:
                raise RuntimeError(f"Saved shallow setting/shot conservation failed for {molecule}")
            if mode != "augment" or extra < 1:
                raise RuntimeError(f"Saved collector mode/L mismatch for {molecule}")
            givens = int(row.two_qubit_givens_gates_each)
            cnot_count = int(row.compiled_cnot_gates_each)
            channels = int(row.depolarizing_channels_each)
            if (
                int(row.channels_per_givens) != CHANNELS_PER_GIVENS
                or cnot_count != CNOTS_PER_GIVENS * givens
                or channels != cnot_count
                or not parse_bool(row.same_support_channel_composition_exact)
            ):
                raise RuntimeError(f"Saved per-CNOT channel audit failed for {molecule}")
            if int(row.maximum_logical_cx_depth) != 4 * depth:
                raise RuntimeError(f"Saved shallow depth audit failed for {molecule}")
            if parse_bool(row.independent_collector_setting_present):
                raise RuntimeError(f"Independent collector found in saved rows for {molecule}")
            if not parse_bool(row.deterministic_constant_added_once):
                raise RuntimeError(f"Constant-offset audit failed for {molecule}")
            if float(row.collector_matrix_relative_reconstruction_residual) > 5.0e-10:
                raise RuntimeError(f"Collector reconstruction audit failed for {molecule}")
        ordered.append(selected)
    return pd.concat(ordered, ignore_index=True).to_dict(orient="records")


def validate_recovered_circuit_archive(frame: pd.DataFrame) -> None:
    """Reject stale collector or pre-CNOT-compilation archives on assembly."""
    if not ANGLE_NPZ.exists():
        raise FileNotFoundError(ANGLE_NPZ)
    with np.load(ANGLE_NPZ, allow_pickle=False) as archive:
        names = set(archive.files)
        if any("collector_equivalent" in name or "_leaf_angles" in name for name in names):
            raise RuntimeError("Recovered circuit archive contains legacy collector branches")
        for molecule in MOLECULES:
            selected = frame.loc[frame["molecule"] == molecule].iloc[0]
            settings = int(selected["measurement_settings"])
            expected = {
                f"{molecule}_setting_angles",
                f"{molecule}_setting_rotation_replay_errors",
                f"{molecule}_shot_vector",
                f"{molecule}_setting_cnot_counts",
                f"{molecule}_setting_givens_decomposition_errors",
            }
            if not expected.issubset(names):
                raise RuntimeError(f"Recovered circuit archive is incomplete for {molecule}")
            if (
                archive[f"{molecule}_setting_angles"].shape[0] != settings
                or len(archive[f"{molecule}_setting_rotation_replay_errors"]) != settings
                or len(archive[f"{molecule}_shot_vector"]) != settings
                or int(archive[f"{molecule}_shot_vector"].sum()) != T_FIXED
                or len(archive[f"{molecule}_setting_cnot_counts"]) != settings
                or not np.all(
                    archive[f"{molecule}_setting_cnot_counts"]
                    == int(selected["compiled_cnot_gates_each"])
                )
                or len(
                    archive[f"{molecule}_setting_givens_decomposition_errors"]
                )
                != settings
                or float(
                    np.max(
                        archive[
                            f"{molecule}_setting_givens_decomposition_errors"
                        ]
                    )
                )
                > 2.0e-12
            ):
                raise RuntimeError(f"Recovered circuit archive has stale dimensions for {molecule}")


def inverse_accuracy_rows(
    fc_inverse: list[dict[str, Any]], fc_audits: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats = pd.read_csv(RESOURCE_STATS)
    variance = pd.read_csv(VARIANCE_T3000)
    curves: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for molecule in MOLECULES:
        benchmark_prefix = molecule + ","
        subset = stats.loc[stats["benchmark"].str.startswith(benchmark_prefix)]
        if subset.empty:
            raise RuntimeError(f"No resource rows for {molecule}")
        for method in METHODS:
            row = subset.loc[subset["method"] == method]
            if len(row) != 1:
                raise RuntimeError(f"Expected one resource row for {molecule}/{method}")
            record = row.iloc[0]
            variance_row = variance.loc[
                (variance["molecule"] == molecule)
                & (variance["method"] == method)
                & (variance["T_total_shots"] == T_FIXED)
            ]
            if len(variance_row) != 1:
                raise RuntimeError(
                    f"Expected one T0 variance row for {molecule}/{method}"
                )
            variance_record = variance_row.iloc[0]
            if str(variance_record.get("availability", "available")) != "available":
                raise RuntimeError(f"T0 variance is unavailable for {molecule}/{method}")
            reference_variance = float(
                variance_record["estimator_variance_hartree2"]
            )
            variance_coefficient = float(
                variance_record["T_times_fixed_variance_hartree2"]
            )
            bias = float(variance_record["signed_approximation_bias_hartree"])
            analytic_mse = float(variance_record["analytic_MSE_hartree2"])
            if not np.isfinite(reference_variance) or reference_variance <= 0.0:
                raise RuntimeError(f"Invalid T0 variance for {molecule}/{method}")
            if not math.isclose(
                variance_coefficient,
                T_FIXED * reference_variance,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(f"Invalid frozen coefficient for {molecule}/{method}")
            if not math.isclose(
                analytic_mse,
                reference_variance + bias**2,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(f"Invalid T0 analytic MSE for {molecule}/{method}")
            if method in {"OGM", "SG", "Derand"} and not math.isclose(
                bias, 0.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise RuntimeError(f"Exact Pauli method has nonzero bias: {molecule}/{method}")
            if int(record["reference_total_shots"]) != T_FIXED:
                raise RuntimeError(f"Unexpected resource reference budget for {molecule}/{method}")
            if not math.isclose(
                float(record["variance_coefficient"]),
                variance_coefficient,
                rel_tol=1.0e-11,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(f"Resource variance mismatch for {molecule}/{method}")
            if not math.isclose(
                float(record["signed_bias"]), bias, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise RuntimeError(f"Resource bias mismatch for {molecule}/{method}")
            minimum = int(record["decomposition_items"])
            # Every redistributed SRDD setting is represented by
            # decomposition_items; the Pauli methods likewise retain every
            # nonzero setting or distinct basis in the frozen T0 design.
            denominator = ERROR_GRID**2 - bias**2
            raw = np.full_like(ERROR_GRID, np.inf, dtype=float)
            valid = denominator > 0.0
            raw[valid] = variance_coefficient / denominator[valid]
            model = "frozen T0=3000 analytic bias-variance: RMSE^2=b^2+V/T"
            audits.append({
                "molecule": molecule,
                "method": method,
                "model": model,
                "reference_T0_shots": T_FIXED,
                "reference_variance_hartree2": reference_variance,
                "variance_coefficient": variance_coefficient,
                "signed_bias_hartree": bias,
                "sampling_scaling_exponent": 0.5,
                "minimum_executable_measurements": minimum,
            })
            values = np.maximum(raw, minimum)
            if np.any(np.diff(values) > 1.0e-10 * np.maximum(values[:-1], 1.0)):
                raise RuntimeError(f"Inverse curve is not monotone for {molecule}/{method}")
            for error, value in zip(ERROR_GRID, values):
                curves.append({
                    "molecule": molecule,
                    "curve": "estimated_measurements_vs_error",
                    "method": method,
                    "x": float(error),
                    "y": float(min(value, REPORTING_LIMIT)),
                    "raw_y": float(value),
                    "right_censored": bool(value > REPORTING_LIMIT),
                    "model": model,
                    "T_fixed": T_FIXED,
                })
    curves.extend(fc_inverse)
    audits.extend(fc_audits)
    return curves, audits


def recover_shallow_angles(rotation: np.ndarray, depth: int, key: str) -> tuple[np.ndarray, float, int]:
    m = rotation.shape[0]
    count = depth * (m - 1)

    def residual(values: np.ndarray) -> np.ndarray:
        current, _ = shallow.rotation_and_derivatives(values, m, depth)
        return (current - rotation).reshape(-1)

    def jacobian(values: np.ndarray) -> np.ndarray:
        _, derivatives = shallow.rotation_and_derivatives(values, m, depth)
        return derivatives.reshape(count, -1).T

    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    starts = [np.zeros(count)]
    starts.extend(rng.uniform(-math.pi, math.pi, count) for _ in range(7))
    best = None
    for start_index, start in enumerate(starts, start=1):
        fit = least_squares(
            residual,
            start,
            jac=jacobian,
            method="trf",
            max_nfev=2500,
            ftol=1.0e-13,
            xtol=1.0e-13,
            gtol=1.0e-13,
        )
        error = float(np.linalg.norm(residual(fit.x)))
        if best is None or error < best[0]:
            best = (error, np.asarray(fit.x), start_index)
        if error < 5.0e-12:
            break
    assert best is not None
    if best[0] > 5.0e-10:
        raise RuntimeError(f"Failed to reconstruct {key}: ||dR||_F={best[0]:.3e}")
    return best[1], best[0], best[2]


def quadratic_diagonal_mpo(
    constant: float,
    linear: np.ndarray,
    quadratic: np.ndarray,
) -> qtn.MatrixProductOperator:
    """Exact MPO for c + sum_i a_i n_i + sum_{i<j} b_ij n_i n_j."""
    linear = np.asarray(linear, dtype=float)
    quadratic = np.asarray(quadratic, dtype=float)
    n = len(linear)
    dimension = n + 2
    accumulator = n + 1
    initial = np.zeros(dimension)
    initial[0] = 1.0
    initial[accumulator] = float(constant)
    final = np.zeros(dimension)
    final[accumulator] = 1.0
    local: list[np.ndarray] = []
    for site in range(n):
        pair = np.zeros((2, dimension, dimension), dtype=np.complex128)
        for bit in (0, 1):
            transition = np.zeros((dimension, dimension), dtype=np.complex128)
            transition[0, 0] = 1.0
            transition[accumulator, accumulator] = 1.0
            for past in range(site):
                transition[past + 1, past + 1] = 1.0
            if bit:
                transition[site + 1, 0] = 1.0
                transition[accumulator, 0] += linear[site]
                for past in range(site):
                    transition[accumulator, past + 1] += quadratic[past, site]
            pair[bit] = transition.T
        local.append(pair)
    arrays = []
    for site, pair in enumerate(local):
        if n == 1:
            array = np.zeros((2, 2), dtype=np.complex128)
            for bit in (0, 1):
                array[bit, bit] = initial @ pair[bit] @ final
        elif site == 0:
            array = np.zeros((dimension, 2, 2), dtype=np.complex128)
            for bit in (0, 1):
                array[:, bit, bit] = initial @ pair[bit]
        elif site == n - 1:
            array = np.zeros((dimension, 2, 2), dtype=np.complex128)
            for bit in (0, 1):
                array[:, bit, bit] = pair[bit] @ final
        else:
            array = np.zeros((dimension, dimension, 2, 2), dtype=np.complex128)
            for bit in (0, 1):
                array[:, :, bit, bit] = pair[bit]
        arrays.append(array)
    return qtn.MatrixProductOperator(arrays, shape="lrud")


def mpo_to_operator_mps(operator: qtn.MatrixProductOperator) -> qtn.MatrixProductState:
    arrays = []
    for site, array in enumerate(operator.arrays):
        if array.ndim == 3:
            if site == 0:
                arrays.append(array.reshape(array.shape[0], 4))
            else:
                arrays.append(array.reshape(array.shape[0], 4))
        else:
            arrays.append(array.reshape(array.shape[0], array.shape[1], 4))
    return qtn.MatrixProductState(arrays, shape="lrp")


def operator_mps_to_mpo(vector: qtn.MatrixProductState) -> qtn.MatrixProductOperator:
    arrays = []
    for site, array in enumerate(vector.arrays):
        if array.ndim == 2:
            arrays.append(array.reshape(array.shape[0], 2, 2))
        else:
            arrays.append(array.reshape(array.shape[0], array.shape[1], 2, 2))
    return qtn.MatrixProductOperator(arrays, shape="lrud")


def propagate_adjoint(
    operator: qtn.MatrixProductOperator,
    gates: list[CompiledGivens],
    p: float,
    *,
    cutoff: float,
    max_bond: int,
) -> qtn.MatrixProductOperator:
    vector = mpo_to_operator_mps(operator)
    for gate in reversed(gates):
        # Each fixed Givens definition is L2 CX L1 CX L0 on one two-qubit
        # support.  The full-support depolarizing channel commutes with every
        # ideal unitary on that same support.  Therefore the two channels after
        # the two CNOTs compose *exactly* to p_eff=1-(1-p)^2 after the aggregate
        # Givens unitary.  This is an algebraic acceleration, not a different
        # arbitrary-two-qubit-gate noise model.
        effective_p = composed_depolarizing_probability(p, gate.cnot_count)
        superoperator = gate_superoperator(gate.unitary, effective_p)
        vector.gate_(
            superoperator,
            where=(gate.first, gate.second),
            contract="split",
            cutoff=cutoff,
            max_bond=max_bond,
        )
    return operator_mps_to_mpo(vector)


def expectation(state: qtn.MatrixProductState, operator: qtn.MatrixProductOperator) -> float:
    value = complex(qtn.expec_TN_1D(state.H, operator, state, compress=None))
    if abs(value.imag) > 2.0e-8:
        raise RuntimeError(f"Nonreal expectation {value}")
    return float(value.real)


def setting_polynomial(
    case: dict[str, Any],
    setting: dict[str, Any],
    tensors: np.ndarray,
    alpha: np.ndarray,
    index: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the spin-orbital diagonal polynomial for one shallow setting.

    Source settings contain their density--density polynomial plus their share
    of the redistributed one-body collector.  Extra settings contain only the
    redistributed one-body polynomial.  The molecular scalar is deterministic
    and is attached to setting zero exactly once so that both O and O^2 follow
    the same noisy propagation path.
    """
    m = case["m"]
    n = case["n"]
    linear = np.zeros(n)
    quadratic = np.zeros((n, n))
    constant = float(case["constant"]) if index == 0 else 0.0
    family = str(setting.get("family", ""))
    if family not in {"shallow_rcdf_leaf", "shallow_one_body_leaf"}:
        raise RuntimeError(f"Unexpected publication SRDD setting family: {family!r}")
    coefficients = np.asarray(
        setting.get("collector_linear_coefficients"), dtype=float
    )
    if coefficients.shape != (m,) or not np.all(np.isfinite(coefficients)):
        raise RuntimeError("Missing or invalid redistributed collector coefficients")
    linear[:m] += coefficients
    linear[m:] += coefficients

    source_index = setting.get("source_index")
    if family == "shallow_one_body_leaf":
        if source_index is not None:
            raise RuntimeError("Extra one-body leaf unexpectedly references a source tensor")
        return constant, linear, quadratic
    if source_index is None:
        raise RuntimeError("Source SRDD leaf is missing source_index")
    source_index = int(source_index)
    if not 0 <= source_index < len(tensors):
        raise RuntimeError("Source SRDD leaf index is out of range")
    z = np.asarray(tensors[source_index], dtype=float)
    direction = z.sum(axis=1) - 0.5 * np.diag(z)
    for orbital in range(m):
        base_linear = -alpha[source_index] * direction[orbital]
        linear[orbital] += base_linear
        linear[m + orbital] += base_linear
        i, j = orbital, m + orbital
        quadratic[min(i, j), max(i, j)] += z[orbital, orbital]
    for first in range(m):
        for second in range(first + 1, m):
            coefficient = z[first, second]
            for i in (first, m + first):
                for j in (second, m + second):
                    quadratic[min(i, j), max(i, j)] += coefficient
    return constant, linear, quadratic


def diagonal_polynomial_values(
    constant: float,
    linear: np.ndarray,
    quadratic: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    """Evaluate the stored upper-triangular occupation polynomial."""
    values = float(constant) + occupations @ np.asarray(linear, dtype=float)
    first, second = np.triu_indices(occupations.shape[1])
    coefficients = np.asarray(quadratic, dtype=float)[first, second]
    values += (occupations[:, first] * occupations[:, second]) @ coefficients
    return np.asarray(values)


def shallow_forward_gates(
    angles: np.ndarray,
    m: int,
    depth: int,
) -> list[CompiledGivens]:
    underlying = []
    for theta, (_, _, first, second) in zip(angles, shallow.givens_layout(m, depth)):
        c, s = math.cos(float(theta)), math.sin(float(theta))
        underlying.append(
            (first, second, float(theta), np.array([[c, -s], [s, c]]))
        )
    # The measurement state uses wedge(R)^T, so reverse and transpose the
    # factors whose forward product is R.
    spatial = [(i, j, theta, gate.T) for i, j, theta, gate in reversed(underlying)]
    gates: list[CompiledGivens] = []
    for first, second, theta, gate in spatial:
        unitary = fock_gate(gate)
        compiled, synthesis_error = fixed_two_cx_givens_unitary(theta)
        error = unitary_equivalence_error(unitary, compiled)
        if synthesis_error > 2.0e-12 or error > 2.0e-12:
            raise RuntimeError(
                f"Fixed two-CNOT circuit does not reproduce G({theta:.17g}): "
                f"aggregate={error:.3e}, synthesis={synthesis_error:.3e}"
            )
        gates.append(
            CompiledGivens(
                first=first,
                second=second,
                unitary=unitary,
                cnot_count=CNOTS_PER_GIVENS,
                decomposition_error=error,
            )
        )
        gates.append(
            CompiledGivens(
                first=m + first,
                second=m + second,
                unitary=unitary,
                cnot_count=CNOTS_PER_GIVENS,
                decomposition_error=error,
            )
        )
    return gates


def noise_rows_for_case(
    molecule: str,
    summary_path: Path,
    candidate_path: Path,
    case_spec,
    case_root: Path,
    *,
    cutoff: float,
    max_bond: int,
    p_grid: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    summary = selected_summary(summary_path, molecule)
    selected_k = int(summary["selected_K"])
    depth = int(summary["selected_depth"])
    collector_mode = str(summary["collector_mode"])
    extra_leaves = int(summary["collector_extra_leaves"])
    shots, candidate = selected_shots(candidate_path, summary)
    case_audit_path = case_root / "audit.json"
    case_audit = json.loads(case_audit_path.read_text(encoding="utf-8"))
    collector_config = case_audit.get("collector_configuration")
    if not isinstance(collector_config, dict):
        raise RuntimeError(f"Missing shallow-collector audit configuration: {case_audit_path}")
    required_controls = {
        "mode", "extra_leaf_grid", "objective", "proxy_state", "reference_shots",
        "proxy_cycles", "equality_tolerance", "extra_angle_steps",
        "f3_spectral_maxfev",
    }
    if not required_controls.issubset(collector_config):
        missing = sorted(required_controls - set(collector_config))
        raise RuntimeError(f"Incomplete shallow-collector replay controls: {missing}")
    if str(collector_config.get("mode")) != collector_mode:
        raise RuntimeError("Summary and audit collector modes disagree")
    if extra_leaves not in [int(value) for value in collector_config.get("extra_leaf_grid", [])]:
        raise RuntimeError("Selected extra-leaf count is absent from the audited L grid")
    if str(collector_config.get("proxy_state")) != "hf":
        raise RuntimeError("Publication collector redistribution must use the HF proxy")
    if bool(case_audit.get("sRCDF_ground_state_used_to_optimize_collector")):
        raise RuntimeError("Publication replay requires a state-independent collector proxy")
    f3_spectral_maxfev = int(collector_config["f3_spectral_maxfev"])
    if f3_spectral_maxfev < 1:
        raise RuntimeError("Invalid audited F3 spectral refinement budget")
    case = engine.load_case(case_spec)
    state = engine.load_mps(case["paths"]["state"])
    ci, configs, _ = engine.ci_state(case)
    occupations = engine.spatial_occupations(configs, case["m"])
    all_occ = engine.all_spatial_occupations(case["m"])
    cache = case_root / "srcdf_fit_cache" / f"K{selected_k:02d}_dR{depth}.npz"
    with np.load(cache, allow_pickle=False) as payload:
        rotations = np.asarray(payload["rotations"])
        tensors = np.asarray(payload["tensors"])
    settings, alpha, f3, collector_audit = engine.measurement_settings(
        case,
        {"rotations": rotations, "tensors": tensors, "depth": depth},
        configs,
        occupations,
        all_occ,
        ci,
        f3_spectral_maxfev,
        collector_mode=collector_mode,
        collector_extra_leaves=extra_leaves,
        collector_objective=str(collector_config["objective"]),
        collector_proxy_state=str(collector_config["proxy_state"]),
        collector_reference_shots=int(collector_config["reference_shots"]),
        collector_proxy_cycles=int(collector_config["proxy_cycles"]),
        collector_equality_tolerance=float(collector_config["equality_tolerance"]),
        collector_extra_angle_steps=int(collector_config["extra_angle_steps"]),
    )
    if collector_audit.get("independent_collector_setting_present") is not False:
        raise RuntimeError("Independent one-body collector entered publication noise replay")
    tolerance = float(collector_config["equality_tolerance"])
    reconstruction_residual = float(
        collector_audit["matrix_relative_reconstruction_residual"]
    )
    if reconstruction_residual > 5.0 * tolerance:
        raise RuntimeError(
            f"Collector reconstruction failed closed: {reconstruction_residual:.3e}"
        )
    stored_reconstruction_residual = float(
        candidate["collector_matrix_relative_reconstruction_residual"]
    )
    if abs(reconstruction_residual - stored_reconstruction_residual) > max(
        5.0 * tolerance, 1.0e-12
    ):
        raise RuntimeError("Replayed collector reconstruction disagrees with candidate audit")
    if len(settings) != len(shots) or len(settings) != selected_k + extra_leaves:
        raise RuntimeError("Replayed shallow setting count does not equal K+L")
    families = [str(setting.get("family")) for setting in settings]
    if (
        families.count("shallow_rcdf_leaf") != selected_k
        or families.count("shallow_one_body_leaf") != extra_leaves
    ):
        raise RuntimeError("Replayed source/extra shallow setting families are inconsistent")
    if int(candidate["measurement_settings"]) != len(settings):
        raise RuntimeError("Replayed setting count disagrees with selected candidate")
    ideal_models = engine.ground_srcdf_models(ci, configs, settings)
    recovered = []
    angle_errors = []
    start_counts = []
    for index, setting in enumerate(settings):
        rotation = np.asarray(setting["rotation"], dtype=float)
        angles, error, starts = recover_shallow_angles(
            rotation,
            depth,
            (
                f"{molecule}|K{selected_k}|L{extra_leaves}|"
                f"{setting['family']}|setting{index}"
            ),
        )
        recovered.append(angles)
        angle_errors.append(error)
        start_counts.append(starts)
    gate_lists = [
        shallow_forward_gates(angles, case["m"], depth) for angles in recovered
    ]
    expected_givens_per_setting = 2 * depth * (case["m"] - 1)
    if any(len(gates) != expected_givens_per_setting for gates in gate_lists):
        raise RuntimeError("A replayed setting violates the declared depth-d_R Givens layout")
    if any(
        gate.cnot_count != CNOTS_PER_GIVENS
        for gates in gate_lists
        for gate in gates
    ):
        raise RuntimeError("A replayed Givens does not use the fixed two-CNOT template")
    maximum_givens_decomposition_error = max(
        gate.decomposition_error for gates in gate_lists for gate in gates
    )
    expected_cnot_count_per_setting = (
        CNOTS_PER_GIVENS * expected_givens_per_setting
    )
    maximum_two_qubit_depth = 4 * depth

    alpha_rows = np.broadcast_to(
        occupations[:, None, :], (len(occupations), len(occupations), case["m"])
    )
    beta_rows = np.broadcast_to(
        occupations[None, :, :], (len(occupations), len(occupations), case["m"])
    )
    sector_spin_occupations = np.concatenate((alpha_rows, beta_rows), axis=2).reshape(
        -1, case["n"]
    )
    operators = []
    operator_squares = []
    maximum_polynomial_replay = 0.0
    deterministic_constants = []
    for index in range(len(settings)):
        constant, linear, quadratic = setting_polynomial(
            case, settings[index], tensors, alpha, index
        )
        replay_values = diagonal_polynomial_values(
            0.0, linear, quadratic, sector_spin_occupations
        )
        saved_values = np.asarray(settings[index]["values"], dtype=float).reshape(-1)
        replay_error = float(np.max(np.abs(replay_values - saved_values)))
        maximum_polynomial_replay = max(maximum_polynomial_replay, replay_error)
        if replay_error > 2.0e-9:
            raise RuntimeError(
                f"{molecule} setting {index} polynomial replay failed: {replay_error:.3e}"
            )
        deterministic_constants.append(constant)
        operator = quadratic_diagonal_mpo(constant, linear, quadratic)
        operator_square = operator.apply(
            operator, compress=True, cutoff=min(cutoff * 0.1, 1.0e-12), max_bond=max_bond
        )
        operators.append(operator)
        operator_squares.append(operator_square)
    if (
        not math.isclose(
            deterministic_constants[0], float(case["constant"]),
            rel_tol=0.0, abs_tol=1.0e-14,
        )
        or any(abs(value) > 1.0e-14 for value in deterministic_constants[1:])
    ):
        raise RuntimeError("Molecular scalar was not attached exactly once")

    approximate_mean = float(
        case["constant"] + sum(float(model[2]) for model in ideal_models)
    )
    expected_approximate_mean = float(
        case["exact_energy"] + summary["deterministic_bias_hartree"]
    )
    expectation_reconstruction_error = abs(
        approximate_mean - expected_approximate_mean
    )
    if expectation_reconstruction_error > 1.0e-7:
        raise RuntimeError(
            f"{molecule} selected-ground expectation replay failed: "
            f"{expectation_reconstruction_error:.3e}"
        )

    rows: list[dict[str, Any]] = []
    maximum_mean_replay = 0.0
    maximum_second_replay = 0.0
    for p in p_grid:
        means = []
        variances = []
        second_moments = []
        max_mpo_bond = 0
        for index, (operator, operator_square, gates) in enumerate(
            zip(operators, operator_squares, gate_lists)
        ):
            if abs(float(p)) < 1.0e-15:
                offset = float(case["constant"]) if index == 0 else 0.0
                mean = float(ideal_models[index][2]) + offset
                variance = float(ideal_models[index][3])
                second = mean**2 + variance
            else:
                evolved = propagate_adjoint(
                    operator, gates, float(p), cutoff=cutoff, max_bond=max_bond
                )
                evolved_square = propagate_adjoint(
                    operator_square, gates, float(p), cutoff=cutoff, max_bond=max_bond
                )
                mean = expectation(state, evolved)
                second = expectation(state, evolved_square)
                variance = max(second - mean**2, 0.0)
                max_mpo_bond = max(
                    max_mpo_bond,
                    int(max(evolved.bond_sizes() or [1])),
                    int(max(evolved_square.bond_sizes() or [1])),
                )
            means.append(mean)
            second_moments.append(second)
            variances.append(variance)
            if abs(float(p)) < 1.0e-15:
                offset = float(case["constant"]) if index == 0 else 0.0
                ideal_mean = float(ideal_models[index][2]) + offset
                ideal_second = ideal_mean**2 + float(ideal_models[index][3])
                maximum_mean_replay = max(maximum_mean_replay, abs(mean - ideal_mean))
                maximum_second_replay = max(
                    maximum_second_replay, abs(second - ideal_second)
                )
        signed_bias = float(sum(means) - case["exact_energy"])
        uncovered_variance = max(
            [value for value, count in zip(variances, shots) if count == 0] or [0.0]
        )
        if uncovered_variance > 1.0e-10:
            raise RuntimeError("A nonconstant noisy setting has zero allocated shots")
        sampling_variance = float(sum(
            value / count for value, count in zip(variances, shots) if count > 0
        ))
        rmse = math.sqrt(signed_bias**2 + sampling_variance)
        rows.append({
            "molecule": molecule,
            "p": float(p),
            "T_total_shots": T_FIXED,
            "selected_K": selected_k,
            "selected_L": extra_leaves,
            "selected_depth": depth,
            "collector_mode": collector_mode,
            "measurement_settings": len(settings),
            "two_qubit_givens_gates_each": expected_givens_per_setting,
            "compiled_cnot_gates_each": expected_cnot_count_per_setting,
            "depolarizing_channels_each": expected_cnot_count_per_setting,
            "channels_per_givens": CHANNELS_PER_GIVENS,
            "givens_compilation": "fixed XXPlusYY(2theta,pi/2) = CNOT--single-qubit--CNOT",
            "same_support_channel_composition_exact": True,
            "maximum_logical_cx_depth": maximum_two_qubit_depth,
            "collector_matrix_relative_reconstruction_residual": reconstruction_residual,
            "independent_collector_setting_present": False,
            "deterministic_constant_added_once": True,
            "signed_bias_hartree": signed_bias,
            "sampling_variance_hartree2": sampling_variance,
            "error_hartree": rmse,
            "maximum_evolved_MPO_bond": max_mpo_bond,
            "setting_means_json": json.dumps(means, separators=(",", ":")),
            "setting_second_moments_json": json.dumps(second_moments, separators=(",", ":")),
            "setting_variances_json": json.dumps(variances, separators=(",", ":")),
            "shot_vector": " ".join(map(str, shots.tolist())),
        })
    expected_zero = float(summary["analytic_total_RMSE_hartree"])
    zero_error = float(rows[0]["error_hartree"])
    if abs(zero_error - expected_zero) > 2.0e-6:
        raise RuntimeError(
            f"{molecule} p=0 RMSE replay mismatch: {zero_error} vs {expected_zero}"
        )
    audit = {
        "molecule": molecule,
        "case_spec": asdict(case_spec),
        "selected_K": selected_k,
        "selected_L": extra_leaves,
        "selected_depth": depth,
        "collector_mode": collector_mode,
        "shot_vector_sum": int(shots.sum()),
        "shot_vector_length": int(len(shots)),
        "settings": len(settings),
        "settings_equal_K_plus_L": len(settings) == selected_k + extra_leaves,
        "all_settings_depth_limited": True,
        "independent_collector_setting_present": False,
        "source_setting_operator": "density-density plus redistributed one-body",
        "extra_setting_operator": "redistributed one-body only",
        "f3_spectral_maxfev": f3_spectral_maxfev,
        "maximum_shallow_rotation_replay_frobenius": max(angle_errors),
        "maximum_deterministic_start_index": max(start_counts),
        "maximum_setting_polynomial_replay_hartree": maximum_polynomial_replay,
        "collector_matrix_relative_reconstruction_residual": reconstruction_residual,
        "two_qubit_givens_gates_each": expected_givens_per_setting,
        "compiled_cnot_gates_each": expected_cnot_count_per_setting,
        "depolarizing_channels_each": expected_cnot_count_per_setting,
        "channels_per_givens": CHANNELS_PER_GIVENS,
        "givens_compilation": "fixed XXPlusYY(2theta,pi/2) = CNOT--single-qubit--CNOT",
        "maximum_givens_decomposition_frobenius": maximum_givens_decomposition_error,
        "same_support_channel_composition": "p_eff=1-(1-p)^2 (exact unitary covariance)",
        "maximum_logical_cx_depth": maximum_two_qubit_depth,
        "deterministic_constant_policy": "case constant attached to setting zero exactly once",
        "ground_expectation_reconstruction_error_hartree": expectation_reconstruction_error,
        "maximum_p0_setting_mean_replay_hartree": maximum_mean_replay,
        "maximum_p0_setting_second_moment_replay_hartree2": maximum_second_replay,
        "p0_error_hartree": zero_error,
        "selected_summary_p0_error_hartree": expected_zero,
        "f3": f3,
    }
    arrays = {
        f"{molecule}_setting_angles": np.asarray(recovered),
        f"{molecule}_setting_rotation_replay_errors": np.asarray(angle_errors),
        f"{molecule}_shot_vector": shots,
        f"{molecule}_setting_cnot_counts": np.asarray(
            [sum(gate.cnot_count for gate in gates) for gates in gate_lists],
            dtype=np.int64,
        ),
        f"{molecule}_setting_givens_decomposition_errors": np.asarray(
            [max(gate.decomposition_error for gate in gates) for gates in gate_lists],
            dtype=float,
        ),
    }
    return rows, audit, arrays


def build_curve_rows(
    noise_rows: list[dict[str, Any]], fc_noise: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    variance = pd.read_csv(VARIANCE_T3000)
    result = []
    for row in noise_rows:
        result.append({
            "molecule": row["molecule"],
            "curve": "error_vs_depolarizing_rate",
            "method": SRDD_METHOD,
            "x": row["p"],
            "y": row["error_hartree"],
            "raw_y": row["error_hartree"],
            "right_censored": False,
            "model": (
                "same local two-qubit depolarizing channel after every CNOT; "
                "each shallow Givens is fixed CNOT--single-qubit--CNOT"
            ),
            "T_fixed": T_FIXED,
        })
    for molecule in MOLECULES:
        for method in ("OGM", "SG", "Derand"):
            selected = variance.loc[
                (variance["molecule"] == molecule) & (variance["method"] == method)
            ]
            if len(selected) != 1:
                raise RuntimeError(f"Missing fixed-schedule variance for {molecule}/{method}")
            value = math.sqrt(float(selected.iloc[0]["analytic_MSE_hartree2"]))
            for p in NOISE_GRID:
                result.append({
                    "molecule": molecule,
                    "curve": "error_vs_depolarizing_rate",
                    "method": method,
                    "x": float(p),
                    "y": value,
                    "raw_y": value,
                    "right_censored": False,
                    "model": "no two-qubit measurement gates",
                    "T_fixed": T_FIXED,
                })
    result.extend(fc_noise)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=float, default=1.0e-8)
    parser.add_argument("--max-bond", type=int, default=96)
    parser.add_argument("--quick", action="store_true", help="Evaluate p=0 only")
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Rebuild inverse/noise plotting rows from an existing noise table",
    )
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    required = [
        ENGINE_PATH,
        SHALLOW_PATH,
        COLLECTOR_PATH,
        BEH2_SUMMARY,
        N2_SUMMARY,
        BEH2_CANDIDATES,
        N2_CANDIDATES,
        BEH2_AUDIT,
        N2_AUDIT,
        VARIANCE_T3000,
        RESOURCE_STATS,
        FC_ERROR_CURVES,
        FC_ERROR_SUMMARY,
        FC_NOISE_CURVES,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    fc_budget, fc_inverse, fc_model_audits = fc_error_rows()
    fc_noise, fc_noise_audits = fc_noise_rows()
    inverse_rows, model_rows = inverse_accuracy_rows(fc_inverse, fc_model_audits)
    if args.assemble_only:
        if not NOISE_CSV.exists():
            raise FileNotFoundError(NOISE_CSV)
        noise_frame = pd.read_csv(NOISE_CSV)
        noise_rows = validated_srcdf_noise_rows(noise_frame)
        validate_recovered_circuit_archive(noise_frame)
        write_csv(
            OUTPUT_CSV,
            fc_budget + inverse_rows + build_curve_rows(noise_rows, fc_noise),
        )
        write_csv(MODEL_AUDIT_CSV, model_rows)
        if not AUDIT_JSON.exists():
            raise FileNotFoundError(AUDIT_JSON)
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        audit["noise_grid"] = NOISE_GRID.tolist()
        audit["collector_policy"] = (
            "local shallow augment-only one-body redistribution over K+L "
            "depth-d_R settings with L>=1; independent collector forbidden"
        )
        audit["all_srcdf_setting_shots_are_included"] = True
        audit["maximum_srcdf_logical_cx_depth_formula"] = "4*d_R"
        audit["srcdf_givens_compilation"] = (
            "fixed XXPlusYY(2theta,pi/2) = CNOT--single-qubit--CNOT"
        )
        audit["srcdf_channels_per_givens"] = CHANNELS_PER_GIVENS
        audit["same_support_channel_composition"] = (
            "p_eff=1-(1-p)^2; exact by unitary covariance"
        )
        audit["deterministic_constant_policy"] = "added exactly once to setting zero"
        audit.setdefault("inputs", {}).update({
            str(path.resolve()): sha256(path) for path in required
        })
        audit["inputs"][str(Path(__file__).resolve())] = sha256(Path(__file__))
        output_paths = (OUTPUT_CSV, MODEL_AUDIT_CSV, NOISE_CSV, ANGLE_NPZ)
        audit["outputs"] = {
            str(path.resolve()): sha256(path) for path in output_paths
        }
        audit["fc_inputs"] = {
            str(path.resolve()): sha256(path)
            for path in (FC_ERROR_CURVES, FC_ERROR_SUMMARY, FC_NOISE_CURVES)
        }
        audit["fc_noise_cases"] = fc_noise_audits
        AUDIT_JSON.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[PASS] reassembled {OUTPUT_CSV}")
        return
    p_grid = np.asarray([0.0]) if args.quick else NOISE_GRID
    cases = [
        (
            "BeH2",
            BEH2_SUMMARY,
            BEH2_CANDIDATES,
            engine.CaseSpec("BeH2", "05_BeH2_R1.33376A_ArchivedJW", "linear Be-H=1.33376 A", 6),
            BEH2_ROOT / "BeH2",
        ),
        (
            "N2",
            N2_SUMMARY,
            N2_CANDIDATES,
            engine.CASES["N2"],
            N2_ROOT,
        ),
    ]
    noise_rows = []
    case_audits = []
    angle_arrays: dict[str, np.ndarray] = {}
    for molecule, summary, candidates, spec, case_root in cases:
        rows, audit, arrays = noise_rows_for_case(
            molecule,
            summary,
            candidates,
            spec,
            case_root,
            cutoff=args.cutoff,
            max_bond=args.max_bond,
            p_grid=p_grid,
        )
        noise_rows.extend(rows)
        case_audits.append(audit)
        angle_arrays.update(arrays)
        print(f"[noise] {molecule}: {len(rows)} p values, p=0 error={rows[0]['error_hartree']:.9g}", flush=True)
    if args.quick:
        print("[quick] p=0 validation complete; no publication outputs written")
        return
    curve_rows = fc_budget + inverse_rows + build_curve_rows(noise_rows, fc_noise)
    write_csv(OUTPUT_CSV, curve_rows)
    write_csv(MODEL_AUDIT_CSV, model_rows)
    write_csv(NOISE_CSV, noise_rows)
    np.savez_compressed(ANGLE_NPZ, **angle_arrays)
    validate_recovered_circuit_archive(pd.DataFrame(noise_rows))
    inputs = {
        str(path.resolve()): sha256(path)
        for path in required
    }
    inputs[str(Path(__file__).resolve())] = sha256(Path(__file__))
    for molecule, _, _, _, case_root in cases:
        summary = selected_summary(BEH2_SUMMARY if molecule == "BeH2" else N2_SUMMARY, molecule)
        cache = case_root / "srcdf_fit_cache" / f"K{int(summary['selected_K']):02d}_dR{int(summary['selected_depth'])}.npz"
        inputs[str(cache.resolve())] = sha256(cache)
    outputs = [OUTPUT_CSV, MODEL_AUDIT_CSV, NOISE_CSV, ANGLE_NPZ]
    audit = {
        "status": "PASS",
        "channel": "D_p^(ij)(rho)=(1-p)rho+(p/4)I_ij tensor Tr_ij(rho)",
        "state_preparation_noise_included": False,
        "measurement_circuit_noise_locations": "after every compiled CNOT",
        "srcdf_givens_compilation": (
            "fixed XXPlusYY(2theta,pi/2) = CNOT--single-qubit--CNOT"
        ),
        "srcdf_channels_per_givens": CHANNELS_PER_GIVENS,
        "same_support_channel_composition": (
            "p_eff=1-(1-p)^2; exact by unitary covariance"
        ),
        "collector_policy": (
            "local shallow augment-only one-body redistribution over K+L "
            "depth-d_R settings with L>=1; independent collector forbidden"
        ),
        "all_srcdf_setting_shots_are_included": True,
        "maximum_srcdf_logical_cx_depth_formula": "4*d_R",
        "deterministic_constant_policy": "added exactly once to setting zero",
        "T_fixed": T_FIXED,
        "noise_grid": NOISE_GRID.tolist(),
        "inverse_error_grid": [float(ERROR_GRID[0]), float(ERROR_GRID[-1]), len(ERROR_GRID)],
        "inverse_accuracy_model": "RMSE(T)^2=b^2+V/T with frozen T0=3000 estimator",
        "reporting_limit_measurements": REPORTING_LIMIT,
        "cutoff": args.cutoff,
        "max_bond": args.max_bond,
        "cases": case_audits,
        "fc_noise_cases": fc_noise_audits,
        "inputs": inputs,
        "outputs": {str(path.resolve()): sha256(path) for path in outputs},
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[PASS] wrote {OUTPUT_CSV}")
    print(f"[PASS] wrote {AUDIT_JSON}")


if __name__ == "__main__":
    main()
