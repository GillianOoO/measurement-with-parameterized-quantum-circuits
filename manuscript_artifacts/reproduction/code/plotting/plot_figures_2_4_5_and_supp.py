"""Render the manuscript figures for the stabilizer-calibrated GPD protocol.

The molecular figures contain SRDD and the applicable Pauli/FC-IMA baselines.
GPD is shown only for the sparse and dense random-Hamiltonian ensembles, using
the audited stabilizer-calibrated balanced-prefix results in this directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PLOTS = Path(
    os.environ.get("PAPER_REPRO_FIGURE_DIR", ROOT / "figures" / "rebuilt")
).resolve()
PRESENTATIONS = Path(
    os.environ.get(
        "PAPER_REPRO_PRESENTATION_DIR",
        ROOT / "data" / "shared_processed" / "presentations",
    )
).resolve()
MANIFEST_PATH = Path(
    os.environ.get(
        "PAPER_REPRO_PLOT_MANIFEST",
        ROOT / "validation" / "plot_rebuild_manifest.json",
    )
).resolve()
RANDOM_SUMMARY = (
    ROOT
    / "data"
    / "random_sparse_dense"
    / "processed"
    / "review_figures"
    / "random_calibrated_gpd_empirical50_summary.csv"
)
REVIEW_MANIFEST = RANDOM_SUMMARY.with_name("review_manifest.json")
RANDOM_ANALYTIC_VARIANCE = (
    ROOT
    / "data"
    / "random_sparse_dense"
    / "results"
    / "random_all_methods_empirical50_instance.csv"
)
RANDOM_ANALYTIC_MANIFEST = RANDOM_ANALYTIC_VARIANCE.with_name(
    "random_all_methods_empirical50_manifest.json"
)
RANDOM_ANALYTIC_AUDIT = RANDOM_ANALYTIC_VARIANCE.with_name(
    "random_all_methods_empirical50_audit.json"
)
RANDOM_ANALYTIC_COMMIT = RANDOM_ANALYTIC_VARIANCE.with_name(
    "random_all_methods_empirical50_audit.commit.json"
)
RANDOM_CASES = ROOT / "data" / "random_sparse_dense" / "results" / "gpd_balanced"
RANDOM_BENCHMARKS = ("sparse", "dense")
RANDOM_METHODS = ("GPD", "OGM", "SG", "Derand", "LCS", "AP")
RANDOM_BUDGETS = (12, 45, 160, 572, 2038, 7259, 25848)

os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))


import plotting_backend as base
from matplotlib.ticker import FuncFormatter, FixedLocator

plt = base.plt
np = base.np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, stem: str) -> list[Path]:
    return base.save(fig, PLOTS, stem)


def validate_random_summary() -> list[dict[str, str]]:
    manifest = json.loads(REVIEW_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "REVIEW_ONLY_CALIBRATED_GPD":
        raise RuntimeError("Unexpected stabilizer-calibrated review status")
    if "stabilizer-calibrated" not in str(manifest.get("selection_semantics", "")):
        raise RuntimeError("The random GPD bundle is not stabilizer calibrated")
    expected_hash = manifest.get("review_data_sha256", {}).get(RANDOM_SUMMARY.name)
    if expected_hash != sha256(RANDOM_SUMMARY):
        raise RuntimeError("Random stabilizer-calibrated summary hash mismatch")
    rows = read_csv(RANDOM_SUMMARY)
    methods = ("GPD", "OGM", "SG", "Derand", "LCS", "AP")
    budgets = (12, 45, 160, 572, 2038, 7259, 25848)
    for benchmark in ("sparse", "dense"):
        for method in methods:
            selected = [
                row
                for row in rows
                if row["benchmark"] == benchmark and row["method"] == method
            ]
            if tuple(sorted(int(row["shots"]) for row in selected)) != budgets:
                raise RuntimeError(f"Incomplete Fig. 5 grid: {benchmark}/{method}")
            if any(int(row["instances"]) != 5 for row in selected):
                raise RuntimeError(f"Fig. 5 must contain five {benchmark} instances")
            if any(int(row["empirical_repeats_per_instance"]) != 50 for row in selected):
                raise RuntimeError(f"Fig. 5 must use 50 empirical repeats: {benchmark}/{method}")
    return rows


def render_random_figure() -> tuple[list[Path], Path]:
    rows = validate_random_summary()
    methods = ("GPD", "OGM", "SG", "Derand", "LCS", "AP")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), sharey=False)
    for ax, benchmark, label in zip(axes, ("sparse", "dense"), ("(a)", "(b)")):
        for method in methods:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["benchmark"] == benchmark and row["method"] == method
                ),
                key=lambda row: int(row["shots"]),
            )
            base.line(
                ax,
                [int(row["shots"]) for row in selected],
                [float(row["mean_instance_empirical_rmse"]) for row in selected],
                method,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("The number of measurements")
        ax.set_title(f"{benchmark.capitalize()} four-qubit ensemble")
        ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.38)
        base.panel_label(ax, label)
    axes[0].set_ylabel("Error")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.9,
        frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.18, top=0.78, wspace=0.28)
    outputs = save(fig, "general_hamiltonian_gpd_average_errors")
    plt.close(fig)
    presentation = PRESENTATIONS / "random_stabilizer_gpd_comparison.csv"
    write_csv(
        presentation,
        [
            {
                **row,
                "selection": "stabilizer-calibrated balanced GPD" if row["method"] == "GPD" else "baseline",
                "rmse_kind": "empirical; 50 finite-shot estimates per instance and budget",
                "source": str(RANDOM_SUMMARY.resolve()),
            }
            for row in rows
        ],
    )
    return outputs, presentation


def h4_scan_bond(row: dict[str, str]) -> float:
    return float(row["case"].split("R", 1)[1].replace("p", "."))


def render_molecular_errors() -> tuple[list[Path], Path]:
    scan_pauli, fixed_pauli = base.require_h4_pauli_rerun()
    srcdf = sorted(read_csv(base.H4_SRDD), key=lambda row: float(row["bond_length_angstrom"]))
    if len(srcdf) != 21:
        raise RuntimeError("H4 SRDD bond scan must contain 21 geometries")
    scan_lookup = {(h4_scan_bond(row), row["method"]): row for row in scan_pauli}
    x = [float(row["bond_length_angstrom"]) for row in srcdf]
    h4_rows = read_csv(base.H4_COMPARISON)
    h6_rows = [row for row in read_csv(base.COMMON_COMPARISON) if row["molecule"] == "H6"]

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65))
    ax = axes[0]
    values = {
        "SRDD": [float(row["SRDD_empirical_RMSE"]) for row in srcdf],
        "OGM": [float(scan_lookup[(value, "OGM")]["empirical_total_RMSE_hartree"]) for value in x],
        "SG": [float(scan_lookup[(value, "SG")]["empirical_total_RMSE_hartree"]) for value in x],
        "Derand": [float(scan_lookup[(value, "Derand")]["empirical_total_RMSE_hartree"]) for value in x],
    }
    for method in ("SRDD", "OGM", "SG", "Derand"):
        base.line(ax, x, values[method], method)
    ax.set_xlabel(r"Bond length $R$ ($\AA$)")
    ax.set_ylabel("Error (Ha)")
    ax.set_yscale("log")
    ax.set_title(r"H$_4$, $T=2038$")
    base.panel_label(ax, "(a)")

    ax = axes[1]
    budgets = sorted({int(row["T_total_shots"]) for row in fixed_pauli})
    positions = list(range(len(budgets)))
    srdd_rows = sorted(
        (row for row in h4_rows if row["method"] == "standalone s-RCDF-F"),
        key=lambda row: int(row["T_total_shots"]),
    )
    if [int(row["T_total_shots"]) for row in srdd_rows] != budgets:
        raise RuntimeError("H4 SRDD fixed-budget grid mismatch")
    base.line(ax, positions, [float(row["empirical_total_RMSE"]) for row in srdd_rows], "SRDD")
    for method in ("OGM", "SG", "Derand"):
        selected = sorted(
            (row for row in fixed_pauli if row["method"] == method),
            key=lambda row: int(row["T_total_shots"]),
        )
        if [int(row["T_total_shots"]) for row in selected] != budgets:
            raise RuntimeError(f"H4 fixed-budget grid mismatch for {method}")
        base.line(
            ax,
            positions,
            [float(row["empirical_total_RMSE_hartree"]) for row in selected],
            method,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in budgets], rotation=28, ha="right")
    ax.set_xlabel("The number of measurements")
    ax.set_ylabel("Error (Ha)")
    ax.set_yscale("log")
    ax.set_title(r"H$_4$, $R=1.20\,\AA$")
    base.panel_label(ax, "(b)")

    ax = axes[2]
    positions_h6: list[int] = []
    for stored, method in (("standalone s-RCDF-F", "SRDD"), ("SG", "SG")):
        selected = sorted(
            (row for row in h6_rows if row["method"] == stored),
            key=lambda row: int(row["T_total_shots"]),
        )
        shots = [int(row["T_total_shots"]) for row in selected]
        positions_h6 = list(range(len(shots)))
        base.line(ax, positions_h6, [float(row["empirical_total_RMSE"]) for row in selected], method)
    ax.set_xticks(positions_h6)
    ax.set_xticklabels([str(value) for value in shots], rotation=28, ha="right")
    ax.set_xlabel("The number of measurements")
    ax.set_ylabel("Error (Ha)")
    ax.set_yscale("log")
    ax.set_title(r"H$_6$, $R=3.40\,\AA$")
    base.panel_label(ax, "(c)")

    for ax in axes:
        ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.38)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.9,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.23, top=0.78, wspace=0.48)
    outputs = save(fig, "molecular_hamiltonian_errors")
    plt.close(fig)

    presentation = PRESENTATIONS / "molecular_errors.csv"
    presentation_rows: list[dict[str, Any]] = []
    for method, method_values in values.items():
        presentation_rows.extend(
            {
                "panel": "H4 bond scan",
                "method": method,
                "bond_length_angstrom": bond,
                "shots": 2038,
                "empirical_rmse": value,
            }
            for bond, value in zip(x, method_values)
        )
    write_csv(presentation, presentation_rows)
    return outputs, presentation


def resource_rows_without_gpd() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    _, fixed_pauli = base.require_h4_pauli_rerun()
    chart = [row for row in read_csv(base.RESOURCE_CHART) if row["method"] not in {"AGPD", "GPD"}]
    variance_rows = [row for row in read_csv(base.STATE_VARIANCE) if row["method"] not in {"AGPD", "GPD"}]
    variance_rows.extend(
        row
        for row in read_csv(base.STATE_VARIANCE_BY_BUDGET)
        if row["method"] == "FC-IMA" and int(row["T_total_shots"]) == 3000
    )
    variance_by_key = {
        (row["molecule"], row["method"]): float(row["estimator_variance_hartree2"])
        for row in variance_rows
        if row["molecule"] in {"H4", "H6", "BeH2", "N2"}
    }
    rows: list[dict[str, Any]] = []
    label_to_molecule = {"H4": "H4", "H6": "H6", "BeH2": "BeH2", "N2": "N2"}
    for row in chart:
        label = row["plot_label"]
        method = row["method"]
        if label == "H4" and method in {"OGM", "SG", "Derand"}:
            continue
        if label not in label_to_molecule:
            continue
        key = (label_to_molecule[label], method)
        if key not in variance_by_key:
            continue
        rows.append(
            {
                "benchmark": row["benchmark"],
                "plot_label": label,
                "method": method,
                "reference_total_shots": int(row["reference_total_shots"]),
                "sampling_variance": variance_by_key[key],
                "signed_bias": "",
                "decomposition_items": float(row["decomposition_items"]),
                "shots_to_error_0p01": row["shots_to_error_0p01"],
                "variance_coefficient": "",
                "source": str(base.RESOURCE_CHART.resolve()),
            }
        )
    h4_pauli = [row for row in fixed_pauli if int(row["T_total_shots"]) == 3000]
    if {row["method"] for row in h4_pauli} != {"OGM", "SG", "Derand"}:
        raise RuntimeError("H4 Pauli resource rows are incomplete")
    for row in h4_pauli:
        variance = float(row["analytic_sampling_variance_hartree2"])
        settings = int(row["measurement_settings_or_distinct_bases"])
        coefficient = 3000.0 * variance
        rows.append(
            {
                "benchmark": "H4, R=1.20 A",
                "plot_label": "H4",
                "method": row["method"],
                "reference_total_shots": 3000,
                "sampling_variance": variance,
                "signed_bias": 0.0,
                "decomposition_items": settings,
                "shots_to_error_0p01": max(settings, math.ceil(coefficient / 0.01**2)),
                "variance_coefficient": coefficient,
                "source": str(base.H4_PAULI_FIXED.resolve()),
            }
        )
    return rows, fixed_pauli


def render_resource_figure() -> tuple[list[Path], Path]:
    rows, _ = resource_rows_without_gpd()
    labels = ("H4", "H6", "BeH2", "N2")
    order = ("SRDD", "FC-IMA", "OGM", "SG", "Derand")
    colors = {method: base.COLORS[method] for method in order}

    def slots(label: str) -> tuple[str, ...]:
        if label in {"H4", "H6"}:
            return tuple(method for method in order if method != "FC-IMA")
        return order

    lookup = {(str(row["plot_label"]), str(row["method"])): row for row in rows}
    for label in labels:
        missing = [method for method in slots(label) if (label, method) not in lookup]
        if missing:
            raise RuntimeError(f"Missing Fig. 4 entries for {label}: {missing}")

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.72))
    metrics = (
        ("sampling_variance", "Exact variance"),
        ("decomposition_items", "Distinct measurement settings"),
        ("shots_to_error_0p01", "Required samples"),
    )
    x = np.arange(len(labels), dtype=float)
    width = 0.13
    for panel, (ax, (field, ylabel)) in enumerate(zip(axes, metrics)):
        positive: list[float] = []
        for label_index, label in enumerate(labels):
            slot_methods = slots(label)
            for method_index, method in enumerate(slot_methods):
                row = lookup[(label, method)]
                try:
                    numeric = float(row[field])
                except (TypeError, ValueError):
                    continue
                if numeric <= 0.0 or not np.isfinite(numeric):
                    continue
                offset = (method_index - (len(slot_methods) - 1) / 2) * width
                ax.bar(
                    x[label_index] + offset,
                    numeric,
                    width=width * 0.92,
                    color=colors[method],
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=3,
                )
                positive.append(numeric)
        ax.set_yscale("log")
        if field == "sampling_variance":
            ax.set_ylim(min(positive) / 2.5, max(positive) * 3.5)
        else:
            ax.set_ylim(bottom=0.8, top=max(positive) * 7.5)
        ax.set_xticks(x)
        ax.set_xticklabels((r"H$_4$", r"H$_6$", r"BeH$_2$", r"N$_2$"))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", which="major", linestyle=":", linewidth=0.55, alpha=0.55, zorder=0)
        ax.tick_params(direction="in", top=True, right=True, labelsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.annotate(
            f"({chr(ord('a') + panel)})",
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-34.0, 0.0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
            clip_on=False,
            annotation_clip=False,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[method], edgecolor="black", linewidth=0.45)
        for method in order
    ]
    fig.legend(handles, order, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.19, top=0.79, wspace=0.62)
    outputs = save(fig, "molecular_resource_summary")
    plt.close(fig)
    presentation = PRESENTATIONS / "molecular_resource_summary.csv"
    write_csv(presentation, rows)
    return outputs, presentation


def random_variance_source_paths() -> list[Path]:
    paths = [
        RANDOM_ANALYTIC_VARIANCE,
        RANDOM_ANALYTIC_MANIFEST,
        RANDOM_ANALYTIC_AUDIT,
        RANDOM_ANALYTIC_COMMIT,
    ]
    for benchmark in RANDOM_BENCHMARKS:
        for instance in range(1, 6):
            case_dir = RANDOM_CASES / f"{benchmark}4_seed0_{instance}"
            paths.extend((case_dir / "manifest.json", case_dir / "selected_results.csv"))
    return paths


def build_random_variance_rows() -> list[dict[str, Any]]:
    """Average exact state-dependent variances over the five main-text instances."""

    source_manifest = json.loads(RANDOM_ANALYTIC_MANIFEST.read_text(encoding="utf-8"))
    source_audit = json.loads(RANDOM_ANALYTIC_AUDIT.read_text(encoding="utf-8"))
    source_commit = json.loads(RANDOM_ANALYTIC_COMMIT.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETE" or source_audit.get("status") != "PASS":
        raise RuntimeError("The random Pauli variance source is not fully audited")
    if source_commit.get("status") != "COMMITTED":
        raise RuntimeError("The random Pauli variance audit is not committed")
    expected_variance_hash = source_manifest.get("outputs_sha256", {}).get(
        RANDOM_ANALYTIC_VARIANCE.name
    )
    if expected_variance_hash != sha256(RANDOM_ANALYTIC_VARIANCE):
        raise RuntimeError("Random Pauli variance source hash mismatch")
    if source_audit.get("output_sha256", {}).get(RANDOM_ANALYTIC_VARIANCE.name) != expected_variance_hash:
        raise RuntimeError("Random Pauli variance audit hash mismatch")
    if source_commit.get("manifest_sha256") != sha256(RANDOM_ANALYTIC_MANIFEST):
        raise RuntimeError("Random Pauli variance manifest commit mismatch")
    if source_commit.get("audit_json_sha256") != sha256(RANDOM_ANALYTIC_AUDIT):
        raise RuntimeError("Random Pauli variance audit commit mismatch")

    instance_rows: list[dict[str, Any]] = []
    expected_hashes: dict[tuple[str, int], tuple[str, str]] = {}
    for benchmark in RANDOM_BENCHMARKS:
        for instance in range(1, 6):
            slug = f"{benchmark}4_seed0_{instance}"
            case_dir = RANDOM_CASES / slug
            manifest_path = case_dir / "manifest.json"
            selected_path = case_dir / "selected_results.csv"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise RuntimeError(f"Incomplete stabilizer-calibrated GPD case: {slug}")
            if manifest.get("version") != "periodic-iswap-gpd-stabilizer-balanced-v2":
                raise RuntimeError(f"Unexpected GPD version for {slug}")
            if manifest.get("selection", {}).get("loss") != "hypot(w_a*x_a, w_s*x_s)":
                raise RuntimeError(f"Unexpected GPD selection loss for {slug}")
            expected_selected_hash = manifest.get("artifacts", {}).get("selected_results.csv")
            if expected_selected_hash != sha256(selected_path):
                raise RuntimeError(f"Selected-result hash mismatch for {slug}")

            case = manifest["case"]
            target_hash = str(case["target_hash"])
            state_hash = str(case["state_hash"])
            expected_hashes[(benchmark, instance)] = (target_hash, state_hash)
            selected = read_csv(selected_path)
            if tuple(sorted(int(row["shots"]) for row in selected)) != RANDOM_BUDGETS:
                raise RuntimeError(f"Incomplete GPD variance budget grid for {slug}")
            for row in selected:
                variance = float(row["variance"])
                bias = float(row["bias"])
                expectation_difference = float(row["approximate_expectation"]) - float(row["target_expectation"])
                if not math.isfinite(bias) or not math.isclose(
                    bias, expectation_difference, rel_tol=1.0e-10, abs_tol=1.0e-12
                ):
                    raise RuntimeError(f"GPD expectation/bias mismatch for {slug}/T={row['shots']}")
                if not math.isfinite(variance) or variance < 0.0:
                    raise RuntimeError(f"Invalid GPD variance for {slug}/T={row['shots']}")
                if str(row["selection_uses_benchmark_state"]).lower() != "false":
                    raise RuntimeError(f"State-dependent GPD selection detected for {slug}")
                instance_rows.append(
                    {
                        "benchmark": benchmark,
                        "instance": instance,
                        "slug": slug,
                        "method": "GPD",
                        "shots": int(row["shots"]),
                        "state_dependent_variance_hartree2": variance,
                        "absolute_approximation_bias": abs(bias),
                        "squared_approximation_bias": bias * bias,
                        "selected_k": int(row["k"]),
                        "target_hash": target_hash,
                        "state_hash": state_hash,
                        "variance_protocol": (
                            "exact fixed-integer-allocation variance of the "
                            "stabilizer-calibrated selected GPD prefix"
                        ),
                        "source": str(selected_path.resolve()),
                    }
                )

    pauli_rows = [
        row
        for row in read_csv(RANDOM_ANALYTIC_VARIANCE)
        if row["method"] in RANDOM_METHODS[1:]
    ]
    for benchmark in RANDOM_BENCHMARKS:
        for instance in range(1, 6):
            target_hash, state_hash = expected_hashes[(benchmark, instance)]
            slug = f"{benchmark}4_seed0_{instance}"
            for method in RANDOM_METHODS[1:]:
                for shots in RANDOM_BUDGETS:
                    selected = [
                        row
                        for row in pauli_rows
                        if row["benchmark"] == benchmark
                        and int(row["instance"]) == instance
                        and row["method"] == method
                        and int(row["shots"]) == shots
                    ]
                    if len(selected) != 1:
                        raise RuntimeError(
                            f"Expected one Pauli variance row for {slug}/{method}/T={shots}"
                        )
                    row = selected[0]
                    if row["target_hash"] != target_hash or row["state_hash"] != state_hash:
                        raise RuntimeError(f"Random-Hamiltonian input mismatch for {slug}/{method}")
                    variance = float(row["analytic_variance"])
                    if not math.isfinite(variance) or variance < 0.0:
                        raise RuntimeError(f"Invalid Pauli variance for {slug}/{method}/T={shots}")
                    instance_rows.append(
                        {
                            "benchmark": benchmark,
                            "instance": instance,
                            "slug": slug,
                            "method": method,
                            "shots": shots,
                            "state_dependent_variance_hartree2": variance,
                            "selected_k": "",
                            "target_hash": target_hash,
                            "state_hash": state_hash,
                            "variance_protocol": str(row["protocol"]),
                            "source": str(RANDOM_ANALYTIC_VARIANCE.resolve()),
                        }
                    )

    summary: list[dict[str, Any]] = []
    for benchmark in RANDOM_BENCHMARKS:
        for method in RANDOM_METHODS:
            for shots in RANDOM_BUDGETS:
                selected = [
                    row
                    for row in instance_rows
                    if row["benchmark"] == benchmark
                    and row["method"] == method
                    and int(row["shots"]) == shots
                ]
                if len(selected) != 5:
                    raise RuntimeError(
                        f"Expected five variances for {benchmark}/{method}/T={shots}"
                    )
                values = np.asarray(
                    [float(row["state_dependent_variance_hartree2"]) for row in selected],
                    dtype=float,
                )
                summary.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "shots": shots,
                        "instances": 5,
                        "mean_instance_state_dependent_variance_hartree2": float(np.mean(values)),
                        "instance_variance_sample_sd_hartree2": float(np.std(values, ddof=1)),
                        "zero_variance_instances": int(np.count_nonzero(values == 0.0)),
                        "averaging_rule": "arithmetic mean of five instance-level exact variances",
                        "selection": (
                            "depth-L=4 stabilizer-calibrated GPD prefix"
                            if method == "GPD"
                            else "same deployed Pauli protocol as the main-text comparison"
                        ),
                    }
                )
                if method == "GPD":
                    summary[-1].update(
                        {
                            "mean_instance_absolute_approximation_bias": float(
                                np.mean([row["absolute_approximation_bias"] for row in selected])
                            ),
                            "mean_instance_squared_approximation_bias": float(
                                np.mean([row["squared_approximation_bias"] for row in selected])
                            ),
                            "bias_averaging_rule": "arithmetic mean of five absolute expectation differences",
                        }
                    )
    return summary


def render_supplement_variance() -> tuple[list[Path], Path]:
    molecular_rows: list[dict[str, Any]] = [
        dict(row)
        for row in read_csv(base.STATE_VARIANCE_BY_BUDGET)
        if row["method"] not in {"AGPD", "GPD"}
    ]
    molecules = ("H4", "H6", "BeH2", "N2")
    methods = {
        "H4": ("SRDD", "OGM", "SG", "Derand"),
        "H6": ("SRDD", "OGM", "SG", "Derand"),
        "BeH2": ("SRDD", "FC-IMA", "OGM", "SG", "Derand"),
        "N2": ("SRDD", "FC-IMA", "OGM", "SG", "Derand"),
    }
    budgets = {
        molecule: sorted(
            {
                int(row["T_total_shots"])
                for row in molecular_rows
                if row["molecule"] == molecule
            }
        )
        for molecule in molecules
    }
    for molecule in molecules:
        for method in methods[molecule]:
            observed = sorted(
                int(row["T_total_shots"])
                for row in molecular_rows
                if row["molecule"] == molecule and row["method"] == method
            )
            if observed != budgets[molecule]:
                raise RuntimeError(f"Incomplete SI variance grid for {molecule}/{method}")
    molecular_rows.sort(
        key=lambda row: (
            molecules.index(str(row["molecule"])),
            int(row["T_total_shots"]),
            methods[str(row["molecule"])].index(str(row["method"])),
        )
    )
    random_rows = build_random_variance_rows()
    # Use one common budget domain for all upper curves and lower bars.
    # Zero GPD variance cannot be shown on a logarithmic variance axis.
    random_plot_budgets = {
        benchmark: {
            int(row["shots"]) for row in random_rows
            if row["benchmark"] == benchmark and row["method"] == "GPD"
            and float(row["mean_instance_state_dependent_variance_hartree2"]) > 0.0
        }
        for benchmark in RANDOM_BENCHMARKS
    }
    if any(not counts for counts in random_plot_budgets.values()):
        raise RuntimeError("No positive-variance GPD measurement counts for an SI ensemble")
    panels = (
        ("molecule", "H4", r"H$_4$"),
        ("molecule", "H6", r"H$_6$"),
        ("molecule", "BeH2", r"BeH$_2$"),
        ("molecule", "N2", r"N$_2$"),
        ("ensemble", "sparse", "Sparse"),
        ("ensemble", "dense", "Dense"),
    )
    fig = plt.figure(figsize=(7.25, 6.35))
    grid = fig.add_gridspec(
        2, 3, left=0.115, right=0.985, bottom=0.09, top=0.895,
        wspace=0.57, hspace=0.45,
    )
    bias_rows: list[dict[str, Any]] = []
    axes_by_panel = []
    for panel, (kind, target, title) in enumerate(panels):
        inner = grid[panel // 3, panel % 3].subgridspec(
            2, 1, height_ratios=(3.0, 1.0), hspace=0.0
        )
        ax = fig.add_subplot(inner[0])
        bias_ax = fig.add_subplot(inner[1], sharex=ax)
        axes_by_panel.append((ax, bias_ax))
        panel_methods = methods[target] if kind == "molecule" else RANDOM_METHODS
        for method in panel_methods:
            if kind == "molecule":
                selected = sorted(
                    (
                        row
                        for row in molecular_rows
                        if row["molecule"] == target and row["method"] == method
                    ),
                    key=lambda row: int(row["T_total_shots"]),
                )
                x_values = [int(row["T_total_shots"]) for row in selected]
                y_values = [float(row["estimator_variance_hartree2"]) for row in selected]
            else:
                selected = sorted(
                    (
                        row
                        for row in random_rows
                        if row["benchmark"] == target and row["method"] == method
                        and int(row["shots"]) in random_plot_budgets[target]
                    ),
                    key=lambda row: int(row["shots"]),
                )
                positive = [
                    row
                    for row in selected
                    if float(row["mean_instance_state_dependent_variance_hartree2"]) > 0.0
                ]
                x_values = [int(row["shots"]) for row in positive]
                y_values = [
                    float(row["mean_instance_state_dependent_variance_hartree2"])
                    for row in positive
                ]
            base.line(
                ax,
                x_values,
                y_values,
                method,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.35)
        ax.tick_params(which="both", direction="in", top=True, right=True, labelbottom=False)
        base.panel_label(ax, f"({chr(ord('a') + panel)})")
        if panel in (0, 3):
            ax.set_ylabel(r"$\operatorname{Var}_{\rho}(\widehat E_T)$")
        if kind == "molecule":
            selected_bias = sorted(
                (row for row in molecular_rows if row["molecule"] == target and row["method"] == "SRDD"),
                key=lambda row: int(row["T_total_shots"]),
            )
            bias_method = "SRDD"
            bias_x = [int(row["T_total_shots"]) for row in selected_bias]
            bias_y = [abs(float(row["signed_approximation_bias_hartree"])) for row in selected_bias]
            for row, absolute_bias in zip(selected_bias, bias_y):
                variance = float(row["estimator_variance_hartree2"])
                mse = float(row["analytic_MSE_hartree2"])
                if not math.isclose(mse, variance + absolute_bias**2, rel_tol=1.0e-10, abs_tol=1.0e-14):
                    raise RuntimeError(f"Molecular bias/MSE mismatch for {target}/T={row['T_total_shots']}")
        else:
            selected_bias = sorted(
                (row for row in random_rows if row["benchmark"] == target and row["method"] == "GPD"
                 and int(row["shots"]) in random_plot_budgets[target]),
                key=lambda row: int(row["shots"]),
            )
            bias_method = "GPD"
            bias_x = [int(row["shots"]) for row in selected_bias]
            bias_y = [float(row["mean_instance_absolute_approximation_bias"]) for row in selected_bias]
        if not bias_y or any(not math.isfinite(value) or value < 0.0 for value in bias_y):
            raise RuntimeError(f"Invalid absolute-bias data for {target}")
        # One bar per evaluated T, with equal visual widths on the log axis.
        # Negative plotting coordinates place absolute-bias magnitudes below zero.
        # The underlying bias table and the displayed tick magnitudes stay nonnegative.
        shared_xlim = ax.get_xlim()
        log_x = np.log10(np.asarray(bias_x, dtype=float))
        log_width = min(0.6 * float(np.min(np.diff(log_x))), 0.08 * float(np.ptp(log_x)))
        bar_left = 10.0 ** (log_x - log_width / 2)
        bar_right = 10.0 ** (log_x + log_width / 2)
        bias_ax.bar(
            bar_left, -np.asarray(bias_y), width=bar_right - bar_left,
            align="edge", bottom=0.0, color=base.COLORS[bias_method],
            edgecolor=base.COLORS[bias_method], linewidth=0.55, alpha=0.90,
            label=bias_method, zorder=3,
        )
        ax.set_xlim(shared_xlim)
        bias_ax.set_yscale("linear")
        # Reserve space below the longest bar for the scientific-scale label.
        requested_top = max(bias_y) * 1.4 if max(bias_y) > 0.0 else 1.0
        exponent = math.floor(math.log10(requested_top))
        scale = 10.0**exponent
        mantissa = requested_top / scale
        nice_top = next(value for value in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if value >= mantissa)
        top = nice_top * scale
        bias_ax.set_ylim(-top, 0.0)
        bias_ax.yaxis.set_major_locator(FixedLocator([-top, -top / 2, 0.0]))
        bias_ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position, unit=scale: f"{abs(value) / unit:g}")
        )
        if exponent != 0:
            bias_ax.text(
                0.97, 0.02, rf"$\times 10^{{{exponent}}}$", transform=bias_ax.transAxes,
                ha="right", va="bottom", fontsize=8.0,
            )
        bias_ax.yaxis.tick_left()
        bias_ax.yaxis.set_label_position("left")
        bias_ax.set_ylabel("Absolute bias", fontsize=8.0, labelpad=4)
        bias_ax.tick_params(axis="y", direction="in", left=True, right=False, labelsize=8.0)
        bias_ax.tick_params(axis="x", which="both", direction="in", top=True, labelsize=8.5)
        bias_ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.35)
        bias_ax.set_xlabel("The number of measurements", fontsize=8.5)
        for total, value in zip(bias_x, bias_y):
            bias_rows.append({
                "panel": chr(ord("a") + panel),
                "benchmark": target,
                "method": bias_method,
                "measurements": total,
                "absolute_bias": value,
                "instances": 1 if kind == "molecule" else 5,
                "aggregation": "absolute expectation difference" if kind == "molecule" else "mean of instance-level absolute expectation differences",
            })
    legend_order = ("GPD", "SRDD", "FC-IMA", "OGM", "SG", "Derand", "LCS", "AP")
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=base.COLORS[method],
            marker=base.MARKERS[method],
            linestyle=base.LINESTYLES[method],
            label=method,
        )
        for method in legend_order
    ]
    fig.legend(
        handles=handles,
        ncol=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.62,
        handlelength=2.0,
        handletextpad=0.35,
        fontsize=7.4,
        frameon=False,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for upper, lower in axes_by_panel:
        # The touching frames share a border. Keep the bias zero label below
        # that border and any nearby variance label above it.
        lower.yaxis.get_major_ticks()[-1].label1.set_verticalalignment("top")
        for tick in upper.yaxis.get_major_ticks() + upper.yaxis.get_minor_ticks():
            tick_y = upper.transData.transform((0.0, tick.get_loc()))[1]
            if 0.0 <= tick_y - upper.bbox.y0 < 10.0 * fig.dpi / 72.0:
                tick.label1.set_verticalalignment("bottom")
        tick_width = max(
            (label.get_window_extent(renderer).width for label in upper.get_yticklabels() if label.get_visible()),
            default=0.0,
        )
        upper.texts[0].set_position((-(tick_width * 72.0 / fig.dpi + 8.0), 0))
    outputs = save(fig, "supp_state_dependent_variance_vs_budget")
    # Each panel PNG includes its aligned variance and bias axes.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_dir = PLOTS / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    from matplotlib.transforms import Bbox
    for panel, pair in enumerate(axes_by_panel):
        bounds = Bbox.union([axis.get_tightbbox(renderer) for axis in pair])
        bounds = bounds.transformed(fig.dpi_scale_trans.inverted()).padded(0.04)
        panel_path = panel_dir / f"supp_variance_bias_panel_{chr(ord('a') + panel)}.png"
        fig.savefig(panel_path, dpi=320, bbox_inches=bounds)
        outputs.append(panel_path)
    plt.close(fig)
    write_csv(PRESENTATIONS / "supp_absolute_bias_by_budget.csv", bias_rows)
    presentation = PRESENTATIONS / "random_state_dependent_variance_by_budget.csv"
    write_csv(presentation, random_rows)
    return outputs, presentation


def manifest_inputs() -> list[Path]:
    return [
        RANDOM_SUMMARY,
        REVIEW_MANIFEST,
        base.H4_SRDD,
        base.H4_COMPARISON,
        base.H4_PAULI_SCAN,
        base.H4_PAULI_FIXED,
        base.H4_PAULI_AUDIT,
        base.H4_PAULI_MANIFEST,
        base.COMMON_COMPARISON,
        base.RESOURCE_CHART,
        base.STATE_VARIANCE,
        base.STATE_VARIANCE_BY_BUDGET,
        base.STATE_VARIANCE_FULL_BY_BUDGET,
        *random_variance_source_paths(),
    ]


def main(*, supplement_variance_only: bool = False, figure_ids=None) -> None:
    base.configure()
    renderers = {"2": render_molecular_errors, "4": render_resource_figure,
                 "5": render_random_figure, "s1": render_supplement_variance}
    chosen = ["s1"] if supplement_variance_only else list(figure_ids or renderers)
    outputs: list[Path] = []
    presentations: list[Path] = []
    for figure_id in dict.fromkeys(chosen):
        rendered, presentation = renderers[figure_id]()
        outputs.extend(rendered)
        presentations.append(presentation)
    inputs = manifest_inputs()
    if "s1" in chosen:
        presentations.append(PRESENTATIONS / "supp_absolute_bias_by_budget.csv")
    manifest = {
        "schema": 2,
        "status": "PASS",
        "figures_rendered": chosen,
        "renderer": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "selection_semantics": "GPD uses stabilizer-calibrated hypot(w_a*x_a,w_s*x_s)",
        "scope": (
            "GPD numerical results appear in the sparse/dense random-Hamiltonian "
            "error comparison and supplementary average-variance panels"
        ),
        "inputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in inputs},
        "presentations": {str(path.resolve()): sha256(path) for path in presentations},
        "outputs": {str(path.resolve()): sha256(path) for path in outputs},
    }
    manifest_path = MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--supplement-variance-only", action="store_true")
    options.add_argument("--figures", nargs="+", choices=("2", "4", "5", "s1"))
    args = parser.parse_args()
    main(supplement_variance_only=args.supplement_variance_only, figure_ids=args.figures)
