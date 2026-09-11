#!/usr/bin/env python3
"""Render publication variance figures and the detailed SI table from audited CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANUSCRIPT = (
    ROOT
    / "RMeasurementAnsatz"
    / "manuscript"
    / "Journal_chemical_theory_computation"
)
PLOT_DIRECTORY = MANUSCRIPT / "plots_figures"
DETAIL_CSV = HERE / "state_dependent_variance_by_budget.csv"
SUMMARY_CSV = HERE / "state_dependent_variance_T3000.csv"
FC_IMA_CURVES = (
    ROOT
    / "outputs"
    / "traditional_cm_benchmarks"
    / "error_eval"
    / "results"
    / "sampling_curves_all.csv"
)
PUBLICATION_DETAIL_CSV = HERE / "supp_state_dependent_variance_by_budget.csv"
PUBLICATION_SUMMARY_CSV = HERE / "supp_state_dependent_variance_T3000.csv"
MAIN_STEM = "state_dependent_variance_T3000"
SUPP_STEM = "supp_state_dependent_variance_vs_budget"
SUPP_TABLE = HERE / "state_dependent_variance_details_T3000_table.tex"
PLOT_MANIFEST = HERE / "figure_table_manifest.json"

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MOLECULES = ("H4", "H6", "BeH2", "N2")
MOLECULE_LABELS = {
    "H4": r"H$_4$",
    "H6": r"H$_6$",
    "BeH2": r"BeH$_2$",
    "N2": r"N$_2$",
}
BUDGET_PANEL_TITLES = {
    "H4": r"H$_4$",
    "H6": r"H$_6$",
    "BeH2": r"BeH$_2$",
    "N2": r"N$_2$",
}
METHODS_BY_MOLECULE = {
    "H4": ("AGPD", "SRDD", "OGM", "SG", "Derand"),
    "H6": ("SRDD", "OGM", "SG", "Derand"),
    "BeH2": ("SRDD", "FC-IMA", "OGM", "SG", "Derand"),
    "N2": ("SRDD", "FC-IMA", "OGM", "SG", "Derand"),
}
LEGEND_METHODS = ("AGPD", "SRDD", "FC-IMA", "OGM", "SG", "Derand")
STYLES = {
    "AGPD": {"color": "#3F6F98", "marker": "*", "linestyle": "-", "label": "AGPD"},
    "SRDD": {"color": "#4F8A70", "marker": "D", "linestyle": "-", "label": "SRDD"},
    "FC-IMA": {"color": "#B45A45", "marker": "o", "linestyle": (0, (3, 1, 1, 1)), "label": "FC-IMA"},
    "OGM": {"color": "#777472", "marker": "v", "linestyle": ":", "label": "OGM"},
    "SG": {"color": "#4D4D4D", "marker": "s", "linestyle": "--", "label": "SG"},
    "Derand": {"color": "#A48974", "marker": "^", "linestyle": "-.", "label": "Derand"},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty publication table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def publication_rows(detail: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Filter to main-text targets and add the analytic FC-IMA variance curves."""
    output: list[dict[str, Any]] = [
        dict(row)
        for row in detail
        if row["molecule"] in MOLECULES
        and row["method"] in METHODS_BY_MOLECULE[row["molecule"]]
    ]
    fc_settings = {"BeH2": 36, "N2": 111}
    fc_rows = [
        row
        for row in read_rows(FC_IMA_CURVES)
        if row["molecule"] in fc_settings
        and row["method"] == "FC-IMA"
        and row["allocation"] == "fc_ima_primary"
    ]
    expected_budgets = {1300, 1600, 2000, 2400, 3000}
    for molecule in fc_settings:
        budgets = {
            int(row["T_total_shots"])
            for row in fc_rows
            if row["molecule"] == molecule
        }
        if budgets != expected_budgets:
            raise RuntimeError(f"Unexpected FC-IMA budget grid for {molecule}: {sorted(budgets)}")
    for row in fc_rows:
        molecule = row["molecule"]
        total = int(row["T_total_shots"])
        if int(row["T_actual_shots"]) != total:
            raise RuntimeError(f"FC-IMA shot mismatch for {molecule} at T={total}")
        if int(row["group_count"]) != fc_settings[molecule]:
            raise RuntimeError(f"FC-IMA group-count mismatch for {molecule}")
        standard_error = float(row["analytic_rmse_hartree"])
        variance = standard_error**2
        output.append(
            {
                "molecule": molecule,
                "geometry_and_representation": (
                    "linear Be-H=1.33376 A, full STO-3G (6e,7o), 14 qubits"
                    if molecule == "BeH2"
                    else "R=2.25 A, full STO-3G (14e,10o), 20 qubits"
                ),
                "qubits": 14 if molecule == "BeH2" else 20,
                "method": "FC-IMA",
                "T_total_shots": total,
                "estimator_variance_hartree2": variance,
                "T_times_fixed_variance_hartree2": total * variance,
                "iid_between_basis_variance_hartree2": "",
                "iid_empirical_distribution_one_shot_variance_hartree2": "",
                "analytic_sampling_SE_hartree": standard_error,
                "minimum_term_hits": "",
                "covered_pauli_terms": "",
                "measurement_settings": fc_settings[molecule],
                "signed_approximation_bias_hartree": 0.0,
                "analytic_MSE_hartree2": variance,
                "variance_protocol": "Nature-2023 overlapping FC-IMA analytic variance with exact integer allocation",
                "source_status": "audited Nature-2023 overlapping FC-IMA analytic variance",
            }
        )
    output.sort(
        key=lambda row: (
            MOLECULES.index(str(row["molecule"])),
            int(row["T_total_shots"]),
            METHODS_BY_MOLECULE[str(row["molecule"])].index(str(row["method"])),
        )
    )
    expected = (
        sum(len(METHODS_BY_MOLECULE[molecule]) for molecule in ("H4", "H6")) * 8
        + sum(len(METHODS_BY_MOLECULE[molecule]) for molecule in ("BeH2", "N2")) * 5
    )
    if len(output) != expected:
        raise RuntimeError(f"Unexpected publication variance row count: {len(output)} != {expected}")
    return output


def publication_summary(detail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in detail if int(row["T_total_shots"]) == 3000]
    expected = sum(len(METHODS_BY_MOLECULE[molecule]) for molecule in MOLECULES)
    if len(rows) != expected:
        raise RuntimeError(f"Unexpected publication T=3000 row count: {len(rows)} != {expected}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 350,
        }
    )


def save_figure(
    figure: plt.Figure,
    stem: str,
    *,
    copy_to_manuscript: bool = True,
) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "png"):
        path = HERE / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
        if copy_to_manuscript:
            destination = PLOT_DIRECTORY / path.name
            shutil.copyfile(path, destination)
            outputs.append(destination)
    return outputs


def plot_t3000(summary: list[dict[str, str]]) -> list[Path]:
    lookup = {
        (row["molecule"], row["method"]): row for row in summary
    }
    figure, axis = plt.subplots(figsize=(7.25, 3.35))
    x = np.arange(len(MOLECULES), dtype=float)
    width = 0.155
    available_values = [
        float(row["estimator_variance_hartree2"])
        for row in summary
        if row["estimator_variance_hartree2"]
    ]
    floor = 10 ** math.floor(math.log10(min(available_values))) / 1.8
    ceiling = 10 ** math.ceil(math.log10(max(available_values)))
    for molecule_index, molecule in enumerate(MOLECULES):
        methods = METHODS_BY_MOLECULE[molecule]
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
        for method_index, method in enumerate(methods):
            raw = lookup[(molecule, method)]["estimator_variance_hartree2"]
            style = STYLES[method]
            axis.bar(
                x[molecule_index] + offsets[method_index],
                float(raw),
                width=width * 0.90,
                color=style["color"],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.93,
                zorder=3,
            )
    axis.set_yscale("log")
    axis.set_ylim(floor, ceiling)
    axis.set_ylabel(r"$\mathrm{Var}_{\rho}(\widehat E_T)$")
    axis.set_xticks(x, [MOLECULE_LABELS[item] for item in MOLECULES])
    axis.set_title(r"Exact state-dependent sampling variance at $T=3000$")
    axis.grid(which="major", axis="y", color="#D1D5DB", linewidth=0.65, zorder=0)
    axis.grid(which="minor", axis="y", color="#E5E7EB", linewidth=0.35, zorder=0)
    legend_handles = [
        Line2D(
            [0], [0], color=STYLES[method]["color"],
            marker=STYLES[method]["marker"],
            linestyle=STYLES[method]["linestyle"], label=method,
        )
        for method in LEGEND_METHODS
    ]
    axis.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=6, frameon=False, handlelength=1.5, columnspacing=1.05,
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
    figure.subplots_adjust(bottom=0.25, left=0.09, right=0.99, top=0.90)
    # The response-letter copy of this panel is generated together with the
    # main-text resource figure so that both share the same method matrix.
    outputs = save_figure(figure, MAIN_STEM, copy_to_manuscript=False)
    plt.close(figure)
    return outputs


def plot_budget_curves(detail: list[dict[str, str]]) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.30))
    flat_axes = list(axes.flat)
    for panel_index, molecule in enumerate(MOLECULES):
        axis = flat_axes[panel_index]
        for method in METHODS_BY_MOLECULE[molecule]:
            selected = sorted(
                (
                    row
                    for row in detail
                    if row["molecule"] == molecule and row["method"] == method
                ),
                key=lambda row: int(row["T_total_shots"]),
            )
            if not selected:
                continue
            shots = np.asarray([int(row["T_total_shots"]) for row in selected])
            variances = np.asarray(
                [float(row["estimator_variance_hartree2"]) for row in selected]
            )
            style = STYLES[method]
            axis.plot(
                shots,
                variances,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                label=style["label"],
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(BUDGET_PANEL_TITLES[molecule])
        axis.annotate(
            f"({chr(97 + panel_index)})",
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-42.0 if panel_index in (0, 2) else -20.0, 0.0),
            textcoords="offset points",
            ha="right",
            va="top",
            weight="bold",
            clip_on=False,
            annotation_clip=False,
        )
        axis.grid(which="major", color="#D1D5DB", linewidth=0.55)
        axis.grid(which="minor", color="#E5E7EB", linewidth=0.30)
        for spine in axis.spines.values():
            spine.set_visible(True)
        if panel_index in (0, 2):
            axis.set_ylabel(r"$\mathrm{Var}_{\rho}(\widehat E_T)$")
        if panel_index >= 2:
            axis.set_xlabel("Shot budget $T$")
    handles = [
        Line2D(
            [0], [0], color=STYLES[method]["color"],
            marker=STYLES[method]["marker"],
            linestyle=STYLES[method]["linestyle"], label=method,
        )
        for method in LEGEND_METHODS
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        ncol=6,
        columnspacing=1.0,
    )
    figure.tight_layout(rect=(0.0, 0.075, 1.0, 1.0), w_pad=5.0, h_pad=1.2)
    outputs = save_figure(figure, SUPP_STEM)
    plt.close(figure)
    return outputs


def tex_number(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return r"\textemdash"
    number = float(value)
    if number == 0.0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(number))))
    mantissa = number / (10**exponent)
    decimals = max(digits - 1, 0)
    return f"${mantissa:.{decimals}f}\\!\\times\\!10^{{{exponent}}}$"


def write_supp_table(summary: list[dict[str, str]]) -> None:
    rows = []
    for molecule in MOLECULES:
        molecule_rows = [row for row in summary if row["molecule"] == molecule]
        molecule_rows.sort(
            key=lambda row: METHODS_BY_MOLECULE[molecule].index(row["method"])
        )
        for index, row in enumerate(molecule_rows):
            molecule_cell = MOLECULE_LABELS[molecule] if index == 0 else ""
            rows.append(
                " & ".join(
                    [
                        molecule_cell,
                        row["method"],
                        tex_number(row["estimator_variance_hartree2"]),
                        tex_number(row["T_times_fixed_variance_hartree2"]),
                        tex_number(row["iid_empirical_distribution_one_shot_variance_hartree2"]),
                        tex_number(row["signed_approximation_bias_hartree"]),
                        tex_number(row["analytic_MSE_hartree2"]),
                    ]
                )
                + r" \\" 
            )
        if molecule != MOLECULES[-1]:
            rows.append(r"\\addlinespace[1pt]")
    # The manuscript does not load booktabs, so replace the small group spacer
    # with a zero-content row that works with the existing tabular packages.
    rows = [r"\\[-2pt]" if row == r"\\addlinespace[1pt]" else row for row in rows]
    text = """\\begin{table*}[t]
\\color{BLUE}
\\captionsetup{labelfont={color=BLUE},textfont={color=BLUE}}
\\centering
\\caption{Detailed $T=3000$ state-dependent variance audit.  $V_{\\rm fix}=\\operatorname{Var}_{\\rho}(\\widehat E_T)$ is the actual fixed-schedule or fixed-stratum variance; $T V_{\\rm fix}$ is its normalized coefficient.  For Derand, OGM, and SG, $V_{\\rm iid}^{\\rm(emp)}$ is the OGM one-shot i.i.d. variance after setting $p_P=N(P)/T$; it is not the variance used for the fixed schedule. FC-IMA uses its exact analytic variance under the optimized integer allocation. The signed approximation bias $b$ is in Ha and $\\mathrm{MSE}=V_{\\rm fix}+b^2$ is in Ha$^2$.}
\\label{tab:si_state_dependent_variance_details}
\\scriptsize
\\begin{adjustbox}{max width=\\textwidth}
\\begin{tabular}{@{}llccccc@{}}
\\hline
Molecule & Method & $V_{\\rm fix}$ (Ha$^2$) & $T V_{\\rm fix}$ (Ha$^2$) & $V_{\\rm iid}^{\\rm(emp)}$ (Ha$^2$) & $b$ (Ha) & MSE (Ha$^2$) \\\\
\\hline
""" + "\n".join(rows) + """
\\hline
\\end{tabular}
\\end{adjustbox}
\\end{table*}
"""
    SUPP_TABLE.write_text(text, encoding="utf-8")


def main() -> None:
    if not DETAIL_CSV.is_file() or not SUMMARY_CSV.is_file() or not FC_IMA_CURVES.is_file():
        raise FileNotFoundError("Run build_state_dependent_variance.py first")
    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    detail = publication_rows(read_rows(DETAIL_CSV))
    summary = publication_summary(detail)
    write_rows(PUBLICATION_DETAIL_CSV, detail)
    write_rows(PUBLICATION_SUMMARY_CSV, summary)
    output_paths = []
    output_paths.extend(plot_t3000(summary))
    output_paths.extend(plot_budget_curves(detail))
    write_supp_table(summary)
    output_paths.extend([SUPP_TABLE, PUBLICATION_DETAIL_CSV, PUBLICATION_SUMMARY_CSV])
    manifest = {
        "status": "PASS",
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "AGPD_protocol_labels": {
            "H4": "joint dense-sector coefficient refit",
        },
        "publication_method_targets": {
            molecule: list(METHODS_BY_MOLECULE[molecule]) for molecule in MOLECULES
        },
        "inputs": {
            str(DETAIL_CSV.resolve()): sha256(DETAIL_CSV),
            str(SUMMARY_CSV.resolve()): sha256(SUMMARY_CSV),
            str(FC_IMA_CURVES.resolve()): sha256(FC_IMA_CURVES),
        },
        "outputs": {
            str(path.resolve()): sha256(path) for path in output_paths
        },
    }
    PLOT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[PASS] wrote {MAIN_STEM}.pdf/png and {SUPP_STEM}.pdf/png")
    print(f"[PASS] copied the SI figure to {PLOT_DIRECTORY}")
    print(f"[PASS] wrote {SUPP_TABLE.name}")


if __name__ == "__main__":
    main()
