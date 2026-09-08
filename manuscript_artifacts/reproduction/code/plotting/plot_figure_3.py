"""Render the BeH2/N2 six-panel comparison from the curated result tables."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ARCHIVE / "validation" / "matplotlib"))

import numpy as np
from matplotlib.ticker import NullFormatter

import plotting_backend as style


plt = style.plt

BEH2 = (
    ARCHIVE
    / "data"
    / "BeH2"
    / "results"
    / "srdd_and_pauli"
    / "BeH2"
    / "sampling_summary.csv"
)
N2 = (
    ARCHIVE
    / "data"
    / "N2"
    / "results"
    / "srdd_and_pauli"
    / "sampling_summary.csv"
)
EXTENDED = (
    ARCHIVE
    / "data"
    / "shared_processed"
    / "figure3"
    / "fig3_extended_curves.csv"
)
EMPIRICAL = ARCHIVE / "data" / "shared_processed" / "figure3" / "empirical50"
METHODS = ("SRDD", "FC-IMA", "OGM", "SG", "Derand")


def manifest_inputs() -> list[Path]:
    return [EXTENDED, EMPIRICAL / "manifest.json", EMPIRICAL / "fig3_empirical_curves.csv"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def method_name(value: str) -> str:
    return "SRDD" if value in {"s-RCDF", "SRDD"} else value


def line(ax, x, y, method: str, **kwargs) -> None:
    style.line(ax, x, y, method, **kwargs)


def main() -> None:
    style.configure()
    manifest = json.loads((EMPIRICAL / "manifest.json").read_text())
    empirical_path = EMPIRICAL / "fig3_empirical_curves.csv"
    if manifest["status"] != "PASS" or manifest["repeats_per_point"] != 50:
        raise RuntimeError("Figure 3 requires a validated 50-repeat empirical replay")
    if hashlib.sha256(empirical_path.read_bytes()).hexdigest() != manifest["outputs"][empirical_path.name]:
        raise RuntimeError("Empirical Figure 3 data hash mismatch")
    empirical = read_csv(empirical_path)
    if len(empirical) != 150 or any(int(row["repeat_count"]) != 50 for row in empirical):
        raise RuntimeError("Incomplete Figure 3 empirical error data")
    extended = read_csv(EXTENDED)
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.65))

    for row_index, molecule in enumerate(("BeH2", "N2")):
        ax = axes[row_index, 0]
        for method in METHODS:
            group = sorted(
                (row for row in empirical
                 if row["molecule"] == molecule
                 and row["curve"] == "error_vs_measurements"
                 and row["method"] == method),
                key=lambda row: float(row["x"]),
            )
            x = [float(row["x"]) for row in group]
            y = [float(row["y"]) for row in group]
            if not group:
                raise RuntimeError(f"Missing error curve for {molecule}/{method}")
            line(ax, x, y, method)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("The number of measurements")
        ax.set_ylabel("Error")

        ax = axes[row_index, 1]
        inverse = [
            row
            for row in extended
            if row["molecule"] == molecule
            and row["curve"] == "estimated_measurements_vs_error"
        ]
        for method in METHODS:
            group = sorted(
                (row for row in inverse if method_name(row["method"]) == method),
                key=lambda row: float(row["x"]),
            )
            if not group:
                raise RuntimeError(f"Missing inverse curve for {molecule}/{method}")
            indices = np.linspace(0, len(group) - 1, 10, dtype=int)
            shown = [group[index] for index in indices]
            line(
                ax,
                [float(row["x"]) for row in shown],
                [float(row["y"]) for row in shown],
                method,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01, 0.5)
        ax.set_ylim(top={"BeH2": 2.0e5, "N2": 2.0e7}[molecule])
        ax.set_xticks((0.01, 0.03, 0.1, 0.3, 0.5), labels=("0.01", "0.03", "0.1", "0.3", "0.5"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Target error")
        ax.set_ylabel("Required samples")

        ax = axes[row_index, 2]
        noisy = [
            row
            for row in empirical
            if row["molecule"] == molecule
            and row["curve"] == "error_vs_depolarizing_rate"
        ]
        for method in METHODS:
            group = sorted(
                (row for row in noisy if method_name(row["method"]) == method),
                key=lambda row: float(row["x"]),
            )
            if not group:
                raise RuntimeError(f"Missing noise curve for {molecule}/{method}")
            line(
                ax,
                [float(row["x"]) for row in group],
                [float(row["y"]) for row in group],
                method,
                markevery=1,
            )
        ax.set_yscale("log")
        ax.set_xlim(0.0, 0.003)
        ax.set_xticks((0.0, 0.001, 0.002, 0.003))
        ax.set_xlabel("Depolarizing error rate $p$")
        ax.set_ylabel("Error")

        title = r"BeH$_2$" if molecule == "BeH2" else r"N$_2$"
        for column_index in range(3):
            current = axes[row_index, column_index]
            current.set_title(title, pad=5)
            current.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.38)
            current.tick_params(direction="in", top=True, right=True)
            style.panel_label(current, f"({chr(ord('a') + 3 * row_index + column_index)})")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        columnspacing=0.95,
        handlelength=2.5,
        frameon=False,
    )
    fig.subplots_adjust(left=0.095, right=0.97, bottom=0.105, top=0.895, hspace=0.50, wspace=0.54)
    if not os.environ.get("PAPER_REPRO_PANELS_ONLY"):
        style.save(fig, style.PLOTS, "additional_molecular_hamiltonian_errors")

    # Keep the archive's individual panels synchronized with the combined figure.
    for legend in list(fig.legends):
        legend.remove()
    fig.set_size_inches(3.6, 3.2)
    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=7, frameon=False,
               columnspacing=1.0, handlelength=2.3)
    for current in axes.flat:
        current.set_visible(False)
    for index, current in enumerate(axes.flat, 1):
        current.set_visible(True)
        current.set_position((0.23, 0.19, 0.73, 0.59))
        style.save(fig, style.PLOTS / "panels",
                   f"additional_molecular_errors_panel_{index}")
        current.set_visible(False)
    plt.close(fig)
    audit_path = Path(os.environ.get(
        "PAPER_REPRO_FIG3_MANIFEST", ARCHIVE / "validation/figure3_plot_manifest.json"))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "schema": 1, "status": "PASS", "figures_rendered": ["3"],
        "renderer": Path(__file__).relative_to(ARCHIVE).as_posix(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {p.relative_to(ARCHIVE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in manifest_inputs()},
        "error_data": "50 finite-measurement repetitions per point",
        "inverse_curve_display_points_per_method": 10,
        "noise_range": [0.0, 0.003], "noise_measurements": 3000,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
