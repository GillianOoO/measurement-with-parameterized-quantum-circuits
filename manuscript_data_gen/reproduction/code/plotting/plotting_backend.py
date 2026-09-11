"""Shared plotting style and curated data bindings for the published figures."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib

_ARCHIVE_HINT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_ARCHIVE_HINT / "validation" / "matplotlib"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent.parent
PLOTS = Path(
    os.environ.get("PAPER_REPRO_FIGURE_DIR", ARCHIVE / "figures" / "rebuilt")
).resolve()

H4_SRDD = ARCHIVE / "data" / "H4" / "processed" / "bond_scan_plot_data.csv"
H4_COMPARISON = (
    ARCHIVE / "data" / "H4" / "processed" / "fixed_geometry_method_curves.csv"
)
H4_PAULI_RERUN = ARCHIVE / "data" / "H4" / "results" / "pauli_bond_scan"
H4_PAULI_SCAN = H4_PAULI_RERUN / "scan_sampling_error_summary.csv"
H4_PAULI_FIXED = H4_PAULI_RERUN / "fixed_sampling_error_summary.csv"
H4_PAULI_AUDIT = H4_PAULI_RERUN / "audit.json"
H4_PAULI_MANIFEST = H4_PAULI_RERUN / "manifest.json"
COMMON_COMPARISON = ARCHIVE / "data" / "H6" / "processed" / "method_curves.csv"
RESOURCE_CHART = (
    ARCHIVE / "data" / "shared_processed" / "resources" / "resource_chart_summary.csv"
)
STATE_VARIANCE = (
    ARCHIVE
    / "data"
    / "shared_processed"
    / "variance"
    / "state_dependent_variance_T3000.csv"
)
STATE_VARIANCE_BY_BUDGET = (
    ARCHIVE
    / "data"
    / "shared_processed"
    / "variance"
    / "supp_state_dependent_variance_by_budget.csv"
)
STATE_VARIANCE_FULL_BY_BUDGET = (
    ARCHIVE
    / "data"
    / "shared_processed"
    / "variance"
    / "state_dependent_variance_by_budget.csv"
)

COLORS = {
    "GPD": "#3F6F98",
    "SRDD": "#4F8A70",
    "FC-IMA": "#B45A45",
    "OGM": "#777472",
    "SG": "#4D4D4D",
    "Derand": "#A48974",
    "LCS": "#89796C",
    "AP": "#B89A7E",
}
MARKERS = {
    "GPD": "*",
    "SRDD": "D",
    "FC-IMA": "o",
    "OGM": "v",
    "SG": "s",
    "Derand": "^",
    "LCS": "o",
    "AP": "P",
}
LINESTYLES = {
    "GPD": "-",
    "SRDD": "-",
    "FC-IMA": (0, (3, 1, 1, 1)),
    "OGM": ":",
    "SG": "--",
    "Derand": "-.",
    "LCS": "--",
    "AP": "--",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("pdf", "png"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.03)
        paths.append(path)
    return paths


def line(
    ax: plt.Axes,
    x: Iterable[float],
    y: Iterable[float],
    method: str,
    **kwargs,
) -> None:
    highlighted = method in {"GPD", "SRDD"}
    ax.plot(
        list(x),
        list(y),
        color=COLORS[method],
        marker=MARKERS[method],
        linestyle=LINESTYLES[method],
        linewidth=1.65 if highlighted else 1.05,
        markersize=4.8 if method == "GPD" else (4.0 if highlighted else 3.4),
        markeredgewidth=0.65,
        alpha=1.0 if highlighted else 0.82,
        zorder=5 if highlighted else 2,
        label=method,
        **kwargs,
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-28, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontweight="normal",
        fontsize=10,
    )


def require_h4_pauli_rerun() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Validate and return the identical-Hamiltonian H4 Pauli schedules."""

    audit = json.loads(H4_PAULI_AUDIT.read_text(encoding="utf-8"))
    manifest = json.loads(H4_PAULI_MANIFEST.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or manifest.get("audit", {}).get("status") != "PASS":
        raise RuntimeError("The identical-input H4 Pauli rerun did not pass its audit")
    if audit.get("scan_case_count") != 21 or audit.get("scan_shot_budget") != 2038:
        raise RuntimeError("Unexpected H4 Pauli bond-scan contract")
    if audit.get("fixed_shot_budgets") != [100, 200, 300, 500, 800, 1200, 2000, 3000]:
        raise RuntimeError("Unexpected fixed-H4 Pauli measurement grid")
    for path in (H4_PAULI_SCAN, H4_PAULI_FIXED, H4_PAULI_AUDIT):
        relative = str(path.relative_to(H4_PAULI_RERUN)).replace("\\", "/")
        expected = manifest.get("output_hashes", {}).get(relative)
        if expected != _sha256(path):
            raise RuntimeError(f"H4 Pauli rerun hash mismatch for {path.name}")
    scan = _read_csv(H4_PAULI_SCAN)
    fixed = _read_csv(H4_PAULI_FIXED)
    if len(scan) != 63 or len(fixed) != 24:
        raise RuntimeError("Incomplete H4 Pauli rerun tables")
    if any(int(row["repeat_count"]) != 50 for row in scan):
        raise RuntimeError("H4 Pauli bond scan must use 50 independent replays")
    if any(int(row["repeat_count"]) != 200 for row in fixed):
        raise RuntimeError("Fixed-H4 Pauli scan must use 200 independent replays")
    return scan, fixed
