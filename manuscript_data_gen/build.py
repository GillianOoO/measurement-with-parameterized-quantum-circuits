#!/usr/bin/env python3
"""Build one manuscript artifact from a machine-readable CSV and artifact.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any


METHOD_STYLES = {
    "GPD": {"color": "#3F6F98", "marker": "*", "linestyle": "-"},
    "SRDD": {"color": "#4F8A70", "marker": "D", "linestyle": "-"},
    "FC-IMA": {"color": "#B45A45", "marker": "o", "linestyle": (0, (3, 1, 1, 1))},
    "OGM": {"color": "#777472", "marker": "v", "linestyle": ":"},
    "SG": {"color": "#4D4D4D", "marker": "s", "linestyle": "--"},
    "Derand": {"color": "#A48974", "marker": "^", "linestyle": "-."},
    "LCS": {"color": "#89796C", "marker": "o", "linestyle": "--"},
    "AP": {"color": "#B89A7E", "marker": "P", "linestyle": "--"},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_fields(rows: list[dict[str, str]], fields: set[str]) -> None:
    if not rows:
        raise ValueError("input CSV is empty")
    missing = fields.difference(rows[0])
    if missing:
        raise ValueError(f"input CSV is missing fields: {sorted(missing)}")


def validate_row_requirements(config: dict[str, Any], rows: list[dict[str, str]]) -> None:
    for rule in config.get("row_requirements", []):
        selected = [
            row
            for row in rows
            if all(row.get(field) == value for field, value in rule.get("when", {}).items())
        ]
        if not selected:
            raise ValueError(f"no rows satisfy required selector {rule.get('when', {})}")
        for row in selected:
            for field, value in rule.get("require", {}).items():
                if row.get(field) != value:
                    raise ValueError(
                        f"row requirement failed for {rule.get('when', {})}: "
                        f"{field}={row.get(field)!r}, expected {value!r}"
                    )


def configure_matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    return plt


def panel_rows(
    rows: list[dict[str, str]], panel_field: str, panel_value: str, filters: dict[str, str]
) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get(panel_field) == panel_value]
    for field, value in filters.items():
        selected = [row for row in selected if row.get(field) == value]
    return selected


def build_line_figure(config: dict[str, Any], rows: list[dict[str, str]], output: Path) -> None:
    plt = configure_matplotlib()
    panels = config["panels"]
    required = {config["panel_field"], config["method_field"]}
    for panel in panels:
        required.update((panel["x_field"], panel["y_field"]))
    require_fields(rows, required)
    validate_row_requirements(config, rows)

    ncols = int(config.get("ncols", len(panels)))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=tuple(config.get("figsize", [3.6 * ncols, 2.8 * nrows])),
        squeeze=False,
    )
    handles: dict[str, Any] = {}
    for index, panel in enumerate(panels):
        ax = axes.flat[index]
        selected = panel_rows(
            rows,
            config["panel_field"],
            panel["value"],
            panel.get("filters", {}),
        )
        methods = panel.get("methods") or sorted({row[config["method_field"]] for row in selected})
        for method in methods:
            method_rows = [row for row in selected if row[config["method_field"]] == method]
            method_rows.sort(key=lambda row: float(row[panel["x_field"]]))
            if not method_rows:
                continue
            style = dict(METHOD_STYLES.get(method, {}))
            style.update(panel.get("style_overrides", {}).get(method, {}))
            highlighted = method in {"GPD", "SRDD"}
            line = ax.plot(
                [float(row[panel["x_field"]]) for row in method_rows],
                [float(row[panel["y_field"]]) for row in method_rows],
                label=method,
                linewidth=1.65 if highlighted else 1.05,
                markersize=4.8 if method == "GPD" else (4.0 if highlighted else 3.4),
                markeredgewidth=0.65,
                alpha=1.0 if highlighted else 0.82,
                zorder=5 if highlighted else 2,
                **style,
            )[0]
            handles.setdefault(method, line)
        ax.set_title(panel["title"])
        ax.set_xlabel(panel.get("xlabel", config.get("xlabel", "The number of measurements")))
        ax.set_ylabel(panel.get("ylabel", config.get("ylabel", "Error")))
        ax.set_xscale(panel.get("xscale", config.get("xscale", "linear")))
        ax.set_yscale(panel.get("yscale", config.get("yscale", "log")))
        if panel.get("grid", config.get("grid", False)):
            ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.38)
        ax.annotate(
            f"({chr(97 + index)})",
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-28, 0),
            textcoords="offset points",
            ha="left",
            va="top",
        )
    for ax in axes.flat[len(panels) :]:
        ax.remove()
    order = config.get("legend_methods", list(handles))
    legend_handles = [handles[name] for name in order if name in handles]
    if legend_handles:
        fig.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc="upper center",
            ncol=min(len(legend_handles), int(config.get("legend_ncol", 6))),
            bbox_to_anchor=(0.5, 1.01),
        )
    if "subplots_adjust" in config:
        fig.subplots_adjust(**config["subplots_adjust"])
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight", pad_inches=0.03)
    if output.suffix.lower() == ".pdf":
        fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_bar_figure(config: dict[str, Any], rows: list[dict[str, str]], output: Path) -> None:
    plt = configure_matplotlib()
    panel_field = config["panel_field"]
    method_field = config["method_field"]
    require_fields(rows, {panel_field, method_field, *[p["y_field"] for p in config["panels"]]})
    panels = config["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=tuple(config.get("figsize", [11, 3.1])))
    axes = [axes] if len(panels) == 1 else list(axes)
    methods = config["methods"]
    labels = config["panel_values"]
    width = 0.8 / len(methods)
    for p_index, (ax, panel) in enumerate(zip(axes, panels)):
        for m_index, method in enumerate(methods):
            values = []
            for label in labels:
                matches = [
                    row
                    for row in rows
                    if row.get(panel_field) == label and row.get(method_field) == method
                ]
                values.append(float(matches[0][panel["y_field"]]) if matches and matches[0][panel["y_field"]] else math.nan)
            positions = [index - 0.4 + width / 2 + m_index * width for index in range(len(labels))]
            ax.bar(positions, values, width=width, color=METHOD_STYLES.get(method, {}).get("color"), label=method)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title(panel["title"])
        ax.set_ylabel(panel["ylabel"])
        ax.set_yscale(panel.get("yscale", "log"))
        ax.text(0.02, 0.96, f"({chr(97 + p_index)})", transform=ax.transAxes, va="top")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(methods), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if output.suffix.lower() == ".pdf":
        fig.savefig(output.with_suffix(".png"), dpi=350)
    plt.close(fig)


def format_value(value: str, spec: str) -> str:
    if value == "":
        return r"\textemdash"
    if spec == "int":
        return f"{int(round(float(value))):,}"
    if spec == "sci":
        return f"${float(value):.3e}$"
    return value.replace("_", r"\_")


def build_table(config: dict[str, Any], rows: list[dict[str, str]], output: Path) -> None:
    fields = {column["field"] for column in config["columns"]}
    require_fields(rows, fields)
    filters = config.get("filters", {})
    selected = rows
    for field, accepted in filters.items():
        accepted_values = accepted if isinstance(accepted, list) else [accepted]
        selected = [row for row in selected if row.get(field) in accepted_values]
    sort_fields = config.get("sort", [])
    selected.sort(key=lambda row: tuple(row.get(field, "") for field in sort_fields))
    headers = " & ".join(column["header"] for column in config["columns"])
    body = [
        " & ".join(format_value(row[column["field"]], column.get("format", "text")) for column in config["columns"])
        + r" \\"
        for row in selected
    ]
    align = config.get("align", "l" * len(config["columns"]))
    payload = "\n".join(
        [
            f"\\begin{{tabular}}{{{align}}}",
            r"\hline",
            headers + r" \\",
            r"\hline",
            *body,
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")


def build_matrix_table(config: dict[str, Any], rows: list[dict[str, str]], output: Path) -> None:
    row_field = config["row_field"]
    column_field = config["column_field"]
    value_field = config["value_field"]
    require_fields(rows, {row_field, column_field, value_field})
    row_values = config["rows"]
    column_values = config["columns"]
    lookup = {(row[row_field], row[column_field]): row[value_field] for row in rows}
    body = []
    for row_value in row_values:
        cells = [row_value]
        for column_value in column_values:
            cells.append(
                format_value(
                    lookup.get((row_value, column_value), ""),
                    config.get("format", "text"),
                )
            )
        body.append(" & ".join(cells) + r" \\")
    align = config.get("align", "l" + "r" * len(column_values))
    payload = "\n".join(
        [
            f"\\begin{{tabular}}{{{align}}}",
            r"\hline",
            "Benchmark & " + " & ".join(column_values) + r" \\",
            r"\hline",
            *body,
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path, help="artifact directory containing artifact.json")
    parser.add_argument("--input", type=Path, help="Required for CSV table/custom builders; not used for current frozen figures.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-reference-check", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.artifact / "artifact.json").read_text(encoding="utf-8"))
    if config["kind"] == "canonical":
        if args.input is not None:
            parser.error("Current figure builders use the bundled audited data; omit --input. See the artifact README.")
        if args.output.suffix.lower() != ".pdf":
            parser.error("Current figure output must be a .pdf path; a PNG is also emitted.")
        from build_all import verify_bundle
        verify_bundle()
        bundle = Path(__file__).resolve().parent / "reproduction"
        if args.output.resolve().is_relative_to(bundle):
            parser.error("Write outputs outside the frozen reproduction bundle")
        work = args.output.resolve().parent / (args.output.stem + "_rebuild")
        command = [sys.executable, str(bundle / "reproduce_manuscript_figures.py"),
                   "--figures", config["figure_id"], "--output", str(work)]
        if args.skip_reference_check:
            command.append("--skip-reference-check")
        subprocess.run(command, check=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / "pdf" / config["filename"], args.output)
        preview = work / "png" / Path(config["filename"]).with_suffix(".png")
        if preview.exists():
            shutil.copy2(preview, args.output.with_suffix(".png"))
        return
    if args.input is None:
        parser.error("--input is required for CSV table/custom builders")
    rows = read_rows(args.input)
    if config["kind"] == "line":
        build_line_figure(config, rows, args.output)
    elif config["kind"] == "bar":
        build_bar_figure(config, rows, args.output)
    elif config["kind"] == "table":
        build_table(config, rows, args.output)
    elif config["kind"] == "matrix_table":
        build_matrix_table(config, rows, args.output)
    else:
        raise ValueError(f"unsupported artifact kind: {config['kind']}")


if __name__ == "__main__":
    main()
