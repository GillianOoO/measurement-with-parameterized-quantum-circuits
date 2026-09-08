"""Check the 2026-09-08 bias presentation without rewriting figure assets."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "Journal_chemical_theory_computation"
PRESENTATIONS = ROOT / "data/shared_processed/presentations"
FIGURE4_SHA256 = "317b1e316abe117dab31ef4807eca7ee1149086f46c45060d5bdb497ceec3218"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(left, right):
    assert math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-14), (left, right)


def main(*, si_only=False, plot_only=False, rebuild_dir=None):
    inputs = []
    molecular_path = ROOT / "data/shared_processed/variance/supp_state_dependent_variance_by_budget.csv"
    inputs.append(molecular_path)
    molecular = rows(molecular_path)
    expected_bias = {}
    expected_variance = {}
    for row in molecular:
        if row["method"] in {"AGPD", "GPD"}:
            continue
        key = row["molecule"], row["method"], int(row["T_total_shots"])
        expected_variance[key] = float(row["estimator_variance_hartree2"])
        if row["method"] == "SRDD":
            value = abs(float(row["signed_approximation_bias_hartree"]))
            expected_bias[key] = value
            same(row["analytic_MSE_hartree2"], expected_variance[key] + value**2)

    for molecule in ("BeH2", "N2"):
        relative = ("data/BeH2/results/srdd_and_pauli/BeH2/sampling_summary.csv"
                    if molecule == "BeH2" else "data/N2/results/srdd_and_pauli/sampling_summary.csv")
        path = ROOT / relative
        inputs.append(path)
        for row in rows(path):
            if row["method"] in {"SRDD", "s-RCDF"}:
                same(expected_bias[(molecule, "SRDD", int(row["T_total_shots"]))],
                     abs(float(row["deterministic_bias_hartree"])))

    for benchmark in ("sparse", "dense"):
        grouped = {}
        for instance in range(1, 6):
            path = ROOT / f"data/random_sparse_dense/results/gpd_balanced/{benchmark}4_seed0_{instance}/selected_results.csv"
            inputs.append(path)
            for row in rows(path):
                bias = float(row["bias"])
                same(bias, float(row["approximate_expectation"]) - float(row["target_expectation"]))
                same(row["mse"], float(row["variance"]) + bias**2)
                grouped.setdefault(int(row["shots"]), []).append(row)
        for total, cases in grouped.items():
            assert len(cases) == 5
            key = benchmark, "GPD", total
            expected_bias[key] = statistics.mean(abs(float(row["bias"])) for row in cases)
            expected_variance[key] = statistics.mean(float(row["variance"]) for row in cases)

    pauli_path = ROOT / "data/random_sparse_dense/results/random_all_methods_empirical50_instance.csv"
    inputs.append(pauli_path)
    grouped = {}
    for row in rows(pauli_path):
        if row["method"] == "GPD":  # Historical GPD rows are not the current protocol.
            continue
        key = row["benchmark"], row["method"], int(row["shots"])
        grouped.setdefault(key, []).append(float(row["analytic_variance"]))
    for key, values in grouped.items():
        assert len(values) == 5
        expected_variance[key] = statistics.mean(values)

    table = PRESENTATIONS / "supp_absolute_bias_by_budget.csv"
    displayed_bias = {
        key: value for key, value in expected_bias.items()
        if key[0] not in {"sparse", "dense"} or expected_variance[(key[0], "GPD", key[2])] > 0.0
    }
    actual = {(r["benchmark"], r["method"], int(r["measurements"])): r for r in rows(table)}
    assert len(actual) == len(rows(table)) == 38
    assert actual.keys() == displayed_bias.keys()
    for key, row in actual.items():
        same(row["absolute_bias"], expected_bias[key])
        assert int(row["instances"]) == (5 if key[1] == "GPD" else 1)

    # Inspect actual Matplotlib artists; suppress all figure/table writes.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation/matplotlib"))
    sys.path.insert(0, str(ROOT / "code/plotting"))
    import plot_figures_2_4_5_and_supp as plot
    from matplotlib.figure import Figure

    plot.base.configure()
    with patch.object(Figure, "savefig"), patch.object(plot, "write_csv"), patch.object(plot.plt, "close"):
        plot.render_supplement_variance()
        figure = plot.plt.gcf()
    assert len(figure.axes) == 12
    panel_order = ("H4", "H6", "BeH2", "N2", "sparse", "dense")
    for index, benchmark in enumerate(panel_order):
        upper, lower = figure.axes[2 * index:2 * index + 2]
        assert upper.get_shared_x_axes().joined(upper, lower)
        same(upper.get_position().y0, lower.get_position().y1)
        zero_label = lower.yaxis.get_major_ticks()[-1].label1
        renderer = figure.canvas.get_renderer()
        for tick in upper.yaxis.get_major_ticks() + upper.yaxis.get_minor_ticks():
            if upper.get_ylim()[0] <= tick.get_loc() <= upper.get_ylim()[1] and tick.label1.get_visible():
                assert not tick.label1.get_window_extent(renderer).overlaps(zero_label.get_window_extent(renderer))
        assert lower.get_xscale() == "log" and lower.get_yscale() == "linear"
        assert lower.get_ylim()[0] < 0.0 and lower.get_ylim()[1] == 0.0
        assert lower.yaxis.get_label_position() == "left"
        assert lower.yaxis.get_ticks_position() == "left"
        assert all(s.get_visible() for ax in (upper, lower) for s in ax.spines.values())
        assert lower.get_xlabel() == "The number of measurements"
        for line in upper.lines:
            if benchmark in {"sparse", "dense"}:
                expected_counts = sorted(key[2] for key in displayed_bias if key[0] == benchmark)
                assert [int(total) for total in line.get_xdata()] == expected_counts
                assert expected_counts[0] == 45 and 12 not in expected_counts
            for total, value in zip(line.get_xdata(), line.get_ydata()):
                same(value, expected_variance[(benchmark, line.get_label(), int(total))])
        assert len(lower.lines) == 0 and len(lower.containers) == 1
        bars = lower.containers[0]
        keys = sorted((key for key in displayed_bias if key[0] == benchmark), key=lambda key: key[2])
        assert len(bars) == len(keys)
        for bar, key in zip(bars, keys):
            assert bar.get_y() == 0.0 and bar.get_height() <= 0.0
            same(-bar.get_height(), expected_bias[key])
            # Geometric centering is appropriate on the shared logarithmic x axis.
            same(math.sqrt(bar.get_x() * (bar.get_x() + bar.get_width())), key[2])
            assert lower.get_xlim()[0] <= bar.get_x()
            assert bar.get_x() + bar.get_width() <= lower.get_xlim()[1]
    plot.plt.close(figure)

    fig4 = (ROOT / "figures/published/molecular_resource_summary.pdf" if plot_only
            else PAPER / "figs/molecular_resource_summary.pdf")
    assert digest(fig4) == FIGURE4_SHA256
    stem = "supp_state_dependent_variance_vs_budget"
    pdfs = [ROOT / f"figures/rebuilt/{stem}.pdf", ROOT / f"figures/published/{stem}.pdf"]
    if not plot_only:
        pdfs.append(PAPER / f"figs/{stem}.pdf")
    assert len({digest(path) for path in pdfs}) == 1
    for panel in "abcdef":
        name = f"supp_variance_bias_panel_{panel}.png"
        assert digest(ROOT / "figures/rebuilt/panels" / name) == digest(ROOT / "figures/panels" / name)

    documents = () if plot_only else ("main_jctc_revise", "supp_jctc_revise", "response")
    for name in documents:
        log = (PAPER / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"undefined|multiply defined|Label\(s\) may have changed|Overfull|Float too large|^!", log, re.I | re.M)

    if si_only and rebuild_dir is None:
        redraw = ROOT / "validation/si_bias_bars_rebuild_check"
        reproduction_path = redraw / "plot_rebuild_manifest.json"
        assert digest(redraw / f"figures/{stem}.png") == digest(ROOT / f"figures/rebuilt/{stem}.png")
        for panel in "abcdef":
            filename = f"supp_variance_bias_panel_{panel}.png"
            assert digest(redraw / "figures/panels" / filename) == digest(ROOT / "figures/rebuilt/panels" / filename)
    else:
        redraw = Path(rebuild_dir or ROOT / "build/current_figures").resolve()
        reproduction_path = redraw / "validation/reproduction_manifest.json"
        reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
        assert reproduction["status"] == "PASS"
        assert reproduction["entrypoint_sha256"] == digest(ROOT / "reproduce_manuscript_figures.py")
        assert "s1" in reproduction["figures_requested"]
        for relative, value in reproduction["input_sha256"].items():
            assert value == digest(ROOT / relative), f"Stale input in reproduction manifest: {relative}"
        for filename, value in reproduction["plotting_programs"].items():
            assert value == digest(ROOT / "code/plotting" / filename), f"Stale renderer: {filename}"
        assert all(item["byte_identical_to_rebuilt_reference"] for item in reproduction["numerical_figures"].values())
        for name, item in reproduction["numerical_figures"].items():
            assert item["generated_png_sha256"] == digest(redraw / f"png/{name}.png") == digest(ROOT / f"figures/rebuilt/{name}.png")
        for name, item in reproduction["individual_panels"].items():
            assert item["generated_png_sha256"] == digest(redraw / "rendered/panels" / name) == digest(ROOT / "figures/rebuilt/panels" / name)
    outputs = [table, fig4, *pdfs, *[PAPER / f"{name}.pdf" for name in documents]]
    report = {
        "status": "PASS", "tex_checked": not plot_only,
        "bias_rows": len(actual), "source_bias_rows": len(expected_bias), "panels": 6,
        "excluded_display_counts": {benchmark: [key[2] for key in expected_bias if key[0] == benchmark and key not in displayed_bias]
                                    for benchmark in ("sparse", "dense")},
        "caption_absolute_bias": {m: expected_bias[(m, "SRDD", 3000)] for m in ("BeH2", "N2")},
        "checks": ["bias versus saved deterministic expectations", "molecular and GPD MSE identities",
                   "mean absolute bias across five random instances", "all upper-curve values versus current sources",
                   "shared logarithmic x; downward bars from zero; left-hand linear magnitude axes; full boxes",
                   "upper/lower panel borders touch; junction tick labels do not overlap",
                   "all random-ensemble curves and bias bars use the same six counts starting at T=45; T=12 omitted only from SI display",
                   "main Figure 4 PDF unchanged", "published/rebuilt SI PDF synchronization",
                   "six individual panels synchronized", "fresh redraw hashes match current sources and references"],
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT / "code/plotting/plot_figures_2_4_5_and_supp.py"]},
        "output_sha256": {str(p): digest(p) for p in outputs},
        "reproduction_manifest": str(reproduction_path),
    }
    report_path = ROOT / ("validation/si_bias_bars_20260908.json" if si_only else "validation/current_bias_validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--si-only", action="store_true", help="Check the isolated SI-only redraw and SI/response compilation.")
    parser.add_argument("--plot-only", action="store_true", help="Validate plot data/layout without private TeX files or compile logs.")
    parser.add_argument("--rebuild-dir", type=Path, help="Fresh output from reproduce_manuscript_figures.py.")
    args = parser.parse_args()
    main(si_only=args.si_only, plot_only=args.plot_only, rebuild_dir=args.rebuild_dir)
