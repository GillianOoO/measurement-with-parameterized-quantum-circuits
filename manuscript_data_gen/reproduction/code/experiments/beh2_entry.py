#!/usr/bin/env python3
"""BeH2 archived-JW benchmark: standalone s-RCDF versus Pauli methods.

The numerical engine is the audited H2O/N2 full-space driver.  This entrypoint
binds it to the exact inverse-JW BeH2 case and supplies a one-panel publication
figure plus an independent artifact consistency audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import large_molecule_srdd_pauli as engine


VERSION = "beh2-archived-jw-fullspace-srcdf-vs-pauli-shallow-collector-v3"
DEFAULT_OUTPUT = engine.RUNS / "beh2_srdd_pauli"
CASE_DIRECTORY = "05_BeH2_R1.33376A_ArchivedJW"
PREPARATION_SCRIPT = HERE / "prepare_beh2_input.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_comparison(rows, output: Path):
    plt = engine.plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, axis = plt.subplots(1, 1, figsize=(6.6, 4.2))
    for method in engine.METHODS:
        data = sorted(
            [row for row in rows if row["molecule"] == "BeH2" and row["method"] == method],
            key=lambda row: row["T_total_shots"],
        )
        x = np.asarray([row["T_total_shots"] for row in data])
        y = np.asarray([row["empirical_total_RMSE_hartree"] for row in data])
        lower = np.asarray([row["bootstrap_RMSE_95_lower_hartree"] for row in data])
        upper = np.asarray([row["bootstrap_RMSE_95_upper_hartree"] for row in data])
        axis.plot(
            x, y, color=engine.COLORS[method], marker=engine.MARKERS[method],
            linewidth=1.8, markersize=5.2, label=method,
        )
        axis.fill_between(x, lower, upper, color=engine.COLORS[method], alpha=0.12, linewidth=0)
        if method == "s-RCDF":
            analytic = np.asarray([row["analytic_total_RMSE_hartree"] for row in data])
            axis.plot(x, analytic, color=engine.COLORS[method], linestyle="--", linewidth=1.15)
            for xx, yy, row in zip(x, y, data):
                suffix = "+" if row.get("right_censored_at_Kmax") else ""
                label = f"K={row['selected_K']}{suffix}"
                if row.get("collector_mode") == "augment":
                    label += f", L={row.get('collector_extra_leaves', 0)}"
                axis.annotate(
                    label, (xx, yy), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=engine.COLORS[method],
                )
    axis.set_xscale("log")
    axis.set_yscale("log")
    # The 1.6 mHa chemical-accuracy line is more than one decade below every
    # curve in this shot window.  Drawing it on the same axis creates a large
    # empty band, so retain the information as an out-of-range annotation.
    axis.set_ylim(1.5e-2, 1.05e-1)
    axis.grid(which="both", alpha=0.18, linewidth=0.6)
    axis.set_xlabel("Total state preparations, $T$")
    axis.set_ylabel("Total energy RMSE (Ha)")
    axis.set_title(r"BeH$_2$, linear, $R_{\mathrm{BeH}}=1.33376$ Å", fontsize=11)
    axis.text(
        0.985, 0.025, "1.6 mHa chemical accuracy lies below displayed range",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=7.2,
        color="0.35",
    )
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        fontsize=8.5, bbox_to_anchor=(0.5, 0.895),
    )
    fig.suptitle(
        "Same archived JW Hamiltonian, exact FCI state, and coverage-enforced budgets",
        y=0.985, fontsize=10,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.76)
    png = output / "BeH2_sRCDF_vs_Pauli_RMSE.png"
    pdf = output / "BeH2_sRCDF_vs_Pauli_RMSE.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def typed_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def independent_audit(output: Path) -> dict:
    summary = typed_rows(output / "sampling_summary_all.csv")
    replicates = typed_rows(output / "sampling_replicates_all.csv")
    candidates = typed_rows(output / "srcdf_candidate_by_T_K_all.csv")
    errors: list[str] = []
    methods = set(engine.METHODS)
    shots = sorted({int(row["T_total_shots"]) for row in summary})
    if len(summary) != len(methods) * len(shots):
        errors.append("summary row count")
    if len(replicates) != len(summary) * int(summary[0]["repeat_count"]):
        errors.append("replicate row count")
    max_rmse_replay = 0.0
    for row in summary:
        key = (row["method"], int(row["T_total_shots"]))
        subset = [r for r in replicates if (r["method"], int(r["T_total_shots"])) == key]
        values = np.asarray([float(r["signed_total_error_hartree"]) for r in subset])
        replay = math.sqrt(float(np.mean(values**2)))
        mismatch = abs(replay - float(row["empirical_total_RMSE_hartree"]))
        max_rmse_replay = max(max_rmse_replay, mismatch)
    for total in shots:
        subset = [r for r in candidates if int(r["T_total_shots"]) == total]
        expected = min(subset, key=lambda r: (float(r["predicted_total_RMSE"]), int(r["K"])))
        selected = next(r for r in summary if r["method"] == "s-RCDF" and int(r["T_total_shots"]) == total)
        if int(float(selected["selected_K"])) != int(expected["K"]):
            errors.append(f"K argmin at T={total}")
        if int(float(selected.get("collector_extra_leaves", 0))) != int(
            expected.get("collector_extra_leaves", 0)
        ):
            errors.append(f"collector L argmin at T={total}")
    case_audit = json.loads((output / "BeH2" / "audit.json").read_text(encoding="utf-8"))
    if case_audit.get("status") != "PASS":
        errors.append("case audit")
    if not case_audit.get("Pauli_all_terms_hit_at_every_T"):
        errors.append("Pauli coverage")
    return {
        "status": "PASS" if not errors and max_rmse_replay < 1.0e-12 else "FAIL",
        "errors": errors,
        "summary_rows": len(summary),
        "replicate_rows": len(replicates),
        "shot_budgets": shots,
        "maximum_RMSE_replay_error_hartree": max_rmse_replay,
        "all_sRCDF_K_choices_replay_argmin": not any("K argmin" in item for item in errors),
        "all_collector_L_choices_replay_argmin": not any(
            "collector L argmin" in item for item in errors
        ),
        "all_Pauli_terms_covered_at_every_budget": bool(case_audit.get("Pauli_all_terms_hit_at_every_T")),
    }


def main() -> None:
    engine.VERSION = VERSION
    engine.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    engine.CASES = {
        "BeH2": engine.CaseSpec(
            "BeH2", CASE_DIRECTORY, "linear Be-H=1.33376 A", 6
        )
    }
    engine.plot_comparison = plot_comparison
    engine.main()
    # Resolve the actual --output used by the shared parser without reparsing it.
    output = engine.args_global.output.resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "case_preparation_script": str(PREPARATION_SCRIPT.resolve()),
        "case_preparation_script_sha256": sha256_file(PREPARATION_SCRIPT),
        "cRCDF_interpretation": "User term c-RCDF interpreted as the established shallow-RCDF (s-RCDF); no distinct c-RCDF implementation exists in the repository.",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit = independent_audit(output)
    (output / "independent_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(f"Independent audit failed: {audit}")
    output_hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "output_hashes.json"
    }
    (output / "output_hashes.json").write_text(json.dumps(output_hashes, indent=2) + "\n", encoding="utf-8")
    print(f"[independent audit] PASS {output / 'independent_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
