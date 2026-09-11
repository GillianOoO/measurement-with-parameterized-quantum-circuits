"""Export an allowlisted, hash-tracked plotting snapshot; never deletes files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--rebuild-dir", type=Path, default=ROOT / "build/current_figures")
    parser.add_argument("--apply", action="store_true", help="Without this flag, report the allowlist only.")
    parser.add_argument("--include-local-inputs", action="store_true",
                        help="Also copy numerical inputs/assets locally. These must remain excluded from GitHub without explicit publication approval.")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination == ROOT or destination in ROOT.parents:
        raise ValueError("Destination must not be the source archive or its ancestor")
    report = json.loads((args.rebuild_dir / "validation/reproduction_manifest.json").read_text())
    if report["status"] != "PASS" or set(report["figures_requested"]) != {"1", "2", "3", "4", "5", "s1"}:
        raise ValueError("A complete, verified current rebuild is required")
    if report["entrypoint_sha256"] != sha(ROOT / "reproduce_manuscript_figures.py"):
        raise ValueError("Rebuild report predates the current runner")
    if report["figure_1"]["exporter_sha256"] != sha(ROOT / "code/plotting/export_framework.py"):
        raise ValueError("Rebuild report predates the current framework exporter")
    for name, value in report["plotting_programs"].items():
        if sha(ROOT / "code/plotting" / name) != value:
            raise ValueError(f"Rebuild report predates {name}")
    selected = set(report["input_sha256"])
    for relative, value in report["input_sha256"].items():
        if sha(ROOT / relative) != value:
            raise ValueError(f"Changed source: {relative}")
    selected.update({
        "reproduce_manuscript_figures.py", "FIGURE_REPRODUCTION.md",
        "code/plotting/plot_figure_3.py", "code/plotting/plot_figures_2_4_5_and_supp.py",
        "code/plotting/plotting_backend.py", "code/plotting/README.md",
        "validation/validate_bias_update.py", "validation/export_plot_bundle.py",
        "code/plotting/export_framework.py", "code/plotting/export_framework_powerpoint.ps1",
        "code/plotting/framework_source.json", "code/plotting/framework_provenance.json",
        "data/BeH2/results/srdd_and_pauli/BeH2/sampling_summary.csv",
        "data/N2/results/srdd_and_pauli/sampling_summary.csv",
        "data/shared_processed/presentations/supp_absolute_bias_by_budget.csv",
        "data/shared_processed/presentations/random_state_dependent_variance_by_budget.csv",
        "data/shared_processed/variance/README.md",
        "figures/published/main_sketch_revise.pdf", "figures/published/main_sketch_revise.png",
        "paper/original/Figure1_revised_source.pptx",
    })
    for stem, item in report["numerical_figures"].items():
        for location in ("rebuilt", "published"):
            for extension in ("pdf", "png"):
                relative = f"figures/{location}/{stem}.{extension}"
                selected.add(relative)
                if location == "rebuilt" and extension == "png" and sha(ROOT / relative) != item["generated_png_sha256"]:
                    raise ValueError(f"Stale figure reference: {relative}")
                if location == "published" and extension == "pdf":
                    tex_asset = ROOT.parent / "Journal_chemical_theory_computation/figs" / f"{stem}.pdf"
                    if tex_asset.exists() and sha(ROOT / relative) != sha(tex_asset):
                        raise ValueError(f"Published PDF differs from current TeX asset: {relative}")
    for name, item in report["individual_panels"].items():
        for location in ("rebuilt/panels", "panels"):
            relative = f"figures/{location}/{name}"
            if sha(ROOT / relative) != item["generated_png_sha256"]:
                raise ValueError(f"Stale panel reference: {relative}")
            selected.add(relative)
    hashes = {relative: sha(ROOT / relative) for relative in sorted(selected)}
    public_framework = {"paper/original/Figure1_revised_source.pptx",
                        "figures/published/main_sketch_revise.pdf",
                        "figures/published/main_sketch_revise.png"}
    external = {p: h for p, h in hashes.items()
                if p.split("/")[0] in {"data", "figures", "paper"} and p not in public_framework}
    code_files = {p: h for p, h in hashes.items() if p not in external}
    copied = hashes if args.include_local_inputs else code_files
    previous_path = destination / "bundle_manifest.json"
    previous = json.loads(previous_path.read_text()).get("files", {}) if previous_path.exists() else {}
    for relative, value in copied.items():
        target = destination / relative
        if target.exists() and sha(target) not in {value, previous.get(relative)}:
            raise ValueError(f"Preserving an unrecognized destination edit: {target}")
    manifest = {
        "schema": 2, "scope": "Current plotting code and framework PPTX/PDF/PNG; numerical inputs and other assets are supplied locally",
        "source_archive": "paper_reproducibility", "files": code_files,
        "required_inputs": "required_inputs.json",
        "reference_environment": report["environment"],
        "not_included": ["private manuscript/reviewer documents", "backup ZIPs", "runtimes and caches",
                         "full raw optimization and measurement replay archive"],
    }
    if args.apply:
        for relative in copied:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Leave an identical user source untouched, including PPTX files
            # that are currently open in PowerPoint.
            if not target.is_file() or sha(target) != hashes[relative]:
                shutil.copy2(ROOT / relative, target)
        previous_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "required_inputs.json").write_text(json.dumps({
            "schema": 1, "scope": "Required local filenames and hashes only; no numerical table or audit contents",
            "files": external}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"mode": "copied" if args.apply else "preview", "files": len(copied),
                      "required_local_inputs": len(external),
                      "bytes": sum((ROOT / relative).stat().st_size for relative in copied),
                      "destination": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
