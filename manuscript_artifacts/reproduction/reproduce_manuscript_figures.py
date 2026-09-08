#!/usr/bin/env python3
"""Rebuild every figure used by the current JCTC main and SI manuscripts.

Figures 2--5 and the SI variance figure are rendered from the curated numerical
tables by the archived plotting programs. Figure 1 uses the reviewed vector
export of PPTX slide 1, verified against its source and export provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PLOTTING = HERE / "code" / "plotting"
REFERENCE = HERE / "figures" / "rebuilt"
PUBLISHED = HERE / "figures" / "published"
DEFAULT_OUTPUT = HERE / "build" / "current_figures"

NUMERICAL_STEMS = (
    "molecular_hamiltonian_errors",
    "additional_molecular_hamiltonian_errors",
    "molecular_resource_summary",
    "general_hamiltonian_gpd_average_errors",
    "supp_state_dependent_variance_vs_budget",
)
FIGURE_STEMS = dict(zip(("2", "3", "4", "5", "s1"), NUMERICAL_STEMS))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], environment: dict[str, str]) -> None:
    subprocess.run(command, check=True, cwd=PLOTTING, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", nargs="+", choices=("1", *FIGURE_STEMS),
                        default=["1", *FIGURE_STEMS], help="Default: every current main/SI figure.")
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
        help="Render without requiring byte-identical PNGs in figures/rebuilt.",
    )
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.figures))
    output = args.output.resolve()
    pdf_dir = output / "pdf"
    png_dir = output / "png"
    validation_dir = output / "validation"
    cache_dir = output / "cache" / "matplotlib"
    for directory in (pdf_dir, png_dir, validation_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(cache_dir)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PAPER_REPRO_PANELS_ONLY", None)
    environment["PAPER_REPRO_FIGURE_DIR"] = str(output / "rendered")
    environment["PAPER_REPRO_PRESENTATION_DIR"] = str(output / "presentations")
    environment["PAPER_REPRO_PLOT_MANIFEST"] = str(
        validation_dir / "plot_rebuild_manifest.json"
    )
    environment["PAPER_REPRO_FIG3_MANIFEST"] = str(validation_dir / "figure3_plot_manifest.json")
    audit_paths = []
    if "3" in selected:
        run([sys.executable, str(PLOTTING / "plot_figure_3.py")], environment)
        audit_paths.append(Path(environment["PAPER_REPRO_FIG3_MANIFEST"]))
    others = [key for key in selected if key in {"2", "4", "5", "s1"}]
    if others:
        run([sys.executable, str(PLOTTING / "plot_figures_2_4_5_and_supp.py"),
             "--figures", *others], environment)
        audit_paths.append(Path(environment["PAPER_REPRO_PLOT_MANIFEST"]))

    rendered = Path(environment["PAPER_REPRO_FIGURE_DIR"])
    comparisons: dict[str, dict[str, object]] = {}
    current_pdfs, current_pngs = [], []
    for key in selected:
        if key == "1":
            continue
        stem = FIGURE_STEMS[key]
        source_png = rendered / f"{stem}.png"
        source_pdf = rendered / f"{stem}.pdf"
        if not source_png.is_file() or not source_pdf.is_file():
            raise FileNotFoundError(f"Missing rendered artifact for {stem}")
        shutil.copy2(source_png, png_dir / source_png.name)
        shutil.copy2(source_pdf, pdf_dir / source_pdf.name)
        current_pdfs.append(pdf_dir / source_pdf.name)
        current_pngs.append(png_dir / source_png.name)
        reference_png = REFERENCE / source_png.name
        reference_hash = sha256(reference_png) if reference_png.is_file() else None
        current_hash = sha256(source_png)
        exact = reference_hash == current_hash if reference_hash else None
        comparisons[stem] = {
            "generated_png_sha256": current_hash,
            "reference_png_sha256": reference_hash,
            "byte_identical_to_rebuilt_reference": exact,
            "generated_pdf_sha256": sha256(source_pdf),
        }
        if not args.skip_reference_check and exact is not True:
            raise RuntimeError(f"PNG reference check failed for {stem}")

    schematic = PUBLISHED / "main_sketch_revise.pdf"
    framework_record = None
    if "1" in selected:
        sys.path.insert(0, str(PLOTTING))
        from export_framework import copy_reviewed
        framework_record = copy_reviewed(pdf_dir / schematic.name,
                                         png_dir / "main_sketch_revise.png")
        current_pdfs.append(pdf_dir / schematic.name)
        current_pngs.append(png_dir / "main_sketch_revise.png")

    inputs = {}
    if framework_record is not None:
        inputs.update(framework_record["files"])
    for path in audit_paths:
        inputs.update(json.loads(path.read_text(encoding="utf-8"))["inputs"])
    for relative, expected_hash in inputs.items():
        if sha256(HERE / relative) != expected_hash:
            raise RuntimeError(f"Input changed during rendering: {relative}")

    panel_comparisons = {}
    expected_panels = []
    if "3" in selected:
        expected_panels += [f"additional_molecular_errors_panel_{i}.png" for i in range(1, 7)]
    if "s1" in selected:
        expected_panels += [f"supp_variance_bias_panel_{letter}.png" for letter in "abcdef"]
    for name in expected_panels:
        actual = sha256(rendered / "panels" / name)
        reference = sha256(REFERENCE / "panels" / name)
        panel_comparisons[name] = {
            "generated_png_sha256": actual, "reference_png_sha256": reference,
            "byte_identical_to_rebuilt_reference": actual == reference,
        }
        if not args.skip_reference_check and actual != reference:
            raise RuntimeError(f"Panel reference check failed for {name}")

    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib
    import numpy
    from matplotlib.font_manager import FontProperties, findfont
    font_path = Path(findfont(FontProperties(family=["Times New Roman", "Times", "DejaVu Serif"])))

    manifest = {
        "schema": 2,
        "status": "RENDERED_UNCHECKED" if args.skip_reference_check else "PASS",
        "figures_requested": selected,
        "scope": "Plot reproduction from saved results; no new optimization or Born-outcome replay.",
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "entrypoint": Path(__file__).name,
        "entrypoint_sha256": sha256(Path(__file__)),
        "environment": {"numpy": numpy.__version__, "matplotlib": matplotlib.__version__,
                        "serif_font": font_path.name, "serif_font_sha256": sha256(font_path)},
        "input_sha256": inputs,
        "plotting_programs": {
            path.name: sha256(path)
            for path in (
                PLOTTING / "plot_figure_3.py",
                PLOTTING / "plot_figures_2_4_5_and_supp.py",
                PLOTTING / "plotting_backend.py",
            )
        },
        "numerical_figures": comparisons,
        "individual_panels": panel_comparisons,
        "figure_1": {
            "type": "reviewed vector export of PPTX slide 1",
            "included": "1" in selected,
            "source_pdf": schematic.relative_to(HERE).as_posix(),
            "source_pdf_sha256": sha256(schematic) if "1" in selected else None,
            "editable_source": "paper/original/Figure1_revised_source.pptx",
            "provenance": framework_record,
            "exporter_sha256": sha256(PLOTTING / "export_framework.py"),
        },
        "output_pdfs": {
            path.name: sha256(path) for path in current_pdfs
        },
        "output_pngs": {
            path.name: sha256(path) for path in current_pngs
        },
    }
    (validation_dir / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{manifest['status']}: rendered {len(manifest['output_pdfs'])} manuscript figures in {output}")


if __name__ == "__main__":
    main()
