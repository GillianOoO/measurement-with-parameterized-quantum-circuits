"""Build Figure 1 from slide 1 of the editable PPTX, or verify its reviewed export.

Export requires pypdf and Poppler (pdftoppm). PowerPoint is required only when
--source-pdf is omitted. The original presentation is opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("framework_source.json")
PROVENANCE = Path(__file__).with_name("framework_provenance.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_source():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    record = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if record["config_sha256"] != sha256(CONFIG):
        raise ValueError("Figure 1 export settings changed; regenerate and review the export")
    for relative, expected in record["files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Figure 1 source/export mismatch: {relative}")
    return config, record


def copy_reviewed(pdf_output, png_output=None):
    config, record = verified_source()
    for key, target in (("pdf", pdf_output), ("png", png_output)):
        if target is None:
            continue
        target = Path(target).resolve()
        source = ROOT / config[key]
        if target == source.resolve():
            raise ValueError("Output must not overwrite the reviewed source")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return record


def export_slide(output, source_pdf=None):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = ROOT / config["pptx"]
    source_hash = sha256(source)
    output = Path(output).resolve()
    if output in {source.resolve(), (ROOT / config["pdf"]).resolve()}:
        raise ValueError("Export to a review directory before promoting the asset")
    if output.suffix.lower() != ".pdf":
        raise ValueError("Output must end in .pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="framework_", dir=output.parent) as temporary:
        if source_pdf is None:
            if os.name != "nt":
                raise RuntimeError("Supply --source-pdf exported by PowerPoint on non-Windows systems")
            source_pdf = Path(temporary) / "powerpoint.pdf"
            shell = shutil.which("pwsh") or shutil.which("powershell.exe")
            if shell is None:
                raise RuntimeError("A PowerShell host is required for PowerPoint export")
            subprocess.run([
                shell, "-NoProfile", "-File",
                str(Path(__file__).with_name("export_framework_powerpoint.ps1")),
                "-Source", str(source), "-Output", str(source_pdf),
            ], check=True, timeout=180)
        reader = PdfReader(source_pdf)
        page = reader.pages[config["slide"] - 1]
        if [float(page.mediabox.width), float(page.mediabox.height)] != config["slide_size_points"]:
            raise ValueError("Unexpected source PDF slide dimensions")
        page.mediabox = RectangleObject(config["crop_box_points"])
        page.cropbox = RectangleObject(config["crop_box_points"])
        writer = PdfWriter()
        writer.add_page(page)
        writer.add_metadata({"/Title": "Expectation value estimation framework", "/Producer": "PowerPoint export; pypdf first-slide crop"})
        with output.open("wb") as handle:
            writer.write(handle)
    subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", str(config["preview_dpi"]),
                    str(output), str(output.with_suffix(""))], check=True)
    if sha256(source) != source_hash:
        raise RuntimeError("PPTX changed during export")
    record = {
        "schema": 1, "slide": config["slide"], "config_sha256": sha256(CONFIG),
        "export_method": "PowerPoint PDF export; first slide; vector-preserving crop",
        "crop_box_points": config["crop_box_points"],
        "files": {config["pptx"]: source_hash, config["pdf"]: sha256(output),
                  config["png"]: sha256(output.with_suffix(".png"))},
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reviewed", "export"), default="reviewed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pdf", type=Path, help="Uncropped PDF exported from this PPTX by PowerPoint")
    args = parser.parse_args()
    if args.mode == "export":
        export_slide(args.output, args.source_pdf)
    else:
        if args.source_pdf:
            parser.error("--source-pdf requires --mode export")
        copy_reviewed(args.output, args.output.with_suffix(".png"))
    print(f"Figure 1: {args.output}")


if __name__ == "__main__":
    main()
