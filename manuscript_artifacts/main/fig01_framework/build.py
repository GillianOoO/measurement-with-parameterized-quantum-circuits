#!/usr/bin/env python3
"""Apply the publication overlay to the source framework PDF."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def build(source: Path, target: Path, sans_font: Path, italic_font: Path) -> None:
    reader = PdfReader(str(source))
    page = reader.pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    if (width, height) != (633.0, 158.0):
        raise ValueError(f"unexpected source page size: {(width, height)}")

    pdfmetrics.registerFont(TTFont("ArtifactSans", str(sans_font)))
    pdfmetrics.registerFont(TTFont("ArtifactItalic", str(italic_font)))
    buffer = BytesIO()
    overlay = canvas.Canvas(buffer, pagesize=(width, height), pageCompression=1)

    overlay.setFillColorRGB(1, 1, 1)
    for x, y, w, h in ((28.7, 134.2, 44.0, 12.7), (20.8, 103.8, 147.0, 38.0), (99.8, 93.1, 84.0, 12.8), (205.3, 55.4, 4.6, 8.9)):
        overlay.rect(x, y, w, h, stroke=0, fill=1)
    overlay.setStrokeColorRGB(241 / 255, 241 / 255, 241 / 255)
    overlay.setLineWidth(0.65)
    overlay.roundRect(22.0, 105.2, 144.0, 34.0, 6.3, stroke=1, fill=1)

    overlay.setFillColorRGB(232 / 255, 155 / 255, 92 / 255)
    overlay.setFont("ArtifactSans", 10.91)
    overlay.drawString(29.49, 136.1, "s-RCDF")
    gpd_width = pdfmetrics.stringWidth("GPD", "ArtifactSans", 10.91)
    overlay.drawString(141.98 - gpd_width / 2, 95.9, "GPD")

    rail_color = (119 / 255, 119 / 255, 119 / 255)
    node_fill = (247 / 255, 227 / 255, 154 / 255)
    overlay.setStrokeColorRGB(*rail_color)
    overlay.setLineWidth(0.36)
    rail_y = [109.8, 117.4, 125.0, 132.6]
    for y in rail_y:
        overlay.line(56.0, y, 159.0, y)
    columns = [72.0, 94.0, 116.0, 138.0]
    pairs_by_column = [[(0, 1), (2, 3)], [(1, 2)], [(0, 1), (2, 3)], [(1, 2)]]
    overlay.setLineWidth(0.48)
    for x, pairs in zip(columns, pairs_by_column):
        for upper, lower in pairs:
            overlay.line(x, rail_y[upper], x, rail_y[lower])
            overlay.setFillColorRGB(*node_fill)
            for rail_index in (upper, lower):
                overlay.circle(x, rail_y[rail_index], 2.45, stroke=1, fill=1)

    overlay.setFillColorRGB(113 / 255, 143 / 255, 190 / 255)
    overlay.setFont("ArtifactItalic", 8.1)
    overlay.drawString(28.0, 117.8, "O")
    overlay.setFont("ArtifactItalic", 4.8)
    overlay.drawString(33.8, 116.3, "t")
    overlay.setFont("ArtifactItalic", 4.4)
    overlay.drawString(34.8, 123.6, "[d_R]")
    overlay.setFillColorRGB(197 / 255, 90 / 255, 17 / 255)
    overlay.setFont("ArtifactItalic", 6.87)
    overlay.drawString(205.95, 56.6, "k")
    overlay.showPage()
    overlay.save()

    buffer.seek(0)
    page.merge_page(PdfReader(buffer).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({"/Title": "Hamiltonian approximation and measurement framework"})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        writer.write(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sans-font", type=Path, default=Path(r"C:\Windows\Fonts\calibri.ttf"))
    parser.add_argument("--italic-font", type=Path, default=Path(r"C:\Windows\Fonts\cambriai.ttf"))
    args = parser.parse_args()
    build(args.source, args.output, args.sans_font, args.italic_font)


if __name__ == "__main__":
    main()
