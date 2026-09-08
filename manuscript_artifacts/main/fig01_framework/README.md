# Main Figure 1

Figure 1 is slide **1** of `reproduction/paper/original/Figure1_revised_source.pptx`.
The PPTX and its reviewed PDF/PNG are included in this branch. Slides 2 and 3
are not part of the manuscript figure. The original slide objects and equations
are preserved; only the blank page margins are cropped.

## Build the reviewed figure

This command requires Python's standard library only and no numerical inputs:

```powershell
python manuscript_artifacts/main/fig01_framework/build.py --output build/main_sketch_revise.pdf
```

It checks the PPTX, PDF and PNG hashes against
`reproduction/code/plotting/framework_provenance.json`, then writes both the
PDF and PNG. A changed source fails verification instead of silently using
an outdated PDF. The full manuscript runner invokes this same implementation.

## Export after editing the PPTX

Install `pypdf`, make `pdftoppm` (Poppler) available, and run on Windows with
PowerPoint installed:

```powershell
python manuscript_artifacts/reproduction/code/plotting/export_framework.py --mode export --output build/framework_review/main_sketch_revise.pdf
```

PowerPoint opens the PPTX read-only without a slide window and exports it to
PDF. `export_framework.py` retains its first page and applies the reviewed
crop from `framework_source.json`, preserving vector graphics. On another
platform, provide an uncropped PowerPoint PDF with `--source-pdf PATH`.
The Artifact Tool preview is not suitable as the final image because its
renderer cannot decode the embedded GPD EMF or all equation notation here.

Review the output visually. If the layout changed, update the crop rectangle
and review again. Promote the reviewed PDF/PNG to `reproduction/figures/published/`
and its adjacent `.provenance.json` to
`reproduction/code/plotting/framework_provenance.json`. Update the canonical
local archive and manuscript `figs/main_sketch_revise.pdf` with the same files,
then run the complete reproduction and `validation/export_plot_bundle.py`
to refresh bundle hashes. Keep the source PPTX byte-identical in both locations.
The exporter never edits it or promotes an unreviewed export.

## Scientific implementation mapping

- GPD projections: `src/mpqc_measurement/gpd.py`.
- SRDD orbital rotations: `src/mpqc_measurement/srdd_tensor.py`; each connected
  pair of yellow circles represents a two-mode Givens rotation, consistent
  with the even/odd nearest-neighbor layout in `givens_layout`.
- Rotated-diagonal fragments and their constant contribution:
  `src/mpqc_measurement/models.py` and `src/mpqc_measurement/srdd.py`.
- Balanced candidate selection and allocation:
  `src/mpqc_measurement/selection.py` and `allocation.py`.

The slide illustrates decomposition and a measurement circuit. Its `k ~ p_k`
block illustrates randomized term selection in the general framework; the
fixed-number-of-measurements experiments use integer circuit allocations as
specified in the manuscript. This layout update does not alter those algorithms,
fit coefficients, selected ranks, or numerical results.
