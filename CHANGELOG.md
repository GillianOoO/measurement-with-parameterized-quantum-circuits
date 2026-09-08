# Changelog

## Framework first-slide export, 2026-09-08

- Use the supplied PPTX first slide for Figure 1, preserving its objects and
  equations and cropping only blank margins from the vector PDF export.
- Include the source PPTX and reviewed PDF/PNG, PowerPoint export/crop scripts,
  source/export hash checks and tests for stale-source detection.
- Keep numerical datasets private and retain the current GPD/SRDD numerical
  implementations, fixed-measurement selection rules and Figures 2--5/SI.

## Manuscript figures, 2026-09-08

- Route all six figure commands to the current archived renderers and supply
  a code-only release with a hash-indexed importer for 91 local inputs/assets.
- Include current Figure 3 empirical50 curves, ten displayed inverse-accuracy
  points, current axis labels and the reviewed noise range.
- Add SI joined variance/inverted-bias panels with left magnitude axes and a
  common T>=45 display grid for all methods in panels (e,f); retain raw T=12.
- Preserve reviewed Figure 1 and Figure 4 assets; record Figure 4 caption biases.
- Add isolated all-figure and individual-figure commands, current provenance,
  regression tests, output mapping and renderer-version caveats.
- Keep the GPD/SRDD algorithm package and numerical source records unchanged.

## 0.1.0

- Added a self-contained GPD library and CLI for arbitrary explicit Hermitian matrices.
- Added the SRDD electronic-integral backend with shallow tensor fitting, F3–R2 redistribution, and depth-matched one-body completion.
- Implemented the SRDD universal product-Pauli residual-completion wrapper.
- Added exact integer fixed-shot allocation, stabilizer-calibrated selection, rigorous state-independent error bounds, and optional state-resolved analytic RMSE.
- Added versioned dense, Pauli JSON, and electronic-integral input contracts.
