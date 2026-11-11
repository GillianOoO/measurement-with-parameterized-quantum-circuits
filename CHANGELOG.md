# Changelog

## Current experiment release, 2026-11-11

- Add the completed 50-sparse/50-dense experiment with lossless inputs,
  GPD fits and selections, Pauli schedules, and 210,000 individual estimates.
- Update Main Figure 5 and SI Figure 1(e,f), retaining the original five-case
  data as regression fixtures rather than the active ensemble.
- Add portable experiment drivers, independent saved-data audits, setting-count
  tables and fixed-Frobenius-tolerance diagnostics.
- Clarify the two molecular SRDD calibration/completion profiles without
  changing their numerical algorithms or saved results.
- Verify all numerical figure/table builds in an isolated repository tree;
  exclude generated font caches from the source dependency manifest.
- Distinguish saved-result reconstruction from optional de novo computation.

## Manuscript figures, 2026-09-08

- Route all numerical figure commands to the current archived renderers and supply
  a code-only release with a hash-indexed importer for 91 local inputs/assets.
- Include current Figure 3 empirical50 curves, ten displayed inverse-accuracy
  points, current axis labels and the reviewed noise range.
- Add SI joined variance/inverted-bias panels with left magnitude axes and a
  common T>=45 display grid for all methods in panels (e,f); retain raw T=12.
- Preserve reviewed Figure 4 assets; record Figure 4 caption biases.
- Add isolated all-figure and individual-figure commands, current provenance,
  regression tests, output mapping and renderer-version caveats.
- Keep the GPD/SRDD algorithm package and numerical source records unchanged.

## 0.1.0

- Added a self-contained GPD library and CLI for arbitrary explicit Hermitian matrices.
- Added the SRDD electronic-integral backend with shallow tensor fitting, F3–R2 redistribution, and depth-matched one-body completion.
- Implemented the SRDD universal product-Pauli residual-completion wrapper.
- Added exact integer fixed-shot allocation, stabilizer-calibrated selection, rigorous state-independent error bounds, and optional state-resolved analytic RMSE.
- Added versioned dense, Pauli JSON, and electronic-integral input contracts.
