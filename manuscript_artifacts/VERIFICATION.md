# Verification of the 2026-09-08 figure update

- The standalone repository checkout rebuilt all five numerical figures from
  locally imported saved results. Their PNGs and all twelve Figure 3/SI panel PNGs
  matched the Matplotlib 3.11.1 rebuild references byte-for-byte.
- Figure 1's copied reviewed PDF matched its source hash.
- The SI validator passed all 38 displayed / 40 source bias checks, source MSE
  identities, variance curves, joined axes, bar direction and magnitude labels.
- The independent SI-only artifact command succeeded in a fresh output
  directory. It did not require an earlier manifest.
- Three artifact regression tests passed; the eight code and 91 local-input file hashes matched.
- The full package suite ran 33 tests in the plotting environment: 31 passed;
  two existing numerical GPD tests could not run because optional JAX was not
  installed. The GPD/SRDD package sources were not changed. Run the full suite
  in the documented package environment with `.[gpd]` installed.

These checks cover figure reconstruction from archived results, not a new
optimization or finite-measurement simulation. Approved Main 2/4/5 PDFs were
produced with Matplotlib 3.10.5 and are preserved; see the renderer-version
qualification in `reproduction/FIGURE_REPRODUCTION.md`.

Commands are in `README.md`. Each new run writes its own input, code, font and
output hashes. This note records the tested update and is not a substitute for
rerunning those checks after future edits.
