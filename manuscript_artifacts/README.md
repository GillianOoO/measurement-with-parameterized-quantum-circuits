# Current manuscript artifacts

The current main-text Figures 1--5 and SI Figure 1 use the canonical archived
plotters in `reproduction/code/plotting/`. Their required numerical inputs, audits and reviewed assets are supplied
from the user's local classified archive. This branch publishes code and input
filenames/hashes and the Figure 1 PPTX/PDF/PNG, not the detailed research datasets
or audit records.

## Rebuild and validate

Only NumPy and Matplotlib are needed for the figure rebuild. The reference
environment is Python 3.14.7, NumPy 2.5.2 and Matplotlib 3.11.1 with Times New
Roman; the font is not redistributed. The method package has separate
installation requirements.

```powershell
python manuscript_artifacts/prepare_inputs.py --archive-root PATH_TO_paper_reproducibility
python manuscript_artifacts/build_all.py --output build/manuscript
python manuscript_artifacts/reproduction/validation/validate_bias_update.py --plot-only --rebuild-dir build/manuscript
```

Code, documentation, framework assets and locally imported input/asset hashes
are checked before rendering. Imported numerical data, audits and other figure
assets are gitignored. Five numerical
combination PNGs and twelve Figure 3/SI panels are checked against the archived
rebuild references. Figure 1 uses the source-verified first-slide PDF/PNG export;
the editable PPTX is included. See [its export instructions](main/fig01_framework/README.md).
Output PDFs, PNGs, individual panels, presentation CSVs, input
hashes and runtime information are saved in the chosen output directory.

Use `--figures s1` for only SI Figure 1, or an individual artifact command:

```powershell
python manuscript_artifacts/build.py manuscript_artifacts/supp/fig_s1_state_dependent_variance --output build/supp_state_dependent_variance_vs_budget.pdf
```

A different font/Matplotlib environment may require `--skip-reference-check`;
this produces `RENDERED_UNCHECKED`, not a verified visual match.
The approved Main 2/4/5 PDFs were created with Matplotlib 3.10.5 and are retained
unchanged under `reproduction/figures/published/`. They can differ in rendering
from the 3.11.1 references under `figures/rebuilt/`. Byte-identity checks
refer to the rebuild references, not to all approved PDFs.

## Current changes and source selection

- Figure 3 reads the current empirical50 error/noise data, displays ten inverse
  accuracy points per curve, and uses the approved labels and frameless legend.
- Figure 4 keeps its three-panel plot; the BeH2/N2 SRDD bias values are caption
  metadata in its artifact description.
- SI Figure 1 contains joined variance/inverted-bias panels with left bias
  axes. SI (e,f) omit T=12 from all comparison curves and bars; Main 5 and the
  raw records retain it. The derived bias CSV contains 38 values.
- Main 3--5/SI axes omit Ha/Ha-squared suffixes, required-sample labels are
  `Required samples`, and panel letters are nonbold.
- Random GPD data use stabilizer-calibrated balanced prefix selection; the
  historical GPD rows in the Pauli replay CSV are not used as current results.

This is plot reproduction from frozen, audited results, not a fresh
Hamiltonian optimization, calibration or Born-outcome replay. The algorithm
package in `src/mpqc_measurement/` is unchanged by the plotting update.
The published code does not contain numerical tables, research-audit records,
numerical figure binaries, the complete simulation archive or private manuscript/reviewer
documents. `required_inputs.json` lists local filenames and hashes only. See
[FIGURE_REPRODUCTION.md](reproduction/FIGURE_REPRODUCTION.md) for the full
figure-to-data mapping, statistical definitions and local TeX/response paths.

## Artifact folders

- `main/fig01_framework`: first-slide SRDD/GPD schematic, editable PPTX and vector PDF exporter.
- `main/fig02_molecular_errors`: H4/H6 errors.
- `main/fig03_additional_molecules`: BeH2/N2 error, inverse accuracy and noise.
- `main/fig04_resource_summary`: variance, measurement settings and samples.
- `main/fig05_random_hamiltonian_errors`: sparse/dense empirical errors.
- `supp/fig_s1_state_dependent_variance`: exact variance and absolute bias.
- `supp/table_s1_state_dependent_variance`, `table_s2_required_measurements`,
  `table_s3_cnot_counts`: existing CSV-to-TeX table builders. These retain
  their explicit `--input` interfaces; they are not new plots or new data.

Each figure's `artifact.json` routes to the canonical renderer, preventing
the former generic line/bar templates from silently drawing an older layout.
