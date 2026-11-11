# Current manuscript figures

This directory is distributed on the **main** branch under
`manuscript_data_gen/reproduction/`, together with lossless numerical archives
and public figure references. From the repository root, first unpack them:

```powershell
python manuscript_data_gen/materialize_data.py
```

Expanded numerical data are gitignored; their source ZIPs are included. Run the commands below from
`manuscript_data_gen/reproduction/`, or use the repository's `build_all.py`.

```powershell
python reproduce_manuscript_figures.py --output build/current_figures
python validation/validate_bias_update.py --plot-only --rebuild-dir build/current_figures
```

The first command reproduces all five numerical figures from saved results.
It does not rerun Hamiltonian construction,
optimization, calibration or finite-measurement simulations. The second checks
SI variances and biases against their saved sources, the actual plotted artists,
and fresh render hashes. The public workflow uses `--plot-only`: private TeX
compilation is outside this release. The validator's older document-checking
mode refers to the historical manuscript filenames, not the current drafts.

The reference environment is Python 3.14.7, NumPy 2.5.2, Matplotlib 3.11.1,
with Times New Roman. The generated manifest records the actual versions and
font hash. Only NumPy and Matplotlib are needed for rendering. Other platforms
can use `--skip-reference-check` if font or renderer differences prevent exact
PNG matching; such a run is marked `RENDERED_UNCHECKED`, not verified reproduction.
Fonts are not redistributed. The algorithm package has its own Python and
dependency requirements; installing that package is unnecessary for plotting.

The approved Main 2 and 4 PDFs were created with Matplotlib 3.10.5; their
rendering can differ from the 3.11.1 rebuild references (for example automatic
log ticks in Main 4(c)). `figures/published/` preserves those approved assets,
while `figures/rebuilt/` is the tested 3.11.1 reference. Byte-identity claims
apply to the latter, not to every approved PDF. PDF metadata timestamps also
vary between runs. Main 3 and the latest 50-instance Main 5/SI 1 use the current renderer in both
locations. Main 5 assets were imported byte-for-byte from the manuscript's
`plots_figures/random_ensemble_50_20261102/`; SI 1 from
`plots_figures/si_variance_50_20261110/`. Independent rendering matched both
combined PNGs and all six SI panels before those references replaced the old
five-instance assets. The corresponding small numerical reference tables are
in `figures/reference_data/random_ensemble_50/`.

## Figure, code and data mapping

| Figure | Filename stem | Plot source and inputs |
|---|---|---|
| Main 2 | `molecular_hamiltonian_errors` | `code/plotting/plot_figures_2_4_5_and_supp.py`; H4/H6 processed tables and the audited same-Hamiltonian H4 Pauli rerun. |
| Main 3 | `additional_molecular_hamiltonian_errors` | `code/plotting/plot_figure_3.py`; empirical50 curves and their manifest in `data/shared_processed/figure3/empirical50/`, plus `fig3_extended_curves.csv` for target-error projections. |
| Main 4 | `molecular_resource_summary` | Shared plotter; `data/shared_processed/resources/resource_chart_summary.csv` and `variance/` tables. |
| Main 5 | `general_hamiltonian_gpd_average_errors` | Shared plotter delegates to `code/experiments/plot_random_ensemble.py`; `data/random_ensemble_50/` contains 50 paired instances per regime, seven budgets and 50 estimates per point. |
| SI 1 | `supp_state_dependent_variance_vs_budget` | Shared plotter plus `random_variance_ensemble50.py`; molecular variance-by-budget table, 100 current GPD selected-results files, and exact Pauli instance variances from the same 50-instance release. |

All input filenames and SHA-256 hashes are recorded in
`build/current_figures/validation/reproduction_manifest.json`. Method construction
is not changed by rendering. Historical AGPD molecular rows and historical GPD
rows in the Pauli replay table are excluded by the current plotter.

## Current display conventions

- Main 5: the arithmetic mean of 50 instance-level empirical RMSEs, each
  computed from 50 independent complete estimates; not pooled RMSE. The
  `plot_random_ensemble.py` audit independently reaggregates all 210,000
  estimates and checks paired input/state hashes and the recorded stopping rule.

- Main 3(a,c,d,f): empirical error from 50 complete estimates per point, not a
  Gaussian or analytic-error surrogate. Panels (b,e) display ten points per
  inverse-accuracy curve. Noise panels use 3000 measurements and ten rates in
  `[0, 0.003]`. No error band is drawn.
- Main 4: the plot is unchanged by the bias-caption update. BeH2 and N2 SRDD
  absolute biases are respectively `8.941657405614478e-7` and
  `2.5943606516420914e-6`. These belong in the main/response caption, not bars.
  The reviewed PDF SHA-256 is
  `317b1e316abe117dab31ef4807eca7ee1149086f46c45060d5bdb497ceec3218`.
- SI 1: each upper log-log variance panel touches its lower inverted-bias
  histogram. Both share the logarithmic measurement axis. The lower left axis
  is linear, independently scaled and zero-based. Negative plotting coordinates
  reflect the bars across zero; their lengths and labels mean nonnegative
  absolute bias, not signed bias.
- Molecular bias is `abs(Tr[rho(H_hat-H)])` for the decomposition selected at
  each measurement count. Random-ensemble bias is the mean of 50 individual
  absolute biases, not the absolute mean signed bias or the RMS bias.
- SI 1(e,f) retain the common manuscript display range T >= 45: `45,160,572,2038,7259,25848`. All methods and bias bars omit `T=12`
  in these two panels. Source data and the full variance summary retain that
  count; Main Figure 5 still displays it. The derived bias table has 38 rows. Every mean includes all 50 instances,
  including zero variances. AP uses the mean exact conditional variance over
  its 50 saved schedules, not the finite-sample spread of estimates.
- Axes have full boxes; panel labels are nonbold, outside the upper-left box
  corner and aligned with its top. Legends have no frames. Main 3--5 and SI 1
  omit Ha/Ha-squared axis suffixes. Required-sample axes read `Required samples`.

## Output locations and individual figures

```powershell
python reproduce_manuscript_figures.py --figures s1 --output build/si_only
python reproduce_manuscript_figures.py --figures 3 4 --output build/main_3_4
```

Each output directory contains `pdf/`, `png/`, `rendered/panels/`,
`presentations/`, and `validation/`. Figure 3 and SI 1 each export six individual
panels. The combined PNGs and these twelve panels are checked against
`figures/rebuilt/`. The manifest contains only figures requested in the current
run; older outputs in a reused directory are not presented as newly verified.
The focused SI command also works in an empty output directory.

Direct plotter overrides are `PAPER_REPRO_FIGURE_DIR`,
`PAPER_REPRO_PRESENTATION_DIR`, `PAPER_REPRO_PLOT_MANIFEST`, and
`PAPER_REPRO_FIG3_MANIFEST`, and `PAPER_REPRO_RANDOM_ENSEMBLE_DIR`. Without overrides, plotters write
`figures/rebuilt/` and processed presentation tables. Prefer the isolated
runner above for verification.

The private canonical TeX directory is not included in this public release.
The current response reuses Main 3, Main 4 and SI 1 assets; it has no separate
renderer or dataset.
After an intentional visual change, inspect the PDFs and explicitly identify
the approved manuscript assets and runtime-specific references before updating
either set, then run a fresh verification. Never replace source simulation
records merely to make a plotting comparison pass.

## Release manifests

`bundle_manifest.json` records source and reference hashes;
`required_inputs.json` records the current plotting/table inputs and reference
assets, including the complete 50-instance dependency set.
Their numerical contents are supplied in `datasets/*.zip`, whose independent
manifest covers the larger numerical replay dataset as well. Experiment
programs are in `code/methods/`, `code/experiments/` and `code/analysis/`.
Manuscript/reviewer documents, backup archives, caches and runtimes are excluded.
The figure runner reconstructs plots from saved results; numerical replay is a
separate operation described in those directories and in `../VERIFICATION.md`.

Dated files under `validation/` and `reproduction_20260908/` are historical run
records. They are not evidence for the current figure revision; use fresh
`build/current_figures/validation/reproduction_manifest.json` and
`validation/current_bias_validation.json`.
