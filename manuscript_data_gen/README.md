# Manuscript data generation and reproduction

This directory on the repository's **main branch** contains the inputs, saved
numerical results, method implementations, and plotting programs for the
updated main manuscript and Supporting Information. The arXiv paper
[2407.19499](https://arxiv.org/abs/2407.19499) is to be updated.

## 1. Unpack and verify the numerical inputs

Run from the repository root, with Python 3.12 or later:

```powershell
python manuscript_data_gen/materialize_data.py
python manuscript_data_gen/materialize_data.py --verify-only
```

No external archive, login, or network download is required after cloning.
The ZIP shards in `reproduction/datasets/` contain the numerical inputs/results and metadata
files. `datasets/manifest.json` records every original filename, byte size,
SHA-256 hash and archive. Extraction preserves original bytes and refuses to
overwrite a different existing file. The expanded `reproduction/data/` is
gitignored to avoid storing a second copy. It requires approximately 1.2 GB.
The 256 MB H6 dense Hamiltonian is losslessly compressed rather than truncated.

The numerical release includes Hamiltonians, electronic integrals, reference
states, optimized factors/angles, calibration and selection records, Pauli/FC
schedules, repetitions, and processed figure tables. Historical rows in frozen
CSV files remain intact; current plotters select only the reported methods.
Private manuscript/response drafts, revision notes, logs, caches and QA
screenshots are excluded.

## 2. Reconstruct the main and SI numerical figures

```powershell
python -m pip install -e ".[artifacts]"
python manuscript_data_gen/build_all.py --output build/manuscript
python manuscript_data_gen/reproduction/validation/validate_bias_update.py --plot-only --rebuild-dir build/manuscript
```

The reference renderer is Python 3.14.7, NumPy 2.5.2 and Matplotlib 3.11.1 with
Times New Roman (the font is not redistributed). Python 3.12 can also render,
but different library/font versions can change pixels. In that case use
`--skip-reference-check`; the result is explicitly `RENDERED_UNCHECKED`, not a
verified pixel match. The package's optional numerical backends have separate
dependencies from this plotting-only workflow.

Outputs are `build/manuscript/pdf/`, `png/`, `rendered/panels/`,
`presentations/`, `tables/` and `validation/`. The full build also reconstructs
the current main circuit-resource table and SI required-measurement table.
For tables alone, run `python manuscript_data_gen/build_tables.py`.
The runner checks five numerical combination
PNGs and twelve individual panels against saved render references.
Main 2/4/5 published PDFs used Matplotlib 3.10.5 and are preserved separately
from the 3.11.1 render references.

For one figure: `python manuscript_data_gen/build_all.py --figures s1 --output build/si`.
See [the exact figure/data map](reproduction/FIGURE_REPRODUCTION.md).

## 3. Numerical replay and new simulations

Replotting saved CSVs is not equivalent to rerunning optimization. The
following directories separate the two operations:

| Directory | Role |
|---|---|
| `reproduction/code/methods/gpd/` | Sequential GPD construction, frozen calibration, selection and random-case replay |
| `reproduction/code/methods/srdd/` | Shallow rotated-density fitting, depth selection and one-body redistribution |
| `reproduction/code/methods/pauli_product/` | OGM, SG, Derand, AP and LCS schedules and sampling |
| `reproduction/code/methods/fully_commuting/` | FC grouping, compilation, allocation and noisy sampling |
| `reproduction/code/experiments/` | H4 scan/fixed geometry, H6 and BeH2/N2 drivers |
| `reproduction/code/analysis/` | Born-outcome replay, curve assembly and resource/variance verification |
| `reproduction/code/plotting/` | Current figure renderers |
| `reproduction/data/` | Materialized inputs, saved trajectories and statistical results |

Use the README in each code directory for exact commands and dependencies.
`requirements-replay.txt` records the tested Python 3.12 outcome-replay
environment; `requirements-plot-reference.txt` specifies the separate Python
3.14 reference renderer. Use separate virtual environments when reproducing
those exact versions.
The calibrated GPD replay instead uses `requirements-gpd-replay.txt` with
NumPy 2.0.2; changing NumPy can change multinomial streams for the same seed.
The standalone API in `src/mpqc_measurement/` is also supplied, but its example
defaults are **not** a substitute for the manuscript's saved experiment
configurations. In particular, large-molecule SRDD uses its saved feasible-rank
set and calibration protocol, not the small-system API's default grid.

Run new experiments into a separate output directory; retain the frozen data
for comparison. Numerical optimization and random-number/library differences
can change a de novo trajectory. Consult [VERIFICATION.md](VERIFICATION.md)
for what has actually been rerun, what is verified from saved artifacts, and
remaining limitations. No claim that every expensive simulation has been
rerun is implied by a successful plot build.

The generic `supp/table_s1_state_dependent_variance` and
`supp/table_s3_cnot_counts` CSV formatters are auxiliary historical outputs,
not active tables in the current SI. Use `build_tables.py` for the active tables.

## Maintenance

`prepare_inputs.py --archive-root ...` is an optional maintainer importer;
it is not required for a public clone. `reproduction/validation/package_data.py`
creates the curated lossless archives from a local scientific dataset.
`reproduction/bundle_manifest.json` records source/reference hashes;
`required_inputs.json` is the smaller plotting dependency list. The data
archive manifest additionally covers inputs and checkpoints used by numerical
replay. Never replace scientific results merely to satisfy an image hash.
