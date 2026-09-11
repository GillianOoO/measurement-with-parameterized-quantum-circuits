# Molecular construction and numerical replay

Run the commands below from `manuscript_data_gen/reproduction`, after
materializing the numerical-data archives as described in the release guide.
Every driver locates its modules and `data/` from the release location; the
checkout name and current working directory are not dependencies.
`MEASUREMENT_DATA_ROOT` can optionally select another extracted data tree.
New calculations write to fresh directories below `runs/` by default.

Use the release's documented Python environment. These molecular drivers need
NumPy, SciPy, matplotlib, pandas, quimb, and OpenFermion. The construction checks
were exercised with Python 3.12.14, SciPy 1.18.1, quimb 1.15.0, OpenFermion 1.8.1,
and both NumPy 2.0.2 and 2.5.3. The `portable/` library additionally uses the
repository's `mpqc-measurement` package (`python -m pip install -e .` from the
repository root). No machine-specific dependency directory is used by the code.

## Bounded checks

```sh
python code/experiments/check_molecular_release.py
python code/methods/srdd/test_large_molecule_calibration_outputs.py
python code/methods/srdd/test_shallow_collector_redistribution.py
python code/experiments/check_frozen_srdd_sampling.py
```

The input check compares recovered H4/H6 active Hamiltonians with the actual
blocks of the original dense inputs, loads BeH2/N2 inputs, and checks selected
rank arrays. The sampling diagnostic verifies the published H4/H6 empirical
RMSE values against the archived repeated-run estimates, then reports any
differences from newly drawn samples. It does not modify published results.

## H4 and H6

Rebuild shallow collector settings, refit SRDD-specific calibration weights,
reselect ranks for all eight shot budgets, and generate 200 repeated estimates
per budget from the released native rank fits and 500 saved probes:

```sh
python code/experiments/molecular_srdd_selection.py --molecules H4 H6 --output runs/h4_h6_reselected
```

Refit the native rank/depth frontier first (K=1 through 10, depths 1 through 3):

```sh
python code/experiments/molecular_srdd_rank_fit.py --molecules H4 H6 --output runs/h4_h6_refitted
python code/experiments/molecular_srdd_selection.py --molecules H4 H6 --source runs/h4_h6_refitted --output runs/h4_h6_refitted_selected
```

The rank-fit stage uses the released frozen calibration weights for its initial
diagnostic selection; the second command fits SRDD-specific weights on the new
frontier before producing the final selection. Full tensor refitting is more
expensive than replaying the saved frontier and was not repeated in the release
portability check.

For H4's 21-geometry bond scan, the default reuses the released native rank fits:

```sh
python code/experiments/h4_bond_scan_srdd.py --output runs/h4_bond_scan
python code/experiments/h4_bond_scan_pauli.py --output runs/h4_pauli
```

For a bounded single-geometry check add `--bonds 1.2` to the SRDD command.
To refit all bond-scan tensors, pass `--frozen-rank-source runs/no_cached_ranks`
where that directory does not contain any rank files. Individual bond outputs
are checkpoints; aggregation requires all 21 geometries. The historical
non-SRDD columns in the SRDD comparison table are read from their archived
numerical export; the separate Pauli command recalculates OGM/SG/Derand.

## BeH2 and N2

```sh
python code/experiments/prepare_beh2_input.py --output runs/beh2_recovered_input
python code/experiments/beh2_entry.py --output runs/beh2_full
python code/experiments/large_molecule_srdd_pauli.py --molecules N2 --output runs/n2_full
```

BeH2 input preparation exactly inverts the released Jordan–Wigner Pauli
Hamiltonian into spatial integrals and computes its fixed-spin ground state.
It does not perform a new SCF calculation. The released N2 integrals, geometry,
and exact FCI MPS are the starting inputs for N2. The full comparison commands
fit 16 ranks at three depths, calibrate settings, and run 50 repeated estimates
at T=1300,1600,2000,2400,3000. `--resume` continues an interrupted output using
its fit caches. These full large-molecule optimizations were not repeated in
the portability check; the commands can be computationally expensive, especially
for the 20-qubit N2 input. Their standalone plots are diagnostics; use the
release figure scripts for the manuscript layout.

## Verification results and exact replay limits

The H4/H6 full recalibration/reselection runs completed and recovered all 16
published selected ranks. Relative to the published results, maximum absolute
analytic-RMSE differences were 4.24e-9 Ha for H4 and 2.01e-11 Ha for H6. A
complete R=1.2 Angstrom H4 bond replay took about 14 seconds, with analytic RMSE
agreement to 2e-16 Ha. BeH2 input recovery took about 7 seconds: the largest
reconstructed Pauli-coefficient error was 1.78e-15 Ha and the ground energy
agreed within 5.4e-14 Ha. These timings are illustrative, not a full-run cost
estimate. Six focused calibration/collector tests and all seven driver imports
passed. The separate noisy-circuit figure replay is documented under analysis.

Fresh seeded SRDD samples are **not guaranteed to equal the original finite
sample values**. The retained algorithm diagonalizes dense setting matrices.
Degenerate eigenspaces can acquire different eigenvector bases across numerical
environments or after small roundoff changes. The same operator, state, and
measurement-outcome distribution can therefore be represented by different
per-eigenvector probability vectors, changing seeded multinomial draws.
In an isolated test on the same saved H4 setting, NumPy 2.0.2 and 2.5.3 produced
identical eigenvalues (rounded to 12 decimals) and identical draws for a fixed
test probability vector, but different eigenspace Born-probability vectors.
The setting had 256 eigenvalues and only 81 distinct values. Pinning NumPy alone
did not recover the old empirical SRDD values. The algorithm's distribution
has not been changed to force agreement with those samples.

The exact empirical values used in Figure 2 can instead be recomputed from
the released **per-repetition energy estimates** (these are not per-shot
bitstrings):

- `data/H4/results/srdd_fixed_selected/sampling_replicates.csv`: 200 repetitions
  at each of eight shot budgets.
- `data/H6/results/srdd_selected/sampling_replicates.csv`: the same structure.
- `data/H4/results/srdd_bond_scan_selected/R*/sampling_replicates.csv`: 50
  repetitions per bond geometry.

For each row group, RMSE is `sqrt(mean(signed_total_error**2))`. The input
archives, native fits, selected settings, calibration data, and saved energy
estimates serve different reproducibility purposes; none should be substituted
for another when claiming an exact replay.
