# Manuscript reproduction verification — 2026-11-11

This release packages the latest completed experiments and rebuilds their
numerical figures and tables. It does not claim that all nonlinear
optimizations were rerun during this repository update.

## Current release checks

An isolated copy containing only repository files was created without the
original project, local runtimes, caches or extracted data. Its own ZIP shards
were then unpacked and the commands in the reproduction guide executed.
The first isolated check exposed a generated font-cache entry in the source
manifest; that entry was removed and the manifest generator now excludes
caches. The subsequent isolated build passed.

| Scope | Result |
|---|---|
| Numerical archives | All 28 archives and 15,239 materialized files verified against recorded SHA-256 hashes; no original numerical member changed. |
| Current random ensembles | All 100 matched Hamiltonians, 4,200 instance/method/measurement-count records, 210,000 complete estimates and 84 ensemble means independently audited. |
| GPD selection | Saved calibrated stopping, smallest minimizing prefix, integer allocations and cross-method Hamiltonian/state bindings verified for every current random instance. |
| Main Figures 2–5 and SI Figure 1 | Five combined PNGs and twelve individual panels exactly match the documented render references in the isolated tree. Latest Main 5 and SI 1 reference assets were copied from the current manuscript and independently reproduced. |
| SI variance and bias | All displayed values checked against saved moments; 38 bias bars, their inverted orientation, zero-based left axes and joined panel borders verified. All 50 random instances enter each mean. |
| Tables and projections | Four main circuit-resource rows, 18 required-measurement entries, 1,208 inverse-accuracy points, 30 setting-count entries, 600 random setting-count records and 104 fixed-tolerance records reconstructed. |
| Original-ten-case regression | 420 instance records, 21,000 estimates, 40 Pauli schedule sets and 84 original five-instance means agree with the retained legacy data. |

The fixed-Frobenius-tolerance analysis uses the recorded candidate grids and
includes molecular completion-residual bounds. It is not a global optimality
claim. LCS's reported 81 settings count its complete distribution support;
individual LCS basis draws were not archived, so this is not an audit of
distinct executed circuits. Other counting rules are recorded per output row.

The package/release test suite and separate large-molecule calibration tests
are also rerun. The machine-readable current summary in
[`reproduction/validation/release_checks/current_20261111.json`](reproduction/validation/release_checks/current_20261111.json)
records their counts, the figure hashes, and the isolated-build results.

The reference plot environment is Python 3.14.7, NumPy 2.5.2 and Matplotlib
3.11.1 with Times New Roman. PNG identity applies to those references, not to
PDF timestamps or every renderer version. Published Main 2/4 PDFs preserve
their earlier Matplotlib 3.10.5 rendering separately. Fonts are not distributed.

## Algorithm consistency

The SRDD core fitting and redistribution files were checked against the local
production implementations. No algorithmic behavior needed changing. The
documentation now distinguishes H4/H6 equal-stratum calibration and full-rank
completion from BeH2/N2 through-origin calibration and one-added-rotation
completion. API defaults are not presented as the configuration of every
manuscript benchmark. Positive-only BeH2/N2 candidates are preserved.

New ensemble entry points resolve paths relative to the repository or explicit
arguments. Historical paths inside frozen records are provenance, not execution
dependencies. Optional fresh experiments use separate output directories and
the documented GPD and Pauli environments.

## Earlier numerical replay checks

The following were performed in the 2026-09-11 release audit and were not
rerun as full sampling or optimization jobs in this update. Their dated
evidence remains in `reproduction/validation/release_checks/`:

- Figure 2: 148 values reaggregated from 17,000 archived estimates;
  maximum estimation-error difference 1.39e-17.
- Figure 3: 209 noisy setting/group jobs freshly sampled with matching
  archived outcomes; 150 error points reassembled from 7,500 complete
  estimates with maximum difference 1.33e-15.
- Original five-per-regime GPD/Pauli checks: 70 GPD instance rows and
  17,500 Pauli estimates reproduced. These are legacy regression checks,
  not evidence for a new 100-instance sampling run.
- H4/H6: Hamiltonian reconstruction errors below 7e-14 and all 16
  fixed-geometry selected SRDD candidates recovered after recalibration.
- BeH2 input recovery, molecular driver imports, collector/FC/Derand tests
  and gate/noise checks passed in the earlier recorded environment.

## Reproduction limits

1. Full molecular electronic-structure calculations and expensive SRDD
   rank/depth refits were not repeated here. Saved fits, inputs and driver
   commands are included. The current random experiment was completed on
   2026-11-02; this update audits and publishes its saved outputs.
2. The current environment did not contain `quimb`, so the older Quimb-based
   collector integration test was not rerun. Its dependency is documented;
   this does not affect the plotting-only or current saved-data audits.
3. Fresh H4/H6 dense diagonalization may choose different bases inside
   degenerate eigenspaces. The outcome distribution is unchanged, but seeded
   bitstrings can differ. Published errors are exactly reconstructible from
   the archived complete estimates.
4. Exact stochastic replay requires the recorded environments: GPD uses
   NumPy 2.0.2; current random-Pauli experiments use NumPy 2.5.2. A single
   unconstrained environment is not an exact-replay guarantee.
5. Historical helpers are identified in their READMEs and are not the
   supported current entry points. Public plotting checks do not compile
   private manuscripts or reviewer correspondence.

Each command writes fresh verification evidence. Dated release records do not
replace rerunning the checks after subsequent changes.
