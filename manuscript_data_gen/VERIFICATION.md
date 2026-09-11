# Manuscript reproduction verification — 2026-09-11

The release checks below ran from the relocated `manuscript_data_gen` tree,
using the data extracted from its own ZIP archives. No original project path
is needed for the supported commands. Numerical inputs and published results
retain their original hashes. Small, machine-readable evidence is in
[`reproduction/validation/release_checks/`](reproduction/validation/release_checks/).

## Verified results

| Scope | Result |
|---|---|
| Main Figures 1–5 and SI Figure 1 | All six reconstructed. Five numerical combination PNGs and twelve subpanel PNGs exactly match the saved reference renderer; Figure 1's PPTX/export hashes verified. |
| Figure 2 | All 148 input/summary values reproduced from 17,000 archived repeated estimates, including recovered H6 Pauli records; maximum RMSE difference 1.39e-17. |
| Figure 3 noisy SRDD/FC-IMA sampling | All 209 setting/group jobs freshly run under the frozen designs; all outcome and contribution arrays exactly match the saved arrays. |
| Figure 3 empirical errors | All 150 points reconstructed from 7,500 complete estimates; maximum RMSE difference 1.33e-15. Zero-noise and Pauli estimates are the saved repetitions, not newly optimized experiments. |
| Figure 3 inverse accuracy | All 1,208 source curve points verified from saved variance/bias and allocation records. |
| Figure 5 GPD | All ten cases and seven measurement counts recalibrated, reselected and resampled from saved fragments; all 70 instance rows agree exactly using NumPy 2.0.2. |
| Figure 5 Pauli methods | All 17,500 fresh estimates agree exactly; all 70 ensemble plot points and 350 variance entries checked. Schedules are the frozen published schedules. |
| Main/SI tables | All four main circuit-resource rows and 18 SI required-measurement entries reproduced, including maximum CNOT counts and depths. |
| SI variance and bias | Source MSE identities, variances, 38 displayed bias values, bar magnitudes/direction, joined axes and twelve panel images checked. |

PNG identity refers to the documented Matplotlib 3.11.1 reference renderer,
not every PDF's metadata bytes. The published Main 2/4/5 PDFs retain their
original Matplotlib 3.10.5 rendering. Times New Roman is not redistributed.

## Method and input checks

- Missing canonical GPD backend, Pauli support kernels, and three extended
  GPD source trees were recovered. Their numerical files were checked against
  the original recorded hashes.
- All ten GPD source/calibration/selection integrity audits passed; no
  published case is right-censored. A deliberately bounded nonlinear
  optimizer smoke test also passed; it is not a production fit.
- H4/H6 Hamiltonian reconstruction errors are below 7e-14. All 16 fixed-geometry
  SRDD ranks were recovered after recalibration/reselection; maximum analytic
  RMSE differences are 4.24e-9 (H4) and 2.01e-11 (H6).
- BeH2 inverse-Jordan–Wigner input recovery agreed in energy within 5.4e-14.
  BeH2/N2 input loading, seven molecular driver imports, and 21 source-module
  compilation checks passed.
- Six SRDD collector/calibration tests, 24 FC tests, six Derand tests, 17
  gate/noise tests, a 200,000-draw SRDD check and 32 FC fault-propagation checks
  passed. All 51 root package/release tests also passed with optional JAX installed.
- Every released archive and every extracted numerical file is hash-verified.
  The original 256 MB H6 matrix is included losslessly; no matrix truncation
  or numerical rounding was performed.

## Reproduction limits

1. **Not every nonlinear optimization was rerun from initialization.**
   Full rank/depth refits, all molecular electronic-structure calculations and
   the expensive BeH2/N2 optimization frontiers have not been independently
   rerun in this release audit. Their input data, saved fits, current scripts
   and commands are included. Saved-fit replay is explicitly distinguished
   from new fitting in each method README.
2. **H4/H6 fresh empirical SRDD values need not be bitwise identical.**
   Dense setting diagonalization can select different bases inside degenerate
   eigenspaces. This preserves the measurement-outcome distribution but can
   change seeded multinomial realizations. The exact published RMSEs are
   reproduced from the included per-repetition energy estimates, not claimed
   from identical newly generated per-shot bitstrings. See
   [the molecular sampling diagnosis](reproduction/code/experiments/README.md).
3. Exact stochastic replay requires the documented environment. GPD's
   multinomial stream uses NumPy 2.0.2; the later random-Pauli replay uses
   NumPy 2.5.2. A single unconstrained environment is not an exact-replay
   guarantee.
4. Historical helper front ends are retained only where needed by current
   kernels and are clearly identified in their READMEs. They are not the
   supported all-manuscript entry points.
5. Repository visibility and licensing are separate from reproducibility.
   GitHub reported this repository as private during the update. Visibility
   was not changed. A public journal data/software declaration additionally
   requires author-approved public access and appropriate licensing.

Each command writes a new verification report. These release records are not
a substitute for rerunning checks after later changes.
