# The 50-instance sparse/dense random ensembles

Current Figure 5 and SI Figure 1(e,f) use 50 independent sparse and 50
independent dense four-qubit Hamiltonians. Every method uses the same
Hamiltonian and common input state within each case, with 50 complete
energy estimates per instance/method/measurement count.

Production finished on 2026-11-02. This release update packages and verifies
those saved results; it does not claim a new full optimization or
finite-measurement rerun. The original five cases per regime remain
available for a separate legacy check.

## Reconstruct and verify saved results

Run from the repository root with NumPy and Matplotlib installed:

```sh
python manuscript_data_gen/materialize_data.py
python manuscript_data_gen/reproduction/code/experiments/plot_random_ensemble.py --audit-only
python manuscript_data_gen/reproduction/code/experiments/plot_random_ensemble.py
```

The default input is `manuscript_data_gen/reproduction/data/random_ensemble_50/`.
The default output is `build/random_ensemble_50/`. Override these with
`--run` and `--output`. `PAPER_REPRO_RANDOM_ENSEMBLE_DIR` overrides the run
directory; `MEASUREMENT_DATA_ROOT` overrides the materialized `data/` root.
Outputs must be outside the input experiment and frozen data tree.

The independent audit checks all 4,200 instance records and 210,000 complete
estimates, the 100 matched Hamiltonians and common state, hashes, all 84
ensemble means, calibrated stopping, selected K and integer allocations.
It uses only saved results: JAX, optimization and fresh sampling are
unnecessary. `--audit-only` writes checked numerical summaries without
rendering. The ordinary command also writes combined and individual
sparse/dense PDF and PNG figures.

For the integrated manuscript figure build:

```sh
python manuscript_data_gen/build.py manuscript_data_gen/main/fig05_random_hamiltonian_errors --output build/general_hamiltonian_gpd_average_errors.pdf
```

The original-ten-case comparison is separate:

```sh
python manuscript_data_gen/reproduction/code/experiments/check_random_legacy_prefix.py
```

It checks 420 instance records, 21,000 estimates, 40 Pauli schedule sets and
84 original five-instance means, writing
`build/random_ensemble_50/legacy_prefix_reproduction.json`. It does not
replace the 100-case audit.

## Statistics and counts

Each instance RMSE is the square root of the mean squared error of its 50
complete estimates. Figure 5 plots the arithmetic mean of those RMSEs over
50 Hamiltonians, not the pooled RMSE. Instance standard deviations and
standard errors remain in `results/ensemble_summary.csv`; no error bands
are drawn.

Variance summaries average exact instance-level variances. GPD bias
summaries average absolute expectation differences, not signed biases. AP's
MSE uses the saved mean squared conditional bias, not the square of its
mean bias. Zero values remain in every average. Figure 5 includes T=12;
SI Figure 1(e,f) displays the common range T>=45 and retains T=12 in data.

The SI comparison counts constructed OGM groups, distinct bases in the
saved SG/Derand/AP schedules, and the 81-setting full support for LCS.
LCS does not archive individual basis draws; its support count is not an
audit of distinct executed circuits. Dense OGM's mean constructed-group
count is 42.74; it differs from the 42.34 mean of bases with positive
allocations at T=25,848. GPD mean selected K at that T is 8.22/15.14
(sparse/dense). A separate Frobenius-tolerance check at 0.01 gives mean
minimum K=4.90/16.98. These selection criteria are not interchangeable.

## Saved-data layout

```text
data/random_ensemble_50/
  inputs/
    hamiltonians/                 100 shared H.npy inputs
    generation_matrices/          raw A before symmetrization
    common_input_state.npy        common pure-state density matrix
    manifest.json                 hashes and generation protocol
    input_checksums.json
  timing_decision.json            original pilot and 50/20 decision
  legacy_prefix_reproduction.json original ten-case validation
  cases/<regime>4_seed0_<i>/
    complete.json                 accepted case and hashes
    gpd_source/                   parameters, diagonals and residuals
    gpd_selected/                 probes, weights, frontiers, K, allocations
                                  and individual estimates
    pauli/                        schedules and individual estimates
  results/
    replicates.csv                210,000 estimates and errors
    instance_results.csv          4,200 RMSE/variance/bias records
    ensemble_summary.csv          84 mean/SD/SE records
    paired_input_audit.csv         100 input-pairing checks
    manifest.json                 completeness and output hashes
```

Historical absolute paths in immutable manifests record production
provenance. Audits resolve files from this portable layout without accessing
those machines. Logs, earlier diagnostic runs, runtimes and two redundant
per-probe CSVs (`calibration_approximation_components.csv` and
`calibration_sampling_components.csv`) are not packaged. Their original
hashes remain in historical manifests; source fragments, probes, weights
and allocations are retained. The normal audit does not require the omitted
CSVs; fresh calibration regenerates them.

## Recorded experiment

| Quantity | Setting |
|---|---|
| Qubits | 4 |
| Sparse A | Four distinct uniformly chosen positions of a zero 16x16 matrix, Uniform[0,1] values |
| Dense A | All 256 entries independently Uniform[0,1] |
| Observable | H=(A+A.T)/2, without rescaling |
| Measurement counts | 12, 45, 160, 572, 2038, 7259, 25848 |
| GPD circuit | Periodic alternating iSWAP, L=4, five Euler layers, 60 angles |
| Fitting | One start, 600 Adam updates, learning rate 0.01, seed 20260808, float32/complex64 |
| Calibration | 496 grouped stabilizer states; seed 918273; K=1..10; T=100,200,300,500,800,1200,2000,3000 |
| Stopping | Five post-calibration boundaries without relative material improvement greater than 0.002 |
| Selection | Smallest minimizer of frozen calibrated balanced loss at each T |
| GPD allocation | Minimum-one, half-range-proportional, largest remainder |
| Repetitions | 50 complete estimates per Hamiltonian/method/T |
| Schedule seed | 20260826 with original case/method hash rule |
| Outcome seed | 20260902 with original method-specific hash rule |

The generator initially supplies 30 fragments and extends by ten if needed.
Historical uncalibrated stopping is suppressed. Failure to reach calibrated
termination by 100 fragments fails the batch; no partial ensemble is plotted.
Fragments after the calibrated stop do not enter fixed-measurement selection.

SG and Derand use prefixes of one maximum-T schedule. OGM optimizes one
distribution per Hamiltonian and uses T-specific allocation. LCS samples
uniform product-Pauli settings jointly with Born outcomes. AP uses 50
independent schedules and one Born replay per schedule at each T.

The input stream is `numpy.random.RandomState(0)`: 16 real and 16 imaginary
normal amplitudes for the state, five dense matrices, then five sparse
matrices. From instance 6 onward, one dense and one sparse matrix are drawn
at each index. The original ten inputs are checked and copied byte for byte.
The common density-matrix file has SHA-256
`c524a2d52b13c42a3db9e541a7af9ee11cfed32e78412a0c0e0f858adf35378d`.

## Optional de novo computation

Use a new output directory. Fresh GPD fitting uses recorded Python 3.12,
JAX 0.4.38, NumPy 2.0.2 and SciPy 1.18.1; install
`manuscript_data_gen/requirements-random-ensemble-gpd.txt` into that
environment. Pauli calculations use separate Python 3.14.7, NumPy 2.5.2 and
SciPy 1.18.0, with `manuscript_data_gen/requirements-random-pauli.txt`.
Install requirements using each environment's `python -m pip install -r`.

Pass the two Python executables explicitly. Defaults use the current
interpreter without machine-local package paths; `--gpd-runtime` is an
optional override for externally managed package directories. In this
PowerShell example, replace the environment paths with your installations
(Unix virtual environments use `bin/python`):

```powershell
$runDir = 'manuscript_data_gen/reproduction/regenerated/random_ensemble_new'
$gpdPython = '.venv-gpd/Scripts/python.exe'
$pauliPython = '.venv-pauli/Scripts/python.exe'
& $pauliPython manuscript_data_gen/reproduction/code/experiments/random_ensemble_inputs.py --instances 50 --output "$runDir/inputs"
& $pauliPython manuscript_data_gen/reproduction/code/experiments/run_random_ensemble.py pilot --output $runDir --gpd-python $gpdPython --pauli-python $pauliPython
& $pauliPython manuscript_data_gen/reproduction/code/experiments/run_random_ensemble.py run --output $runDir --workers 4 --gpd-python $gpdPython --pauli-python $pauliPython
& $pauliPython manuscript_data_gen/reproduction/code/experiments/check_random_legacy_prefix.py --run-root $runDir --output build/random_ensemble_new/legacy_prefix_reproduction.json
python manuscript_data_gen/reproduction/code/experiments/plot_random_ensemble.py --run $runDir --output build/random_ensemble_new
```

The pilot executes one complete sparse/dense pair and projects serial
runtime with 50% margin. It selects 50 instances per regime if the estimate
is <=24 hours, otherwise 20. Only the 50-instance result reproduces the
current manuscript ensemble; production selected 50. Resume accepts only
completed cases whose input, code and environment bindings match.

The original Pauli arithmetic is `trace(P @ H)/16`. Changes to floating-point
reductions or BLAS threading can alter tied OGM probabilities or Derand
schedules. The driver preserves the recorded thread policy. Exact equality
on other dependency versions/platforms is not promised; the legacy check
reports differences.
