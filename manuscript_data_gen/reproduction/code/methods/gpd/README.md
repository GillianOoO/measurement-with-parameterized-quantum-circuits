# GPD computation and validation

Run from `manuscript_data_gen/reproduction` after materializing the release data.
The code defaults to `data/`; `MEASUREMENT_DATA_ROOT` can point to an equivalent
extracted tree. Place new outputs in `regenerated/` or pass `--output`.

The restored `recompute_h4_gpd.py` is the canonical native-iSWAP backend.
`generate_frobenius_prefixes.py` fits sequential full-diagonal residual
fragments. `reselect_stabilizer_balanced.py` calibrates approximation and
sampling coefficients on grouped stabilizer probes, then selects K and
integer shot allocations before loading the benchmark state.

Exact finite-shot replay requires the recorded NumPy **2.0.2**. The original
selector used JAX 0.4.38; JAX 0.6.2 also reproduced the saved selector results
with NumPy 2.0.2 in release validation. Newer NumPy 2.5.3 changed some
multinomial draws despite identical seeds. Use separate environments for
this selector and the later random-Pauli experiments.

Audit the ten released random cases, then recompute every calibrated
selection and its 50 replicates and compare the 70 Fig. 5 instance rows:

```sh
python code/methods/gpd/audit_batch_reselection.py
python code/methods/gpd/check_reproduction.py --output regenerated/gpd_validation
```

The check reports per-case errors and exits unsuccessfully on a numerical
mismatch. It validates saved-fragment calibration, selection and finite-shot
evaluation; it does not rerun nonlinear optimization from initial parameters.

For a single calibrated case:

```sh
python code/methods/gpd/reselect_stabilizer_balanced.py --source data/random_sparse_dense/results/gpd_source_prefixes/sparse4_seed0_1 --output regenerated/gpd_balanced/sparse4_seed0_1 --empirical-repeats 50
```

Sparse instances 2, 3 and 5 require their extended source frontiers under
`data/random_sparse_dense/results/gpd_extended_sources`, rather than the
shorter original frontiers. Historical manifest paths are resolved against
the released tree while Hamiltonian, state and fragment content checks are
retained.

For a bounded optimizer smoke test (deliberately incomplete):

```sh
python code/methods/gpd/generate_frobenius_prefixes.py case --benchmark sparse --slug sparse4_seed0_1_smoke --input data/random_sparse_dense/inputs/hamiltonians/sparse_hamiltonian_4_1.npy --state data/random_sparse_dense/inputs/common_input_state.npy --output regenerated/gpd_smoke --paper-depth 4 --shots 2038 --starts 1 --steps 2 --max-k 1 --minimum-k 1 --x64
```

Production optimization must use the parameters and seed in each source
`manifest.json`. This smoke test completed successfully and records a
right-censored frontier. It is not a publication-curve reproduction.
`run_iswap_gpd_main_loss.py` supplies the original ansatz interface; its old
uncalibrated aggregate/AGPD-comparison branches are historical, and are not
the current manuscript plotting workflow. Use the top-level figure runner
for the current main and supplementary plots.

iSWAP is applied directly as a native unitary. For even n, L entangling layers
contain nL/2 iSWAPs, entangling depth L and 3n(L+1) Euler angles. Thus n=4, L=4
has 60 angles and 8 iSWAPs without CNOT decomposition. State preparation,
one-qubit gates, routing and readout are excluded from this resource count.
