# Product-Pauli methods

`run_same_hamiltonian_pauli_methods.py` supplies the molecular OGM, SG and
Derand schedule and measurement routines. `derand_huang2021.py` implements
the separate Appendix-C and repository-target rules of Huang et al.
`run_general_hamiltonian_revision.py` supplies random-Hamiltonian OGM, SG,
Derand, LCS and AP routines. The current molecular experiment drivers use
these as imports with explicit released Hamiltonian/state inputs.

The restored support files `estimate_sampling_errors.py`,
`run_lih_r1p50_sampling_comparison.py`,
`compare_fixed_shot_lih_baselines.py` and
`count_measurement_settings_n2038.py` provide the original measurement
kernels used by those experiments. Their standalone historical LiH,
TFIM and AGPD front-end commands require inputs outside the current
H4/H6/BeH2/N2/random release; they are not all-manuscript entrypoints.
See `../../experiments/README.md` for the current experiment commands.

Run the six Derand tests from `manuscript_data_gen/reproduction`:

```sh
python -m unittest discover -s code/methods/pauli_product -p "test_*.py"
python code/methods/pauli_product/validate_molecular_schedules.py --help
```

Those tests and imports of all five supporting baseline modules pass in
Python 3.12 with NumPy 2.5.3, SciPy 1.18.1, pandas and Matplotlib 3.11.1.
This does not establish a complete rerun of all stochastic Pauli curves.

The current random-Hamiltonian Pauli replay is now available separately:

```sh
python code/analysis/replay_random_pauli.py --output build/random_pauli_replay
```

All ten instances, seven budgets and five Pauli methods were freshly replayed
with 50 repetitions each: all 17,500 estimates exactly matched the released
records in Python 3.14.7, NumPy 2.5.2, SciPy 1.18.0. The wrapper independently
checks 350 exact variance entries and the 70 ensemble-mean Figure 5 points.
It retains the published measurement schedules and does not re-optimize them.

`replay_random_all_methods_empirical50.py` is the historical uncalibrated-GPD
all-method replay, not the current calibrated GPD curve. It records an exact
Python 3.14.7 / NumPy 2.5.2 / SciPy 1.18.0 baseline environment and strict
original-source hash checks. Its complete historical standalone workflow
has not been certified on this release. Current calibrated GPD data and
checks are in `../gpd/`; current random-Pauli figure data are in
`data/random_sparse_dense/` relative to the reproduction root.

The `shadowgrouping/` kernels were restored from the original
ShadowGrouping source snapshot used by the manuscript. Its upstream
Apache-2.0 `LICENSE` is included in that directory and governs those files;
no new license is assigned to third-party code.
