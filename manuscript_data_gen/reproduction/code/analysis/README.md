# Current numerical analysis

Run from manuscript_data_gen/reproduction/ after materializing the released
data. Results go to build/; published records under data/ are retained.

## Main and Supporting Information tables

~~~powershell
python code/analysis/rebuild_manuscript_tables.py --output build/manuscript_tables
~~~

This rebuilds all 18 numerical entries in the SI required-measurement table,
the four main-table circuit-resource rows, and the 1,208 continuous
inverse-accuracy points used by Figure 3(b,e). It checks the new numbers against
the released figure inputs. Outputs include CSV tables, the two TeX tabular
blocks, and a source-hash manifest. The main table includes maximum compiled
CNOT counts (SRDD: 72/108; FC-IMA: 86/181 for BeH2/N2), setting counts and
depth. Only NumPy and the Python standard library are required.

These are calculations from saved exact variance, bias, circuit and allocation
records. They do not perform a new molecular electronic-structure calculation,
decomposition optimization, or measurement experiment.

## Figure 2 recorded molecular estimates

~~~powershell
python code/analysis/verify_molecular_replicates.py --output build/molecular_replicates
~~~

This independently reaggregates 4,250 saved SRDD estimates (200 repetitions per
fixed H4/H6 point and 50 per H4 bond-scan point), plus 7,950 saved H4 Pauli
and 4,800 H6 Pauli estimates, and verifies all 148 corresponding figure-input
and benchmark-summary values. It writes
a CSV and a JSON report with source hashes. This is a check of recorded
estimates, not a new stochastic replay. The H6 Pauli repeated estimates were
recovered from the original v4 aggregate CSVs and retained byte-for-byte under
data/H6/results/pauli_replay/. The verifier checks their original hashes and
selects H6 OGM, SG and Derand only; historical rows for other experiments are
retained solely to preserve those original source files.

## Figure 3 empirical-data reconstruction

Use Python 3.12 with NumPy, SciPy, Numba, Qiskit, Quimb and Pandas installed as
specified by the release requirements:

~~~powershell
python code/analysis/test_fig3_empirical50.py
python code/analysis/replay_fig3_fc_empirical50.py --validate
python code/analysis/assemble_fig3_empirical50.py --output build/fig3_assembly
~~~

The assembler reconstructs scores from released bitstrings and recomputes all
150 empirical error points from 7,500 complete estimates. It checks noiseless
SRDD setting moments, FC-IMA circuit counts and 3,000-measurement allocations.
It reads the released settings/, fc_groups/ and frozen *_settings.npz by
default, and writes a separate newly assembled table and manifest. This is
an outcome-data reconstruction, not a new draw.

To generate fresh noisy Born outcomes under the published circuit design:

~~~powershell
python code/analysis/run_fig3_empirical50.py --output build/fresh_fig3 --workers 6
$env:PAPER_REPRO_FIG3_OUTPUT = 'build/fresh_fig3'
python code/analysis/test_fig3_empirical50.py
python code/analysis/replay_fig3_fc_empirical50.py --validate
Remove-Item Env:PAPER_REPRO_FIG3_OUTPUT
python code/analysis/assemble_fig3_empirical50.py --input build/fresh_fig3 --output build/fresh_fig3_assembled
python code/analysis/verify_fig3_replay.py --replay build/fresh_fig3
~~~

A fresh run executes 209 SRDD-setting/FC-IMA-group jobs and can be expensive,
especially for 20-qubit N2. Existing job files are resumed; choose a new empty
output directory when a fresh replay is intended. Stable seeds reproduce the
specified Monte Carlo streams, subject to numerical/runtime compatibility.
The zero-noise and product-Pauli curves reuse the published noiseless repeated
estimates; this workflow is not a new zero-noise experiment.

Individual bounded jobs:

~~~powershell
python code/analysis/replay_fig3_empirical50.py --molecule BeH2 --setting 0 --output build/test_replay
python code/analysis/replay_fig3_fc_empirical50.py --molecule BeH2 --group 0 --output build/test_replay
~~~

SRDD replay uses the released frozen circuits, coefficients and allocations.
The separate command below reconstructs settings from archived fits. It retains
published angles if numerical optimizers choose a different near-degenerate
solution:

~~~powershell
python code/analysis/replay_fig3_empirical50.py --prepare BeH2 --rebuild-settings --output build/reconstructed_settings
~~~

PAPER_REPRO_DATA_ROOT can select another extracted data directory.
PAPER_REPRO_FIG3_OUTPUT controls sampler outputs; --output overrides it.
No private runtime directory is added to Python's module path.

## Figure 5 Pauli baselines

~~~powershell
python code/analysis/replay_random_pauli.py --output build/random_pauli_replay
~~~

This generates 17,500 fresh Born estimates for OGM, SG, Derand, LCS and AP
across all ten four-qubit random Hamiltonians, seven budgets and 50 repetitions.
It preserves the published Hamiltonian, common state, frozen schedules and
outcome seeds. It recomputes the 350 instance-level exact variances and compares
the 70 ensemble-mean error points with the current Figure 5 input table.
GPD is checked separately by code/methods/gpd/check_reproduction.py.

The reference Pauli runtime is Python 3.14.7, NumPy 2.5.2, SciPy 1.18.0.
The earlier GPD optimization environment is separate. Other runtimes can run
this wrapper, but the report only passes when numerical comparisons agree.
Use --case sparse4_seed0_1 for a bounded single-case check; a complete Figure 5
ensemble comparison requires all five cases in each regime.

The wrapper imports the estimator helpers in
methods/pauli_product/replay_random_all_methods_empirical50.py, but does not run
that historical file's main, which refers to an obsolete uncalibrated GPD
source and removed experiment paths. All output CSVs and the machine-readable
comparison report are isolated under the selected output directory.

## Historical analysis sources

build_resource_statistics.py, build_state_dependent_variance.py,
build_fig3_extended_data.py and generate_variance_data.py are historical
sources for earlier analysis stages. They retain obsolete experiment names,
paths and non-current method branches; they are not clean-release execution
entrypoints. Do not use them to overwrite current tables or figures.

The supported entrypoints above rebuild current tables and Figure 3 outcome
summaries. The figure runner at the reproduction root rebuilds all current
plots and SI bias bars from released source tables.

## Gate accounting

native_gate_resources.py counts iSWAP and CNOT gates separately. Periodic GPD
uses n*L/2 native iSWAP gates and entangling depth L for even n; at n=4, L=4
this is 8 iSWAPs, depth 4 and 60 angles. Legacy open-chain variants keep their
actual matchings. Counts exclude preparation, single-qubit gates, routing and
readout. Current molecular Figure 4 uses SRDD and FC-IMA compiled CNOT resources,
not historical AGPD resource rows.

~~~powershell
python -m unittest discover -s code/analysis -p test_native_gate_resources.py -v
python -m unittest discover -s code/analysis -p test_cnot_noise_semantics.py -v
~~~
