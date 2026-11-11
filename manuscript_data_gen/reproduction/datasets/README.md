# Numerical data archive

From the repository root run `python manuscript_data_gen/materialize_data.py`.
The ZIP shards are parts of one logical dataset but are ordinary independent
ZIP archives, not a proprietary split-ZIP format. Each member retains its full
`data/...` path. `manifest.json` is the authoritative member/size/SHA-256 index.

## Expanded directory layout

| Directory under `reproduction/data` | Contents |
|---|---|
| `H4/inputs`, `H6/inputs` | Dense and electronic-integral Hamiltonian inputs and geometry information |
| `H4/results`, `H6/results` | SRDD rank/depth factors, calibration/selection records, reference states, Pauli schedules and numerical outputs |
| `H4/processed`, `H6/processed` | Molecular error curves consumed by Main Figure 2 |
| `BeH2/inputs`, `N2/inputs` | Integrals, Pauli coefficients, blocked-spin ground-state MPS and metadata |
| `BeH2/results`, `N2/results` | Large-molecule SRDD fits, selected designs and Pauli comparison outputs |
| `fully_commuting/` | FC-IMA groups, compiled-circuit counts/depths and error/noise records |
| `random_sparse_dense/inputs` | Ten four-qubit Hamiltonians in original Pauli order and dense form, plus the common state |
| `random_sparse_dense/results/gpd_source_prefixes` | Initial sequential GPD source trajectories |
| `random_sparse_dense/results/gpd_extended_sources` | Complete extended trajectories for sparse instances 2, 3 and 5 |
| `random_sparse_dense/results/gpd_balanced` | Calibrated GPD selection, integer allocations and source fingerprints |
| `random_sparse_dense/results` | Pauli schedules, individual repeated estimates and statistical summaries |
| `random_ensemble_50/inputs` | Current 50 sparse and 50 dense Hamiltonians, generation matrices, common state and hashes |
| `random_ensemble_50/cases` | Per-case GPD fragments, calibration, selection, allocations, Pauli schedules and individual estimates |
| `random_ensemble_50/results` | 210,000 estimates, 4,200 instance results, 84 ensemble summaries and paired-input checks |
| `shared_processed/figure3/empirical50` | Frozen SRDD/FC settings, noisy Born bitstrings, per-setting contributions and 50-repeat curves |
| `shared_processed/figure3` | Inverse-accuracy data and recovered measurement circuits |
| `shared_processed/resources`, `variance` | Exact variance/bias and fixed-accuracy measurement/circuit summaries |

NPY/NPZ files store numerical arrays; CSV files store coefficients, samples or
summaries with named columns; JSON files store configuration, selection and
provenance metadata. Numerical TXT files preserve the original Pauli-term
ordering, which matters for exact seeded replay. Paths appearing inside
historical provenance records refer to the original computation; supported
release drivers resolve their inputs inside the extracted tree and verify
the saved hashes.

These are preserved scientific records. Historical method rows may coexist
with current ones in a source table; the current plotters explicitly filter
them. Neither molecular Hamiltonian coefficients nor simulation outcomes were
rounded, regenerated, or relabeled during lossless packaging. New simulations
must write outside this frozen data directory.

## Current random-Hamiltonian ensemble

The `random_ensemble_50` supplement contains the completed 2026-11-02
experiment used by current Main Figure 5 and SI Figure 1(e,f). The original
five instances per regime in `random_sparse_dense` remain available for
regression checks; they are not the current ensemble average.

Logs, superseded diagnostic runs and two redundant per-probe CSVs are omitted:
`calibration_approximation_components.csv` and
`calibration_sampling_components.csv`. The source fragments, calibration
states, fitted weights, allocations and individual estimates are retained;
fresh calibration regenerates the omitted CSVs. Original provenance manifests
are preserved byte for byte, including their historical paths and hashes.
The supported audits resolve inputs within the extracted release tree.

See [the ensemble protocol and commands](../code/experiments/RANDOM_ENSEMBLE_README.md).
For maintainers, `../validation/package_random_ensemble.py --run-root PATH`
adds this curated supplement without replacing existing numerical members;
`--check-only` verifies every released byte against the production run.
