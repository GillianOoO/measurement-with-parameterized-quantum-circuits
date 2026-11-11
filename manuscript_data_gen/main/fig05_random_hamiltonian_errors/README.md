# Main Figure 5

Current Figure 5 uses 50 independent sparse and 50 independent dense
four-qubit Hamiltonians, with 50 complete estimates per Hamiltonian, method
and measurement count. Methods share identical inputs. Each curve is the
arithmetic mean of instance-level RMSEs over 50 Hamiltonians, not a pooled
RMSE. All seven measurement counts, including T=12, are retained.

## Build from frozen production results

Run from the repository root:

```sh
python manuscript_data_gen/materialize_data.py
python manuscript_data_gen/build.py manuscript_data_gen/main/fig05_random_hamiltonian_errors --output build/general_hamiltonian_gpd_average_errors.pdf
```

The current data are materialized at
`manuscript_data_gen/reproduction/data/random_ensemble_50/` from the
completed 2026-11-02 production experiment. The output-adjacent
`*_rebuild/` directory retains summaries, panels and validation records.
This release update verifies and reconstructs saved results; it does not
rerun all experimental optimization or measurement sampling.

For independent validation and combined/single-panel outputs:

```sh
python manuscript_data_gen/reproduction/code/experiments/plot_random_ensemble.py --audit-only
python manuscript_data_gen/reproduction/code/experiments/plot_random_ensemble.py
```

These commands write to `build/random_ensemble_50/` by default and audit
4,200 instance records, 210,000 complete estimates, paired inputs, stopping
and 84 ensemble means. The original five-instance-per-regime data remain
available for legacy checks; they are not the current figure source.

See the [ensemble guide](../../reproduction/code/experiments/RANDOM_ENSEMBLE_README.md)
for data schemas, statistical conventions, portable overrides, the legacy
comparison and optional de novo commands. See the
[reproduction guide](../../reproduction/FIGURE_REPRODUCTION.md) for renderer
and font differences. JAX is unnecessary for saved-result reconstruction.
