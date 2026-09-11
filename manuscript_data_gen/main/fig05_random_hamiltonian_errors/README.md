# Main Figure 5

Output: `general_hamiltonian_gpd_average_errors.pdf` and a matching PNG.

Sparse/dense ensembles use the current stabilizer-calibrated GPD prefixes and the audited Pauli comparison methods. Every method retains all seven measurement counts, including T=12; the SI-only display restriction is not applied here.

```powershell
python manuscript_data_gen/build.py manuscript_data_gen/main/fig05_random_hamiltonian_errors --output build/general_hamiltonian_gpd_average_errors.pdf
```

First run `python manuscript_data_gen/materialize_data.py`.
The figure command reads these hash-verified local inputs and the current plotter;
no manually normalized `--input` CSV is needed. `artifact.json` records the
figure mapping and policies; the Python plotter is the layout source of truth.
The output-adjacent `*_rebuild/` directory retains data presentations,
individual panels where available, and validation manifests.

See [the reproduction guide](../../reproduction/FIGURE_REPRODUCTION.md) for
input hashes, data conventions and the distinction between approved PDF assets
and runtime-specific rebuild references. The response document reuses the
same Figure 3, Figure 4 and SI Figure 1 assets.
