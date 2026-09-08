# Main Figure 3

Output: `additional_molecular_hamiltonian_errors.pdf` and a matching PNG.

BeH2 and N2 each have error/measurements, required-samples/target-error and error/noise panels. Error and noise curves use 50 complete estimates per point. Inverse curves display ten points; noise uses 3000 measurements and p in [0,0.003]. Labels are unit-free and legends have no frame.

```powershell
python manuscript_artifacts/build.py manuscript_artifacts/main/fig03_additional_molecules --output build/additional_molecular_hamiltonian_errors.pdf
```

First run `python manuscript_artifacts/prepare_inputs.py --archive-root PATH_TO_paper_reproducibility`.
The figure command reads these hash-verified local inputs and the current plotter;
no manually normalized `--input` CSV is needed. `artifact.json` records the
figure mapping and policies; the Python plotter is the layout source of truth.
The output-adjacent `*_rebuild/` directory retains data presentations,
individual panels where available, and validation manifests.

See [the reproduction guide](../../reproduction/FIGURE_REPRODUCTION.md) for
input hashes, data conventions and the distinction between approved PDF assets
and runtime-specific rebuild references. The response document reuses the
same Figure 3, Figure 4 and SI Figure 1 assets.
