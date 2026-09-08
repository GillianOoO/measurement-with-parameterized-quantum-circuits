# Main Figure 2

Output: `molecular_hamiltonian_errors.pdf` and a matching PNG.

H4 bond scan, fixed H4 R=1.20 A, and fixed H6 R=3.40 A. The H4 Pauli curves use the audited same-Hamiltonian rerun.

```powershell
python manuscript_artifacts/build.py manuscript_artifacts/main/fig02_molecular_errors --output build/molecular_hamiltonian_errors.pdf
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
