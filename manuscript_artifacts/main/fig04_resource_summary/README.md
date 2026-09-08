# Main Figure 4

Output: `molecular_resource_summary.pdf` and a matching PNG.

The three panels report exact sampling variance, distinct measurement settings and required samples for target error 0.01. H4 uses the fixed R=1.20 A result at 3000 measurements. The two SRDD absolute biases (BeH2: 8.94e-7; N2: 2.59e-6) are caption information, not extra plot bars. The approved PDF remains unchanged.

```powershell
python manuscript_artifacts/build.py manuscript_artifacts/main/fig04_resource_summary --output build/molecular_resource_summary.pdf
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
