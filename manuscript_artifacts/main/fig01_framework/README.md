# Main Figure 1

Output: `main_sketch_revise.pdf`.

The default operation copies the reviewed static SRDD/GPD schematic. Its editable PPTX is retained in the local archive. No older s-RCDF overlay is applied.

```powershell
python manuscript_artifacts/build.py manuscript_artifacts/main/fig01_framework --output build/main_sketch_revise.pdf
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
