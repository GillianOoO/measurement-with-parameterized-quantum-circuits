# Main Figure 1: framework diagram

Output: `main_sketch_revise.pdf`.

The builder preserves the source vector PDF and applies the publication overlay
for the GPD and s-RCDF labels and shallow Givens-ladder glyph. Font paths are
command-line options; the defaults match the Windows manuscript workstation.

```powershell
python manuscript_artifacts/main/fig01_framework/build.py `
  --source path/to/main_sketch.pdf `
  --output build/main_sketch_revise.pdf
```
