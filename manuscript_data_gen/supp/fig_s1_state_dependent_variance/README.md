# SI Figure 1

Output: `supp_state_dependent_variance_vs_budget.pdf` and a matching PNG.

Each of six variance panels touches a lower inverted absolute-bias histogram, sharing a logarithmic x-axis. Lower axes are independent, linear, on the left, and start at zero at the common border. Tick labels show nonnegative bias magnitudes. Molecular bars use SRDD absolute bias; random bars use the mean of five individual absolute GPD biases. All methods and bars in panels (e,f) use 45,160,572,2038,7259,25848, omitting T=12 only from this SI display. There are 38 plotted bias values; raw data retain T=12.

```powershell
python manuscript_data_gen/build.py manuscript_data_gen/supp/fig_s1_state_dependent_variance --output build/supp_state_dependent_variance_vs_budget.pdf
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
