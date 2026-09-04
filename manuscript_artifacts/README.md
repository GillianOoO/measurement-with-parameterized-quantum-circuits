# Manuscript artifact builders

This directory maps every generated main-text figure and Supplementary
Information figure/table to a dedicated artifact folder. Shared rendering code
lives in `build.py`; each folder contains an `artifact.json` with the exact
panel, field, scale, method-order, and output configuration for that artifact.

The directory is the publication layer of the existing `mpqc_measurement`
package rather than a second implementation of the measurement algorithms.
The package writes normalized CSV presentations, and the builders read those
CSVs directly without digitizing PDF or PNG files. GPD rows used by the
random-Hamiltonian artifacts must be produced by the stabilizer-calibrated
balanced selector implemented in `mpqc_measurement.selection`:

```text
x_a(K)   = ||Pi (H - H_hat_K) Pi||_F
x_s(K,T) = sqrt(sum_i lambda_i(K)^2 / T_i)
L(K,T)   = hypot(w_a x_a(K), w_s x_s(K,T))
```

Run a builder from the repository root:

```powershell
python manuscript_artifacts/build.py `
  manuscript_artifacts/main/fig05_random_hamiltonian_errors `
  --input path/to/manuscript_random_stabilizer_gpd_comparison.csv `
  --output build/general_hamiltonian_gpd_average_errors.pdf
```

Each artifact README names the expected manuscript filename and CSV schema.
The source Hamiltonians, finite-measurement replay records, fitted calibration
metadata, and hash manifests remain experiment outputs and are not embedded in
the plotting code.

## Artifact index

- `main/fig01_framework`: framework diagram overlay builder.
- `main/fig02_molecular_errors`: H4/H6 molecular error panels.
- `main/fig03_additional_molecules`: BeH2/N2 error, resource, and noise panels.
- `main/fig04_resource_summary`: variance, allocation, and required-measurement bars.
- `main/fig05_random_hamiltonian_errors`: sparse/dense empirical-RMSE panels.
- `supp/fig_s1_state_dependent_variance`: molecular and sparse/dense variance panels.
- `supp/table_s1_state_dependent_variance`: exact variance, bias, and MSE table.
- `supp/table_s2_required_measurements`: projected required measurements.
- `supp/table_s3_cnot_counts`: projected executed CNOT counts.
