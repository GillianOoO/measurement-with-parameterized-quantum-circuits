# Main Figure 2: molecular estimation errors

Output: `molecular_hamiltonian_errors.pdf` and a matching PNG.

The normalized input columns are `panel`, `method`, `bond_length_angstrom`,
`measurements`, and `empirical_rmse`. Each value is an empirical RMSE from the
registered finite-measurement replays. GPD is intentionally absent from these
molecular panels.

```powershell
python manuscript_artifacts/build.py manuscript_artifacts/main/fig02_molecular_errors `
  --input build/molecular_errors.csv `
  --output build/molecular_hamiltonian_errors.pdf
```
