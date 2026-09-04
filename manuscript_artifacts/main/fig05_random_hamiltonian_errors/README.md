# Main Figure 5: sparse and dense Hamiltonians

Output: `general_hamiltonian_gpd_average_errors.pdf` and a matching PNG.

Use `manuscript_random_stabilizer_gpd_comparison.csv`. Required columns are
`benchmark`, `method`, `shots`, and `mean_instance_empirical_rmse`. Every curve
is the arithmetic mean of five instance-level empirical RMSEs, each computed
from 50 finite-measurement estimates. GPD must have selection metadata equal to
`stabilizer-calibrated balanced GPD`; analytic-RMSE and AGPD curves are not
rendered.
