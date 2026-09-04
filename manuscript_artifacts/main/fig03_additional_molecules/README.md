# Main Figure 3: BeH2 and N2

Output: `additional_molecular_hamiltonian_errors.pdf` and a matching PNG.

The normalized input has columns `panel`, `method`, `x`, and `value`. The six
panel identifiers are listed in `artifact.json`. GPD is absent; the methods are
SRDD and the applicable FC-IMA/Pauli baselines. The error panels use empirical
finite-measurement RMSE, the required-measurement panels use frozen-design
projections, and the noise panels use the fixed `T=3000` designs.
