"""Bounded import, input, and saved-rank checks, without fitting new ranks.

Run from the reproduction directory after materialize_data.py. This verifies
Hamiltonian recovery and the numerical contents of released rank artifacts;
it is not a claim that every stochastic simulation was rerun.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "methods" / "srdd"))
from srdd_release_paths import DATA, molecular_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true")
    args = parser.parse_args()
    modules = (
        "h4_bond_scan_srdd", "h4_bond_scan_pauli", "molecular_srdd_rank_fit",
        "molecular_srdd_selection", "large_molecule_srdd_pauli", "beh2_entry",
        "prepare_beh2_input",
    )
    for name in modules:
        importlib.import_module(name)
    print(json.dumps({"imports": list(modules), "status": "PASS"}))
    if args.imports_only:
        return
    from five_molecule_cases import (
        SOURCE_SPECS, _active_embedding_indices, _open_dense_source,
        load_five_molecule_case,
    )
    from stabilizer_calibration import ground_state
    import large_molecule_srdd_pauli as large
    for molecule in ("H4", "H6"):
        case = load_five_molecule_case(molecule)
        spec = SOURCE_SPECS[molecule]
        indices = _active_embedding_indices(spec)
        with _open_dense_source(spec) as source:
            original_block = np.asarray(source[np.ix_(indices, indices)], dtype=np.complex128)
        delta = float(np.linalg.norm(case.target - original_block))
        if delta > 1.0e-10:
            raise AssertionError(f"{molecule}: Hamiltonian recovery error {delta}")
        state, energy = ground_state(case)
        checked = []
        for rank in (1, 10):
            path = molecular_result(DATA, molecule, "selected") / f"selected_rank_K{rank:02d}.npz"
            with np.load(path, allow_pickle=False) as values:
                for key in values.files:
                    if np.issubdtype(values[key].dtype, np.number) and not np.all(np.isfinite(values[key])):
                        raise AssertionError(f"Nonfinite {molecule} K={rank} {key}")
                checked.append({"rank": rank, "arrays": values.files})
        print(json.dumps({"molecule": molecule, "energy_hartree": energy,
                          "Hamiltonian_reconstruction_frobenius": delta,
                          "eigen_residual": float(np.linalg.norm(case.target @ state - energy * state)),
                          "rank_files": checked, "status": "PASS"}))
    for molecule, electrons in (("BeH2", 6), ("N2", 14)):
        case = large.load_case(large.CaseSpec(molecule, "", "released geometry", electrons))
        print(json.dumps({"molecule": molecule, "spatial_orbitals": case["m"],
                          "qubits": case["n"], "status": "PASS"}))


if __name__ == "__main__":
    main()
