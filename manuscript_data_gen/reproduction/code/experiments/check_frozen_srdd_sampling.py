"""Compare fresh finite-shot draws from saved settings with published samples.

This diagnostic isolates sampling/eigensolver differences from rank fitting.
It prints differences and does not replace any published results. A successful
process exit means the diagnostic ran, not that all differences were zero.
"""

from pathlib import Path
import csv
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "methods" / "srdd"))
from srdd_release_paths import DATA, molecular_result
from five_molecule_cases import load_five_molecule_case
from stabilizer_calibration import ground_state


def main():
    reports = []
    for molecule in ("H4", "H6"):
        root = molecular_result(DATA, molecule, "selected")
        state, energy = ground_state(load_five_molecule_case(molecule))
        with (root / "sampling_summary.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        with (root / "sampling_replicates.csv").open(newline="") as handle:
            replicates = list(csv.DictReader(handle))
        for row in rows:
            rank = int(row["K_source_terms"])
            with np.load(root / f"selected_rank_K{rank:02d}.npz") as saved:
                matrices = saved["setting_matrices"]
            rng = np.random.default_rng(int(row["sampling_seed"]))
            estimates = np.zeros(int(row["sampling_repeats"]))
            shots = list(map(int, row["shot_vector"].split()))
            for matrix, count in zip(matrices, shots):
                values, vectors = np.linalg.eigh(matrix)
                probabilities = np.maximum(np.abs(vectors.conj().T @ state) ** 2, 0.0)
                probabilities /= probabilities.sum()
                if count <= 0:
                    if values[-1] != values[0]:
                        raise RuntimeError("Uncertified zero-shot setting")
                    estimates += values[0]
                else:
                    estimates += rng.multinomial(count, probabilities, size=len(estimates)) @ values / count
            actual = float(np.sqrt(np.mean((estimates - energy) ** 2)))
            expected = float(row["empirical_total_RMSE"])
            errors = np.asarray([
                float(record["signed_total_error"]) for record in replicates
                if int(record["T_total_shots"]) == int(row["T_total_shots"])
                and int(record["K_source_terms"]) == rank
            ])
            if len(errors) != int(row["sampling_repeats"]):
                raise AssertionError("Incomplete archived repeated-run estimates")
            archived_rmse = float(np.sqrt(np.mean(errors ** 2)))
            if abs(archived_rmse - expected) > 1.0e-12:
                raise AssertionError("Archived summary disagrees with archived repeated-run estimates")
            reports.append({"molecule": molecule, "shots": int(row["T_total_shots"]),
                            "published_RMSE": expected, "redrawn_RMSE": actual,
                            "archived_replicate_RMSE": archived_rmse,
                            "archive_summary_difference": abs(archived_rmse - expected),
                            "absolute_difference": abs(actual - expected)})
    print(json.dumps({"numpy": np.__version__, "runs": reports,
                      "max_absolute_difference": max(r["absolute_difference"] for r in reports)}, indent=2))


if __name__ == "__main__":
    main()
