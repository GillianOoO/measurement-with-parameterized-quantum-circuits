#!/usr/bin/env python3
"""Run SG, Derandomization, and OGM on the exact same LiH H and state.

The Hamiltonian is LiH R=1.50 Angstrom in the blocked-spin Jordan-Wigner
ordering, and every method samples the same saved exact FCI ground-state MPS.
Shadow Grouping is imported from the retained comparison package.
Derandomization uses the stable Huang--Kueng--Preskill Appendix-C implementation
in ``derand_huang2021.py``; the retained package is used only for a short-prefix
compatibility check. OGM candidates are overlapping QWC groups generated from
the same Pauli coefficients; their probabilities minimize the standard diagonal
OGM proxy sum_i w_i^2 / q_i on the probability simplex.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from release_paths import BUNDLE_ROOT, DATA_ROOT, OUTPUT_ROOT
SCAN_DIR = DATA_ROOT / "legacy_lih"
CURVES_DIR = DATA_ROOT
SELECTED_DIR = DATA_ROOT / "LiH" / "inputs"
PAULI_CSV = SELECTED_DIR / "hamiltonian_pauli_blocked_spin.csv"
METADATA_PATH = SELECTED_DIR / "metadata.json"
STATE_PATH = SELECTED_DIR / "ground_state_mps_blocked_spin.npz"
HUANG_DERAND_PATH = HERE / "derand_huang2021.py"
REVIEWED_PACKAGE = HERE
sys.path.insert(0, str(HERE))
import estimate_sampling_errors as sampling
from derand_huang2021 import appendix_c_schedule
from shadowgrouping.measurement_schemes import (
    Derandomization,
    Shadow_Grouping,
)
from shadowgrouping.weight_functions import Bernstein_bound


SHOT_BUDGETS = (12, 45, 160, 572, 2038, 7259, 25848, 92041)
PAULI_SHOT_BUDGETS = SHOT_BUDGETS
REPEATS = 50
SEED = 7259
EPSILON = 0.1
PAULI_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
S_DAGGER = np.diag([1.0, -1.0j]).astype(np.complex128)
BASIS_ROTATIONS = {
    0: np.eye(2, dtype=np.complex128),
    1: HADAMARD,
    2: HADAMARD @ S_DAGGER,
    3: np.eye(2, dtype=np.complex128),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_problem():
    frame = pd.read_csv(PAULI_CSV)
    coefficients = (
        frame["coefficient_real_hartree"].to_numpy(dtype=float)
        + 1j * frame["coefficient_imag_hartree"].to_numpy(dtype=float)
    )
    if float(np.max(np.abs(coefficients.imag))) > 1e-12:
        raise ValueError("Hamiltonian has unexpectedly complex coefficients")
    labels = frame["full_label_site0_to_siteNminus1"].astype(str).to_list()
    observables = np.array(
        [[PAULI_CODE[character] for character in label] for label in labels],
        dtype=np.int8,
    )
    identity = np.all(observables == 0, axis=1)
    if int(identity.sum()) != 1:
        raise ValueError(f"Expected one identity term, got {identity.sum()}")
    offset = float(coefficients.real[identity][0])
    observables = observables[~identity]
    weights = coefficients.real[~identity]
    state, state_max_bond = sampling.load_dense_mps(STATE_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    exact_energy = float(metadata["fci_ground_energy_hartree"])
    return observables, weights, offset, state, state_max_bond, exact_energy


def hit_matrix(observables: np.ndarray, settings: np.ndarray) -> np.ndarray:
    return np.all(
        (observables[:, None, :] == 0)
        | (observables[:, None, :] == settings[None, :, :]),
        axis=2,
    )


def setting_hits(observables: np.ndarray, setting: np.ndarray) -> np.ndarray:
    return np.all((observables == 0) | (observables == setting), axis=1)


def vectorized_shadow_grouping_schedule(
    observables: np.ndarray, weights: np.ndarray, max_shots: int
) -> np.ndarray:
    alpha = float(np.max(np.abs(weights)) / np.min(np.abs(weights)) + np.min(np.abs(weights)))
    hits = np.zeros(weights.size, dtype=np.int64)
    base = alpha * np.abs(weights)
    rows = np.empty((max_shots, observables.shape[1]), dtype=np.int8)
    for shot in range(max_shots):
        scores = base.copy()
        measured = hits != 0
        if np.any(measured):
            root_n = np.sqrt(hits[measured])
            root_next = np.sqrt(hits[measured] + 1)
            divisor = alpha * root_n * root_next / (root_next - root_n)
            scores[measured] /= divisor
        # Match the reviewed implementation's default np.argsort tie order.
        order = np.argsort(scores)
        setting = np.zeros(observables.shape[1], dtype=np.int8)
        for term_index in order[::-1]:
            term = observables[term_index]
            if np.all((term == 0) | (setting == 0) | (term == setting)):
                mask = term != 0
                setting[mask] = term[mask]
                if np.all(setting != 0):
                    break
        hits += setting_hits(observables, setting)
        rows[shot] = setting
    return rows


def vectorized_derandomization_schedule(
    observables: np.ndarray,
    weights: np.ndarray,
    max_shots: int,
    *,
    initial_hits: np.ndarray | None = None,
    return_audit: bool = False,
):
    """Compatibility wrapper for the paper-C8/C11 fixed-budget stream.

    ``initial_hits`` is used when a separately constructed QWC coverage prefix
    has already been measured.  It prevents the historical error of prepending
    a cover while restarting Huang's multiplicative-weight state from zero.
    """

    return appendix_c_schedule(
        observables,
        weights,
        max_shots,
        initial_hits=initial_hits,
        return_audit=return_audit,
    )


def verify_vectorized_prefix(
    observables: np.ndarray,
    weights: np.ndarray,
    sg_schedule: np.ndarray,
    derand_schedule: np.ndarray,
    count: int = 24,
) -> dict[str, object]:
    alpha = float(np.max(np.abs(weights)) / np.min(np.abs(weights)) + np.min(np.abs(weights)))
    reviewed_sg = Shadow_Grouping(
        observables, weights, EPSILON, Bernstein_bound(alpha=alpha)()
    )
    # This retained third-party class is not the production implementation.  It
    # is kept only to verify that the corrected paper implementation preserves
    # the historical short prefix used in earlier archived calculations.
    reviewed_derand = Derandomization(
        observables, weights, math.sqrt(0.9), use_one_norm=True
    )
    reviewed = {}
    for name, scheme in (
        ("Shadow Grouping", reviewed_sg),
        ("Derandomization", reviewed_derand),
    ):
        rows = []
        for _ in range(count):
            setting, _ = scheme.find_setting()
            rows.append(np.asarray(setting, dtype=np.int8))
        reviewed[name] = np.asarray(rows)
    sg_equal = bool(np.array_equal(reviewed["Shadow Grouping"], sg_schedule[:count]))
    derand_equal = bool(
        np.array_equal(reviewed["Derandomization"], derand_schedule[:count])
    )
    if not sg_equal:
        raise AssertionError(
            "Vectorized Shadow Grouping schedule differs from its retained "
            f"reference prefix: SG={sg_equal}"
        )
    equal_rows = np.all(
        reviewed["Derandomization"] == derand_schedule[:count], axis=1
    )
    mismatch = np.flatnonzero(~equal_rows)
    return {
        "checked_prefix_settings": count,
        "Shadow Grouping_exact_match": sg_equal,
        "Derandomization_exact_match": derand_equal,
        "Derandomization_historical_equal_prefix_length": (
            int(mismatch[0]) if mismatch.size else int(count)
        ),
        "Derandomization_historical_comparison_is_release_gate": False,
        "Derandomization_authoritative_validation": (
            "outputs/derand_huang2021_validation: official C++ golden and "
            "independent paper-C8/C11 scalar tests"
        ),
    }


def generate_reviewed_schedules(
    observables: np.ndarray, weights: np.ndarray, max_shots: int
):
    sg = vectorized_shadow_grouping_schedule(observables, weights, max_shots)
    derand = vectorized_derandomization_schedule(observables, weights, max_shots)
    verification = verify_vectorized_prefix(observables, weights, sg, derand)
    schedules = {"Shadow Grouping": sg, "Derandomization": derand}
    zero_counts = {
        method: int(np.count_nonzero(schedule == 0))
        for method, schedule in schedules.items()
    }
    if any(zero_counts.values()):
        raise RuntimeError(f"Incomplete measurement settings: {zero_counts}")
    return schedules, zero_counts, verification


def generate_ogm_candidates(
    observables: np.ndarray,
    weights: np.ndarray,
    extra_settings: list[np.ndarray],
):
    order = np.argsort(-np.abs(weights), kind="stable")
    candidate_tuples = set()
    for seed_index in order:
        setting = observables[seed_index].copy()
        for term_index in order:
            term = observables[term_index]
            if np.all((term == 0) | (setting == 0) | (term == setting)):
                mask = (setting == 0) & (term != 0)
                setting[mask] = term[mask]
        setting[setting == 0] = 3
        candidate_tuples.add(tuple(int(value) for value in setting))
    for setting in extra_settings:
        completed = np.asarray(setting, dtype=np.int8).copy()
        completed[completed == 0] = 3
        candidate_tuples.add(tuple(int(value) for value in completed))

    candidates = np.array(sorted(candidate_tuples), dtype=np.int8)
    hits = hit_matrix(observables, candidates)
    if np.any(hits.sum(axis=1) == 0):
        raise RuntimeError("At least one Pauli term is uncovered by OGM candidates")

    # Columns with identical coverage have the same diagonal proxy. Keep one
    # deterministic representative to make the convex optimization smaller.
    unique_columns = {}
    for column in range(hits.shape[1]):
        key = np.packbits(hits[:, column]).tobytes()
        unique_columns.setdefault(key, column)
    keep = np.array(list(unique_columns.values()), dtype=int)
    return candidates[keep], hits[:, keep]


def optimize_ogm_probabilities(weights: np.ndarray, hits: np.ndarray):
    coverage = hits.astype(np.float64)
    weights_sq = weights * weights
    initial_scores = np.abs(weights) @ coverage
    initial = np.maximum(initial_scores, 1e-12)
    initial /= initial.sum()

    def objective(probabilities: np.ndarray) -> float:
        q = coverage @ probabilities
        return float(np.sum(weights_sq / np.maximum(q, 1e-15)))

    def gradient(probabilities: np.ndarray) -> np.ndarray:
        q = coverage @ probabilities
        return -(coverage.T @ (weights_sq / np.maximum(q, 1e-15) ** 2))

    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        values = np.exp(shifted)
        return values / values.sum()

    def objective_logits(logits: np.ndarray) -> float:
        return objective(softmax(logits))

    def gradient_logits(logits: np.ndarray) -> np.ndarray:
        probabilities = softmax(logits)
        probability_gradient = gradient(probabilities)
        return probabilities * (
            probability_gradient
            - float(np.dot(probability_gradient, probabilities))
        )

    result = minimize(
        objective_logits,
        np.log(initial),
        jac=gradient_logits,
        method="L-BFGS-B",
        options={
            "maxiter": 2000,
            "ftol": 1e-14,
            "gtol": 1e-9,
            "maxls": 50,
        },
    )
    probabilities = softmax(np.asarray(result.x, dtype=float))
    return probabilities, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "initial_objective": objective(initial),
        "final_objective": objective(probabilities),
        "probability_sum": float(probabilities.sum()),
        "nonzero_probabilities_above_1e-10": int(
            np.count_nonzero(probabilities > 1e-10)
        ),
        "optimizer": "L-BFGS-B on softmax logits",
    }


def build_sign_table(observables: np.ndarray):
    nsites = observables.shape[1]
    masks = np.zeros(observables.shape[0], dtype=np.uint16)
    for site in range(nsites):
        masks |= ((observables[:, site] != 0).astype(np.uint16) << (nsites - 1 - site))
    unique_masks, inverse = np.unique(masks, return_inverse=True)
    outcomes = np.arange(2**nsites, dtype=np.uint16)
    parity = np.array(
        [bin(int(value)).count("1") & 1 for value in outcomes],
        dtype=np.int8,
    )
    signs = 1 - 2 * parity[np.bitwise_and(unique_masks[:, None], outcomes[None, :])]
    return signs.astype(np.int8), inverse


class ExactStateMeasurementSampler:
    def __init__(self, state: np.ndarray, observables: np.ndarray):
        self.state = np.asarray(state, dtype=np.complex128)
        self.observables = observables
        self.nsites = observables.shape[1]
        self.probability_cache: dict[tuple[int, ...], np.ndarray] = {}
        self.sign_table, self.support_indices = build_sign_table(observables)

    def probabilities(self, setting_tuple: tuple[int, ...]) -> np.ndarray:
        cached = self.probability_cache.get(setting_tuple)
        if cached is not None:
            return cached
        transformed = self.state.copy()
        for site, basis in enumerate(setting_tuple):
            gate = BASIS_ROTATIONS[int(basis)]
            if int(basis) in (1, 2):
                transformed = sampling.apply_one_site_gate(
                    transformed, gate, site, self.nsites
                )
        probabilities = np.abs(transformed) ** 2
        probabilities /= probabilities.sum()
        self.probability_cache[setting_tuple] = probabilities
        return probabilities

    def estimate_energy(
        self,
        settings_counts: Counter,
        weights: np.ndarray,
        offset: float,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        running_sum = np.zeros(weights.size, dtype=np.float64)
        running_count = np.zeros(weights.size, dtype=np.int64)
        for setting_tuple, repetitions in settings_counts.items():
            setting = np.asarray(setting_tuple, dtype=np.int8)
            probabilities = self.probabilities(setting_tuple)
            outcome_counts = rng.multinomial(int(repetitions), probabilities)
            hit = setting_hits(self.observables, setting)
            indices = np.flatnonzero(hit)
            if indices.size:
                products = (
                    self.sign_table[self.support_indices[indices]].astype(np.int64)
                    @ outcome_counts.astype(np.int64)
                )
                running_sum[indices] += products
                running_count[indices] += int(repetitions)
        averages = np.zeros_like(running_sum)
        measured = running_count > 0
        averages[measured] = running_sum[measured] / running_count[measured]
        energy = float(offset + np.dot(weights, averages))
        return energy, int(np.count_nonzero(~measured))


def counter_from_schedule(schedule: np.ndarray, shots: int) -> Counter:
    return Counter(tuple(int(value) for value in row) for row in schedule[:shots])


def counter_from_ogm(
    candidates: np.ndarray,
    probabilities: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> Counter:
    counts = rng.multinomial(shots, probabilities)
    return Counter(
        {
            tuple(int(value) for value in candidates[index]): int(count)
            for index, count in enumerate(counts)
            if count > 0
        }
    )


def run_estimators(
    observables,
    weights,
    offset,
    state,
    exact_energy,
    schedules,
    ogm_candidates,
    ogm_probabilities,
):
    simulator = ExactStateMeasurementSampler(state, observables)
    repeat_rows = []
    summary_rows = []
    for method_index, method in enumerate(
        ("Shadow Grouping", "Derandomization", "OGM")
    ):
        for shots in PAULI_SHOT_BUDGETS:
            estimates = np.empty(REPEATS, dtype=float)
            missing_terms = np.empty(REPEATS, dtype=int)
            fixed_counter = (
                counter_from_schedule(schedules[method], shots)
                if method != "OGM"
                else None
            )
            for repeat in range(REPEATS):
                rng = np.random.default_rng(
                    SEED + 1000003 * method_index + 1009 * shots + repeat
                )
                counter = (
                    fixed_counter
                    if fixed_counter is not None
                    else counter_from_ogm(
                        ogm_candidates, ogm_probabilities, shots, rng
                    )
                )
                estimate, missing = simulator.estimate_energy(
                    counter, weights, offset, rng
                )
                estimates[repeat] = estimate
                missing_terms[repeat] = missing
                repeat_rows.append(
                    {
                        "molecule": "LiH",
                        "bond_length_angstrom": 1.50,
                        "mapping": "blocked-spin Jordan-Wigner",
                        "method": method,
                        "requested_shots": shots,
                        "effective_shots": int(sum(counter.values())),
                        "repeat": repeat + 1,
                        "energy_estimate_hartree": estimate,
                        "signed_error_vs_exact_H_hartree": estimate - exact_energy,
                        "absolute_error_vs_exact_H_hartree": abs(
                            estimate - exact_energy
                        ),
                        "unmeasured_nonidentity_pauli_terms": missing,
                    }
                )
            residuals = estimates - exact_energy
            summary_rows.append(
                {
                    "molecule": "LiH",
                    "bond_length_angstrom": 1.50,
                    "mapping": "blocked-spin Jordan-Wigner",
                    "method": method,
                    "repeat_count": REPEATS,
                    "requested_shots": shots,
                    "effective_shots": shots,
                    "exact_H_ground_energy_hartree": exact_energy,
                    "mean_estimated_energy_hartree": float(estimates.mean()),
                    "empirical_rmse_vs_exact_H_hartree": float(
                        np.sqrt(np.mean(residuals**2))
                    ),
                    "empirical_mae_vs_exact_H_hartree": float(
                        np.mean(np.abs(residuals))
                    ),
                    "empirical_standard_deviation_hartree": float(
                        np.std(estimates, ddof=1)
                    ),
                    "mean_unmeasured_nonidentity_pauli_terms": float(
                        missing_terms.mean()
                    ),
                    "maximum_unmeasured_nonidentity_pauli_terms": int(
                        missing_terms.max()
                    ),
                }
            )
            print(
                f"[{method}] shots={shots}: "
                f"RMSE={summary_rows[-1]['empirical_rmse_vs_exact_H_hartree']:.9e}, "
                f"cached_bases={len(simulator.probability_cache)}",
                flush=True,
            )
    return pd.DataFrame(repeat_rows), pd.DataFrame(summary_rows), simulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-shots", type=int, default=max(PAULI_SHOT_BUDGETS),
        help="Generate deterministic schedules to this many settings.",
    )
    args = parser.parse_args()
    if args.max_shots < max(PAULI_SHOT_BUDGETS):
        raise ValueError(
            f"--max-shots must be at least {max(PAULI_SHOT_BUDGETS)}"
        )
    started = time.time()
    observables, weights, offset, state, state_max_bond, exact_energy = (
        load_problem()
    )
    print(
        f"[input] terms={weights.size}, qubits={observables.shape[1]}, "
        f"state_dim={state.size}, E={exact_energy:.15f}",
        flush=True,
    )
    schedules, zero_counts, prefix_verification = generate_reviewed_schedules(
        observables, weights, args.max_shots
    )
    unique_sg = np.unique(schedules["Shadow Grouping"], axis=0)
    unique_derand = np.unique(schedules["Derandomization"], axis=0)
    ogm_candidates, ogm_hits = generate_ogm_candidates(
        observables,
        weights,
        [],
    )
    print(
        f"[OGM] optimizing {ogm_candidates.shape[0]} candidate settings",
        flush=True,
    )
    ogm_probabilities, optimization = optimize_ogm_probabilities(
        weights, ogm_hits
    )
    if not optimization["success"]:
        raise RuntimeError(f"OGM optimization failed: {optimization}")
    print(
        f"[OGM] candidates={ogm_candidates.shape[0]}, "
        f"active={optimization['nonzero_probabilities_above_1e-10']}, "
        f"objective={optimization['final_objective']:.9e}",
        flush=True,
    )
    np.savez_compressed(
        HERE / "same_H_ogm_distribution.npz",
        settings=ogm_candidates,
        probabilities=ogm_probabilities,
        hit_matrix=ogm_hits,
    )
    repeats, summary, simulator = run_estimators(
        observables,
        weights,
        offset,
        state,
        exact_energy,
        schedules,
        ogm_candidates,
        ogm_probabilities,
    )
    repeats_path = HERE / "same_H_pauli_method_repeats.csv"
    summary_path = HERE / "same_H_pauli_method_summary.csv"
    repeats.to_csv(repeats_path, index=False)
    summary.to_csv(summary_path, index=False)

    settings_rows = []
    for method in ("Shadow Grouping", "Derandomization"):
        for shots in PAULI_SHOT_BUDGETS:
            counter = counter_from_schedule(schedules[method], shots)
            setting_array = np.array(list(counter), dtype=np.int8)
            hits = hit_matrix(observables, setting_array)
            weighted_hits = hits @ np.array(list(counter.values()), dtype=int)
            settings_rows.append(
                {
                    "method": method,
                    "requested_shots": shots,
                    "unique_settings": len(counter),
                    "minimum_term_hits": int(weighted_hits.min()),
                    "maximum_term_hits": int(weighted_hits.max()),
                    "unmeasured_terms": int(np.count_nonzero(weighted_hits == 0)),
                }
            )
    pd.DataFrame(settings_rows).to_csv(
        HERE / "same_H_measurement_settings_summary.csv", index=False
    )

    manifest = {
        "configuration": {
            "molecule": "LiH",
            "bond_length_angstrom": 1.50,
            "basis": "STO-3G",
            "mapping": "blocked-spin Jordan-Wigner",
            "number_qubits": int(observables.shape[1]),
            "nonidentity_pauli_terms": int(weights.size),
            "ground_state": "same exact FCI MPS used by corrected TND sampling",
            "ground_state_mps_max_bond": state_max_bond,
            "exact_energy_hartree": exact_energy,
        },
        "sampling": {
            "shot_budgets": list(PAULI_SHOT_BUDGETS),
            "repeat_count": REPEATS,
            "seed": SEED,
            "error": "RMSE against the exact H energy over 50 repeats",
            "reviewed_schedule_prefix_verification": prefix_verification,
        },
        "methods": {
            "Shadow Grouping": {
                "setting_generator": (
                    "reviewed shadowgrouping.measurement_schemes.Shadow_Grouping"
                ),
                "epsilon": EPSILON,
                "zero_basis_entries_over_full_schedule": zero_counts[
                    "Shadow Grouping"
                ],
            },
            "Derandomization": {
                "setting_generator": (
                    "Huang-2021 Appendix C C8/C11 stable affected-term delta"
                ),
                "algorithm_id": "huang2021-appendix-c-fixed-budget-stable-delta-v1",
                "eta": 0.9,
                "implementation_path": str(HUANG_DERAND_PATH),
                "implementation_sha256": sha256_file(HUANG_DERAND_PATH),
                "zero_basis_entries_over_full_schedule": zero_counts[
                    "Derandomization"
                ],
            },
            "OGM": {
                "candidate_generation": (
                    "overlapping greedy QWC groups seeded by every Hamiltonian "
                    "term and deduplicated by Pauli-term coverage"
                ),
                "probability_objective": "minimize sum_i w_i^2/q_i",
                "candidate_settings": int(ogm_candidates.shape[0]),
                "optimization": optimization,
            },
        },
        "basis_probability_vectors_cached": len(simulator.probability_cache),
        "inputs": {
            "pauli_csv": str(PAULI_CSV),
            "pauli_csv_sha256": sha256_file(PAULI_CSV),
            "metadata": str(METADATA_PATH),
            "metadata_sha256": sha256_file(METADATA_PATH),
            "ground_state_mps": str(STATE_PATH),
            "ground_state_mps_sha256": sha256_file(STATE_PATH),
        },
        "reviewed_code": {
            "role": "Shadow Grouping production; Derandomization historical-prefix diagnostic only",
            "package_root": str(REVIEWED_PACKAGE),
            "measurement_schemes_sha256": sha256_file(
                REVIEWED_PACKAGE / "shadowgrouping/measurement_schemes.py"
            ),
            "weight_functions_sha256": sha256_file(
                REVIEWED_PACKAGE / "shadowgrouping/weight_functions.py"
            ),
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (
                repeats_path,
                summary_path,
                HERE / "same_H_ogm_distribution.npz",
                HERE / "same_H_measurement_settings_summary.csv",
            )
        },
        "elapsed_seconds": time.time() - started,
    }
    (HERE / "same_H_pauli_method_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"[complete] elapsed={time.time() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
