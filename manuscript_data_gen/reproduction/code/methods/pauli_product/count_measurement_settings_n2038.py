#!/usr/bin/env python3
"""Count distinct local-Pauli measurement bases at a fixed shot budget.

The non-OGM schemes follow ``shadowgrouping-master/tutorial.ipynb``.  OGM
follows the final allocation loop in ``gen_errors_all_algs/Sample_main.m``
(called by ``Sample_OGMMain.m``) after loading the optimized distribution made
by ``OGM_optimization/main_CutOGM.m``.  For audit purposes the output also
reports how many positive rows and total shots the intermediate
``OGM_meas_convert.py`` floor rule would produce.

Stochastic schemes (AP and LCS) use an explicit NumPy seed so that the reported
counts are reproducible.  SG and Derand use the same parameters as the tutorial.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
SHADOW_ROOT = ROOT
sys.path.insert(0, str(SHADOW_ROOT))

from shadowgrouping.measurement_schemes import (  # noqa: E402
    AdaptiveShadows,
    Derandomization,
    Shadow_Grouping,
)
from shadowgrouping.weight_functions import Bernstein_bound  # noqa: E402


DATASETS = {
    "Structured_8KLocal_plot_input": {
        "hamiltonian": PROJECT_ROOT
        / "Comparison_codes"
        / "OGM"
        / "gen_errors_all_algs"
        / "Hamiltonian"
        / "8-K-Local-H.txt",
        "ogm": PROJECT_ROOT
        / "Comparison_codes"
        / "OGM"
        / "gen_errors_all_algs"
        / "CutSet"
        / "OGM_8-K-Local-H.txt",
    },
    # The plot-input file above uses coefficient 8 for X^8.  The manuscript
    # equation at n=8, J=g=1 instead gives coefficient 2.  Keeping this as a
    # separate audit case prevents those two similarly named models from being
    # silently conflated.  There is no matching archived OGM distribution.
    "Structured_8KLocal_manuscript_equation": {
        "hamiltonian": PROJECT_ROOT
        / "Comparison_codes"
        / "OGM"
        / "gen_errors_all_algs"
        / "Hamiltonian"
        / "8-K-Local-H.txt",
        "weight_overrides": {0: 2.0},
        "ogm": None,
    },
    "TFIM_8": {
        "hamiltonian": SHADOW_ROOT / "Hamiltonians" / "TFIM_8.txt",
        "ogm": SHADOW_ROOT / "OGM_probabilities" / "OGM_TFIM_8.txt",
    },
    "H6_3.4_sto-3g_12": {
        "hamiltonian": SHADOW_ROOT
        / "Hamiltonians"
        / "H6_3.4_sto-3g_ogm_12.txt",
        "ogm": None,
        "ogm_recompute": True,
    },
    "H4_R1.2_sto-3g_8": {
        "hamiltonian": ROOT / "legacy_inputs" / "H4_1.2_sto-3g.pkl",
        "ogm": None,
        "ogm_recompute": True,
    },
    "rand_H_4": {
        "hamiltonian": ROOT / "Hamiltonian" / "rand_H_4.txt",
        "ogm": ROOT / "OGM_optimization" / "CutSet" / "OGM_rand_H_4.txt",
    },
    "sym_rand_H_4": {
        "hamiltonian": ROOT / "Hamiltonian" / "sym_rand_H_4.txt",
        "ogm": ROOT
        / "OGM_optimization"
        / "CutSet"
        / "OGM_sym_rand_H_4.txt",
    },
}


def load_hamiltonian(
    path: Path, weight_overrides: dict[int, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            data = pickle.load(handle)
        operator = data["qubit_hamiltonian"]
        n_qubits = int(data["num_qubits"])
        rows: list[list[int]] = []
        coefficients: list[float] = []
        pauli_code = {"X": 1, "Y": 2, "Z": 3}
        for term, coefficient in operator.terms.items():
            row = [0] * n_qubits
            for qubit, pauli in term:
                row[int(qubit)] = pauli_code[pauli]
            rows.append(row)
            coefficients.append(float(np.real_if_close(coefficient)))
        observables = np.asarray(rows, dtype=int)
        weights = np.asarray(coefficients, dtype=float)
    else:
        data = np.atleast_2d(np.loadtxt(path, dtype=float))
        weights = data[:, 0].copy()
        observables = data[:, 1:].astype(int)
    if weight_overrides:
        for index, value in weight_overrides.items():
            weights[index] = value
    nonidentity = np.any(observables != 0, axis=1)
    return observables[nonidentity], weights[nonidentity]


def qwc_compatible(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all((left == right) | (left == 0) | (right == 0)))


def qwc_merge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.where(left == 0, right, left)


def greedy_ogm_candidates(
    observables: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Translate the grouping loop in OGM_optimization/main_CutOGM.m."""
    order = np.argsort(-np.abs(weights), kind="stable")
    obs = observables[order]
    coeff = weights[order]
    added = np.zeros(len(obs), dtype=bool)
    candidates: list[np.ndarray] = []
    initial_weights: list[float] = []

    while not np.all(added):
        start = int(np.flatnonzero(~added)[0])
        basis = obs[start].copy()
        added[start] = True
        initial_weight = abs(coeff[start])
        for index in range(start + 1, len(obs)):
            if qwc_compatible(basis, obs[index]):
                basis = qwc_merge(basis, obs[index])
                if not added[index]:
                    added[index] = True
                    initial_weight += abs(coeff[index])
        for index in range(start):
            if qwc_compatible(basis, obs[index]):
                basis = qwc_merge(basis, obs[index])
        candidates.append(basis)
        initial_weights.append(initial_weight)

    return np.asarray(candidates), np.asarray(initial_weights, dtype=float)


def optimize_ogm_distribution(
    observables: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Recompute OGM probabilities with the same diagonal-variance objective."""
    from scipy.optimize import minimize

    settings, initial_weights = greedy_ogm_candidates(observables, weights)
    coverage = np.asarray(
        [
            [qwc_compatible(observable, setting) for setting in settings]
            for observable in observables
        ],
        dtype=float,
    )
    squared_weights = np.square(weights)

    def objective(probabilities: np.ndarray) -> float:
        covered_probability = coverage @ probabilities
        if np.any(covered_probability <= 0):
            return 1e100
        return float(np.sum(squared_weights / covered_probability))

    initial = initial_weights / np.sum(initial_weights)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(settings),
        constraints={"type": "eq", "fun": lambda probabilities: np.sum(probabilities) - 1},
        options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
    )
    probabilities = np.clip(result.x, 0.0, None)
    probabilities /= np.sum(probabilities)
    return settings, probabilities, objective(probabilities), bool(result.success)


def allocate_ogm_bases(
    settings: np.ndarray, probabilities: np.ndarray, shots: int, seed: int
) -> int:
    order = np.argsort(-probabilities, kind="stable")
    rng = np.random.RandomState(seed)
    allocated = 0
    used_bases: set[tuple[int, ...]] = set()
    for index in order:
        amount = shots * probabilities[index]
        if amount > 1:
            number = int(np.floor(amount))
            if rng.rand() < np.mod(amount, number):
                number += 1
        else:
            number = 1
        allocated += number
        used_bases.add(tuple(settings[index]))
        if allocated >= shots:
            break
    return len(used_bases)


def count_recomputed_ogm_bases(
    observables: np.ndarray, weights: np.ndarray, shots: int, seed: int
) -> tuple[int, int, int, float, bool]:
    settings, probabilities, objective, success = optimize_ogm_distribution(
        observables, weights
    )
    final_distinct = allocate_ogm_bases(settings, probabilities, shots, seed)
    floor_counts = np.floor(shots * probabilities).astype(int)
    active = floor_counts > 0
    distinct_active = len({tuple(row) for row in settings[active]})
    return (
        final_distinct,
        distinct_active,
        int(np.sum(floor_counts)),
        objective,
        success,
    )


def tutorial_scheme(name: str, observables: np.ndarray, weights: np.ndarray):
    epsilon = 0.1
    if name == "SG":
        abs_weights = np.abs(weights)
        alpha = np.max(abs_weights) / np.min(abs_weights) + np.min(abs_weights)
        return Shadow_Grouping(
            observables,
            weights,
            epsilon,
            Bernstein_bound(alpha=alpha)(),
        )
    if name == "Derand":
        return Derandomization(
            observables,
            weights,
            np.sqrt(0.9),
            use_one_norm=True,
        )
    if name == "AP":
        return AdaptiveShadows(observables, weights)
    if name == "LCS":
        # ``RandomPaulis`` in tutorial.ipynb is local classical shadows.
        return Derandomization(observables, weights, epsilon, delta=1)
    raise ValueError(f"Unknown method: {name}")


def count_tutorial_bases(
    name: str,
    observables: np.ndarray,
    weights: np.ndarray,
    shots: int,
    seed: int,
) -> int:
    np.random.seed(seed)
    scheme = tutorial_scheme(name, observables, weights)
    bases: set[tuple[int, ...]] = set()
    for _ in range(shots):
        setting, _ = scheme.find_setting()
        bases.add(tuple(np.asarray(setting, dtype=int)))
    return len(bases)


def count_ogm_bases(path: Path, shots: int, seed: int) -> tuple[int, int, int]:
    data = np.atleast_2d(np.loadtxt(path, dtype=float))
    settings = data[:-1].T.astype(int)
    probabilities = data[-1]
    # Reproduce the allocation loop in Sample_main.m.  It scans settings in
    # descending probability order, stochastically rounds N*p, assigns at
    # least one shot to each visited setting, and stops upon reaching N shots.
    final_distinct = allocate_ogm_bases(settings, probabilities, shots, seed)

    floor_counts = (shots * probabilities).astype(int)
    converter_positive = int(np.count_nonzero(floor_counts > 0))
    converter_shots = int(np.sum(floor_counts))
    return final_distinct, converter_positive, converter_shots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=2038)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--recompute-ogm",
        action="store_true",
        help="optimize missing molecular OGM distributions in memory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("SG", "Derand", "AP", "LCS", "OGM"),
        default=["SG", "Derand", "AP", "LCS", "OGM"],
    )
    args = parser.parse_args()

    print("dataset\tmethod\tdistinct_bases\tshots\tseed_or_rule")
    for dataset_name in args.datasets:
        files = DATASETS[dataset_name]
        observables, weights = load_hamiltonian(
            files["hamiltonian"], files.get("weight_overrides")
        )
        for method in (name for name in ("SG", "Derand", "AP", "LCS") if name in args.methods):
            count = count_tutorial_bases(
                method,
                observables,
                weights,
                args.shots,
                args.seed,
            )
            rule = "deterministic" if method in {"SG", "Derand"} else str(args.seed)
            print(f"{dataset_name}\t{method}\t{count}\t{args.shots}\t{rule}")

        if "OGM" not in args.methods:
            continue
        ogm_file = files["ogm"]
        if ogm_file is None and args.recompute_ogm and files.get("ogm_recompute"):
            count, converter_positive, converter_shots, objective, success = (
                count_recomputed_ogm_bases(
                    observables, weights, args.shots, args.seed
                )
            )
            print(
                f"{dataset_name}\tOGM\t{count}\t{args.shots}\t"
                f"recomputed seed={args.seed}; converter="
                f"{converter_positive} bases/{converter_shots} shots; "
                f"objective={objective:.12g}; success={success}"
            )
        elif ogm_file is None:
            print(f"{dataset_name}\tOGM\tNA\t{args.shots}\tno optimized distribution")
        else:
            count, converter_positive, converter_shots = count_ogm_bases(
                ogm_file,
                args.shots,
                args.seed,
            )
            print(
                f"{dataset_name}\tOGM\t{count}\t{args.shots}\t"
                f"seed={args.seed}; converter="
                f"{converter_positive} bases/{converter_shots} shots"
            )


if __name__ == "__main__":
    main()
