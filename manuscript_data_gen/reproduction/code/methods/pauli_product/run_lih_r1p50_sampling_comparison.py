#!/usr/bin/env python3
"""Reproduce OGM/ShadowGrouping-family sampling on the LiH GPD input.

The legacy repository uses MATLAB for OGM and a Python-3.9/qibo stack for the
other methods.  This driver keeps the original measurement-allocation rules,
but evaluates repeated quantum measurement outcomes in batches from the exact
joint distribution of the commuting Pauli terms hit by each setting.  The
batched evaluator is statistically identical to drawing shots one at a time.

The compared methods are the five methods used by the repository's N=2038
audit: OGM, ShadowGrouping (SG), Derandomization, AdaptivePaulis (AP), and
RandomPaulis/local classical shadows (LCS).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from release_paths import DATA_ROOT
PROJECT_ROOT = DATA_ROOT
HUANG_DERAND_DIR = HERE
HUANG_DERAND_PATH = HERE / "derand_huang2021.py"
sys.path.insert(0, str(HERE))
from derand_huang2021 import appendix_c_schedule

LIH_DIR = DATA_ROOT / "LiH" / "inputs"
HAMILTONIAN_PATH = LIH_DIR / "hamiltonian_pauli_blocked_spin.txt"
STATE_PATH = LIH_DIR / "ground_state_mps_blocked_spin.npz"
METADATA_PATH = LIH_DIR / "metadata.json"

DEFAULT_BUDGETS = (12, 45, 160, 572, 2038, 7259, 25848, 92041)
DEFAULT_METHODS = ("OGM", "SG", "Derand", "AP", "LCS")
PAULI_TO_INT = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pauli_hamiltonian(path: Path):
    rows = np.loadtxt(path, dtype=object, comments="#")
    weights = rows[:, 0].astype(float)
    observables = np.asarray(
        [[PAULI_TO_INT[c] for c in label] for label in rows[:, 1]], dtype=np.int8
    )
    identity = np.all(observables == 0, axis=1)
    if np.count_nonzero(identity) != 1:
        raise ValueError("Expected exactly one identity term")
    offset = float(weights[identity][0])
    return observables[~identity], weights[~identity], offset


def load_dense_pauli_hamiltonian(
    path: Path,
    dense_index: int,
    threshold: float = 1.0e-10,
):
    """Load one dense Hamiltonian slice and decompose it in the Pauli basis."""
    archive = np.load(path, mmap_mode="r")
    dense = np.asarray(archive[dense_index], dtype=np.complex128)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError(f"Expected a square dense Hamiltonian, got {dense.shape}")
    num_qubits = int(round(math.log2(dense.shape[0])))
    if dense.shape != (1 << num_qubits, 1 << num_qubits):
        raise ValueError(f"Dense dimension {dense.shape[0]} is not a power of two")
    dense = 0.5 * (dense + dense.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    state /= np.linalg.norm(state)
    exact_energy = float(eigenvalues[0])

    local_paulis = np.asarray(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=np.complex128,
    )
    # c_P = Tr(P H) / 2^n.  With H[row, column], the local contraction
    # therefore uses P[column, row] / 2 at every site.
    local_transform = np.transpose(local_paulis, (0, 2, 1)) / 2.0
    dense_tensor = dense.reshape((2,) * (2 * num_qubits))
    dense_subscripts = list(range(2 * num_qubits))
    einsum_arguments = [dense_tensor, dense_subscripts]
    output_subscripts = []
    for site in range(num_qubits):
        pauli_axis = 2 * num_qubits + site
        einsum_arguments.extend(
            [local_transform, [pauli_axis, site, num_qubits + site]]
        )
        output_subscripts.append(pauli_axis)
    coefficients = np.einsum(
        *einsum_arguments,
        output_subscripts,
        optimize="greedy",
    )
    maximum_imaginary = float(np.max(np.abs(coefficients.imag)))
    if maximum_imaginary > 1.0e-9:
        raise RuntimeError(
            f"Pauli coefficients are not real: max imaginary={maximum_imaginary}"
        )
    coefficients = coefficients.real

    coefficient_norm_hartree2 = float(
        (1 << num_qubits) * np.sum(coefficients**2)
    )
    dense_norm_hartree2 = float(np.linalg.norm(dense, "fro") ** 2)
    norm_audit_error = coefficient_norm_hartree2 - dense_norm_hartree2
    if abs(norm_audit_error) > 1.0e-8:
        raise RuntimeError(
            "Dense-to-Pauli Frobenius audit failed: "
            f"difference={norm_audit_error:.3e}"
        )

    active_indices = np.argwhere(np.abs(coefficients) > threshold)
    observables = np.asarray(active_indices, dtype=np.int8)
    weights = np.asarray(
        [coefficients[tuple(index)] for index in active_indices], dtype=float
    )
    identity = np.all(observables == 0, axis=1)
    if np.count_nonzero(identity) != 1:
        raise ValueError("Expected exactly one identity term after decomposition")
    offset = float(weights[identity][0])
    audit = {
        "dense_slice_index": int(dense_index),
        "number_qubits": num_qubits,
        "pauli_threshold": threshold,
        "retained_terms_including_identity": int(len(observables)),
        "maximum_coefficient_imaginary_part": maximum_imaginary,
        "dense_frobenius_norm_squared": dense_norm_hartree2,
        "pauli_frobenius_norm_squared": coefficient_norm_hartree2,
        "frobenius_norm_squared_difference": norm_audit_error,
    }
    return (
        observables[~identity],
        weights[~identity],
        offset,
        state,
        exact_energy,
        audit,
    )


def load_mps_state(path: Path) -> np.ndarray:
    with np.load(path) as data:
        tensors = [np.asarray(data[key]) for key in sorted(data.files)]
    contracted = tensors[0]
    for tensor in tensors[1:]:
        contracted = np.tensordot(contracted, tensor, axes=([-1], [0]))
    state = np.squeeze(contracted, axis=(0, -1)).reshape(-1).astype(complex)
    state /= np.linalg.norm(state)
    return state


def pauli_expectation(state: np.ndarray, observable: np.ndarray) -> float:
    """Return <state|P|state>; observable site 0 is the most-significant bit."""
    n = len(observable)
    indices = np.arange(len(state), dtype=np.uint64)
    flip = 0
    phase_mask = 0
    num_y = 0
    for site, pauli in enumerate(observable):
        bit = 1 << (n - 1 - site)
        if pauli in (1, 2):
            flip |= bit
        if pauli in (2, 3):
            phase_mask |= bit
        if pauli == 2:
            num_y += 1
    parity = np.zeros(len(state), dtype=np.uint8)
    masked = indices & np.uint64(phase_mask)
    while np.any(masked):
        parity ^= (masked & 1).astype(np.uint8)
        masked >>= np.uint64(1)
    phase = (1j**num_y) * (1 - 2 * parity.astype(np.int8))
    value = np.vdot(state[indices ^ np.uint64(flip)], phase * state)
    return float(np.real_if_close(value, tol=1000).real)


def audit_energy(observables, weights, offset, state):
    expectations = np.asarray(
        [pauli_expectation(state, observable) for observable in observables]
    )
    return float(offset + weights @ expectations), expectations


def qwc_compatible(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all((left == right) | (left == 0) | (right == 0)))


def greedy_ogm_candidates(observables, weights):
    order = np.argsort(-np.abs(weights), kind="stable")
    obs = observables[order]
    coeff = weights[order]
    added = np.zeros(len(obs), dtype=bool)
    candidates = []
    initial_weights = []
    while not np.all(added):
        start = int(np.flatnonzero(~added)[0])
        basis = obs[start].copy()
        added[start] = True
        initial_weight = abs(coeff[start])
        for index in range(start + 1, len(obs)):
            if qwc_compatible(basis, obs[index]):
                basis = np.where(basis == 0, obs[index], basis)
                if not added[index]:
                    added[index] = True
                    initial_weight += abs(coeff[index])
        for index in range(start):
            if qwc_compatible(basis, obs[index]):
                basis = np.where(basis == 0, obs[index], basis)
        candidates.append(basis)
        initial_weights.append(initial_weight)
    return np.asarray(candidates, dtype=np.int8), np.asarray(initial_weights)


def optimize_ogm(observables, weights):
    settings, initial_weights = greedy_ogm_candidates(observables, weights)
    coverage = np.asarray(
        [
            [qwc_compatible(observable, setting) for setting in settings]
            for observable in observables
        ],
        dtype=float,
    )
    squared_weights = weights**2

    def objective(probabilities):
        beta = coverage @ probabilities
        if np.any(beta <= 0):
            return 1e100
        return float(np.sum(squared_weights / beta))

    initial = initial_weights / initial_weights.sum()
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(settings),
        constraints={"type": "eq", "fun": lambda p: np.sum(p) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
    )
    probabilities = np.clip(result.x, 0.0, None)
    probabilities /= probabilities.sum()
    return settings, probabilities, {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": objective(probabilities),
        "candidate_count": int(len(settings)),
    }


def legacy_ogm_schedule(settings, probabilities, shots, seed):
    """Translate lines 147--177 of gen_errors_all_algs/Sample_main.m."""
    order = np.argsort(-probabilities, kind="stable")
    rng = np.random.RandomState(seed)
    schedule = []
    for index in order:
        amount = shots * probabilities[index]
        draw = rng.rand()  # MATLAB draws even when amount <= 1.
        if amount > 1:
            number = int(math.floor(amount))
            if draw < np.mod(amount, number):
                number += 1
        else:
            number = 1
        take = min(number, shots - len(schedule))
        schedule.extend([settings[index]] * take)
        if len(schedule) >= shots:
            break
    if len(schedule) != shots:
        raise RuntimeError(f"OGM allocation produced {len(schedule)} of {shots} shots")
    out = np.asarray(schedule, dtype=np.int8)
    out[out == 0] = 3
    return out


def shadow_grouping_schedule(observables, weights, shots):
    """Vectorized equivalent of Shadow_Grouping + Bernstein_bound."""
    m, n = observables.shape
    hits = np.zeros(m, dtype=np.int64)
    alpha = np.max(np.abs(weights)) / np.min(np.abs(weights)) + np.min(
        np.abs(weights)
    )
    schedule = np.empty((shots, n), dtype=np.int8)
    for shot in range(shots):
        score = alpha * np.abs(weights)
        nonzero = hits != 0
        roots = np.sqrt(hits[nonzero])
        roots_plus_one = np.sqrt(hits[nonzero] + 1)
        # Keep the retained Shadow_Grouping/Bernstein_bound arithmetic exactly,
        # rather than using its algebraically simplified form.  The two forms
        # differ by roundoff for symmetry-related coefficients (notably F2),
        # which can change np.argsort tie order and thus the greedy QWC basis.
        score[nonzero] /= (
            alpha * roots * roots_plus_one / (roots_plus_one - roots)
        )
        order = np.argsort(score)
        setting = np.zeros(n, dtype=np.int8)
        for index in order[::-1]:
            observable = observables[index]
            if qwc_compatible(observable, setting):
                setting = np.where(setting == 0, observable, setting)
                if np.min(setting) > 0:
                    break
        setting[setting == 0] = 3
        schedule[shot] = setting
        hits += np.all(
            (observables == 0) | (observables == setting), axis=1
        )
    return schedule


def derandomized_schedule(
    observables,
    weights,
    shots,
    epsilon=math.sqrt(0.9),
    *,
    initial_hits=None,
    return_audit=False,
):
    """Paper-C8/C11 Derandomization with stable affected-term scoring."""

    return appendix_c_schedule(
        observables,
        weights,
        shots,
        initial_hits=initial_hits,
        eta=float(epsilon) ** 2,
        return_audit=return_audit,
    )


def adaptive_paulis_schedule(observables, weights, shots, seed):
    rng = np.random.RandomState(seed)
    squared_weights = weights**2
    m, n = observables.shape
    schedule = np.empty((shots, n), dtype=np.int8)
    for shot in range(shots):
        permutation = rng.permutation(n)
        compatible = np.ones(m, dtype=bool)
        setting = np.empty(n, dtype=np.int8)
        for site in permutation:
            constants = np.bincount(
                observables[compatible, site],
                weights=squared_weights[compatible],
                minlength=4,
            )[1:4]
            probabilities = np.sqrt(constants)
            if probabilities.sum() == 0:
                probabilities[:] = 1 / 3
            else:
                probabilities /= probabilities.sum()
            action = int(rng.choice([1, 2, 3], p=probabilities))
            setting[site] = action
            compatible &= (observables[:, site] == 0) | (
                observables[:, site] == action
            )
        schedule[shot] = setting
    return schedule


def local_classical_shadow_schedule(num_qubits, shots, seed):
    """Match Derandomization(delta=1), including its RNG consumption order."""
    rng = np.random.RandomState(seed)
    schedule = np.empty((shots, num_qubits), dtype=np.int8)
    for shot in range(shots):
        for site in range(num_qubits):
            rng.rand()
            schedule[shot, site] = int(rng.choice(3)) + 1
    return schedule


def distinct_rows(array):
    return int(len(np.unique(array, axis=0)))


def packed_hit_patterns(schedule, observables, batch_size=256):
    """Count identical sets of Hamiltonian terms hit by schedule settings."""
    counts = Counter()
    for start in range(0, len(schedule), batch_size):
        batch = schedule[start : start + batch_size]
        hit = np.all(
            (observables[None, :, :] == 0)
            | (observables[None, :, :] == batch[:, None, :]),
            axis=2,
        )
        packed = np.packbits(hit, axis=1, bitorder="little")
        unique, repetitions = np.unique(packed, axis=0, return_counts=True)
        for row, number in zip(unique, repetitions):
            counts[row.tobytes()] += int(number)
    return counts


def unpack_pattern(key: bytes, num_observables: int):
    packed = np.frombuffer(key, dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little")[:num_observables].astype(bool)


def fwht(values):
    out = np.asarray(values, dtype=float).copy()
    step = 1
    while step < len(out):
        for start in range(0, len(out), 2 * step):
            left = out[start : start + step].copy()
            right = out[start + step : start + 2 * step].copy()
            out[start : start + step] = left + right
            out[start + step : start + 2 * step] = left - right
        step *= 2
    return out


class JointPatternCache:
    def __init__(self, observables, state):
        self.observables = observables
        self.state = state
        self.expectation_cache = {tuple([0] * observables.shape[1]): 1.0}
        self.pattern_cache = {}

    def expectation(self, observable):
        key = tuple(int(value) for value in observable)
        if key not in self.expectation_cache:
            self.expectation_cache[key] = pauli_expectation(self.state, observable)
        return self.expectation_cache[key]

    def get(self, packed_key):
        if packed_key in self.pattern_cache:
            return self.pattern_cache[packed_key]
        selected = unpack_pattern(packed_key, len(self.observables))
        indices = np.flatnonzero(selected)
        if len(indices) == 0:
            value = (indices, np.ones(1), np.ones((1, 0), dtype=np.int8))
            self.pattern_cache[packed_key] = value
            return value

        selected_obs = self.observables[indices]
        axes = np.max(selected_obs, axis=0)
        masks = []
        for observable in selected_obs:
            mask = 0
            for site in np.flatnonzero(observable):
                mask |= 1 << int(site)
            masks.append(mask)

        span = {0: 0}
        generators = []
        for mask in masks:
            if mask in span:
                continue
            generator_index = len(generators)
            additions = {
                old_mask ^ mask: subset | (1 << generator_index)
                for old_mask, subset in list(span.items())
            }
            span.update(additions)
            generators.append(mask)
        rank = len(generators)
        xor_by_subset = np.empty(1 << rank, dtype=np.int64)
        for xor_mask, subset in span.items():
            xor_by_subset[subset] = xor_mask

        moments = np.empty(1 << rank)
        for subset, xor_mask in enumerate(xor_by_subset):
            observable = np.zeros(self.observables.shape[1], dtype=np.int8)
            for site in range(self.observables.shape[1]):
                if int(xor_mask) & (1 << site):
                    observable[site] = axes[site]
            moments[subset] = self.expectation(observable)
        probabilities = fwht(moments) / (1 << rank)
        probabilities[np.abs(probabilities) < 1e-14] = 0
        if probabilities.min() < -1e-9:
            raise RuntimeError(
                f"Invalid joint Pauli probability: minimum {probabilities.min()}"
            )
        probabilities = np.clip(probabilities, 0, None)
        probabilities /= probabilities.sum()

        categories = np.arange(1 << rank, dtype=np.int64)
        parity = np.asarray([int(value).bit_count() & 1 for value in categories])
        signs = np.empty((1 << rank, len(indices)), dtype=np.int8)
        for column, mask in enumerate(masks):
            coordinate = span[mask]
            signs[:, column] = 1 - 2 * parity[categories & coordinate]
        value = (indices, probabilities, signs)
        self.pattern_cache[packed_key] = value
        return value


def sample_fixed_schedule(
    schedule,
    observables,
    weights,
    offset,
    repeats,
    seed,
    cache,
):
    pattern_counts = packed_hit_patterns(schedule, observables)
    num_hits = np.zeros(len(observables), dtype=np.int64)
    unpacked = {}
    for key, count in pattern_counts.items():
        selected = unpack_pattern(key, len(observables))
        unpacked[key] = selected
        num_hits[selected] += count

    rng = np.random.default_rng(seed)
    energies = np.full(repeats, offset, dtype=float)
    for key, count in pattern_counts.items():
        if not np.any(unpacked[key]):
            continue
        indices, probabilities, signs = cache.get(key)
        coefficients = weights[indices] / num_hits[indices]
        category_energy = signs @ coefficients
        draws = rng.multinomial(count, probabilities, size=repeats)
        energies += draws @ category_energy
    return energies, num_hits, len(pattern_counts)


def save_schedule(path, schedule):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, settings=schedule)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7259)
    parser.add_argument(
        "--dense-hamiltonian",
        type=Path,
        default=None,
        help="Optional dense Hamiltonian archive; enables exact diagonalization.",
    )
    parser.add_argument(
        "--dense-index",
        type=int,
        default=10,
        help="Slice index used with --dense-hamiltonian.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "sampling_comparison_lih_r1p50",
    )
    args = parser.parse_args()
    budgets = sorted(set(args.budgets))
    max_shots = max(budgets)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "schedules").mkdir(exist_ok=True)

    started = time.time()
    pauli_decomposition_audit = None
    if args.dense_hamiltonian is not None:
        dense_path = args.dense_hamiltonian.resolve()
        print(
            f"Loading dense LiH Hamiltonian {dense_path}, slice {args.dense_index}..."
        )
        (
            observables,
            weights,
            offset,
            state,
            recorded_energy,
            pauli_decomposition_audit,
        ) = load_dense_pauli_hamiltonian(dense_path, args.dense_index)
        input_paths = (dense_path,)
        source_description = f"dense Hamiltonian slice {args.dense_index}"
        basis_description = "archived eight-qubit LiH Hamiltonian"
        qubit_order = "archived dense basis; site 0 is the most-significant bit"
    else:
        print("Loading exact LiH R=1.50 A Hamiltonian and MPS...")
        observables, weights, offset = load_pauli_hamiltonian(HAMILTONIAN_PATH)
        state = load_mps_state(STATE_PATH)
        with METADATA_PATH.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        recorded_energy = float(metadata["fci_ground_energy_hartree"])
        input_paths = (HAMILTONIAN_PATH, STATE_PATH, METADATA_PATH)
        source_description = "blocked-spin Pauli Hamiltonian and exact MPS"
        basis_description = "sto-3g"
        qubit_order = "blocked spin: alpha orbitals, then beta orbitals"
    energy, term_expectations = audit_energy(observables, weights, offset, state)
    energy_difference = energy - recorded_energy
    print(
        f"Energy audit: calculated={energy:.15f}, recorded={recorded_energy:.15f}, "
        f"difference={energy_difference:.3e}"
    )
    if abs(energy_difference) > 1e-10:
        raise RuntimeError("Hamiltonian/state ordering audit failed")

    schedules = {}
    ogm_info = None
    ogm_settings = None
    ogm_probabilities = None
    for method in args.methods:
        method_start = time.time()
        print(f"Generating {method} measurement schedule(s)...", flush=True)
        if method == "OGM":
            ogm_settings, ogm_probabilities, ogm_info = optimize_ogm(
                observables, weights
            )
            np.savetxt(
                output / "ogm_distribution.txt",
                np.vstack([ogm_settings.T, ogm_probabilities]),
                fmt="%.16g",
            )
            schedules[method] = {
                budget: legacy_ogm_schedule(
                    ogm_settings, ogm_probabilities, budget, args.seed
                )
                for budget in budgets
            }
            for budget, schedule in schedules[method].items():
                save_schedule(
                    output / "schedules" / f"{method}_{budget}.npz", schedule
                )
        elif method == "SG":
            schedule = shadow_grouping_schedule(observables, weights, max_shots)
            schedules[method] = schedule
            save_schedule(output / "schedules" / f"{method}_{max_shots}.npz", schedule)
        elif method == "Derand":
            schedule = derandomized_schedule(observables, weights, max_shots)
            schedules[method] = schedule
            save_schedule(output / "schedules" / f"{method}_{max_shots}.npz", schedule)
        elif method == "AP":
            schedule = adaptive_paulis_schedule(
                observables, weights, max_shots, args.seed
            )
            schedules[method] = schedule
            save_schedule(output / "schedules" / f"{method}_{max_shots}.npz", schedule)
        elif method == "LCS":
            schedule = local_classical_shadow_schedule(
                observables.shape[1], max_shots, args.seed
            )
            schedules[method] = schedule
            save_schedule(output / "schedules" / f"{method}_{max_shots}.npz", schedule)
        print(
            f"Generated {method} in {time.time() - method_start:.1f} s", flush=True
        )

    summary_rows = []
    repeat_rows = []
    cache = JointPatternCache(observables, state)
    method_seed_offset = {name: i * 100_000 for i, name in enumerate(DEFAULT_METHODS)}
    for method in args.methods:
        for budget in budgets:
            point_start = time.time()
            schedule = (
                schedules[method][budget]
                if method == "OGM"
                else schedules[method][:budget]
            )
            energies, num_hits, pattern_count = sample_fixed_schedule(
                schedule,
                observables,
                weights,
                offset,
                args.repeats,
                args.seed + method_seed_offset[method] + budget,
                cache,
            )
            errors = energies - recorded_energy
            rmse = float(np.sqrt(np.mean(errors**2)))
            row = {
                "method": method,
                "shots": budget,
                "repeat_count": args.repeats,
                "rmse_hartree": rmse,
                "mean_absolute_error_hartree": float(np.mean(np.abs(errors))),
                "mean_energy_hartree": float(np.mean(energies)),
                "energy_bias_hartree": float(np.mean(errors)),
                "energy_std_hartree": float(np.std(energies, ddof=0)),
                "distinct_measurement_bases": distinct_rows(schedule),
                "distinct_hit_patterns": pattern_count,
                "unhit_pauli_terms": int(np.count_nonzero(num_hits == 0)),
                "min_hits_nonzero": int(np.min(num_hits[num_hits > 0]))
                if np.any(num_hits > 0)
                else 0,
                "wall_seconds": float(time.time() - point_start),
            }
            summary_rows.append(row)
            for repeat, (estimate, error) in enumerate(zip(energies, errors)):
                repeat_rows.append(
                    {
                        "method": method,
                        "shots": budget,
                        "repeat": repeat,
                        "estimated_energy_hartree": float(estimate),
                        "error_hartree": float(error),
                    }
                )
            print(
                f"{method:6s} shots={budget:6d} RMSE={rmse:.8g} "
                f"bases={row['distinct_measurement_bases']} "
                f"unhit={row['unhit_pauli_terms']}",
                flush=True,
            )
            write_csv(output / "sampling_summary.csv", list(row), summary_rows)
            write_csv(
                output / "sampling_repeats.csv", list(repeat_rows[0]), repeat_rows
            )

    manifest = {
        "created_unix_time": time.time(),
        "wall_seconds": time.time() - started,
        "molecule": "LiH",
        "bond_length_angstrom": 1.5,
        "basis": basis_description,
        "qubit_order": qubit_order,
        "source_description": source_description,
        "number_qubits": int(observables.shape[1]),
        "pauli_terms_including_identity": int(len(observables) + 1),
        "methods": list(args.methods),
        "shot_budgets": budgets,
        "repeat_count": args.repeats,
        "seed": args.seed,
        "reference_energy_hartree": recorded_energy,
        "recomputed_pauli_expectation_hartree": energy,
        "energy_audit_difference_hartree": energy_difference,
        "sampling": (
            "Exact multinomial draws from each hit-pattern's joint commuting-Pauli "
            "distribution; equivalent to per-shot projective sampling. Fixed schedule "
            f"and {args.repeats} independent outcome repetitions per method/budget."
        ),
        "algorithm_parameters": {
            "SG": "Bernstein_bound with alpha=max(|w|)/min(|w|)+min(|w|), epsilon=0.1",
            "Derand": (
                "Huang-2021 Appendix C C8/C11 fixed-budget stream; eta=0.9; "
                "stable affected-term delta"
            ),
            "AP": "AdaptiveShadows conditional beta rule",
            "LCS": "Derandomization delta=1 (uniform local Pauli bases)",
            "OGM": "main_CutOGM greedy candidates + diagonal-variance SLSQP; Sample_main legacy allocation",
        },
        "ogm_optimizer": ogm_info,
        "pauli_decomposition_audit": pauli_decomposition_audit,
        "inputs": {str(path): sha256(path) for path in input_paths},
        "derand_implementation": {
            "algorithm_id": "huang2021-appendix-c-fixed-budget-stable-delta-v1",
            "path": str(HUANG_DERAND_PATH),
            "sha256": sha256(HUANG_DERAND_PATH),
        },
        "output_hashes": {
            "sampling_summary.csv": sha256(output / "sampling_summary.csv"),
            "sampling_repeats.csv": sha256(output / "sampling_repeats.csv"),
        },
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Completed in {manifest['wall_seconds']:.1f} s. Results: {output}")


if __name__ == "__main__":
    main()
