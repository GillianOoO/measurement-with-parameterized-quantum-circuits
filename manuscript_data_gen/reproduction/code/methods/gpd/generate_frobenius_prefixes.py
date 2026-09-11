#!/usr/bin/env python3
"""Periodic-iSWAP GPD source-prefix generator.

The nonlinear source fit follows the original GPD semantics:

    Lambda_k = diag(U_k R_{k-1} U_k^dagger),
    R_k = R_{k-1} - U_k^dagger Lambda_k U_k,

where U_k is the archived periodic fixed-iSWAP plus local Rx-Ry-Rx ansatz and
the unrestricted full computational diagonal is retained.  For every saved
prefix and fixed shot budget T, this runner records the two uncalibrated
features

    x_a(K)   = ||Pi (H-Hhat_K) Pi||_F,
    x_s(K,T) = sqrt(sum_i lambda_i^2 / T_i),

using full-domain diagonal half-ranges and minimum-one/largest-remainder
integer shot allocation.  The stabilizer-calibrated coefficients and final
prefix selection are applied by the calibrated selector.  State-resolved
moments recorded here are reporting quantities and do not enter the source
features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pauli_product"))
from release_paths import BUNDLE_ROOT, DATA_ROOT, OUTPUT_ROOT

WORKSPACE = BUNDLE_ROOT
CANONICAL_GPD = HERE / "recompute_h4_gpd.py"
H4_INPUT_ROOT = DATA_ROOT / "H4" / "inputs" / "bond_scan"
H4_BASELINE = DATA_ROOT / "H4" / "processed" / "h4_fully_corrective_v10_validation.csv"
RANDOM_ROOT = DATA_ROOT / "random_sparse_dense" / "inputs"
RANDOM_STATE = RANDOM_ROOT / "common_input_state.npy"
RANDOM_PAULI_ROWS = DATA_ROOT / "random_sparse_dense" / "provenance" / "instance_error_rows.csv"
RANDOM_BASELINE_MANIFEST = RANDOM_PAULI_ROWS.parent / "manifest.json"
RANDOM_V10_ROWS = DATA_ROOT / "random_sparse_dense" / "provenance" / "review_instance_rows.csv"

PROFILE = "periodic-iswap-full-diagonal-sequential-gpd-frobenius-source-v2"
SOURCE_OUTPUT_ROOT = OUTPUT_ROOT / "gpd_source_prefixes"
H4_SHOTS = (2038,)
RANDOM_SHOTS = (12, 45, 160, 572, 2038, 7259, 25848)
PAULI_METHODS = ("OGM", "SG", "Derand", "LCS", "AP")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def matrix_sha256(values: np.ndarray) -> str:
    """Hash convention used by the retained H4 AGPD bond-scan baseline."""
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.dtype("<c16")).tobytes()
    ).hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(item) for item in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def canonical_fragment_key(base_seed: int, k: int) -> jax.Array:
    """Return the kth subkey from the archived one-start sequential key stream."""
    key = jax.random.PRNGKey(int(base_seed))
    fragment_key = key
    for _ in range(int(k)):
        key, fragment_key = jax.random.split(key)
    return fragment_key


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def load_canonical_gpd_module():
    spec = importlib.util.spec_from_file_location("canonical_periodic_iswap_gpd", CANONICAL_GPD)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical GPD implementation: {CANONICAL_GPD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


canonical = load_canonical_gpd_module()


@dataclass
class CaseData:
    benchmark: str
    slug: str
    label: str
    input_path: str
    state_path: str
    hamiltonian: np.ndarray
    state: np.ndarray
    stored_energy: float | None
    shots: tuple[int, ...]
    particle_number: int | None

    @property
    def n_qubits(self) -> int:
        return int(round(math.log2(self.hamiltonian.shape[0])))

    @property
    def target_expectation(self) -> float:
        return float(np.real(np.vdot(self.state, self.hamiltonian @ self.state)))

    @property
    def target_hash(self) -> str:
        return array_sha256(self.hamiltonian)

    @property
    def state_hash(self) -> str:
        return array_sha256(self.state)


@dataclass
class FragmentRecord:
    k: int
    seed_namespace: str
    starts: int
    winning_start: int
    winning_seed: int
    winning_prng_key: list[int]
    optimizer_steps: int
    final_offdiagonal_frobenius_squared: float
    final_losses_all_starts: list[float]
    residual_frobenius: float
    residual_projected_frobenius: float
    diagonal_min: float
    diagonal_max: float
    diagonal_midpoint: float
    full_half_range: float
    allocation_range_branch: str
    allocation_midpoint: float
    centered_half_range: float
    state_mean: float
    state_single_shot_variance: float
    parameter_file: str
    diagonal_file: str
    probability_file: str
    parameter_sha256: str
    diagonal_sha256: str
    probability_sha256: str
    elapsed_seconds: float


def pure_state(values: np.ndarray) -> np.ndarray:
    saved = np.asarray(values, dtype=np.complex128)
    if saved.ndim == 1:
        state = saved.copy()
    elif saved.ndim == 2 and saved.shape[0] == saved.shape[1]:
        eigenvalues, eigenvectors = np.linalg.eigh((saved + saved.conj().T) / 2.0)
        state = eigenvectors[:, int(np.argmax(eigenvalues))]
        projector_error = float(np.linalg.norm(saved - np.outer(state, state.conj())))
        if projector_error > 2.0e-7:
            raise ValueError(f"Saved density matrix is not a pure-state projector: {projector_error}")
    else:
        raise ValueError(f"Unsupported saved state shape: {saved.shape}")
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Invalid saved state norm")
    state /= norm
    pivot = int(np.argmax(np.abs(state)))
    state *= np.exp(-1j * np.angle(state[pivot]))
    return state


def load_case(
    *,
    benchmark: str,
    slug: str,
    label: str,
    input_path: Path,
    state_path: Path | None,
    shots: Sequence[int],
    particle_number: int | None,
) -> CaseData:
    if input_path.suffix.lower() == ".npz":
        archive = np.load(input_path)
        if "H" not in archive.files or "state" not in archive.files:
            raise ValueError(f"NPZ must contain H and state: {input_path}")
        hamiltonian = np.asarray(archive["H"], dtype=np.complex128)
        state = pure_state(archive["state"])
        stored_energy = float(np.asarray(archive["energy"])) if "energy" in archive.files else None
        state_source = input_path
    else:
        if state_path is None:
            raise ValueError("A separate --state path is required for .npy Hamiltonians")
        hamiltonian = np.asarray(np.load(input_path), dtype=np.complex128)
        state = pure_state(np.load(state_path))
        stored_energy = None
        state_source = state_path
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(f"Invalid Hamiltonian shape: {hamiltonian.shape}")
    dimension = hamiltonian.shape[0]
    n_qubits = int(round(math.log2(dimension)))
    if 2**n_qubits != dimension or state.shape != (dimension,):
        raise ValueError("Hamiltonian/state dimension is not a common qubit Hilbert space")
    anti = float(np.linalg.norm(hamiltonian - hamiltonian.conj().T))
    if anti > 2.0e-9:
        raise ValueError(f"Hamiltonian is not Hermitian: ||H-Hdag||F={anti}")
    hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2.0
    target = float(np.real(np.vdot(state, hamiltonian @ state)))
    if stored_energy is not None and abs(target - stored_energy) > 3.0e-8:
        raise ValueError(f"Stored energy mismatch: expectation={target}, stored={stored_energy}")
    return CaseData(
        benchmark=benchmark,
        slug=slug,
        label=label,
        input_path=str(input_path.resolve()),
        state_path=str(state_source.resolve()),
        hamiltonian=hamiltonian,
        state=state,
        stored_energy=stored_energy,
        shots=tuple(int(item) for item in shots),
        particle_number=particle_number,
    )


def sector_indices(n_qubits: int, particle_number: int | None) -> np.ndarray:
    if particle_number is None:
        return np.arange(2**n_qubits, dtype=int)
    if not 0 <= int(particle_number) <= int(n_qubits):
        raise ValueError(
            f"particle_number={particle_number} is outside [0,{n_qubits}]"
        )
    return np.asarray(
        [index for index in range(2**n_qubits) if index.bit_count() == particle_number],
        dtype=int,
    )


def projected_frobenius_feature(residual: np.ndarray, indices: np.ndarray) -> float:
    block = residual[np.ix_(indices, indices)]
    return float(np.linalg.norm(block, "fro"))


def full_domain_allocation_range(
    full_diagonal: np.ndarray,
) -> dict[str, float | str]:
    """Return the allocation center and half-range of the retained diagonal."""
    full_minimum = float(np.min(full_diagonal))
    full_maximum = float(np.max(full_diagonal))
    full_midpoint = 0.5 * (full_minimum + full_maximum)
    full_half_range = 0.5 * (full_maximum - full_minimum)
    if not np.isfinite(full_half_range) or full_half_range < 0.0:
        raise FloatingPointError("Invalid full-domain allocation half-range")
    return {
        "full_min": full_minimum,
        "full_max": full_maximum,
        "full_midpoint": full_midpoint,
        "full_half_range": full_half_range,
        "allocation_half_range": full_half_range,
        "allocation_midpoint": full_midpoint,
        "branch": "full-domain",
    }


def integer_range_allocation(
    ranges: Sequence[float], total_shots: int
) -> tuple[np.ndarray, float]:
    values = np.asarray(ranges, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Allocation ranges must be finite and nonnegative")
    # The revised supplement deliberately treats every finite positive range
    # as active.  Only a mathematically exact floating-point zero is constant.
    active = values > 0.0
    count = int(np.count_nonzero(active))
    if count > total_shots:
        raise ValueError(f"{count} active settings exceed T={total_shots}")
    allocation = np.zeros(len(values), dtype=int)
    if count == 0:
        return allocation, 0.0
    active_indices = np.flatnonzero(active)
    allocation[active_indices] = 1
    remaining = int(total_shots - count)
    if remaining:
        quotas = remaining * values[active] / float(np.sum(values[active]))
        floors = np.floor(quotas).astype(int)
        allocation[active_indices] += floors
        leftover = remaining - int(np.sum(floors))
        if leftover:
            fractions = quotas - floors
            order = np.lexsort((active_indices, -fractions))
            allocation[active_indices[order[:leftover]]] += 1
    if int(np.sum(allocation)) != int(total_shots):
        raise AssertionError("Integer allocation failed exact shot conservation")
    sampling_squared = float(np.sum(np.square(values[active]) / allocation[active]))
    return allocation, math.sqrt(max(sampling_squared, 0.0))


def prefix_rows(
    *,
    case: CaseData,
    k: int,
    residual_frobenius: float,
    approximation_feature: float,
    ranges: Sequence[float],
    centers: Sequence[float],
    variances: Sequence[float],
    approximate_expectation: float,
) -> list[dict[str, object]]:
    bias = float(approximate_expectation - case.target_expectation)
    rows: list[dict[str, object]] = []
    for total in case.shots:
        try:
            allocation, sampling_proxy = integer_range_allocation(ranges, int(total))
        except ValueError:
            continue
        variance = float(
            sum(
                float(value) / int(count)
                for value, count in zip(variances, allocation)
                if int(count) > 0
            )
        )
        mse = bias * bias + variance
        rows.append(
            {
                "benchmark": case.benchmark,
                "slug": case.slug,
                "k": int(k),
                "shots": int(total),
                "active_settings": int(np.count_nonzero(allocation)),
                "loss": math.hypot(approximation_feature, sampling_proxy),
                "approximation_feature_xa": approximation_feature,
                "sampling_feature_xs": sampling_proxy,
                "residual_frobenius": residual_frobenius,
                "approximate_expectation": approximate_expectation,
                "target_expectation": case.target_expectation,
                "bias": bias,
                "bias_squared": bias * bias,
                "variance": variance,
                "mse": mse,
                "analytic_rmse": math.sqrt(max(mse, 0.0)),
                "allocation_json": json.dumps(allocation.tolist()),
                "centered_half_ranges_json": json.dumps([float(item) for item in ranges]),
                "centering_offsets_json": json.dumps([float(item) for item in centers]),
            }
        )
    return rows


def replay_empirical_rmse(
    case: CaseData,
    output: Path,
    selected: dict[str, object],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, list[float]]:
    k = int(selected["k"])
    if k == 0:
        estimates = np.full(repeats, float(selected["approximate_expectation"]))
    else:
        allocation = np.asarray(json.loads(str(selected["allocation_json"])), dtype=int)
        centers = np.asarray(
            json.loads(str(selected["centering_offsets_json"])), dtype=float
        )
        if len(allocation) != k:
            raise ValueError("Selected allocation length does not match K")
        if len(centers) != k:
            raise ValueError("Selected centering-offset vector does not match K")
        rng = np.random.default_rng(seed)
        estimates = np.full(repeats, float(np.trace(case.hamiltonian).real / len(case.hamiltonian)))
        for index, count in enumerate(allocation, start=1):
            if int(count) <= 0:
                estimates += float(centers[index - 1])
                continue
            fragment_dir = output / "fragments" / f"k{index:03d}"
            diagonal = np.asarray(np.load(fragment_dir / "diagonal.npy"), dtype=float)
            probabilities = np.asarray(np.load(fragment_dir / "probabilities.npy"), dtype=float)
            probabilities = np.maximum(probabilities, 0.0)
            probabilities /= float(np.sum(probabilities))
            counts = rng.multinomial(int(count), probabilities, size=repeats)
            estimates += counts @ diagonal / int(count)
    errors = estimates - case.target_expectation
    return float(np.sqrt(np.mean(np.square(errors)))), [float(item) for item in estimates]


def select_rows(frontier_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    shots_values = sorted({int(row["shots"]) for row in frontier_rows})
    for total in shots_values:
        candidates = [row for row in frontier_rows if int(row["shots"]) == total]
        if not candidates:
            continue
        minimum = min(float(row["loss"]) for row in candidates)
        exact_ties = [row for row in candidates if float(row["loss"]) == minimum]
        selected.append(min(exact_ties, key=lambda row: (int(row["k"]), float(row["loss"]))).copy())
    return selected


def configuration_signature(case: CaseData, args: argparse.Namespace) -> str:
    payload = {
        "profile": PROFILE,
        "benchmark": case.benchmark,
        "slug": case.slug,
        "target_hash": case.target_hash,
        "state_hash": case.state_hash,
        "shots": list(case.shots),
        "particle_number": case.particle_number,
        "paper_depth": args.paper_depth,
        "internal_layers": args.paper_depth + 1,
        "starts": args.starts,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "base_seed": args.base_seed,
        "x64": bool(args.x64),
        "minimum_k": args.minimum_k,
        "stall_patience": args.stall_patience,
        "relative_improvement": args.relative_improvement,
        "empirical_repeats": args.empirical_repeats,
        "canonical_gpd_sha256": sha256(CANONICAL_GPD),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_case(args: argparse.Namespace) -> None:
    if int(args.starts) < 1 or int(args.steps) < 1:
        raise ValueError("--starts and --steps must be positive")
    if int(args.progress_every) < 1 or int(args.max_k) < 1:
        raise ValueError("--progress-every and --max-k must be positive")
    if int(args.minimum_k) < 0 or int(args.stall_patience) < 1:
        raise ValueError("--minimum-k must be nonnegative and --stall-patience positive")
    if int(args.empirical_repeats) < 0 or any(int(item) <= 0 for item in args.shots):
        raise ValueError("shot budgets must be positive and empirical repeats nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", bool(args.x64))
    shots = tuple(int(item) for item in args.shots)
    case = load_case(
        benchmark=args.benchmark,
        slug=args.slug,
        label=args.label or args.slug,
        input_path=args.input.resolve(),
        state_path=args.state.resolve() if args.state else None,
        shots=shots,
        particle_number=args.particle_number,
    )
    signature = configuration_signature(case, args)
    completed_path = output / "selected_results.csv"
    manifest_path = output / "manifest.json"
    if completed_path.exists() and manifest_path.exists() and not args.force:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("configuration_signature") == signature and saved.get("status") == "complete":
            saved_frontier = saved.get("frontier", {})
            saved_terminal = int(saved_frontier.get("terminal_k", 0))
            saved_right_censored = bool(saved_frontier.get("right_censored", False))
            if not saved_right_censored or int(args.max_k) <= saved_terminal:
                print(f"{case.slug}: complete cache matches; skipping", flush=True)
                return
            print(
                f"{case.slug}: extending right-censored frontier "
                f"K={saved_terminal} to guard K={args.max_k}",
                flush=True,
            )

    complex_dtype = jnp.complex128 if args.x64 else jnp.complex64
    real_dtype = jnp.float64 if args.x64 else jnp.float32
    internal_layers = int(args.paper_depth + 1)
    actions = canonical.CircuitActions(case.n_qubits, internal_layers)

    def loss_fn(parameters: jnp.ndarray, current_residual: jnp.ndarray) -> jnp.ndarray:
        rotated = actions.conjugate_forward(current_residual, parameters)
        diagonal = jnp.diag(rotated)
        # Match the archived GPD objective and its floating-point gradient.
        return jnp.real(jnp.vdot(rotated, rotated)) - jnp.real(
            jnp.vdot(diagonal, diagonal)
        )

    single_value_grad = jax.jit(jax.value_and_grad(loss_fn))
    single_loss = jax.jit(loss_fn)
    forward_operator = jax.jit(actions.conjugate_forward)
    inverse_operator = jax.jit(actions.conjugate_inverse)
    forward_state = jax.jit(actions.evolve_state_forward)

    dimension = len(case.state)
    exact_constant = float(np.trace(case.hamiltonian).real / dimension)
    centered_target = case.hamiltonian - exact_constant * np.eye(dimension, dtype=np.complex128)
    centered_target = (centered_target + centered_target.conj().T) / 2.0
    indices = sector_indices(case.n_qubits, case.particle_number)
    state_sector_leakage = 0.0
    if case.particle_number is not None:
        state_sector_weight = float(np.sum(np.square(np.abs(case.state[indices]))))
        state_sector_leakage = max(0.0, 1.0 - state_sector_weight)
        if state_sector_leakage > 2.0e-10:
            raise ValueError(
                f"{case.slug}: target-state weight outside particle-number sector is "
                f"{state_sector_leakage:.6g}"
            )
    # The first nonlinear fit sees the same full H as the archived generator.
    # Its trace component is peeled out of the first saved diagonal afterward,
    # leaving the exact K=0 scalar without perturbing the historical trajectory.
    residual_np = case.hamiltonian.copy()
    approximate_expectation = exact_constant
    fragments: list[dict[str, object]] = []
    ranges: list[float] = []
    centers: list[float] = []
    variances: list[float] = []
    frontier_rows: list[dict[str, object]] = prefix_rows(
        case=case,
        k=0,
        residual_frobenius=float(np.linalg.norm(centered_target, "fro")),
        approximation_feature=projected_frobenius_feature(centered_target, indices),
        ranges=ranges,
        centers=centers,
        variances=variances,
        approximate_expectation=approximate_expectation,
    )
    start_k = 1
    stalled = 0
    best_scores = {int(row["shots"]): float(row["loss"]) for row in frontier_rows}
    checkpoint_json = output / "checkpoint.json"
    checkpoint_residual = output / "residual_latest.npy"
    if args.resume and not args.force and checkpoint_json.exists() and checkpoint_residual.exists():
        saved = json.loads(checkpoint_json.read_text(encoding="utf-8"))
        if saved.get("configuration_signature") == signature:
            if saved.get("residual_sha256") != sha256(checkpoint_residual):
                raise RuntimeError(
                    f"{case.slug}: checkpoint JSON/residual hash mismatch; "
                    "rerun with --force to discard the incomplete checkpoint"
                )
            fragments = list(saved["fragments"])
            ranges = [float(item) for item in saved["ranges"]]
            centers = [float(item) for item in saved["centers"]]
            variances = [float(item) for item in saved["variances"]]
            frontier_rows = list(saved["frontier_rows"])
            approximate_expectation = float(saved["approximate_expectation"])
            residual_np = np.asarray(np.load(checkpoint_residual), dtype=np.complex128)
            best_scores = {int(key): float(value) for key, value in saved["best_scores"].items()}
            stalled = int(saved["stalled"])
            start_k = len(fragments) + 1
            print(f"{case.slug}: resuming at K={start_k}", flush=True)

    residual = jnp.asarray(residual_np, dtype=complex_dtype)
    state = jnp.asarray(case.state, dtype=complex_dtype)
    beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8
    run_started = time.perf_counter()
    stop_reason = ""
    for k in range(start_k, int(args.max_k) + 1):
        fragment_started = time.perf_counter()
        parameters = []
        seeds = []
        prng_keys: list[list[int]] = []
        primary_key = canonical_fragment_key(int(args.base_seed), k)
        for start in range(int(args.starts)):
            # start=0 exactly follows recompute_h4_gpd.py. Extra starts are
            # deterministic fold-ins and do not perturb that canonical stream.
            key = primary_key if start == 0 else jax.random.fold_in(primary_key, start)
            key_words = [int(item) for item in np.asarray(key, dtype=np.uint32)]
            seed = (key_words[0] << 32) | key_words[1]
            seeds.append(seed)
            prng_keys.append(key_words)
            parameters.append(
                jax.random.uniform(
                    key,
                    (internal_layers, case.n_qubits, 3),
                    minval=-0.1 * np.pi,
                    maxval=0.1 * np.pi,
                    dtype=real_dtype,
                )
            )
        optimized_parameters: list[jax.Array] = []
        final_loss_values: list[float] = []
        for start, current_parameters in enumerate(parameters):
            first_moment = jnp.zeros_like(current_parameters)
            second_moment = jnp.zeros_like(current_parameters)
            for step in range(1, int(args.steps) + 1):
                _, gradient = single_value_grad(current_parameters, residual)
                first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
                second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
                corrected_first = first_moment / (1.0 - beta1**step)
                corrected_second = second_moment / (1.0 - beta2**step)
                current_parameters = current_parameters - float(args.learning_rate) * corrected_first / (
                    jnp.sqrt(corrected_second) + epsilon
                )
                if step == 1 or step == int(args.steps) or step % int(args.progress_every) == 0:
                    value = float(single_loss(current_parameters, residual))
                    print(
                        f"{case.slug} K={k:03d} start={start:02d} step={step:04d} "
                        f"offdiag_F2={value:.10g}",
                        flush=True,
                    )
            optimized_parameters.append(current_parameters)
            final_rotated = forward_operator(residual, current_parameters)
            final_diagonal = jnp.real(jnp.diag(final_rotated))
            final_off_diagonal = final_rotated - jnp.diag(
                final_diagonal.astype(final_rotated.dtype)
            )
            final_loss_values.append(
                float(jnp.real(jnp.vdot(final_off_diagonal, final_off_diagonal)))
            )
        final_losses = np.asarray(final_loss_values, dtype=float)
        if not np.all(np.isfinite(final_losses)):
            raise FloatingPointError(f"{case.slug} K={k}: non-finite final losses")
        winner = int(np.argmin(final_losses))
        winner_parameters = optimized_parameters[winner]
        rotated = forward_operator(residual, winner_parameters)
        raw_diagonal = jnp.real(jnp.diag(rotated))
        off_diagonal = rotated - jnp.diag(raw_diagonal.astype(rotated.dtype))
        winner_final_loss = float(jnp.real(jnp.vdot(off_diagonal, off_diagonal)))
        if not np.isfinite(winner_final_loss) or winner_final_loss < 0.0:
            raise FloatingPointError(
                f"{case.slug} K={k}: invalid explicit off-diagonal loss "
                f"{winner_final_loss}"
            )
        final_losses[winner] = winner_final_loss
        diagonal_np = np.asarray(raw_diagonal, dtype=np.float64)
        if k == 1:
            diagonal_np = diagonal_np - exact_constant
        measurement_diagonal = jnp.asarray(diagonal_np, dtype=real_dtype)
        fragment = inverse_operator(
            jnp.diag(measurement_diagonal.astype(rotated.dtype)), winner_parameters
        )
        residual = inverse_operator(off_diagonal, winner_parameters)
        residual.block_until_ready()
        fragment.block_until_ready()
        residual_np = np.asarray(residual, dtype=np.complex128)
        residual_np = (residual_np + residual_np.conj().T) / 2.0
        residual = jnp.asarray(residual_np, dtype=complex_dtype)
        fragment_np = np.asarray(fragment, dtype=np.complex128)
        fragment_np = (fragment_np + fragment_np.conj().T) / 2.0
        rotated_state = np.asarray(forward_state(state, winner_parameters), dtype=np.complex128)
        probabilities = np.square(np.abs(rotated_state))
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= float(np.sum(probabilities))
        mean = float(probabilities @ diagonal_np)
        second = float(probabilities @ np.square(diagonal_np))
        single_variance = max(second - mean * mean, 0.0)
        range_audit = full_domain_allocation_range(diagonal_np)
        minimum = float(range_audit["full_min"])
        maximum = float(range_audit["full_max"])
        midpoint = float(range_audit["full_midpoint"])
        half_range = float(range_audit["allocation_half_range"])
        center = float(range_audit["allocation_midpoint"])
        approximate_expectation += mean
        ranges.append(half_range)
        centers.append(center)
        variances.append(single_variance)
        approximation_feature = projected_frobenius_feature(residual_np, indices)
        residual_frobenius = float(np.linalg.norm(residual_np, "fro"))

        fragment_dir = output / "fragments" / f"k{k:03d}"
        fragment_dir.mkdir(parents=True, exist_ok=True)
        parameter_path = fragment_dir / "parameters.npy"
        diagonal_path = fragment_dir / "diagonal.npy"
        probability_path = fragment_dir / "probabilities.npy"
        write_npy(parameter_path, np.asarray(winner_parameters, dtype=np.float64))
        write_npy(diagonal_path, diagonal_np)
        write_npy(probability_path, probabilities)
        record = FragmentRecord(
            k=k,
            seed_namespace=(
                f"archived sequential JAX stream|base_seed={args.base_seed}|"
                f"fragment={k}|start0 canonical; extra starts use fold_in(start)"
            ),
            starts=int(args.starts),
            winning_start=winner,
            winning_seed=int(seeds[winner]),
            winning_prng_key=prng_keys[winner],
            optimizer_steps=int(args.steps),
            final_offdiagonal_frobenius_squared=float(final_losses[winner]),
            final_losses_all_starts=[float(item) for item in final_losses],
            residual_frobenius=residual_frobenius,
            residual_projected_frobenius=approximation_feature,
            diagonal_min=minimum,
            diagonal_max=maximum,
            diagonal_midpoint=midpoint,
            full_half_range=float(range_audit["full_half_range"]),
            allocation_range_branch=str(range_audit["branch"]),
            allocation_midpoint=center,
            centered_half_range=half_range,
            state_mean=mean,
            state_single_shot_variance=single_variance,
            parameter_file=str(parameter_path.relative_to(output)),
            diagonal_file=str(diagonal_path.relative_to(output)),
            probability_file=str(probability_path.relative_to(output)),
            parameter_sha256=sha256(parameter_path),
            diagonal_sha256=sha256(diagonal_path),
            probability_sha256=sha256(probability_path),
            elapsed_seconds=time.perf_counter() - fragment_started,
        )
        fragments.append(asdict(record))
        new_rows = prefix_rows(
            case=case,
            k=k,
            residual_frobenius=residual_frobenius,
            approximation_feature=approximation_feature,
            ranges=ranges,
            centers=centers,
            variances=variances,
            approximate_expectation=approximate_expectation,
        )
        frontier_rows.extend(new_rows)
        materially_improved = False
        score_map = {int(row["shots"]): float(row["loss"]) for row in new_rows}
        for total, score in score_map.items():
            previous = best_scores.get(total, math.inf)
            if score < previous * (1.0 - float(args.relative_improvement)):
                best_scores[total] = score
                materially_improved = True
        stalled = 0 if materially_improved else stalled + 1
        write_npy(checkpoint_residual, residual_np)
        write_json(
            checkpoint_json,
            {
                "configuration_signature": signature,
                "fragments": fragments,
                "ranges": ranges,
                "centers": centers,
                "variances": variances,
                "frontier_rows": frontier_rows,
                "approximate_expectation": approximate_expectation,
                "best_scores": best_scores,
                "stalled": stalled,
                "residual_sha256": sha256(checkpoint_residual),
                "updated_utc": utc_now(),
            },
        )
        write_csv(output / "frontier_rows.csv", frontier_rows)
        print(
            f"{case.slug} K={k:03d} complete xa={approximation_feature:.8g} "
            f"F={residual_frobenius:.8g} stalled={stalled} "
            f"elapsed={record.elapsed_seconds:.1f}s",
            flush=True,
        )
        if k >= int(args.minimum_k) and stalled >= int(args.stall_patience):
            stop_reason = (
                f"no global-best balanced-loss improvement above "
                f"{float(args.relative_improvement):g} for {int(args.stall_patience)} consecutive K"
            )
            break
    terminal_k = len(fragments)
    right_censored = not bool(stop_reason)
    if right_censored:
        stop_reason = f"reached max_k={args.max_k} before the stall rule"

    selected_rows = select_rows(frontier_rows)
    for row in selected_rows:
        row["method"] = "Periodic-iSWAP GPD + balanced-K"
        row["paper_depth_L"] = int(args.paper_depth)
        row["internal_layers"] = internal_layers
        row["starts_per_fragment"] = int(args.starts)
        row["optimizer_steps_per_start"] = int(args.steps)
        row["frontier_terminal_k"] = terminal_k
        row["frontier_stop_reason"] = stop_reason
        row["frontier_right_censored"] = right_censored
        row["selected_at_terminal_k"] = int(row["k"]) == terminal_k
        row["selection_is_provisional"] = right_censored
        row["target_hash"] = case.target_hash
        row["state_hash"] = case.state_hash
        row["matrix_sha256_h4_convention"] = matrix_sha256(case.hamiltonian)
        row["input_sha256"] = sha256(Path(case.input_path))
        row["input_path"] = case.input_path
        row["empirical_repeats"] = int(args.empirical_repeats)
        if int(args.empirical_repeats) > 0:
            empirical_seed = stable_seed(PROFILE, "empirical", case.slug, int(row["shots"]), args.base_seed)
            empirical_rmse, estimates = replay_empirical_rmse(
                case,
                output,
                row,
                repeats=int(args.empirical_repeats),
                seed=empirical_seed,
            )
            row["empirical_rmse"] = empirical_rmse
            row["empirical_seed"] = empirical_seed
            write_csv(
                output / f"empirical_replicates_T{int(row['shots'])}.csv",
                [
                    {
                        "repeat": index,
                        "estimate": estimate,
                        "error": estimate - case.target_expectation,
                    }
                    for index, estimate in enumerate(estimates)
                ],
            )
        else:
            row["empirical_rmse"] = ""
            row["empirical_seed"] = ""
    write_csv(completed_path, selected_rows)
    manifest = {
        "profile": PROFILE,
        "status": "complete",
        "completed_utc": utc_now(),
        "configuration_signature": signature,
        "case": {
            "benchmark": case.benchmark,
            "slug": case.slug,
            "label": case.label,
            "input_path": case.input_path,
            "state_path": case.state_path,
            "target_hash": case.target_hash,
            "state_hash": case.state_hash,
            "n_qubits": case.n_qubits,
            "particle_number": case.particle_number,
            "state_sector_leakage": state_sector_leakage,
            "target_expectation": case.target_expectation,
            "stored_energy": case.stored_energy,
            "shots": list(case.shots),
        },
        "algorithm": {
            "source_fit": "sequential full-diagonal GPD",
            "ansatz": "periodic alternating fixed-iSWAP ring plus local Rx-Ry-Rx",
            "paper_depth_L": int(args.paper_depth),
            "internal_layers": internal_layers,
            "starts_per_fragment": int(args.starts),
            "steps_per_start": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "x64": bool(args.x64),
            "source_features": "projected residual Frobenius norm and integer full-domain half-range proxy",
            "shot_allocation": "minimum-one proportional centered-half-range largest remainder",
            "coefficient_update": "original greedy only; no fully-corrective refit",
            "K0": "exact trace(H)/D scalar only",
            "state_used_for_selection": False,
        },
        "frontier": {
            "terminal_k": terminal_k,
            "requested_max_k": int(args.max_k),
            "stop_reason": stop_reason,
            "right_censored": right_censored,
            "selected_k_by_shots": {
                str(row["shots"]): int(row["k"]) for row in selected_rows
            },
        },
        "runtime": {
            "elapsed_seconds_this_process": time.perf_counter() - run_started,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "jax": jax.__version__,
            "jax_devices": [str(item) for item in jax.devices()],
        },
        "source_files": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(CANONICAL_GPD): sha256(CANONICAL_GPD),
            case.input_path: sha256(Path(case.input_path)),
            case.state_path: sha256(Path(case.state_path)),
        },
        "fragments": fragments,
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "slug": case.slug,
                "terminal_k": terminal_k,
                "stop_reason": stop_reason,
                "right_censored": right_censored,
                "selected": [
                    {
                        "T": row["shots"],
                        "K": row["k"],
                        "loss": row["loss"],
                        "analytic_rmse": row["analytic_rmse"],
                        "empirical_rmse": row["empirical_rmse"],
                    }
                    for row in selected_rows
                ],
            },
            indent=2,
        ),
        flush=True,
    )


def suite_case_commands(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    commands: list[tuple[str, list[str], Path]] = []
    selected = set(args.benchmarks)
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "case",
        "--steps",
        str(args.steps),
        "--starts",
        str(args.starts),
        "--learning-rate",
        str(args.learning_rate),
        "--base-seed",
        str(args.base_seed),
        "--minimum-k",
        str(args.minimum_k),
        "--stall-patience",
        str(args.stall_patience),
        "--relative-improvement",
        str(args.relative_improvement),
        "--max-k",
        str(args.max_k),
        "--progress-every",
        str(args.progress_every),
        "--resume",
    ]
    if args.x64:
        common.append("--x64")
    if args.force:
        common.append("--force")
    if "h4" in selected:
        for index in range(21):
            bond = 0.4 + 0.2 * index
            tag = f"{bond:.1f}".replace(".", "p")
            slug = f"h4_R{tag}"
            output = SOURCE_OUTPUT_ROOT / "cases" / slug
            command = [
                *common,
                "--benchmark",
                "h4",
                "--slug",
                slug,
                "--label",
                f"H4 R={bond:.1f} Angstrom",
                "--input",
                str(H4_INPUT_ROOT / f"H4_R{bond:.1f}.npz"),
                "--output",
                str(output),
                "--paper-depth",
                str(args.h4_depth),
                "--particle-number",
                "4",
                "--shots",
                *[str(item) for item in H4_SHOTS],
                "--empirical-repeats",
                str(args.h4_repeats),
            ]
            commands.append((slug, command, output / "run.log"))
    for benchmark in ("sparse", "dense"):
        if benchmark not in selected:
            continue
        for instance in range(1, 6):
            slug = f"{benchmark}4_seed0_{instance}"
            output = SOURCE_OUTPUT_ROOT / "cases" / slug
            command = [
                *common,
                "--benchmark",
                benchmark,
                "--slug",
                slug,
                "--label",
                f"seed-0 {benchmark} random Hamiltonian {instance}",
                "--input",
                str(RANDOM_ROOT / "hamiltonians" / f"{benchmark}_hamiltonian_4_{instance}.npy"),
                "--state",
                str(RANDOM_STATE),
                "--output",
                str(output),
                "--paper-depth",
                str(args.random_depth),
                "--shots",
                *[str(item) for item in RANDOM_SHOTS],
                "--empirical-repeats",
                "0",
            ]
            commands.append((slug, command, output / "run.log"))
    return commands


def run_suite(args: argparse.Namespace) -> None:
    commands = suite_case_commands(args)
    pending = list(commands)
    running: dict[subprocess.Popen, tuple[str, object, Path]] = {}
    failed: list[tuple[str, int]] = []
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    while pending or running:
        while pending and len(running) < int(args.workers):
            slug, command, log_path = pending.pop(0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handle.write(f"\n===== suite launch {utc_now()} =====\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=WORKSPACE,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            running[process] = (slug, handle, log_path)
            print(f"launched {slug} pid={process.pid}", flush=True)
        time.sleep(2.0)
        for process, (slug, handle, log_path) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[process]
            print(f"finished {slug} exit={code} log={log_path}", flush=True)
            if code:
                failed.append((slug, int(code)))
    if failed:
        raise RuntimeError(f"Suite failures: {failed}")
    aggregate_results(args.benchmarks)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def configure_matplotlib() -> None:
    cache = WORKSPACE / ".matplotlib-cache" / "iswap_gpd_main_loss_v1"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def aggregate_results(benchmarks: Iterable[str]) -> None:
    selected_benchmarks = set(benchmarks)
    output_rows: list[dict[str, object]] = []
    for case_dir in sorted((SOURCE_OUTPUT_ROOT / "cases").glob("*")):
        path = case_dir / "selected_results.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            if row["benchmark"] in selected_benchmarks:
                output_rows.append(dict(row))
    if not output_rows:
        raise RuntimeError("No completed selected_results.csv files found")
    provisional = sorted(
        {
            row["slug"]
            for row in output_rows
            if str(row.get("selection_is_provisional", "")).strip().lower() == "true"
        }
    )
    if provisional:
        raise RuntimeError(
            "Refusing to publish a comparison figure from right-censored frontiers: "
            + ", ".join(provisional)
        )
    write_csv(SOURCE_OUTPUT_ROOT / "selected_results_all.csv", output_rows)

    figures = SOURCE_OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    import matplotlib.pyplot as plt

    if "h4" in selected_benchmarks:
        new_h4 = sorted(
            (row for row in output_rows if row["benchmark"] == "h4"),
            key=lambda row: float(row["slug"].split("R", 1)[1].replace("p", ".")),
        )
        if len(new_h4) != 21:
            raise RuntimeError(f"Expected 21 completed H4 rows, got {len(new_h4)}")
        baseline = sorted(
            read_csv(H4_BASELINE), key=lambda row: float(row["bond_length_angstrom"])
        )
        if len(baseline) != 21:
            raise RuntimeError("H4 baseline does not contain 21 bond lengths")
        h4_rows: list[dict[str, object]] = []
        for new, old in zip(new_h4, baseline):
            bond = float(old["bond_length_angstrom"])
            expected_slug = f"h4_R{bond:.1f}".replace(".", "p")
            if new["slug"] != expected_slug:
                raise RuntimeError(f"H4 row alignment failure: {new['slug']} != {expected_slug}")
            if new["matrix_sha256_h4_convention"] != old["target_sha256"]:
                raise RuntimeError(
                    f"H4 target hash mismatch at R={bond:.1f}: "
                    f"{new['matrix_sha256_h4_convention']} != {old['target_sha256']}"
                )
            if new["input_sha256"] != old["source_sha256"]:
                raise RuntimeError(
                    f"H4 source NPZ hash mismatch at R={bond:.1f}: "
                    f"{new['input_sha256']} != {old['source_sha256']}"
                )
            if int(new["empirical_repeats"]) != 50:
                raise RuntimeError(
                    f"H4 {new['slug']} has {new['empirical_repeats']} empirical repeats, expected 50"
                )
            h4_rows.append(
                {
                    "bond_length_angstrom": bond,
                    "new_selected_k": int(new["k"]),
                    "new_loss": float(new["loss"]),
                    "new_xa": float(new["approximation_feature_xa"]),
                    "new_xs": float(new["sampling_feature_xs"]),
                    "new_bias": float(new["bias"]),
                    "new_variance": float(new["variance"]),
                    "new_analytic_rmse": float(new["analytic_rmse"]),
                    "new_empirical_rmse": float(new["empirical_rmse"]),
                    "agpd_v10_analytic_rmse": float(old["v10_analytic_total_RMSE"]),
                    "agpd_v10_empirical_rmse": float(old["v10_empirical_RMSE"]),
                    "Derand_empirical_rmse": float(old["Derand_empirical_RMSE"]),
                    "OGM_empirical_rmse": float(old["OGM_empirical_RMSE"]),
                    "SG_empirical_rmse": float(old["SG_empirical_RMSE"]),
                    "target_sha256": old["target_sha256"],
                }
            )
        write_csv(HERE / "h4_bond_scan_comparison.csv", h4_rows)
        x = np.asarray([row["bond_length_angstrom"] for row in h4_rows])
        fig, axes = plt.subplots(2, 1, figsize=(8.3, 7.1), sharex=True, height_ratios=(3.2, 1.15))
        ax = axes[0]
        ax.plot(x, [row["new_analytic_rmse"] for row in h4_rows], color="#0B3C5D", lw=2.5, label="Periodic-iSWAP GPD + balanced-K (analytic)")
        ax.scatter(x, [row["new_empirical_rmse"] for row in h4_rows], color="#0B3C5D", s=22, zorder=5, label="Periodic-iSWAP GPD + balanced-K (50 repeats)")
        ax.plot(x, [row["agpd_v10_analytic_rmse"] for row in h4_rows], color="#777777", lw=1.7, ls=":", label="AGPD v10-FC (analytic)")
        styles = {"Derand": ("#D95F02", "--"), "OGM": ("#1B9E77", "-."), "SG": ("#7570B3", (0, (4, 2)))}
        for method in ("Derand", "OGM", "SG"):
            color, style = styles[method]
            ax.plot(x, [row[f"{method}_empirical_rmse"] for row in h4_rows], color=color, lw=1.6, ls=style, label=f"{method} (archived empirical)")
        ax.set_ylabel("Energy-estimation RMSE (Ha)")
        ax.grid(alpha=0.22)
        ax.legend(ncol=2, fontsize=8.2, frameon=False)
        axes[1].plot(x, [row["new_selected_k"] for row in h4_rows], color="#0B3C5D", marker="o", ms=3.5)
        axes[1].set_ylabel(r"Selected $K^\star$")
        axes[1].set_xlabel(r"H--H distance $R$ ($\AA$)")
        axes[1].grid(alpha=0.22)
        fig.suptitle(r"H$_4$ bond-length scan, $T=2038$, fixed periodic-iSWAP depth $L=12$", y=0.995)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"h4_periodic_iswap_gpd_balanced_k_vs_pauli.{suffix}", dpi=300)
        plt.close(fig)

    if {"sparse", "dense"} & selected_benchmarks:
        new_random = [row for row in output_rows if row["benchmark"] in {"sparse", "dense"}]
        baseline_manifest = json.loads(RANDOM_BASELINE_MANIFEST.read_text(encoding="utf-8"))
        case_audit = {
            row["slug"]: row
            for row in baseline_manifest["case_audit"]
            if row["benchmark"] in {"sparse", "dense"}
        }
        for row in new_random:
            expected = case_audit.get(row["slug"])
            if expected is None:
                raise RuntimeError(f"No retained baseline audit for {row['slug']}")
            if row["target_hash"] != expected["target_hash"]:
                raise RuntimeError(f"Random target hash mismatch for {row['slug']}")
            if row["state_hash"] != expected["state_hash"]:
                raise RuntimeError(f"Random state hash mismatch for {row['slug']}")
            if abs(float(row["target_expectation"]) - float(expected["target_expectation"])) > 2.0e-12:
                raise RuntimeError(f"Random target expectation mismatch for {row['slug']}")
        pauli_rows = [row for row in read_csv(RANDOM_PAULI_ROWS) if row["benchmark"] in {"sparse", "dense"} and row["method"] in PAULI_METHODS]
        v10_rows = [row for row in read_csv(RANDOM_V10_ROWS) if row["benchmark"] in {"sparse", "dense"} and row["method"] == "AGPD v10-FC adapter"]
        long_rows: list[dict[str, object]] = []
        for row in new_random:
            long_rows.append({"benchmark": row["benchmark"], "instance": int(row["slug"].rsplit("_", 1)[1]), "shots": int(row["shots"]), "method": "Periodic-iSWAP GPD + balanced-K", "rmse": float(row["analytic_rmse"]), "mse": float(row["mse"]), "selected_k": int(row["k"])})
        for source in (pauli_rows, v10_rows):
            for row in source:
                long_rows.append({"benchmark": row["benchmark"], "instance": int(row["instance"]), "shots": int(row["shots"]), "method": row["method"], "rmse": float(row["rmse"]), "mse": float(row["mse"]), "selected_k": row.get("selected_k", "")})
        write_csv(HERE / "random_instance_comparison.csv", long_rows)
        summary: list[dict[str, object]] = []
        for benchmark in ("sparse", "dense"):
            if benchmark not in selected_benchmarks:
                continue
            for method in ("Periodic-iSWAP GPD + balanced-K", "AGPD v10-FC adapter", *PAULI_METHODS):
                for total in RANDOM_SHOTS:
                    chosen = [row for row in long_rows if row["benchmark"] == benchmark and row["method"] == method and int(row["shots"]) == total]
                    if len(chosen) != 5:
                        raise RuntimeError(f"Expected 5 rows for {benchmark}/{method}/T={total}, got {len(chosen)}")
                    mean, sd = mean_sd([float(row["rmse"]) for row in chosen])
                    mean_mse = float(np.mean([float(row["mse"]) for row in chosen]))
                    summary.append({"benchmark": benchmark, "method": method, "shots": total, "mean_instance_rmse": mean, "rmse_instance_sd": sd, "mean_instance_mse": mean_mse, "pooled_rmse": math.sqrt(mean_mse), "instances": 5})
        write_csv(HERE / "random_summary.csv", summary)
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.35), sharey=False)
        method_styles = {
            "Periodic-iSWAP GPD + balanced-K": ("#0B3C5D", "-", 2.7),
            "AGPD v10-FC adapter": ("#777777", ":", 1.7),
            "OGM": ("#1B9E77", "--", 1.5),
            "SG": ("#7570B3", "-.", 1.5),
            "Derand": ("#D95F02", (0, (4, 2)), 1.5),
            "LCS": ("#E7298A", (0, (2, 2)), 1.35),
            "AP": ("#66A61E", (0, (6, 2, 1, 2)), 1.35),
        }
        for ax, benchmark in zip(axes, ("sparse", "dense")):
            for method in method_styles:
                selected = sorted((row for row in summary if row["benchmark"] == benchmark and row["method"] == method), key=lambda row: int(row["shots"]))
                totals = np.asarray([int(row["shots"]) for row in selected])
                means = np.asarray([float(row["mean_instance_rmse"]) for row in selected])
                sds = np.asarray([float(row["rmse_instance_sd"]) for row in selected])
                color, style, width = method_styles[method]
                ax.plot(totals, means, color=color, ls=style, lw=width, marker="o" if method == "Periodic-iSWAP GPD + balanced-K" else None, ms=3.5, label=method)
                if method == "Periodic-iSWAP GPD + balanced-K":
                    lower = np.maximum(means - sds, np.finfo(float).tiny)
                    ax.fill_between(totals, lower, means + sds, color=color, alpha=0.12, linewidth=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Total shots T")
            ax.set_title(f"{benchmark.capitalize()} 4-qubit ensemble (5 instances)")
            ax.grid(which="both", alpha=0.2)
        axes[0].set_ylabel("Mean of five instance RMSEs")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(r"Fixed periodic-iSWAP depth $L=4$; $K$ selected by the manuscript loss")
        fig.tight_layout(rect=(0, 0.10, 1, 0.95))
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"random_periodic_iswap_gpd_balanced_k_vs_pauli.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    manifest = {
        "profile": PROFILE,
        "generated_utc": utc_now(),
        "benchmarks": sorted(selected_benchmarks),
        "completed_case_count": len({row["slug"] for row in output_rows}),
        "algorithm_script": str(Path(__file__).resolve()),
        "algorithm_script_sha256": sha256(Path(__file__).resolve()),
        "canonical_gpd": str(CANONICAL_GPD),
        "canonical_gpd_sha256": sha256(CANONICAL_GPD),
        "baselines": {
            str(H4_BASELINE): sha256(H4_BASELINE) if H4_BASELINE.exists() else None,
            str(RANDOM_PAULI_ROWS): sha256(RANDOM_PAULI_ROWS) if RANDOM_PAULI_ROWS.exists() else None,
            str(RANDOM_BASELINE_MANIFEST): sha256(RANDOM_BASELINE_MANIFEST) if RANDOM_BASELINE_MANIFEST.exists() else None,
            str(RANDOM_V10_ROWS): sha256(RANDOM_V10_ROWS) if RANDOM_V10_ROWS.exists() else None,
        },
    }
    write_json(SOURCE_OUTPUT_ROOT / "aggregate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    case = subparsers.add_parser("case", help="Run or resume one Hamiltonian frontier")
    case.add_argument("--benchmark", choices=("h4", "sparse", "dense"), required=True)
    case.add_argument("--slug", required=True)
    case.add_argument("--label", default="")
    case.add_argument("--input", type=Path, required=True)
    case.add_argument("--state", type=Path)
    case.add_argument("--output", type=Path, required=True)
    case.add_argument("--paper-depth", type=int, required=True)
    case.add_argument("--particle-number", type=int)
    case.add_argument("--shots", type=int, nargs="+", required=True)
    case.add_argument("--starts", type=int, default=1)
    case.add_argument("--steps", type=int, default=600)
    case.add_argument("--learning-rate", type=float, default=1.0e-2)
    case.add_argument("--base-seed", type=int, default=20260808)
    case.add_argument("--minimum-k", type=int, default=8)
    case.add_argument("--stall-patience", type=int, default=5)
    case.add_argument("--relative-improvement", type=float, default=2.0e-3)
    case.add_argument("--max-k", type=int, default=40)
    case.add_argument("--progress-every", type=int, default=100)
    case.add_argument("--empirical-repeats", type=int, default=0)
    case.add_argument("--x64", action="store_true")
    case.add_argument("--resume", action="store_true")
    case.add_argument("--force", action="store_true")
    case.set_defaults(func=run_case)

    suite = subparsers.add_parser("suite", help="Run H4/random cases with bounded process parallelism")
    suite.add_argument("--benchmarks", nargs="+", choices=("h4", "sparse", "dense"), default=("h4", "sparse", "dense"))
    suite.add_argument("--workers", type=int, default=4)
    suite.add_argument("--h4-depth", type=int, default=12)
    suite.add_argument("--random-depth", type=int, default=4)
    suite.add_argument("--h4-repeats", type=int, default=50)
    suite.add_argument("--starts", type=int, default=1)
    suite.add_argument("--steps", type=int, default=600)
    suite.add_argument("--learning-rate", type=float, default=1.0e-2)
    suite.add_argument("--base-seed", type=int, default=20260808)
    suite.add_argument("--minimum-k", type=int, default=8)
    suite.add_argument("--stall-patience", type=int, default=5)
    suite.add_argument("--relative-improvement", type=float, default=2.0e-3)
    suite.add_argument("--max-k", type=int, default=40)
    suite.add_argument("--progress-every", type=int, default=100)
    suite.add_argument("--x64", action="store_true")
    suite.add_argument("--force", action="store_true")
    suite.set_defaults(func=run_suite)

    aggregate = subparsers.add_parser("aggregate", help="Regenerate CSV summaries and review figures")
    aggregate.add_argument("--benchmarks", nargs="+", choices=("h4", "sparse", "dense"), default=("h4", "sparse", "dense"))
    aggregate.set_defaults(func=lambda args: aggregate_results(args.benchmarks))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
