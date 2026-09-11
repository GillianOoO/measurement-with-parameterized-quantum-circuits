#!/usr/bin/env python3
"""Full-space H2O/N2 comparison: standalone s-RCDF versus Pauli methods.

The molecular inputs are exactly the H2O R=0.80 A and N2 R=2.25 A archives
documented in ``Molecules_origin.md``.  The code never materializes a dense
Hamiltonian.  Shallow RCDF is fitted in the spatial ERI tensor, its fixed-N
Frobenius residual is evaluated analytically from Jordan--Wigner Pauli
coefficients, and its diagonal measurement distributions are evaluated in the
fixed-particle CI representation of the archived exact FCI MPS.

The production default removes the independently diagonalized dense one-body
collector.  It is exactly redistributed into the existing shallow rotations,
or into a collector-tailored depth-matched augmentation bank, under an audited
matrix equality constraint.  ``--collector-mode exact`` remains available as
a diagnostic benchmark rather than the production path.

SG, Derandomization, and OGM use the archived full JW Pauli Hamiltonian and the
same exact FCI MPS.  A common sorted-insertion QWC cover is measured once before
method-specific shots are added; consequently every nonzero Pauli term is hit
at least once and the pooled term-mean estimator is unbiased.  Exact Born
samples are generated from locally rotated MPS objects, preserving same-basis
covariances.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import matplotlib
import numpy as np
import pandas as pd
import quimb.tensor as qtn
import scipy.linalg as la
import scipy.sparse as sp
from scipy.optimize import linprog, minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SRDD_METHODS = HERE.parent / "methods" / "srdd"
sys.path.insert(0, str(SRDD_METHODS))
from srdd_release_paths import BUNDLE as WORKSPACE, DATA, METHODS as METHOD_ROOT, RUNS

LEGACY_PAULI_DRIVER = METHOD_ROOT / "pauli_product" / "run_same_hamiltonian_pauli_methods.py"
HUANG_DERAND_IMPLEMENTATION = LEGACY_PAULI_DRIVER.with_name("derand_huang2021.py")
DEFAULT_OUTPUT = RUNS / "n2_srdd_pauli"
VERSION = "h2o-n2-fullspace-srcdf-vs-pauli-shallow-collector-v5-huang2021"
DERAND_ALGORITHM_ID = "qwc-seeded-huang2021-appendix-c-c8-c11-stable-delta-v1"
OGM_ITERATION_SAFETY_GUARD = 100_000
PAULI_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
METHODS = ("s-RCDF", "OGM", "SG", "Derand")
COLORS = {
    "s-RCDF": "#CC3311",
    "OGM": "#E69F00",
    "SG": "#009E73",
    "Derand": "#7A5195",
}
MARKERS = {"s-RCDF": "o", "OGM": "s", "SG": "^", "Derand": "D"}
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)
S_DAGGER = np.diag([1.0, -1.0j]).astype(np.complex128)
BASIS_GATES = {1: HADAMARD, 2: HADAMARD @ S_DAGGER, 3: np.eye(2)}

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from shallow_rcdf import (
    VERSION as SHALLOW_VERSION,
    optimize_shallow_rcdf,
)
import shallow_collector_redistribution as collector_core


@dataclass(frozen=True)
class CaseSpec:
    name: str
    directory: str
    geometry: str
    electrons: int


CASES = {
    "N2": CaseSpec("N2", "03_N2_R2.25A", "N-N=2.25 A", 14),
}


def load_module(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAULI_IMPL = load_module(LEGACY_PAULI_DRIVER, "fullspace_pauli_schedule_impl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join([VERSION, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_typed(path: Path) -> list[dict[str, Any]]:
    """Read a saved comparison table while preserving numeric plot fields."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value == "":
                parsed[key] = ""
            elif value == "True":
                parsed[key] = True
            elif value == "False":
                parsed[key] = False
            else:
                try:
                    numeric = float(value)
                except ValueError:
                    parsed[key] = value
                else:
                    parsed[key] = int(numeric) if numeric.is_integer() else numeric
        result.append(parsed)
    return result


def load_reused_pauli_rows(
    source_root: Path,
    molecule: str,
    budgets: list[int],
    repeats: int,
    current_input_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load an audited same-target Pauli baseline without rerunning schedules."""
    source_root = source_root.resolve()
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    if list(map(int, source_manifest["shot_budgets"])) != list(map(int, budgets)):
        raise RuntimeError("Reused Pauli artifact has a different shot grid")
    if int(source_manifest["sampling_repeats"]) != int(repeats):
        raise RuntimeError("Reused Pauli artifact has a different repeat count")
    audit_path = source_root / molecule / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError(f"Reused {molecule} Pauli audit is not PASS")
    if audit.get("input_sha256") != current_input_hashes:
        raise RuntimeError(f"Reused {molecule} Pauli artifact has different molecular inputs")
    reused_schedule = audit.get("schedule", {})
    if reused_schedule.get("Derand_algorithm_id") != DERAND_ALGORITHM_ID:
        raise RuntimeError(
            f"Reused {molecule} Pauli artifact predates the corrected Huang-2021 Derand"
        )
    current_derand_hash = sha256_file(HUANG_DERAND_IMPLEMENTATION)
    if reused_schedule.get("Derand_implementation_sha256") != current_derand_hash:
        raise RuntimeError(
            f"Reused {molecule} Pauli artifact has a different Derand implementation"
        )
    methods = {"OGM", "SG", "Derand"}
    summaries = [
        row for row in read_csv_typed(source_root / molecule / "sampling_summary.csv")
        if row.get("method") in methods and int(row["T_total_shots"]) in budgets
    ]
    replicates = [
        row for row in read_csv_typed(source_root / molecule / "sampling_replicates.csv")
        if row.get("method") in methods and int(row["T_total_shots"]) in budgets
    ]
    expected_summary = len(methods) * len(budgets)
    expected_replicates = expected_summary * repeats
    if len(summaries) != expected_summary or len(replicates) != expected_replicates:
        raise RuntimeError(
            f"Incomplete reused {molecule} Pauli rows: "
            f"{len(summaries)}/{expected_summary}, {len(replicates)}/{expected_replicates}"
        )
    for row in summaries:
        if int(row["T_actual_shots"]) != int(row["T_total_shots"]):
            raise RuntimeError(f"Reused {molecule} Pauli schedule does not conserve shots")
        if int(row["covered_pauli_terms"]) <= 0 or int(row["minimum_term_hits"]) < 1:
            raise RuntimeError(f"Reused {molecule} Pauli schedule lacks full coverage")
    provenance = {
        "source_root": str(source_root),
        "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "source_case_audit_sha256": sha256_file(audit_path),
        "source_summary_sha256": sha256_file(source_root / molecule / "sampling_summary.csv"),
        "source_replicates_sha256": sha256_file(source_root / molecule / "sampling_replicates.csv"),
    }
    return summaries, replicates, {"schedule": audit["schedule"], "provenance": provenance}


def load_mps(path: Path) -> qtn.MatrixProductState:
    with np.load(path, allow_pickle=False) as payload:
        arrays = [np.asarray(payload[key]) for key in sorted(payload.files)]
    state = qtn.MatrixProductState(arrays, shape="lpr")
    norm = complex(state.H @ state)
    if abs(norm - 1.0) > 2.0e-10:
        raise ValueError(f"MPS norm mismatch {norm} in {path}")
    return state


def dense_mps(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = [np.asarray(payload[key]) for key in sorted(payload.files)]
    value = arrays[0][0, :, :]
    for array in arrays[1:]:
        value = np.tensordot(value, array, axes=(-1, 0))
    state = np.asarray(value[..., 0], dtype=np.complex128).reshape(-1)
    state /= np.linalg.norm(state)
    return state, max(max(a.shape[0], a.shape[2]) for a in arrays)


def load_case(spec: CaseSpec) -> dict[str, Any]:
    directory = DATA / spec.name / "inputs"
    integral_path = directory / "integrals_mo.npz"
    pauli_path = directory / "hamiltonian_pauli_blocked_spin.csv"
    metadata_path = directory / "metadata.json"
    state_path = directory / "ground_state_mps_blocked_spin.npz"
    with np.load(integral_path, allow_pickle=False) as saved:
        one = np.asarray(saved["one_body_integrals"], dtype=float)
        two_open = np.asarray(saved["two_body_integrals_openfermion_order"], dtype=float)
        constant = float(saved["nuclear_repulsion_hartree"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(pauli_path)
    coefficients = frame["coefficient_real_hartree"].to_numpy(dtype=float)
    imaginary = frame["coefficient_imag_hartree"].to_numpy(dtype=float)
    # A small anti-Hermitian numerical residue is present in the archived N2
    # CSV (purely imaginary coefficients up to about 3e-8 Ha).  Measurement
    # observables must be Hermitian, so use (H+H^dagger)/2, i.e. retain the
    # real Pauli coefficients, and audit the discarded residue explicitly.
    imaginary_max = float(np.max(np.abs(imaginary)))
    imaginary_l2 = float(np.linalg.norm(imaginary))
    labels = frame["full_label_site0_to_siteNminus1"].astype(str).tolist()
    observables = np.asarray(
        [[PAULI_CODE[c] for c in label] for label in labels], dtype=np.int8
    )
    identity = np.all(observables == 0, axis=1)
    if int(identity.sum()) != 1:
        raise ValueError(f"{spec.name}: identity count {identity.sum()}")
    offset = float(coefficients[identity][0])
    keep = (~identity) & (np.abs(coefficients) > 1.0e-14)
    discarded_zero = int(np.count_nonzero((~identity) & (~keep)))
    observables = observables[keep]
    weights = coefficients[keep]
    m = one.shape[0]
    n = 2 * m
    if observables.shape[1] != n or spec.electrons % 2:
        raise ValueError(f"{spec.name}: inconsistent orbital/electron metadata")
    one_raw = one.copy()
    one = 0.5 * (one + one.T)
    eri_raw = two_open.transpose(0, 3, 1, 2).copy()
    eri_chemist = sum(
        np.transpose(eri_raw, axes)
        for axes in (
            (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
            (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
        )
    ) / 8.0
    two_open = eri_chemist.transpose(0, 2, 3, 1).copy()
    return {
        "spec": spec,
        "directory": directory,
        "one": one,
        "two_open": two_open,
        "eri": eri_chemist,
        "constant": constant,
        "m": m,
        "n": n,
        "nalpha": spec.electrons // 2,
        "metadata": metadata,
        "exact_energy": float(metadata["fci_ground_energy_hartree"]),
        "observables": observables,
        "weights": weights,
        "offset": offset,
        "discarded_zero_pauli_rows": discarded_zero,
        "discarded_antihermitian_pauli_imaginary_max": imaginary_max,
        "discarded_antihermitian_pauli_imaginary_l2": imaginary_l2,
        "one_body_Hermitian_projection_frobenius": float(np.linalg.norm(one_raw - one)),
        "two_body_eightfold_projection_frobenius": float(np.linalg.norm(eri_raw - eri_chemist)),
        "paths": {
            "integrals": integral_path,
            "pauli": pauli_path,
            "metadata": metadata_path,
            "state": state_path,
        },
    }


def blocked_spin_coefficients(one_spatial: np.ndarray, two_open: np.ndarray):
    from openfermion.chem.molecular_data import spinorb_from_spatial

    one_i, two_i = spinorb_from_spatial(one_spatial, two_open)
    m = one_spatial.shape[0]
    permutation = np.asarray([2 * p for p in range(m)] + [2 * p + 1 for p in range(m)])
    return (
        one_i[np.ix_(permutation, permutation)],
        two_i[np.ix_(permutation, permutation, permutation, permutation)],
    )


def residual_pauli_terms(two_open_residual: np.ndarray) -> list[tuple[int, int, float, int]]:
    from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner

    m = two_open_residual.shape[0]
    one_spin, two_spin = blocked_spin_coefficients(
        np.zeros((m, m)), two_open_residual
    )
    interaction = InteractionOperator(0.0, one_spin, 0.5 * two_spin)
    qop = jordan_wigner(get_fermion_operator(interaction))
    result = []
    for term, coefficient in qop.terms.items():
        value = complex(coefficient)
        # Match the Hermitian observable convention used by the dense legacy
        # implementation: keep the real coefficient of each Hermitian Pauli
        # string.  N2 carries an archived O(1e-8) anti-Hermitian roundoff
        # residue which is separately recorded at input loading.
        if abs(value.real) <= 1.0e-13:
            continue
        xmask = 0
        zmask = 0
        ny = 0
        for site, letter in term:
            if letter in ("X", "Y"):
                xmask |= 1 << site
            if letter in ("Z", "Y"):
                zmask |= 1 << site
            if letter == "Y":
                ny += 1
        result.append((xmask, zmask, float(value.real), ny))
    return result


def krawtchouk_trace(n: int, particles: int, z_weight: int) -> int:
    lower = max(0, particles - (n - z_weight))
    upper = min(particles, z_weight)
    return sum(
        (-1) ** overlap
        * math.comb(z_weight, overlap)
        * math.comb(n - z_weight, particles - overlap)
        for overlap in range(lower, upper + 1)
    )


def sector_frobenius(terms: list[tuple[int, int, float, int]], n: int, particles: int) -> float:
    dimension = math.comb(n, particles)
    trace_cache = {weight: krawtchouk_trace(n, particles, weight) for weight in range(n + 1)}
    groups: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    for xmask, zmask, coefficient, ny in terms:
        groups[xmask].append((zmask, coefficient, ny))
    norm_sq = 0.0
    for xmask, group in groups.items():
        for i, (zi, ci, nyi) in enumerate(group):
            norm_sq += ci * ci * dimension
            for zj, cj, nyj in group[i + 1 :]:
                exponent = (nyi + nyj) % 4
                phase = (1j ** exponent) * ((-1) ** ((zi & xmask).bit_count()))
                trace = trace_cache[(zi ^ zj).bit_count()]
                norm_sq += 2.0 * ci * cj * float(np.real(phase)) * trace
    return math.sqrt(max(norm_sq, 0.0))


def full_frobenius(terms: list[tuple[int, int, float, int]], n: int) -> float:
    return math.sqrt((2**n) * sum(coefficient**2 for _, _, coefficient, _ in terms))


def determinant_configs(m: int, count: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(m), count))


def determinant_masks(configs: list[tuple[int, ...]], offset: int = 0) -> np.ndarray:
    return np.asarray([sum(1 << (offset + q) for q in config) for config in configs], dtype=np.uint64)


def wedge_rotation(rotation: np.ndarray, configs: list[tuple[int, ...]]) -> np.ndarray:
    indices = np.asarray(configs, dtype=int)
    submatrices = rotation[
        indices[:, None, :, None], indices[None, :, None, :]
    ]
    return np.linalg.det(submatrices)


def ci_state(case: dict[str, Any]) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    state, max_bond = dense_mps(case["paths"]["state"])
    configs = determinant_configs(case["m"], case["nalpha"])
    alpha_masks = determinant_masks(configs, 0)
    beta_masks = determinant_masks(configs, case["m"])
    dense_indices = np.empty((len(configs), len(configs)), dtype=np.int64)
    for ia, alpha in enumerate(alpha_masks):
        for ib, beta in enumerate(beta_masks):
            occupation_mask = int(alpha | beta)
            index = 0
            for q in range(case["n"]):
                if occupation_mask & (1 << q):
                    index |= 1 << (case["n"] - 1 - q)
            dense_indices[ia, ib] = index
    ci = state[dense_indices]
    leakage = math.sqrt(max(1.0 - float(np.vdot(ci, ci).real), 0.0))
    if leakage > 1.0e-7:
        raise ValueError(f"{case['spec'].name}: fixed-spin-sector leakage {leakage}")
    ci /= np.linalg.norm(ci)
    masks = (alpha_masks[:, None] | beta_masks[None, :]).reshape(-1)
    case["ground_state_max_bond"] = int(max_bond)
    case["ground_sector_leakage"] = leakage
    return ci, configs, masks


def spatial_occupations(configs: list[tuple[int, ...]], m: int) -> np.ndarray:
    values = np.zeros((len(configs), m), dtype=float)
    for row, config in enumerate(configs):
        values[row, list(config)] = 1.0
    return values


def all_spatial_occupations(m: int) -> np.ndarray:
    return np.asarray(list(itertools.product((0.0, 1.0, 2.0), repeat=m)))


def integer_allocation(total: int, lambdas: np.ndarray) -> np.ndarray:
    values = np.asarray(lambdas, dtype=float)
    active = values > 1.0e-14
    if total < int(active.sum()):
        raise ValueError("Shot budget cannot cover all nonconstant settings")
    result = np.zeros(len(values), dtype=int)
    result[active] = 1
    remaining = total - int(active.sum())
    if remaining:
        probabilities = values[active] / values[active].sum()
        quotas = remaining * probabilities
        base = np.floor(quotas).astype(int)
        base[np.argsort(-(quotas - base), kind="stable")[: remaining - int(base.sum())]] += 1
        result[active] += base
    if int(result.sum()) != total:
        raise RuntimeError("Integer allocation failed to conserve shots")
    return result


def one_body_covariance(a: np.ndarray, b: np.ndarray, occupied: int) -> float:
    gamma = np.zeros_like(a)
    gamma[:occupied, :occupied] = np.eye(occupied)
    value = 2.0 * np.trace(gamma @ a @ (np.eye(len(a)) - gamma) @ b)
    return float(np.real(value))


def diagonal_leaf_values(occupation: np.ndarray, z: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    quadratic = 0.5 * np.einsum("...i,ij,...j->...", occupation, z, occupation)
    linear = 0.5 * occupation @ np.diag(z) + alpha * (occupation @ direction)
    return quadratic - linear


class CollectorRepresentationError(RuntimeError):
    """Raised when a declared shallow bank cannot reproduce the collector."""

    def __init__(self, audit: dict[str, Any]):
        self.audit = audit
        super().__init__(
            "Shallow collector equality constraint is infeasible: "
            f"relative residual={audit['relative_reconstruction_residual']:.3e}, "
            f"rank={audit['dictionary_rank']}/{audit['symmetric_target_dimension']}"
        )


def collector_proxy_moments(
    rotations: np.ndarray,
    base_values: np.ndarray,
    configs: list[tuple[int, ...]],
    occupations: np.ndarray,
    ci: np.ndarray,
    proxy_state: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Var(base), Cov(base,N), and Cov(N,N) for each setting."""
    if proxy_state not in {"hf", "ground"}:
        raise ValueError(f"Unknown collector proxy state: {proxy_state}")
    occ_pair = occupations[:, None, :] + occupations[None, :, :]
    occupation_rows = occ_pair.reshape(-1, occ_pair.shape[-1])
    base_variances = []
    base_covariances = []
    occupation_covariances = []
    for rotation, values in zip(rotations, base_values):
        wedge = wedge_rotation(rotation, configs)
        if proxy_state == "ground":
            amplitude = wedge.T @ ci @ wedge
            probabilities = np.abs(amplitude.reshape(-1)) ** 2
        else:
            probabilities_spin = np.abs(wedge[0]) ** 2
            probabilities = np.outer(probabilities_spin, probabilities_spin).reshape(-1)
        probabilities /= probabilities.sum()
        diagonal = np.asarray(values, dtype=float).reshape(-1)
        mean_n = probabilities @ occupation_rows
        centered_n = occupation_rows - mean_n
        mean_base = float(probabilities @ diagonal)
        centered_base = diagonal - mean_base
        base_variances.append(max(float(probabilities @ centered_base**2), 0.0))
        base_covariances.append((probabilities * centered_base) @ centered_n)
        occupation_covariances.append(
            centered_n.T @ (probabilities[:, None] * centered_n)
        )
    return (
        np.asarray(base_variances),
        np.asarray(base_covariances),
        np.asarray(occupation_covariances),
    )


def collector_proxy_variances(
    vector: np.ndarray,
    base_variances: np.ndarray,
    base_covariances: np.ndarray,
    occupation_covariances: np.ndarray,
) -> np.ndarray:
    eta = np.asarray(vector, dtype=float).reshape(len(base_variances), -1)
    variances = (
        base_variances
        + 2.0 * np.einsum("ti,ti->t", base_covariances, eta)
        + np.einsum("ti,tij,tj->t", eta, occupation_covariances, eta)
    )
    return np.maximum(variances, 0.0)


def redistribution_metrics(
    vector: np.ndarray,
    base_full_values: np.ndarray,
    all_occupations: np.ndarray,
    base_variances: np.ndarray,
    base_covariances: np.ndarray,
    occupation_covariances: np.ndarray,
    reference_shots: int,
) -> dict[str, Any]:
    eta = np.asarray(vector, dtype=float).reshape(len(base_full_values), -1)
    full_values = base_full_values + eta @ all_occupations.T
    lambdas = 0.5 * (np.max(full_values, axis=1) - np.min(full_values, axis=1))
    variances = collector_proxy_variances(
        vector, base_variances, base_covariances, occupation_covariances
    )
    shots = integer_allocation(reference_shots, lambdas)
    estimator_variance = 0.0
    for variance, count in zip(variances, shots):
        if count:
            estimator_variance += float(variance / count)
        elif variance > 1.0e-14:
            estimator_variance = float("inf")
            break
    return {
        "proxy_estimator_SE_at_reference_shots": math.sqrt(max(estimator_variance, 0.0)),
        "centered_range_sum": float(lambdas.sum()),
        "coefficient_l1": float(np.abs(vector).sum()),
        "active_settings_at_reference_shots": int(np.sum(shots > 0)),
        "reference_shot_vector": shots.tolist(),
        "lambdas": lambdas,
        "proxy_variances": variances,
    }


def variance_optimized_collector(
    particular: np.ndarray,
    null_basis: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    base_full_values: np.ndarray,
    all_occupations: np.ndarray,
    base_variances: np.ndarray,
    base_covariances: np.ndarray,
    occupation_covariances: np.ndarray,
    reference_shots: int,
    cycles: int,
) -> np.ndarray:
    """Equality-constrained proxy-variance optimization in null(A)."""
    if null_basis.shape[1] == 0:
        return particular.copy()
    settings, m = base_covariances.shape
    vector = particular.copy()
    for _ in range(cycles):
        metrics = redistribution_metrics(
            vector,
            base_full_values,
            all_occupations,
            base_variances,
            base_covariances,
            occupation_covariances,
            reference_shots,
        )
        shots = np.asarray(metrics["reference_shot_vector"], dtype=float)
        fractions = shots / reference_shots
        hessian = np.zeros((settings * m, settings * m))
        gradient = np.zeros(settings * m)
        for setting in range(settings):
            block = slice(setting * m, (setting + 1) * m)
            if fractions[setting] > 0.0:
                hessian[block, block] = occupation_covariances[setting] / fractions[setting]
                gradient[block] = base_covariances[setting] / fractions[setting]
        reduced = null_basis.T @ hessian @ null_basis
        reduced_rhs = -null_basis.T @ (hessian @ particular + gradient)
        ridge = 1.0e-11 * max(1.0, float(np.trace(reduced)) / max(len(reduced), 1))
        reduced.flat[:: len(reduced) + 1] += ridge
        try:
            coordinates = la.solve(reduced, reduced_rhs, assume_a="sym", check_finite=False)
        except la.LinAlgError:
            coordinates = np.linalg.lstsq(reduced, reduced_rhs, rcond=1.0e-12)[0]
        candidate = particular + null_basis @ coordinates
        if not np.all(np.isfinite(candidate)):
            break
        vector = equality_project(candidate, design, target)
    return vector


def range_optimized_collector(
    particular: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize the L1 upper bound on total centered one-body range."""
    count = design.shape[1]
    objective = np.r_[np.zeros(count), np.ones(count)]
    a_ub = np.block([
        [np.eye(count), -np.eye(count)],
        [-np.eye(count), -np.eye(count)],
    ])
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.zeros(2 * count),
        A_eq=np.c_[design, np.zeros_like(design)],
        b_eq=target,
        bounds=[(None, None)] * count + [(0.0, None)] * count,
        method="highs",
        options={"dual_feasibility_tolerance": 1.0e-9, "primal_feasibility_tolerance": 1.0e-9},
    )
    if not result.success:
        return particular.copy(), {
            "success": False,
            "status": int(result.status),
            "message": result.message,
        }
    return equality_project(np.asarray(result.x[:count]), design, target), {
        "success": True,
        "status": int(result.status),
        "message": result.message,
        "iterations": int(result.nit),
    }


# Backward-compatible public aliases.  The shared module is the authoritative
# implementation used by both this full-space driver and the four-orbital
# standalone pipeline.
symmetric_vector = collector_core.symmetric_vector
collector_dictionary = collector_core.collector_dictionary
collector_matrix = collector_core.collector_matrix
dictionary_diagnostics = collector_core.dictionary_diagnostics
equality_project = collector_core.equality_project
collector_tailored_extra_shallow_rotations = (
    collector_core.collector_tailored_extra_shallow_rotations
)


def shallow_collector_redistribution(
    collector: np.ndarray,
    source_rotations: np.ndarray,
    base_sector_values: np.ndarray,
    base_full_values: np.ndarray,
    configs: list[tuple[int, ...]],
    occupations: np.ndarray,
    all_occupations: np.ndarray,
    ci: np.ndarray,
    depth: int,
    extra_leaves: int,
    objective: str,
    proxy_state: str,
    reference_shots: int,
    proxy_cycles: int,
    equality_tolerance: float,
    extra_angle_steps: int,
    seed_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Exactly expand a one-body collector in a depth-limited rotation bank."""
    if objective not in {"greedy", "variance", "range", "hybrid"}:
        raise ValueError(f"Unknown collector redistribution objective: {objective}")
    if reference_shots < 1 or proxy_cycles < 1 or equality_tolerance <= 0.0:
        raise ValueError("Invalid shallow collector optimization controls")
    extra, greedy_extra_coefficients, extra_audit = (
        collector_core.collector_tailored_extra_shallow_rotations(
        source_rotations,
        collector,
        depth,
        extra_leaves,
        seed_key,
        extra_angle_steps,
        # The physical collector construction must not change when this engine
        # is entered through the BeH2 wrapper rather than the H2O/N2 CLI.
        seed_namespace=collector_core.VERSION,
    ))
    rotations = np.concatenate((np.asarray(source_rotations), extra), axis=0)
    sector_zeros = np.zeros((extra_leaves, *base_sector_values.shape[1:]))
    full_zeros = np.zeros((extra_leaves, base_full_values.shape[1]))
    sector_values = np.concatenate((np.asarray(base_sector_values), sector_zeros), axis=0)
    full_values = np.concatenate((np.asarray(base_full_values), full_zeros), axis=0)
    design = collector_core.collector_dictionary(rotations)
    target = collector_core.symmetric_vector(collector)
    rank, condition, singular_values, null_basis = collector_core.dictionary_diagnostics(
        design, equality_tolerance
    )
    particular = np.linalg.lstsq(design, target, rcond=equality_tolerance)[0]
    particular = equality_project(particular, design, target)
    relative_residual = float(
        np.linalg.norm(design @ particular - target) / max(np.linalg.norm(target), 1.0)
    )
    feasibility = {
        "dictionary_rank": rank,
        "dictionary_columns": int(design.shape[1]),
        "symmetric_target_dimension": int(design.shape[0]),
        "dictionary_nullity": int(design.shape[1] - rank),
        "dictionary_condition_on_numerical_range": condition,
        "smallest_retained_singular_value": (
            float(singular_values[rank - 1]) if rank else 0.0
        ),
        "relative_reconstruction_residual": relative_residual,
        "equality_tolerance": equality_tolerance,
        "source_rotation_count": int(len(source_rotations)),
        "extra_rotation_count": int(extra_leaves),
        "extra_rotation_seed_namespace": collector_core.VERSION,
        "extra_rotation_construction": extra_audit,
    }
    if relative_residual > equality_tolerance:
        raise CollectorRepresentationError(feasibility)

    source_columns = len(source_rotations) * rotations.shape[1]
    greedy_seed, source_fill_residual = collector_core.greedy_exact_fill(
        design,
        target,
        source_columns,
        greedy_extra_coefficients,
        equality_tolerance,
    )

    base_variances, base_covariances, occupation_covariances = collector_proxy_moments(
        rotations, sector_values, configs, occupations, ci, proxy_state
    )
    variance_vector = variance_optimized_collector(
        particular,
        null_basis,
        design,
        target,
        full_values,
        all_occupations,
        base_variances,
        base_covariances,
        occupation_covariances,
        reference_shots,
        proxy_cycles,
    )
    range_vector, range_audit = range_optimized_collector(particular, design, target)
    raw_candidates: list[tuple[str, np.ndarray]] = [
        ("minimum_norm", particular),
        ("collector_tailored_greedy_exact_fill", greedy_seed),
        ("proxy_variance_nullspace", variance_vector),
        ("l1_range_nullspace", range_vector),
    ]
    if objective == "hybrid":
        for weight in np.linspace(0.1, 0.9, 9):
            raw_candidates.append((
                f"hybrid_interpolation_{weight:.1f}",
                (1.0 - weight) * variance_vector + weight * range_vector,
            ))
        for weight in (0.25, 0.5, 0.75):
            raw_candidates.extend((
                (
                    f"greedy_variance_interpolation_{weight:.2f}",
                    (1.0 - weight) * greedy_seed + weight * variance_vector,
                ),
                (
                    f"greedy_range_interpolation_{weight:.2f}",
                    (1.0 - weight) * greedy_seed + weight * range_vector,
                ),
            ))
    candidate_rows = []
    for name, vector in raw_candidates:
        feasible_vector = equality_project(np.asarray(vector), design, target)
        metrics = redistribution_metrics(
            feasible_vector,
            full_values,
            all_occupations,
            base_variances,
            base_covariances,
            occupation_covariances,
            reference_shots,
        )
        candidate_rows.append({"name": name, "vector": feasible_vector, **metrics})
    if objective == "greedy":
        selected = next(
            row
            for row in candidate_rows
            if row["name"] == "collector_tailored_greedy_exact_fill"
        )
    elif objective == "variance":
        allowed = [row for row in candidate_rows if row["name"] != "l1_range_nullspace"]
        selected = min(
            allowed,
            key=lambda row: (
                row["proxy_estimator_SE_at_reference_shots"],
                row["centered_range_sum"],
            ),
        )
    elif objective == "range":
        selected = min(
            candidate_rows,
            key=lambda row: (
                row["centered_range_sum"],
                row["proxy_estimator_SE_at_reference_shots"],
            ),
        )
    else:
        se_scale = max(
            min(row["proxy_estimator_SE_at_reference_shots"] for row in candidate_rows),
            1.0e-16,
        )
        range_scale = max(
            min(row["centered_range_sum"] for row in candidate_rows), 1.0e-16
        )
        selected = min(
            candidate_rows,
            key=lambda row: (
                math.hypot(
                    row["proxy_estimator_SE_at_reference_shots"] / se_scale,
                    row["centered_range_sum"] / range_scale,
                ),
                row["proxy_estimator_SE_at_reference_shots"],
            ),
        )
    vector = selected["vector"]
    coefficients = vector.reshape(len(rotations), rotations.shape[1])
    reconstructed = collector_core.collector_matrix(rotations, coefficients)
    matrix_residual = float(np.linalg.norm(reconstructed - collector))
    matrix_relative = matrix_residual / max(float(np.linalg.norm(collector)), 1.0)
    if matrix_relative > 5.0 * equality_tolerance:
        feasibility["relative_reconstruction_residual"] = matrix_relative
        raise CollectorRepresentationError(feasibility)
    audit_candidates = []
    for row in candidate_rows:
        audit_candidates.append({
            key: value
            for key, value in row.items()
            if key not in {"vector", "lambdas", "proxy_variances"}
        })
    audit = {
        **feasibility,
        "status": "PASS",
        "mode": "augment" if extra_leaves else "absorb",
        "objective": objective,
        "proxy_state": proxy_state,
        "reference_shots": reference_shots,
        "selected_candidate": selected["name"],
        "selected_proxy_estimator_SE_at_reference_shots": selected[
            "proxy_estimator_SE_at_reference_shots"
        ],
        "selected_centered_range_sum": selected["centered_range_sum"],
        "selected_reference_shot_vector": selected["reference_shot_vector"],
        "matrix_reconstruction_residual_frobenius": matrix_residual,
        "matrix_relative_reconstruction_residual": matrix_relative,
        "range_linear_program": range_audit,
        "greedy_existing_bank_fill_relative_residual": source_fill_residual,
        "candidate_metrics": audit_candidates,
        "coefficients": coefficients.tolist(),
    }
    return rotations, coefficients, sector_values, audit


def f3_r2_alpha(
    one: np.ndarray,
    rotations: np.ndarray,
    tensors: np.ndarray,
    configs: list[tuple[int, ...]],
    occupations: np.ndarray,
    all_occupations: np.ndarray,
    nalpha: int,
    maxfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    k = len(rotations)
    directions = np.asarray([z.sum(axis=1) - 0.5 * np.diag(z) for z in tensors])
    q_matrices = np.asarray([r @ np.diag(d) @ r.T for r, d in zip(rotations, directions)])
    gram = np.empty((k, k))
    rhs0 = np.empty(k)
    for i in range(k):
        rhs0[i] = one_body_covariance(q_matrices[i], one, nalpha)
        for j in range(k):
            gram[i, j] = one_body_covariance(q_matrices[i], q_matrices[j], nalpha)
    var0 = max(one_body_covariance(one, one, nalpha), 0.0)
    var_leaves = np.empty(k)
    cov_leaf_q = np.empty(k)
    for t, (rotation, z, direction) in enumerate(zip(rotations, tensors, directions)):
        wedge = wedge_rotation(rotation, configs)
        probabilities_spin = np.abs(wedge[0]) ** 2
        # The HF determinant occupies orbitals 0..nalpha-1, i.e. config row 0.
        joint = np.outer(probabilities_spin, probabilities_spin)
        occ = occupations[:, None, :] + occupations[None, :, :]
        leaf = diagonal_leaf_values(occ, z, direction, 0.0)
        qvalue = np.einsum("...i,i->...", occ, direction)
        mean_leaf = float(np.sum(joint * leaf))
        mean_q = float(np.sum(joint * qvalue))
        var_leaves[t] = max(float(np.sum(joint * leaf**2)) - mean_leaf**2, 0.0)
        cov_leaf_q[t] = float(np.sum(joint * leaf * qvalue)) - mean_leaf * mean_q
    variances = np.r_[var0, var_leaves]
    allocations = np.sqrt(np.maximum(variances, 1.0e-16))
    allocations /= allocations.sum()
    alpha = np.zeros(k)
    for _ in range(6):
        normal = gram / allocations[0]
        rhs = -rhs0 / allocations[0]
        diagonal = np.diag_indices(k)
        normal[diagonal] += np.diag(gram) / allocations[1:]
        rhs += cov_leaf_q / allocations[1:]
        ridge = 1.0e-11 * max(1.0, float(np.trace(normal)) / k)
        normal[diagonal] += ridge
        try:
            alpha = la.solve(normal, rhs, assume_a="pos", check_finite=False)
        except la.LinAlgError:
            alpha = np.linalg.lstsq(normal, rhs, rcond=1.0e-12)[0]
        collector_var = var0 + 2.0 * alpha @ rhs0 + alpha @ gram @ alpha
        leaf_var = var_leaves - 2.0 * alpha * cov_leaf_q + alpha**2 * np.diag(gram)
        variances = np.r_[max(collector_var, 0.0), np.maximum(leaf_var, 0.0)]
        allocations = np.sqrt(np.maximum(variances, 1.0e-16))
        allocations /= allocations.sum()

    def spectral(values: np.ndarray) -> float:
        collector = one + np.einsum("t,tpq->pq", values, q_matrices)
        collector_range = float(np.sum(np.abs(np.linalg.eigvalsh(collector))))
        total = collector_range
        for value, z, direction in zip(values, tensors, directions):
            diagonal = diagonal_leaf_values(all_occupations, z, direction, float(value))
            total += 0.5 * float(np.max(diagonal) - np.min(diagonal))
        return total

    before = spectral(np.zeros(k))
    proxy = spectral(alpha)
    optimization = minimize(
        spectral,
        alpha,
        method="Powell",
        options={"maxiter": 3, "maxfev": maxfev, "xtol": 2.0e-3, "ftol": 2.0e-5},
    )
    candidates = [
        (before, np.zeros(k), "zero_safeguard"),
        (proxy, alpha.copy(), "hf_proxy"),
        (float(optimization.fun), np.asarray(optimization.x), "spectral_refinement"),
    ]
    final_value, final_alpha, selected = min(candidates, key=lambda item: item[0])
    if spectral(final_alpha) > before + 1.0e-10:
        final_alpha = np.zeros(k)
        final_value = before
        selected = "zero_safeguard_after_materialization"
    return final_alpha, {
        "selected": selected,
        "spectral_sum_before_R2": before,
        "spectral_sum_after_R2": final_value,
        "powell_evaluations": int(optimization.nfev),
    }


def measurement_settings(
    case: dict[str, Any],
    fit: dict[str, Any],
    configs,
    occupations,
    all_occ,
    ci,
    maxfev,
    collector_mode="absorb",
    collector_extra_leaves=0,
    collector_objective="hybrid",
    collector_proxy_state="hf",
    collector_reference_shots=3000,
    collector_proxy_cycles=6,
    collector_equality_tolerance=1.0e-10,
    collector_extra_angle_steps=400,
    precomputed_alpha_f3=None,
):
    if collector_mode not in {"exact", "absorb", "augment"}:
        raise ValueError(f"Unknown collector mode: {collector_mode}")
    if collector_mode != "augment" and collector_extra_leaves:
        raise ValueError("Extra collector leaves are only valid in augment mode")
    rotations = fit["rotations"]
    tensors = fit["tensors"]
    if precomputed_alpha_f3 is None:
        alpha, f3 = f3_r2_alpha(
            case["one"], rotations, tensors, configs, occupations, all_occ,
            case["nalpha"], maxfev,
        )
    else:
        alpha, f3 = precomputed_alpha_f3
    directions = np.asarray([z.sum(axis=1) - 0.5 * np.diag(z) for z in tensors])
    q_matrices = np.asarray([r @ np.diag(d) @ r.T for r, d in zip(rotations, directions)])
    collector = case["one"] + np.einsum("t,tpq->pq", alpha, q_matrices)
    occ_pair = occupations[:, None, :] + occupations[None, :, :]
    base_sector_values = []
    base_full_values = []
    for t, (rotation, z, direction, value) in enumerate(zip(rotations, tensors, directions, alpha)):
        diagonal = diagonal_leaf_values(occ_pair, z, direction, float(value))
        full_diagonal = diagonal_leaf_values(all_occ, z, direction, float(value))
        base_sector_values.append(diagonal)
        base_full_values.append(full_diagonal)
    base_sector_values = np.asarray(base_sector_values)
    base_full_values = np.asarray(base_full_values)

    if collector_mode == "exact":
        eigenvalues, collector_rotation = np.linalg.eigh(collector)
        collector_values = np.einsum("...i,i->...", occ_pair, eigenvalues)
        settings = [{
            "family": "one_body_collector",
            "rotation": collector_rotation,
            "values": collector_values,
            "lambda": float(np.sum(np.abs(eigenvalues))),
        }]
        for t, (rotation, diagonal, full_diagonal) in enumerate(
            zip(rotations, base_sector_values, base_full_values)
        ):
            settings.append({
                "family": "shallow_rcdf_leaf",
                "source_index": t,
                "rotation": rotation,
                "values": diagonal,
                "lambda": 0.5 * float(np.max(full_diagonal) - np.min(full_diagonal)),
            })
        collector_audit = {
            "status": "PASS",
            "mode": "exact",
            "objective": "exact_eigendecomposition",
            "proxy_state": "not_applicable",
            "reference_shots": collector_reference_shots,
            "source_rotation_count": int(len(rotations)),
            "extra_rotation_count": 0,
            "measurement_setting_count": int(len(settings)),
            "independent_collector_setting_present": True,
            "matrix_reconstruction_residual_frobenius": 0.0,
            "matrix_relative_reconstruction_residual": 0.0,
        }
        return settings, alpha, f3, collector_audit

    bank_rotations, eta, bank_base_sector, collector_audit = shallow_collector_redistribution(
        collector,
        rotations,
        base_sector_values,
        base_full_values,
        configs,
        occupations,
        all_occ,
        ci,
        fit["depth"],
        collector_extra_leaves,
        collector_objective,
        collector_proxy_state,
        collector_reference_shots,
        collector_proxy_cycles,
        collector_equality_tolerance,
        collector_extra_angle_steps,
        f"{case['spec'].name}|K{len(rotations)}|dR{fit['depth']}",
    )
    if collector_extra_leaves:
        bank_base_full = np.concatenate((
            base_full_values,
            np.zeros((collector_extra_leaves, base_full_values.shape[1])),
        ))
    else:
        bank_base_full = base_full_values
    settings = []
    for setting_index, (rotation, coefficients, sector_base, full_base) in enumerate(
        zip(bank_rotations, eta, bank_base_sector, bank_base_full)
    ):
        diagonal = sector_base + np.einsum("...i,i->...", occ_pair, coefficients)
        full_diagonal = full_base + all_occ @ coefficients
        settings.append({
            "family": (
                "shallow_rcdf_leaf"
                if setting_index < len(rotations)
                else "shallow_one_body_leaf"
            ),
            "source_index": setting_index if setting_index < len(rotations) else None,
            "rotation": rotation,
            "values": diagonal,
            "collector_linear_coefficients": coefficients,
            "lambda": 0.5 * float(np.max(full_diagonal) - np.min(full_diagonal)),
        })
    collector_audit.update({
        "measurement_setting_count": int(len(settings)),
        "independent_collector_setting_present": False,
    })
    return settings, alpha, f3, collector_audit


def state_expectation(terms, supports: list[tuple[int, complex]]) -> float:
    coefficients = dict(supports)
    value = 0.0 + 0.0j
    for xmask, zmask, coefficient, ny in terms:
        phase_y = 1j**ny
        for source, amplitude in supports:
            target = source ^ xmask
            target_amplitude = coefficients.get(target)
            if target_amplitude is None:
                continue
            matrix_element = phase_y * ((-1) ** ((zmask & source).bit_count()))
            value += np.conj(target_amplitude) * amplitude * coefficient * matrix_element
    if abs(value.imag) > 2.0e-8:
        raise ValueError(f"Complex stabilizer expectation {value}")
    return float(value.real)


def sector_state_expectation(terms, masks: np.ndarray, amplitudes: np.ndarray, n: int) -> float:
    """Vectorized Pauli expectation for one fixed-spin CI state."""
    masks_u = np.asarray(masks, dtype=np.uint64)
    values = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
    lookup = np.full(1 << n, -1, dtype=np.int32)
    lookup[masks_u.astype(np.int64)] = np.arange(len(masks_u), dtype=np.int32)
    parity = np.fromiter((index.bit_count() & 1 for index in range(1 << n)), dtype=np.int8)
    expectation = 0.0 + 0.0j
    for xmask, zmask, coefficient, ny in terms:
        targets = lookup[np.bitwise_xor(masks_u, np.uint64(xmask)).astype(np.int64)]
        valid = targets >= 0
        if not np.any(valid):
            continue
        signs = 1 - 2 * parity[np.bitwise_and(masks_u[valid], np.uint64(zmask)).astype(np.int64)]
        matrix_elements = (1j**ny) * signs
        expectation += coefficient * np.vdot(values[targets[valid]], values[valid] * matrix_elements)
    if abs(expectation.imag) > 2.0e-7:
        raise ValueError(f"Complex sector-state expectation {expectation}")
    return float(expectation.real)


def make_stabilizer_catalog(case, configs, determinant_sector_masks, count: int = 120):
    rng = np.random.default_rng(stable_seed(case["spec"].name, "stabilizer-catalog"))
    size = len(determinant_sector_masks)
    records = []
    determinant_count = min(40, size)
    chosen = np.unique(np.r_[0, rng.choice(size, size=determinant_count - 1, replace=False)])
    for index in chosen[:determinant_count]:
        mask = int(determinant_sector_masks[index])
        records.append({"group": f"det:{mask}", "supports": [(mask, 1.0 + 0.0j)]})
    pair_groups = max((count - len(records)) // 4, 1)
    used = set()
    while len(used) < pair_groups:
        a, b = sorted(rng.choice(size, size=2, replace=False).tolist())
        used.add((a, b))
    for a, b in sorted(used):
        ma, mb = int(determinant_sector_masks[a]), int(determinant_sector_masks[b])
        for phase_name, phase in (("+", 1.0), ("-", -1.0), ("+i", 1.0j), ("-i", -1.0j)):
            records.append({
                "group": f"pair:{ma}|{mb}",
                "phase": phase_name,
                "supports": [(ma, 1 / math.sqrt(2)), (mb, phase / math.sqrt(2))],
            })
    records = records[:count]
    group_split = {}
    for record in records:
        group = record["group"]
        if group not in group_split:
            bucket = stable_seed(case["spec"].name, group) % 10
            group_split[group] = "train" if bucket < 6 else "validation" if bucket < 8 else "test"
        record["split"] = group_split[group]
        index_supports = []
        for mask, amplitude in record["supports"]:
            alpha_mask = mask & ((1 << case["m"]) - 1)
            beta_mask = mask >> case["m"]
            ia = next(i for i, config in enumerate(configs) if sum(1 << q for q in config) == alpha_mask)
            ib = next(i for i, config in enumerate(configs) if sum(1 << q for q in config) == beta_mask)
            index_supports.append((ia, ib, amplitude))
        record["index_supports"] = index_supports
    return records


def setting_moments_for_state(settings, wedges, record):
    means = []
    variances = []
    for setting, wedge in zip(settings, wedges):
        amplitude = np.zeros_like(setting["values"], dtype=np.complex128)
        for ia, ib, coefficient in record["index_supports"]:
            amplitude += coefficient * np.outer(wedge[ia], wedge[ib])
        probabilities = np.abs(amplitude) ** 2
        probabilities /= probabilities.sum()
        values = setting["values"]
        mean = float(np.sum(probabilities * values))
        variance = max(float(np.sum(probabilities * values**2)) - mean**2, 0.0)
        means.append(mean)
        variances.append(variance)
    return np.asarray(means), np.asarray(variances)


def fit_weights(case, fits, catalogs, configs, budgets):
    rows = []
    for fit in fits:
        terms = fit["residual_terms"]
        lambdas = np.asarray([s["lambda"] for s in fit["settings"]])
        wedges = [wedge_rotation(setting["rotation"], configs) for setting in fit["settings"]]
        for state_index, record in enumerate(catalogs):
            residual_mean = state_expectation(terms, record["supports"])
            setting_means, setting_variances = setting_moments_for_state(
                fit["settings"], wedges, record
            )
            for total in budgets:
                shots = integer_allocation(total, lambdas)
                active = shots > 0
                x_sampling_sq = float(np.sum(lambdas[active] ** 2 / shots[active]))
                sampling_variance = float(np.sum(setting_variances[active] / shots[active]))
                rows.append({
                    "molecule": case["spec"].name,
                    "state_index": state_index,
                    "group": record["group"],
                    "split": record["split"],
                    "K": fit["K"],
                    "T": total,
                    "x_approximation": fit["sector_F"],
                    "absolute_approximation_error": abs(residual_mean),
                    "x_sampling_sq": x_sampling_sq,
                    "sampling_variance": sampling_variance,
                    "setting_mean_sum": float(case["constant"] + setting_means.sum()),
                    "collector_mode": fit["collector_mode"],
                    "collector_extra_leaves": fit["collector_extra_leaves"],
                })
    training = [row for row in rows if row["split"] == "train"]
    xa2 = np.asarray([row["x_approximation"] ** 2 for row in training])
    ya2 = np.asarray([row["absolute_approximation_error"] ** 2 for row in training])
    xs2 = np.asarray([row["x_sampling_sq"] for row in training])
    ys2 = np.asarray([row["sampling_variance"] for row in training])
    theta_a = max(float(np.dot(xa2, ya2) / np.dot(xa2, xa2)), 0.0)
    theta_s = max(float(np.dot(xs2, ys2) / np.dot(xs2, xs2)), 0.0)
    weights = {"weight_approximation": math.sqrt(theta_a), "weight_sampling": math.sqrt(theta_s)}
    for split in ("train", "validation", "test"):
        weights[f"{split}_rows"] = sum(row["split"] == split for row in rows)
    return weights, rows


def fit_srcdf(case, args, case_output, configs, occupations, all_occ, ci):
    fits = []
    trial_rows = []
    collector_rows = []
    for k in args.rank_grid:
        candidates = []
        for depth in (1, 2, 3):
            cache = case_output / "srcdf_fit_cache" / f"K{k:02d}_dR{depth}.npz"
            if cache.exists():
                with np.load(cache, allow_pickle=False) as saved:
                    rotations = saved["rotations"]
                    tensors = saved["tensors"]
                    fitted = saved["fitted_eri"]
                elapsed = 0.0
                cache_hit = True
            else:
                started = time.perf_counter()
                rotations, tensors, fitted, _ = optimize_shallow_rcdf(
                    case["eri"], k, depth, args.rho, args.cycles,
                    args.angle_steps, 1.0e-7, f"{case['spec'].name}|K{k}",
                )
                elapsed = time.perf_counter() - started
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, rotations=rotations, tensors=tensors, fitted_eri=fitted)
                cache_hit = False
            fitted_open = fitted.transpose(0, 2, 3, 1)
            terms = residual_pauli_terms(case["two_open"] - fitted_open)
            sector_f = sector_frobenius(terms, case["n"], case["spec"].electrons)
            full_f = full_frobenius(terms, case["n"])
            row = {
                "molecule": case["spec"].name,
                "K": k,
                "depth": depth,
                "sector_Frobenius_distance": sector_f,
                "full_Frobenius_distance": full_f,
                "eri_Frobenius_distance": float(np.linalg.norm(case["eri"] - fitted)),
                "elapsed_seconds": elapsed,
                "cache_hit": cache_hit,
                "selected_depth": False,
            }
            trial_rows.append(row)
            candidates.append((row, rotations, tensors, fitted, terms))
            print(f"[s-RCDF] {case['spec'].name} K={k} dR={depth} sector-F={sector_f:.6e}", flush=True)
        minimum = min(item[0]["sector_Frobenius_distance"] for item in candidates)
        selected = min(
            [item for item in candidates if item[0]["sector_Frobenius_distance"] <= minimum + 1.0e-12],
            key=lambda item: item[0]["depth"],
        )
        selected[0]["selected_depth"] = True
        row, rotations, tensors, fitted, terms = selected
        alpha_f3 = f3_r2_alpha(
            case["one"], rotations, tensors, configs, occupations, all_occ,
            case["nalpha"], args.spectral_maxfev,
        )
        extra_grid = (
            tuple(args.collector_extra_leaf_grid)
            if args.collector_mode == "augment"
            else (0,)
        )
        for extra_leaves in extra_grid:
            base_record = {
                "molecule": case["spec"].name,
                "K": k,
                "depth": row["depth"],
                "collector_mode": args.collector_mode,
                "collector_extra_leaves": extra_leaves,
                "collector_objective": args.collector_objective,
                "collector_proxy_state": args.collector_proxy_state,
                "collector_reference_shots": args.collector_reference_shots,
            }
            try:
                settings, alpha, f3, collector_audit = measurement_settings(
                    case,
                    {"rotations": rotations, "tensors": tensors, "depth": row["depth"]},
                    configs,
                    occupations,
                    all_occ,
                    ci,
                    args.spectral_maxfev,
                    collector_mode=args.collector_mode,
                    collector_extra_leaves=extra_leaves,
                    collector_objective=args.collector_objective,
                    collector_proxy_state=args.collector_proxy_state,
                    collector_reference_shots=args.collector_reference_shots,
                    collector_proxy_cycles=args.collector_proxy_cycles,
                    collector_equality_tolerance=args.collector_equality_tolerance,
                    collector_extra_angle_steps=args.collector_extra_angle_steps,
                    precomputed_alpha_f3=alpha_f3,
                )
            except CollectorRepresentationError as error:
                collector_rows.append({
                    **base_record,
                    "feasible": False,
                    **{
                        key: value
                        for key, value in error.audit.items()
                        if not isinstance(value, (dict, list))
                    },
                })
                print(
                    f"[collector] {case['spec'].name} K={k} dR={row['depth']} "
                    f"mode={args.collector_mode} L={extra_leaves} infeasible: {error}",
                    flush=True,
                )
                continue
            collector_rows.append({
                **base_record,
                "feasible": True,
                **{
                    key: value
                    for key, value in collector_audit.items()
                    if not isinstance(value, (dict, list))
                },
            })
            fits.append({
                "K": k,
                "depth": row["depth"],
                "sector_F": row["sector_Frobenius_distance"],
                "full_F": row["full_Frobenius_distance"],
                "rotations": rotations,
                "tensors": tensors,
                "fitted": fitted,
                "residual_terms": terms,
                "settings": settings,
                "alpha": alpha,
                "f3": f3,
                "collector_mode": args.collector_mode,
                "collector_extra_leaves": extra_leaves,
                "collector_audit": collector_audit,
            })
    if not fits:
        raise RuntimeError(
            f"{case['spec'].name}: no feasible s-RCDF fit for collector mode "
            f"{args.collector_mode}; extend the K/L grids or use exact mode"
        )
    return fits, trial_rows, collector_rows


def ground_srcdf_models(ci, configs, settings):
    models = []
    for setting in settings:
        wedge = wedge_rotation(setting["rotation"], configs)
        rotated = wedge.T @ ci @ wedge
        probabilities = np.abs(rotated) ** 2
        probabilities /= probabilities.sum()
        values = np.asarray(setting["values"])
        mean = float(np.sum(probabilities * values))
        variance = max(float(np.sum(probabilities * values**2)) - mean**2, 0.0)
        models.append((probabilities.reshape(-1), values.reshape(-1), mean, variance))
    return models


def sample_srcdf(case, fits, weights, ci, configs, sector_masks, budgets, repeats):
    summaries = []
    replicate_rows = []
    candidate_rows = []
    ground_cache = {}
    reconstruction_cache = {}
    for total in budgets:
        candidates = []
        for fit in fits:
            lambdas = np.asarray([setting["lambda"] for setting in fit["settings"]])
            shots = integer_allocation(total, lambdas)
            active = shots > 0
            xs = math.sqrt(float(np.sum(lambdas[active] ** 2 / shots[active])))
            loss = math.hypot(
                weights["weight_approximation"] * fit["sector_F"],
                weights["weight_sampling"] * xs,
            )
            record = {
                "molecule": case["spec"].name,
                "T_total_shots": total,
                "K": fit["K"],
                "depth": fit["depth"],
                "collector_mode": fit["collector_mode"],
                "collector_extra_leaves": fit["collector_extra_leaves"],
                "collector_objective": fit["collector_audit"]["objective"],
                "collector_proxy_state": fit["collector_audit"]["proxy_state"],
                "collector_reference_shots": fit["collector_audit"]["reference_shots"],
                "collector_dictionary_rank": fit["collector_audit"].get(
                    "dictionary_rank", "N/A"
                ),
                "collector_dictionary_nullity": fit["collector_audit"].get(
                    "dictionary_nullity", "N/A"
                ),
                "collector_proxy_SE_at_reference_shots": fit["collector_audit"].get(
                    "selected_proxy_estimator_SE_at_reference_shots", "N/A"
                ),
                "collector_centered_range_sum": fit["collector_audit"].get(
                    "selected_centered_range_sum", "N/A"
                ),
                "collector_matrix_relative_reconstruction_residual": fit[
                    "collector_audit"
                ]["matrix_relative_reconstruction_residual"],
                "measurement_settings": len(fit["settings"]),
                "sector_Frobenius_distance": fit["sector_F"],
                "x_sampling": xs,
                "predicted_total_RMSE": loss,
                "shot_vector": " ".join(map(str, shots.tolist())),
            }
            candidate_rows.append(record)
            candidates.append((loss, fit["K"], fit["collector_extra_leaves"], fit, shots))
        _, _, _, selected, shots = min(
            candidates, key=lambda value: (value[0], value[1], value[2])
        )
        if selected["K"] == max(args_global.rank_grid):
            raise RuntimeError(
                f"{case['spec'].name} T={total}: the selected s-RCDF rank lies at "
                "the audited continuation endpoint; refusing to publish a "
                "right-censored rank selection. Extend --rank-grid and rerun."
            )
        selected_key = (
            selected["K"],
            selected["depth"],
            selected["collector_mode"],
            selected["collector_extra_leaves"],
        )
        if selected_key not in ground_cache:
            ground_cache[selected_key] = ground_srcdf_models(ci, configs, selected["settings"])
        models = ground_cache[selected_key]
        approximate_mean = case["constant"] + sum(model[2] for model in models)
        if selected_key not in reconstruction_cache:
            residual_mean = sector_state_expectation(
                selected["residual_terms"], sector_masks, ci.reshape(-1), case["n"]
            )
            reconstruction_cache[selected_key] = abs(
                approximate_mean - (case["exact_energy"] - residual_mean)
            )
        if reconstruction_cache[selected_key] > 1.0e-6:
            raise RuntimeError(
                f"{case['spec'].name} K={selected['K']}: s-RCDF ground expectation "
                f"replay mismatch {reconstruction_cache[selected_key]:.3e}"
            )
        analytic_variance = sum(model[3] / count for model, count in zip(models, shots) if count)
        rng = np.random.default_rng(stable_seed(case["spec"].name, "s-RCDF", total))
        estimates = np.full(repeats, case["constant"], dtype=float)
        for model, count in zip(models, shots):
            if not count:
                continue
            probabilities, values, _, _ = model
            cdf = np.cumsum(probabilities)
            cdf[-1] = 1.0
            outcomes = np.searchsorted(cdf, rng.random((repeats, count)), side="right")
            estimates += values[outcomes].mean(axis=1)
        errors = estimates - case["exact_energy"]
        summary = summarize_errors(case, "s-RCDF", total, estimates, selected["K"], selected["depth"])
        summary.update({
            "deterministic_bias_hartree": approximate_mean - case["exact_energy"],
            "analytic_sampling_SE_hartree": math.sqrt(max(analytic_variance, 0.0)),
            "analytic_total_RMSE_hartree": math.hypot(approximate_mean - case["exact_energy"], math.sqrt(max(analytic_variance, 0.0))),
            "right_censored_at_Kmax": False,
            "measurement_settings": len(models),
            "active_measurement_settings": int(np.sum(shots > 0)),
            "collector_mode": selected["collector_mode"],
            "collector_extra_leaves": selected["collector_extra_leaves"],
            "collector_objective": selected["collector_audit"]["objective"],
            "collector_proxy_state": selected["collector_audit"]["proxy_state"],
            "collector_reference_shots": selected["collector_audit"]["reference_shots"],
            "collector_dictionary_rank": selected["collector_audit"].get(
                "dictionary_rank", "N/A"
            ),
            "collector_dictionary_nullity": selected["collector_audit"].get(
                "dictionary_nullity", "N/A"
            ),
            "collector_proxy_SE_at_reference_shots": selected["collector_audit"].get(
                "selected_proxy_estimator_SE_at_reference_shots", "N/A"
            ),
            "collector_centered_range_sum": selected["collector_audit"].get(
                "selected_centered_range_sum", "N/A"
            ),
            "collector_selected_candidate": selected["collector_audit"].get(
                "selected_candidate", "exact_eigendecomposition"
            ),
            "collector_matrix_reconstruction_residual_frobenius": selected[
                "collector_audit"
            ]["matrix_reconstruction_residual_frobenius"],
            "collector_matrix_relative_reconstruction_residual": selected[
                "collector_audit"
            ]["matrix_relative_reconstruction_residual"],
            "ground_expectation_reconstruction_error": reconstruction_cache[selected_key],
            "shot_vector": " ".join(map(str, shots.tolist())),
        })
        summaries.append(summary)
        for repeat, estimate in enumerate(estimates):
            replicate_rows.append(replicate_row(case, "s-RCDF", total, repeat, estimate))
    return summaries, replicate_rows, candidate_rows


def sorted_insertion_cover(observables: np.ndarray, weights: np.ndarray) -> np.ndarray:
    groups: list[np.ndarray] = []
    for index in np.argsort(-np.abs(weights), kind="stable"):
        term = observables[index]
        for setting in groups:
            if np.all((term == 0) | (setting == 0) | (term == setting)):
                mask = (setting == 0) & (term != 0)
                setting[mask] = term[mask]
                break
        else:
            groups.append(term.copy())
    result = np.asarray(groups, dtype=np.int8)
    result[result == 0] = 3
    hits = PAULI_IMPL.hit_matrix(observables, result)
    if np.any(hits.sum(axis=1) == 0):
        raise RuntimeError("Sorted-insertion cover missed a Pauli term")
    return result


def unique_settings(rows: Iterable[np.ndarray]) -> np.ndarray:
    seen = {}
    for row in rows:
        key = tuple(int(value) for value in row)
        seen.setdefault(key, np.asarray(row, dtype=np.int8))
    return np.asarray(list(seen.values()), dtype=np.int8)


def sparse_hit_matrix(observables: np.ndarray, settings: np.ndarray) -> sp.csc_matrix:
    row_indices = []
    column_indices = []
    for column, setting in enumerate(settings):
        hit = np.flatnonzero(PAULI_IMPL.setting_hits(observables, setting))
        row_indices.extend(hit.tolist())
        column_indices.extend([column] * len(hit))
    data = np.ones(len(row_indices), dtype=float)
    matrix = sp.csc_matrix((data, (row_indices, column_indices)), shape=(len(observables), len(settings)))
    if np.any(np.asarray(matrix.sum(axis=1)).reshape(-1) == 0):
        raise RuntimeError("OGM bank does not cover every Pauli term")
    return matrix


def optimize_ogm_sparse(weights: np.ndarray, coverage: sp.csc_matrix):
    weights_sq = weights**2
    initial = np.asarray(np.abs(weights) @ coverage).reshape(-1)
    initial = np.maximum(initial, 1.0e-12)
    initial /= initial.sum()

    def softmax(logits):
        shifted = logits - np.max(logits)
        value = np.exp(shifted)
        return value / value.sum()

    def value_gradient(logits):
        p = softmax(logits)
        q = np.asarray(coverage @ p).reshape(-1)
        q = np.maximum(q, 1.0e-14)
        value = float(np.sum(weights_sq / q))
        gp = -np.asarray(coverage.T @ (weights_sq / q**2)).reshape(-1)
        gradient = p * (gp - float(gp @ p))
        return value, gradient

    result = minimize(
        value_gradient, np.log(initial), jac=True, method="L-BFGS-B",
        options={
            "maxiter": OGM_ITERATION_SAFETY_GUARD,
            "maxfun": 500_000,
            "ftol": 1.0e-13,
            "gtol": 1.0e-8,
            "maxls": 100,
        },
    )
    probabilities = softmax(np.asarray(result.x))
    audit = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "iteration_safety_guard": OGM_ITERATION_SAFETY_GUARD,
        "right_censored_at_iteration_guard": bool(
            not result.success and int(result.nit) >= OGM_ITERATION_SAFETY_GUARD
        ),
        "terminal_gradient_infinity_norm": float(
            np.linalg.norm(np.asarray(result.jac), ord=np.inf)
        ),
        "objective": float(result.fun),
        "candidate_count": int(coverage.shape[1]),
        "nonzero_probability_count": int(np.count_nonzero(probabilities > 1.0e-12)),
    }
    if not result.success:
        raise RuntimeError(
            "OGM optimization did not converge; refusing to publish a solver-censored "
            f"schedule: {audit}"
        )
    return probabilities, audit


def largest_remainder_counts(total: int, probabilities: np.ndarray) -> np.ndarray:
    quotas = total * probabilities
    result = np.floor(quotas).astype(int)
    missing = total - int(result.sum())
    if missing:
        result[np.argsort(-(quotas - result), kind="stable")[:missing]] += 1
    return result


def pauli_method_counts(case, budgets):
    observables, weights = case["observables"], case["weights"]
    maximum = max(budgets)
    core = sorted_insertion_cover(observables, weights)
    if min(budgets) < len(core):
        raise ValueError(f"{case['spec'].name}: min T must be at least QWC cover size {len(core)}")
    sg_raw = PAULI_IMPL.vectorized_shadow_grouping_schedule(observables, weights, maximum)
    core_hits = PAULI_IMPL.hit_matrix(observables, core).sum(axis=1).astype(np.int64)
    der_raw, derand_generator_audit = PAULI_IMPL.vectorized_derandomization_schedule(
        observables,
        weights,
        maximum,
        initial_hits=core_hits,
        return_audit=True,
    )
    sg_raw[sg_raw == 0] = 3
    der_raw[der_raw == 0] = 3
    bank = unique_settings(itertools.chain(core, sg_raw, der_raw))
    coverage = sparse_hit_matrix(observables, bank)
    probabilities, ogm_audit = optimize_ogm_sparse(weights, coverage)
    index = {tuple(map(int, row)): i for i, row in enumerate(bank)}
    core_indices = np.asarray([index[tuple(map(int, row))] for row in core], dtype=int)
    results: dict[str, dict[int, Counter]] = {"SG": {}, "Derand": {}, "OGM": {}}
    for total in budgets:
        remainder = total - len(core)
        for method, raw in (("SG", sg_raw), ("Derand", der_raw)):
            counter = Counter(tuple(map(int, row)) for row in core)
            counter.update(tuple(map(int, row)) for row in raw[:remainder])
            results[method][total] = counter
        counts = largest_remainder_counts(remainder, probabilities)
        counts[core_indices] += 1
        results["OGM"][total] = Counter({tuple(map(int, bank[i])): int(c) for i, c in enumerate(counts) if c})
    for method in results:
        for total, counter in results[method].items():
            if sum(counter.values()) != total:
                raise RuntimeError(f"{method} T={total}: shot conservation failure")
            settings = np.asarray(list(counter), dtype=np.int8)
            counts = np.asarray(list(counter.values()), dtype=int)
            hits = PAULI_IMPL.hit_matrix(observables, settings) @ counts
            if np.any(hits == 0):
                raise RuntimeError(f"{method} T={total}: unhit Pauli term")
    return results, {
        "common_QWC_cover_size": len(core),
        "SG_raw_unique_at_max_T": len(np.unique(sg_raw, axis=0)),
        "Derand_raw_unique_at_max_T": len(np.unique(der_raw, axis=0)),
        "Derand_algorithm_id": DERAND_ALGORITHM_ID,
        "Derand_implementation_sha256": sha256_file(HUANG_DERAND_IMPLEMENTATION),
        "Derand_cover_seed_min_hits": int(np.min(core_hits)),
        "Derand_cover_seed_max_hits": int(np.max(core_hits)),
        "Derand_generator_audit": derand_generator_audit.as_dict(),
        "OGM_candidate_bank_size": len(bank),
        "OGM_optimization": ogm_audit,
    }


def sample_pauli_method(case, state_mps, method, counts_by_t, repeats):
    observables, weights = case["observables"], case["weights"]
    budgets = sorted(counts_by_t)
    all_settings = sorted(set().union(*(set(counter) for counter in counts_by_t.values())))
    running = {total: np.zeros((repeats, len(weights)), dtype=float) for total in budgets}
    hit_counts = {total: np.zeros(len(weights), dtype=int) for total in budgets}
    for setting_number, setting_tuple in enumerate(all_settings, 1):
        max_count = max(counts_by_t[total].get(setting_tuple, 0) for total in budgets)
        transformed = state_mps.copy()
        for site, basis in enumerate(setting_tuple):
            if int(basis) in (1, 2):
                transformed.gate_(BASIS_GATES[int(basis)], site, contract=True)
        sample_count = repeats * max_count
        sample_seed = stable_seed(case["spec"].name, method, setting_tuple)
        if sample_count >= 1000:
            # Sequential MPS sampling is preferable for the many one-shot
            # bases in SG/OGM.  Derandomization can repeat one basis thousands
            # of times; there it is much faster to contract this one rotated
            # MPS once and draw all outcomes from its exact dense probability
            # vector (at most 2^20 entries in the present benchmark).
            dense = np.asarray(transformed.to_dense(), dtype=np.complex128).reshape(-1)
            probabilities = np.abs(dense) ** 2
            probabilities /= probabilities.sum()
            cdf = np.cumsum(probabilities)
            cdf[-1] = 1.0
            rng = np.random.default_rng(sample_seed)
            outcomes = np.searchsorted(cdf, rng.random(sample_count), side="right")
            shifts = np.arange(case["n"] - 1, -1, -1, dtype=np.int64)
            bits = ((outcomes[:, None] >> shifts[None, :]) & 1).astype(np.int8)
        else:
            samples = list(transformed.sample(sample_count, seed=sample_seed))
            bits = np.asarray([sample[0] for sample in samples], dtype=np.int8)
        setting = np.asarray(setting_tuple, dtype=np.int8)
        hit = np.flatnonzero(PAULI_IMPL.setting_hits(observables, setting))
        supports = (observables[hit] != 0).astype(np.int16)
        signs = (1 - 2 * ((bits.astype(np.int16) @ supports.T) & 1)).reshape(repeats, max_count, len(hit))
        for total in budgets:
            count = counts_by_t[total].get(setting_tuple, 0)
            if count:
                running[total][:, hit] += signs[:, :count, :].sum(axis=1)
                hit_counts[total][hit] += count
        if setting_number % 200 == 0 or setting_number == len(all_settings):
            print(f"[Born] {case['spec'].name} {method}: {setting_number}/{len(all_settings)} bases", flush=True)
    summaries = []
    replicate_rows = []
    for total in budgets:
        if np.any(hit_counts[total] == 0):
            raise RuntimeError(f"{method} T={total}: sampling replay found unhit term")
        estimates = case["offset"] + (running[total] / hit_counts[total]) @ weights
        summaries.append(summarize_errors(case, method, total, estimates, None, None))
        summaries[-1].update({
            "deterministic_bias_hartree": 0.0,
            "right_censored_at_Kmax": False,
            "measurement_settings": len(counts_by_t[total]),
            "covered_pauli_terms": len(weights),
            "minimum_term_hits": int(hit_counts[total].min()),
        })
        for repeat, estimate in enumerate(estimates):
            replicate_rows.append(replicate_row(case, method, total, repeat, estimate))
    return summaries, replicate_rows


def bootstrap_rmse(errors: np.ndarray, seed: int, resamples: int = 2000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(errors), size=(resamples, len(errors)))
    values = np.sqrt(np.mean(errors[indices] ** 2, axis=1))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize_errors(case, method, total, estimates, selected_k, depth):
    errors = np.asarray(estimates) - case["exact_energy"]
    lower, upper = bootstrap_rmse(errors, stable_seed(case["spec"].name, method, total, "bootstrap"))
    return {
        "molecule": case["spec"].name,
        "method": method,
        "T_total_shots": total,
        "T_actual_shots": total,
        "selected_K": selected_k if selected_k is not None else "N/A",
        "selected_depth": depth if depth is not None else "N/A",
        "exact_ground_energy_hartree": case["exact_energy"],
        "empirical_mean_energy_hartree": float(np.mean(estimates)),
        "empirical_bias_hartree": float(np.mean(errors)),
        "empirical_MAE_hartree": float(np.mean(np.abs(errors))),
        "empirical_total_RMSE_hartree": float(np.sqrt(np.mean(errors**2))),
        "bootstrap_RMSE_95_lower_hartree": lower,
        "bootstrap_RMSE_95_upper_hartree": upper,
        "repeat_count": len(estimates),
    }


def replicate_row(case, method, total, repeat, estimate):
    return {
        "molecule": case["spec"].name,
        "method": method,
        "T_total_shots": total,
        "repeat": repeat,
        "energy_estimate_hartree": float(estimate),
        "signed_total_error_hartree": float(estimate - case["exact_energy"]),
    }


def plot_comparison(rows, output: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    molecules = tuple(dict.fromkeys(row["molecule"] for row in rows))
    fig, axes_grid = plt.subplots(1, len(molecules), figsize=(5.7 * len(molecules), 4.7), sharey=True, squeeze=False)
    axes = axes_grid[0]
    for axis, molecule in zip(axes, molecules):
        selected = [row for row in rows if row["molecule"] == molecule]
        for method in METHODS:
            data = sorted([row for row in selected if row["method"] == method], key=lambda row: row["T_total_shots"])
            x = np.asarray([row["T_total_shots"] for row in data])
            y = np.asarray([row["empirical_total_RMSE_hartree"] for row in data])
            lower = np.asarray([row["bootstrap_RMSE_95_lower_hartree"] for row in data])
            upper = np.asarray([row["bootstrap_RMSE_95_upper_hartree"] for row in data])
            axis.plot(x, y, color=COLORS[method], marker=MARKERS[method], linewidth=1.8, markersize=5.2, label=method)
            axis.fill_between(x, lower, upper, color=COLORS[method], alpha=0.12, linewidth=0)
            if method == "s-RCDF":
                analytic = np.asarray([row["analytic_total_RMSE_hartree"] for row in data])
                axis.plot(x, analytic, color=COLORS[method], linestyle="--", linewidth=1.15, alpha=0.9)
                for xx, yy, row in zip(x, y, data):
                    suffix = "+" if row.get("right_censored_at_Kmax") else ""
                    label = f"K={row['selected_K']}{suffix}"
                    if row.get("collector_mode") == "augment":
                        label += f", L={row.get('collector_extra_leaves', 0)}"
                    axis.annotate(label, (xx, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=COLORS[method])
        axis.axhline(1.6e-3, color="0.35", linestyle=":", linewidth=1.0, label="1.6 mHa" if molecule == "H2O" else None)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(which="both", alpha=0.18, linewidth=0.6)
        axis.set_xlabel("Total state preparations, $T$")
        title = r"H$_2$O, $R_{\mathrm{OH}}=0.80$ Å" if molecule == "H2O" else r"N$_2$, $R_{\mathrm{NN}}=2.25$ Å"
        axis.set_title(title, fontsize=11)
    axes[0].set_ylabel("Total energy RMSE (Ha)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Same Hamiltonian, exact FCI state, and coverage-enforced fixed-shot budgets", y=1.09, fontsize=11)
    fig.tight_layout()
    png = output / ("_".join(molecules) + "_sRCDF_vs_Pauli_RMSE.png")
    pdf = output / ("_".join(molecules) + "_sRCDF_vs_Pauli_RMSE.pdf")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--molecules", nargs="+", choices=tuple(CASES), default=tuple(CASES))
    parser.add_argument("--shots", nargs="+", type=int, default=(1300, 1600, 2000, 2400, 3000))
    parser.add_argument(
        "--rank-grid", nargs="+", type=int,
        default=(1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 45, 50),
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rho", type=float, default=1.0e-6)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--angle-steps", type=int, default=20)
    parser.add_argument("--spectral-maxfev", type=int, default=40)
    parser.add_argument(
        "--collector-mode",
        choices=("exact", "absorb", "augment"),
        default="augment",
        help=(
            "exact keeps the independently diagonalized one-body collector; "
            "absorb expands it exactly in the K existing shallow rotations; "
            "augment permits additional depth-d_R shallow one-body rotations"
        ),
    )
    parser.add_argument(
        "--collector-extra-leaf-grid",
        nargs="+",
        type=int,
        default=(1,),
        help=(
            "L values compared in augment mode.  The formal default fixes L=1; "
            "pass 1 2 4 8 explicitly for the diagnostic frontier only."
        ),
    )
    parser.add_argument(
        "--collector-objective",
        choices=("greedy", "variance", "range", "hybrid"),
        default="greedy",
        help=(
            "Exact shallow collector coefficient policy.  greedy preserves the "
            "collector-tailored seed; the other modes explore its affine null space."
        ),
    )
    parser.add_argument(
        "--collector-proxy-state",
        choices=("hf", "ground"),
        default="hf",
        help=(
            "State used only for redistribution proxy covariances.  ground is "
            "the exact archived benchmark state; hf is a deployable state proxy."
        ),
    )
    parser.add_argument(
        "--collector-reference-shots",
        type=int,
        default=3000,
        help="Reference T used by the variance/range redistribution objective.",
    )
    parser.add_argument("--collector-proxy-cycles", type=int, default=6)
    parser.add_argument(
        "--collector-extra-angle-steps",
        type=int,
        default=400,
        help="Converged L-BFGS-B cap for each collector-tailored shallow rotation.",
    )
    parser.add_argument("--collector-equality-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--resume", action="store_true", help="Reuse rank caches in a partial output directory")
    parser.add_argument(
        "--reuse-pauli-from", type=Path,
        help="Reuse audited OGM/SG/Derand rows from this same-target result root",
    )
    args = parser.parse_args()
    if any(value < 1 for value in args.collector_extra_leaf_grid):
        parser.error("--collector-extra-leaf-grid values must be positive")
    if (
        args.collector_reference_shots < 1
        or args.collector_proxy_cycles < 1
        or args.collector_extra_angle_steps < 1
    ):
        parser.error("collector reference shots, proxy cycles, and angle steps must be positive")
    if args.collector_equality_tolerance <= 0.0:
        parser.error("collector equality tolerance must be positive")
    return args


args_global = None


def main():
    global args_global
    args = parse_args()
    args_global = args
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_summaries = []
    all_replicates = []
    all_candidates = []
    all_trials = []
    all_collector_trials = []
    all_calibration = []
    case_audits = []
    for name in args.molecules:
        case = load_case(CASES[name])
        case_output = output / name
        case_output.mkdir(parents=True, exist_ok=True)
        ci, configs, sector_masks = ci_state(case)
        occupations = spatial_occupations(configs, case["m"])
        all_occ = all_spatial_occupations(case["m"])
        fits, trials, collector_trials = fit_srcdf(
            case, args, case_output, configs, occupations, all_occ, ci
        )
        catalog = make_stabilizer_catalog(case, configs, sector_masks)
        weights, calibration = fit_weights(case, fits, catalog, configs, args.shots)
        write_json(case_output / "stabilizer_weights.json", weights)
        srcdf_summaries, srcdf_replicates, candidate_rows = sample_srcdf(
            case, fits, weights, ci, configs, sector_masks, args.shots, args.repeats
        )
        source_hashes = {key: sha256_file(path) for key, path in case["paths"].items()}
        pauli_reuse = None
        if args.reuse_pauli_from is not None:
            pauli_summaries, pauli_replicates, pauli_reuse = load_reused_pauli_rows(
                args.reuse_pauli_from, name, list(args.shots), args.repeats, source_hashes
            )
            schedule_audit = pauli_reuse["schedule"]
            print(f"[Pauli reuse] {name} <- {args.reuse_pauli_from.resolve()}", flush=True)
        else:
            method_counts, schedule_audit = pauli_method_counts(case, args.shots)
            state_mps = load_mps(case["paths"]["state"])
            pauli_summaries = []
            pauli_replicates = []
            for method in ("OGM", "SG", "Derand"):
                summaries, replicates = sample_pauli_method(
                    case, state_mps, method, method_counts[method], args.repeats
                )
                pauli_summaries.extend(summaries)
                pauli_replicates.extend(replicates)
        case_summaries = srcdf_summaries + pauli_summaries
        case_replicates = srcdf_replicates + pauli_replicates
        write_csv(case_output / "sampling_summary.csv", case_summaries)
        write_csv(case_output / "sampling_replicates.csv", case_replicates)
        write_csv(case_output / "srcdf_rank_depth_trials.csv", trials)
        write_csv(case_output / "collector_redistribution_trials.csv", collector_trials)
        write_csv(case_output / "srcdf_candidate_by_T_K.csv", candidate_rows)
        write_csv(case_output / "stabilizer_calibration_labels.csv", calibration)
        audit = {
            "status": "PASS",
            "molecule": name,
            "number_qubits": case["n"],
            "number_electrons": case["spec"].electrons,
            "exact_energy_hartree": case["exact_energy"],
            "ground_state_MPS_max_bond": case["ground_state_max_bond"],
            "ground_fixed_spin_sector_leakage": case["ground_sector_leakage"],
            "nonzero_nonidentity_Pauli_terms": len(case["weights"]),
            "discarded_numerically_zero_Pauli_rows": case["discarded_zero_pauli_rows"],
            "Hermitian_projection_discarded_imaginary_Pauli_max": case["discarded_antihermitian_pauli_imaginary_max"],
            "Hermitian_projection_discarded_imaginary_Pauli_l2": case["discarded_antihermitian_pauli_imaginary_l2"],
            "one_body_Hermitian_projection_frobenius": case["one_body_Hermitian_projection_frobenius"],
            "two_body_eightfold_projection_frobenius": case["two_body_eightfold_projection_frobenius"],
            "Pauli_all_terms_hit_at_every_T": True,
            "all_methods_use_exactly_T_state_preparations": True,
            "sRCDF_ground_state_not_used_to_fit_weights_or_select_K": (
                args.collector_proxy_state != "ground" or args.collector_mode == "exact"
            ),
            "sRCDF_ground_state_used_to_optimize_collector": (
                args.collector_proxy_state == "ground" and args.collector_mode != "exact"
            ),
            "sRCDF_ground_state_used_only_for_post_selection_sampling": (
                args.collector_proxy_state != "ground" or args.collector_mode == "exact"
            ),
            "collector_configuration": {
                "mode": args.collector_mode,
                "extra_leaf_grid": (
                    list(args.collector_extra_leaf_grid)
                    if args.collector_mode == "augment"
                    else [0]
                ),
                "objective": args.collector_objective,
                "proxy_state": args.collector_proxy_state,
                "reference_shots": args.collector_reference_shots,
                "proxy_cycles": args.collector_proxy_cycles,
                "extra_angle_steps": args.collector_extra_angle_steps,
                "equality_tolerance": args.collector_equality_tolerance,
                "f3_spectral_maxfev": args.spectral_maxfev,
            },
            "all_feasible_collector_matrix_reconstructions_pass": all(
                (not row["feasible"])
                or float(row.get("matrix_relative_reconstruction_residual", 0.0))
                <= 5.0 * args.collector_equality_tolerance
                for row in collector_trials
            ),
            "schedule": schedule_audit,
            "Pauli_rows_reused": pauli_reuse is not None,
            "Pauli_reuse_provenance": None if pauli_reuse is None else pauli_reuse["provenance"],
            "weights": weights,
            "input_sha256": source_hashes,
        }
        write_json(case_output / "audit.json", audit)
        all_summaries.extend(case_summaries)
        all_replicates.extend(case_replicates)
        all_candidates.extend(candidate_rows)
        all_trials.extend(trials)
        all_collector_trials.extend(collector_trials)
        all_calibration.extend(calibration)
        case_audits.append(audit)
    write_csv(output / "sampling_summary_all.csv", all_summaries)
    write_csv(output / "sampling_replicates_all.csv", all_replicates)
    write_csv(output / "srcdf_candidate_by_T_K_all.csv", all_candidates)
    write_csv(output / "srcdf_rank_depth_trials_all.csv", all_trials)
    write_csv(output / "collector_redistribution_trials_all.csv", all_collector_trials)
    write_csv(output / "stabilizer_calibration_labels_all.csv", all_calibration)
    png, pdf = plot_comparison(all_summaries, output)
    manifest = {
        "version": VERSION,
        "description_source": str((HERE / "README.md").resolve()),
        "description_source_sha256": sha256_file(HERE / "README.md"),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__)),
        "shallow_rcdf_implementation": str((SRDD_METHODS / "shallow_rcdf.py").resolve()),
        "shallow_rcdf_implementation_sha256": sha256_file(SRDD_METHODS / "shallow_rcdf.py"),
        "shallow_rcdf_version": SHALLOW_VERSION,
        "collector_redistribution_implementation": str(
            (SRDD_METHODS / "shallow_collector_redistribution.py").resolve()
        ),
        "collector_redistribution_implementation_sha256": sha256_file(
            SRDD_METHODS / "shallow_collector_redistribution.py"
        ),
        "collector_redistribution_version": collector_core.VERSION,
        "pauli_schedule_implementation": str(LEGACY_PAULI_DRIVER.resolve()),
        "pauli_schedule_implementation_sha256": sha256_file(LEGACY_PAULI_DRIVER),
        "molecules": list(args.molecules),
        "shot_budgets": list(args.shots),
        "sampling_repeats": args.repeats,
        "sRCDF_rank_grid": list(args.rank_grid),
        "sRCDF_depth_grid": [1, 2, 3],
        "sRCDF_optimizer": {
            "rho": args.rho,
            "cycles": args.cycles,
            "angle_steps": args.angle_steps,
            "f3_spectral_maxfev": args.spectral_maxfev,
        },
        "collector_redistribution": {
            "mode": args.collector_mode,
            "extra_leaf_grid": (
                list(args.collector_extra_leaf_grid)
                if args.collector_mode == "augment"
                else [0]
            ),
            "extra_rotation_construction": (
                "collector-tailored greedy depth-d_R nearest-neighbor Givens "
                "diagonalization followed by exact fill in the full shallow bank"
            ),
            "seed_namespace": collector_core.VERSION,
            "equality_constraint": "A eta = svec(C) with fail-closed residual audit",
            "objective": args.collector_objective,
            "proxy_state": args.collector_proxy_state,
            "reference_shots": args.collector_reference_shots,
            "proxy_cycles": args.collector_proxy_cycles,
            "extra_angle_steps": args.collector_extra_angle_steps,
            "equality_tolerance": args.collector_equality_tolerance,
            "constant_policy": "molecular constant is a deterministic classical offset",
        },
        "OGM_optimizer": {
            "method": "L-BFGS-B in softmax logits with analytic gradient",
            "iteration_safety_guard": OGM_ITERATION_SAFETY_GUARD,
            "guard_policy": "abort rather than publish if convergence is not reported",
        },
        "Pauli_policy": "common one-shot sorted-insertion QWC cover plus method-specific remaining shots; pooled term means; exact MPS Born replay",
        "Pauli_reuse_root": None if args.reuse_pauli_from is None else str(args.reuse_pauli_from.resolve()),
        "Pauli_reuse_manifest_sha256": None if args.reuse_pauli_from is None else sha256_file(args.reuse_pauli_from.resolve() / "manifest.json"),
        "error_metric": "RMSE of sampled energy relative to exact archived FCI ground energy",
        "elapsed_seconds": time.time() - started,
        "all_case_audits_pass": all(item["status"] == "PASS" for item in case_audits),
    }
    write_json(output / "manifest.json", manifest)
    output_files = [path for path in output.rglob("*") if path.is_file() and path.name != "output_hashes.json"]
    write_json(output / "output_hashes.json", {str(path.relative_to(output)): sha256_file(path) for path in output_files})
    print(f"[complete] {output} elapsed={time.time()-started:.1f}s", flush=True)
    print(f"[figure] {png}", flush=True)
    print(f"[figure] {pdf}", flush=True)


if __name__ == "__main__":
    main()
