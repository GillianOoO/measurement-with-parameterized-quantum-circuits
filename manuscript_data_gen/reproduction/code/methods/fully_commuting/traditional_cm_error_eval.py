#!/usr/bin/env python3
"""Reproducible finite-shot error audit for Nature-2023 FC-IMA measurement.

The public FC baseline follows Eqs. (12)--(22) of
Yen--Ganeshram--Izmaylov, npj Quantum Information 9, 14 (2023): an extended
sorted-insertion (SI) collection of overlapping fully commuting groups,
Eq. (14) pooling of repeated Pauli estimates, and iterative measurement
allocation (IMA).  The initial IMA proportions are the exact-state optimal
proportions for ordinary non-overlapping SI.  Ten IMA cycles are evaluated
and the lowest exact pooled-estimator variance is selected.

For an executable integer budget every FC-IMA setting receives one shot and
the remaining shots are rounded by deterministic largest remainder.  The
reported analytic error is then recomputed from Eq. (17) with those integer
counts.  Empirical errors are generated from genuine joint Born outcomes of
each commuting group and are pooled Pauli-by-Pauli exactly as in Eq. (14),
not from a Gaussian replay.

This program is intentionally isolated from manuscript/figure sources.  It
only writes below ``outputs/traditional_cm_benchmarks/error_eval/results``.
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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


VERSION = "full-commuting-fc-ima-nature-2023-v3"
COEFFICIENT_TOLERANCE = 1.0e-14
RANGE_TOLERANCE = 1.0e-14
AMPLITUDE_TOLERANCE = 1.0e-12
DEFAULT_BUDGETS = (1300, 1600, 2000, 2400, 3000)
DEFAULT_REPEATS = 50
DEFAULT_IMA_CYCLES = 10
IMA_VARIANCE_FLOOR = 1.0e-8
BORN_PROBABILITY_TOLERANCE = 1.0e-18

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pauli_product"))
from release_paths import BUNDLE_ROOT, DATA_ROOT, OUTPUT_ROOT as RELEASE_OUTPUT
WORKSPACE = BUNDLE_ROOT
SELECTED_ROOT = DATA_ROOT
OUTPUT_ROOT = RELEASE_OUTPUT / "fully_commuting" / "error_eval"
RESULTS_ROOT = OUTPUT_ROOT / "results"


@dataclass(frozen=True)
class CaseSpec:
    molecule: str
    directory: str
    srcdf_csv: Path


CASES = (
    CaseSpec(
        "BeH2",
        "BeH2/inputs",
        DATA_ROOT / "BeH2" / "results" / "srdd_and_pauli" / "BeH2" / "sampling_summary.csv",
    ),
    CaseSpec(
        "N2",
        "N2/inputs",
        DATA_ROOT / "N2" / "results" / "srdd_and_pauli" / "sampling_summary.csv",
    ),
)


@dataclass(frozen=True)
class PauliTerm:
    source_row: int
    label: str
    coefficient: float
    xmask: int
    zmask: int
    y_count: int


@dataclass(frozen=True)
class SignedPauli:
    """A Hermitian operator ``sign * i^y X^x Z^z``."""

    sign: int
    xmask: int
    zmask: int

    @property
    def y_count(self) -> int:
        return (self.xmask & self.zmask).bit_count()


@dataclass(frozen=True)
class BornDistribution:
    """Exact nonzero computational outcomes after an FC diagonalizer."""

    basis_indices: np.ndarray
    probabilities: np.ndarray
    cumulative_probabilities: np.ndarray
    term_z_masks: np.ndarray
    term_signs: np.ndarray

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


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def label_to_masks(label: str) -> tuple[int, int, int]:
    """Map site0-to-siteN-1 label to little-endian integer masks.

    Dense MPS flattening uses site 0 as the most-significant tensor-product
    index, hence site q maps to integer bit n - 1 - q.
    """
    n = len(label)
    xmask = 0
    zmask = 0
    y_count = 0
    for site, letter in enumerate(label):
        bit = 1 << (n - 1 - site)
        if letter in ("X", "Y"):
            xmask |= bit
        if letter in ("Z", "Y"):
            zmask |= bit
        if letter == "Y":
            y_count += 1
        if letter not in "IXYZ":
            raise ValueError(f"Invalid Pauli letter {letter!r} in {label!r}")
    return xmask, zmask, y_count


def pack_symplectic(term: PauliTerm, n: int) -> int:
    return term.xmask | (term.zmask << n)


def fully_commutes(left: PauliTerm, right: PauliTerm) -> bool:
    parity = (
        (left.xmask & right.zmask).bit_count()
        + (left.zmask & right.xmask).bit_count()
    ) & 1
    return parity == 0


def load_pauli_terms(path: Path) -> tuple[list[PauliTerm], float, dict[str, Any]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty Pauli CSV: {path}")
    label_key = "full_label_site0_to_siteNminus1"
    identity = 0.0
    terms: list[PauliTerm] = []
    imaginary_values: list[float] = []
    discarded = 0
    n = len(rows[0][label_key])
    for source_row, row in enumerate(rows, start=2):
        label = row[label_key].strip()
        if len(label) != n:
            raise ValueError(f"Inconsistent Pauli length at row {source_row}")
        real = float(row["coefficient_real_hartree"])
        imag = float(row["coefficient_imag_hartree"])
        imaginary_values.append(imag)
        if set(label) <= {"I"}:
            identity += real
            continue
        if abs(real) <= COEFFICIENT_TOLERANCE:
            discarded += 1
            continue
        xmask, zmask, y_count = label_to_masks(label)
        terms.append(PauliTerm(source_row, label, real, xmask, zmask, y_count))
    audit = {
        "csv_rows": len(rows),
        "number_qubits": n,
        "retained_nonidentity_terms": len(terms),
        "discarded_nonidentity_real_below_tolerance": discarded,
        "coefficient_tolerance_hartree": COEFFICIENT_TOLERANCE,
        "identity_offset_hartree": identity,
        "discarded_imaginary_coefficient_max_abs_hartree": float(
            np.max(np.abs(imaginary_values))
        ),
        "discarded_imaginary_coefficient_l2_hartree": float(
            np.linalg.norm(imaginary_values)
        ),
        "hermitian_projection": "retain real coefficients, matching existing s-RCDF input loader",
    }
    return terms, identity, audit


def sorted_insertion_fc(terms: Sequence[PauliTerm]) -> list[list[PauliTerm]]:
    """Coefficient-sorted non-overlapping SI used only to initialize IMA."""
    ordered = sorted(terms, key=lambda item: (-abs(item.coefficient), item.source_row))
    groups: list[list[PauliTerm]] = []
    for term in ordered:
        for group in groups:
            if all(fully_commutes(term, other) for other in group):
                group.append(term)
                break
        else:
            groups.append([term])
    if sum(map(len, groups)) != len(terms):
        raise AssertionError("Disjoint grouping lost or duplicated terms")
    if len({term.source_row for group in groups for term in group}) != len(terms):
        raise AssertionError("A Pauli CSV row appears in more than one group")
    for index, group in enumerate(groups):
        for i, left in enumerate(group):
            if not all(fully_commutes(left, right) for right in group[i + 1 :]):
                raise AssertionError(f"Group {index} is not fully commuting")
    return groups


def extended_sorted_insertion_fc(
    terms: Sequence[PauliTerm],
) -> tuple[list[list[PauliTerm]], list[set[int]]]:
    """Nature-2023 extended SI with overlap (``order='pi'`` semantics).

    This is a dependency-free, row-stable mirror of
    ``ToBW/utils/grouping_utils.py::greedy_grouping_with_overlap``.  Each
    iteration first builds the next maximal SI group from unassigned terms,
    then inserts every previously assigned term that commutes with the whole
    new group.  ``seed_rows[g]`` records the newly assigned (ordinary-SI)
    members of group ``g`` for provenance.
    """
    remaining = sorted(
        terms, key=lambda item: (-abs(item.coefficient), item.source_row)
    )
    previously_grouped: list[PauliTerm] = []
    groups: list[list[PauliTerm]] = []
    seed_rows: list[set[int]] = []
    while remaining:
        group: list[PauliTerm] = []
        selected_indices: list[int] = []
        for index, term in enumerate(remaining):
            if all(fully_commutes(term, other) for other in group):
                group.append(term)
                selected_indices.append(index)
        seeds = {remaining[index].source_row for index in selected_indices}
        for term in previously_grouped:
            if all(fully_commutes(term, other) for other in group):
                group.append(term)
        groups.append(group)
        seed_rows.append(seeds)
        # Supplementary Note 1 appends P* in the order in which the new P_alpha
        # was assembled.  The historical ToBW code accidentally reversed this
        # order while popping list indices; preserve the paper order here.
        selected_terms = [remaining[index] for index in selected_indices]
        selected_set = set(selected_indices)
        remaining = [
            term for index, term in enumerate(remaining) if index not in selected_set
        ]
        previously_grouped.extend(selected_terms)

    expected_rows = {term.source_row for term in terms}
    seeded_rows = set().union(*seed_rows)
    if seeded_rows != expected_rows or sum(map(len, seed_rows)) != len(terms):
        raise AssertionError("Extended SI seeds did not partition the Hamiltonian")
    for group_index, group in enumerate(groups):
        if len({term.source_row for term in group}) != len(group):
            raise AssertionError(f"Extended SI group {group_index} has a duplicate")
        for index, left in enumerate(group):
            if not all(fully_commutes(left, right) for right in group[index + 1 :]):
                raise AssertionError(f"Extended SI group {group_index} is not FC")
    return groups, seed_rows


def pauli_memberships(
    groups: Sequence[Sequence[PauliTerm]],
) -> dict[int, tuple[int, ...]]:
    memberships: dict[int, list[int]] = {}
    for group_index, group in enumerate(groups):
        for term in group:
            memberships.setdefault(term.source_row, []).append(group_index)
    return {row: tuple(indices) for row, indices in memberships.items()}


def load_dense_mps(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = [np.asarray(payload[key]) for key in sorted(payload.files)]
    value = arrays[0][0, :, :]
    for array in arrays[1:]:
        value = np.tensordot(value, array, axes=(-1, 0))
    state = np.asarray(value[..., 0], dtype=np.complex128).reshape(-1)
    raw_norm = float(np.linalg.norm(state))
    state /= raw_norm
    support = np.flatnonzero(np.abs(state) > AMPLITUDE_TOLERANCE)
    discarded_probability = max(
        float(1.0 - np.sum(np.abs(state[support]) ** 2)), 0.0
    )
    sparse = np.zeros_like(state)
    sparse[support] = state[support]
    sparse /= np.linalg.norm(sparse)
    audit = {
        "tensor_count": len(arrays),
        "tensor_shapes": [list(array.shape) for array in arrays],
        "max_mps_bond": max(max(array.shape[0], array.shape[2]) for array in arrays),
        "dense_dimension": len(state),
        "raw_mps_norm": raw_norm,
        "amplitude_support_tolerance": AMPLITUDE_TOLERANCE,
        "retained_support_size": len(support),
        "discarded_probability_before_sparse_renormalization": discarded_probability,
    }
    if discarded_probability > 1.0e-18:
        raise RuntimeError(
            f"Sparse support cutoff discards probability {discarded_probability:.3e}"
        )
    return sparse, audit


def parity_table(n: int) -> np.ndarray:
    table = np.zeros(1 << n, dtype=np.int8)
    for value in range(1, 1 << n):
        table[value] = table[value >> 1] ^ (value & 1)
    return table


def apply_pauli_dense(state: np.ndarray, term: PauliTerm, parity: np.ndarray) -> np.ndarray:
    indices = np.arange(len(state), dtype=np.uint32)
    signs = 1 - 2 * parity[np.bitwise_and(indices, term.zmask)]
    output = np.empty_like(state)
    output[np.bitwise_xor(indices, term.xmask)] = (
        (1j ** term.y_count) * signs * state
    )
    return output


def group_moments_sparse(
    group: Sequence[PauliTerm],
    state: np.ndarray,
    support: np.ndarray,
    parity: np.ndarray,
) -> tuple[float, float, float]:
    """Return exact mean, variance and ||H_g psi||^2.

    The state is sparse in a fixed particle/spin sector (1,225 amplitudes for
    BeH2 and 14,400 for N2).  Every Pauli maps that support bijectively, so the
    exact group action costs O(group_size * support_size), not O(2^n) per term.
    """
    amplitudes = state[support]
    acted = np.zeros_like(state)
    for term in group:
        signs = 1 - 2 * parity[np.bitwise_and(support, term.zmask)]
        targets = np.bitwise_xor(support, term.xmask)
        acted[targets] += (
            term.coefficient * (1j ** term.y_count) * signs * amplitudes
        )
    mean_complex = np.vdot(amplitudes, acted[support])
    norm_squared = float(np.vdot(acted, acted).real)
    if abs(mean_complex.imag) > 5.0e-10:
        raise RuntimeError(f"Non-real group mean {mean_complex}")
    mean = float(mean_complex.real)
    raw_variance = norm_squared - mean * mean
    if raw_variance < -5.0e-10:
        raise RuntimeError(f"Negative group variance {raw_variance}")
    return mean, max(raw_variance, 0.0), norm_squared


def gf2_coordinates(
    group: Sequence[PauliTerm], n: int
) -> tuple[list[PauliTerm], list[int]]:
    """Express all group Paulis in an independent generator basis over GF(2)."""
    rows: dict[int, tuple[int, int]] = {}
    generators: list[PauliTerm] = []
    coordinates: list[int] = []
    for term in group:
        vector = pack_symplectic(term, n)
        reduced = vector
        combination = 0
        for pivot in sorted(rows, reverse=True):
            row_vector, row_coordinates = rows[pivot]
            if (reduced >> pivot) & 1:
                reduced ^= row_vector
                combination ^= row_coordinates
        if reduced:
            generator_index = len(generators)
            generators.append(term)
            pivot = reduced.bit_length() - 1
            rows[pivot] = (reduced, combination ^ (1 << generator_index))
            coordinates.append(1 << generator_index)
        else:
            coordinates.append(combination)
    if len(generators) > n:
        raise AssertionError(
            f"Commuting Pauli rank {len(generators)} exceeds n={n}"
        )
    return generators, coordinates


def phase_on_basis(term: PauliTerm, basis_index: int) -> complex:
    sign = -1 if ((basis_index & term.zmask).bit_count() & 1) else 1
    return (1j ** term.y_count) * sign


def pauli_product_sign(
    term: PauliTerm, generators: Sequence[PauliTerm], coordinates: int
) -> int:
    basis = 0
    phase = 1.0 + 0.0j
    for index, generator in enumerate(generators):
        if (coordinates >> index) & 1:
            phase *= phase_on_basis(generator, basis)
            basis ^= generator.xmask
    if basis != term.xmask:
        raise AssertionError("Generator product has the wrong X mask")
    canonical_phase = 1j ** term.y_count
    ratio = phase / canonical_phase
    if abs(ratio.imag) > 1.0e-10 or abs(abs(ratio.real) - 1.0) > 1.0e-10:
        raise AssertionError(f"Unexpected Pauli product phase {ratio}")
    return 1 if ratio.real > 0 else -1


def fwht_in_place(values: np.ndarray) -> None:
    width = 1
    size = len(values)
    while width < size:
        blocks = values.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2


def exact_group_range(
    group: Sequence[PauliTerm], n: int
) -> tuple[float, float, float, int, list[int], list[int]]:
    """Exact min/max spectrum via generator polynomial and a Walsh transform."""
    generators, coordinates = gf2_coordinates(group, n)
    rank = len(generators)
    coefficients = np.zeros(1 << rank, dtype=float)
    signs: list[int] = []
    for term, coordinate in zip(group, coordinates):
        sign = pauli_product_sign(term, generators, coordinate)
        signs.append(sign)
        coefficients[coordinate] += sign * term.coefficient
    fwht_in_place(coefficients)
    minimum = float(np.min(coefficients))
    maximum = float(np.max(coefficients))
    centered_half_range = 0.5 * (maximum - minimum)
    return minimum, maximum, centered_half_range, rank, coordinates, signs


def largest_remainder_allocation(
    total_shots: int, weights: Sequence[float], active_mask: np.ndarray | None = None
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    active = weights > RANGE_TOLERANCE if active_mask is None else np.asarray(active_mask, dtype=bool)
    active_indices = np.flatnonzero(active)
    if total_shots < len(active_indices):
        raise ValueError(
            f"T={total_shots} is smaller than {len(active_indices)} nonconstant groups"
        )
    allocation = np.zeros(len(weights), dtype=np.int64)
    if not len(active_indices):
        return allocation
    allocation[active_indices] = 1
    remaining = total_shots - len(active_indices)
    if remaining:
        active_weights = weights[active]
        if float(np.sum(active_weights)) <= 0.0:
            active_weights = np.ones_like(active_weights)
        quotas = remaining * active_weights / float(np.sum(active_weights))
        floors = np.floor(quotas).astype(np.int64)
        allocation[active_indices] += floors
        leftover = remaining - int(np.sum(floors))
        order = np.lexsort((active_indices, -(quotas - floors)))
        allocation[active_indices[order[:leftover]]] += 1
    if int(np.sum(allocation)) != total_shots:
        raise AssertionError("Integer allocation did not conserve shots")
    if np.any(allocation[active] < 1):
        raise AssertionError("An active group received no shot")
    return allocation


def load_srcdf_reference(path: Path, molecule: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        if row["molecule"] != molecule or row["method"] != "s-RCDF":
            continue
        budget = int(row["T_total_shots"])
        result[budget] = {
            "empirical_rmse_hartree": float(row["empirical_total_RMSE_hartree"]),
            "analytic_rmse_hartree": float(row["analytic_total_RMSE_hartree"]),
            "selected_K": int(row["selected_K"]),
            "selected_depth": int(row["selected_depth"]),
            "measurement_settings": int(row["measurement_settings"]),
            "repeat_count": int(row["repeat_count"]),
        }
    if 3000 not in result:
        raise RuntimeError(f"Missing T=3000 s-RCDF reference in {path}")
    return result


def redistributed_fragments(
    groups: Sequence[Sequence[PauliTerm]],
    group_measurements: Sequence[float],
    memberships: dict[int, tuple[int, ...]],
) -> tuple[list[list[PauliTerm]], dict[int, float]]:
    """Eq. (22) fragments and Eq. (16) pooled counts/proportions ``M_j``."""
    measurement = np.asarray(group_measurements, dtype=float)
    if measurement.shape != (len(groups),) or np.any(measurement <= 0.0):
        raise ValueError("Every overlapping FC group needs a positive allocation")
    pooled = {
        row: float(np.sum(measurement[list(group_indices)]))
        for row, group_indices in memberships.items()
    }
    fragments: list[list[PauliTerm]] = []
    for group_index, group in enumerate(groups):
        fragment: list[PauliTerm] = []
        for term in group:
            split = term.coefficient * measurement[group_index] / pooled[term.source_row]
            fragment.append(
                PauliTerm(
                    term.source_row,
                    term.label,
                    split,
                    term.xmask,
                    term.zmask,
                    term.y_count,
                )
            )
        fragments.append(fragment)
    return fragments, pooled


def pooled_estimator_variance(
    groups: Sequence[Sequence[PauliTerm]],
    group_measurements: Sequence[float],
    memberships: dict[int, tuple[int, ...]],
    state: np.ndarray,
    support: np.ndarray,
    parity: np.ndarray,
) -> tuple[float, np.ndarray, dict[int, float], float]:
    """Exact Eq. (17) variance using the equivalent Eq. (22) fragments."""
    measurement = np.asarray(group_measurements, dtype=float)
    fragments, pooled = redistributed_fragments(groups, measurement, memberships)
    fragment_variances = np.zeros(len(groups), dtype=float)
    reconstructed_mean = 0.0
    for group_index, fragment in enumerate(fragments):
        mean, variance, _ = group_moments_sparse(fragment, state, support, parity)
        reconstructed_mean += mean
        fragment_variances[group_index] = variance
    total = float(np.sum(fragment_variances / measurement))
    if total < -5.0e-10:
        raise RuntimeError(f"Negative pooled estimator variance {total}")
    return max(total, 0.0), fragment_variances, pooled, reconstructed_mean


def variance_proportions(
    variances: Sequence[float], variance_floor: float = IMA_VARIANCE_FLOOR
) -> np.ndarray:
    """Nature/ToBW allocation ``m_g proportional sqrt(Var(A_g))``.

    ``run_ima.py`` initializes from ``measurement_utils``, which uses the
    exact nonnegative SI variances without a floor.  Subsequent IMA updates
    use ``cov_dict_utils`` and its ``1e-8`` numerical variance floor.
    """
    values = np.asarray(variances, dtype=float)
    weights = np.sqrt(np.maximum(values, variance_floor))
    if not float(np.sum(weights)) > 0.0:
        raise RuntimeError("All FC fragment allocation weights vanished")
    return weights / float(np.sum(weights))


def optimize_fc_ima(
    nonoverlap_groups: Sequence[Sequence[PauliTerm]],
    overlap_groups: Sequence[Sequence[PauliTerm]],
    state: np.ndarray,
    support: np.ndarray,
    parity: np.ndarray,
    cycles: int = DEFAULT_IMA_CYCLES,
) -> tuple[
    np.ndarray,
    list[dict[str, Any]],
    list[np.ndarray],
    dict[int, tuple[int, ...]],
]:
    """Eqs. (19)--(22), selecting the lowest exact variance over all cycles."""
    if len(nonoverlap_groups) != len(overlap_groups):
        raise AssertionError("Extended SI must preserve the ordinary-SI group count")
    initial_variances = []
    for group in nonoverlap_groups:
        _, variance, _ = group_moments_sparse(group, state, support, parity)
        initial_variances.append(variance)
    allocation = variance_proportions(initial_variances, variance_floor=0.0)
    memberships = pauli_memberships(overlap_groups)
    cycle_rows: list[dict[str, Any]] = []
    allocations: list[np.ndarray] = []
    for cycle in range(cycles + 1):
        variance, fragment_variances, pooled, reconstructed_mean = (
            pooled_estimator_variance(
                overlap_groups,
                allocation,
                memberships,
                state,
                support,
                parity,
            )
        )
        allocations.append(allocation.copy())
        cycle_rows.append(
            {
                "cycle": cycle,
                "continuous_variance_coefficient_hartree2_shots": variance,
                "continuous_rmse_coefficient_hartree_sqrt_shots": math.sqrt(variance),
                "minimum_group_fraction": float(np.min(allocation)),
                "maximum_group_fraction": float(np.max(allocation)),
                "sum_group_fractions": float(np.sum(allocation)),
                "minimum_pooled_pauli_fraction": float(min(pooled.values())),
                "maximum_pooled_pauli_fraction": float(max(pooled.values())),
                "reconstructed_nonidentity_mean_hartree": reconstructed_mean,
            }
        )
        if cycle < cycles:
            allocation = variance_proportions(fragment_variances)
    selected_cycle = min(
        range(len(cycle_rows)),
        key=lambda index: cycle_rows[index][
            "continuous_variance_coefficient_hartree2_shots"
        ],
    )
    for row in cycle_rows:
        row["selected_lowest_variance_cycle"] = row["cycle"] == selected_cycle
    return allocations[selected_cycle], cycle_rows, allocations, memberships


_DEPTH_MODULE: Any | None = None


def load_depth_module() -> Any:
    """Load the audited Yen diagonalizer without making this file a package."""
    global _DEPTH_MODULE
    if _DEPTH_MODULE is None:
        path = HERE / "traditional_cm_depth.py"
        module_spec = importlib.util.spec_from_file_location(
            "traditional_cm_depth_for_fc_ima", path
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"Cannot load FC diagonalizer from {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        _DEPTH_MODULE = module
    return _DEPTH_MODULE


def exact_group_born_distribution(
    group: Sequence[PauliTerm], state: np.ndarray
) -> BornDistribution:
    """Joint Born distribution after the audited Yen FC Clifford.

    Only nonzero probabilities are retained.  At 20 qubits this uses one
    statevector of length ``2**20`` and therefore remains practical without a
    density matrix or an exponentially larger object.
    """
    from qiskit.quantum_info import Clifford, Pauli, Statevector

    depth = load_depth_module()
    complete_basis = depth.complete_commuting_basis([term.label for term in group])
    circuit = depth.build_expanded_yen_circuit(complete_basis)
    core = Clifford(circuit)
    circuit.compose(depth.local_qwc_diagonalizer(core, complete_basis), inplace=True)
    final_clifford = Clifford(circuit)
    rotated = Statevector(state).evolve(circuit).data
    all_probabilities = np.abs(rotated) ** 2
    indices = np.flatnonzero(all_probabilities > BORN_PROBABILITY_TOLERANCE).astype(
        np.uint32
    )
    probabilities = np.asarray(all_probabilities[indices], dtype=float)
    probabilities /= float(np.sum(probabilities))
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    masks: list[int] = []
    signs: list[int] = []
    for term in group:
        transformed = Pauli(term.label).evolve(final_clifford, frame="s")
        if np.any(transformed.x):
            raise AssertionError(f"FC diagonalizer left X support for {term.label}")
        sign, _ = depth.strip_pauli_prefix(transformed.to_label())
        signs.append(sign)
        masks.append(depth.qiskit_z_mask(transformed))
    return BornDistribution(
        basis_indices=indices,
        probabilities=probabilities,
        cumulative_probabilities=cumulative,
        term_z_masks=np.asarray(masks, dtype=np.uint32),
        term_signs=np.asarray(signs, dtype=np.int8),
    )


def sample_fragment_means(
    distribution: BornDistribution,
    coefficients: Sequence[float],
    shots: int,
    repeats: int,
    rng: np.random.Generator,
    parity: np.ndarray,
) -> np.ndarray:
    """Sample one Eq. (22) fragment mean from genuine joint Born outcomes."""
    if shots < 1:
        raise ValueError("A measured FC group needs at least one shot")
    draws = rng.random((repeats, shots))
    selected = np.searchsorted(
        distribution.cumulative_probabilities, draws, side="right"
    )
    basis_indices = distribution.basis_indices[selected]
    values = np.zeros((repeats, shots), dtype=float)
    for coefficient, mask, sign in zip(
        coefficients, distribution.term_z_masks, distribution.term_signs
    ):
        eigenvalues = 1 - 2 * parity[np.bitwise_and(basis_indices, mask)].astype(
            np.int8
        )
        values += float(coefficient) * int(sign) * eigenvalues
    return np.mean(values, axis=1)


def evaluate_case(
    spec: CaseSpec,
    budgets: Sequence[int],
    repeats: int,
    results_root: Path,
    ima_cycles: int = DEFAULT_IMA_CYCLES,
) -> dict[str, Any]:
    """Run the publication FC-IMA estimator, including joint Born sampling."""
    started = time.perf_counter()
    if 3000 not in budgets:
        raise ValueError("The publication audit requires T=3000 in --budgets")
    case_dir = SELECTED_ROOT / spec.directory
    pauli_path = case_dir / "hamiltonian_pauli_blocked_spin.csv"
    mps_path = case_dir / "ground_state_mps_blocked_spin.npz"
    metadata_path = case_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    exact_energy = float(metadata["fci_ground_energy_hartree"])
    print(f"[{spec.molecule}] loading Hamiltonian and exact-state MPS", flush=True)
    terms, identity, pauli_audit = load_pauli_terms(pauli_path)
    state, state_audit = load_dense_mps(mps_path)
    n = int(pauli_audit["number_qubits"])
    if len(state) != 1 << n:
        raise RuntimeError(f"{spec.molecule}: state dimension does not match n={n}")
    support = np.flatnonzero(np.abs(state) > 0.0).astype(np.uint32)
    parity = parity_table(n)

    nonoverlap_groups = sorted_insertion_fc(terms)
    groups, seed_rows = extended_sorted_insertion_fc(terms)
    print(
        f"[{spec.molecule}] extended SI: {len(groups)} settings, "
        f"{sum(map(len, groups))} overlapping occurrences",
        flush=True,
    )
    selected_fractions, cycle_rows, cycle_allocations, memberships = optimize_fc_ima(
        nonoverlap_groups, groups, state, support, parity, cycles=ima_cycles
    )
    selected_cycle = next(
        int(row["cycle"])
        for row in cycle_rows
        if row["selected_lowest_variance_cycle"]
    )
    continuous_variance, continuous_fragment_variances, _, continuous_mean = (
        pooled_estimator_variance(
            groups, selected_fractions, memberships, state, support, parity
        )
    )
    if abs(identity + continuous_mean - exact_energy) > 2.0e-8:
        raise RuntimeError("Continuous IMA fragments do not reconstruct H")
    continuous_fragments, _ = redistributed_fragments(
        groups, selected_fractions, memberships
    )

    source_index = {term.source_row: index for index, term in enumerate(terms)}
    group_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        _, _, half_range, rank, coordinates, signs = exact_group_range(group, n)
        fragment_mean, _, _ = group_moments_sparse(
            continuous_fragments[group_index], state, support, parity
        )
        group_rows.append(
            {
                "molecule": spec.molecule,
                "group_index": group_index,
                "term_count": len(group),
                "ordinary_si_seed_term_count": len(seed_rows[group_index]),
                "overlap_reused_term_count": len(group) - len(seed_rows[group_index]),
                "generator_rank": rank,
                "centered_half_range_hartree_audit_only": half_range,
                "selected_ima_cycle": selected_cycle,
                "continuous_ima_fraction": selected_fractions[group_index],
                "continuous_fragment_mean_hartree": fragment_mean,
                "continuous_fragment_variance_hartree2": continuous_fragment_variances[
                    group_index
                ],
                "continuous_variance_contribution_hartree2_shots": continuous_fragment_variances[
                    group_index
                ]
                / selected_fractions[group_index],
            }
        )
        generator_rows = {item.source_row for item in gf2_coordinates(group, n)[0]}
        for term, coordinate, sign in zip(group, coordinates, signs):
            membership_rows.append(
                {
                    "molecule": spec.molecule,
                    "group_index": group_index,
                    "source_row": term.source_row,
                    "source_csv_row_1based_including_header": term.source_row,
                    "source_index_zero_based_nonidentity": source_index[term.source_row],
                    "pauli_label_site0_to_siteNminus1": term.label,
                    "coefficient_hartree": term.coefficient,
                    "coefficient_real_hartree": term.coefficient,
                    "occurrence_count": len(memberships[term.source_row]),
                    "membership_group_indices": " ".join(
                        map(str, memberships[term.source_row])
                    ),
                    "is_disjoint_si_seed": term.source_row in seed_rows[group_index],
                    "is_selected_independent_generator": term.source_row in generator_rows,
                    "generator_coordinate_mask_hex": hex(coordinate),
                    "generator_product_sign": sign,
                }
            )

    srcdf = load_srcdf_reference(spec.srcdf_csv, spec.molecule)
    curve_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    budget_cache: dict[int, dict[str, Any]] = {}
    all_groups_active = np.ones(len(groups), dtype=bool)
    for total_shots in budgets:
        counts = largest_remainder_allocation(
            total_shots, selected_fractions, active_mask=all_groups_active
        )
        variance, fragment_variances, pooled, integer_mean = pooled_estimator_variance(
            groups, counts, memberships, state, support, parity
        )
        if abs(identity + integer_mean - exact_energy) > 2.0e-8:
            raise RuntimeError("Integer IMA fragments do not reconstruct H")
        fragments, _ = redistributed_fragments(groups, counts, memberships)
        seed = stable_seed(spec.molecule, "fc_ima_primary", total_shots)
        reference = srcdf.get(total_shots, {})
        curve_rows.append(
            {
                "molecule": spec.molecule,
                "method": "FC-IMA",
                "allocation": "fc_ima_primary",
                "T_total_shots": total_shots,
                "T_actual_shots": int(np.sum(counts)),
                "group_count": len(groups),
                "overlapping_term_occurrence_count": sum(map(len, groups)),
                "analytic_rmse_hartree": math.sqrt(variance),
                "empirical_born_rmse_hartree": "",
                "empirical_repeats": repeats,
                "empirical_model": "joint Born outcomes; Eq. (14) pooled Pauli estimator",
                "sampling_seed": seed,
                "selected_ima_cycle": selected_cycle,
                "srcdf_empirical_rmse_hartree": reference.get(
                    "empirical_rmse_hartree", ""
                ),
                "srcdf_analytic_rmse_hartree": reference.get(
                    "analytic_rmse_hartree", ""
                ),
            }
        )
        budget_cache[total_shots] = {
            "counts": counts,
            "fragments": fragments,
            "analytic_variance": variance,
            "seed": seed,
            "rng": np.random.default_rng(seed),
            "estimates": np.full(repeats, identity, dtype=float),
        }
        for group_index, count in enumerate(counts):
            allocation_rows.append(
                {
                    "molecule": spec.molecule,
                    "allocation": "fc_ima_primary",
                    "T_total_shots": total_shots,
                    "group_index": group_index,
                    "allocated_shots": int(count),
                    "continuous_ima_fraction": selected_fractions[group_index],
                    "integer_fragment_variance_hartree2": fragment_variances[
                        group_index
                    ],
                    "integer_pooled_variance_contribution_hartree2": fragment_variances[
                        group_index
                    ]
                    / count,
                }
            )
        for term in terms:
            pooled_rows.append(
                {
                    "molecule": spec.molecule,
                    "allocation": "fc_ima_primary",
                    "T_total_shots": total_shots,
                    "source_row": term.source_row,
                    "source_csv_row_1based_including_header": term.source_row,
                    "source_index_zero_based_nonidentity": source_index[term.source_row],
                    "pauli_label_site0_to_siteNminus1": term.label,
                    "coefficient_hartree": term.coefficient,
                    "membership_count": len(memberships[term.source_row]),
                    "membership_group_indices": " ".join(
                        map(str, memberships[term.source_row])
                    ),
                    "total_pooled_shots_Mj": int(round(pooled[term.source_row])),
                }
            )

    print(f"[{spec.molecule}] sampling genuine joint Born outcomes", flush=True)
    for group_index, group in enumerate(groups):
        distribution = exact_group_born_distribution(group, state)
        for total_shots in budgets:
            cache = budget_cache[total_shots]
            fragment = cache["fragments"][group_index]
            cache["estimates"] += sample_fragment_means(
                distribution,
                [term.coefficient for term in fragment],
                int(cache["counts"][group_index]),
                repeats,
                cache["rng"],
                parity,
            )
        if (group_index + 1) % 10 == 0 or group_index + 1 == len(groups):
            print(
                f"[{spec.molecule}] Born settings {group_index + 1}/{len(groups)}",
                flush=True,
            )
    for row in curve_rows:
        cache = budget_cache[int(row["T_total_shots"])]
        estimates = cache["estimates"]
        row["empirical_born_rmse_hartree"] = math.sqrt(
            float(np.mean((estimates - exact_energy) ** 2))
        )
        for repeat, estimate in enumerate(estimates):
            replicate_rows.append(
                {
                    "molecule": spec.molecule,
                    "allocation": "fc_ima_primary",
                    "T_total_shots": row["T_total_shots"],
                    "repeat_index": repeat,
                    "estimated_energy_hartree": float(estimate),
                    "error_hartree": float(estimate - exact_energy),
                    "sampling_seed": row["sampling_seed"],
                    "model": "joint Born outcomes; Eq. (14) pooled Pauli estimator",
                }
            )

    reference_variance = budget_cache[3000]["analytic_variance"]
    frozen_coefficient = 3000 * reference_variance
    inverse_rows: list[dict[str, Any]] = []
    for target_error in np.geomspace(0.01, 0.5, 120):
        unbounded = frozen_coefficient / float(target_error**2)
        bounded = max(float(len(groups)), unbounded)
        planned_total = int(math.ceil(bounded))
        inverse_rows.append(
            {
                "molecule": spec.molecule,
                "method": "FC-IMA",
                "allocation": "fc_ima_primary",
                "target_error_hartree": float(target_error),
                "frozen_reference_T0_shots": 3000,
                "frozen_T0_analytic_variance_hartree2": reference_variance,
                "variance_coefficient_V_equals_T0_variance_hartree2_shots": frozen_coefficient,
                "continuous_T_unbounded_V_over_epsilon2": unbounded,
                "continuous_T_lower_bounded_by_group_count": bounded,
                "integer_T_ceiling": planned_total,
                "frozen_scaling_rmse_hartree": math.sqrt(
                    frozen_coefficient / planned_total
                ),
                "zero_decomposition_bias": True,
                "group_count_lower_bound": len(groups),
            }
        )

    cycle_allocation_rows: list[dict[str, Any]] = []
    for cycle_row, allocation in zip(cycle_rows, cycle_allocations):
        cycle_row["molecule"] = spec.molecule
        for group_index, fraction in enumerate(allocation):
            cycle_allocation_rows.append(
                {
                    "molecule": spec.molecule,
                    "cycle": cycle_row["cycle"],
                    "group_index": group_index,
                    "continuous_fraction": fraction,
                    "cycle_variance_coefficient_hartree2_shots": cycle_row[
                        "continuous_variance_coefficient_hartree2_shots"
                    ],
                    "selected_lowest_variance_cycle": cycle_row[
                        "selected_lowest_variance_cycle"
                    ],
                }
            )

    primary = next(row for row in curve_rows if row["T_total_shots"] == 3000)
    srcdf_3000 = srcdf[3000]
    summary = {
        "molecule": spec.molecule,
        "number_qubits": n,
        "retained_pauli_terms": len(terms),
        "identity_offset_hartree": identity,
        "fc_ordinary_si_group_count": len(nonoverlap_groups),
        "fc_overlapping_group_count": len(groups),
        "fc_overlapping_term_occurrence_count": sum(map(len, groups)),
        "grouping": "Nature-2023 extended coefficient-sorted SI, FC overlap, paper-order P*",
        "estimator": "Eq. (14) pooled Pauli means; Eq. (17) variance",
        "allocation_optimization": "Eqs. (19)-(22) exact-state FC-IMA",
        "ima_cycles": ima_cycles,
        "selected_ima_cycle": selected_cycle,
        "selected_continuous_variance_coefficient_hartree2_shots": continuous_variance,
        "exact_energy_hartree": exact_energy,
        "reconstructed_hamiltonian_expectation_hartree": identity + continuous_mean,
        "expectation_minus_exact_hartree": identity + continuous_mean - exact_energy,
        "T_gate": 3000,
        "inverse_error_frozen_reference_T0_shots": 3000,
        "inverse_error_frozen_T0_variance_hartree2": reference_variance,
        "inverse_error_frozen_variance_coefficient_T0_times_variance_hartree2_shots": frozen_coefficient,
        "fc_primary_analytic_rmse_hartree": primary["analytic_rmse_hartree"],
        "fc_primary_born_rmse_hartree": primary["empirical_born_rmse_hartree"],
        "srcdf_analytic_rmse_hartree": srcdf_3000["analytic_rmse_hartree"],
        "srcdf_empirical_rmse_hartree": srcdf_3000["empirical_rmse_hartree"],
        "fc_analytic_strictly_worse_than_srcdf": primary["analytic_rmse_hartree"]
        > srcdf_3000["analytic_rmse_hartree"],
        "fc_born_strictly_worse_than_srcdf_empirical": primary[
            "empirical_born_rmse_hartree"
        ]
        > srcdf_3000["empirical_rmse_hartree"],
        "srcdf_T3000_selected_K": srcdf_3000["selected_K"],
        "srcdf_T3000_selected_depth": srcdf_3000["selected_depth"],
        "srcdf_T3000_measurement_settings": srcdf_3000["measurement_settings"],
        "pauli_input_audit": pauli_audit,
        "state_input_audit": state_audit,
        "elapsed_seconds": time.perf_counter() - started,
    }
    case_output = results_root / spec.molecule
    write_csv(case_output / "group_statistics.csv", group_rows)
    write_csv(case_output / "group_membership.csv", membership_rows)
    write_csv(case_output / "shot_allocations.csv", allocation_rows)
    write_csv(case_output / "pooled_term_allocations.csv", pooled_rows)
    write_csv(case_output / "ima_cycles.csv", cycle_rows)
    write_csv(case_output / "ima_cycle_allocations.csv", cycle_allocation_rows)
    write_csv(case_output / "sampling_curve.csv", curve_rows)
    write_csv(case_output / "inverse_error_curve.csv", inverse_rows)
    write_csv(case_output / "born_replicates.csv", replicate_rows)
    write_json(case_output / "summary.json", summary)
    return {
        "summary": summary,
        "curve_rows": curve_rows,
        "inverse_rows": inverse_rows,
        "input_hashes": {
            "hamiltonian_pauli_blocked_spin.csv": sha256_file(pauli_path),
            "ground_state_mps_blocked_spin.npz": sha256_file(mps_path),
            "metadata.json": sha256_file(metadata_path),
            "srcdf_reference_csv": sha256_file(spec.srcdf_csv),
        },
    }


def make_plot(
    all_curves: Sequence[dict[str, Any]], results_root: Path
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), sharey=True)
    styles = {
        "fc_ima_primary": ("#d62728", "o", "FC-IMA (Nature 2023)"),
    }
    for ax, molecule in zip(axes, ("BeH2", "N2")):
        selected = [row for row in all_curves if row["molecule"] == molecule]
        for allocation, (color, marker, label) in styles.items():
            rows = sorted(
                [row for row in selected if row["allocation"] == allocation],
                key=lambda row: row["T_total_shots"],
            )
            ax.plot(
                [row["T_total_shots"] for row in rows],
                [row["analytic_rmse_hartree"] for row in rows],
                color=color,
                marker=marker,
                linewidth=1.4,
                markersize=4.2,
                label=label,
            )
        primary = sorted(
            [row for row in selected if row["allocation"] == "fc_ima_primary"],
            key=lambda row: row["T_total_shots"],
        )
        ax.plot(
            [row["T_total_shots"] for row in primary],
            [float(row["srcdf_analytic_rmse_hartree"]) for row in primary],
            color="#1f77b4",
            marker="x",
            linewidth=1.4,
            markersize=5.0,
            label="s-RCDF analytic",
        )
        ax.plot(
            [row["T_total_shots"] for row in primary],
            [float(row["srcdf_empirical_rmse_hartree"]) for row in primary],
            color="#1f77b4",
            linestyle="--",
            marker="+",
            linewidth=1.0,
            markersize=5.0,
            label="s-RCDF empirical (50)",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total measurements T")
        ax.set_title(r"BeH$_2$" if molecule == "BeH2" else r"N$_2$")
        ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.4)
    axes[0].set_ylabel("RMSE (Ha)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8.0)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.76, wspace=0.12)
    path = results_root / "fc_vs_srcdf_error.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    # Keep the historical filename as a byte-equivalent visual compatibility
    # alias.  Its plotted legend and all public labels are nevertheless FC.
    fig.savefig(results_root / "cm_vs_srcdf_error.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.molecule for case in CASES],
        default=[case.molecule for case in CASES],
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--ima-cycles", type=int, default=DEFAULT_IMA_CYCLES)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [case for case in CASES if case.molecule in set(args.cases)]
    outputs = [
        evaluate_case(
            case, args.budgets, args.repeats, args.output, ima_cycles=args.ima_cycles
        )
        for case in selected
    ]
    summaries = [item["summary"] for item in outputs]
    curves = [row for item in outputs for row in item["curve_rows"]]
    inverse_rows = [row for item in outputs for row in item["inverse_rows"]]
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "sampling_curves_all.csv", curves)
    write_csv(args.output / "inverse_error_curves_all.csv", inverse_rows)
    fig3_ready: list[dict[str, Any]] = []
    for row in curves:
        if row["allocation"] != "fc_ima_primary":
            continue
        fig3_ready.append(
            {
                "molecule": row["molecule"],
                "curve": "error_vs_measurements",
                "method": "FC-IMA",
                "x": row["T_total_shots"],
                "y": row["analytic_rmse_hartree"],
                "raw_y": row["analytic_rmse_hartree"],
                "right_censored": False,
                "model": "Nature-2023 extended SI + exact-state IMA; integer Eq. (17)",
                "T_fixed": "",
            }
        )
    for row in inverse_rows:
        fig3_ready.append(
            {
                "molecule": row["molecule"],
                "curve": "estimated_measurements_vs_error",
                "method": "FC-IMA",
                "x": row["target_error_hartree"],
                "y": row["continuous_T_lower_bounded_by_group_count"],
                "raw_y": row["continuous_T_unbounded_V_over_epsilon2"],
                "right_censored": False,
                "model": "frozen T0=3000 exact-analytic variance; zero bias",
                "T_fixed": 3000,
            }
        )
    write_csv(args.output / "fc_fig3_ready_curves.csv", fig3_ready)
    # Historical path compatibility only; never emit the former public method
    # label in the aliased table.
    write_csv(args.output / "cm_fig3_ready_curves.csv", fig3_ready)
    plot_path = make_plot(curves, args.output)
    both_present = {row["molecule"] for row in summaries} == {"BeH2", "N2"}
    overall = {
        "version": VERSION,
        "primary_error_metric": "integer Eq. (17) RMSE for Nature-2023 FC-IMA at T=3000",
        "both_molecules_present": both_present,
        "both_molecules_fc_analytic_strictly_worse_than_srcdf": bool(
            both_present
            and all(row["fc_analytic_strictly_worse_than_srcdf"] for row in summaries)
        ),
        "both_molecules_fc_born_strictly_worse_than_srcdf_empirical": bool(
            both_present
            and all(
                row["fc_born_strictly_worse_than_srcdf_empirical"]
                for row in summaries
            )
        ),
        "budgets": list(args.budgets),
        "repeats": args.repeats,
        "ima_cycles": args.ima_cycles,
        "cases": summaries,
        "input_sha256": {
            summary["molecule"]: output["input_hashes"]
            for summary, output in zip(summaries, outputs)
        },
        "implementation_sha256": sha256_file(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "plot": str(plot_path),
        "legacy_aliases_refreshed_with_fc_content": [
            str(args.output / "cm_fig3_ready_curves.csv"),
            str(args.output / "cm_vs_srcdf_error.png"),
        ],
        "formal_manuscript_files_modified": False,
    }
    write_json(args.output / "manifest.json", overall)
    print(json.dumps(overall, indent=2, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
