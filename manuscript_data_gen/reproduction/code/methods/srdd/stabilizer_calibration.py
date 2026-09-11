#!/usr/bin/env python3
"""Stabilizer calibration and dynamic-depth validation utilities.

This script is deliberately independent of the formal ``results`` directory.
It can replay the saved LiH pilot to fit two non-negative component weights, then
builds one F-dominant source frontier in which circuit depths 1 through 7 are
tried in ascending order with first-no-benefit stopping inside every circuit
family.  RCDF has no circuit depth.  The frozen
weights select K offline for several shot budgets, and exact eight-qubit Born
sampling checks the selected ground-state energy errors.

The production AGPD prefix keeps every source whole and realizes the fixed
molecular scalar-plus-one-body reference with an exact depth-bounded shallow
cover.  It therefore has no independent dense collector and performs no
source-to-collector transfer.  Dense matrices, state vectors, and exact
variances are used only as an eight-qubit validation oracle; this file is not
a scalable algorithmic implementation or a complexity claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import weakref

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "measansatz_matplotlib_cache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import linprog

from adaptive_f3_pool import (
    BLOCK_SIZE,
    DIMENSION,
    FAMILIES,
    LABELS,
    N_QUBITS,
    SPATIAL_OCCUPATIONS,
    SPATIAL_ORBITALS,
    STANDARD_F3_FAMILIES,
    Fragment,
    best_rcdf_fragment,
    diagonal_centered_norm,
    f3_r2,
    hermitian,
    interaction_dense,
    load_active_case,
    load_circuit_module,
    make_rcdf_bank,
    make_shallow_rcdf_bank,
    matrix_centered_norm,
    optimize_circuit_fragment,
)
from shallow_collector_redistribution import (
    collector_dictionary,
    collector_matrix,
    collector_tailored_extra_shallow_rotations,
    dictionary_diagnostics,
    equality_project,
    greedy_exact_fill,
    symmetric_vector,
)


HERE = Path(__file__).resolve().parent
from srdd_release_paths import BUNDLE as WORKSPACE
PILOT_INPUT = HERE / "sample_aware_pilot_v1" / "LiH"
DEFAULT_OUTPUT = HERE / "stabilizer_calibration_v2"
FORMAL_RESULTS = HERE / "results"
ALGORITHM_DOCUMENT = WORKSPACE / "FIGURE_REPRODUCTION.md"
TEX_CANDIDATE = WORKSPACE / "paper" / "original" / "supplementary_labelled.tex"

VERSION = "stabilizer-calibration-agpd-shallow-cover-sector-range-v6"
CIRCUIT_FAMILIES = tuple(value for value in FAMILIES if value != "shallow_rcdf")
DEPTHS = tuple(range(1, 8))
DEPTH_RELATIVE_IMPROVEMENT = 2.0e-3
CALIBRATION_K_VALUES = tuple(range(1, 11))
SHOT_GRID = (100, 200, 300, 500, 800, 1_200, 2_000, 3_000)
STABILIZER_PROBE_SEED = 918273
STABILIZER_PROBE_COUNT = 500
STABILIZER_SPLIT_COUNTS = {"train": 300, "validation": 100, "test": 100}
SAMPLING_RATIO_RELATIVE_THRESHOLD = 1.0e-12
SAMPLING_RATIO_ABSOLUTE_THRESHOLD_HA2 = 1.0e-30
COLORS = {
    "gfro": "#2563eb",
    "operator_pool": "#f97316",
    "nnk_uccgsdi": "#16a34a",
    "shallow_iswap_su2": "#9333ea",
    "rcdf": "#d62728",
    "base_collector_only": "#4b5563",
}

# AGPD uses a common, exact, depth-bounded materialization of the molecular
# scalar-plus-one-body reference.  Source fragments are kept whole, so no
# non-Gaussian conjugated direction is ever hidden in a dense collector.
AGPD_SHALLOW_COLLECTOR_DEPTHS = (1, 2, 3)
AGPD_SHALLOW_COLLECTOR_MIN_EXTRA_LEAVES = 1
AGPD_SHALLOW_COLLECTOR_MAX_EXTRA_LEAVES = 8
AGPD_SHALLOW_COLLECTOR_EQUALITY_TOLERANCE = 1.0e-10
AGPD_SHALLOW_COLLECTOR_MATRIX_TOLERANCE = 2.0e-8
AGPD_SHALLOW_COLLECTOR_ANGLE_STEPS = 400
AGPD_SECTOR_LEAKAGE_TOLERANCE = 1.0e-10
_AGPD_SHALLOW_COLLECTOR_CACHE: dict[tuple[Any, ...], tuple[PrefixDecomposition, int]] = {}
_AGPD_SECTOR_RANGE_CACHE: dict[tuple[Any, ...], tuple[Any, dict[str, Any]]] = {}


@dataclass
class Setting:
    matrix: np.ndarray
    centered_range: float
    source_index: int
    family: str
    interface: str
    native_minimum: float
    native_maximum: float
    # The fields below are populated by the depth-bounded s-RCDF and AGPD
    # one-body-cover paths.  They stay optional for archived dense diagnostics.
    rotation: np.ndarray | None = None
    native_diagonal: np.ndarray | None = None
    rotation_depth: int | None = None
    setting_kind: str | None = None
    collector_linear_coefficients: np.ndarray | None = None
    # Production AGPD allocations use the spectrum reachable from the fixed
    # particle-number sector.  The full-Fock value is retained as an audit
    # baseline, and leakage is checked before the restricted range is trusted.
    range_domain: str | None = None
    full_fock_centered_range: float | None = None
    sector_spectral_centered_range: float | None = None
    allocation_centered_range: float | None = None
    sector_minimum: float | None = None
    sector_maximum: float | None = None
    sector_dimension: int | None = None
    sector_leakage_frobenius: float | None = None
    leakage_operator_bound: float | None = None
    sector_leakage_relative_frobenius: float | None = None
    sector_range_eligible: bool | None = None


@dataclass
class PrefixDecomposition:
    k: int
    settings: list[Setting]
    alpha_by_source: np.ndarray
    spectral_sum: float
    spectral_sum_before_r2: float
    reconstruction_error: float
    f3_optimizer: dict[str, Any]
    collector_audit: dict[str, Any] = field(default_factory=dict)


class ShallowCollectorRepresentationError(RuntimeError):
    """Raised when a declared depth-limited bank cannot reproduce C exactly."""

    def __init__(self, audit: dict[str, Any]):
        self.audit = audit
        super().__init__(
            "Shallow collector equality constraint is infeasible: "
            f"relative residual={audit.get('relative_reconstruction_residual', float('nan')):.3e}, "
            f"rank={audit.get('dictionary_rank', 0)}/"
            f"{audit.get('symmetric_target_dimension', 0)}"
        )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sector_range_metrics(
    matrix: np.ndarray,
    sector_indices: np.ndarray,
    *,
    full_fock_centered_range: float | None = None,
    leakage_tolerance: float = AGPD_SECTOR_LEAKAGE_TOLERANCE,
) -> dict[str, Any]:
    """Return the reachable half-range and a fail-closed sector audit.

    A restricted spectral range is a valid outcome bound only if the setting
    preserves the declared physical sector.  This helper therefore reports the
    off-sector block norm alongside the range.  Callers that use the restricted
    value for shot allocation must reject ``passes_leakage_check=False``.

    ``full_fock_centered_range`` may be supplied when the native diagonal is
    already known.  This avoids an unnecessary full-space diagonalization in
    the inner AGPD search while retaining the unrestricted audit baseline.
    """
    raw = np.asarray(matrix, dtype=np.complex128)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array")
    if not np.all(np.isfinite(raw)):
        raise ValueError("matrix contains non-finite entries")
    if not np.isfinite(leakage_tolerance) or leakage_tolerance < 0.0:
        raise ValueError("leakage_tolerance must be finite and non-negative")

    dimension = int(raw.shape[0])
    indices = np.asarray(sector_indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("sector_indices must be a non-empty one-dimensional array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("sector_indices must contain integers")
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= dimension):
        raise ValueError("sector_indices contains an out-of-bounds index")
    if np.unique(indices).size != indices.size:
        raise ValueError("sector_indices must not contain duplicates")

    matrix_scale = max(float(np.linalg.norm(raw, "fro")), 1.0)
    antihermitian_relative = float(
        np.linalg.norm(raw - raw.conj().T, "fro") / matrix_scale
    )
    if antihermitian_relative > 1.0e-12:
        raise ValueError(
            "matrix is not Hermitian within tolerance: "
            f"relative anti-Hermitian norm={antihermitian_relative:.3e}"
        )
    value = hermitian(raw)
    sector = value[np.ix_(indices, indices)]
    sector_spectrum = np.linalg.eigvalsh(sector)
    sector_minimum = float(sector_spectrum[0])
    sector_maximum = float(sector_spectrum[-1])
    sector_spectral_range = 0.5 * (sector_maximum - sector_minimum)

    outside = np.ones(dimension, dtype=bool)
    outside[indices] = False
    leakage_block = value[np.ix_(outside, indices)]
    leakage_frobenius = float(np.linalg.norm(leakage_block, "fro"))
    leakage_relative = leakage_frobenius / max(
        float(np.linalg.norm(value, "fro")), 1.0
    )

    if full_fock_centered_range is None:
        full_spectrum = np.linalg.eigvalsh(value)
        full_range = 0.5 * float(full_spectrum[-1] - full_spectrum[0])
    else:
        full_range = float(full_fock_centered_range)
        if not np.isfinite(full_range) or full_range < 0.0:
            raise ValueError(
                "full_fock_centered_range must be finite and non-negative"
            )
    # A principal submatrix cannot have a wider spectrum than the full matrix.
    # Leave a small absolute tolerance for native-diagonal roundoff.
    if sector_spectral_range > full_range + 1.0e-9 * max(1.0, full_range):
        raise ValueError(
            "sector range exceeds the supplied full-Fock range: "
            f"{sector_spectral_range:.12g} > {full_range:.12g}"
        )

    # For a state |psi> in P,
    #   Var_M(psi) = Var_{PMP}(psi) + ||QMP psi||^2.
    # The restricted half-range bounds the first term and the off-block
    # Frobenius norm conservatively bounds the second.  Cap their quadrature by
    # the independently valid full-Fock half-range.  Exact sector-preserving
    # chemistry settings have a zero guard term, so this reduces exactly to the
    # requested physical-sector range.
    leakage_operator_bound = leakage_frobenius
    allocation_range = min(
        full_range,
        math.hypot(sector_spectral_range, leakage_operator_bound),
    )

    return {
        "range_domain": "fixed_particle_number_sector",
        "centered_range": float(allocation_range),
        "allocation_centered_range": float(allocation_range),
        "sector_spectral_centered_range": float(sector_spectral_range),
        "full_fock_centered_range": float(full_range),
        "sector_minimum": sector_minimum,
        "sector_maximum": sector_maximum,
        "sector_dimension": int(indices.size),
        "leakage_frobenius": leakage_frobenius,
        "leakage_operator_bound": leakage_operator_bound,
        "leakage_relative_frobenius": float(leakage_relative),
        "leakage_tolerance": float(leakage_tolerance),
        "passes_leakage_check": bool(leakage_relative <= leakage_tolerance),
        "relative_antihermitian_norm": antihermitian_relative,
    }


def _cached_sector_range_metrics(
    matrix: np.ndarray,
    sector_indices: np.ndarray,
    *,
    full_fock_centered_range: float,
    leakage_tolerance: float,
) -> dict[str, Any]:
    """Cache immutable range audits for matrices reused across AGPD trials."""
    value = np.asarray(matrix)
    indices = np.asarray(sector_indices, dtype=np.int64)
    key = (
        id(value),
        value.shape,
        indices.tobytes(),
        float(full_fock_centered_range),
        float(leakage_tolerance),
    )
    cached = _AGPD_SECTOR_RANGE_CACHE.get(key)
    if cached is not None and cached[0]() is value:
        return dict(cached[1])
    metrics = _sector_range_metrics(
        value,
        indices,
        full_fock_centered_range=full_fock_centered_range,
        leakage_tolerance=leakage_tolerance,
    )

    reference = None

    def discard(_: Any) -> None:
        current = _AGPD_SECTOR_RANGE_CACHE.get(key)
        if current is not None and current[0] is reference:
            _AGPD_SECTOR_RANGE_CACHE.pop(key, None)

    reference = weakref.ref(value, discard)
    _AGPD_SECTOR_RANGE_CACHE[key] = (reference, dict(metrics))
    return metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return "MISSING"
    files = sorted(value for value in path.rglob("*") if value.is_file())
    for value in files:
        digest.update(str(value.relative_to(path)).replace("\\", "/").encode())
        digest.update(bytes.fromhex(sha256(value)))
    return digest.hexdigest()


def protected_snapshot() -> dict[str, str]:
    return {
        "algorithm_update.md": sha256(ALGORITHM_DOCUMENT),
        "recommended_tex": sha256(TEX_CANDIDATE),
        "formal_results_tree": tree_digest(FORMAL_RESULTS),
    }


def validate_output_path(output: Path) -> Path:
    resolved = output.resolve()
    formal = FORMAL_RESULTS.resolve()
    if resolved == formal or formal in resolved.parents:
        raise ValueError("Pilot output must not be inside the formal results directory")
    if resolved in (ALGORITHM_DOCUMENT.resolve(), TEX_CANDIDATE.resolve()):
        raise ValueError("Pilot output collides with a protected document")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Output directory is non-empty: {resolved}. Choose a fresh --output path."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "figures").mkdir(exist_ok=True)
    return resolved


def expectation(states: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    action = matrix @ states
    return np.real(np.einsum("ij,ij->j", states.conj(), action))


def expectation_and_variance(
    states: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    action = matrix @ states
    mean = np.real(np.einsum("ij,ij->j", states.conj(), action))
    second = np.real(np.einsum("ij,ij->j", action.conj(), action))
    return mean, np.maximum(second - mean * mean, 0.0)


def ground_state(case) -> tuple[np.ndarray, float]:
    block = case.target[np.ix_(case.sector_indices, case.sector_indices)]
    eigenvalues, eigenvectors = np.linalg.eigh(block)
    state = np.zeros(DIMENSION, dtype=np.complex128)
    state[case.sector_indices] = eigenvectors[:, 0]
    return state, float(eigenvalues[0])


def save_fragments(path: Path, fragments: list[Fragment], depths: list[int]) -> None:
    # Current AGPD keeps every source whole.  The legacy-named transfer fields
    # remain in the NPZ schema only so old readers see an explicit all-false
    # contract instead of silently assuming GFRO transfer.
    transfer_mask = np.zeros(len(fragments), dtype=bool)
    zero_matrices = [np.zeros_like(value.matrix) for value in fragments]
    zero_diagonals = [np.zeros_like(value.diagonal) for value in fragments]
    np.savez_compressed(
        path,
        matrices=np.asarray([value.matrix for value in fragments]),
        raw_source_one_matrices=np.asarray(
            [value.one_matrix for value in fragments]
        ),
        raw_source_q_directions=np.asarray(
            [value.q_direction for value in fragments]
        ),
        effective_one_matrices=np.asarray(
            [
                value.one_matrix if enabled else zero
                for value, enabled, zero in zip(fragments, transfer_mask, zero_matrices)
            ]
        ),
        effective_q_directions=np.asarray(
            [
                value.q_direction if enabled else zero
                for value, enabled, zero in zip(fragments, transfer_mask, zero_matrices)
            ]
        ),
        diagonals=np.asarray([value.diagonal for value in fragments]),
        effective_source_diagonals=np.asarray(
            [value.diagonal for value in fragments]
        ),
        raw_source_one_diagonals=np.asarray(
            [value.one_diagonal for value in fragments]
        ),
        remainder_diagonals=np.asarray(
            [value.remainder_diagonal for value in fragments]
        ),
        raw_source_q_diagonals=np.asarray(
            [value.q_diagonal for value in fragments]
        ),
        effective_one_diagonals=np.asarray(
            [
                value.one_diagonal if enabled else zero
                for value, enabled, zero in zip(fragments, transfer_mask, zero_diagonals)
            ]
        ),
        effective_q_diagonals=np.asarray(
            [
                value.q_diagonal if enabled else zero
                for value, enabled, zero in zip(fragments, transfer_mask, zero_diagonals)
            ]
        ),
        families=np.asarray([value.family for value in fragments]),
        raw_source_f3_interfaces=np.asarray(
            [value.f3_interface for value in fragments]
        ),
        effective_f3_interfaces=np.asarray(
            ["native_shallow_source_no_collector_transfer"] * len(fragments)
        ),
        standard_f3_transfer_mask=transfer_mask,
        source_to_collector_transfer_mask=transfer_mask,
        chosen_depth=np.asarray(depths, dtype=int),
    )


def replay_saved_pilot(case, circuits, starts: int, iterations: int):
    source = PILOT_INPUT / "selected_fragments.npz"
    with np.load(source, allow_pickle=False) as saved:
        saved_matrices = np.asarray(saved["matrices"])
        families = [str(value) for value in saved["families"]]
    residual = hermitian(case.target - case.base_collector)
    fragments: list[Fragment] = []
    rows: list[dict[str, Any]] = []
    for term, (family, expected) in enumerate(
        zip(families, saved_matrices), start=1
    ):
        started = time.perf_counter()
        reduction, fragment = optimize_circuit_fragment(
            circuits,
            case.name,
            residual,
            family,
            1,
            term,
            starts,
            iterations,
        )
        replay_error = float(np.linalg.norm(fragment.matrix - expected, "fro"))
        if replay_error > 2.0e-8:
            raise RuntimeError(
                f"Saved LiH pilot replay failed at K={term}: {replay_error:.6g}"
            )
        before = float(np.vdot(residual, residual).real)
        residual = hermitian(residual - fragment.matrix)
        actual = before - float(np.vdot(residual, residual).real)
        if abs(actual - reduction) > 2.0e-7 * max(1.0, abs(reduction)):
            raise RuntimeError(f"Replay reduction identity failed at K={term}")
        fragments.append(fragment)
        rows.append(
            {
                "K": term,
                "family": family,
                "family_label": LABELS[family],
                "depth": 1,
                "saved_matrix_replay_Frobenius_error": replay_error,
                "Frobenius_distance": float(np.linalg.norm(residual, "fro")),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"[replay] K={term:02d} {LABELS[family]} "
            f"error={replay_error:.3e}",
            flush=True,
        )
    return fragments, rows


def build_exact_dense_prefix_decomposition(
    case,
    fragments: list[Fragment],
    spectral_max_evaluations: int,
) -> PrefixDecomposition:
    result = f3_r2(
        fragments,
        case.proxy_state,
        None,
        case.base_collector,
        spectral_max_evaluations=spectral_max_evaluations,
    )
    collector0 = hermitian(
        case.base_collector
        + sum(
            (fragment.one_matrix for fragment in fragments),
            start=np.zeros_like(case.target),
        )
    )

    def materialize_settings(alpha_values, collector):
        collector_values = np.linalg.eigvalsh(collector)
        collector_minimum = float(collector_values[0])
        collector_maximum = float(collector_values[-1])
        rows = [
            Setting(
                matrix=collector,
                centered_range=0.5 * (collector_maximum - collector_minimum),
                source_index=0,
                family="collector",
                interface=(
                    "standard_one_body_collector_dense_oracle"
                    if all(
                        fragment.family in STANDARD_F3_FAMILIES
                        for fragment in fragments
                    )
                    else "generalized_dense_collector_oracle"
                ),
                native_minimum=collector_minimum,
                native_maximum=collector_maximum,
            )
        ]
        for source, (fragment, alpha) in enumerate(
            zip(fragments, alpha_values), start=1
        ):
            alpha = float(alpha)
            matrix = hermitian(
                fragment.remainder_matrix - alpha * fragment.q_direction
            )
            native = np.real(
                fragment.remainder_diagonal - alpha * fragment.q_diagonal
            )
            rows.append(
                Setting(
                    matrix=matrix,
                    centered_range=diagonal_centered_norm(native),
                    source_index=source,
                    family=fragment.family,
                    interface=fragment.f3_interface,
                    native_minimum=float(np.min(native)),
                    native_maximum=float(np.max(native)),
                )
            )
        return rows

    zero_alpha = np.zeros(len(fragments), dtype=float)
    zero_settings = materialize_settings(zero_alpha, collector0)
    physical_before = float(sum(value.centered_range for value in zero_settings))
    alpha_by_source = np.asarray(result.alpha, dtype=float).copy()
    settings = materialize_settings(alpha_by_source, result.collector)
    physical_after = float(sum(value.centered_range for value in settings))
    setting_level_safeguard = bool(physical_after > physical_before)
    if setting_level_safeguard:
        alpha_by_source = zero_alpha
        settings = zero_settings
        physical_after = physical_before
        result.optimizer = {
            **result.optimizer,
            "selected": "zero_safeguard_at_materialized_setting_level",
        }
    result.optimizer = {
        **result.optimizer,
        "materialized_setting_level_safeguard_triggered": setting_level_safeguard,
        "materialized_setting_zero_value": physical_before,
        "materialized_setting_selected_value": physical_after,
    }
    reconstructed = sum(
        (setting.matrix for setting in settings),
        start=np.zeros_like(case.target),
    )
    expected = case.base_collector + sum(
        (fragment.matrix for fragment in fragments),
        start=np.zeros_like(case.target),
    )
    error = float(np.linalg.norm(reconstructed - expected, "fro"))
    if error > 2.0e-7:
        raise RuntimeError(f"Legacy exact-dense F3 reconstruction failed: {error:.6g}")
    return PrefixDecomposition(
        k=len(fragments),
        settings=settings,
        alpha_by_source=alpha_by_source,
        spectral_sum=physical_after,
        spectral_sum_before_r2=physical_before,
        reconstruction_error=error,
        f3_optimizer=result.optimizer,
    )


def _shallow_fragment_data(
    fragments: list[Fragment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the native rotations, Z tensors, and F3 one-body directions."""
    rotations = []
    tensors = []
    for fragment in fragments:
        if fragment.family != "shallow_rcdf":
            raise ValueError(
                "The shallow-collector builder accepts only standalone s-RCDF leaves"
            )
        metadata = fragment.candidate_metadata
        if "spatial_rotation" not in metadata or "z_tensor" not in metadata:
            raise ValueError("An s-RCDF fragment is missing its native rotation/Z tensor")
        rotations.append(np.asarray(metadata["spatial_rotation"], dtype=float))
        tensors.append(np.asarray(metadata["z_tensor"], dtype=float))
    rotations_array = (
        np.asarray(rotations, dtype=float)
        if rotations
        else np.empty((0, SPATIAL_ORBITALS, SPATIAL_ORBITALS), dtype=float)
    )
    tensors_array = (
        np.asarray(tensors, dtype=float)
        if tensors
        else np.empty((0, SPATIAL_ORBITALS, SPATIAL_ORBITALS), dtype=float)
    )
    directions = np.asarray(
        [np.sum(value, axis=1) - 0.5 * np.diag(value) for value in tensors_array],
        dtype=float,
    )
    if not len(directions):
        directions = np.empty((0, SPATIAL_ORBITALS), dtype=float)
    return rotations_array, tensors_array, directions


def _fixed_dimension_interaction_dense(
    constant: float, one_spatial: np.ndarray, two_openfermion: np.ndarray
) -> np.ndarray:
    """Return an eight-qubit dense operator even when trailing terms are zero.

    Some OpenFermion releases trim inactive trailing qubits for an
    ``InteractionOperator`` despite the requested qubit count.  Padding with
    identities restores the fixed register convention used by this benchmark.
    """
    # OpenFermion removes coefficients below a fixed absolute threshold while
    # constructing the sparse operator.  A shallow cover can legitimately
    # contain tiny cancellation leaves, so mapping each leaf independently at
    # its native scale would make this nominally linear map nonlinear.  A
    # common power-of-two rescaling keeps every nonzero coefficient above that
    # threshold without adding decimal-rounding error, after which we scale the
    # dense result back exactly in binary arithmetic.
    linear_scale = float(2**40)
    value = interaction_dense(
        float(constant) * linear_scale,
        np.asarray(one_spatial) * linear_scale,
        np.asarray(two_openfermion) * linear_scale,
    ) / linear_scale
    if value.shape == (DIMENSION, DIMENSION):
        return value
    if DIMENSION % value.shape[0] != 0:
        raise RuntimeError(f"Cannot pad interaction matrix of shape {value.shape}")
    factor = DIMENSION // value.shape[0]
    if factor & (factor - 1):
        raise RuntimeError(f"Non-qubit interaction padding factor: {factor}")
    return hermitian(np.kron(value, np.eye(factor, dtype=np.complex128)))


def _dense_collector_proxy_moments(
    case,
    rotations: np.ndarray,
    base_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HF variances and number covariances for a depth-limited setting bank."""
    state = np.asarray(case.proxy_state, dtype=np.complex128)
    zero_two = np.zeros_like(case.two_openfermion)
    base_variances = []
    base_covariances = []
    occupation_covariances = []
    for rotation, base in zip(rotations, base_matrices):
        base_action = np.asarray(base) @ state
        base_action -= np.vdot(state, base_action) * state
        number_actions = []
        for orbital in range(SPATIAL_ORBITALS):
            coefficient = np.zeros(SPATIAL_ORBITALS)
            coefficient[orbital] = 1.0
            one_spatial = rotation @ np.diag(coefficient) @ rotation.T
            number_action = (
                _fixed_dimension_interaction_dense(0.0, one_spatial, zero_two)
                @ state
            )
            number_action -= np.vdot(state, number_action) * state
            number_actions.append(number_action)
        number_actions = np.column_stack(number_actions)
        base_variances.append(max(float(np.vdot(base_action, base_action).real), 0.0))
        base_covariances.append(np.real(number_actions.conj().T @ base_action))
        covariance = np.real(number_actions.conj().T @ number_actions)
        occupation_covariances.append(0.5 * (covariance + covariance.T))
    return (
        np.asarray(base_variances, dtype=float),
        np.asarray(base_covariances, dtype=float),
        np.asarray(occupation_covariances, dtype=float),
    )


def _collector_proxy_variances(
    vector: np.ndarray,
    base_variances: np.ndarray,
    base_covariances: np.ndarray,
    occupation_covariances: np.ndarray,
) -> np.ndarray:
    eta = np.asarray(vector, dtype=float).reshape(len(base_variances), -1)
    values = (
        base_variances
        + 2.0 * np.einsum("ti,ti->t", base_covariances, eta)
        + np.einsum("ti,tij,tj->t", eta, occupation_covariances, eta)
    )
    return np.maximum(values, 0.0)


def _collector_metrics(
    vector: np.ndarray,
    base_full_values: np.ndarray,
    base_variances: np.ndarray,
    base_covariances: np.ndarray,
    occupation_covariances: np.ndarray,
    reference_shots: int,
) -> dict[str, Any]:
    eta = np.asarray(vector, dtype=float).reshape(len(base_full_values), -1)
    full_values = base_full_values + eta @ SPATIAL_OCCUPATIONS.T
    lambdas = 0.5 * (np.max(full_values, axis=1) - np.min(full_values, axis=1))
    variances = _collector_proxy_variances(
        vector, base_variances, base_covariances, occupation_covariances
    )
    shots = integer_range_allocation(reference_shots, lambdas)
    estimator_variance = 0.0
    for variance, count in zip(variances, shots):
        if count:
            estimator_variance += float(variance / count)
        elif variance > 1.0e-14:
            estimator_variance = float("inf")
            break
    return {
        "proxy_estimator_SE_at_reference_shots": math.sqrt(
            max(estimator_variance, 0.0)
        ),
        "centered_range_sum": float(np.sum(lambdas)),
        "coefficient_l1": float(np.sum(np.abs(vector))),
        "active_settings_at_reference_shots": int(np.sum(shots > 0)),
        "reference_shot_vector": shots.tolist(),
        "lambdas": lambdas,
        "proxy_variances": variances,
    }


def _variance_optimized_collector(
    particular: np.ndarray,
    null_basis: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    base_full_values: np.ndarray,
    base_variances: np.ndarray,
    base_covariances: np.ndarray,
    occupation_covariances: np.ndarray,
    reference_shots: int,
    cycles: int,
    equality_tolerance: float,
) -> np.ndarray:
    if null_basis.shape[1] == 0:
        return particular.copy()
    settings, orbitals = base_covariances.shape
    vector = particular.copy()
    for _ in range(cycles):
        metrics = _collector_metrics(
            vector,
            base_full_values,
            base_variances,
            base_covariances,
            occupation_covariances,
            reference_shots,
        )
        fractions = np.asarray(metrics["reference_shot_vector"], dtype=float)
        fractions /= reference_shots
        hessian = np.zeros((settings * orbitals, settings * orbitals))
        gradient = np.zeros(settings * orbitals)
        for setting in range(settings):
            block = slice(setting * orbitals, (setting + 1) * orbitals)
            if fractions[setting] > 0.0:
                hessian[block, block] = (
                    occupation_covariances[setting] / fractions[setting]
                )
                gradient[block] = base_covariances[setting] / fractions[setting]
        reduced = null_basis.T @ hessian @ null_basis
        reduced_rhs = -null_basis.T @ (hessian @ particular + gradient)
        ridge = 1.0e-11 * max(
            1.0, float(np.trace(reduced)) / max(len(reduced), 1)
        )
        reduced.flat[:: len(reduced) + 1] += ridge
        try:
            coordinates = la.solve(
                reduced, reduced_rhs, assume_a="sym", check_finite=False
            )
        except la.LinAlgError:
            coordinates = np.linalg.lstsq(reduced, reduced_rhs, rcond=1.0e-12)[0]
        candidate = particular + null_basis @ coordinates
        if not np.all(np.isfinite(candidate)):
            break
        vector = equality_project(
            candidate, design, target, rcond=equality_tolerance
        )
    return vector


def _range_optimized_collector(
    particular: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    equality_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    count = design.shape[1]
    objective = np.r_[np.zeros(count), np.ones(count)]
    inequalities = np.block(
        [
            [np.eye(count), -np.eye(count)],
            [-np.eye(count), -np.eye(count)],
        ]
    )
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(2 * count),
        A_eq=np.c_[design, np.zeros_like(design)],
        b_eq=target,
        bounds=[(None, None)] * count + [(0.0, None)] * count,
        method="highs",
        options={
            "dual_feasibility_tolerance": 1.0e-9,
            "primal_feasibility_tolerance": 1.0e-9,
        },
    )
    audit = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "iterations": int(result.nit),
    }
    if not result.success:
        return particular.copy(), audit
    return (
        equality_project(
            np.asarray(result.x[:count]),
            design,
            target,
            rcond=equality_tolerance,
        ),
        audit,
    )


def _collector_extra_schedule(minimum: int, maximum: int) -> list[int]:
    if minimum < 1 or maximum < minimum:
        raise ValueError("Collector augmentation requires 1 <= minimum <= maximum")
    # Four orbitals are small enough to test every integer L and therefore
    # certify that the first feasible bank is the minimum-depth-matched
    # augmentation, rather than merely the first point on a logarithmic grid.
    return list(range(minimum, maximum + 1))


def build_shallow_srcdf_prefix_decomposition(
    case,
    fragments: list[Fragment],
    rotation_depth: int,
    spectral_max_evaluations: int,
    *,
    collector_mode: str = "augment",
    collector_min_extra_leaves: int = 1,
    collector_max_extra_leaves: int = 8,
    collector_objective: str = "greedy",
    collector_proxy_state: str = "hf",
    collector_reference_shots: int = 3000,
    collector_proxy_cycles: int = 6,
    collector_equality_tolerance: float = 1.0e-10,
    collector_extra_angle_steps: int = 400,
) -> PrefixDecomposition:
    """Build an s-RCDF prefix with no independently diagonalized collector.

    Production ``augment`` mode adds at least one collector-tailored rotation at
    the same depth as the two-body leaves.  The bank is enlarged until its
    one-body projector dictionary spans Sym(m), and the collector is then
    redistributed under the audited equality constraint.  ``exact`` is retained
    only as an explicit diagnostic replay of the historical implementation.
    """
    if collector_mode not in {"exact", "absorb", "augment"}:
        raise ValueError(f"Unknown collector mode: {collector_mode}")
    if collector_objective not in {"greedy", "variance", "range", "hybrid"}:
        raise ValueError(f"Unknown collector objective: {collector_objective}")
    if collector_proxy_state != "hf":
        raise ValueError(
            "The four-orbital production wrapper permits only the deployable HF proxy"
        )
    if (
        rotation_depth < 1
        or collector_reference_shots < 1
        or collector_proxy_cycles < 1
        or collector_equality_tolerance <= 0.0
        or collector_extra_angle_steps < 1
    ):
        raise ValueError("Invalid shallow collector controls")

    source_rotations, _, directions = _shallow_fragment_data(fragments)
    if collector_mode == "exact":
        prefix = build_exact_dense_prefix_decomposition(
            case, fragments, spectral_max_evaluations
        )
        collector_spatial = np.asarray(case.one_spatial, dtype=float).copy()
        for alpha, rotation, direction in zip(
            prefix.alpha_by_source, source_rotations, directions
        ):
            collector_spatial += float(alpha) * (
                rotation @ np.diag(direction) @ rotation.T
            )
        eigenvalues, collector_rotation = np.linalg.eigh(collector_spatial)
        prefix.settings[0].rotation = collector_rotation
        prefix.settings[0].native_diagonal = (
            float(case.active_constant) + SPATIAL_OCCUPATIONS @ eigenvalues
        )
        prefix.settings[0].rotation_depth = None
        prefix.settings[0].setting_kind = "independent_exact_collector_diagnostic"
        prefix.settings[0].collector_linear_coefficients = eigenvalues
        for setting, fragment, rotation in zip(
            prefix.settings[1:], fragments, source_rotations
        ):
            setting.rotation = rotation
            setting.native_diagonal = np.real(
                fragment.remainder_diagonal
                - float(prefix.alpha_by_source[setting.source_index - 1])
                * fragment.q_diagonal
            )
            setting.rotation_depth = rotation_depth
            setting.setting_kind = "source_leaf"
            setting.collector_linear_coefficients = np.zeros(SPATIAL_ORBITALS)
        prefix.collector_audit = {
            "status": "PASS",
            "mode": "exact",
            "objective": "exact_eigendecomposition_diagnostic",
            "proxy_state": "not_applicable",
            "source_rotation_count": len(fragments),
            "extra_rotation_count": 0,
            "measurement_setting_count": len(prefix.settings),
            "dictionary_rank": 0,
            "symmetric_target_dimension": (
                SPATIAL_ORBITALS * (SPATIAL_ORBITALS + 1) // 2
            ),
            "independent_collector_setting_present": True,
            "matrix_reconstruction_residual_frobenius": prefix.reconstruction_error,
            "matrix_relative_reconstruction_residual": 0.0,
        }
        return prefix

    result = f3_r2(
        fragments,
        case.proxy_state,
        None,
        case.base_collector,
        spectral_max_evaluations=spectral_max_evaluations,
    )
    alpha = np.asarray(result.alpha, dtype=float).copy()
    collector_spatial = np.asarray(case.one_spatial, dtype=float).copy()
    for value, rotation, direction in zip(alpha, source_rotations, directions):
        collector_spatial += float(value) * (
            rotation @ np.diag(direction) @ rotation.T
        )
    collector_spatial = np.real_if_close(
        0.5 * (collector_spatial + collector_spatial.T)
    ).real
    source_base_matrices = np.asarray(
        [
            hermitian(fragment.remainder_matrix - float(value) * fragment.q_direction)
            for fragment, value in zip(fragments, alpha)
        ],
        dtype=np.complex128,
    )
    if not len(source_base_matrices):
        source_base_matrices = np.empty((0, DIMENSION, DIMENSION), dtype=np.complex128)
    source_base_values = np.asarray(
        [
            np.real(fragment.remainder_diagonal - float(value) * fragment.q_diagonal)
            for fragment, value in zip(fragments, alpha)
        ],
        dtype=float,
    )
    if not len(source_base_values):
        source_base_values = np.empty((0, DIMENSION), dtype=float)

    target = symmetric_vector(collector_spatial)
    target_dimension = len(target)
    attempts = []
    selected_bank = None
    counts = (
        [0]
        if collector_mode == "absorb"
        else _collector_extra_schedule(
            collector_min_extra_leaves, collector_max_extra_leaves
        )
    )
    last_feasibility = None
    for extra_count in counts:
        extra_rotations, greedy_extra, extra_audit = (
            collector_tailored_extra_shallow_rotations(
                source_rotations,
                collector_spatial,
                rotation_depth,
                extra_count,
                f"{case.name}|K{len(fragments)}|dR{rotation_depth}",
                collector_extra_angle_steps,
            )
        )
        rotations = np.concatenate((source_rotations, extra_rotations), axis=0)
        design = collector_dictionary(rotations)
        rank, condition, singular_values, null_basis = dictionary_diagnostics(
            design, collector_equality_tolerance
        )
        particular = np.linalg.lstsq(
            design, target, rcond=collector_equality_tolerance
        )[0]
        particular = equality_project(
            particular,
            design,
            target,
            rcond=collector_equality_tolerance,
        )
        relative_residual = float(
            np.linalg.norm(design @ particular - target)
            / max(np.linalg.norm(target), 1.0)
        )
        greedy_trial, greedy_source_fill_residual = greedy_exact_fill(
            design,
            target,
            len(source_rotations) * SPATIAL_ORBITALS,
            greedy_extra,
            collector_equality_tolerance,
        )
        greedy_relative_residual = float(
            np.linalg.norm(design @ greedy_trial - target)
            / max(np.linalg.norm(target), 1.0)
        )
        feasibility = {
            "extra_rotation_count": extra_count,
            "dictionary_rank": rank,
            "dictionary_columns": int(design.shape[1]),
            "symmetric_target_dimension": target_dimension,
            "dictionary_condition_on_numerical_range": condition,
            "smallest_retained_singular_value": (
                float(singular_values[rank - 1]) if rank else 0.0
            ),
            "relative_reconstruction_residual": relative_residual,
            "greedy_exact_fill_relative_reconstruction_residual": (
                greedy_relative_residual
            ),
            "full_symmetric_dictionary_rank": rank == target_dimension,
        }
        attempts.append(feasibility)
        last_feasibility = feasibility
        feasible = relative_residual <= collector_equality_tolerance
        if collector_mode == "augment":
            feasible = feasible and rank == target_dimension
        if collector_objective == "greedy":
            feasible = feasible and (
                greedy_relative_residual <= collector_equality_tolerance
            )
        if feasible:
            selected_bank = (
                rotations,
                design,
                null_basis,
                particular,
                greedy_extra,
                extra_audit,
                condition,
                singular_values,
                rank,
                extra_count,
                greedy_trial,
                greedy_source_fill_residual,
            )
            break
    if selected_bank is None:
        audit = {
            **(last_feasibility or {}),
            "mode": collector_mode,
            "attempted_extra_counts": attempts,
            "equality_tolerance": collector_equality_tolerance,
        }
        raise ShallowCollectorRepresentationError(audit)

    (
        rotations,
        design,
        null_basis,
        particular,
        greedy_extra,
        extra_audit,
        condition,
        singular_values,
        rank,
        extra_count,
        greedy_vector,
        source_fill_residual,
    ) = selected_bank
    base_matrices = np.concatenate(
        (
            source_base_matrices,
            np.zeros((extra_count, DIMENSION, DIMENSION), dtype=np.complex128),
        ),
        axis=0,
    )
    base_full_values = np.concatenate(
        (source_base_values, np.zeros((extra_count, DIMENSION), dtype=float)),
        axis=0,
    )
    base_variances, base_covariances, occupation_covariances = (
        _dense_collector_proxy_moments(case, rotations, base_matrices)
    )
    variance_vector = _variance_optimized_collector(
        particular,
        null_basis,
        design,
        target,
        base_full_values,
        base_variances,
        base_covariances,
        occupation_covariances,
        collector_reference_shots,
        collector_proxy_cycles,
        collector_equality_tolerance,
    )
    range_vector, range_audit = _range_optimized_collector(
        particular, design, target, collector_equality_tolerance
    )
    raw_candidates: list[tuple[str, np.ndarray]] = [
        ("minimum_norm", particular),
        ("collector_tailored_greedy_exact_fill", greedy_vector),
        ("proxy_variance_nullspace", variance_vector),
        ("l1_range_nullspace", range_vector),
    ]
    if collector_objective == "hybrid":
        for weight in np.linspace(0.1, 0.9, 9):
            raw_candidates.append(
                (
                    f"hybrid_interpolation_{weight:.1f}",
                    (1.0 - weight) * variance_vector + weight * range_vector,
                )
            )
        for weight in (0.25, 0.5, 0.75):
            raw_candidates.extend(
                (
                    (
                        f"greedy_variance_interpolation_{weight:.2f}",
                        (1.0 - weight) * greedy_vector + weight * variance_vector,
                    ),
                    (
                        f"greedy_range_interpolation_{weight:.2f}",
                        (1.0 - weight) * greedy_vector + weight * range_vector,
                    ),
                )
            )
    candidate_rows = []
    for name, vector in raw_candidates:
        feasible_vector = equality_project(
            np.asarray(vector),
            design,
            target,
            rcond=collector_equality_tolerance,
        )
        candidate_rows.append(
            {
                "name": name,
                "vector": feasible_vector,
                **_collector_metrics(
                    feasible_vector,
                    base_full_values,
                    base_variances,
                    base_covariances,
                    occupation_covariances,
                    collector_reference_shots,
                ),
            }
        )
    if collector_objective == "greedy":
        selected = next(
            row
            for row in candidate_rows
            if row["name"] == "collector_tailored_greedy_exact_fill"
        )
    elif collector_objective == "variance":
        selected = min(
            [row for row in candidate_rows if row["name"] != "l1_range_nullspace"],
            key=lambda row: (
                row["proxy_estimator_SE_at_reference_shots"],
                row["centered_range_sum"],
            ),
        )
    elif collector_objective == "range":
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
    eta = vector.reshape(len(rotations), SPATIAL_ORBITALS)
    reconstructed_collector = collector_matrix(rotations, eta)
    matrix_residual = float(
        np.linalg.norm(reconstructed_collector - collector_spatial, "fro")
    )
    matrix_relative = matrix_residual / max(
        float(np.linalg.norm(collector_spatial, "fro")), 1.0
    )
    if matrix_relative > 5.0 * collector_equality_tolerance:
        raise ShallowCollectorRepresentationError(
            {
                "relative_reconstruction_residual": matrix_relative,
                "dictionary_rank": rank,
                "symmetric_target_dimension": target_dimension,
            }
        )

    zero_two = np.zeros_like(case.two_openfermion)
    settings = []
    maximum_spectral_audit = 0.0
    for index, (rotation, coefficients, base_matrix, base_values) in enumerate(
        zip(rotations, eta, base_matrices, base_full_values)
    ):
        one_spatial = rotation @ np.diag(coefficients) @ rotation.T
        constant = float(case.active_constant) if index == 0 else 0.0
        matrix = hermitian(
            base_matrix
            + _fixed_dimension_interaction_dense(constant, one_spatial, zero_two)
        )
        native = np.real(base_values + SPATIAL_OCCUPATIONS @ coefficients + constant)
        spectrum = np.linalg.eigvalsh(matrix)
        scale = max(
            1.0,
            float(np.max(np.abs(spectrum))),
            float(np.max(np.abs(native))),
        )
        spectral_audit = float(
            np.max(np.abs(np.sort(spectrum) - np.sort(native)))
        )
        maximum_spectral_audit = max(maximum_spectral_audit, spectral_audit / scale)
        settings.append(
            Setting(
                matrix=matrix,
                centered_range=diagonal_centered_norm(native),
                source_index=index + 1,
                family=(
                    "shallow_rcdf"
                    if index < len(fragments)
                    else "shallow_collector_extra"
                ),
                interface=(
                    "standard_shallow_tensor_native_f3_r2_with_redistributed_collector"
                    if index < len(fragments)
                    else "collector_tailored_depth_matched_one_body_leaf"
                ),
                native_minimum=float(np.min(native)),
                native_maximum=float(np.max(native)),
                rotation=rotation,
                native_diagonal=native,
                rotation_depth=rotation_depth,
                setting_kind=(
                    "source_leaf_with_redistributed_collector"
                    if index < len(fragments)
                    else "collector_tailored_extra_leaf"
                ),
                collector_linear_coefficients=coefficients,
            )
        )
    expected = case.base_collector + sum(
        (fragment.matrix for fragment in fragments),
        start=np.zeros_like(case.target),
    )
    reconstructed = sum(
        (setting.matrix for setting in settings), start=np.zeros_like(case.target)
    )
    reconstruction_error = float(np.linalg.norm(reconstructed - expected, "fro"))
    # The frozen four-orbital artifacts were materialized through OpenFermion's
    # sparse map one leaf at a time.  Re-materializing the same one-body sum can
    # differ at a few 1e-7 in absolute dense Frobenius norm, while the spatial
    # collector equality above is enforced at 1e-10 relative tolerance.
    if reconstruction_error > 1.0e-6:
        raise RuntimeError(
            f"Shallow collector F3 reconstruction failed: {reconstruction_error:.6g}"
        )
    if maximum_spectral_audit > 2.0e-7:
        raise RuntimeError(
            "A shallow collector setting/native diagonal mismatch exceeded tolerance: "
            f"{maximum_spectral_audit:.3e}"
        )
    audit_candidates = [
        {
            key: value
            for key, value in row.items()
            if key not in {"vector", "lambdas", "proxy_variances"}
        }
        for row in candidate_rows
    ]
    collector_audit = {
        "status": "PASS",
        "mode": collector_mode,
        "objective": collector_objective,
        "proxy_state": collector_proxy_state,
        "reference_shots": collector_reference_shots,
        "source_rotation_count": len(source_rotations),
        "extra_rotation_count": extra_count,
        "measurement_setting_count": len(settings),
        "dictionary_rank": rank,
        "dictionary_columns": int(design.shape[1]),
        "dictionary_nullity": int(design.shape[1] - rank),
        "symmetric_target_dimension": target_dimension,
        "dictionary_condition_on_numerical_range": condition,
        "smallest_retained_singular_value": (
            float(singular_values[rank - 1]) if rank else 0.0
        ),
        "attempted_extra_counts": attempts,
        "extra_rotation_construction": extra_audit,
        "equality_tolerance": collector_equality_tolerance,
        "relative_reconstruction_residual": matrix_relative,
        "matrix_reconstruction_residual_frobenius": matrix_residual,
        "matrix_relative_reconstruction_residual": matrix_relative,
        "many_body_reconstruction_residual_frobenius": reconstruction_error,
        "maximum_native_diagonal_spectral_relative_audit": maximum_spectral_audit,
        "selected_candidate": selected["name"],
        "selected_proxy_estimator_SE_at_reference_shots": selected[
            "proxy_estimator_SE_at_reference_shots"
        ],
        "selected_centered_range_sum": selected["centered_range_sum"],
        "selected_reference_shot_vector": selected["reference_shot_vector"],
        "range_linear_program": range_audit,
        "greedy_existing_bank_fill_relative_residual": source_fill_residual,
        "candidate_metrics": audit_candidates,
        "coefficients": eta.tolist(),
        "all_measurement_settings_depth_limited": True,
        "independent_collector_setting_present": False,
    }
    return PrefixDecomposition(
        k=len(fragments),
        settings=settings,
        alpha_by_source=alpha,
        spectral_sum=float(sum(setting.centered_range for setting in settings)),
        spectral_sum_before_r2=float(result.spectral_norm_sum_before_r2),
        reconstruction_error=reconstruction_error,
        f3_optimizer={
            **result.optimizer,
            "collector_redistribution_selected_candidate": selected["name"],
            "collector_redistribution_setting_level_range": selected[
                "centered_range_sum"
            ],
        },
        collector_audit=collector_audit,
    )


def build_minimum_depth_zero_rank_shallow_collector(
    case,
    rotation_depths: tuple[int, ...] | list[int],
    spectral_max_evaluations: int,
    **collector_options,
) -> tuple[PrefixDecomposition, int]:
    """Select the smallest d_R whose K=0 augmented dictionary is full rank."""
    depths = tuple(sorted(set(int(value) for value in rotation_depths)))
    if not depths or any(value < 1 for value in depths):
        raise ValueError("The zero-rank collector depth grid must be positive")
    attempts = []
    last_error = None
    for depth in depths:
        try:
            prefix = build_shallow_srcdf_prefix_decomposition(
                case,
                [],
                depth,
                spectral_max_evaluations,
                **collector_options,
            )
        except ShallowCollectorRepresentationError as error:
            last_error = error
            attempts.append(
                {"rotation_depth": depth, "status": "INFEASIBLE", **error.audit}
            )
            continue
        attempts.append(
            {
                "rotation_depth": depth,
                "status": "PASS",
                "extra_rotation_count": prefix.collector_audit.get(
                    "extra_rotation_count", 0
                ),
                "dictionary_rank": prefix.collector_audit.get("dictionary_rank", 0),
                "matrix_relative_reconstruction_residual": (
                    prefix.collector_audit.get(
                        "matrix_relative_reconstruction_residual", 0.0
                    )
                ),
            }
        )
        prefix.collector_audit.update(
            {
                "zero_rank_depth_selection": "smallest_strictly_feasible_depth",
                "zero_rank_depth_grid": list(depths),
                "zero_rank_depth_attempts": attempts,
                "selected_zero_rank_depth": depth,
            }
        )
        return prefix, depth
    if last_error is not None:
        audit = {**last_error.audit, "zero_rank_depth_attempts": attempts}
        raise ShallowCollectorRepresentationError(audit)
    raise RuntimeError("No zero-rank collector depth candidate was evaluated")


def _agpd_shallow_collector_cache_key(case) -> tuple[Any, ...]:
    one = np.ascontiguousarray(np.asarray(case.one_spatial, dtype="<f8"))
    digest = hashlib.sha256(one.view(np.uint8)).hexdigest()
    return (
        str(case.name),
        float(case.active_constant),
        digest,
        AGPD_SHALLOW_COLLECTOR_DEPTHS,
        AGPD_SHALLOW_COLLECTOR_MIN_EXTRA_LEAVES,
        AGPD_SHALLOW_COLLECTOR_MAX_EXTRA_LEAVES,
        AGPD_SHALLOW_COLLECTOR_EQUALITY_TOLERANCE,
        AGPD_SHALLOW_COLLECTOR_MATRIX_TOLERANCE,
        AGPD_SHALLOW_COLLECTOR_ANGLE_STEPS,
    )


def _agpd_shallow_one_body_cover(
    case, spectral_max_evaluations: int
) -> tuple[PrefixDecomposition, int]:
    """Return the cached exact shallow cover of the molecular one-body reference."""
    key = _agpd_shallow_collector_cache_key(case)
    cached = _AGPD_SHALLOW_COLLECTOR_CACHE.get(key)
    if cached is None:
        cached = build_minimum_depth_zero_rank_shallow_collector(
            case,
            AGPD_SHALLOW_COLLECTOR_DEPTHS,
            spectral_max_evaluations,
            collector_mode="augment",
            collector_min_extra_leaves=AGPD_SHALLOW_COLLECTOR_MIN_EXTRA_LEAVES,
            collector_max_extra_leaves=AGPD_SHALLOW_COLLECTOR_MAX_EXTRA_LEAVES,
            collector_objective="greedy",
            collector_proxy_state="hf",
            collector_reference_shots=3000,
            collector_proxy_cycles=6,
            collector_equality_tolerance=AGPD_SHALLOW_COLLECTOR_EQUALITY_TOLERANCE,
            collector_extra_angle_steps=AGPD_SHALLOW_COLLECTOR_ANGLE_STEPS,
        )
        if cached[0].collector_audit.get("independent_collector_setting_present"):
            raise RuntimeError("AGPD shallow cover unexpectedly retained a collector")
        _AGPD_SHALLOW_COLLECTOR_CACHE[key] = cached
    return cached


def _copy_setting(setting: Setting, **updates: Any) -> Setting:
    values = {
        "matrix": setting.matrix,
        "centered_range": setting.centered_range,
        "source_index": setting.source_index,
        "family": setting.family,
        "interface": setting.interface,
        "native_minimum": setting.native_minimum,
        "native_maximum": setting.native_maximum,
        "rotation": setting.rotation,
        "native_diagonal": setting.native_diagonal,
        "rotation_depth": setting.rotation_depth,
        "setting_kind": setting.setting_kind,
        "collector_linear_coefficients": setting.collector_linear_coefficients,
        "range_domain": setting.range_domain,
        "full_fock_centered_range": setting.full_fock_centered_range,
        "sector_spectral_centered_range": (
            setting.sector_spectral_centered_range
        ),
        "allocation_centered_range": setting.allocation_centered_range,
        "sector_minimum": setting.sector_minimum,
        "sector_maximum": setting.sector_maximum,
        "sector_dimension": setting.sector_dimension,
        "sector_leakage_frobenius": setting.sector_leakage_frobenius,
        "leakage_operator_bound": setting.leakage_operator_bound,
        "sector_leakage_relative_frobenius": (
            setting.sector_leakage_relative_frobenius
        ),
        "sector_range_eligible": setting.sector_range_eligible,
    }
    values.update(updates)
    return Setting(**values)


def _setting_with_sector_range(
    setting: Setting,
    sector_indices: np.ndarray,
    *,
    leakage_tolerance: float = AGPD_SECTOR_LEAKAGE_TOLERANCE,
) -> Setting:
    """Return a setting whose allocation range is certified in the sector."""
    full_range = (
        setting.centered_range
        if setting.full_fock_centered_range is None
        else setting.full_fock_centered_range
    )
    metrics = _cached_sector_range_metrics(
        setting.matrix,
        sector_indices,
        full_fock_centered_range=float(full_range),
        leakage_tolerance=leakage_tolerance,
    )
    sector_eligible = bool(metrics["passes_leakage_check"])
    # Fail closed without aborting the four-family search: a leaky candidate
    # receives the independently valid full-Fock range.  Only settings below
    # the declared leakage tolerance may use the tighter projected-sector
    # bound (including its small leakage guard).
    allocation_range = (
        float(metrics["allocation_centered_range"])
        if sector_eligible
        else float(metrics["full_fock_centered_range"])
    )
    range_domain = (
        "fixed_particle_number_sector"
        if sector_eligible
        else "full_fock_fallback_due_to_sector_leakage"
    )
    return _copy_setting(
        setting,
        centered_range=allocation_range,
        range_domain=range_domain,
        full_fock_centered_range=float(metrics["full_fock_centered_range"]),
        sector_spectral_centered_range=float(
            metrics["sector_spectral_centered_range"]
        ),
        allocation_centered_range=allocation_range,
        sector_minimum=float(metrics["sector_minimum"]),
        sector_maximum=float(metrics["sector_maximum"]),
        sector_dimension=int(metrics["sector_dimension"]),
        sector_leakage_frobenius=float(metrics["leakage_frobenius"]),
        leakage_operator_bound=float(metrics["leakage_operator_bound"]),
        sector_leakage_relative_frobenius=float(
            metrics["leakage_relative_frobenius"]
        ),
        sector_range_eligible=sector_eligible,
    )


def build_prefix_decomposition(
    case,
    fragments: list[Fragment],
    spectral_max_evaluations: int,
) -> PrefixDecomposition:
    """Build the production AGPD prefix with no independent collector circuit.

    The exact molecular scalar-plus-one-body reference is deployed through a
    cached minimum-depth shallow one-body cover.  Every accepted source remains
    a complete native rotated-diagonal setting.  Consequently no conjugated
    non-Gaussian direction is transferred to a dense Hermitian collector.
    """
    cover, collector_depth = _agpd_shallow_one_body_cover(
        case, spectral_max_evaluations
    )
    settings = []
    for setting in cover.settings:
        settings.append(
            _setting_with_sector_range(
                _copy_setting(
                    setting,
                    source_index=0,
                    family="agpd_shallow_one_body",
                    interface="agpd_depth_bounded_one_body_cover",
                    setting_kind="agpd_shallow_one_body_leaf",
                ),
                case.sector_indices,
            )
        )
    for source_index, fragment in enumerate(fragments, start=1):
        native = np.real(np.asarray(fragment.diagonal, dtype=float))
        source_depth = fragment.candidate_metadata.get("ansatz_depth")
        if source_depth is None and fragment.family == "shallow_rcdf":
            source_depth = fragment.candidate_metadata.get("shallow_rcdf_depth")
        settings.append(
            _setting_with_sector_range(
                Setting(
                # Preserve the fragment matrix identity so its certified
                # sector audit can be reused across depth/family trials.
                matrix=np.asarray(fragment.matrix, dtype=np.complex128),
                centered_range=diagonal_centered_norm(native),
                source_index=source_index,
                family=fragment.family,
                interface="native_shallow_source_no_collector_transfer",
                native_minimum=float(np.min(native)),
                native_maximum=float(np.max(native)),
                native_diagonal=native,
                rotation_depth=(
                    None if source_depth is None else int(source_depth)
                ),
                setting_kind="source_leaf",
                ),
                case.sector_indices,
            )
        )
    expected = case.base_collector + sum(
        (fragment.matrix for fragment in fragments),
        start=np.zeros_like(case.target),
    )
    reconstructed = sum(
        (setting.matrix for setting in settings),
        start=np.zeros_like(case.target),
    )
    error = float(np.linalg.norm(reconstructed - expected, "fro"))
    if error > AGPD_SHALLOW_COLLECTOR_MATRIX_TOLERANCE:
        raise RuntimeError(f"AGPD shallow-cover reconstruction failed: {error:.6g}")
    spectral_sum = float(sum(setting.centered_range for setting in settings))
    full_fock_spectral_sum = float(
        sum(float(setting.full_fock_centered_range) for setting in settings)
    )
    sector_spectral_sum = float(
        sum(float(setting.sector_spectral_centered_range) for setting in settings)
    )
    maximum_sector_leakage = max(
        (
            float(setting.sector_leakage_relative_frobenius)
            for setting in settings
        ),
        default=0.0,
    )
    range_domain_counts = {
        domain: sum(setting.range_domain == domain for setting in settings)
        for domain in (
            "fixed_particle_number_sector",
            "full_fock_fallback_due_to_sector_leakage",
        )
    }
    sector_range_setting_count = range_domain_counts[
        "fixed_particle_number_sector"
    ]
    full_fock_fallback_setting_count = range_domain_counts[
        "full_fock_fallback_due_to_sector_leakage"
    ]
    aggregate_range_domain = (
        "fixed_particle_number_sector"
        if full_fock_fallback_setting_count == 0
        else "mixed_sector_and_full_fock_fallback"
    )
    cover_audit = {
        **cover.collector_audit,
        "method": "AGPD",
        "policy": "fixed_shallow_one_body_cover_and_complete_native_sources",
        "source_transfer_policy": "disabled_for_all_families",
        "source_transfer_count": 0,
        "collector_depth": int(collector_depth),
        "cover_setting_count": len(cover.settings),
        "source_setting_count": len(fragments),
        "measurement_setting_count": len(settings),
        "independent_collector_setting_present": False,
        "complete_source_matrices_retained": True,
        "centered_range_domain": aggregate_range_domain,
        "sector_dimension": int(len(case.sector_indices)),
        "all_setting_ranges_sector_restricted": all(
            setting.range_domain == "fixed_particle_number_sector"
            for setting in settings
        ),
        "all_settings_pass_sector_leakage_check": bool(
            maximum_sector_leakage <= AGPD_SECTOR_LEAKAGE_TOLERANCE
        ),
        "sector_range_setting_count": int(sector_range_setting_count),
        "full_fock_fallback_setting_count": int(
            full_fock_fallback_setting_count
        ),
        "range_domain_counts": range_domain_counts,
        "all_settings_sector_safe_or_full_fock_fallback": bool(
            sector_range_setting_count + full_fock_fallback_setting_count
            == len(settings)
        ),
        "maximum_sector_leakage_relative_frobenius": maximum_sector_leakage,
        "sector_leakage_tolerance": AGPD_SECTOR_LEAKAGE_TOLERANCE,
        "full_fock_centered_range_sum": full_fock_spectral_sum,
        "sector_centered_range_sum": spectral_sum,
        "sector_spectral_centered_range_sum": sector_spectral_sum,
        "sector_leakage_guard_increment_sum": spectral_sum - sector_spectral_sum,
        "sector_range_sum_reduction_fraction": (
            0.0
            if full_fock_spectral_sum <= 0.0
            else 1.0 - spectral_sum / full_fock_spectral_sum
        ),
        "matrix_reconstruction_residual_frobenius": error,
        "matrix_reconstruction_tolerance_frobenius": (
            AGPD_SHALLOW_COLLECTOR_MATRIX_TOLERANCE
        ),
    }
    return PrefixDecomposition(
        k=len(fragments),
        settings=settings,
        alpha_by_source=np.zeros(len(fragments), dtype=float),
        spectral_sum=spectral_sum,
        spectral_sum_before_r2=spectral_sum,
        reconstruction_error=error,
        f3_optimizer={
            "status": "disabled",
            "selected": "no_source_to_collector_transfer",
            "reason": (
                "A conjugated diagonal one-body direction is not generally a "
                "one-body operator for non-Gaussian AGPD sources."
            ),
            "shallow_one_body_cover_depth": int(collector_depth),
            "shallow_one_body_cover_settings": len(cover.settings),
        },
        collector_audit=cover_audit,
    )


def build_all_prefixes(case, fragments, spectral_max_evaluations):
    values = []
    for k in range(len(fragments) + 1):
        print(f"[AGPD] rebuilding shallow-cover prefix K={k:02d}", flush=True)
        values.append(
            build_prefix_decomposition(
                case, fragments[:k], spectral_max_evaluations
            )
        )
    return values


def generate_stabilizer_probes(
    sector_indices: np.ndarray, seed: int = STABILIZER_PROBE_SEED
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return the frozen 500-probe grouped stabilizer design.

    The physical sector may contain more than 28 determinants (for example,
    the four-electron eight-qubit sector contains 70).  We sample 28 determinant
    groups without replacement and 118 unordered two-determinant groups with
    four Clifford phases each.  Whole groups are assigned to a 300/100/100
    train/validation/test split.  The same seed and sector therefore produce
    exactly the same probes for every molecule in that sector.
    """
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    if len(sector_indices) < 28:
        raise ValueError("The common stabilizer design requires at least 28 sector states")
    determinant_order = rng.permutation(len(sector_indices))[:28]
    determinant_split: dict[int, str] = {}
    for position, slot in enumerate(determinant_order):
        determinant_split[int(slot)] = (
            "train" if position < 20 else "validation" if position < 24 else "test"
        )
    for slot, sector_slot in enumerate(determinant_order):
        index = int(sector_indices[int(sector_slot)])
        vector = np.zeros(DIMENSION, dtype=np.complex128)
        vector[index] = 1.0
        vectors.append(vector)
        records.append(
            {
                "probe_id": len(records),
                "group_id": f"determinant_{int(index):03d}",
                "group_size": 1,
                "probe_type": "determinant",
                "basis_a": format(int(index), f"0{N_QUBITS}b"),
                "basis_b": "",
                "relative_phase_radians": 0.0,
                "split": determinant_split[int(sector_slot)],
                "probe_seed": seed,
            }
        )
    pair_candidates = [
        (int(a), int(b))
        for left, a in enumerate(sector_indices)
        for b in sector_indices[left + 1 :]
    ]
    chosen_pairs = rng.choice(len(pair_candidates), size=118, replace=False)
    pair_order = rng.permutation(118)
    pair_split: dict[int, str] = {}
    for position, slot in enumerate(pair_order):
        pair_split[int(slot)] = (
            "train" if position < 70 else "validation" if position < 94 else "test"
        )
    for pair_slot, candidate_index in enumerate(chosen_pairs):
        a, b = pair_candidates[int(candidate_index)]
        group_id = f"pair_{a:03d}_{b:03d}"
        for phase in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            vector = np.zeros(DIMENSION, dtype=np.complex128)
            vector[a] = 1.0 / math.sqrt(2.0)
            vector[b] = np.exp(1j * phase) / math.sqrt(2.0)
            vectors.append(vector)
            records.append(
                {
                    "probe_id": len(records),
                    "group_id": group_id,
                    "group_size": 4,
                    "probe_type": "two_determinant_equal_amplitude",
                    "basis_a": format(a, f"0{N_QUBITS}b"),
                    "basis_b": format(b, f"0{N_QUBITS}b"),
                    "relative_phase_radians": phase,
                    "split": pair_split[pair_slot],
                    "probe_seed": seed,
                }
            )
    states = np.column_stack(vectors)
    norms = np.sum(np.abs(states) ** 2, axis=0)
    outside = np.setdiff1d(np.arange(DIMENSION), sector_indices)
    leakage = float(np.max(np.sum(np.abs(states[outside]) ** 2, axis=0)))
    if float(np.max(np.abs(norms - 1.0))) > 1.0e-12 or leakage > 1.0e-12:
        raise RuntimeError("Generated calibration probes failed normalization/sector audit")
    split_counts = {
        value: sum(record["split"] == value for record in records)
        for value in ("train", "validation", "test")
    }
    if split_counts != STABILIZER_SPLIT_COUNTS:
        raise RuntimeError(f"Unexpected grouped probe split: {split_counts}")
    if states.shape[1] != STABILIZER_PROBE_COUNT:
        raise RuntimeError(f"Unexpected probe count: {states.shape[1]}")
    group_splits: dict[str, set[str]] = {}
    for record in records:
        group_splits.setdefault(str(record["group_id"]), set()).add(str(record["split"]))
    if any(len(values) != 1 for values in group_splits.values()):
        raise RuntimeError("A stabilizer group crosses train/validation/test splits")
    return states, records


def integer_range_allocation(total_shots: int, lambdas: np.ndarray) -> np.ndarray:
    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.ndim != 1 or not np.all(np.isfinite(lambdas)):
        raise ValueError("lambdas must be a finite one-dimensional array")
    if np.any(lambdas < 0.0):
        raise ValueError("allocation ranges must be non-negative")
    # Only a mathematically exact zero may be exempted from measurement.  A
    # small positive leakage guard still represents nonzero variance and must
    # receive shots.
    active = np.flatnonzero(lambdas > 0.0)
    if total_shots < len(active):
        raise ValueError(
            f"T={total_shots} is smaller than {len(active)} nonconstant settings"
        )
    shots = np.zeros(len(lambdas), dtype=np.int64)
    if not len(active):
        return shots
    shots[active] = 1
    remaining = total_shots - len(active)
    if remaining:
        raw = remaining * lambdas[active] / float(np.sum(lambdas[active]))
        extra = np.floor(raw).astype(np.int64)
        shots[active] += extra
        remainder = remaining - int(np.sum(extra))
        order = np.lexsort((active, -(raw - extra)))
        shots[active[order[:remainder]]] += 1
    if int(np.sum(shots)) != total_shots:
        raise RuntimeError("Integer allocation did not conserve the shot budget")
    return shots


def fit_nonnegative_squared(feature: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Fit target = theta * feature through the origin and return theta, sqrt(theta)."""
    feature = np.asarray(feature, dtype=float)
    target = np.asarray(target, dtype=float)
    denominator = float(np.dot(feature, feature))
    if denominator <= 1.0e-30:
        raise RuntimeError("Degenerate calibration feature")
    theta = max(0.0, float(np.dot(feature, target)) / denominator)
    return theta, math.sqrt(theta)


def equal_stratum_ratio_fit(
    rows: list[dict[str, Any]],
    strata: tuple[str, ...],
    numerator: str,
    denominator: str,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Fit one nonnegative ratio per stratum, then average strata equally."""
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        value = float(row[denominator])
        if value <= 0.0:
            raise RuntimeError(f"Nonpositive ratio denominator in calibration row: {value}")
        key = tuple(row[field] for field in strata)
        grouped.setdefault(key, []).append(float(row[numerator]) / value)
    if not grouped:
        raise RuntimeError("No eligible calibration strata")
    stratum_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ratios = np.asarray(grouped[key], dtype=float)
        theta = max(0.0, float(np.mean(ratios)))
        record = {field: key[index] for index, field in enumerate(strata)}
        record.update(
            {
                "rows": int(len(ratios)),
                "theta_ratio_mean": theta,
                "ratio_minimum": float(np.min(ratios)),
                "ratio_median": float(np.median(ratios)),
                "ratio_maximum": float(np.max(ratios)),
            }
        )
        stratum_rows.append(record)
    theta_global = max(
        0.0, float(np.mean([row["theta_ratio_mean"] for row in stratum_rows]))
    )
    return theta_global, math.sqrt(theta_global), stratum_rows


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual = target - prediction
    ss_res = float(np.dot(residual, residual))
    centered = target - float(np.mean(target))
    ss_total = float(np.dot(centered, centered))
    return {
        "rows": int(len(target)),
        "RMSE": math.sqrt(ss_res / max(len(target), 1)),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": 1.0 - ss_res / ss_total if ss_total > 1.0e-30 else float("nan"),
    }


def calibrate_weights(
    case,
    fragments: list[Fragment],
    prefixes: list[PrefixDecomposition],
    states: np.ndarray,
    probe_records: list[dict[str, Any]],
    shot_grid: tuple[int, ...],
    fit_k_values: tuple[int, ...] = CALIBRATION_K_VALUES,
    independent_approximations: list[np.ndarray] | None = None,
):
    fit_k_values = tuple(sorted(set(int(value) for value in fit_k_values)))
    shot_grid = tuple(sorted(set(int(value) for value in shot_grid)))
    if fit_k_values != CALIBRATION_K_VALUES:
        raise ValueError(
            f"Active calibration requires K={CALIBRATION_K_VALUES}, got {fit_k_values}"
        )
    if shot_grid != SHOT_GRID:
        raise ValueError(f"Active calibration requires T={SHOT_GRID}, got {shot_grid}")
    available_k = tuple(prefix.k for prefix in prefixes)
    if available_k != tuple(range(max(fit_k_values) + 1)):
        raise ValueError(
            "Calibration prefixes must be contiguous K=0,...,10 so K=0 can be "
            f"audited while only K={fit_k_values} are fitted; got {available_k}"
        )
    if len(probe_records) != STABILIZER_PROBE_COUNT:
        raise ValueError(
            f"Active calibration requires {STABILIZER_PROBE_COUNT} probes"
        )
    if independent_approximations is not None:
        if len(independent_approximations) != len(prefixes):
            raise ValueError(
                "Independent calibration requires one approximate Hamiltonian "
                "for every K=0,...,10 prefix"
            )
        approximation_sequence = [
            hermitian(np.asarray(value, dtype=np.complex128))
            for value in independent_approximations
        ]
        if any(value.shape != case.target.shape for value in approximation_sequence):
            raise ValueError("Independent calibration approximation shape mismatch")
    else:
        if len(fragments) != max(fit_k_values):
            raise ValueError(
                "Nested calibration requires exactly ten source fragments"
            )
        approximation_sequence = []
        approximation = case.base_collector.copy()
        approximation_sequence.append(hermitian(approximation))
        for fragment in fragments:
            approximation = approximation + fragment.matrix
            approximation_sequence.append(hermitian(approximation))
    split = np.asarray([value["split"] for value in probe_records])
    approximation_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    exact_energies = expectation(states, case.target)
    for k, (prefix, approximation) in enumerate(
        zip(prefixes, approximation_sequence)
    ):
        residual = hermitian(case.target - approximation)
        residual_sector = residual[np.ix_(case.sector_indices, case.sector_indices)]
        sector_frobenius = float(np.linalg.norm(residual_sector, "fro"))
        # Keep the approximation feature in its native energy units.  An
        # n-dependent normalization only reparameterizes the fitted weight and
        # obscures comparisons across active spaces.
        x_approximation = sector_frobenius
        approximate_means = expectation(states, approximation)
        signed_bias = approximate_means - exact_energies
        setting_variances = []
        for setting in prefix.settings:
            _, variance = expectation_and_variance(states, setting.matrix)
            setting_variances.append(variance)
        variance_array = np.asarray(setting_variances)
        lambdas = np.asarray(
            [setting.centered_range for setting in prefix.settings], dtype=float
        )
        for probe in range(states.shape[1]):
            approximation_rows.append(
                {
                    "probe_id": probe,
                    "group_id": str(probe_records[probe]["group_id"]),
                    "split": str(split[probe]),
                    "K": k,
                    "fit_eligible": k in fit_k_values,
                    "sector_Frobenius_distance": sector_frobenius,
                    "approximation_feature": x_approximation,
                    "exact_energy_E": float(exact_energies[probe]),
                    "approximate_mean_mu_K": float(approximate_means[probe]),
                    "signed_energy_error": float(signed_bias[probe]),
                    "absolute_energy_error": abs(float(signed_bias[probe])),
                    "squared_energy_error": float(signed_bias[probe] ** 2),
                }
            )
        for total_shots in shot_grid:
            shots = integer_range_allocation(total_shots, lambdas)
            active = shots > 0
            allocation_key = f"pilot_T{total_shots}_K{k}"
            for setting_index, (setting, count) in enumerate(
                zip(prefix.settings, shots)
            ):
                allocation_rows.append(
                    {
                        "allocation_key": allocation_key,
                        "T_total_shots": total_shots,
                        "K": k,
                        "setting_index": setting_index,
                        "source_index": setting.source_index,
                        "family": setting.family,
                        "centered_native_range": setting.centered_range,
                        "allocated_shots": int(count),
                    }
                )
            range_variance_proxy = float(
                np.sum(lambdas[active] ** 2 / shots[active])
            )
            sampling_feature = math.sqrt(max(range_variance_proxy, 0.0))
            actual_variance = np.sum(
                variance_array[active] / shots[active, None], axis=0
            )
            for probe in range(states.shape[1]):
                sampling_rows.append(
                    {
                        "probe_id": probe,
                        "group_id": str(probe_records[probe]["group_id"]),
                        "split": str(split[probe]),
                        "K": k,
                        "T_total_shots": total_shots,
                        "fit_eligible_K": k in fit_k_values,
                        "allocation_key": allocation_key,
                        "allocated_total_shots": int(np.sum(shots)),
                        "nonconstant_settings": int(np.sum(active)),
                        "exact_energy_E": float(exact_energies[probe]),
                        "approximate_mean_mu_K": float(approximate_means[probe]),
                        "signed_approximation_error_mu_minus_E": float(
                            signed_bias[probe]
                        ),
                        "sampling_feature": sampling_feature,
                        "range_variance_proxy": range_variance_proxy,
                        "actual_sampling_variance": float(actual_variance[probe]),
                        "actual_sampling_standard_error": math.sqrt(
                            max(float(actual_variance[probe]), 0.0)
                        ),
                    }
                )
    train_approximation = [
        row
        for row in approximation_rows
        if row["split"] == "train" and bool(row["fit_eligible"])
    ]
    # x_a,K is state-independent within a K stratum.  Each K first receives
    # its own mean squared-error/F-sector-squared ratio; K=1,...,10 are then
    # averaged equally so large K=0 residuals and row multiplicity cannot
    # dominate the transferable molecule-specific slope.
    for row in approximation_rows:
        row["approximation_ratio_denominator"] = float(
            row["approximation_feature"]
        ) ** 2
    theta_a, weight_a, approximation_strata = equal_stratum_ratio_fit(
        train_approximation,
        ("K",),
        "squared_energy_error",
        "approximation_ratio_denominator",
    )

    positive_q = np.asarray(
        [
            float(row["range_variance_proxy"])
            for row in sampling_rows
            if bool(row["fit_eligible_K"])
            and float(row["range_variance_proxy"]) > 0.0
        ],
        dtype=float,
    )
    if not len(positive_q):
        raise RuntimeError("No positive sampling range-variance proxy")
    q_reference = float(np.median(positive_q))
    q_threshold = max(
        SAMPLING_RATIO_ABSOLUTE_THRESHOLD_HA2,
        SAMPLING_RATIO_RELATIVE_THRESHOLD * q_reference,
    )
    for row in sampling_rows:
        q_value = float(row["range_variance_proxy"])
        eligible = bool(row["fit_eligible_K"]) and q_value > q_threshold
        row["sampling_ratio_q_reference"] = q_reference
        row["sampling_ratio_q_threshold"] = q_threshold
        row["fit_eligible"] = eligible
        row["sampling_variance_ratio"] = (
            float(row["actual_sampling_variance"]) / q_value
            if eligible
            else "N/A"
        )
    train_sampling = [
        row
        for row in sampling_rows
        if row["split"] == "train" and bool(row["fit_eligible"])
    ]
    theta_s, weight_s, sampling_strata = equal_stratum_ratio_fit(
        train_sampling,
        ("K", "T_total_shots"),
        "actual_sampling_variance",
        "range_variance_proxy",
    )
    for row in approximation_rows:
        row["predicted_absolute_component"] = weight_a * float(
            row["approximation_feature"]
        )
        row["predicted_squared_component"] = theta_a * float(
            row["approximation_feature"]
        ) ** 2
    for row in sampling_rows:
        row["predicted_sampling_standard_error"] = weight_s * float(
            row["sampling_feature"]
        )
        row["predicted_sampling_variance"] = theta_s * float(
            row["range_variance_proxy"]
        )
    diagnostics: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        a_rows = [
            row
            for row in approximation_rows
            if row["split"] == split_name and bool(row["fit_eligible"])
        ]
        s_rows = [
            row
            for row in sampling_rows
            if row["split"] == split_name and bool(row["fit_eligible"])
        ]
        diagnostics[split_name] = {
            "approximation_amplitude": regression_metrics(
                np.asarray([row["absolute_energy_error"] for row in a_rows]),
                np.asarray([row["predicted_absolute_component"] for row in a_rows]),
            ),
            "approximation_squared": regression_metrics(
                np.asarray([row["squared_energy_error"] for row in a_rows]),
                np.asarray([row["predicted_squared_component"] for row in a_rows]),
            ),
            "sampling_standard_error": regression_metrics(
                np.asarray([row["actual_sampling_standard_error"] for row in s_rows]),
                np.asarray(
                    [row["predicted_sampling_standard_error"] for row in s_rows]
                ),
            ),
            "sampling_variance": regression_metrics(
                np.asarray([row["actual_sampling_variance"] for row in s_rows]),
                np.asarray([row["predicted_sampling_variance"] for row in s_rows]),
            ),
        }
    weights = {
        "weight_approximation": weight_a,
        "weight_sampling": weight_s,
        "theta_approximation": theta_a,
        "theta_sampling": theta_s,
        "fit_definition": {
            "approximation_feature": (
                "x_a,K = ||P(H-H_hat_K)P||_F; no n-dependent normalization"
            ),
            "approximation": (
                "For each K=1,...,10, average d_sK^2/x_a,K^2 over training "
                "probes; average the ten nonnegative K slopes equally. K=0 is "
                "saved for audit and excluded from fitting."
            ),
            "sampling": (
                "For every eligible (K,T), average [sum_i sigma_siK^2/T_i] / "
                "[sum_i lambda_iK^2/T_i] over training probes; average all "
                "K=1,...,10 and eight T strata equally."
            ),
            "weight_mapping": "weight = sqrt(theta)",
            "lambda_definition": (
                "leakage-certified fixed-particle-number-sector allocation half-range "
                "for every complete source and depth-bounded one-body-cover leaf, "
                "with fail-closed full-Fock fallback; no independent collector"
            ),
            "allocation": (
                "minimum-one proportional allocation-range heuristic: give one shot "
                "to each nonconstant setting, then largest-remainder apportion T-m "
                "by the certified sector range or full-Fock fallback"
            ),
        },
        "calibration_K_fit": list(fit_k_values),
        "calibration_K_audit_only": [0],
        "calibration_shot_grid": list(shot_grid),
        "probe_seed": int(probe_records[0]["probe_seed"]),
        "probe_count": len(probe_records),
        "state_group_split": dict(STABILIZER_SPLIT_COUNTS),
        "sampling_ratio_filter": {
            "q_reference_median_positive_Ha2": q_reference,
            "relative_threshold": SAMPLING_RATIO_RELATIVE_THRESHOLD,
            "absolute_threshold_Ha2": SAMPLING_RATIO_ABSOLUTE_THRESHOLD_HA2,
            "q_threshold_Ha2": q_threshold,
            "excluded_K_T_strata": [
                {"K": int(k), "T_total_shots": int(total_shots)}
                for k in fit_k_values
                for total_shots in shot_grid
                if not any(
                    row["K"] == k
                    and row["T_total_shots"] == total_shots
                    and bool(row["fit_eligible"])
                    for row in sampling_rows
                )
            ],
            "rule": "exclude q_KT <= max(1e-30 Ha^2, 1e-12 median_positive_q); never add epsilon to the ratio denominator",
        },
        "equal_weight_strata": {
            "approximation_by_K": approximation_strata,
            "sampling_by_K_T": sampling_strata,
        },
        "sampling_ratio_bound_audit": {
            "minimum": float(
                min(float(row["sampling_variance_ratio"]) for row in sampling_rows if bool(row["fit_eligible"]))
            ),
            "maximum": float(
                max(float(row["sampling_variance_ratio"]) for row in sampling_rows if bool(row["fit_eligible"]))
            ),
            "violations_above_one_plus_1e-10": int(
                sum(
                    float(row["sampling_variance_ratio"]) > 1.0 + 1.0e-10
                    for row in sampling_rows
                    if bool(row["fit_eligible"])
                )
            ),
        },
        "diagnostics": diagnostics,
    }
    return weights, approximation_rows, sampling_rows, allocation_rows


def _select_with_tolerance(
    values: list[Any], score, secondary, relative_tolerance: float = 1.0e-12
):
    minimum = min(float(score(value)) for value in values)
    tolerance = relative_tolerance * max(1.0, abs(minimum))
    tied = [value for value in values if float(score(value)) <= minimum + tolerance]
    return min(tied, key=secondary)


def source_block_trial(
    case,
    circuits,
    residual: np.ndarray,
    family: str,
    depth: int | None,
    boundary: int,
    block_size: int,
    starts: int,
    iterations: int,
    rcdf_bank: list[Fragment],
    used_rcdf_indices: set[int] | None = None,
):
    trial_residual = residual.copy()
    trial_fragments: list[Fragment] = []
    used_rcdf: set[int] = set() if used_rcdf_indices is None else set(used_rcdf_indices)
    term_rows = []
    started = time.perf_counter()
    for offset in range(block_size):
        term = boundary + offset + 1
        before = float(np.vdot(trial_residual, trial_residual).real)
        if family in {"rcdf", "shallow_rcdf"}:
            reduction, fragment = best_rcdf_fragment(
                trial_residual, rcdf_bank, used_rcdf, term
            )
            used_rcdf.add(int(fragment.rcdf_bank_index))
        else:
            if depth is None:
                raise RuntimeError("Circuit trial requires an explicit depth")
            reduction, fragment = optimize_circuit_fragment(
                circuits,
                case.name,
                trial_residual,
                family,
                depth,
                term,
                starts,
                iterations,
            )
        trial_residual = hermitian(trial_residual - fragment.matrix)
        after = float(np.vdot(trial_residual, trial_residual).real)
        actual = before - after
        if actual < -1.0e-8:
            raise RuntimeError(f"{family} depth={depth} increased F at K={term}")
        if abs(actual - reduction) > 2.0e-7 * max(1.0, abs(reduction)):
            raise RuntimeError(
                f"Reduction identity failed for {family} depth={depth} K={term}"
            )
        trial_fragments.append(fragment)
        random_start_count = (
            "N/A"
            if family in {"rcdf", "shallow_rcdf"}
            else int(fragment.candidate_metadata.get("random_start_count", 0))
        )
        term_rows.append(
            {
                "K": term,
                "Frobenius_distance": math.sqrt(max(after, 0.0)),
                "Frobenius_sq_reduction": actual,
                "random_start_count": random_start_count,
                "winner_start": fragment.candidate_metadata.get("winner_start", "N/A"),
                "winner_seed": fragment.candidate_metadata.get("winner_seed", "N/A"),
                "all_start_trials": fragment.candidate_metadata.get("start_trials", []),
                "shallow_rcdf_depth": fragment.candidate_metadata.get(
                    "shallow_rcdf_depth"
                ),
                "logical_spin_givens_per_leaf": fragment.candidate_metadata.get(
                    "logical_spin_givens_per_leaf"
                ),
                "native_givens_depth_per_leaf": fragment.candidate_metadata.get(
                    "native_givens_depth_per_leaf"
                ),
                "even_odd_angle_record": fragment.candidate_metadata.get(
                    "even_odd_angle_record"
                ),
            }
        )
    return {
        "family": family,
        "family_label": LABELS[family],
        "depth": depth,
        "residual": trial_residual,
        "fragments": trial_fragments,
        "term_rows": term_rows,
        "random_start_contract": (
            "shallow_RCDF_uses_tensor_native_optimizer"
            if family in {"rcdf", "shallow_rcdf"}
            else all(int(row["random_start_count"]) == starts for row in term_rows)
        ),
        "endpoint_Frobenius_distance": float(np.linalg.norm(trial_residual, "fro")),
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_dynamic_depth_frontier(
    case,
    circuits,
    max_terms: int,
    starts: int,
    iterations: int,
    rcdf_rho: float,
    rcdf_cycles: int,
    rcdf_u_steps: int,
):
    if max_terms % BLOCK_SIZE:
        raise ValueError("This validation pilot requires max_terms divisible by four")
    residual = hermitian(case.target - case.base_collector)
    accepted_fragments: list[Fragment] = []
    accepted_depths: list[int] = []
    selection_log: list[dict[str, Any]] = []
    flat_trials: list[dict[str, Any]] = []
    family_order = {family: index for index, family in enumerate(FAMILIES)}
    for boundary in range(0, max_terms, BLOCK_SIZE):
        rcdf_leaves = min(10, boundary + BLOCK_SIZE)
        print(
            f"[frontier] boundary K={boundary:02d}: fitting RCDF bank ({rcdf_leaves} leaves)",
            flush=True,
        )
        rcdf_bank, rcdf_audit = make_rcdf_bank(
            case, rcdf_leaves, rcdf_rho, rcdf_cycles, rcdf_u_steps
        )
        family_winners = []
        boundary_trials = []
        for family in CIRCUIT_FAMILIES:
            depth_trials = []
            incumbent = None
            for depth in DEPTHS:
                print(
                    f"[frontier] K={boundary:02d} trial {LABELS[family]} depth={depth}",
                    flush=True,
                )
                trial = source_block_trial(
                    case,
                    circuits,
                    residual,
                    family,
                    depth,
                    boundary,
                    BLOCK_SIZE,
                    starts,
                    iterations,
                    rcdf_bank,
                )
                depth_trials.append(trial)
                print(
                    f"[frontier] {LABELS[family]} d={depth} -> "
                    f"F={trial['endpoint_Frobenius_distance']:.8g}",
                    flush=True,
                )
                if incumbent is None:
                    incumbent = trial
                    trial["depth_relative_improvement"] = None
                    trial["depth_search_decision"] = "initialize_incumbent"
                    continue
                relative_improvement = (
                    incumbent["endpoint_Frobenius_distance"]
                    - trial["endpoint_Frobenius_distance"]
                ) / max(incumbent["endpoint_Frobenius_distance"], 1.0e-15)
                trial["depth_relative_improvement"] = relative_improvement
                if relative_improvement > DEPTH_RELATIVE_IMPROVEMENT:
                    incumbent = trial
                    trial["depth_search_decision"] = "accept_and_continue"
                else:
                    trial["depth_search_decision"] = "reject_and_stop"
                    break
            if incumbent is None:
                raise RuntimeError(f"No depth trial generated for {family}")
            winner = incumbent
            family_winners.append(winner)
            boundary_trials.extend(depth_trials)
        print(f"[frontier] K={boundary:02d} trial {LABELS['rcdf']} depth=N/A", flush=True)
        rcdf_trial = source_block_trial(
            case,
            circuits,
            residual,
            "rcdf",
            None,
            boundary,
            BLOCK_SIZE,
            starts,
            iterations,
            rcdf_bank,
        )
        family_winners.append(rcdf_trial)
        boundary_trials.append(rcdf_trial)
        chosen = _select_with_tolerance(
            family_winners,
            score=lambda value: value["endpoint_Frobenius_distance"],
            secondary=lambda value: family_order[str(value["family"])],
        )
        for trial in boundary_trials:
            selected_depth = (
                any(trial is winner for winner in family_winners)
                and trial["family"] != "rcdf"
            )
            record = {
                "boundary_after_K": boundary,
                "candidate_end_K": boundary + BLOCK_SIZE,
                "family": trial["family"],
                "family_label": trial["family_label"],
                "depth": "N/A" if trial["depth"] is None else trial["depth"],
                "endpoint_Frobenius_distance": trial[
                    "endpoint_Frobenius_distance"
                ],
                "selected_depth_within_family": bool(selected_depth),
                "selected_family": trial is chosen,
                "elapsed_seconds": trial["elapsed_seconds"],
            }
            flat_trials.append(record)
        depth_value = -1 if chosen["depth"] is None else int(chosen["depth"])
        accepted_fragments.extend(chosen["fragments"])
        accepted_depths.extend([depth_value] * BLOCK_SIZE)
        residual = chosen["residual"]
        selection_log.append(
            {
                "boundary_after_K": boundary,
                "candidate_end_K": boundary + BLOCK_SIZE,
                "depth_rule": (
                    "calibration exploration only: try d=1,...,7 in ascending order; "
                    "continue only when endpoint full-Fock Frobenius distance improves "
                    f"by more than {DEPTH_RELATIVE_IMPROVEMENT:g}; otherwise retain the "
                    "latest beneficial depth and stop. The production v3 runner applies "
                    "the same rule to the frozen stabilizer loss."
                ),
                "family_rule": "minimum endpoint full-Fock Frobenius distance; ties use declared pool order",
                "circuit_depth_trials": [
                    {
                        "family": trial["family"],
                        "depth": trial["depth"],
                        "endpoint_Frobenius_distance": trial[
                            "endpoint_Frobenius_distance"
                        ],
                        "term_rows": trial["term_rows"],
                        "elapsed_seconds": trial["elapsed_seconds"],
                        "depth_relative_improvement": trial.get(
                            "depth_relative_improvement"
                        ),
                        "depth_search_decision": trial.get(
                            "depth_search_decision"
                        ),
                    }
                    for trial in boundary_trials
                    if trial["family"] != "rcdf"
                ],
                "RCDF_trial": {
                    "depth": None,
                    "endpoint_Frobenius_distance": rcdf_trial[
                        "endpoint_Frobenius_distance"
                    ],
                    "term_rows": rcdf_trial["term_rows"],
                    "elapsed_seconds": rcdf_trial["elapsed_seconds"],
                    "bank_audit": rcdf_audit,
                },
                "family_depth_winners": [
                    {
                        "family": trial["family"],
                        "depth": trial["depth"],
                        "endpoint_Frobenius_distance": trial[
                            "endpoint_Frobenius_distance"
                        ],
                    }
                    for trial in family_winners
                ],
                "selected_family": chosen["family"],
                "selected_family_label": chosen["family_label"],
                "selected_depth": chosen["depth"],
                "accepted_endpoint_Frobenius_distance": chosen[
                    "endpoint_Frobenius_distance"
                ],
            }
        )
        print(
            f"[frontier] accepted K={boundary + 1}-{boundary + BLOCK_SIZE}: "
            f"{chosen['family_label']} depth="
            f"{'N/A' if chosen['depth'] is None else chosen['depth']}",
            flush=True,
        )
    return accepted_fragments, accepted_depths, selection_log, flat_trials


def prefix_setting_replay_rows(prefixes: list[PrefixDecomposition]):
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        cumulative = 0.0
        for setting_index, setting in enumerate(prefix.settings):
            cumulative += setting.centered_range
            alpha = (
                0.0
                if setting.source_index == 0
                else float(prefix.alpha_by_source[setting.source_index - 1])
            )
            rows.append(
                {
                    "K": prefix.k,
                    "setting_index": setting_index,
                    "source_index": setting.source_index,
                    "family": setting.family,
                    "interface": setting.interface,
                    "F3_alpha": alpha,
                    "native_minimum": setting.native_minimum,
                    "native_maximum": setting.native_maximum,
                    "range_domain": setting.range_domain,
                    "full_fock_centered_range": setting.full_fock_centered_range,
                    "sector_spectral_centered_range": (
                        setting.sector_spectral_centered_range
                    ),
                    "allocation_centered_range": setting.allocation_centered_range,
                    "sector_minimum": setting.sector_minimum,
                    "sector_maximum": setting.sector_maximum,
                    "sector_dimension": setting.sector_dimension,
                    "sector_leakage_frobenius": (
                        setting.sector_leakage_frobenius
                    ),
                    "leakage_operator_bound": setting.leakage_operator_bound,
                    "sector_leakage_relative_frobenius": (
                        setting.sector_leakage_relative_frobenius
                    ),
                    "sector_range_eligible": setting.sector_range_eligible,
                    "centered_native_range": setting.centered_range,
                    "cumulative_centered_native_range": cumulative,
                }
            )
    return rows


def dynamic_prefix_metrics(case, fragments, depths, prefixes, ground):
    rows: list[dict[str, Any]] = []
    approximation = case.base_collector.copy()
    for k, prefix in enumerate(prefixes):
        if k:
            approximation = approximation + fragments[k - 1].matrix
        residual = hermitian(case.target - approximation)
        sector_residual = residual[np.ix_(case.sector_indices, case.sector_indices)]
        # Signed convention used everywhere below: approximate minus exact.
        bias = -float(np.real(np.vdot(ground, residual @ ground)))
        family = "base_collector_only" if k == 0 else fragments[k - 1].family
        depth = "N/A" if k == 0 or depths[k - 1] < 0 else depths[k - 1]
        rows.append(
            {
                "K": k,
                "total_measurement_settings": len(prefix.settings),
                "accepted_family_at_K": family,
                "accepted_family_label_at_K": (
                    "base collector only" if k == 0 else LABELS[family]
                ),
                "chosen_depth_at_K": depth,
                "full_Frobenius_distance": float(np.linalg.norm(residual, "fro")),
                "sector_Frobenius_distance": float(
                    np.linalg.norm(sector_residual, "fro")
                ),
                "centered_native_range_sum": prefix.spectral_sum,
                "signed_ground_state_approximation_error": bias,
                "absolute_ground_state_approximation_error": abs(bias),
                "F3_reconstruction_error": prefix.reconstruction_error,
            }
        )
    return rows, prefix_setting_replay_rows(prefixes)


def setting_variances(state: np.ndarray, prefix: PrefixDecomposition) -> np.ndarray:
    states = state[:, None]
    values = []
    for setting in prefix.settings:
        _, variance = expectation_and_variance(states, setting.matrix)
        values.append(float(variance[0]))
    return np.asarray(values, dtype=float)


def stable_sampling_seed(total_shots: int, k: int) -> int:
    payload = f"{VERSION}|LiH|T{total_shots}|K{k}|ground".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def outcome_models(state: np.ndarray, prefix: PrefixDecomposition):
    models = []
    for setting in prefix.settings:
        eigenvalues, eigenvectors = np.linalg.eigh(setting.matrix)
        amplitudes = eigenvectors.conj().T @ state
        probabilities = np.maximum(np.abs(amplitudes) ** 2, 0.0)
        probabilities /= float(np.sum(probabilities))
        mean = float(np.dot(probabilities, eigenvalues))
        variance = max(
            0.0, float(np.dot(probabilities, eigenvalues * eigenvalues) - mean * mean)
        )
        deterministic_value = None
        if setting.centered_range == 0.0:
            if (
                setting.range_domain == "fixed_particle_number_sector"
                and setting.sector_minimum is not None
                and setting.sector_maximum is not None
                and setting.sector_leakage_frobenius == 0.0
            ):
                deterministic_value = 0.5 * (
                    float(setting.sector_minimum) + float(setting.sector_maximum)
                )
            elif float(eigenvalues[-1] - eigenvalues[0]) == 0.0:
                deterministic_value = float(eigenvalues[0])
            else:
                raise RuntimeError(
                    "A zero-shot setting is not certified constant on its allocation domain"
                )
        models.append(
            (eigenvalues, probabilities, mean, variance, deterministic_value)
        )
    return models


def monte_carlo_estimates(
    models,
    shots: np.ndarray,
    repeats: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    estimates = np.zeros(repeats, dtype=float)
    for (
        eigenvalues,
        probabilities,
        _,
        _,
        deterministic_value,
    ), count in zip(models, shots):
        if count <= 0:
            if deterministic_value is None:
                raise RuntimeError(
                    "A zero-shot estimator term lacks a state-independent constant"
                )
            estimates += deterministic_value
            continue
        sampled = rng.multinomial(int(count), probabilities, size=repeats)
        estimates += sampled @ eigenvalues / float(count)
    return estimates


def select_k_and_sample(
    metrics: list[dict[str, Any]],
    prefixes: list[PrefixDecomposition],
    weights: dict[str, Any],
    ground: np.ndarray,
    ground_energy: float,
    shot_grid: tuple[int, ...],
    repeats: int,
):
    weight_a = float(weights["weight_approximation"])
    weight_s = float(weights["weight_sampling"])
    variance_cache = {
        prefix.k: setting_variances(ground, prefix) for prefix in prefixes
    }
    grid_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    outcome_cache: dict[int, Any] = {}
    for total_shots in shot_grid:
        candidates = []
        current_allocation_rows: list[dict[str, Any]] = []
        for metric, prefix in zip(metrics, prefixes):
            lambdas = np.asarray(
                [setting.centered_range for setting in prefix.settings], dtype=float
            )
            shots = integer_range_allocation(total_shots, lambdas)
            active = shots > 0
            x_sampling_sq = float(
                np.sum(lambdas[active] ** 2 / shots[active])
            )
            approximation_component = weight_a * float(
                metric["sector_Frobenius_distance"]
            )
            sampling_component = weight_s * math.sqrt(max(x_sampling_sq, 0.0))
            objective = math.hypot(approximation_component, sampling_component)
            actual_variance = float(
                np.sum(variance_cache[prefix.k][active] / shots[active])
            )
            record = {
                "T_total_shots": total_shots,
                "K": prefix.k,
                "allocation_key": f"dynamic_T{total_shots}_K{prefix.k}",
                "total_measurement_settings": len(prefix.settings),
                "accepted_family_at_K": metric["accepted_family_at_K"],
                "chosen_depth_at_K": metric["chosen_depth_at_K"],
                "predicted_approximation_component": approximation_component,
                "predicted_sampling_component": sampling_component,
                "predicted_total_RMSE": objective,
                "integer_range_proxy_variance": x_sampling_sq,
                "ground_state_sampling_variance": actual_variance,
                "ground_state_sampling_standard_error": math.sqrt(
                    max(actual_variance, 0.0)
                ),
                "signed_ground_state_approximation_error": metric[
                    "signed_ground_state_approximation_error"
                ],
            }
            candidates.append((record, shots))
            grid_rows.append(record)
            for setting_index, (setting, count) in enumerate(
                zip(prefix.settings, shots)
            ):
                current_allocation_rows.append(
                    {
                        "allocation_key": f"dynamic_T{total_shots}_K{prefix.k}",
                        "T_total_shots": total_shots,
                        "K": prefix.k,
                        "setting_index": setting_index,
                        "source_index": setting.source_index,
                        "family": setting.family,
                        "interface": setting.interface,
                        "centered_native_range": setting.centered_range,
                        "allocated_shots": int(count),
                    }
                )
        selected_record, selected_shots = min(
            candidates, key=lambda item: (item[0]["predicted_total_RMSE"], item[0]["K"])
        )
        k = int(selected_record["K"])
        if k == len(prefixes) - 1:
            raise RuntimeError(
                f"T={total_shots}: selected LiH rank lies at the available "
                "frontier endpoint; extend the frontier before publishing the "
                "fixed-shot result"
            )
        for allocation_row in current_allocation_rows:
            allocation_row["selected_K_for_T"] = int(allocation_row["K"]) == k
        allocation_rows.extend(current_allocation_rows)
        if k not in outcome_cache:
            print(f"[sampling] diagonalizing selected prefix K={k:02d}", flush=True)
            outcome_cache[k] = outcome_models(ground, prefixes[k])
        models = outcome_cache[k]
        approximate_energy = float(sum(model[2] for model in models))
        signed_bias = approximate_energy - ground_energy
        estimates = monte_carlo_estimates(
            models,
            selected_shots,
            repeats,
            stable_sampling_seed(total_shots, k),
        )
        sampling_errors = estimates - approximate_energy
        total_errors = estimates - ground_energy
        summary = dict(selected_record)
        summary.update(
            {
                "right_censored_at_Kmax": False,
                "exact_ground_energy": ground_energy,
                "exact_approximate_energy": approximate_energy,
                "signed_ground_state_approximation_error": signed_bias,
                "absolute_ground_state_approximation_error": abs(signed_bias),
                "analytic_total_RMSE": math.hypot(
                    signed_bias,
                    float(selected_record["ground_state_sampling_standard_error"]),
                ),
                "Monte_Carlo_repeats": repeats,
                "empirical_sampling_bias": float(np.mean(sampling_errors)),
                "empirical_sampling_standard_deviation": float(
                    np.std(sampling_errors, ddof=1)
                ),
                "empirical_sampling_RMSE": float(
                    np.sqrt(np.mean(sampling_errors**2))
                ),
                "empirical_total_bias": float(np.mean(total_errors)),
                "empirical_total_MAE": float(np.mean(np.abs(total_errors))),
                "empirical_total_RMSE": float(np.sqrt(np.mean(total_errors**2))),
                "allocation_minimum_nonzero_shots": int(
                    np.min(selected_shots[selected_shots > 0])
                ),
                "allocation_maximum_shots": int(np.max(selected_shots)),
                "allocation_conserved": int(np.sum(selected_shots)) == total_shots,
            }
        )
        selected_rows.append(summary)
        for repeat, (estimate, sampling_error, total_error) in enumerate(
            zip(estimates, sampling_errors, total_errors)
        ):
            replicate_rows.append(
                {
                    "T_total_shots": total_shots,
                    "K": k,
                    "repeat": repeat,
                    "energy_estimate": float(estimate),
                    "sampling_error_about_approximation": float(sampling_error),
                    "total_error_about_exact_ground_energy": float(total_error),
                }
            )
        print(
            f"[sampling] T={total_shots:>8d} K={k:02d} "
            f"empirical total RMSE={summary['empirical_total_RMSE']:.4g}",
            flush=True,
        )
    return grid_rows, selected_rows, replicate_rows, allocation_rows


def audit_native_ranges(
    prefix: PrefixDecomposition, sector_indices: np.ndarray
) -> float:
    """Replay both restricted and unrestricted ranges against dense spectra."""
    maximum = 0.0
    for setting in prefix.settings:
        metrics = _sector_range_metrics(setting.matrix, sector_indices)
        maximum = max(
            maximum,
            abs(float(metrics["centered_range"]) - setting.centered_range),
            abs(
                float(metrics["full_fock_centered_range"])
                - float(setting.full_fock_centered_range)
            ),
        )
    return maximum


def make_plots(
    output: Path,
    weights: dict[str, Any],
    approximation_rows: list[dict[str, Any]],
    dynamic_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> None:
    figure_dir = output / "figures"
    test_rows = [row for row in approximation_rows if row["split"] == "test"]
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    x = np.asarray([row["absolute_energy_error"] for row in test_rows])
    y = np.asarray([row["predicted_absolute_component"] for row in test_rows])
    axis.scatter(x, y, s=18, alpha=0.55, color="#2563eb")
    limit = 1.05 * max(float(np.max(x)), float(np.max(y)), 1.0e-8)
    axis.plot([0, limit], [0, limit], color="#111827", linestyle="--", linewidth=1.0)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("Exact stabilizer-probe |energy error| (Ha)")
    axis.set_ylabel("Calibrated approximation component (Ha)")
    axis.set_title("LiH held-out stabilizer calibration")
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "LiH_stabilizer_calibration.png", dpi=220)
    figure.savefig(figure_dir / "LiH_stabilizer_calibration.pdf")
    plt.close(figure)

    figure = plt.figure(figsize=(7.6, 6.3))
    grid = figure.add_gridspec(2, 1, height_ratios=(4.2, 1.05), hspace=0.16)
    top_axis = figure.add_subplot(grid[0, 0])
    schedule_axis = figure.add_subplot(grid[1, 0], sharex=top_axis)
    axes = [top_axis, schedule_axis]
    k_values = np.asarray([int(row["K"]) for row in dynamic_rows])
    distances = np.asarray(
        [float(row["full_Frobenius_distance"]) for row in dynamic_rows]
    )
    axes[0].plot(k_values, distances, color="#20242a", linewidth=1.6)
    for row in dynamic_rows[1:]:
        family = str(row["accepted_family_at_K"])
        axes[0].scatter(
            int(row["K"]),
            float(row["full_Frobenius_distance"]),
            color=COLORS[family],
            s=38,
            zorder=3,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Full-Fock Frobenius distance (Ha)")
    axes[0].grid(True, which="both", alpha=0.2)
    axes[0].set_title("LiH F-dominant dynamic-depth source frontier")
    terminal_k = int(dynamic_rows[-1]["K"])
    for start in range(1, terminal_k + 1, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE - 1, terminal_k)
        row = dynamic_rows[end]
        family = str(row["accepted_family_at_K"])
        depth = row["chosen_depth_at_K"]
        label = f"{LABELS[family]}\n" + (
            "depth=N/A" if depth == "N/A" else f"d={depth}"
        )
        axes[1].barh(
            0,
            end - start + 1,
            left=start - 0.5,
            height=0.62,
            color=COLORS[family],
            edgecolor="white",
            linewidth=1.0,
        )
        axes[1].text(
            0.5 * (start + end),
            0,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=7.5,
            fontweight="bold",
        )
    axes[1].set_ylim(-0.55, 0.55)
    axes[1].set_yticks([])
    axes[1].set_xlim(-0.1, terminal_k + 0.5)
    axes[1].set_xticks(np.arange(0, terminal_k + 1, dtype=int))
    axes[1].set_xlabel("Accepted source terms K")
    axes[1].set_title("Accepted ansatz schedule (family and selected depth)", fontsize=9)
    axes[1].grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
    axes[0].tick_params(axis="x", labelbottom=False)
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.94)
    figure.savefig(figure_dir / "LiH_dynamic_depth_frontier.png", dpi=220)
    figure.savefig(figure_dir / "LiH_dynamic_depth_frontier.pdf")
    plt.close(figure)

    total_shots = np.asarray([int(row["T_total_shots"]) for row in selected_rows])
    chosen_k = np.asarray([int(row["K"]) for row in selected_rows])
    figure, axis = plt.subplots(figsize=(6.4, 4.5))
    axis.semilogx(total_shots, chosen_k, marker="o", color="#1d4ed8", linewidth=1.6)
    axis.set_xlabel("Total shots T")
    axis.set_ylabel("Selected source terms K")
    axis.set_yticks(np.arange(0, int(np.max(chosen_k)) + 1))
    axis.set_title("LiH stabilizer-calibrated K(T)")
    axis.grid(True, which="both", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "LiH_K_vs_T.png", dpi=220)
    figure.savefig(figure_dir / "LiH_K_vs_T.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    series = (
        ("ground_state_sampling_standard_error", "Analytic sampling SE", "#2563eb", "-"),
        ("empirical_sampling_RMSE", "Monte Carlo sampling RMSE", "#60a5fa", "--"),
        ("absolute_ground_state_approximation_error", "Approximation |bias|", "#d62728", ":"),
        ("analytic_total_RMSE", "Analytic total RMSE", "#111827", "-"),
        ("empirical_total_RMSE", "Monte Carlo total RMSE", "#16a34a", "--"),
    )
    for key, label, color, linestyle in series:
        values = np.asarray([float(row[key]) for row in selected_rows])
        axis.loglog(
            total_shots,
            np.maximum(values, 1.0e-12),
            marker="o",
            markersize=3.8,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )
    axis.set_xlabel("Total shots T")
    axis.set_ylabel("Ground-state energy error (Ha)")
    axis.set_title("LiH ground-state sampling error at selected K(T)")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "LiH_ground_state_sampling_error.png", dpi=220)
    figure.savefig(figure_dir / "LiH_ground_state_sampling_error.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-terms", type=int, default=12)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--spectral-max-evaluations", type=int, default=30)
    parser.add_argument("--rcdf-rho", type=float, default=1.0e-6)
    parser.add_argument("--rcdf-cycles", type=int, default=4)
    parser.add_argument("--rcdf-u-steps", type=int, default=2)
    parser.add_argument("--sampling-repeats", type=int, default=400)
    parser.add_argument("--probe-seed", type=int, default=918273)
    parser.add_argument("--shot-grid", type=int, nargs="+", default=SHOT_GRID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_terms < BLOCK_SIZE or args.max_terms % BLOCK_SIZE:
        raise ValueError("--max-terms must be a positive multiple of four")
    if args.starts < 1 or args.iterations < 0 or args.sampling_repeats < 2:
        raise ValueError("Invalid optimization or sampling budget")
    shot_grid = tuple(sorted(set(int(value) for value in args.shot_grid)))
    if not shot_grid or min(shot_grid) < args.max_terms + 1:
        raise ValueError("Every shot budget must cover all possible settings")
    before = protected_snapshot()
    output = validate_output_path(args.output)
    started = time.perf_counter()
    case = load_active_case("LiH")
    circuits = load_circuit_module()
    circuits.validate_operator_pool_definition()
    ground, ground_energy = ground_state(case)

    print("[stage 1/4] replaying the saved depth-one LiH calibration path", flush=True)
    pilot_fragments, replay_rows = replay_saved_pilot(
        case, circuits, args.starts, args.iterations
    )
    save_fragments(
        output / "pilot_replayed_fragments.npz",
        pilot_fragments,
        [1] * len(pilot_fragments),
    )
    write_csv(output / "pilot_replay_audit.csv", replay_rows)
    pilot_prefixes = build_all_prefixes(
        case, pilot_fragments, args.spectral_max_evaluations
    )
    write_csv(
        output / "pilot_prefix_setting_ranges.csv",
        prefix_setting_replay_rows(pilot_prefixes),
    )
    probes, probe_records = generate_stabilizer_probes(
        case.sector_indices, args.probe_seed
    )
    np.savez_compressed(output / "stabilizer_probes.npz", states=probes)
    write_csv(output / "stabilizer_probe_manifest.csv", probe_records)
    weights, approximation_rows, sampling_rows, calibration_allocation_rows = calibrate_weights(
        case,
        pilot_fragments,
        pilot_prefixes,
        probes,
        probe_records,
        shot_grid,
    )
    write_json(output / "stabilizer_weights.json", weights)
    write_csv(output / "stabilizer_approximation_components.csv", approximation_rows)
    write_csv(output / "stabilizer_sampling_components.csv", sampling_rows)
    write_csv(
        output / "stabilizer_calibration_shot_allocations.csv",
        calibration_allocation_rows,
    )
    print(
        "[calibration] weight_approximation="
        f"{weights['weight_approximation']:.8g}, weight_sampling="
        f"{weights['weight_sampling']:.8g}",
        flush=True,
    )

    print("[stage 2/4] building the dynamic-depth F-dominant frontier", flush=True)
    dynamic_fragments, dynamic_depths, selection_log, flat_trials = (
        build_dynamic_depth_frontier(
            case,
            circuits,
            args.max_terms,
            args.starts,
            args.iterations,
            args.rcdf_rho,
            args.rcdf_cycles,
            args.rcdf_u_steps,
        )
    )
    save_fragments(
        output / "dynamic_depth_fragments.npz",
        dynamic_fragments,
        dynamic_depths,
    )
    np.savez_compressed(
        output / "LiH_active_case_operators.npz",
        target=case.target,
        exact_base_collector=case.base_collector,
        proxy_state=case.proxy_state,
        sector_indices=case.sector_indices,
    )
    write_json(output / "dynamic_depth_selection_log.json", selection_log)
    write_csv(output / "dynamic_depth_all_trials.csv", flat_trials)

    print("[stage 3/4] rebuilding accepted prefixes and selecting K(T)", flush=True)
    dynamic_prefixes = build_all_prefixes(
        case, dynamic_fragments, args.spectral_max_evaluations
    )
    dynamic_rows, setting_rows = dynamic_prefix_metrics(
        case, dynamic_fragments, dynamic_depths, dynamic_prefixes, ground
    )
    write_csv(output / "dynamic_depth_metrics_by_k.csv", dynamic_rows)
    write_csv(output / "dynamic_prefix_setting_ranges.csv", setting_rows)
    candidate_grid, selected_rows, replicate_rows, dynamic_allocation_rows = select_k_and_sample(
        dynamic_rows,
        dynamic_prefixes,
        weights,
        ground,
        ground_energy,
        shot_grid,
        args.sampling_repeats,
    )
    write_csv(output / "LiH_K_by_T_candidate_grid.csv", candidate_grid)
    write_csv(output / "LiH_K_and_ground_sampling_error_by_T.csv", selected_rows)
    write_csv(output / "LiH_ground_sampling_replicates.csv", replicate_rows)
    write_csv(output / "LiH_dynamic_shot_allocations_by_T_K.csv", dynamic_allocation_rows)

    print("[stage 4/4] plots and audits", flush=True)
    make_plots(output, weights, approximation_rows, dynamic_rows, selected_rows)
    after = protected_snapshot()
    terminal_range_error = audit_native_ranges(
        dynamic_prefixes[-1], case.sector_indices
    )
    replay_maximum = max(
        float(row["saved_matrix_replay_Frobenius_error"]) for row in replay_rows
    )
    reconstruction_maximum = max(
        prefix.reconstruction_error for prefix in dynamic_prefixes
    )
    full_distances = np.asarray(
        [float(row["full_Frobenius_distance"]) for row in dynamic_rows]
    )
    protected_unchanged = before == after
    probe_group_splits: dict[str, set[str]] = {}
    for record in probe_records:
        probe_group_splits.setdefault(str(record["group_id"]), set()).add(
            str(record["split"])
        )
    probe_groups_are_split_disjoint = all(
        len(values) == 1 for values in probe_group_splits.values()
    )
    selected_signed_error_consistency = max(
        abs(
            float(row["signed_ground_state_approximation_error"])
            - (
                float(row["exact_approximate_energy"])
                - float(row["exact_ground_energy"])
            )
        )
        for row in selected_rows
    )
    stabilizer_signed_error_consistency = max(
        abs(
            float(row["signed_energy_error"])
            - (
                float(row["approximate_mean_mu_K"])
                - float(row["exact_energy_E"])
            )
        )
        for row in approximation_rows
    )
    allocation_sums: dict[str, int] = {}
    allocation_targets: dict[str, int] = {}
    for row in calibration_allocation_rows + dynamic_allocation_rows:
        key = str(row["allocation_key"])
        allocation_sums[key] = allocation_sums.get(key, 0) + int(
            row["allocated_shots"]
        )
        allocation_targets[key] = int(row["T_total_shots"])
    every_saved_shot_vector_conserved = all(
        allocation_sums[key] == allocation_targets[key] for key in allocation_sums
    )
    audit = {
        "status": "PASS",
        "protected_files_before": before,
        "protected_files_after": after,
        "protected_files_unchanged": protected_unchanged,
        "saved_pilot_maximum_matrix_replay_Frobenius_error": replay_maximum,
        "dynamic_full_Frobenius_monotone": bool(
            np.all(np.diff(full_distances) <= 1.0e-9)
        ),
        "maximum_AGPD_shallow_cover_reconstruction_error": reconstruction_maximum,
        "terminal_maximum_sector_and_full_range_vs_dense_oracle_error": (
            terminal_range_error
        ),
        "all_shot_allocations_conserved": all(
            bool(row["allocation_conserved"]) for row in selected_rows
        ),
        "every_saved_shot_vector_conserved": every_saved_shot_vector_conserved,
        "saved_shot_vector_count": len(allocation_sums),
        "stabilizer_group_count": len(probe_group_splits),
        "stabilizer_groups_are_train_validation_test_disjoint": probe_groups_are_split_disjoint,
        "maximum_selected_signed_error_convention_mismatch": selected_signed_error_consistency,
        "maximum_stabilizer_signed_error_convention_mismatch": stabilizer_signed_error_consistency,
        "signed_error_convention": "approximate energy minus exact energy",
        "depth_trial_count": len(flat_trials),
        "minimum_expected_depth_trial_count": (len(CIRCUIT_FAMILIES) + 1)
        * (args.max_terms // BLOCK_SIZE),
        "maximum_expected_depth_trial_count": (
            len(CIRCUIT_FAMILIES) * len(DEPTHS) + 1
        ) * (args.max_terms // BLOCK_SIZE),
        "RCDF_depth_encoding": "N/A in CSV/JSON; -1 only in numeric NPZ chosen_depth array",
        "nonstandard_family_policy": "original native leaf; no transferred direction and no generalized dense collector",
        "dense_validation_only": True,
    }
    conditions = (
        protected_unchanged,
        replay_maximum <= 2.0e-8,
        bool(np.all(np.diff(full_distances) <= 1.0e-9)),
        reconstruction_maximum <= AGPD_SHALLOW_COLLECTOR_MATRIX_TOLERANCE,
        terminal_range_error <= 2.0e-7,
        audit["all_shot_allocations_conserved"],
        every_saved_shot_vector_conserved,
        probe_groups_are_split_disjoint,
        selected_signed_error_consistency <= 1.0e-12,
        stabilizer_signed_error_consistency <= 1.0e-12,
        audit["minimum_expected_depth_trial_count"]
        <= audit["depth_trial_count"]
        <= audit["maximum_expected_depth_trial_count"],
    )
    if not all(conditions):
        audit["status"] = "FAIL"
    write_json(output / "audit.json", audit)

    manifest = {
        "version": VERSION,
        "scope": "LiH CAS(2e,4o) isolated numerical calibration pilot",
        "molecule": case.metadata,
        "input_pilot": str(PILOT_INPUT),
        "input_pilot_fragment_sha256": sha256(
            PILOT_INPUT / "selected_fragments.npz"
        ),
        "script_sha256": sha256(Path(__file__)),
        "output": str(output),
        "protected_formal_results": str(FORMAL_RESULTS),
        "recommended_future_tex_location": {
            "path": str(TEX_CANDIDATE),
            "insertion": (
                "after the existing hybrid_ansatz section ending near lines "
                "2438-2445 and before Primary literature near line 2447"
            ),
            "document_modified_by_this_run": False,
        },
        "calibration": {
            "probe_definition": (
                "28 one-state determinant groups plus 118 unordered determinant-pair "
                "groups; every pair group contains all four equal-amplitude Clifford "
                "states with relative phases 0, pi/2, pi, and 3pi/2"
            ),
            "state_group_split": {"train": 60, "validation": 20, "test": 20},
            "group_split_rule": (
                "All four phases of one unordered determinant pair share one group_id "
                "and one train/validation/test assignment. The split is group-disjoint "
                "but intentionally not determinant-support-disjoint: a determinant may "
                "also occur in a different pair group assigned to another split."
            ),
            "weights": weights,
            "source_trajectory": "saved depth-one LiH pilot replay",
            "predictive_caveat": (
                "These molecule/geometry-specific pilot weights are diagnostics, not "
                "a transferable predictor. Validation/test fit quality, including "
                "possibly weak or negative R2, is reported without clipping in "
                "stabilizer_weights.json."
            ),
        },
        "dynamic_depth_frontier": {
            "maximum_source_terms": args.max_terms,
            "block_size": BLOCK_SIZE,
            "circuit_depths_compared": list(DEPTHS),
            "circuit_depth_constraint": "integer 1 <= d < 8",
            "depth_selection": (
                "ascending d=1,...,7; stop at the first trial without more than "
                f"{DEPTH_RELATIVE_IMPROVEMENT:g} relative endpoint-F improvement "
                "and retain the latest beneficial depth"
            ),
            "RCDF_depth": None,
            "family_selection": "minimum block-end full-Fock Frobenius distance; tie -> declared family order",
            "offline_K_selection_note": (
                "This file remains the calibration/exploration pilot: one F-dominant "
                "frontier is built once and K is selected offline. The production v3 "
                "runner reruns family/depth selection for every T using frozen loss."
            ),
            "extension_policy": (
                "The diagnostic frontier intentionally continues to max_terms even "
                "after F < 1 so that finite-shot K(T) optima can be checked for right censoring; "
                "it is not the formal stopping run."
            ),
            "accepted_schedule": [
                {
                    "K": index + 1,
                    "family": fragment.family,
                    "family_label": fragment.family_label,
                    "chosen_depth": None if dynamic_depths[index] < 0 else dynamic_depths[index],
                }
                for index, fragment in enumerate(dynamic_fragments)
            ],
        },
        "AGPD_shallow_collector_policy": {
            "source_to_collector_transfer_families": [],
            "complete_native_source_families": [
                LABELS[value] for value in CIRCUIT_FAMILIES
            ],
            "source_leaf_range": "half native diagonal maximum-minus-minimum",
            "one_body_cover": {
                "depth_grid": list(AGPD_SHALLOW_COLLECTOR_DEPTHS),
                "selection": "smallest strictly feasible depth",
                "minimum_extra_leaves": AGPD_SHALLOW_COLLECTOR_MIN_EXTRA_LEAVES,
                "maximum_extra_leaves": AGPD_SHALLOW_COLLECTOR_MAX_EXTRA_LEAVES,
                "equality_tolerance": AGPD_SHALLOW_COLLECTOR_EQUALITY_TOLERANCE,
                "dense_replay_tolerance": AGPD_SHALLOW_COLLECTOR_MATRIX_TOLERANCE,
                "independent_collector_setting_present": False,
            },
            "independent_dense_replay": {
                "operators": "LiH_active_case_operators.npz",
                "source_primitives": (
                    "pilot_replayed_fragments.npz or dynamic_depth_fragments.npz; "
                    "every source matrix and native diagonal is retained whole"
                ),
                "formula": (
                    "H_tilde_K=sum(shallow_one_body_cover_leaves)+"
                    "sum(complete_native_source_leaves)"
                ),
            },
        },
        "shots": {
            "T_grid": list(shot_grid),
            "allocation": (
                "minimum-one proportional-range heuristic: one shot per "
                "nonconstant setting, then largest-remainder apportionment of "
                "the remaining T-m shots by lambda; not claimed discrete-optimal"
            ),
            "replay_tables": [
                "stabilizer_calibration_shot_allocations.csv",
                "LiH_dynamic_shot_allocations_by_T_K.csv",
            ],
            "Monte_Carlo_repeats": args.sampling_repeats,
        },
        "complexity_scope": (
            "The analytic ground-state sampling SE uses matrix-vector actions "
            "at at most O(2^(2n)) in this dense representation and does not "
            "require dense Hermitian diagonalization. Dense 2^8 source optimization, "
            "spectral-range eigendecompositions, and Born multinomials are validation "
            "oracles only. A production path must obtain source ranges/outcomes from "
            "native diagonals and circuit-basis sampling, or use an admissible fast "
            "Hermitian method; the formal algorithm must not depend on O(2^(3n)) LAPACK."
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "audit_status": audit["status"],
    }
    write_json(output / "manifest.json", manifest)
    readme = f"""# LiH stabilizer-calibration and dynamic-depth pilot

This isolated directory was generated by `stabilizer_calibration.py`.
It does not replace or modify the formal benchmark under `{FORMAL_RESULTS}`.

The saved depth-one LiH pilot supplies a frozen calibration trajectory.  The
100 fixed-particle stabilizer probes are split by probe identity into 60 train,
20 validation, and 20 test states.  Each unordered-pair group keeps all four
phases in one split.  This is group-disjoint, not determinant-support-disjoint:
a determinant may occur in another pair group assigned to another split.  Two
non-negative squared-component fits
produce `weight_approximation` and `weight_sampling`.  The model contains only
these learned components.  The approximation feature is the unnormalized
sector Frobenius distance `||P(H-H_hat_K)P||_F`; no factor of `1/n^2` is used.

The exploratory source frontier scans circuit depths 1 through 7 in ascending
order for GFRO, Operator pool, NN-Pair, and iSWAP+SU(2) at every four-term
boundary, stopping within a family at the first nonmaterial improvement.  A
separate historical RCDF trial records depth as N/A.  This calibration pilot
uses endpoint Frobenius distance for its family choice; the production runner
instead applies the frozen balanced loss and reruns the family schedule for
every shot budget T.
The frontier deliberately continues to `max_terms` after F falls below 1; this
is an extended K(T) diagnostic, not the formal stopping trajectory.

Every AGPD family retains its complete native shallow source setting.  No
source direction is transferred.  The fixed molecular scalar-plus-one-body
reference is expanded into the minimum feasible depth-bounded one-body cover,
so a prefix has `K+L_A` settings and no independent dense collector.  Source
ranges come from native diagonal maxima and minima; cover ranges come from the
corresponding occupation-linear diagonals.

Important complexity boundary: every dense 256-by-256 matrix operation, exact
source optimization, spectral-range eigendecomposition, and Born multinomial in
this folder is an eight-qubit validation oracle.  The analytic ground-state
sampling standard error itself only needs setting matrix-vector actions here
(at most O(2^(2n))) and can be obtained from native setting bases in production;
it does not require the dense Hermitian eigendecompositions used for Monte Carlo
validation.  A scalable implementation must use native diagonals/circuit-basis
sampling or an admissible fast Hermitian method and must not depend on
O(2^(3n)) LAPACK.

Main files:

- `stabilizer_weights.json`: learned weights and train/validation/test metrics;
- `stabilizer_approximation_components.csv`: explicit exact energy E,
  approximate mean mu_K, and the signed convention mu_K-E for every probe/K;
- `stabilizer_calibration_shot_allocations.csv` and
  `LiH_dynamic_shot_allocations_by_T_K.csv`: complete integer shot vectors;
- `dynamic_depth_all_trials.csv`: all depths for all circuit families plus RCDF;
- `dynamic_depth_metrics_by_k.csv`: accepted family and chosen depth for every K;
- `LiH_K_and_ground_sampling_error_by_T.csv`: K(T), analytic errors, and Monte Carlo errors;
- `LiH_ground_sampling_replicates.csv`: deterministic replicate-level errors;
- `pilot_replayed_fragments.npz` and `dynamic_depth_fragments.npz`: full source,
  one-body, and diagnostic transfer-direction matrices plus native diagonals;
  `source_to_collector_transfer_mask` is explicitly all false;
- `audit.json`: replay, reconstruction, range, shot, and protected-file checks.

Every prefix is independently reconstructible from
`LiH_active_case_operators.npz`, the corresponding fragment NPZ, and its
shallow-cover setting table.  The dense replay is the sum of all shallow
one-body cover leaves plus all complete native source leaves.

Shot allocation is a minimum-one proportional-range heuristic: each
nonconstant setting first receives one shot, and the remaining shots are
apportioned by lambda with largest remainders.  It is internally replayable
from the saved vectors, but is not claimed to be the discrete-optimal range
allocation.

The fitted sampling weight is a molecule- and geometry-specific diagnostic.
Held-out scores (including weak or negative R-squared values) are retained in
`stabilizer_weights.json`; they must not be presented as a validated universal
predictor.

If this method is later documented, the most coherent TeX insertion point is
`{TEX_CANDIDATE}`, after the existing `hybrid_ansatz` section and immediately
before `Primary literature`.  This run did not edit that file or
`{ALGORITHM_DOCUMENT}`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(f"Pilot audit failed; see {output / 'audit.json'}")
    print(f"[done] PASS: {output}", flush=True)


if __name__ == "__main__":
    main()
