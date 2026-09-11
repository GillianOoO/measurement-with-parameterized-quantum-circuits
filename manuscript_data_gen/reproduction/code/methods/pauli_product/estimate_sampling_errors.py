#!/usr/bin/env python3
"""Estimate finite-shot energy errors for the selected molecular H-hat models.

This is a scalable adaptation of
``Decompose-by-tensornetwork-feature-KongGit/estimation.py``.  It keeps the
legacy importance rule, shot grid, RMSE definition, and 50 independent
repetitions, while reading the saved TND/GPD MPO decomposition terms directly.
Unlike the legacy use of ``floor(T * p_k)``, the integer allocation uses the
largest-remainder rule so that the executed total is exactly T (in particular,
the corrected 7259-shot point really uses 7259 samples).

For a saved approximation

    H_hat_K = sum_k alpha_k U_k^dagger Lambda_k U_k,

each term is measured by sampling the computational-basis distribution of
``U_k |psi_0>`` and evaluating the real diagonal of ``alpha_k Lambda_k``.
The exact FCI ground-state MPS is used as the input state, matching the legacy
program's use of an exact ground-state vector for simulator experiments.
"""

from __future__ import annotations

import hashlib
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-lih-h2o-n2-sampling")
# Avoid the macOS Accelerate/OpenBLAS abort observed when the 20-qubit N2
# probability vectors and Matplotlib are handled in the same process.
for thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = "1"

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SCAN_DIR = HERE.parent
CURVES_DIR = SCAN_DIR.parent
SELECTED_DIR = CURVES_DIR / "mps_bond_dimension_analysis" / "selected_hamiltonians"
DECOMPOSITIONS_DIR = SCAN_DIR / "decompositions"

SHOT_BUDGETS = (12, 45, 160, 572, 2038, 7259, 25848, 92041)
REPEATS = 50
SEED = 7259

RX_RY_RX_AXES = (0, 1, 2)
ISWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.complex128,
)


@dataclass(frozen=True)
class Case:
    molecule: str
    bond_length_angstrom: float
    selected_directory: str
    run_directory: str
    method: str
    mpo_bond_dimension: int
    circuit_depth: int
    decomposition_terms: int


CASES = (
    Case(
        molecule="LiH",
        bond_length_angstrom=1.50,
        selected_directory="01_LiH_R1.50A",
        run_directory="LiH_R1.50A_TND_L3_chi64",
        method="TND",
        mpo_bond_dimension=64,
        circuit_depth=3,
        decomposition_terms=2,
    ),
    Case(
        molecule="H2O",
        bond_length_angstrom=0.80,
        selected_directory="02_H2O_R0.80A",
        run_directory="H2O_R0.80A_TND_L3_chi64",
        method="TND",
        mpo_bond_dimension=64,
        circuit_depth=3,
        decomposition_terms=3,
    ),
    Case(
        molecule="N2",
        bond_length_angstrom=2.25,
        selected_directory="03_N2_R2.25A",
        run_directory="N2_R2.25A_GPD-MPO-reference_L3_chi128",
        method="GPD-MPO-reference",
        mpo_bond_dimension=128,
        circuit_depth=3,
        decomposition_terms=5,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dense_mps(path: Path) -> Tuple[np.ndarray, int]:
    """Contract arrays A000... into a state ordered by sites 0...n-1."""
    with np.load(path, allow_pickle=False) as payload:
        keys = sorted(payload.files)
        arrays = [np.asarray(payload[key]) for key in keys]
    if not arrays:
        raise ValueError(f"No MPS arrays found in {path}")
    if any(array.ndim != 3 or array.shape[1] != 2 for array in arrays):
        raise ValueError(f"Unexpected MPS tensor convention in {path}")
    state = arrays[0][0, :, :]
    for array in arrays[1:]:
        state = np.tensordot(state, array, axes=(-1, 0))
    state = np.asarray(state[..., 0], dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid MPS norm in {path}: {norm}")
    state /= norm
    return state, max(max(array.shape[0], array.shape[2]) for array in arrays)


def _diagonal_site_tensor(array: np.ndarray, site: int, nsites: int) -> np.ndarray:
    """Return one diagonal MPO tensor with shape (left, physical, right)."""
    diagonal = np.diagonal(array, axis1=-2, axis2=-1)
    if array.ndim == 4:
        return diagonal.transpose(0, 2, 1)
    if array.ndim != 3:
        raise ValueError(f"Unexpected MPO tensor rank {array.ndim}")
    if site == 0:
        return diagonal.T[None, :, :]
    if site == nsites - 1:
        return diagonal[:, :, None]
    raise ValueError("Only boundary MPO tensors may have rank three")


def load_diagonal_mpo(path: Path) -> Tuple[np.ndarray, Dict[str, object], float]:
    """Contract the computational-basis diagonal of a saved MPO."""
    with np.load(path, allow_pickle=False) as payload:
        nsites = int(payload["number_sites"][0])
        arrays = [
            np.asarray(payload[f"tensor_{site:03d}"]) for site in range(nsites)
        ]
        metadata = json.loads(str(payload["metadata_json"][0]))

    values = np.ones((1, 1), dtype=np.complex128)
    for site, array in enumerate(arrays):
        local = _diagonal_site_tensor(array, site, nsites)
        values = np.einsum("cl,lsr->csr", values, local, optimize=True)
        values = values.reshape(-1, local.shape[2])
    if values.shape[1] != 1:
        raise ValueError(f"Uncontracted right MPO bond in {path}: {values.shape}")

    complex_diagonal = values[:, 0]
    imaginary_max = float(np.max(np.abs(np.imag(complex_diagonal))))
    # The original estimation code saves/uses real Lambda diagonals.  Finite
    # MPO recompression can leave a small (and for later terms non-negligible)
    # anti-Hermitian numerical component, which is not a measurable observable.
    real_diagonal = np.real(complex_diagonal).astype(np.float64, copy=False)
    return real_diagonal, metadata, imaginary_max


def rx(theta: float) -> np.ndarray:
    cosine = math.cos(theta / 2.0)
    sine = math.sin(theta / 2.0)
    return np.array(
        [[cosine, -1j * sine], [-1j * sine, cosine]],
        dtype=np.complex128,
    )


def ry(theta: float) -> np.ndarray:
    cosine = math.cos(theta / 2.0)
    sine = math.sin(theta / 2.0)
    return np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.complex128,
    )


def apply_one_site_gate(
    state: np.ndarray, gate: np.ndarray, site: int, nsites: int
) -> np.ndarray:
    tensor = state.reshape((2,) * nsites)
    tensor = np.moveaxis(tensor, site, 0).reshape(2, -1)
    tensor = gate @ tensor
    tensor = tensor.reshape((2,) + (2,) * (nsites - 1))
    return np.moveaxis(tensor, 0, site).reshape(-1)


def apply_two_site_gate(
    state: np.ndarray, gate: np.ndarray, site: int, nsites: int
) -> np.ndarray:
    if site + 1 >= nsites:
        raise ValueError("Two-site gate exceeds the chain")
    tensor = state.reshape((2,) * nsites)
    tensor = np.moveaxis(tensor, (site, site + 1), (0, 1)).reshape(4, -1)
    tensor = gate.reshape(4, 4) @ tensor
    tensor = tensor.reshape((2, 2) + (2,) * (nsites - 2))
    tensor = np.moveaxis(tensor, (0, 1), (site, site + 1))
    return tensor.reshape(-1)


def apply_ansatz(state: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Apply the same forward U(theta) used to create U^dag Lambda U."""
    depth, nsites, axes = theta.shape
    if axes != 3:
        raise ValueError(f"Expected theta shape (depth, nsites, 3), got {theta.shape}")
    transformed = np.array(state, dtype=np.complex128, copy=True)
    for layer in range(depth):
        for site in range(layer % 2, nsites - 1, 2):
            transformed = apply_two_site_gate(
                transformed, ISWAP, site, nsites
            )
        for axis, gate_function in ((0, rx), (1, ry), (2, rx)):
            for site in range(nsites):
                angle = float(theta[layer, site, axis])
                if abs(angle) > 1e-15:
                    transformed = apply_one_site_gate(
                        transformed, gate_function(angle), site, nsites
                    )
    return transformed


def importance_shot_allocation(
    total_shots: int, probabilities: np.ndarray
) -> np.ndarray:
    """Integerize the legacy importance weights while using exactly T shots."""
    quotas = total_shots * probabilities
    allocation = np.floor(quotas).astype(np.int64)
    missing = int(total_shots - allocation.sum())
    if missing:
        fractional = quotas - allocation
        order = np.argsort(-fractional, kind="stable")
        allocation[order[:missing]] += 1
    if int(allocation.sum()) != total_shots:
        raise ValueError(
            f"Shot allocation sums to {allocation.sum()}, expected {total_shots}"
        )
    return allocation


def sample_observable_mean(
    cdf: np.ndarray,
    observable: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> float:
    if shots <= 0:
        return 0.0
    uniforms = rng.random(shots)
    outcomes = np.searchsorted(cdf, uniforms, side="right")
    outcomes = np.minimum(outcomes, observable.size - 1)
    return float(np.mean(observable[outcomes]))


def prepare_case(case: Case) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    selected = SELECTED_DIR / case.selected_directory
    run_dir = DECOMPOSITIONS_DIR / case.run_directory
    metadata_path = selected / "metadata.json"
    state_path = selected / "ground_state_mps_blocked_spin.npz"
    run_manifest_path = run_dir / "run_manifest.json"

    with metadata_path.open() as handle:
        molecular_metadata = json.load(handle)
    with run_manifest_path.open() as handle:
        run_manifest = json.load(handle)

    config = run_manifest["config"]
    expected = (
        config["method"],
        int(config["depth"]),
        int(config["mpo_bond"]),
    )
    requested = (case.method, case.circuit_depth, case.mpo_bond_dimension)
    if expected != requested:
        raise ValueError(
            f"{case.molecule}: run manifest {expected} does not match {requested}"
        )

    state, state_max_bond = load_dense_mps(state_path)
    nsites = int(round(math.log2(state.size)))
    if 2**nsites != state.size:
        raise ValueError(f"{case.molecule}: state length is not a power of two")

    terms: List[Dict[str, object]] = []
    for term_index in range(1, case.decomposition_terms + 1):
        term_dir = run_dir / f"term_{term_index:02d}"
        theta_path = term_dir / "theta.npy"
        diagonal_path = term_dir / "lambda_diagonal_mpo.npz"
        theta = np.load(theta_path, allow_pickle=False)
        if theta.shape != (case.circuit_depth, nsites, 3):
            raise ValueError(
                f"{case.molecule} K={term_index}: unexpected theta {theta.shape}"
            )
        diagonal, diagonal_metadata, imaginary_max = load_diagonal_mpo(
            diagonal_path
        )
        alpha = float(diagonal_metadata["alpha"])
        observable = alpha * diagonal
        transformed = apply_ansatz(state, theta)
        probabilities = np.abs(transformed) ** 2
        probability_sum = float(probabilities.sum())
        probabilities /= probability_sum
        cdf = np.cumsum(probabilities)
        cdf[-1] = 1.0
        exact_mean = float(np.dot(probabilities, observable))
        exact_variance = float(
            max(np.dot(probabilities, observable * observable) - exact_mean**2, 0.0)
        )
        terms.append(
            {
                "term": term_index,
                "alpha": alpha,
                "observable": observable,
                "cdf": cdf,
                "exact_mean": exact_mean,
                "exact_variance": exact_variance,
                "importance": float(np.max(np.abs(observable))),
                "probability_normalization_before_rescale": probability_sum,
                "lambda_imaginary_max_before_real_projection": imaginary_max,
                "theta_sha256": sha256_file(theta_path),
                "lambda_mpo_sha256": sha256_file(diagonal_path),
            }
        )

    importances = np.array([term["importance"] for term in terms], dtype=float)
    if not np.all(np.isfinite(importances)) or np.any(importances <= 0.0):
        raise ValueError(f"{case.molecule}: invalid term importances {importances}")
    importance_probabilities = importances / importances.sum()
    for term, probability in zip(terms, importance_probabilities):
        term["importance_probability"] = float(probability)

    exact_hhat_energy = float(sum(term["exact_mean"] for term in terms))
    exact_h_energy = float(molecular_metadata["fci_ground_energy_hartree"])
    summary = {
        "molecule": case.molecule,
        "bond_length_angstrom": case.bond_length_angstrom,
        "number_qubits": nsites,
        "method": case.method,
        "mpo_bond_dimension": case.mpo_bond_dimension,
        "circuit_depth": case.circuit_depth,
        "decomposition_terms": case.decomposition_terms,
        "repeat_count": REPEATS,
        "exact_H_ground_energy_hartree": exact_h_energy,
        "exact_Hhat_expectation_hartree": exact_hhat_energy,
        "signed_decomposition_bias_hartree": exact_hhat_energy - exact_h_energy,
        "absolute_decomposition_bias_hartree": abs(
            exact_hhat_energy - exact_h_energy
        ),
        "ground_state_mps_max_bond": state_max_bond,
        "ground_state_mps_sha256": sha256_file(state_path),
        "metadata_sha256": sha256_file(metadata_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
    }
    return terms, summary


def estimate_case(
    case: Case,
    terms: Sequence[Dict[str, object]],
    case_summary: Dict[str, object],
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    exact_h_energy = float(case_summary["exact_H_ground_energy_hartree"])
    exact_hhat_energy = float(case_summary["exact_Hhat_expectation_hartree"])
    probabilities = np.array(
        [term["importance_probability"] for term in terms], dtype=float
    )

    repeat_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for requested_shots in SHOT_BUDGETS:
        allocation = importance_shot_allocation(requested_shots, probabilities)
        effective_shots = int(allocation.sum())
        expected_finite_shot_mean = float(
            sum(
                float(term["exact_mean"])
                for term, shots in zip(terms, allocation)
                if shots > 0
            )
        )
        predicted_variance = float(
            sum(
                float(term["exact_variance"]) / int(shots)
                for term, shots in zip(terms, allocation)
                if shots > 0
            )
        )
        estimates = np.empty(REPEATS, dtype=float)
        for repeat in range(REPEATS):
            estimate = 0.0
            for term, shots in zip(terms, allocation):
                estimate += sample_observable_mean(
                    np.asarray(term["cdf"]),
                    np.asarray(term["observable"]),
                    int(shots),
                    rng,
                )
            estimates[repeat] = estimate
            repeat_rows.append(
                {
                    "molecule": case.molecule,
                    "method": case.method,
                    "mpo_bond_dimension": case.mpo_bond_dimension,
                    "circuit_depth": case.circuit_depth,
                    "decomposition_terms": case.decomposition_terms,
                    "requested_shots": requested_shots,
                    "effective_shots": effective_shots,
                    "repeat": repeat + 1,
                    "energy_estimate_hartree": estimate,
                    "signed_error_vs_exact_H_hartree": estimate - exact_h_energy,
                    "absolute_error_vs_exact_H_hartree": abs(
                        estimate - exact_h_energy
                    ),
                    "signed_error_vs_Hhat_hartree": estimate - exact_hhat_energy,
                    "absolute_error_vs_Hhat_hartree": abs(
                        estimate - exact_hhat_energy
                    ),
                }
            )

        residuals_h = estimates - exact_h_energy
        residuals_hhat = estimates - exact_hhat_energy
        predicted_rmse_h = math.sqrt(
            predicted_variance + (expected_finite_shot_mean - exact_h_energy) ** 2
        )
        predicted_rmse_hhat = math.sqrt(
            predicted_variance + (expected_finite_shot_mean - exact_hhat_energy) ** 2
        )
        summary_rows.append(
            {
                "molecule": case.molecule,
                "bond_length_angstrom": case.bond_length_angstrom,
                "number_qubits": case_summary["number_qubits"],
                "method": case.method,
                "mpo_bond_dimension": case.mpo_bond_dimension,
                "circuit_depth": case.circuit_depth,
                "decomposition_terms": case.decomposition_terms,
                "repeat_count": REPEATS,
                "requested_shots": requested_shots,
                "effective_shots": effective_shots,
                "shot_allocation_per_term": ";".join(
                    str(int(value)) for value in allocation
                ),
                "terms_receiving_zero_shots": int(np.count_nonzero(allocation == 0)),
                "exact_H_ground_energy_hartree": exact_h_energy,
                "exact_Hhat_expectation_hartree": exact_hhat_energy,
                "absolute_decomposition_bias_hartree": abs(
                    exact_hhat_energy - exact_h_energy
                ),
                "mean_estimated_energy_hartree": float(np.mean(estimates)),
                "empirical_rmse_vs_exact_H_hartree": float(
                    np.sqrt(np.mean(residuals_h**2))
                ),
                "empirical_rmse_vs_Hhat_hartree": float(
                    np.sqrt(np.mean(residuals_hhat**2))
                ),
                "empirical_mae_vs_exact_H_hartree": float(
                    np.mean(np.abs(residuals_h))
                ),
                "empirical_standard_deviation_hartree": float(
                    np.std(estimates, ddof=1)
                ),
                "predicted_rmse_vs_exact_H_hartree": predicted_rmse_h,
                "predicted_rmse_vs_Hhat_hartree": predicted_rmse_hhat,
                "predicted_standard_deviation_hartree": math.sqrt(
                    predicted_variance
                ),
                "expected_estimator_mean_with_integer_allocation_hartree": (
                    expected_finite_shot_mean
                ),
            }
        )
    return repeat_rows, summary_rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    all_repeat_rows: List[Dict[str, object]] = []
    all_summary_rows: List[Dict[str, object]] = []
    case_manifests: List[Dict[str, object]] = []

    for case in CASES:
        print(
            f"[prepare] {case.molecule}: {case.method}, "
            f"chi={case.mpo_bond_dimension}, L={case.circuit_depth}, "
            f"K={case.decomposition_terms}",
            flush=True,
        )
        terms, case_summary = prepare_case(case)
        repeat_rows, summary_rows = estimate_case(case, terms, case_summary, rng)
        all_repeat_rows.extend(repeat_rows)
        all_summary_rows.extend(summary_rows)
        case_manifests.append(
            {
                **case_summary,
                "terms": [
                    {
                        key: value
                        for key, value in term.items()
                        if key not in {"observable", "cdf"}
                    }
                    for term in terms
                ],
            }
        )
        # The N2 CDF and observable arrays occupy tens of MiB per term.  They
        # are no longer needed after sampling, so release them before the plot
        # backend allocates its PDF/PNG rendering buffers.
        del terms
        gc.collect()

    repeats = pd.DataFrame(all_repeat_rows)
    summary = pd.DataFrame(all_summary_rows)
    repeats.to_csv(HERE / "sampling_error_repeats.csv", index=False)
    summary.to_csv(HERE / "sampling_error_summary.csv", index=False)

    manifest = {
        "seed": SEED,
        "shot_budgets": list(SHOT_BUDGETS),
        "repeat_count": REPEATS,
        "error_definition": (
            "sqrt(mean((estimated_energy-reference_energy)^2)) over repeats"
        ),
        "shot_allocation": (
            "legacy importance allocation: p_k proportional to "
            "max(abs(alpha_k * real(diag(Lambda_k)))); integerized by the "
            "largest-remainder rule so sum_k n_k equals T exactly"
        ),
        "corrected_legacy_shot_label": {"from": 7256, "to": 7259},
        "input_state": "exact FCI ground-state MPS in blocked-spin site order",
        "complex_diagonal_handling": (
            "use real part, matching legacy estimation.py measurable-observable path"
        ),
        "cases": case_manifests,
        "outputs": {
            name: sha256_file(HERE / name)
            for name in (
                "sampling_error_repeats.csv",
                "sampling_error_summary.csv",
            )
        },
    }
    (HERE / "sampling_error_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    print("\n[complete]", flush=True)
    for row in summary[summary["requested_shots"] == max(SHOT_BUDGETS)].itertuples():
        print(
            f"  {row.molecule}: RMSE(H)={row.empirical_rmse_vs_exact_H_hartree:.6e}, "
            f"RMSE(Hhat)={row.empirical_rmse_vs_Hhat_hartree:.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()

