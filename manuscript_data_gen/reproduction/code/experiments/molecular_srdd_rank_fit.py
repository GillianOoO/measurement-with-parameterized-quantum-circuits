#!/usr/bin/env python3
"""Independent shallow-RC-DF-F benchmark on the five active-space targets.

This is deliberately *not* an ansatz-pool member.  At every source rank K and
every shallow rotation depth d_R in {1,2,3}, all K density-density leaves and
all even/odd Givens angles are jointly refitted to the original Hamiltonian.
The d_R trial with the smallest physical-sector Frobenius residual is retained.
F3-R2 is then replayed on the retained complete rank-K factorization.  Frozen
stabilizer weights select K separately for every total-shot budget, after which
the exact target ground state is used only as a sampling benchmark oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
SRDD_METHODS = HERE.parent / "methods" / "srdd"
sys.path.insert(0, str(SRDD_METHODS))
from srdd_release_paths import DATA, RUNS, molecular_result

from adaptive_f3_pool import (  # noqa: E402
    SHALLOW_RCDF_DEPTHS,
    VERSION as CORE_VERSION,
    hermitian,
    make_shallow_rcdf_bank,
)
from five_molecule_cases import load_five_molecule_case  # noqa: E402
from stabilizer_calibration import (  # noqa: E402
    build_minimum_depth_zero_rank_shallow_collector,
    build_shallow_srcdf_prefix_decomposition,
    ground_state,
    integer_range_allocation,
    monte_carlo_estimates,
    outcome_models,
)


VERSION = "standalone-shallow-rcdf-joint-rank-k-v3-shallow-collector"
MOLECULES = ("H4", "H6")
DEFAULT_SHOTS = (100, 200, 300, 500, 800, 1_200, 2_000, 3_000)
DEFAULT_HYBRID_SOURCE = DATA
DEFAULT_OUTPUT = RUNS / "molecular_srdd_rank_fits"
SELECTION_TOLERANCE = 1.0e-12


def collector_kwargs(args) -> dict[str, Any]:
    return {
        "collector_mode": args.collector_mode,
        "collector_min_extra_leaves": args.collector_min_extra_leaves,
        "collector_max_extra_leaves": args.collector_max_extra_leaves,
        "collector_objective": args.collector_objective,
        "collector_proxy_state": args.collector_proxy_state,
        "collector_reference_shots": args.collector_reference_shots,
        "collector_proxy_cycles": args.collector_proxy_cycles,
        "collector_equality_tolerance": args.collector_equality_tolerance,
        "collector_extra_angle_steps": args.collector_extra_angle_steps,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=json_default) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sampling_seed(molecule: str, total_shots: int, k: int) -> int:
    payload = f"{VERSION}|{molecule}|T{total_shots}|K{k}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def approximation_matrix(case, fragments) -> np.ndarray:
    return case.base_collector + sum(
        (fragment.matrix for fragment in fragments), start=np.zeros_like(case.target)
    )


def residual_metrics(case, fragments) -> dict[str, float]:
    residual = hermitian(case.target - approximation_matrix(case, fragments))
    sector = residual[np.ix_(case.sector_indices, case.sector_indices)]
    return {
        "full_Frobenius_distance": float(np.linalg.norm(residual, "fro")),
        "sector_Frobenius_distance": float(np.linalg.norm(sector, "fro")),
    }


def save_selected_rank(path: Path, case, fragments, prefix, depth: int) -> None:
    np.savez_compressed(
        path,
        target=case.target,
        base_collector=case.base_collector,
        sector_indices=case.sector_indices,
        source_matrices=np.asarray([fragment.matrix for fragment in fragments]),
        source_diagonals=np.asarray([fragment.diagonal for fragment in fragments]),
        source_one_matrices=np.asarray([fragment.one_matrix for fragment in fragments]),
        source_q_directions=np.asarray([fragment.q_direction for fragment in fragments]),
        source_one_diagonals=np.asarray([fragment.one_diagonal for fragment in fragments]),
        source_q_diagonals=np.asarray([fragment.q_diagonal for fragment in fragments]),
        spatial_rotations=np.asarray(
            [fragment.candidate_metadata["spatial_rotation"] for fragment in fragments]
        ),
        z_tensors=np.asarray(
            [fragment.candidate_metadata["z_tensor"] for fragment in fragments]
        ),
        selected_depth=np.asarray(depth),
        f3_alpha=prefix.alpha_by_source,
        setting_matrices=np.asarray([setting.matrix for setting in prefix.settings]),
        setting_centered_ranges=np.asarray(
            [setting.centered_range for setting in prefix.settings]
        ),
        setting_rotations=np.asarray(
            [setting.rotation for setting in prefix.settings], dtype=float
        ),
        setting_native_diagonals=np.asarray(
            [setting.native_diagonal for setting in prefix.settings], dtype=float
        ),
        setting_kinds=np.asarray(
            [setting.setting_kind for setting in prefix.settings]
        ),
        setting_collector_linear_coefficients=np.asarray(
            [setting.collector_linear_coefficients for setting in prefix.settings],
            dtype=float,
        ),
        collector_audit_json=np.asarray(json.dumps(prefix.collector_audit)),
    )


def fit_rank(case, k: int, args, case_output: Path):
    candidates = []
    depth_rows = []
    for depth in SHALLOW_RCDF_DEPTHS:
        cache = case_output / "rank_depth" / f"K{k:02d}_dR{depth}.npz"
        # The cache stores complete fragments, but we intentionally do not try to
        # reverse-engineer Fragment objects from it.  Resume is rank-level: a
        # selected-rank artifact is consumed only after the corresponding JSON is
        # complete, otherwise this finite optimization is rerun.
        started = time.perf_counter()
        bank, bank_audit = make_shallow_rcdf_bank(
            case,
            k,
            depth,
            args.rho,
            args.cycles,
            args.angle_steps,
        )
        metrics = residual_metrics(case, bank)
        row = {
            "molecule": case.name,
            "K_source_terms": k,
            "shallow_rotation_depth": depth,
            **metrics,
            "eri_residual_frobenius": float(bank_audit["eri_residual_frobenius"]),
            "full_fock_operator_residual_frobenius": float(
                bank_audit["full_fock_operator_residual_frobenius"]
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "selected_depth": False,
            "fit_semantics": "independent_joint_rank_K_refit",
        }
        depth_rows.append(row)
        candidates.append((row, bank, bank_audit))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            source_matrices=np.asarray([fragment.matrix for fragment in bank]),
            source_diagonals=np.asarray([fragment.diagonal for fragment in bank]),
            spatial_rotations=np.asarray(
                [fragment.candidate_metadata["spatial_rotation"] for fragment in bank]
            ),
            z_tensors=np.asarray(
                [fragment.candidate_metadata["z_tensor"] for fragment in bank]
            ),
        )
    minimum = min(value[0]["sector_Frobenius_distance"] for value in candidates)
    tied = [
        value
        for value in candidates
        if value[0]["sector_Frobenius_distance"] <= minimum + SELECTION_TOLERANCE
    ]
    selected = min(tied, key=lambda value: value[0]["shallow_rotation_depth"])
    selected[0]["selected_depth"] = True
    return selected, depth_rows


def run_case(args, molecule: str, output: Path):
    case = load_five_molecule_case(molecule)
    case_output = output / molecule
    case_output.mkdir(parents=True, exist_ok=True)
    weights_path = molecular_result(args.weight_source, molecule, "selected") / "stabilizer_weights.json"
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    weight_a = float(weights["weight_approximation"])
    weight_s = float(weights["weight_sampling"])
    ground, exact_energy = ground_state(case)

    base_prefix, base_depth = build_minimum_depth_zero_rank_shallow_collector(
        case,
        args.collector_zero_rank_depths,
        args.spectral_max_evaluations,
        **collector_kwargs(args),
    )
    prefixes = [base_prefix]
    fragment_banks = [[]]
    rank_rows = [{
        "molecule": molecule,
        "K_source_terms": 0,
        "shallow_rotation_depth": base_depth,
        **residual_metrics(case, []),
        "F3_reconstruction_error": prefixes[0].reconstruction_error,
        "centered_spectral_norm_sum": prefixes[0].spectral_sum,
        "centered_spectral_norm_sum_before_R2": prefixes[0].spectral_sum_before_r2,
        "measurement_settings": len(prefixes[0].settings),
        "collector_extra_settings": int(
            prefixes[0].collector_audit.get("extra_rotation_count", 0)
        ),
        "collector_dictionary_rank": int(
            prefixes[0].collector_audit.get("dictionary_rank", 0)
        ),
        "collector_reconstruction_residual": float(
            prefixes[0].collector_audit.get(
                "matrix_reconstruction_residual_frobenius", 0.0
            )
        ),
        "selected_depth": True,
        "fit_semantics": "base_one_body_shallow_collector_bank",
    }]
    depth_rows = []
    max_reconstruction = prefixes[0].reconstruction_error
    for k in range(1, args.max_rank + 1):
        print(f"[standalone s-RCDF] {molecule} joint refit K={k}", flush=True)
        (selected_row, bank, _), trials = fit_rank(case, k, args, case_output)
        prefix = build_shallow_srcdf_prefix_decomposition(
            case,
            bank,
            int(selected_row["shallow_rotation_depth"]),
            args.spectral_max_evaluations,
            **collector_kwargs(args),
        )
        selected_row = {
            **selected_row,
            "F3_reconstruction_error": prefix.reconstruction_error,
            "centered_spectral_norm_sum": prefix.spectral_sum,
            "centered_spectral_norm_sum_before_R2": prefix.spectral_sum_before_r2,
            "measurement_settings": len(prefix.settings),
            "collector_extra_settings": int(
                prefix.collector_audit.get("extra_rotation_count", 0)
            ),
            "collector_dictionary_rank": int(
                prefix.collector_audit.get("dictionary_rank", 0)
            ),
            "collector_reconstruction_residual": float(
                prefix.collector_audit.get(
                    "matrix_reconstruction_residual_frobenius", 0.0
                )
            ),
        }
        rank_rows.append(selected_row)
        depth_rows.extend(trials)
        prefixes.append(prefix)
        fragment_banks.append(bank)
        max_reconstruction = max(max_reconstruction, prefix.reconstruction_error)
        save_selected_rank(
            case_output / f"selected_rank_K{k:02d}.npz",
            case,
            bank,
            prefix,
            int(selected_row["shallow_rotation_depth"]),
        )

    candidate_rows = []
    summary_rows = []
    replicate_rows = []
    for total_shots in args.shots:
        candidates = []
        for row, prefix, bank in zip(rank_rows, prefixes, fragment_banks):
            lambdas = np.asarray(
                [setting.centered_range for setting in prefix.settings], dtype=float
            )
            shots = integer_range_allocation(total_shots, lambdas)
            active = shots > 0
            x_sampling = math.sqrt(
                max(float(np.sum(lambdas[active] ** 2 / shots[active])), 0.0)
            )
            x_approximation = float(row["sector_Frobenius_distance"])
            loss = math.hypot(weight_a * x_approximation, weight_s * x_sampling)
            record = {
                "molecule": molecule,
                "T_total_shots": total_shots,
                "K_source_terms": int(row["K_source_terms"]),
                "shallow_rotation_depth": int(row["shallow_rotation_depth"]),
                "sector_Frobenius_distance": x_approximation,
                "full_Frobenius_distance": float(row["full_Frobenius_distance"]),
                "x_sampling": x_sampling,
                "stabilizer_loss": loss,
                "total_settings": len(prefix.settings),
                "collector_mode": prefix.collector_audit.get(
                    "mode", args.collector_mode
                ),
                "collector_extra_settings": int(
                    prefix.collector_audit.get("extra_rotation_count", 0)
                ),
                "independent_collector_setting_present": bool(
                    prefix.collector_audit.get(
                        "independent_collector_setting_present", False
                    )
                ),
                "shot_vector": " ".join(str(int(value)) for value in shots),
            }
            candidate_rows.append(record)
            candidates.append((record, prefix, bank, shots))
        selected = min(
            candidates,
            key=lambda value: (value[0]["stabilizer_loss"], value[0]["K_source_terms"]),
        )
        record, prefix, bank, shots = selected
        if int(record["K_source_terms"]) == args.max_rank:
            raise RuntimeError(
                f"{molecule} T={total_shots}: the selected s-RCDF rank lies at "
                "the continuation endpoint; refusing to publish a right-censored "
                "rank selection. Increase --max-rank and rerun."
            )
        models = outcome_models(ground, prefix)
        approximate_energy = float(sum(model[2] for model in models))
        sampling_variance = float(
            sum(model[3] / count for model, count in zip(models, shots) if count > 0)
        )
        seed = sampling_seed(molecule, total_shots, int(record["K_source_terms"]))
        estimates = monte_carlo_estimates(models, shots, args.repeats, seed)
        errors = estimates - exact_energy
        summary_rows.append({
            **record,
            "right_censored_at_rank_10": False,
            "exact_ground_energy": exact_energy,
            "approximate_ground_state_mean": approximate_energy,
            "signed_approximation_bias": approximate_energy - exact_energy,
            "analytic_sampling_SE": math.sqrt(max(sampling_variance, 0.0)),
            "analytic_total_RMSE": math.hypot(
                approximate_energy - exact_energy,
                math.sqrt(max(sampling_variance, 0.0)),
            ),
            "empirical_total_RMSE": float(np.sqrt(np.mean(errors**2))),
            "empirical_total_MAE": float(np.mean(np.abs(errors))),
            "sampling_repeats": args.repeats,
            "sampling_seed": seed,
        })
        replicate_rows.extend(
            {
                "molecule": molecule,
                "T_total_shots": total_shots,
                "K_source_terms": int(record["K_source_terms"]),
                "repeat": repeat,
                "energy_estimate": float(value),
                "signed_total_error": float(value - exact_energy),
            }
            for repeat, value in enumerate(estimates)
        )

    write_csv(case_output / "rank_depth_trials.csv", depth_rows)
    write_csv(case_output / "rank_selected.csv", rank_rows)
    write_csv(case_output / "candidate_by_T_K.csv", candidate_rows)
    write_csv(case_output / "sampling_summary.csv", summary_rows)
    write_csv(case_output / "sampling_replicates.csv", replicate_rows)
    maximum_collector_residual = max(
        float(
            prefix.collector_audit.get(
                "matrix_relative_reconstruction_residual", 0.0
            )
        )
        for prefix in prefixes
    )
    all_settings_depth_limited = all(
        setting.rotation_depth is not None
        for prefix in prefixes
        for setting in prefix.settings
    )
    independent_collector_present = any(
        bool(
            prefix.collector_audit.get(
                "independent_collector_setting_present", False
            )
        )
        for prefix in prefixes
    )
    production_collector_pass = (
        args.collector_mode != "augment"
        or (
            all_settings_depth_limited
            and not independent_collector_present
            and maximum_collector_residual
            <= 5.0 * args.collector_equality_tolerance
            and all(
                int(prefix.collector_audit.get("extra_rotation_count", 0))
                >= args.collector_min_extra_leaves
                for prefix in prefixes
            )
        )
    )
    audit = {
        "status": (
            "PASS"
            if max_reconstruction <= 1.0e-6 and production_collector_pass
            else "FAIL"
        ),
        "molecule": molecule,
        "all_rank_K_points_independently_jointly_refitted": True,
        "depths": list(SHALLOW_RCDF_DEPTHS),
        "maximum_F3_reconstruction_error": max_reconstruction,
        "collector_mode": args.collector_mode,
        "collector_objective": args.collector_objective,
        "collector_proxy_state": args.collector_proxy_state,
        "minimum_collector_extra_settings": min(
            int(prefix.collector_audit.get("extra_rotation_count", 0))
            for prefix in prefixes
        ),
        "maximum_collector_extra_settings": max(
            int(prefix.collector_audit.get("extra_rotation_count", 0))
            for prefix in prefixes
        ),
        "maximum_collector_reconstruction_residual_frobenius": max(
            float(
                prefix.collector_audit.get(
                    "matrix_reconstruction_residual_frobenius", 0.0
                )
            )
            for prefix in prefixes
        ),
        "maximum_collector_relative_reconstruction_residual": maximum_collector_residual,
        "collector_equality_tolerance": args.collector_equality_tolerance,
        "collector_extra_angle_steps": args.collector_extra_angle_steps,
        "all_formal_settings_are_depth_limited": all_settings_depth_limited,
        "independent_collector_setting_present": independent_collector_present,
        "production_collector_contract_passed": production_collector_pass,
        "exact_ground_state_used_for_selection": False,
        "exact_ground_state_used_only_for_post_selection_sampling_benchmark": True,
    }
    write_json(case_output / "audit.json", audit)
    return rank_rows, summary_rows, audit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", nargs="+", choices=MOLECULES, default=MOLECULES)
    parser.add_argument("--shots", nargs="+", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--rho", type=float, default=1.0e-6)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--angle-steps", type=int, default=40)
    parser.add_argument("--spectral-max-evaluations", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--collector-mode",
        choices=("augment", "absorb", "exact"),
        default="augment",
        help=(
            "Production uses augment. absorb/exact are explicit diagnostics and "
            "are never selected automatically."
        ),
    )
    parser.add_argument("--collector-min-extra-leaves", type=int, default=1)
    parser.add_argument("--collector-max-extra-leaves", type=int, default=8)
    parser.add_argument(
        "--collector-objective",
        choices=("greedy", "variance", "range", "hybrid"),
        default="greedy",
        help=(
            "Production keeps the collector-tailored greedy exact-fill solution; "
            "the other objectives are diagnostic benchmarks."
        ),
    )
    parser.add_argument(
        "--collector-proxy-state", choices=("hf",), default="hf"
    )
    parser.add_argument("--collector-reference-shots", type=int, default=3000)
    parser.add_argument("--collector-proxy-cycles", type=int, default=6)
    parser.add_argument("--collector-equality-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--collector-extra-angle-steps", type=int, default=400)
    parser.add_argument(
        "--collector-zero-rank-depths",
        nargs="+",
        type=int,
        choices=SHALLOW_RCDF_DEPTHS,
        default=SHALLOW_RCDF_DEPTHS,
        help=(
            "Ascending grid used to select the smallest strictly feasible K=0 "
            "collector depth."
        ),
    )
    parser.add_argument("--weight-source", type=Path, default=DEFAULT_HYBRID_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_rank < 1:
        raise ValueError("--max-rank must be positive")
    if any(value < args.max_rank + 1 for value in args.shots):
        raise ValueError("Every T must cover all nonconstant settings")
    if args.collector_mode == "augment" and (
        args.collector_min_extra_leaves < 1
        or args.collector_max_extra_leaves < args.collector_min_extra_leaves
    ):
        raise ValueError(
            "augment requires 1 <= --collector-min-extra-leaves <= "
            "--collector-max-extra-leaves"
        )
    if (
        args.collector_reference_shots < 1
        or args.collector_proxy_cycles < 1
        or args.collector_equality_tolerance <= 0.0
        or args.collector_extra_angle_steps < 1
    ):
        raise ValueError("Invalid collector optimization controls")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    all_rank_rows = []
    all_summary_rows = []
    audits = []
    for molecule in args.molecules:
        ranks, summaries, audit = run_case(args, molecule, output)
        all_rank_rows.extend(ranks)
        all_summary_rows.extend(summaries)
        audits.append(audit)
    write_csv(output / "rank_selected_all.csv", all_rank_rows)
    write_csv(output / "sampling_summary_all.csv", all_summary_rows)
    manifest = {
        "version": VERSION,
        "adaptive_core_version": CORE_VERSION,
        "runner_sha256": sha256(Path(__file__)),
        "adaptive_core_sha256": sha256(SRDD_METHODS / "adaptive_f3_pool.py"),
        "shallow_rcdf_optimizer_sha256": sha256(SRDD_METHODS / "shallow_rcdf.py"),
        "shallow_collector_core_sha256": sha256(
            SRDD_METHODS / "shallow_collector_redistribution.py"
        ),
        "case_factory_sha256": sha256(SRDD_METHODS / "five_molecule_cases.py"),
        "molecules": list(args.molecules),
        "shots": list(args.shots),
        "maximum_rank": args.max_rank,
        "shallow_rotation_depths": list(SHALLOW_RCDF_DEPTHS),
        "rank_semantics": "independent_joint_rank_K_refit_to_original_target",
        "optimizer_budget": {
            "rho": args.rho,
            "cycles": args.cycles,
            "angle_steps_per_cycle": args.angle_steps,
        },
        "collector_redistribution": {
            "mode": args.collector_mode,
            "minimum_extra_leaves": args.collector_min_extra_leaves,
            "maximum_extra_leaves": args.collector_max_extra_leaves,
            "objective": args.collector_objective,
            "proxy_state": args.collector_proxy_state,
            "reference_shots": args.collector_reference_shots,
            "proxy_cycles": args.collector_proxy_cycles,
            "equality_tolerance": args.collector_equality_tolerance,
            "extra_angle_steps": args.collector_extra_angle_steps,
            "zero_rank_depth_grid": list(args.collector_zero_rank_depths),
            "exact_and_absorb_are_diagnostic_only": True,
        },
        "weight_source": str(args.weight_source.resolve()),
        "all_passed": all(row["status"] == "PASS" for row in audits),
    }
    write_json(output / "manifest.json", manifest)
    if not manifest["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
