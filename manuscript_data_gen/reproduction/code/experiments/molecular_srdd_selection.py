#!/usr/bin/env python3
"""Fit s-RCDF-specific stabilizer weights, then reselect and resample ranks.

The audited independent joint rank-K shallow-RCDF factorizations are unchanged:
their construction does not depend on stabilizer labels.  Their historical
independently diagonalized one-body collector settings are *not* reused.  This
runner reconstructs each frozen Fragment bank and exactly redistributes the
collector over a minimum-size, depth-matched shallow augmentation bank before
calibration, selection, and sampling are repeated.  K=0 remains audit-only for
fitting.  The exact molecular ground state is used only after the weights are
frozen and K*(T) has been selected; it never enters calibration or rank
selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
SRDD_METHODS = HERE.parent / "methods" / "srdd"
MOLECULES = ("H4", "H6")
SHOTS = (100, 200, 300, 500, 800, 1200, 2000, 3000)
VERSION = "standalone-shallow-rcdf-v5-shallow-collector-calibration-resample"


import sys

if str(SRDD_METHODS) not in sys.path:
    sys.path.insert(0, str(SRDD_METHODS))

from srdd_release_paths import DATA, RUNS, molecular_result
SOURCE = DATA
PROBE_SOURCE = DATA
OUTPUT = RUNS / "molecular_srdd_selected"

from five_molecule_cases import load_five_molecule_case  # noqa: E402
from adaptive_f3_pool import Fragment, LABELS, SHALLOW_RCDF_DEPTHS  # noqa: E402
from stabilizer_calibration import (  # noqa: E402
    CALIBRATION_K_VALUES,
    build_minimum_depth_zero_rank_shallow_collector,
    build_shallow_srcdf_prefix_decomposition,
    calibrate_weights,
    ground_state,
    integer_range_allocation,
    monte_carlo_estimates,
    outcome_models,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sampling_seed(molecule: str, total_shots: int, k: int) -> int:
    payload = f"{VERSION}|{molecule}|T{total_shots}|K{k}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def frozen_fragments(data, k: int) -> list[Fragment]:
    """Reconstruct the native rank-K bank; ignore its historical settings."""
    fragments = []
    for index in range(k):
        fragments.append(
            Fragment(
                family="shallow_rcdf",
                family_label=LABELS["shallow_rcdf"],
                matrix=np.asarray(data["source_matrices"][index]),
                diagonal=np.asarray(data["source_diagonals"][index]),
                one_matrix=np.asarray(data["source_one_matrices"][index]),
                one_diagonal=np.asarray(data["source_one_diagonals"][index]),
                remainder_diagonal=np.asarray(
                    data["source_diagonals"][index]
                    - data["source_one_diagonals"][index]
                ),
                q_direction=np.asarray(data["source_q_directions"][index]),
                q_diagonal=np.asarray(data["source_q_diagonals"][index]),
                f3_interface="standard_shallow_tensor_native_f3_r2",
                source_term=index,
                candidate_metadata={
                    "spatial_rotation": np.asarray(
                        data["spatial_rotations"][index]
                    ).tolist(),
                    "z_tensor": np.asarray(data["z_tensors"][index]).tolist(),
                    "shallow_rcdf_depth": int(data["selected_depth"]),
                },
                rcdf_bank_index=index,
            )
        )
    return fragments


def collector_kwargs(args) -> dict[str, Any]:
    return {
        "collector_mode": args.collector_mode,
        "collector_min_extra_leaves": args.collector_min_extra_leaves,
        "collector_max_extra_leaves": args.collector_max_extra_leaves,
        "collector_objective": args.collector_objective,
        "collector_proxy_state": "hf",
        "collector_reference_shots": args.collector_reference_shots,
        "collector_proxy_cycles": args.collector_proxy_cycles,
        "collector_equality_tolerance": args.collector_equality_tolerance,
        "collector_extra_angle_steps": args.collector_extra_angle_steps,
    }


def save_shallow_rank(path: Path, prefix, source_path: Path, depth: int) -> None:
    np.savez_compressed(
        path,
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
        frozen_rank_source=np.asarray(str(source_path)),
        frozen_rank_source_sha256=np.asarray(sha256(source_path)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--probe-source", type=Path, default=PROBE_SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--molecules", nargs="+", choices=MOLECULES, default=MOLECULES
    )
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--collector-mode",
        choices=("augment", "absorb", "exact"),
        default="augment",
    )
    parser.add_argument("--collector-min-extra-leaves", type=int, default=1)
    parser.add_argument("--collector-max-extra-leaves", type=int, default=8)
    parser.add_argument(
        "--collector-objective",
        choices=("greedy", "variance", "range", "hybrid"),
        default="greedy",
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
    )
    args = parser.parse_args()
    if args.collector_mode == "augment" and (
        args.collector_min_extra_leaves < 1
        or args.collector_max_extra_leaves < args.collector_min_extra_leaves
    ):
        parser.error(
            "augment requires 1 <= collector-min-extra-leaves <= "
            "collector-max-extra-leaves"
        )
    if (
        args.collector_reference_shots < 1
        or args.collector_proxy_cycles < 1
        or args.collector_equality_tolerance <= 0.0
        or args.collector_extra_angle_steps < 1
    ):
        parser.error("invalid collector controls")
    source = args.source.resolve()
    probe_root = args.probe_source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a fresh output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    all_ranks: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    source_hashes: dict[str, Any] = {}
    for molecule in args.molecules:
        rank_source = molecular_result(source, molecule, "ranks")
        case = load_five_molecule_case(molecule)
        case_output = output / molecule
        case_output.mkdir(parents=True, exist_ok=True)
        source_rank_rows = read_csv(rank_source / "rank_selected.csv")
        if [int(row["K_source_terms"]) for row in source_rank_rows] != list(range(11)):
            raise RuntimeError(f"{molecule}: source ranks are not K=0,...,10")
        rank_data = []
        rank_rows = []
        base_prefix, base_depth = build_minimum_depth_zero_rank_shallow_collector(
            case,
            args.collector_zero_rank_depths,
            8,
            **collector_kwargs(args),
        )
        base_settings = np.asarray(
            [setting.matrix for setting in base_prefix.settings]
        )
        base_ranges = np.asarray(
            [setting.centered_range for setting in base_prefix.settings]
        )
        base_row = {
            **source_rank_rows[0],
            "shallow_rotation_depth": base_depth,
            "F3_reconstruction_error": base_prefix.reconstruction_error,
            "centered_spectral_norm_sum": base_prefix.spectral_sum,
            "centered_spectral_norm_sum_before_R2": (
                base_prefix.spectral_sum_before_r2
            ),
            "measurement_settings": len(base_prefix.settings),
            "collector_mode": base_prefix.collector_audit["mode"],
            "collector_extra_settings": base_prefix.collector_audit[
                "extra_rotation_count"
            ],
            "collector_dictionary_rank": base_prefix.collector_audit[
                "dictionary_rank"
            ],
            "collector_reconstruction_residual": base_prefix.collector_audit[
                "matrix_reconstruction_residual_frobenius"
            ],
            "independent_collector_setting_present": base_prefix.collector_audit[
                "independent_collector_setting_present"
            ],
        }
        rank_rows.append(base_row)
        rank_data.append(
            {
                "K": 0,
                "prefix": base_prefix,
                "approximation": np.sum(base_settings, axis=0),
                "settings": base_settings,
                "ranges": base_ranges,
                "source_matrices": np.empty((0, *case.target.shape)),
            }
        )
        if args.collector_mode == "augment" and (
            base_prefix.collector_audit["independent_collector_setting_present"]
            or base_prefix.collector_audit["extra_rotation_count"]
            < args.collector_min_extra_leaves
            or base_prefix.collector_audit[
                "matrix_relative_reconstruction_residual"
            ]
            > 5.0 * args.collector_equality_tolerance
            or not base_prefix.collector_audit[
                "all_measurement_settings_depth_limited"
            ]
        ):
            raise RuntimeError(f"{molecule} K=0: production collector contract failed")
        save_shallow_rank(
            case_output / "selected_rank_K00.npz",
            base_prefix,
            rank_source / "rank_selected.csv",
            base_depth,
        )
        case_source_hashes = {
            "rank_selected.csv": sha256(rank_source / "rank_selected.csv")
        }
        for k in range(1, 11):
            path = rank_source / f"selected_rank_K{k:02d}.npz"
            with np.load(path, allow_pickle=False) as data:
                target = np.asarray(data["target"], dtype=np.complex128)
                source_matrices = np.asarray(data["source_matrices"], dtype=np.complex128)
                depth = int(data["selected_depth"])
                fragments = frozen_fragments(data, k)
            if float(np.linalg.norm(target - case.target, "fro")) > 1.0e-10:
                raise RuntimeError(f"{molecule} K={k}: frozen rank target changed")
            prefix = build_shallow_srcdf_prefix_decomposition(
                case,
                fragments,
                depth,
                8,
                **collector_kwargs(args),
            )
            settings = np.asarray(
                [setting.matrix for setting in prefix.settings], dtype=np.complex128
            )
            ranges = np.asarray(
                [setting.centered_range for setting in prefix.settings], dtype=float
            )
            reconstruction = float(prefix.reconstruction_error)
            if reconstruction > 1.0e-6:
                raise RuntimeError(
                    f"{molecule} K={k}: shallow collector F3 replay failed"
                )
            audit = prefix.collector_audit
            if args.collector_mode == "augment" and (
                audit["independent_collector_setting_present"]
                or audit["extra_rotation_count"] < args.collector_min_extra_leaves
                or audit["matrix_relative_reconstruction_residual"]
                > 5.0 * args.collector_equality_tolerance
                or not audit["all_measurement_settings_depth_limited"]
            ):
                raise RuntimeError(
                    f"{molecule} K={k}: production collector contract failed"
                )
            rank_row = {
                **source_rank_rows[k],
                "shallow_rotation_depth": depth,
                "F3_reconstruction_error": reconstruction,
                "centered_spectral_norm_sum": prefix.spectral_sum,
                "centered_spectral_norm_sum_before_R2": (
                    prefix.spectral_sum_before_r2
                ),
                "measurement_settings": len(prefix.settings),
                "collector_mode": audit["mode"],
                "collector_extra_settings": audit["extra_rotation_count"],
                "collector_dictionary_rank": audit["dictionary_rank"],
                "collector_reconstruction_residual": audit[
                    "matrix_reconstruction_residual_frobenius"
                ],
                "independent_collector_setting_present": audit[
                    "independent_collector_setting_present"
                ],
            }
            rank_rows.append(rank_row)
            rank_data.append(
                {
                    "K": k,
                    "prefix": prefix,
                    "approximation": np.sum(settings, axis=0),
                    "settings": settings,
                    "ranges": ranges,
                    "source_matrices": source_matrices,
                }
            )
            save_shallow_rank(
                case_output / f"selected_rank_K{k:02d}.npz",
                prefix,
                path,
                depth,
            )
            case_source_hashes[f"selected_rank_K{k:02d}.npz"] = sha256(path)

        probe_case = molecular_result(probe_root, molecule, "selected")
        with np.load(probe_case / "stabilizer_probes.npz") as data:
            probes = np.asarray(data["states"], dtype=np.complex128)
        probe_records = read_csv(probe_case / "stabilizer_probe_manifest.csv")
        np.savez_compressed(case_output / "stabilizer_probes.npz", states=probes)
        write_csv(case_output / "stabilizer_probe_manifest.csv", probe_records)
        prefixes = [row["prefix"] for row in rank_data]
        approximations = [row["approximation"] for row in rank_data]
        weights, approximation_rows, sampling_rows, allocation_rows = calibrate_weights(
            case,
            [],
            prefixes,
            probes,
            probe_records,
            SHOTS,
            CALIBRATION_K_VALUES,
            independent_approximations=approximations,
        )
        weights.update(
            {
                "calibration_source_family": "standalone_shallow_rcdf",
                "calibration_source_semantics": (
                    "independent complete joint rank-K shallow-RCDF fits; "
                    "selected d_R per K; collector-tailored greedy exact-fill "
                    "settings reconstructed from frozen native leaves"
                ),
                "calibration_rank_source": str(rank_source),
                "probe_source": str(probe_case),
                "probe_manifest_sha256": sha256(
                    probe_case / "stabilizer_probe_manifest.csv"
                ),
                "probe_vectors_sha256": sha256(probe_case / "stabilizer_probes.npz"),
            }
        )
        write_json(case_output / "stabilizer_weights.json", weights)
        write_csv(
            case_output / "stabilizer_approximation_components.csv",
            approximation_rows,
        )
        write_csv(
            case_output / "stabilizer_sampling_components.csv", sampling_rows
        )
        write_csv(
            case_output / "stabilizer_calibration_shot_allocations.csv",
            allocation_rows,
        )
        weight_a = float(weights["weight_approximation"])
        weight_s = float(weights["weight_sampling"])

        # The exact ground state is constructed only after calibration.  It is
        # not read by the candidate-loss loop and is used only after K*(T) is
        # selected for the corresponding sampling benchmark.
        state, exact_energy = ground_state(case)

        candidate_rows = []
        summary_rows = []
        replicate_rows = []
        max_fitted_rank = max(int(data["K"]) for data in rank_data)
        for total_shots in SHOTS:
            candidates = []
            for source_row, data in zip(rank_rows, rank_data):
                shots = integer_range_allocation(total_shots, data["ranges"])
                active = shots > 0
                x_sampling = math.sqrt(
                    max(
                        float(
                            np.sum(data["ranges"][active] ** 2 / shots[active])
                        ),
                        0.0,
                    )
                )
                x_approximation = float(source_row["sector_Frobenius_distance"])
                loss = math.hypot(
                    weight_a * x_approximation, weight_s * x_sampling
                )
                record = {
                    "molecule": molecule,
                    "T_total_shots": total_shots,
                    "K_source_terms": int(source_row["K_source_terms"]),
                    "shallow_rotation_depth": int(
                        source_row["shallow_rotation_depth"]
                    ),
                    "sector_Frobenius_distance": x_approximation,
                    "full_Frobenius_distance": float(
                        source_row["full_Frobenius_distance"]
                    ),
                    "x_sampling": x_sampling,
                    "stabilizer_loss": loss,
                    "total_settings": len(data["settings"]),
                    "collector_mode": data["prefix"].collector_audit["mode"],
                    "collector_extra_settings": data["prefix"].collector_audit[
                        "extra_rotation_count"
                    ],
                    "collector_dictionary_rank": data["prefix"].collector_audit[
                        "dictionary_rank"
                    ],
                    "collector_reconstruction_residual": data[
                        "prefix"
                    ].collector_audit[
                        "matrix_reconstruction_residual_frobenius"
                    ],
                    "independent_collector_setting_present": data[
                        "prefix"
                    ].collector_audit["independent_collector_setting_present"],
                    "shot_vector": " ".join(str(int(value)) for value in shots),
                }
                candidate_rows.append(record)
                candidates.append((loss, int(data["K"]), record, data, shots))
            _, _, record, data, shots = min(candidates, key=lambda value: value[:2])
            if int(record["K_source_terms"]) == max_fitted_rank:
                raise RuntimeError(
                    f"{molecule} T={total_shots}: the selected s-RCDF rank lies "
                    "at the fitted endpoint; extend the rank grid and rerun "
                    "before publishing."
                )
            models = outcome_models(state, data["prefix"])
            approximate_energy = float(sum(model[2] for model in models))
            sampling_variance = float(
                sum(model[3] / count for model, count in zip(models, shots) if count)
            )
            seed = sampling_seed(
                molecule, total_shots, int(record["K_source_terms"])
            )
            estimates = monte_carlo_estimates(models, shots, args.repeats, seed)
            errors = estimates - exact_energy
            summary = {
                **record,
                "right_censored_at_rank_10": int(record["K_source_terms"]) == 10,
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
            }
            summary_rows.append(summary)
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
        write_csv(case_output / "rank_selected.csv", rank_rows)
        write_csv(case_output / "candidate_by_T_K.csv", candidate_rows)
        write_csv(case_output / "sampling_summary.csv", summary_rows)
        write_csv(case_output / "sampling_replicates.csv", replicate_rows)
        maximum_collector_relative_residual = max(
            float(
                data["prefix"].collector_audit.get(
                    "matrix_relative_reconstruction_residual", 0.0
                )
            )
            for data in rank_data
        )
        maximum_many_body_reconstruction = max(
            float(data["prefix"].reconstruction_error) for data in rank_data
        )
        all_depth_limited = all(
            setting.rotation_depth is not None
            for data in rank_data
            for setting in data["prefix"].settings
        )
        independent_collector_present = any(
            bool(
                data["prefix"].collector_audit.get(
                    "independent_collector_setting_present", False
                )
            )
            for data in rank_data
        )
        write_json(
            case_output / "audit.json",
            {
                "status": "PASS",
                "molecule": molecule,
                "rank_fit_reused_without_refitting": True,
                "historical_exact_collector_settings_reused": False,
                "frozen_native_fragments_reconstructed": True,
                "rank_source": str(rank_source),
                "weight_calibration_family": "standalone_shallow_rcdf",
                "gfro_weights_used_for_srcdf_selection": False,
                "srcdf_specific_weights_and_all_eight_T_points_recomputed": True,
                "calibration_fit_K": list(CALIBRATION_K_VALUES),
                "calibration_audit_only_K": [0],
                "calibration_T": list(SHOTS),
                "probe_count": len(probe_records),
                "probe_split_counts": weights["state_group_split"],
                "sampling_ratio_bound_audit": weights[
                    "sampling_ratio_bound_audit"
                ],
                "exact_ground_state_used_only_after_selection": True,
                "collector_mode": args.collector_mode,
                "collector_objective": args.collector_objective,
                "collector_proxy_state": "hf",
                "collector_extra_angle_steps": args.collector_extra_angle_steps,
                "minimum_extra_settings_over_rank_grid": min(
                    int(data["prefix"].collector_audit["extra_rotation_count"])
                    for data in rank_data
                ),
                "maximum_extra_settings_over_rank_grid": max(
                    int(data["prefix"].collector_audit["extra_rotation_count"])
                    for data in rank_data
                ),
                "maximum_collector_relative_reconstruction_residual": (
                    maximum_collector_relative_residual
                ),
                "maximum_many_body_reconstruction_residual_frobenius": (
                    maximum_many_body_reconstruction
                ),
                "all_measurement_settings_depth_limited": all_depth_limited,
                "independent_collector_setting_present": (
                    independent_collector_present
                ),
                "rank_source_hashes": case_source_hashes,
            },
        )
        source_hashes[molecule] = case_source_hashes
        all_ranks.extend(rank_rows)
        all_summary.extend(summary_rows)
    write_csv(output / "rank_selected_all.csv", all_ranks)
    write_csv(output / "sampling_summary_all.csv", all_summary)
    manifest = {
        "version": VERSION,
        "molecules": list(args.molecules),
        "shots": list(SHOTS),
        "maximum_rank": 10,
        "rank_semantics": (
            "frozen audited independent_joint_rank_K_refits; no numerical refit; "
            "historical exact collector settings discarded and reconstructed"
        ),
        "rank_source": str(source),
        "rank_source_manifest_sha256": (
            sha256(source / "manifest.json") if (source / "manifest.json").is_file() else None
        ),
        "weight_fit_source_family": "standalone_shallow_rcdf",
        "gfro_weight_transfer": False,
        "probe_source": str(probe_root),
        "probe_source_manifest_sha256": (
            sha256(probe_root / "manifest.json") if (probe_root / "manifest.json").is_file() else None
        ),
        "calibration_K_fit": list(CALIBRATION_K_VALUES),
        "calibration_K_audit_only": [0],
        "calibration_shot_grid": list(SHOTS),
        "collector_redistribution": {
            "mode": args.collector_mode,
            "minimum_extra_leaves": args.collector_min_extra_leaves,
            "maximum_extra_leaves": args.collector_max_extra_leaves,
            "formal_objective": args.collector_objective,
            "proxy_state": "hf",
            "reference_shots": args.collector_reference_shots,
            "proxy_cycles": args.collector_proxy_cycles,
            "equality_tolerance": args.collector_equality_tolerance,
            "extra_angle_steps": args.collector_extra_angle_steps,
            "zero_rank_depth_grid": list(args.collector_zero_rank_depths),
            "historical_exact_collector_reused": False,
            "exact_and_absorb_are_diagnostic_only": True,
        },
        "new_selection_and_sampling_repeats": args.repeats,
        "rank_source_hashes": source_hashes,
        "runner_sha256": sha256(Path(__file__)),
        "shallow_collector_core_sha256": sha256(
            SRDD_METHODS / "shallow_collector_redistribution.py"
        ),
        "all_passed": True,
    }
    write_json(output / "manifest.json", manifest)
    print(f"PASS: reweighted/resampled standalone s-RCDF -> {output}")


if __name__ == "__main__":
    main()
