"""Command-line interface for validating and decomposing Hamiltonians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import run_gpd, run_srdd
from .gpd import GPDConfig
from .io import load_hamiltonian
from .srdd import SRDDConfig


def _summary(result, result_path: Path) -> dict[str, object]:
    return {
        "method": result.method,
        "shots": result.shots,
        "selected_size": result.selected_size,
        "settings": len(result.fragments),
        "shot_allocation": result.shot_allocation.tolist(),
        "approximation_spectral_norm": result.error.approximation_spectral_norm,
        "approximation_frobenius_norm": result.error.approximation_frobenius_norm,
        "sampling_variance_bound": result.error.sampling_variance_bound,
        "state_independent_rmse_bound": result.error.state_independent_rmse_bound,
        "calibrated_proxy_rmse": result.error.calibrated_proxy_rmse,
        "state_bias": result.error.state_bias,
        "state_sampling_variance": result.error.state_sampling_variance,
        "state_rmse": result.error.state_rmse,
        "backend": result.metadata.get("backend"),
        "frontier_status": result.metadata.get("frontier"),
        "selection_is_provisional": result.metadata.get("selection_is_provisional", False),
        "unused_shots": result.metadata.get("unused_shots", 0),
        "result": str(result_path),
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--particle-number", type=int)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--calibration-seed", type=int, default=918273)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mpqc-measure",
        description="GPD and SRDD fixed-shot Hamiltonian decompositions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate and summarize an input")
    validate.add_argument("input", type=Path)
    validate.add_argument("--state", type=Path)
    validate.add_argument("--particle-number", type=int)

    gpd = subparsers.add_parser("gpd", help="run Greedy Projection Decomposition")
    _common(gpd)
    gpd.add_argument("--paper-depth", type=int, default=4)
    gpd.add_argument("--max-terms", type=int, default=40)
    gpd.add_argument("--starts", type=int, default=1)
    gpd.add_argument("--steps", type=int, default=600)
    gpd.add_argument("--learning-rate", type=float, default=1.0e-2)
    gpd.add_argument("--seed", type=int, default=20260808)
    precision = gpd.add_mutually_exclusive_group()
    precision.add_argument(
        "--x64",
        action="store_true",
        help="use the double-precision extension instead of the paper float32 fit",
    )
    precision.add_argument(
        "--float32",
        action="store_true",
        help="explicitly select the default paper artifact precision",
    )
    gpd.add_argument("--relative-improvement", type=float, default=2.0e-3)
    gpd.add_argument("--stall-patience", type=int, default=5)
    gpd.add_argument("--minimum-k", type=int, default=10)
    gpd.add_argument("--odd-open-chain", action="store_true")
    gpd.add_argument("--progress", action="store_true")

    srdd = subparsers.add_parser("srdd", help="run SRDD or its universal wrapper")
    _common(srdd)
    srdd.add_argument("--rank-grid", type=int, nargs="+", default=tuple(range(1, 11)))
    srdd.add_argument("--depth-grid", type=int, nargs="+", default=(1, 2, 3))
    srdd.add_argument("--rho", type=float, default=1.0e-6)
    srdd.add_argument("--cycles", type=int, default=4)
    srdd.add_argument("--angle-steps", type=int, default=20)
    srdd.add_argument("--seed", type=int, default=20260826)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        data = load_hamiltonian(
            args.input, state_path=args.state, particle_number=args.particle_number
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "source_format": data.source_format,
                    "n_qubits": data.n_qubits,
                    "dimension": data.dimension,
                    "state_supplied": data.state is not None,
                    "particle_number": data.particle_number,
                    "electronic_integrals": data.electronic is not None,
                },
                indent=2,
            )
        )
        return
    if args.command == "gpd":
        config = GPDConfig(
            paper_depth=args.paper_depth,
            max_terms=args.max_terms,
            starts=args.starts,
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            x64=args.x64,
            calibrate=not args.no_calibrate,
            calibration_seed=args.calibration_seed,
            relative_improvement=args.relative_improvement,
            stall_patience=args.stall_patience,
            minimum_k=args.minimum_k,
            odd_qubit_mode="open-chain" if args.odd_open_chain else "reject",
            progress=args.progress,
        )
        result = run_gpd(
            args.input,
            args.shots,
            state=args.state,
            particle_number=args.particle_number,
            config=config,
            output=args.output,
        )
    else:
        config = SRDDConfig(
            rank_grid=tuple(args.rank_grid),
            depth_grid=tuple(args.depth_grid),
            rho=args.rho,
            cycles=args.cycles,
            angle_steps=args.angle_steps,
            seed=args.seed,
            calibrate=not args.no_calibrate,
            calibration_seed=args.calibration_seed,
        )
        result = run_srdd(
            args.input,
            args.shots,
            state=args.state,
            particle_number=args.particle_number,
            config=config,
            output=args.output,
        )
    print(json.dumps(_summary(result, args.output / "result.json"), indent=2))


if __name__ == "__main__":
    main()
