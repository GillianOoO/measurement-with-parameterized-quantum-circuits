"""Genuine 50-repeat measurement replay for Figure 3(a,d,c,f).

SRDD noise is sampled as Pauli quantum trajectories. The two spin circuits
act on disjoint registers, allowing exact conditional Born sampling using
1024-by-120 amplitude matrices instead of a 20-qubit density matrix. No MPO
truncation, Gaussian draws, moment fitting, or variance clipping is used.
Common random numbers across noise rates do not couple different repetitions.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ARCHIVE = Path(__file__).resolve().parents[2]
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
import numpy as np
from numba import njit, prange
import scipy.linalg as la
from scipy.optimize import linprog, minimize

DATA = Path(os.environ.get("PAPER_REPRO_DATA_ROOT", ARCHIVE / "data")).resolve()
FROZEN = DATA / "shared_processed" / "figure3" / "empirical50"
OUT = Path(os.environ.get("PAPER_REPRO_FIG3_OUTPUT", ARCHIVE / "build" / "fig3_empirical50")).resolve()
REPEATS = 50
P_GRID = np.linspace(0.0, 0.003, 10)
VERSION = "fig3-exact-pauli-trajectories-empirical50-v1"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256((VERSION + "|" + "|".join(map(str, parts))).encode()).digest()[:8], "little")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def archived_srdd_functions():
    """Reuse the original fitting functions without executing obsolete drivers."""
    core = ARCHIVE / "code" / "methods" / "srdd"
    sys.path.insert(0, str(core))
    import shallow_collector_redistribution as collector_core
    source = ARCHIVE / "code" / "experiments" / "large_molecule_srdd_pauli.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    needed = {"measurement_settings", "ground_srcdf_models"}
    while True:
        expanded = needed | {node.id for name in needed for node in ast.walk(definitions[name])
                             if isinstance(node, ast.Name) and node.id in definitions}
        if expanded == needed:
            break
        needed = expanded
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in needed)
    namespace = dict(np=np, la=la, linprog=linprog, minimize=minimize, math=math,
                     itertools=itertools, hashlib=hashlib, collector_core=collector_core,
                     VERSION="h2o-n2-fullspace-srcdf-vs-pauli-shallow-collector-v5-huang2021")
    namespace.update({name: getattr(collector_core, name) for name in dir(collector_core)
                      if not name.startswith("_") and name not in namespace})
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


def dense_state(molecule):
    path = DATA / molecule / "inputs" / "ground_state_mps_blocked_spin.npz"
    with np.load(path) as payload:
        tensors = [payload[key] for key in sorted(payload.files)]
    value = tensors[0][0]
    for tensor in tensors[1:]:
        value = np.tensordot(value, tensor, axes=(-1, 0))
    state = value[..., 0].reshape(-1)
    state = state * np.exp(-1j*np.angle(state[np.argmax(np.abs(state))]))
    if np.max(np.abs(np.imag(state))) < 1e-14:
        state = np.asarray(state.real, dtype=float)
    return state / np.linalg.norm(state)


def prepare_srdd(molecule, rebuild=False):
    destination = OUT / f"{molecule}_settings.npz"
    if destination.exists():
        return destination
    frozen = FROZEN / destination.name
    if not rebuild and frozen.is_file():
        # These are the published circuits and coefficients. A new noise replay
        # must keep the design fixed, including the near-degenerate N2 angles.
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frozen, destination)
        return destination
    started = time.perf_counter()
    engine = archived_srdd_functions()
    root = DATA / molecule / "results" / "srdd_and_pauli"
    if molecule == "BeH2":
        root /= "BeH2"
    summary = next(row for row in read_csv(root / "sampling_summary.csv")
                   if row["method"] == "s-RCDF" and int(row["T_total_shots"]) == 3000)
    metadata = json.loads((DATA / molecule / "inputs" / "metadata.json").read_text())
    m = int(metadata["number_spatial_orbitals"])
    occupied = int(metadata["number_electrons"]) // 2
    with np.load(DATA / molecule / "inputs" / "integrals_mo.npz") as payload:
        one = payload["one_body_integrals"]
        constant = float(payload["nuclear_repulsion_hartree"])
    one = (one + one.T) / 2
    case = dict(m=m, n=2*m, nalpha=occupied, one=one, constant=constant,
                spec=SimpleNamespace(name=molecule))
    configs = list(itertools.combinations(range(m), occupied))
    occupations = np.zeros((len(configs), m))
    for i, config in enumerate(configs):
        occupations[i, list(config)] = 1
    masks = np.asarray([sum(1 << (m-1-q) for q in config) for config in configs], dtype=np.int64)
    ci = dense_state(molecule).reshape(2**m, 2**m)[np.ix_(masks, masks)]
    assert abs(float(np.vdot(ci, ci).real)-1) < 1e-12
    all_occ = np.asarray(list(itertools.product((0., 1., 2.), repeat=m)))
    k, depth = int(summary["selected_K"]), int(summary["selected_depth"])
    cache = root / "srcdf_fit_cache" / f"K{k:02d}_dR{depth}.npz"
    with np.load(cache) as payload:
        rotations, tensors = payload["rotations"], payload["tensors"]
    controls = json.loads((root / "audit.json").read_text())["collector_configuration"]
    kwargs = {"collector_"+key: controls[key] for key in
              ("mode", "objective", "proxy_state", "reference_shots", "proxy_cycles", "equality_tolerance", "extra_angle_steps")}
    kwargs["collector_extra_leaves"] = int(summary["collector_extra_leaves"])
    settings, alpha, f3, audit = engine.measurement_settings(
        case, dict(rotations=rotations, tensors=tensors, depth=depth), configs,
        occupations, all_occ, ci, int(controls["f3_spectral_maxfev"]), **kwargs)
    from shallow_rcdf import givens_layout, rotation_and_derivatives
    with np.load(DATA / "shared_processed" / "figure3" / "recovered_measurement_circuits.npz") as payload:
        angles = payload[f"{molecule}_setting_angles"]
        shots = payload[f"{molecule}_shot_vector"]
    replay = max(np.linalg.norm(rotation_and_derivatives(a, m, depth)[0]-setting["rotation"])
                 for a, setting in zip(angles, settings))
    if replay > 5e-8:
        # Optimization libraries can select a different near-degenerate extra
        # rotation. Freeze the published angles and replay the SAME greedy
        # exact-fill coefficient rule; never change the measurement circuit.
        if controls["objective"] != "greedy":
            raise RuntimeError("Archived-rotation replay requires the audited greedy rule")
        bank = np.asarray([rotation_and_derivatives(a, m, depth)[0] for a in angles])
        if np.max(np.abs(bank[:k]-rotations)) > 5e-8:
            raise RuntimeError("Archived source rotations differ from the fitted factors")
        directions = tensors.sum(axis=2)-.5*np.diagonal(tensors, axis1=1, axis2=2)
        collector = one+np.einsum("t,tpi,ti,tqi->pq", alpha, rotations, directions, rotations)
        residual = (collector+collector.T)/2
        extra_coefficients = []
        for rotation in bank[k:]:
            diagonal = np.diag(rotation.T@residual@rotation)
            extra_coefficients.append(diagonal)
            residual -= rotation@np.diag(diagonal)@rotation.T
        core = engine.collector_core
        design = core.collector_dictionary(bank)
        target = core.symmetric_vector(collector)
        eta, _ = core.greedy_exact_fill(design, target, k*m,
            np.asarray(extra_coefficients), float(controls["equality_tolerance"]))
        eta = core.equality_project(eta, design, target).reshape(len(bank), m)
        occ = occupations[:, None, :]+occupations[None, :, :]
        for i, setting in enumerate(settings):
            base = engine.diagonal_leaf_values(occ, tensors[i], directions[i], alpha[i]) if i < k else np.zeros(occ.shape[:2])
            setting.update(rotation=bank[i], collector_linear_coefficients=eta[i],
                           values=base+occ@eta[i])
        audit["matrix_relative_reconstruction_residual"] = float(
            np.linalg.norm(core.collector_matrix(bank, eta)-collector)/max(np.linalg.norm(collector),1.))
        replay = max(np.linalg.norm(rotation_and_derivatives(a,m,depth)[0]-setting["rotation"])
                     for a,setting in zip(angles,settings))
    # The spatial occupation score is 1/2 n^T Z n plus its linear correction.
    zbank = np.zeros((len(settings), m, m))
    linear = np.zeros((len(settings), m))
    for i, setting in enumerate(settings):
        linear[i] = setting["collector_linear_coefficients"]
        if i < k:
            zbank[i] = tensors[i]
            linear[i] = setting["collector_linear_coefficients"] - alpha[i]*(tensors[i].sum(axis=1)-.5*np.diag(tensors[i])) - .5*np.diag(tensors[i])
        occ = occupations[:, None, :] + occupations[None, :, :]
        values = .5*np.einsum("...i,ij,...j->...", occ, zbank[i], occ) + occ @ linear[i]
        if np.max(np.abs(values-setting["values"])) > 2e-9:
            raise RuntimeError("Full-domain diagonal score reconstruction failed")
    models = engine.ground_srcdf_models(ci, configs, settings)
    mean = constant + sum(model[2] for model in models)
    variance = sum(model[3]/count for model, count in zip(models, shots))
    exact_energy = float(metadata["fci_ground_energy_hartree"])
    rmse = math.hypot(mean-exact_energy, math.sqrt(variance))
    if abs(rmse-float(summary["analytic_total_RMSE_hartree"])) > 2e-7:
        raise RuntimeError(f"Noiseless RMSE reconstruction failed: {rmse}")
    layout = np.asarray([(m-1-i, m-1-j) for _, _, i, j in reversed(givens_layout(m, depth))], dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, ci=ci, masks=masks, layout=layout,
                        angles=angles[:, ::-1], shots=shots, linear=linear, zbank=zbank,
                        constant=constant, exact_energy=exact_energy, m=m)
    (OUT / f"{molecule}_settings_audit.json").write_text(json.dumps(dict(
        status="PASS", circuit_replay_error=float(replay), analytic_rmse_check=rmse,
        original_analytic_rmse=float(summary["analytic_total_RMSE_hartree"]),
        collector_reconstruction=audit["matrix_relative_reconstruction_residual"],
        preparation_seconds=time.perf_counter()-started, source_fit_sha256=sha(cache)), indent=2))
    print(f"[prepared] {molecule}, {len(settings)} settings, {time.perf_counter()-started:.1f}s", flush=True)
    return destination


@njit(cache=True)
def gate_matrix(a, bit0, bit1, theta):
    c, s = math.cos(theta), math.sin(theta)
    mask0, mask1 = 1 << bit0, 1 << bit1
    for row in range(a.shape[0]):
        if row & mask0 and not row & mask1:
            other = row ^ mask0 ^ mask1
            for column in range(a.shape[1]):
                x, y = a[row, column], a[other, column]
                a[row, column] = c*x+s*y
                a[other, column] = -s*x+c*y


@njit(cache=True)
def pauli_matrix(a, bit0, bit1, code):
    xmask = ((code & 1) << bit0) | (((code >> 1) & 1) << bit1)
    zmask = (((code >> 2) & 1) << bit0) | (((code >> 3) & 1) << bit1)
    out = np.empty_like(a)
    for row in range(a.shape[0]):
        parity = ((row & zmask) >> bit0 & 1) ^ ((row & zmask) >> bit1 & 1)
        sign = 1.-2.*parity
        for col in range(a.shape[1]):
            out[row ^ xmask, col] = sign*a[row, col]
    return out


@njit(cache=True)
def evolved_alpha(initial, layout, angles, pattern):
    a = initial.copy()
    for g in range(len(angles)):
        gate_matrix(a, layout[g, 0], layout[g, 1], angles[g])
        if pattern[g]:
            a = pauli_matrix(a, layout[g, 0], layout[g, 1], pattern[g])
    return a


@njit(cache=True, parallel=True)
def sample_beta_pairs(conditions, patterns, pairs, order, offsets, uniforms, masks, layout, angles, dimension):
    output = np.empty(len(uniforms), dtype=np.int64)
    norm_error = np.zeros(len(pairs))
    for group in prange(len(pairs)):
        conditional, pattern_index = pairs[group]
        a = np.zeros((dimension, 1), dtype=conditions.dtype)
        for k in range(len(masks)):
            a[masks[k], 0] = conditions[conditional, k]
        a = evolved_alpha(a, layout, angles, patterns[pattern_index])
        cumulative = np.empty(dimension)
        total = 0.
        for k in range(dimension):
            total += a[k, 0].real**2 + a[k, 0].imag**2
            cumulative[k] = total
        norm_error[group] = abs(total-1.)
        for j in range(offsets[group], offsets[group+1]):
            index = order[j]
            u = uniforms[index]*total
            lo, hi = 0, dimension-1
            while lo < hi:
                mid = (lo+hi)//2
                if u < cumulative[mid]:
                    hi = mid
                else:
                    lo = mid+1
            output[index] = lo
    return output, np.max(norm_error)


def patterns_for_rates(rng, shots, gate_count):
    draws = rng.random((shots, gate_count))
    paulis = rng.integers(1, 16, size=(shots, gate_count), dtype=np.uint8)
    threshold = (15./16.)*(1.-(1.-P_GRID[1:])**2)
    table = np.where(draws[None, :, :] < threshold[:, None, None], paulis[None, :, :], 0).astype(np.uint8)
    patterns, inverse = np.unique(table.reshape(-1, gate_count), axis=0, return_inverse=True)
    return patterns, inverse


def replay_srdd_setting(molecule, setting):
    destination = OUT / "settings" / f"{molecule}_{setting:03d}.npz"
    if destination.exists():
        return destination
    started = time.perf_counter()
    with np.load(prepare_srdd(molecule)) as f:
        ci, masks, layout = f["ci"], f["masks"], f["layout"]
        angles, count = f["angles"][setting], int(f["shots"][setting])
        linear, z, m = f["linear"][setting], f["zbank"][setting], int(f["m"])
    rng = np.random.default_rng(seed(molecule, setting))
    n = REPEATS*count
    apatterns, ainverse = patterns_for_rates(rng, n, len(angles))
    bpatterns, binverse = patterns_for_rates(rng, n, len(angles))
    # Reusing each shot's uniforms across p couples curves, never repetitions.
    ua = np.tile(rng.random(n), len(P_GRID)-1)
    ub = np.tile(rng.random(n), len(P_GRID)-1)
    initial = np.zeros((2**m, len(masks)), dtype=ci.dtype)
    initial[masks] = ci
    alpha_bits = np.empty(len(ainverse), dtype=np.int64)
    condition_ids = np.empty_like(alpha_bits)
    conditions = []
    order = np.argsort(ainverse, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(ainverse, minlength=len(apatterns)))]
    max_norm_error = 0.
    for pattern_id, pattern in enumerate(apatterns):
        a = evolved_alpha(initial, layout, angles, pattern)
        probabilities = np.sum(np.abs(a)**2, axis=1)
        max_norm_error = max(max_norm_error, abs(float(probabilities.sum())-1))
        indices = order[offsets[pattern_id]:offsets[pattern_id+1]]
        samples = np.searchsorted(np.cumsum(probabilities), ua[indices]*probabilities.sum(), side="right")
        alpha_bits[indices] = samples
        unique, inverse = np.unique(samples, return_inverse=True)
        condition_ids[indices] = len(conditions)+inverse
        conditions.extend(a[unique]/np.sqrt(probabilities[unique, None]))
    pairs, pinverse = np.unique(np.column_stack((condition_ids, binverse)), axis=0, return_inverse=True)
    order = np.argsort(pinverse, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(pinverse, minlength=len(pairs)))]
    beta_bits, beta_norm_error = sample_beta_pairs(np.asarray(conditions), bpatterns, pairs,
        order, offsets, ub, masks, layout, angles, 2**m)
    if max(max_norm_error, beta_norm_error) > 1e-10:
        raise RuntimeError("Trajectory norm was not preserved")
    shifts = np.arange(m-1, -1, -1)
    occ = ((alpha_bits[:, None] >> shifts) & 1) + ((beta_bits[:, None] >> shifts) & 1)
    scores = .5*np.einsum("ni,ij,nj->n", occ, z, occ, optimize=True) + occ @ linear
    scores = scores.reshape(len(P_GRID)-1, REPEATS, count)
    contributions = scores.mean(axis=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, contributions=contributions,
        alpha_outcomes=alpha_bits.reshape(len(P_GRID)-1, REPEATS, count).astype(np.uint16),
        beta_outcomes=beta_bits.reshape(len(P_GRID)-1, REPEATS, count).astype(np.uint16),
        seed=np.uint64(seed(molecule, setting)), count=count,
        maximum_norm_error=max(max_norm_error, beta_norm_error))
    print(f"[SRDD] {molecule} setting {setting+1}, shots {count}, alpha patterns {len(apatterns)}, beta pairs {len(pairs)}, {time.perf_counter()-started:.1f}s", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", choices=("BeH2", "N2"))
    parser.add_argument("--molecule", choices=("BeH2", "N2"))
    parser.add_argument("--setting", type=int)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--rebuild-settings", action="store_true",
                        help="Reconstruct settings from fits instead of using the published frozen settings.")
    args = parser.parse_args()
    globals()["OUT"] = args.output.resolve()
    if args.prepare:
        prepare_srdd(args.prepare, rebuild=args.rebuild_settings)
    elif args.molecule is not None and args.setting is not None:
        replay_srdd_setting(args.molecule, args.setting)
    else:
        parser.error("Choose --prepare or --molecule with --setting")


if __name__ == "__main__":
    main()
