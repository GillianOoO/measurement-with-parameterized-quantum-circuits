"""Exact variance/bias aggregation for the 50-instance-per-regime Fig. 5 run.

No optimization or sampling is performed. Inputs are immutable saved results.
An environment override makes the plotting entry point portable:
PAPER_REPRO_RANDOM_ENSEMBLE_DIR=<run directory containing results/, cases/, inputs/>.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.environ.get(
    "PAPER_REPRO_RANDOM_ENSEMBLE_DIR",
    ROOT / "data/random_ensemble_50",
)).resolve()
METHODS = ("GPD", "OGM", "SG", "Derand", "LCS", "AP")
REGIMES = ("sparse", "dense")
COUNTS = (12, 45, 160, 572, 2038, 7259, 25848)
INSTANCES = 50

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_paths():
    paths = [RUN / "results" / name for name in
             ("manifest.json", "instance_results.csv", "paired_input_audit.csv")]
    paths.append(RUN / "inputs/common_input_state.npy")
    for regime in REGIMES:
        for i in range(1, INSTANCES + 1):
            case = RUN / "cases" / f"{regime}4_seed0_{i}"
            paths += [case / "gpd_selected/manifest.json", case / "gpd_selected/selected_results.csv",
                      case / "pauli/manifest.json", case / "pauli/instance_results.csv",
                      RUN / "inputs/hamiltonians" / f"{regime}_hamiltonian_4_{i}.npy"]
    return paths

def close(a, b, context):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-13):
        raise ValueError(f"Inconsistent saved quantities: {context}: {a} vs {b}")

def aggregate(output_dir=None):
    manifest = json.loads((RUN / "results/manifest.json").read_text())
    if manifest["status"] != "COMPLETE" or manifest["instances_per_regime"] != INSTANCES:
        raise ValueError("Expected a complete 50-instance-per-regime run")
    for name in ("instance_results.csv", "paired_input_audit.csv"):
        if digest(RUN / "results" / name) != manifest["output_sha256"][name]:
            raise ValueError(f"Source hash mismatch: {name}")
    raw = read_csv(RUN / "results/instance_results.csv")
    expected = {(b, i, m, t) for b in REGIMES for i in range(1, INSTANCES+1)
                for m in METHODS for t in COUNTS}
    indexed = {(r["benchmark"], int(r["instance"]), r["method"], int(r["shots"])): r for r in raw}
    if len(raw) != len(expected) or set(indexed) != expected:
        raise ValueError("Missing or duplicate instance/method/measurement-count records")
    paired = read_csv(RUN / "results/paired_input_audit.csv")
    if len(paired) != 100 or {(r['benchmark'],int(r['instance'])) for r in paired} != {(b,i) for b in REGIMES for i in range(1,51)}:
        raise ValueError("Incomplete paired-input audit")
    if any(r["status"] != "PASS" or r["all_methods_same_input_files"] != "True" for r in paired):
        raise ValueError("Paired-input audit failed")
    state_hash = digest(RUN / "inputs/common_input_state.npy")
    maximum_mse_difference = 0.0
    for b in REGIMES:
        for i in range(1, INSTANCES+1):
            case = RUN / "cases" / f"{b}4_seed0_{i}"
            gm = json.loads((case / "gpd_selected/manifest.json").read_text())
            gs = case / "gpd_selected/selected_results.csv"
            if gm['status'] != 'complete' or digest(gs) != gm['artifacts']['selected_results.csv']:
                raise ValueError(f"GPD source verification failed: {case}")
            if gm['selection']['loss'] != 'hypot(w_a*x_a, w_s*x_s)' or gm['selection']['right_censored']:
                raise ValueError(f"Unverified GPD selection: {case}")
            gpd = {int(r["shots"]): r for r in read_csv(gs)}
            if set(gpd) != set(COUNTS):
                raise ValueError(f"GPD measurement-count grid incomplete: {case}")
            pauli = {(r['method'],int(r['shots'])):r for r in read_csv(case / 'pauli/instance_results.csv')}
            hhash = digest(RUN / "inputs/hamiltonians" / f"{b}_hamiltonian_4_{i}.npy")
            for m in METHODS:
                for t in COUNTS:
                    r = indexed[b,i,m,t]
                    if r['input_file_sha256'] != hhash or r['state_file_sha256'] != state_hash:
                        raise ValueError(f"Hamiltonian/state mismatch: {(b,i,m,t)}")
                    close(r['target_expectation'], gm['case']['target_expectation'], (b,i,m,t,'target'))
                    var, mse = [float(r[k]) for k in ('analytic_variance','analytic_mse')]
                    # The ensemble exporter leaves GPD analytic_bias_squared empty.
                    # Its selected-results table stores the exact squared bias.
                    bias2 = (float(gpd[t]['bias_squared']) if m == 'GPD'
                             else float(r['analytic_bias_squared']))
                    if not all(math.isfinite(v) and v >= 0 for v in (var,bias2,mse)):
                        raise ValueError(f"Invalid moments: {(b,i,m,t)}")
                    close(mse, var+bias2, (b,i,m,t,'MSE=variance+mean squared bias'))
                    maximum_mse_difference = max(maximum_mse_difference, abs(mse-var-bias2))
                    if m == 'GPD':
                        s = gpd[t]
                        if s['selection_uses_benchmark_state'] != 'False':
                            raise ValueError(f"Unexpected state-dependent selection: {case}")
                        for a, k in [('analytic_variance','variance'),('analytic_bias','bias'),('analytic_mse','mse')]:
                            close(r[a],s[k],(b,i,m,t,a))
                        close(s['bias'], float(s['approximate_expectation'])-float(s['target_expectation']), (b,i,t,'expectation bias'))
                        close(bias2,float(s['bias'])**2,(b,i,t,'bias squared'))
                    else:
                        s = pauli[m,t]
                        for k in ('analytic_variance','analytic_bias_squared','analytic_mse'):
                            close(r[k],s[k],(b,i,m,t,k))
                        if r['target_hash'] != gm['case']['target_hash'] or r['state_hash'] != gm['case']['state_hash']:
                            raise ValueError(f"In-memory target/state hash mismatch: {(b,i,m,t)}")
    summary = []
    for b in REGIMES:
        for m in METHODS:
            for t in COUNTS:
                rows = [indexed[b,i,m,t] for i in range(1,INSTANCES+1)]
                values = np.array([float(r['analytic_variance']) for r in rows])
                item = dict(benchmark=b, method=m, shots=t, instances=INSTANCES,
                    mean_instance_state_dependent_variance_hartree2=float(np.mean(values)),
                    instance_variance_sample_sd_hartree2=float(np.std(values,ddof=1)),
                    zero_variance_instances=int(np.count_nonzero(values==0)),
                    averaging_rule='arithmetic mean of 50 instance-level exact variances',
                    selection='L=4 stabilizer-calibrated GPD prefix' if m=='GPD' else 'same deployed Pauli protocol as main Fig. 5')
                if m=='GPD':
                    biases=np.array([float(r['analytic_bias']) for r in rows])
                    item.update(mean_instance_absolute_approximation_bias=float(np.mean(np.abs(biases))),
                                mean_instance_squared_approximation_bias=float(np.mean(biases**2)),
                                bias_averaging_rule='arithmetic mean of 50 absolute expectation differences')
                summary.append(item)
    validation = dict(status='PASS',source=str(RUN),instances_per_regime=50,
        instance_ids={b:list(range(1,51)) for b in REGIMES}, methods=list(METHODS),
        measurement_counts=list(COUNTS), input_rows=len(raw), summary_rows=len(summary),
        every_statistic_has_exactly_50_instances=True,
        paired_input_files_verified=True, selected_gpd_moments_verified=True,
        pauli_moments_verified=True, max_mse_identity_difference=maximum_mse_difference,
        zero_variance_GPD_instances={b:{str(t):sum(float(indexed[b,i,'GPD',t]['analytic_variance'])==0 for i in range(1,51)) for t in COUNTS} for b in REGIMES},
        display_counts=list(COUNTS[1:]),
        display_policy='Both random panels retain the common T>=45 display range; all 50 instances, including zero-variance instances, enter every mean. T=12 remains in the raw and summary tables.',
        protocols={'GPD':'exact variance conditional on selected integer allocation',
                   'OGM/SG/Derand':'exact variance conditional on frozen schedule',
                   'LCS':'exact iid variance divided by T',
                   'AP':'mean exact conditional variance over the 50 preregistered schedules for each Hamiltonian'},
        source_sha256={str(p):digest(p) for p in source_paths()})
    if output_dir is not None:
        output=Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output/'random_ensemble50_variance_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    return summary
