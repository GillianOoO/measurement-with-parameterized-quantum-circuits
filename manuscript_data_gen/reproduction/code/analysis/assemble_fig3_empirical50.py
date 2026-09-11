"""Assemble and independently re-aggregate all Figure 3 empirical error points."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import math
import numpy as np
import replay_fig3_empirical50 as base
import replay_fig3_fc_empirical50 as fc

def main(input_dir=None,output_dir=None):
    out=Path(input_dir or base.FROZEN).resolve()
    destination=Path(output_dir or base.ARCHIVE/"build"/"fig3_assembly").resolve()
    if destination==base.FROZEN:
        raise ValueError("Choose an output directory other than the published source data")
    destination.mkdir(parents=True,exist_ok=True)
    jobs=json.loads((out/"replay_jobs.json").read_text())
    assert jobs["status"]=="PASS" and len(jobs["jobs"])==209
    for name in ("sampler_validation.json","fc_sampler_validation.json"):
        assert json.loads((out/name).read_text())["status"]=="PASS"
    inputs=set()
    curves=[];replicates=[];checks=[]
    def read(path):
        inputs.add(path)
        return base.read_csv(path)
    def add(molecule,method,curve,x,estimates,energy,total,source):
        estimates=np.asarray(estimates,float)
        assert estimates.shape==(50,) and np.all(np.isfinite(estimates))
        errors=estimates-energy
        rmse=float(np.sqrt(np.mean(errors**2)))
        curves.append(dict(molecule=molecule,method=method,curve=curve,x=float(x),y=rmse,
            repeat_count=50,T_total_shots=total,empirical_bias_hartree=float(errors.mean()),
            empirical_variance_hartree2=float(np.mean((estimates-estimates.mean())**2)),
            source=source))
        for r,(estimate,error) in enumerate(zip(estimates,errors)):
            replicates.append(dict(molecule=molecule,method=method,curve=curve,x=float(x),
                T_total_shots=total,repeat=r,estimate_hartree=float(estimate),
                exact_energy_hartree=energy,signed_error_hartree=float(error)))
        assert abs(rmse**2-float(errors.mean())**2-float(np.mean((estimates-estimates.mean())**2)))<1e-11
        return rmse
    for molecule,k,groups in [("BeH2",21,36),("N2",41,111)]:
        metadata=base.DATA/molecule/"inputs"/"metadata.json"
        inputs.add(metadata)
        energy=float(json.loads(metadata.read_text())["fci_ground_energy_hartree"])
        source=base.DATA/molecule/"results"/"srdd_and_pauli"
        if molecule=="BeH2":source/="BeH2"
        existing=read(source/"sampling_replicates.csv")
        by_key=defaultdict(list)
        for row in existing:
            method="SRDD" if row["method"]=="s-RCDF" else row["method"]
            by_key[(method,int(row["T_total_shots"]))].append((int(row["repeat"]),float(row["energy_estimate_hartree"])))
        fcroot=base.DATA/"fully_commuting"/"BeH2_N2_FC_IMA"/"error_eval"/"results"/molecule
        for row in read(fcroot/"born_replicates.csv"):
            by_key[("FC-IMA",int(row["T_total_shots"]))].append((int(row["repeat_index"]),float(row["estimated_energy_hartree"])))
        for (method,total),values in by_key.items():
            values=sorted(values)
            assert [r for r,_ in values]==list(range(50))
            add(molecule,method,"error_vs_measurements",total,[v for _,v in values],energy,total,"archived joint Born repetitions")
        for method in ("SRDD","FC-IMA","OGM","SG","Derand"):
            estimates=np.array([v for _,v in sorted(by_key[(method,3000)])])
            add(molecule,method,"error_vs_depolarizing_rate",0.,estimates,energy,3000,"same 50 archived zero-noise repetitions as the budget panel")
            if method in ("OGM","SG","Derand"):
                for p in base.P_GRID[1:]:
                    add(molecule,method,"error_vs_depolarizing_rate",p,estimates,energy,3000,"same 50 repetitions; no two-qubit measurement gates")
        inputs.add(out/f"{molecule}_settings.npz")
        with np.load(out/f"{molecule}_settings.npz") as settings:
            totals=np.full((9,50),float(settings["constant"]))
            ci,masks,layout=settings["ci"],settings["masks"],settings["layout"]
            m=int(settings["m"])
            shifts=np.arange(m-1,-1,-1)
            occupations=(masks[:,None]>>shifts)&1
            sector_occ=occupations[:,None,:]+occupations[None,:,:]
            old=next(row for row in read(base.DATA/"shared_processed"/"figure3"/"srcdf_local_depolarizing_moments.csv")
                     if row["molecule"]==molecule and float(row["p"])==0.)
            oldmeans=np.array(json.loads(old["setting_means_json"]))
            oldseconds=np.array(json.loads(old["setting_second_moments_json"]))
            mean_errors=[];second_errors=[]
            for index in range(k):
                path=out/"settings"/f"{molecule}_{index:03d}.npz"
                inputs.add(path)
                with np.load(path) as saved:
                    a,b=saved["alpha_outcomes"],saved["beta_outcomes"]
                    assert a.shape==b.shape==(9,50,int(settings["shots"][index]))
                    occupation=((a.reshape(-1,1)>>shifts)&1)+((b.reshape(-1,1)>>shifts)&1)
                    scores=.5*np.einsum("ni,ij,nj->n",occupation,settings["zbank"][index],occupation,optimize=True)+occupation@settings["linear"][index]
                    expected=scores.reshape(a.shape).mean(axis=2)
                    assert np.max(np.abs(expected-saved["contributions"]))<1e-11
                    totals+=expected
                initial=np.zeros((2**m,len(masks)),dtype=ci.dtype);initial[masks]=ci
                angles=settings["angles"][index]
                alpha=base.evolved_alpha(initial,layout,angles,np.zeros(len(angles),np.uint8))[masks]
                initial[masks]=alpha.T
                both=base.evolved_alpha(initial,layout,angles,np.zeros(len(angles),np.uint8))[masks].T
                probabilities=np.abs(both)**2
                values=.5*np.einsum("...i,ij,...j->...",sector_occ,settings["zbank"][index],sector_occ)+sector_occ@settings["linear"][index]
                if index==0:values=values+float(settings["constant"])
                mean_errors.append(abs(float(np.sum(probabilities*values))-oldmeans[index]))
                second_errors.append(abs(float(np.sum(probabilities*values**2))-oldseconds[index]))
            assert max(mean_errors)<2e-7 and max(second_errors)<2e-6,(molecule,max(mean_errors),max(second_errors))
            checks.append(dict(molecule=molecule,SRDD_noiseless_setting_mean_max_error=max(mean_errors),SRDD_noiseless_setting_second_moment_max_error=max(second_errors)))
            assert int(settings["shots"].sum())==3000
        for pi,p in enumerate(base.P_GRID[1:]):
            add(molecule,"SRDD","error_vs_depolarizing_rate",p,totals[pi],energy,3000,"direct Pauli trajectories and joint Born outcomes")
        reference=json.loads((fcroot/"summary.json").read_text())
        # The identity scalar is the sum of identity Pauli coefficients.
        paulis=read(base.DATA/molecule/"inputs"/"hamiltonian_pauli_blocked_spin.csv")
        identity=sum(float(row["coefficient_real_hartree"]) for row in paulis if set(row["full_label_site0_to_siteNminus1"])=={"I"})
        totals=np.full((9,50),identity)
        audit=read(base.DATA/"fully_commuting"/"BeH2_N2_FC_IMA"/"noise_eval"/"results"/molecule/"circuit_audit.csv")
        audit={int(row["group_index"]):row for row in audit}
        count_sum=0
        for index in range(groups):
            path=out/"fc_groups"/f"{molecule}_{index:03d}.npz";inputs.add(path)
            with np.load(path) as saved:
                outcomes=saved["outcomes"]
                assert outcomes.shape==(9,50,int(saved["count"]))
                assert int(saved["cnot_count"])==int(audit[index]["compiled_cx_count"])
                values=fc.scores(outcomes.reshape(-1),saved["output_z_masks"],saved["pooled_score_coefficients"])
                expected=values.reshape(outcomes.shape).sum(axis=2)
                assert np.max(np.abs(expected-saved["contributions"]))<1e-11
                totals+=expected;count_sum+=int(saved["count"])
        assert count_sum==3000
        for pi,p in enumerate(base.P_GRID[1:]):
            add(molecule,"FC-IMA","error_vs_depolarizing_rate",p,totals[pi],energy,3000,"joint Born outcomes with exact propagated per-CNOT Pauli faults")
    assert len(curves)==150 and len(replicates)==7500
    curve_path=destination/"fig3_empirical_curves.csv"
    replicate_path=destination/"fig3_empirical_replicates.csv"
    base.write_csv(curve_path,curves);base.write_csv(replicate_path,replicates)
    reference_curves=read(base.FROZEN/"fig3_empirical_curves.csv")
    key=lambda row:(row["molecule"],row["method"],row["curve"],float(row["x"]))
    expected={key(row):row for row in reference_curves}
    assert set(expected)=={key(row) for row in curves}
    largest=max(abs(float(row["y"])-float(expected[key(row)]["y"])) for row in curves)
    reference_comparison=dict(points=len(curves),maximum_absolute_RMSE_difference=largest,
                              tolerance=1e-11,matches_within_tolerance=largest<=1e-11)
    scripts=[Path(__file__).with_name(name) for name in ("replay_fig3_empirical50.py","replay_fig3_fc_empirical50.py","run_fig3_empirical50.py","test_fig3_empirical50.py",Path(__file__).name)]
    inputs.update(scripts)
    manifest=dict(status="PASS",definition="sqrt(mean((estimate-Tr(rho0 H))**2))",
        repeats_per_point=50,curve_points=150,complete_estimates=7500,
        published_reference_comparison=reference_comparison,
        noise_repetitions="Independent within each point; common random numbers across p. The p=0 and zero-CNOT rows reuse archived empirical repetitions.",
        checks=checks,inputs={str(p.relative_to(base.ARCHIVE)) if p.is_relative_to(base.ARCHIVE)
                            else "data/"+p.relative_to(base.DATA).as_posix():base.sha(p) for p in sorted(inputs)},
        outputs={p.name:base.sha(p) for p in [curve_path,replicate_path]})
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2))
    for row in curves:
        if row["curve"]=="error_vs_depolarizing_rate" and row["method"] in ("SRDD","FC-IMA"):
            print(row["molecule"],row["method"],f"p={row['x']:.6g}",f"RMSE={row['y']:.9g}",flush=True)
    print(json.dumps(dict(status="PASS",points=len(curves),repetitions=len(replicates),checks=checks)),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=base.FROZEN,
                        help="Directory with saved or freshly replayed setting/group outcomes.")
    parser.add_argument("--output",type=Path,default=base.ARCHIVE/"build"/"fig3_assembly")
    args=parser.parse_args()
    main(args.input,args.output)
