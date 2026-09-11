"""Run all independent Figure 3 noise-replay settings with resumable outputs."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import replay_fig3_empirical50 as base

def run(job):
    method,molecule,index=job
    cache=base.OUT/("settings" if method=="SRDD" else "fc_groups")/f"{molecule}_{index:03d}.npz"
    if cache.exists():
        return dict(job=job,status="PASS",generated_this_run=False,detail="saved replay")
    script=Path(__file__).with_name("replay_fig3_empirical50.py" if method=="SRDD" else "replay_fig3_fc_empirical50.py")
    flag="--setting" if method=="SRDD" else "--group"
    log=base.OUT/"logs"/f"{method}_{molecule}_{index:03d}.log"
    log.parent.mkdir(parents=True,exist_ok=True)
    environment=os.environ.copy()
    environment["NUMBA_NUM_THREADS"]="2"
    with log.open("w",encoding="utf-8") as output:
        result=subprocess.run([sys.executable,str(script),"--molecule",molecule,flag,str(index)],
            stdout=output,stderr=subprocess.STDOUT,env=environment)
    content=log.read_text(encoding="utf-8",errors="replace")
    if result.returncode:
        return dict(job=job,status="FAILED",log=str(log),detail=content[-2400:])
    return dict(job=job,status="PASS",generated_this_run=True,detail=content.strip())

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=base.OUT)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    if args.workers<1:parser.error("--workers must be positive")
    base.OUT=args.output.resolve()
    base.OUT.mkdir(parents=True,exist_ok=True)
    os.environ["PAPER_REPRO_FIG3_OUTPUT"]=str(base.OUT)
    # Prepare each shared settings file before workers read it; copying a
    # partially written NPZ concurrently is not a valid cache hit.
    for molecule in ("BeH2","N2"):
        base.prepare_srdd(molecule)
    jobs=[("SRDD",m,i) for m,n in [("BeH2",21),("N2",41)] for i in range(n)]
    jobs += [("FC-IMA",m,i) for m,n in [("BeH2",36),("N2",111)] for i in range(n)]
    started=time.perf_counter()
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in concurrent.futures.as_completed([executor.submit(run,job) for job in jobs]):
            result=future.result();results.append(result)
            print(f"[{len(results)}/{len(jobs)}] {result['status']} {result['job']}: {result['detail']}",flush=True)
    status="PASS" if all(row["status"]=="PASS" for row in results) else "FAILED"
    generated=sum(row.get("generated_this_run",False) for row in results)
    (base.OUT/"replay_jobs.json").write_text(json.dumps(dict(
        status=status,seconds=time.perf_counter()-started,jobs=results,
        generated_jobs=generated,resumed_jobs=len(results)-generated,
        scope="Fresh noisy Born outcomes for frozen published designs; archived noiseless estimates are retained"),indent=2))
    print(status,flush=True)
    raise SystemExit(0 if status=="PASS" else 1)
