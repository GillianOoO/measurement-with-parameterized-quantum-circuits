"""Joint Born samples with exact propagated Pauli faults for noisy FC-IMA."""
from __future__ import annotations
import ast
import sys
import types
from pathlib import Path
import replay_fig3_empirical50 as base
import numpy as np
from numba import njit
from qiskit import transpile
from qiskit.quantum_info import Clifford, Pauli
import time
import json
import argparse

def dependencies():
    root=base.ARCHIVE/"code"/"methods"/"fully_commuting"
    depth=base.load_module("empirical50_fc_depth",root/"traditional_cm_depth.py")
    error=base.load_module("empirical50_fc_error",root/"traditional_cm_error_eval.py")
    error._DEPTH_MODULE=depth
    path=root/"traditional_cm_noise_eval.py"
    tree=ast.parse(path.read_text())
    tree.body=[node for node in tree.body if not (
        isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"depth","error"} for t in node.targets))]
    noise=types.ModuleType("empirical50_fc_noise")
    noise.__file__=str(path)
    noise.depth,noise.error=depth,error
    sys.modules[noise.__name__]=noise
    exec(compile(tree,str(path),"exec"),noise.__dict__)
    noise.ERROR_RESULTS=base.DATA/"fully_commuting"/"BeH2_N2_FC_IMA"/"error_eval"/"results"
    return depth,error,noise

def compiled_group(depth,noise,group):
    complete=depth.complete_commuting_basis([term.label for term in group])
    expanded=depth.build_expanded_yen_circuit(complete)
    core=Clifford(expanded)
    local=depth.local_qwc_diagonalizer(core,complete)
    expanded.compose(local,inplace=True)
    final=Clifford(expanded)
    raw=core.to_circuit()
    raw.compose(local,inplace=True)
    optimized=transpile(raw,basis_gates=list(depth.BASIS_GATES),
        optimization_level=1,seed_transpiler=depth.DEFAULT_SEED)
    if Clifford(optimized)!=final:
        raise RuntimeError("FC compilation changed the unitary")
    masks=[];signs=[]
    for term in group:
        transformed=Pauli(term.label).evolve(final,frame="s")
        if np.any(transformed.x):
            raise RuntimeError("FC observable is not diagonal after rotation")
        masks.append(noise.qiskit_mask(transformed,"z"))
        signs.append(noise.pauli_sign(transformed))
    return optimized,np.array(masks,np.uint64),np.array(signs)

def flip_maps(noise,circuit):
    operations=noise.compiled_operations(circuit)
    x=np.zeros(circuit.num_qubits,np.uint64)
    z=np.array([1<<q for q in range(circuit.num_qubits)],np.uint64)
    maps=[]
    for name,qubits,params in reversed(operations):
        if name=="cx":
            control,target=qubits
            table=[]
            for code in range(16):
                xf=((code&1)<<control)|(((code>>1)&1)<<target)
                zf=(((code>>2)&1)<<control)|(((code>>3)&1)<<target)
                mask=0
                for output in range(circuit.num_qubits):
                    if ((xf&int(z[output]))^(zf&int(x[output]))).bit_count()&1:
                        mask|=1<<output
                table.append(mask)
            maps.append(table)
            x^=((x>>np.uint64(control))&np.uint64(1))<<np.uint64(target)
            z^=((z>>np.uint64(target))&np.uint64(1))<<np.uint64(control)
        elif name=="rz":
            turns=int(round(params[0]/(np.pi/2)))
            assert abs(params[0]-turns*np.pi/2)<2e-8
            if turns&1:
                q=qubits[0]
                z^=((x>>np.uint64(q))&np.uint64(1))<<np.uint64(q)
        elif name in ("sx","sxdg"):
            q=qubits[0]
            x^=((z>>np.uint64(q))&np.uint64(1))<<np.uint64(q)
        elif name not in ("x","id","barrier"):
            raise RuntimeError(f"Unsupported compiled operation {name}")
    return np.asarray(maps[::-1],np.uint32).reshape(-1,16),operations

@njit(cache=True)
def evolve_compiled(state,codes,first,second,angles):
    a=state.astype(np.complex128)
    for g in range(len(codes)):
        q=first[g]
        bit=1<<q
        if codes[g]==0:
            p0=np.exp(-.5j*angles[g]);p1=np.exp(.5j*angles[g])
            for i in range(len(a)):
                a[i]*=p1 if i&bit else p0
        elif codes[g]==1:
            for i in range(len(a)):
                if not i&bit:
                    j=i^bit
                    x,y=a[i],a[j]
                    a[i]=(.5+.5j)*x+(.5-.5j)*y
                    a[j]=(.5-.5j)*x+(.5+.5j)*y
        elif codes[g]==2:
            for i in range(len(a)):
                if not i&bit:
                    j=i^bit
                    a[i],a[j]=a[j],a[i]
        elif codes[g]==3:
            target=1<<second[g]
            for i in range(len(a)):
                if i&bit and not i&target:
                    j=i^target
                    a[i],a[j]=a[j],a[i]
    return a

@njit(cache=True)
def parity(value):
    value^=value>>32
    value^=value>>16
    value^=value>>8
    value^=value>>4
    return (0x6996>>(value&15))&1

@njit(cache=True)
def scores(samples,masks,coefficients):
    out=np.zeros(len(samples))
    for i in range(len(samples)):
        value=0.
        for j in range(len(masks)):
            value+=coefficients[j]*(1.-2.*parity(np.uint64(samples[i])&masks[j]))
        out[i]=value
    return out

def run_group(molecule,group_index):
    destination=base.OUT/"fc_groups"/f"{molecule}_{group_index:03d}.npz"
    if destination.exists():
        return destination
    started=time.perf_counter()
    depth,error,noise=dependencies()
    terms,identity,_=error.load_pauli_terms(base.DATA/molecule/"inputs"/"hamiltonian_pauli_blocked_spin.csv")
    groups,_=noise.load_ima_groups(molecule,terms)
    shots=noise.load_frozen_shots(molecule,len(groups))
    pooled=noise.pooled_shot_counts(groups,shots)
    group=groups[group_index]
    count=int(shots[group_index])
    circuit,masks,signs=compiled_group(depth,noise,group)
    maps,operations=flip_maps(noise,circuit)
    opcode={"rz":0,"sx":1,"x":2,"cx":3}
    active=[item for item in operations if item[0] in opcode]
    codes=np.array([opcode[name] for name,_,_ in active],dtype=np.int64)
    first=np.array([qs[0] for _,qs,_ in active],dtype=np.int64)
    second=np.array([qs[1] if len(qs)>1 else -1 for _,qs,_ in active],dtype=np.int64)
    angles=np.array([params[0] if params else 0. for _,_,params in active])
    state=base.dense_state(molecule)
    rotated=evolve_compiled(state,codes,first,second,angles)
    probabilities=np.abs(rotated)**2
    norm_error=abs(float(probabilities.sum())-1.)
    if norm_error>2e-10:
        raise RuntimeError("FC circuit does not preserve state norm")
    cumulative=np.cumsum(probabilities)
    cumulative/=cumulative[-1]
    rng=np.random.default_rng(base.seed(molecule,"FC-IMA",group_index))
    n=base.REPEATS*count
    ideal=np.searchsorted(cumulative,rng.random(n),side="right").astype(np.uint32)
    uniforms=rng.random((len(maps),n))
    faults=rng.integers(1,16,size=(len(maps),n),dtype=np.uint8)
    outcomes=np.tile(ideal,(len(base.P_GRID)-1,1))
    for pi,p in enumerate(base.P_GRID[1:]):
        for location,table in enumerate(maps):
            mask=uniforms[location]<15*p/16
            outcomes[pi,mask]^=table[faults[location,mask]]
    coefficients=signs*np.array([term.coefficient/pooled[term.source_row] for term in group])
    contributions=scores(outcomes.reshape(-1),masks,coefficients).reshape(len(base.P_GRID)-1,base.REPEATS,count).sum(axis=2)
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,contributions=contributions,
        outcomes=outcomes.reshape(len(base.P_GRID)-1,base.REPEATS,count),
        count=count,seed=np.uint64(base.seed(molecule,"FC-IMA",group_index)),
        cnot_count=len(maps),maximum_norm_error=norm_error,output_z_masks=masks,
        output_signs=signs,pooled_score_coefficients=coefficients)
    print(f"[FC-IMA] {molecule} group {group_index+1}/{len(groups)}, shots {count}, CNOTs {len(maps)}, {time.perf_counter()-started:.1f}s",flush=True)
    return destination

def validate():
    depth,error,noise=dependencies()
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    rng=np.random.default_rng(3172)
    qc=QuantumCircuit(3)
    qc.sx(0);qc.cx(0,1);qc.rz(np.pi/2,1);qc.sx(2);qc.cx(1,2);qc.x(0);qc.rz(-np.pi,2)
    maps,ops=flip_maps(noise,qc)
    v=rng.normal(size=8)+1j*rng.normal(size=8);v/=np.linalg.norm(v)
    actual=evolve_compiled(v,np.array([{"rz":0,"sx":1,"x":2,"cx":3}[x[0]] for x in ops]),
        np.array([x[1][0] for x in ops]),np.array([x[1][1] if len(x[1])>1 else -1 for x in ops]),
        np.array([x[2][0] if x[2] else 0. for x in ops]))
    expected=Statevector(v).evolve(qc).data
    state_error=float(np.max(np.abs(actual-expected)))
    assert state_error<2e-12
    loc=0
    for index,(name,qubits,params) in enumerate(ops):
        if name!="cx":continue
        suffix=QuantumCircuit(3)
        for instruction in qc.data[index+1:]:
            suffix.append(instruction.operation,[qc.find_bit(q).index for q in instruction.qubits])
        for code in range(16):
            xmask=((code&1)<<qubits[0])|(((code>>1)&1)<<qubits[1])
            zmask=(((code>>2)&1)<<qubits[0])|(((code>>3)&1)<<qubits[1])
            pauli=Pauli((np.array([(zmask>>q)&1 for q in range(3)],bool),
                         np.array([(xmask>>q)&1 for q in range(3)],bool)))
            propagated=pauli.evolve(Clifford(suffix),frame="s")
            expected_mask=sum(int(b)<<q for q,b in enumerate(propagated.x))
            assert expected_mask==int(maps[loc,code])
        loc+=1
    result=dict(status="PASS",statevector_error=state_error,independent_single_fault_checks=loc*16)
    base.OUT.mkdir(parents=True,exist_ok=True)
    (base.OUT/"fc_sampler_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--validate",action="store_true")
    parser.add_argument("--molecule",choices=("BeH2","N2"))
    parser.add_argument("--group",type=int)
    parser.add_argument("--output",type=Path,default=base.OUT)
    args=parser.parse_args()
    base.OUT=args.output.resolve()
    if args.validate:validate()
    elif args.molecule is not None and args.group is not None:run_group(args.molecule,args.group)
    else:parser.error("Choose --validate or --molecule with --group")
