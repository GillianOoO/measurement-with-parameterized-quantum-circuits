"""Independent checks for the Figure 3 Born/Pauli trajectory sampler."""
from pathlib import Path
import itertools
import json
import math
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_fig3_empirical50 as replay
# Import NumPy after replay's optional runtime path selection when run directly.
np = replay.np

def run():
    rng = np.random.default_rng(571902)
    errors = {}
    from shallow_rcdf import givens_layout, rotation_and_derivatives
    worst = 0.
    for m in (2, 3, 4, 5):
        k = m//2
        configs = list(itertools.combinations(range(m), k))
        masks = np.array([sum(1 << (m-1-q) for q in c) for c in configs])
        angles = rng.normal(size=2*(m-1))
        rotation, _ = rotation_and_derivatives(angles, m, 2)
        indices = np.array(configs)
        wedge = np.linalg.det(rotation[indices[:,None,:,None], indices[None,:,None,:]])
        ci = rng.normal(size=(len(masks), len(masks))) + 1j*rng.normal(size=(len(masks), len(masks)))
        ci /= np.linalg.norm(ci)
        initial = np.zeros((2**m, len(masks)), dtype=complex)
        initial[masks] = ci
        layout = np.array([(m-1-i,m-1-j) for _,_,i,j in reversed(givens_layout(m,2))])
        actual = replay.evolved_alpha(initial, layout, angles[::-1], np.zeros(len(angles),np.uint8))
        expected = np.zeros_like(actual)
        expected[masks] = wedge.T@ci
        worst = max(worst, float(np.max(np.abs(expected-actual))))
    assert worst < 2e-12
    errors["gate_vs_independent_exterior_power"] = worst

    eye = np.eye(2)
    x = np.array([[0,1],[1,0]])
    z = np.diag([1,-1])
    matrices = []
    for code in range(16):
        p0 = (x if code&1 else eye)@(z if code&4 else eye)
        p1 = (x if code&2 else eye)@(z if code&8 else eye)
        exact = np.kron(p0,p1)
        actual = replay.pauli_matrix(np.eye(4),1,0,code)
        assert np.array_equal(actual,exact)
        matrices.append(exact)
    vector = rng.normal(size=4)+1j*rng.normal(size=4)
    vector /= np.linalg.norm(vector)
    rho = np.outer(vector,vector.conj())
    worst = 0.
    for p in (0., .000333333333333, .003, .27, 1.):
        actual = (1-15*p/16)*rho
        for pauli in matrices[1:]:
            actual += (p/16)*pauli@rho@pauli.T
        expected = (1-p)*rho + p*np.eye(4)/4
        worst = max(worst,float(np.max(np.abs(actual-expected))))
    assert worst < 2e-12
    errors["pauli_mixture_vs_depolarizing_channel"] = worst

    ci = rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
    ci /= np.linalg.norm(ci)
    masks = np.array([1,2])
    initial = np.zeros((4,2),complex)
    initial[masks]=ci
    full = np.zeros((4,4),complex)
    full[np.ix_(masks,masks)]=ci
    theta=.418
    c,s=math.cos(theta),math.sin(theta)
    gate=np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]],complex)
    layout=np.array([[1,0]],np.int64)
    angles=np.array([theta])
    p=.17
    weights=np.full(16,p/16);weights[0]=1-15*p/16
    conditional_joint=np.zeros((4,4))
    direct_joint=np.zeros((4,4))
    worst=0.
    for acode in range(16):
        alpha=replay.evolved_alpha(initial,layout,angles,np.array([acode],np.uint8))
        for bcode in range(16):
            beta_map=matrices[bcode]@gate
            conditional=alpha@beta_map[:,masks].T
            direct=(matrices[acode]@gate)@full@beta_map.T
            worst=max(worst,float(np.max(np.abs(conditional-direct))))
            conditional_joint+=weights[acode]*weights[bcode]*np.abs(conditional)**2
            direct_joint+=weights[acode]*weights[bcode]*np.abs(direct)**2
    assert worst < 2e-12
    assert np.max(np.abs(conditional_joint-direct_joint))<2e-12
    assert abs(conditional_joint.sum()-1)<2e-12
    errors["conditional_spin_sampling_vs_full_four_qubit_amplitudes"]=worst

    # Check the actual CDF sampling kernel against exact conditional probabilities.
    acode,bcode=5,11
    alpha=replay.evolved_alpha(initial,layout,angles,np.array([acode],np.uint8))
    probabilities=np.sum(np.abs(alpha)**2,axis=1)
    nonzero=np.flatnonzero(probabilities>1e-15)
    conditions=alpha[nonzero]/np.sqrt(probabilities[nonzero,None])
    count=200000
    draws=rng.choice(len(nonzero),size=count,p=probabilities[nonzero]/probabilities.sum())
    pairs=np.column_stack((np.arange(len(nonzero)),np.zeros(len(nonzero),dtype=int)))
    order=np.argsort(draws,kind="stable")
    offsets=np.r_[0,np.cumsum(np.bincount(draws,minlength=len(nonzero)))]
    beta,norm=replay.sample_beta_pairs(conditions,np.array([[bcode]],np.uint8),
        pairs,order,offsets,rng.random(count),masks,layout,angles,4)
    actual=np.bincount(nonzero[draws]*4+beta,minlength=16).reshape(4,4)/count
    expected=np.abs((matrices[acode]@gate)@full@(matrices[bcode]@gate).T)**2
    delta=float(np.max(np.abs(actual-expected)))
    assert delta < .008
    assert norm < 2e-12
    errors["200000_joint_born_draws_max_probability_error"]=delta
    result=dict(status="PASS",checks=errors)
    replay.OUT.mkdir(parents=True,exist_ok=True)
    (replay.OUT/"sampler_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    sys.path.insert(0,str(replay.ARCHIVE/"code"/"methods"/"srdd"))
    run()
