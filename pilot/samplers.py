"""Candidate engines. They see only features, the shared prefix and their own oracle replies."""
from time import perf_counter
from itertools import product
import numpy as np

from .model import GROUPS, BITS, SUPPORTS, LOOKUP

def mix_pairs():
    pairs=[]
    for g in GROUPS:
        for i,j in zip(g,g[1:]+g[:1]):
            a=np.where((BITS[:,i]==1)&(BITS[:,j]==0))[0]
            b=np.array([LOOKUP[tuple(sorted((set(SUPPORTS[k])-{i})|{j}))] for k in a])
            pairs.append((a,b))
    return pairs

XY_PAIRS=mix_pairs()
NEIGHBORS=[]
for s in SUPPORTS:
    ns=set()
    for g in GROUPS:
        for i in set(s)&set(g):
            for j in set(g)-set(s):
                ns.add(LOOKUP[tuple(sorted((set(s)-{i})|{j}))])
    NEIGHBORS.append(sorted(ns))

def state(energy, gamma, beta):
    a=np.exp(-1j*gamma*energy)/np.sqrt(len(energy))
    c,z=np.cos(beta),-1j*np.sin(beta)
    for i,j in XY_PAIRS:
        u,v=a[i].copy(),a[j].copy()
        a[i],a[j]=c*u+z*v,z*u+c*v
    return a

def xy_fit(q, config):
    z=(q-q.mean())/max(q.std(),1e-10)
    order=np.argsort(z)
    alpha=config['xy_train_alpha']
    best=None
    n=0
    for gamma,beta in product(config['xy_gamma_grid'], config['xy_beta_grid']):
        p=np.abs(state(z,gamma,beta))**2
        assert abs(p.sum()-1)<1e-10
        po=p[order]
        mass=np.minimum(po,np.maximum(alpha-np.r_[0.,np.cumsum(po)[:-1]],0.))
        value=float(z[order]@mass/alpha)
        n+=1
        if best is None or value<best[0]:
            best=(value,p/p.sum(),gamma,beta)
    return best[1],n,best[2],best[3]

def propose(method, p, prefix, oracle, q0, qhat, config, seed):
    started=perf_counter()
    rng=np.random.default_rng(seed)
    seen=dict(prefix)
    cap=config['proposal_cap']; budget=config['additional_calls']
    q=q0 if method.endswith('_proxy') else qhat
    states=0;gamma=beta=None
    if method.startswith('xy'):
        probs, states, gamma,beta=xy_fit(q,config)
    elif method=='gibbs_residual':
        energy=(q-q.mean())/max(q.std(),1e-10)
        mass=np.exp(-config['gibbs_beta']*(energy-energy.min()))
        probs=mass/mass.sum()
    elif method=='uniform':
        probs=np.ones(len(BITS))/len(BITS)
    else:
        probs=None

    emitted=cap if probs is not None else 0
    candidates=rng.choice(len(BITS),size=cap,p=probs).tolist() if probs is not None else []
    if method.startswith('rank'):
        candidates=np.argsort(q,kind='stable').tolist()
    if method=='diverse_residual':
        # Cycle through bins defined using only the known equal-weight exposures.
        D=(BITS/4)@p.duration; Y=(BITS/4)@p.yld
        buckets={}
        for j in np.argsort(q,kind='stable'):
            key=tuple([np.digitize(D[j],config['exposure_bins']['duration_years']),
                       np.digitize(Y[j],config['exposure_bins']['distribution_yield'])])
            buckets.setdefault(key,[]).append(int(j))
        order=sorted(buckets,key=lambda k:(q[buckets[k][0]],k))
        candidates=[buckets[k][i] for i in range(max(map(len,buckets.values()))) for k in order if i<len(buckets[k])]

    rows=[]; got=0; oracle_lookup_s=0.
    for step in range(cap):
        if got>=budget:
            break
        if method=='swap':
            good=sorted((j for j,v in seen.items() if v['status']=='feasible'),key=lambda j:seen[j]['cost'])
            if not good or rng.random()<.15:
                j=int(rng.integers(len(BITS)))
            else:
                j=int(rng.choice(NEIGHBORS[int(rng.choice(good[:4]))]))
        else:
            if step==len(candidates):
                break
            j=int(candidates[step])
        hit=j in seen
        if not hit:
            ts=perf_counter();seen[j]=oracle(j);oracle_lookup_s+=perf_counter()-ts
            got+=1
        rows.append({'draw':step+1,'support_id':j,'duplicate':hit,
                     'new_oracle_calls':got,'status':seen[j]['status']})
    seconds=perf_counter()-started-oracle_lookup_s
    return seen,rows,{'raw_generated':emitted or len(rows),'raw_examined':len(rows),'new_calls':got,
                      'duplicate_examined':sum(r['duplicate'] for r in rows),
                      'circuit_evaluations':states,'simulation_training_shots':0,
                      'simulated_readout_draws':emitted if method.startswith('xy') else 0,
                      'gamma':gamma,'beta':beta,'proposal_cpu_seconds':seconds,
                      'proxy_states_scored':len(q) if method!='swap' and method!='uniform' else 0}
