"""Exploratory follow-up on fresh synthetic instances: exposure-conditioned acquisition."""
from itertools import product
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json

import numpy as np

from .model import BITS, SUPPORTS, evaluate, fit_models, instance, proxy, short_mask
from .run import CONFIG, atomic, cost_best, metrics, CODE_SHA
from .samplers import propose, state

HERE=Path(__file__).resolve().parent
PROTO=HERE/'tilt_protocol.json'
P=json.loads(PROTO.read_text())
SIGNATURE=hashlib.sha256(PROTO.read_bytes()+Path(__file__).read_bytes()+CODE_SHA.encode()).hexdigest()

def standardized(x):
    return (x-x.mean())/max(x.std(),1e-10)

def tilted_state(energy, grid):
    v=standardized(energy)
    order=np.argsort(v);alpha=CONFIG['xy_train_alpha']
    best=None
    for gamma,beta in grid:
        p=abs(state(v,gamma,beta))**2
        po=p[order]
        mass=np.minimum(po,np.maximum(alpha-np.r_[0.,np.cumsum(po)[:-1]],0.))
        loss=float(v[order]@mass/alpha)
        if best is None or loss<best[0]:
            best=(loss,p/p.sum())
    return best[1]

def tilted_proposals(method,prefix,field,q,D,Y,rng):
    cfg=CONFIG; count=P['raw_proposal_cap']; energy=standardized(q)
    arms=[energy,energy+P['tilt_strength_sd']*standardized(D),energy-P['tilt_strength_sd']*standardized(Y)]
    grid=list(product(cfg['xy_gamma_grid'],cfg['xy_beta_grid']))
    states=0; arrays=[]
    for k,e in enumerate(arms):
        size=len(range(k,count,3))
        if method=='xy_tilted':
            # Divide the same 40 state evolutions among three acquisitions.
            n=len(range(k,P['total_xy_evaluations'],3))
            chosen=np.linspace(0,len(grid)-1,n,dtype=int)
            prob=tilted_state(e,[grid[j] for j in chosen]);states+=n
        elif method=='gibbs_tilted':
            v=standardized(e);mass=np.exp(-cfg['gibbs_beta']*(v-v.min()));prob=mass/mass.sum()
        else:
            arrays.append(np.argsort(e,kind='stable')[:size])
            continue
        arrays.append(rng.choice(len(BITS),size=size,p=prob))
    seen=dict(prefix);new=0;consumed=0;logged=[]
    for i in range(count):
        if new>=P['additional_calls']:
            break
        j=int(arrays[i%3][i//3]);consumed+=1
        duplicate=j in seen
        if not duplicate:
            seen[j]=field[j];new+=1
        logged.append((consumed,j,duplicate,new))
    return seen,{'new_calls':new,'raw_generated':count,'raw_examined':consumed,
                 'circuit_evaluations':states,'simulated_readout_draws':count if method=='xy_tilted' else 0},logged

def one(family,seed):
    p=instance(family,seed); q0=proxy(p); S=short_mask(p,CONFIG['shortlist_per_sleeve'])
    field={j:evaluate(p,j) for j in range(len(SUPPORTS))}
    fs=cost_best(field); sc=float(np.median([v['cost'] for v in field.values() if v['status']=='feasible'])-fs)
    max_bins=metrics(field,fs,sc,CONFIG)[0]
    D=(BITS/4)@p.duration; Y=(BITS/4)@p.yld
    rows=[];draws=[]
    methods=['rank_residual','gibbs_residual','xy_residual','rank_tilted','gibbs_tilted','xy_tilted']
    for r in range(P['repeats']):
        rng=np.random.default_rng(np.random.SeedSequence([seed,r,1401]))
        n=CONFIG['initial_calls']//2
        ids=np.r_[rng.choice(np.flatnonzero(S),n,replace=False),rng.choice(np.flatnonzero(~S),n,replace=False)]
        prefix={int(j):field[int(j)] for j in ids};q,_=fit_models(q0,prefix,CONFIG['ridge_alpha'])
        for method in methods:
            started=perf_counter()
            if method.endswith('_tilted'):
                seen,counts,trace=tilted_proposals(method,prefix,field,q,D,Y,np.random.default_rng(seed*1000+r+17))
            else:
                seen,t,counts=propose(method,p,prefix,lambda j:field[j],q0,q,CONFIG,seed*1000+r+17)
                trace=[(x['draw'],x['support_id'],x['duplicate'],x['new_oracle_calls']) for x in t]
            elapsed=perf_counter()-started
            m,_=metrics(seen,fs,sc,CONFIG)
            rows.append({'family':family,'instance_seed':seed,'replicate':r,'method':method,
                         'initial_calls':len(prefix),'new_calls':counts['new_calls'],
                         'raw_generated':counts['raw_generated'],'raw_examined':counts['raw_examined'],
                         'circuit_evaluations':counts['circuit_evaluations'],
                         'simulated_readout_draws':counts['simulated_readout_draws'],
                         'wall_seconds_cached':elapsed,
                         'reference_bins_5pct':max_bins['near_opt_exposure_bins'],
                         'reference_bins_15pct':max_bins['near_opt_exposure_bins_15pct'],**m})
            for t,j,dup,new in trace:
                draws.append({'family':family,'instance_seed':seed,'replicate':r,'method':method,
                              'proposal':t,'support_id':j,'duplicate':dup,'new_calls':new,
                              'status':seen[j]['status'],'true_cost':seen[j]['cost'],
                              'duration':seen[j].get('duration'),'yield':seen[j].get('yield')})
    return {'signature':SIGNATURE,'rows':rows,'draws':draws}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE.parent/'runs'/'raw')
    parser.add_argument('--checkpoints',type=Path,default=HERE.parent/'runs'/'checkpoints')
    a=parser.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True);a.checkpoints.mkdir(parents=True,exist_ok=True)
    all_runs=[]
    for family,seed in P['instances']:
        p=a.checkpoints/f'tilt_{seed}.json'
        x=json.loads(p.read_text()) if p.exists() else {}
        if x.get('signature')!=SIGNATURE:
            x=one(family,seed);atomic(p,x)
            status='checkpoint saved'
        else:
            status='resumed'
        all_runs.append(x)
        atomic(a.output/'tilts.csv',[row for run in all_runs for row in run['rows']])
        atomic(a.output/'tilt_draws.csv',[row for run in all_runs for row in run['draws']])
        print(f'Exploratory {family}/{seed} {status}',flush=True)

if __name__=='__main__':
    main()
