"""Run the frozen pilot. Each instance is checkpointed before the next one starts."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import spearmanr

from .model import SUPPORTS, instance, short_mask, evaluate, proxy, fit_models
from .samplers import propose

HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'protocol.json'
CONFIG=json.loads(PROTOCOL.read_text())
PROTOCOL_SHA=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
CODE_SHA=hashlib.sha256(b''.join((HERE/f).read_bytes() for f in ('model.py','samplers.py','run.py'))).hexdigest()

def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        if path.suffix=='.json':
            json.dump(value,f,ensure_ascii=False,allow_nan=False)
        else:
            fields=list(dict.fromkeys(k for row in value for k in row))
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
            w.writeheader();w.writerows(value)
        f.flush();os.fsync(f.fileno())
    temp.replace(path)

def bucket(v,cfg):
    bins=cfg['exposure_bins']
    return (int(np.digitize(v['duration'],bins['duration_years'])),int(np.digitize(v['yield'],bins['distribution_yield'])))

def cost_best(obs):
    return min((x['cost'] for x in obs.values() if x['status']=='feasible'),default=None)

def metrics(obs, fstar, scale, cfg):
    best=cost_best(obs)
    good=[v for v in obs.values() if v['status']=='feasible']
    near=[v for v in good if v['cost']-fstar <= cfg['relative_regret_threshold']*scale+1e-8]
    wide=[v for v in good if v['cost']-fstar <= .15*scale+1e-8]
    bins={bucket(v,cfg) for v in near}
    return {'best_cost':best,'normalized_regret':(best-fstar)/scale if best is not None else None,
            'feasible_seen':len(good),'near_opt_supports':len(near),'near_opt_exposure_bins':len(bins),
            'near_opt_exposure_bins_15pct':len({bucket(v,cfg) for v in wide}),
            'hit_near_opt':int(bool(near))}, bins

def eval_model(q, truth, ids):
    t=np.array([truth[j]['cost'] for j in ids])
    vals=q[ids]
    k=max(1,int(np.ceil(.10*len(ids))))
    best=set(np.argsort(t,kind='stable')[:k])
    predicted=set(np.argsort(vals,kind='stable')[:k])
    corr=float(spearmanr(t, vals).statistic) if len(ids)>1 and np.ptp(vals)>1e-12 and np.ptp(t)>1e-12 else None
    return corr, len(best&predicted)/k, float(np.mean(abs(t-vals)))

def run_instance(family,seed,split,cfg):
    p=instance(family,seed)
    q0=proxy(p); short=short_mask(p,cfg['shortlist_per_sleeve'])
    # Materialize the exact small reference *offline*. Candidate engines cannot inspect this table.
    field={j:evaluate(p,j) for j in range(len(SUPPORTS))}
    good={j:v for j,v in field.items() if v['status']=='feasible'}
    if not good:
        raise RuntimeError(f'No feasible support: {family}/{seed}')
    best_id=min(good,key=lambda j:good[j]['cost']); fstar=good[best_id]['cost']
    bound=min(v['cost']-v['certificate_gap'] for v in good.values())
    scale=float(np.median([v['cost'] for v in good.values()])-fstar)
    assert scale>0
    meta={'split':split,'family':family,'instance_seed':seed}
    screen=[]
    for width in range(2,7):
        mask=short_mask(p,width)
        fshort={j:v for j,v in good.items() if mask[j]}
        short_bound=min((v['cost']-v['certificate_gap'] for v in fshort.values()), default=None)
        sid=min(fshort,key=lambda j:fshort[j]['cost']) if fshort else None
        fid=fshort[sid]['cost'] if sid is not None else None
        eq=[j for j in fshort if field[j]['equal_ok']]
        e=min(eq,key=lambda j:field[j]['equal_cost']) if eq else None
        wa=field[e]['equal_cost'] if e is not None else None
        fa=field[e]['cost'] if e is not None else None
        total=wa-fstar if wa is not None else None
        if total is not None:
            assert abs(total-((wa-fa)+(fa-fid)+(fid-fstar)))<1e-9
        screen.append({**meta,'retained_per_sleeve':width,'supports':int(mask.sum()),
                       'feasible_supports':len(fshort),'full_opt_support':best_id,'short_opt_support':sid,
                       'equal_weight_selected_support':e,'full_opt_cost':fstar,'full_opt_lower_bound':bound,
                       'short_opt_cost':fid,'short_opt_lower_bound':short_bound,'equal_weight_cost':wa,'same_support_reweighted_cost':fa,
                       'screening_gap':fid-fstar if fid is not None else None,
                       'support_search_gap':fa-fid if fa is not None else None,
                       'weight_gap':wa-fa if wa is not None else None,'total_gap':total,'reference_scale':scale})
    chosen=screen[cfg['shortlist_per_sleeve']-2]
    fshort=chosen['short_opt_cost']
    reference_bins=metrics(good,fstar,scale,cfg)[0]['near_opt_exposure_bins']
    metrics_rows=[];draws=[];fit_rows=[];curves=[]
    for r in range(cfg['proposal_repeats']):
        rng=np.random.default_rng(np.random.SeedSequence([seed,r,1401]))
        half=cfg['initial_calls']//2
        warm_ids=np.r_[rng.choice(np.flatnonzero(short),half,replace=False),
                       rng.choice(np.flatnonzero(~short),cfg['initial_calls']-half,replace=False)]
        prefix={int(j):field[int(j)] for j in warm_ids}
        t=perf_counter()
        qhat,qdirect=fit_models(q0,prefix,cfg['ridge_alpha'])
        fit_seconds=perf_counter()-t
        held=[j for j in good if j not in prefix]
        rank=[('equal_weight_proxy',q0),('direct_ridge',qdirect),('residual_ridge',qhat)]
        for name,q in rank:
            corr,recall,mae=eval_model(q,field,held)
            fit_rows.append({**meta,'replicate':r,'model':name,'training_feasible':sum(j in good for j in prefix),
                             'heldout_feasible':len(held),'spearman':corr,'top_decile_recall':recall,'mae':mae})
        prefix_metric,prefix_bins=metrics(prefix,fstar,scale,cfg)
        draws.extend({**meta,'replicate':r,'method':'shared_prefix','draw':i+1,'support_id':j,
                      'duplicate':False,'new_oracle_calls':i+1,'status':field[j]['status'],
                      'outside_shortlist':not bool(short[j])} for i,j in enumerate(prefix))
        for method in cfg['methods']:
            observed, trace, account=propose(method,p,prefix,lambda j:field[j],q0,qhat,cfg,seed*1000+r+17)
            m,bins=metrics(observed,fstar,scale,cfg)
            unique_new=set(observed)-set(prefix)
            new_metrics,_=metrics({j:observed[j] for j in unique_new},fstar,scale,cfg)
            assert account['new_calls']==len(unique_new)
            best=m['best_cost']
            recovery=next((n for n,(j,v) in enumerate(observed.items(),1)
                           if v['status']=='feasible' and fshort is not None and v['cost']<fshort-1e-5),None)
            for limit in cfg['budget_checkpoints']:
                partial=dict(list(observed.items())[:len(prefix)+limit])
                summary,_=metrics(partial,fstar,scale,cfg)
                curves.append({**meta,'replicate':r,'method':method,'additional_budget':limit,
                               'new_calls':len(partial)-len(prefix),**summary,
                               'beats_shortlist':int(fshort is not None and summary['best_cost'] is not None
                                                     and summary['best_cost']<fshort-1e-5)})
            metrics_rows.append({**meta,'replicate':r,'method':method,**m,**account,
                       'initial_calls':len(prefix),'total_calls':len(observed),
                       'initial_best':prefix_metric['best_cost'],'initial_near_opt_bins':len(prefix_bins),
                       'reference_near_opt_bins':reference_bins,'new_exposure_bins':len(bins-prefix_bins),
                       'new_near_opt_supports':new_metrics['near_opt_supports'],
                       'new_feasible':new_metrics['feasible_seen'],
                       'new_outside_shortlist':sum(not short[j] for j in unique_new),
                       'beats_shortlist':int(best is not None and fshort is not None and best<fshort-1e-5),
                       'first_recovery_total_call':recovery,
                       'improves_prefix':int(best is not None and (prefix_metric['best_cost'] is None or best<prefix_metric['best_cost']-1e-5)),
                       'fit_cpu_seconds':fit_seconds if method.endswith('_residual') else 0.,
                       'oracle_cpu_seconds_replayed':sum(v['seconds'] for v in observed.values())})
            draws.extend({**meta,'replicate':r,'method':method,**x,
                          'outside_shortlist':not bool(short[x['support_id']])} for x in trace)
    return {'protocol_sha256':PROTOCOL_SHA,'code_sha256':CODE_SHA,'metrics':metrics_rows,'screening':screen,
            'surrogate':fit_rows,'draws':draws,'curves':curves,
            'field':[{**meta,'support_id':j,'fund_ids':';'.join('F%02d'%(i+1) for i in SUPPORTS[j]),
                      'inside_shortlist':bool(short[j]),**{k:v for k,v in val.items() if k!='weights'},
                      'weights':';'.join(f'{w:.12g}' for w in val.get('weights',[]))}
                     for j,val in field.items()],
            'inputs':[{**meta,'fund_id':'F%02d'%(i+1),'sleeve':'income' if i<6 else 'short',
                       'distribution_yield':float(p.yld[i]),'fee':float(p.fee[i]),
                       'duration_years':float(p.duration[i]),'ranking_score':float(p.rank[i]),
                       'yield_floor':p.yield_floor,'duration_cap':p.duration_cap} for i in range(12)]}

def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--split',choices=['dev','evaluation','all'],default='all')
    a.add_argument('--output',type=Path,default=HERE.parent/'runs'/'raw')
    a.add_argument('--checkpoints',type=Path,default=HERE.parent/'runs'/'checkpoints')
    args=a.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True);args.checkpoints.mkdir(parents=True,exist_ok=True)
    splits=['dev','evaluation'] if args.split=='all' else [args.split]
    completed=[]
    for split in splits:
        for family,seed in CONFIG[split]:
            path=args.checkpoints/f'{split}_{family}_{seed}.json'
            data=json.loads(path.read_text()) if path.exists() else {}
            if data.get('protocol_sha256')!=PROTOCOL_SHA or data.get('code_sha256')!=CODE_SHA:
                ts=perf_counter()
                data=run_instance(family,seed,split,CONFIG)
                atomic(path,data)
                print(f'{split} {family}/{seed} complete in {perf_counter()-ts:.2f}s; checkpoint saved',flush=True)
            else:
                print(f'{split} {family}/{seed} resumed',flush=True)
            completed.append(data)
            # Publish complete CSV snapshots after every certified instance.
            for key in ('metrics','screening','surrogate','draws','curves','field','inputs'):
                atomic(args.output/(key+'.csv'),[row for d in completed for row in d[key]])

if __name__=='__main__':
    main()
