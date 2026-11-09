"""Synthetic bond ETF instances and a constrained portfolio evaluator."""
from dataclasses import dataclass
from itertools import combinations, product
from time import perf_counter

import numpy as np
from scipy.optimize import linprog, minimize

GROUPS = (tuple(range(6)), tuple(range(6, 12)))
SUPPORTS = tuple(a + b for a, b in product(*(tuple(combinations(g, 2)) for g in GROUPS)))
BITS = np.zeros((len(SUPPORTS), 12), dtype=int)
for j, s in enumerate(SUPPORTS):
    BITS[j, list(s)] = 1
LOOKUP = {s: j for j, s in enumerate(SUPPORTS)}
PAIRS = tuple(combinations(range(12), 2))
FEATURES = np.column_stack((BITS, *(BITS[:, a] * BITS[:, b] for a, b in PAIRS)))

@dataclass
class Instance:
    family: str
    seed: int
    yld: np.ndarray
    fee: np.ndarray
    duration: np.ndarray
    cov: np.ndarray
    rank: np.ndarray
    yield_floor: float
    duration_cap: float

def instance(family, seed):
    rng = np.random.default_rng(seed)
    y = np.r_[rng.uniform(.042, .060, 6), rng.uniform(.029, .047, 6)]
    fee = rng.uniform(.0007, .0032, 12)
    d = np.r_[rng.uniform(1.8, 3.4, 6), rng.uniform(.35, 1.8, 6)]
    idio = rng.uniform(.006, .015, 12)
    credit = rng.uniform(.45, 1.15, 12)
    if family == 'complement':
        # Individually lower-ranked hedges share different factor exposure.
        # Both remain eligible and carry a positive distribution yield.
        y[[4, 5, 10, 11]] -= .006
        credit[[4, 5, 10, 11]] *= -.85
    factors = np.c_[.012*d, .020*credit, .012*rng.normal(0, .55, 12)]
    cov = factors @ factors.T + np.diag(idio**2)
    rank = 100*(y-fee) if family != 'aligned' else -(np.diag(cov)/1e-4 - .30*100*(y-fee))
    if family == 'complement':
        # Controlled soft rank bias (e.g. lower liquidity, still over the hard floor).
        # This deliberately challenges a marginal screen; it is not a prevalence model.
        rank[[4, 5, 10, 11]] -= 5.
    floor, cap = (.043, 1.90) if family == 'tight' else (.038, 2.35)
    return Instance(family, seed, y, fee, d, cov, rank, floor, cap)

def short_mask(p, width):
    keep = {j for g in GROUPS for j in sorted(g, key=lambda j:(-p.rank[j],j))[:width]}
    return np.array([set(s) <= keep for s in SUPPORTS])

def cost_terms(p, s):
    idx = np.array(s)
    return p.cov[np.ix_(idx,idx)]/1e-4, -.30*100*(p.yld-p.fee)[idx]

def constraints(p,s):
    idx = np.array(s)
    return np.array([-100*p.yld[idx], p.duration[idx], [1,1,0,0], [-1,-1,0,0]]), np.array([-100*p.yield_floor, p.duration_cap, .65, -.45])

def violation(w,A,b):
    return float(max(abs(w.sum()-1), np.max(A@w-b), .08-w.min(), w.max()-.42))

def evaluate(p, j):
    """Optimize the weights on one support and certify its cost numerically."""
    start = perf_counter()
    s = SUPPORTS[j]
    H, c = cost_terms(p,s)
    A, b = constraints(p,s)
    bounds = [(.08,.42)]*4
    one = np.ones((1,4))
    def lp(g):
        return linprog(g, A_ub=A, b_ub=b, A_eq=one, b_eq=[1.], bounds=bounds, method='highs')
    def f(w): return float(w@H@w+c@w)
    def g(w): return 2*H@w+c
    eq = np.full(4,.25)
    eq_cost = f(eq)
    eq_ok = violation(eq,A,b) <= 1e-9
    if not eq_ok:
        feasible=lp(np.zeros(4))
        if feasible.status == 2:
            return {'status':'infeasible','cost':None,'equal_cost':eq_cost,'equal_ok':False,
                    'seconds':perf_counter()-start}
        if not feasible.success:
            raise RuntimeError(f'Feasibility solver failure for {p.seed}/{j}: {feasible.message}')
        x0=feasible.x
    else:
        x0=eq
    cons=[{'type':'eq','fun':lambda w: w.sum()-1,'jac':lambda w: one[0]},
          {'type':'ineq','fun':lambda w:b-A@w,'jac':lambda w:-A}]
    for ftol in (1e-12,1e-14):
        r=minimize(f,x0,jac=g,bounds=bounds,constraints=cons,method='SLSQP',
                   options={'ftol':ftol,'maxiter':180})
        w=r.x
        v=lp(g(w))
        if not v.success:
            raise RuntimeError(f'Certificate LP failure for {p.seed}/{j}: {v.message}')
        gap=max(0., float(g(w)@(w-v.x)))
        feas=violation(w,A,b)
        if feas <= 1e-8 and gap <= 2e-6:
            break
        x0=(v.x+x0)/2
    else:
        raise RuntimeError(f'Uncertified support {p.seed}/{j}: violation={feas}, gap={gap}')
    return {'status':'feasible','cost':f(w),'certificate_gap':gap,'violation':feas,'weights':w.tolist(),
            'equal_cost':eq_cost,'equal_ok':bool(eq_ok),'duration':float(p.duration[list(s)]@w),
            'yield':float(p.yld[list(s)]@w),'fee':float(p.fee[list(s)]@w),
            'volatility':float(np.sqrt(w@H@w*1e-4)),'seconds':perf_counter()-start}

def proxy(p):
    w=BITS/4
    return np.einsum('ij,jk,ik->i',w,p.cov/1e-4,w) - .30*100*(w@(p.yld-p.fee))

def ridge(ids, y, alpha):
    if not len(ids):
        return np.zeros(len(BITS))
    x = FEATURES.astype(float)
    a=x[ids]
    offset=a.mean(axis=0)
    a=a-offset
    beta=np.linalg.solve(a@a.T+alpha*np.eye(len(a)), y-np.mean(y))
    return (x-offset)@a.T@beta+np.mean(y)

def fit_models(prior, observed, alpha):
    ids=[j for j,v in observed.items() if v['status']=='feasible']
    y=np.array([observed[j]['cost'] for j in ids])
    return prior+ridge(ids,y-prior[ids],alpha), ridge(ids,y,alpha)
