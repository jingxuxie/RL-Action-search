"""Rank-sensitive, residual-certified stopping of budget-matched sweeps.

All certification costs are counted. A diagnostic backup is reused if another
sweep is needed. This routine allocates refresh work for a *specified* K; it
is not an optimal joint compute scheduler over all possible K and m.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .menus import menu_envelope,selection_probabilities


def residual_bands(model,q,k,model_box=None,next_q=None):
    if next_q is None: next_q=model.backup(q,k)
    errors=np.zeros_like(q) if model_box is None else model_box.backup_errors(q,k)
    resid=np.abs(next_q-q)+errors
    b=np.zeros(model.H+1)
    for h in range(model.H-1,-1,-1):
        b[h]=float(np.max(resid[(model.level==h)[:,None] & (model.beta>0)]))+model.gamma*b[h+1]
    # No clipping: the untruncated boxes make comparison to 2*b transparent.
    lower=q-b[model.level,None]; upper=q+b[model.level,None]
    return lower,upper,b[:-1]


def certificate(model,q,k,model_box=None,next_q=None):
    lower,upper,b=residual_bands(model,q,k,model_box,next_q)
    envelope,ambiguity=menu_envelope(q,model.beta,k,lower,upper)
    rank_bound=sum(model.gamma**h*float(np.max(envelope[model.level==h])) for h in range(model.H))
    generic_bound=sum(model.gamma**h*2*b[h] for h in range(model.H))
    return float(rank_bound),float(generic_bound),envelope,ambiguity,b


@dataclass
class RefreshResult:
    critic: np.ndarray
    policy: np.ndarray
    updates: int
    backup_calls: int
    diagnostic_calls: int
    bound: float
    certified: bool
    trace: list
    elapsed: float
    transition_products: int
    menu_pairs: int


def stopped_refresh(model,k,tolerance=0.01,initial=None,model_box=None,
                    max_updates=None,rule='rank'):
    """Return the first certified iterate, or the cap with certified=False.
    The m=0 diagnostic costs one backup. Final failed diagnostics also count.
    The baseline critic construction is shared by compared methods and its
    separate cost must be added in end-to-end comparisons.
    """
    if rule not in ('rank','generic'): raise ValueError("unknown rule")
    if tolerance<0: raise ValueError("negative tolerance")
    start=time.perf_counter()
    if initial is None: initial=model.evaluate(model.beta)[2]
    q=initial.copy(); cap=model.H if max_updates is None else int(max_updates)
    trace=[]; pairs=0; bound=float('inf'); certified=False
    for m in range(cap+1):
        next_q=model.backup(q,k)
        if rule=='rank':
            rb,gb,env,ambiguity,b=certificate(model,q,k,model_box,next_q)
            bound=rb; pairs+=model.S*model.A*(model.A-1)//2
        else:
            _,_,b=residual_bands(model,q,k,model_box,next_q)
            gb=sum(model.gamma**h*2*b[h] for h in range(model.H))
            rb=np.nan; bound=gb; ambiguity=np.zeros(model.S)
        trace.append(dict(updates=m,bound=float(bound),generic=float(gb),
                          max_radius=float(b.max()),ambiguity=float(ambiguity.max())))
        if bound<=tolerance:
            certified=True; break
        if m<cap: q=next_q
    pi=selection_probabilities(q,model.beta,k)
    calls=m+1
    return RefreshResult(q,pi,m,calls,calls,float(bound),certified,trace,
                         time.perf_counter()-start,calls*model.edges,pairs)


def candidate_cost(result,model,k,deployment_decisions=0):
    """Explicit operation proxy, NOT measured time or a universal FLOP model.
    Sorting and powers are charged; finite-data model-error calculations and
    rank integration cost additional work. Robust safety gates are excluded
    here and must be reported separately. Binary sample policies could be
    compiled to avoid K scores; this proxy models uncompiled action scoring.
    """
    per_backup=model.edges+3*model.S*model.A+model.S*model.A*np.log2(max(2,model.A))
    diagnostic=result.menu_pairs*(1+np.log2(max(2,model.A)))
    return float(result.backup_calls*per_backup+diagnostic+deployment_decisions*k)
