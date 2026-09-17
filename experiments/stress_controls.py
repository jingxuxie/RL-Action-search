#!/usr/bin/env python3
"""Additional mechanism controls and diagnostic-cost ablation."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from search_refresh.mdp import rare_action_control,delayed_fork
from search_refresh.confidence import sample_model_box
from search_refresh.menus import selection_probabilities,box_advantage_lower
from search_refresh.refresh import stopped_refresh,candidate_cost


def rare():
    rows=[]
    for n in [100,300,1000,10000,100000]:
        for seed in range(100):
            true=rare_action_control(); box=sample_model_box(true,n,seed+491000)
            m=box.model; q=m.evaluate(m.beta)[2]; old=m.beta.copy(); oldvalue=true.evaluate(old)[0]
            contains=box.contains(true)
            for k in [2,8,32]:
                candidate=selection_probabilities(q,m.beta,k); val=true.evaluate(candidate)[0]
                lo,hi,_,_=box.value_bands(pi=old)
                margin=box_advantage_lower(candidate,old,lo,hi)
                accept=bool(np.all(margin>=0))
                harm=val<oldvalue-1e-10
                if contains: assert not (accept and harm)
                rows.append(dict(n_per_state=n,seed=seed,k=k,samples=int(box.counts.sum()),
                    model_contains=contains,accepted=accept,proposal_value=val,old_value=oldvalue,
                    harmful_proposal=harm,false_accept=accept and harm,
                    accepted_value=val if accept else oldvalue,
                    baseline=true.evaluate(true.beta)[0],matched=true.optimal_budget(k)[0]))
                if accept: old=candidate.copy(); oldvalue=val
    pd.DataFrame(rows).to_csv(ROOT/'results/rare_action_control.csv',index=False,float_format='%.12g')
    print('rare_action_control',len(rows),flush=True)


def exposure_stopping():
    rows=[]
    for depth in [2,4,8,16]:
        for p in [.00001,.0001,.001,.01,.1]:
            # Unchanged large root decision gap versus rare downstream search.
            m=delayed_fork(depth=depth,safe=.2,good_prob=p)
            q=m.evaluate(m.beta)[2]
            for k in [2,8,32]:
                for tol in [.001,.01,.05]:
                    for rule in ['rank','generic']:
                        out=stopped_refresh(m,k,tol,initial=q,rule=rule)
                        value=m.evaluate(out.policy)[0]; opt=m.optimal_budget(k)[0]
                        assert opt-value<=out.bound+1e-10
                        rows.append(dict(depth=depth,good_prob=p,k=k,tolerance=tol,rule=rule,
                            updates=out.updates,backups=out.backup_calls,value=value,matched=opt,
                            gap=opt-value,bound=out.bound,cost=candidate_cost(out,m,k)))
    pd.DataFrame(rows).to_csv(ROOT/'results/exposure_stopping.csv',index=False,float_format='%.12g')
    print('exposure_stopping',len(rows),flush=True)

if __name__=='__main__': rare();exposure_stopping()
