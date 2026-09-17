#!/usr/bin/env python3
"""Deterministic CPU experiments. Figures are built separately.

No result selection is based on true-MDP performance. True evaluation is used
only to audit methods and define explicitly named hindsight oracles.
"""
from __future__ import annotations
import argparse,hashlib,json,os,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from search_refresh.menus import (selection_probabilities,expected_max,menu_envelope,
    enumerate_menu_statistics,fixed_box_worst_regret)
from search_refresh.mdp import delayed_fork,random_layered
from search_refresh.confidence import sample_model_box
from search_refresh.refresh import stopped_refresh,certificate,candidate_cost
from search_refresh.linear import LayerwiseRidge

OUT=ROOT/'results'
TIMINGS={}

def save(name,rows):
    df=pd.DataFrame(rows); df.to_csv(OUT/(name+'.csv'),index=False,float_format='%.12g')
    print(name,len(df),flush=True)
    return df


def geometry():
    rows=[]
    for a in [2,3,4,5]:
        for seed in range(20):
            rng=np.random.default_rng(10000*a+seed)
            q=rng.normal(size=a); b=rng.dirichlet(np.full(a,.6))
            lo=rng.normal(size=a); hi=lo+rng.uniform(.01,2,a)
            for k in [1,2,3,8,64]:
                g,amb=menu_envelope(q,b,k,lo,hi)
                robust,_=fixed_box_worst_regret(q,b,k,lo,hi)
                err=0; enumerated=0
                if k<=3:
                    _,g2,p2,_=enumerate_menu_statistics(q,b,k,lo,hi)
                    err=max(abs(g-g2),abs(amb-p2)); enumerated=a**k
                assert robust<=g+1e-9 and g<=4*robust+1e-9 and err<1e-9
                rows.append(dict(actions=a,seed=seed,k=k,envelope=g,robust=robust,
                                 ratio=g/robust if robust>1e-10 else np.nan,
                                 ambiguity=amb,enumeration_error=err,menus_enumerated=enumerated,
                                 vertices_checked=2**a))
    save('geometry',rows)
    rows=[]
    for k in range(1,65):
        q=np.array([0.,1.,3.]); v=np.array([1.,0.,3.]); b=np.ones(3)/3
        g,_=menu_envelope(q,b,k,v,v)
        predicted=(2/3)**k-2*(1/3)**k
        assert abs(g-predicted)<1e-12
        rows.append(dict(k=k,loss=g,formula=predicted))
    save('screening',rows)


def forks():
    rows=[]
    for depth in [1,2,4,8,16]:
        for goodp in [.02,.1,.25]:
            for safep in [.03,.1,.4]:
                m=delayed_fork(depth,safe=.2,good_prob=goodp,safe_prob=safep)
                q0=m.evaluate(m.beta)[2]
                for k in [1,2,3,4,8,16,32,64]:
                    q=q0.copy(); optimal=m.optimal_budget(k)[0]
                    for it in range(depth+2):
                        value=m.return_of(q,k)
                        rows.append(dict(depth=depth,goodp=goodp,safep=safep,k=k,
                            updates=it,value=value,optimal=optimal,gap=optimal-value))
                        q=m.backup(q,k)
    save('forks',rows)
    m=delayed_fork(safe=2.,good=10.)
    q=m.evaluate(m.beta)[2]
    save('opening_fork',[dict(k=k,frozen=m.return_of(q,k),refreshed=m.return_of(m.backup(q,k),k),
                             matched=m.optimal_budget(k)[0]) for k in range(1,65)])


def exact_stopping():
    rows=[]; grids=[]
    configurations=[(h,a,seed) for h in [3,6,10] for a in [2,8] for seed in range(12)]
    configurations += [(h,32,seed) for h in [3,6] for seed in range(3)]
    for h,a,seed in configurations:
        mdp=random_layered(seed+100*a+10000*h,horizon=h,width=2,actions=a)
        q0=mdp.evaluate(mdp.beta)[2]; baseline=mdp.evaluate(mdp.beta)[0]
        for k in [2,8,32]:
            opt=mdp.optimal_budget(k)[0]; q=q0.copy(); first=None
            for it in range(h+1):
                value=mdp.return_of(q,k)
                if opt-value<=.02 and first is None: first=it
                grids.append(dict(horizon=h,actions=a,seed=seed,k=k,updates=it,
                                  value=value,matched=opt,gap=opt-value,baseline=baseline))
                q=mdp.backup(q,k)
            for rule in ['rank','generic']:
                out=stopped_refresh(mdp,k,.02,initial=q0,rule=rule)
                value=mdp.evaluate(out.policy)[0]
                assert opt-value<=out.bound+1e-9
                per=mdp.edges+3*mdp.S*a+mdp.S*a*np.log2(max(2,a))
                rows.append(dict(horizon=h,actions=a,seed=seed,k=k,rule=rule,
                    updates=out.updates,backups=out.backup_calls,bound=out.bound,
                    certified=out.certified,value=value,matched=opt,gap=opt-value,
                    oracle_updates=first,transition_products=out.transition_products,
                    menu_pairs=out.menu_pairs,operation_proxy=candidate_cost(out,mdp,k),
                    fixed_h_sweep_proxy=h*per,fixed_full_sweep_proxy=(h-1)*per,backward_dp_proxy=per,
                    elapsed_seconds=out.elapsed))
    # Runtime is nondeterministic metadata, deliberately separate from scientific CSV.
    df=pd.DataFrame(rows); df[['horizon','actions','seed','k','rule','elapsed_seconds']].to_json(OUT/'timing_stopping.json',orient='records',indent=2)
    save('exact_stopping',df.drop(columns='elapsed_seconds').to_dict('records'))
    save('exact_grid',grids)


def offline():
    rows=[]; data_rows=[]
    families=['fork1','fork4','stochastic3','stochastic5']
    for family in families:
        for n in [200,2000,20000,200000]:
            for seed in range(30):
                if family.startswith('fork'):
                    true=delayed_fork(depth=int(family[-1]))
                else:
                    h=int(family[-1]); true=random_layered(seed+80000,horizon=h,width=2,actions=4 if h==3 else 8)
                box=sample_model_box(true,n,seed+7_000_000,known_transitions=False)
                model=box.model; q0=model.evaluate(model.beta)[2]
                old=model.beta.copy(); old_true=true.evaluate(old)[0]
                contains=box.contains(true); base=old_true
                data_rows.append(dict(family=family,n_per_state=n,seed=seed,samples=int(box.counts.sum()),
                                      min_count=int(box.counts.min()),model_contains=contains))
                total_backups=0; total_pairs=0; total_gate_evals=0
                for k in [2,8,32]:
                    out=stopped_refresh(model,k,.03,initial=q0,model_box=box)
                    proposal=out.policy; new_true=true.evaluate(proposal)[0]
                    opt=true.optimal_budget(k)[0]
                    lo_old,hi_old,vlold,vuold=box.value_bands(pi=old)
                    lo_new,hi_new,vlnew,vunew=box.value_bands(pi=proposal)
                    from search_refresh.menus import box_advantage_lower
                    margins=box_advantage_lower(proposal,old,lo_old,hi_old)
                    unchanged=np.all(proposal==old,axis=-1); margins[unchanged]=0
                    advantage_accept=bool(np.all(margins>=0))
                    lower_difference=float(model.initial@(vlnew-vuold))
                    value_accept=lower_difference>=0
                    accept=advantage_accept or value_accept
                    false=bool(accept and new_true<old_true-1e-10)
                    false_near=bool(out.certified and opt-new_true>.03+1e-10)
                    if contains:
                        assert not false and not false_near
                        assert opt-new_true<=out.bound+1e-9
                    before=old_true
                    if accept: old=proposal.copy(); old_true=new_true
                    total_backups+=out.backup_calls; total_pairs+=out.menu_pairs; total_gate_evals+=2
                    # A deliberately unsafe plug-in-only gate for comparison.
                    plug_in_improvement=model.evaluate(proposal)[0]-model.evaluate(model.beta)[0]
                    rows.append(dict(family=family,n_per_state=n,seed=seed,k=k,
                        model_contains=contains,certified=out.certified,accepted=accept,
                        advantage_accept=advantage_accept,value_accept=value_accept,
                        false_accept=false,false_nearopt=false_near,updates=out.updates,
                        backups=out.backup_calls,bound=out.bound,proposal_value=new_true,
                        old_value=before,accepted_value=old_true,baseline=base,matched=opt,
                        gap=opt-new_true,value_lower_difference=lower_difference,
                        min_advantage=float(margins.min()),
                        cumulative_backups=total_backups,cumulative_menu_pairs=total_pairs,
                        cumulative_gate_policy_evaluations=total_gate_evals,
                        plugin_improvement=plug_in_improvement,
                        plugin_harm=bool(plug_in_improvement>=0 and new_true<base-1e-10)))
    save('offline_datasets',data_rows); save('offline_selection',rows)


def learned():
    rows=[]
    for taskseed in range(20):
        true=random_layered(taskseed+190000,horizon=5,width=3,actions=8)
        for dataseed in range(3):
            box=sample_model_box(true,5000,dataseed+1000*taskseed+195000)
            m=box.model; reg=LayerwiseRidge(m,box.counts,seed=taskseed)
            q0=reg.fit_predict(m.evaluate(m.beta)[2])
            for k in [2,8,32]:
                q=q0.copy(); opt=true.optimal_budget(k)[0]
                for step in range(m.H+1):
                    rb,gb,_,_,_=certificate(m,q,k,box)
                    val=true.return_of(q,k)
                    if box.contains(true): assert opt-val<=rb+1e-9
                    rows.append(dict(taskseed=taskseed,dataseed=dataseed,k=k,updates=step,
                                     value=val,matched=opt,gap=opt-val,rank_bound=rb,generic_bound=gb,
                                     empirical_full_value=true.return_of(m.optimal_budget(k)[2],k),
                                     baseline=true.evaluate(true.beta)[0]))
                    q=reg.fit_predict(m.backup(q,k))
    save('linear_critics',rows)


def statistical_limit():
    rows=[]
    for delta in [.01,.025,.05,.1,.2]:
        for n in [1,4,16,64,256,1024,4096]:
            # Optimal equal-prior likelihood test for Bernoulli(.5 +/- delta).
            # The safe action mean is .5 and exactly known.
            x=np.arange(n+1); pp=binom.pmf(x,n,.5+delta); pm=binom.pmf(x,n,.5-delta)
            bayes=.5*np.minimum(pp,pm).sum()
            kl=2*delta*np.log((.5+delta)/(.5-delta))
            for k in [1,2,4,8,32]:
                chi=1-2*(.5**k)
                lb=chi*delta*.5*max(0,1-np.sqrt(n*kl/2))
                actual=chi*delta*bayes
                assert actual>=lb-1e-12
                rows.append(dict(delta=delta,n=n,k=k,bayes_error=bayes,
                                 regret=actual,pinsker_lower=lb,constant_regime=n<=1/(32*delta**2)))
    save('statistical_limit',rows)


def summarize():
    get=lambda n:pd.read_csv(OUT/(n+'.csv'))
    geometry=get('geometry'); exact=get('exact_stopping'); off=get('offline_selection'); ds=get('offline_datasets')
    linear=get('linear_critics'); opening=get('opening_fork')
    pair=exact.pivot(index=['horizon','actions','seed','k'],columns='rule',values=['backups','operation_proxy','gap'])
    learned_pair=linear.pivot(index=['taskseed','dataseed','k'],columns='updates',values='value')
    summary={
        'geometry_instances':len(geometry),'menus_enumerated':int(geometry.menus_enumerated.sum()),
        'box_vertices_checked':int(geometry.vertices_checked.sum()),
        'max_enumeration_error':float(geometry.enumeration_error.max()),
        'max_envelope_to_robust_ratio':float(geometry.ratio.max()),
        'exact_mdp_budget_configs':len(pair),
        'rank_mean_backups':float(pair['backups']['rank'].mean()),
        'generic_mean_backups':float(pair['backups']['generic'].mean()),
        'rank_backup_reduction_vs_generic':float(1-pair['backups']['rank'].sum()/pair['backups']['generic'].sum()),
        'rank_operation_reduction_vs_generic':float(1-pair['operation_proxy']['rank'].sum()/pair['operation_proxy']['generic'].sum()),
        'rank_mean_gap':float(pair['gap']['rank'].mean()),'rank_max_gap':float(pair['gap']['rank'].max()),
        'offline_datasets':len(ds),'model_confidence_failures':int((~ds.model_contains).sum()),
        'offline_proposals':len(off),'accepted_proposals':int(off.accepted.sum()),
        'near_optimal_certificates':int(off.certified.sum()),
        'observed_false_accepts':int(off.false_accept.sum()),
        'observed_false_nearoptimal_certificates':int(off.false_nearopt.sum()),
        'plugin_harmful_proposals':int(off.plugin_harm.sum()),
        'linear_critic_configs':len(learned_pair),
        'linear_mean_refresh_gain':float((learned_pair[5]-learned_pair[0]).mean()),
        'linear_refresh_improved_fraction':float(((learned_pair[5]-learned_pair[0])>1e-10).mean()),
        'linear_refresh_harmed_fraction':float(((learned_pair[5]-learned_pair[0])<-1e-10).mean()),
    }
    if (OUT/'rare_action_control.csv').exists():
        rare=get('rare_action_control'); exposure=get('exposure_stopping')
        ep=exposure.pivot(index=['depth','good_prob','k','tolerance'],columns='rule',values=['backups','cost'])
        summary.update(rare_control_datasets=int(rare[['n_per_state','seed']].drop_duplicates().shape[0]),
            rare_control_proposals=len(rare),rare_harmful_proposals=int(rare.harmful_proposal.sum()),
            rare_accepted_proposals=int(rare.accepted.sum()),rare_false_accepts=int(rare.false_accept.sum()),
            exposure_configs=len(ep),exposure_rank_mean_backups=float(ep.backups['rank'].mean()),
            exposure_generic_mean_backups=float(ep.backups['generic'].mean()),
            exposure_backup_reduction=float(1-ep.backups['rank'].sum()/ep.backups['generic'].sum()),
            exposure_operation_reduction=float(1-ep.cost['rank'].sum()/ep.cost['generic'].sum()))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    (OUT/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.csv'))},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--suite',default='all',choices=['all','geometry','forks','exact','offline','learned','statistics','summary'])
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    functions={'geometry':geometry,'forks':forks,'exact':exact_stopping,'offline':offline,'learned':learned,'statistics':statistical_limit,'summary':summarize}
    for name,fn in functions.items():
        if args.suite in ('all',name):
            t=time.perf_counter(); fn(); TIMINGS[name]=time.perf_counter()-t
    (OUT/f'runtime_{args.suite}.json').write_text(json.dumps({'seconds':TIMINGS,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__},indent=2)+'\n')

if __name__=='__main__': main()
