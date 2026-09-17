#!/usr/bin/env python3
"""Create separate, default-color matplotlib figures from recorded CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)

def get(name): return pd.read_csv(ROOT/'results'/f'{name}.csv')
def finish(fig,name):
    fig.tight_layout()
    for ext in ['pdf','png','svg']:
        fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    x=get('opening_fork'); fig,ax=plt.subplots(figsize=(6.1,3.5))
    ax.plot(x.k,x.frozen,label='Frozen exact baseline critic',linewidth=2)
    ax.plot(x.k,x.refreshed,label='One matched backup',linewidth=2)
    ax.set(xlabel='Candidate budget K',ylabel='Expected return',xscale='log',title='Search can harm an exact reference-policy critic')
    ax.legend(frameon=False,fontsize=9); finish(fig,'opening_fork')

    x=get('forks'); x=x[(x.depth==8)&(x.goodp==.1)&(x.safep==.1)]
    z=x.pivot(index='updates',columns='k',values='value')
    fig,ax=plt.subplots(figsize=(6.1,3.5))
    im=ax.imshow(z.values,origin='lower',aspect='auto')
    ax.set_xticks(range(len(z.columns)),labels=z.columns)
    ax.set_yticks(range(len(z.index)),labels=z.index)
    ax.set(xlabel='Candidate budget K',ylabel='Synchronous refresh sweeps',title='Eight-step propagation delay')
    fig.colorbar(im,ax=ax,label='Expected return'); finish(fig,'refresh_heatmap')

    x=get('geometry'); x=x[x.robust>1e-10]
    fig,ax=plt.subplots(figsize=(5.4,3.5)); ax.scatter(x.robust,x.envelope,s=13,alpha=.65)
    lim=max(x.envelope.max(),x.robust.max())
    ax.plot([0,lim],[0,lim],linestyle='--',label='Exact equality')
    ax.set(xlabel='Exact fixed-vector worst regret',ylabel='Menu-wise envelope',title='400 interval geometries; exhaustive corner checks')
    ax.legend(frameon=False,fontsize=9); finish(fig,'geometry')

    x=get('screening'); fig,ax=plt.subplots(figsize=(5.6,3.2))
    ax.plot(x.k,x.loss,linewidth=2)
    ax.set(xlabel='Candidate budget K',ylabel='One-state ranking regret',title='More search can screen a low-ranked inversion')
    finish(fig,'screening')

    x=get('exact_stopping'); y=get('exposure_stopping')
    fig,ax=plt.subplots(figsize=(6.2,3.6))
    for index,(name,df) in enumerate([('Random graphs',x),('Rare downstream choices',y)]):
        vals=df.groupby('rule').backups.mean()
        pos=np.array([index*3,index*3+1])
        ax.bar(pos,[vals['generic'],vals['rank']])
    ax.set_xticks([0,1,3,4],['Generic\nrandom','Exposure\nrandom','Generic\nrare','Exposure\nrare'])
    ax.set(ylabel='Mean Bellman backup calls',title='Savings are strongly instance-dependent')
    finish(fig,'backup_counts')

    fig,ax=plt.subplots(figsize=(6.2,3.6))
    vals=[]
    for df,col in [(x,'operation_proxy'),(y,'cost')]:
        v=df.groupby('rule')[col].sum(); vals += [1.,v['rank']/v['generic']]
    ax.bar([0,1,3,4],vals)
    ax.axhline(1,linestyle='--',linewidth=1)
    ax.set_xticks([0,1,3,4],['Generic\nrandom','Exposure\nrandom','Generic\nrare','Exposure\nrare'])
    ax.set(ylabel='Operation proxy relative to generic stopping',title='Count diagnostic work, not only saved sweeps')
    finish(fig,'operation_counts')

    x=get('rare_action_control'); x=x[x.k==32]
    fig,ax=plt.subplots(figsize=(6.0,3.6))
    for col,label in [('proposal_value','Ungated point-critic extraction'),('accepted_value','Conservative retained policy')]:
        g=x.groupby('n_per_state')[col]; mu=g.mean(); se=g.std()/np.sqrt(g.count())
        ax.errorbar(mu.index,mu.values,yerr=1.96*se.values,label=label,marker='o',capsize=3)
    ax.axhline(.59,linestyle='--',label='Behavior baseline')
    ax.set(xscale='log',xlabel='Offline samples allocated per state',ylabel='Expected return',title='Estimation-only control (K = 32)')
    ax.legend(frameon=False,fontsize=8); finish(fig,'rare_data')

    x=get('offline_selection'); x=x[(x.k==32)&(x.family=='stochastic5')]
    fig,ax=plt.subplots(figsize=(6,3.6))
    for col,label in [('proposal_value','Proposed refreshed policy'),('accepted_value','Retained certified policy'),('baseline','Behavior baseline')]:
        g=x.groupby('n_per_state')[col]; mu=g.mean(); se=g.std()/np.sqrt(g.count())
        ax.errorbar(mu.index,mu.values,yerr=1.96*se.values,label=label,marker='o',capsize=3)
    ax.set(xscale='log',xlabel='Offline samples allocated per state',ylabel='Expected return',title='Conservatism on stochastic five-layer graphs')
    ax.legend(frameon=False,fontsize=8); finish(fig,'offline_returns')

    x=get('linear_critics'); p=x.pivot(index=['taskseed','dataseed','k'],columns='updates',values='value')
    gain=p[5]-p[0]
    fig,ax=plt.subplots(figsize=(5.8,3.4)); ax.hist(gain.values,bins=22)
    ax.axvline(0,linestyle='--',linewidth=1)
    ax.set(xlabel='Return after five fitted sweeps minus frozen return',ylabel='Configuration count',title='Projected critic refresh is not uniformly beneficial')
    finish(fig,'linear_gains')
    print(f'Wrote 9 separate figures to {FIG}')

if __name__=='__main__': main()
