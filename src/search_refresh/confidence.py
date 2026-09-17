"""One simultaneous rectangular model confidence event, reusable adaptively.

Data are stratified independent one-step samples, NOT dependent trajectories.
Reward samples are Bernoulli. Transition support (time layer plus termination)
is known; the transition probabilities need not be. Exact binomial intervals
and a finite-alphabet TV concentration bound are used. No ensemble is called a
confidence interval. Safe acceptance does not use the true MDP.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta as beta_dist
from .mdp import LayeredMDP
from .menus import expected_max, box_advantage_lower


def extremal_expectation(prob, values, tv, maximize=True):
    """Exact extremum on a probability simplex intersected with a TV ball.

    `prob` and `values` are restricted to the known allowed support. Movement
    of mass tv from low-valued to high-valued coordinates solves this LP.
    """
    prob=np.asarray(prob,float); values=np.asarray(values,float)
    if prob.shape!=values.shape or prob.ndim!=1 or not np.isclose(prob.sum(),1):
        raise ValueError("invalid simplex")
    if not 0<=tv<=1+1e-12: raise ValueError("TV must lie in [0,1]")
    if not maximize: return -extremal_expectation(prob,-values,tv,True)
    order=np.argsort(values,kind='stable'); p=prob[order].copy(); v=values[order]
    left,right=0,len(p)-1; remaining=min(float(tv),1.0)
    while left<right and remaining>1e-15:
        if p[left]<=1e-15: left+=1; continue
        if p[right]>=1-1e-15: right-=1; continue
        move=min(remaining,p[left],1-p[right])
        p[left]-=move; p[right]+=move; remaining-=move
    return float(p@v)


@dataclass
class ModelBox:
    model: LayeredMDP
    rlo: np.ndarray
    rhi: np.ndarray
    tv: np.ndarray
    support: np.ndarray
    counts: np.ndarray
    delta: float

    def transition_extreme(self,s,a,values,maximize):
        use=self.support[s,a]
        return extremal_expectation(self.model.transition[s,a,use],values[use],
                                    float(self.tv[s,a]),maximize)

    def value_bands(self, pi=None, k=None):
        """Robust backward DP, valid for any data-adaptively chosen policy/K.

        Exactly one of pi/k is passed. k gives bounds on matched-budget Q_K;
        pi gives bounds on Q^pi. All policy and model optimization occurs on
        the confidence set, not the unknown true MDP.
        """
        if (pi is None)==(k is None): raise ValueError("pass pi or k, exclusively")
        m=self.model; vl=np.zeros(m.S+1); vu=np.zeros(m.S+1)
        ql=np.zeros_like(m.reward); qu=np.zeros_like(m.reward)
        for h in range(m.H-1,-1,-1):
            states=np.flatnonzero(m.level==h)
            for s in states:
                for a in range(m.A):
                    ql[s,a]=self.rlo[s,a]+m.gamma*self.transition_extreme(s,a,vl,False)
                    qu[s,a]=self.rhi[s,a]+m.gamma*self.transition_extreme(s,a,vu,True)
            if pi is not None:
                vl[states]=(pi[states]*ql[states]).sum(-1)
                vu[states]=(pi[states]*qu[states]).sum(-1)
            else:
                vl[states]=expected_max(ql[states],m.beta[states],k)
                vu[states]=expected_max(qu[states],m.beta[states],k)
        return ql,qu,vl[:-1],vu[:-1]

    def backup_errors(self,q,k):
        """Uniform model event implies this bound even for data-dependent q.
        Uses the actual span of F_K(q) on each row's allowed support.
        """
        m=self.model; v=np.r_[expected_max(q,m.beta,k),0.0]
        reward_error=np.maximum(m.reward-self.rlo,self.rhi-m.reward)
        err=reward_error.copy()
        for s in range(m.S):
            for a in range(m.A):
                vals=v[self.support[s,a]]
                err[s,a]+=m.gamma*self.tv[s,a]*np.ptp(vals)
        return err

    def contains(self,true):
        in_rewards=np.all(true.reward>=self.rlo-1e-12) and np.all(true.reward<=self.rhi+1e-12)
        distances=0.5*np.abs(true.transition-self.model.transition).sum(-1)
        return bool(in_rewards and np.all(distances<=self.tv+1e-12)
                    and np.all(true.transition[~self.support]==0))

    def safe_gate(self,new_policy,old_policy):
        """All-state advantage gate. Sufficiency only; rejection != harm.
        Returns certificate and raw per-state margins, without oracle access.
        """
        lo,hi,_,_=self.value_bands(pi=old_policy)
        margins=box_advantage_lower(new_policy,old_policy,lo,hi)
        # Identical rows imply an algebraically exact zero margin.
        same=np.all(new_policy==old_policy,axis=-1)
        margins[same]=0.0
        # Strictly nonnegative acceptance; no tolerance-based false claim.
        return bool(np.all(margins>=0.0)),margins


def sample_model_box(true, n_per_state, seed, delta=0.05,
                     known_transitions=False, uniform_allocation=False):
    """Generate a fixed offline dataset in sufficient-statistic form.

    Each state has a deterministic budget proportional to beta, unless uniform
    allocation is requested. Counts do not depend on observed outcomes. The
    data contain independent reward/transition draws at each state-action.
    Terminal-only/zero rewards are *not* revealed to the learner.
    """
    if not np.all((true.reward>=0)&(true.reward<=1)):
        raise ValueError("Bernoulli experiments require rewards in [0,1]")
    if n_per_state<1 or not 0<delta<1: raise ValueError("invalid sample size/delta")
    rng=np.random.default_rng(seed); s,a=true.reward.shape; M=s*a
    allocation=np.full_like(true.beta,1/a) if uniform_allocation else true.beta
    counts=np.floor(n_per_state*allocation).astype(int)
    # Every state has at most the advertised budget; unsupported actions may
    # receive zero samples. Floor loss is reported through actual counts.
    rh=np.zeros((s,a)); lo=np.zeros((s,a)); hi=np.ones((s,a))
    ph=np.zeros_like(true.transition); tv=np.zeros((s,a)); support=np.zeros_like(ph,dtype=bool)
    alpha=delta/(2*M)
    for i in range(s):
        targets=np.r_[np.flatnonzero(true.level==true.level[i]+1),s]
        for j in range(a):
            if known_transitions:
                targets_row=np.flatnonzero(true.transition[i,j]>0)
            else:
                targets_row=targets
            support[i,j,targets_row]=True
            n=int(counts[i,j]); success=int(rng.binomial(n,true.reward[i,j]))
            rh[i,j]=success/n if n else 0.5
            if n:
                lo[i,j]=0 if success==0 else beta_dist.ppf(alpha/2,success,n-success+1)
                hi[i,j]=1 if success==n else beta_dist.ppf(1-alpha/2,success+1,n-success)
            d=len(targets_row)
            if known_transitions:
                ph[i,j]=true.transition[i,j]
            elif n:
                trans=rng.multinomial(n,true.transition[i,j,targets_row])
                ph[i,j,targets_row]=trans/n
                tv[i,j]=0 if d==1 else min(1.,np.sqrt((d*np.log(2)+np.log(2*M/delta))/(2*n)))
            else:
                ph[i,j,targets_row]=1/d
                tv[i,j]=0 if d==1 else 1.0
    model=LayeredMDP(rh,ph,true.beta.copy(),true.level.copy(),true.initial.copy(),true.gamma)
    return ModelBox(model,lo,hi,tv,support,counts,delta)
