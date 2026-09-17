"""Finite-horizon MDPs with a terminal state and known time-layer support."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .menus import expected_max, selection_probabilities


@dataclass
class LayeredMDP:
    reward: np.ndarray          # S x A, expected bounded rewards
    transition: np.ndarray      # S x A x (S+1), last index terminal
    beta: np.ndarray            # S x A, fixed proposal
    level: np.ndarray           # S, integer time layer
    initial: np.ndarray         # S, initial distribution (layer zero)
    gamma: float = 1.0

    def __post_init__(self):
        self.reward = np.asarray(self.reward, float)
        self.transition = np.asarray(self.transition, float)
        self.beta = np.asarray(self.beta, float)
        self.level = np.asarray(self.level, int)
        self.initial = np.asarray(self.initial, float)
        s, a = self.reward.shape
        if self.transition.shape != (s,a,s+1) or self.beta.shape != (s,a):
            raise ValueError("incompatible arrays")
        if self.level.shape != (s,) or self.initial.shape != (s,):
            raise ValueError("incompatible state arrays")
        if not 0 < self.gamma <= 1 or np.min(self.level) != 0:
            raise ValueError("invalid discount or levels")
        if np.min(self.transition)<0 or not np.allclose(self.transition.sum(-1),1):
            raise ValueError("invalid transition distribution")
        if np.min(self.beta)<0 or not np.allclose(self.beta.sum(-1),1):
            raise ValueError("invalid proposal")
        if np.min(self.initial)<0 or not np.isclose(self.initial.sum(),1):
            raise ValueError("invalid initial distribution")
        if np.any(self.initial[self.level != 0] > 0):
            raise ValueError("initial states must be in layer zero")
        for state in range(s):
            targets = np.flatnonzero(self.transition[state,:,:s].sum(0)>0)
            if np.any(self.level[targets] != self.level[state]+1):
                raise ValueError("nonterminal transitions must advance exactly one layer")
        if not np.isfinite(self.reward).all():
            raise ValueError("nonfinite rewards")

    @property
    def S(self): return self.reward.shape[0]
    @property
    def A(self): return self.reward.shape[1]
    @property
    def H(self): return int(self.level.max())+1
    @property
    def edges(self): return int(np.count_nonzero(self.transition))

    def backup(self, q, k):
        v = np.r_[expected_max(q, self.beta, k), 0.0]
        return self.reward+self.gamma*np.einsum('saj,j->sa', self.transition,v)

    def evaluate(self, pi):
        pi=np.asarray(pi,float)
        if pi.shape != self.beta.shape or np.min(pi)<-1e-12 or not np.allclose(pi.sum(-1),1):
            raise ValueError("invalid policy")
        v=np.zeros(self.S+1); q=np.zeros_like(self.reward)
        for h in range(self.H-1,-1,-1):
            states=np.flatnonzero(self.level==h)
            q[states]=self.reward[states]+self.gamma*np.einsum('saj,j->sa', self.transition[states],v)
            v[states]=np.sum(pi[states]*q[states],axis=-1)
        return float(self.initial@v[:-1]), v[:-1], q

    def optimal_budget(self,k):
        v=np.zeros(self.S+1); q=np.zeros_like(self.reward)
        for h in range(self.H-1,-1,-1):
            states=np.flatnonzero(self.level==h)
            q[states]=self.reward[states]+self.gamma*np.einsum('saj,j->sa',self.transition[states],v)
            v[states]=expected_max(q[states],self.beta[states],k)
        return float(self.initial@v[:-1]), v[:-1], q

    def optimal_full_action(self):
        v=np.zeros(self.S+1); q=np.zeros_like(self.reward)
        for h in range(self.H-1,-1,-1):
            states=np.flatnonzero(self.level==h)
            q[states]=self.reward[states]+self.gamma*np.einsum('saj,j->sa',self.transition[states],v)
            v[states]=q[states].max(-1)
        pi=np.eye(self.A)[np.argmax(q,axis=-1)]
        return float(self.initial@v[:-1]),pi,q

    def return_of(self,q,k):
        return self.evaluate(selection_probabilities(q,self.beta,k))[0]

    def propagate_loss(self,loss,pi):
        v=np.zeros(self.S+1)
        for h in range(self.H-1,-1,-1):
            states=np.flatnonzero(self.level==h)
            trans=np.einsum('sa,saj->sj',pi[states],self.transition[states])
            v[states]=loss[states]+self.gamma*trans@v
        return float(self.initial@v[:-1])


def delayed_fork(depth=1, safe=0.2, good=1.0, good_prob=0.1,
                 safe_prob=0.1, gamma=1.0):
    """Root followed by depth transitions to a final two-action state.
    Internal states have only one supported action. Rewards may be >1 for the
    illustrative 2-versus-10 example, but confidence experiments use [0,1].
    """
    if depth<1: raise ValueError("depth must be >=1")
    s=depth+1; r=np.zeros((s,2)); p=np.zeros((s,2,s+1))
    b=np.tile([1.,0.],(s,1)); b[0]=[safe_prob,1-safe_prob]; b[-1]=[1-good_prob,good_prob]
    r[0,0]=safe; p[0,0,-1]=1; p[0,1,1]=1
    for i in range(1,depth): p[i,:,i+1]=1
    r[-1,1]=good; p[-1,:,-1]=1
    initial=np.zeros(s); initial[0]=1
    return LayeredMDP(r,p,b,np.arange(s),initial,gamma)


def random_layered(seed=0,horizon=5,width=3,actions=8,gamma=1.0,
                   terminal_only=True, proposal_skew=0.35):
    """Stochastic action-rich graph. Every reward distribution is Bernoulli."""
    rng=np.random.default_rng(seed); s=horizon*width
    level=np.repeat(np.arange(horizon),width)
    r=rng.uniform(0.05,0.95,(s,actions))
    if terminal_only: r[level<horizon-1]=0
    p=np.zeros((s,actions,s+1))
    for i in range(s):
        if level[i]==horizon-1:
            p[i,:,-1]=1
        else:
            targets=np.r_[np.flatnonzero(level==level[i]+1),s]
            # Some early termination, but no hidden transition support supplied.
            for a in range(actions):
                p[i,a,targets]=rng.dirichlet(np.r_[np.full(width,0.6),0.12])
    beta=rng.dirichlet(np.full(actions,proposal_skew),size=s)
    # Avoid exact numerical undercoverage without assuming uniform coverage.
    beta=0.96*beta+0.04/actions
    initial=np.zeros(s); initial[:width]=1/width
    return LayeredMDP(r,p,beta,level,initial,gamma)


def rare_action_control(actions=8,good_reward=.6,bad_reward=.5,good_mass=.9):
    """One-step control: no continuation mismatch, only estimation error."""
    if actions<2: raise ValueError('at least two actions')
    r=np.full((1,actions),bad_reward); r[0,0]=good_reward
    p=np.zeros((1,actions,2)); p[0,:,1]=1
    b=np.full((1,actions),(1-good_mass)/(actions-1)); b[0,0]=good_mass
    return LayeredMDP(r,p,b,np.array([0]),np.array([1.]))
