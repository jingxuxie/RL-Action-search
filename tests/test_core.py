import itertools
import numpy as np
import pytest
from scipy.optimize import linprog
from search_refresh.menus import (selection_probabilities,expected_max,menu_envelope,
    enumerate_menu_statistics,fixed_box_worst_regret,box_advantage_lower)
from search_refresh.mdp import delayed_fork,random_layered
from search_refresh.confidence import extremal_expectation,sample_model_box
from search_refresh.refresh import stopped_refresh,certificate,residual_bands

@pytest.mark.parametrize('seed',range(10))
@pytest.mark.parametrize('k',[1,2,3,4,8])
def test_selection(seed,k):
    rng=np.random.default_rng(seed); q=rng.integers(-2,3,3).astype(float)
    b=rng.dirichlet(np.ones(3)); exact=enumerate_menu_statistics(q,b,k)[0]
    np.testing.assert_allclose(selection_probabilities(q,b,k),exact,atol=3e-12)

@pytest.mark.parametrize('seed',range(20))
@pytest.mark.parametrize('k',[1,2,3,5])
def test_menu_envelope(seed,k):
    rng=np.random.default_rng(seed); q=rng.normal(size=4); b=rng.dirichlet(np.ones(4))
    lo=rng.normal(size=4); hi=lo+rng.uniform(0,2,4)
    fast,amb=menu_envelope(q,b,k,lo,hi)
    _,slow,amb_slow,_=enumerate_menu_statistics(q,b,k,lo,hi)
    np.testing.assert_allclose([fast,amb],[slow,amb_slow],atol=2e-12)
    robust,_=fixed_box_worst_regret(q,b,k,lo,hi)
    assert robust<=fast+2e-12
    assert fast<=4*robust+2e-12
    shifted=menu_envelope(q+11,b,k,lo+7,hi+7)
    np.testing.assert_allclose(shifted,[fast,amb],atol=1e-12)

@pytest.mark.parametrize('seed',range(8))
@pytest.mark.parametrize('k',[1,2,4,16,10000])
def test_two_action_exact(seed,k):
    rng=np.random.default_rng(seed); p=rng.uniform(.001,.999)
    q=np.array([0.,1.]); lo=rng.normal(size=2); hi=lo+rng.uniform(0,2,2)
    loss=max(0,hi[0]-lo[1])*(1-p**k-(1-p)**k)
    out=menu_envelope(q,np.array([p,1-p]),k,lo,hi)[0]
    np.testing.assert_allclose(out,loss,atol=5e-12)

@pytest.mark.parametrize('depth',[1,2,3,5,10])
@pytest.mark.parametrize('k',[3,4,8,16,32])
def test_delayed_exact(depth,k):
    mdp=delayed_fork(depth=depth); q=mdp.evaluate(mdp.beta)[2]
    high=1-.9**k; c=.2
    frozen=c*(1-.9**k)+.9**k*high
    optimal=c*.1**k+(1-.1**k)*high
    for m in range(depth+2):
        np.testing.assert_allclose(mdp.return_of(q,k),frozen if m<depth else optimal,atol=2e-12)
        q=mdp.backup(q,k)
    loss=(1-.1**k-.9**k)*(high-c)
    np.testing.assert_allclose(optimal-frozen,loss,atol=1e-12)

@pytest.mark.parametrize('seed',range(20))
def test_tv_lp(seed):
    rng=np.random.default_rng(seed); d=5; p=rng.dirichlet(np.ones(d)); v=rng.normal(size=d); tv=rng.random()
    # Variables new distribution x, absolute deviations z.
    c=np.r_[-v,np.zeros(d)]; eye=np.eye(d)
    aub=np.block([[eye,-eye],[-eye,-eye],[np.zeros((1,d)),np.ones((1,d))]])
    bub=np.r_[p,-p,2*tv]
    sol=linprog(c,A_ub=aub,b_ub=bub,A_eq=np.r_[np.ones(d),np.zeros(d)][None,:],b_eq=[1],bounds=(0,None),method='highs')
    assert sol.success
    np.testing.assert_allclose(extremal_expectation(p,v,tv),-sol.fun,atol=1e-9)

@pytest.mark.parametrize('seed',range(15))
@pytest.mark.parametrize('k',[1,4,16])
def test_certificates(seed,k):
    mdp=random_layered(seed,horizon=4,width=2,actions=4)
    _,_,qstar=mdp.optimal_budget(k); opt=mdp.return_of(qstar,k)
    q=mdp.evaluate(mdp.beta)[2]
    for m in range(mdp.H+1):
        rb,gb,env,_,_=certificate(mdp,q,k)
        lo,hi,_=residual_bands(mdp,q,k)
        assert np.all(qstar>=lo-1e-10) and np.all(qstar<=hi+1e-10)
        pi=selection_probabilities(q,mdp.beta,k)
        loss=expected_max(qstar,mdp.beta,k)-np.sum(pi*qstar,axis=-1)
        actual=opt-mdp.return_of(q,k)
        np.testing.assert_allclose(mdp.propagate_loss(loss,pi),actual,atol=2e-10)
        assert actual<=rb+2e-10 and rb<=gb+2e-10
        q=mdp.backup(q,k)

@pytest.mark.parametrize('seed',range(12))
def test_finite_model(seed):
    mdp=random_layered(seed,horizon=3,width=2,actions=3)
    box=sample_model_box(mdp,10000,seed+1000)
    if not box.contains(mdp): pytest.skip('Legitimate confidence event failure')
    for k in [1,4,16]:
        _,_,trueq=mdp.optimal_budget(k)
        lo,hi,_,_=box.value_bands(k=k)
        assert np.all(trueq>=lo-1e-10) and np.all(trueq<=hi+1e-10)
        q=box.model.evaluate(box.model.beta)[2]
        for _ in range(mdp.H+1):
            rb,_,_,_,_=certificate(box.model,q,k,box)
            truegap=mdp.optimal_budget(k)[0]-mdp.return_of(q,k)
            assert truegap<=rb+1e-10
            q=box.model.backup(q,k)
        new=selection_probabilities(q,mdp.beta,k)
        accept,margin=box.safe_gate(new,mdp.beta)
        if accept: assert mdp.evaluate(new)[0]>=mdp.evaluate(mdp.beta)[0]-1e-10

@pytest.mark.parametrize('seed',range(15))
def test_box_advantage_exact(seed):
    rng=np.random.default_rng(seed); a=4
    pi,pj=rng.dirichlet(np.ones(a),size=2); lo=rng.normal(size=a); hi=lo+rng.random(a)
    brute=min((pi-pj)@np.where(bits,hi,lo) for bits in itertools.product((0,1),repeat=a))
    np.testing.assert_allclose(box_advantage_lower(pi,pj,lo,hi),brute,atol=1e-12)

@pytest.mark.parametrize('seed',range(10))
def test_stopper(seed):
    m=random_layered(seed,horizon=5,width=2,actions=4)
    out=stopped_refresh(m,8,.03)
    assert out.certified
    assert out.backup_calls==out.updates+1
    assert out.diagnostic_calls==len(out.trace)
    gap=m.optimal_budget(8)[0]-m.evaluate(out.policy)[0]
    assert gap<=out.bound+1e-10


def test_interval_quantifiers_not_equal():
    q=np.arange(3.); b=np.ones(3)/3
    g=menu_envelope(q,b,3,np.zeros(3),np.ones(3))[0]
    r,_=fixed_box_worst_regret(q,b,3,np.zeros(3),np.ones(3))
    np.testing.assert_allclose([g,r],[8/9,2/3],atol=1e-12)


def test_single_candidate_and_absent_actions():
    q=np.array([0.,1.,2.]); b=np.array([1.,0.,0.])
    for k in [1,2,100000]:
        np.testing.assert_equal(selection_probabilities(q,b,k),[1,0,0])
        assert menu_envelope(q,b,k,q-100,q+100)==(0.,0.)

@pytest.mark.parametrize('k',[0,-1,1.5])
def test_bad_k(k):
    with pytest.raises(ValueError): selection_probabilities([0,1],[.5,.5],k)


def test_empty_rows_confidence():
    m=delayed_fork(depth=2)
    b=sample_model_box(m,1,123)
    assert np.any(b.counts==0)
    assert np.isfinite(b.value_bands(k=4)[0]).all()
    assert b.safe_gate(m.beta,m.beta)[0]
