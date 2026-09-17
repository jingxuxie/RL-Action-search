"""Exact best-of-K probabilities and candidate-exposure certificates.

Ties use a fixed priority: the larger action index wins. This affects only which
policy is executed, never the value of the expected maximum. No candidate menus
are sampled by the core experiments; small tests independently enumerate them.
"""
from __future__ import annotations
import itertools
import numpy as np


def _check(q, beta, k):
    q = np.asarray(q, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if q.shape != beta.shape or q.ndim not in (1, 2):
        raise ValueError("q and beta must have the same one- or two-dimensional shape")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(q).all() or not np.isfinite(beta).all():
        raise ValueError("nonfinite input")
    if np.min(beta) < 0 or not np.allclose(beta.sum(axis=-1), 1, atol=1e-12):
        raise ValueError("beta must be a probability distribution")
    return q, beta


def power_difference(x, y, k):
    """x**k-y**k, stabilized when x and y nearly coincide (0 <= y <= x <= 1)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    x, y = np.clip(x, 0, 1), np.clip(y, 0, 1)
    ratio = np.divide(y, x, out=np.zeros_like(x), where=x > 0)
    ratio = np.clip(ratio, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ans = x ** k * (-np.expm1(k * np.log(ratio)))
    return np.where(x == 0, 0.0, ans)


def selection_probabilities(q, beta, k):
    q, beta = _check(q, beta, k)
    order = np.argsort(q, axis=-1, kind="stable")
    bs = np.take_along_axis(beta, order, axis=-1)
    f = np.cumsum(bs, axis=-1)
    f[..., -1] = 1.0
    before = np.concatenate([np.zeros_like(f[..., :1]), f[..., :-1]], axis=-1)
    mass = power_difference(f, before, k)
    out = np.empty_like(mass)
    np.put_along_axis(out, order, mass, axis=-1)
    # Correct only floating point roundoff; tests also check the raw formula.
    out /= out.sum(axis=-1, keepdims=True)
    return out


def expected_max(q, beta, k):
    return np.sum(selection_probabilities(q, beta, k) * np.asarray(q), axis=-1)


def menu_envelope(q, beta, k, lower, upper):
    """E_C sup_{v in [lower,upper]} (max_{a in C} v_a-v_{I_q(C)}).

    This is the exact expectation of the *menu-wise* box envelope. It is an
    upper bound, NOT generally the exact sup_v E_C of fixed-vector regret.
    Complexity O(A^2 log A) per state, memory O(A). Also returns the probability
    of strictly positive menu-wise ambiguity.
    """
    q, beta = _check(q, beta, k)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != q.shape or hi.shape != q.shape or np.any(lo > hi + 1e-12):
        raise ValueError("invalid interval box")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("nonfinite interval")
    if q.ndim == 2:
        rows = [menu_envelope(qi, bi, k, li, ui) for qi, bi, li, ui in zip(q, beta, lo, hi)]
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    order = np.argsort(q, kind="stable")
    b, l, u = beta[order], lo[order], hi[order]
    f = np.cumsum(b)
    f[-1] = 1.0
    value = ambiguity = 0.0
    if k == 1:
        return 0.0, 0.0
    for i in range(1, len(q)):
        if b[i] == 0:
            continue
        fi, prev = f[i], f[i-1]
        wi = float(power_difference(fi, prev, k))
        # At threshold t, event = i is selected and some lower-ranked j has
        # u_j-l_i > t. Its probability follows from exclusion of that set.
        gaps = np.maximum(u[:i] - l[i], 0.0)
        perm = np.argsort(gaps, kind="stable")
        positive = [int(j) for j in perm if gaps[j] > 0 and b[j] > 0]
        if not positive:
            continue
        threat_mass = float(b[positive].sum())
        last = 0.0
        for index, j in enumerate(positive):
            event = wi - float(power_difference(fi-threat_mass, prev-threat_mass, k))
            event = max(0.0, min(wi, event))
            if index == 0:
                ambiguity += event
            value += (gaps[j]-last) * event
            last = gaps[j]
            threat_mass = max(0.0, threat_mass-b[j])
    return float(value), float(min(1.0, ambiguity))


def enumerate_menu_statistics(q, beta, k, lower=None, upper=None, true_values=None):
    """Independent exponential-time reference implementation for tiny tests."""
    q, beta = _check(q, beta, k)
    if q.ndim != 1:
        raise ValueError("enumeration only supports a single state")
    p = np.zeros_like(q)
    envelope = ambiguity = regret = 0.0
    for menu in itertools.product(range(len(q)), repeat=k):
        prob = float(np.prod(beta[list(menu)]))
        chosen = max(menu, key=lambda a: (q[a], a))
        p[chosen] += prob
        if lower is not None:
            alternatives = [a for a in set(menu) if a != chosen]
            loss = max([0.0] + [upper[a]-lower[chosen] for a in alternatives])
            envelope += prob * loss
            ambiguity += prob * (loss > 0)
        if true_values is not None:
            regret += prob * (max(true_values[a] for a in menu)-true_values[chosen])
    return p, envelope, ambiguity, regret


def box_advantage_lower(new_policy, old_policy, lower, upper):
    """Exact min_{v in box} <new_policy-old_policy, v>, separately per state."""
    d = np.asarray(new_policy)-np.asarray(old_policy)
    return np.sum(np.where(d >= 0, d*np.asarray(lower), d*np.asarray(upper)), axis=-1)


def fixed_box_worst_regret(q,beta,k,lower,upper):
    """Exact sup_v E_C regret by vertex enumeration; for small A only.
    The objective is convex in v. This is a verification baseline, not the
    polynomial-time deployed certificate.
    """
    q,beta=_check(q,beta,k)
    if q.ndim!=1 or len(q)>20: raise ValueError("use <=20 actions in one state")
    pi=selection_probabilities(q,beta,k); best=-np.inf; best_v=None
    for bits in itertools.product((0,1),repeat=len(q)):
        v=np.where(bits,upper,lower)
        val=float(expected_max(v,beta,k)-pi@v)
        if val>best: best,best_v=val,v.copy()
    return max(0.0,best),best_v
