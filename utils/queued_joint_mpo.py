"""Finite-support joint improvement and public short-window queue forecasts.

No workload identifier, hidden geometry, future tape or complete Engine is
accepted by online forecasting. Action indices are local to each saved bank.
"""
import copy
import math
import numpy as np
from environment.queued_service import Engine, Task, distance
from utils.queued_service_learning import state_features
from utils.queued_service_controllers import Controller, assignments


def softmax(logits, axis=-1):
    x = np.asarray(logits, dtype=np.float64)
    z = np.exp(x-np.max(x, axis=axis, keepdims=True))
    return z/z.sum(axis=axis, keepdims=True)


def divergence(p, q):
    p, q = np.asarray(p), np.asarray(q)
    active = p > 0
    if np.any(q[active] <= 0): return float('inf')
    return float(np.sum(p[active]*(np.log(p[active])-np.log(q[active]))))


def improve(probabilities, values, budget=.1):
    """One global dual temperature for mean-state KL(q||old) <= budget."""
    if budget < 0 or not probabilities: raise ValueError('nonnegative budget and nonempty states required')
    ps = [np.asarray(p, dtype=np.float64) for p in probabilities]
    vs = [np.asarray(v, dtype=np.float64) for v in values]
    if len(ps) != len(vs): raise ValueError('one value matrix per state')
    for p, v in zip(ps, vs):
        if p.shape != v.shape or np.any(p < 0) or not np.isclose(p.sum(), 1) or not np.all(np.isfinite(v)):
            raise ValueError('invalid support/value matrix')
    centered = [v-np.max(v[p > 0]) for p, v in zip(ps, vs)]
    def at(eta):
        result = []
        for p, v in zip(ps, centered):
            q = np.zeros_like(p); active = p > 0
            q[active] = softmax(np.log(p[active])+v[active]/eta)
            result.append(q)
        return result, float(np.mean([divergence(q, p) for q, p in zip(result, ps)]))
    scale = max([float(np.ptp(v[p > 0])) for p, v in zip(ps, vs)]+[1e-12])
    if budget == 0 or scale <= 1e-12:
        return [p.copy() for p in ps], dict(eta=None, kl=0., gain=0.)
    lo, hi = scale*1e-12, scale
    while at(hi)[1] > budget: hi *= 2
    if at(lo)[1] <= budget: hi = lo
    else:
        for _ in range(60):
            mid = (lo+hi)/2
            if at(mid)[1] > budget: lo = mid
            else: hi = mid
    qs, kl = at(hi)
    gain = float(np.mean([np.sum((q-p)*v) for q, p, v in zip(qs, ps, vs)]))
    if gain < -1e-8 or kl > budget+1e-8: raise AssertionError('E-step contract')
    return qs, dict(eta=hi, kl=kl, gain=gain)


def unique_plans(plans):
    seen, result = set(), []
    for plan in plans:
        key = tuple(sorted(plan.items()))
        if key not in seen: seen.add(key); result.append(dict(key))
    return result


def restricted_prior(log_masses, exploration=.1):
    """Normalize exact unique plan masses once, then explicit support mixture.

    This defines a NEW finite-bank policy, not an unbiased full-actor update.
    The bank's retained full-actor mass is reported independently.
    """
    x = np.asarray(log_masses, dtype=float)
    if not len(x) or np.any(~np.isfinite(x)) or not 0 <= exploration <= 1: raise ValueError('invalid masses')
    z = float(x.max()+np.log(np.exp(x-x.max()).sum()))
    return (1-exploration)*softmax(x)+exploration/len(x), float(np.exp(z))


def value_features(obs, history):
    """Task-count value features, including spatial demand and time basis."""
    now = obs['time']; weights = history.weights
    result = [1., max(0., 200-now)/200, max(0., 100-now)/100, max(0., 50-now)/50]
    result += state_features(obs, weights)
    for a in obs['positions']:
        for radius in (100., 200., 460.):
            result.append(sum(weights.get(u, 0.)*math.exp(-distance(xy, a)**2/(2*radius**2)) for u, xy in obs['ues'].items())/100)
    # The finite horizon is known. No value beyond the actual empty terminal.
    return np.asarray(result, dtype=np.float64)


def arrival_prediction(obs, history, seed, horizon=3):
    """One common public-history plug-in forecast; no latent phase/hotspot.

    EWMA newly observed task mass / 20 estimates rate. Fixed size/cycle/SLA
    support is the frozen business prior. Visibility/cold-start bias is retained
    and tested; future arrivals after the known t=200 cutoff are zero.
    """
    rng = np.random.default_rng(seed); ids = sorted(obs['ues'])
    weights = np.asarray([history.weights.get(u, 0.) for u in ids], dtype=float)
    rate = float(weights.sum()/20)
    probs = weights/weights.sum() if weights.sum() else np.full(len(ids), 1/max(len(ids), 1))
    output = []
    for offset in range(1, horizon):
        t = int(obs['time'])+offset; row = []
        count = int(rng.poisson(rate)) if t < 200 and ids else 0
        for j in range(count):
            ue = ids[int(rng.choice(len(ids), p=probs))]
            bits = float(8*rng.integers(1_500_000, 3_000_001)); cycles = bits*int(rng.integers(500, 1001))
            row.append(Task(-(t*10000+j+1), ue, float(t), t+2+float(rng.random()), bits, cycles))
        output.append(row)
    return output


def forecast_window(obs, history, plan, motion, arrivals, horizon=3):
    """Only initial action is varied; suffix uses public EDF routing and hold.

    Exactly H slots, no forced drain. Tail value uses the before-arrival state
    at t+H (same timing as the value-training rows).
    """
    if horizon < 1 or len(arrivals) != horizon-1: raise ValueError('forecast horizon mismatch')
    e = Engine.from_observation(obs); h = copy.deepcopy(history)
    e.assign(plan); e.advance(motion)
    zero = [(0., 0.)]*len(e.positions)
    for row in arrivals:
        e.add(copy.deepcopy(row)); e.admit(); public = e.observe()
        h._update_history(public); e.assign(assignments(public)); e.advance(zero)
    e.audit()
    x = value_features(e.observe(), h)
    return dict(success=int(e.counts['success']), expired=int(e.counts['expired']),
                terminal=x, terminal_empty=not e.tasks and e.t >= 200,
                end_time=e.t, generated=int(e.counts['generated']), active=len(e.tasks))
