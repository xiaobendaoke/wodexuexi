"""Cascade-safe EDF assignment, capacity-guarded rescue, and analytical queue amortization."""
from collections import defaultdict
import copy
import numpy as np

from environment.queued_service import Engine, distance
from utils.queued_joint_mpo import forecast_window, value_features
from utils.queued_waiting_rescue import rescue_candidates, rescue_value, service_bounds

REVISION = 'queued_quality_amortization_20261105'
RESCUE_PRESSURE_LIMIT = 0.85
RESCUE_RATE_LIMIT = 25.0
CONGESTED_PRESSURE_LIMIT = 1.0


def safe_assignments(obs):
    """Cascade-safe EDF assignment that isolates doomed tasks from busy UAV CPUs.

    Uses only public current observation fields (`protocol`, `positions`, `time`,
    `tasks`, `ues`). Never accesses private future arrivals or hidden states.
    """
    p = obs['protocol']
    positions = obs['positions']
    n = len(positions)
    now = obs['time']
    tasks = obs['tasks']
    plan = {}
    work = [[] for _ in p.capacity]
    inter_bits = sum(x['remaining_transfer'] for x in tasks if x['stage'] == 'transfer' and x['target'] != n)
    back_bits = sum(x['remaining_transfer'] for x in tasks if x['stage'] == 'transfer' and x['target'] == n)
    source_ues = defaultdict(set)
    for x in tasks:
        if x['source'] is not None and x['stage'] == 'upload':
            source_ues[x['source']].add(x['ue'])
        if x['target'] is not None:
            work[x['target']].append((x['deadline'], x['remaining_cycles']))
    radio = Engine(p, positions, obs['ues'])
    for x in tasks:
        if x['source'] is None or x['target'] is not None:
            continue
        source = x['source']
        ue = obs['ues'][x['ue']]
        upload_rate = radio.rate('uplink', (*ue, 0.), (*positions[source], p.altitude))
        upload = x['remaining_upload'] * max(len(source_ues[source]), 1) / upload_rate
        slack = x['deadline'] - now
        choices = []
        for node in range(n + 1):
            if node != n and node != source and distance(positions[source], positions[node]) > p.inter_range:
                continue
            transfer = 0.
            if node != source:
                target = p.mbs if node == n else (*positions[node], p.altitude)
                kind = 'backhaul' if node == n else 'inter'
                rate = radio.rate(kind, (*positions[source], p.altitude), target)
                backlog = back_bits if node == n else inter_bits
                transfer = (backlog + x['bits']) / rate
            cap = p.capacity[node]
            items = sorted(work[node], key=lambda z: z[0])
            load_before = sum(c for d, c in items if d <= x['deadline'])
            latency = upload + transfer + (load_before + x['remaining_cycles']) / cap
            self_miss = latency > slack
            total_load = sum(c for _, c in items) / cap
            cascade_misses = 0
            if total_load > 1.2:
                cum = 0.0
                for d, c in items:
                    cum += c
                    if d > x['deadline']:
                        was_ok = (cum / cap) <= (d - now)
                        now_ok = ((cum + x['remaining_cycles']) / cap) <= (d - now)
                        if was_ok and not now_ok:
                            cascade_misses += 1
            if not self_miss and cascade_misses == 0:
                score = (0, 0, latency, node)
            elif not self_miss:
                score = (1, cascade_misses, latency, node)
            else:
                congested_uav = 1 if (node < n and total_load > 1.0) else 0
                score = (2, cascade_misses, congested_uav, total_load, latency, node)
            choices.append((score, node))
        best_score, node = min(choices)
        plan[x['uid']] = node
        if best_score[0] < 2:
            work[node].append((x['deadline'], x['remaining_cycles']))
        if node != source and best_score[0] < 2:
            if node == n:
                back_bits += x['bits']
            else:
                inter_bits += x['bits']
    return plan


def expand_safe_plan_bank(obs, plans, prior):
    """Ensure `safe_assignments(obs)` is in the candidate allocation plan bank."""
    s_plan = safe_assignments(obs)
    out_plans = [dict(p) for p in plans]
    out_prior = [float(x) for x in prior]
    if s_plan not in out_plans:
        out_plans.append(s_plan)
        out_prior.append(max(out_prior))
    else:
        idx = out_plans.index(s_plan)
        out_prior[idx] = max(out_prior)
    arr = np.asarray(out_prior, dtype=float)
    arr /= arr.sum()
    return out_plans, arr


def should_expand_rescue(pressure, rate3):
    """Dual public congestion & causal arrival-rate gate for rescue motion proposals."""
    return bool(float(pressure) <= RESCUE_PRESSURE_LIMIT and float(rate3) <= RESCUE_RATE_LIMIT)


def forecast_quality_window(obs, history, plan, motion, arrivals, horizon=3):
    """Short-window public forecast using cascade-safe continuation and admission accounting."""
    if horizon < 1 or len(arrivals) != horizon - 1:
        raise ValueError('forecast horizon mismatch')
    e = Engine.from_observation(obs)
    h = copy.deepcopy(history)
    e.assign(plan)
    e.advance(motion)
    zero = [(0., 0.)] * len(e.positions)
    for row in arrivals:
        e.add(copy.deepcopy(row))
        e.admit()
        public = e.observe()
        h._update_history(public)
        e.assign(safe_assignments(public))
        e.advance(zero)
    e.audit()
    x = value_features(e.observe(), h)
    suc = int(e.counts['success'])
    adm = int(e.counts['admitted'])
    return dict(
        success=suc,
        admitted=adm,
        uncompleted_admitted=max(0, adm - suc),
        expired=int(e.counts['expired']),
        terminal=x,
        terminal_empty=not e.tasks and e.t >= 200,
        end_time=e.t,
        generated=int(e.counts['generated']),
        active=len(e.tasks),
    )


def evaluate_quality_bank(obs, history, plans, motions, arrivals, pressure):
    """Evaluate `(plans, motions)` with overload-aware tie-breaking under congestion."""
    if float(pressure) <= CONGESTED_PRESSURE_LIMIT:
        return np.array([
            [forecast_window(obs, history, plan, motion, arrivals, 3)['success'] for motion in motions]
            for plan in plans
        ], dtype=float)
    raw = np.array([
        [
            (lambda res: float(res['success']) - 0.10 * float(res['uncompleted_admitted']))(
                forecast_quality_window(obs, history, plan, motion, arrivals, 3)
            )
            for motion in motions
        ]
        for plan in plans
    ], dtype=float)
    return np.round(raw * 10.0) / 10.0


def analytical_arrival_outcomes(obs, history, plan, motion, rate3):
    """Closed-form expected 2-slot arrivals split into completable vs overflow per UAV."""
    after = [(a[0] + d[0], a[1] + d[1]) for a, d in zip(obs['positions'], motion)]
    p = obs['protocol']
    n = len(after)
    tot_w = sum(history.weights.get(u, 0.) for u in obs['ues'])
    if tot_w <= 1e-12:
        return 0.0, 0.0
    load_s = [0.0] * n
    for x in obs['tasks']:
        if x['source'] is not None:
            target = x['target'] if x['target'] is not None else plan.get(x['uid'], x['source'])
            if target < n:
                load_s[target] += x['remaining_cycles'] / p.capacity[target]
        else:
            xy = obs['ues'][x['ue']]
            nearest = min(range(n), key=lambda i: (distance(xy, after[i]), i))
            if distance(xy, after[nearest]) <= p.coverage + 1e-9:
                load_s[nearest] += x['remaining_cycles'] / p.capacity[nearest]
    uav_w = [0.0] * n
    for u, xy in obs['ues'].items():
        w = history.weights.get(u, 0.)
        if w <= 0:
            continue
        nearest = min(range(n), key=lambda i: (distance(xy, after[i]), i))
        if distance(xy, after[nearest]) <= p.coverage + 1e-9:
            uav_w[nearest] += w
    fut_suc = 0.0
    fut_over = 0.0
    for i in range(n):
        arr_i = 2.0 * float(rate3) * (uav_w[i] / tot_w)
        cap_i = max(0.0, (1.05 - load_s[i]) / 0.38)
        s_i = min(arr_i, cap_i)
        fut_suc += s_i
        fut_over += max(0.0, arr_i - s_i)
    return float(fut_suc), float(fut_over)


def evaluate_amortized_bank(obs, history, plans, motions, pressure, rate3, base_neural_scores):
    """Zero-future-arrival deterministic public lookahead + analytical queueing + neural score."""
    bounds = service_bounds(obs)
    centered_neural = np.asarray(base_neural_scores, dtype=float)
    centered_neural = centered_neural - centered_neural.mean()
    k_orig, m_orig = centered_neural.shape
    congested = not should_expand_rescue(pressure, rate3)
    scores = np.zeros((len(plans), len(motions)), dtype=float)
    zero = [(0., 0.)] * len(obs['positions'])
    for pi, plan in enumerate(plans):
        for mi, motion in enumerate(motions):
            e = Engine.from_observation(obs)
            e.assign(plan)
            e.advance(motion)
            for _ in range(2):
                e.admit()
                pub = e.observe()
                e.assign(safe_assignments(pub))
                e.advance(zero)
            suc = float(e.counts['success'])
            adm = int(e.counts['admitted'])
            uncompleted_cur = max(0, adm - int(e.counts['success']))
            fut_suc, fut_over = analytical_arrival_outcomes(obs, history, plan, motion, rate3)
            n_sc = float(centered_neural[min(pi, k_orig - 1), min(mi, m_orig - 1)])
            if not congested:
                r_val = rescue_value(obs, motion, bounds)
                val = suc + 0.45 * fut_suc + 0.30 * r_val - 0.25 * (uncompleted_cur + fut_over) + 0.12 * n_sc
            else:
                val = suc + 0.20 * fut_suc - 0.55 * (uncompleted_cur + fut_over)
            scores[pi, mi] = val
    return np.round(scores * 10.0) / 10.0
