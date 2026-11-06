"""Upper-lower joint training analytical feature extraction, teacher targets, and closed-loop inference.

Implements the audited, unconfounded pipeline for `queued_joint_training_20261105`:
1. True zero-simulation online inference (`deterministic_engine_steps = 0`, `stochastic_forecast_calls = 0`)
   for all neural arms (`uncoordinated_init`, `lower_only_trained`, `upper_only_trained`,
   `decomposed_composed`, `decomposed_simultaneous`, `joint_trained`).
   `extract_actor_row` never calls `Engine.from_observation` or any multi-step rollout.
2. Strictly shared candidate bank (`include_safe_plan=True, rescue_mode='unguarded'`) across all
   neural arms—no external hand-crafted congestion gate filters candidates for `joint_trained`.
3. Separate evaluation of:
   - `decomposed_composed`: direct assembly of independently trained `lower_only_trained` and
     `upper_only_trained` parameters (`compose_decomposed_actor`).
   - `decomposed_simultaneous`: simultaneous two-branch training on the exact same dataset,
     candidate bank, and joint teacher target `target_joint` as `joint_trained`, differing only
     in masking cross-layer conditioning.
4. Genuine `joint_neural_mpc`: uses the trained `joint_trained` actor to prune the candidate space
   to the top-6 `(plan, motion)` pairs and refines only those pairs with 3-step online lookahead.
"""
import copy
import math
import time
import numpy as np

from environment.queued_service import Engine, distance
from marl_models.queued_joint_training import actor_policy
from utils.queued_arrival_adaptation import RateHistory, future_arrivals, rate, rate_features
from utils.queued_planner_distillation import planner_target
from utils.queued_quality_amortization import (
    analytical_arrival_outcomes,
    evaluate_amortized_bank,
    expand_safe_plan_bank,
    safe_assignments,
    should_expand_rescue,
)
from utils.queued_score_control import congestion_pressure, row_scores
from utils.queued_waiting_rescue import rescue_candidates, rescue_value, service_bounds

REVISION = 'queued_joint_training_20261105'
TEACHER_TEMPERATURE = 0.015

# Four neural training jobs per parent lineage; `decomposed_composed` is assembled from
# `lower_only_trained` + `upper_only_trained` without separate training.
TRAINING_ARMS = (
    'lower_only_trained',
    'upper_only_trained',
    'decomposed_simultaneous',
    'joint_trained',
)

NEURAL_EVAL_ARMS = (
    'uncoordinated_init',
    'lower_only_trained',
    'upper_only_trained',
    'decomposed_composed',
    'decomposed_simultaneous',
    'joint_trained',
)

ALL_EVAL_ARMS = (
    'uncoordinated_init',
    'lower_only_trained',
    'upper_only_trained',
    'decomposed_composed',
    'decomposed_simultaneous',
    'joint_trained',
    'quality_amortized',
    'old_plan',
    'joint_neural_mpc',
)


def mode_for_arm(arm):
    if arm == 'joint_trained':
        return 'joint'
    if arm in ('decomposed_composed', 'decomposed_simultaneous'):
        return 'decomposed'
    if arm == 'upper_only_trained':
        return 'upper_only'
    if arm == 'lower_only_trained':
        return 'lower_only'
    raise ValueError(f'unknown neural training arm: {arm}')


def _normalize_vector(vec):
    arr = np.asarray(vec, dtype=np.float64)
    if arr.size <= 1:
        return np.zeros_like(arr)
    centered = arr - arr.mean()
    scale = max(float(np.std(centered)), float(np.max(np.abs(centered))), 1e-6)
    return np.clip(centered / scale, -3.0, 3.0)


def advantage_weighted_target(scores, prior, motion_prior, temperature=TEACHER_TEMPERATURE):
    """Smooth advantage-weighted MPO E-step target distribution over (K, M) candidates.

    Computes q(k, m) proportional to pi_0(k, m) * exp((Q(k, m) - max Q) / temperature).
    """
    q = np.asarray(scores, dtype=np.float64)
    p0 = np.asarray(prior, dtype=np.float64)[:, None] * np.asarray(motion_prior, dtype=np.float64)
    p0 = np.clip(p0, 1e-12, None)
    p0 = p0 / p0.sum()
    if temperature <= 0:
        raise ValueError('positive temperature required')
    adv = (q - float(np.max(q))) / float(temperature)
    adv = np.clip(adv, -60.0, 0.0)
    weights = p0 * np.exp(adv)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return p0
    return weights / total


def evaluate_uncoupled_upper_bank(obs, history, plans, motions, rate3, base_neural_scores):
    """Offline teacher helper for uncoupled upper coverage utility (used ONLY during training)."""
    bounds = service_bounds(obs)
    centered_neural = np.asarray(base_neural_scores, dtype=float)
    centered_neural = centered_neural - centered_neural.mean()
    k_orig, m_orig = centered_neural.shape
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
            fut_suc, _ = analytical_arrival_outcomes(obs, history, plan, motion, rate3)
            r_val = rescue_value(obs, motion, bounds)
            n_sc = float(centered_neural[min(pi, k_orig - 1), min(mi, m_orig - 1)])
            val = suc + 0.45 * fut_suc + 0.40 * r_val + 0.12 * n_sc
            scores[pi, mi] = val
    return np.round(scores * 10.0) / 10.0


def prepare_slot_base(root, prototype_module, proposer, score_model, obs, adapter,
                      parent, workload, replica, now):
    """Update adapter once for the current macro slot and compute shared base bank & scores."""
    t = prototype_module.t
    bank = t.bank(proposer, obs, adapter, t.seed(prototype_module.REVISION, 'bank', parent, workload, replica, now))
    raw = prototype_module.previous.validation.original.public_row(bank, obs, adapter.history)
    aug = prototype_module.previous.augment(raw, prototype_module.previous.old.previous.extra(obs, bank, 'pool'))
    base_prob, _ = prototype_module.previous.old.distribution(root, 'hard_joint', parent * 2, aug, obs, bank)
    base_neural = row_scores(aug, score_model, 'joint')
    return dict(bank=bank, base_prob=base_prob, base_neural=base_neural)


def build_joint_bank(root, prototype_module, proposer, score_model, obs, adapter, temporal,
                     parent, workload, replica, now,
                     include_safe_plan=True, rescue_mode='unguarded', precomputed_base=None):
    """Construct candidate allocation & motion bank and baseline anchor distribution.

    All neural arms in `NEURAL_EVAL_ARMS` use `include_safe_plan=True, rescue_mode='unguarded'`
    so every neural arm sees the exact same candidate set without any external hand-crafted gate.
    """
    if precomputed_base is None:
        precomputed_base = prepare_slot_base(
            root, prototype_module, proposer, score_model, obs, adapter,
            parent, workload, replica, now,
        )
    bank = precomputed_base['bank']
    base_prob = precomputed_base['base_prob']
    base_neural = precomputed_base['base_neural']

    original_count = len(bank['actions'])
    original_plan_count = len(bank['plans'])
    pressure = congestion_pressure(obs, bank['plans'], bank['prior'])
    rate3 = float(np.mean(temporal.counts[-3:])) if temporal.counts else 20.0

    if rescue_mode == 'none':
        expand = False
    elif rescue_mode == 'unguarded':
        expand = True
    elif rescue_mode == 'gated':
        expand = should_expand_rescue(pressure, rate3)
    else:
        raise ValueError(f'unknown rescue_mode: {rescue_mode}')

    if expand:
        actions, metadata = rescue_candidates(obs, bank['actions'])
    else:
        actions = list(bank['actions'])
        metadata = dict(added=0, waiting=sum(x['source'] is None for x in obs['tasks']))

    s_plan = safe_assignments(obs)
    base_plan_prior = base_prob.sum(axis=1)
    base_plan_prior = np.clip(base_plan_prior, 1e-8, None)
    base_plan_prior = base_plan_prior / base_plan_prior.sum()
    if include_safe_plan and rescue_mode == 'gated':
        plans, prior = expand_safe_plan_bank(obs, bank['plans'], bank['prior'])
    elif include_safe_plan:
        plans = [dict(p) for p in bank['plans']]
        safe_init_mass = 0.08
        if s_plan not in plans:
            plans.append(s_plan)
            prior = np.r_[(1.0 - safe_init_mass) * base_plan_prior, safe_init_mass]
        else:
            idx = plans.index(s_plan)
            prior = (1.0 - safe_init_mass) * base_plan_prior.copy()
            prior[idx] += safe_init_mass
        prior = np.clip(prior, 1e-8, None)
        prior = prior / prior.sum()
    else:
        plans = [dict(p) for p in bank['plans']]
        prior = base_plan_prior.copy()

    base_motion_cond = np.divide(
        base_prob, base_plan_prior[:, None],
        out=np.full_like(base_prob, 1.0 / original_count),
        where=base_plan_prior[:, None] > 1e-12,
    )
    motion_prior = np.zeros((len(plans), len(actions)), dtype=np.float64)
    for i in range(len(plans)):
        base_m = base_motion_cond[i] if i < original_plan_count else base_motion_cond[0]
        if len(actions) > original_count:
            added_count = len(actions) - original_count
            rescue_init_mass = 0.08
            motion_prior[i, :original_count] = (1.0 - rescue_init_mass) * base_m
            motion_prior[i, original_count:] = rescue_init_mass / added_count
        else:
            motion_prior[i, :] = base_m
        motion_prior[i, :] = np.clip(motion_prior[i, :], 1e-8, None)
        motion_prior[i, :] /= motion_prior[i, :].sum()

    metadata.update(
        pressure=float(pressure),
        rate3=float(rate3),
        guard_blocked=not expand,
        safe_plan_added=bool(len(plans) > original_plan_count),
    )

    return dict(
        plans=plans,
        actions=actions,
        prior=prior,
        motion_prior=motion_prior,
        original_count=original_count,
        original_plan_count=original_plan_count,
        safe_plan=s_plan,
        pressure=float(pressure),
        rate3=float(rate3),
        base_neural=base_neural,
        metadata=metadata,
    )


def _shannon_rate(protocol, kind, a_xyz, b_xyz):
    fixed = getattr(protocol, kind + '_bps', None)
    if fixed is not None:
        return float(fixed)
    bandwidth = float(getattr(protocol, kind + '_bandwidth'))
    dsq = sum((x - y) ** 2 for x, y in zip(a_xyz, b_xyz))
    return bandwidth * math.log2(1.0 + protocol.power * protocol.gain / (max(dsq, 1e-9) * protocol.noise))


def _lindley_plan_motion_eval(obs, admitted, waiting, plan, motion):
    """Closed-form two-stage Lindley tandem-queue completion and overflow recursion.

    Evaluates exact work-conserving EDF service across upload, inter-UAV/backhaul
    transfer, and per-node CPU queues over `[now, now + 3.0]` without instantiating
    or stepping `Engine`.
    """
    protocol = obs['protocol']
    positions = obs['positions']
    n_uav = len(positions)
    mbs_idx = n_uav
    now = float(obs['time'])
    after = [(positions[u][0] + motion[u][0], positions[u][1] + motion[u][1]) for u in range(n_uav)]
    mid = [(positions[u][0] + 0.5 * motion[u][0], positions[u][1] + 0.5 * motion[u][1]) for u in range(n_uav)]

    u_free = [0.0] * n_uav
    t_inter_free = 0.0
    t_bh_free = 0.0
    c_free = [0.0] * (n_uav + 1)
    comp_cnt = 0.0
    over_cnt = 0.0

    for t in admitted:
        src = t['source']
        tgt = t['target'] if t['target'] is not None else plan.get(t['uid'], src)
        slack = float(t['deadline']) - now
        win_slack = min(3.0, slack)

        if t['stage'] == 'upload' and t['remaining_upload'] > 1e-8:
            ue_xy = obs['ues'][t['ue']]
            d0 = distance(ue_xy, positions[src]) <= protocol.coverage + 1e-9
            dm = distance(ue_xy, mid[src]) <= protocol.coverage + 1e-9
            da = distance(ue_xy, after[src]) <= protocol.coverage + 1e-9
            if dm and da:
                r_up = _shannon_rate(protocol, 'uplink', (*ue_xy, 0.0), (*mid[src], protocol.altitude))
                dt_up = float(t['remaining_upload']) / max(r_up, 1e-6)
                t_up_done = u_free[src] + dt_up
            elif d0 and dm:
                r_up = _shannon_rate(protocol, 'uplink', (*ue_xy, 0.0), (*mid[src], protocol.altitude))
                dt_up = float(t['remaining_upload']) / max(r_up, 1e-6)
                t_up_done = u_free[src] + dt_up
                if t_up_done > 0.75:
                    t_up_done = 10.0
            else:
                t_up_done = 10.0
        else:
            t_up_done = 0.0

        rem_tr = float(t['remaining_transfer']) if t['target'] is not None else (float(t['bits']) if tgt != src else 0.0)
        if t['stage'] in ('upload', 'transfer') and tgt != src and rem_tr > 1e-8:
            if tgt == mbs_idx:
                r_tr = _shannon_rate(protocol, 'backhaul', (*mid[src], protocol.altitude), protocol.mbs)
                dt_tr = rem_tr / max(r_tr, 1e-6)
                t_tr_done = max(t_bh_free, t_up_done) + dt_tr
            elif (distance(mid[src], mid[tgt]) <= protocol.inter_range + 1e-9
                  and distance(after[src], after[tgt]) <= protocol.inter_range + 1e-9):
                r_tr = _shannon_rate(protocol, 'inter', (*mid[src], protocol.altitude), (*mid[tgt], protocol.altitude))
                dt_tr = rem_tr / max(r_tr, 1e-6)
                t_tr_done = max(t_inter_free, t_up_done) + dt_tr
            else:
                t_tr_done = 10.0
        else:
            t_tr_done = t_up_done

        dt_cpu = float(t['remaining_cycles']) / float(protocol.capacity[tgt])
        t_cpu_done = max(c_free[tgt], t_tr_done) + dt_cpu

        if t_cpu_done <= slack + 1e-9:
            if t['stage'] == 'upload' and t['remaining_upload'] > 1e-8:
                u_free[src] = t_up_done
            if t['stage'] in ('upload', 'transfer') and tgt != src and rem_tr > 1e-8:
                if tgt == mbs_idx:
                    t_bh_free = t_tr_done
                else:
                    t_inter_free = t_tr_done
            c_free[tgt] = t_cpu_done
            if t_cpu_done <= win_slack + 1e-9:
                comp_cnt += 1.0
            else:
                comp_cnt += 0.85
                over_cnt += 0.15
        else:
            over_cnt += 1.0
            if slack > 0.0:
                if t['stage'] == 'upload' and t['remaining_upload'] > 1e-8 and t_up_done < 9.0:
                    u_free[src] = max(u_free[src], min(slack, t_up_done))
                if t_up_done < slack and t_tr_done < slack:
                    c_free[tgt] = max(c_free[tgt], min(slack, t_cpu_done))

    for wt in waiting:
        slack = float(wt['deadline']) - now
        if slack <= 1.0 + 1e-9:
            continue
        win_slack = min(3.0, slack)
        ue_xy = obs['ues'][wt['ue']]
        src = min(range(n_uav), key=lambda u: (distance(ue_xy, after[u]), u))
        if distance(ue_xy, after[src]) > protocol.coverage + 1e-9:
            continue

        r_up = _shannon_rate(protocol, 'uplink', (*ue_xy, 0.0), (*after[src], protocol.altitude))
        dt_up = float(wt['bits']) / max(r_up, 1e-6)
        t_up_done = max(1.0, u_free[src]) + dt_up

        best_choice = None
        for node in range(n_uav + 1):
            if node != mbs_idx and node != src and distance(after[src], after[node]) > protocol.inter_range + 1e-9:
                continue
            if node == src:
                t_tr_done = t_up_done
            elif node == mbs_idx:
                r_tr = _shannon_rate(protocol, 'backhaul', (*after[src], protocol.altitude), protocol.mbs)
                t_tr_done = max(1.0, t_bh_free, t_up_done) + float(wt['bits']) / max(r_tr, 1e-6)
            else:
                r_tr = _shannon_rate(protocol, 'inter', (*after[src], protocol.altitude), (*after[node], protocol.altitude))
                t_tr_done = max(1.0, t_inter_free, t_up_done) + float(wt['bits']) / max(r_tr, 1e-6)
            dt_cpu = float(wt['cycles']) / float(protocol.capacity[node])
            t_cpu_done = max(1.0, c_free[node], t_tr_done) + dt_cpu
            cand = (t_cpu_done > slack + 1e-9, t_cpu_done, node, t_tr_done)
            if best_choice is None or cand < best_choice:
                best_choice = cand

        miss, t_cpu_done, best_node, t_tr_done = best_choice
        if not miss:
            u_free[src] = t_up_done
            if best_node == mbs_idx:
                t_bh_free = t_tr_done
            elif best_node != src:
                t_inter_free = t_tr_done
            c_free[best_node] = t_cpu_done
            if t_cpu_done <= win_slack + 1e-9:
                comp_cnt += 1.0
            else:
                over_cnt += 1.0
        else:
            over_cnt += 1.0

    return comp_cnt, over_cnt


def _instantaneous_plan_motion_physics(obs, history, plans, actions, s_plan, pressure, rate3, base_neural):
    """Pure closed-form analytical queue, channel, and coverage features at time `t`.

    Performs ZERO `Engine` instantiations and ZERO multi-step simulation rollouts.
    """
    protocol = obs['protocol']
    positions = obs['positions']
    n_uav = len(positions)
    mbs_idx = n_uav
    admitted = sorted(
        [t for t in obs['tasks'] if t['source'] is not None],
        key=lambda z: (z['deadline'], z['uid']),
    )
    waiting = sorted(
        [t for t in obs['tasks'] if t['source'] is None],
        key=lambda z: (z['deadline'], z['uid']),
    )
    unassigned_uids = [t['uid'] for t in admitted if t['target'] is None]

    k_plans = len(plans)
    m_actions = len(actions)
    centered_neural = np.asarray(base_neural, dtype=np.float64)
    centered_neural = centered_neural - centered_neural.mean()
    k_orig, m_orig = centered_neural.shape

    plan_node_loads = np.zeros((k_plans, n_uav + 1), dtype=np.float64)
    plan_static_completable = np.zeros(k_plans, dtype=np.float64)
    plan_static_overloaded = np.zeros(k_plans, dtype=np.float64)
    plan_safe_agree = np.zeros(k_plans, dtype=np.float64)
    plan_transfer_counts = np.zeros(k_plans, dtype=np.float64)
    plan_backhaul_counts = np.zeros(k_plans, dtype=np.float64)

    zero_motion = [(0.0, 0.0)] * n_uav
    for pi, plan in enumerate(plans):
        if unassigned_uids:
            plan_safe_agree[pi] = float(np.mean([
                1.0 if plan.get(uid) == s_plan.get(uid) else 0.0 for uid in unassigned_uids
            ]))
        else:
            plan_safe_agree[pi] = 1.0

        inter_cnt = sum(
            1 for t in admitted
            if t['stage'] in ('upload', 'transfer')
            and (t['target'] if t['target'] is not None else plan.get(t['uid'], t['source'])) not in (t['source'], mbs_idx)
        )
        bh_cnt = sum(
            1 for t in admitted
            if t['stage'] in ('upload', 'transfer')
            and (t['target'] if t['target'] is not None else plan.get(t['uid'], t['source'])) == mbs_idx
        )
        plan_transfer_counts[pi] = float(inter_cnt)
        plan_backhaul_counts[pi] = float(bh_cnt)

        for t in admitted:
            src = t['source']
            tgt = t['target'] if t['target'] is not None else plan.get(t['uid'], src)
            cpu_sec = float(t['remaining_cycles']) / float(protocol.capacity[tgt])
            plan_node_loads[pi, tgt] += cpu_sec

        c_stat, o_stat = _lindley_plan_motion_eval(obs, admitted, [], plan, zero_motion)
        plan_static_completable[pi] = c_stat
        plan_static_overloaded[pi] = o_stat

    bounds = service_bounds(obs)
    rescue_vals = np.array([rescue_value(obs, m, bounds) for m in actions], dtype=np.float64)
    move_norms = np.array([
        sum(math.hypot(dx, dy) for dx, dy in m) / max(1.0, len(m) * protocol.speed)
        for m in actions
    ], dtype=np.float64)

    inst_completable = np.zeros((k_plans, m_actions), dtype=np.float64)
    inst_overflow = np.zeros((k_plans, m_actions), dtype=np.float64)
    fut_suc_mat = np.zeros((k_plans, m_actions), dtype=np.float64)
    fut_over_mat = np.zeros((k_plans, m_actions), dtype=np.float64)
    neural_mat = np.zeros((k_plans, m_actions), dtype=np.float64)

    for mi, motion in enumerate(actions):
        f_suc_ref, f_over_ref = analytical_arrival_outcomes(obs, history, s_plan, motion, rate3)
        for pi, plan in enumerate(plans):
            comp_cnt, over_cnt = _lindley_plan_motion_eval(obs, admitted, waiting, plan, motion)
            inst_completable[pi, mi] = comp_cnt
            inst_overflow[pi, mi] = over_cnt
            fut_suc_mat[pi, mi] = f_suc_ref
            fut_over_mat[pi, mi] = f_over_ref
            src_pi = pi if pi < k_orig else 0
            neural_mat[pi, mi] = float(centered_neural[src_pi, min(mi, m_orig - 1)])

    return dict(
        plan_node_loads=plan_node_loads,
        plan_static_completable=plan_static_completable,
        plan_static_overloaded=plan_static_overloaded,
        plan_safe_agree=plan_safe_agree,
        plan_transfer_counts=plan_transfer_counts,
        plan_backhaul_counts=plan_backhaul_counts,
        rescue_vals=rescue_vals,
        move_norms=move_norms,
        inst_completable=inst_completable,
        inst_overflow=inst_overflow,
        fut_suc_mat=fut_suc_mat,
        fut_over_mat=fut_over_mat,
        neural_mat=neural_mat,
    )


def extract_actor_row(obs, bank, adapter, temporal):
    """Extract purely analytical public context, plan features `a`, and plan-motion features `m`.

    Executes ZERO `Engine` simulations (`deterministic_engine_steps = 0`) and ZERO
    stochastic future arrival rollouts (`stochastic_forecast_calls = 0`), and contains
    NO pre-computed argmax winner indicator flags.
    """
    plans = bank['plans']
    actions = bank['actions']
    original_count = bank['original_count']
    s_plan = bank['safe_plan']
    pressure = bank['pressure']
    rate3 = bank['rate3']
    base_neural = bank['base_neural']

    rates = rate_features(obs, adapter.history, temporal)
    admitted_tasks = [t for t in obs['tasks'] if t['source'] is not None]
    waiting_tasks = [t for t in obs['tasks'] if t['source'] is None]
    up_count = sum(t['stage'] == 'upload' for t in admitted_tasks)
    rel_count = sum(t['stage'] == 'transfer' for t in admitted_tasks)
    comp_count = sum(t['stage'] == 'compute' for t in admitted_tasks)
    is_congested = 1.0 if not should_expand_rescue(pressure, rate3) else 0.0

    context = np.r_[
        rates,
        [
            float(np.tanh(pressure / 2.0)),
            rate3 / 60.0,
            is_congested,
            len(admitted_tasks) / 40.0,
            len(waiting_tasks) / 40.0,
            (up_count + rel_count + comp_count) / 40.0,
        ],
    ]

    phys = _instantaneous_plan_motion_physics(
        obs, adapter.history, plans, actions, s_plan, pressure, rate3, base_neural
    )
    plan_node_loads = phys['plan_node_loads']
    plan_static_completable = phys['plan_static_completable']
    plan_static_overloaded = phys['plan_static_overloaded']
    plan_safe_agree = phys['plan_safe_agree']
    plan_transfer_counts = phys['plan_transfer_counts']
    plan_backhaul_counts = phys['plan_backhaul_counts']
    rescue_vals = phys['rescue_vals']
    move_norms = phys['move_norms']
    inst_completable = phys['inst_completable']
    inst_overflow = phys['inst_overflow']
    fut_suc_mat = phys['fut_suc_mat']
    fut_over_mat = phys['fut_over_mat']
    neural_mat = phys['neural_mat']

    n_uav = len(obs['positions'])
    is_rescue_vec = np.array(
        [1.0 if j >= original_count else 0.0 for j in range(len(actions))],
        dtype=np.float64,
    )
    is_safe_vec = np.array(
        [1.0 if plan == s_plan else 0.0 for plan in plans],
        dtype=np.float64,
    )
    plan_overload_sq = np.array([
        sum(max(0.0, float(plan_node_loads[pi, u]) - 1.0) ** 2 for u in range(n_uav))
        for pi in range(len(plans))
    ], dtype=np.float64)

    # Continuous analytical quality proxies (no Engine simulation, no discrete argmax indicators)
    total_overflow = inst_overflow + fut_over_mat + 0.65 * plan_overload_sq[:, None]
    coupled_utility_proxy = (
        inst_completable
        + (0.20 if is_congested > 0.5 else 0.45) * fut_suc_mat
        + (0.0 if is_congested > 0.5 else 0.30) * rescue_vals[None, :]
        - (0.55 if is_congested > 0.5 else 0.25) * total_overflow
        + (0.0 if is_congested > 0.5 else 0.12) * neural_mat
        - 5.00 * is_congested * is_rescue_vec[None, :]
        + 0.15 * is_safe_vec[:, None]
    )

    plan_comp_norm = _normalize_vector(plan_static_completable - 0.50 * plan_overload_sq)
    plan_over_norm = _normalize_vector(plan_static_overloaded + 0.50 * plan_overload_sq)
    plan_neural_mean_norm = _normalize_vector(neural_mat.mean(axis=1))
    log_prior_norm = _normalize_vector(np.log(np.clip(bank['prior'], 1e-12, None)))
    n_adm = max(1.0, float(len(admitted_tasks)))

    a_rows = []
    for i, plan in enumerate(plans):
        uav_loads = plan_node_loads[i, :n_uav]
        mbs_load = float(plan_node_loads[i, n_uav])
        is_safe = is_safe_vec[i]
        a_rows.append([
            plan_comp_norm[i],
            plan_over_norm[i],
            float(plan_transfer_counts[i] / n_adm),
            float(plan_backhaul_counts[i] / n_adm),
            is_safe,
            plan_safe_agree[i],
            float(np.tanh(np.max(uav_loads) / 2.0)),
            float(np.tanh(np.mean(uav_loads) / 2.0)),
            float(np.tanh(np.std(uav_loads) / 2.0)),
            float(np.tanh(mbs_load / 2.0)),
            plan_neural_mean_norm[i],
            log_prior_norm[i],
            float(np.tanh(pressure / 2.0)),
            is_congested,
        ])
    a_mat = np.asarray(a_rows, dtype=np.float64)

    # Global (K, M) normalization preserves cross-plan utility & overflow differences in C_upper!
    rescue_norm = _normalize_vector(rescue_vals)
    fut_suc_norm_global = _normalize_vector(fut_suc_mat)
    overflow_norm_global = _normalize_vector(total_overflow)
    net_gain_norm_global = _normalize_vector(coupled_utility_proxy)
    comp_norm_global = _normalize_vector(inst_completable)
    neural_norm_global = _normalize_vector(neural_mat)

    m_tensor = np.zeros((len(plans), len(actions), 16), dtype=np.float64)
    for i, plan in enumerate(plans):
        log_mp_norm = _normalize_vector(np.log(np.clip(bank['motion_prior'][i], 1e-12, None)))
        is_safe = is_safe_vec[i]
        for j in range(len(actions)):
            is_hover = 1.0 if move_norms[j] < 1e-6 else 0.0
            is_rescue = is_rescue_vec[j]
            # First 4 channels form `phi_couple` for the 4D upper-policy expectation vector C_upper:
            #   0: rescue_norm (upper waiting rescue potential)
            #   1: fut_suc_norm_global (expected future arrival service rate)
            #   2: overflow_norm_global (cross-layer queue & deadline overflow)
            #   3: net_gain_norm_global (cross-layer congestion-adjusted net gain)
            # Channels (2, 3, 4, 9, 10, 11, 12, 13) are masked in uncoupled modes.
            m_tensor[i, j, :] = [
                rescue_norm[j],
                fut_suc_norm_global[i, j],
                overflow_norm_global[i, j],
                net_gain_norm_global[i, j],
                comp_norm_global[i, j],
                neural_norm_global[i, j],
                move_norms[j],
                is_hover,
                is_rescue,
                is_congested,
                is_congested * is_rescue,
                (1.0 - is_congested) * rescue_norm[j],
                float(np.tanh(pressure / 2.0)) * move_norms[j],
                is_safe,
                plan_safe_agree[i],
                log_mp_norm[j],
            ]

    return dict(
        a=a_mat,
        m=m_tensor,
        context=context,
        prior=bank['prior'],
        motion_prior=bank['motion_prior'],
    )


def compute_offline_teacher_targets(obs, bank, adapter):
    """Compute offline teacher target distributions & value matrices for training data collection ONLY.

    Uses smooth advantage-weighted softmax (`TEACHER_TEMPERATURE = 0.015`) over the offline
    multi-step teacher values. Never called during online evaluation (`neural_actor_step`).
    """
    plans = bank['plans']
    actions = bank['actions']
    prior = bank['prior']
    motion_prior = bank['motion_prior']
    pressure = bank['pressure']
    rate3 = bank['rate3']
    base_neural = bank['base_neural']
    s_plan = bank['safe_plan']
    protocol = obs['protocol']
    n_uav = len(obs['positions'])
    admitted = sorted(
        [t for t in obs['tasks'] if t['source'] is not None],
        key=lambda z: (z['deadline'], z['uid']),
    )
    waiting = sorted(
        [t for t in obs['tasks'] if t['source'] is None],
        key=lambda z: (z['deadline'], z['uid']),
    )

    # Evaluate 3-step Engine teacher with canonical reference future arrival capacity,
    # full-deadline Lindley cascade completion, and concave per-UAV overload penalty.
    q_raw = evaluate_amortized_bank(
        obs, adapter.history, plans, actions, pressure, rate3, base_neural
    ).copy()
    congested = not should_expand_rescue(pressure, rate3)
    w_fut_suc = 0.20 if congested else 0.45
    w_fut_over = 0.55 if congested else 0.25

    for mi, motion in enumerate(actions):
        f_suc_ref, f_over_ref = analytical_arrival_outcomes(obs, adapter.history, s_plan, motion, rate3)
        for pi, plan in enumerate(plans):
            f_suc_p, f_over_p = analytical_arrival_outcomes(obs, adapter.history, plan, motion, rate3)
            comp_cnt, over_cnt = _lindley_plan_motion_eval(obs, admitted, waiting, plan, motion)
            q_raw[pi, mi] += (
                w_fut_suc * (f_suc_ref - f_suc_p)
                - w_fut_over * (f_over_ref - f_over_p)
                + 0.35 * (comp_cnt - 0.65 * over_cnt)
            )

    for pi, plan in enumerate(plans):
        uav_loads = [0.0] * n_uav
        for t in admitted:
            tgt = t['target'] if t['target'] is not None else plan.get(t['uid'], t['source'])
            if tgt < n_uav:
                uav_loads[tgt] += float(t['remaining_cycles']) / float(protocol.capacity[tgt])
        overload_sq = sum(max(0.0, ld - 1.0) ** 2 for ld in uav_loads)
        q_raw[pi, :] -= w_fut_over * 0.65 * overload_sq
        if plan == s_plan:
            q_raw[pi, :] += 0.15

    if congested and bank['original_count'] < len(actions):
        q_raw[:, bank['original_count']:] -= 5.00

    q_joint = np.round(q_raw * 100.0) / 100.0

    q_upper = evaluate_uncoupled_upper_bank(
        obs, adapter.history, plans, actions, rate3, base_neural
    )

    _, teacher_plan_prior = expand_safe_plan_bank(obs, plans, prior)

    # 1. Joint target (used by both `joint_trained` and `decomposed_simultaneous`!)
    target_joint = advantage_weighted_target(
        q_joint, teacher_plan_prior, motion_prior, temperature=TEACHER_TEMPERATURE
    )

    # 2. Upper-only target: uncoupled coverage utility under baseline lower allocation prior
    target_upper_full = advantage_weighted_target(q_upper, prior, motion_prior, temperature=TEACHER_TEMPERATURE)
    qm_upper = np.divide(
        target_upper_full, target_upper_full.sum(axis=1, keepdims=True),
        out=np.full_like(target_upper_full, 1.0 / q_upper.shape[1]),
        where=target_upper_full.sum(axis=1, keepdims=True) > 1e-12,
    )
    target_upper = prior[:, None] * qm_upper
    target_upper /= target_upper.sum()

    # 3. Lower-only target: exact marginal allocation target and soft-value over motions
    qa_lower = target_joint.sum(axis=1, keepdims=True)
    target_lower = qa_lower * motion_prior
    target_lower /= target_lower.sum()
    q_row_max = np.max(q_joint, axis=1, keepdims=True)
    q_lower_plan = q_row_max + TEACHER_TEMPERATURE * np.log(
        np.clip(
            np.sum(motion_prior * np.exp((q_joint - q_row_max) / TEACHER_TEMPERATURE), axis=1, keepdims=True),
            1e-12, None,
        )
    )
    v_lower = np.broadcast_to(q_lower_plan, q_joint.shape).copy()

    return dict(
        joint=(target_joint, q_joint),
        decomposed=(target_joint, q_joint),  # Identical target & value matrix as `joint`!
        upper_only=(target_upper, q_upper),
        lower_only=(target_lower, v_lower),
    )


def compute_teacher_targets(obs, bank, row_or_adapter):
    """Backward-compatible wrapper around `compute_offline_teacher_targets`."""
    if hasattr(row_or_adapter, 'history'):
        return compute_offline_teacher_targets(obs, bank, row_or_adapter)
    raise ValueError('compute_teacher_targets requires `adapter` and may only be called offline')


def neural_actor_step(root, prototype_module, proposer, score_model, obs, adapter, temporal,
                      arm, model, parent, workload, replica, now, precomputed_base=None):
    """Execute one closed-loop decision step.

    - All neural arms (`NEURAL_EVAL_ARMS`) share the exact same candidate bank
      (`include_safe_plan=True, rescue_mode='unguarded'`) and execute with
      `deterministic_engine_steps = 0` and `stochastic_forecast_calls = 0`.
    - `quality_amortized` serves as the online 3-step deterministic rollout reference
      (`deterministic_engine_steps = 3 * K * M`, `stochastic_forecast_calls = 0`).
    - `old_plan` serves as the online 3-step stochastic rollout baseline.
    - `joint_neural_mpc` uses the trained `joint_trained` neural network to select the
      top-6 `(plan, motion)` pairs and refines only those 6 pairs with 3-step online lookahead.
    """
    t = prototype_module.t
    started = time.perf_counter()

    if arm in NEURAL_EVAL_ARMS:
        bank = build_joint_bank(
            root, prototype_module, proposer, score_model, obs, adapter, temporal,
            parent, workload, replica, now,
            include_safe_plan=True, rescue_mode='unguarded', precomputed_base=precomputed_base,
        )
        row = extract_actor_row(obs, bank, adapter, temporal)
        if arm == 'uncoordinated_init':
            probability = row['prior'][:, None] * row['motion_prior']
            probability /= probability.sum()
        else:
            mode = mode_for_arm(arm)
            probability = actor_policy(row, model, mode=mode)
        rng = np.random.default_rng(t.seed(prototype_module.REVISION, 'action', parent, workload, replica, now))
        choice = int(rng.choice(probability.size, p=probability.ravel()))
        alloc_idx, motion_idx = np.unravel_index(choice, probability.shape)
        det_engine_steps = 0
        forecast_calls = 0
    elif arm == 'quality_amortized':
        bank = build_joint_bank(
            root, prototype_module, proposer, score_model, obs, adapter, temporal,
            parent, workload, replica, now,
            include_safe_plan=True, rescue_mode='gated', precomputed_base=precomputed_base,
        )
        q_amort = evaluate_amortized_bank(
            obs, adapter.history, bank['plans'], bank['actions'],
            bank['pressure'], bank['rate3'], bank['base_neural'],
        )
        probability = planner_target(q_amort, bank['prior'])
        rng = np.random.default_rng(t.seed(prototype_module.REVISION, 'action', parent, workload, replica, now))
        choice = int(rng.choice(probability.size, p=probability.ravel()))
        alloc_idx, motion_idx = np.unravel_index(choice, probability.shape)
        det_engine_steps = 3 * len(bank['plans']) * len(bank['actions'])
        forecast_calls = 0
    elif arm == 'old_plan':
        bank = build_joint_bank(
            root, prototype_module, proposer, score_model, obs, adapter, temporal,
            parent, workload, replica, now,
            include_safe_plan=False, rescue_mode='none', precomputed_base=precomputed_base,
        )
        f_seed = t.seed(prototype_module.REVISION, 'forecast', parent, workload, replica, now)
        arr = future_arrivals(obs, adapter.history, temporal, 'window3', f_seed)
        q = prototype_module.evaluate_bank(obs, adapter.history, bank['plans'], bank['actions'], arr)
        probability = planner_target(q, bank['prior'])
        rng = np.random.default_rng(t.seed(prototype_module.REVISION, 'action', parent, workload, replica, now))
        choice = int(rng.choice(probability.size, p=probability.ravel()))
        alloc_idx, motion_idx = np.unravel_index(choice, probability.shape)
        det_engine_steps = 0
        forecast_calls = len(bank['plans']) * len(bank['actions'])
    elif arm == 'joint_neural_mpc':
        bank = build_joint_bank(
            root, prototype_module, proposer, score_model, obs, adapter, temporal,
            parent, workload, replica, now,
            include_safe_plan=True, rescue_mode='unguarded', precomputed_base=precomputed_base,
        )
        row = extract_actor_row(obs, bank, adapter, temporal)
        neural_prob = actor_policy(row, model, mode='joint')
        flat_prob = neural_prob.ravel()
        top_k = min(6, flat_prob.size)
        top_flat_idx = np.argsort(flat_prob)[-top_k:]

        f_seed = t.seed(prototype_module.REVISION, 'forecast', parent, workload, replica, now)
        arr = future_arrivals(obs, adapter.history, temporal, 'window3', f_seed)
        zero = [(0., 0.)] * len(obs['positions'])
        bounds = service_bounds(obs)
        congested = not should_expand_rescue(bank['pressure'], bank['rate3'])
        best_val = -1e18
        best_pairs = []
        for f_idx in top_flat_idx:
            pi, mi = np.unravel_index(int(f_idx), neural_prob.shape)
            plan = bank['plans'][pi]
            motion = bank['actions'][mi]
            e = Engine.from_observation(obs)
            e.assign(plan)
            e.advance(motion)
            for batch in arr:
                e.add(copy.deepcopy(batch))
                e.admit()
                pub = e.observe()
                e.assign(safe_assignments(pub))
                e.advance(zero)
            suc = float(e.counts['success'])
            adm = int(e.counts['admitted'])
            uncompleted = max(0, adm - int(e.counts['success']))
            r_val = rescue_value(obs, motion, bounds)
            val = (
                suc + (0.0 if congested else 0.25 * r_val)
                - (0.45 if congested else 0.15) * uncompleted
                + 0.20 * math.log(max(float(neural_prob[pi, mi]), 1e-12))
            )
            val = round(val * 100.0) / 100.0
            if val > best_val + 1e-9:
                best_val = val
                best_pairs = [(int(pi), int(mi), float(neural_prob[pi, mi]))]
            elif abs(val - best_val) <= 1e-9:
                best_pairs.append((int(pi), int(mi), float(neural_prob[pi, mi])))
        weights = np.array([w for _, _, w in best_pairs], dtype=np.float64)
        weights = weights / weights.sum()
        rng = np.random.default_rng(t.seed(prototype_module.REVISION, 'action', parent, workload, replica, now))
        sel = int(rng.choice(len(best_pairs), p=weights))
        alloc_idx, motion_idx, _ = best_pairs[sel]
        det_engine_steps = 0
        forecast_calls = int(top_k)
    else:
        raise ValueError(f'unexpected evaluation arm: {arm}')

    elapsed = time.perf_counter() - started
    selected_plan = bank['plans'][alloc_idx]
    selected_motion = bank['actions'][motion_idx]
    original_count = bank['original_count']
    metadata = dict(bank['metadata'])
    metadata['selected_safe_plan'] = bool(selected_plan == bank['safe_plan'])
    metadata['arm'] = arm

    return dict(
        plan=selected_plan,
        motion=selected_motion,
        original_candidates=original_count,
        candidates=len(bank['actions']),
        selected_new=bool(motion_idx >= original_count),
        metadata=metadata,
        decision_seconds=elapsed,
        deterministic_engine_steps=det_engine_steps,
        forecast_calls=forecast_calls,
    )
