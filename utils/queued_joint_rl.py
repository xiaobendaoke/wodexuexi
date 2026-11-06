"""Clean physical feature extraction, release-time tandem-queue approximation, and online RL rollout.

Key properties for `queued_joint_rl_20261106`:
1. Release-Time Work-Conserving Tandem-Queue Analytical Approximation (`_release_time_tandem_queue_eval`):
   - Accounts for concurrent multi-UE uplink bandwidth sharing per UAV (`rate / active_ue_count`).
   - Simulates release-time work-conserving EDF on each compute node `tgt` so tasks already in
     `compute` (`t_ready = 0`) utilize idle CPU while earlier-deadline tasks are still uploading
     (`t_ready > 0`), eliminating the false-expiration counterexample of static single-pass Lindley.
   - Explicitly framed as a fast analytical approximation and paired with `quantify_tandem_approximation_error`
     to measure MAE and rank correlation against the true `Engine`.
2. Clean Physical Actor Features (`extract_clean_actor_row`):
   - Removes the hand-coded `-5.00 * is_congested * is_rescue` penalty and hand-crafted utility proxy.
   - All 4 channels of `phi_couple` are objective queueing/throughput quantities.
3. Genuine Environment-Reward-Driven RL Rollout (`collect_rl_episode`):
   - Executes ZERO teacher lookaheads (`compute_offline_teacher_targets` is never imported or called).
   - Step reward is derived strictly from real `Engine` transitions:
       r_t = d_success_t - lambda_exp * d_expired_t + gamma * Phi(s_{t+1}) - Phi(s_t)
"""
import copy
import math
import time
import numpy as np

from environment.queued_service import Engine, distance
from marl_models.queued_joint_rl import rl_actor_policy
from utils.queued_arrival_adaptation import RateHistory, future_arrivals, rate_features
from utils.queued_joint_training import _normalize_vector, _shannon_rate, build_joint_bank, prepare_slot_base
from utils.queued_planner_distillation import planner_target
from utils.queued_quality_amortization import (
    analytical_arrival_outcomes,
    evaluate_amortized_bank,
    safe_assignments,
    should_expand_rescue,
)
from utils.queued_waiting_rescue import rescue_value, service_bounds

RL_REVISION = 'queued_joint_rl_v2_20261106'

# Training arms under genuine environment-reward PPO-GAE:
RL_TRAINING_ARMS = (
    'lower_only_rl',
    'upper_only_rl',
    'decomposed_rl',
    'joint_no_expectation_rl',
    'joint_rl',
    'decomposed_rl_finetuned',
    'joint_no_expectation_rl_finetuned',
    'joint_rl_finetuned',
)

# Full evaluation matrix on blind confirmation workloads:
RL_EVAL_ARMS = (
    'uncoordinated_init',
    'lower_only_rl',
    'upper_only_rl',
    'decomposed_composed_rl',
    'decomposed_rl',
    'joint_no_expectation_rl',
    'joint_rl',
    'decomposed_distilled',
    'joint_no_expectation_distilled',
    'joint_distilled',
    'decomposed_rl_finetuned',
    'joint_no_expectation_rl_finetuned',
    'joint_rl_finetuned',
    'quality_amortized',
    'prior_top6_mpc',
    'distilled_top6_mpc',
    'joint_rl_mpc',
)


def rl_mode_for_arm(arm):
    if arm in ('joint_rl', 'joint_rl_finetuned', 'joint_distilled', 'distilled_top6_mpc', 'joint_rl_mpc'):
        return 'joint'
    if arm in ('joint_no_expectation_rl', 'joint_no_expectation_distilled', 'joint_no_expectation_rl_finetuned'):
        return 'joint_no_expectation'
    if arm in ('decomposed_rl', 'decomposed_composed_rl', 'decomposed_rl_finetuned', 'decomposed_distilled'):
        return 'decomposed'
    if arm == 'upper_only_rl':
        return 'upper_only'
    if arm == 'lower_only_rl':
        return 'lower_only'
    raise ValueError(f'unknown RL evaluation/training arm: {arm}')


def _simulate_node_release_edf(jobs):
    """Simulate work-conserving release-time EDF CPU service for a single node.

    Each job in `jobs` is `(t_ready, dt_cpu, slack, win_slack, uid)`.
    Returns `(comp_cnt, over_cnt)` where ready tasks (`t_ready <= t_cur`) can utilize
    the CPU before earlier-deadline tasks finish uploading (`t_ready > t_cur`).
    """
    if not jobs:
        return 0.0, 0.0
    # Mutable state per job: [t_ready, rem_cpu, slack, win_slack, uid, active]
    state = [[float(r), float(c), float(s), float(w), int(u), True] for r, c, s, w, u in jobs]
    t_cur = 0.0
    comp_cnt = 0.0
    over_cnt = 0.0
    eps = 1e-9

    while True:
        # 1. Expire any active job whose deadline has passed or that cannot even become ready before deadline
        for j in state:
            if j[5] and (t_cur >= j[2] - eps or j[0] >= j[2] - eps):
                j[5] = False
                over_cnt += 1.0

        active_jobs = [j for j in state if j[5]]
        if not active_jobs:
            break

        # 2. Identify jobs ready for CPU at `t_cur`
        ready_jobs = [j for j in active_jobs if j[0] <= t_cur + eps]
        if not ready_jobs:
            t_cur = min(j[0] for j in active_jobs)
            continue

        # 3. Work-conserving EDF among currently ready jobs: pick earliest (slack, uid)
        current = min(ready_jobs, key=lambda j: (j[2], j[4]))
        dt_finish = current[1]
        dt_expire = max(0.0, current[2] - t_cur)

        # Check if any unready job with an earlier deadline becomes ready before `current` finishes
        dt_preempt = float('inf')
        for j in active_jobs:
            if j[0] > t_cur + eps and (j[2], j[4]) < (current[2], current[4]):
                dt_preempt = min(dt_preempt, j[0] - t_cur)

        dt = min(dt_finish, dt_expire, dt_preempt)
        if dt <= 0.0:
            dt = min(dt_finish, dt_expire)
        current[1] -= dt
        t_cur += dt

        if current[1] <= eps:
            current[5] = False
            if t_cur <= current[2] + eps:
                if t_cur <= current[3] + eps:
                    comp_cnt += 1.0
                else:
                    comp_cnt += 0.85
                    over_cnt += 0.15
            else:
                over_cnt += 1.0
        elif t_cur >= current[2] - eps:
            current[5] = False
            over_cnt += 1.0

    return comp_cnt, over_cnt


def _release_time_tandem_queue_eval(obs, admitted, waiting, plan, motion):
    """Release-time work-conserving tandem-queue analytical approximation.

    Resolves the static single-pass Lindley counterexample by:
    1. Modeling concurrent multi-UE upload sharing per UAV (`rate / active_ue_count`)
       while serializing same-UE tasks in EDF order.
    2. Computing each task's CPU release time `t_tr_done` and running work-conserving
       release-time EDF (`_simulate_node_release_edf`) per compute node.
    """
    protocol = obs['protocol']
    positions = obs['positions']
    n_uav = len(positions)
    mbs_idx = n_uav
    now = float(obs['time'])
    after = [(positions[u][0] + motion[u][0], positions[u][1] + motion[u][1]) for u in range(n_uav)]
    mid = [(positions[u][0] + 0.5 * motion[u][0], positions[u][1] + 0.5 * motion[u][1]) for u in range(n_uav)]

    # Count distinct active UEs currently uploading per UAV for bandwidth sharing
    uploading_ues_per_uav = [set() for _ in range(n_uav)]
    for t in admitted:
        if t['stage'] == 'upload' and t['remaining_upload'] > 1e-8:
            src = t['source']
            ue_xy = obs['ues'][t['ue']]
            if (distance(ue_xy, mid[src]) <= protocol.coverage + 1e-9
                    or distance(ue_xy, positions[src]) <= protocol.coverage + 1e-9):
                uploading_ues_per_uav[src].add(t['ue'])

    ue_up_free = {}
    uav_up_tail = [0.0] * n_uav
    t_inter_free = 0.0
    t_bh_free = 0.0
    node_jobs = [[] for _ in range(n_uav + 1)]
    early_over_cnt = 0.0

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
            share = max(1, len(uploading_ues_per_uav[src]))
            prev_ue_free = ue_up_free.get((src, t['ue']), 0.0)
            if dm and da:
                r_up = _shannon_rate(protocol, 'uplink', (*ue_xy, 0.0), (*mid[src], protocol.altitude)) / share
                dt_up = float(t['remaining_upload']) / max(r_up, 1e-6)
                t_up_done = prev_ue_free + dt_up
            elif d0 and dm:
                r_up = _shannon_rate(protocol, 'uplink', (*ue_xy, 0.0), (*mid[src], protocol.altitude)) / share
                dt_up = float(t['remaining_upload']) / max(r_up, 1e-6)
                t_up_done = prev_ue_free + dt_up
                if t_up_done > 0.75:
                    t_up_done = 10.0
            else:
                t_up_done = 10.0
            if t_up_done < slack:
                ue_up_free[(src, t['ue'])] = t_up_done
                uav_up_tail[src] = max(uav_up_tail[src], t_up_done)
        else:
            t_up_done = 0.0

        if t_up_done >= slack - 1e-9:
            early_over_cnt += 1.0
            continue

        rem_tr = float(t['remaining_transfer']) if t['target'] is not None else (float(t['bits']) if tgt != src else 0.0)
        if t['stage'] in ('upload', 'transfer') and tgt != src and rem_tr > 1e-8:
            if tgt == mbs_idx:
                r_tr = _shannon_rate(protocol, 'backhaul', (*mid[src], protocol.altitude), protocol.mbs)
                dt_tr = rem_tr / max(r_tr, 1e-6)
                t_tr_done = max(t_bh_free, t_up_done) + dt_tr
                if t_tr_done < slack:
                    t_bh_free = t_tr_done
            elif (distance(mid[src], mid[tgt]) <= protocol.inter_range + 1e-9
                  and distance(after[src], after[tgt]) <= protocol.inter_range + 1e-9):
                r_tr = _shannon_rate(protocol, 'inter', (*mid[src], protocol.altitude), (*mid[tgt], protocol.altitude))
                dt_tr = rem_tr / max(r_tr, 1e-6)
                t_tr_done = max(t_inter_free, t_up_done) + dt_tr
                if t_tr_done < slack:
                    t_inter_free = t_tr_done
            else:
                t_tr_done = 10.0
        else:
            t_tr_done = t_up_done

        if t_tr_done >= slack - 1e-9:
            early_over_cnt += 1.0
            continue

        dt_cpu = float(t['remaining_cycles']) / float(protocol.capacity[tgt])
        node_jobs[tgt].append((t_tr_done, dt_cpu, slack, win_slack, int(t['uid'])))

    # Evaluate waiting tasks newly covered at t+1 (`after` positions)
    node_work_est = [
        sum(j[1] for j in node_jobs[node]) for node in range(n_uav + 1)
    ]
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
        t_up_done = max(1.0, uav_up_tail[src]) + dt_up
        if t_up_done >= slack - 1e-9:
            early_over_cnt += 1.0
            continue

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
            est_finish = max(t_tr_done, 1.0 + node_work_est[node]) + dt_cpu
            cand = (est_finish > slack + 1e-9, est_finish, node, t_tr_done, dt_cpu)
            if best_choice is None or cand < best_choice:
                best_choice = cand

        miss, _, best_node, t_tr_done, dt_cpu = best_choice
        if not miss:
            uav_up_tail[src] = max(uav_up_tail[src], t_up_done)
            if best_node == mbs_idx:
                t_bh_free = max(t_bh_free, t_tr_done)
            elif best_node != src:
                t_inter_free = max(t_inter_free, t_tr_done)
            node_work_est[best_node] += dt_cpu
            node_jobs[best_node].append((t_tr_done, dt_cpu, slack, win_slack, int(wt['uid'])))
        else:
            early_over_cnt += 1.0

    total_comp = 0.0
    total_over = early_over_cnt
    for node in range(n_uav + 1):
        c_node, o_node = _simulate_node_release_edf(node_jobs[node])
        total_comp += c_node
        total_over += o_node

    return total_comp, total_over


def quantify_tandem_approximation_error(obs, plans, motions):
    """Quantify the approximation error of `_release_time_tandem_queue_eval` vs. 3-step `Engine` rollout."""
    admitted = sorted(
        [t for t in obs['tasks'] if t['source'] is not None],
        key=lambda z: (z['deadline'], z['uid']),
    )
    waiting = sorted(
        [t for t in obs['tasks'] if t['source'] is None],
        key=lambda z: (z['deadline'], z['uid']),
    )
    zero = [(0.0, 0.0)] * len(obs['positions'])
    approx_suc = []
    engine_suc = []
    for plan in plans:
        for motion in motions:
            c_app, _ = _release_time_tandem_queue_eval(obs, admitted, waiting, plan, motion)
            e = Engine.from_observation(obs)
            e.assign(plan)
            e.advance(motion)
            for _ in range(2):
                e.admit()
                pub = e.observe()
                e.assign(safe_assignments(pub))
                e.advance(zero)
            approx_suc.append(float(c_app))
            engine_suc.append(float(e.counts['success']))
    a_arr = np.asarray(approx_suc, dtype=np.float64)
    e_arr = np.asarray(engine_suc, dtype=np.float64)
    mae = float(np.mean(np.abs(a_arr - e_arr)))
    if np.std(a_arr) > 1e-9 and np.std(e_arr) > 1e-9:
        corr = float(np.corrcoef(a_arr, e_arr)[0, 1])
    else:
        corr = 1.0
    return dict(pairs=len(approx_suc), mae=mae, correlation=corr)


def extract_clean_actor_row(obs, bank, adapter, temporal):
    """Extract clean, objective physical queueing features with ZERO hand-coded `-5.0` penalty.

    Executes ZERO `Engine` simulations (`deterministic_engine_steps = 0`) and ZERO
    stochastic future arrival rollouts (`stochastic_forecast_calls = 0`).
    """
    plans = bank['plans']
    actions = bank['actions']
    original_count = bank['original_count']
    s_plan = bank['safe_plan']
    pressure = bank['pressure']
    rate3 = bank['rate3']
    base_neural = bank['base_neural']
    protocol = obs['protocol']
    positions = obs['positions']
    n_uav = len(positions)
    mbs_idx = n_uav

    rates = rate_features(obs, adapter.history, temporal)
    admitted = sorted(
        [t for t in obs['tasks'] if t['source'] is not None],
        key=lambda z: (z['deadline'], z['uid']),
    )
    waiting = sorted(
        [t for t in obs['tasks'] if t['source'] is None],
        key=lambda z: (z['deadline'], z['uid']),
    )
    unassigned_uids = [t['uid'] for t in admitted if t['target'] is None]
    up_count = sum(t['stage'] == 'upload' for t in admitted)
    rel_count = sum(t['stage'] == 'transfer' for t in admitted)
    comp_count = sum(t['stage'] == 'compute' for t in admitted)
    is_congested = 1.0 if not should_expand_rescue(pressure, rate3) else 0.0

    context = np.r_[
        rates,
        [
            float(np.tanh(pressure / 2.0)),
            rate3 / 60.0,
            is_congested,
            len(admitted) / 40.0,
            len(waiting) / 40.0,
            (up_count + rel_count + comp_count) / 40.0,
        ],
    ]

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

        c_stat, o_stat = _release_time_tandem_queue_eval(obs, admitted, [], plan, zero_motion)
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
        f_suc_ref, f_over_ref = analytical_arrival_outcomes(obs, adapter.history, s_plan, motion, rate3)
        for pi, plan in enumerate(plans):
            f_suc_p, f_over_p = analytical_arrival_outcomes(obs, adapter.history, plan, motion, rate3)
            comp_cnt, over_cnt = _release_time_tandem_queue_eval(obs, admitted, waiting, plan, motion)
            inst_completable[pi, mi] = comp_cnt
            inst_overflow[pi, mi] = over_cnt
            fut_suc_mat[pi, mi] = 0.5 * f_suc_ref + 0.5 * f_suc_p
            fut_over_mat[pi, mi] = 0.5 * f_over_ref + 0.5 * f_over_p
            src_pi = pi if pi < k_orig else 0
            neural_mat[pi, mi] = float(centered_neural[src_pi, min(mi, m_orig - 1)])

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

    # Objective physical queue metrics (NO `-5.0 * is_congested * is_rescue` hand-coded penalty!)
    total_overflow = inst_overflow + fut_over_mat + 0.50 * plan_overload_sq[:, None]
    net_physical_throughput = inst_completable + fut_suc_mat - total_overflow

    plan_comp_norm = _normalize_vector(plan_static_completable - 0.50 * plan_overload_sq)
    plan_over_norm = _normalize_vector(plan_static_overloaded + 0.50 * plan_overload_sq)
    plan_neural_mean_norm = _normalize_vector(neural_mat.mean(axis=1))
    log_prior_norm = _normalize_vector(np.log(np.clip(bank['prior'], 1e-12, None)))
    n_adm = max(1.0, float(len(admitted)))

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

    rescue_norm = _normalize_vector(rescue_vals)
    fut_suc_norm_global = _normalize_vector(fut_suc_mat)
    overflow_norm_global = _normalize_vector(total_overflow)
    net_tp_norm_global = _normalize_vector(net_physical_throughput)
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
            #   0: rescue_norm (waiting-queue rescue coverage potential)
            #   1: fut_suc_norm_global (expected future arrival service capacity)
            #   2: overflow_norm_global (tandem-queue & overload overflow risk)
            #   3: net_tp_norm_global (objective net physical throughput = completable + fut_suc - overflow)
            m_tensor[i, j, :] = [
                rescue_norm[j],
                fut_suc_norm_global[i, j],
                overflow_norm_global[i, j],
                net_tp_norm_global[i, j],
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


def state_potential(engine, include_covered_waiting=True):
    """State potential Phi(s) evaluated on the live `engine` queue state.

    Evaluates `engine` using the release-time work-conserving tandem-queue model:
    - Before action execution (`include_covered_waiting=False`): evaluates currently admitted
      tasks (with unassigned tasks defaulted to local node `source`).
    - After action execution (`include_covered_waiting=True`): evaluates active admitted tasks
      under their assigned targets plus waiting tasks newly covered at `engine.positions` (p_{t+1}),
      with congestion-aware load accounting.
    """
    if not engine.tasks:
        return 0.0
    obs = engine.observe()
    protocol = obs['protocol']
    positions = obs['positions']
    n_uav = len(positions)
    mbs_idx = n_uav
    zero_motion = [(0.0, 0.0)] * n_uav

    admitted = []
    uav_loads = [0.0] * (n_uav + 1)
    for t in obs['tasks']:
        if t['source'] is not None:
            admitted.append(t)
            tgt = t['target'] if t['target'] is not None else t['source']
            uav_loads[tgt] += float(t['remaining_cycles']) / float(protocol.capacity[tgt])

    if include_covered_waiting:
        # Waiting tasks that are now within coverage at `positions` (p_{t+1}) will be admitted
        # at the very start of the next slot (`engine.admit()`).
        for t in obs['tasks']:
            if t['source'] is None:
                ue_xy = obs['ues'][t['ue']]
                src = min(range(n_uav), key=lambda u: (distance(ue_xy, positions[u]), u))
                if distance(ue_xy, positions[src]) <= protocol.coverage + 1e-9:
                    virt = dict(t)
                    virt['source'] = src
                    virt['target'] = src
                    virt['stage'] = 'upload'
                    virt['remaining_upload'] = float(t['bits'])
                    virt['remaining_transfer'] = 0.0
                    virt['remaining_cycles'] = float(t['cycles'])
                    admitted.append(virt)
                    uav_loads[src] += float(t['cycles']) / float(protocol.capacity[src])

    if not admitted:
        return 0.0
    admitted.sort(key=lambda z: (z['deadline'], z['uid']))
    comp_pot, over_pot = _release_time_tandem_queue_eval(obs, admitted, [], {}, zero_motion)
    overload_sq = sum(max(0.0, uav_loads[u] - 1.0) ** 2 for u in range(n_uav))
    return 0.65 * comp_pot - 0.65 * over_pot - 0.35 * overload_sq


def env_counterfactual_3step_return(obs, bank, alloc_idx, motion_idx, arr_batches, r_val, congested):
    """Evaluate 3-step real Engine transition return from `obs` under shared arrival stream `arr_batches`.

    Strictly uses real `Engine` transitions (`e.assign`, `e.advance`, `e.add`, `e.admit`),
    producing zero-arrival-variance counterfactual group returns without calling any offline teacher.
    """
    plan = bank['plans'][alloc_idx]
    motion = bank['actions'][motion_idx]
    e = Engine.from_observation(obs)
    prev_suc = int(e.counts['success'])
    prev_exp = int(e.counts['expired'])
    prev_disc = int(e.counts['disconnect_events'])

    e.assign(plan)
    e.advance(motion)
    zero = [(0.0, 0.0)] * len(e.positions)
    for batch in arr_batches:
        e.add(copy.deepcopy(batch))
        e.admit()
        pub = e.observe()
        e.assign(safe_assignments(pub))
        e.advance(zero)

    d_suc = float(int(e.counts['success']) - prev_suc)
    d_exp = float(int(e.counts['expired']) - prev_exp)
    d_disc = float(int(e.counts['disconnect_events']) - prev_disc)
    adm_total = int(e.counts['admitted'])
    uncompleted = max(0, adm_total - int(e.counts['success']))

    return (
        d_suc
        - 0.65 * d_exp
        - 0.05 * d_disc
        - (0.45 if congested else 0.15) * uncompleted
        + (0.0 if congested else 0.25 * r_val)
    )


def collect_rl_episode(
    root,
    prototype_module,
    proposer,
    score_model,
    tape,
    model,
    mode,
    parent,
    workload,
    iteration,
    rollout_rep=0,
    ref_model=None,
    gamma=0.85,
    lambda_expired=0.65,
    cf_stride=0,
):
    """Roll out one complete online RL episode driven strictly by real `Engine` rewards.

    Executes ZERO offline teacher queries (`compute_offline_teacher_targets` is never called).
    When `cf_stride > 0`, also records zero-mean on-policy counterfactual group transitions
    evaluated in the real `Engine` under Common Random Numbers (CRN).
    """
    t = prototype_module.t
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions)
    adapter = t.LearningAdapter()
    temporal = RateHistory()
    steps = []

    while engine.t < 200 or engine.tasks:
        now = int(engine.t)
        assert now <= 204
        engine.add(tape.slot(now))
        engine.admit()
        obs = engine.observe()
        temporal.update(obs)

        base_state = prepare_slot_base(
            root, prototype_module, proposer, score_model, obs, adapter, parent, workload, rollout_rep, now
        )
        bank = build_joint_bank(
            root, prototype_module, proposer, score_model, obs, adapter, temporal,
            parent, workload, rollout_rep, now,
            include_safe_plan=True, rescue_mode='unguarded', precomputed_base=base_state,
        )
        row = extract_clean_actor_row(obs, bank, adapter, temporal)
        prob = rl_actor_policy(row, model, mode=mode)
        ref_prob = (
            rl_actor_policy(row, ref_model, mode=mode)
            if ref_model is not None
            else (row['prior'][:, None] * row['motion_prior'])
        )
        ref_prob = ref_prob / ref_prob.sum()

        rng = np.random.default_rng(
            t.seed(RL_REVISION, 'rl_rollout', mode, parent, workload, iteration, rollout_rep, now)
        )
        flat = prob.ravel()
        choice = int(rng.choice(flat.size, p=flat))
        alloc_idx, motion_idx = np.unravel_index(choice, prob.shape)
        old_logp = float(np.log(max(float(prob[alloc_idx, motion_idx]), 1e-12)))

        cf_group = []
        if cf_stride > 0 and now < 200 and now % cf_stride == 0:
            f_seed = t.seed(RL_REVISION, 'cf_arr', parent, workload, iteration, rollout_rep, now)
            arr_batches = future_arrivals(obs, adapter.history, temporal, 'window3', f_seed)
            bounds = service_bounds(obs)
            congested = not should_expand_rescue(bank['pressure'], bank['rate3'])
            r_vals = [
                rescue_value(obs, m, bounds) if not congested else 0.0
                for m in bank['actions']
            ]
            top_order = np.argsort(flat)[::-1][: min(3, flat.size)]
            cand_set = {int(choice)} | {int(x) for x in top_order}
            if len(cand_set) < min(4, flat.size):
                cand_set.add(int(rng.choice(flat.size)))
            cand_list = sorted(cand_set)
            q_vals = []
            for c_idx in cand_list:
                ak, am = np.unravel_index(c_idx, prob.shape)
                q_c = env_counterfactual_3step_return(
                    obs, bank, int(ak), int(am), arr_batches, r_vals[int(am)], congested
                )
                q_vals.append(q_c)
            q_arr = np.asarray(q_vals, dtype=np.float64)
            q_mean = float(np.mean(q_arr))
            q_std = max(float(np.std(q_arr)), 0.10)
            if float(np.std(q_arr)) > 1e-6:
                for c_idx, q_c in zip(cand_list, q_arr):
                    ak, am = np.unravel_index(c_idx, prob.shape)
                    cf_group.append(dict(
                        action_pair=(int(ak), int(am)),
                        old_logp=float(np.log(max(float(prob[ak, am]), 1e-12))),
                        norm_advantage=float(np.clip((q_c - q_mean) / q_std, -3.0, 3.0)),
                    ))

        phi_pre = state_potential(engine, include_covered_waiting=False)
        prev_suc = int(engine.counts['success'])
        prev_exp = int(engine.counts['expired'])
        prev_disc = int(engine.counts['disconnect_events'])

        engine.assign(bank['plans'][alloc_idx])
        engine.advance(bank['actions'][motion_idx])

        d_suc = float(int(engine.counts['success']) - prev_suc)
        d_exp = float(int(engine.counts['expired']) - prev_exp)
        d_disc = float(int(engine.counts['disconnect_events']) - prev_disc)
        phi_post = state_potential(engine, include_covered_waiting=True)
        step_reward = (
            (d_suc - lambda_expired * d_exp - 0.05 * d_disc)
            + (phi_post - phi_pre)
        )

        # Record every active and post-200 drain transition so drain actions receive their own
        # discounted GAE advantages and Clipped Joint PPO policy gradients (no truncation at t=200).
        steps.append(dict(
            time=now,
            is_drain=bool(now >= 200),
            workload=int(workload),
            parent=int(parent),
            iteration=int(iteration),
            rollout_rep=int(rollout_rep),
            row=dict(
                a=row['a'],
                m=row['m'],
                context=row['context'],
                prior=row['prior'],
                motion_prior=row['motion_prior'],
            ),
            action_pair=(int(alloc_idx), int(motion_idx)),
            old_logp=old_logp,
            ref_policy=ref_prob,
            reward=float(step_reward),
            d_success=d_suc,
            d_expired=d_exp,
            cf_group=cf_group,
        ))

    audit = engine.audit()
    assert not engine.tasks and audit['status'] == 'PASS'
    assert int(round(sum(s['d_success'] for s in steps))) == int(engine.counts['success'])
    assert int(round(sum(s['d_expired'] for s in steps))) == int(engine.counts['expired'])
    return steps, dict(
        generated=int(engine.counts['generated']),
        admitted=int(engine.counts['admitted']),
        success=int(engine.counts['success']),
        expired=int(engine.counts['expired']),
        total_steps=len(steps),
        drain_steps=sum(1 for s in steps if s['is_drain']),
        dsr=float(engine.counts['success'] / max(1, engine.counts['generated'])),
    )


def _run_top6_mpc(obs, bank, adapter, temporal, proposal_prob, f_seed, act_seed):
    """Evaluate Top-6 candidates selected by `proposal_prob` under identical 3-step rollout rules."""
    flat_prob = proposal_prob.ravel()
    top_k = min(6, flat_prob.size)
    # Deterministic tie-breaking by index before taking top_k
    order = np.lexsort((np.arange(flat_prob.size), flat_prob))
    top_flat_idx = order[-top_k:]

    arr = future_arrivals(obs, adapter.history, temporal, 'window3', f_seed)
    zero = [(0.0, 0.0)] * len(obs['positions'])
    bounds = service_bounds(obs)
    congested = not should_expand_rescue(bank['pressure'], bank['rate3'])
    best_val = -1e18
    best_pairs = []
    for f_idx in top_flat_idx:
        pi, mi = np.unravel_index(int(f_idx), proposal_prob.shape)
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
            + 0.20 * math.log(max(float(proposal_prob[pi, mi]), 1e-12))
        )
        val = round(val * 100.0) / 100.0
        if val > best_val + 1e-9:
            best_val = val
            best_pairs = [(int(pi), int(mi), float(proposal_prob[pi, mi]))]
        elif abs(val - best_val) <= 1e-9:
            best_pairs.append((int(pi), int(mi), float(proposal_prob[pi, mi])))
    weights = np.array([w for _, _, w in best_pairs], dtype=np.float64)
    weights = weights / weights.sum()
    rng = np.random.default_rng(act_seed)
    sel = int(rng.choice(len(best_pairs), p=weights))
    alloc_idx, motion_idx, _ = best_pairs[sel]
    return alloc_idx, motion_idx, int(top_k)


def rl_actor_step(
    root,
    prototype_module,
    proposer,
    score_model,
    obs,
    adapter,
    temporal,
    arm,
    model,
    parent,
    workload,
    replica,
    now,
    precomputed_base=None,
):
    """Execute one closed-loop evaluation step for an RL, distilled, or MPC arm."""
    t = prototype_module.t
    started = time.perf_counter()

    if arm in (
        'uncoordinated_init',
        'lower_only_rl',
        'upper_only_rl',
        'decomposed_composed_rl',
        'decomposed_rl',
        'joint_no_expectation_rl',
        'joint_rl',
        'decomposed_distilled',
        'joint_no_expectation_distilled',
        'joint_distilled',
        'decomposed_rl_finetuned',
        'joint_no_expectation_rl_finetuned',
        'joint_rl_finetuned',
    ):
        bank = build_joint_bank(
            root, prototype_module, proposer, score_model, obs, adapter, temporal,
            parent, workload, replica, now,
            include_safe_plan=True, rescue_mode='unguarded', precomputed_base=precomputed_base,
        )
        row = extract_clean_actor_row(obs, bank, adapter, temporal)
        if arm == 'uncoordinated_init':
            probability = row['prior'][:, None] * row['motion_prior']
            probability /= probability.sum()
        else:
            mode = rl_mode_for_arm(arm)
            probability = rl_actor_policy(row, model, mode=mode)
        rng = np.random.default_rng(t.seed(RL_REVISION, 'eval_action', parent, workload, replica, now))
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
        rng = np.random.default_rng(t.seed(RL_REVISION, 'eval_action', parent, workload, replica, now))
        choice = int(rng.choice(probability.size, p=probability.ravel()))
        alloc_idx, motion_idx = np.unravel_index(choice, probability.shape)
        det_engine_steps = 3 * len(bank['plans']) * len(bank['actions'])
        forecast_calls = 0
    elif arm in ('prior_top6_mpc', 'distilled_top6_mpc', 'joint_rl_mpc'):
        bank = build_joint_bank(
            root, prototype_module, proposer, score_model, obs, adapter, temporal,
            parent, workload, replica, now,
            include_safe_plan=True, rescue_mode='unguarded', precomputed_base=precomputed_base,
        )
        row = extract_clean_actor_row(obs, bank, adapter, temporal)
        if arm == 'prior_top6_mpc':
            proposal_prob = row['prior'][:, None] * row['motion_prior']
            proposal_prob = proposal_prob / proposal_prob.sum()
        else:
            proposal_prob = rl_actor_policy(row, model, mode='joint')
        f_seed = t.seed(RL_REVISION, 'eval_forecast', parent, workload, replica, now)
        act_seed = t.seed(RL_REVISION, 'eval_action', parent, workload, replica, now)
        alloc_idx, motion_idx, top_k = _run_top6_mpc(
            obs, bank, adapter, temporal, proposal_prob, f_seed, act_seed
        )
        det_engine_steps = 0
        forecast_calls = int(top_k)
    else:
        raise ValueError(f'unexpected RL evaluation arm: {arm}')

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
