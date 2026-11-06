"""Two-branch candidate-equivariant Actor-Critic for true environment-reward-driven Joint RL (PPO-GAE).

Implements genuine reinforcement learning driven strictly by online environment trajectory rewards,
Generalized Advantage Estimation (GAE), a state-value Critic V_psi(s), and Clipped Joint PPO policy
gradients with exact cross-layer backpropagation through the 4D upper-policy expectation vector:
    C_upper[k] = E_{m ~ pi_upper(. | s, k)}[phi_couple(s, k, m)]

Supports five unconfounded modes on a shared neural architecture:
- 'joint': plan-conditioned motion features + 4D upper expectation vector C_upper[k] + exact cross-layer
  policy gradient from log p_a(k* | s) through C_upper[k] to upper motion parameters theta_motion.
- 'joint_no_expectation': identical plan-conditioned motion features as 'joint', but sets C_upper[k] = 0
  and cuts off cross-layer backpropagation. Comparing 'joint' vs 'joint_no_expectation' isolates the
  exact causal contribution of C_upper and cross-layer backpropagation.
- 'decomposed': retains all global congestion and upper motion features (never masking global `is_congested`
  or `is_rescue`), replaces only lower-plan-specific channels with their across-plan mean so upper motion
  is unconditioned on the lower plan k, and sets C_upper[k] = 0.
- 'upper_only': updates only the upper motion branch via PPO while keeping the lower branch at its prior.
- 'lower_only': updates only the lower allocation branch via PPO while keeping the upper branch at its prior.
"""
import copy
import numpy as np

from utils.queued_service_learning import finite_gae

RL_MODES = ('joint', 'joint_no_expectation', 'decomposed', 'upper_only', 'lower_only')
COUPLING_DIM = 4
# All 8 channels of `m` that depend on the specific lower allocation plan `k` are averaged across
# plans (weighted by `prior`) in uncoupled modes (`decomposed`, `upper_only`, `lower_only`), and the
# upper motion prior is replaced by the plan-marginalized prior `sum_k prior[k] * motion_prior[k, m]`.
# This guarantees exact mathematical independence `p_m(m | s, k) == p_m(m | s, k')` for all k, k'
# while keeping global congestion (channel 9) and rescue indicators (channels 0, 8, 10, 11, 12) visible.
PLAN_DEPENDENT_MOTION_CHANNELS = (1, 2, 3, 4, 5, 13, 14, 15)


def _softmax(logits, axis=-1):
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _log_softmax(logits, axis=-1):
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    active = p > 0
    if np.any(q[active] <= 0):
        return float('inf')
    return float(np.sum(p[active] * (np.log(p[active]) - np.log(q[active]))))


def _validate_row(row):
    a = np.asarray(row['a'], dtype=np.float64)
    m = np.asarray(row['m'], dtype=np.float64)
    c = np.asarray(row['context'], dtype=np.float64)
    prior = np.asarray(row['prior'], dtype=np.float64)
    motion_prior = np.asarray(row['motion_prior'], dtype=np.float64)
    if (a.ndim != 2 or m.ndim != 3 or c.ndim != 1 or prior.ndim != 1
            or motion_prior.shape != m.shape[:2] or a.shape[0] != m.shape[0]
            or m.shape[2] < COUPLING_DIM
            or prior.shape != (a.shape[0],) or np.any(prior <= 0)
            or np.any(motion_prior <= 0)
            or not np.isclose(prior.sum(), 1.0)
            or not np.allclose(motion_prior.sum(axis=1), 1.0)
            or not all(np.isfinite(arr).all() for arr in (a, m, c, prior, motion_prior))):
        raise ValueError('invalid or nonfinite actor row features')
    return a, m, c, prior, motion_prior


def extract_critic_state(row):
    """Construct state feature vector x_v for the state-value Critic V_psi(s)."""
    a, m, c, prior, motion_prior = _validate_row(row)
    p0 = prior[:, None] * motion_prior
    p0 = p0 / p0.sum()
    a_expected = np.sum(prior[:, None] * a, axis=0)
    a_max = np.max(a, axis=0)
    m_expected = np.sum(p0[:, :, None] * m, axis=(0, 1))
    m_max = np.max(m[:, :, :COUPLING_DIM], axis=(0, 1))
    return np.concatenate([c, a_expected[:6], a_max[:4], m_expected[:6], m_max], axis=0)


def prepare_rl_branch_inputs(row, mode='joint', pm_override=None):
    """Construct allocation inputs `xa` (K, Da+4+Dc) and motion inputs `xm` (K, M, Dm+Da+Dc).

    - In `joint`:
        `xm` conditions on plan `a[k]` and full plan-motion physics `m[k, m]`.
        `xa` includes the 4D upper-policy expectation vector `C_upper[k] = sum_m pm[k, m] * phi_couple[k, m]`.
    - In `joint_no_expectation`:
        `xm` is identical to `joint` (plan-conditioned), while `xa` sets `C_upper[k] = 0`.
    - In `decomposed`, `upper_only`, `lower_only`:
        `xm` retains all global context and upper motion channels (including `is_congested` and `is_rescue`),
        replacing only lower-plan-specific channels with their across-plan average. `xa` sets `C_upper[k] = 0`.
    """
    if mode not in RL_MODES:
        raise ValueError(f'unknown RL training mode: {mode}')
    a, m, c, prior, motion_prior = _validate_row(row)
    k, n, dm = m.shape
    da = a.shape[1]
    dc = len(c)

    if mode in ('joint', 'joint_no_expectation'):
        alloc_ctx = np.broadcast_to(a[:, None, :], (k, n, da))
        xm = np.concatenate([m, alloc_ctx, np.broadcast_to(c, (k, n, dc))], axis=2)
        eff_motion_prior = motion_prior
    else:
        w_plan = prior[:, None]
        w_plan3 = prior[:, None, None]
        a_mean = np.broadcast_to(np.sum(w_plan * a, axis=0, keepdims=True), a.shape)
        m_uncond = m.copy()
        plan_mean_m = np.broadcast_to(np.sum(w_plan3 * m, axis=0, keepdims=True), m.shape)
        for ch in PLAN_DEPENDENT_MOTION_CHANNELS:
            if ch < dm:
                m_uncond[:, :, ch] = plan_mean_m[:, :, ch]
        alloc_ctx = np.broadcast_to(a_mean[:, None, :], (k, n, da))
        xm = np.concatenate([m_uncond, alloc_ctx, np.broadcast_to(c, (k, n, dc))], axis=2)
        mp_marginal = np.sum(w_plan * motion_prior, axis=0, keepdims=True)
        mp_marginal = np.clip(mp_marginal, 1e-12, None)
        mp_marginal = mp_marginal / np.sum(mp_marginal, axis=1, keepdims=True)
        eff_motion_prior = np.broadcast_to(mp_marginal, motion_prior.shape).copy()

    if mode == 'joint':
        phi_couple = m[:, :, :COUPLING_DIM]
        weights = eff_motion_prior if pm_override is None else np.asarray(pm_override, dtype=np.float64)
        c_upper = np.sum(weights[:, :, None] * phi_couple, axis=1)
    else:
        phi_couple = np.zeros((k, n, COUPLING_DIM), dtype=np.float64)
        c_upper = np.zeros((k, COUPLING_DIM), dtype=np.float64)

    xa = np.concatenate([a, c_upper, np.broadcast_to(c, (k, dc))], axis=1)
    return xa, xm, prior, eff_motion_prior, phi_couple


def initialize_rl_actor_critic(row, hidden=48, critic_hidden=48, seed=0):
    """Initialize two-branch neural Actor and state-value Critic V_psi(s)."""
    xa, xm, _, _, _ = prepare_rl_branch_inputs(row, mode='joint')
    xv = extract_critic_state(row)
    rng = np.random.default_rng(seed)
    params = {}
    for name, dim in (('allocation', xa.shape[-1]), ('motion', xm.shape[-1])):
        params[name] = dict(
            w=np.zeros(dim, dtype=np.float64),
            W=rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, hidden)),
            b=np.zeros(hidden, dtype=np.float64),
            v=np.zeros(hidden, dtype=np.float64),
        )
    dv = len(xv)
    critic = dict(
        w=np.zeros(dv, dtype=np.float64),
        W=rng.normal(0.0, 1.0 / np.sqrt(dv), size=(dv, critic_hidden)),
        b=np.zeros(critic_hidden, dtype=np.float64),
        v=np.zeros(critic_hidden, dtype=np.float64),
        c0=np.zeros(1, dtype=np.float64),
    )
    return dict(
        parameters=params,
        critic=critic,
        hidden=hidden,
        critic_hidden=critic_hidden,
        context_dim=len(row['context']),
        coupling_dim=COUPLING_DIM,
        input_dimensions=[int(xa.shape[-1]), int(xm.shape[-1])],
        critic_dim=int(dv),
    )


def warmstart_from_distilled_actor(distilled_model, row, critic_hidden=48, seed=0):
    """Initialize an Actor-Critic model warm-started from a frozen teacher-distilled actor checkpoint."""
    model = initialize_rl_actor_critic(
        row, hidden=int(distilled_model.get('hidden', 48)), critic_hidden=critic_hidden, seed=seed
    )
    for branch in ('allocation', 'motion'):
        for k in ('w', 'W', 'b', 'v'):
            src = np.asarray(distilled_model['parameters'][branch][k], dtype=np.float64)
            if src.shape != model['parameters'][branch][k].shape:
                raise ValueError(f'warmstart shape mismatch on {branch}.{k}: {src.shape} vs {model["parameters"][branch][k].shape}')
            model['parameters'][branch][k] = src.copy()
    model['warmstarted_from'] = 'distilled_checkpoint'
    return model


def compose_decomposed_rl_actor(lower_only_model, upper_only_model):
    """Directly assemble a decomposed RL actor from independently RL-trained lower and upper models."""
    if list(lower_only_model['input_dimensions']) != list(upper_only_model['input_dimensions']):
        raise ValueError('cannot compose RL models with mismatched input dimensions')
    composed = copy.deepcopy(lower_only_model)
    composed['parameters']['allocation'] = copy.deepcopy(lower_only_model['parameters']['allocation'])
    composed['parameters']['motion'] = copy.deepcopy(upper_only_model['parameters']['motion'])
    composed['composed_from'] = ('lower_only_rl', 'upper_only_rl')
    return composed


def _branch_score(x, branch):
    h = np.tanh(x @ branch['W'] + branch['b'])
    score = x @ branch['w'] + h @ branch['v']
    return score, h


def _branch_backward(x, h, grad_output, branch):
    flat_x = x.reshape(-1, x.shape[-1])
    flat_h = h.reshape(-1, h.shape[-1])
    flat_g = grad_output.reshape(-1)
    dw = flat_x.T @ flat_g
    dv = flat_h.T @ flat_g
    dh = (flat_g[:, None] * branch['v'][None, :]) * (1.0 - flat_h * flat_h)
    dW = flat_x.T @ dh
    db = dh.sum(axis=0)
    dx = (flat_g[:, None] * branch['w'][None, :] + dh @ branch['W'].T).reshape(x.shape)
    return dict(w=dw, W=dW, b=db, v=dv), dx


def rl_actor_forward(row, model, mode='joint'):
    """Forward pass computing lower allocation distribution `pa` and upper motion `pm`."""
    _, xm, prior, motion_prior, phi_couple = prepare_rl_branch_inputs(row, mode=mode)
    params = model['parameters']
    if int(xm.shape[-1]) != int(model['input_dimensions'][1]):
        raise ValueError('motion branch input dimension mismatch')

    zm_raw, hm = _branch_score(xm, params['motion'])
    if mode == 'lower_only':
        zm_raw = np.zeros_like(zm_raw)
    zm = np.log(np.clip(motion_prior, 1e-12, None)) + zm_raw
    pm = _softmax(zm, axis=-1)

    xa, _, _, _, _ = prepare_rl_branch_inputs(row, mode=mode, pm_override=pm)
    if int(xa.shape[-1]) != int(model['input_dimensions'][0]):
        raise ValueError('allocation branch input dimension mismatch')

    za_raw, ha = _branch_score(xa, params['allocation'])
    if mode == 'upper_only':
        za_raw = np.zeros_like(za_raw)
    za = np.log(np.clip(prior, 1e-12, None)) + za_raw
    pa = _softmax(za, axis=-1)

    return pa, pm, za, zm, xa, xm, ha, hm, phi_couple


def rl_actor_policy(row, model, mode='joint'):
    pa, pm, *_ = rl_actor_forward(row, model, mode=mode)
    return pa[:, None] * pm


def critic_forward(row, model):
    """Evaluate state-value Critic V_psi(s)."""
    xv = extract_critic_state(row)
    c_params = model['critic']
    hv = np.tanh(xv @ c_params['W'] + c_params['b'])
    val = float(xv @ c_params['w'] + hv @ c_params['v'] + c_params['c0'][0])
    return val, xv, hv


def critic_loss_and_gradient(row, model, target_return, huber_delta=5.0, l2_reg=1e-4):
    """Compute smooth Huber value loss and analytical gradients for Critic V_psi(s)."""
    val, xv, hv = critic_forward(row, model)
    c_params = model['critic']
    diff = val - float(target_return)
    abs_diff = abs(diff)
    if abs_diff <= huber_delta:
        loss = 0.5 * diff * diff
        dval = diff
    else:
        loss = huber_delta * (abs_diff - 0.5 * huber_delta)
        dval = huber_delta * (1.0 if diff > 0 else -1.0)

    dw = dval * xv + l2_reg * c_params['w']
    dv = dval * hv + l2_reg * c_params['v']
    dh = (dval * c_params['v']) * (1.0 - hv * hv)
    dW = np.outer(xv, dh) + l2_reg * c_params['W']
    db = dh
    dc0 = np.array([dval], dtype=np.float64)

    reg_loss = 0.5 * l2_reg * float(
        np.sum(c_params['w'] ** 2) + np.sum(c_params['v'] ** 2) + np.sum(c_params['W'] ** 2)
    )
    return float(loss + reg_loss), dict(w=dw, W=dW, b=db, v=dv, c0=dc0)


def ppo_actor_loss_and_gradient(
    row,
    model,
    action_pair,
    old_logp,
    advantage,
    mode='joint',
    clip_eps=0.20,
    ent_coef=0.002,
    kl_coef=0.02,
    ref_policy=None,
    l2_reg=1e-5,
):
    """Compute Clipped Joint PPO loss and exact analytical gradients across upper and lower branches.

    Includes:
    1. Clipped PPO policy gradient surrogate `-min(rho * A, clip(rho, 1-eps, 1+eps) * A)`
       for the environment-sampled joint action `(k_star, m_star)`.
    2. Entropy regularization `-ent_coef * H(P_theta)` over valid joint candidates.
    3. Optional trust-region KL anchor `kl_coef * KL(P_ref || P_theta)`.
    4. Exact cross-layer chain-rule gradient from lower allocation logits `za` through
       `C_upper[k] = sum_m pm(m|k) phi_couple[k, m]` into upper motion logits `zm` when `mode == 'joint'`.
    """
    pa, pm, za, zm, xa, xm, ha, hm, phi_couple = rl_actor_forward(row, model, mode=mode)
    k_star, m_star = int(action_pair[0]), int(action_pair[1])
    adv = float(advantage)
    old_lp = float(old_logp)

    log_pa = _log_softmax(za, axis=-1)
    log_pm = _log_softmax(zm, axis=-1)
    log_p_joint = log_pa[:, None] + log_pm
    p_joint = pa[:, None] * pm

    new_lp = float(log_p_joint[k_star, m_star])
    ratio = float(np.exp(np.clip(new_lp - old_lp, -20.0, 20.0)))
    clipped_ratio = float(np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps))

    u1 = ratio * adv
    u2 = clipped_ratio * adv
    if u1 <= u2:
        surr_loss = -u1
        d_new_lp = -ratio * adv
    else:
        surr_loss = -u2
        d_new_lp = 0.0

    # Gradient table w.r.t. joint log-probabilities `log_p_joint[k, m] = log_pa[k] + log_pm[k, m]`
    g_log_joint = np.zeros_like(p_joint)
    g_log_joint[k_star, m_star] += d_new_lp

    ent_loss = 0.0
    if ent_coef != 0.0:
        # -ent_coef * H(P) = ent_coef * sum_{k,m} P[k,m] * log P[k,m]
        ent_loss = float(ent_coef * np.sum(p_joint * log_p_joint))
        g_log_joint += ent_coef * p_joint * (1.0 + log_p_joint)

    kl_loss = 0.0
    if kl_coef != 0.0 and ref_policy is not None:
        p_ref = np.asarray(ref_policy, dtype=np.float64)
        p_ref = np.clip(p_ref, 1e-12, None)
        p_ref = p_ref / p_ref.sum()
        kl_loss = float(kl_coef * np.sum(p_ref * (np.log(p_ref) - log_p_joint)))
        g_log_joint -= kl_coef * p_ref

    # Project `g_log_joint` to `za` and `zm` via log-softmax Jacobians
    if mode != 'upper_only':
        g_log_pa = np.sum(g_log_joint, axis=1)
        grad_za = g_log_pa - pa * float(np.sum(g_log_pa))
    else:
        grad_za = np.zeros_like(pa)

    params = model['parameters']
    grad_alloc, dxa = _branch_backward(xa, ha, grad_za, params['allocation'])

    if mode != 'lower_only':
        grad_zm_direct = g_log_joint - pm * np.sum(g_log_joint, axis=1, keepdims=True)
        if mode == 'joint':
            da = np.asarray(row['a'], dtype=np.float64).shape[1]
            grad_c_upper = dxa[:, da:da + COUPLING_DIM]
            grad_pm_via_c = np.sum(phi_couple * grad_c_upper[:, None, :], axis=2)
            grad_zm_via_c = pm * (grad_pm_via_c - np.sum(pm * grad_pm_via_c, axis=1, keepdims=True))
            grad_zm = grad_zm_direct + grad_zm_via_c
        else:
            grad_zm = grad_zm_direct
    else:
        grad_zm = np.zeros_like(pm)

    grad_motion, _ = _branch_backward(xm, hm, grad_zm, params['motion'])

    reg_loss = 0.0
    for branch_name, grad_dict in (('allocation', grad_alloc), ('motion', grad_motion)):
        active = (
            (mode in ('joint', 'joint_no_expectation', 'decomposed'))
            or (mode == 'lower_only' and branch_name == 'allocation')
            or (mode == 'upper_only' and branch_name == 'motion')
        )
        if not active:
            for k in grad_dict:
                grad_dict[k] = np.zeros_like(grad_dict[k])
        else:
            for k in ('w', 'v', 'W'):
                reg_loss += 0.5 * l2_reg * float(np.sum(params[branch_name][k] ** 2))
                grad_dict[k] += l2_reg * params[branch_name][k]

    total_loss = float(surr_loss + ent_loss + kl_loss + reg_loss)
    return total_loss, dict(allocation=grad_alloc, motion=grad_motion)


def compute_episode_gae_and_returns(episodes, model, gamma=0.85, gae_lambda=0.90):
    """Compute Critic values V_psi(s_t), GAE advantages, and LOO-CRN variance-reduced advantages."""
    transitions = []
    ep_returns = []
    slot_groups = {}
    for ep in episodes:
        if not ep:
            continue
        rewards = [float(step['reward']) for step in ep]
        values = [critic_forward(step['row'], model)[0] for step in ep]
        advantages, returns = finite_gae(rewards, values, gamma=gamma, lam=gae_lambda)
        ep_returns.append(float(sum(rewards)))
        for step_idx, (step, adv, ret, val) in enumerate(zip(ep, advantages, returns, values)):
            tr_idx = len(transitions)
            w_key = step.get('workload')
            t_key = step.get('time', step_idx)
            if w_key is not None:
                slot_groups.setdefault((int(w_key), int(t_key)), []).append(tr_idx)
            transitions.append(dict(
                row=step['row'],
                action_pair=step['action_pair'],
                old_logp=float(step['old_logp']),
                ref_policy=step.get('ref_policy'),
                raw_gae_advantage=float(adv),
                advantage=float(adv),
                target_return=float(ret),
                old_value=float(val),
                reward=float(step['reward']),
            ))

    # Leave-One-Out Common-Random-Numbers (LOO-CRN) baseline across independent rollouts sharing (workload, time)
    for _, idx_list in slot_groups.items():
        if len(idx_list) >= 2:
            group_advs = [transitions[i]['raw_gae_advantage'] for i in idx_list]
            total_adv = float(sum(group_advs))
            n_g = len(idx_list)
            for i in idx_list:
                loo_baseline = (total_adv - transitions[i]['raw_gae_advantage']) / (n_g - 1)
                transitions[i]['advantage'] = float(transitions[i]['raw_gae_advantage'] - 0.50 * loo_baseline)

    if transitions:
        adv_arr = np.array([t['advantage'] for t in transitions], dtype=np.float64)
        adv_mean = float(np.mean(adv_arr))
        adv_std = max(float(np.std(adv_arr)), 1e-6)
        for t in transitions:
            t['norm_advantage'] = float(np.clip((t['advantage'] - adv_mean) / adv_std, -4.0, 4.0))
    return transitions, dict(
        episodes=len(episodes),
        transitions=len(transitions),
        mean_episode_reward=float(np.mean(ep_returns)) if ep_returns else 0.0,
    )


def ppo_update_actor_critic(
    episodes,
    model,
    mode='joint',
    ppo_epochs=6,
    critic_epochs=8,
    batch_size=32,
    actor_lr=0.003,
    critic_lr=0.008,
    clip_eps=0.20,
    ent_coef=0.002,
    kl_coef=0.02,
    gamma=0.85,
    gae_lambda=0.90,
    cf_weight=1.25,
    seed=0,
):
    """Run one on-policy PPO-GAE update iteration on collected environment episodes."""
    if mode not in RL_MODES:
        raise ValueError(f'unknown RL mode: {mode}')
    updated = copy.deepcopy(model)
    transitions, stats = compute_episode_gae_and_returns(
        episodes, updated, gamma=gamma, gae_lambda=gae_lambda
    )
    if not transitions:
        return updated, stats

    rng = np.random.default_rng(seed)

    # 1. Update state-value Critic V_psi(s) strictly on real trajectory GAE returns
    c_params = updated['critic']
    m_crit = {k: np.zeros_like(v) for k, v in c_params.items()}
    v_crit = {k: np.zeros_like(v) for k, v in c_params.items()}
    c_step = 0
    critic_losses = []
    for _ in range(critic_epochs):
        order = rng.permutation(len(transitions))
        for start in range(0, len(transitions), batch_size):
            batch_idx = order[start:start + batch_size]
            b_grad = {k: np.zeros_like(v) for k, v in c_params.items()}
            for idx in batch_idx:
                tr = transitions[int(idx)]
                c_loss, g_c = critic_loss_and_gradient(tr['row'], updated, tr['target_return'])
                critic_losses.append(c_loss)
                for k in c_params:
                    b_grad[k] += g_c[k] / len(batch_idx)
            gnorm = np.sqrt(sum(float(np.sum(g * g)) for g in b_grad.values()))
            scale = min(1.0, 5.0 / max(gnorm, 1e-30))
            c_step += 1
            for k in c_params:
                g = b_grad[k] * scale
                m_crit[k] = 0.9 * m_crit[k] + 0.1 * g
                v_crit[k] = 0.999 * v_crit[k] + 0.001 * (g * g)
                m_hat = m_crit[k] / (1.0 - 0.9 ** c_step)
                v_hat = v_crit[k] / (1.0 - 0.999 ** c_step)
                c_params[k] = c_params[k] - critic_lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Recompute GAE advantages with updated Critic for sharper variance reduction
    transitions, _ = compute_episode_gae_and_returns(
        episodes, updated, gamma=gamma, gae_lambda=gae_lambda
    )

    # Build combined Actor PPO transitions: trajectory GAE transitions + zero-mean counterfactual group transitions
    actor_transitions = list(transitions)
    if cf_weight > 0.0:
        for ep in episodes:
            for step in ep:
                for cf in step.get('cf_group', []):
                    actor_transitions.append(dict(
                        row=step['row'],
                        action_pair=cf['action_pair'],
                        old_logp=float(cf['old_logp']),
                        ref_policy=step.get('ref_policy'),
                        norm_advantage=float(cf_weight * cf['norm_advantage']),
                    ))

    # 2. Update two-branch Actor via Clipped Joint PPO
    a_params = updated['parameters']
    m_act = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in a_params.items()}
    v_act = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in a_params.items()}
    a_step = 0
    actor_losses = []
    for _ in range(ppo_epochs):
        order = rng.permutation(len(actor_transitions))
        for start in range(0, len(actor_transitions), batch_size):
            batch_idx = order[start:start + batch_size]
            b_grad = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in a_params.items()}
            for idx in batch_idx:
                tr = actor_transitions[int(idx)]
                a_loss, g_a = ppo_actor_loss_and_gradient(
                    tr['row'],
                    updated,
                    tr['action_pair'],
                    tr['old_logp'],
                    tr['norm_advantage'],
                    mode=mode,
                    clip_eps=clip_eps,
                    ent_coef=ent_coef,
                    kl_coef=kl_coef,
                    ref_policy=tr.get('ref_policy'),
                )
                actor_losses.append(a_loss)
                for b in a_params:
                    for k in a_params[b]:
                        b_grad[b][k] += g_a[b][k] / len(batch_idx)
            gnorm = np.sqrt(
                sum(float(np.sum(g * g)) for b_dict in b_grad.values() for g in b_dict.values())
            )
            scale = min(1.0, 5.0 / max(gnorm, 1e-30))
            a_step += 1
            for b in a_params:
                for k in a_params[b]:
                    g = b_grad[b][k] * scale
                    m_act[b][k] = 0.9 * m_act[b][k] + 0.1 * g
                    v_act[b][k] = 0.999 * v_act[b][k] + 0.001 * (g * g)
                    m_hat = m_act[b][k] / (1.0 - 0.9 ** a_step)
                    v_hat = v_act[b][k] / (1.0 - 0.999 ** a_step)
                    a_params[b][k] = a_params[b][k] - actor_lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    stats.update(
        critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
        actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
        actor_steps=a_step,
        critic_steps=c_step,
        actor_transitions=len(actor_transitions),
    )
    return updated, stats


def _distill_loss_and_grad(row, target, model, mode='joint', l2_reg=1e-4):
    """Cross-entropy distillation loss and gradient on clean features for warm-start baselines."""
    t_joint = np.asarray(target, dtype=np.float64)
    t_joint = t_joint / t_joint.sum()
    t_alloc = t_joint.sum(axis=1)
    pa, pm, za, zm, xa, xm, ha, hm, phi_couple = rl_actor_forward(row, model, mode=mode)
    log_pa = _log_softmax(za, axis=-1)
    log_pm = _log_softmax(zm, axis=-1)
    ce_loss = -float(np.sum(t_joint * (log_pa[:, None] + log_pm)))

    grad_za = (pa - t_alloc) if mode != 'upper_only' else np.zeros_like(pa)
    params = model['parameters']
    grad_alloc, dxa = _branch_backward(xa, ha, grad_za, params['allocation'])

    if mode != 'lower_only':
        grad_zm_direct = t_alloc[:, None] * pm - t_joint
        if mode == 'joint':
            da = np.asarray(row['a'], dtype=np.float64).shape[1]
            grad_c_upper = dxa[:, da:da + COUPLING_DIM]
            grad_pm_via_c = np.sum(phi_couple * grad_c_upper[:, None, :], axis=2)
            grad_zm_via_c = pm * (grad_pm_via_c - np.sum(pm * grad_pm_via_c, axis=1, keepdims=True))
            grad_zm = grad_zm_direct + grad_zm_via_c
        else:
            grad_zm = grad_zm_direct
    else:
        grad_zm = np.zeros_like(pm)

    grad_motion, _ = _branch_backward(xm, hm, grad_zm, params['motion'])
    reg_loss = 0.0
    for b_name, g_dict in (('allocation', grad_alloc), ('motion', grad_motion)):
        for k in ('w', 'v', 'W'):
            reg_loss += 0.5 * l2_reg * float(np.sum(params[b_name][k] ** 2))
            g_dict[k] += l2_reg * params[b_name][k]
    return float(ce_loss + reg_loss), dict(allocation=grad_alloc, motion=grad_motion)


def fit_clean_distilled_actor(
    train_rows,
    train_targets,
    val_rows,
    val_targets,
    init_model,
    mode='joint',
    epochs=32,
    batch_size=16,
    learning_rate=0.008,
    seed=0,
):
    """Fit a clean-feature teacher-distilled baseline (`joint_distilled` or `decomposed_distilled`)."""
    model = copy.deepcopy(init_model)
    best_model = copy.deepcopy(model)
    best_val_kl = float('inf')
    rng = np.random.default_rng(seed)
    a_params = model['parameters']
    m_act = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in a_params.items()}
    v_act = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in a_params.items()}
    step = 0

    for ep in range(1, epochs + 1):
        order = rng.permutation(len(train_rows))
        for start in range(0, len(train_rows), batch_size):
            batch_idx = order[start:start + batch_size]
            b_grad = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in a_params.items()}
            for idx in batch_idx:
                _, g = _distill_loss_and_grad(train_rows[int(idx)], train_targets[int(idx)], model, mode=mode)
                for b in a_params:
                    for k in a_params[b]:
                        b_grad[b][k] += g[b][k] / len(batch_idx)
            gnorm = np.sqrt(
                sum(float(np.sum(g * g)) for b_dict in b_grad.values() for g in b_dict.values())
            )
            scale = min(1.0, 5.0 / max(gnorm, 1e-30))
            step += 1
            lr_t = learning_rate * (0.5 * (1.0 + np.cos(np.pi * (ep - 1) / max(1, epochs))))
            for b in a_params:
                for k in a_params[b]:
                    g = b_grad[b][k] * scale
                    m_act[b][k] = 0.9 * m_act[b][k] + 0.1 * g
                    v_act[b][k] = 0.999 * v_act[b][k] + 0.001 * (g * g)
                    m_hat = m_act[b][k] / (1.0 - 0.9 ** step)
                    v_hat = v_act[b][k] / (1.0 - 0.999 ** step)
                    a_params[b][k] = a_params[b][k] - lr_t * m_hat / (np.sqrt(v_hat) + 1e-8)

        val_kls = [
            kl_divergence(vt, rl_actor_policy(vr, model, mode=mode))
            for vr, vt in zip(val_rows, val_targets)
        ]
        mean_val_kl = float(np.mean(val_kls)) if val_kls else 0.0
        if mean_val_kl < best_val_kl - 1e-9:
            best_val_kl = mean_val_kl
            best_model = copy.deepcopy(model)

    return best_model, dict(best_val_kl=best_val_kl, epochs=epochs)

