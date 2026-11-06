"""Two-branch candidate-equivariant neural actor for upper-lower joint training.

Supports four training modes on a shared neural architecture:
- 'joint': plan-conditioned motion features + 4D upper-policy expectation coupling
  vector C_upper[k] = E_{m ~ pi_upper(.|k)}[phi_couple(k, m)], updating both upper
  motion and lower allocation parameters jointly with exact end-to-end gradients.
- 'decomposed': plan-averaged motion features with cross-layer congestion/overflow
  channels masked + zero cross-layer expectation coupling, updating upper and lower
  branches without cross-layer conditioning.
- 'upper_only': trains only the upper motion branch (uncoupled from lower-layer
  congestion/allocation) while keeping the lower allocation branch frozen at zero.
- 'lower_only': trains only the lower allocation branch (uncoupled from upper-layer
  learned motion) while keeping the upper motion branch frozen at zero.

Also provides `compose_decomposed_actor(lower_only_model, upper_only_model)` to
directly assemble an independently trained lower allocation branch and an
independently trained upper motion branch without joint retraining.
"""
import copy
import numpy as np

MODES = ('joint', 'decomposed', 'upper_only', 'lower_only')
COUPLING_DIM = 4
# Channels of motion tensor `m` that encode cross-layer queue-overflow / congestion coupling:
#   2: cross_overflow_norm
#   3: cross_net_gain_norm
#   4: cross_comp_norm
#   9: is_congested
#  10: is_congested * is_rescue
#  11: (1 - is_congested) * rescue_norm
#  12: pressure_scaled_move
#  13: is_safe_plan
UNCOUPLED_MASKED_MOTION_CHANNELS = (2, 3, 4, 9, 10, 11, 12, 13)


def _softmax(logits, axis=-1):
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _log_softmax(logits, axis=-1):
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def kl_divergence(target, predicted):
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    active = target > 0
    if np.any(predicted[active] <= 0):
        return float('inf')
    return float(np.sum(target[active] * (np.log(target[active]) - np.log(predicted[active]))))


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


def prepare_branch_inputs(row, mode='joint', pm_override=None):
    """Construct allocation inputs xa (K, Da+4+Dc) and motion inputs xm (K, M, Dm+Da+Dc).

    In `joint` mode:
      - `xm` conditions each motion candidate on the specific lower allocation plan `a[k]`
        and retains all cross-layer queue-overflow & congestion interaction channels.
      - `xa` includes the 4D expected upper-policy coupling vector:
            C_upper[k, d] = sum_m pm[k, m] * phi_couple[k, m, d]
        where `pm` is the upper motion policy distribution (or `motion_prior` when
        inspecting static input dimensions prior to the forward pass).
    In uncoupled modes (`decomposed`, `upper_only`, `lower_only`):
      - `xm` replaces `a[k]` with its across-plan average and masks cross-layer
        congestion/overflow channels so upper motion selection is unconditioned on the lower plan.
      - `xa` sets `C_upper` to zero so lower allocation is unconditioned on learned upper motion.
    """
    if mode not in MODES:
        raise ValueError(f'unknown training mode: {mode}')
    a, m, c, prior, motion_prior = _validate_row(row)
    k, n, dm = m.shape
    da = a.shape[1]
    dc = len(c)

    if mode == 'joint':
        alloc_ctx = np.broadcast_to(a[:, None, :], (k, n, da))
        xm = np.concatenate([m, alloc_ctx, np.broadcast_to(c, (k, n, dc))], axis=2)
        phi_couple = m[:, :, :COUPLING_DIM]
        weights = motion_prior if pm_override is None else np.asarray(pm_override, dtype=np.float64)
        c_upper = np.sum(weights[:, :, None] * phi_couple, axis=1)
        xa = np.concatenate([a, c_upper, np.broadcast_to(c, (k, dc))], axis=1)
    else:
        a_mean = np.broadcast_to(a.mean(axis=0, keepdims=True), a.shape)
        m_masked = np.broadcast_to(m.mean(axis=0, keepdims=True), m.shape).copy()
        valid_mask_idx = [idx for idx in UNCOUPLED_MASKED_MOTION_CHANNELS if idx < dm]
        if valid_mask_idx:
            m_masked[..., valid_mask_idx] = 0.0
        alloc_ctx = np.broadcast_to(a_mean[:, None, :], (k, n, da))
        xm = np.concatenate([m_masked, alloc_ctx, np.broadcast_to(c, (k, n, dc))], axis=2)
        phi_couple = np.zeros((k, n, COUPLING_DIM), dtype=np.float64)
        c_upper = np.zeros((k, COUPLING_DIM), dtype=np.float64)
        xa = np.concatenate([a, c_upper, np.broadcast_to(c, (k, dc))], axis=1)

    return xa, xm, prior, motion_prior, phi_couple


def initialize_joint_actor(row, hidden=48, seed=0):
    """Initialize 2-layer residual MLP actor with zero initial output weights.

    Because both the linear skip weight `w` and the output layer `v` are initialized
    to zero, Epoch 0 exactly reproduces the baseline prior distribution.
    """
    if not isinstance(hidden, int) or hidden <= 0:
        raise ValueError('positive hidden dimension required')
    xa, xm, _, _, _ = prepare_branch_inputs(row, mode='joint')
    rng = np.random.default_rng(seed)
    params = {}
    for name, dim in (('allocation', xa.shape[-1]), ('motion', xm.shape[-1])):
        params[name] = dict(
            w=np.zeros(dim, dtype=np.float64),
            W=rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, hidden)),
            b=np.zeros(hidden, dtype=np.float64),
            v=np.zeros(hidden, dtype=np.float64),
        )
    return dict(
        parameters=params,
        hidden=hidden,
        context_dim=len(row['context']),
        coupling_dim=COUPLING_DIM,
        input_dimensions=[int(xa.shape[-1]), int(xm.shape[-1])],
    )


def compose_decomposed_actor(lower_only_model, upper_only_model):
    """Directly assemble a decomposed actor from independently trained lower and upper models.

    Takes the trained `allocation` parameters from `lower_only_model` and the trained
    `motion` parameters from `upper_only_model` without any joint co-training.
    """
    if list(lower_only_model['input_dimensions']) != list(upper_only_model['input_dimensions']):
        raise ValueError('cannot compose models with mismatched input dimensions')
    composed = copy.deepcopy(lower_only_model)
    composed['parameters']['allocation'] = copy.deepcopy(lower_only_model['parameters']['allocation'])
    composed['parameters']['motion'] = copy.deepcopy(upper_only_model['parameters']['motion'])
    composed['composed_from'] = ('lower_only_trained', 'upper_only_trained')
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


def actor_forward(row, model, mode='joint'):
    """Forward pass computing lower allocation distribution `pa` and upper motion `pm`.

    In `joint` mode:
      1. Evaluates the upper motion branch on plan-conditioned inputs `xm` to obtain `pm(m|k)`.
      2. Computes the 4D expected upper coupling vector `C_upper[k] = sum_m pm(m|k) phi_couple[k, m]`.
      3. Evaluates the lower allocation branch on `xa = [a, C_upper, context]` to obtain `pa(k)`.
    """
    _, xm, prior, motion_prior, phi_couple = prepare_branch_inputs(row, mode=mode)
    params = model['parameters']
    if int(xm.shape[-1]) != int(model['input_dimensions'][1]):
        raise ValueError('motion branch input dimension mismatch')

    zm_raw, hm = _branch_score(xm, params['motion'])
    if mode == 'lower_only':
        zm_raw = np.zeros_like(zm_raw)
    zm = np.log(np.clip(motion_prior, 1e-12, None)) + zm_raw
    pm = _softmax(zm, axis=-1)

    xa, _, _, _, _ = prepare_branch_inputs(row, mode=mode, pm_override=pm)
    if int(xa.shape[-1]) != int(model['input_dimensions'][0]):
        raise ValueError('allocation branch input dimension mismatch')

    za_raw, ha = _branch_score(xa, params['allocation'])
    if mode == 'upper_only':
        za_raw = np.zeros_like(za_raw)
    za = np.log(np.clip(prior, 1e-12, None)) + za_raw
    pa = _softmax(za, axis=-1)

    return pa, pm, za, zm, xa, xm, ha, hm, phi_couple


def actor_policy(row, model, mode='joint'):
    pa, pm, *_ = actor_forward(row, model, mode=mode)
    return pa[:, None] * pm


def actor_loss_and_gradient(row, model, target, mode='joint', l2_reg=1e-5):
    """Compute exact cross-entropy + L2 loss and analytical gradients for all active branches."""
    pa, pm, za, zm, xa, xm, ha, hm, phi_couple = actor_forward(row, model, mode=mode)
    target = np.asarray(target, dtype=np.float64)
    if target.shape != pm.shape or np.any(target < 0) or not np.isclose(target.sum(), 1.0):
        raise ValueError('valid normalized joint target distribution required')

    log_pa = _log_softmax(za, axis=-1)
    log_pm = _log_softmax(zm, axis=-1)
    ce_loss = -float(np.sum(target * (log_pa[:, None] + log_pm)))

    qa = target.sum(axis=1)
    grad_za = (pa - qa) if mode != 'upper_only' else np.zeros_like(pa)

    params = model['parameters']
    grad_alloc, dxa = _branch_backward(xa, ha, grad_za, params['allocation'])

    grad_zm_direct = qa[:, None] * pm - target
    if mode == 'joint':
        da = np.asarray(row['a'], dtype=np.float64).shape[1]
        grad_c_upper = dxa[:, da:da + COUPLING_DIM]
        grad_pm_via_c = np.sum(phi_couple * grad_c_upper[:, None, :], axis=2)
        grad_zm_via_c = pm * (grad_pm_via_c - np.sum(pm * grad_pm_via_c, axis=1, keepdims=True))
        grad_zm = grad_zm_direct + grad_zm_via_c
    elif mode == 'lower_only':
        grad_zm = np.zeros_like(pm)
    else:
        grad_zm = grad_zm_direct

    grad_motion, _ = _branch_backward(xm, hm, grad_zm, params['motion'])

    reg_loss = 0.0
    for branch_name, grad_dict in (('allocation', grad_alloc), ('motion', grad_motion)):
        active = (
            (mode in ('joint', 'decomposed'))
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

    return ce_loss + reg_loss, dict(allocation=grad_alloc, motion=grad_motion)


def evaluate_actor_dataset(rows, targets, values, model, mode='joint'):
    """Compute offline KL, top-1 value regret, optimal mass, expected value, and pairwise accuracy."""
    if not rows:
        return dict(states=0, kl=0.0, top1_regret=0.0, optimal_probability=0.0,
                    expected_value=0.0, pairwise_accuracy=0.0)
    kls = []
    regrets = []
    opt_probs = []
    expected_vals = []
    accuracies = []
    for row, target, val_mat in zip(rows, targets, values):
        p = actor_policy(row, model, mode=mode)
        kls.append(kl_divergence(target, p))
        v = np.asarray(val_mat, dtype=np.float64)
        expected_vals.append(float(np.sum(p * v)))
        best_true = v >= (v.max() - 1e-12)
        opt_probs.append(float(p[best_true].sum()))
        best_pred = p >= (p.max() - 1e-12)
        regrets.append(float(v.max() - v[best_pred].mean()))
        flat_p = p.ravel()
        flat_v = v.ravel()
        idx_i, idx_j = np.triu_indices(len(flat_v), 1)
        dv = flat_v[idx_i] - flat_v[idx_j]
        dp = flat_p[idx_i] - flat_p[idx_j]
        mask = np.abs(dv) > 1e-9
        if np.any(mask):
            correct = np.where(np.abs(dp[mask]) < 1e-12, 0.5, np.sign(dp[mask]) == np.sign(dv[mask]))
            weights = np.abs(dv[mask])
            accuracies.append(float(np.sum(weights * correct) / np.sum(weights)))
    return dict(
        states=len(rows),
        kl=float(np.mean(kls)),
        top1_regret=float(np.mean(regrets)),
        optimal_probability=float(np.mean(opt_probs)),
        expected_value=float(np.mean(expected_vals)),
        pairwise_accuracy=float(np.mean(accuracies)) if accuracies else 1.0,
    )


def fit_joint_actor(train_rows, train_targets, train_values,
                    val_rows, val_targets, val_values,
                    initial_model, mode='joint', epochs=32, batch_size=16,
                    learning_rate=0.008, seed=0, checkpoint_epochs=(0, 2, 4, 8, 16, 32),
                    use_warmup=True):
    """Train the two-branch neural actor with Adam and record learning curves & checkpoints."""
    if mode not in MODES:
        raise ValueError(f'unknown mode: {mode}')
    model = copy.deepcopy(initial_model)
    params = model['parameters']
    m_adam = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in params.items()}
    v_adam = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in params.items()}
    rng = np.random.default_rng(seed)
    step = 0
    curve = []
    checkpoints = {}

    for epoch in range(epochs + 1):
        if epoch in checkpoint_epochs:
            train_metrics = evaluate_actor_dataset(train_rows, train_targets, train_values, model, mode=mode)
            val_metrics = evaluate_actor_dataset(val_rows, val_targets, val_values, model, mode=mode)
            curve.append(dict(epoch=epoch, step=step, train=train_metrics, validation=val_metrics))
            checkpoints[epoch] = copy.deepcopy(model)
        if epoch == epochs:
            break
        if use_warmup:
            warmup = 0.20 if epoch < 2 else (0.35 if epoch < 4 else (0.55 if epoch < 8 else (0.80 if epoch < 16 else 1.0)))
        else:
            warmup = 1.0
        eff_lr = learning_rate * warmup
        order = rng.permutation(len(train_rows))
        for start in range(0, len(train_rows), batch_size):
            batch_idx = order[start:start + batch_size]
            batch_grad = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in params.items()}
            for idx in batch_idx:
                loss, grad = actor_loss_and_gradient(
                    train_rows[int(idx)], model, train_targets[int(idx)], mode=mode
                )
                if not np.isfinite(loss):
                    raise FloatingPointError('nonfinite actor training loss')
                for b in params:
                    for k in params[b]:
                        batch_grad[b][k] += grad[b][k] / len(batch_idx)
            grad_norm = np.sqrt(
                sum(float(np.sum(g * g)) for b_dict in batch_grad.values() for g in b_dict.values())
            )
            scale = min(1.0, 5.0 / max(grad_norm, 1e-30))
            step += 1
            for b in params:
                for k in params[b]:
                    g = batch_grad[b][k] * scale
                    m_adam[b][k] = 0.9 * m_adam[b][k] + 0.1 * g
                    v_adam[b][k] = 0.999 * v_adam[b][k] + 0.001 * (g * g)
                    m_hat = m_adam[b][k] / (1.0 - 0.9 ** step)
                    v_hat = v_adam[b][k] / (1.0 - 0.999 ** step)
                    params[b][k] = params[b][k] - eff_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return model, dict(mode=mode, epochs=epochs, batch_size=batch_size,
                       learning_rate=learning_rate, steps=step, curve=curve), checkpoints
