"""Small reproducible finite-bank actors and transactional full-batch fitting.

Both allocation-plan and conditional motion probabilities are learned. The
autoregressive plan PROPOSAL network stays frozen; this is stated explicitly.
Pure NumPy permits state-sharded exact gradient/acceptance reductions on CPUs.
"""
import copy
import numpy as np
from utils.queued_joint_mpo import softmax


def policy(row, theta, mode='joint'):
    a, m, prior = row['a'], row['m'], row['prior']; cut = a.shape[-1]
    if mode == 'decomposed': m = np.broadcast_to(m.mean(axis=0, keepdims=True), m.shape)
    if mode == 'flat':
        logits = np.log(prior)[:, None]+a@theta[:cut, None]+m@theta[cut:]
        return softmax(logits.reshape(-1)).reshape(m.shape[:2])
    pa = softmax(np.log(prior)+a@theta[:cut])
    pm = softmax(m@theta[cut:], axis=1)
    return pa[:, None]*pm


def loss_gradient(row, theta, target, mode='joint'):
    a, m = row['a'], row['m']; cut = a.shape[-1]
    if mode == 'decomposed': m = np.broadcast_to(m.mean(axis=0, keepdims=True), m.shape)
    p = policy(row, theta, mode); loss = -float(np.sum(target*np.log(p.clip(1e-300))))
    if mode == 'flat':
        diff = p-target
        return loss, np.r_[a.T@diff.sum(1), np.einsum('ij,ijk->k', diff, m)]
    qa, pa = target.sum(1), p.sum(1)
    # Motion CE is weighted by q_A, not by a uniform plan distribution.
    diff = qa[:, None]*(p/pa[:, None])-target
    return loss, np.r_[a.T@(pa-qa), np.einsum('ij,ijk->k', diff, m)]


def ppo_gradient(row, theta, old, advantage, mode='joint'):
    """Exact finite-support clipped surrogate diagnostic, not trajectory PPO."""
    p = policy(row, theta, mode); ratio = p/old
    active = ((advantage >= 0) & (ratio <= 1.2)) | ((advantage < 0) & (ratio >= .8))
    weights = old*advantage*ratio*active
    # Derivative of -sum(weights log pi), with weights detached for this step.
    a, m = row['a'], row['m']; pa = p.sum(1)
    ga = a.T@(pa*weights.sum()-weights.sum(1))
    gm = np.einsum('ij,ijk->k', weights.sum(1)[:, None]*p/pa[:, None]-weights, m)
    loss = -float(np.sum(old*np.minimum(ratio*advantage, np.clip(ratio, .8, 1.2)*advantage)))
    return loss, np.r_[ga, gm]


def transactional_step(theta, state, gradient, accept, base_lr=.03):
    """Functional Adam: no mutation until a full-state test accepts a proposal."""
    original = copy.deepcopy(state); state = copy.deepcopy(state)
    state['step'] += 1; step = state['step']
    state['m'] = .9*state['m']+.1*gradient
    state['v'] = .999*state['v']+.001*gradient**2
    direction = state['m']/(1-.9**step)/(np.sqrt(state['v']/(1-.999**step))+1e-8)
    trials = []
    # Functional-space calibration can explore above one nominal step.
    for scale in (8., 4., 2., 1., .5, .25, .125, .0625, .03125, .015625):
        candidate = theta-base_lr*scale*direction
        allowed = bool(np.all(np.isfinite(candidate))) and accept(candidate)
        trials.append(dict(scale=scale, accepted=allowed))
        if allowed: return candidate, state, dict(accepted=True, scale=scale, trials=trials)
    return theta.copy(), original, dict(accepted=False, scale=0., trials=trials)
