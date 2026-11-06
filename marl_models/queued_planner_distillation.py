"""Candidate-equivariant conditional actor; pure NumPy numerical implementation.

Inference reads only public context and existing bank features. No teacher,
terminal forecast, workload identifier, action index or true future is an input.
The proposal network is inherited and frozen, not trained by this module.
"""
import copy
import numpy as np
from utils.queued_planner_distillation import planner_target, check_rows


def inputs(row, mode='joint'):
    if mode not in ('joint', 'decomposed'):
        raise ValueError('joint or decomposed required')
    a, m, c, prior = (np.asarray(row[k], float) for k in ('a', 'm', 'context', 'prior'))
    if (a.ndim != 2 or m.ndim != 3 or c.ndim != 1 or not a.size or not m.size
            or a.shape[0] != m.shape[0] or prior.shape != (a.shape[0],)
            or np.any(prior <= 0) or not np.isclose(prior.sum(), 1.)
            or not all(np.isfinite(x).all() for x in (a, m, c, prior))):
        raise ValueError('finite compatible public bank features required')
    k, n = m.shape[:2]
    xa = np.concatenate([a, np.broadcast_to(c, (k, len(c)))], axis=1)
    if mode == 'decomposed':
        # Same feature dimensions and model capacity; remove dependence on
        # the selected plan by averaging its public features before inference.
        motion = np.broadcast_to(m.mean(0, keepdims=True), m.shape)
        allocation = np.broadcast_to(a.mean(0), (k, n, a.shape[1]))
    else:
        motion = m
        allocation = np.broadcast_to(a[:, None, :], (k, n, a.shape[1]))
    xm = np.concatenate([motion, allocation, np.broadcast_to(c, (k, n, len(c)))], axis=2)
    return xa, xm, prior


def initialize(row, hidden=32, seed=0):
    if not isinstance(hidden, int) or hidden < 0:
        raise ValueError('nonnegative hidden width required')
    xa, xm, _ = inputs(row)
    rng = np.random.default_rng(seed)
    params = {}
    for name, dim in [('allocation', xa.shape[-1]), ('motion', xm.shape[-1])]:
        if hidden:
            params[name] = dict(W=rng.normal(size=(dim, hidden))/np.sqrt(dim),
                                b=np.zeros(hidden), v=np.zeros(hidden))
        else:
            params[name] = dict(w=np.zeros(dim))
    return dict(parameters=params, hidden=hidden, context_dim=len(row['context']),
                input_dimensions=[xa.shape[-1], xm.shape[-1]])


def _score(x, branch):
    if 'w' in branch:
        return x@branch['w'], None
    h = np.tanh(x@branch['W']+branch['b'])
    return h@branch['v'], h


def _softmax(x):
    y = np.exp(x-np.max(x, axis=-1, keepdims=True))
    return y/y.sum(axis=-1, keepdims=True)


def _forward(row, model, mode):
    xa, xm, prior = inputs(row, mode)
    if ([xa.shape[-1], xm.shape[-1]] != model['input_dimensions']
            or len(row['context']) != model['context_dim']):
        raise ValueError('actor feature schema mismatch')
    if not all(np.isfinite(p).all() for branch in model['parameters'].values() for p in branch.values()):
        raise ValueError('nonfinite actor parameters')
    za, ha = _score(xa, model['parameters']['allocation'])
    zm, hm = _score(xm, model['parameters']['motion'])
    pa, pm = _softmax(np.log(prior)+za), _softmax(zm)
    return pa, pm, xa, xm, ha, hm


def policy(row, model, mode='joint'):
    pa, pm, *_ = _forward(row, model, mode)
    return pa[:, None]*pm


def _backward(x, h, derivative, branch):
    if h is None:
        return dict(w=x.reshape(-1, x.shape[-1]).T@derivative.reshape(-1))
    dx = derivative[..., None]*branch['v']*(1-h*h)
    return dict(W=x.reshape(-1, x.shape[-1]).T@dx.reshape(-1, dx.shape[-1]),
                b=dx.reshape(-1, dx.shape[-1]).sum(0),
                v=h.reshape(-1, h.shape[-1]).T@derivative.reshape(-1))


def loss_gradient(row, model, target, mode='joint'):
    pa, pm, xa, xm, ha, hm = _forward(row, model, mode)
    target = np.asarray(target, float)
    if (target.shape != pm.shape or np.any(target < 0) or not np.isfinite(target).all()
            or not np.isclose(target.sum(), 1.)):
        raise ValueError('normalized paired teacher target required')
    # Log softmax avoids underflow in the loss; gradients match this objective.
    za, _ = _score(xa, model['parameters']['allocation'])
    za = za+np.log(np.asarray(row['prior']))
    zm, _ = _score(xm, model['parameters']['motion'])
    def logsoftmax(x):
        z = x-np.max(x, axis=-1, keepdims=True)
        return z-np.log(np.exp(z).sum(axis=-1, keepdims=True))
    loss = -float(np.sum(target*(logsoftmax(za)[:, None]+logsoftmax(zm))))
    qa = target.sum(1)
    grad = dict(allocation=_backward(xa, ha, pa-qa, model['parameters']['allocation']),
                motion=_backward(xm, hm, qa[:, None]*pm-target, model['parameters']['motion']))
    return loss, grad


def fit(rows, initial, mode='joint', epochs=4, batch_size=8, learning_rate=.003, seed=0):
    """Fixed-budget aggregate imitation. No held-out data or early stopping.

    New rounds restart Adam moments but retain actor weights. A failed numerical
    proposal raises before mutation of the caller's model or saved checkpoint.
    """
    check_rows(rows)
    if (not isinstance(epochs, int) or epochs <= 0 or not isinstance(batch_size, int)
            or batch_size <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('finite positive optimizer budget required')
    model = copy.deepcopy(initial)
    params = model['parameters']
    moments = [{b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in params.items()} for _ in range(2)]
    rng = np.random.default_rng(seed)
    targets = [planner_target(r['q'], r['prior']) for r in rows]
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), batch_size):
            indices = order[start:start+batch_size]
            grads = {b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in params.items()}
            for i in indices:
                loss, grad = loss_gradient(rows[i], model, targets[i], mode)
                if not np.isfinite(loss):
                    raise FloatingPointError('nonfinite imitation loss')
                for b in params:
                    for k in params[b]:
                        grads[b][k] += grad[b][k]/len(indices)
            norm = np.sqrt(sum(float(np.sum(g*g)) for p in grads.values() for g in p.values()))
            if not np.isfinite(norm):
                raise FloatingPointError('nonfinite imitation gradient')
            scale = min(1., 5./max(norm, 1e-300))
            step += 1
            for b in params:
                for k, p in params[b].items():
                    g = grads[b][k]*scale
                    moments[0][b][k] = .9*moments[0][b][k]+.1*g
                    moments[1][b][k] = .999*moments[1][b][k]+.001*g*g
                    update = moments[0][b][k]/(1-.9**step)/(np.sqrt(moments[1][b][k]/(1-.999**step))+1e-8)
                    candidate = p-learning_rate*update
                    if not np.isfinite(candidate).all():
                        raise FloatingPointError('nonfinite imitation proposal')
                    params[b][k] = candidate
    return model, dict(states=len(rows), workloads=len({r['workload'] for r in rows}),
                       epochs=epochs, batch_size=batch_size, optimizer_steps=step,
                       learning_rate=learning_rate, optimizer='Adam, fresh moments per aggregation round',
                       heldout_used=False, early_stopping=False, gradient_clip=5.)
