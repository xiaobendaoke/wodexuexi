"""Matched supervision and node/deadline features for bounded cause tests."""
import copy
import numpy as np
from utils.queued_learning_validation import pack, subset, statistics, evaluate
from utils.queued_closed_loop import _masked_softmax
from utils.queued_planner_distillation import planner_target

REVISION = 'queued_learning_causes_20261104'
FEATURES = ('pool', 'node')
OBJECTIVES = ('hard', 'vote', 'rank')
MODES = ('joint', 'decomposed')


def node_features(obs, plan, motion):
    """64 public, complete-task resource statistics with explicit node slots."""
    protocol = obs['protocol']
    positions = np.asarray(obs['positions']) + np.asarray(motion)
    nodes = len(protocol.capacity)
    if nodes != 6:
        raise ValueError('registered six-node protocol required')
    values = np.zeros((nodes, 10))
    work = [[] for _ in range(nodes)]
    inter = backhaul = 0.
    for task in obs['tasks']:
        ue = np.asarray(obs['ues'][task['ue']])
        if task['source'] is None:
            distances = np.linalg.norm(positions - ue, axis=1)
            source = int(np.argmin(distances))
            values[source, 8] += float(distances[source] <= protocol.coverage)/20
            continue
        source = task['source']
        target = task['target'] if task['target'] is not None else plan.get(task['uid'], source)
        slack = task['deadline'] - obs['time']
        bucket = 0 if slack <= 1 else 1 if slack <= 2 else 2
        seconds = task['remaining_cycles']/protocol.capacity[target]
        values[target, bucket] += seconds
        values[source, 3] += task['remaining_upload']/24e6
        transfer = task['remaining_transfer'] if task['target'] is not None else task['bits'] if target != source else 0.
        values[target, 4] += transfer/24e6
        values[target, 5] += seconds if task['stage'] == 'compute' else 0.
        values[target, 7] += 1/20
        work[target].append((slack, seconds))
        if target != source:
            target_pos = np.asarray(protocol.mbs[:2]) if target == len(positions) else positions[target]
            disconnected = target != len(positions) and np.linalg.norm(positions[source]-target_pos) > protocol.inter_range
            values[target, 9] += float(disconnected)*transfer/24e6
            if target == len(positions):backhaul += transfer/24e6
            else:inter += transfer/24e6
    for node, tasks in enumerate(work):
        if tasks:
            ordered = sorted(tasks)
            # EDF demand includes all tasks with an equal deadline.
            deadlines = np.asarray([x[0] for x in ordered])
            prefix = np.r_[0., np.cumsum([x[1] for x in ordered])]
            load = prefix[np.searchsorted(deadlines, deadlines, side='right')]
            values[node, 6] = np.max(load/np.maximum(deadlines, 1e-6))
    count = len(obs['tasks'])
    result = np.arcsinh(np.r_[values.ravel(), inter, backhaul, count/100,
                              sum(x['source'] is None for x in obs['tasks'])/100])
    if result.shape != (64,) or not np.isfinite(result).all():
        raise ValueError('finite 64-channel public representation required')
    return result


def prepared(rows, forecasts, mode):
    if len(rows) != len(forecasts):
        raise ValueError('one forecast bank per row required')
    data = pack(rows, mode)
    batch, plans, motions = data['mm'].shape
    count = plans*motions
    preferences = np.zeros((batch, count, count))
    weights = np.zeros_like(preferences)
    votes = np.zeros_like(data['target'])
    for index, (row, scores) in enumerate(zip(rows, forecasts)):
        scores = np.asarray(scores, float)
        if scores.ndim != 3 or scores.shape[1:] != row['m'].shape[:2] or not np.isfinite(scores).all() or len(scores) != 16:
            raise ValueError('sixteen finite public forecast matrices required')
        k, m = scores.shape[1:]
        votes[index, :k, :m] = np.mean([planner_target(q, row['prior']) for q in scores], axis=0)
        ids = np.array([a*motions+b for a in range(k) for b in range(m)])
        samples = scores.reshape(len(scores), -1)
        differences = samples[:, :, None]-samples[:, None, :]
        delta = differences.mean(0)
        scale = 1+differences.std(0, ddof=1)/np.sqrt(len(scores))
        preference = 1/(1+np.exp(-np.clip(delta/scale, -700, 700)))
        weight = np.triu(np.abs(delta)/(np.abs(delta)+scale), 1)
        weight /= max(weight.sum(), 1e-300)
        preferences[index][np.ix_(ids, ids)] = preference
        weights[index][np.ix_(ids, ids)] = weight
    data.update(votes=votes, preferences=preferences, weights=weights)
    return data


def loss_gradient(data, model, objective):
    if objective == 'hard':
        result, gradient = statistics(data, model)
        return result['loss'], gradient
    if objective == 'vote':
        voted = dict(data, target=data['votes'])
        result, gradient = statistics(voted, model)
        return result['loss'], gradient
    if objective != 'rank':raise ValueError('unknown registered objective')
    params = model['parameters']
    activations = {}
    scores = {}
    for branch, key in (('allocation', 'xa'), ('motion', 'xm')):
        p = params[branch]
        h = np.tanh(data[key]@p['W']+p['b'])
        activations[branch] = h
        scores[branch] = h@p['v']
    pa = _masked_softmax(np.log(data['prior'].clip(1e-300))+scores['allocation'], data['ma'], 1)
    pm = _masked_softmax(scores['motion'], data['mm'], 2)
    logp = (np.log(pa.clip(1e-300))[:, :, None]+np.log(pm.clip(1e-300))).reshape(len(pa), -1)
    difference = logp[:, :, None]-logp[:, None, :]
    preference = data['preferences']
    weight = data['weights']
    loss = float(np.mean(np.sum(weight*(np.logaddexp(0, difference)-preference*difference), axis=(1, 2))))
    sigmoid = 1/(1+np.exp(-np.clip(difference, -700, 700)))
    derivative = weight*(sigmoid-preference)
    dlogp = (derivative.sum(2)-derivative.sum(1)).reshape(pm.shape)
    da = dlogp.sum(2)
    derivatives = dict(allocation=da-pa*da.sum(1, keepdims=True),
                       motion=dlogp-pm*dlogp.sum(2, keepdims=True))
    gradient = {}
    for branch, key in (('allocation', 'xa'), ('motion', 'xm')):
        x = data[key];h = activations[branch];d = derivatives[branch]
        dh = d[..., None]*params[branch]['v']*(1-h*h)
        gradient[branch] = dict(W=x.reshape(-1, x.shape[-1]).T@dh.reshape(-1, dh.shape[-1])/len(pa),
                               b=dh.reshape(-1, dh.shape[-1]).sum(0)/len(pa),
                               v=h.reshape(-1, h.shape[-1]).T@d.reshape(-1)/len(pa))
    return loss, gradient


def fit(rows, forecasts, heldout, hold_forecasts, initial, mode, objective, epochs, seed):
    from utils.queued_planner_distillation import check_rows
    check_rows(rows, heldout)
    train = prepared(rows, forecasts, mode)
    hold = prepared(heldout, hold_forecasts, mode)
    model = copy.deepcopy(initial)
    params = model['parameters']
    moments = [{b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in params.items()} for _ in range(2)]
    rng = np.random.default_rng(seed)
    step = 0;curve = []
    for epoch in range(epochs+1):
        if epoch in (0, 4, 16, 32, 64):
            curve.append(dict(epoch=epoch, steps=step, train=evaluate(train, model), holdout=evaluate(hold, model)))
        if epoch == epochs:break
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), 8):
            loss, gradient = loss_gradient(subset(train, order[start:start+8]), model, objective)
            norm = np.sqrt(sum(float(np.sum(g*g)) for b in gradient.values() for g in b.values()))
            if not np.isfinite(loss) or not np.isfinite(norm):raise FloatingPointError('nonfinite fitting proposal')
            scale = min(1., 5/max(norm, 1e-300));step += 1
            for branch, values in params.items():
                for key, value in values.items():
                    g = gradient[branch][key]*scale
                    moments[0][branch][key] = .9*moments[0][branch][key]+.1*g
                    moments[1][branch][key] = .999*moments[1][branch][key]+.001*g*g
                    update = moments[0][branch][key]/(1-.9**step)/(np.sqrt(moments[1][branch][key]/(1-.999**step))+1e-8)
                    candidate = value-.003*update
                    if not np.isfinite(candidate).all():raise FloatingPointError('nonfinite actor')
                    params[branch][key] = candidate
    return model, curve


def quality_details(tasks):
    result = {}
    for phase in ('pre', 'peak', 'post'):
        selected = [x for x in tasks if ('pre' if x.arrival < 50 else 'peak' if x.arrival < 100 else 'post') == phase]
        nodes = []
        for node in range(6):
            xs = [x for x in selected if x.target == node]
            compute = [x for x in xs if x.stage == 'completed' or x.expired_stage == 'compute']
            failed = [x for x in xs if x.stage == 'expired']
            nodes.append(dict(node=node, assigned=len(xs), success=sum(x.stage == 'completed' for x in xs),
                compute_expired=sum(x.expired_stage == 'compute' for x in xs),
                compute_wait=sum(x.time_in['compute_wait'] for x in xs),
                compute_entries=len(compute),
                entry_slack_sum=sum(x.deadline-(x.finished-x.time_in['compute']) for x in compute),
                served_cycles=sum(x.served_cycles for x in xs),
                wasted_cycles=sum(x.served_cycles for x in failed),
                missed_cycles=sum(x.cancelled_cycles for x in failed)))
        result[phase] = nodes
    return result
