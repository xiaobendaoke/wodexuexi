"""Independent candidate scores and bounded, public-pressure policy control."""
import copy
import numpy as np
from marl_models.queued_planner_distillation import _score, _backward, inputs
from utils.queued_learning_causes import prepared
from utils.queued_learning_validation import subset
from utils.queued_planner_distillation import check_rows

REVISION = 'queued_score_control_20261104'


def score_data(rows, forecasts, mode):
    data = prepared(rows, forecasts, mode)
    weights = np.zeros_like(data['q'])
    for index, scores in enumerate(forecasts):
        centered = scores-scores.mean((1, 2), keepdims=True)
        k,m = scores.shape[1:]
        weights[index, :k, :m] = 1/(1+centered.var(0, ddof=1)/len(scores))
    data['score_weights'] = weights/weights.sum((1, 2), keepdims=True)
    return data


def scores(data, model):
    params = model['parameters']
    allocation, _ = _score(data['xa'], params['allocation'])
    motion, _ = _score(data['xm'], params['motion'])
    return allocation[:, :, None]+motion


def row_scores(row, model, mode):
    allocation, motion, _ = inputs(row, mode)
    return scores(dict(xa=allocation[None], xm=motion[None]), model)[0]


def score_gradient(data, model, objective):
    params = model['parameters']
    allocation, ha = _score(data['xa'], params['allocation'])
    motion, hm = _score(data['xm'], params['motion'])
    prediction = allocation[:, :, None]+motion
    batch = len(prediction)
    if objective == 'reg':
        weights = data['score_weights']
        error = prediction-data['q']
        error -= np.sum(weights*error, axis=(1, 2), keepdims=True)
        loss = float(np.mean(np.sum(weights*error**2, axis=(1, 2)))/2)
        derivative = weights*error
        derivative -= weights*derivative.sum((1, 2), keepdims=True)
    elif objective == 'rawrank':
        flat = prediction.reshape(batch, -1)
        difference = flat[:, :, None]-flat[:, None, :]
        weight = data['weights'];preference = data['preferences']
        loss = float(np.mean(np.sum(weight*(np.logaddexp(0, difference)-preference*difference), axis=(1, 2))))
        sigmoid = 1/(1+np.exp(-np.clip(difference, -700, 700)))
        pair_derivative = weight*(sigmoid-preference)
        derivative = (pair_derivative.sum(2)-pair_derivative.sum(1)).reshape(prediction.shape)
    else:
        raise ValueError('reg or rawrank required')
    gradient = dict(
        allocation=_backward(data['xa'], ha, derivative.sum(2)/batch, params['allocation']),
        motion=_backward(data['xm'], hm, derivative/batch, params['motion']))
    return loss, gradient


def score_metrics(data, model, objective):
    prediction = scores(data, model)
    output = {}
    for phase in ('all', 'peak', 'nonpeak'):
        peak = (data['times'] >= 50)&(data['times'] < 100)
        mask = np.ones(len(peak), bool) if phase == 'all' else peak if phase == 'peak' else ~peak
        if not mask.any():continue
        part = subset(data, mask)
        records = []
        for predicted,q,valid in zip(prediction[mask], part['q'], part['mm']):
            x = predicted[valid];y = q[valid]
            i,j = np.triu_indices(len(x), 1)
            delta = y[i]-y[j];difference = x[i]-x[j]
            informative = np.abs(delta) > 1e-12
            correct = np.where(np.abs(difference) < 1e-12, .5, np.sign(delta) == np.sign(difference))
            weight = np.abs(delta[informative])
            best = x >= x.max()-1e-12
            records.append(dict(greedy_regret=float(y.max()-y[best].mean()),
                centered_rmse=float(np.sqrt(np.mean(((x-x.mean())-(y-y.mean()))**2))),
                weighted_pairwise_accuracy=float(np.sum(weight*correct[informative])/weight.sum()) if weight.size else None))
        output[phase] = dict(states=len(records), objective=score_gradient(part, model, objective)[0],
            **{key:float(np.mean([r[key] for r in records if r[key] is not None]))
               if any(r[key] is not None for r in records) else None for key in records[0]})
    return output


def fit_scores(rows, forecasts, holdout, hold_forecasts, initial, mode, objective, seed):
    check_rows(rows, holdout)
    train = score_data(rows, forecasts, mode)
    held = score_data(holdout, hold_forecasts, mode)
    model = copy.deepcopy(initial)
    params = model['parameters']
    moments = [{b:{k:np.zeros_like(v) for k,v in p.items()} for b,p in params.items()} for _ in range(2)]
    rng = np.random.default_rng(seed);step = 0;curve = []
    for epoch in range(65):
        if epoch in (0,4,16,32,64):
            curve.append(dict(epoch=epoch, steps=step, train=score_metrics(train, model, objective),
                              holdout=score_metrics(held, model, objective)))
        if epoch == 64:break
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), 8):
            loss,gradient = score_gradient(subset(train, order[start:start+8]), model, objective)
            norm = np.sqrt(sum(float(np.sum(x*x)) for branch in gradient.values() for x in branch.values()))
            if not np.isfinite(loss) or not np.isfinite(norm):raise FloatingPointError('nonfinite score update')
            scale = min(1., 5/max(norm, 1e-300));step += 1
            for branch,values in params.items():
                for key,value in values.items():
                    g = gradient[branch][key]*scale
                    moments[0][branch][key] = .9*moments[0][branch][key]+.1*g
                    moments[1][branch][key] = .999*moments[1][branch][key]+.001*g*g
                    update = moments[0][branch][key]/(1-.9**step)/(np.sqrt(moments[1][branch][key]/(1-.999**step))+1e-8)
                    candidate = value-.003*update
                    if not np.isfinite(candidate).all():raise FloatingPointError('nonfinite score parameters')
                    params[branch][key] = candidate
    return model,curve


def congestion_pressure(obs, plans, plan_probability):
    plan_probability = np.asarray(plan_probability, float)
    if plan_probability.shape != (len(plans),) or not np.isfinite(plan_probability).all() or np.any(plan_probability < 0) or not np.isclose(plan_probability.sum(), 1.):
        raise ValueError('normalized frozen anchor plan probabilities required')
    pressures = []
    for plan in plans:
        work = [[] for _ in obs['protocol'].capacity]
        for task in obs['tasks']:
            if task['source'] is None:continue
            target = task['target'] if task['target'] is not None else plan.get(task['uid'], task['source'])
            slack = task['deadline']-obs['time']
            work[target].append((slack, task['remaining_cycles']/obs['protocol'].capacity[target]))
        pressure = 0.
        for node in work:
            if not node:continue
            ordered = sorted(node)
            deadlines = np.array([x[0] for x in ordered])
            prefix = np.r_[0., np.cumsum([x[1] for x in ordered])]
            demand = prefix[np.searchsorted(deadlines, deadlines, side='right')]
            pressure = max(pressure, float(np.max(demand/np.maximum(deadlines, 1e-6))))
        pressures.append(pressure)
    return float(plan_probability@pressures)


def controlled_policy(anchor, predicted, pressure, decoder, branch='both', probability_target=None):
    anchor = np.asarray(anchor, float)
    predicted = np.asarray(predicted, float)
    if anchor.ndim != 2 or anchor.shape != predicted.shape or not anchor.size or not np.isfinite(predicted).all() or not np.isfinite(anchor).all() or np.any(anchor < 0) or not np.isclose(anchor.sum(), 1.):
        raise ValueError('finite compatible anchor and score bank required')
    if not np.isfinite(pressure) or pressure < 0 or decoder not in ('uniform','gated') or branch not in ('both','allocation','motion'):
        raise ValueError('registered pressure, decoder and branch required')
    alpha = .25 if decoder == 'uniform' or pressure > 1 else 0.
    if alpha == 0:
        return anchor.copy(), dict(alpha=0., pressure=float(pressure), tv=0., predicted_gain=0.)
    if probability_target is None:
        supported = anchor > 0
        best = supported & (predicted >= predicted[supported].max()-1e-12)
        target = anchor*best
        target /= target.sum()
    else:
        target = np.asarray(probability_target, float)
        if target.shape != anchor.shape or not np.isfinite(target).all() or np.any(target < 0) or not np.isclose(target.sum(),1.):
            raise ValueError('normalized matched probability target required')
    combined = (1-alpha)*anchor+alpha*target
    if branch != 'both':
        pa = anchor.sum(1, keepdims=True);qa = combined.sum(1, keepdims=True)
        pm = np.divide(anchor, pa, out=np.zeros_like(anchor), where=pa > 0)
        qm = np.divide(combined, qa, out=np.zeros_like(combined), where=qa > 0)
        combined = qa*pm if branch == 'allocation' else pa*qm
    tv = float(np.sum(np.abs(combined-anchor))/2)
    if tv > alpha+1e-10 or not np.isclose(combined.sum(),1.):raise AssertionError('bounded controller contract')
    return combined, dict(alpha=alpha, pressure=float(pressure), tv=tv,
                          predicted_gain=float(np.sum((combined-anchor)*predicted)))
