"""Fixed-data fitting and public-information diagnostic primitives."""
import copy
import itertools
from functools import lru_cache
import numpy as np
from marl_models.queued_planner_distillation import inputs
from utils.queued_closed_loop import _masked_softmax
from utils.queued_planner_distillation import planner_target

REVISION = 'queued_learning_validation_20261104'
ROOT_TIMES = (10, 30, 50, 60, 70, 80, 90, 100, 120, 150, 176, 190)
CASE_TIMES = (40, 60, 80, 100, 150, 180)


def phase(t):
    return 'peak' if 50 <= t < 100 else 'nonpeak'


def case_grid():
    return [dict(id=i, peak=p, range=r, sla=s) for i, (p, r, s) in enumerate(
        itertools.product((.45, .6, .75), (460., 360., 260.), ('tight', 'normal', 'relaxed')))]


def tiny_indices(rows):
    first = min(r['workload'] for r in rows)
    selected = []
    for p in ('peak', 'nonpeak'):
        ids = sorted((i for i, r in enumerate(rows) if r['workload'] == first and phase(r['time']) == p),
                     key=lambda i: rows[i]['time'])
        if len(ids) < 16:
            raise ValueError('need 16 states in each phase of the first workload')
        selected.extend(ids[:16])
    return sorted(selected)


@lru_cache(maxsize=12)
def _projection_matrix(dim, width):
    rng = np.random.default_rng(2026110401+dim)
    return rng.normal(size=(dim, width))/np.sqrt(dim)


def projection(x, width=64):
    x = np.asarray(x, float)
    return np.tanh(np.arcsinh(x)@_projection_matrix(x.shape[-1], width))


def pack(rows, mode):
    xs = [inputs(r, mode) for r in rows]
    n = len(rows);k = max(x[0].shape[0] for x in xs);m = max(x[1].shape[1] for x in xs)
    data = dict(xa=np.zeros((n, k, xs[0][0].shape[-1])), xm=np.zeros((n, k, m, xs[0][1].shape[-1])),
                prior=np.zeros((n, k)), ma=np.zeros((n, k), bool), mm=np.zeros((n, k, m), bool),
                target=np.zeros((n, k, m)), q=np.zeros((n, k, m)), times=np.array([r['time'] for r in rows]))
    for i, (r, (xa, xm, prior)) in enumerate(zip(rows, xs)):
        ki, mi = xm.shape[:2]
        data['xa'][i, :ki] = xa;data['xm'][i, :ki, :mi] = xm;data['prior'][i, :ki] = prior
        data['ma'][i, :ki] = True;data['mm'][i, :ki, :mi] = True
        data['target'][i, :ki, :mi] = planner_target(r['q'], prior);data['q'][i, :ki, :mi] = r['q']
    return data


def subset(data, ids):
    return {k: v[ids] for k, v in data.items()}


def statistics(data, model):
    params = model['parameters'];n = len(data['xa'])
    def forward(x, b):
        if 'w' in b:
            return x@b['w'], None
        h = np.tanh(x@b['W']+b['b'])
        return h@b['v'], h
    za, ha = forward(data['xa'], params['allocation'])
    zm, hm = forward(data['xm'], params['motion'])
    pa = _masked_softmax(np.log(data['prior'].clip(1e-300))+za, data['ma'], 1)
    pm = _masked_softmax(zm, data['mm'], 2)
    p = pa[:, :, None]*pm;t = data['target'];qa = t.sum(2)
    loss = -float(np.sum(t*(np.log(pa.clip(1e-300))[:, :, None]+np.log(pm.clip(1e-300))))/n)
    def back(x, h, d, b):
        if h is None:
            return dict(w=x.reshape(-1, x.shape[-1]).T@d.reshape(-1)/n)
        dh = d[..., None]*b['v']*(1-h*h)
        return dict(W=x.reshape(-1, x.shape[-1]).T@dh.reshape(-1, dh.shape[-1])/n,
                    b=dh.reshape(-1, dh.shape[-1]).sum(0)/n,
                    v=h.reshape(-1, h.shape[-1]).T@d.reshape(-1)/n)
    gradient = dict(allocation=back(data['xa'], ha, pa-qa, params['allocation']),
                    motion=back(data['xm'], hm, qa[:, :, None]*pm-t, params['motion']))
    q = data['q'];maximum = np.max(np.where(data['mm'], q, -np.inf), axis=(1, 2))
    uniform_motion = np.divide(data['mm'], data['mm'].sum(2, keepdims=True),
        out=np.zeros_like(p), where=data['mm'].sum(2, keepdims=True) > 0)
    baseline = np.sum(data['prior'][:, :, None]*uniform_motion*q, axis=(1, 2))
    expected = np.sum(p*q, axis=(1, 2));available = maximum-baseline
    entropy = -float(np.sum(t*np.log(t.clip(1e-300)))/n)
    result = dict(states=n, loss=loss, teacher_kl=loss-entropy,
                  score_regret=float(np.mean(maximum-expected)),
                  optimal_probability=float(np.mean(np.sum(p*((q >= maximum[:, None, None]-1e-12)&data['mm']), axis=(1, 2)))),
                  retained_planning_gain=float(np.sum(expected-baseline)/available.sum()) if available.sum() > 1e-12 else None)
    return result, gradient


def evaluate(data, model):
    result = {'all': statistics(data, model)[0]}
    for p in ('peak', 'nonpeak'):
        mask = (data['times'] >= 50)&(data['times'] < 100)
        if p == 'nonpeak':mask = ~mask
        result[p] = statistics(subset(data, mask), model)[0] if mask.any() else None
    return result


def train_curve(rows, heldout, initial, mode, epochs, checkpoints, seed):
    """Continuous Adam on frozen data, fixed epochs, no holdout-based decisions."""
    data = pack(rows, mode);hold = pack(heldout, mode) if heldout else None
    model = copy.deepcopy(initial);params = model['parameters'];step = 0
    moments = [{b: {k: np.zeros_like(v) for k, v in p.items()} for b, p in params.items()} for _ in range(2)]
    rng = np.random.default_rng(seed);curve = []
    def record(epoch):
        curve.append(dict(epoch=epoch, steps=step, train=evaluate(data, model),
                          holdout=evaluate(hold, model) if hold is not None else None))
    record(0)
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), 8):
            stat, gradient = statistics(subset(data, order[start:start+8]), model)
            norm = np.sqrt(sum(float(np.sum(g*g)) for b in gradient.values() for g in b.values()))
            if not np.isfinite(stat['loss']) or not np.isfinite(norm):
                raise FloatingPointError('nonfinite fixed-data diagnostic')
            scale = min(1., 5./max(norm, 1e-300));step += 1
            for b, values in params.items():
                for k, value in values.items():
                    g = gradient[b][k]*scale
                    moments[0][b][k] = .9*moments[0][b][k]+.1*g
                    moments[1][b][k] = .999*moments[1][b][k]+.001*g*g
                    update = moments[0][b][k]/(1-.9**step)/(np.sqrt(moments[1][b][k]/(1-.999**step))+1e-8)
                    proposal = value-.003*update
                    if not np.isfinite(proposal).all():raise FloatingPointError('nonfinite proposal')
                    params[b][k] = proposal
        if epoch in checkpoints:record(epoch)
    return model, curve


def label_stability(qs, prior):
    qs = np.asarray(qs, float)
    if qs.ndim != 3 or len(qs) < 4 or len(qs) % 2:
        raise ValueError('even paired forecast repetitions required')
    half = len(qs)//2;a = qs[:half].mean(0);b = qs[half:].mean(0)
    ta, tb = planner_target(a, prior), planner_target(b, prior)
    centered = qs-qs.mean((1, 2), keepdims=True)
    return dict(single_prediction_variance=float(np.mean(centered.var(0, ddof=1))),
                split_selection_overlap=float(np.minimum(ta, tb).sum()),
                split_selection_regret=float((b.max()-np.sum(ta*b)+a.max()-np.sum(tb*a))/2),
                mean_prediction_range=float(np.ptp(qs.mean(0))))


def task_descriptors(obs, plan, motion):
    """32 EDF task records x 16 public fields, no task IDs in feature values."""
    from environment.queued_service import distance
    tasks = sorted(obs['tasks'], key=lambda x: (x['deadline'], x['uid']))
    p = obs['protocol'];pos = [(a[0]+d[0], a[1]+d[1]) for a, d in zip(obs['positions'], motion)]
    values = np.zeros((32, 16))
    for i, x in enumerate(tasks[:32]):
        original_source = x['source'];waiting = original_source is None
        source = min(range(len(pos)), key=lambda j: distance(pos[j], obs['ues'][x['ue']])) if waiting else original_source
        target = x['target'] if x['target'] is not None else plan.get(x['uid'], source)
        target_position = p.mbs if target == len(pos) else pos[target]
        link_distance = distance(pos[source], target_position)
        transfer = x['remaining_transfer'] if x['target'] is not None else x['bits'] if target != source else 0.
        queued = sum(z['remaining_cycles'] for z in tasks if (z['target'] if z['target'] is not None else plan.get(z['uid'])) == target
                     and z['deadline'] <= x['deadline'])/p.capacity[target]
        upload_distance = distance(pos[source], obs['ues'][x['ue']])
        values[i] = [(x['deadline']-obs['time'])/3, x['remaining_upload']/24e6, transfer/24e6,
            x['remaining_cycles']/13.5e9, (obs['time']-x['arrival'])/3,
            ('waiting', 'upload', 'transfer', 'compute').index(x['stage'])/3,
            p.capacity[source]/85e9, p.capacity[target]/85e9, upload_distance/p.coverage,
            link_distance/p.inter_range, queued/3, float(upload_distance <= p.coverage),
            float(target == source or target == len(pos) or link_distance <= p.inter_range),
            float(x['target'] is None), float(target == source), float(waiting)]
    return values.ravel(), max(0, len(tasks)-32)


def augment(row, extra):
    extra = np.asarray(extra, float)
    if extra.shape[:2] != row['m'].shape[:2]:raise ValueError('one public feature per paired candidate')
    result = dict(row)
    result['m'] = np.concatenate([row['m'], extra], axis=2)
    result['a'] = np.concatenate([row['a'], extra.mean(1)], axis=1)
    return result
