"""Bounded server-only public-model / finite-bank joint MPO experiment.

Invocation always requires --task-root. Existing run directories are refused;
--resume only executes stages without a completed stage manifest. Scientific
source snapshots are made by the launcher, not changed by this runner.
"""
import argparse
import copy
from functools import lru_cache
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
import numpy as np
from environment.queued_service import Engine, Protocol
from utils.queued_service_business_refinement import RecoveryScenario
from utils.queued_service_learning import LearningAdapter, VirtualQueue, state_features
from utils.queued_service_controllers import assignments
from utils.queued_joint_opportunity import expanded_bank
from utils.queued_joint_mpo import (value_features, arrival_prediction, forecast_window,
    unique_plans, restricted_prior, improve, divergence)
from marl_models.queued_joint_mpo import policy, loss_gradient, ppo_gradient
from scripts.analysis.queued_service_validation import START

TASK = 'chapter5_dynamic_service_20261028'
REV = 'queued_joint_mpo_20261102'
ARMS = ('joint', 'no_tail', 'decomposed', 'alternating', 'flat', 'motion_only', 'allocation_only', 'ppo_operator')
METHODS = ('initial', 'model', 'model_tail', 'parent')+ARMS
TIMES = tuple(range(5, 200, 10))


def seed(*parts): return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], 'big') % (2**63-1)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def out(root): return Path(root)/'results'/TASK/REV
def validation(root): return Path(root)/'output'/TASK/'validation'/REV
def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data, indent=2), encoding='utf-8'); os.replace(tmp, path)
def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp'); tmp.write_bytes(gzip.compress(pickle.dumps(data, 5), compresslevel=1, mtime=0)); os.replace(tmp, path)
@lru_cache(maxsize=8)
def read(path): return pickle.loads(gzip.decompress(Path(path).read_bytes()))
def parent_path(root, b): return Path(root)/'results'/TASK/'queued_long_value_20261102'/'models'/'expanded'/str(b)/'2.pt'


@lru_cache(maxsize=3)
def parent(root, b):
    import torch
    from scripts.analysis.queued_long_value_experiment import factory
    torch.set_num_threads(1)
    model = factory(); model.load_state_dict(torch.load(parent_path(root, b), map_location='cpu', weights_only=False)['model'])
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def proposal(model, obs, weights, generator_seed, fixed=None, greedy=False):
    import torch
    from marl_models.queued_service_joint import tensor
    prefix = VirtualQueue(obs); result = {}; logp = 0.
    generator = torch.Generator().manual_seed(generator_seed)
    with torch.no_grad():
        for task in obs['tasks']:
            if task['source'] is None or task['target'] is not None: continue
            state, features, mask, prior = prefix.features(task['uid'], weights)
            dist = model.allocation(tensor([state]), tensor([features]), tensor([mask], torch.bool), tensor([prior]))
            if fixed is not None: action = fixed[task['uid']]
            elif greedy: action = int(dist.probs[0].argmax())
            else: action = int(torch.multinomial(dist.probs[0], 1, generator=generator))
            assert mask[action]; logp += float(dist.log_prob(torch.tensor([action]))[0])
            result[task['uid']] = action; prefix.assign(task['uid'], action)
    return result, logp


def bank(model, obs, adapter, action_seed):
    adapter.prepare(obs); adapter.plans = expanded_bank(obs, adapter.plans)
    weights = adapter.history.weights
    plans = [proposal(model, obs, weights, seed(action_seed, 'greedy'), greedy=True)[0],
             proposal(model, obs, weights, seed(action_seed, 0))[0],
             proposal(model, obs, weights, seed(action_seed, 1))[0], assignments(obs)]
    plans = unique_plans(plans)
    logs = [proposal(model, obs, weights, 0, fixed=p)[1] for p in plans]
    prior, mass = restricted_prior(logs)
    n = len(adapter.plans); motion_features = []; allocation_features = []
    for plan in plans:
        prefix = VirtualQueue(obs)
        for uid, node in plan.items(): prefix.assign(uid, node)
        candidate = adapter.motion(prefix.obs)
        assert candidate['actions'] == adapter.plans
        m = np.asarray(candidate['features'][:n], dtype=np.float64)
        # asinh bounds extreme congestion while keeping physical sign/direction.
        m = np.arcsinh(m)
        a = np.r_[np.arcsinh(state_features(prefix.obs, weights)), m.mean(0), m.max(0), m.std(0)]
        motion_features.append(m); allocation_features.append(a)
    a = np.asarray(allocation_features); m = np.asarray(motion_features)
    # Subtract state-common components, not divide by a test-fitted scale.
    a = a-a.mean(0); m = m-m.mean(1, keepdims=True)
    return dict(a=a, m=m, prior=prior, retained_mass=mass, log_masses=logs,
                plans=plans, actions=copy.deepcopy(adapter.plans))


def mode_for(arm): return arm if arm in ('flat', 'decomposed') else 'joint'
def theta_path(root, arm, b, version): return out(root)/'models'/arm/str(b)/f'{version}.pkl.gz'


def tail_predict(tail, x, now, terminal_empty=False):
    if terminal_empty: return 0.
    times, means, y = tail['times'], tail['mean_x'], tail['mean_y']
    baseline = np.interp(now, times, y)
    center = np.array([np.interp(now, times, means[:, j]) for j in range(means.shape[1])])
    # Finite-count lower bound only; no use of actual remaining future count.
    return max(0., float(baseline+(x-center)@tail['coef']))


def teacher(row, obs, history, forecast_seed, tail):
    arrivals = arrival_prediction(obs, history, forecast_seed, 3)
    success = []; future = []; terminals = []
    for plan in row['plans']:
        sq, tq, xs = [], [], []
        for motion in row['actions']:
            prediction = forecast_window(obs, history, plan, motion, arrivals, 3)
            sq.append(prediction['success'])
            tq.append(tail_predict(tail, prediction['terminal'], prediction['end_time'], prediction['terminal_empty']))
            xs.append(prediction['terminal'])
        success.append(sq); future.append(tq); terminals.append(xs)
    row['short'] = np.asarray(success, dtype=float); row['tail'] = np.asarray(future, dtype=float)
    row['forecast_arrivals'] = [len(a) for a in arrivals]
    row['terminal_features'] = np.asarray(terminals)
    return row


def episode(args):
    root, b, w, arm, version, kind = args
    model = parent(root, b); tape = RecoveryScenario(w, 'burst', 'normal', .6)
    e = Engine(Protocol(), START, tape.ue_positions)
    rows, value_rows, rewards, decisions = [], [], [], []
    if kind == 'tail' or arm == 'parent':
        from utils.queued_long_value import LongAdapter
        adapter = LongAdapter(True)
    else: adapter = LearningAdapter()
    tail = None if kind == 'tail' or arm == 'parent' else read(str(out(root)/'tail'/f'{b}.pkl.gz'))
    theta = None if kind == 'tail' or arm in ('parent', 'initial', 'model', 'model_tail') else read(str(theta_path(root, arm, b, version)))['theta']
    while e.t < 200 or e.tasks:
        t = int(e.t); assert t <= 204
        if kind == 'tail' and (t % 5 == 0 or t >= 200):
            value_rows.append(dict(time=t, x=value_features(e.observe(), adapter.history), index=len(rewards)))
        e.add(tape.slot(t)); e.admit(); obs = e.observe(); begin = time.perf_counter()
        if kind == 'tail' or arm == 'parent':
            _, plan, move = model.act(obs, adapter, seed(REV, 'parent', b, w, t))
        else:
            row = bank(model, obs, adapter, seed(REV, 'bank', b, w, t))
            if theta is None: local_theta = np.zeros(row['a'].shape[1]+row['m'].shape[2])
            else: local_theta = theta
            probabilities = policy(row, local_theta, mode_for(arm))
            if arm in ('model', 'model_tail') or (kind == 'train' and t in TIMES):
                teacher(row, obs, adapter.history, seed(REV, 'forecast', b, w, t), tail)
            if arm in ('model', 'model_tail'):
                q = row['short']+(row['tail'] if arm == 'model_tail' else 0.)
                # Equal-valued maxima are sampled from the common prior.
                probabilities = probabilities*(q >= q.max()-1e-10); probabilities /= probabilities.sum()
            rng = np.random.default_rng(seed(REV, 'act', b, w, t))
            choice = int(rng.choice(probabilities.size, p=probabilities.ravel()))
            i, j = np.unravel_index(choice, probabilities.shape)
            plan, move = row['plans'][i], row['actions'][j]
            if kind == 'train' and t in TIMES:
                row.update(time=t, b=b, workload=w, chosen=(int(i), int(j)), probabilities=probabilities,
                           observed=value_features(obs, adapter.history))
                rows.append(row)
        decisions.append(time.perf_counter()-begin)
        e.assign(plan); transition = e.advance(move); rewards.append(transition['success'])
    audit = e.audit(); counts = dict(e.counts)
    assert counts['generated'] == counts.get('success', 0)+counts.get('expired', 0)
    assert sum(rewards) == counts.get('success', 0) and not e.tasks
    mc = np.cumsum(np.asarray(rewards, dtype=float)[::-1])[::-1]
    for v in value_rows: v['return'] = float(mc[v['index']])
    group = f'{kind}_{arm}_{version}'; key = f'{b}_{w}'
    record = dict(b=b, workload=w, arm=arm, version=version, kind=kind, counts=counts, audit=audit,
                  fingerprint=tape.fingerprint, parent_sha=sha(parent_path(root, b)),
                  decision_mean=float(np.mean(decisions)), decision_max=max(decisions),
                  over_second=sum(d > 1 for d in decisions), slots=len(rewards), successes=rewards,
                  training_rows=len(rows), value_rows=len(value_rows))
    write(out(root)/'episodes'/group/f'{key}.json', record)
    if rows: dump(out(root)/'experience'/group/f'{key}.pkl.gz', rows)
    if value_rows: dump(out(root)/'value_data'/f'{key}.pkl.gz', value_rows)
    return dict(file=str(Path('episodes')/group/f'{key}.json'), record=record)


def fit_tail(root, b):
    train, dev = [], []
    for w in range(11000, 11072): train += read(str(out(root)/'value_data'/f'{b}_{w}.pkl.gz'))
    for w in range(11100, 11116): dev += read(str(out(root)/'value_data'/f'{b}_{w}.pkl.gz'))
    times = sorted({r['time'] for r in train}); means = []; ys = []
    xx = []; yy = []
    for t in times:
        group = [r for r in train if r['time'] == t]; x = np.stack([r['x'] for r in group]); y = np.array([r['return'] for r in group])
        means.append(x.mean(0)); ys.append(y.mean()); xx.extend(x-x.mean(0)); yy.extend(y-y.mean())
    x = np.asarray(xx); y = np.asarray(yy)
    coef = np.linalg.solve(x.T@x+np.eye(x.shape[1]), x.T@y)
    tail = dict(times=np.asarray(times), mean_x=np.asarray(means), mean_y=np.asarray(ys), coef=coef,
                train_workloads=list(range(11000, 11072)), dev_workloads=list(range(11100, 11116)))
    diagnostics = {}
    for name, data in (('train', train), ('dev', dev)):
        truth = np.array([r['return'] for r in data]); pred = np.array([tail_predict(tail, r['x'], r['time']) for r in data])
        cy, cp = [], []
        for t in sorted({r['time'] for r in data}):
            ix = [i for i, r in enumerate(data) if r['time'] == t]
            cy.extend(truth[ix]-truth[ix].mean()); cp.extend(pred[ix]-pred[ix].mean())
        cy, cp = np.asarray(cy), np.asarray(cp)
        diagnostics[name] = dict(n=len(data), rmse=float(np.sqrt(np.mean((truth-pred)**2))),
            ev=float(1-np.var(truth-pred)/max(np.var(truth), 1e-12)),
            centered_ev=float(1-np.var(cy-cp)/max(np.var(cy), 1e-12)),
            centered_rmse=float(np.sqrt(np.mean((cy-cp)**2))))
    dump(out(root)/'tail'/f'{b}.pkl.gz', tail); write(out(root)/'tail'/f'{b}.json', diagnostics)
    return diagnostics


def init_worker(cpus):
    os.sched_setaffinity(0, {cpus[(mp.current_process()._identity[-1]-1) % len(cpus)]})


def timed_job(payload):
    function, args = payload; start = time.time(); cpu = time.process_time()
    try:
        result = globals()[function](*args)
        return dict(ok=True, result=result, start=start, end=time.time(), cpu_seconds=time.process_time()-cpu,
                    pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)), threads=1)
    except Exception:
        import traceback
        return dict(ok=False, error=traceback.format_exc(), start=start, end=time.time(), cpu_seconds=time.process_time()-cpu,
                    pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)), threads=1)


def parallel(pool, root, label, function, args):
    started = time.perf_counter(); records = []
    futures = {pool.submit(timed_job, (function, (a,))): i for i, a in enumerate(args)}
    ordered = [None]*len(args)
    for future in as_completed(futures):
        d = future.result(); i = futures[future]; d['job'] = i; records.append(d); ordered[i] = d.get('result')
        if len(records) % 32 == 0: print(label, len(records), '/', len(args), flush=True)
    elapsed = time.perf_counter()-started; cpu = sum(d['cpu_seconds'] for d in records)
    events = sorted([(d['start'], 1) for d in records]+[(d['end'], -1) for d in records])
    active = peak = 0
    for _, change in events: active += change; peak = max(peak, active)
    report = dict(label=label, tasks=len(args), allowed_cpus=sorted(os.sched_getaffinity(0)),
                  workers=len(os.sched_getaffinity(0)), wall_seconds=elapsed, worker_cpu_seconds=cpu,
                  equivalent_cpus=cpu/max(elapsed, 1e-9), utilization=cpu/max(elapsed, 1e-9)/len(os.sched_getaffinity(0)),
                  peak_active=peak, failures=sum(not d['ok'] for d in records), retries=0,
                  records=records)
    write(validation(root)/'resources'/f'{label}.json', report)
    if report['failures']: raise RuntimeError(f'{label}: {report["failures"]} failures; inspect resource manifest')
    print(label, 'DONE', round(elapsed, 1), 'seconds; effective CPUs', round(report['equivalent_cpus'], 1), flush=True)
    return ordered


def reduce_fit(args):
    path, theta, oldtheta, previous, mode, operator = args
    rows, targets, oldps, advantages = read(path)
    loss = 0.; gradient = np.zeros_like(theta); kl = delta = qkl = expected = akl = mkl = 0.
    for row, target, oldp, advantage in zip(rows, targets, oldps, advantages):
        p = policy(row, theta, mode); prev = policy(row, previous, mode)
        l, g = ppo_gradient(row, theta, oldp, advantage) if operator == 'ppo' else loss_gradient(row, theta, target, mode)
        loss += l; gradient += g; kl += divergence(oldp, p); delta += divergence(prev, p); qkl += divergence(target, p)
        pa, oa = p.sum(1), oldp.sum(1); akl += divergence(oa, pa)
        mkl += sum(oa[i]*divergence(oldp[i]/oa[i], p[i]/pa[i]) for i in range(len(pa)))
        expected += float(np.sum((p-oldp)*row['q']))
    return dict(n=len(rows), loss=loss, gradient=gradient, kl=kl, delta=delta, qkl=qkl, expected=expected, akl=akl, mkl=mkl)


def small_parallel(pool, payloads):
    """Frequent full-batch reductions; timings accumulated by caller, no logs per trial."""
    futures = [pool.submit(timed_job, ('reduce_fit', (x,))) for x in payloads]
    results = [f.result() for f in futures]
    errors = [r for r in results if not r['ok']]
    if errors: raise RuntimeError(errors[0]['error'])
    return results


def fit_all(pool, root, version):
    started = time.perf_counter(); parent_cpu = time.process_time(); sessions = []; timing = []
    for arm in ARMS:
        for b in range(3):
            rows = []
            for w in range(11200+(version-1)*100, 11216+(version-1)*100):
                rows += read(str(out(root)/'experience'/f'train_{arm}_{version-1}'/f'{b}_{w}.pkl.gz'))
            mode = mode_for(arm)
            theta = np.zeros(rows[0]['a'].shape[1]+rows[0]['m'].shape[2]) if version == 1 else read(str(theta_path(root, arm, b, version-1)))['theta'].copy()
            oldps = [policy(r, theta, mode) for r in rows]
            values = [r['short']+(r['tail'] if arm != 'no_tail' else 0.) for r in rows]
            targets, estep = improve(oldps, values, .1)
            advantages = [v-float(np.sum(p*v)) for p, v in zip(oldps, values)]
            rms = max(float(np.sqrt(np.mean([np.sum(p*a*a) for p, a in zip(oldps, advantages)]))), 1e-8)
            advantages = [a/rms for a in advantages]
            paths = []
            for index in range(0, len(rows), 16):
                subset = rows[index:index+16]
                for r, v in zip(subset, values[index:index+16]): r['q'] = v
                path = out(root)/'fit_chunks'/f'{version}_{arm}_{b}_{index}.pkl.gz'
                dump(path, (subset, targets[index:index+16], oldps[index:index+16], advantages[index:index+16])); paths.append(str(path))
            sessions.append(dict(arm=arm, b=b, mode=mode, theta=theta, old=theta.copy(), m=np.zeros_like(theta), v=np.zeros_like(theta), adam_step=0,
                                 paths=paths, estep=estep, accepted=[], rejected=0))
    def reduce_all(candidates, previous, selected=None):
        payloads = []; owners = []
        for i, (s, theta, prev) in enumerate(zip(sessions, candidates, previous)):
            if selected is not None and i not in selected: continue
            for path in s['paths']:
                payloads.append((path, theta, s['old'], prev, s['mode'], 'ppo' if s['arm'] == 'ppo_operator' else 'mpo')); owners.append(i)
        results = small_parallel(pool, payloads); sums = [None]*len(sessions)
        for i, r in zip(owners, results):
            timing.append({k: r[k] for k in ('start', 'end', 'cpu_seconds', 'pid', 'affinity', 'threads')})
            v = r['result']
            if sums[i] is None: sums[i] = v
            else:
                for key in v: sums[i][key] += v[key]
        for d in sums:
            if d is None: continue
            for key in d:
                if key != 'n': d[key] /= d['n']
        return sums
    current = reduce_all([s['theta'] for s in sessions], [s['theta'] for s in sessions])
    for step in range(12):
        proposed_states = []
        for s, d in zip(sessions, current):
            g = d['gradient'].copy(); cut = len(g)-44
            active = 'allocation' if s['arm'] == 'allocation_only' or (s['arm'] == 'alternating' and version == 1) else 'motion' if s['arm'] == 'motion_only' or (s['arm'] == 'alternating' and version == 2) else 'both'
            if active == 'allocation': g[cut:] = 0
            if active == 'motion': g[:cut] = 0
            m = .9*s['m']+.1*g; v = .999*s['v']+.001*g*g; count = s['adam_step']+1
            direction = m/(1-.9**count)/(np.sqrt(v/(1-.999**count))+1e-8)
            proposed_states.append((m, v, count, direction))
        pending = set(range(len(sessions))); accepted = {}
        before = [s['theta'].copy() for s in sessions]
        for scale in (8., 4., 2., 1., .5, .25, .125, .0625, .03125, .015625):
            candidates = [before[i]-.03*scale*proposed_states[i][3] if i in pending else sessions[i]['theta'] for i in range(len(sessions))]
            checks = reduce_all(candidates, before, pending)
            for i in list(pending):
                d = checks[i]
                if np.isfinite(d['loss']) and d['loss'] <= current[i]['loss']+1e-12 and d['kl'] <= .02+1e-10 and d['delta'] <= .002+1e-10:
                    sessions[i]['theta'] = candidates[i]; sessions[i]['m'], sessions[i]['v'], sessions[i]['adam_step'], _ = proposed_states[i]
                    sessions[i]['accepted'].append(dict(step=step, scale=scale, kl=d['kl'], delta=d['delta'], loss=d['loss']))
                    accepted[i] = d; pending.remove(i)
            if not pending: break
        for i in pending: sessions[i]['rejected'] += 1
        # Full failure preserves theta, m, v AND Adam step. No stale accepted gradient.
        current = reduce_all([s['theta'] for s in sessions], [s['theta'] for s in sessions])
        print('fit', version, step, 'accepted', len(accepted), '/', len(sessions), flush=True)
    for s, d in zip(sessions, current):
        dump(theta_path(root, s['arm'], s['b'], version), dict(theta=s['theta'], adam_m=s['m'], adam_v=s['v'], adam_step=s['adam_step']))
        write(out(root)/'updates'/f'{version}_{s["arm"]}_{s["b"]}.json', dict(arm=s['arm'], b=s['b'], version=version, estep=s['estep'],
              final={k: v for k, v in d.items() if k != 'gradient'}, accepted=s['accepted'], rejected=s['rejected'],
              allocation_change=float(np.linalg.norm((s['theta']-s['old'])[:-44])), motion_change=float(np.linalg.norm((s['theta']-s['old'])[-44:]))))
    elapsed = time.perf_counter()-started; cpu = sum(r['cpu_seconds'] for r in timing); own = time.process_time()-parent_cpu
    write(validation(root)/'resources'/f'fit_{version}.json', dict(tasks=len(timing), allowed_cpus=sorted(os.sched_getaffinity(0)),
        workers=len(os.sched_getaffinity(0)), threads=1, wall_seconds=elapsed, worker_cpu_seconds=cpu, coordinator_cpu_seconds=own,
        equivalent_cpus=(cpu+own)/elapsed, utilization=(cpu+own)/elapsed/len(os.sched_getaffinity(0)),
        failures=0, retries=0, records=timing, note='Sequential Adam and global E-step coordinator are measured serial sections. All gradient/KL candidate evaluations are state-sharded.'))


def protect(root):
    paths = []
    for name in ('source_long_value_20261102', 'source_long_unfreeze_20261102', 'source_motion_trust_20261102',
                 'source_motion_trust_collectfix_20261102', 'source_probability_decoupling_20261102'):
        paths += [p for p in (Path(root)/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for name in ('queued_long_value_20261102', 'queued_probability_decoupling_20261102'):
        paths += list((Path(root)/'results'/TASK/name/'models').rglob('*.pt'))
    return {str(p.relative_to(root)): sha(p) for p in sorted(set(paths))}


def analyze(root):
    results = {}; records = {}
    for arm in METHODS:
        entries = [json.loads(p.read_text()) for p in (out(root)/'episodes'/f'test_{arm}_2').glob('*.json')]
        assert len(entries) == 48, (arm, len(entries)); records[arm] = entries
        results[arm] = []
        for b in range(3):
            rows = [r for r in entries if r['b'] == b]; total = sum(r['counts']['generated'] for r in rows); good = sum(r['counts'].get('success', 0) for r in rows)
            results[arm].append(dict(b=b, dsr=good/total, success=good, generated=total, decision_mean=statistics.mean(r['decision_mean'] for r in rows)))
    comparison = {}
    rng = np.random.default_rng(20261102)
    for other in METHODS:
        if other == 'joint': continue
        a = np.array([[next(r for r in records['joint'] if r['b'] == b and r['workload'] == w)['counts'].get('success', 0) for w in range(11400,11416)] for b in range(3)])
        z = np.array([[next(r for r in records[other] if r['b'] == b and r['workload'] == w)['counts'].get('success', 0) for w in range(11400,11416)] for b in range(3)])
        n = np.array([next(r for r in records['joint'] if r['b'] == 0 and r['workload'] == w)['counts']['generated'] for w in range(11400,11416)])
        diffs = (a-z).sum(1)/n.sum(); boot = []
        for _ in range(4000):
            bs = rng.integers(0,3,3); ws = rng.integers(0,16,16)
            boot.append(float(np.mean((a-z)[bs][:,ws].sum(1)/n[ws].sum())))
        comparison[other] = dict(mean=float(diffs.mean()), by_lineage=diffs.tolist(), crossed_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                                 note='Three inherited lineages; exploratory crossed lineage/workload bootstrap, not independent confirmation.')
    updates = [json.loads(p.read_text()) for p in (out(root)/'updates').glob('*.json')]
    assert len(updates) == 48
    tails = [json.loads((out(root)/'tail'/f'{b}.json').read_text()) for b in range(3)]
    mechanisms = all(u['final']['kl'] <= .02000001 and abs(u['final']['kl']-u['final']['akl']-u['final']['mkl']) < 1e-8 for u in updates)
    both_changed = all(u['allocation_change'] > 0 and u['motion_change'] > 0 for u in updates if u['arm'] == 'joint')
    performance = all(comparison[k]['crossed_bootstrap95'][0] > 0 and min(comparison[k]['by_lineage']) > 0 for k in ('initial','no_tail','decomposed','alternating','flat','model','model_tail'))
    result = dict(status='COMPLETE', summary=results, comparisons=comparison, tail=tails,
                  mechanism=mechanisms, joint_both_branches_changed=both_changed,
                  performance_gate=performance, tail_centered_gate=all(x['dev']['centered_ev'] > 0 for x in tails))
    write(out(root)/'analysis.json', result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'run', 'analyze'), default='run'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); root = args.task_root.resolve()
    if os.name != 'posix' or not hasattr(os, 'sched_getaffinity'): raise RuntimeError('server only')
    if args.mode == 'analyze': analyze(root); return
    target = out(root)
    if args.mode == 'smoke':
        smoke(root); return
    lock = validation(root)/'run.lock'; lock.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not args.resume: raise FileExistsError('existing scientific output; refuse initialization')
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        target.mkdir(parents=True, exist_ok=True)
        protection = validation(root)/'protection_before.json'
        if not protection.exists(): write(protection, protect(root))
        cpus = sorted(os.sched_getaffinity(0))
        contract = dict(revision=REV, parent={str(b): sha(parent_path(root,b)) for b in range(3)},
                        source={str(p.relative_to(REPO_ROOT)): sha(p) for folder in ('environment','utils','marl_models','scripts/analysis','tests') for p in (REPO_ROOT/folder).rglob('*.py')},
                        cpus=cpus, arms=ARMS, train=[list(range(11200,11216)),list(range(11300,11316))], test=list(range(11400,11416)), horizon=3)
        if not (target/'protocol.json').exists(): write(target/'protocol.json', contract)
        def done(name): return (validation(root)/'stages'/f'{name}.json').exists()
        def mark(name): write(validation(root)/'stages'/f'{name}.json', dict(status='COMPLETE', time=time.time()))
        # No Torch import/initialization in the coordinator before Linux fork.
        with ProcessPoolExecutor(max_workers=len(cpus), mp_context=mp.get_context('fork'), initializer=init_worker, initargs=(cpus,)) as pool:
            if not done('tail_collection'):
                jobs = [(str(root),b,w,'parent',0,'tail') for b in range(3) for w in (*range(11000,11072),*range(11100,11116))]
                parallel(pool, root, 'tail_collection', 'episode', jobs); mark('tail_collection')
            if not done('tail_fit'):
                start, cpu = time.perf_counter(), time.process_time()
                for b in range(3): fit_tail(root, b)
                elapsed = time.perf_counter()-start; used = time.process_time()-cpu
                write(validation(root)/'resources'/'tail_fit.json', dict(tasks=3, wall_seconds=elapsed, worker_cpu_seconds=used,
                    equivalent_cpus=used/elapsed, utilization=used/elapsed/len(cpus), allowed_cpus=cpus, workers=1, threads=1,
                    note='99-dimensional closed-form solve; serial measured microstage; expensive trajectory/value data collected on all CPUs.'))
                mark('tail_fit')
            for version in (1,2):
                # Initial actors are created only after output/lock protection.
                if version == 1:
                    for arm in ARMS:
                        for b in range(3):
                            path = theta_path(root,arm,b,0)
                            if not path.exists(): dump(path, dict(theta=np.zeros(256)))
                label = f'train_{version}'
                if not done(label):
                    jobs = [(str(root),b,w,arm,version-1,'train') for arm in ARMS for b in range(3) for w in range(11200+(version-1)*100,11216+(version-1)*100)]
                    parallel(pool, root, label, 'episode', jobs); mark(label)
                label = f'fit_{version}'
                if not done(label): fit_all(pool,root,version); mark(label)
            if not done('test'):
                jobs = [(str(root),b,w,arm,2,'test') for arm in METHODS for b in range(3) for w in range(11400,11416)]
                parallel(pool,root,'test','episode',jobs); mark('test')
        previous = json.loads(protection.read_text()); current = protect(root); assert previous == current
        write(validation(root)/'protection_after.json', dict(status='PASS', files=len(current), hashes=current))
        analyze(root); mark('analysis')
    finally:
        lock.unlink(missing_ok=True)


def smoke(root):
    """Real complete collector + public forecasts; separate from scientific IDs."""
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_queued_joint_mpo')
    result = unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful(): raise AssertionError('mechanism tests')
    model = parent(str(root),0); tape = RecoveryScenario(11999,'burst','normal',.6)
    e = Engine(Protocol(),START,tape.ue_positions); adapter=LearningAdapter(); probes=[]
    while e.t < 200 or e.tasks:
        t=int(e.t); e.add(tape.slot(t)); e.admit(); obs=e.observe()
        row=bank(model,obs,adapter,seed('smoke',t)); theta=np.zeros(row['a'].shape[1]+row['m'].shape[2])
        assert len(theta)==256
        if t in (5,75,175,200):
            before=pickle.dumps((obs,adapter.history),5); arrivals=arrival_prediction(obs,adapter.history,seed('forecast',t))
            values=[]
            for plan in row['plans']:
                values.append([forecast_window(obs,adapter.history,plan,m,arrivals)['success'] for m in row['actions']])
            assert before==pickle.dumps((obs,adapter.history),5)
            p=policy(row,theta); q,info=improve([p],[np.asarray(values)],.1)
            probes.append(dict(time=t,plans=len(row['plans']),motions=len(row['actions']),retained_mass=row['retained_mass'],estep=info))
        p=policy(row,theta); rng=np.random.default_rng(seed('smoke_act',t)); k=int(rng.choice(p.size,p=p.ravel())); i,j=np.unravel_index(k,p.shape)
        e.assign(row['plans'][i]); e.advance(row['actions'][j])
    audit=e.audit(); assert e.counts['generated']==e.counts['success']+e.counts['expired']
    write(validation(root)/'smoke.json',dict(status='PASS',counts=dict(e.counts),audit=audit,probes=probes,parent_sha=sha(parent_path(root,0))))


if __name__=='__main__': main()
