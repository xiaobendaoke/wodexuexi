"""Bounded, ordered fit / information / applicability diagnostics on server.

Historical source, actors and results are immutable. New case grids have their
own protocol; frozen-policy transfer is not same-condition retrained MARL.
"""
import argparse
import copy
from dataclasses import replace
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_planner_distillation as original
from scripts.analysis import queued_joint_mpo_experiment as t
from scripts.analysis.analyze_queued_planner_distillation import conditional_opportunity, teacher_projection
from marl_models.queued_planner_distillation import initialize, policy
from utils.queued_planner_distillation import specification, check_rows
from utils.queued_learning_validation import (REVISION, ROOT_TIMES, CASE_TIMES, case_grid, phase,
    tiny_indices, projection, task_descriptors, augment, train_curve, label_stability)
from utils.queued_service_business import SLA

TASK = original.TASK


def out(root):return Path(root)/'results'/TASK/REVISION
def val(root):return Path(root)/'output'/TASK/'validation'/REVISION
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read(p):return pickle.loads(gzip.decompress(Path(p).read_bytes()))


def write(p, value):
    if Path(p).exists():raise FileExistsError(str(p))
    t.write(p, value)


def dump(p, value):
    if Path(p).exists():raise FileExistsError(str(p))
    t.dump(p, value)


def source_protection(root):
    r = original.out(root)
    return {**{str(p.relative_to(root)): t.sha(p) for p in r.rglob('*') if p.is_file()}, **original.protected(root)}


def old_rows(root, arm, run, split):
    rows = []
    for p in sorted((original.out(root)/'experience'/split/arm).glob(f'{run}_*.pkl.gz')):
        rows.extend(read(p))
    return rows


def fit_task(root, arm, run, small):
    train = old_rows(root, arm, run, 'training');hold = old_rows(root, arm, run, 'holdout')
    check_rows(train, hold)
    if small:train = [train[i] for i in tiny_indices(train)]
    model = read(original.model_path(root, arm, run, 2));mode = original.actor_mode(arm)
    epochs = 256 if small else 64;checkpoints = (1, 4, 16, 32, 64, 128, 256) if small else (1, 4, 16, 32, 64)
    started = time.perf_counter()
    fitted, curve = train_curve(train, [] if small else hold, model, mode, epochs, checkpoints,
                                 t.seed(REVISION, 'fit', run, small))
    stem = f'{arm}_{run}_'+('tiny' if small else 'full')
    dump(out(root)/'stage1/models'/(stem+'.pkl.gz'), fitted)
    report = dict(arm=arm, run=run, small=small, train_states=len(train), train_workloads=len({r['workload'] for r in train}),
                  fixed_epochs=epochs, optimizer_reset_once=True, holdout_used_for_selection=False,
                  seconds=time.perf_counter()-started, curve=curve)
    write(out(root)/'stage1/fits'/(stem+'.json'), report)
    return report['seconds']


def feature_extras(row, bank, obs):
    k, m = row['m'].shape[:2]
    raw = np.concatenate([np.broadcast_to(row['a'][:, None, :], (k, m, row['a'].shape[1])),
                          row['m'], np.broadcast_to(row['context'], (k, m, len(row['context'])))], axis=2)
    task = [];truncated = 0
    for plan in bank['plans']:
        line = []
        for action in bank['actions']:
            descriptor, skipped = task_descriptors(obs, plan, action)
            line.append(descriptor);truncated = max(truncated, skipped)
        task.append(line)
    return projection(raw), projection(np.asarray(task)), truncated


def replay_task(root, run, version, split, workload):
    cfg = next(r for r in specification()['runs'] if r['id'] == run);parent = cfg['parent']
    stem = f'{run}_{parent}_{workload}_0_{version}'
    historical = load(original.out(root)/'episodes'/split/'mlp_joint'/(stem+'.json'))
    expected = read(original.out(root)/'experience'/split/'mlp_joint'/(stem+'.pkl.gz'))
    by_time = {r['time']: r for r in expected}
    proposer = t.parent(root, parent);tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions);adapter = t.LearningAdapter()
    actor = None if version == 0 else read(original.model_path(root, 'mlp_joint', run, version))
    states = [];successes = [];checked = 0
    while engine.t < 200 or engine.tasks:
        now = int(engine.t)
        if now > 204:raise AssertionError('replay tail')
        engine.add(tape.slot(now));engine.admit();obs = engine.observe()
        bank = t.bank(proposer, obs, adapter, t.seed(original.REVISION, 'bank', parent, workload, 0, now))
        row = original.public_row(bank, obs, adapter.history)
        if actor is None:actor = original.initial_model(row, 'mlp_joint', run)
        p = policy(row, actor, 'joint')
        if now in by_time:
            for key in ('a', 'm', 'prior', 'context'):
                np.testing.assert_allclose(row[key], by_time[now][key], atol=1e-12, rtol=1e-12)
            checked += 1
        if now in ROOT_TIMES:
            q = original.short_scores(bank, obs, adapter.history, t.seed(original.REVISION, 'forecast', parent, workload, 0, now))
            np.testing.assert_allclose(q, by_time[now]['q'], atol=1e-12)
            summary, detailed, truncated = feature_extras(row, bank, obs)
            row.update(q=q, time=now, split=split, workload=workload, parent=parent, run=run, version=version)
            states.append(dict(row=row, bank=bank, obs=obs, history=copy.deepcopy(adapter.history),
                               summary=summary, task=detailed, truncated=truncated))
        rng = np.random.default_rng(t.seed(original.REVISION, 'action', parent, workload, 0, now))
        i, j = np.unravel_index(int(rng.choice(p.size, p=p.ravel())), p.shape)
        engine.assign(bank['plans'][i]);successes.append(engine.advance(bank['actions'][j])['success'])
    assert checked == 100 and len(states) == 12
    assert dict(engine.counts) == historical['counts'] and successes == historical['successes']
    assert engine.audit()['status'] == 'PASS'
    name = f'{run}_{version}_{split}_{workload}'
    dump(out(root)/'stage2/states'/(name+'.pkl.gz'), states)
    write(out(root)/'stage2/replay'/(name+'.json'), dict(status='PASS', run=run, workload=workload,
        split=split, version=version, states=len(states), saved_feature_states_checked=checked,
        exact_terminal_counts=True, exact_slot_successes=True))
    return len(states)


def predict_scores(state, seed, sla='normal'):
    obs, history, bank = state['obs'], state['history'], state['bank']
    arrivals = t.arrival_prediction(obs, history, seed, 3)
    if sla != 'normal':
        lo, hi = SLA[sla]
        for row in arrivals:
            for task in row:
                mark = task.deadline-task.arrival-2.
                task.deadline = task.arrival+lo+(hi-lo)*mark
    return np.array([[t.forecast_window(obs, history, plan, move, arrivals, 3)['success']
                      for move in bank['actions']] for plan in bank['plans']], dtype=float)


def label_task(root, state_file, index):
    state = read(Path(root)/state_file)[index];row = state['row']
    scores = [row['q']]
    for rep in range(1, 16):
        scores.append(predict_scores(state, t.seed(REVISION, 'forecast', row['parent'], row['workload'], row['time'], rep)))
    scores = np.asarray(scores)
    name = Path(state_file).stem.replace('.pkl', '')+f'_{index}'
    result = {k: state[k] for k in ('row', 'summary', 'task', 'truncated')}
    result['forecasts'] = scores
    result['stability'] = label_stability(scores, row['prior'])
    dump(out(root)/'stage2/labels'/(name+'.pkl.gz'), result)
    return name


def feature_fit_task(root, run, feature, label, mode):
    records = [read(p) for p in sorted((out(root)/'stage2/labels').glob(f'{run}_*.pkl.gz'))]
    train = [];hold = []
    for item in records:
        row = augment(item['row'], item[feature]);row['q'] = item['forecasts'][0] if label == 'single' else item['forecasts'].mean(0)
        (train if row['split'] == 'training' else hold).append(row)
    check_rows(train, hold);assert len(train) == 144 and len(hold) == 48
    actor = initialize(train[0], hidden=32, seed=t.seed(REVISION, 'feature_actor', run))
    fitted, curve = train_curve(train, hold, actor, mode, 64, (4, 16, 32, 64), t.seed(REVISION, 'feature_fit', run))
    # Evaluate every arm against the same 16-forecast reference on held-out roots.
    evaluation = []
    from utils.queued_planner_distillation import imitation_metrics
    for item in records:
        if item['row']['split'] != 'holdout':continue
        row = augment(item['row'], item[feature]);q = item['forecasts'].mean(0)
        z = imitation_metrics(q, policy(row, fitted, mode), row['prior']);z.update(time=row['time'], workload=row['workload'])
        evaluation.append(z)
    stem = f'{run}_{feature}_{label}_{mode}'
    dump(out(root)/'stage2/models'/(stem+'.pkl.gz'), fitted)
    write(out(root)/'stage2/fits'/(stem+'.json'), dict(run=run, feature=feature, label=label, mode=mode,
        curve=curve, heldout_common_reference=evaluation, parameter_count=sum(v.size for p in fitted['parameters'].values() for v in p.values()),
        no_holdout_selection=True))
    return stem


class CaseScenario:
    def __init__(self, workload, case):
        base = t.RecoveryScenario(workload, 'burst', 'normal', case['peak'])
        self.ue_positions = base.ue_positions;self.requests = copy.deepcopy(base.requests)
        lo, hi = SLA[case['sla']]
        for row in self.requests:
            for task in row:
                mark = task.deadline-task.arrival-2.
                task.deadline = task.arrival+lo+(hi-lo)*mark
        fields = [[(x.uid, x.ue, x.arrival, x.bits, x.cycles, x.deadline) for x in row] for row in self.requests]
        self.fingerprint = hashlib.sha256(json.dumps(fields).encode()).hexdigest()
        self.arrival_fingerprint = base.arrival_fingerprint
    def slot(self, now):return self.requests[now] if now < 200 else ()


def condition_task(root, case, run, workload, arm):
    parent = run//2;proposer = t.parent(root, parent)
    actor = read(original.model_path(root, arm, run, 2));mode = original.actor_mode(arm)
    tape = CaseScenario(workload, case)
    engine = t.Engine(replace(t.Protocol(), inter_range=case['range']), t.START, tape.ue_positions)
    adapter = t.LearningAdapter();successes = [];states = []
    while engine.t < 200 or engine.tasks:
        now = int(engine.t)
        if now > 206:raise AssertionError('condition tail not drained')
        engine.add(tape.slot(now));engine.admit();obs = engine.observe()
        bank = t.bank(proposer, obs, adapter, t.seed(REVISION, 'case_bank', parent, workload, now))
        row = original.public_row(bank, obs, adapter.history);p = policy(row, actor, mode)
        if arm == 'mlp_decomposed' and run % 2 == 0 and now in CASE_TIMES:
            states.append(dict(row=row, bank=bank, obs=obs, history=copy.deepcopy(adapter.history),
                               case=case, run=run, workload=workload, time=now))
        rng = np.random.default_rng(t.seed(REVISION, 'case_action', parent, workload, now))
        i, j = np.unravel_index(int(rng.choice(p.size, p=p.ravel())), p.shape)
        engine.assign(bank['plans'][i]);successes.append(engine.advance(bank['actions'][j])['success'])
    c = dict(engine.counts)
    assert not engine.tasks and c['generated'] == c.get('success', 0)+c.get('expired', 0)
    assert sum(successes) == c.get('success', 0) and engine.audit()['status'] == 'PASS'
    stem = f'{case["id"]}_{run}_{workload}_{arm}'
    write(out(root)/'stage3/episodes'/(stem+'.json'), dict(case=case, run=run, parent=parent, workload=workload,
        arm=arm, counts=c, metrics=engine.summary(), successes=successes, fingerprint=tape.fingerprint,
        arrival_fingerprint=tape.arrival_fingerprint, frozen_policy_zero_shot=True))
    if states:
        assert len(states) == 6;dump(out(root)/'stage3/states'/(stem+'.pkl.gz'), states)
    return len(states)


def condition_label_task(root, state_file, index):
    state = read(Path(root)/state_file)[index];case = state['case']
    scores = np.array([predict_scores(state, t.seed(REVISION, 'case_forecast', state['workload'], state['time'], rep), case['sla'])
                       for rep in range(4)])
    q = scores.mean(0);row = state['row'];prior = row['prior']
    opportunity = conditional_opportunity(q, prior);projection_result = teacher_projection(q, prior)
    p = state['obs']['protocol'];obs = state['obs'];margins = []
    for plan in state['bank']['plans']:
        by_uid = {x['uid']: x for x in obs['tasks']}
        for uid, target in plan.items():
            source = by_uid[uid]['source']
            if source is not None and target not in (source, len(obs['positions'])):
                d = np.linalg.norm(np.asarray(obs['positions'][source])-obs['positions'][target])
                margins.append((p.inter_range-d)/(2*p.speed))
    record = dict(case=case, run=state['run'], workload=state['workload'], time=state['time'],
                  opportunity=opportunity, projection=projection_result,
                  label_stability=label_stability(scores, prior), public_tasks=len(obs['tasks']),
                  admitted_tasks=sum(x['source'] is not None for x in obs['tasks']),
                  new_link_margins_seconds=margins)
    name = Path(state_file).stem.replace('.pkl', '')+f'_{index}'
    write(out(root)/'stage3/labels'/(name+'.json'), record)
    return name


def init_worker(cpus):
    os.sched_setaffinity(0, {cpus[(mp.current_process()._identity[-1]-1) % len(cpus)]})


def timed(payload):
    kind, key, args = payload;start = time.time();cpu = time.process_time()
    functions = dict(fit=fit_task, replay=replay_task, label=label_task, feature_fit=feature_fit_task,
                     condition=condition_task, condition_label=condition_label_task)
    result = functions[kind](*args)
    return dict(kind=kind, key=key, start=start, end=time.time(), cpu_seconds=time.process_time()-cpu,
                pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)), result=result)


def dispatch(root, kind, entries):
    marker = val(root)/(kind+'_complete.json')
    if marker.exists():raise FileExistsError('stage already completed; no implicit rerun')
    cpus = sorted(os.sched_getaffinity(0));start = time.perf_counter();records = []
    with ProcessPoolExecutor(max_workers=min(len(cpus), len(entries)), mp_context=mp.get_context('spawn'),
                             initializer=init_worker, initargs=(cpus,)) as pool:
        futures = [pool.submit(timed, (kind, key, (str(root),)+args)) for key, args in entries]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 12 == 0:print(kind, len(records), '/', len(entries), flush=True)
    elapsed = time.perf_counter()-start;used = sum(r['cpu_seconds'] for r in records)
    report = dict(status='COMPLETE', kind=kind, tasks=len(records), wall_seconds=elapsed,
                  worker_cpu_seconds=used, equivalent_cpus=used/elapsed,
                  utilization=used/elapsed/len(cpus), allowed_cpus=cpus, records=records)
    write(marker, report)
    return report


def run(root):
    original.require_server();root = Path(root).resolve()
    if (out(root)/'protocol.json').exists():raise FileExistsError('no restart or overwrite')
    before = source_protection(root)
    text = (REPO_ROOT/'docs/plans/chapter5_dynamic_service_20261028.md').read_text(encoding='utf-8')
    write(out(root)/'protocol.json', dict(revision=REVISION, original_result_hashes=before,
        source={str(p.relative_to(REPO_ROOT)): t.sha(p) for folder in ('utils', 'marl_models', 'scripts/analysis', 'tests')
                for p in (REPO_ROOT/folder).rglob('*.py')},
        protocol_text=text.split('## 2026-11-03 规划到策略')[0], cases=case_grid(),
        formal_test_access=False, order=['fit', 'replay', 'label', 'feature_fit', 'condition', 'condition_label']))
    try:
        jobs = [(f'{a}:{r}:{s}', (a, r, s)) for a in ('mlp_joint', 'mlp_decomposed', 'linear_joint') for r in range(6) for s in (False, True)]
        dispatch(root, 'fit', jobs)
        jobs = []
        for run in specification()['runs']:
            for version in (0, 1):
                for split in ('training', 'holdout'):
                    for w in run[split][version]:jobs.append((f'{run["id"]}:{version}:{split}:{w}', (run['id'], version, split, w)))
        dispatch(root, 'replay', jobs)
        jobs = [(f'{p.name}:{i}', (str(p.relative_to(root)), i)) for p in sorted((out(root)/'stage2/states').glob('*.pkl.gz')) for i in range(12)]
        assert len(jobs) == 1152;dispatch(root, 'label', jobs)
        jobs = [(f'{r}:{f}:{l}:{m}', (r, f, l, m)) for r in range(6) for f in ('summary', 'task')
                for l in ('single', 'mean16') for m in ('joint', 'decomposed')]
        dispatch(root, 'feature_fit', jobs)
        jobs = [(f'{c["id"]}:{r}:{w}:{a}', (c, r, w, a)) for c in case_grid() for r in range(6)
                for w in range(942000, 942004) for a in ('mlp_joint', 'mlp_decomposed')]
        assert len(jobs) == 1296;dispatch(root, 'condition', jobs)
        jobs = [(f'{p.name}:{i}', (str(p.relative_to(root)), i)) for p in sorted((out(root)/'stage3/states').glob('*.pkl.gz')) for i in range(6)]
        assert len(jobs) == 1944;dispatch(root, 'condition_label', jobs)
        assert source_protection(root) == before, 'historical result mutated'
        summarize(root)
        write(val(root)/'run_complete.json', dict(status='COMPLETE', original_results_unchanged=True,
            factual_replay_episodes=96, frozen_transfer_episodes=1296, independent_review=False, formal_test_access=False))
    except Exception:
        write(val(root)/'run_failed.json', dict(error=traceback.format_exc(), automatic_retry=False));raise


def summarize(root):
    # Detailed aggregation is kept outside training and cannot select checkpoints.
    from scripts.analysis.review_queued_learning_validation import review
    return review(root)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--task-root', type=Path, default=REPO_ROOT)
    p.add_argument('--mode', choices=('run', 'review'), required=True);args = p.parse_args()
    run(args.task_root) if args.mode == 'run' else summarize(args.task_root)


if __name__ == '__main__':main()
