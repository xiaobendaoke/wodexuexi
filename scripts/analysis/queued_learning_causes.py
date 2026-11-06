"""Ordered supervision, structure, own-state and joint-mechanism experiments."""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_learning_validation as previous
from scripts.analysis import queued_taskset_closed_loop as closed
from scripts.analysis import queued_joint_mpo_experiment as engine_tools
from marl_models.queued_planner_distillation import initialize, policy
from utils.queued_learning_validation import augment, evaluate, pack, ROOT_TIMES, phase
from utils.queued_taskset_validation import taskset_features
from utils.queued_learning_causes import REVISION, FEATURES, OBJECTIVES, MODES, node_features, fit, quality_details
from utils.queued_taskset_closed_loop import cohort_statistics, calibration_record, calibration_summary
from utils.queued_planner_distillation import planner_target, imitation_metrics
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

TASK = previous.TASK
BASE_ARMS = tuple(f'{f}_{o}_{m}' for f in FEATURES for o in OBJECTIVES for m in MODES)
ADDED_ARMS = tuple(f'{data}_{m}' for data in ('replay', 'mix') for m in MODES)
TEST_WORKLOADS = tuple(range(946000, 946008))


def out(root):return Path(root)/'results'/TASK/REVISION
def val(root):return Path(root)/'output'/TASK/'validation'/REVISION
def saved(path):return previous.read(path)
def load(path):return previous.load(path)
def write(path, data):previous.write(path, data)
def dump(path, data):previous.dump(path, data)


def extra(obs, bank, feature):
    function = taskset_features if feature == 'pool' else node_features
    return np.array([[function(obs, plan, motion)[0] if feature == 'pool' else function(obs, plan, motion)
                      for motion in bank['actions']] for plan in bank['plans']])


def extract(root, state_file):
    states = saved(Path(root)/state_file)
    records = []
    for index, state in enumerate(states):
        label = Path(state_file).name.replace('.pkl.gz', '')+f'_{index}.pkl.gz'
        record = saved(previous.out(root)/'stage2/labels'/label)
        item = dict(row=record['row'], forecasts=record['forecasts'],
                    node=extra(state['obs'], state['bank'], 'node'))
        records.append(item)
    assert len(records) == 12
    dump(out(root)/'fixed_data'/Path(state_file).name, records)
    return len(records)


def fixed_data(root, run, feature):
    # Pool features are the exact frozen descriptors from the previous trial.
    old_features = {x['label']:x['extra'] for path in (previous.out(root)/'stage2_taskset_followup/features').glob(f'{run}_*.pkl.gz') for x in saved(path)}
    records = []
    for path in sorted((out(root)/'fixed_data').glob(f'{run}_*.pkl.gz')):
        for index, item in enumerate(saved(path)):
            label = path.name.replace('.pkl.gz', '')+f'_{index}.pkl.gz'
            channels = old_features[label] if feature == 'pool' else item['node']
            row = augment(item['row'], channels)
            row['q'] = item['forecasts'].mean(0)
            records.append((label, row, item['forecasts']))
    records = [(row, scores) for _,row,scores in sorted(records, key=lambda x:x[0])]
    train = [z for z in records if z[0]['split'] == 'training']
    hold = [z for z in records if z[0]['split'] == 'holdout']
    assert len(train) == 144 and len(hold) == 48
    return train, hold


def model_path(root, arm, run):return out(root)/'models'/arm/f'{run}.pkl.gz'


def train_base(root, feature, objective, mode, run):
    train, hold = fixed_data(root, run, feature)
    rows, qs = zip(*train);hs, hq = zip(*hold)
    initial = initialize(rows[0], hidden=32, seed=engine_tools.seed(previous.REVISION, 'feature_actor', run))
    model, curve = fit(rows, qs, hs, hq, initial, mode, objective, 64,
                       engine_tools.seed(previous.REVISION, 'feature_fit', run))
    arm = f'{feature}_{objective}_{mode}'
    identity_error = None
    if feature == 'pool' and objective == 'hard':
        old = saved(previous.out(root)/'stage2_taskset_followup/models'/f'{run}_mean16_{mode}.pkl.gz')
        identity_error = max(float(np.max(np.abs(v-old['parameters'][b][k]))) for b,p in model['parameters'].items() for k,v in p.items())
        assert identity_error < 1e-10, 'matched hard-target baseline changed'
    dump(model_path(root, arm, run), model)
    write(out(root)/'fits'/arm/f'{run}.json', dict(arm=arm, run=run, curve=curve,
        old_baseline_identity_error=identity_error, train_states=len(rows), heldout_states=len(hs),
        parameter_count=sum(v.size for p in model['parameters'].values() for v in p.values()),
        data_workloads={'training':sorted({r['workload'] for r in rows}), 'holdout':sorted({r['workload'] for r in hs})}))
    return arm


def choose(root):
    scores = {}
    for feature in FEATURES:
        for objective in OBJECTIVES:
            key = f'{feature}_{objective}'
            values = [load(out(root)/'fits'/(key+'_joint')/f'{run}.json')['curve'][-1]['holdout']['peak']['score_regret'] for run in range(6)]
            scores[key] = dict(mean=float(np.mean(values)), by_run=values)
    selected = min(scores, key=lambda k:(scores[k]['mean'], k))
    write(out(root)/'selection.json', dict(selected=selected, scores=scores,
        criterion='Lowest final fixed 64-epoch public heldout peak regret averaged over six runs; lexicographic tie break.',
        selected_before_new_collection_and_test=True, tested_workloads=list(TEST_WORKLOADS)))
    return selected


def arm_feature(root, arm):
    return load(out(root)/'selection.json')['selected'].split('_')[0] if arm.startswith(('mix_', 'replay_')) else arm.split('_')[0]


def collection_workloads(run, split):
    return list(range(944000+4*run, 944000+4*run+4)) if split == 'training' else list(range(945000+2*run, 945000+2*run+2))


def episode(root, arm, run, workload, replica, collect=False, split='evaluation'):
    previous.original.require_server()
    baseline = arm in ('initial', 'plan_short')
    parent = run//2
    if collect:
        assert workload in collection_workloads(run, split) and replica == 0
    else:
        assert workload in TEST_WORKLOADS and replica in (0, 1)
    proposer = engine_tools.parent(root, parent)
    model = None if baseline else saved(model_path(root, arm, run))
    feature = None if baseline else arm_feature(root, arm)
    mode = 'decomposed' if arm.endswith('decomposed') else 'joint'
    tape = engine_tools.RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = engine_tools.Engine(engine_tools.Protocol(), engine_tools.START, tape.ue_positions)
    adapter = engine_tools.LearningAdapter()
    states = [];successes = [];decisions = [];flight = 0.;peak_load = []
    while engine.t < 200 or engine.tasks:
        now = int(engine.t)
        assert now <= 204
        engine.add(tape.slot(now));engine.admit();obs = engine.observe()
        started = time.perf_counter()
        bank = engine_tools.bank(proposer, obs, adapter, engine_tools.seed(REVISION, 'bank', parent, workload, replica, now))
        row = previous.original.public_row(bank, obs, adapter.history)
        if baseline:
            probability = (planner_target(previous.original.short_scores(bank, obs, adapter.history,
                engine_tools.seed(REVISION, 'forecast', parent, workload, replica, now)), row['prior'])
                if arm == 'plan_short' else np.broadcast_to(row['prior'][:, None]/row['m'].shape[1], row['m'].shape[:2]))
        else:
            row = augment(row, extra(obs, bank, feature))
            probability = policy(row, model, mode)
        rng = np.random.default_rng(engine_tools.seed(REVISION, 'action', parent, workload, replica, now))
        i, j = np.unravel_index(rng.choice(probability.size, p=probability.ravel()), probability.shape)
        elapsed = time.perf_counter()-started
        decisions.append((now, elapsed))
        if collect and now in ROOT_TIMES:
            row.update(time=now, workload=workload, split=split, run=run)
            # Only public state/history; no full engine or realized future.
            states.append(dict(row=row, obs=obs, bank=bank, history=copy.deepcopy(adapter.history)))
        if 50 <= now < 100:
            peak_load.append([sum(x.remaining_cycles for x in engine.tasks.values() if x.target == node)/engine.p.capacity[node] for node in range(6)])
        motion = bank['actions'][j]
        flight += sum(float(np.hypot(*d)) for d in motion)
        engine.assign(bank['plans'][i]);successes.append(engine.advance(motion)['success'])
    counts = dict(engine.counts)
    assert sum(successes) == counts.get('success', 0)
    assert counts['generated'] == counts.get('success', 0)+counts.get('expired', 0)
    assert engine.audit()['status'] == 'PASS'
    record = dict(arm=arm, run=run, parent=parent, workload=workload, replica=replica, counts=counts,
        cohorts=cohort_statistics(engine.finished.values()), metrics=engine.summary(),
        node_quality=quality_details(list(engine.finished.values())), peak_node_load_mean=np.mean(peak_load, 0).tolist(),
        peak_node_load_max=np.max(peak_load, 0).tolist(), decisions=decisions, flight_distance_m=flight,
        fingerprint=tape.fingerprint, successes=successes, public_task_truncation=0, audit=engine.audit())
    folder = 'collection' if collect else 'episodes'
    name = f'{run}_{workload}_{replica}'
    write(out(root)/folder/arm/(name+'.json'), record)
    if collect:
        assert len(states) == 12
        dump(out(root)/'own_states'/arm/(name+'.pkl.gz'), states)
    return dict(arm=arm, states=len(states))


def label(root, state_file, index):
    state = saved(Path(root)/state_file)[index]
    scores = np.array([previous.predict_scores(state, engine_tools.seed(REVISION, 'own_label',
        state['row']['run']//2, state['row']['workload'], state['row']['time'], rep)) for rep in range(16)])
    state['row']['q'] = scores.mean(0)
    arm = Path(state_file).parent.name
    dump(out(root)/'own_labels'/arm/(Path(state_file).name.replace('.pkl.gz', '')+f'_{index}.pkl.gz'),
         dict(row=state['row'], forecasts=scores))
    return 1


def train_continued(root, data_source, mode, run):
    selected = load(out(root)/'selection.json')['selected']
    feature, objective = selected.split('_')
    arm = f'{selected}_{mode}'
    fixed_train, fixed_hold = fixed_data(root, run, feature)
    own = [saved(p) for p in sorted((out(root)/'own_labels'/arm).glob(f'{run}_*.pkl.gz'))]
    new_train = [(z['row'], z['forecasts']) for z in own if z['row']['split'] == 'training']
    new_hold = [(z['row'], z['forecasts']) for z in own if z['row']['split'] == 'holdout']
    assert len(new_train) == 48 and len(new_hold) == 24
    # Four repeated roots per old workload, rotating across all temporal slots.
    replay = [fixed_train[i] for i in range(144) if (i % 12) % 3 == (i // 12) % 3]
    training = fixed_train+(new_train if data_source == 'mix' else replay)
    heldout = fixed_hold+new_hold
    rows, qs = zip(*training);hs, hq = zip(*heldout)
    model, curve = fit(rows, qs, hs, hq, saved(model_path(root, arm, run)), mode, objective, 16,
                       engine_tools.seed(REVISION, 'continue', run))
    target = f'{data_source}_{mode}'
    separate = {name:evaluate(pack([z[0] for z in pairs], mode), model) for name,pairs in [('old', fixed_hold), ('own', new_hold)]}
    dump(model_path(root, target, run), model)
    write(out(root)/'fits'/target/f'{run}.json', dict(arm=target, run=run, selected=selected, curve=curve,
        separate_heldout=separate, train_states=len(training), optimizer_steps=16*24,
        data_source=data_source, new_training_workloads=collection_workloads(run, 'training') if data_source == 'mix' else []))
    return target


def cross_audit(root, state_file):
    records = []
    states = saved(Path(root)/state_file)
    for state in states:
        if state['run'] < 0:continue
        name = f'{state["arm"]}/{state["run"]}_{state["parent"]}_{state["workload"]}_{state["time"]}.pkl.gz'
        matrices = saved(closed.out(root)/'audit_matrices'/name)
        # Explicit public-only input. Private state/future keys are unused.
        features = {f:extra(state['obs'], state['bank'], f) for f in FEATURES}
        result = dict(source_arm=state['arm'], run=state['run'], parent=state['parent'],
                      workload=state['workload'], time=state['time'], methods={})
        for arm in BASE_ARMS+ADDED_ARMS:
            row = augment(state['row'], features[arm_feature(root, arm)])
            p = policy(row, saved(model_path(root, arm, state['run'])), arm.split('_')[-1])
            result['methods'][arm] = {kind:calibration_record(p, reference, q) for kind,reference,q in (
                ('public', matrices['reference'], matrices['forecasts'].mean(0)),
                ('realized', planner_target(matrices['realized'], matrices['prior']), matrices['realized']))}
        records.append(result)
    dump(out(root)/'cross_audit'/Path(state_file).parent.name/Path(state_file).name, records)
    return len(records)


def worker(payload):
    kind, args = payload
    started = time.perf_counter();cpu = time.process_time()
    functions = dict(extract=extract, base=train_base, episode=episode, label=label, continued=train_continued, cross=cross_audit)
    result = functions[kind](*args)
    return dict(kind=kind, cpu_seconds=time.process_time()-cpu, seconds=time.perf_counter()-started, result=result)


def dispatch(root, stage, kind, jobs):
    assert not (val(root)/(stage+'_complete.json')).exists()
    cpus = sorted(os.sched_getaffinity(0));started = time.perf_counter();records = []
    with ProcessPoolExecutor(max_workers=min(len(cpus), len(jobs)), mp_context=mp.get_context('spawn'),
                             initializer=previous.init_worker, initargs=(cpus,)) as pool:
        futures = [pool.submit(worker, (kind, (str(root),)+tuple(args))) for args in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%12 == 0:print(stage, len(records), '/', len(jobs), flush=True)
    elapsed = time.perf_counter()-started;used = sum(r['cpu_seconds'] for r in records)
    write(val(root)/(stage+'_complete.json'), dict(status='COMPLETE', tasks=len(jobs), wall_seconds=elapsed,
        worker_cpu_seconds=used, equivalent_cpus=used/elapsed, utilization=used/elapsed/len(cpus), records=records))


def protected(root):
    paths = list((previous.out(root)).rglob('*'))+list(closed.out(root).rglob('*'))
    return {str(p.relative_to(root)):engine_tools.sha(p) for p in paths if p.is_file()} | previous.source_protection(root)


def run(root):
    previous.original.require_server();root = Path(root).resolve()
    assert not (out(root)/'protocol.json').exists(), 'immutable run already started'
    before = protected(root)
    write(out(root)/'protocol.json', dict(revision=REVISION, protected=before,
        protocol=(REPO_ROOT/'docs/plans/chapter5_dynamic_service_20261028.md').read_text(encoding='utf-8').split('## 2026-11-04 无截断任务表示闭环')[0],
        source={str(p.relative_to(REPO_ROOT)):engine_tools.sha(p) for folder in ('utils', 'marl_models', 'scripts/analysis', 'tests') for p in (REPO_ROOT/folder).rglob('*.py')},
        formal_test_access=False, independent_review=False, new_parent_training=False))
    try:
        dispatch(root, 'extraction', 'extract', [(str(p.relative_to(root)),) for p in sorted((previous.out(root)/'stage2/states').glob('*.pkl.gz'))])
        for feature in FEATURES:
            dispatch(root, feature+'_fits', 'base', [(feature,o,m,r) for o in OBJECTIVES for m in MODES for r in range(6)])
        selected = choose(root);print('SELECTED', selected, flush=True)
        jobs = [(selected+'_'+m, r, w, 0, True, split) for m in MODES for r in range(6)
                for split in ('training', 'holdout') for w in collection_workloads(r, split)]
        assert len(jobs) == 72;dispatch(root, 'own_collection', 'episode', jobs)
        jobs = [(str(p.relative_to(root)), i) for p in sorted((out(root)/'own_states').glob('*/*.pkl.gz')) for i in range(12)]
        assert len(jobs) == 864;dispatch(root, 'own_labels', 'label', jobs)
        dispatch(root, 'continued_fits', 'continued', [(d,m,r) for d in ('replay', 'mix') for m in MODES for r in range(6)])
        jobs = [('plan_short', r, w, rep) for r in (0,2,4) for w in TEST_WORKLOADS for rep in (0,1)]
        jobs += [(arm,r,w,rep) for arm in BASE_ARMS+ADDED_ARMS for r in range(6) for w in TEST_WORKLOADS for rep in (0,1)]
        jobs += [('initial',r,w,rep) for r in (0,2,4) for w in TEST_WORKLOADS for rep in (0,1)]
        assert len(jobs) == 1632;dispatch(root, 'evaluation', 'episode', jobs)
        jobs = [(str(p.relative_to(root)),) for p in sorted((closed.out(root)/'states').glob('*/*.pkl.gz')) if p.parent.name not in closed.BASELINES]
        assert len(jobs) == 120;dispatch(root, 'cross_audit', 'cross', jobs)
        assert protected(root) == before, 'historical artifacts changed'
        from scripts.analysis.review_queued_learning_causes import review
        review(root)
        write(val(root)/'run_complete.json', dict(status='COMPLETE', episodes=1632, collection_episodes=72,
            new_public_label_roots=864, fixed_fits=72, continued_fits=24, historical_artifacts_unchanged=True,
            independent_review=False, formal_test_access=False))
    except Exception:
        write(val(root)/'run_failed.json', dict(error=traceback.format_exc()));raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        previous.original.require_server()
        state = saved(next((previous.out(args.task_root)/'stage2/states').glob('*.pkl.gz')))[0]
        for feature in FEATURES:
            channels = extra(state['obs'], state['bank'], feature)
            assert channels.shape[-1] == 64 and np.isfinite(channels).all()
        print('public feature preflight passed', flush=True)
    else:run(args.task_root)
