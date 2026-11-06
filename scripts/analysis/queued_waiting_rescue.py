"""Server-only staged evaluation of public waiting-request motion proposals."""
import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from paths import REPO_ROOT
from scripts.analysis import queued_arrival_adaptation as previous
from utils.queued_arrival_adaptation import RateHistory, future_arrivals
from utils.queued_joint_mpo import forecast_window
from utils.queued_planner_distillation import planner_target
from utils.queued_taskset_closed_loop import cohort_statistics
from utils.queued_learning_causes import quality_details
from utils.queued_score_control import congestion_pressure
from utils.queued_waiting_rescue import REVISION, rescue_candidates

TASK = previous.TASK
t = previous.t
ARMS = ('old_hard', 'old_plan', 'rescue_plan')
PILOT_WORKLOADS = tuple(range(951000, 951004))
CONFIRMATION_WORKLOADS = tuple(range(952000, 952008))


def out(root):
    return Path(root)/'results'/TASK/REVISION


def val(root):
    return Path(root)/'output'/TASK/'validation'/REVISION


def write(path, record):
    if Path(path).exists():
        raise FileExistsError(path)
    t.write(path, record)


def evaluate_bank(obs, history, plans, motions, arrivals):
    return np.array([
        [forecast_window(obs, history, plan, motion, arrivals, 3)['success'] for motion in motions]
        for plan in plans
    ], dtype=float)


def offline(root, relative):
    state = previous.saved(Path(root)/relative)
    key = previous.recovery.root_key(state)
    labels = previous.saved(previous.out(root)/'diagnostic_labels_compact'/(key+'.pkl.gz'))
    obs = state['obs']
    bank = state['bank']
    history = state['adapter'].history
    actions, metadata = rescue_candidates(obs, bank['actions'])
    old_count = len(bank['actions'])
    added = actions[old_count:]
    old_q = labels['fast'][:4].mean(0)
    old_actual = labels['realized']
    old_distribution = planner_target(old_q, bank['prior'])
    new_q = old_q.copy()
    new_actual = old_actual.copy()
    if added:
        # Only the public stored three-slot mean is needed to restore this forecast.
        temporal = RateHistory()
        temporal.update_count(int(obs['time']), float(labels['rates'][3]*60))
        predictions = []
        for replica in range(4):
            seed = t.seed(previous.recovery.REVISION, 'public_prediction', state['parent'],
                          state['workload'], state['time'], replica)
            arrivals = future_arrivals(obs, history, temporal, 'window3', seed)
            reproduced = evaluate_bank(obs, history, bank['plans'][:1], bank['actions'][:1], arrivals)
            assert reproduced[0, 0] == labels['fast'][replica, 0, 0]
            predictions.append(evaluate_bank(obs, history, bank['plans'], added, arrivals))
        new_q = np.concatenate([old_q, np.mean(predictions, axis=0)], axis=1)
        actual_extra = evaluate_bank(state['private']['full'], history, bank['plans'], added,
                                     state['private']['future'])
        new_actual = np.concatenate([old_actual, actual_extra], axis=1)
    new_distribution = planner_target(new_q, bank['prior'])
    selected_gain = float(np.sum(new_distribution*new_actual)-np.sum(old_distribution*old_actual))
    capacity_gain = float(new_actual.max()-old_actual.max())
    assert capacity_gain >= -1e-12
    record = dict(behavior=state['behavior'], run=state['run'], parent=state['parent'],
                  workload=state['workload'], time=state['time'], metadata=metadata,
                  prior=bank['prior'].tolist(),
                  old_actions=old_count, new_actions=len(actions), actions=actions,
                  public_old=old_q.tolist(), public_new=new_q.tolist(),
                  actual_old=old_actual.tolist(), actual_new=new_actual.tolist(),
                  selected_gain=selected_gain, capacity_gain=capacity_gain)
    write(out(root)/'offline'/(key+'.json'), record)
    return dict(added=len(added), selected_gain=selected_gain)


def episode(root, stage, arm, parent, workload, replica, output_folder=None):
    allowed = PILOT_WORKLOADS if stage == 'pilot' else CONFIRMATION_WORKLOADS
    assert stage in ('pilot', 'confirmation') and workload in allowed and arm in ARMS+('guarded_plan',)
    tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions)
    adapter = t.LearningAdapter()
    temporal = RateHistory()
    proposer = t.parent(root, parent)
    successes = []
    trajectory = []
    while engine.t < 200 or engine.tasks:
        now = int(engine.t)
        assert now <= 204
        engine.add(tape.slot(now))
        admitted = engine.admit()
        obs = engine.observe()
        temporal.update(obs)
        started = time.perf_counter()
        bank = t.bank(proposer, obs, adapter, t.seed(REVISION, 'bank', parent, workload, replica, now))
        original_count = len(bank['actions'])
        metadata = dict(added=0, waiting=sum(x['source'] is None for x in obs['tasks']))
        pressure = congestion_pressure(obs, bank['plans'], bank['prior']) if arm == 'guarded_plan' else None
        expand = arm == 'rescue_plan' or (arm == 'guarded_plan' and pressure <= 1.)
        if expand:
            actions, metadata = rescue_candidates(obs, bank['actions'])
        else:
            actions = bank['actions']
        if arm == 'guarded_plan':
            metadata.update(pressure=pressure, guard_blocked=pressure > 1.)
        calls = 0
        if arm == 'old_hard':
            raw = previous.validation.original.public_row(bank, obs, adapter.history)
            row = previous.augment(raw, previous.old.previous.extra(obs, bank, 'pool'))
            probability, _ = previous.old.distribution(root, 'hard_joint', parent*2, row, obs, bank)
        else:
            future = future_arrivals(obs, adapter.history, temporal, 'window3',
                                     t.seed(REVISION, 'forecast', parent, workload, replica, now))
            scores = evaluate_bank(obs, adapter.history, bank['plans'], actions, future)
            probability = planner_target(scores, bank['prior'])
            calls = scores.size
        rng = np.random.default_rng(t.seed(REVISION, 'action', parent, workload, replica, now))
        allocation, motion = np.unravel_index(rng.choice(probability.size, p=probability.ravel()), probability.shape)
        seconds = time.perf_counter()-started
        selected = actions[motion]
        trajectory.append(dict(time=now, positions=list(engine.positions), motion=selected,
            admitted=len(admitted), original_candidates=original_count, candidates=len(actions),
            selected_new=bool(motion >= original_count), metadata=metadata,
            decision_seconds=seconds, forecast_calls=calls))
        engine.assign(bank['plans'][allocation])
        successes.append(engine.advance(selected)['success'])
    audit = engine.audit()
    counts = dict(engine.counts)
    assert not engine.tasks and audit['status'] == 'PASS'
    assert sum(successes) == counts.get('success', 0)
    assert counts['generated'] == counts.get('success', 0)+counts.get('expired', 0)
    record = dict(stage=stage, arm=arm, parent=parent, workload=workload, replica=replica,
                  counts=counts, cohorts=cohort_statistics(engine.finished.values()),
                  quality=quality_details(list(engine.finished.values())),
                  successes=successes, trajectory=trajectory, fingerprint=tape.fingerprint, audit=audit)
    destination = out(root)/stage if output_folder is None else Path(output_folder)
    write(destination/arm/f'{parent}_{workload}_{replica}.json', record)
    return dict(arm=arm, success=counts.get('success', 0))


def worker(payload):
    kind, args = payload
    started = time.perf_counter()
    cpu = time.process_time()
    result = offline(*args) if kind == 'offline' else episode(*args)
    return dict(cpu_seconds=time.process_time()-cpu, seconds=time.perf_counter()-started, result=result)


def dispatch(root, stage, kind, jobs):
    cpus = sorted(os.sched_getaffinity(0))
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=min(len(cpus), len(jobs)), mp_context=mp.get_context('spawn'),
            initializer=previous.validation.init_worker, initargs=(cpus,)) as pool:
        futures = [pool.submit(worker, (kind, (str(root),)+tuple(args))) for args in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 6 == 0:
                print(stage, len(records), '/', len(jobs), flush=True)
    wall = time.perf_counter()-started
    cpu = sum(record['cpu_seconds'] for record in records)
    write(val(root)/(stage+'_complete.json'), dict(tasks=len(jobs), wall_seconds=wall,
          worker_cpu_seconds=cpu, equivalent_cpus=cpu/wall, utilization=cpu/wall/len(cpus), records=records))


def protected(root):
    return previous.protection(root) | {
        str(path.relative_to(root)):t.sha(path) for path in previous.out(root).rglob('*') if path.is_file()
    }


def preflight(root):
    previous.validation.original.require_server()
    path = next((previous.recovery.out(root)/'states').glob('*/*.pkl.gz'))
    state = previous.saved(path)
    actions, _ = rescue_candidates(state['obs'], state['bank']['actions'])
    assert actions[:len(state['bank']['actions'])] == state['bank']['actions']
    scores = evaluate_bank(state['obs'], state['adapter'].history, state['bank']['plans'][:1],
                           actions[:1], [[], []])
    assert np.isfinite(scores).all()
    print('public rescue and H3 preflight passed', flush=True)


def run(root):
    previous.validation.original.require_server()
    root = Path(root).resolve()
    before = protected(root)
    protocol = (REPO_ROOT/'docs/plans/chapter5_dynamic_service_20261028.md').read_text(encoding='utf-8')
    write(out(root)/'protocol.json', dict(revision=REVISION, protected=before,
        protocol=protocol.split('## 2026-11-04 快速到达预测、修正评分与运动尾值')[0],
        sources={str(path.relative_to(REPO_ROOT)):t.sha(path) for directory in ('environment','utils','marl_models','scripts/analysis','tests')
                 for path in (REPO_ROOT/directory).rglob('*.py')},
        formal_test_access=False, new_training_steps=0, independent_review=False))
    try:
        paths = sorted((previous.recovery.out(root)/'states').glob('*/*.pkl.gz'))
        assert len(paths) == 96
        dispatch(root, 'offline', 'offline', [(str(path.relative_to(root)),) for path in paths])
        from scripts.analysis.review_queued_waiting_rescue import offline_review, stage_review
        audit = offline_review(root)
        write(out(root)/'offline_analysis.json', audit)
        if audit['gate_passed']:
            jobs = [('pilot', arm, parent, workload, 0) for arm in ARMS for parent in range(3) for workload in PILOT_WORKLOADS]
            dispatch(root, 'pilot', 'episode', jobs)
            pilot = stage_review(root, 'pilot')
            write(out(root)/'pilot_analysis.json', pilot)
            if pilot['gate_passed']:
                jobs = [('confirmation', arm, parent, workload, replica) for arm in ARMS for parent in range(3)
                        for workload in CONFIRMATION_WORKLOADS for replica in (0, 1)]
                dispatch(root, 'confirmation', 'episode', jobs)
                write(out(root)/'confirmation_analysis.json', stage_review(root, 'confirmation'))
        assert protected(root) == before
        completion = dict(status='COMPLETE', offline_roots=96,
            pilot_episodes=len(list((out(root)/'pilot').glob('*/*.json'))),
            confirmation_episodes=len(list((out(root)/'confirmation').glob('*/*.json'))),
            new_training_steps=0, formal_test_access=False, historical_artifacts_unchanged=True, independent_review=False)
        write(val(root)/'run_complete.json', completion)
        print(json.dumps(completion), flush=True)
    except Exception:
        write(val(root)/'run_failed.json', dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    preflight(args.task_root) if args.preflight else run(args.task_root)
