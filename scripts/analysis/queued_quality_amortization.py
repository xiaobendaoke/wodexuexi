"""Server-only staged evaluation of cascade-safe planning and analytical queue amortization."""
import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
import time
import traceback

if __name__ == '__main__':
    for env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[env_key] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from paths import REPO_ROOT
from scripts.analysis import queued_waiting_rescue as prototype
from scripts.analysis import queued_waiting_guard as guard_exp
from scripts.analysis.review_queued_quality_amortization import ARMS, stage_review
from utils.queued_arrival_adaptation import RateHistory, future_arrivals
from utils.queued_planner_distillation import planner_target
from utils.queued_taskset_closed_loop import cohort_statistics
from utils.queued_learning_causes import quality_details
from utils.queued_score_control import congestion_pressure, row_scores
from utils.queued_waiting_rescue import rescue_candidates
from utils.queued_quality_amortization import (
    REVISION,
    evaluate_amortized_bank,
    evaluate_quality_bank,
    expand_safe_plan_bank,
    safe_assignments,
    should_expand_rescue,
)

TASK = prototype.TASK
t = prototype.t


def out(root):
    return Path(root) / 'results' / TASK / REVISION


def val(root):
    return Path(root) / 'output' / TASK / 'validation' / REVISION


def episode(root, stage, arm, parent, workload, replica, output_folder=None):
    allowed = prototype.PILOT_WORKLOADS if stage == 'pilot' else prototype.CONFIRMATION_WORKLOADS
    assert stage in ('pilot', 'confirmation') and workload in allowed and arm in ARMS
    if arm in ('old_hard', 'old_plan'):
        return prototype.episode(root, stage, arm, parent, workload, replica,
                                 out(root) / stage if output_folder is None else output_folder)

    tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions)
    adapter = t.LearningAdapter()
    temporal = RateHistory()
    proposer = t.parent(root, parent)
    score_model = prototype.previous.old.scorer(str(root), 'reg', 'joint', parent * 2) if arm == 'quality_amortized' else None
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
        bank = t.bank(proposer, obs, adapter, t.seed(prototype.REVISION, 'bank', parent, workload, replica, now))
        original_count = len(bank['actions'])
        original_plan_count = len(bank['plans'])
        pressure = congestion_pressure(obs, bank['plans'], bank['prior'])
        rate3 = float(np.mean(temporal.counts[-3:]))
        expand = should_expand_rescue(pressure, rate3)
        if expand:
            actions, metadata = rescue_candidates(obs, bank['actions'])
        else:
            actions = bank['actions']
            metadata = dict(added=0, waiting=sum(x['source'] is None for x in obs['tasks']))
        plans, prior = expand_safe_plan_bank(obs, bank['plans'], bank['prior'])
        s_plan = safe_assignments(obs)
        safe_plan_idx = plans.index(s_plan)
        metadata.update(
            pressure=pressure,
            rate3=rate3,
            guard_blocked=not expand,
            safe_plan_added=bool(len(plans) > original_plan_count),
        )
        calls = 0
        if arm == 'quality_plan':
            future = future_arrivals(obs, adapter.history, temporal, 'window3',
                                     t.seed(prototype.REVISION, 'forecast', parent, workload, replica, now))
            scores = evaluate_quality_bank(obs, adapter.history, plans, actions, future, pressure)
            calls = int(scores.size)
        elif arm == 'quality_amortized':
            raw = prototype.previous.validation.original.public_row(bank, obs, adapter.history)
            row = prototype.previous.augment(raw, prototype.previous.old.previous.extra(obs, bank, 'pool'))
            base_neural = row_scores(row, score_model, 'joint')
            scores = evaluate_amortized_bank(obs, adapter.history, plans, actions, pressure, rate3, base_neural)
            calls = 0
        else:
            raise ValueError(f'unexpected arm: {arm}')
        probability = planner_target(scores, prior)
        rng = np.random.default_rng(t.seed(prototype.REVISION, 'action', parent, workload, replica, now))
        allocation, motion = np.unravel_index(rng.choice(probability.size, p=probability.ravel()), probability.shape)
        seconds = time.perf_counter() - started
        selected = actions[motion]
        metadata['selected_safe_plan'] = bool(allocation == safe_plan_idx)
        trajectory.append(dict(
            time=now, positions=list(engine.positions), motion=selected,
            admitted=len(admitted), original_candidates=original_count, candidates=len(actions),
            selected_new=bool(motion >= original_count), metadata=metadata,
            decision_seconds=seconds, forecast_calls=calls,
        ))
        engine.assign(plans[allocation])
        successes.append(engine.advance(selected)['success'])
    audit = engine.audit()
    counts = dict(engine.counts)
    assert not engine.tasks and audit['status'] == 'PASS'
    assert sum(successes) == counts.get('success', 0)
    assert counts['generated'] == counts.get('success', 0) + counts.get('expired', 0)
    record = dict(
        stage=stage, arm=arm, parent=parent, workload=workload, replica=replica,
        counts=counts, cohorts=cohort_statistics(engine.finished.values()),
        quality=quality_details(list(engine.finished.values())),
        successes=successes, trajectory=trajectory, fingerprint=tape.fingerprint, audit=audit,
    )
    destination = out(root) / stage if output_folder is None else Path(output_folder)
    prototype.write(destination / arm / f'{parent}_{workload}_{replica}.json', record)
    return dict(arm=arm, success=counts.get('success', 0))


def worker(payload):
    root, stage, arm, parent, workload, replica = payload
    started = time.perf_counter()
    cpu = time.process_time()
    record = episode(root, stage, arm, parent, workload, replica, out(root) / stage)
    return dict(seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu, result=record)


def dispatch(root, stage, jobs):
    cpus = sorted(os.sched_getaffinity(0))
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(
        max_workers=min(len(cpus), len(jobs)),
        mp_context=mp.get_context('spawn'),
        initializer=prototype.previous.validation.init_worker,
        initargs=(cpus,),
    ) as pool:
        futures = [pool.submit(worker, (str(root),) + tuple(job)) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 12 == 0 or len(records) == len(jobs):
                print(stage, len(records), '/', len(jobs), flush=True)
    seconds = time.perf_counter() - started
    cpu = sum(row['cpu_seconds'] for row in records)
    prototype.write(val(root) / (stage + '_complete.json'), dict(
        tasks=len(jobs), wall_seconds=seconds,
        worker_cpu_seconds=cpu, equivalent_cpus=cpu / seconds,
        utilization=cpu / seconds / len(cpus), records=records,
    ))


def protected(root):
    return guard_exp.protected(root) | {
        str(path.relative_to(root)): prototype.t.sha(path)
        for path in guard_exp.out(root).rglob('*') if path.is_file()
    }


def run(root):
    prototype.previous.validation.original.require_server()
    root = Path(root).resolve()
    before = protected(root)
    assert not json.loads((guard_exp.out(root) / 'pilot_analysis.json').read_text())['gate_passed']
    prototype.write(out(root) / 'protocol.json', dict(
        revision=REVISION,
        protected=before,
        protocol=(REPO_ROOT / 'docs/plans/chapter5_dynamic_service_20261028.md').read_text(encoding='utf-8').split(
            '## 2026-11-04 待接纳任务运动候选第一版')[0],
        sources={
            str(path.relative_to(REPO_ROOT)): prototype.t.sha(path)
            for directory in ('environment', 'utils', 'marl_models', 'scripts/analysis', 'tests')
            for path in (REPO_ROOT / directory).rglob('*.py')
        },
        arms=list(ARMS),
        rescue_pressure_limit=0.85,
        rescue_rate_limit=25.0,
        congested_pressure_limit=1.0,
        selection='Cascade-safe EDF assignment + dual pressure/arrival-rate rescue guard + overload-aware tie-breaking and zero-future-sampling analytical queue amortization.',
        new_training_steps=0,
        formal_test_access=False,
        independent_review=False,
    ))
    try:
        baseline_files = {}
        for arm in ('old_hard', 'old_plan'):
            for source in (prototype.out(root) / 'pilot' / arm).glob('*.json'):
                destination = out(root) / 'pilot' / arm / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                assert not destination.exists()
                shutil.copy2(source, destination)
                baseline_files[str(source.relative_to(root))] = prototype.t.sha(source)
        assert len(baseline_files) == 24
        prototype.write(out(root) / 'reused_pilot_baselines.json', baseline_files)
        pilot_jobs = [
            ('pilot', arm, parent, workload, 0)
            for arm in ('quality_plan', 'quality_amortized')
            for parent in range(3)
            for workload in prototype.PILOT_WORKLOADS
        ]
        dispatch(root, 'pilot', pilot_jobs)
        pilot = stage_review(root, 'pilot', out(root), 'quality_plan')
        prototype.write(out(root) / 'pilot_analysis.json', pilot)
        if pilot['gate_passed']:
            conf_jobs = [
                ('confirmation', arm, parent, workload, replica)
                for arm in ARMS
                for parent in range(3)
                for workload in prototype.CONFIRMATION_WORKLOADS
                for replica in (0, 1)
            ]
            dispatch(root, 'confirmation', conf_jobs)
            prototype.write(
                out(root) / 'confirmation_analysis.json',
                stage_review(root, 'confirmation', out(root), 'quality_plan'),
            )
        assert protected(root) == before
        completion = dict(
            status='COMPLETE',
            new_pilot_episodes=len(pilot_jobs),
            reused_pilot_baseline_episodes=24,
            confirmation_episodes=len(list((out(root) / 'confirmation').glob('*/*.json'))),
            pilot_gate_passed=pilot['gate_passed'],
            pilot_gates=pilot['gates'],
            new_training_steps=0,
            historical_artifacts_unchanged=True,
            formal_test_access=False,
            independent_review=False,
        )
        prototype.write(val(root) / 'run_complete.json', completion)
        print(json.dumps(completion), flush=True)
    except Exception:
        prototype.write(val(root) / 'run_failed.json', dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    run(args.task_root)
