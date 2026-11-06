"""Fixed public EDF pressure gate for the waiting-request prototype."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_waiting_rescue as prototype
from scripts.analysis.review_queued_waiting_rescue import stage_review

REVISION = 'queued_waiting_guard_20261104'
ARMS = ('old_hard', 'old_plan', 'guarded_plan')


def out(root):
    return Path(root)/'results'/prototype.TASK/REVISION


def val(root):
    return Path(root)/'output'/prototype.TASK/'validation'/REVISION


def worker(payload):
    root, stage, arm, parent, workload, replica = payload
    started = time.perf_counter()
    cpu = time.process_time()
    record = prototype.episode(root, stage, arm, parent, workload, replica, out(root)/stage)
    return dict(seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu, result=record)


def dispatch(root, stage, jobs):
    cpus = sorted(os.sched_getaffinity(0))
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=min(len(cpus), len(jobs)), mp_context=mp.get_context('spawn'),
            initializer=prototype.previous.validation.init_worker, initargs=(cpus,)) as pool:
        futures = [pool.submit(worker, (str(root),)+tuple(job)) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 6 == 0:
                print(stage, len(records), '/', len(jobs), flush=True)
    seconds = time.perf_counter()-started
    cpu = sum(row['cpu_seconds'] for row in records)
    prototype.write(val(root)/(stage+'_complete.json'), dict(tasks=len(jobs), wall_seconds=seconds,
        worker_cpu_seconds=cpu, equivalent_cpus=cpu/seconds, utilization=cpu/seconds/len(cpus), records=records))


def protected(root):
    return prototype.protected(root) | {
        str(path.relative_to(root)):prototype.t.sha(path) for path in prototype.out(root).rglob('*') if path.is_file()
    }


def run(root):
    prototype.previous.validation.original.require_server()
    root = Path(root).resolve()
    before = protected(root)
    assert not json.loads((prototype.out(root)/'pilot_analysis.json').read_text())['gate_passed']
    assert not list((prototype.out(root)/'confirmation').glob('*/*.json'))
    prototype.write(out(root)/'protocol.json', dict(revision=REVISION, protected=before,
        protocol=(REPO_ROOT/'docs/plans/chapter5_dynamic_service_20261028.md').read_text(encoding='utf-8').split(
            '## 2026-11-04 待接纳任务运动候选第一版')[0],
        sources={str(path.relative_to(REPO_ROOT)):prototype.t.sha(path) for directory in ('environment','utils','marl_models','scripts/analysis','tests')
                 for path in (REPO_ROOT/directory).rglob('*.py')},
        threshold=1., selection='Public current EDF pressure weighted by frozen allocation prior.',
        new_training_steps=0, formal_test_access=False, independent_review=False))
    try:
        baseline_files = {}
        for arm in ('old_hard', 'old_plan'):
            for source in (prototype.out(root)/'pilot'/arm).glob('*.json'):
                destination = out(root)/'pilot'/arm/source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                assert not destination.exists()
                shutil.copy2(source, destination)
                baseline_files[str(source.relative_to(root))] = prototype.t.sha(source)
        assert len(baseline_files) == 24
        prototype.write(out(root)/'reused_pilot_baselines.json', baseline_files)
        jobs = [('pilot', 'guarded_plan', parent, workload, 0)
                for parent in range(3) for workload in prototype.PILOT_WORKLOADS]
        dispatch(root, 'pilot', jobs)
        pilot = stage_review(root, 'pilot', out(root), 'guarded_plan')
        prototype.write(out(root)/'pilot_analysis.json', pilot)
        if pilot['gate_passed']:
            jobs = [('confirmation', arm, parent, workload, replica) for arm in ARMS for parent in range(3)
                    for workload in prototype.CONFIRMATION_WORKLOADS for replica in (0, 1)]
            dispatch(root, 'confirmation', jobs)
            prototype.write(out(root)/'confirmation_analysis.json', stage_review(root, 'confirmation', out(root), 'guarded_plan'))
        assert protected(root) == before
        completion = dict(status='COMPLETE', new_pilot_episodes=12, reused_pilot_baseline_episodes=24,
            confirmation_episodes=len(list((out(root)/'confirmation').glob('*/*.json'))),
            new_training_steps=0, historical_artifacts_unchanged=True, formal_test_access=False, independent_review=False)
        prototype.write(val(root)/'run_complete.json', completion)
        print(json.dumps(completion), flush=True)
    except Exception:
        prototype.write(val(root)/'run_failed.json', dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    run(args.task_root)
