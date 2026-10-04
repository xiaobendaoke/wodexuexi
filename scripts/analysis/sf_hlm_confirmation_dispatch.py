"""Dispatch registered confirmation jobs by dependencies, using frozen workers.

Only execution order changes. Each training trajectory stays sequential and the
existing collector retains its frozen-policy barrier and deterministic seeds.
All input/release/output roots are explicit external paths, independent of cwd.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

BASE = ('learned', 'heuristic', 'random', 'greedy', 'original_frozen', 'original_adapted')
PATH_ARGS = ('source_root', 'suite_root', 'extension_root', 'output', 'legacy_results',
    'resources', 'adaptation_root', 'followup_root', 'results_root')
PRIORITY = {'foundation_training': 0, 'confirmation_original_training': 1,
    'component_merge': 2, 'component_lower': 3, 'component_dataset': 4,
    'confirmation_eval': 5, 'confirmation_budget': 6, 'confirmation_timing': 7}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dependency_graph(protocol):
    selected = [tuple(t) for t in protocol['tasks'] if t[0] in PRIORITY
        and (t[0] not in ('confirmation_eval', 'confirmation_timing') or t[1] in BASE)]
    if len(selected) != len(set(selected)):
        raise ValueError('Duplicate registered task')
    shared = frozenset(t for t in selected if t[0] != 'confirmation_timing')
    graph = {}
    for t in selected:
        phase, method, block, _, _ = t
        own = [x for x in selected if x[2] == block]
        def one(name):
            candidates = [x for x in own if x[0] == name]
            if len(candidates) != 1:
                raise ValueError('Missing or ambiguous dependency: ' + str((block, name)))
            return candidates[0]
        if phase == 'foundation_training':
            deps = frozenset()
        elif phase == 'component_merge':
            deps = frozenset(x for x in own if x[0] == 'component_dataset')
            if len(deps) != protocol['dataset_episodes']:
                raise ValueError('Incomplete training graph population')
        elif phase == 'component_lower':
            deps = frozenset([one('component_merge')])
        elif phase == 'confirmation_timing':
            deps = shared
        else:
            deps = {one('foundation_training')}
            if phase == 'confirmation_eval' and method == 'learned':
                deps.add(one('component_lower'))
            if phase == 'confirmation_eval' and method == 'original_adapted':
                deps.add(one('confirmation_original_training'))
            deps = frozenset(deps)
        graph[t] = deps
    return graph


def eligible(graph, completed, active, failed=False):
    if failed:
        return []
    return sorted((t for t, deps in graph.items() if t not in completed
        and t not in active and deps <= completed), key=lambda t: (PRIORITY[t[0]], t))


def submissions(ready, active, capacity, timing_capacity):
    """Limit isolated timing workers before spawning CPU lease waiters."""
    if capacity < 1 or timing_capacity < 1:
        raise ValueError('Dispatch capacities must be positive')
    room = max(0, capacity - len(active))
    timing_room = max(0, timing_capacity - sum(t[0] == 'confirmation_timing' for t in active))
    selected = []
    for task in ready:
        if len(selected) >= room:
            break
        if task[0] == 'confirmation_timing':
            if not timing_room:
                continue
            timing_room -= 1
        selected.append(task)
    return selected


def result_folder(args, task):
    phase, method, block, workload, episode = task
    root = args.results_root
    if phase == 'component_dataset':
        return root / 'components/dataset_episode/complete/greedy' / f'block_{block}_workload_{workload}' / f'episode_{episode:05d}'
    if phase == 'component_merge':
        return root / 'components/dataset' / f'block_{block}'
    if phase == 'component_lower':
        return root / 'components/lower_training/complete/learned' / f'block_{block}'
    folder = root / phase / method / f'block_{block}_workload_{workload}'
    return folder if episode < 0 else folder / f'episode_{episode:05d}'


def verify_result(args, protocol, task):
    folder = result_folder(args, task)
    file = folder / 'result.json'
    if not file.exists():
        return False
    doc = read(file)
    expected = dict(zip(('phase', 'method', 'block', 'workload', 'episode'), task))
    expected.update(source_hash=protocol['source_hash'],
        followup_protocol_sha256=sha(args.results_root / 'protocol.json'))
    if doc.get('status') != 'PASS' or any(doc['contract'].get(k) != v for k, v in expected.items()):
        raise ValueError('Result identity mismatch: ' + str(task))
    if task[0].startswith('component') and doc['contract']['component_protocol_sha256'] != sha(args.results_root / 'components/protocol.json'):
        raise ValueError('Component protocol changed')
    if task[0] in ('foundation_training', 'confirmation_original_training'):
        foundation = task[0] == 'foundation_training'
        count = protocol['foundation_episodes' if foundation else 'original_adaptation_episodes']
        training = folder / 'canonical' if foundation else folder
        if (doc['episodes'] != count or {f.name for f in (training / 'episodes').glob('*.json')}
                != {f'{e:05d}.json' for e in range(count)}):
            raise ValueError('Training episode files are incomplete: ' + str(task))
        if foundation and read(training / 'latest.json')['completed_episodes'] != count:
            raise ValueError('Foundation final checkpoint is incomplete')
        if not foundation and not (doc['upper_fixed'] and doc['optimizer_restored']):
            raise ValueError('Original lower adaptation contract failed')
    if task[0] == 'component_lower' and read(folder / 'latest.json')['iterations'] != protocol['lower_iterations']:
        raise ValueError('Lower training updates are incomplete')
    if task[0] in ('component_dataset', 'component_merge'):
        points = len(read(args.results_root / 'components/protocol.json')['dataset_steps'])
        observed = doc['graphs'] if task[0] == 'component_dataset' else doc['metadata']['graphs']
        expected = points if task[0] == 'component_dataset' else points * protocol['dataset_episodes']
        if observed != expected or (task[0] == 'component_merge' and doc['test_data_used']):
            raise ValueError('Training graph population/partition mismatch')
    for name, h in doc['files'].items():
        if sha(folder / name) != h:
            raise ValueError('Result file changed: ' + str(folder / name))
    return True


def worker_command(args, task):
    command = [sys.executable, str(args.followup_root / 'scripts/analysis/sf_hlm_followup.py'), 'worker']
    for key in PATH_ARGS:
        command += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    for key, value in zip(('phase', 'method', 'block', 'workload', 'episode'), task):
        command += ['--' + key, str(value)]
    return command


def identity(pid, proc=Path('/proc')):
    try:
        fields = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None


def discover_workers(args, graph, proc=Path('/proc')):
    found, orphans = {}, {}
    canonical = {str(result_folder(args, t) / 'canonical'): t for t in graph if t[0] == 'foundation_training'}
    for file in proc.glob('[0-9]*/cmdline'):
        try:
            words = file.read_bytes().decode().strip('\0').split('\0')
        except (OSError, UnicodeError):
            continue
        pid = int(file.parent.name)
        token = identity(pid, proc)
        if token is None or len(words) < 3:
            continue
        if words[1] == str(args.followup_root / 'scripts/analysis/sf_hlm_followup.py') and words[2] == 'worker':
            flags = dict(zip(words[3::2], words[4::2]))
            if flags.get('--results-root') != str(args.results_root):
                continue
            task = tuple(flags['--' + k] if k in ('phase', 'method') else int(flags['--' + k])
                for k in ('phase', 'method', 'block', 'workload', 'episode'))
            if task not in graph or task in found:
                raise ValueError('Unregistered or duplicate in-flight worker: ' + str(task))
            found[task] = dict(pid=pid, identity=token, orphan_canonical=False)
        elif words[1] == str(args.source_root / 'run_canonical_experiment.py') and '--output' in words:
            output = words[words.index('--output') + 1]
            if output in canonical:
                task = canonical[output]
                if task in orphans:
                    raise ValueError('Duplicate canonical trajectory: ' + str(task))
                orphans[task] = dict(pid=pid, identity=token, orphan_canonical=True)
    return {**orphans, **found}


def takeover(args, graph, res):
    pid = args.takeover_pid
    if not pid:
        return discover_workers(args, graph)
    words = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    launch = read(args.validation / 'followup_confirmation_launch.json')
    frozen_controller = words[1:3] == [str(args.followup_root / 'scripts/analysis/sf_hlm_followup.py'), 'pipeline']
    registered_dispatcher = (Path(words[1]).name == 'sf_hlm_confirmation_dispatch.py'
        and words in (launch['command'], launch.get('previous_command', [])))
    if (pid == os.getpid() or not (frozen_controller or registered_dispatcher)
            or '--results-root' not in words or words[words.index('--results-root') + 1] != str(args.results_root)):
        raise ValueError('Refusing to signal an unrelated controller')
    os.kill(pid, signal.SIGSTOP)
    terminating = False
    try:
        active = discover_workers(args, graph)
        event = dict(
            status='DISPATCH_ONLY', old_controller_pid=pid, new_controller_pid=os.getpid(),
            unix=time.time(), adopted=[dict(task=t, **v) for t, v in active.items()],
            workers_kept_running=True, source_and_protocol_unchanged=True)
        record = args.validation / 'followup_confirmation_takeover.json'
        prior = read(record) if record.exists() else None
        history = prior.get('history', [prior]) if prior else []
        res.atomic_json(record, dict(**event, history=history + [event]))
        os.kill(pid, signal.SIGTERM)
        terminating = True
        os.kill(pid, signal.SIGCONT)
        for _ in range(100):
            if identity(pid) is None:
                return active
            time.sleep(.05)
        raise RuntimeError('Previous dispatcher did not exit')
    finally:
        if not terminating and identity(pid) is not None:
            os.kill(pid, signal.SIGCONT)


def execute(args, protocol, task, adopted=None):
    if adopted:
        while identity(adopted['pid']) == adopted['identity']:
            time.sleep(2)
        if verify_result(args, protocol, task):
            return task
        # A detached canonical child may finish without its wrapper receipt.
        # The frozen entrypoint finalizes it or restores its saved state.
    log = args.results_root / 'job_logs' / ('_'.join(map(str, task)) + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        subprocess.run(worker_command(args, task), cwd=args.source_root, stdout=stream,
            stderr=subprocess.STDOUT, check=True)
    if not verify_result(args, protocol, task):
        raise RuntimeError('Worker exited without a verified result: ' + str(task))
    return task


def run(args):
    if sys.platform != 'linux':
        raise RuntimeError('Confirmation execution is remote Linux only')
    spec = importlib.util.spec_from_file_location('confirmation_frozen', args.followup_root / 'scripts/analysis/sf_hlm_followup.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    r = m.runner(args)
    protocol = read(args.results_root / 'protocol.json')
    if protocol != m.make_protocol(args):
        raise ValueError('Frozen protocol changed')
    authorization = read(args.results_root / 'confirmation_authorization.json')
    selection = read(args.results_root / 'confirmation_selection.json')
    if (not authorization['authorized'] or authorization['protocol_sha256'] != sha(args.results_root / 'protocol.json')
            or selection != dict(candidates=['learned'], residual_promoted=False, methods=list(BASE), locked_before_confirmation=True)
            or protocol['smoke']):
        raise ValueError('Confirmation selection/authorization mismatch')
    graph = dependency_graph(protocol)
    active = takeover(args, graph, r.res)
    with r.res.task_lock(args.results_root / 'controller.lock'):
        completed = {t for t in graph if t not in active and verify_result(args, protocol, t)}
        profile = read(args.resources / 'profile.json')
        capacity = profile['slots']
        timing_capacity = read(args.resources / 'isolated/profile.json')['slots']
        # Controller/hash threads stay off the reserved timing cores. Frozen
        # workers acquire their own leases and may expand this inherited mask.
        os.sched_setaffinity(0, set(profile['cpus'][:capacity]))
        failures, futures = [], {}
        stop = threading.Event()
        telemetry = threading.Thread(target=r.res.telemetry, args=(args.resources, stop), daemon=True)
        telemetry.start()
        def status(phase):
            r.res.atomic_json(args.results_root / 'status.json', dict(phase=phase,
                completed=len(completed), expected=len(graph),
                completed_by_phase=dict(Counter(t[0] for t in completed)),
                expected_by_phase=dict(Counter(t[0] for t in graph)),
                active_by_phase=dict(Counter(t[0] for t in futures.values())),
                failures=failures, pid=os.getpid(), updated_unix=time.time(),
                scientific_success_claim=False, execution='dependency_dispatch_v1'))
        try:
            with ThreadPoolExecutor(max_workers=capacity) as pool:
                for task, owner in active.items():
                    futures[pool.submit(execute, args, protocol, task, owner)] = task
                while len(completed) < len(graph):
                    ready = eligible(graph, completed, set(futures.values()), failed=bool(failures))
                    for task in submissions(ready, list(futures.values()), capacity, timing_capacity):
                        futures[pool.submit(execute, args, protocol, task)] = task
                    status('CONFIRMATION_RUNNING' if not failures else 'FAILED_DRAINING')
                    if not futures:
                        if failures:
                            raise RuntimeError('Confirmation tasks failed: ' + str(failures))
                        raise RuntimeError('Dependency deadlock')
                    finished, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                    for future in finished:
                        task = futures.pop(future)
                        try:
                            completed.add(future.result())
                        except Exception as exc:
                            failures.append(dict(task=task, error=str(exc)))
            if failures:
                raise RuntimeError(str(failures))
            status('CONFIRMATION_AGGREGATING')
            quality = m.aggregate(args, r, protocol, 'confirmation_eval', BASE)
            timing = m.aggregate(args, r, protocol, 'confirmation_timing', BASE)
            outcomes = m.confirmation_analysis(args, r, protocol, quality, timing, False)
            r.res.atomic_json(args.results_root / 'status.json', dict(phase='CONFIRMATION_COMPLETE',
                completed=True, completed_tasks=len(completed), expected=len(graph), independent_confirmation=True,
                outcomes=outcomes, pid=os.getpid(), execution='dependency_dispatch_v1',
                scientific_success_claim=any(x['primary_passed'] for x in outcomes.values())))
        finally:
            stop.set()
            telemetry.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in PATH_ARGS:
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--takeover-pid', type=int, default=0)
    args = parser.parse_args()
    for key in (*PATH_ARGS, 'validation'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.smoke = False
    run(args)


if __name__ == '__main__':
    main()
