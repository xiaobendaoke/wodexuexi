"""Scheduling-only recovery for the unchanged b4b20295 scientific release.

Adopts live workers using their existing OS locks, serializes protocol creation,
and resumes partial checkpoints. Never dispatches confirmation tasks.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def identity(arguments):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command')
    parser.add_argument('--split', default='dev')
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--method', default='full')
    parser.add_argument('--kind', default='D')
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--workload', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=0)
    args, _ = parser.parse_known_args(arguments)
    if args.split != 'dev':
        raise RuntimeError('recovery dispatch is development-only')
    return args


def completed(arguments, expected, output, contract):
    if not expected.exists():
        return False
    args = identity(arguments)
    if args.command == 'train':
        last = output/'training/dev'/str(args.block)/args.method/'0399.json'
        if not last.exists():
            return False
        record = json.loads(last.read_text())
        return record['episode'] == 400 and record['protocol'] == contract
    record = json.loads(expected.read_text())
    return record['protocol'] == contract and not record['smoke'] and len(record['rows']) == 1000


def live_worker(arguments, output, source):
    args = identity(arguments)
    name = f'{args.command}_{args.split}_{args.block}_{args.method}_{args.kind}_{args.scale}_{args.workload}_{args.repeat}.lock'
    file = output/'locks'/name
    if not file.exists():
        return None
    with file.open('r') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = int(handle.read().strip())
            cmdline = (Path('/proc')/str(pid)/'cmdline').read_bytes()
            expected_tokens = [str(source/'scripts/analysis/dynamic_service_experiment.py').encode(),
                               b'--split\x00dev', f'--block\x00{args.block}'.encode(),
                               f'--method\x00{args.method}'.encode()]
            if not all(token in cmdline for token in expected_tokens):
                raise RuntimeError('lock owner does not match the frozen task')
            cpu = os.sched_getaffinity(pid)
            if len(cpu) != 1:
                raise RuntimeError('live worker is not pinned to one CPU')
            return dict(pid=pid, cpu=next(iter(cpu)), cmdline=cmdline)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    source, task = args.source.resolve(), args.task.resolve()
    if source.name != 'source_b4b20295' or not source.is_relative_to(task):
        raise RuntimeError('unexpected frozen source')
    validation = task/'output/chapter5_dynamic_service_20261028/validation'
    output = task/'results/chapter5_dynamic_service_20261028'
    sys.path.insert(0, str(source))
    from scripts.analysis import dynamic_service_pipeline as pipeline
    from scripts.analysis.dynamic_service_experiment import atomic_json
    if args.inspect_only:
        found = []
        for block in range(5):
            for method in ('full', 'independent'):
                parameters = ['train', '--split', 'dev', '--block', str(block), '--method', method]
                worker = live_worker(parameters, output, source)
                if worker:
                    found.append(dict(block=block, method=method, pid=worker['pid'], cpu=worker['cpu']))
        print(json.dumps(dict(live_workers=found)))
        return

    def recover_jobs(jobs, unused_output, cpus, parallel, common):
        pending, active = [], []
        # A single serial invocation creates/validates protocol.json before
        # simultaneous children can attempt the first write.
        subprocess.run([sys.executable, str(source/'scripts/analysis/dynamic_service_experiment.py'),
                        'protocol', '--output', str(output)], cwd=source, check=True)
        for job, arguments, expected in jobs:
            worker = live_worker(arguments, output, source)
            if worker:
                if worker['cpu'] not in cpus[:parallel]:
                    raise RuntimeError('adopted worker uses an unexpected CPU')
                active.append(dict(job=job, arguments=arguments, expected=expected, process=None,
                                   handle=None, **worker))
            elif not completed(arguments, expected, output, common['contract']):
                pending.append((job, arguments, expected))
        while active or pending:
            busy = {item['cpu'] for item in active}
            for cpu in cpus[:parallel]:
                if cpu in busy or not pending:
                    continue
                job, arguments, expected = pending.pop(0)
                log = validation/'execution'/f'{job}.log'
                handle = log.open('a')
                command = ['taskset', '-c', str(cpu), sys.executable,
                           str(source/'scripts/analysis/dynamic_service_experiment.py'),
                           *arguments, '--output', str(output)]
                process = subprocess.Popen(command, cwd=source, stdout=handle, stderr=subprocess.STDOUT)
                active.append(dict(job=job, arguments=arguments, expected=expected, process=process,
                                   handle=handle, pid=process.pid, cpu=cpu, cmdline=None))
            for item in active[:]:
                process = item['process']
                if process is None:
                    proc_file = Path('/proc')/str(item['pid'])/'cmdline'
                    alive = proc_file.exists() and proc_file.read_bytes() == item['cmdline']
                    done = not alive
                    code = 0 if not alive and completed(item['arguments'], item['expected'], output, common['contract']) else 1
                else:
                    code = process.poll()
                    done = code is not None
                if done:
                    if item['handle']:
                        item['handle'].close()
                    active.remove(item)
                    if code:
                        atomic_json(validation/'execution/final_status.json', dict(
                            status='STOPPED_WORKER_ERROR', failed_job=item['job'],
                            preserved_live_workers=[x['pid'] for x in active], confirmation_dispatch=False))
                        raise RuntimeError(f'{item["job"]} failed; other live workers are preserved')
            atomic_json(validation/'execution/status.json', dict(stage=common['stage'], pending=len(pending),
                        active=[dict(pid=x['pid'], cpu=x['cpu'], job=x['job'], adopted=x['process'] is None) for x in active],
                        time=time.time(), confirmation_dispatch=False))
            if active:
                time.sleep(2)

    original = validation/'development_dispatch.py'
    old_launch = validation/'development_launch.json'
    if old_launch.exists() and not (validation/'development_launch_initial.json').exists():
        (validation/'development_launch_initial.json').write_bytes(old_launch.read_bytes())
    atomic_json(validation/'development_recovery.json', dict(
        source=str(source), scientific_source_changed=False, confirmation_dispatch=False,
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_dispatcher_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        fix='serial protocol initialization and adoption of still-running task-locked workers'))
    spec = importlib.util.spec_from_file_location('scoped_development_dispatch', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipeline.run_jobs = recover_jobs  # Scheduling only; child scientific source stays immutable.
    module.main()


if __name__ == '__main__':
    main()
