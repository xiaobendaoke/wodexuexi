"""Run the frozen formal matrix with bounded workers, durable jobs and resume."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

from utils.canonical_experiment import ROOT, digest, source_manifest, write_json
from utils.formal_protocol import NONLEARNED, matrix, verify_completed_job, verify_release


def groups(plan):
    for variant in plan['variants']:
        for method in variant['methods']:
            for seed in ([42] if method in NONLEARNED else plan['training_seeds']):
                yield variant, method, seed


def run_suite(plan, output, source_lock=None, validation_report=None, workers=1, threads=4):
    if plan.get('protocol', '').startswith('batch_offload_'):
        from scripts.analysis.batch_offload_protocol import validate_plan
        validate_plan(plan)
    source = source_manifest()
    if plan['purpose'] == 'formal':
        lock = verify_release(source_lock, validation_report, source)
        source.update({key: lock.get(key) for key in ('git_sha', 'branch', 'git_clean')})
    if workers < 1 or threads < 1 or workers * threads > (os.cpu_count() or 1):
        raise ValueError('Invalid CPU allocation: workers × threads must fit available CPUs')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 10 * 1024**3:
        raise RuntimeError('Less than 10 GiB free disk; free space or choose another output volume')
    if Path('/proc/meminfo').exists():
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        if workers * 8 * 1024**2 > memory['MemAvailable']:
            raise RuntimeError('Insufficient available RAM: reserve 8 GiB per worker')
    # One controller owns a suite directory. OS releases this lock on process exit.
    import fcntl
    with (output / '.controller.lock').open('w') as controller:
        try:
            fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This suite already has an active controller')
        identity = {'plan': plan, 'source_hash': source['source_hash'], 'threads': threads,
                    'git_sha': source.get('git_sha'), 'git_clean': source.get('git_clean')}
        manifest = output / 'suite.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError('Suite configuration/source changed; use a new output directory')
        else:
            write_json(manifest, identity)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        def job(relative, variant, method, seed, workload, checkpoint=None):
            directory = output / relative
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory / 'completed.json'
            if marker.exists():
                return verify_completed_job(directory, source['source_hash'])
            attempts = sorted(directory.glob('attempt_*'))
            resume = False
            if checkpoint is None and method not in NONLEARNED and attempts and (attempts[-1] / 'latest.json').exists():
                attempt, resume = attempts[-1], True
                count = json.loads((attempt / 'latest.json').read_text())['completed_episodes']
                if count >= plan['train_episodes']:
                    # Crash after final checkpoint, before parent completion marker.
                    resume = False
                    complete(directory, attempt, marker)
                    return attempt
            else:
                attempt = directory / f'attempt_{len(attempts):04d}'
            cmd = [sys.executable, str(ROOT / 'run_canonical_experiment.py'), '--method', method,
                   '--training-seed', str(seed), '--workload-seed', str(workload),
                   '--episodes', str(plan['eval_episodes'] if checkpoint or method in NONLEARNED else plan['train_episodes']),
                   '--steps', str(plan['steps']), '--rollout', str(plan['rollout']),
                   '--ppo-epochs', str(plan['ppo_epochs']), '--batch-size', str(plan['batch_size']),
                   '--threads', str(threads), '--checkpoint-every', str(plan['checkpoint_every']),
                   '--ue-count', str(variant['ue_count']), '--compute-scale', str(variant['compute_scale']),
                   '--purpose', plan['purpose'], '--output', str(attempt)]
            if source_lock:
                cmd += ['--source-lock', str(Path(source_lock).resolve()), '--validation-report', str(Path(validation_report).resolve())]
            if checkpoint:
                cmd += ['--evaluate-checkpoint', str(checkpoint)]
            if resume:
                cmd += ['--resume']
            write_json(directory / 'state.json', {'status': 'RUNNING', 'command': cmd, 'started': time.time()})
            with (directory / 'worker.log').open('a') as log:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(directory / 'state.json', {'status': 'FAILED', 'returncode': result.returncode, 'command': cmd})
                raise RuntimeError(f'Experiment failed; see {directory / "worker.log"}')
            complete(directory, attempt, marker)
            return attempt

        def complete(directory, attempt, marker):
            hashes = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in attempt.rglob('*') if p.is_file()}
            if not hashes:
                raise RuntimeError('Empty completed experiment')
            write_json(marker, {'status': 'PASS', 'attempt': attempt.name, 'hashes': hashes})
            write_json(directory / 'state.json', {'status': 'PASS'})

        def group(item):
            variant, method, seed = item
            base = Path(variant['name']) / method
            checkpoint = None
            if method not in NONLEARNED:
                training_workload = plan.get('training_workload_seeds', {}).get(
                    str(seed), plan.get('training_workload_seed'))
                if training_workload is None:
                    raise ValueError('Plan needs training_workload_seed or a per-seed training_workload_seeds map')
                trained = job(base / f'seed_{seed}' / 'train', variant, method, seed, training_workload)
                checkpoint = trained / json.loads((trained / 'latest.json').read_text())['checkpoint']
                base /= f'seed_{seed}'
            for workload in plan['workload_seeds']:
                job(base / f'workload_{workload}', variant, method, seed, workload, checkpoint)
            return {'variant': variant['name'], 'method': method, 'seed': None if method in NONLEARNED else seed}
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                completed = list(pool.map(group, groups(plan)))
            from analyze_canonical_suite import analyze
            analyze(output)
            write_json(output / 'status.json', {'status': 'PASS', 'groups': completed, 'purpose': plan['purpose']})
        except Exception as error:
            write_json(output / 'status.json', {'status': 'FAILED', 'error': str(error), 'resume': 'Rerun the identical command'})
            raise
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['all', 'main', 'sensitivity'], default='all')
    parser.add_argument('--plan-file', help='Frozen JSON plan for a separately named experiment')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-lock', default=str(ROOT / 'source-lock.json'))
    parser.add_argument('--validation-report', default=str(ROOT / 'validation.json'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true', help='Small verification matrix, never formal data')
    parser.add_argument('--plan-only', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    if args.plan_file and args.smoke:
        raise ValueError('Use a separate verification plan file; --smoke cannot modify a frozen plan')
    plan = json.loads(Path(args.plan_file).read_text()) if args.plan_file else matrix(args.suite, args.smoke)
    required = {'protocol', 'purpose', 'training_seeds', 'workload_seeds', 'train_episodes',
                'eval_episodes', 'steps', 'rollout', 'ppo_epochs', 'batch_size',
                'checkpoint_every', 'variants'}
    if required - plan.keys() or plan['purpose'] not in ('verification', 'pilot', 'formal'):
        raise ValueError('Invalid experiment plan')
    if len(set(plan['training_seeds'])) != len(plan['training_seeds']) or len(set(plan['workload_seeds'])) != len(plan['workload_seeds']):
        raise ValueError('Repeated seed in experiment plan')
    if any(len(set(variant['methods'])) != len(variant['methods']) for variant in plan['variants']):
        raise ValueError('Repeated method in experiment plan')
    if args.plan_only:
        write_json(Path(args.output), plan)
        print(json.dumps({'groups': len(list(groups(plan))), 'purpose': plan['purpose'], 'plan': args.output}))
    else:
        formal = plan['purpose'] == 'formal'
        print(run_suite(plan, args.output, args.source_lock if formal else None,
                        args.validation_report if formal else None, args.workers, args.threads))
