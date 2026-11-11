"""Reproducible fixed-trajectory Chapter 4 experiments on the remote server."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import config
from environment.env import Env
from utils.canonical_experiment import ROOT, config_snapshot, digest, scenario_manifest, source_manifest, write_json
from utils.chapter4 import PROTOCOL, heuristic_control, solve_snapshot
from utils.formal_protocol import WORKLOAD_SEEDS, verify_release
from utils.snapshot_evaluator import capture
from utils.logger import refresh_derived_config_fields


def plan(suite='all', smoke=False):
    variants = []
    if suite in ('all', 'main'):
        variants.append({'name': 'main', 'ues': 6 if smoke else 100, 'scale': 1., 'exact': False})
    if suite in ('all', 'sensitivity'):
        for n in ([4, 8] if smoke else [60, 80, 120, 140]):
            variants.append({'name': f'ue_{n}', 'ues': n, 'scale': 1., 'exact': False})
        for scale in ([.8, 1.2] if smoke else [.6, .8, 1.2, 1.4]):
            variants.append({'name': f'compute_{scale}', 'ues': 6 if smoke else 100, 'scale': scale, 'exact': False})
    if suite in ('all', 'exact'):
        variants.append({'name': 'exact_small', 'ues': 2 if smoke else 3, 'scale': 1., 'exact': True})
    if not variants:
        raise ValueError('Unknown Chapter 4 suite')
    return {'protocol': PROTOCOL, 'purpose': 'verification' if smoke else 'formal',
            'variants': variants, 'workloads': [19] if smoke else WORKLOAD_SEEDS,
            'episodes': 1 if smoke else 6, 'steps': 2 if smoke else 1000,
            'snapshot_steps': [0, 1] if smoke else list(range(0, 1000, 100)),
            'max_rounds': 20 if smoke else 1000, 'exact_budget': 10000,
            'trajectory': 'fixed open-loop phase-shifted circular actions; amplitude 0.4, period 1000 slots',
            'statistical_unit': 'workload seed; average snapshots and episodes within seed; no training replicates'}


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _motion(step, trajectory):
    if trajectory is not None:
        return np.asarray(trajectory[step], dtype=float)
    phase = 2 * np.pi * (step / 1000 + np.arange(config.NUM_UAVS) / config.NUM_UAVS)
    return .4 * np.column_stack((np.cos(phase), np.sin(phase)))


def _actions(snapshot, assignment):
    result = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=int)
    for request, target in zip(snapshot.requests, assignment):
        result[request[0], request[1]] = target
    return result


def worker(job):
    """Each process owns its config, environment, reference history and output."""
    import torch
    torch.set_num_threads(job['threads'])
    importlib.reload(config)
    config.NUM_UES = job['variant']['ues']
    config.UAV_COMPUTING_CAPACITY = config.UAV_COMPUTING_CAPACITY * job['variant']['scale']
    refresh_derived_config_fields()
    output = Path(job['directory'])
    output.mkdir(parents=True, exist_ok=True)
    identity = {k: v for k, v in job.items() if k != 'directory'}
    identity['config'] = config_snapshot()
    identity['scenario'] = scenario_manifest(job['seed'])
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Chapter 4 job identity changed')
    write_json(manifest, identity)
    done = output / 'completed.json'
    if done.exists():
        completed = json.loads(done.read_text())
        for name, expected in completed['files'].items():
            if _file_hash(output / name) != expected:
                raise ValueError('Chapter 4 completed output changed')
        return str(output)
    write_json(output / 'state.json', {'status': 'RUNNING'})
    try:
        env = Env(scenario_seed=job['seed'])
        files = {}
        for episode in range(job['plan']['episodes']):
            env.reset()
            for step in range(job['plan']['steps']):
                motion = _motion(step, job['trajectory'])
                snapshot = capture(env, motion)
                reference = heuristic_control(snapshot)
                if step in job['plan']['snapshot_steps']:
                    name = f'episode_{episode:03d}_step_{step:04d}.json'
                    path = output / name
                    if path.exists():
                        old = json.loads(path.read_text())
                        # Partial results carry their own payload digest too.
                        attestation = old.pop('payload_hash')
                        if digest(old) != attestation or old['snapshot_hash'] != snapshot.snapshot_hash:
                            raise ValueError('Partial result changed or reference history is not reproducible')
                    else:
                        result = {'episode': episode, 'step': step, 'snapshot_hash': snapshot.snapshot_hash,
                                  'requests': snapshot.requests, 'candidates': snapshot.candidates,
                                  'motion': snapshot.motion, 'reference_assignment': reference,
                                  'uav_positions': [u.pos.tolist() for u in env.uavs],
                                  'ue_positions': [u.pos.tolist() for u in env.ues],
                                  'methods': solve_snapshot(snapshot, max_rounds=job['plan']['max_rounds'],
                                      exact_budget=job['plan']['exact_budget'] if job['variant']['exact'] else 0)}
                        result['payload_hash'] = digest(result)
                        write_json(path, result)
                    files[name] = _file_hash(path)
                    write_json(output / 'state.json', {'status': 'RUNNING', 'episode': episode, 'step': step})
                # Counterfactual policies never drive the shared reference history.
                env.step(motion, offloading_actions=_actions(snapshot, reference))
        files['manifest.json'] = _file_hash(manifest)
        write_json(done, {'status': 'PASS', 'files': files})
        write_json(output / 'state.json', {'status': 'PASS'})
        return str(output)
    except Exception as error:
        write_json(output / 'state.json', {'status': 'FAILED', 'error': repr(error)})
        raise


def aggregate(root, jobs):
    """Paired differences; episodes/snapshots do not inflate the sample size."""
    units = []
    for job in jobs:
        path = Path(job['directory'])
        done = json.loads((path / 'completed.json').read_text())
        values = {}
        expected_count = job['plan']['episodes'] * len(job['plan']['snapshot_steps'])
        names = [n for n in done['files'] if n.startswith('episode_')]
        if len(names) != expected_count:
            raise ValueError('Incomplete snapshot grid')
        for name, expected in done['files'].items():
            if _file_hash(path / name) != expected:
                raise ValueError('Cannot aggregate modified outputs')
        for name in names:
            record = json.loads((path / name).read_text())
            methods = record['methods']
            for method, result in methods.items():
                if 'objective' not in result:
                    continue
                metrics = result['result']['metrics']
                row = [*result['objective'], result['wall_seconds'],
                       float(result.get('local_optimality_verified', False)),
                       metrics['lower_admitted_service_count'], float(not result['result']['requests'])]
                values.setdefault(method, []).append(row)
            exact = methods['exact_small']
            if job['variant']['exact'] and not exact['optimality_complete']:
                raise ValueError('Exact reference profile did not complete enumeration')
            if exact['optimality_complete']:
                optimum = tuple(exact['objective'])
                for result in methods.values():
                    if tuple(result['objective']) < optimum:
                        raise ValueError('A heuristic appears better than the exact reference')
        for method, rows in values.items():
            units.append({'variant': job['variant']['name'], 'workload_seed': job['seed'], 'method': method,
                          'snapshots': len(rows), 'mean': np.mean(rows, axis=0).tolist()})
    comparisons = []
    for variant in sorted({u['variant'] for u in units}):
        selected = [u for u in units if u['variant'] == variant]
        proposed = {u['workload_seed']: np.asarray(u['mean']) for u in selected if u['method'] == 'mc_itr'}
        for method in sorted({u['method'] for u in selected} - {'mc_itr'}):
            reference = {u['workload_seed']: np.asarray(u['mean']) for u in selected if u['method'] == method}
            if proposed.keys() != reference.keys():
                raise ValueError('Paired workload units are incomplete')
            differences = np.array([proposed[s] - reference[s] for s in sorted(proposed)])
            interval = None
            if len(differences) > 1:
                rng = np.random.default_rng(2026)
                draws = differences[rng.integers(0, len(differences), size=(5000, len(differences)))].mean(axis=1)
                interval = np.quantile(draws, [.025, .975], axis=0).tolist()
            comparisons.append({'variant': variant, 'reference': method, 'n_workload_seeds': len(differences),
                                'mean_difference': differences.mean(axis=0).tolist(), 'ci95': interval})
    report = {'status': 'PASS', 'units': units, 'comparisons': comparisons,
              'metrics': ['deadline_misses', 'mean_normalized_latency',
                          'solver_wall_seconds', 'local_optimality_verified_fraction',
                          'admitted_service_requests', 'empty_snapshot_fraction'],
              'note': 'Component differences are reported separately; no scalar relative gap for a lexicographic objective.'}
    write_json(Path(root) / 'analysis.json', report)
    with (Path(root) / 'evaluation_units.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'workload_seed', 'method', 'snapshots', *report['metrics']])
        for unit in units:
            writer.writerow([unit['variant'], unit['workload_seed'], unit['method'], unit['snapshots'], *unit['mean']])
    return report


def run(plan_spec, output, *, workers=1, threads=4, source_lock=None, validation_report=None, trajectory=None):
    if platform.system() != 'Linux':
        raise RuntimeError('Chapter 4 experiments run only on remote Linux')
    if min(workers, threads) < 1 or workers * threads > (os.cpu_count() or 1):
        raise ValueError('Invalid CPU allocation')
    if trajectory is not None:
        array = np.asarray(trajectory, dtype=float)
        if array.shape != (plan_spec['steps'], config.NUM_UAVS, 2) or not np.isfinite(array).all() or np.any(abs(array) > 1):
            raise ValueError('Trajectory must be finite normalized actions [steps, UAVs, 2]')
        trajectory = array.tolist()
        plan_spec = {**plan_spec, 'trajectory': 'provided frozen action trace', 'trajectory_hash': digest(trajectory)}
    source = source_manifest()
    if plan_spec['purpose'] == 'formal':
        lock = verify_release(source_lock, validation_report, source)
        source.update({key: lock.get(key) for key in ('git_sha', 'branch', 'git_clean')})
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise RuntimeError('Less than 10 GiB free output space')
    memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    min_mem_kb = (8 * 1024**2) if plan_spec.get('purpose') == 'formal' else (1024**2)
    if memory['MemAvailable'] < workers * min_mem_kb:
        raise RuntimeError('Insufficient available RAM per worker')
    with (root / '.controller.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'plan': plan_spec, 'source': source, 'trajectory': trajectory, 'threads': threads}
        manifest = root / 'suite.json'
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError('Chapter 4 suite configuration changed')
        write_json(manifest, identity)
        jobs = [{'plan': plan_spec, 'variant': v, 'seed': s, 'source_hash': source['source_hash'],
                 'git_sha': source.get('git_sha'), 'trajectory': trajectory, 'threads': threads,
                 'directory': str(root / v['name'] / f'workload_{s}')}
                for v in plan_spec['variants'] for s in plan_spec['workloads']]
        write_json(root / 'status.json', {'status': 'RUNNING', 'jobs': len(jobs)})
        try:
            if workers == 1:
                for job in jobs:
                    worker(job)
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                    list(pool.map(worker, jobs))
            report = aggregate(root, jobs)
            write_json(root / 'status.json', {'status': 'PASS', 'jobs': len(jobs), 'units': len(report['units'])})
            return report
        except Exception as error:
            write_json(root / 'status.json', {'status': 'FAILED', 'error': repr(error)})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--suite', choices=['all', 'main', 'sensitivity', 'exact'], default='all')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--trajectory-json', help='Optional frozen normalized actions, shape [steps, UAVs, 2]')
    parser.add_argument('--source-lock', default=str(ROOT / 'source-lock.json'))
    parser.add_argument('--validation-report', default=str(ROOT / 'validation.json'))
    args = parser.parse_args()
    specification = plan(args.suite, args.smoke)
    if args.plan_only:
        write_json(Path(args.output), specification)
        print(json.dumps({'status': 'PLAN_ONLY', 'jobs': len(specification['variants']) * len(specification['workloads'])}))
    else:
        result = run(specification, args.output, workers=args.workers, threads=args.threads,
                     source_lock=args.source_lock, validation_report=args.validation_report,
                     trajectory=json.loads(Path(args.trajectory_json).read_text()) if args.trajectory_json else None)
        print(json.dumps({'status': result['status'], 'units': len(result['units'])}))
