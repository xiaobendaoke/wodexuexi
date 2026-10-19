"""Replay the Chapter 5 candidate and compare frozen Chapter 4 assignments.

Run only on the remote Linux host from the frozen whole-system source directory.
Counterfactual assignments never advance the online environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
import torch

import config
from environment.env import Env
from run_canonical_experiment import configure_method, policy_step
from utils.canonical_experiment import config_snapshot, digest, exogenous_frame, source_manifest, write_json
from utils.chapter4 import objective, solve_snapshot
from utils.logger import refresh_derived_config_fields
from utils.snapshot_evaluator import capture, evaluate


METHODS = ('online_lower', 'heuristic_control', 'greedy_single_pass', 'iterative_base', 'mc_itr')
FULL_SNAPSHOT_STEPS = (0, 250, 500, 750)
SMOKE_SNAPSHOT_STEPS = (0, 250)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models(source_root: Path, suite_root: Path, plan: dict, seed: int) -> dict:
    importlib.reload(config)
    variant = plan['variants'][0]
    config.NUM_UES = variant['ue_count']
    config.UAV_COMPUTING_CAPACITY = config.UAV_COMPUTING_CAPACITY * variant['compute_scale']
    refresh_derived_config_fields()
    config.PPO_EPOCHS = plan['ppo_epochs']
    config.PPO_BATCH_SIZE = plan['batch_size']
    config.PPO_ROLLOUT_LENGTH = plan['rollout']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    models = configure_method(plan['candidate_method'])
    checkpoint = (suite_root / 'main' / plan['candidate_method'] / f'seed_{seed}' /
                  'train' / 'attempt_0000' / 'checkpoints' /
                  f'episode_{plan["train_episodes"] - 1:05d}')
    contract = json.loads((checkpoint / 'contract.json').read_text())
    expected_source = source_manifest()['source_hash']
    if (contract['method'] != plan['candidate_method'] or
            contract['config_hash'] != digest(config_snapshot()) or
            contract['source_hash'] != expected_source):
        raise ValueError(f'Checkpoint contract mismatch: {checkpoint}')
    for relative, expected in contract['files'].items():
        if file_sha(checkpoint / relative) != expected:
            raise ValueError(f'Checkpoint file mismatch: {checkpoint / relative}')
    for name, model in models.items():
        model.load(str(checkpoint / name))
    return models


def original_episode(suite_root: Path, method: str, seed: int, workload: int, episode: int) -> dict:
    job = suite_root / 'main' / method / f'seed_{seed}' / f'workload_{workload}'
    completed = json.loads((job / 'completed.json').read_text())
    if completed['status'] != 'PASS':
        raise ValueError(f'Original evaluation incomplete: {job}')
    path = job / completed['attempt'] / 'last_invocation.json'
    if file_sha(path) != completed['hashes'][str(path.relative_to(job))]:
        raise ValueError(f'Original evaluation summary changed: {path}')
    return json.loads(path.read_text())['completed'][episode]


def assignment_from_policy(snapshot, offload: np.ndarray) -> tuple[int, ...]:
    assignment = tuple(int(offload[request[0], request[1]]) for request in snapshot.requests)
    for action, candidates in zip(assignment, snapshot.candidates):
        if action not in candidates:
            raise ValueError('Online lower action is outside the hard-feasible set')
    return assignment


def compact_solver(result: dict) -> dict:
    return {
        'objective': list(result['objective']),
        'wall_seconds': float(result['wall_seconds']),
        'termination': result.get('termination'),
        'local_optimality_verified': result.get('local_optimality_verified'),
    }


def run_job(suite_root: Path, output_root: Path, plan: dict, seed: int, workload: int,
            episodes: int, snapshot_steps: tuple[int, ...], script_hash: str) -> Path:
    output = output_root / f'seed_{seed}_workload_{workload}.json'
    identity = {
        'source_hash': source_manifest()['source_hash'],
        'analysis_script_hash': script_hash,
        'original_protocol': plan['protocol'],
        'method': plan['candidate_method'],
        'training_seed': seed, 'workload_seed': workload,
        'episodes': episodes, 'steps_per_episode': plan['steps'],
        'snapshot_steps': snapshot_steps,
    }
    if output.exists():
        old = json.loads(output.read_text())
        if old['identity'] != json.loads(json.dumps(identity)) or old['status'] != 'PASS':
            raise ValueError(f'Existing job has a different identity: {output}')
        return output
    models = load_models(Path.cwd(), suite_root, plan, seed)
    env = Env(scenario_seed=workload)
    selected = set(snapshot_steps)
    snapshots = []
    replay_checks = []
    for episode in range(episodes):
        obs = np.asarray(env.reset(), dtype=np.float32)
        for model in models.values():
            model.reset()
        trace = hashlib.sha256()
        generated = successful = 0
        for step in range(plan['steps']):
            trace.update(digest(exogenous_frame(env)).encode())
            started = time.perf_counter()
            motion, offload, _ = policy_step(models, env, obs, exploration=False)
            policy_seconds = time.perf_counter() - started
            motion = np.clip(motion, -1, 1)
            pending = None
            if step in selected:
                started = time.perf_counter()
                snapshot = capture(env, motion)
                capture_seconds = time.perf_counter() - started
                online_assignment = assignment_from_policy(snapshot, offload)
                online_result = evaluate(snapshot, online_assignment)
                solved = solve_snapshot(snapshot, max_rounds=1000, exact_budget=0)
                methods = {'online_lower': {'objective': list(objective(online_result)),
                                            'wall_seconds': float(policy_seconds)}}
                for name in METHODS[1:]:
                    methods[name] = compact_solver(solved[name])
                methods['mc_itr']['full_path_wall_seconds'] = sum(
                    float(solved[name]['wall_seconds'])
                    for name in ('heuristic_control', 'greedy_single_pass', 'mc_itr'))
                pending = {
                    'episode': episode, 'step': step,
                    'snapshot_hash': snapshot.snapshot_hash,
                    'admitted_service_requests': len(snapshot.requests),
                    'capture_wall_seconds': capture_seconds,
                    'methods': methods,
                    'online_snapshot_success': int(online_result['metrics']['deadline_satisfied_count']),
                }
            next_obs, _, metrics = env.step(motion, offloading_actions=offload)
            generated += int(metrics['generated_service_count'])
            successful += int(metrics['deadline_satisfied_count'])
            if pending is not None:
                if (pending['admitted_service_requests'] != int(metrics['admitted_service_count']) or
                        pending['online_snapshot_success'] != int(metrics['deadline_satisfied_count'])):
                    raise ValueError(f'Snapshot/online step mismatch at {seed}, {workload}, {episode}, {step}')
                snapshots.append(pending)
            obs = np.asarray(next_obs, dtype=np.float32)
        archived = original_episode(suite_root, plan['candidate_method'], seed, workload, episode)
        if (trace.hexdigest() != archived['exogenous_trace_hash'] or
                generated != int(archived['generated_service_count']) or
                successful != int(archived['deadline_satisfied_count'])):
            raise ValueError(f'Online replay differs from archived evaluation: {seed}, {workload}, {episode}')
        replay_checks.append({'episode': episode, 'exogenous_trace_hash': trace.hexdigest(),
                              'generated_service_count': generated,
                              'deadline_satisfied_count': successful})
    result = {'status': 'PASS', 'identity': identity, 'replay_checks': replay_checks,
              'snapshots': snapshots}
    temporary = output.with_suffix('.json.tmp')
    write_json(temporary, result)
    temporary.replace(output)
    return output


def aggregate(paths: list[Path], plan: dict, seeds: list[int], workloads: list[int],
              expected_per_job: int) -> dict:
    lookup = {}
    for path in paths:
        row = json.loads(path.read_text())
        if row['status'] != 'PASS' or len(row['snapshots']) != expected_per_job:
            raise ValueError(f'Incomplete bridge job: {path}')
        identity = row['identity']
        key = (identity['training_seed'], identity['workload_seed'])
        if key in lookup:
            raise ValueError(f'Duplicate bridge job: {key}')
        lookup[key] = row
    units = []
    for seed in seeds:
        for workload in workloads:
            row = lookup[(seed, workload)]
            unit = {'training_seed': seed, 'workload_seed': workload, 'snapshots': len(row['snapshots'])}
            for method in METHODS:
                values = np.array([s['methods'][method]['objective'] for s in row['snapshots']], dtype=float)
                unit[method] = {'deadline_misses': float(values[:, 0].mean()),
                                'mean_normalized_latency': float(values[:, 1].mean()),
                                'normalized_assignment_energy': float(values[:, 2].mean()),
                                'wall_seconds': float(np.mean([s['methods'][method]['wall_seconds']
                                                               for s in row['snapshots']]))}
                if method == 'mc_itr':
                    unit[method]['full_path_wall_seconds'] = float(np.mean([
                        s['methods'][method]['full_path_wall_seconds'] for s in row['snapshots']]))
            units.append(unit)
    differences = np.array([[next(u for u in units if u['training_seed'] == seed and
                                   u['workload_seed'] == workload)['mc_itr']['deadline_misses'] -
                             next(u for u in units if u['training_seed'] == seed and
                                   u['workload_seed'] == workload)['online_lower']['deadline_misses']
                             for workload in workloads] for seed in seeds], dtype=float)
    seed_means = differences.mean(axis=1)
    if len(seeds) > 1:
        sem = stats.sem(seed_means)
        crit = stats.t.ppf(.975, len(seeds) - 1)
        interval = [float(seed_means.mean() - crit * sem), float(seed_means.mean() + crit * sem)]
    else:
        interval = None
    return {'status': 'PASS', 'study': 'chapter4_chapter5_bridge',
            'source_hash': source_manifest()['source_hash'],
            'statistical_unit': 'training seed after averaging paired workload means',
            'training_seeds': seeds, 'workload_seeds': workloads,
            'evaluation_jobs': len(paths), 'snapshots': len(paths) * expected_per_job,
            'units': units,
            'mc_itr_minus_online_lower_deadline_misses': {
                'mean_difference': float(differences.mean()),
                'training_seed_t_ci95': interval,
                'seed_mean_differences': [float(v) for v in seed_means],
            },
            'interpretation_boundary': 'Frozen counterfactuals on candidate-policy trajectories, not online integration or long-run DSR.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    suite_root = args.suite_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    suite = json.loads((suite_root / 'suite.json').read_text())
    status = json.loads((suite_root / 'status.json').read_text())
    if status['status'] != 'PASS' or suite['source_hash'] != source_manifest()['source_hash']:
        raise ValueError('Original whole-system suite or frozen source is not valid')
    plan = suite['plan']
    if (len(plan['variants']) != 1 or plan['variants'][0]['name'] != 'main' or
            plan['variants'][0]['ue_count'] != 100 or plan['variants'][0]['compute_scale'] != 1.):
        raise ValueError('Bridge requires the original 100-UE main setting')
    seeds = plan['training_seeds'][:1] if args.smoke else plan['training_seeds']
    workloads = plan['workload_seeds'][:1] if args.smoke else plan['workload_seeds']
    episodes = 1 if args.smoke else 2
    snapshot_steps = SMOKE_SNAPSHOT_STEPS if args.smoke else FULL_SNAPSHOT_STEPS
    script_hash = file_sha(Path(__file__))
    write_json(output_root / 'protocol.json', {
        'study': 'chapter4_chapter5_bridge', 'smoke': args.smoke,
        'source_hash': suite['source_hash'], 'analysis_script_hash': script_hash,
        'training_seeds': seeds, 'workload_seeds': workloads,
        'episodes': episodes, 'steps_per_episode': plan['steps'],
        'snapshot_steps': snapshot_steps,
    })
    paths = []
    for seed in seeds:
        for workload in workloads:
            path = run_job(suite_root, output_root, plan, seed, workload,
                           episodes, snapshot_steps, script_hash)
            paths.append(path)
            print(f'PASS {seed=} {workload=} jobs={len(paths)}/{len(seeds)*len(workloads)}', flush=True)
    result = aggregate(paths, plan, seeds, workloads, episodes * len(snapshot_steps))
    write_json(output_root / 'analysis.json', result)
    print(json.dumps({'status': result['status'], 'jobs': result['evaluation_jobs'],
                      'snapshots': result['snapshots'],
                      'mc_itr_minus_online_lower': result['mc_itr_minus_online_lower_deadline_misses']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
