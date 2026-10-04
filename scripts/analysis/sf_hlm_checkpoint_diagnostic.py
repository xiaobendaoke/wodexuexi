"""Read-only checkpoint replay against the frozen SF-HLM v2 protocol.

All model/environment execution is Linux-only. Diagnostic outputs are separate
from the frozen experiment. No optimizer steps, checkpoint selection or gates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_runner(args):
    path = args.extension_root / 'scripts/analysis/sf_hlm_v2_experiment.py'
    spec = importlib.util.spec_from_file_location('checkpoint_v2_runner', path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner


def unit_path(args):
    return (args.diagnostic_root / 'units' / args.method / f'ep_{args.checkpoint_episode:03d}'
            / f'block_{args.block}_workload_{args.workload}' / f'episode_{args.episode:05d}')


def load_upper(args, runner, p, models, episode):
    import config
    from marl_models.sf_hlm_v2 import PlainMAPPO, ServiceMAPPO
    if args.method == 'baseline':
        return None, None
    cls = PlainMAPPO if args.method == 'greedy' else ServiceMAPPO
    upper = cls(config.OBS_DIM_SINGLE, config.NUM_UAVS, enabled=args.method == 'full')
    if episode == 0:
        upper.warm_start(models['upper'])
        return upper, None
    train = (args.output / 'pilot_training/complete' / args.method
             / f'block_{args.block}_workload_{p["training_workloads"][str(args.block)]}')
    doc = read(train / 'result.json')
    folder = train / 'recovery' / f'episode_{episode - 1:05d}'
    rel = str((folder / 'feedback_v2.pt').relative_to(train))
    if sha(folder / 'feedback_v2.pt') != doc['files'][rel]:
        raise ValueError('Recovery checkpoint hash differs')
    upper.load(folder, doc['system_contract'])
    return upper, dict(path=str(folder / 'feedback_v2.pt'), sha256=doc['files'][rel])


def replay(args, runner):
    import numpy as np
    import torch
    from torch.distributions import Normal, kl_divergence
    p, _ = runner.bootstrap(args)
    models, source_contract = runner.base.configure(args, p, historical=True)
    lower = runner.read_lower(args, p) if args.method != 'greedy' else None
    upper, checkpoint = load_upper(args, runner, p, models, args.checkpoint_episode)
    previous, _ = load_upper(args, runner, p, models, max(0, args.checkpoint_episode - 10))
    from utils.canonical_experiment import digest, exogenous_frame
    method = 'learned' if args.method == 'baseline' else args.method
    fingerprint = runner.base.policy_fingerprint([models['upper'], models['lower'], lower, upper, previous])
    env, obs = runner.episode_env(args.workload, args.episode)
    rng = np.random.default_rng(runner.res.trajectory_seed(sha(args.output / 'protocol.json'),
                                                         args.block, args.workload, args.episode, 'search'))
    reference = None
    if args.method == 'baseline' or args.checkpoint_episode == 50:
        phase = 'lower_eval' if args.method == 'baseline' else 'pilot_eval'
        original = (args.output / phase / 'complete' / method
                    / f'block_{args.block}_workload_{args.workload}'
                    / f'episode_{args.episode:05d}' / 'episode.json')
        reference = read(original)
    totals = {k: 0. for k in ('generated', 'admitted', 'success', 'potential', 'energy', 'mbs', 'fairness')}
    frames, samples = [], []
    trace = hashlib.sha256()
    action_trace = hashlib.sha256()
    for step in range(p['steps']):
        trace.update(digest(exogenous_frame(env)).encode())
        potential = sum(u.latent_request[0] < __import__('config').NUM_SERVICES for u in env.ues)
        motion, actions, info, entry, _, _ = runner.policy(
            env, obs, method, models, family='complete', lower=lower, upper=upper,
            rng=rng, budget=p['matching_budget'])
        action_trace.update(digest(info['assignment']).encode())
        if step % 25 == 0:
            observation = torch.as_tensor(obs, dtype=torch.float32)[None]
            view = torch.as_tensor(entry['feedback'], dtype=torch.float32)[None]
            with torch.no_grad():
                ref_raw = models['upper'].actor(observation.reshape(-1, observation.shape[-1]))
                ref = Normal(ref_raw.mean[None], ref_raw.stddev[None])
                if upper is None:
                    dist, raw, prior = ref, ref_raw, ref
                    residual_rms = 0.
                else:
                    residuals = []
                    hook = upper.actor_feedback.residual.register_forward_hook(
                        lambda module, inputs, output: residuals.append(float(output.square().mean().sqrt()))) if args.method != 'greedy' else None
                    try:
                        dist, _ = upper.distribution(observation, view)
                    finally:
                        if hook is not None:
                            hook.remove()
                    residual_rms = residuals[0] if residuals else 0.
                    raw = upper.actor(observation.reshape(-1, observation.shape[-1]))
                    prior, _ = previous.distribution(observation, view)
                base_dist = Normal(raw.mean[None], raw.stddev[None])
                zeros = view.clone(); zeros[..., :6] = 0.
                without_pressure = upper.distribution(observation, zeros)[0] if upper is not None else ref
                samples.append(dict(step=step,
                    action_delta=float((dist.mean - ref.mean).square().sum(-1).sqrt().mean()),
                    base_action_delta=float((base_dist.mean - ref.mean).square().sum(-1).sqrt().mean()),
                    branch_action_delta=float((dist.mean - base_dist.mean).square().sum(-1).sqrt().mean()),
                    pressure_action_delta=float((dist.mean - without_pressure.mean).square().sum(-1).sqrt().mean()),
                    kl_original=float(kl_divergence(dist, ref).sum(-1).mean()),
                    kl_previous=float(kl_divergence(dist, prior).sum(-1).mean()),
                    std=float(dist.stddev.mean()), std_ratio=float((dist.stddev / ref.stddev).mean()),
                    residual_rms=residual_rms,
                    pressure_abs_max=view[..., :6].abs().reshape(-1, 6).max(0).values.tolist()))
            uav = np.stack([u.pos[:2] for u in env.uavs]); ue = np.stack([u.pos[:2] for u in env.ues])
            config = __import__('config')
            distance = np.linalg.norm(ue[:, None] - uav[None], axis=-1)
            pair = np.linalg.norm(uav[:, None] - uav[None], axis=-1)
            count = (distance <= config.UAV_COVERAGE_RADIUS).sum(1)
            grid = np.minimum((uav / np.array([config.AREA_WIDTH, config.AREA_HEIGHT]) * 7).astype(int), 6).clip(0, 6)
            frames.append(dict(step=step, coverage=float((count > 0).mean()), overlap=float((count > 1).mean()),
                               pair_distance=float(pair[np.triu_indices(len(uav), 1)].mean()),
                               uav_grid=grid.tolist()))
        obs, _, metrics = env.step(motion, offloading_actions=actions)
        runner.base.assert_step(metrics, info)
        obs = np.asarray(obs, np.float32)
        if reference is not None:
            old = reference['steps'][step]
            if list(info['assignment']) != old['assignment']:
                raise ValueError(f'Replay assignment mismatch at {step}')
            for name in ('generated_service_count', 'admitted_service_count', 'deadline_satisfied_count', 'energy', 'fairness'):
                if not math.isclose(metrics[name], old['metrics'][name], rel_tol=1e-10, abs_tol=1e-9):
                    raise ValueError(f'Replay {name} mismatch at {step}')
        totals['potential'] += potential
        for k, name in [('generated', 'generated_service_count'), ('admitted', 'admitted_service_count'),
                        ('success', 'deadline_satisfied_count'), ('energy', 'energy'), ('mbs', 'service_offloads_mbs'), ('fairness', 'fairness')]:
            totals[k] += float(metrics[name])
    if fingerprint != runner.base.policy_fingerprint([models['upper'], models['lower'], lower, upper, previous]):
        raise ValueError('Diagnostic modified model parameters')
    if reference is not None and trace.hexdigest() != reference['summary']['exogenous_trace_hash']:
        raise ValueError('Replay exogenous stream differs')
    result = dict(status='PASS', source_contract=source_contract, checkpoint=checkpoint,
                  replay_verified=reference is not None, summary=totals, frames=frames, diagnostics=samples,
                  exogenous_trace_hash=trace.hexdigest(), assignment_trace_hash=action_trace.hexdigest(), policy_fingerprint=fingerprint)
    runner.res.atomic_json(unit_path(args) / 'episode.json', result)


def worker(args, runner):
    folder = unit_path(args)
    with runner.res.task_lock(folder / '.task.lock'):
        contract = dict(diagnostic_protocol_sha256=sha(args.diagnostic_root / 'protocol.json'), method=args.method,
                        checkpoint_episode=args.checkpoint_episode, block=args.block, workload=args.workload, episode=args.episode)
        if (folder / 'result.json').exists():
            receipt = read(folder / 'result.json')
            if receipt['contract'] != contract or receipt['sha256'] != sha(folder / 'episode.json'):
                raise ValueError('Diagnostic recovery identity mismatch')
            return
        with runner.res.CpuLease(args.resources, task=['checkpoint_diagnostic', *contract.values()], kind='environment'):
            replay(args, runner)
        runner.res.atomic_json(folder / 'result.json', dict(status='PASS', contract=contract, sha256=sha(folder / 'episode.json')))


def pipeline(args, runner):
    original = read(args.output / 'protocol.json')
    tasks = [('baseline', 0, b, w, ep) for b in original['historical_blocks'] for w in original['eval_workloads'] for ep in range(2)]
    tasks += [(m, c, b, w, ep) for m in ('full', 'no_feedback', 'greedy') for c in (10, 20, 30, 40, 50)
              for b in original['historical_blocks'] for w in original['eval_workloads'] for ep in range(2)]
    if args.mode == 'smoke':
        tasks = [tasks[0], ('full', 50, original['historical_blocks'][0], original['eval_workloads'][0], 0)]
    protocol = dict(version='sf_hlm_checkpoint_diagnostic_v1', original_protocol_sha256=sha(args.output / 'protocol.json'),
                    script_sha256=sha(__file__), tasks=tasks, diagnostic_stride=25, no_training=True,
                    checkpoint_selection=False, independent_confirmation=False)
    target = args.diagnostic_root / 'protocol.json'
    if target.exists() and read(target) != json.loads(json.dumps(protocol)):
        raise ValueError('Diagnostic protocol changed')
    runner.res.atomic_json(target, protocol)
    def one(task):
        cmd = [sys.executable, str(Path(__file__).resolve()), 'worker']
        for k in ('source_root', 'suite_root', 'extension_root', 'output', 'legacy_results', 'resources', 'diagnostic_root'):
            cmd += ['--' + k.replace('_', '-'), str(getattr(args, k))]
        for k, v in zip(('method', 'checkpoint_episode', 'block', 'workload', 'episode'), task):
            cmd += ['--' + k.replace('_', '-'), str(v)]
        log = args.diagnostic_root / 'job_logs' / ('_'.join(map(str, task)) + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT)
    with runner.res.task_lock(args.diagnostic_root / 'controller.lock'):
        done, failures = 0, []
        stop = threading.Event()
        telemetry = threading.Thread(target=runner.res.telemetry, args=(args.resources, stop), daemon=True)
        telemetry.start()
        try:
            slots = min(args.workers, read(args.resources / 'profile.json')['slots'])
            with ThreadPoolExecutor(max_workers=slots) as pool:
                futures = {pool.submit(one, t): t for t in tasks}
                for future in as_completed(futures):
                    try:
                        future.result(); done += 1
                    except Exception as exc:
                        failures.append(dict(task=futures[future], error=str(exc)))
                    runner.res.atomic_json(args.diagnostic_root / 'status.json', dict(completed=done, expected=len(tasks), failures=failures, pid=os.getpid()))
            if failures:
                raise RuntimeError(f'{len(failures)} diagnostic jobs failed')
            runner.res.atomic_json(args.diagnostic_root / 'status.json', dict(completed=done, expected=len(tasks), failures=[], finished=True))
        finally:
            stop.set(); telemetry.join(timeout=5)


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Run model and environment diagnostics on the remote Linux server')
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('worker', 'pipeline', 'smoke'))
    for k in ('source-root', 'suite-root', 'extension-root', 'output', 'legacy-results', 'resources', 'diagnostic-root'):
        parser.add_argument('--' + k, type=Path, required=True)
    parser.add_argument('--method'); parser.add_argument('--workers', type=int, default=192)
    for k in ('checkpoint-episode', 'block', 'workload', 'episode'):
        parser.add_argument('--' + k, type=int, default=0)
    parser.add_argument('--family', default='complete'); parser.add_argument('--seed', type=int, default=690001)
    args = parser.parse_args()
    for k in ('source_root', 'suite_root', 'extension_root', 'output', 'legacy_results', 'resources', 'diagnostic_root'):
        setattr(args, k, getattr(args, k).expanduser().resolve())
    runner = load_runner(args)
    if args.mode == 'worker':
        worker(args, runner)
    else:
        pipeline(args, runner)


if __name__ == '__main__':
    main()
