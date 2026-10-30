#!/usr/bin/env python3
"""Frozen service-only experiments. Numerical subcommands require the server."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import REPO_ROOT
ROOT = REPO_ROOT
from utils.dynamic_service_scenarios import NAMESPACE, Scenario, seed_for

TASK = 'chapter5_dynamic_service_20261028'
METHODS = ('full', 'no_memory', 'no_feedback', 'parallel', 'independent',
           'greedy_lower', 'matching_lower', 'history', 'mpc')
TRAINED = METHODS[:5]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def source_contract():
    # Include imported legacy radio/motion/config dependencies in the lock.
    selected = [ROOT/'config.py', ROOT/'paths.py']
    for folder in ('environment', 'marl_models', 'utils', 'scripts/analysis', 'tests'):
        selected.extend((ROOT/folder).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(selected))}


def protocol():
    return dict(namespace=NAMESPACE, version=1, source=source_contract(), methods=METHODS,
                trained=TRAINED, cells=[(k, s) for k in 'ABCD' for s in (.6, 1., 1.4)],
                primary=('D', 1.), episodes=400, horizon=1000, main_margin=.01,
                upper=dict(hidden=128, sequence=32, burn_in=16, gamma=.99, gae=.95),
                lower=dict(hidden=128, layers=2, gamma=1, minibatch_graphs=8), ppo=dict(lr=.0003, clip=.2, epochs=4, grad_clip=.5),
                warmup=50, upper_only_until=200, alternating_block=5,
                dev_seeds=[seed_for('dev/train', i) for i in range(5)],
                confirm_seeds=[seed_for('confirm/train', i) for i in range(10)],
                development_gate=dict(comparisons=['history', 'independent', 'no_memory', 'no_feedback', 'parallel'],
                                      minimum_positive_blocks=4, minimum_mean_difference=0.))


def configure_runtime():
    if sys.platform != 'linux':
        raise RuntimeError('Numerical workloads are server-only; local host may run scenario/static checks')
    os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return torch


def models(method, device='cpu'):
    from marl_models.dynamic_service_hmarl import UpperPolicy, LowerPolicy
    upper = UpperPolicy(memory=method not in ('no_memory', 'independent'), feedback=method != 'no_feedback').to(device)
    mode = 'parallel' if method == 'parallel' else ('independent' if method == 'independent' else 'autoregressive')
    return upper, LowerPolicy(mode).to(device)


def checkpoint_path(out, split, block, method):
    return out/'models'/split/str(block)/(method+'.pt')


def train(args, torch):
    import numpy as np
    from environment.dynamic_service import DynamicServiceEnv
    from marl_models.dynamic_service_hmarl import update_lower, update_upper
    from utils.dynamic_service_baselines import HistoryController
    seed = seed_for(f'{args.split}/train', args.block)
    torch.manual_seed(seed)
    upper, lower = models(args.method)
    uopt, lopt = (torch.optim.Adam(m.parameters(), lr=3e-4) for m in (upper, lower))
    file = checkpoint_path(args.output, args.split, args.block, args.method)
    frozen = None
    if args.method in ('no_memory', 'no_feedback', 'parallel'):
        frozen_file = checkpoint_path(args.output, args.split, args.block, 'full')
        frozen = torch.load(frozen_file, map_location='cpu', weights_only=False)
        if (frozen['episode'] != args.episodes or frozen['protocol'] != args.contract
                or frozen['method'] != 'full' or frozen['seed'] != seed):
            raise RuntimeError('ablations require the complete common full-policy checkpoint')
        if args.method == 'parallel':
            upper.load_state_dict(frozen['upper'])
        else:
            lower.load_state_dict(frozen['lower'])
    first = 0
    if file.exists():
        saved = torch.load(file, map_location='cpu', weights_only=False)
        if (saved['protocol'] != args.contract or saved['method'] != args.method or saved['seed'] != seed
                or saved['episode'] > args.episodes):
            raise RuntimeError('source/protocol mismatch; refusing resume')
        upper.load_state_dict(saved['upper']); lower.load_state_dict(saved['lower'])
        uopt.load_state_dict(saved['uopt']); lopt.load_state_dict(saved['lopt'])
        torch.set_rng_state(saved['rng'])
        first = saved['episode']
        last_log = args.output/'training'/args.split/str(args.block)/args.method/f'{first-1:04}.json'
        if first and not last_log.exists() and 'last_record' in saved:
            atomic_json(last_log, saved['last_record'])
    if first == args.episodes:
        return
    file.parent.mkdir(parents=True, exist_ok=True)
    for episode in range(first, args.episodes):
        selection = np.random.RandomState(seed_for(f'{args.split}/cell/{args.block}', episode))
        kind = 'ABCD'[selection.randint(4)]
        scale = (.6, 1., 1.4)[selection.randint(3)]
        env = DynamicServiceEnv(Scenario(seed_for(f'{args.split}/workload/{args.block}', episode), kind), scale)
        controller, hidden = HistoryController(), upper.initial_state()
        warm = episode < 50
        lower_active = warm or (episode >= 200 and ((episode-200)//5)%2 == 1)
        upper_active = not warm and not lower_active
        if args.method in ('no_memory', 'no_feedback'):
            lower_active = False
        if args.method == 'parallel':
            upper_active = False
        upper_rollout, lower_rollout = [], []
        totals = dict(generated=0, admitted=0, success=0)
        started = time.perf_counter()
        for t in range(args.steps):
            observation = env.observe()
            batch = env.batch()
            controller.update(observation, env.positions)
            hidden_before = hidden.detach().cpu().numpy()
            upper_action = upper.act(observation, hidden, stochastic=upper_active)
            hidden = upper_action['hidden']
            if warm and args.method != 'parallel':
                actions = controller.actions(env.positions)
            else:
                actions = upper_action['actions']
            lower_action = lower.act(batch, stochastic=lower_active)
            stats = env.step(actions, lower_action['assignment'])
            if lower_active:
                lower_rollout.append(dict(batch=batch, **lower_action, reward=stats['success']/max(len(batch), 1)))
            if upper_active:
                upper_rollout.append(dict(observation=observation, hidden_before=hidden_before,
                                          raw=upper_action['raw'], logprob=upper_action['logprob'],
                                          value=upper_action['value'], reward=stats['reward']))
            for key in totals:
                totals[key] += stats[key]
        # Smoke trajectories terminate here; they are never promotion evidence.
        lloss = update_lower(lower, lopt, lower_rollout) if lower_active else None
        uloss = update_upper(upper, uopt, upper_rollout) if upper_active else None
        record = dict(**totals, episode=episode+1, kind=kind, scale=scale, lower_loss=lloss, upper_loss=uloss,
                      seconds=time.perf_counter()-started, lower_active=lower_active, upper_active=upper_active,
                      workload_fingerprint=env.tape.fingerprint(), protocol=args.contract, seed=seed)
        saved = dict(upper=upper.state_dict(), lower=lower.state_dict(), uopt=uopt.state_dict(), lopt=lopt.state_dict(),
                     episode=episode+1, rng=torch.get_rng_state(), protocol=args.contract,
                     method=args.method, seed=seed, smoke=args.steps != 1000 or args.episodes != 400, last_record=record)
        tmp = file.with_suffix('.tmp')
        torch.save(saved, tmp); os.replace(tmp, file)
        atomic_json(args.output/'training'/args.split/str(args.block)/args.method/f'{episode:04}.json',
                    record)
        print(json.dumps(dict(method=args.method, episode=episode+1, **totals)), flush=True)


def evaluate_job(args, torch):
    import numpy as np
    from environment.dynamic_service import DynamicServiceEnv
    from utils.dynamic_service_baselines import HistoryController, greedy, matching_search
    method = args.method
    name = f'{args.kind}_{args.scale}_{args.workload}_{args.repeat}.json'
    destination = args.output/'evaluation'/args.split/str(args.block)/method/name
    if destination.exists():
        existing = json.loads(destination.read_text(encoding='utf-8'))
        if existing['protocol'] != args.contract:
            raise RuntimeError('refusing to replace existing evaluation')
        print(json.dumps(dict(status='REUSED_IMMUTABLE_RESULT', path=str(destination))), flush=True)
        return
    model_method = 'full' if method in ('greedy_lower', 'matching_lower') else method
    learned = method not in ('history', 'mpc')
    upper, lower = models(model_method if learned else 'full')
    if learned:
        model_file = checkpoint_path(args.output, args.split, args.block, model_method)
        saved = torch.load(model_file, map_location='cpu', weights_only=False)
        if (saved['protocol'] != args.contract or saved['episode'] != args.episodes
                or saved['method'] != model_method or saved['seed'] != seed_for(f'{args.split}/train', args.block)):
            raise RuntimeError('only final matching-contract checkpoint may be evaluated')
        upper.load_state_dict(saved['upper']); lower.load_state_dict(saved['lower'])
        upper.eval(); lower.eval()
    workload = seed_for(f'{args.split}/evaluation/{args.workload}', args.repeat)
    env = DynamicServiceEnv(Scenario(workload, args.kind), args.scale)
    hidden, controller = upper.initial_state(), HistoryController()
    totals = dict(generated=0, admitted=0, success=0, uncovered=0, deadline_misses=0, decision_timeout=0, mbs=0)
    times, rows, snapshots = [], [], []
    snapshot_policies = {}
    if method == 'full':
        for control in ('parallel', 'independent'):
            file = checkpoint_path(args.output, args.split, args.block, control)
            if file.exists():
                state = torch.load(file, map_location='cpu', weights_only=False)
                if (state['protocol'] != args.contract or state['episode'] != args.episodes
                        or state['seed'] != seed_for(f'{args.split}/train', args.block)):
                    raise RuntimeError('snapshot control is not the matching final model')
                _, control_lower = models(control)
                control_lower.load_state_dict(state['lower']); control_lower.eval()
                snapshot_policies[control] = control_lower
    for t in range(args.steps):
        started = time.perf_counter()
        observation, batch = env.observe(), env.batch()
        controller.update(observation, env.positions)
        if learned:
            action = upper.act(observation, hidden)
            actions, hidden = action['actions'], action['hidden']
        else:
            actions = controller.mpc(env, batch) if method == 'mpc' else controller.actions(env.positions)
        if method in ('history', 'mpc', 'matching_lower'):
            assignment = matching_search(batch)
        elif method == 'greedy_lower':
            assignment = greedy(batch)
        else:
            assignment = lower.act(batch)['assignment']
        elapsed = time.perf_counter()-started
        timeout = elapsed > 1.
        if method == 'full' and args.repeat < 2 and t in (0, 250, 500, 750):
            # Same observed batch; counterfactual results never enter env.step.
            from environment.dynamic_service import evaluate
            counterfactual = {}
            for control, allocation in (('full', assignment), ('greedy', greedy(batch)),
                                        ('matching', matching_search(batch))):
                result = evaluate(batch, allocation)
                counterfactual[control] = dict(success=result['success'], assignment=result['assignment'].tolist())
            for control, policy in snapshot_policies.items():
                result = evaluate(batch, policy.act(batch)['assignment'])
                counterfactual[control] = dict(success=result['success'], assignment=result['assignment'].tolist())
            snapshots.append(dict(t=t, batch={key: getattr(batch, key).tolist() for key in batch.__dataclass_fields__},
                                  controls=counterfactual))
        row = env.step(actions, assignment, decision_timeout=timeout)
        if timeout:
            # Discard hypothetical GRU history when actual executed motion differs.
            hidden = upper.initial_state()
        for key in totals:
            totals[key] += row[key]
        times.append(elapsed)
        rows.append(dict(t=t, seconds=elapsed, **row))
    out = dict(split=args.split, block=args.block, method=method, kind=args.kind, scale=args.scale,
               workload=args.workload, repeat=args.repeat, workload_seed=workload, protocol=args.contract,
               workload_fingerprint=env.tape.fingerprint(), totals=totals,
               dsr=totals['success']/max(totals['generated'], 1), fairness=rows[-1]['fairness'],
               timing=dict(median=float(np.median(times)), p95=float(np.percentile(times, 95))),
               smoke=args.steps != 1000 or args.episodes != 400, rows=rows, snapshots=snapshots)
    if learned:
        out['checkpoint_sha256'] = hashlib.sha256(model_file.read_bytes()).hexdigest()
    atomic_json(destination, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'snapshots')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('protocol', 'train', 'evaluate', 'analyze'))
    parser.add_argument('--output', type=Path, default=ROOT/'results'/TASK)
    parser.add_argument('--split', choices=('dev', 'confirm', 'smoke'), default='dev')
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--method', choices=METHODS, default='full')
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--kind', choices=tuple('ABCD'), default='D')
    parser.add_argument('--scale', type=float, choices=(.6, 1., 1.4), default=1.)
    parser.add_argument('--workload', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--primary-only', action='store_true')
    args = parser.parse_args()
    if args.block < 0 or (args.split != 'smoke' and args.block >= (5 if args.split == 'dev' else 10)):
        parser.error('block outside registered split')
    if not 0 <= args.workload < 5 or not 0 <= args.repeat < 6:
        parser.error('workload/repeat outside registered evaluation grid')
    definition = protocol()
    args.contract = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    protocol_file = args.output/'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text(encoding='utf-8')) != json.loads(json.dumps(definition)):
            raise RuntimeError('existing output is frozen to a different source/protocol')
    else:
        atomic_json(protocol_file, definition)
    if args.command == 'protocol':
        print(args.contract)
        return
    if args.command == 'analyze':
        configure_runtime()
        from utils.dynamic_service_analysis import analyze
        result = analyze(args.output, args.split, args.primary_only)
        atomic_json(args.output/'analysis'/f'{args.split}.json', result)
        if args.split == 'dev' and result['status'] == 'COMPLETE':
            atomic_json(args.output/'development_gate.json', dict(**result['development_gate'], protocol=args.contract,
                        evidence_file='analysis/dev.json', evidence_sha256=hashlib.sha256((args.output/'analysis/dev.json').read_bytes()).hexdigest()))
        print(json.dumps({k: v for k, v in result.items() if k not in ('results', 'file_hashes', 'comparisons', 'missing')}))
        return
    if args.method not in TRAINED and args.command == 'train':
        parser.error('this control has no training job')
    if args.split != 'smoke' and (args.episodes != 400 or args.steps != 1000):
        parser.error('reduced runs require --split smoke')
    if args.split == 'confirm':
        gate = args.output/'development_gate.json'
        if not gate.exists():
            raise RuntimeError('confirmation requires a verified development gate')
        gate_data = json.loads(gate.read_text())
        evidence = args.output/'analysis/dev.json'
        if (gate_data.get('status') != 'PASS' or gate_data.get('protocol') != args.contract
                or not evidence.exists() or hashlib.sha256(evidence.read_bytes()).hexdigest() != gate_data['evidence_sha256']):
            raise RuntimeError('confirmation requires unchanged complete development evidence')
    lock = args.output/'locks'/f'{args.command}_{args.split}_{args.block}_{args.method}_{args.kind}_{args.scale}_{args.workload}_{args.repeat}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    # OS lock is released on process exit; no stale lock deletion required.
    import fcntl
    with lock.open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0); handle.truncate(); handle.write(str(os.getpid())); handle.flush()
        torch = configure_runtime()
        if args.command == 'train':
            train(args, torch)
        else:
            evaluate_job(args, torch)


if __name__ == '__main__':
    main()
