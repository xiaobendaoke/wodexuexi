"""Frozen SF-HLM staged runner. All simulation and Torch execution is Linux-only.

Stages have durable contracts, fresh train/test namespaces and explicit gates.
No stage may reinterpret a failed DSR gate as success on another metric.
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
import platform
import statistics
import subprocess
import sys
import time

UPPER = 'lower_mlp_upper_uncovered_obs'
METHODS = ('full', 'no_feedback', 'heuristic', 'greedy', 'original')
FILES = ('utils/deadline_repair.py', 'utils/deadline_repair_online.py', 'utils/node_offload_observation.py',
         'utils/sf_hlm_matching.py', 'utils/sf_hlm_search.py', 'utils/sf_hlm_coordination.py', 'marl_models/sf_hlm.py',
         'scripts/analysis/sf_hlm_experiment.py', 'tests/test_sf_hlm.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if hasattr(value, 'tolist'): return value.tolist()
    if hasattr(value, 'item'): return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, default=serial, indent=2) + '\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def bootstrap(args):
    if platform.system() != 'Linux': raise RuntimeError('Simulation, matching and training must run on remote Linux')
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import torch
    from utils.canonical_experiment import source_manifest
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if sha(args.suite_root / 'suite.json') != protocol['suite_sha256']: raise ValueError('Historical suite changed')
    if source_manifest()['source_hash'] != protocol['source_hash']: raise ValueError('Historical source changed')
    for rel, checksum in protocol['extension_files'].items():
        if sha(args.extension_root / rel) != checksum: raise ValueError('Frozen extension changed: ' + rel)
    for name in ('utils.deadline_repair', 'utils.deadline_repair_online', 'utils.node_offload_observation',
                 'utils.sf_hlm_matching', 'utils.sf_hlm_search', 'utils.sf_hlm_coordination', 'marl_models.sf_hlm'):
        load_module(name, args.extension_root / (name.replace('.', '/') + '.py'))
    torch.set_num_threads(1); np.random.seed(args.seed); torch.manual_seed(args.seed)
    return protocol


def configure(args, protocol, historical=True):
    import config
    from utils.logger import refresh_derived_config_fields
    from run_canonical_experiment import configure_method
    from utils.canonical_experiment import config_snapshot, digest
    suite = json.loads((args.suite_root / 'suite.json').read_text()); plan = suite['plan']
    v = plan['variants'][0]; config.NUM_UES = v['ue_count']; config.UAV_COMPUTING_CAPACITY = config.UAV_COMPUTING_CAPACITY * v['compute_scale']
    refresh_derived_config_fields()
    config.PPO_EPOCHS = plan['ppo_epochs']; config.PPO_BATCH_SIZE = plan['batch_size']; config.PPO_ROLLOUT_LENGTH = plan['rollout']
    if not historical:
        import numpy as np
        import torch
        np.random.seed(args.block); torch.manual_seed(args.block)
    models = configure_method(UPPER)
    checkpoint_hash = None
    if historical:
        from utils.formal_protocol import verify_completed_job
        base = args.suite_root / 'main' / UPPER / f'seed_{args.block}' / 'train'
        training = verify_completed_job(base, protocol['source_hash'])
        latest = json.loads((training / 'latest.json').read_text())
        if latest['completed_episodes'] != 200: raise ValueError('Historical final checkpoint missing')
        checkpoint = training / latest['checkpoint']; contract = json.loads((checkpoint / 'contract.json').read_text())
        if contract['method'] != UPPER or contract['config_hash'] != digest(config_snapshot()): raise ValueError('Historical config/method mismatch')
        for rel, checksum in contract['files'].items():
            if sha(checkpoint / rel) != checksum: raise ValueError('Historical checkpoint changed')
        for name, model in models.items(): model.load(str(checkpoint / name))
        checkpoint_hash = sha(checkpoint / 'contract.json')
    return models, dict(protocol_sha256=sha(args.output / 'protocol.json'), source_hash=protocol['source_hash'],
                        block=args.block, config_hash=digest(config_snapshot()), historical_checkpoint_sha256=checkpoint_hash)


def prepare(env):
    import config
    from utils.deadline_repair_online import prepare as proposal
    from utils.node_offload_observation import describe_candidates
    lower_obs, masks = env.get_offloading_obs_and_masks(); requests, rows = proposal(env, masks)
    descriptions = describe_candidates(env, requests, rows); ues = {u.id: u for u in env.ues}
    for request, row, description in zip(requests, rows, descriptions):
        req = ues[request[2]].current_request
        row['size'] = int(req.req_size); row['cycles'] = float(config.CPU_CYCLES_PER_BYTE[req.req_id]) * req.req_size
        for action, option in row['options'].items():
            option['cache_belief'] = description[action]['belief']; option['capacity'] = description[action]['capacity']
    return lower_obs, masks, requests, rows


def actions_from_assignment(requests, assignment, masks):
    import numpy as np
    if len(requests) != len(assignment): raise ValueError('Lost admitted requests')
    actions = np.zeros(masks.shape[:2], dtype=np.int64)
    for request, action in zip(requests, assignment):
        if not masks[request[0], request[1], action]: raise ValueError('Quality mask changed')
        actions[request[0], request[1]] = action
    return actions


def common_repair(rows, initial, budget=128, wall_seconds=None):
    from utils.sf_hlm_search import common_repair as execute
    return execute(rows, initial, budget, wall_seconds)


def original_proposal(model, lower_obs, masks, exploration=False):
    import numpy as np
    if masks.shape[1] == model.max_requests: return model.get_action_and_value(lower_obs, masks=masks, exploration=exploration)
    if exploration: raise ValueError('Changed request capacity is evaluation-only')
    request_dim = (lower_obs.shape[-1] - 5) // masks.shape[1]
    requests = lower_obs[:, 5:].reshape(len(lower_obs), masks.shape[1], request_dim)
    pieces = []
    for start in range(0, masks.shape[1], model.max_requests):
        n = min(model.max_requests, masks.shape[1] - start)
        obs = np.zeros((len(lower_obs), 5 + model.max_requests * request_dim), np.float32); obs[:, :5] = lower_obs[:, :5]
        obs[:, 5:5 + n * request_dim] = requests[:, start:start + n].reshape(len(lower_obs), -1)
        mask = np.zeros((len(lower_obs), model.max_requests, masks.shape[2]), np.float32); mask[:, :n] = masks[:, start:start + n]
        pieces.append(model.get_action_and_value(obs, masks=mask, exploration=False)[0][:, :n])
    return np.concatenate(pieces, 1), None, None


def policy_fingerprint(models):
    h = hashlib.sha256()
    for model in models:
        if model is None: continue
        for name in ('network', 'actor', 'critic', 'actor_feedback', 'critic_feedback'):
            part = getattr(model, name, None)
            if part is None: continue
            for key, value in sorted(part.state_dict().items()):
                h.update(name.encode()); h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def get_policy(env, obs, method, models, *, lower=None, upper=None, rng=None, train_upper=False, budget=16, wall_seconds=None, signaling=None):
    import numpy as np
    import config
    from utils.deadline_repair_online import greedy_assignment
    from utils.sf_hlm_search import run_search
    from utils.sf_hlm_matching import feedback
    started = time.perf_counter(); lower_obs, masks, requests, rows = prepare(env)
    control = None; round_lost = False
    if signaling is not None:
        from environment import comm_model as comms
        from utils.sf_hlm_coordination import control_cost
        rates = [comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(u.pos, config.MBS_POS)) for u in env.uavs]
        control = control_cost(rows, rates, config.UAV_BACKHAUL_TX_POWER, config.UAV_BACKHAUL_RX_POWER, additional_delay=signaling['additional_delay'])
        for row in rows:
            for option in row['options'].values(): option['base'] += control['delay_seconds']
        if rng is None: raise ValueError('Explicit packet-loss RNG required')
        round_lost = rng.random() < signaling['round_loss_probability']
    entry = None
    if round_lost:
        # Source-local single-request fallback, no global proposal coordinator.
        from utils.deadline_repair import assess
        assignment = tuple(min(row['options'], key=lambda a: (row['options'][a]['base'] + row['options'][a]['work'], a)) for row in rows)
        predicted = assess(rows, assignment)
        diagnostic = dict(chosen_predicted=predicted, reference_predicted=predicted, common_energy_cap=predicted['energy'],
                          matching_calls=0, matching_budget=0, guard_bypassed_due_packet_loss=True)
    elif method == 'original':
        raw, lp, value = original_proposal(models['lower'], lower_obs, masks, exploration=train_upper)
        initial = tuple(int(raw[r[0], r[1]]) for r in requests)
        assignment, diagnostic = common_repair(rows, initial)
        entry = (lower_obs, raw, masks, lp, value)
    elif method == 'greedy':
        initial, _ = greedy_assignment(rows); assignment, diagnostic = common_repair(rows, initial)
    else:
        kind = 'learned' if method in ('full', 'no_feedback', 'learned') else method
        assignment, diagnostic, _ = run_search(rows, num_uavs=config.NUM_UAVS, kind=kind, model=lower, rng=rng,
                                                budget=budget, wall_seconds=wall_seconds)
    summary = feedback(rows, assignment, config.NUM_UAVS, energy_reference=config.OFFLOAD_ENERGY_NORM_REF)
    if round_lost: summary.fill(0.)  # No central feedback can arrive after a failed round.
    if upper is None:
        raw_motion = models['upper'].select_actions(obs, exploration=False); motion_lp = motion_value = None
    else: raw_motion, motion_lp, motion_value = upper.act(obs, summary, exploration=train_upper)
    actions = actions_from_assignment(requests, assignment, masks)
    upload = len(json.dumps(dict(rows=rows), separators=(',', ':')).encode())
    download = len(json.dumps(dict(ue_ids=[r[2] for r in requests], actions=assignment, feedback=summary.tolist()), separators=(',', ':')).encode())
    info = dict(request_ids=[r[2] for r in requests], assignment=assignment, diagnostic=diagnostic,
                logical_upload_bytes=upload, logical_download_bytes=download, policy_seconds=time.perf_counter() - started)
    if control is not None: info.update(control=control, coordination_round_lost=round_lost)
    return np.clip(raw_motion, -1, 1), actions, info, dict(obs=obs.copy(), feedback=summary, action=raw_motion,
                                                       log_prob=motion_lp, value=motion_value), entry, rows


def assert_step(metrics, info):
    if metrics['forced_service_admissions'] or metrics['service_fallback_count'] or metrics['service_predict_exception_fallback_count']:
        raise ValueError('Forced admission/fallback violates protocol')
    if metrics['admitted_service_count'] != len(info['request_ids']) or metrics['service_offloads_mbs'] != info['assignment'].count(1):
        raise ValueError('Execution disagrees with assignments')
    if not all(math.isfinite(v) for v in metrics.values()): raise FloatingPointError('Nonfinite environment')
    d = info['diagnostic']
    if d['chosen_predicted']['energy'] > d['common_energy_cap'] + 1e-7: raise ValueError('Common predictor guard violated')
    if d['matching_calls'] > d['matching_budget']: raise ValueError('Matching budget exceeded')


def result_path(root, phase, method, block, workload):
    return root / phase / method / f'block_{block}_workload_{workload}'


def evaluation_root(args):
    return args.output / 'robustness_variants' / args.variant if args.phase == 'robustness' else args.output


def apply_physical_variant(variant):
    import config
    from utils.logger import refresh_derived_config_fields
    # Freeze the trained attention input contract across workload sizes.
    upper_capacity, upper_dim = config.MAX_ASSOCIATED_UES, config.OBS_DIM_SINGLE
    config.NUM_UES = variant['ue_count']; config.UAV_COMPUTING_CAPACITY = config.UAV_COMPUTING_CAPACITY * variant['compute_scale']
    refresh_derived_config_fields(); config.MAX_ASSOCIATED_UES = upper_capacity; config.OBS_DIM_SINGLE = upper_dim


def save_result(out, contract, summaries, **extra):
    files = {str(p.relative_to(out)): sha(p) for p in sorted((out / 'episodes').glob('*.json'))}
    result = dict(contract); result.update(status='PASS', episodes=summaries, episode_files=files); result.update(extra)
    write(out / 'result.json', result)


def verify_result(out, contract):
    doc = json.loads((out / 'result.json').read_text())
    if doc['status'] != 'PASS' or any(doc.get(k) != v for k, v in contract.items()): raise ValueError('Incompatible result')
    for rel, checksum in doc['episode_files'].items():
        if sha(out / rel) != checksum: raise ValueError('Raw episode changed')
    if len(doc['episode_files']) != len(doc['episodes']): raise ValueError('Incomplete episode manifest')
    if [e['episode'] for e in doc['episodes']] != list(range(len(doc['episodes']))): raise ValueError('Episode ordering changed')
    return doc


def snapshot_job(args, protocol, models, contract):
    import numpy as np
    import config
    from environment.env import Env
    from run_canonical_experiment import policy_step
    from utils.snapshot_evaluator import capture, evaluate
    from utils.canonical_experiment import digest, exogenous_frame
    from utils.sf_hlm_search import run_search
    from utils.deadline_repair_online import greedy_assignment
    from utils.formal_protocol import verify_completed_job
    out = result_path(evaluation_root(args), args.phase, args.method, args.block, args.workload)
    if (out / 'result.json').exists(): return verify_result(out, contract)
    env = Env(scenario_seed=args.workload); summaries = []
    lower = load_quota(args, protocol) if args.phase == 'learned_snapshots' else None
    raw_dir = verify_completed_job(args.suite_root / 'main' / UPPER / f'seed_{args.block}' / f'workload_{args.workload}', protocol['source_hash'])
    for episode in range(protocol['snapshot_episodes']):
        original = json.loads((raw_dir / 'episodes' / f'{episode:05d}.json').read_text())
        obs = np.asarray(env.reset(), dtype=np.float32); records = []; trace = hashlib.sha256()
        for step in range(protocol['steps']):
            trace.update(digest(exogenous_frame(env)).encode())
            motion, original_actions, _ = policy_step(models, env, obs, exploration=False)
            if step in protocol['snapshot_steps']:
                _, masks, requests, rows = prepare(env); snap = capture(env, np.clip(motion, -1, 1))
                if tuple(r[2] for r in requests) != tuple(r[2] for r in snap.requests): raise ValueError('Snapshot row order changed')
                rng = np.random.default_rng(args.seed + episode * 10000 + step)
                methods = ('greedy', 'fixed', 'heuristic', 'random') + (('learned',) if lower is not None else ())
                comparisons = {}
                for method in methods:
                    t = time.perf_counter()
                    if method == 'greedy':
                        initial, _ = greedy_assignment(rows); assignment, diagnostic = common_repair(rows, initial)
                    else:
                        assignment, diagnostic, _ = run_search(rows, num_uavs=config.NUM_UAVS, kind=method, rng=rng, model=lower, budget=protocol['matching_budget'])
                    proposal_seconds = time.perf_counter() - t; actual = evaluate(snap, assignment)
                    comparisons[method] = dict(assignment=assignment, diagnostic=diagnostic, proposal_seconds=proposal_seconds,
                        actual_misses=sum(not r['success'] for r in actual['requests']),
                        actual_assignment_energy=float(actual['metrics']['lower_assignment_energy']), actual_requests=actual['requests'])
                # Fair timing curves: complete proposal paths, not equal evaluator counts.
                curves = []
                for wall in protocol['wall_budgets']:
                    for kind in ('greedy', 'heuristic') + (('learned',) if lower else ()):
                        t = time.perf_counter()
                        if kind == 'greedy':
                            initial, _ = greedy_assignment(rows); a, d = common_repair(rows, initial, budget=100000, wall_seconds=wall)
                        else: a, d, _ = run_search(rows, num_uavs=config.NUM_UAVS, kind=kind, model=lower, budget=100000, wall_seconds=wall, base_repair_budget=128)
                        elapsed = time.perf_counter() - t; actual = evaluate(snap, a)
                        curves.append(dict(method=kind, wall_budget_seconds=wall, elapsed_seconds=elapsed, predicted=d['chosen_predicted'], matching_calls=d['matching_calls'],
                                           actual_misses=sum(not r['success'] for r in actual['requests']), actual_assignment_energy=float(actual['metrics']['lower_assignment_energy'])))
                records.append(dict(step=step, snapshot_hash=snap.snapshot_hash, requests=len(rows), comparisons=comparisons, wall_curves=curves))
            next_obs, _, metrics = env.step(np.clip(motion, -1, 1), offloading_actions=original_actions)
            old = original['steps'][step]['metrics']
            for name in ('generated_service_count', 'admitted_service_count', 'deadline_satisfied_count', 'lower_assignment_energy'):
                if not np.isclose(metrics[name], old[name], rtol=1e-10, atol=1e-10): raise ValueError('Historical replay differs: ' + name)
            obs = np.asarray(next_obs, dtype=np.float32)
        if not protocol['smoke'] and trace.hexdigest() != original['exogenous_trace_hash']: raise ValueError('Historical random stream changed')
        write(out / 'episodes' / f'{episode:05d}.json', dict(snapshots=records, trace=trace.hexdigest()))
        summaries.append(dict(episode=episode, snapshots=len(records), trace=trace.hexdigest()))
    save_result(out, contract, summaries, workload=args.workload)


def teacher_samples(rows, num_uavs, total_budget=128):
    """Allowed predictor only; a finite expert search, not the true simulator."""
    import numpy as np
    from utils.sf_hlm_search import QuotaSearch, graph_state
    from utils.sf_hlm_matching import solve_fixed_counts
    engine = QuotaSearch(rows, num_uavs=num_uavs, matching_budget=16)
    samples = []; calls = 0
    while not engine.done and calls + engine.calls < total_budget:
        state = graph_state(engine); choices = np.flatnonzero(state['mask'][1:]) + 1
        best = engine.current_predicted['objective'][:2]; chosen = 0
        for action in choices:
            if calls + engine.calls + 1 >= total_budget: break  # Reserve selected transition's matching call.
            i, j, amount = engine.actions[action]; q = engine.current.copy(); q[i] -= amount; q[j] += amount
            candidate = solve_fixed_counts(rows, q, num_uavs, problem=engine.problem); calls += 1
            if candidate is not None and candidate['predicted']['energy'] <= engine.cap and candidate['predicted']['objective'][:2] < best:
                chosen = int(action); best = candidate['predicted']['objective'][:2]
        samples.append(dict(state=state, action=chosen)); engine.step(chosen)
    return samples, calls + engine.calls


def dataset_job(args, protocol, models, contract):
    import numpy as np
    import torch
    import config
    from environment.env import Env
    out = result_path(evaluation_root(args), args.phase, args.method, args.block, args.workload)
    if (out / 'result.json').exists():
        doc = verify_result(out, contract)
        if sha(out / 'dataset.pt') != doc['dataset_sha256']: raise ValueError('Training dataset changed')
        return
    env = Env(scenario_seed=args.workload); rows_list = []; teacher = []; calls = 0
    for episode in range(protocol['dataset_episodes']):
        obs = np.asarray(env.reset(), dtype=np.float32)
        for step in range(protocol['steps']):
            motion, action, info, _, _, rows = get_policy(env, obs, 'greedy', models)
            if step in protocol['dataset_steps']:
                sample, evaluations = teacher_samples(rows, config.NUM_UAVS, protocol['teacher_budget'])
                rows_list.append(rows); teacher.extend(sample); calls += evaluations
            obs, _, metrics = env.step(motion, offloading_actions=action); assert_step(metrics, info); obs = np.asarray(obs, dtype=np.float32)
    out.mkdir(parents=True, exist_ok=True); torch.save(dict(rows=rows_list, teacher=teacher), out / 'dataset.pt')
    save_result(out, contract, [], dataset_sha256=sha(out / 'dataset.pt'), graphs=len(rows_list), teacher_samples=len(teacher), teacher_matching_calls=calls,
                test_data_used=False, training_workload=args.workload)


def quota_contract(args, protocol):
    return dict(protocol_sha256=sha(args.output / 'protocol.json'), block=args.block,
                lower_seed=protocol['lower_seeds'][str(args.block)], data_split='training_only')


def quota_path(args): return args.output / 'lower_training' / f'block_{args.block}'


def load_quota(args, protocol):
    from marl_models.sf_hlm import QuotaPPO
    import config
    path = quota_path(args); doc = verify_result(path, quota_contract(args, protocol))
    if sha(path / 'checkpoint/quota_ppo.pt') != doc['checkpoint_sha256']: raise ValueError('Lower checkpoint checksum changed')
    model = QuotaPPO(config.NUM_UAVS); model.load(path / 'checkpoint', quota_contract(args, protocol)); return model


def lower_job(args, protocol):
    import numpy as np
    import torch
    import config
    from marl_models.sf_hlm import QuotaPPO
    from utils.sf_hlm_search import run_search
    contract = quota_contract(args, protocol); out = quota_path(args)
    if (out / 'result.json').exists():
        doc = verify_result(out, contract)
        if sha(out / 'checkpoint/quota_ppo.pt') != doc['checkpoint_sha256']: raise ValueError('Lower checkpoint changed')
        return
    data_path = result_path(args.output, 'dataset', 'greedy', args.block, protocol['training_workloads'][str(args.block)])
    data_doc = verify_result(data_path, dict(protocol_sha256=contract['protocol_sha256'], phase='dataset', method='greedy',
                                           block=args.block, workload=protocol['training_workloads'][str(args.block)]))
    if sha(data_path / 'dataset.pt') != data_doc['dataset_sha256'] or data_doc['protocol_sha256'] != contract['protocol_sha256']:
        raise ValueError('Dataset provenance changed')
    data = torch.load(data_path / 'dataset.pt', weights_only=False)
    torch.manual_seed(protocol['lower_seeds'][str(args.block)]); model = QuotaPPO(config.NUM_UAVS)
    rng = np.random.default_rng(protocol['lower_seeds'][str(args.block)])
    start = 0; losses = []
    if (out / 'latest.json').exists():
        latest = json.loads((out / 'latest.json').read_text()); recovery = out / latest['checkpoint']
        verify_system_checkpoint(recovery, contract); model.load(recovery, contract)
        state = torch.load(recovery / 'state.pt', weights_only=False); start = state['next_iteration']; losses = state['losses']
        torch.set_rng_state(state['torch_rng']); rng.bit_generator.state = state['search_rng']
    elif (out / 'progress.json').exists(): raise ValueError('Partial lower has no valid recovery contract')
    else: losses = [model.pretrain(data['teacher'], epochs=protocol['supervised_epochs'])]
    for iteration in range(start, protocol['lower_iterations']):
        records = []
        for i in rng.integers(0, len(data['rows']), size=protocol['graphs_per_update']):
            _, _, sampled = run_search(data['rows'][int(i)], num_uavs=config.NUM_UAVS, kind='learned', model=model, exploration=True, budget=protocol['matching_budget'])
            records.extend(sampled)
        loss = model.update(records, epochs=protocol['ppo_epochs']); losses.append(loss)
        recovery = out / 'recovery' / f'iteration_{iteration:05d}'; model.save(recovery, contract)
        torch.save(dict(next_iteration=iteration + 1, losses=losses, torch_rng=torch.get_rng_state(), search_rng=rng.bit_generator.state), recovery / 'state.pt')
        write(recovery / 'contract.json', dict(system_contract=contract, files={p.name: sha(p) for p in recovery.iterdir() if p.is_file() and p.name != 'contract.json'}))
        write(out / 'latest.json', dict(checkpoint=str(recovery.relative_to(out)), iterations=iteration + 1))
        write(out / 'progress.json', dict(iteration=iteration + 1, iterations=protocol['lower_iterations'], losses=loss, pid=os.getpid()))
    model.save(out / 'checkpoint', contract)
    save_result(out, contract, [], checkpoint_sha256=sha(out / 'checkpoint/quota_ppo.pt'), losses=losses, dataset_sha256=data_doc['dataset_sha256'])


def evaluate_job(args, protocol, models, contract):
    import numpy as np
    import config
    from environment.env import Env
    from utils.canonical_experiment import exogenous_frame, digest
    from marl_models.sf_hlm import FeedbackMAPPO
    out = result_path(evaluation_root(args), args.phase, args.method, args.block, args.workload)
    if (out / 'result.json').exists(): return verify_result(out, contract)
    lower = load_quota(args, protocol) if args.method in ('learned', 'full', 'no_feedback') else None
    upper = None
    if args.phase in ('confirmation_eval', 'robustness'):
        train_path = result_path(args.output, 'system_training', args.method, args.block, protocol['training_workloads'][str(args.block)])
        doc = json.loads((train_path / 'result.json').read_text())
        checkpoint = train_path / 'checkpoint'; verify_system_checkpoint(checkpoint, doc['system_contract'])
        upper = FeedbackMAPPO(config.OBS_DIM_SINGLE, config.NUM_UAVS, enabled=args.method in ('full', 'heuristic'))
        upper.load(checkpoint, doc['system_contract'])
        if lower is not None: lower.load(checkpoint, doc['system_contract'])
        if args.method == 'original': models['lower'].load(str(checkpoint / 'original_lower'))
    signaling = None
    if args.phase == 'robustness':
        from utils.canonical_experiment import config_snapshot, digest
        variant = protocol['robustness_variants'][args.variant]
        apply_physical_variant(variant); signaling = variant.get('signaling')
        contract.update(variant=args.variant, evaluation_config_hash=digest(config_snapshot()), signaling=signaling)
    env = Env(scenario_seed=args.workload); summaries = []; rng = np.random.default_rng(args.seed + args.workload)
    frozen_hash = policy_fingerprint([models['upper'], models['lower'], lower, upper])
    eval_episodes = protocol['confirmation_eval_episodes'] if args.phase in ('confirmation_eval', 'robustness') else protocol['eval_episodes']
    for episode in range(eval_episodes):
        obs = np.asarray(env.reset(), dtype=np.float32); trace = hashlib.sha256(); actions_hash = hashlib.sha256()
        totals = {k: 0. for k in ('generated', 'admitted', 'success', 'potential', 'energy', 'assignment_energy', 'mbs', 'fairness')}
        times = []; records = []
        for step in range(protocol['steps']):
            trace.update(digest(exogenous_frame(env)).encode()); potential = sum(ue.latent_request[0] < config.NUM_SERVICES for ue in env.ues)
            motion, action, info, _, _, _ = get_policy(env, obs, args.method, models, lower=lower, upper=upper, rng=rng, budget=protocol['matching_budget'], signaling=signaling)
            if signaling is None: obs, _, metrics = env.step(motion, offloading_actions=action)
            else:
                from utils.sf_hlm_coordination import execute_controlled_slot
                obs, _, metrics = execute_controlled_slot(env, motion, action, info['control'])
            assert_step(metrics, info)
            if not np.isfinite(obs).all(): raise FloatingPointError('Nonfinite next observation')
            obs = np.asarray(obs, dtype=np.float32); times.append(info['policy_seconds']); actions_hash.update(digest(info['assignment']).encode())
            for key, name in [('generated', 'generated_service_count'), ('admitted', 'admitted_service_count'), ('success', 'deadline_satisfied_count'),
                              ('energy', 'energy'), ('assignment_energy', 'lower_assignment_energy'), ('mbs', 'service_offloads_mbs'), ('fairness', 'fairness')]: totals[key] += float(metrics[name])
            totals['potential'] += potential
            records.append(dict(step=step, metrics=metrics, potential=potential, **info))
        summary = dict(episode=episode, **totals, exogenous_trace_hash=trace.hexdigest(), action_trace_hash=actions_hash.hexdigest(),
                       dsr=totals['success'] / max(totals['generated'], 1), p95_seconds=float(np.quantile(times, .95)))
        write(out / 'episodes' / f'{episode:05d}.json', dict(summary=summary, steps=records)); summaries.append(summary)
    if policy_fingerprint([models['upper'], models['lower'], lower, upper]) != frozen_hash: raise ValueError('Evaluation mutated frozen policies')
    save_result(out, contract, summaries, workload=args.workload, method=args.method, policy_fingerprint=frozen_hash,
                lower_policy_sha256=sha(quota_path(args) / 'checkpoint/quota_ppo.pt') if lower else None)


def system_contract(args, protocol):
    return dict(protocol_sha256=sha(args.output / 'protocol.json'), method=args.method, seed=args.block,
                algorithm_id=protocol['method_identifiers'][args.method],
                episodes=protocol['system_episodes'], feedback=args.method in ('full', 'heuristic'), action_contract='raw upper Gaussian; sampled lower quota transfer')


def verify_system_checkpoint(path, contract):
    doc = json.loads((path / 'contract.json').read_text())
    if doc['system_contract'] != contract: raise ValueError('System checkpoint incompatible')
    for rel, checksum in doc['files'].items():
        if sha(path / rel) != checksum: raise ValueError('System checkpoint changed')


def system_job(args, protocol, models, contract):
    import numpy as np
    import torch
    import config
    from environment.env import Env
    from marl_models.sf_hlm import FeedbackMAPPO
    from utils.sf_hlm_search import run_search
    from marl_models.buffer_and_helpers import DiscreteOffloadRolloutBuffer
    from run_hierarchical_mappo_experiment import lower_rewards_from_metrics, update_on_policy_model
    out = result_path(args.output, args.phase, args.method, args.block, args.workload)
    if (out / 'result.json').exists(): return verify_result(out, contract)
    upper = FeedbackMAPPO(config.OBS_DIM_SINGLE, config.NUM_UAVS, enabled=args.method in ('full', 'heuristic'))
    lower = load_quota(args, protocol) if args.method in ('full', 'no_feedback') else None
    sys_contract = system_contract(args, protocol); env = Env(scenario_seed=args.workload); summaries = []; graphs = []
    rho_dsr = rho_mbs = 0.; start = 0
    if (out / 'latest.json').exists():
        latest = json.loads((out / 'latest.json').read_text()); recovery = out / latest['checkpoint']
        verify_system_checkpoint(recovery, sys_contract); upper.load(recovery, sys_contract)
        if lower: lower.load(recovery, sys_contract)
        if args.method == 'original': models['lower'].load(str(recovery / 'original_lower'))
        state = torch.load(recovery / 'state.pt', weights_only=False)
        env = state['env']; start = state['next_episode']; summaries = state['summaries']; graphs = state['graphs']
        rho_dsr, rho_mbs = state['rho']; torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
    elif (out / 'progress.json').exists(): raise ValueError('Partial job has no valid recovery contract')
    for episode in range(start, protocol['system_episodes']):
        obs = np.asarray(env.reset(), dtype=np.float32); rollout = []; totals = {k: 0. for k in ('generated', 'admitted', 'success', 'energy')}; violations = []
        action_trace = hashlib.sha256()
        original_buffer = DiscreteOffloadRolloutBuffer(config.NUM_UAVS, config.OFFLOAD_OBS_DIM_SINGLE, config.MAX_OFFLOAD_REQUESTS_PER_UAV,
                                    config.OFFLOAD_NUM_ACTIONS, protocol['steps'], 'cpu') if args.method == 'original' else None
        for step in range(protocol['steps']):
            motion, action, info, entry, lower_entry, rows = get_policy(env, obs, args.method, models, lower=lower, upper=upper, train_upper=True, budget=protocol['matching_budget'])
            next_obs, rewards, metrics = env.step(motion, offloading_actions=action); assert_step(metrics, info)
            action_trace.update(motion.tobytes()); action_trace.update(action.tobytes())
            entry.update(reward=np.asarray(rewards, dtype=np.float32), done=step + 1 == protocol['steps']); rollout.append(entry)
            if lower is not None and step in protocol['dataset_steps']: graphs.append(rows)
            if original_buffer is not None:
                lower_rewards, v = lower_rewards_from_metrics(metrics, rewards, rho_dsr, rho_mbs); violations.append(v)
                original_buffer.add(*lower_entry[:4], lower_rewards, entry['done'], lower_entry[4])
            for k, name in [('generated', 'generated_service_count'), ('admitted', 'admitted_service_count'), ('success', 'deadline_satisfied_count'), ('energy', 'energy')]: totals[k] += float(metrics[name])
            obs = np.asarray(next_obs, dtype=np.float32)
        upper_loss = upper.update(rollout, epochs=protocol['ppo_epochs']); lower_loss = {}
        if original_buffer is not None:
            lower_loss = update_on_policy_model(models['lower'], original_buffer, np.zeros(config.NUM_UAVS, np.float32))
            rho_dsr = float(np.clip(rho_dsr + config.OFFLOAD_LAGRANGE_LR * np.mean([v['dsr_violation'] for v in violations]), 0, config.OFFLOAD_LAGRANGE_MAX))
            rho_mbs = float(np.clip(rho_mbs + config.OFFLOAD_LAGRANGE_LR * np.mean([v['mbs_load_violation'] for v in violations]), 0, config.OFFLOAD_LAGRANGE_MAX))
        if lower is not None and episode + 1 > protocol['upper_warmup_episodes'] and (episode + 1) % 5 == 0:
            samples = []
            for rows in graphs:
                _, _, sampled = run_search(rows, num_uavs=config.NUM_UAVS, kind='learned', model=lower, exploration=True, budget=protocol['matching_budget'])
                samples.extend(sampled)
            lower_loss = lower.update(samples, epochs=protocol['ppo_epochs']); graphs = []
        if episode + 1 <= protocol['upper_warmup_episodes']: graphs = []
        summary = dict(episode=episode, **totals, upper_loss=upper_loss, lower_loss=lower_loss, action_trace_hash=action_trace.hexdigest())
        write(out / 'episodes' / f'{episode:05d}.json', dict(summary=summary)); summaries.append(summary)
        checkpoint = out / 'recovery' / f'episode_{episode:05d}'; upper.save(checkpoint, sys_contract)
        if lower: lower.save(checkpoint, sys_contract)
        if original_buffer is not None:
            (checkpoint / 'original_lower').mkdir(parents=True, exist_ok=True); models['lower'].save(str(checkpoint / 'original_lower'))
        torch.save(dict(env=env, next_episode=episode + 1, summaries=summaries, graphs=graphs, rho=(rho_dsr, rho_mbs),
                        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state()), checkpoint / 'state.pt')
        write(checkpoint / 'contract.json', dict(system_contract=sys_contract, files={str(p.relative_to(checkpoint)): sha(p) for p in checkpoint.rglob('*') if p.is_file() and p.name != 'contract.json'}))
        write(out / 'latest.json', dict(checkpoint=str(checkpoint.relative_to(out)), episodes=episode + 1))
        write(out / 'progress.json', dict(episode=episode + 1, episodes=protocol['system_episodes'], pid=os.getpid()))
    checkpoint = out / 'checkpoint'; upper.save(checkpoint, sys_contract)
    if lower: lower.save(checkpoint, sys_contract)
    if args.method == 'original':
        (checkpoint / 'original_lower').mkdir(parents=True, exist_ok=True); models['lower'].save(str(checkpoint / 'original_lower'))
    write(checkpoint / 'contract.json', dict(system_contract=sys_contract, files={str(p.relative_to(checkpoint)): sha(p) for p in checkpoint.rglob('*') if p.is_file() and p.name != 'contract.json'}))
    save_result(out, contract, summaries, system_contract=sys_contract)


def worker(args):
    protocol = bootstrap(args)
    if [args.phase, args.method, args.block, args.workload] not in protocol['jobs']: raise ValueError('Undeclared job')
    if args.phase == 'robustness' and args.variant not in protocol['robustness_variants']: raise ValueError('Undeclared physical variant')
    if args.phase != 'robustness' and args.variant is not None: raise ValueError('Variant applies only to robustness')
    historical = args.block in protocol['historical_blocks']
    models, contract = configure(args, protocol, historical=historical)
    contract.update(phase=args.phase, method=args.method, workload=args.workload)
    if args.phase in ('snapshots', 'learned_snapshots'): snapshot_job(args, protocol, models, contract)
    elif args.phase == 'dataset': dataset_job(args, protocol, models, contract)
    elif args.phase == 'lower_training': lower_job(args, protocol)
    elif args.phase == 'system_training': system_job(args, protocol, models, contract)
    else: evaluate_job(args, protocol, models, contract)
    print(json.dumps(dict(status='PASS', phase=args.phase, method=args.method, block=args.block, workload=args.workload)), flush=True)


def make_protocol(args):
    suite = json.loads((args.suite_root / 'suite.json').read_text()); plan = suite['plan']; smoke = args.smoke
    historical = plan['training_seeds'][:1] if smoke else plan['training_seeds']
    snapshots = plan['workload_seeds'][:1] if smoke else plan['workload_seeds']
    eval_workloads = [780001] if smoke else list(range(780001, 780006))
    confirmation = [700001] if smoke else list(range(700001, 700011))
    confirm_workloads = [790001] if smoke else list(range(790001, 790006))
    robust_workloads = [791001] if smoke else list(range(791001, 791006))
    variants = {f'ue_{n}_compute_{int(scale * 100)}': dict(ue_count=n, compute_scale=scale)
                for n in (60, 100, 140) for scale in (.6, 1., 1.4)}
    variants.update({f'charged_delay_{int(delay * 1000)}ms_loss_{int(loss * 100)}':
                     dict(ue_count=100, compute_scale=1., signaling=dict(additional_delay=delay, round_loss_probability=loss))
                     for delay, loss in ((.02, 0.), (.1, 0.), (.02, .1))})
    if smoke: variants = {k: variants[k] for k in ('ue_60_compute_60', 'ue_140_compute_140', 'charged_delay_20ms_loss_10')}
    all_blocks = historical + confirmation
    runtime = dict(python=sys.version, platform=platform.system())
    if platform.system() == 'Linux':
        import numpy as np
        import scipy
        import torch
        runtime.update(numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__)
    lower_seeds = {str(b): 690001 + i for i, b in enumerate(all_blocks)}
    train = {str(b): 890001 + i if b in historical else 900001 + i for i, b in enumerate(all_blocks)}
    jobs = []
    for b in historical:
        jobs += [['snapshots', 'all', b, w] for w in snapshots]
        jobs += [['screen_eval', k, b, w] for k in ('greedy', 'fixed', 'heuristic', 'random') for w in eval_workloads]
        jobs += [['dataset', 'greedy', b, train[str(b)]], ['lower_training', 'learned', b, 0]]
        jobs += [['learned_snapshots', 'all', b, w] for w in snapshots]
        jobs += [['learned_eval', k, b, w] for k in ('greedy', 'heuristic', 'random', 'learned') for w in eval_workloads]
    for b in confirmation:
        jobs += [['dataset', 'greedy', b, train[str(b)]], ['lower_training', 'learned', b, 0]]
        jobs += [['system_training', k, b, train[str(b)]] for k in METHODS]
        jobs += [['confirmation_eval', k, b, w] for k in METHODS for w in confirm_workloads]
        jobs += [['robustness', k, b, w] for k in ('full', 'greedy', 'original') for w in robust_workloads]
    return dict(version='sf_hlm_staged_v1', purpose='verification' if smoke else 'development_then_independent_confirmation', smoke=smoke,
        source_hash=suite['source_hash'], suite_sha256=sha(args.suite_root / 'suite.json'), extension_files={f: sha(args.extension_root / f) for f in FILES}, runtime=runtime,
        method_identifiers=dict(full='sf_hlm', no_feedback='sf_hlm_no_feedback', heuristic='sf_hlm_heuristic', greedy='attention_greedy_common_repair', original='attention_retrained_mlp_common_repair'),
        historical_blocks=historical, snapshot_workloads=snapshots, eval_workloads=eval_workloads, confirmation_blocks=confirmation,
        confirmation_workloads=confirm_workloads, robustness_workloads=robust_workloads, robustness_variants=variants,
        robustness_policy='Frozen policies, upper observation capacity stays 30; original fixed-size MLP packs or chunks requests for changed UE count; no retraining.',
        signaling_protocol='Separate serial symmetric MBS control channel; fixed binary message sizes; source-local fallback and zero feedback on round loss; predicted guard waived only on loss.',
        lower_seeds=lower_seeds, training_workloads=train, jobs=jobs,
        steps=4 if smoke else 1000, snapshot_steps=[0] if smoke else [0, 250, 500, 750], snapshot_episodes=1 if smoke else 2,
        eval_episodes=1 if smoke else 2, confirmation_eval_episodes=1 if smoke else 6, dataset_episodes=1 if smoke else 25,
        dataset_steps=[0, 2] if smoke else list(range(0, 1000, 125)), teacher_budget=4 if smoke else 128,
        supervised_epochs=1 if smoke else 5, lower_iterations=3 if smoke else 100, graphs_per_update=2 if smoke else 32,
        system_episodes=6 if smoke else 200, upper_warmup_episodes=1 if smoke else 50,
        matching_budget=3 if smoke else 16, wall_budgets=[.001] if smoke else [.02, .05, .1], ppo_epochs=1 if smoke else 8,
        upper_lower_alternation_episodes=5, shared_greedy_energy_ratio=1.05, workers=args.workers, threads=1,
        gate=dict(confirmation_dsr_gain=.01, development_dsr_gain=0., energy_ratio=1.05, policy_p95_seconds=1.),
        primary_comparisons=['full-greedy', 'full-original'], checkpoint_selection='last prespecified checkpoint only',
        coordination='ideal lossless uncharged central rows; physical signaling extension is separate',
        restrictions=['No future requests, postexecution truth or neighbor cache truth in online selection.',
          'Fixed-count optimum excludes the energy side constraint; filtering is not a constrained optimum.',
          'Matching calls are not comparable to individual assessment calls; report wall-clock curves.',
          'Historical blocks are exploratory; independent confirmation uses new training/workload seeds.',
          'No manuscript performance claim is updated before audited independent confirmation.'])


def paired(values):
    from scipy.stats import t, ttest_1samp
    n = len(values); mean = float(np.mean(values))
    if n < 2: return dict(mean=mean, ci95=None, p=None, values=values, n=n)
    se = statistics.stdev(values) / math.sqrt(n); half = float(t.ppf(.975, n - 1)) * se
    p = float(ttest_1samp(values, 0).pvalue) if se > 0 else (1. if mean == 0 else 0.)
    return dict(mean=mean, ci95=[mean - half, mean + half], p=p, values=values, n=n)


def analyse(args, phase, protocol):
    # Import numerical/statistical dependencies only on the remote worker host.
    global np
    import numpy as np
    root = evaluation_root(args) if phase == 'robustness' else args.output
    groups = {}; traces = {}; files = {}; snapshots = {}; nblocks = protocol['confirmation_blocks'] if phase in ('confirmation_eval', 'robustness') else protocol['historical_blocks']
    jobs = [j for j in protocol['jobs'] if j[0] == phase]
    for _, method, block, workload in jobs:
        out = result_path(root, phase, method, block, workload); doc = json.loads((out / 'result.json').read_text())
        verify_result(out, dict(protocol_sha256=sha(args.output / 'protocol.json'), phase=phase, method=method, workload=workload))
        files[str(out / 'result.json')] = sha(out / 'result.json')
        expected = protocol['snapshot_episodes'] if phase in ('snapshots', 'learned_snapshots') else protocol['confirmation_eval_episodes'] if phase in ('confirmation_eval', 'robustness') else protocol['eval_episodes']
        if len(doc['episodes']) != expected: raise ValueError('Missing planned evaluation episodes')
        for rel, h in doc['episode_files'].items(): files[str(out / rel)] = h
        for episode in doc['episodes']:
            raw = json.loads((out / 'episodes' / f"{episode['episode']:05d}.json").read_text())
            if 'snapshots' in raw:
                for snap in raw['snapshots']:
                    for k, item in snap['comparisons'].items(): snapshots.setdefault((k, block), []).append(item['actual_misses'] - snap['comparisons']['greedy']['actual_misses'])
                continue
            key = workload, episode['episode']; fingerprint = (episode['exogenous_trace_hash'], episode['potential'])
            if key in traces and traces[key] != fingerprint: raise ValueError('External workloads not paired')
            traces[key] = fingerprint
            unit = groups.setdefault((method, block), {k: 0. for k in ('generated', 'admitted', 'success', 'potential', 'energy', 'mbs', 'fairness', 'steps',
                'miss_error', 'miss_absolute_error', 'energy_error', 'latency_error', 'matching_calls', 'wire_bytes', 'coordination_energy', 'lost_rounds')}); unit.setdefault('times', [])
            for k in ('generated', 'admitted', 'success', 'potential', 'energy', 'mbs', 'fairness'): unit[k] += episode[k]
            # Recompute counters from raw steps, not trusted summary alone.
            for key2, metric in [('generated', 'generated_service_count'), ('admitted', 'admitted_service_count'), ('success', 'deadline_satisfied_count'), ('energy', 'energy')]:
                if not np.isclose(episode[key2], sum(s['metrics'][metric] for s in raw['steps']), rtol=1e-10): raise ValueError('Raw summary mismatch')
            unit['steps'] += len(raw['steps']); unit['times'].extend(s['policy_seconds'] for s in raw['steps'])
            for s in raw['steps']:
                predicted = s['diagnostic']['chosen_predicted']; actual = s['metrics']
                error = predicted['misses'] - (actual['admitted_service_count'] - actual['deadline_satisfied_count'])
                unit['miss_error'] += error; unit['miss_absolute_error'] += abs(error)
                unit['energy_error'] += predicted['energy'] - actual['lower_assignment_energy']
                unit['latency_error'] += sum(predicted['latencies']) - actual['lower_service_latency']
                unit['matching_calls'] += s['diagnostic']['matching_calls']
                unit['wire_bytes'] += s['logical_upload_bytes'] + s['logical_download_bytes']
                unit['coordination_energy'] += actual.get('coordination_uav_energy', 0.)
                unit['lost_rounds'] += bool(s.get('coordination_round_lost', False))
    if snapshots:
        comparisons = {k: paired([statistics.mean(snapshots[k, b]) for b in nblocks]) for k in sorted({k for k, b in snapshots})}
        result = dict(status='PASS', phase=phase, snapshot_miss_differences=comparisons, raw_files=files,
                      actual_snapshot_benefit=any(v['mean'] < 0 for k, v in comparisons.items() if k in ('fixed', 'heuristic')))
    else:
        metrics = {}
        for (k, b), u in groups.items(): metrics[k, b] = dict(dsr=u['success'] / max(u['generated'], 1), potential_dsr=u['success'] / max(u['potential'], 1),
            admission=u['admitted'] / max(u['generated'], 1), conditional=u['success'] / max(u['admitted'], 1), energy=u['energy'],
            fairness=u['fairness'] / max(u['steps'], 1), mbs_share=u['mbs'] / max(u['admitted'], 1), p95=float(np.quantile(u['times'], .95)),
            predicted_miss_bias_per_slot=u['miss_error'] / max(u['steps'], 1), predicted_miss_mae_per_slot=u['miss_absolute_error'] / max(u['steps'], 1),
            predicted_assignment_energy_bias_per_slot=u['energy_error'] / max(u['steps'], 1), predicted_latency_bias_per_request=u['latency_error'] / max(u['admitted'], 1),
            matching_calls_per_slot=u['matching_calls'] / max(u['steps'], 1), logical_bytes_per_slot=u['wire_bytes'] / max(u['steps'], 1),
            coordination_energy=u['coordination_energy'], lost_round_fraction=u['lost_rounds'] / max(u['steps'], 1))
        names = sorted({k for k, _ in groups}); summaries = {k: {m: float(np.mean([metrics[k, b][m] for b in nblocks])) for m in metrics[k, nblocks[0]]} for k in names}
        comparisons = {}; candidates = [k for k in names if k != 'greedy']
        for k in candidates: comparisons[k + '-greedy'] = paired([metrics[k, b]['dsr'] - metrics['greedy', b]['dsr'] for b in nblocks])
        if phase == 'learned_eval':
            for k in ('heuristic', 'random'): comparisons['learned-' + k] = paired([metrics['learned', b]['dsr'] - metrics[k, b]['dsr'] for b in nblocks])
        gates = {}
        for k in candidates:
            diff = comparisons[k + '-greedy']; er = float(np.mean([metrics[k, b]['energy'] / metrics['greedy', b]['energy'] for b in nblocks]))
            gates[k] = dict(pass_gate=diff['mean'] > 0 and er <= 1.05 and max(metrics[k, b]['p95'] for b in nblocks) < 1.,
                           positive_blocks=sum(x > 0 for x in diff['values']), energy_ratio=er)
        if phase == 'confirmation_eval':
            comparisons['full-original'] = paired([metrics['full', b]['dsr'] - metrics['original', b]['dsr'] for b in nblocks])
            primary = [comparisons[k] for k in protocol['primary_comparisons']]
            ordered = sorted(range(len(primary)), key=lambda i: primary[i]['p']); running = 0.
            for rank, i in enumerate(ordered): running = max(running, min(1., primary[i]['p'] * (len(primary) - rank))); primary[i]['holm_p'] = running
            er = np.mean([metrics['full', b]['energy'] / metrics['original', b]['energy'] for b in nblocks])
            gates['full']['independently_confirmed'] = gates['full']['pass_gate'] and er <= 1.05 and all(c['mean'] >= .01 and c['ci95'][0] > 0 and c['holm_p'] < .05 for c in primary)
            for k in ('no_feedback', 'heuristic'): comparisons['full-' + k] = paired([metrics['full', b]['dsr'] - metrics[k, b]['dsr'] for b in nblocks])
        result = dict(status='PASS', phase=phase, groups=summaries, comparisons=comparisons, gates=gates, raw_files=files,
                      independent_blocks=len(nblocks), exogenous_pairing=True, confirmation=phase == 'confirmation_eval')
    if phase == 'robustness': result.update(variant=args.variant, frozen_policy=True, separate_physical_protocol='signaling' in protocol['robustness_variants'][args.variant])
    write(root / 'analysis' / f'{phase}.json', result); return result


def run_jobs(args, protocol, phases, blocks=None, variant=None):
    selected = [j for j in protocol['jobs'] if j[0] in phases and (blocks is None or j[2] in blocks)]
    completed = []; failures = []
    def one(job):
        phase, method, block, workload = job
        cmd = [sys.executable, str(args.extension_root / 'scripts/analysis/sf_hlm_experiment.py'), 'worker',
               '--source-root', str(args.source_root), '--suite-root', str(args.suite_root), '--extension-root', str(args.extension_root),
               '--output', str(args.output), '--phase', phase, '--method', method, '--block', str(block), '--workload', str(workload), '--seed', str(protocol['lower_seeds'][str(block)])]
        if variant is not None: cmd += ['--variant', variant]
        log = args.output / 'job_logs' / f'{phase}_{method}_{block}_{workload}_{variant or "main"}.log'; log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT, cwd=args.source_root,
                           env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1'))
        return job
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, j): j for j in selected}
        for f in as_completed(futures):
            try: completed.append(f.result())
            except Exception as exc: failures.append(dict(job=futures[f], error=str(exc)))
            write(args.output / 'status.json', dict(phase=','.join(phases), variant=variant, pid=os.getpid(), completed=len(completed), expected=len(selected), failures=failures, updated_unix=time.time()))
    if failures: raise RuntimeError('Jobs failed; inspect retained job logs')


def pipeline(args):
    if platform.system() != 'Linux': raise RuntimeError('Pipeline is remote Linux only')
    protocol = make_protocol(args); path = args.output / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol: raise ValueError('Protocol differs; frozen run cannot be overwritten')
    else: write(path, protocol)
    sys.path.insert(0, str(args.source_root))
    if args.smoke:
        # Exercise each branch regardless of statistical gates; cannot promote.
        for phases, blocks in [(['snapshots'], protocol['historical_blocks']), (['dataset'], protocol['historical_blocks']),
                               (['lower_training'], protocol['historical_blocks']), (['screen_eval'], protocol['historical_blocks']),
                               (['learned_snapshots'], protocol['historical_blocks']), (['learned_eval'], protocol['historical_blocks']),
                               (['dataset'], protocol['confirmation_blocks']), (['lower_training'], protocol['confirmation_blocks']),
                               (['system_training'], protocol['confirmation_blocks']), (['confirmation_eval'], protocol['confirmation_blocks'])]: run_jobs(args, protocol, phases, blocks)
        for variant in protocol['robustness_variants']:
            args.variant = variant; args.phase = 'robustness'; run_jobs(args, protocol, ['robustness'], variant=variant)
            analyse(args, 'robustness', protocol)
        write(args.output / 'status.json', dict(phase='SMOKE_COMPLETE', purpose='verification_only', independent_confirmation=False)); return
    run_jobs(args, protocol, ['snapshots']); snap = analyse(args, 'snapshots', protocol)
    run_jobs(args, protocol, ['screen_eval']); screen = analyse(args, 'screen_eval', protocol)
    if not any(snap['snapshot_miss_differences'][k]['mean'] < 0 and screen['comparisons'][k + '-greedy']['mean'] > 0 for k in ('fixed', 'heuristic')):
        write(args.output / 'status.json', dict(phase='STOPPED_STRUCTURE_GATE', completed=True, passed=False,
              reason='No matching benefit in actual snapshots and frozen-upper closed loop', upper_training_started=False)); return
    run_jobs(args, protocol, ['dataset'], protocol['historical_blocks']); run_jobs(args, protocol, ['lower_training'], protocol['historical_blocks'])
    run_jobs(args, protocol, ['learned_snapshots']); learned_snap = analyse(args, 'learned_snapshots', protocol)
    run_jobs(args, protocol, ['learned_eval']); learned = analyse(args, 'learned_eval', protocol)
    if learned_snap['snapshot_miss_differences']['learned']['mean'] >= 0 or not learned['gates']['learned']['pass_gate']:
        write(args.output / 'status.json', dict(phase='STOPPED_LEARNED_GATE', completed=True, passed=False,
              reason='Learned lower failed actual snapshot or DSR/energy/time gate', upper_training_started=False)); return
    run_jobs(args, protocol, ['dataset'], protocol['confirmation_blocks']); run_jobs(args, protocol, ['lower_training'], protocol['confirmation_blocks'])
    run_jobs(args, protocol, ['system_training']); run_jobs(args, protocol, ['confirmation_eval']); confirmation = analyse(args, 'confirmation_eval', protocol)
    passed = confirmation['gates']['full']['independently_confirmed']
    if passed:
        for variant in protocol['robustness_variants']:
            args.variant = variant; args.phase = 'robustness'; run_jobs(args, protocol, ['robustness'], variant=variant)
            analyse(args, 'robustness', protocol)
    write(args.output / 'status.json', dict(phase='ROBUSTNESS_COMPLETE' if passed else 'STOPPED_CONFIRMATION_GATE', completed=True,
          passed=passed, robustness_completed=passed, independent_confirmation=True))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['protocol', 'pipeline', 'worker', 'analyse'])
    for key in ('source-root', 'suite-root', 'extension-root', 'output'): p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--workers', type=int, default=5); p.add_argument('--smoke', action='store_true')
    p.add_argument('--phase'); p.add_argument('--method'); p.add_argument('--block', type=int); p.add_argument('--workload', type=int); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--variant')
    args = p.parse_args()
    for key in ('source_root', 'suite_root', 'extension_root', 'output'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.workers < 1: raise ValueError('Workers must be positive')
    if args.mode == 'protocol': print(json.dumps(make_protocol(args), ensure_ascii=False))
    elif args.mode == 'pipeline': pipeline(args)
    elif args.mode == 'worker': worker(args)
    else:
        if platform.system() != 'Linux': raise RuntimeError('Statistics are remote-only')
        sys.path.insert(0, str(args.source_root)); analyse(args, args.phase, json.loads((args.output / 'protocol.json').read_text()))


if __name__ == '__main__': main()
