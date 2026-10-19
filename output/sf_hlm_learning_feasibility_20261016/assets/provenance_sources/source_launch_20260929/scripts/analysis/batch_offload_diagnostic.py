"""Replay frozen historical source without weakening its checkpoint contract.

Run as a script with --source-root pointing at the ORIGINAL immutable source.
The diagnostic script lives outside that source and records its own checksum.
No training is performed; counterfactual evaluation never advances live state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--suite-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-seed', type=int, required=True)
    p.add_argument('--workload-seed', type=int, required=True)
    p.add_argument('--reference-method', default='lower_mlp_upper_uncovered_obs')
    p.add_argument('--compare-method', help='Optional promoted method in the SAME completed suite')
    p.add_argument('--mc-itr', action='store_true', help='Expensive fixed-snapshot bridge; not needed for initial diagnosis')
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('Replay belongs on remote Linux')
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import torch
    import config
    from collections import Counter, defaultdict
    from environment.env import Env
    from run_canonical_experiment import configure_method, policy_step
    from utils.canonical_experiment import config_snapshot, digest, exogenous_frame, source_manifest, write_json
    from utils.formal_protocol import verify_completed_job
    from utils.logger import refresh_derived_config_fields
    from utils.snapshot_evaluator import capture, evaluate

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes < 1 or args.threads < 1:
        raise ValueError('Invalid replay size')
    suite = json.loads((args.suite_root/'suite.json').read_text()); plan = suite['plan']
    source = source_manifest()
    if source['source_hash'] != suite['source_hash']:
        raise ValueError('Run with the original frozen source; checkpoint checks remain mandatory')
    method = args.reference_method; seed = args.training_seed; workload = args.workload_seed
    if seed not in plan['training_seeds'] or workload not in plan['workload_seeds'] or args.episodes > plan['eval_episodes']:
        raise ValueError('Replay must be a declared subset of original evaluation')
    variant = plan['variants'][0]
    if method not in variant['methods'] or (args.compare_method and args.compare_method not in variant['methods']):
        raise ValueError('Compared methods must belong to the same frozen suite')
    config.NUM_UES = variant['ue_count']; config.UAV_COMPUTING_CAPACITY = config.UAV_COMPUTING_CAPACITY * variant['compute_scale']
    refresh_derived_config_fields()
    config.PPO_EPOCHS = plan['ppo_epochs']; config.PPO_BATCH_SIZE = plan['batch_size']; config.PPO_ROLLOUT_LENGTH = plan['rollout']
    torch.set_num_threads(args.threads); np.random.seed(seed); torch.manual_seed(seed)
    models = configure_method(method)
    base = args.suite_root/'main'/method/f'seed_{seed}'
    # The completion record protects the complete historical job.
    training = verify_completed_job(base/'train',suite['source_hash'])
    latest = json.loads((training/'latest.json').read_text())
    checkpoint = training/latest['checkpoint']
    contract = json.loads((checkpoint/'contract.json').read_text())
    if contract['method'] != method or contract['source_hash'] != suite['source_hash'] or contract['config_hash'] != digest(config_snapshot()):
        raise ValueError('Checkpoint mismatch')
    for rel, sha in contract['files'].items():
        if hashlib.sha256((checkpoint/rel).read_bytes()).hexdigest() != sha:
            raise ValueError('Checkpoint checksum mismatch')
    for name, model in models.items():
        model.load(str(checkpoint/name))
    comparison = None
    if args.compare_method:
        other_base = args.suite_root/'main'/args.compare_method/f'seed_{seed}'
        other_training = verify_completed_job(other_base/'train',suite['source_hash'])
        other_checkpoint = other_training/json.loads((other_training/'latest.json').read_text())['checkpoint']
        other_contract = json.loads((other_checkpoint/'contract.json').read_text())
        comparison = configure_method(args.compare_method)
        if other_contract['method'] != args.compare_method or other_contract['source_hash'] != suite['source_hash'] or other_contract['config_hash'] != digest(config_snapshot()):
            raise ValueError('Comparison checkpoint contract mismatch')
        for name, model in comparison.items(): model.load(str(other_checkpoint/name))
        # All supported batch-study methods use identical physical/config state.
        if contract['config_hash'] != digest(config_snapshot()):
            raise ValueError('Counterfactual methods must use the identical configuration')
    original = verify_completed_job(base/f'workload_{workload}',suite['source_hash'])
    raw = [json.loads((original/'episodes'/f'{i:05d}.json').read_text()) for i in range(args.episodes)]
    curves = [json.loads(path.read_text())['request_weighted_dsr'] for path in sorted((training/'episodes').glob('*.json'))]
    tail = min(20,len(curves)//2)
    training_tail = {'metric': 'training rollout request-weighted DSR; not held-out convergence',
                     'last_window': float(np.mean(curves[-tail:])) if tail else None,
                     'previous_window': float(np.mean(curves[-2*tail:-tail])) if tail else None,
                     'window_episodes': tail}
    env = Env(scenario_seed=workload); snapshots = []; truncation = Counter(); replay = []
    for episode in range(args.episodes):
        obs = np.asarray(env.reset(),dtype=np.float32)
        for model in models.values(): model.reset()
        trace = hashlib.sha256(); generated = success = 0
        for step in range(plan['steps']):
            trace.update(digest(exogenous_frame(env)).encode())
            for uav in env.uavs:
                covered = sorted(uav.current_covered_ues,key=lambda ue:(float(np.linalg.norm(uav.pos[:2]-ue.pos[:2])),ue.id))
                unseen = sorted((ue for ue in env.ues if not ue.assigned and np.linalg.norm(uav.pos[:2]-ue.pos[:2]) <= config.UAV_SENSING_RANGE),
                                key=lambda ue:(float(np.linalg.norm(uav.pos[:2]-ue.pos[:2])),ue.id))
                slots = max(config.MAX_ASSOCIATED_UES-len(covered),0)
                truncation['uav_observations'] += 1
                truncation['truncated_observations'] += int(len(covered)+len(unseen)>config.MAX_ASSOCIATED_UES)
                truncation['uncovered_service_available'] += sum(ue.current_request.is_service for ue in unseen)
                truncation['uncovered_service_omitted'] += sum(ue.current_request.is_service for ue in unseen[slots:])
            motion, offload, entries = policy_step(models,env,obs,exploration=False)
            if step in (0,250,500,750):
                snapshot = capture(env,np.clip(motion,-1,1))
                assignment = tuple(int(offload[r[0],r[1]]) for r in snapshot.requests)
                result = evaluate(snapshot,assignment)
                lower_obs = entries['lower'][0]
                requests = []
                destination = Counter(str(r['target']) for r in result['requests'])
                source_target = Counter((r['source'],str(r['target'])) for r in result['requests'])
                source_counts = Counter(r['source'] for r in result['requests'])
                target_sources = defaultdict(set)
                for row in result['requests']: target_sources[str(row['target'])].add(row['source'])
                for request, record in zip(snapshot.requests,result['requests']):
                    owner,index = request[:2]
                    features = lower_obs[owner,5:].reshape(config.MAX_OFFLOAD_REQUESTS_PER_UAV,-1)[index]
                    action = record['action']; role = 'local' if action == 0 else 'mbs' if action == 1 else 'best_coop_proxy'
                    ratio = float(features[8 if action == 0 else 10 if action == 1 else 9])
                    estimated = ratio*config.OFFLOAD_LATENCY_RATIO_CLIP*record['deadline']
                    requests.append({**record,'source_request_count': source_counts[record['source']],
                                     'final_target_count': destination[str(record['target'])],
                                     'same_source_target_count': source_target[record['source'],str(record['target'])],
                                     'target_source_count': len(target_sources[str(record['target'])]),
                                     'estimate_role': role,'estimate_clipped': ratio >= 1.,
                                     'estimated_latency': estimated,'actual_minus_estimate': record['latency']-estimated})
                bridge = {}
                if comparison is not None:
                    from utils.chapter4 import objective
                    for label, group in ((method,models),(args.compare_method,comparison)):
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        started = time.perf_counter()
                        alt,_,_ = group['lower'].get_action_and_value(lower_obs,masks=entries['lower'][2],exploration=False)
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        seconds = time.perf_counter()-started
                        alt_assignment = tuple(int(alt[r[0],r[1]]) for r in snapshot.requests)
                        alt_result = evaluate(snapshot,alt_assignment)
                        bridge[label] = {'objective': list(objective(alt_result)), 'lower_inference_seconds': seconds}
                if args.mc_itr:
                    from utils.chapter4 import solve_snapshot
                    solved = solve_snapshot(snapshot,max_rounds=1000,exact_budget=0)
                    bridge['mc_itr'] = {'objective':list(solved['mc_itr']['objective']),
                                        'full_path_seconds':sum(solved[k]['wall_seconds'] for k in ('heuristic_control','greedy_single_pass','mc_itr')),
                                        'termination':solved['mc_itr']['termination']}
                snapshots.append({'episode':episode,'step':step,'snapshot_hash':snapshot.snapshot_hash,
                                  'request_count':len(requests),'requests':requests,'bridge':bridge})
            next_obs,_,metrics = env.step(np.clip(motion,-1,1),offloading_actions=offload)
            # Step-level audit is stronger than only checking episode totals.
            old = raw[episode]['steps'][step]['metrics']
            for key in ('generated_service_count','admitted_service_count','deadline_satisfied_count','lower_assignment_energy'):
                if not np.isclose(metrics[key],old[key],rtol=1e-10,atol=1e-10):
                    raise ValueError(f'Replay changed {key} at {episode}/{step}')
            generated += int(metrics['generated_service_count']); success += int(metrics['deadline_satisfied_count'])
            obs = np.asarray(next_obs,dtype=np.float32)
        if trace.hexdigest() != raw[episode]['exogenous_trace_hash'] or generated != raw[episode]['generated_service_count'] or success != raw[episode]['deadline_satisfied_count']:
            raise ValueError('Replay differs from archived original')
        replay.append({'episode':episode,'generated':generated,'successful':success,'trace':trace.hexdigest()})
    write_json(args.output,{'status':'PASS','protocol':'batch_offload_diagnostic_v1',
                           'source_hash':suite['source_hash'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'training_seed':seed,'workload_seed':workload,'reference_method':method,'compare_method':args.compare_method,'mc_itr':args.mc_itr,'replay':replay,
                           'training_tail':training_tail,'observation_truncation':dict(truncation),'snapshots':snapshots,
                           'limitations':['Descriptive associations, not causal effects.',
                                          'Cooperative estimate is a best-neighbor proxy, not selected-target prediction.',
                                          'Clipped latency estimates must be analyzed separately.']})
    print(json.dumps({'status':'PASS','output':str(args.output),'snapshots':len(snapshots)}))


if __name__ == '__main__':
    main()
