"""Extract immutable SF-HLM logs and replay two prespecified episodes for figures.

All external release/result locations are supplied in a JSON path manifest.
Numerical replay runs on Linux only, uses frozen production functions, and never
updates model parameters or writes into a historical experiment directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2))


def checked_read(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Archived raw file digest differs: ' + str(path))
    return json.loads(raw)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract(args, target):
    parent = read(args.results_root / 'protocol.json')
    manifest = {}
    curves = {}
    lower = {}
    behavior = {}
    for block in parent['confirmation_blocks']:
        workload = parent['training_workloads'][str(block)]
        root = args.results_root / 'foundation_training/shared' / f'block_{block}_workload_{workload}'
        receipt = read(root / 'result.json')
        rows = []
        for path in sorted((root / 'canonical/episodes').glob('*.json')):
            digest = receipt['files'][str(path.relative_to(root))]
            doc = checked_read(path, digest)
            manifest[str(path)] = digest
            rows.append(dict(episode=doc['episode'], dsr=doc['request_weighted_dsr'],
                reward=sum(s['team_reward'] for s in doc['steps']) / len(doc['steps']),
                generated=doc['generated_service_count'], success=doc['deadline_satisfied_count'],
                steps=len(doc['steps'])))
        if len(rows) != parent['foundation_episodes']:
            raise ValueError('Incomplete foundation training series')
        curves[str(block)] = rows
        root = args.results_root / 'components/lower_training/complete/learned' / f'block_{block}'
        path = root / 'losses.json'
        receipt = read(root / 'result.json')
        digest = receipt.get('files', {}).get('losses.json')
        if digest is None:
            digest = sha(path)  # final checkpoint receipt is verified during replay
        doc = checked_read(path, digest)
        manifest[str(path)] = digest
        lower[str(block)] = [dict(update=i, **{k:v for k,v in row.items()
            if k in ('supervised_loss','teacher_samples','ppo_loss','value_loss','matching_calls','initial_ratio_mean')})
            for i, row in enumerate(doc)]
        calls, terms = Counter(), Counter()
        destinations = Counter()
        count = 0
        for workload in parent['confirmation_workloads']:
            for episode in range(6):
                root = args.results_root / 'confirmation_eval/learned' / f'block_{block}_workload_{workload}' / f'episode_{episode:05d}'
                path = root / 'episode.json'
                digest = read(root / 'result.json')['files']['episode.json']
                doc = checked_read(path, digest)
                manifest[str(path)] = digest
                for step in doc['steps']:
                    diag = step.get('diagnostic', {})
                    calls[str(diag.get('matching_calls', 0))] += 1
                    terms[str(diag.get('termination', 'unrecorded'))] += 1
                    metrics = step['metrics']
                    for name in ('local','cooperative','mbs'):
                        destinations[name] += metrics.get('service_offloads_' + name, 0)
                    count += 1
        if count != 30000:
            raise ValueError('Incomplete behavior series')
        behavior[str(block)] = dict(steps=count, matching_calls=dict(calls), termination=dict(terms),
            destinations=dict(destinations))
        print('Extracted block', block, flush=True)
    write(target / 'training_behavior.json', dict(foundation=curves, quota=lower, behavior=behavior,
        source_files=manifest, protocol_sha256=sha(args.results_root / 'protocol.json'),
        observations='All ten blocks; foundation 200 episodes, lower 100 PPO updates; no joint training curve.'))


def actual_attention(actor, obs, config, torch):
    """Reconstruct the weights of the executed SDPA, including head scaling.

    This diagnostic bypasses the historical weight getter's hidden-dim scale.
    It also verifies the reconstructed context against the actual forward.
    """
    actor = getattr(actor, '_orig_mod', actor)
    x = torch.from_numpy(obs).to(next(actor.parameters()).device)
    batch = x.shape[0]
    own = x[:, :actor.own_dim]
    neigh = x[:, actor.own_dim:actor.own_dim + actor.neighbor_block_size].reshape(batch, actor.num_neighbors, actor.neighbor_obs_dim)
    ue = x[:, actor.own_dim + actor.neighbor_block_size:].reshape(batch, actor.num_ues, actor.ue_obs_dim)
    result = {}
    with torch.no_grad():
        own_e = actor.self_encoder(own)
        for name, states, encoder, module in [('neighbor',neigh,actor.neighbor_encoder,actor.neighbor_attn),
                ('ue',ue,actor.ue_encoder,actor.ue_attn)]:
            emb = encoder(states)
            valid = states.abs().sum(-1) > 1e-5
            q = module.query_layer(own_e).view(batch, 1, config.ATTN_NUM_HEADS, module.head_dim).transpose(1, 2)
            k = module.key_layer(emb).view(batch, -1, config.ATTN_NUM_HEADS, module.head_dim).transpose(1, 2)
            v = module.value_layer(emb).view(batch, -1, config.ATTN_NUM_HEADS, module.head_dim).transpose(1, 2)
            logits = q @ k.transpose(-2,-1) / math.sqrt(module.head_dim)
            weights = torch.nan_to_num(torch.softmax(logits.masked_fill(~valid[:,None,None,:], -float('inf')), -1),nan=0.)
            context = (weights @ v).transpose(1,2).reshape(batch,1,-1)
            reconstructed = module.out_proj(context).squeeze(1)
            executed = module(own_e, emb, mask=valid.float())
            if not torch.allclose(reconstructed,executed,atol=2e-5,rtol=2e-5):
                raise ValueError('Attention reconstruction is not executed forward')
            result[name] = dict(weights=weights.squeeze(2).cpu().tolist(), mask=valid.cpu().tolist(),
                states=states.cpu().tolist(), head_dim=module.head_dim,
                context_max_error=float((reconstructed-executed).abs().max()))
    return result


def replay(args, target):
    if sys.platform != 'linux':
        raise RuntimeError('Frozen policy replay is restricted to remote Linux')
    args.seed = args.block = 1060001
    args.workload, args.episode = 1070001, 0
    args.phase, args.family = 'confirmation_eval', 'complete'
    followup = load('thesis_frozen_followup', args.followup_root / 'scripts/analysis/sf_hlm_followup.py')
    runner = followup.runner(args)
    old, _ = followup.initialize(args, runner)
    import numpy as np
    import torch
    import config
    from utils.canonical_experiment import digest, exogenous_frame
    parent = read(args.results_root / 'protocol.json')
    all_records = {}
    for method in ('learned', 'greedy'):
        args.method = method
        models, _, checkpoint, _ = followup.basis(args, runner, old, parent)
        lower, lower_hash = followup.quota(args, runner, old, parent) if method == 'learned' else (None, None)
        fingerprint = runner.base.policy_fingerprint([models['upper'],models['lower'],lower,None])
        root = args.results_root / 'confirmation_eval' / method / f'block_{args.block}_workload_{args.workload}' / 'episode_00000'
        receipt = read(root / 'result.json')
        archived = checked_read(root / 'episode.json', receipt['files']['episode.json'])
        if fingerprint != receipt['model_fingerprint']:
            raise ValueError('Frozen model fingerprint differs')
        env, obs = runner.episode_env(args.workload,args.episode)
        rng = np.random.default_rng(runner.res.trajectory_seed(sha(args.results_root/'protocol.json'),
            args.block,args.workload,args.episode,'search'))
        trace = hashlib.sha256()
        records = []
        attentions = {}
        for step in range(parent['steps']):
            trace.update(digest(exogenous_frame(env)).encode())
            before = [u.pos[:2].tolist() for u in env.uavs]
            users = [u.pos[:2].tolist() for u in env.ues]
            if method == 'learned' and step in (0,250,500,750):
                attentions[str(step)] = actual_attention(models['upper'].actor,obs,config,torch)
            motion, actions, info, _, _, _ = followup.policy(args,runner,parent,models,env,obs,lower=lower,rng=rng)
            obs, rewards, metrics = env.step(motion,offloading_actions=actions)
            runner.base.assert_step(metrics,info)
            obs = np.asarray(obs,np.float32)
            saved = archived['steps'][step]
            if list(info['assignment']) != saved['assignment']:
                raise ValueError(f'Assignment mismatch {method}:{step}')
            for key, value in metrics.items():
                if not math.isclose(value,saved['metrics'][key],rel_tol=1e-11,abs_tol=1e-7):
                    raise ValueError(f'Metric mismatch {method}:{step}:{key}')
            records.append(dict(step=step,uav_before=before,uav_after=[u.pos[:2].tolist() for u in env.uavs],
                ue_positions=users if step == 0 else None,motion=motion.tolist(),
                generated=metrics['generated_service_count'],admitted=metrics['admitted_service_count'],
                success=metrics['deadline_satisfied_count']))
            if step % 250 == 0:
                print('Replay verified',method,step,flush=True)
        if trace.hexdigest() != archived['exogenous_trace_hash']:
            raise ValueError('Exogenous replay fingerprint differs')
        if fingerprint != runner.base.policy_fingerprint([models['upper'],models['lower'],lower,None]):
            raise ValueError('Replay changed a frozen model')
        all_records[method] = dict(steps=records,attention=attentions,
            archived_episode_sha256=receipt['files']['episode.json'],model_fingerprint=fingerprint,
            lower_checkpoint_sha256=lower_hash,exogenous_trace_hash=trace.hexdigest(),
            verified_steps=len(records),all_assignments_and_metrics_match=True)
    write(target / 'replay.json', dict(block=args.block,workload=args.workload,episode=args.episode,
        selection='Prespecified first confirmation block, first workload, first episode.',
        methods=all_records,protocol_sha256=sha(args.results_root/'protocol.json'),
        attention='Actual SDPA head-dimension scale, same masks; context checked against forward.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('extract','replay'))
    p.add_argument('--path-manifest',type=Path,required=True)
    p.add_argument('--visual-output',type=Path,required=True)
    opts = p.parse_args()
    args = SimpleNamespace(**{k:Path(v).expanduser().resolve() for k,v in read(opts.path_manifest).items()})
    if sys.platform == 'linux' and hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    (extract if opts.mode == 'extract' else replay)(args,opts.visual_output)


if __name__ == '__main__':
    main()
