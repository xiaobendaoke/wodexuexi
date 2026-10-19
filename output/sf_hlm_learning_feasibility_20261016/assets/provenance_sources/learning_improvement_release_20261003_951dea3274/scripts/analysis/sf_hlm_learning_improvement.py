"""Resume a registered development pipeline; analyze only its complete population.

All numerical imports are lazy and Linux-only. Existing source, upper policies,
teacher data and confirmation results are read-only external inputs. See the
2026-10-03 contract in docs/plans/sf_hlm_20261001.md.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
import hashlib
import importlib.util
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

FILES = ('scripts/analysis/sf_hlm_learning_improvement.py', 'utils/sf_hlm_learning_improvement.py',
         'marl_models/sf_hlm_action_effect.py', 'tests/test_sf_hlm_learning_improvement.py', 'paths.py')
TRAIN_VARIANTS = ('baseline', 'multistep', 'effects')
METHODS = ('old_learned', 'greedy', 'random', 'baseline_bc', 'baseline_ppo',
           'multistep_bc', 'multistep_ppo', 'effects_bc', 'effects_ppo')
PRIMARY_PAIRS = (('multistep_ppo', 'baseline_ppo'), ('effects_ppo', 'multistep_ppo'),
                 ('multistep_ppo', 'random'), ('effects_ppo', 'random'))
MECHANISM_PAIRS = tuple((v+'_ppo', v+'_bc') for v in TRAIN_VARIANTS)
PATH_ARGS = ('source_root', 'suite_root', 'extension_root', 'output', 'legacy_results', 'resources',
             'adaptation_root', 'followup_root', 'parent_results', 'results_root', 'release_root', 'deliverables')


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2,
                    default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x) if isinstance(x, Path) else fail_json(x))+'\n')
    temp.replace(path)


def fail_json(x): raise TypeError(type(x).__name__)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def registered_tasks(blocks, workloads, episodes, points):
    tasks = []
    for b in blocks:
        tasks.append(('teacher', 'multistep', b, 0, -1, -1))
        tasks.extend(('train', v, b, 0, -1, -1) for v in TRAIN_VARIANTS)
        for w in workloads:
            for e in range(episodes):
                tasks.append(('capture', 'old_learned', b, w, e, -1))
                tasks.extend((phase, 'all', b, w, e, s) for phase in ('diagnostic', 'timing') for s in points)
                tasks.extend(('evaluate', m, b, w, e, -1) for m in METHODS)
    return tasks


def task_id(task): return '_'.join(map(str, task))


def task_folder(args, task):
    phase, method, b, w, e, s = task
    path = args.results_root/phase/method/f'block_{b}_workload_{w}'
    if e >= 0: path = path/f'episode_{e:05d}'
    if s >= 0: path = path/f'step_{s:05d}'
    return path


def receipt(args, p, task, **extra):
    folder = task_folder(args, task)
    files = {str(f.relative_to(folder)): sha(f) for f in folder.rglob('*')
             if f.is_file() and f.name not in ('result.json', 'progress.json') and not f.name.endswith(('.tmp', '.lock'))}
    atomic_json(folder/'result.json', dict(status='PASS', task=list(task),
                protocol_sha256=sha(args.results_root/'protocol.json'), files=files, **extra))


def verify_task(args, p, task, full=True):
    path = task_folder(args, task)
    if not (path/'result.json').exists(): return False
    d = read(path/'result.json')
    if d['status'] != 'PASS' or d['task'] != list(task) or d['protocol_sha256'] != sha(args.results_root/'protocol.json'):
        raise ValueError('Task identity changed: '+task_id(task))
    if task[0] in ('capture', 'evaluate') and d.get('steps') != p['steps']:
        raise ValueError('Incomplete episode: '+task_id(task))
    if task[0] == 'capture' and d.get('snapshots') != len(p['snapshot_steps']):
        raise ValueError('Incomplete capture population')
    if task[0] == 'train' and d.get('iterations') != p.get('iterations'):
        raise ValueError('Incomplete training')
    if not d['files']: raise ValueError('Empty result manifest')
    for f, digest in d['files'].items():
        if full or f.startswith(('bc/', 'checkpoint/')) or f == 'data.pt':
            if sha(path/f) != digest: raise ValueError('Changed result: '+str(path/f))
    return True


def require_population(observed, expected):
    if set(map(tuple, observed)) != set(map(tuple, expected)) or len(observed) != len(expected):
        raise ValueError('Registered population incomplete or duplicated')


def worker_command(args, task, paths=True):
    command = [sys.executable, str(args.release_root/FILES[0]), 'worker', '--release-root', str(args.release_root),
               '--results-root', str(args.results_root), '--task', json.dumps(task, separators=(',', ':'))]
    if paths:
        for key in PATH_ARGS:
            if key not in ('release_root', 'results_root'):
                command += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    return command


def decode_worker(command, args):
    try:
        if command[1] != str(args.release_root/FILES[0]) or command[2] != 'worker': return None
        if command[command.index('--results-root')+1] != str(args.results_root): return None
        if command[command.index('--release-root')+1] != str(args.release_root): return None
        task = json.loads(command[command.index('--task')+1])
        return tuple(task) if len(task) == 6 else None
    except (ValueError, IndexError, TypeError): return None


def live_workers(args):
    live = {}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command = [s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            task = decode_worker(command, args)
            if task is not None:
                identity = (proc/'stat').read_text().rsplit(')', 1)[1].split()
                if identity[0] != 'Z': live.setdefault(task, []).append((int(proc.name), identity[19]))
        except (OSError, ValueError, UnicodeError): pass
    return live


def protocol(args):
    old = read(args.output/'protocol.json'); blocks = old['historical_blocks']; workloads = old['eval_workloads']
    if args.smoke: blocks, workloads = blocks[:1], workloads[:1]
    points = [0, 4] if args.smoke else old['snapshot_steps']; episodes = 1 if args.smoke else 2
    inputs = {'old_protocol': sha(args.output/'protocol.json'), 'parent_protocol': sha(args.parent_results/'protocol.json')}
    for b in blocks:
        for label, rel in [('dataset', f'dataset/block_{b}/dataset.pt'),
                           ('lower', f'lower_training/complete/learned/block_{b}/checkpoint/quota_v2.pt')]:
            inputs[f'{label}_{b}'] = sha(args.output/rel)
    return dict(version='learning_improvement_development_20261003_v1', smoke=args.smoke,
        purpose='verification' if args.smoke else 'development', blocks=blocks, workloads=workloads, episodes=episodes,
        steps=8 if args.smoke else 1000, snapshot_steps=points, matching_budget=16, base_repair_budget=128,
        teacher_budget=24 if args.smoke else 128, beam_width=4, branch_width=8, teacher_depth=6,
        random_repeats=2 if args.smoke else 20, supervised_epochs=1 if args.smoke else 5,
        iterations=2 if args.smoke else 100, graphs_per_update=1 if args.smoke else 32,
        ppo_epochs=1 if args.smoke else old['ppo_epochs'], rollout_workers=2 if args.smoke else 8,
        lower_seeds={str(b): old['lower_seeds'][str(b)] for b in blocks},
        tasks=[list(t) for t in registered_tasks(blocks, workloads, episodes, points)], inputs=inputs,
        files={f: sha(args.release_root/f) for f in FILES},
        external_roots={k: str(getattr(args, k)) for k in PATH_ARGS if k not in ('deliverables', 'release_root', 'results_root')},
        primary_pairs=PRIMARY_PAIRS, mechanism_pairs=MECHANISM_PAIRS,
        gates=dict(development_gain=.003, positive_blocks=4, energy_ratio=1.05, full_policy_p95=1.),
        claim_boundary='Five reused development blocks; no independent confirmation or thesis main-result replacement.',
        teacher_contract='All branch matching calls counted; no STOP inferred from budget exhaustion; only witnessed improving paths.',
        training_contract='Same observed training graphs, seeds, 5 BC epochs, sampled label count and 100x32 PPO opportunity; fixed upper. New teacher has extra offline cost.',
        information_contract='Observed predictor only, same physical observation/admission; no hidden snapshot outcomes in teacher or features.',
        timing_contract='Full prepare+actor+search+mapping+serialization on eight reserved physical cores; simulation execution excluded.')


def initialize_numerical(args):
    if sys.platform != 'linux': raise RuntimeError('Numerical work must run on the Linux experiment server')
    args.seed = 0; args.family = 'complete'
    followup = load('learning_parent_runner', args.followup_root/'scripts/analysis/sf_hlm_followup.py')
    r = followup.runner(args); old, rollout = followup.initialize(args, r)
    for name in ('utils.sf_hlm_learning_improvement', 'marl_models.sf_hlm_action_effect'):
        load(name, args.release_root/(name.replace('.', '/')+'.py'))
    return followup, r, old


def test_runtime_args():
    args = parser().parse_args(['worker', *json.loads(os.environ['SF_HLM_IMPROVEMENT_ARGS'])])
    return args


def basis(args, followup, r, old):
    a = copy.copy(args); a.results_root = args.parent_results; a.phase = 'dev_eval'; a.family = 'complete'; a.seed = 0
    parent = read(args.parent_results/'protocol.json')
    models, source, checkpoint, _ = followup.basis(a, r, old, parent)
    lower, lower_hash = followup.quota(a, r, old, parent)
    current = read(args.results_root/'protocol.json')
    if lower_hash != current['inputs'][f'lower_{args.block}']: raise ValueError('Frozen lower checkpoint differs from registered input')
    return models, lower, dict(source_config=source['config_hash'], upper_contract_sha256=sha(checkpoint/'contract.json'))


def lower_class(variant):
    from marl_models.sf_hlm_v2 import CompleteQuotaPPO
    from marl_models.sf_hlm_action_effect import ActionEffectPPO
    return ActionEffectPPO if variant == 'effects' else CompleteQuotaPPO


def train_contract(args, variant, block):
    return dict(protocol_sha256=sha(args.results_root/'protocol.json'), variant=variant, block=block, training_only=True)


def load_candidate(args, p, method, block):
    if '_' not in method: raise ValueError('No candidate stage')
    variant, stage = method.rsplit('_', 1); task = ('train', variant, block, 0, -1, -1)
    if not verify_task(args, p, task, full=False): raise ValueError('Candidate training incomplete')
    import config
    model = lower_class(variant)(config.NUM_UAVS)
    model.load(task_folder(args, task)/('bc' if stage == 'bc' else 'checkpoint'), train_contract(args, variant, block))
    return model


def policy_step(args, r, p, models, lower, env, obs, method, rng):
    if method in ('old_learned', 'greedy', 'random'):
        return r.policy(env, obs, 'learned' if method == 'old_learned' else method, models,
                        family='complete', lower=lower, rng=rng, budget=p['matching_budget'])
    import config
    import numpy as np
    from utils.sf_hlm_learning_improvement import run_search
    from utils.sf_hlm_matching import feedback
    started = time.perf_counter(); _, masks, requests, rows = r.base.prepare(env)
    assignment, d, _ = run_search(rows, num_uavs=config.NUM_UAVS, kind='learned', model=lower,
                                 budget=p['matching_budget'], effects=method.startswith('effects_'))
    motion = models['upper'].select_actions(obs, exploration=False)
    actions = r.base.actions_from_assignment(requests, assignment, masks)
    summary = feedback(rows, assignment, config.NUM_UAVS, energy_reference=config.OFFLOAD_ENERGY_NORM_REF)
    from marl_models.sf_hlm_v2 import service_view
    service_view(env, summary)  # Include shared coordinator preparation in timing.
    upload = len(json.dumps(dict(rows=rows), default=r.base.serial, separators=(',', ':')).encode())
    download = len(json.dumps(dict(ue_ids=[q[2] for q in requests], actions=assignment, feedback=summary.tolist()),
                              default=r.base.serial, separators=(',', ':')).encode())
    info = dict(request_ids=[q[2] for q in requests], assignment=assignment, diagnostic=d,
                logical_upload_bytes=upload, logical_download_bytes=download, policy_seconds=time.perf_counter()-started)
    return np.clip(motion, -1, 1), actions, info, None, None, rows


def episode(args, p, task, r, models, lower, capture=False):
    import config
    import numpy as np
    from utils.canonical_experiment import digest, exogenous_frame
    from utils.snapshot_evaluator import capture as freeze
    folder = task_folder(args, task); folder.mkdir(parents=True, exist_ok=True)
    method, block, workload, ep = task[1:5]
    env, obs = r.episode_env(workload, ep); steps = []; trace = hashlib.sha256()
    # Method is absent: random seeds remain paired and independent of scheduling.
    rng = np.random.default_rng(r.res.trajectory_seed(p['inputs']['old_protocol'], block, workload, ep, 'search'))
    totals = {k: 0. for k in ('generated', 'admitted', 'success', 'potential', 'energy', 'mbs', 'fairness')}
    frozen = r.base.policy_fingerprint([models['upper'], lower]); points = []
    for step in range(p['steps']):
        trace.update(digest(exogenous_frame(env)).encode()); potential = sum(u.latent_request[0] < config.NUM_SERVICES for u in env.ues)
        if capture and step in p['snapshot_steps']:
            observed = pickle.dumps((env, obs), protocol=pickle.HIGHEST_PROTOCOL)
        motion, actions, info, _, _, rows = policy_step(args, r, p, models, lower, env, obs, method, rng)
        if capture and step in p['snapshot_steps']:
            snap = freeze(env, motion)
            if tuple(info['request_ids']) != tuple(q[2] for q in snap.requests): raise ValueError('Snapshot request order differs')
            target = folder/f'step_{step:05d}'; target.mkdir(parents=True, exist_ok=True)
            (target/'snapshot.bin').write_bytes(pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL))
            (target/'observed.bin').write_bytes(observed)
            atomic_json(target/'input.json', dict(rows=rows, snapshot_hash=snap.snapshot_hash, motion=motion,
                                                 request_ids=info['request_ids'], policy=info))
            points.append(step)
        obs, rewards, metrics = env.step(motion, offloading_actions=actions); r.base.assert_step(metrics, info)
        obs = np.asarray(obs, np.float32); totals['potential'] += potential
        for key, name in [('generated', 'generated_service_count'), ('admitted', 'admitted_service_count'),
                          ('success', 'deadline_satisfied_count'), ('energy', 'energy'), ('mbs', 'service_offloads_mbs'), ('fairness', 'fairness')]:
            totals[key] += float(metrics[name])
        steps.append(dict(step=step, potential=potential, metrics=metrics, team_reward=float(np.mean(rewards)), **info))
        if step % 100 == 0: atomic_json(folder/'progress.json', dict(step=step+1, steps=p['steps'], pid=os.getpid()))
    if frozen != r.base.policy_fingerprint([models['upper'], lower]): raise ValueError('Frozen policy changed')
    atomic_json(folder/'episode.json', dict(summary=totals, steps=steps, exogenous_trace_hash=trace.hexdigest()))
    receipt(args, p, task, steps=p['steps'], snapshots=len(points), model_fingerprint=frozen)


def snapshot_input(args, p, task):
    cap = ('capture', 'old_learned', task[2], task[3], task[4], -1)
    if not verify_task(args, p, cap, full=False): raise ValueError('Capture is incomplete')
    parent = task_folder(args, cap); folder = parent/f'step_{task[5]:05d}'
    manifest = read(parent/'result.json')['files']
    for name in ('snapshot.bin', 'observed.bin', 'input.json'):
        rel = str((folder/name).relative_to(parent))
        if sha(folder/name) != manifest[rel]: raise ValueError('Snapshot input changed')
    d = read(folder/'input.json')
    for row in d['rows']: row['options'] = {int(k): v for k, v in row['options'].items()}
    return folder, d, pickle.loads((folder/'snapshot.bin').read_bytes())


def diagnostic(args, p, task, r, lower):
    import config
    import numpy as np
    from utils.sf_hlm_learning_improvement import run_search
    from utils.snapshot_evaluator import evaluate
    _, d, snapshot = snapshot_input(args, p, task); results = []
    for kind in ('learned', 'no_stop', 'hybrid', 'random'):
        for repetition in range(p['random_repeats'] if kind in ('hybrid', 'random') else 1):
            seed = r.res.trajectory_seed(p['inputs']['old_protocol'], *task[2:], repetition, 'stop_completion')
            assignment, info, _ = run_search(d['rows'], num_uavs=config.NUM_UAVS, kind=kind, model=lower,
                                            rng=np.random.default_rng(seed), budget=p['matching_budget'])
            actual = evaluate(snapshot, assignment)
            results.append(dict(kind=kind, repetition=repetition, seed=seed, assignment=assignment, diagnostic=info,
                                actual_misses=sum(not q['success'] for q in actual['requests']),
                                allocation_energy=float(actual['metrics']['lower_assignment_energy'])))
    learned = results[0]
    if tuple(learned['assignment']) != tuple(d['policy']['assignment']): raise ValueError('Stopped policy replay differs')
    atomic_json(task_folder(args, task)/'diagnostic.json', dict(snapshot_hash=d['snapshot_hash'], requests=len(d['rows']), results=results))
    receipt(args, p, task, trials=len(results))


_NUMERIC = None
_ARGS = None


def child_initialize(values):
    global _NUMERIC, _ARGS
    _ARGS = argparse.Namespace(**values)
    profile = read(_ARGS.resources/'profile.json')
    os.sched_setaffinity(0, set(profile['cpus'][:profile['slots']]))
    _NUMERIC = initialize_numerical(_ARGS)


def numeric_child(job):
    import config
    import numpy as np
    import torch
    from utils.sf_hlm_learning_improvement import multistep_teacher, run_search
    _, r, _ = _NUMERIC; kind = job['kind']
    with r.res.CpuLease(_ARGS.resources, task=['learning_improvement', kind, job['id']], kind='teacher' if kind == 'teacher' else 'rollout_complete'):
        torch.manual_seed(job['seed']); np.random.seed(job['seed'])
        if kind == 'teacher':
            samples, d = multistep_teacher(job['rows'], num_uavs=config.NUM_UAVS, seed=job['seed'],
                total_budget=job['budget'], beam_width=job['beam_width'], branch_width=job['branch_width'], depth=job['depth'])
            return dict(id=job['id'], samples=samples, diagnostic=d)
        model = lower_class(job['variant'])(config.NUM_UAVS); model.network.load_state_dict(job['weights'])
        # Constructor draw counts differ for the effect head; trajectory seed must not.
        torch.manual_seed(job['seed'])
        _, d, records = run_search(job['rows'], num_uavs=config.NUM_UAVS, kind='learned', model=model,
                                   exploration=True, budget=job['budget'], effects=job['variant'] == 'effects')
        return dict(id=job['id'], records=records, diagnostic=d, seed=job['seed'])


def pool(args, p):
    return ProcessPoolExecutor(max_workers=p['rollout_workers'], mp_context=multiprocessing.get_context('spawn'),
                               initializer=child_initialize, initargs=(vars(args),))


def teacher(args, p, task, r, lease):
    import torch
    folder = task_folder(args, task); folder.mkdir(parents=True, exist_ok=True)
    source = args.output/f'dataset/block_{task[2]}/dataset.pt'
    if sha(source) != p['inputs'][f'dataset_{task[2]}']: raise ValueError('Training dataset changed')
    data = torch.load(source, weights_only=False); rows = data['rows'][:4] if p['smoke'] else data['rows']
    jobs = [dict(kind='teacher', id=i, rows=x, seed=r.res.trajectory_seed(p['inputs']['old_protocol'], task[2], i, 'multistep_teacher'),
                 budget=p['teacher_budget'], beam_width=p['beam_width'], branch_width=p['branch_width'], depth=p['teacher_depth']) for i, x in enumerate(rows)]
    with pool(args, p) as executor:
        with lease.suspended(): outputs = list(executor.map(numeric_child, jobs))
    samples = [s for o in outputs for s in o['samples']]
    if not samples: raise ValueError('Multi-step teacher has no witnessed improving paths; no invented training labels')
    torch.save(dict(samples=samples), folder/'data.pt')
    atomic_json(folder/'teacher.json', dict(graphs=len(rows), source_dataset_sha256=sha(source),
                                          matching_calls=sum(o['diagnostic']['matching_calls'] for o in outputs),
                                          labels=len(samples), details=[dict(id=o['id'], **o['diagnostic']) for o in outputs]))
    receipt(args, p, task, graphs=len(rows), labels=len(samples))


def training(args, p, task, r, lease):
    import config
    import numpy as np
    import torch
    folder = task_folder(args, task); folder.mkdir(parents=True, exist_ok=True); variant, block = task[1:3]
    source = args.output/f'dataset/block_{block}/dataset.pt'
    if sha(source) != p['inputs'][f'dataset_{block}']: raise ValueError('Training dataset changed')
    data = torch.load(source, weights_only=False); rows = data['rows']; labels = data['teacher']
    if not labels: raise ValueError('Original teacher is empty')
    sample_count = min(16, len(labels)) if p['smoke'] else len(labels)
    if variant != 'baseline':
        t = ('teacher', 'multistep', block, 0, -1, -1)
        if not verify_task(args, p, t, full=False): raise ValueError('Teacher incomplete')
        labels = torch.load(task_folder(args, t)/'data.pt', weights_only=False)['samples']
        # Match gradient opportunities, never pretend repeated labels are new graphs.
        chosen = np.random.default_rng(p['lower_seeds'][str(block)]).choice(len(labels), sample_count, replace=len(labels) < sample_count)
        labels = [labels[int(i)] for i in chosen]; unique = len(set(map(int, chosen)))
    else: labels = labels[:sample_count]; unique = len(labels)
    seed = p['lower_seeds'][str(block)]; torch.manual_seed(seed)
    lower = lower_class(variant)(config.NUM_UAVS); rng = np.random.default_rng(seed)
    contract = train_contract(args, variant, block); start = 0; losses = []
    latest_path = folder/'latest.json'
    if latest_path.exists():
        latest = read(latest_path); recover = folder/latest['checkpoint']
        for f, digest in latest['files'].items():
            if sha(recover/f) != digest: raise ValueError('Training recovery changed')
        lower.load(recover, contract); state = torch.load(recover/'state.pt', weights_only=False)
        if state['contract'] != contract: raise ValueError('Recovery identity changed')
        start = state['next_iteration']; losses = state['losses']; rng.bit_generator.state = state['rng']; torch.set_rng_state(state['torch_rng'])
    else:
        torch.manual_seed(seed+1); losses = [lower.pretrain(labels, epochs=p['supervised_epochs'])]
        lower.save(folder/'bc', contract)
        # PPO sampling schedule does not depend on architecture initialization.
        torch.manual_seed(seed+2)
        save_recovery(folder, lower, contract, 0, rng, losses)
    with pool(args, p) as executor:
        for iteration in range(start, p['iterations']):
            indices = rng.integers(0, len(rows), size=p['graphs_per_update'])
            weights = {k: v.detach().clone() for k, v in lower.network.state_dict().items()}
            jobs = [dict(kind='rollout', id=i, rows=rows[int(index)], variant=variant, weights=weights,
                         seed=r.res.trajectory_seed(p['inputs']['old_protocol'], seed, iteration, i, int(index), 'paired_rollout'),
                         budget=p['matching_budget']) for i, index in enumerate(indices)]
            with lease.suspended(): outputs = list(executor.map(numeric_child, jobs))
            records = [record for o in outputs for record in o['records']]
            loss = lower.update(records, epochs=p['ppo_epochs']); loss.update(iteration=iteration,
                matching_calls=sum(o['diagnostic']['matching_calls'] for o in outputs), graph_indices=indices.tolist(),
                trajectory_seeds=[o['seed'] for o in outputs])
            losses.append(loss); save_recovery(folder, lower, contract, iteration+1, rng, losses)
            atomic_json(folder/'progress.json', dict(iteration=iteration+1, iterations=p['iterations'], pid=os.getpid()))
    lower.save(folder/'checkpoint', contract); atomic_json(folder/'losses.json', losses)
    receipt(args, p, task, iterations=p['iterations'], supervised_sample_count=len(labels), unique_labels=unique,
            dataset_sha256=sha(source), checkpoint_contract=contract)


def save_recovery(folder, model, contract, next_iteration, rng, losses):
    import torch
    # Alternate recovery directories: publish a manifest only after every file is
    # durable. A killed writer cannot corrupt the previous published checkpoint.
    target = folder/f'recovery_slot_{next_iteration % 2}'; model.save(target, contract)
    torch.save(dict(contract=contract, next_iteration=next_iteration, rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), losses=losses), target/'state.pt')
    atomic_json(folder/'latest.json', dict(checkpoint=target.name, iterations=next_iteration,
                files={f.name: sha(f) for f in target.iterdir() if f.is_file()}))


def timing(args, p, task, r, models, old_lower):
    import numpy as np
    from utils.snapshot_evaluator import evaluate
    folder, inp, snap = snapshot_input(args, p, task); results = []
    order = np.random.default_rng(r.res.trajectory_seed(p['inputs']['old_protocol'], *task[2:], 'timing_order')).permutation(METHODS)
    for method in order:
        lower = old_lower if method in ('old_learned', 'greedy', 'random') else load_candidate(args, p, method, task[2])
        # Restore observed input separately from the private execution snapshot.
        env, obs = pickle.loads((folder/'observed.bin').read_bytes())
        rng = np.random.default_rng(r.res.trajectory_seed(p['inputs']['old_protocol'], *task[2:], 'timing'))
        policy_step(args, r, p, models, lower, env, obs, method, copy.deepcopy(rng))
        env, obs = pickle.loads((folder/'observed.bin').read_bytes())
        motion, _, info, _, _, _ = policy_step(args, r, p, models, lower, env, obs, method, rng)
        if not np.allclose(motion, inp['motion'], atol=1e-7, rtol=0): raise ValueError('Frozen upper motion changed')
        actual = evaluate(snap, info['assignment'])
        results.append(dict(method=method, seconds=info['policy_seconds'], diagnostic=info['diagnostic'],
                            assignment=info['assignment'], actual_misses=sum(not q['success'] for q in actual['requests']),
                            allocation_energy=float(actual['metrics']['lower_assignment_energy'])))
    atomic_json(task_folder(args, task)/'timing.json', dict(snapshot_hash=inp['snapshot_hash'], requests=len(inp['rows']), results=results))
    receipt(args, p, task, methods=len(results))


def check_release(args, p):
    for f, digest in p['files'].items():
        if sha(args.release_root/f) != digest: raise ValueError('Frozen release changed: '+f)
    if sha(args.output/'protocol.json') != p['inputs']['old_protocol'] or sha(args.parent_results/'protocol.json') != p['inputs']['parent_protocol']:
        raise ValueError('Frozen parent protocol changed')
    parent = read(args.parent_results/'protocol.json')
    for root, files in ((args.followup_root, parent['files']), (args.adaptation_root, parent['adaptation_files'])):
        for f, digest in files.items():
            if sha(root/f) != digest: raise ValueError('Frozen parent release changed: '+str(root/f))
    for key, value in p['external_roots'].items():
        if str(getattr(args, key)) != value: raise ValueError('External input path changed: '+key)


def worker(args):
    p = read(args.results_root/'protocol.json'); task = tuple(json.loads(args.task))
    if list(task) not in p['tasks']: raise ValueError('Unregistered task')
    check_release(args, p)
    args.block = task[2]; args.family = 'complete'; args.seed = 0
    followup, r, old = initialize_numerical(args)
    resource = args.resources/'isolated' if task[0] in ('diagnostic', 'timing') else args.resources
    with r.res.task_lock(args.results_root/'task_locks'/(task_id(task)+'.lock')):
        if verify_task(args, p, task): return
        with r.res.CpuLease(resource, task=['learning_improvement', *task], kind='environment') as lease:
            if task[0] == 'teacher': teacher(args, p, task, r, lease)
            elif task[0] == 'train': training(args, p, task, r, lease)
            else:
                models, lower, lock = basis(args, followup, r, old)
                atomic_json(task_folder(args, task)/'basis.json', lock)
                if task[0] == 'diagnostic': diagnostic(args, p, task, r, lower)
                elif task[0] == 'timing': timing(args, p, task, r, models, lower)
                else:
                    if task[1] not in ('old_learned', 'greedy', 'random'): lower = load_candidate(args, p, task[1], task[2])
                    episode(args, p, task, r, models, lower, capture=task[0] == 'capture')


def dispatch(args, p, selected):
    tasks = [tuple(t) for t in p['tasks'] if t[0] in selected[0] and (selected[1] is None or t[1] in selected[1])]
    active = live_workers(args); handles = {}; waiting = [t for t in tasks if not verify_task(args, p, t)]
    phase = '+'.join(selected[0]); completed = len(tasks)-len(waiting)
    for task in waiting:
        if len(active.get(task, [])) > 1: raise RuntimeError('Duplicate live task: '+task_id(task))
    limit = args.timing_workers if phase in ('diagnostic', 'timing') else args.workers
    def one(task):
        live = active.get(task)
        if live:
            pid, identity = live[0]
            while True:
                try:
                    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                    if fields[0] == 'Z' or fields[19] != identity: break
                except OSError: break
                time.sleep(2)
            if not verify_task(args, p, task): raise RuntimeError('Adopted task exited incomplete: '+task_id(task))
        else:
            logfile = args.results_root/'logs'/(task_id(task)+'.log'); logfile.parent.mkdir(parents=True, exist_ok=True)
            with logfile.open('a') as log:
                command = worker_command(args, task); child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                                              cwd=args.release_root, close_fds=True)
                handles[task] = child
                if child.wait() != 0: raise RuntimeError('Task failed: '+task_id(task)+'; log '+str(logfile))
                if not verify_task(args, p, task): raise ValueError('Missing completion receipt: '+task_id(task))
        return task
    executor = ThreadPoolExecutor(max_workers=limit)
    pending = {executor.submit(one, t): t for t in waiting}
    try:
        for future in as_completed(pending):
            future.result(); completed += 1
            atomic_json(args.results_root/'status.json', dict(status='RUNNING', phase=phase, completed=completed,
                        phase_tasks=len(tasks), pid=os.getpid(), updated_unix=time.time()))
    except BaseException:
        for f in pending: f.cancel()
        # Running workers retain independent task/CPU locks and publish receipts;
        # no child from another experiment is killed. Resume adopts live workers.
        executor.shutdown(wait=False, cancel_futures=True); raise
    else: executor.shutdown(wait=True)
    for task in tasks: verify_task(args, p, task)


def paired(values):
    import numpy as np
    from scipy import stats
    a = np.asarray(values, float); n = len(a); mean = float(a.mean()); sd = float(a.std(ddof=1)) if n > 1 else None
    if n < 2: return dict(n=n, mean=mean, ci95=None, p=None, positive=int((a > 0).sum()), exact_signflip_p=None)
    error = sd/np.sqrt(n); delta = float(stats.t.ppf(.975, n-1)*error)
    p = float(stats.ttest_1samp(a, 0).pvalue) if sd else (1. if mean == 0 else 0.)
    exact = sum(abs(float(np.mean(a*np.asarray(signs)))) >= abs(mean)-1e-14 for signs in itertools.product((-1, 1), repeat=n))/2**n
    return dict(n=n, mean=mean, ci95=[mean-delta, mean+delta], p=p, positive=int((a > 0).sum()), exact_signflip_p=exact, differences=a.tolist())


def holm(comparisons):
    available = sorted((d['p'], k) for k, d in comparisons.items() if d['p'] is not None); prior = 0.
    for index, (pv, key) in enumerate(available):
        prior = max(prior, min(1., (len(available)-index)*pv)); comparisons[key]['holm_p'] = prior


def analyze(args, p):
    if sys.platform != 'linux': raise RuntimeError('Scientific aggregation runs on the experiment server')
    import numpy as np
    observed = []
    for t in p['tasks']:
        if verify_task(args, p, tuple(t)): observed.append(t)
    require_population(observed, p['tasks'])
    metrics = {}; traces = {}; timing_values = {}; diagnostics = {}; opportunities = {}
    for method in METHODS:
        seeds = {}; time_seeds = {}
        for b in p['blocks']:
            totals = {k: 0. for k in ('generated', 'admitted', 'success', 'potential', 'energy', 'mbs', 'fairness')}; calls = []; times = []
            for w in p['workloads']:
                for e in range(p['episodes']):
                    d = read(task_folder(args, ('evaluate', method, b, w, e, -1))/'episode.json')
                    if len(d['steps']) != p['steps'] or [s['step'] for s in d['steps']] != list(range(p['steps'])): raise ValueError('Incomplete step sequence')
                    traces.setdefault((b, w, e), set()).add(d['exogenous_trace_hash'])
                    recomputed = {k: 0. for k in totals}
                    for s in d['steps']:
                        recomputed['potential'] += s['potential']
                        for key, name in [('generated', 'generated_service_count'), ('admitted', 'admitted_service_count'), ('success', 'deadline_satisfied_count'),
                                          ('energy', 'energy'), ('mbs', 'service_offloads_mbs'), ('fairness', 'fairness')]: recomputed[key] += s['metrics'][name]
                        dg = s['diagnostic']
                        if dg['chosen_predicted']['energy'] > dg['common_energy_cap']+1e-8: raise ValueError('Predicted guard violated')
                        if dg['matching_calls'] > p['matching_budget']: raise ValueError('Matching budget violated')
                        calls.append(dg['matching_calls'])
                    if not all(np.isclose(totals_value, d['summary'][k], rtol=1e-12, atol=1e-8) for k, totals_value in recomputed.items()): raise ValueError('Step counters do not reproduce summary')
                    for key in totals: totals[key] += recomputed[key]
                    for step in p['snapshot_steps']:
                        td = read(task_folder(args, ('timing', 'all', b, w, e, step))/'timing.json')
                        row = next(x for x in td['results'] if x['method'] == method); times.append(row['seconds'])
            divisor = len(p['workloads'])*p['episodes']*p['steps']
            seeds[str(b)] = dict(dsr=totals['success']/max(totals['generated'], 1), potential_dsr=totals['success']/max(totals['potential'], 1),
                                admission=totals['admitted']/max(totals['generated'], 1), admitted_quality=totals['success']/max(totals['admitted'], 1),
                                energy=totals['energy']/divisor, mbs=totals['mbs']/max(totals['admitted'], 1), fairness=totals['fairness']/divisor,
                                matching_calls=float(np.mean(calls)), **{'counts': totals})
            time_seeds[str(b)] = dict(p95=float(np.quantile(times, .95)), mean=float(np.mean(times)), samples=len(times))
        metrics[method] = dict(seeds=seeds, means={k: float(np.mean([d[k] for d in seeds.values()])) for k in ('dsr', 'potential_dsr', 'admission', 'admitted_quality', 'energy', 'mbs', 'fairness', 'matching_calls')})
        timing_values[method] = dict(seeds=time_seeds, mean_seed_p95=float(np.mean([d['p95'] for d in time_seeds.values()])))
    if any(len(t) != 1 for t in traces.values()): raise ValueError('Paired exogenous streams differ')
    for family, pairs in [('primary', PRIMARY_PAIRS), ('mechanism', MECHANISM_PAIRS)]:
        comparisons = {}
        for a, b in pairs:
            comparisons[a+'-'+b] = paired([metrics[a]['seeds'][str(s)]['dsr']-metrics[b]['seeds'][str(s)]['dsr'] for s in p['blocks']])
        holm(comparisons); diagnostics[family] = comparisons
    stop = {}
    for b in p['blocks']:
        per_kind = {k: [] for k in ('learned', 'no_stop', 'hybrid', 'random')}; count = stopped = 0
        for w in p['workloads']:
            for e in range(p['episodes']):
                for s in p['snapshot_steps']:
                    d = read(task_folder(args, ('diagnostic', 'all', b, w, e, s))/'diagnostic.json')
                    for kind in per_kind:
                        trials = [q for q in d['results'] if q['kind'] == kind]
                        per_kind[kind].append(float(np.mean([q['actual_misses'] for q in trials])))
                    count += 1; stopped += d['results'][0]['diagnostic']['termination'] == 'policy_stop'
        stop[str(b)] = dict(snapshots=count, stopped=stopped, mean_actual_misses={k: float(np.mean(v)) for k, v in per_kind.items()})
    diagnostics['stop'] = dict(seeds=stop, pairs={k+'-learned': paired([stop[str(b)]['mean_actual_misses'][k]-stop[str(b)]['mean_actual_misses']['learned'] for b in p['blocks']]) for k in ('no_stop', 'hybrid', 'random')})
    for b in p['blocks']:
        for v in TRAIN_VARIANTS:
            d = read(task_folder(args, ('train', v, b, 0, -1, -1))/'losses.json')
            opportunities[(b, v)] = [(q['graph_indices'], q['trajectory_seeds']) for q in d[1:]]
        if opportunities[(b, 'baseline')] != opportunities[(b, 'multistep')] or opportunities[(b, 'multistep')] != opportunities[(b, 'effects')]: raise ValueError('Training opportunity schedules differ')
    gates = {}
    for method in ('multistep_ppo', 'effects_ppo'):
        comparison = diagnostics['primary'][method+'-random']
        energy = float(np.mean([metrics[method]['seeds'][str(b)]['energy']/max(metrics['random']['seeds'][str(b)]['energy'], 1e-12) for b in p['blocks']]))
        gates[method] = dict(promising_development_only=not p['smoke'] and comparison['mean'] >= p['gates']['development_gain'] and
                    comparison['positive'] >= p['gates']['positive_blocks'] and energy <= p['gates']['energy_ratio'] and
                    all(d['p95'] < p['gates']['full_policy_p95'] for d in timing_values[method]['seeds'].values()), energy_ratio_vs_random=energy)
    compute = {}
    for b in p['blocks']:
        d = read(task_folder(args, ('teacher', 'multistep', b, 0, -1, -1))/'teacher.json')
        original = read(args.output/f'dataset/block_{b}/result.json')
        compute[str(b)] = dict(old_teacher_matching_calls=original.get('teacher_matching_calls'),
                              new_teacher_matching_calls=d['matching_calls'], new_witnessed_labels=d['labels'],
                              new_teacher_graphs=d['graphs'], improved_graphs=sum(q['improved'] for q in d['details']),
                              unresolved_graphs=sum(q['unresolved_no_improvement'] for q in d['details']),
                              training={v: {k: read(task_folder(args, ('train', v, b, 0, -1, -1))/'result.json')[k]
                                            for k in ('supervised_sample_count', 'unique_labels', 'iterations')} for v in TRAIN_VARIANTS})
    result = dict(status='PASS', purpose=p['purpose'], task_count=len(observed), protocol_sha256=sha(args.results_root/'protocol.json'),
                  metrics=metrics, timing=timing_values, comparisons=diagnostics, gates=gates,
                  paired_external_streams=True, paired_training_opportunities=True,
                  boundary=p['claim_boundary'], all_results_reported=True, no_independent_confirmation_launched=True,
                  offline_compute=compute)
    atomic_json(args.results_root/'analysis/summary.json', result)
    lines = ['# 学习算法优化开发实验统一分析', '', p['claim_boundary'], '',
             '原独立确认结果保持冻结；本轮五个开发训练块用于诊断和筛选。随机重复与快照不是独立训练重复。', '',
             '| 方法 | DSR | 接纳率 | 接纳后按时率 | UAV能耗/步 | MBS占比 | 公平性 | 匹配次数 | 完整策略p95均值/s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        d = metrics[m]['means']; lines.append(f"| {m} | {d['dsr']:.4%} | {d['admission']:.4%} | {d['admitted_quality']:.4%} | {d['energy']:.6f} | {d['mbs']:.4%} | {d['fairness']:.4f} | {d['matching_calls']:.3f} | {timing_values[m]['mean_seed_p95']:.4f} |")
    lines += ['', '完整策略p95先按训练块计算，再取五块均值；在线评估计时与隔离快照计时不可混用。', '', '## 预先指定的DSR比较', '']
    for family in ('primary', 'mechanism'):
        for name, d in diagnostics[family].items():
            ci = '烟雾测试不估计' if d['ci95'] is None else f"[{100*d['ci95'][0]:+.3f}, {100*d['ci95'][1]:+.3f}]"
            lines.append(f"- {name}: {100*d['mean']:+.3f} 个百分点，95% t区间 {ci}，正向 {d['positive']}/{d['n']}；Holm p={d.get('holm_p')}，精确符号翻转 p={d['exact_signflip_p']}。")
    lines += ['', '## 早停诊断', '']
    for name, d in diagnostics['stop']['pairs'].items(): lines.append(f"- {name}: 每快照实际超时数差 {d['mean']:+.3f}，95% t区间 {d['ci95']}（负值较好；固定快照结果）。")
    lines += ['', '## 离线教师计算与标签复用', '',
              '| 训练块 | 原教师匹配调用 | 新教师匹配调用 | 有改进路径图/总图 | 原始新标签 |',
              '|---|---:|---:|---:|---:|']
    for b, d in compute.items(): lines.append(f"| {b} | {d['old_teacher_matching_calls']} | {d['new_teacher_matching_calls']} | {d['improved_graphs']}/{d['new_teacher_graphs']} | {d['new_witnessed_labels']} |")
    lines += ['', '标签抽样/复用数、独有标签数和每次PPO的图索引及种子见原始JSON；多步教师的离线开销不能算成在线学习加速。', '',
              '## 开发筛选', '', '```json', json.dumps(gates, ensure_ascii=False, indent=2), '```', '',
              '负结果同样进入本报告。开发门槛不构成论文性能证明；如筛选出新候选，应另行冻结设计并用新训练种子独立确认。', '']
    args.deliverables.mkdir(parents=True, exist_ok=True); (args.deliverables/'analysis.md').write_text('\n'.join(lines))
    atomic_json(args.results_root/'status.json', dict(status='COMPLETED', tasks=len(observed), analysis=str(args.results_root/'analysis/summary.json'), updated_unix=time.time()))


def pipeline(args):
    if sys.platform != 'linux': raise RuntimeError('Pipeline must run on the Linux experiment server')
    args.results_root.mkdir(parents=True, exist_ok=True)
    profile = read(args.resources/'profile.json')
    os.sched_setaffinity(0, set(profile['cpus'][:profile['slots']]))
    res = load('learning_controller_resources', args.extension_root/'utils/sf_hlm_resources.py')
    with res.task_lock(args.results_root/'controller.lock'):
        path = args.results_root/'protocol.json'
        if path.exists(): p = read(path); check_release(args, p)
        else: p = protocol(args); atomic_json(path, p)
        stages = [(['capture'], None), (['diagnostic'], None), (['teacher'], None),
                  (['train'], ['baseline', 'multistep']),
                  (['evaluate'], [m for m in METHODS if not m.startswith('effects_')]),
                  (['train'], ['effects']), (['evaluate'], ['effects_bc', 'effects_ppo']), (['timing'], None)]
        try:
            for phases, methods in stages:
                print(json.dumps(dict(event='stage', phases=phases, methods=methods, unix=time.time())), flush=True)
                atomic_json(args.results_root/'status.json', dict(status='RUNNING', phase='+'.join(phases), pid=os.getpid(), updated_unix=time.time()))
                dispatch(args, p, (phases, methods))
            analyze(args, p)
        except BaseException as error:
            atomic_json(args.results_root/'status.json', dict(status='FAILED', error=str(error), pid=os.getpid(),
                        updated_unix=time.time(), resume='Re-run the identical pipeline command; live workers are adopted and verified tasks skipped.'))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('pipeline', 'worker', 'analyze', 'status'))
    for key in PATH_ARGS:
        default = REPO_ROOT if key == 'release_root' else REPO_ROOT/'results/sf_hlm_learning_improvement_20261003' if key == 'results_root' else REPO_ROOT/'output/sf_hlm_learning_improvement_20261003/deliverables' if key == 'deliverables' else None
        p.add_argument('--'+key.replace('_', '-'), type=Path, default=default)
    p.add_argument('--task'); p.add_argument('--smoke', action='store_true'); p.add_argument('--workers', type=int, default=192)
    p.add_argument('--timing-workers', type=int, default=8)
    return p


def main():
    args = parser().parse_args()
    for key in PATH_ARGS:
        if getattr(args, key) is not None: setattr(args, key, getattr(args, key).resolve())
    if args.mode == 'status':
        print(json.dumps(read(args.results_root/'status.json'), ensure_ascii=False, indent=2)); return
    if any(getattr(args, k) is None for k in PATH_ARGS): raise ValueError('All frozen external input paths must be explicitly specified')
    if args.workers < 1 or not 1 <= args.timing_workers <= 8: raise ValueError('Invalid concurrency')
    if args.mode == 'worker': worker(args)
    elif args.mode == 'analyze':
        p = read(args.results_root/'protocol.json'); check_release(args, p); analyze(args, p)
    else: pipeline(args)


if __name__ == '__main__': main()
