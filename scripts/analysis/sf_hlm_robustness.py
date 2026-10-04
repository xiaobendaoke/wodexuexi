"""Frozen-policy scenario evaluation and isolated snapshot budget diagnostics.

All numerical work runs on Linux. Historical releases, models and results stay
immutable; the nominal grid cell references the completed confirmation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time

METHODS = ('learned', 'heuristic', 'random', 'greedy', 'original_frozen', 'original_adapted')
SEARCH_METHODS = METHODS[:4]
PHASES = ('capture', 'robustness', 'benchmark')
FIELDS = ('phase', 'variant', 'method', 'block', 'workload', 'episode', 'step')
MATCHING_BUDGET = 16
NOMINAL = 'ue_100_compute_100'
VARIANTS = {f'ue_{n}_compute_{int(s*100)}': dict(ue_count=n, compute_scale=s)
    for n in (60, 100, 140) for s in (.6, 1., 1.4)}
PATH_ARGS = ('source_root', 'suite_root', 'extension_root', 'output', 'legacy_results',
    'resources', 'adaptation_root', 'followup_root', 'confirmation_root', 'results_root', 'release_root')
FILES = ('scripts/analysis/sf_hlm_robustness.py', 'tests/test_sf_hlm_robustness.py')


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def registered_tasks(blocks, workloads, episodes, points, capture_episodes=2):
    tasks = [('capture', NOMINAL, 'learned', b, w, e, -1)
        for b in blocks for w in workloads for e in range(capture_episodes)]
    tasks += [('robustness', variant, m, b, w, e, -1) for variant in VARIANTS
        if variant != NOMINAL for m in METHODS for b in blocks for w in workloads for e in range(episodes)]
    tasks += [('benchmark', NOMINAL, 'all', b, w, e, step)
        for b in blocks for w in workloads for e in range(capture_episodes) for step in points]
    if len(tasks) != len(set(tasks)): raise ValueError('Duplicate registered task')
    return tasks


def conditions(): return [('free', None), ('fixed_mbs', None), ('wall', .02), ('wall', .05), ('wall', .1)]


def remaining_wall(wall, started, now):
    if not math.isfinite(wall) or wall <= 0: raise ValueError('Invalid wall budget')
    return max(0., wall-(now-started))


def decode_rows(rows):
    rows = copy.deepcopy(rows)
    for row in rows: row['options'] = {int(k): v for k,v in row['options'].items()}
    return rows


def eligible(tasks, completed, active):
    timing_ready = all(t in completed for t in tasks if t[0] != 'benchmark')
    return [t for t in tasks if t not in completed and t not in active
        and (t[0] != 'benchmark' or timing_ready)]


def submissions(ready, active, capacity, timing_capacity):
    if min(capacity, timing_capacity) < 1: raise ValueError('Invalid capacity')
    room = max(0, capacity-len(active))
    timing_room = max(0, timing_capacity-sum(t[0] == 'benchmark' for t in active))
    chosen = []
    for task in ready:
        if len(chosen) >= room: break
        if task[0] == 'benchmark':
            if not timing_room: continue
            timing_room -= 1
        chosen.append(task)
    return chosen


def folder(args, task):
    phase, variant, method, block, workload, episode, step = task
    path = args.results_root/phase/variant/method/f'block_{block}_workload_{workload}'/f'episode_{episode:05d}'
    return path if step < 0 else path/f'step_{step:05d}'


def task_contract(args, p, task):
    return dict(zip(FIELDS, task), protocol_sha256=sha(args.results_root/'protocol.json'),
        parent_protocol_sha256=p['parent_protocol_sha256'], source_hash=p['source_hash'])


def verify(args, p, task):
    path = folder(args, task)
    if not (path/'result.json').exists(): return False
    d = read(path/'result.json')
    if d['status'] != 'PASS' or d['contract'] != task_contract(args, p, task):
        raise ValueError('Result identity differs: '+str(task))
    if task[0] in ('robustness', 'capture') and d['steps'] != p['steps']:
        raise ValueError('Partial episode receipt')
    if task[0] == 'capture' and d['snapshots'] != len(p['snapshot_steps']):
        raise ValueError('Partial snapshot population')
    if task[0] == 'benchmark' and d['conditions'] != len(SEARCH_METHODS)*len(conditions()):
        raise ValueError('Partial benchmark population')
    for name, digest in d['files'].items():
        if sha(path/name) != digest: raise ValueError('Result changed: '+str(path/name))
    return True


def receipt(r, path, contract, **extra):
    files = {str(f.relative_to(path)): sha(f) for f in path.rglob('*') if f.is_file()
        and f.name != 'result.json' and not f.name.startswith('.') and not f.name.endswith('.tmp')}
    r.res.atomic_json(path/'result.json', dict(status='PASS', contract=contract, files=files, **extra))


def make_protocol(args):
    parent = read(args.confirmation_root/'protocol.json')
    status = read(args.confirmation_root/'status.json')
    result = read(args.confirmation_root/'analysis/confirmation.json')
    if (status['phase'] != 'CONFIRMATION_COMPLETE' or parent['smoke']
            or not result['outcomes']['learned']['primary_passed']):
        raise ValueError('A completed, successful frozen confirmation is required')
    if tuple(result['groups']) != METHODS: raise ValueError('Parent methods differ')
    blocks = parent['confirmation_blocks'][:1] if args.smoke else parent['confirmation_blocks']
    workloads = parent['confirmation_workloads'][:1] if args.smoke else parent['confirmation_workloads']
    points = [0, 4] if args.smoke else parent['snapshot_steps']
    episodes = 1 if args.smoke else 6
    captures = 1 if args.smoke else 2
    tasks = registered_tasks(blocks, workloads, episodes, points, captures)
    return dict(version='sf_hlm_frozen_post_confirmation_20261003_v1', smoke=args.smoke,
        source_hash=parent['source_hash'], parent_protocol_sha256=sha(args.confirmation_root/'protocol.json'),
        parent_quality_sha256=sha(args.confirmation_root/'analysis/confirmation_eval.json'),
        parent_final_sha256=sha(args.confirmation_root/'analysis/confirmation.json'),
        files={f:sha(args.release_root/f) for f in FILES}, variants=VARIANTS, blocks=blocks,
        workloads=workloads, snapshot_steps=points, steps=8 if args.smoke else 1000,
        episodes=episodes, capture_episodes=captures, tasks=[list(t) for t in tasks], matching_budget=MATCHING_BUDGET,
        ordinary_repair_budget=128, methods=list(METHODS), snapshot_methods=list(SEARCH_METHODS),
        wall_budgets=[.02,.05,.1], wall_policy='Whole policy path, including input preparation, graph/actor inference, search, mapping and message serialization. Original 16/128 call caps retained. Soft deadline; every overrun reported.',
        nominal_reuse='Use unchanged confirmation_eval raw episodes; no nominal grid rerun or new independent training.',
        fixed_mbs='Raw greedy cardinality in the identical frozen state; all initialization/search/fallback maintain it. Snapshot diagnosis only.',
        frozen_policy=True, retraining=False, upper_input_capacity=30,
        uav_compute_scale_only=True, mbs_compute_unchanged=True, reward_unchanged=True,
        natural_admission=True, original_physical_model=True, future_requests_not_observed=True,
        primary_family='24 paired seed comparisons: learned minus G/original-frozen/original-adapted in eight new scenarios, one Holm family.',
        mechanism_family='16 paired seed comparisons: learned minus H/R in eight new scenarios, separate Holm family.',
        snapshot_family='Free/fixed-MBS/wall miss reductions and timing, ten seed summaries; supplementary diagnosis, never long-sequence DSR.',
        information_cost_unchanged='Current ideal coordination; no new communication deployment claim.',
        independent_training_units=len(blocks), not_new_training_confirmation=True,
        smoke_cannot_support_claims=args.smoke)


def backend(args):
    return load('post_confirmation_followup', args.followup_root/'scripts/analysis/sf_hlm_followup.py')


def verify_releases(args,p):
    if sha(args.confirmation_root/'protocol.json') != p['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    parent=read(args.confirmation_root/'protocol.json')
    for files,root in [(p['files'],args.release_root),(parent['files'],args.followup_root),
            (parent['extension_files'],args.extension_root),(parent['adaptation_files'],args.adaptation_root)]:
        for name,h in files.items():
            if sha(root/name) != h:raise ValueError('Frozen release changed: '+str(root/name))
    for name,key in [('confirmation_eval.json','parent_quality_sha256'),('confirmation.json','parent_final_sha256')]:
        if sha(args.confirmation_root/'analysis'/name) != p[key]:raise ValueError('Parent analysis changed')


def basis_args(args):
    a = copy.copy(args); a.results_root = args.confirmation_root
    a.smoke = False; a.seed = 0; a.family = 'complete'; a.phase = 'confirmation_eval'
    return a


def boot(args):
    if sys.platform != 'linux': raise RuntimeError('Numerical work is remote Linux only')
    a = basis_args(args); m = backend(a); r = m.runner(a); old, _ = m.initialize(a, r)
    parent = read(args.confirmation_root/'protocol.json')
    models, _, check, _ = m.basis(a, r, old, parent)
    lower, lower_sha = m.quota(a, r, old, parent)
    if args.method == 'original_adapted':
        location = args.confirmation_root/'confirmation_original_training/original_adapted'/f"block_{args.block}_workload_{parent['lower_training_workloads'][str(args.block)]}"
        m.verified(location, full=False); models['lower'].load(str(location/'checkpoint/lower'))
    return a, m, r, parent, models, lower, dict(basis_checkpoint_sha256=sha(check/'contract.json'), quota_checkpoint_sha256=lower_sha)


def apply_variant(variant):
    import config
    from utils.logger import refresh_derived_config_fields
    upper_capacity, upper_dim = config.MAX_ASSOCIATED_UES, config.OBS_DIM_SINGLE
    mbs = config.MBS_COMPUTING_CAPACITY
    config.NUM_UES = variant['ue_count']
    config.UAV_COMPUTING_CAPACITY = config.UAV_COMPUTING_CAPACITY*variant['compute_scale']
    refresh_derived_config_fields()
    config.MAX_ASSOCIATED_UES = upper_capacity; config.OBS_DIM_SINGLE = upper_dim
    if config.MBS_COMPUTING_CAPACITY != mbs: raise RuntimeError('MBS compute was changed')


def totals(): return dict.fromkeys(('generated','admitted','success','potential','energy','mbs','fairness'), 0.)


def add_totals(total, metrics, potential):
    total['potential'] += potential
    for key,name in [('generated','generated_service_count'),('admitted','admitted_service_count'),
            ('success','deadline_satisfied_count'),('energy','energy'),('mbs','service_offloads_mbs'),('fairness','fairness')]:
        total[key] += float(metrics[name])


def evaluate_episode(args, p, contract, capture=False):
    import numpy as np
    a,m,r,parent,models,lower,model_contract = boot(args)
    import config
    from utils.canonical_experiment import digest, exogenous_frame, config_snapshot
    from utils.snapshot_evaluator import capture as capture_state
    if not capture: apply_variant(p['variants'][args.variant])
    path = folder(args, tuple(getattr(args,k) for k in FIELDS)); path.mkdir(parents=True, exist_ok=True)
    original = read(args.confirmation_root/'confirmation_eval/learned'/f'block_{args.block}_workload_{args.workload}'/f'episode_{args.episode:05d}'/'episode.json') if capture else None
    env,obs = r.episode_env(args.workload,args.episode)
    frozen = r.base.policy_fingerprint([models['upper'],models['lower'],lower])
    rng = np.random.default_rng(r.res.trajectory_seed(sha(args.confirmation_root/'protocol.json'), args.block,args.workload,args.episode,'search'))
    trace = hashlib.sha256(); records = []; total = totals(); count = 0
    for step in range(p['steps']):
        trace.update(digest(exogenous_frame(env)).encode())
        potential = sum(u.latent_request[0] < config.NUM_SERVICES for u in env.ues)
        motion,actions,info,_,_,rows = m.policy(a,r,parent,models,env,obs,lower=lower,rng=rng)
        if capture and step in p['snapshot_steps']:
            snap = capture_state(env,motion)
            if tuple(x[2] for x in snap.requests) != tuple(info['request_ids']): raise ValueError('Snapshot request order changed')
            state = path/f'step_{step:05d}'; state.mkdir(parents=True, exist_ok=True)
            # Only our own task-generated, manifest-verified payloads are read.
            # Observed input state stays separate from the private physics copy.
            (state/'observed_state.bin').write_bytes(pickle.dumps(env, protocol=5))
            (state/'physics_state.bin').write_bytes(snap.payload)
            r.res.atomic_json(state/'snapshot.json',dict(config_hash=snap.config_hash,
                requests=snap.requests,candidates=snap.candidates,motion=snap.motion,
                snapshot_hash=snap.snapshot_hash,obs=obs,rows=rows,rows_hash=digest(rows),
                snapshot_id=[args.block,args.workload,args.episode,step],trusted_task_generated=True))
            count += 1
        obs,_,metrics = env.step(motion,offloading_actions=actions); obs = np.asarray(obs,np.float32)
        r.base.assert_step(metrics,info)
        if capture:
            before = original['steps'][step]
            if metrics != before['metrics'] or list(info['assignment']) != before['assignment']:
                raise ValueError('Confirmation replay changed')
        add_totals(total,metrics,potential)
        records.append(dict(step=step,metrics=metrics,potential=potential,**info))
    if frozen != r.base.policy_fingerprint([models['upper'],models['lower'],lower]): raise ValueError('Frozen models changed')
    if capture and not p['smoke'] and trace.hexdigest() != original['exogenous_trace_hash']: raise ValueError('Replay external stream differs')
    r.res.atomic_json(path/'episode.json',dict(summary=total,steps=records,exogenous_trace_hash=trace.hexdigest(),
        evaluation_config_hash=digest(config_snapshot()),frozen_policy=True,software_timing_condition='shared_cluster'))
    receipt(r,path,contract,steps=p['steps'],snapshots=count,model_fingerprint=frozen,**model_contract)


def timed_policy(env,obs,kind,models,lower,rng,r,wall=None,fixed=False):
    import config
    from utils.deadline_repair_online import greedy_assignment
    from utils.deadline_repair import assess
    from utils.sf_hlm_matching import feedback
    from utils.sf_hlm_v2 import search,validate_metadata
    from utils.sf_hlm_followup import fixed_mbs_search
    from marl_models.sf_hlm_v2 import service_view
    if wall is None and not fixed:
        return r.policy(env,obs,kind,models,family='complete',lower=lower,rng=rng,budget=MATCHING_BUDGET)
    started = time.perf_counter()
    _,masks,requests,rows = r.base.prepare(env); validate_metadata(rows)
    if fixed:
        assignment,d,_ = fixed_mbs_search(rows,num_uavs=config.NUM_UAVS,kind=kind,model=lower,rng=rng,budget=MATCHING_BUDGET)
    else:
        remaining = remaining_wall(wall,started,time.perf_counter())
        if remaining == 0:
            assignment,_ = greedy_assignment(rows); reference = assess(rows,assignment)
            d = dict(matching_calls=0,matching_budget=MATCHING_BUDGET,
                reference_predicted=reference,chosen_predicted=reference,common_energy_cap=1.05*reference['energy'],termination='input_wall_budget')
        elif kind == 'greedy':
            initial,_ = greedy_assignment(rows)
            remaining = remaining_wall(wall,started,time.perf_counter())
            assignment,d = r.base.common_repair(rows,initial,budget=128,wall_seconds=remaining)
        else:
            assignment,d,_ = search(rows,num_uavs=config.NUM_UAVS,kind=kind,family='complete',model=lower,
                rng=rng,budget=MATCHING_BUDGET,wall_seconds=remaining)
    summary = feedback(rows,assignment,config.NUM_UAVS,energy_reference=config.OFFLOAD_ENERGY_NORM_REF)
    view = service_view(env,summary)
    motion = models['upper'].select_actions(obs,exploration=False)
    actions = r.base.actions_from_assignment(requests,assignment,masks)
    upload = len(json.dumps(dict(rows=rows),default=r.base.serial,separators=(',',':')).encode())
    download = len(json.dumps(dict(ue_ids=[x[2] for x in requests],actions=assignment,feedback=summary.tolist()),default=r.base.serial,separators=(',',':')).encode())
    info = dict(request_ids=[x[2] for x in requests],assignment=assignment,diagnostic=d,
        logical_upload_bytes=upload,logical_download_bytes=download,policy_seconds=time.perf_counter()-started)
    import numpy as np
    return np.clip(motion,-1,1),actions,info,dict(obs=obs,feedback=view),None,rows


def benchmark(args,p,contract):
    import numpy as np
    a,m,r,parent,models,lower,model_contract = boot(args)
    from utils.snapshot_evaluator import AssignmentSnapshot,evaluate
    from utils.canonical_experiment import digest
    capture_task = ('capture',NOMINAL,'learned',args.block,args.workload,args.episode,-1)
    if not verify(args,p,capture_task): raise ValueError('Missing verified snapshot capture')
    source = folder(args,capture_task)/f'step_{args.step:05d}'; meta = read(source/'snapshot.json')
    if not meta['trusted_task_generated']: raise ValueError('Untrusted snapshot input')
    snap = AssignmentSnapshot((source/'physics_state.bin').read_bytes(),meta['config_hash'],
        tuple(tuple(x) for x in meta['requests']),tuple(tuple(x) for x in meta['candidates']),tuple(tuple(x) for x in meta['motion']))
    if snap.snapshot_hash != meta['snapshot_hash']: raise ValueError('Snapshot identity differs')
    observed_payload = (source/'observed_state.bin').read_bytes()
    obs = np.asarray(meta['obs'],np.float32); frozen = r.base.policy_fingerprint([models['upper'],models['lower'],lower])
    seed = r.res.trajectory_seed(sha(args.results_root/'protocol.json'),args.block,args.workload,args.episode,args.step,'benchmark_search')
    # Warm imports and both frozen actors before any recorded policy timer.
    timed_policy(pickle.loads(observed_payload),obs,'learned',models,lower,np.random.default_rng(seed),r)
    jobs = [(kind,condition,wall) for condition,wall in conditions() for kind in SEARCH_METHODS]
    order = np.random.default_rng(r.res.trajectory_seed(seed,'order')).permutation(len(jobs)); records = []
    fixed_counts = []
    for index in order:
        kind,condition,wall = jobs[int(index)]
        rng = np.random.default_rng(r.res.trajectory_seed(seed,kind))
        motion,_,info,_,_,rows = timed_policy(pickle.loads(observed_payload),obs,kind,models,lower,rng,r,
            wall=wall,fixed=condition=='fixed_mbs')
        if digest(rows) != meta['rows_hash']: raise ValueError('Observed graph changed on snapshot restoration')
        if tuple(info['request_ids']) != tuple(x[2] for x in snap.requests): raise ValueError('Restored request order differs')
        if not np.allclose(motion,snap.motion,rtol=0,atol=1e-7): raise ValueError('Frozen current motion differs')
        actual = evaluate(snap,info['assignment']); d = info['diagnostic']
        if d['matching_calls'] > MATCHING_BUDGET or d['chosen_predicted']['energy'] > d['common_energy_cap']+1e-7: raise ValueError('Guard failed')
        mbs_count = sum(x == 1 for x in info['assignment'])
        if condition == 'fixed_mbs':
            if mbs_count != d['fixed_mbs_count']: raise ValueError('Fixed MBS cardinality differs')
            fixed_counts.append(mbs_count)
        elapsed = info['policy_seconds']
        records.append(dict(method=kind,condition=condition,wall_budget=wall,policy_seconds=elapsed,
            soft_overrun_seconds=0. if wall is None else max(0.,elapsed-wall),assignment=info['assignment'],
            requests=len(snap.requests),actual_misses=sum(not x['success'] for x in actual['requests']),
            actual_assignment_energy=float(actual['metrics']['lower_assignment_energy']),
            actual_uav_fleet_energy=float(actual['metrics']['uav_fleet_total_energy']),mbs_count=mbs_count,
            matching_calls=d['matching_calls'],diagnostic=d,snapshot_hash=snap.snapshot_hash))
    if {(x['method'],x['condition'],x['wall_budget']) for x in records} != set(jobs):
        raise ValueError('Benchmark conditions differ from registration')
    if len(set(fixed_counts)) != 1 or len(fixed_counts) != 4: raise ValueError('Common MBS quantities differ across methods')
    if frozen != r.base.policy_fingerprint([models['upper'],models['lower'],lower]): raise ValueError('Benchmark changed frozen models')
    path = folder(args,tuple(getattr(args,k) for k in FIELDS)); path.mkdir(parents=True,exist_ok=True)
    r.res.atomic_json(path/'benchmark.json',dict(records=records,snapshot_id=meta['snapshot_id'],snapshot_hash=snap.snapshot_hash,
        current_step_only=True,online_call_caps_preserved=True,software_timing_condition='reserved_physical_cores',order=order))
    receipt(r,path,contract,conditions=len(records),model_fingerprint=frozen,**model_contract)


def worker(args):
    if sys.platform != 'linux': raise RuntimeError('Remote Linux only')
    p = read(args.results_root/'protocol.json'); task = tuple(getattr(args,k) for k in FIELDS)
    if list(task) not in p['tasks']: raise ValueError('Unregistered task')
    verify_releases(args,p)
    r = backend(args).runner(basis_args(args))
    path = folder(args,task); path.mkdir(parents=True,exist_ok=True)
    with r.res.task_lock(path/'.task.lock'):
        if verify(args,p,task): return
        resource = args.resources/'isolated' if args.phase == 'benchmark' else args.resources
        with r.res.CpuLease(resource,task=['robustness_20261003',*task],kind='environment'):
            contract = task_contract(args,p,task)
            if args.phase == 'benchmark': benchmark(args,p,contract)
            else: evaluate_episode(args,p,contract,capture=args.phase=='capture')


def command(args,task):
    cmd = [sys.executable,str(args.release_root/FILES[0]),'worker']
    for k in PATH_ARGS: cmd += ['--'+k.replace('_','-'),str(getattr(args,k))]
    for k,v in zip(FIELDS,task): cmd += ['--'+k,str(v)]
    return cmd


def live_workers(args,tasks,res,proc=Path('/proc')):
    found = {}
    for f in proc.glob('[0-9]*/cmdline'):
        try: words=f.read_bytes().decode().strip('\0').split('\0')
        except (OSError,UnicodeError): continue
        if words[1:3] != [str(args.release_root/FILES[0]),'worker']: continue
        flags = dict(zip(words[3::2],words[4::2]))
        if flags.get('--results-root') != str(args.results_root): continue
        task = tuple(flags['--'+k] if k in FIELDS[:3] else int(flags['--'+k]) for k in FIELDS)
        pid = int(f.parent.name); identity = res.process_identity(pid)
        if identity is None: continue
        if task not in tasks or task in found: raise ValueError('Duplicate/unregistered in-flight task')
        found[task] = (pid,identity)
    return found


def execute(args,p,task,res,owner=None):
    if owner:
        while res.process_identity(owner[0]) == owner[1]: time.sleep(2)
        if verify(args,p,task): return task
    log = args.results_root/'job_logs'/('_'.join(map(str,task))+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream: subprocess.run(command(args,task),cwd=args.source_root,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not verify(args,p,task): raise RuntimeError('Worker has no verified completion')
    return task


def aggregate(args,p,r):
    import numpy as np
    r.base.np=np; units={}; traces={}; raw={}; benchmark_units={}
    for t in p['tasks']:
        task=tuple(t)
        if task[0] == 'capture': continue
        if not verify(args,p,task): raise ValueError('Aggregation population incomplete')
        path=folder(args,task)
        if task[0] == 'robustness':
            row=read(path/'episode.json');s=row['summary'];key=(task[1],task[4],task[5])
            pair=row['exogenous_trace_hash'],s['potential']
            if traces.setdefault(key,pair) != pair: raise ValueError('Scenario external pairing differs')
            unit=units.setdefault((task[1],task[2],task[3]),dict(total=totals(),times=[],steps=0,bias=0.,mae=0.))
            if len(row['steps']) != p['steps']: raise ValueError('Partial scenario episode')
            for k,v in s.items():unit['total'][k]+=v
            for step in row['steps']:
                met=step['metrics'];diag=step['diagnostic']
                if not 0<=met['deadline_satisfied_count']<=met['admitted_service_count']<=met['generated_service_count']<=step['potential']: raise ValueError('Counters invalid')
                if met['forced_service_admissions'] or len(step['assignment'])!=met['admitted_service_count']: raise ValueError('Admission/assignment invalid')
                if diag['matching_calls']>16 or diag['chosen_predicted']['energy']>diag['common_energy_cap']+1e-7:raise ValueError('Online guard invalid')
                error=diag['chosen_predicted']['misses']-(met['admitted_service_count']-met['deadline_satisfied_count'])
                unit['bias']+=error;unit['mae']+=abs(error);unit['times'].append(step['policy_seconds']);unit['steps']+=1
            for k,n in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy')]:
                if not np.isclose(s[k],sum(x['metrics'][n] for x in row['steps']),rtol=1e-10):raise ValueError('Scenario summary counters differ')
            raw[str(path/'episode.json')]=sha(path/'episode.json')
        else:
            row=read(path/'benchmark.json')
            for record in row['records']:
                key=(record['condition'],record['wall_budget'],record['method'],task[3])
                benchmark_units.setdefault(key,[]).append(record)
            raw[str(path/'benchmark.json')]=sha(path/'benchmark.json')
    metrics={}
    for key,u in units.items():
        s=u['total'];metrics[key]=dict(dsr=s['success']/max(s['generated'],1),potential_dsr=s['success']/max(s['potential'],1),
            admission=s['admitted']/max(s['generated'],1),conditional=s['success']/max(s['admitted'],1),energy=s['energy'],
            mbs_share=s['mbs']/max(s['admitted'],1),fairness=s['fairness']/u['steps'],cluster_p95=float(np.quantile(u['times'],.95)),
            predicted_miss_bias=u['bias']/u['steps'],predicted_miss_mae=u['mae']/u['steps'])
    variants={};primary={};mechanisms={}
    for variant in VARIANTS:
        if variant==NOMINAL:continue
        groups={m:{k:float(np.mean([metrics[variant,m,b][k] for b in p['blocks']])) for k in metrics[variant,m,p['blocks'][0]]} for m in METHODS}
        comparisons={}
        for m in METHODS[1:]:
            x=r.base.paired([metrics[variant,'learned',b]['dsr']-metrics[variant,m,b]['dsr'] for b in p['blocks']])
            x['energy_ratio']=float(np.mean([metrics[variant,'learned',b]['energy']/metrics[variant,m,b]['energy'] for b in p['blocks']]))
            name=variant+':learned-'+m;comparisons[name]=x
            (mechanisms if m in ('heuristic','random') else primary)[name]=x
        variants[variant]=dict(groups=groups,comparisons=comparisons)
    if not p['smoke']:
        r.holm(primary,list(primary));r.holm(mechanisms,list(mechanisms))
    snapshot={}
    for condition,wall in conditions():
        key=condition if wall is None else f'wall_{int(wall*1000)}ms';block_values={}
        for m in SEARCH_METHODS:
            for b in p['blocks']:
                rows=benchmark_units[condition,wall,m,b]
                block_values[m,b]=dict(actual_misses=float(np.mean([x['actual_misses'] for x in rows])),
                    actual_assignment_energy=float(np.mean([x['actual_assignment_energy'] for x in rows])),
                    mean_seconds=float(np.mean([x['policy_seconds'] for x in rows])),p95_seconds=float(np.quantile([x['policy_seconds'] for x in rows],.95)),
                    matching_calls=float(np.mean([x['matching_calls'] for x in rows])),mbs_count=float(np.mean([x['mbs_count'] for x in rows])),
                    overrun_fraction=float(np.mean([x['soft_overrun_seconds']>0 for x in rows])),maximum_overrun=max(x['soft_overrun_seconds'] for x in rows))
        groups={m:{k:float(np.mean([block_values[m,b][k] for b in p['blocks']])) for k in block_values[m,p['blocks'][0]]} for m in SEARCH_METHODS}
        snapshot[key]=dict(groups=groups,comparisons={'learned-'+m:r.base.paired([block_values['learned',b]['actual_misses']-block_values[m,b]['actual_misses'] for b in p['blocks']]) for m in SEARCH_METHODS[1:]},snapshot_only=True)
    out=dict(status='PASS',smoke=p['smoke'],smoke_cannot_support_claims=p['smoke'],frozen_policy=True,
        scenario_results=variants,primary_family=list(primary),mechanism_family=list(mechanisms),
        nominal_reference=read(args.confirmation_root/'analysis/confirmation.json'),snapshot_diagnostics=snapshot,
        independent_training_seeds=len(p['blocks']),not_new_training_confirmation=True,
        raw_files=raw,scenario_timing_condition='shared_cluster_not_isolated',snapshot_timing_condition='reserved_cores',
        no_deployment_communication_claim=True)
    r.res.atomic_json(args.results_root/'analysis/robustness.json',out)
    return dict(evaluation_complete=True,performance_claims_require_review=True,smoke=p['smoke'])


def pipeline(args):
    if sys.platform != 'linux': raise RuntimeError('Remote Linux only')
    r=backend(args).runner(basis_args(args));res=r.res
    profile=read(args.resources/'profile.json');isolated=read(args.resources/'isolated/profile.json')
    os.sched_setaffinity(0,set(profile['cpus'][:profile['slots']]))
    p=make_protocol(args);file=args.results_root/'protocol.json'
    if file.exists() and read(file)!=p:raise ValueError('Frozen diagnostic protocol differs')
    verify_releases(args,p)
    res.atomic_json(file,p);tasks=[tuple(t) for t in p['tasks']]
    with res.task_lock(args.results_root/'controller.lock'):
        adopted=live_workers(args,tasks,res)
        completed={t for t in tasks if t not in adopted and verify(args,p,t)}
        futures={};failures=[];stop=threading.Event()
        telemetry=threading.Thread(target=res.telemetry,args=(args.resources,stop),daemon=True);telemetry.start()
        def status(phase):
            res.atomic_json(args.results_root/'status.json',dict(phase=phase,pid=os.getpid(),expected=len(tasks),completed=len(completed),
                expected_by_phase=dict(Counter(t[0] for t in tasks)),completed_by_phase=dict(Counter(t[0] for t in completed)),
                active_by_phase=dict(Counter(t[0] for t in futures.values())),failures=failures,unix=time.time(),
                frozen_policy=True,scientific_success_claim=False,smoke=args.smoke))
        try:
            with ThreadPoolExecutor(max_workers=profile['slots']) as pool:
                for task,owner in adopted.items():futures[pool.submit(execute,args,p,task,res,owner)]=task
                while len(completed)<len(tasks):
                    ready=[] if failures else eligible(tasks,completed,set(futures.values()))
                    for task in submissions(ready,list(futures.values()),profile['slots'],isolated['slots']):
                        futures[pool.submit(execute,args,p,task,res)]=task
                    status('RUNNING' if not failures else 'FAILED_DRAINING')
                    if not futures:
                        if failures:status('FAILED');raise RuntimeError('Diagnostic worker failure; inspect registered job logs')
                        raise RuntimeError('Dependency deadlock')
                    finished,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                    for future in finished:
                        task=futures.pop(future)
                        try:completed.add(future.result())
                        except Exception as exc:failures.append(dict(task=task,error=str(exc)[-1500:]))
            status('AGGREGATING');aggregate(args,p,r);status('COMPLETE')
        finally:stop.set();telemetry.join(timeout=5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['pipeline','worker'])
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    for key in FIELDS:parser.add_argument('--'+key,type=str if key in FIELDS[:3] else int,default=None)
    args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).expanduser().resolve())
    if args.mode=='worker':worker(args)
    else:pipeline(args)


if __name__=='__main__':main()
