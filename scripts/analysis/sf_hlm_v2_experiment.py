"""Gated SF-HLM v2 experiments with immutable releases and episode jobs.

The v1 prepare wrapper already supplies metadata. v2 checks that contract and
separates current/best graph state; it does not invent an input-missing result.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


res=load('sf_v2_resources',ROOT/'utils/sf_hlm_resources.py')
base=load('sf_v1_runner',ROOT/'scripts/analysis/sf_hlm_experiment.py')
EXTRA=('utils/sf_hlm_resources.py','utils/sf_hlm_v2.py','utils/sf_hlm_rollouts.py','marl_models/sf_hlm_v2.py',
       'scripts/analysis/sf_hlm_parallel.py','scripts/analysis/sf_hlm_calibrate.py','scripts/analysis/sf_hlm_v2_experiment.py','tests/test_sf_hlm_resources.py','tests/test_sf_hlm_v2.py')
METHODS=('full','no_feedback','heuristic','random','greedy','original')


def bootstrap(args):
    if sys.platform!='linux':raise RuntimeError('All SF-HLM simulation, learning and matching run on remote Linux')
    protocol=base.bootstrap(args)
    for name in ('utils.sf_hlm_resources','utils.sf_hlm_v2','marl_models.sf_hlm_v2'):
        load(name,args.extension_root/(name.replace('.','/')+'.py'))
    sys.path.insert(0,str(args.extension_root/'utils'))
    rollout=load('sf_hlm_rollouts',args.extension_root/'utils/sf_hlm_rollouts.py')
    import numpy as np
    import torch
    base.np=np;torch.set_num_threads(1);torch.set_num_interop_threads(1)
    return protocol,rollout


def model(family,num_uavs):
    from marl_models.sf_hlm_v2 import CompleteQuotaPPO,ArchivePolicy
    return (CompleteQuotaPPO if family=='complete' else ArchivePolicy)(num_uavs)


def lower_dir(args,family=None,variant='learned'):
    return args.output/'lower_training'/str(family or args.family)/variant/f'block_{args.block}'


def lower_contract(args,p,variant='learned'):
    return dict(protocol_sha256=base.sha(args.output/'protocol.json'),block=args.block,lower_seed=p['lower_seeds'][str(args.block)],
                family=args.family,variant=variant,input='observed_complete_current_best_graph_v2',training_only=True)


def read_lower(args,p,variant='learned'):
    import config
    m=model(args.family,config.NUM_UAVS);path=lower_dir(args,variant=variant);doc=json.loads((path/'result.json').read_text())
    if doc['checkpoint_sha256']!=base.sha(path/'checkpoint/quota_v2.pt'):raise ValueError('Lower checkpoint changed')
    m.load(path/'checkpoint',lower_contract(args,p,variant));return m


def frozen_lower(args):
    from marl_models.sf_hlm import QuotaPPO
    import config
    old=copy.copy(args);old.output=args.legacy_results;p=json.loads((old.output/'protocol.json').read_text())
    return base.load_quota(old,p)


def episode_env(workload,episode):
    import numpy as np
    from environment.env import Env
    env=Env(scenario_seed=workload)
    # Env._rng is consumed only by construction/reset (checked by regression).
    # Skipping preceding transitions preserves the same reset and external UE
    # streams as serial evaluation; it does not look ahead to future requests.
    for _ in range(episode+1):obs=env.reset()
    return env,np.asarray(obs,np.float32)


def policy(env,obs,method,models,*,family,lower=None,upper=None,rng=None,train=False,budget=16,wall=None,fixed_mbs=False):
    import numpy as np
    import config
    from utils.deadline_repair_online import greedy_assignment
    from utils.sf_hlm_matching import feedback
    from utils.sf_hlm_v2 import search,validate_metadata
    from marl_models.sf_hlm_v2 import service_view
    started=time.perf_counter();lower_obs,masks,requests,rows=base.prepare(env);validate_metadata(rows);entry=None
    if method=='original':
        raw,lp,value=base.original_proposal(models['lower'],lower_obs,masks,exploration=train)
        initial=tuple(int(raw[r[0],r[1]]) for r in requests);assignment,d=base.common_repair(rows,initial);entry=(lower_obs,raw,masks,lp,value)
    elif method=='greedy':
        initial,_=greedy_assignment(rows);assignment,d=base.common_repair(rows,initial)
    elif method=='old_frozen':
        from utils.sf_hlm_search import run_search
        assignment,d,_=run_search(rows,num_uavs=config.NUM_UAVS,kind='learned',model=lower,budget=budget)
    else:
        kind='learned' if method in ('learned','full','no_feedback','no_breakpoints','no_archive') else method
        assignment,d,_=search(rows,num_uavs=config.NUM_UAVS,kind=kind,family=family,model=lower,rng=rng,exploration=False,
            budget=budget,wall_seconds=wall,breakpoints=method!='no_breakpoints',multiple=method!='no_archive',fixed_mbs=fixed_mbs)
    summary=feedback(rows,assignment,config.NUM_UAVS,energy_reference=config.OFFLOAD_ENERGY_NORM_REF)
    view=service_view(env,summary)
    if upper is None:motion=models['upper'].select_actions(obs,exploration=False);lp=value=None
    else:motion,lp,value=upper.act(obs,view,exploration=train)
    actions=base.actions_from_assignment(requests,assignment,masks)
    upload=len(json.dumps(dict(rows=rows),default=base.serial,separators=(',',':')).encode())
    download=len(json.dumps(dict(ue_ids=[r[2] for r in requests],actions=assignment,feedback=summary.tolist()),default=base.serial,separators=(',',':')).encode())
    info=dict(request_ids=[r[2] for r in requests],assignment=assignment,diagnostic=d,logical_upload_bytes=upload,logical_download_bytes=download,
              policy_seconds=time.perf_counter()-started)
    upper_entry=dict(obs=obs.copy(),feedback=view,action=motion,log_prob=lp,value=value)
    return np.clip(motion,-1,1),actions,info,upper_entry,entry,rows


def directory(args):
    return args.output/args.phase/args.family/args.method/f'block_{args.block}_workload_{args.workload}'


def completed(path,contract):
    if not (path/'result.json').exists():return False
    doc=json.loads((path/'result.json').read_text())
    if doc['status']!='PASS' or doc['contract']!=contract:raise ValueError('Result contract mismatch')
    for rel,h in doc['files'].items():
        if base.sha(path/rel)!=h:raise ValueError('Result file changed: '+rel)
    return True


def result(path,contract,**extra):
    files={str(p.relative_to(path)):base.sha(p) for p in path.rglob('*') if p.is_file() and p.name not in ('result.json','.task.lock') and not p.name.endswith('.tmp')}
    res.atomic_json(path/'result.json',dict(status='PASS',contract=contract,files=files,**extra))


def dataset_episode(args,p,models,contract):
    import numpy as np
    import torch
    from utils.canonical_experiment import digest,exogenous_frame
    path=directory(args)/f'episode_{args.episode:05d}';path.mkdir(parents=True,exist_ok=True)
    if completed(path,contract):return
    env,obs=episode_env(args.workload,args.episode);graphs=[];trace=hashlib.sha256()
    for step in range(p['steps']):
        trace.update(digest(exogenous_frame(env)).encode());motion,actions,info,_,_,rows=policy(env,obs,'greedy',models,family='complete')
        if step in p['dataset_steps']:graphs.append(rows)
        obs,_,metrics=env.step(motion,offloading_actions=actions);base.assert_step(metrics,info);obs=np.asarray(obs,np.float32)
    torch.save(graphs,path/'graphs.pt');result(path,contract,graphs=len(graphs),trace=trace.hexdigest(),training_only=True)


def merge_dataset(args,p,rollout,lease,contract):
    import torch
    import config
    from utils.canonical_experiment import digest
    path=args.output/'dataset'/f'block_{args.block}';path.mkdir(parents=True,exist_ok=True)
    if completed(path,contract):return
    rows=[];sources={}
    for ep in range(p['dataset_episodes']):
        a=copy.copy(args);a.phase='dataset_episode';a.method='greedy';a.family='complete';a.episode=ep
        folder=directory(a)/f'episode_{ep:05d}';doc=json.loads((folder/'result.json').read_text())
        for rel,h in doc['files'].items():
            if base.sha(folder/rel)!=h:raise ValueError('Dataset fragment changed')
        rows.extend(torch.load(folder/'graphs.pt',weights_only=False));sources[str(folder/'result.json')]=base.sha(folder/'result.json')
    metadata=dict(missing_requests=0,missing_edges=0,graphs=len(rows),empty=sum(not x for x in rows))
    old_path=args.legacy_results/'dataset/greedy'/f'block_{args.block}_workload_{args.workload}'/'dataset.pt'
    if old_path.exists() and not p['smoke']:
        old=torch.load(old_path,weights_only=False)['rows']
        if digest(old)!=digest(rows):raise ValueError('Training replay differs from original observed rows')
        metadata.update(legacy_rows_identical=True,legacy_dataset_sha256=base.sha(old_path))
    selected=json.loads((args.output/'selected.json').read_text()) if (args.output/'selected.json').exists() else {}
    need_teacher=args.block in p['historical_blocks'] or selected.get('family')!='critical'
    collector=rollout.ParallelCollector(args,workers=p['rollout_workers'])
    try:
        tasks=[dict(id=(i,0),kind='teacher',rows=r,num_uavs=config.NUM_UAVS,budget=p['teacher_budget']) for i,r in enumerate(rows)] if need_teacher else []
        outputs=collector.map(tasks,lease);teacher=[x for t in outputs for x in t['teacher'][0]];calls=sum(t['teacher'][1] for t in outputs)
    finally:collector.close()
    torch.save(dict(rows=rows,teacher=teacher),path/'dataset.pt');result(path,contract,metadata=metadata,sources=sources,
        teacher_matching_calls=calls,teacher_samples=len(teacher),test_data_used=False)


def recovery_manifest(folder):
    return {str(f.relative_to(folder)):base.sha(f) for f in folder.rglob('*') if f.is_file()}


def checked_recovery(path):
    latest=json.loads((path/'latest.json').read_text());folder=path/latest['checkpoint']
    for rel,h in latest['files'].items():
        if base.sha(folder/rel)!=h:raise ValueError('Recovery checkpoint changed')
    return latest,folder


def lower_training(args,p,rollout,lease,contract):
    import numpy as np
    import torch
    import config
    path=lower_dir(args,variant=args.method);path.mkdir(parents=True,exist_ok=True)
    if completed(path,contract):return
    data_path=args.output/'dataset'/f'block_{args.block}'/'dataset.pt'
    data=torch.load(data_path,weights_only=False);rows=data['rows'];seed=p['lower_seeds'][str(args.block)]
    torch.manual_seed(seed);rng=np.random.default_rng(seed);lower=model(args.family,config.NUM_UAVS);losses=[];start=0
    checkpoint_contract=lower_contract(args,p,args.method)
    if (path/'latest.json').exists():
        latest,folder=checked_recovery(path);lower.load(folder,checkpoint_contract)
        d=torch.load(folder/'state.pt',weights_only=False);rng.bit_generator.state=d['rng'];torch.set_rng_state(d['torch_rng']);start=d['next_iteration'];losses=d['losses']
    elif args.family=='complete':losses=[lower.pretrain(data['teacher'],epochs=p['supervised_epochs'])]
    collector=rollout.ParallelCollector(args,workers=p['rollout_workers'])
    try:
        for iteration in range(start,p['lower_iterations']):
            indices=rng.integers(0,len(rows),size=32 if args.family=='complete' else 4)
            if p['smoke']:indices=indices[:1]
            tasks=rollout.rollout_tasks(lower,rows,indices,protocol_hash=base.sha(args.output/'protocol.json'),seed=seed,
                iteration=iteration,budget=p['matching_budget'],breakpoints=args.method!='no_breakpoints',multiple=args.method!='no_archive')
            outputs=collector.map(tasks,lease)
            if args.family=='complete':loss=lower.update([r for t in outputs for r in t['records']],epochs=p['ppo_epochs'])
            else:loss=lower.update_groups([[t for t in outputs if t['id'][0]==g] for g in range(len(indices))])
            loss.update(matching_calls=sum(t['diagnostic']['matching_calls'] for t in outputs),policy_version=tasks[0]['version'],trajectory_seeds=[t['seed'] for t in tasks])
            losses.append(loss);folder=path/'recovery'/f'iteration_{iteration:05d}';lower.save(folder,checkpoint_contract)
            torch.save(dict(next_iteration=iteration+1,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),losses=losses),folder/'state.pt')
            res.atomic_json(path/'latest.json',dict(checkpoint=str(folder.relative_to(path)),iterations=iteration+1,files=recovery_manifest(folder)))
            res.atomic_json(path/'progress.json',dict(iteration=iteration+1,iterations=p['lower_iterations'],loss=loss,pid=os.getpid()))
    finally:collector.close()
    lower.save(path/'checkpoint',checkpoint_contract);res.atomic_json(path/'losses.json',losses)
    result(path,contract,checkpoint_sha256=base.sha(path/'checkpoint/quota_v2.pt'),data_sha256=base.sha(data_path),lower_contract=checkpoint_contract)


def eval_episode(args,p,models,contract):
    import numpy as np
    import config
    from utils.canonical_experiment import digest,exogenous_frame
    from marl_models.sf_hlm_v2 import ServiceMAPPO,PlainMAPPO
    path=directory(args)/f'episode_{args.episode:05d}';path.mkdir(parents=True,exist_ok=True)
    if completed(path,contract):return
    lower=read_lower(args,p,args.method if args.method in ('no_breakpoints','no_archive') else 'learned') if args.method in ('learned','full','no_feedback','no_breakpoints','no_archive') else frozen_lower(args) if args.method=='old_frozen' else None
    upper=None
    if args.phase in ('pilot_eval','confirmation_eval','system_timing'):
        phase='pilot_training' if args.block in p['historical_blocks'] else 'system_training'
        train=args.output/phase/args.family/args.method/f'block_{args.block}_workload_{p["training_workloads"][str(args.block)]}'
        doc=json.loads((train/'result.json').read_text());checkpoint=train/'checkpoint'
        for rel,h in doc['checkpoint_files'].items():
            if base.sha(checkpoint/rel)!=h:raise ValueError('System checkpoint changed')
        upper=(PlainMAPPO if args.method in ('greedy','original') else ServiceMAPPO)(config.OBS_DIM_SINGLE,config.NUM_UAVS,enabled=args.method in ('full','heuristic','random'))
        upper.load(checkpoint,doc['system_contract'])
        if lower is not None:lower.load(checkpoint,doc['system_contract'])
        if args.method=='original':models['lower'].load(str(checkpoint/'original_lower'))
    env,obs=episode_env(args.workload,args.episode);trace=hashlib.sha256();actions_hash=hashlib.sha256();records=[]
    rng=np.random.default_rng(res.trajectory_seed(base.sha(args.output/'protocol.json'),args.block,args.workload,args.episode,'search'))
    fingerprint=base.policy_fingerprint([models['upper'],models['lower'],lower,upper])
    totals={k:0. for k in ('generated','admitted','success','potential','energy','assignment_energy','mbs','fairness')}
    for step in range(p['steps']):
        trace.update(digest(exogenous_frame(env)).encode());potential=sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)
        motion,actions,info,_,_,_=policy(env,obs,args.method,models,family=args.family,lower=lower,upper=upper,rng=rng,budget=p['matching_budget'])
        obs,_,metrics=env.step(motion,offloading_actions=actions);base.assert_step(metrics,info);obs=np.asarray(obs,np.float32)
        actions_hash.update(digest(info['assignment']).encode());totals['potential']+=potential
        for k,name in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy'),
            ('assignment_energy','lower_assignment_energy'),('mbs','service_offloads_mbs'),('fairness','fairness')]:totals[k]+=float(metrics[name])
        records.append(dict(step=step,metrics=metrics,potential=potential,**info))
    if base.policy_fingerprint([models['upper'],models['lower'],lower,upper])!=fingerprint:raise ValueError('Evaluation changed a frozen model')
    summary=dict(episode=args.episode,**totals,exogenous_trace_hash=trace.hexdigest(),action_trace_hash=actions_hash.hexdigest(),
        p95_seconds=float(np.quantile([r['policy_seconds'] for r in records],.95)))
    res.atomic_json(path/'episode.json',dict(summary=summary,steps=records));result(path,contract,summary=summary,policy_fingerprint=fingerprint)


def snapshot_episode(args,p,models,contract):
    import numpy as np
    import config
    from run_canonical_experiment import policy_step
    from utils.snapshot_evaluator import capture,evaluate
    from utils.sf_hlm_v2 import search
    from utils.deadline_repair_online import greedy_assignment
    from utils.formal_protocol import verify_completed_job
    from utils.canonical_experiment import digest,exogenous_frame
    path=directory(args)/f'episode_{args.episode:05d}';path.mkdir(parents=True,exist_ok=True)
    if completed(path,contract):return
    lower=read_lower(args,p);old=frozen_lower(args);env,obs=episode_env(args.workload,args.episode);records=[]
    historical=verify_completed_job(args.suite_root/'main'/base.UPPER/f'seed_{args.block}'/f'workload_{args.workload}',p['source_hash'])
    original=json.loads((historical/'episodes'/f'{args.episode:05d}.json').read_text());trace=hashlib.sha256()
    for step in range(p['steps']):
        trace.update(digest(exogenous_frame(env)).encode())
        motion,actions,_=policy_step(models,env,obs,exploration=False)
        if step in p['snapshot_steps']:
            _,masks,requests,rows=base.prepare(env);snap=capture(env,np.clip(motion,-1,1));comparisons={};curves=[]
            if tuple(r[2] for r in requests)!=tuple(r[2] for r in snap.requests):raise ValueError('Snapshot request order changed')
            for kind in ('greedy','heuristic','random','learned','old_frozen'):
                rng=np.random.default_rng(res.trajectory_seed(base.sha(args.output/'protocol.json'),args.block,args.workload,args.episode,step,kind))
                if kind=='greedy':initial,_=greedy_assignment(rows);a,d=base.common_repair(rows,initial)
                elif kind=='old_frozen':
                    from utils.sf_hlm_search import run_search
                    a,d,_=run_search(rows,num_uavs=config.NUM_UAVS,kind='learned',model=old,budget=p['matching_budget'])
                else:a,d,_=search(rows,num_uavs=config.NUM_UAVS,kind=kind,family=args.family,model=lower,rng=rng,budget=p['matching_budget'])
                actual=evaluate(snap,a);comparisons[kind]=dict(assignment=a,diagnostic=d,actual_misses=sum(not x['success'] for x in actual['requests']),
                    actual_assignment_energy=float(actual['metrics']['lower_assignment_energy']))
            for wall in p['wall_budgets']:
                for kind in ('greedy','heuristic','random','learned'):
                    start=time.perf_counter();rng=np.random.default_rng(res.trajectory_seed(args.block,args.workload,args.episode,step,kind))
                    if kind=='greedy':initial,_=greedy_assignment(rows);a,d=base.common_repair(rows,initial,budget=100000,wall_seconds=wall)
                    else:a,d,_=search(rows,num_uavs=config.NUM_UAVS,kind=kind,family=args.family,model=lower,rng=rng,budget=100000,wall_seconds=wall)
                    elapsed=time.perf_counter()-start;actual=evaluate(snap,a)
                    curves.append(dict(kind=kind,budget=wall,elapsed=elapsed,overrun=max(0.,elapsed-wall),calls=d['matching_calls'],actual_misses=sum(not x['success'] for x in actual['requests'])))
            fixed=[]
            if args.family=='critical':
                for kind in ('heuristic','random','learned'):
                    a,d,_=search(rows,num_uavs=config.NUM_UAVS,kind=kind,family=args.family,model=lower,
                        rng=np.random.default_rng(res.trajectory_seed(args.block,args.workload,step,kind,'fixed_mbs')),budget=p['matching_budget'],fixed_mbs=True)
                    actual=evaluate(snap,a);fixed.append(dict(kind=kind,mbs=sum(x==1 for x in a),actual_misses=sum(not x['success'] for x in actual['requests'])))
            records.append(dict(step=step,snapshot_hash=snap.snapshot_hash,requests=len(rows),comparisons=comparisons,wall_curves=curves,fixed_mbs=fixed))
        obs,_,metrics=env.step(np.clip(motion,-1,1),offloading_actions=actions);obs=np.asarray(obs,np.float32)
        for metric in ('generated_service_count','admitted_service_count','deadline_satisfied_count','lower_assignment_energy'):
            if not np.isclose(metrics[metric],original['steps'][step]['metrics'][metric],rtol=1e-10,atol=1e-10):raise ValueError('Historical snapshot replay differs: '+metric)
    if not p['smoke'] and trace.hexdigest()!=original['exogenous_trace_hash']:raise ValueError('Historical snapshot exogenous flow changed')
    res.atomic_json(path/'snapshots.json',records);result(path,contract,snapshots=len(records),historical_replay=True,trace=trace.hexdigest())


def system_training(args,p,models,rollout,lease,contract):
    import config
    import numpy as np
    import torch
    from environment.env import Env
    from marl_models.sf_hlm_v2 import ServiceMAPPO,PlainMAPPO
    from marl_models.buffer_and_helpers import DiscreteOffloadRolloutBuffer
    from run_hierarchical_mappo_experiment import lower_rewards_from_metrics,update_on_policy_model
    path=directory(args);path.mkdir(parents=True,exist_ok=True)
    if completed(path,contract):return
    episodes=p['pilot_episodes'] if args.phase=='pilot_training' else p['system_episodes']
    warmup=p['upper_warmup_episodes']
    upper=(PlainMAPPO if args.method in ('greedy','original') else ServiceMAPPO)(config.OBS_DIM_SINGLE,config.NUM_UAVS,enabled=args.method in ('full','heuristic','random'))
    upper.warm_start(models['upper']);lower=read_lower(args,p) if args.method in ('full','no_feedback') else None
    sys_contract=dict(**contract,episodes=episodes,lower_family=args.family,upper='service_attention' if args.method not in ('greedy','original') else 'original_attention')
    env=Env(scenario_seed=args.workload);start=0;graphs=[];summaries=[];rho_dsr=rho_mbs=0.
    if (path/'latest.json').exists():
        latest,folder=checked_recovery(path);upper.load(folder,sys_contract)
        if lower:lower.load(folder,sys_contract)
        if args.method=='original':models['lower'].load(str(folder/'original_lower'))
        d=torch.load(folder/'state.pt',weights_only=False);env=d['env'];start=d['next_episode'];summaries=d['summaries'];graphs=d['graphs'];rho_dsr,rho_mbs=d['rho']
        torch.set_rng_state(d['torch_rng']);np.random.set_state(d['numpy_rng'])
    collector=rollout.ParallelCollector(args,workers=p['rollout_workers'])
    try:
        for ep in range(start,episodes):
            obs=np.asarray(env.reset(),np.float32);entries=[];violations=[];totals={k:0. for k in ('generated','admitted','success','energy')};action_hash=hashlib.sha256()
            rng=np.random.default_rng(res.trajectory_seed(base.sha(args.output/'protocol.json'),args.block,ep,'online_search'))
            buffer=DiscreteOffloadRolloutBuffer(config.NUM_UAVS,config.OFFLOAD_OBS_DIM_SINGLE,config.MAX_OFFLOAD_REQUESTS_PER_UAV,config.OFFLOAD_NUM_ACTIONS,p['steps'],'cpu') if args.method=='original' else None
            for step in range(p['steps']):
                motion,actions,info,entry,old,rows=policy(env,obs,args.method,models,family=args.family,lower=lower,upper=upper,rng=rng,train=True,budget=p['matching_budget'])
                obs,rewards,metrics=env.step(motion,offloading_actions=actions);base.assert_step(metrics,info);obs=np.asarray(obs,np.float32)
                action_hash.update(motion.tobytes());action_hash.update(actions.tobytes());entry.update(reward=np.asarray(rewards,np.float32),done=step+1==p['steps']);entries.append(entry)
                if lower and step in p['dataset_steps']:graphs.append(rows)
                if buffer is not None:
                    reward,v=lower_rewards_from_metrics(metrics,rewards,rho_dsr,rho_mbs);violations.append(v);buffer.add(*old[:4],reward,entry['done'],old[4])
                for k,name in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy')]:totals[k]+=float(metrics[name])
            uloss=upper.update(entries,epochs=p['ppo_epochs']);lloss={}
            if buffer is not None:
                lloss=update_on_policy_model(models['lower'],buffer,np.zeros(config.NUM_UAVS,np.float32))
                rho_dsr=float(np.clip(rho_dsr+config.OFFLOAD_LAGRANGE_LR*np.mean([v['dsr_violation'] for v in violations]),0,config.OFFLOAD_LAGRANGE_MAX))
                rho_mbs=float(np.clip(rho_mbs+config.OFFLOAD_LAGRANGE_LR*np.mean([v['mbs_load_violation'] for v in violations]),0,config.OFFLOAD_LAGRANGE_MAX))
            if lower and args.phase=='system_training' and ep+1>warmup and (ep+1)%p['upper_lower_alternation_episodes']==0:
                indices=list(range(len(graphs))) if args.family=='complete' else np.random.default_rng(res.trajectory_seed(args.block,ep,'adapt_graphs')).integers(0,len(graphs),4)
                tasks=rollout.rollout_tasks(lower,graphs,indices,protocol_hash=base.sha(args.output/'protocol.json'),seed=args.block,iteration=ep,budget=p['matching_budget'])
                outputs=collector.map(tasks,lease)
                lloss=lower.update([r for t in outputs for r in t['records']],epochs=p['ppo_epochs']) if args.family=='complete' else lower.update_groups([[t for t in outputs if t['id'][0]==g] for g in range(len(indices))])
                graphs=[]
            if args.phase=='pilot_training' or ep+1<=warmup:graphs=[]
            summary=dict(episode=ep,**totals,upper_loss=uloss,lower_loss=lloss,action_trace_hash=action_hash.hexdigest());summaries.append(summary)
            res.atomic_json(path/'episodes'/f'{ep:05d}.json',summary);folder=path/'recovery'/f'episode_{ep:05d}'
            upper.save(folder,sys_contract)
            if lower:lower.save(folder,sys_contract)
            if buffer is not None:(folder/'original_lower').mkdir(exist_ok=True);models['lower'].save(str(folder/'original_lower'))
            torch.save(dict(env=env,next_episode=ep+1,graphs=graphs,summaries=summaries,rho=(rho_dsr,rho_mbs),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state()),folder/'state.pt')
            res.atomic_json(path/'latest.json',dict(checkpoint=str(folder.relative_to(path)),episodes=ep+1,files=recovery_manifest(folder)));res.atomic_json(path/'progress.json',dict(episode=ep+1,episodes=episodes,pid=os.getpid(),dsr=totals['success']/max(totals['generated'],1)))
    finally:collector.close()
    folder=path/'checkpoint';upper.save(folder,sys_contract)
    if lower:lower.save(folder,sys_contract)
    if args.method=='original':(folder/'original_lower').mkdir(exist_ok=True);models['lower'].save(str(folder/'original_lower'))
    checkpoint_files={str(x.relative_to(folder)):base.sha(x) for x in folder.rglob('*') if x.is_file()}
    result(path,contract,system_contract=sys_contract,checkpoint_files=checkpoint_files,episodes=episodes)


def worker(args):
    resource=args.resources
    if args.phase in ('snapshots','timing_eval','system_timing'):resource=args.resources/'isolated'
    kind='teacher' if args.phase=='dataset_merge' else 'rollout_'+args.family if args.phase=='lower_training' else 'environment'
    lease=res.CpuLease(resource,task=[args.phase,args.family,args.method,args.block,args.workload,args.episode],kind=kind)
    with lease:
        p,rollout=bootstrap(args)
        if [args.phase,args.family,args.method,args.block,args.workload,args.episode] not in p['tasks']:raise ValueError('Undeclared task')
        models,source_contract=base.configure(args,p,historical=args.block in p['historical_blocks'])
        contract=dict(**source_contract,phase=args.phase,family=args.family,method=args.method,workload=args.workload,episode=args.episode)
        lock=args.output/'task_locks'/('_'.join(map(str,[args.phase,args.family,args.method,args.block,args.workload,args.episode]))+'.lock')
        with res.task_lock(lock):
            if args.phase=='dataset_episode':dataset_episode(args,p,models,contract)
            elif args.phase=='dataset_merge':merge_dataset(args,p,rollout,lease,contract)
            elif args.phase=='lower_training':lower_training(args,p,rollout,lease,contract)
            elif args.phase in ('pilot_training','system_training'):system_training(args,p,models,rollout,lease,contract)
            elif args.phase=='snapshots':snapshot_episode(args,p,models,contract)
            else:eval_episode(args,p,models,contract)


def make_protocol(args):
    old=base.make_protocol(args);smoke=args.smoke;history=old['historical_blocks'];fresh=[960001] if smoke else list(range(960001,960011))
    train={str(b):old['training_workloads'][str(b)] for b in history};train.update({str(b):980001+i for i,b in enumerate(fresh)})
    lower_train={str(b):train[str(b)] for b in history}
    # Use separate fresh lower and upper workloads; shared upper physical budget.
    lower_train.update({str(b):990001+i for i,b in enumerate(fresh)})
    tasks=[];dev=old['eval_workloads'];confirm=[970001] if smoke else list(range(970001,970006))
    epcount=1 if smoke else 2;dataset_eps=1 if smoke else 25;confirm_eps=1 if smoke else 6
    for b in history+fresh:
        tasks += [['dataset_episode','complete','greedy',b,lower_train[str(b)],ep] for ep in range(dataset_eps)]
        tasks += [['dataset_merge','complete','greedy',b,lower_train[str(b)],-1]]
        for family in ('complete','critical'):
            tasks += [['lower_training',family,k,b,0,-1] for k in (('learned',) if family=='complete' or b in fresh else ('learned','no_breakpoints','no_archive'))]
    for b in history:
        for family in ('complete','critical'):
            tasks += [['snapshots',family,'all',b,w,ep] for w in old['snapshot_workloads'] for ep in range(epcount)]
            names=('greedy','heuristic','random','learned','old_frozen')+(('no_breakpoints','no_archive') if family=='critical' else ())
            tasks += [[phase,family,k,b,w,ep] for phase in ('lower_eval','timing_eval') for k in names for w in dev for ep in range(epcount)]
            tasks += [['pilot_training',family,k,b,train[str(b)],-1] for k in METHODS if k!='original']
            tasks += [[phase,family,k,b,w,ep] for phase in ('pilot_eval','system_timing') for k in METHODS if k!='original' for w in dev for ep in range(epcount)]
    for b in fresh:
        for family in ('complete','critical'):
            tasks += [['system_training',family,k,b,train[str(b)],-1] for k in METHODS]
            tasks += [[phase,family,k,b,w,ep] for phase in ('confirmation_eval','system_timing') for k in METHODS for w in confirm for ep in range(confirm_eps)]
    from utils.sf_hlm_resources import trajectory_seed
    old.update(version='sf_hlm_v2_gated_parallel',historical_blocks=history,confirmation_blocks=fresh,confirmation_workloads=confirm,
        training_workloads=train,lower_training_workloads=lower_train,lower_seeds={str(b):690001+i if b in history else 995001+fresh.index(b) for i,b in enumerate(history+fresh)},
        tasks=tasks,pilot_episodes=2 if smoke else 50,dataset_episodes=dataset_eps,confirmation_eval_episodes=confirm_eps,
        rollout_workers=2 if smoke else 32,method_identifiers={k:'sf_hlm_v2_'+k for k in METHODS},workers=64,maximum_cpu_slots=240,
        extension_files={f:base.sha(args.extension_root/f) for f in (*base.FILES,*EXTRA)},
        primary_comparisons=['full-greedy','full-original'],mechanism_comparisons=['full-no_feedback','full-heuristic','full-random'],
        input_audit_correction='v1 prepare already supplied all four observed fields; no missing-input causal claim.',
        episode_parallelism='Repeated reset prefix is equivalent: environment construction/reset RNG is not consumed in transitions.',
        gates_v2=dict(lower_gain=.003,pilot_gain=.005,confirmation_gain=.01,positive_development_blocks=4,energy_ratio=1.05,p95=1.),
        phase_policy='A; B only if A fails; C only after lower gate; D only after pilot gate. Smoke cannot promote.',
        calibration=dict(concurrency=[32,64,96,120,192,240],repeats=3,seconds=20,within_best=.95),
        state_rng_contract='SHA256(protocol,training seed,update,graph ordinal,trajectory ordinal); results sorted before optimizer.')
    reserved=set(old['historical_blocks']+old['eval_workloads']+old['snapshot_workloads'])
    oldp=json.loads((args.legacy_results/'protocol.json').read_text())
    reserved.update(oldp['confirmation_blocks']+oldp['confirmation_workloads'])
    new=set(fresh+confirm+list(range(980001,980001+len(fresh)))+list(range(990001,990001+len(fresh))))
    if reserved&new:raise ValueError('Independent seed namespace conflicts with history')
    return old


def run_tasks(args,p,phase,family='complete',blocks=None):
    selected=[t for t in p['tasks'] if t[0]==phase and t[1]==family and (blocks is None or t[3] in blocks)]
    failures=[];done=0;started=time.time()
    def one(t):
        phase,family,method,block,workload,episode=t;cmd=[sys.executable,str(args.extension_root/'scripts/analysis/sf_hlm_v2_experiment.py'),'worker']
        for key in ('source_root','suite_root','extension_root','output','legacy_results','resources'):cmd+=['--'+key.replace('_','-'),str(getattr(args,key))]
        for key,value in [('phase',phase),('family',family),('method',method),('block',block),('workload',workload),('episode',episode),('seed',p['lower_seeds'][str(block)])]:cmd+=['--'+key,str(value)]
        log=args.output/'job_logs'/('_'.join(map(str,t))+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
        return t
    # Parent workers never monopolize all slots while awaiting child trajectories.
    profile=json.loads((args.resources/'profile.json').read_text());kind='teacher' if phase=='dataset_merge' else 'rollout_'+family if phase=='lower_training' else 'environment'
    ceiling=min(profile.get('concurrency_by_kind',{}).get(kind,240),len(selected));ceiling=min(ceiling,10) if phase in ('dataset_merge','lower_training') else ceiling
    with ThreadPoolExecutor(max_workers=max(1,ceiling)) as pool:
        fs={pool.submit(one,t):t for t in selected}
        for f in as_completed(fs):
            try:f.result();done+=1
            except Exception as exc:failures.append(dict(task=fs[f],error=str(exc)))
            res.atomic_json(args.output/'status.json',dict(phase=phase,family=family,completed=done,expected=len(selected),failures=failures,
                elapsed_seconds=time.time()-started,completed_tasks_per_second=done/max(time.time()-started,1e-9),pid=os.getpid(),updated_unix=time.time()))
    if failures:raise RuntimeError('Stage tasks failed: '+str(failures))


def aggregate(args,p,phase,family):
    import numpy as np
    names=('greedy','heuristic','random','learned','old_frozen')+(('no_breakpoints','no_archive') if family=='critical' else ()) if phase in ('lower_eval','timing_eval') else tuple(k for k in METHODS if phase=='confirmation_eval' or p['historical_blocks']==p['confirmation_blocks'] or k!='original')
    blocks=p['confirmation_blocks'] if phase=='confirmation_eval' else p['historical_blocks'];units={};traces={};raw={}
    for t in p['tasks']:
        if t[0]!=phase or t[1]!=family or t[3] not in blocks:continue
        _,_,method,block,workload,episode=t;a=copy.copy(args);a.phase=phase;a.family=family;a.method=method;a.block=block;a.workload=workload
        path=directory(a)/f'episode_{episode:05d}';doc=json.loads((path/'result.json').read_text())
        for rel,h in doc['files'].items():
            if base.sha(path/rel)!=h:raise ValueError('Raw evaluation changed')
        row=json.loads((path/'episode.json').read_text());summary=row['summary'];raw[str(path/'episode.json')]=base.sha(path/'episode.json')
        trace=(summary['exogenous_trace_hash'],summary['potential']);key=(workload,episode)
        if key in traces and traces[key]!=trace:raise ValueError('Exogenous evaluation pairing failed')
        traces[key]=trace;unit=units.setdefault((method,block),dict(generated=0.,admitted=0.,success=0.,potential=0.,energy=0.,mbs=0.,fairness=0.,steps=0,times=[],miss_bias=0.,miss_mae=0.))
        for k in ('generated','admitted','success','potential','energy','mbs','fairness'):unit[k]+=summary[k]
        for k,metric in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy')]:
            if not np.isclose(summary[k],sum(s['metrics'][metric] for s in row['steps']),rtol=1e-10):raise ValueError('Raw counters differ from summary')
        unit['steps']+=len(row['steps']);unit['times'].extend(s['policy_seconds'] for s in row['steps'])
        for s in row['steps']:
            d=s['diagnostic'];m=s['metrics'];error=d['chosen_predicted']['misses']-(m['admitted_service_count']-m['deadline_satisfied_count']);unit['miss_bias']+=error;unit['miss_mae']+=abs(error)
            if d['matching_calls']>p['matching_budget'] or d['chosen_predicted']['energy']>d['common_energy_cap']+1e-7:raise ValueError('Online constraint failed')
    metrics={}
    for key,u in units.items():metrics[key]=dict(dsr=u['success']/max(u['generated'],1),potential_dsr=u['success']/max(u['potential'],1),admission=u['admitted']/max(u['generated'],1),
        conditional=u['success']/max(u['admitted'],1),energy=u['energy'],mbs_share=u['mbs']/max(u['admitted'],1),fairness=u['fairness']/max(u['steps'],1),
        p95=float(np.quantile(u['times'],.95)),predicted_miss_bias=u['miss_bias']/max(u['steps'],1),predicted_miss_mae=u['miss_mae']/max(u['steps'],1))
    groups={k:{m:float(np.mean([metrics[k,b][m] for b in blocks])) for m in metrics[k,blocks[0]]} for k in names}
    candidate='learned' if phase in ('lower_eval','timing_eval') else 'full';comparisons={k:base.paired([metrics[candidate,b]['dsr']-metrics[k,b]['dsr'] for b in blocks]) for k in names if k!=candidate}
    energy={k:float(np.mean([metrics[candidate,b]['energy']/metrics[k,b]['energy'] for b in blocks])) for k in ('greedy','original') if k in names}
    out=dict(status='PASS',phase=phase,family=family,groups=groups,comparisons=comparisons,energy_ratios=energy,
        block_metrics={f'{k}:{b}':v for (k,b),v in metrics.items()},raw_files=raw,independent_blocks=len(blocks),exogenous_pairing=True)
    res.atomic_json(args.output/'analysis'/f'{phase}_{family}.json',out);return out


def snapshot_analysis(args,p,family):
    values={};hashes=[]
    for t in p['tasks']:
        if t[:2]!=['snapshots',family]:continue
        a=copy.copy(args);a.phase=t[0];a.family=t[1];a.method=t[2];a.block=t[3];a.workload=t[4]
        path=directory(a)/f'episode_{t[5]:05d}';records=json.loads((path/'snapshots.json').read_text())
        for s in records:
            hashes.append(s['snapshot_hash']);values.setdefault(t[3],[]).append(s['comparisons']['learned']['actual_misses']-s['comparisons']['greedy']['actual_misses'])
    result=dict(status='PASS',family=family,shared_snapshots=len(hashes),mean=float(statistics.mean([statistics.mean(v) for v in values.values()])),
        paired=base.paired([statistics.mean(values[b]) for b in p['historical_blocks']]),snapshot_hashes=hashes)
    res.atomic_json(args.output/'analysis'/f'snapshots_{family}.json',result);return result


def holm(comparisons,names):
    ordered=sorted(names,key=lambda k:comparisons[k]['p']);running=0.
    for rank,k in enumerate(ordered):running=max(running,min(1.,comparisons[k]['p']*(len(names)-rank)));comparisons[k]['holm_p']=running


def pipeline(args):
    sys.path.insert(0,str(args.source_root));load('utils.sf_hlm_resources',args.extension_root/'utils/sf_hlm_resources.py')
    import numpy as np
    base.np=np;res.initialize(args.resources)
    profile=json.loads((args.resources/'profile.json').read_text());isolated=dict(profile,cpus=profile['reserved'],slots=8,calibrated=True)
    # Discover physical sibling groups rather than depending on CPU numbering.
    seen=set();isolated_cpus=[]
    for cpu in profile['reserved']:
        siblings=(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')).read_text()
        if siblings not in seen:seen.add(siblings);isolated_cpus.append(cpu)
    isolated.update(cpus=isolated_cpus,slots=len(isolated_cpus));res.atomic_json(args.resources/'isolated/profile.json',isolated)
    with res.task_lock(args.output/'controller.lock'):
        p=make_protocol(args);file=args.output/'protocol.json'
        if file.exists() and json.loads(file.read_text())!=p:raise ValueError('Frozen v2 protocol differs')
        if not file.exists():res.atomic_json(file,p)
        stop=threading.Event();thread=threading.Thread(target=res.telemetry,args=(args.resources,stop),daemon=True);thread.start()
        try:
            history=p['historical_blocks'];fresh=p['confirmation_blocks']
            run_tasks(args,p,'dataset_episode',blocks=history);run_tasks(args,p,'dataset_merge',blocks=history)
            selected=None
            for family in ('complete','critical'):
                run_tasks(args,p,'lower_training',family,history);run_tasks(args,p,'lower_eval',family,history)
                run_tasks(args,p,'timing_eval',family,history);run_tasks(args,p,'snapshots',family,history)
                quality=aggregate(args,p,'lower_eval',family);timing=aggregate(args,p,'timing_eval',family);snap=snapshot_analysis(args,p,family)
                passed=snap['mean']<0 and quality['energy_ratios']['greedy']<=1.05 and all(quality['comparisons'][k]['mean']>=.003 and sum(v>0 for v in quality['comparisons'][k]['values'])>=4 for k in ('random','heuristic'))
                passed &= max(v['p95'] for k,v in timing['block_metrics'].items() if k.startswith('learned:'))<1.
                res.atomic_json(args.output/'analysis'/f'gate_lower_{family}.json',dict(passed=bool(passed),family=family,exploratory=True,smoke_cannot_promote=p['smoke']))
                if passed and not p['smoke']:selected=family;break
                if p['smoke']:selected=family # Smoke exercises both without scientific promotion.
            if selected is None:
                res.atomic_json(args.output/'status.json',dict(phase='STOPPED_LOWER_GATE',completed=True,passed=False));return
            res.atomic_json(args.output/'selected.json',dict(family=selected,protocol_sha256=base.sha(file),exploratory=True))
            run_tasks(args,p,'pilot_training',selected,history);run_tasks(args,p,'pilot_eval',selected,history);run_tasks(args,p,'system_timing',selected,history)
            pilot=aggregate(args,p,'pilot_eval',selected);timing=aggregate(args,p,'system_timing',selected)
            pilot_pass=pilot['comparisons']['greedy']['mean']>=.005 and all(pilot['comparisons'][k]['mean']>0 and sum(v>0 for v in pilot['comparisons'][k]['values'])>=4 for k in ('no_feedback','heuristic','random'))
            pilot_pass &= pilot['energy_ratios']['greedy']<=1.05 and max(v['p95'] for k,v in timing['block_metrics'].items() if k.startswith('full:'))<1.
            res.atomic_json(args.output/'analysis/gate_pilot.json',dict(passed=bool(pilot_pass),smoke_cannot_promote=p['smoke']))
            if not pilot_pass and not p['smoke']:
                res.atomic_json(args.output/'status.json',dict(phase='STOPPED_PILOT_GATE',completed=True,passed=False));return
            run_tasks(args,p,'dataset_episode',blocks=fresh);run_tasks(args,p,'dataset_merge',blocks=fresh)
            run_tasks(args,p,'lower_training',selected,fresh);run_tasks(args,p,'system_training',selected,fresh)
            run_tasks(args,p,'confirmation_eval',selected,fresh);run_tasks(args,p,'system_timing',selected,fresh)
            confirmation=aggregate(args,p,'confirmation_eval',selected)
            # The timing phase has both pilot and independent tasks; aggregate
            # explicitly using independent blocks rather than mixed repetitions.
            q=copy.deepcopy(p);q['historical_blocks']=fresh;timing=aggregate(args,q,'system_timing',selected)
            if p['smoke']:
                res.atomic_json(args.output/'analysis/confirmation.json',dict(**confirmation,primary_passed=False,mechanisms_passed=False,smoke_cannot_promote=True))
                res.atomic_json(args.output/'status.json',dict(phase='SMOKE_COMPLETE',completed=True,passed=False,independent_confirmation=False));return
            holm(confirmation['comparisons'],['greedy','original']);holm(confirmation['comparisons'],['no_feedback','heuristic','random'])
            success=all(confirmation['comparisons'][k]['mean']>=.01 and confirmation['comparisons'][k]['ci95'][0]>0 and confirmation['comparisons'][k]['holm_p']<.05 and confirmation['energy_ratios'][k]<=1.05 for k in ('greedy','original'))
            mechanisms=all(confirmation['comparisons'][k]['mean']>0 and confirmation['comparisons'][k]['ci95'][0]>0 and confirmation['comparisons'][k]['holm_p']<.05 for k in ('no_feedback','heuristic','random'))
            success &= max(v['p95'] for k,v in timing['block_metrics'].items() if k.startswith('full:'))<1.
            res.atomic_json(args.output/'analysis/confirmation.json',dict(**confirmation,primary_passed=bool(success),mechanisms_passed=bool(mechanisms),smoke_cannot_promote=p['smoke']))
            res.atomic_json(args.output/'status.json',dict(phase='SMOKE_COMPLETE' if p['smoke'] else 'CONFIRMATION_COMPLETE',completed=True,passed=bool(success and mechanisms and not p['smoke']),independent_confirmation=not p['smoke']))
        finally:stop.set();thread.join(timeout=5)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pipeline','worker','protocol'])
    for name in ('source-root','suite-root','extension-root','output','legacy-results','resources'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=64)
    for name in ('phase','method','family'):p.add_argument('--'+name)
    for name in ('block','workload','episode','seed'):p.add_argument('--'+name,type=int,default=-1)
    args=p.parse_args()
    for name in ('source_root','suite_root','extension_root','output','legacy_results','resources'):setattr(args,name,getattr(args,name).expanduser().resolve())
    if args.mode=='worker':worker(args)
    elif args.mode=='pipeline':pipeline(args)
    else:
        sys.path.insert(0,str(args.source_root));load('utils.sf_hlm_resources',args.extension_root/'utils/sf_hlm_resources.py');print(json.dumps(make_protocol(args),default=base.serial))


if __name__=='__main__':main()
