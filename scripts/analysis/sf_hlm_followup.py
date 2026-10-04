"""Registered controls, bounded residual pilot and conditional confirmation.

Never resumes the failed historical D stage. Each child runs on one shared CPU
lease; only independent episodes/blocks and frozen-policy search are parallel.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

FILES=('utils/sf_hlm_followup.py','marl_models/sf_hlm_bounded_residual.py',
       'scripts/analysis/sf_hlm_followup.py','tests/test_sf_hlm_followup.py')
BASE=('learned','heuristic','random','greedy','original_frozen','original_adapted')
PILOT=('bounded_feedback','bounded_zero','unbounded_feedback')
EXTRA=(*PILOT,'bounded_heuristic','bounded_random')
PATH_ARGS=('source_root','suite_root','extension_root','output','legacy_results','resources',
           'adaptation_root','followup_root','results_root')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def runner(args):return load('followup_v2_runner',args.extension_root/'scripts/analysis/sf_hlm_v2_experiment.py')


def initialize(args,r):
    old,rollout=r.bootstrap(args)
    for name,root in [('marl_models.sf_hlm_upper_adaptation',args.adaptation_root),
                      ('utils.sf_hlm_followup',args.followup_root),
                      ('marl_models.sf_hlm_bounded_residual',args.followup_root)]:
        load(name,root/(name.replace('.','/')+'.py'))
    return old,rollout


def make_protocol(args):
    old=read(args.output/'protocol.json');smoke=args.smoke
    blocks=old['historical_blocks'][:1] if smoke else old['historical_blocks']
    workloads=old['eval_workloads'][:1] if smoke else old['eval_workloads']
    # Smoke uses its own namespace, never any scientific confirmation split.
    fresh=[11960001] if smoke else list(range(1060001,1060011))
    confirm=[11970001] if smoke else list(range(1070001,1070006))
    upper_train={str(b):(11980001 if smoke else 1080001+i) for i,b in enumerate(fresh)}
    lower_train={str(b):(11990001 if smoke else 1090001+i) for i,b in enumerate(fresh)}
    lower_seeds={str(b):(11950001 if smoke else 1100001+i) for i,b in enumerate(fresh)}
    train={str(b):old['training_workloads'][str(b)] for b in blocks};train.update(upper_train)
    eps=1 if smoke else 2;ceps=1 if smoke else 6;tasks=[]
    for b in blocks:
        tasks += [['dev_original_training','original_adapted',b,train[str(b)],-1]]
        tasks += [['dev_residual_training',m,b,train[str(b)],-1] for m in PILOT]
        tasks += [['dev_snapshots','fixed_mbs',b,w,e] for w in (old['snapshot_workloads'][:1] if smoke else old['snapshot_workloads']) for e in range(eps)]
        tasks += [[phase,m,b,w,e] for phase in ('dev_eval','dev_timing') for m in (*BASE,*PILOT) for w in workloads for e in range(eps)]
    for b in fresh:
        tasks += [['foundation_training','shared',b,upper_train[str(b)],-1]]
        tasks += [['confirmation_original_training','original_adapted',b,lower_train[str(b)],-1]]
        tasks += [['component_dataset','greedy',b,lower_train[str(b)],e] for e in range(1 if smoke else 25)]
        tasks += [['component_merge','greedy',b,lower_train[str(b)],-1],['component_lower','learned',b,0,-1]]
        tasks += [['confirmation_budget','frozen_capture',b,lower_train[str(b)],e] for e in range(2 if smoke else 50)]
        tasks += [['confirmation_residual_training',m,b,upper_train[str(b)],-1] for m in EXTRA]
        tasks += [[phase,m,b,w,e] for phase in ('confirmation_eval','confirmation_timing') for m in (*BASE,*EXTRA) for w in confirm for e in range(ceps)]
    return dict(version='sf_hlm_followup_registered_20261002_v1',smoke=smoke,
        original_protocol_sha256=sha(args.output/'protocol.json'),source_hash=old['source_hash'],
        extension_files=old['extension_files'],adaptation_files=read(args.output/'upper_adaptation/protocol.json')['files'],
        files={f:sha(args.followup_root/f) for f in FILES},historical_blocks=blocks,eval_workloads=workloads,
        confirmation_blocks=fresh,confirmation_workloads=confirm,training_workloads=train,
        lower_training_workloads=lower_train,lower_seeds=lower_seeds,tasks=tasks,
        steps=8 if smoke else 1000,original_adaptation_episodes=2 if smoke else 75,
        residual_episodes=2 if smoke else 50,foundation_episodes=2 if smoke else 200,
        dataset_episodes=1 if smoke else 25,lower_iterations=2 if smoke else 100,
        supervised_epochs=1 if smoke else 5,ppo_epochs=old['ppo_epochs'],snapshot_steps=[0,4] if smoke else old['snapshot_steps'],
        matching_budget=16,base_repair_budget=128,teacher_budget=128,rollout_workers=2 if smoke else 32,
        base_methods=list(BASE),residual_methods=list(EXTRA),last_checkpoint_only=True,
        residual=dict(delta=.15,std=.5,anchor=.1,actor_lr=1e-4,critic_lr=3e-4,epochs=4,batch_size=256,clip=.1,gradient_clip=.5),
        fixed_mbs='Raw greedy MBS count is held by initialization, repair, matching, search and fallback; snapshot diagnosis only.',
        budgets=dict(environment_per_method=3 if smoke else 275,common_foundation=2 if smoke else 200,
                     lower_train_or_capture=1 if smoke else 25,adaptation_or_capture=2 if smoke else 50,
                     shared_capture='Frozen arms share the same immutable capture opportunity; compute charged per arm and amortized separately.'),
        primary_controls=['greedy','original_frozen','original_adapted'],
        primary_candidates=['learned','bounded_feedback'],
        primary_family='Three comparisons for learned; six jointly Holm corrected if residual candidate is registered after the development gate.',
        mechanism_family='Learned-H/R; if promoted, bounded-zero/unbounded/frozen-learned/H/R. All active mechanism comparisons jointly Holm corrected.',
        gates=dict(residual_gain=.003,positive_blocks=4,confirmation_gain=.01,energy_ratio=1.05,p95=1.),
        reward_unchanged=True,physics_unchanged=True,natural_admission=True,
        confirmed_split_must_be_authorized=True,confirmation_never_resumes_failed_D=True)


def folder(args):
    path=args.results_root/args.phase/args.method/f'block_{args.block}_workload_{args.workload}'
    return path if args.episode<0 else path/f'episode_{args.episode:05d}'


def receipt(r,path,contract,**extra):
    files={str(f.relative_to(path)):sha(f) for f in path.rglob('*') if f.is_file() and f.name!='result.json' and not f.name.endswith('.tmp')}
    r.res.atomic_json(path/'result.json',dict(status='PASS',contract=contract,files=files,**extra))


def verified(path,contract=None,*,full=True):
    doc=read(path/'result.json')
    if doc['status']!='PASS' or (contract is not None and doc['contract']!=contract):raise ValueError('Task result identity mismatch')
    for f,h in doc['files'].items():
        # Whole training manifests are checked at the phase barrier. Consumers
        # verify the actual final weights, avoiding thousands of re-reads of
        # unused recovery checkpoints during episode evaluation.
        if full or f.startswith('checkpoint/') or f in ('latest.json','canonical/latest.json'):
            if sha(path/f)!=h:raise ValueError('Result manifest changed: '+str(path/f))
    return doc


def complete(path,contract):
    if not (path/'result.json').exists():return False
    verified(path,contract);return True


def component_args(args):
    c=copy.copy(args);c.output=args.results_root/'components';c.family='complete'
    return c


def component_protocol(args,p):
    old=read(args.output/'protocol.json');old.update(smoke=p['smoke'],historical_blocks=[],
        confirmation_blocks=p['confirmation_blocks'],training_workloads=p['training_workloads'],
        lower_training_workloads=p['lower_training_workloads'],lower_seeds=p['lower_seeds'],
        dataset_episodes=p['dataset_episodes'],dataset_steps=[0,4] if p['smoke'] else old['dataset_steps'],
        steps=p['steps'],lower_iterations=p['lower_iterations'],supervised_epochs=p['supervised_epochs'],
        ppo_epochs=1 if p['smoke'] else p['ppo_epochs'],rollout_workers=p['rollout_workers'],
        followup_protocol_sha256=sha(args.results_root/'protocol.json'),version='followup_component_quota_v1')
    return old


def checkpoint_contract(contract):
    return {k:contract[k] for k in ('followup_protocol_sha256','block','method','source_hash','basis_checkpoint_sha256','quota_checkpoint_sha256')}


def basis(args,r,old,p):
    import torch
    fresh=args.block in p['confirmation_blocks']
    models,source=r.base.configure(args,old,historical=not fresh)
    if fresh:
        a=copy.copy(args);a.phase='foundation_training';a.method='shared';a.workload=p['training_workloads'][str(args.block)];a.episode=-1
        path=folder(a);manifest=verified(path,full=False);latest=read(path/'canonical/latest.json');checkpoint=path/'canonical'/latest['checkpoint']
        if sha(checkpoint/'contract.json')!=manifest['files'][str((checkpoint/'contract.json').relative_to(path))]:raise ValueError('Foundation checkpoint contract changed')
    else:
        from utils.formal_protocol import verify_completed_job
        training=verify_completed_job(args.suite_root/'main'/r.base.UPPER/f'seed_{args.block}'/'train',old['source_hash'])
        checkpoint=training/read(training/'latest.json')['checkpoint']
    lock=read(checkpoint/'contract.json')
    if lock['source_hash']!=old['source_hash'] or lock['config_hash']!=source['config_hash']:raise ValueError('Common basis source/config differs')
    for f,h in lock['files'].items():
        if sha(checkpoint/f)!=h:raise ValueError('Basis checkpoint file changed')
    if fresh:
        for name,model in models.items():model.load(str(checkpoint/name))
    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
    return models,source,checkpoint,state


def quota(args,r,old,p):
    if args.block in p['historical_blocks']:
        m=r.read_lower(args,old)
        path=args.output/'lower_training/complete/learned'/f'block_{args.block}'
    else:
        c=component_args(args);cp=read(c.output/'protocol.json');m=r.read_lower(c,cp)
        path=r.lower_dir(c)
    return m,read(path/'result.json')['checkpoint_sha256']


def policy(args,r,p,models,env,obs,*,lower=None,upper=None,train=False,rng=None):
    method=args.method
    mapped='original' if method in ('original_frozen','original_adapted') else 'full' if method in PILOT else 'heuristic' if method=='bounded_heuristic' else 'random' if method=='bounded_random' else 'greedy' if method=='frozen_capture' else method
    result=r.policy(env,obs,mapped,models,family='complete',lower=lower,upper=upper,rng=rng,train=train,budget=p['matching_budget'])
    motion,actions,info,entry,original,rows=result
    if upper is not None:
        entry['action']=upper.last_latent.copy();entry['executed_action']=motion.copy()
        entry['old_mean']=upper.last_mean.copy();entry['old_std']=upper.last_std.copy()
        info['latent_action']=upper.last_latent.copy();info['reference_motion']=upper.last_reference.copy()
        info['executed_motion']=motion.copy();info['bounded']=upper.bounded
    return motion,actions,info,entry,original,rows


def original_training(args,r,p,models,contract,basis_state):
    import numpy as np
    import torch
    import config
    from environment.env import Env
    from marl_models.buffer_and_helpers import DiscreteOffloadRolloutBuffer
    from run_hierarchical_mappo_experiment import lower_rewards_from_metrics,update_on_policy_model
    from utils.canonical_experiment import digest,exogenous_frame
    path=folder(args);path.mkdir(parents=True,exist_ok=True)
    if complete(path,contract):return
    lower=models['lower'];upper_hash=r.base.policy_fingerprint([models['upper']])
    initial_hash=r.base.policy_fingerprint([lower]);rho_dsr=basis_state['rho_dsr'];rho_mbs=basis_state['rho_mbs']
    env=Env(scenario_seed=args.workload);start=0;cc=checkpoint_contract(contract)
    if (path/'latest.json').exists():
        latest=read(path/'latest.json');check=path/latest['checkpoint']
        for f,h in latest['files'].items():
            if sha(check/f)!=h:raise ValueError('Original lower recovery changed')
        lower.load(str(check/'lower'));state=torch.load(check/'state.pt',map_location='cpu',weights_only=False)
        if state['contract']!=cc:raise ValueError('Lower recovery contract differs')
        env=state['env'];start=state['next_episode'];rho_dsr=state['rho_dsr'];rho_mbs=state['rho_mbs']
    for ep in range(start,p['original_adaptation_episodes']):
        seed=r.res.trajectory_seed(sha(args.results_root/'protocol.json'),args.block,ep,'original_lower_training')
        torch.manual_seed(seed);np.random.seed(seed);obs=np.asarray(env.reset(),np.float32)
        buffer=DiscreteOffloadRolloutBuffer(num_agents=config.NUM_UAVS,obs_dim=config.OFFLOAD_OBS_DIM_SINGLE,
            max_requests=config.MAX_OFFLOAD_REQUESTS_PER_UAV,num_actions=config.OFFLOAD_NUM_ACTIONS,buffer_size=p['steps'],device='cpu')
        violations=[];totals={k:0. for k in ('generated','admitted','success','potential','energy','fairness')}
        trace=hashlib.sha256();actions_hash=hashlib.sha256()
        for step in range(p['steps']):
            trace.update(digest(exogenous_frame(env)).encode());potential=sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)
            motion,actions,info,_,entry,_=policy(args,r,p,models,env,obs,train=True)
            obs,rewards,metrics=env.step(motion,offloading_actions=actions);r.base.assert_step(metrics,info);obs=np.asarray(obs,np.float32)
            reward,violation=lower_rewards_from_metrics(metrics,rewards,rho_dsr,rho_mbs);violations.append(violation)
            buffer.add(*entry[:4],reward,step+1==p['steps'],entry[4])
            actions_hash.update(actions.tobytes());totals['potential']+=potential
            for k,n in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy'),('fairness','fairness')]:totals[k]+=float(metrics[n])
        loss=update_on_policy_model(lower,buffer,np.zeros(config.NUM_UAVS,np.float32))
        if config.OFFLOAD_CONSTRAINT_MODE!='none':
            rho_dsr=float(np.clip(rho_dsr+config.OFFLOAD_LAGRANGE_LR*np.mean([v['dsr_violation'] for v in violations]),0,config.OFFLOAD_LAGRANGE_MAX))
            rho_mbs=float(np.clip(rho_mbs+config.OFFLOAD_LAGRANGE_LR*np.mean([v['mbs_load_violation'] for v in violations]),0,config.OFFLOAD_LAGRANGE_MAX))
        if upper_hash!=r.base.policy_fingerprint([models['upper']]):raise ValueError('Frozen trajectory model changed')
        r.res.atomic_json(path/'episodes'/f'{ep:05d}.json',dict(episode=ep,seed=seed,**totals,loss=loss,rho_dsr=rho_dsr,rho_mbs=rho_mbs,exogenous_trace_hash=trace.hexdigest(),action_trace_hash=actions_hash.hexdigest()))
        check=path/'recovery'/f'episode_{ep:05d}';(check/'lower').mkdir(parents=True,exist_ok=True);lower.save(str(check/'lower'))
        torch.save(dict(contract=cc,env=env,next_episode=ep+1,rho_dsr=rho_dsr,rho_mbs=rho_mbs,
                        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state()),check/'state.pt')
        r.res.atomic_json(path/'latest.json',dict(checkpoint=str(check.relative_to(path)),files={str(f.relative_to(check)):sha(f) for f in check.rglob('*') if f.is_file()}))
        r.res.atomic_json(path/'progress.json',dict(episode=ep+1,episodes=p['original_adaptation_episodes'],pid=os.getpid()))
        if args.stop_after and ep+1>=args.stop_after:return
    (path/'checkpoint/lower').mkdir(parents=True,exist_ok=True);lower.save(str(path/'checkpoint/lower'))
    receipt(r,path,contract,episodes=p['original_adaptation_episodes'],upper_fixed=True,initial_lower_fingerprint=initial_hash,
        restored_constraint_state=dict(rho_dsr=basis_state['rho_dsr'],rho_mbs=basis_state['rho_mbs']),optimizer_restored=True)


def residual_training(args,r,p,models,lower,contract):
    import numpy as np
    import torch
    import config
    from environment.env import Env
    from marl_models.sf_hlm_bounded_residual import BoundedResidualMAPPO
    from utils.canonical_experiment import digest,exogenous_frame
    path=folder(args);path.mkdir(parents=True,exist_ok=True)
    if complete(path,contract):return
    torch.manual_seed(r.res.trajectory_seed(sha(args.results_root/'protocol.json'),args.block,'residual_init'))
    upper=BoundedResidualMAPPO(config.OBS_DIM_SINGLE,config.NUM_UAVS,kind=args.method);upper.warm_start(models['upper'])
    frozen=r.base.policy_fingerprint([models['upper'],models['lower'],lower])
    backbone={k:{n:v.detach().clone() for n,v in getattr(upper,k).state_dict().items()} for k in ('actor','critic')}
    initial=r.base.policy_fingerprint([upper]);cc=checkpoint_contract(contract)
    env=Env(scenario_seed=args.workload);start=0
    if (path/'latest.json').exists():
        latest=read(path/'latest.json');check=path/latest['checkpoint']
        for f,h in latest['files'].items():
            if sha(check/f)!=h:raise ValueError('Residual recovery changed')
        upper.load(check,cc);state=torch.load(check/'state.pt',map_location='cpu',weights_only=False)
        if state['contract']!=cc:raise ValueError('Residual recovery contract differs')
        env=state['env'];start=state['next_episode']
    for ep in range(start,p['residual_episodes']):
        seed=r.res.trajectory_seed(sha(args.results_root/'protocol.json'),args.block,ep,'residual_training')
        torch.manual_seed(seed);np.random.seed(seed);obs=np.asarray(env.reset(),np.float32)
        rng=np.random.default_rng(r.res.trajectory_seed(seed,'lower_search'));records=[];actions_hash=hashlib.sha256();trace=hashlib.sha256()
        totals={k:0. for k in ('generated','admitted','success','potential','energy','fairness')};max_shift=0.
        for step in range(p['steps']):
            trace.update(digest(exogenous_frame(env)).encode());potential=sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)
            motion,actions,info,entry,_,_=policy(args,r,p,models,env,obs,lower=lower,upper=upper,train=True,rng=rng)
            max_shift=max(max_shift,float(np.abs(motion-upper.last_reference).max()))
            obs,rewards,metrics=env.step(motion,offloading_actions=actions);r.base.assert_step(metrics,info);obs=np.asarray(obs,np.float32)
            entry.update(reward=np.asarray(rewards,np.float32),done=step+1==p['steps']);records.append(entry)
            actions_hash.update(motion.tobytes());actions_hash.update(actions.tobytes());totals['potential']+=potential
            for k,n in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy'),('fairness','fairness')]:totals[k]+=float(metrics[n])
        loss=upper.update_rollout(records,epochs=1 if p['smoke'] else 4,batch_size=256)
        if frozen!=r.base.policy_fingerprint([models['upper'],models['lower'],lower]):raise ValueError('Fixed model changed during residual training')
        for k,values in backbone.items():
            if any(not torch.equal(v,getattr(upper,k).state_dict()[n]) for n,v in values.items()):raise ValueError('Original residual backbone changed')
        r.res.atomic_json(path/'episodes'/f'{ep:05d}.json',dict(episode=ep,seed=seed,**totals,upper_loss=loss,
            action_trace_hash=actions_hash.hexdigest(),exogenous_trace_hash=trace.hexdigest(),maximum_action_residual=max_shift))
        check=path/'recovery'/f'episode_{ep:05d}';upper.save(check,cc)
        torch.save(dict(contract=cc,env=env,next_episode=ep+1,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state()),check/'state.pt')
        r.res.atomic_json(path/'latest.json',dict(checkpoint=str(check.relative_to(path)),files={str(f.relative_to(check)):sha(f) for f in check.iterdir() if f.is_file()}))
        r.res.atomic_json(path/'progress.json',dict(episode=ep+1,episodes=p['residual_episodes'],pid=os.getpid()))
        if args.stop_after and ep+1>=args.stop_after:return
    upper.save(path/'checkpoint',cc)
    receipt(r,path,contract,episodes=p['residual_episodes'],initial_fingerprint=initial,lower_fixed=True,backbone_fixed=True,
            checkpoint_contract=cc,checkpoint_sha256=sha(path/'checkpoint/latent_residual.pt'))


def evaluate(args,r,p,models,lower,contract):
    import config
    import numpy as np
    import torch
    from marl_models.sf_hlm_bounded_residual import BoundedResidualMAPPO
    from utils.canonical_experiment import digest,exogenous_frame
    path=folder(args);path.mkdir(parents=True,exist_ok=True)
    if complete(path,contract):return
    fresh=args.block in p['confirmation_blocks'];upper=None
    if args.method in EXTRA:
        a=copy.copy(args);a.phase='confirmation_residual_training' if fresh else 'dev_residual_training';a.workload=p['training_workloads'][str(args.block)];a.episode=-1
        training=folder(a);d=verified(training,full=False)
        upper=BoundedResidualMAPPO(config.OBS_DIM_SINGLE,config.NUM_UAVS,kind=args.method)
        upper.load(training/'checkpoint',d['checkpoint_contract'])
    if args.method=='original_adapted':
        a=copy.copy(args);a.phase='confirmation_original_training' if fresh else 'dev_original_training';a.workload=p['lower_training_workloads'][str(args.block)] if fresh else p['training_workloads'][str(args.block)];a.episode=-1
        training=folder(a);verified(training,full=False);models['lower'].load(str(training/'checkpoint/lower'))
    fingerprint=r.base.policy_fingerprint([models['upper'],models['lower'],lower,upper])
    env,obs=r.episode_env(args.workload,args.episode)
    # Development frozen trajectories retain the exact historical search seed.
    rng=np.random.default_rng(r.res.trajectory_seed(p['original_protocol_sha256'] if not fresh else sha(args.results_root/'protocol.json'),args.block,args.workload,args.episode,'search'))
    trace=hashlib.sha256();steps=[];totals={k:0. for k in ('generated','admitted','success','potential','energy','mbs','fairness')}
    for step in range(p['steps']):
        trace.update(digest(exogenous_frame(env)).encode());potential=sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)
        motion,actions,info,_,_,_=policy(args,r,p,models,env,obs,lower=lower,upper=upper,rng=rng)
        obs,rewards,metrics=env.step(motion,offloading_actions=actions);r.base.assert_step(metrics,info);obs=np.asarray(obs,np.float32)
        totals['potential']+=potential
        for k,n in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy'),('mbs','service_offloads_mbs'),('fairness','fairness')]:totals[k]+=float(metrics[n])
        steps.append(dict(step=step,metrics=metrics,potential=potential,team_reward=float(np.mean(rewards)),**info))
    if fingerprint!=r.base.policy_fingerprint([models['upper'],models['lower'],lower,upper]):raise ValueError('Evaluation changed models')
    r.res.atomic_json(path/'episode.json',dict(summary=totals,exogenous_trace_hash=trace.hexdigest(),steps=steps))
    receipt(r,path,contract,model_fingerprint=fingerprint)


def snapshots(args,r,p,models,lower,contract):
    import config
    import numpy as np
    from run_canonical_experiment import policy_step
    from utils.snapshot_evaluator import capture,evaluate as execute
    from utils.sf_hlm_followup import fixed_mbs_search
    from utils.canonical_experiment import digest,exogenous_frame
    from utils.formal_protocol import verify_completed_job
    path=folder(args);path.mkdir(parents=True,exist_ok=True)
    if complete(path,contract):return
    env,obs=r.episode_env(args.workload,args.episode);records=[];trace=hashlib.sha256()
    oldpath=args.output/'snapshots/complete/all'/f'block_{args.block}_workload_{args.workload}'/f'episode_{args.episode:05d}'
    verified(oldpath);previous=read(oldpath/'snapshots.json')
    original_path=verify_completed_job(args.suite_root/'main'/r.base.UPPER/f'seed_{args.block}'/f'workload_{args.workload}',p['source_hash'])
    original=read(original_path/'episodes'/f'{args.episode:05d}.json')
    for step in range(p['steps']):
        trace.update(digest(exogenous_frame(env)).encode());motion,actions,_=policy_step(models,env,obs,exploration=False)
        if step in p['snapshot_steps']:
            _,mask,requests,rows=r.base.prepare(env);snap=capture(env,np.clip(motion,-1,1));comparison={}
            if tuple(q[2] for q in requests)!=tuple(q[2] for q in snap.requests):raise ValueError('Frozen request order differs')
            prior=next((x for x in previous if x['step']==step),None)
            if prior is not None and prior['snapshot_hash']!=snap.snapshot_hash:raise ValueError('Frozen snapshot identity differs')
            for m in ('greedy','heuristic','random','learned'):
                rng=np.random.default_rng(r.res.trajectory_seed(sha(args.results_root/'protocol.json'),args.block,args.workload,args.episode,step,m,'fixed_mbs'))
                a,d,_=fixed_mbs_search(rows,num_uavs=config.NUM_UAVS,kind=m,model=lower,rng=rng,budget=p['matching_budget'])
                actual=execute(snap,a)
                comparison[m]=dict(assignment=a,diagnostic=d,actual_misses=sum(not q['success'] for q in actual['requests']),
                    actual_assignment_energy=float(actual['metrics']['lower_assignment_energy']))
            if len({x['diagnostic']['fixed_mbs_count'] for x in comparison.values()})!=1:raise ValueError('Common MBS quantities differ')
            records.append(dict(step=step,snapshot_hash=snap.snapshot_hash,requests=len(requests),comparisons=comparison))
        obs,_,metrics=env.step(np.clip(motion,-1,1),offloading_actions=actions);obs=np.asarray(obs,np.float32)
        for metric in ('generated_service_count','admitted_service_count','deadline_satisfied_count','lower_assignment_energy'):
            if not np.isclose(metrics[metric],original['steps'][step]['metrics'][metric],rtol=1e-10,atol=1e-10):raise ValueError('Historical step replay differs: '+metric)
    if not p['smoke'] and trace.hexdigest()!=original['exogenous_trace_hash']:raise ValueError('Historical exogenous stream differs')
    r.res.atomic_json(path/'snapshots.json',dict(snapshots=records,exogenous_trace_hash=trace.hexdigest()))
    receipt(r,path,contract,snapshots=len(records),diagnosis_only=True)


def foundation(args,r,p,contract,lease):
    import fcntl
    path=folder(args);path.mkdir(parents=True,exist_ok=True)
    if complete(path,contract):return
    plan=read(args.suite_root/'suite.json')['plan']
    target=path/'canonical';cmd=[sys.executable,str(args.source_root/'run_canonical_experiment.py'),
        '--method',r.base.UPPER,'--training-seed',str(args.block),'--workload-seed',str(args.workload),
        '--episodes',str(p['foundation_episodes']),'--steps',str(p['steps']),'--rollout',str(plan['rollout']),
        '--ppo-epochs',str(plan['ppo_epochs']),'--batch-size',str(plan['batch_size']),'--threads','1',
        '--output',str(target),'--purpose','verification' if p['smoke'] else 'pilot','--checkpoint-every','20']
    if (target/'latest.json').exists():cmd.append('--resume')
    # A live canonical child keeps its slot and lock if the wrapper exits.
    with (path/'foundation-child.lock').open('a+') as child_lock:
        try:fcntl.flock(child_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Canonical foundation child is still running')
        already_done=(target/'latest.json').exists() and read(target/'latest.json')['completed_episodes']==p['foundation_episodes']
        if not already_done:
            child=subprocess.Popen(cmd,cwd=args.source_root,pass_fds=(child_lock.fileno(),))
            lease.adopt_owner(child.pid)
            if child.wait()!=0:raise RuntimeError('Canonical foundation training failed')
    latest=read(target/'latest.json')
    if latest['completed_episodes']!=p['foundation_episodes']:raise ValueError('Common basis is incomplete')
    checkpoint=target/latest['checkpoint'];lock=read(checkpoint/'contract.json')
    if lock['source_hash']!=p['source_hash']:raise ValueError('Foundation source differs')
    for f,h in lock['files'].items():
        if sha(checkpoint/f)!=h:raise ValueError('Foundation final checkpoint changed')
    if {f.name for f in (target/'episodes').glob('*.json')}!={f'{e:05d}.json' for e in range(p['foundation_episodes'])}:raise ValueError('Foundation episodes are incomplete')
    receipt(r,path,contract,episodes=p['foundation_episodes'],shared_foundation=True)


def worker(args):
    r=runner(args);p=read(args.results_root/'protocol.json')
    task=[args.phase,args.method,args.block,args.workload,args.episode]
    if task not in p['tasks']:raise ValueError('Unregistered follow-up task')
    if sha(args.output/'protocol.json')!=p['original_protocol_sha256']:raise ValueError('Original protocol changed')
    for files,root in [(p['files'],args.followup_root),(p['extension_files'],args.extension_root),(p['adaptation_files'],args.adaptation_root)]:
        for f,h in files.items():
            if sha(root/f)!=h:raise ValueError('Frozen release changed: '+f)
    if args.phase.startswith('confirmation') or args.phase.startswith('component') or args.phase=='foundation_training':
        if not p['smoke']:
            authorization=read(args.results_root/'confirmation_authorization.json')
            if not authorization.get('authorized') or authorization.get('protocol_sha256')!=sha(args.results_root/'protocol.json'):raise RuntimeError('Confirmation split has not been approved')
    resource=args.resources/'isolated' if args.phase.endswith('_timing') else args.resources
    with r.res.task_lock(args.results_root/'task_locks'/('_'.join(map(str,task))+'.lock')):
        with r.res.CpuLease(resource,task=['followup',*task],kind='environment') as lease:
            old,rollout=initialize(args,r)
            contract=dict(followup_protocol_sha256=sha(args.results_root/'protocol.json'),source_hash=p['source_hash'],
                          phase=args.phase,method=args.method,block=args.block,workload=args.workload,episode=args.episode)
            if args.phase=='foundation_training':foundation(args,r,p,contract,lease);return
            models,source,check,state=basis(args,r,old,p)
            contract.update(config_hash=source['config_hash'],basis_checkpoint_sha256=sha(check/'contract.json'),quota_checkpoint_sha256=None)
            if args.phase.startswith('component'):
                c=component_args(args);cp=read(c.output/'protocol.json')
                c.phase={'component_dataset':'dataset_episode','component_merge':'dataset_merge','component_lower':'lower_training'}[args.phase]
                component_contract=dict(**contract,component_protocol_sha256=sha(c.output/'protocol.json'))
                if c.phase=='dataset_episode':r.dataset_episode(c,cp,models,component_contract)
                elif c.phase=='dataset_merge':r.merge_dataset(c,cp,rollout,lease,component_contract)
                else:r.lower_training(c,cp,rollout,lease,component_contract)
                return
            if args.phase.endswith('original_training'):
                original_training(args,r,p,models,contract,state);return
            lower=None
            if args.method in ('learned',*PILOT) or args.phase=='dev_snapshots':
                lower,contract['quota_checkpoint_sha256']=quota(args,r,old,p)
            if args.phase.endswith('residual_training'):residual_training(args,r,p,models,lower,contract)
            elif args.phase=='dev_snapshots':snapshots(args,r,p,models,lower,contract)
            else:evaluate(args,r,p,models,lower,contract)


def run(args,r,p,phases,methods=None):
    tasks=[t for t in p['tasks'] if t[0] in phases and (methods is None or t[1] in methods)]
    if not tasks:return
    def one(t):
        cmd=[sys.executable,str(args.followup_root/'scripts/analysis/sf_hlm_followup.py'),'worker']
        for k in PATH_ARGS:cmd+=['--'+k.replace('_','-'),str(getattr(args,k))]
        for k,v in zip(('phase','method','block','workload','episode'),t):cmd+=['--'+k,str(v)]
        log=args.results_root/'job_logs'/('_'.join(map(str,t))+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:subprocess.run(cmd,check=True,stdout=stream,stderr=subprocess.STDOUT)
    done=0;failed=[];concurrency=min(len(tasks),read(args.resources/'profile.json')['slots'])
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures={pool.submit(one,t):t for t in tasks}
        for f in as_completed(futures):
            try:f.result();done+=1
            except Exception as exc:failed.append(dict(task=futures[f],error=str(exc)))
            r.res.atomic_json(args.results_root/'status.json',dict(phase='RUNNING',phases=phases,completed=done,expected=len(tasks),failures=failed,pid=os.getpid(),scientific_success_claim=False))
    if failed:raise RuntimeError('Follow-up tasks failed: '+str(failed))
    # Audit complete training state once before any consumer can use it.
    for t in tasks:
        if t[0].endswith('training') or t[0]=='dev_snapshots':
            a=copy.copy(args)
            for k,v in zip(('phase','method','block','workload','episode'),t):setattr(a,k,v)
            d=verified(folder(a))
            if t[0].endswith('training') and t[0]!='foundation_training':
                episodes=p['original_adaptation_episodes'] if t[0].endswith('original_training') else p['residual_episodes']
                if d['episodes']!=episodes or {f.name for f in (folder(a)/'episodes').glob('*.json')}!={f'{e:05d}.json' for e in range(episodes)}:raise ValueError('Training episodes are incomplete')
    if 'dev_residual_training' in phases:
        for b in p['historical_blocks']:
            states={}
            for m in PILOT:
                a=copy.copy(args);a.phase='dev_residual_training';a.method=m;a.block=b;a.workload=p['training_workloads'][str(b)];a.episode=-1
                states[m]=verified(folder(a),full=False)
            if len({d['initial_fingerprint'] for d in states.values()})!=1:raise ValueError('Residual arms have different initial weights')
            a.method='bounded_feedback';first=read(folder(a)/'episodes/00000.json')
            a.method='bounded_zero';zero=read(folder(a)/'episodes/00000.json')
            if first['action_trace_hash']!=zero['action_trace_hash'] or first['exogenous_trace_hash']!=zero['exogenous_trace_hash']:raise ValueError('Bounded first rollouts differ at zero initialization')


def aggregate(args,r,p,phase,names):
    import numpy as np
    r.base.np=np;units={};traces={};raw={}
    blocks=p['historical_blocks'] if phase.startswith('dev') else p['confirmation_blocks']
    for t in p['tasks']:
        if t[0]!=phase or t[1] not in names:continue
        a=copy.copy(args)
        for k,v in zip(('phase','method','block','workload','episode'),t):setattr(a,k,v)
        path=folder(a);verified(path);row=read(path/'episode.json');s=row['summary']
        trace=row['exogenous_trace_hash'],s['potential'];key=(a.workload,a.episode)
        if traces.setdefault(key,trace)!=trace:raise ValueError('Exogenous pairing differs')
        raw[str(path/'episode.json')]=sha(path/'episode.json')
        u=units.setdefault((a.method,a.block),dict(totals={k:0. for k in s},times=[],steps=0,bias=0.,mae=0.))
        for k,v in s.items():u['totals'][k]+=v
        if len(row['steps'])!=p['steps']:raise ValueError('Partial evaluation episode')
        for k,n in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy')]:
            if not np.isclose(s[k],sum(x['metrics'][n] for x in row['steps']),rtol=1e-10):raise ValueError('Evaluation counters differ')
        for x in row['steps']:
            d=x['diagnostic'];m=x['metrics']
            if d['matching_calls']>p['matching_budget'] or d['chosen_predicted']['energy']>d['common_energy_cap']+1e-7:raise ValueError('Online guard failed')
            if x.get('bounded') and np.max(np.abs(np.asarray(x['executed_motion'])-np.asarray(x['reference_motion'])))>.150001:raise ValueError('Evaluation bound failed')
            error=d['chosen_predicted']['misses']-(m['admitted_service_count']-m['deadline_satisfied_count']);u['bias']+=error;u['mae']+=abs(error)
            u['times'].append(x['policy_seconds']);u['steps']+=1
        if phase.endswith('_timing'):
            q=copy.copy(a);q.phase=phase.replace('_timing','_eval');quality=read(folder(q)/'episode.json')
            if row['summary']!=quality['summary'] or [x['assignment'] for x in row['steps']]!=[x['assignment'] for x in quality['steps']]:raise ValueError('Isolated timing changes quality')
    metrics={}
    for (m,b),u in units.items():
        s=u['totals'];metrics[m,b]=dict(dsr=s['success']/max(s['generated'],1),potential_dsr=s['success']/max(s['potential'],1),
            admission=s['admitted']/max(s['generated'],1),conditional=s['success']/max(s['admitted'],1),energy=s['energy'],
            mbs_share=s['mbs']/max(s['admitted'],1),fairness=s['fairness']/u['steps'],p95=float(np.quantile(u['times'],.95)),
            predicted_miss_bias=u['bias']/u['steps'],predicted_miss_mae=u['mae']/u['steps'])
    for m in names:
        for b in blocks:
            if (m,b) not in metrics:raise ValueError('Missing evaluation unit')
    groups={m:{k:float(np.mean([metrics[m,b][k] for b in blocks])) for k in metrics[m,blocks[0]]} for m in names}
    comparisons={}
    for candidate in ('learned','bounded_feedback'):
        if candidate not in names:continue
        for control in names:
            if control==candidate:continue
            label=candidate+'-'+control
            comparisons[label]=dict(**r.base.paired([metrics[candidate,b]['dsr']-metrics[control,b]['dsr'] for b in blocks]),
                energy_ratio=float(np.mean([metrics[candidate,b]['energy']/metrics[control,b]['energy'] for b in blocks])))
    out=dict(status='PASS',phase=phase,groups=groups,comparisons=comparisons,block_metrics={f'{m}:{b}':v for (m,b),v in metrics.items()},
        independent_blocks=len(blocks),exploratory=phase.startswith('dev'),raw_files=raw,exogenous_pairing=True)
    r.res.atomic_json(args.results_root/'analysis'/f'{phase}.json',out);return out


def snapshot_analysis(args,r,p):
    block_values={b:{m:[] for m in ('greedy','heuristic','random','learned')} for b in p['historical_blocks']};raw={};count=0
    for t in p['tasks']:
        if t[0]!='dev_snapshots':continue
        a=copy.copy(args)
        for k,v in zip(('phase','method','block','workload','episode'),t):setattr(a,k,v)
        path=folder(a);verified(path);doc=read(path/'snapshots.json');raw[str(path/'snapshots.json')]=sha(path/'snapshots.json')
        for row in doc['snapshots']:
            count+=1;mbs=[]
            for m,v in row['comparisons'].items():
                d=v['diagnostic'];mbs.append(d['fixed_mbs_count'])
                if sum(x==1 for x in v['assignment'])!=d['fixed_mbs_count']:raise ValueError('Common MBS diagnostic failed')
                if d['matching_calls']>16 or d['chosen_predicted']['energy']>d['common_energy_cap']+1e-7:raise ValueError('Restricted guard failed')
                block_values[a.block][m].append(v['actual_misses'])
            if len(set(mbs))!=1:raise ValueError('Restricted MBS counts differ')
    import numpy as np
    means={str(b):{m:float(np.mean(v)) for m,v in values.items()} for b,values in block_values.items()}
    out=dict(status='PASS',snapshot_count=count,block_means=means,raw_files=raw,diagnosis_only=True,
             comparisons={m:r.base.paired([means[str(b)]['learned']-means[str(b)][m] for b in p['historical_blocks']]) for m in ('greedy','heuristic','random')})
    r.res.atomic_json(args.results_root/'analysis/common_mbs.json',out);return out


def split_audit(args,r,p):
    """Actual result/manifests count as use; mere protocol reservations do not."""
    if p['smoke']:return dict(status='SMOKE_NAMESPACE',independent_confirmation=False)
    wanted=set(p['confirmation_blocks']+p['confirmation_workloads']+[p['training_workloads'][str(b)] for b in p['confirmation_blocks']]+list(p['lower_training_workloads'].values())+list(p['lower_seeds'].values()))
    conflicts=[];inspected=0
    for task_root in args.output.parents[2].iterdir():
        if not task_root.is_dir():continue
        for pattern in ('**/manifest.json','**/result.json','**/completed.json'):
            for file in task_root.glob(pattern):
                if args.results_root in file.parents:continue
                inspected+=1
                try:d=read(file)
                except (ValueError,OSError):continue
                contexts=[d.get('settings',{}),d.get('contract',{}),d.get('job',{})]
                used=set()
                for c in contexts:
                    for k in ('training_seed','workload_seed','block','workload','lower_seed'):
                        if isinstance(c.get(k),int):used.add(c[k])
                overlap=used&wanted
                if overlap:conflicts.append(dict(path=str(file),ids=sorted(overlap)))
    out=dict(status='PASS' if not conflicts else 'CONFLICT',actual_metadata_files=inspected,conflicts=conflicts,
             old_smoke_conflict_disclosed=[960001,970001,980001,990001],selected_new_split=sorted(wanted))
    r.res.atomic_json(args.results_root/'split_audit.json',out)
    if conflicts:raise ValueError('Confirmation namespaces were previously used: '+str(conflicts[:3]))
    return out


def confirmation_analysis(args,r,p,quality,timing,include):
    names=list(BASE)+list(EXTRA if include else ())
    primary=[c+'-'+k for c in ('learned','bounded_feedback') if c in names for k in p['primary_controls']]
    mechanisms=['learned-heuristic','learned-random']
    if include:mechanisms += ['bounded_feedback-'+k for k in ('bounded_zero','unbounded_feedback','learned','bounded_heuristic','bounded_random')]
    if not p['smoke']:
        r.holm(quality['comparisons'],primary);r.holm(quality['comparisons'],mechanisms)
    outcomes={}
    for candidate in ('learned','bounded_feedback'):
        if candidate not in names:continue
        comparisons=[quality['comparisons'][candidate+'-'+k] for k in p['primary_controls']]
        p95=max(timing['block_metrics'][f'{candidate}:{b}']['p95'] for b in p['confirmation_blocks'])
        passed=not p['smoke'] and all(x['mean']>=.01 and x['ci95'][0]>0 and x['holm_p']<.05 and x['energy_ratio']<=1.05 for x in comparisons) and p95<1
        mech=['learned-heuristic','learned-random'] if candidate=='learned' else ['bounded_feedback-'+k for k in ('bounded_zero','unbounded_feedback','learned','bounded_heuristic','bounded_random')]
        mechanism_passed=not p['smoke'] and all(quality['comparisons'][k]['mean']>0 and quality['comparisons'][k]['ci95'][0]>0 and quality['comparisons'][k]['holm_p']<.05 for k in mech)
        outcomes[candidate]=dict(primary_passed=bool(passed),mechanisms_passed=bool(mechanism_passed),maximum_isolated_p95=p95)
    quality.update(outcomes=outcomes,primary_family=primary,mechanism_family=mechanisms,independent_confirmation=not p['smoke'],smoke_cannot_promote=p['smoke'])
    r.res.atomic_json(args.results_root/'analysis/confirmation.json',quality)
    return outcomes


def pipeline(args):
    r=runner(args);r.res.initialize(args.resources);p=make_protocol(args);file=args.results_root/'protocol.json'
    if file.exists() and read(file)!=p:raise ValueError('Frozen follow-up protocol differs')
    r.res.atomic_json(file,p)
    cp=component_protocol(args,p);component=args.results_root/'components/protocol.json'
    if component.exists() and read(component)!=cp:raise ValueError('Frozen component protocol differs')
    r.res.atomic_json(component,cp)
    with r.res.task_lock(args.results_root/'controller.lock'):
        stop=threading.Event();telemetry=threading.Thread(target=r.res.telemetry,args=(args.resources,stop),daemon=True);telemetry.start()
        try:
            # Stage 1 must complete before the single, final residual pilot.
            run(args,r,p,['dev_original_training','dev_snapshots'])
            run(args,r,p,['dev_eval'],BASE);run(args,r,p,['dev_timing'],BASE)
            first=aggregate(args,r,p,'dev_eval',BASE);aggregate(args,r,p,'dev_timing',BASE);snapshot_analysis(args,r,p)
            run(args,r,p,['dev_residual_training']);run(args,r,p,['dev_eval'],PILOT);run(args,r,p,['dev_timing'],PILOT)
            quality=aggregate(args,r,p,'dev_eval',(*BASE,*PILOT));timing=aggregate(args,r,p,'dev_timing',(*BASE,*PILOT))
            compare=[quality['comparisons']['bounded_feedback-'+k] for k in ('bounded_zero','unbounded_feedback','learned')]
            include=not p['smoke'] and all(x['mean']>=.003 and sum(v>0 for v in x['values'])>=4 for x in compare)
            include &= quality['comparisons']['bounded_feedback-learned']['energy_ratio']<=1.05
            include &= max(timing['block_metrics'][f'bounded_feedback:{b}']['p95'] for b in p['historical_blocks'])<1
            r.res.atomic_json(args.results_root/'development_gate.json',dict(status='PASS',residual_promoted=bool(include),
                main_confirmation_authorized_by_plan=True,smoke_cannot_promote=p['smoke'],scientific_success_claim=False))
            if args.development_only or (not args.authorize_confirmation and not p['smoke']):
                r.res.atomic_json(args.results_root/'status.json',dict(phase='DEVELOPMENT_COMPLETE_AWAITING_CONFIRMATION_SPLIT',completed=True,
                    residual_promoted=bool(include),confirmation_started=False,scientific_success_claim=False));return
            split_audit(args,r,p)
            r.res.atomic_json(args.results_root/'confirmation_authorization.json',dict(split='fresh',authorized=bool(args.authorize_confirmation),protocol_sha256=sha(file),
                smoke=p['smoke'],decision_source='Explicit user acceptance of the replacement split; smoke is validation only.'))
            selection=dict(candidates=['learned']+(['bounded_feedback'] if include else []),residual_promoted=bool(include),
                           methods=list(BASE)+list(EXTRA if include or p['smoke'] else ()),locked_before_confirmation=True)
            selected=args.results_root/'confirmation_selection.json'
            if selected.exists() and read(selected)!=selection:raise ValueError('Locked confirmation selection differs')
            r.res.atomic_json(selected,selection)
            run(args,r,p,['foundation_training'])
            run(args,r,p,['confirmation_original_training','component_dataset'])
            run(args,r,p,['component_merge']);run(args,r,p,['component_lower'])
            run(args,r,p,['confirmation_budget'])
            if include or p['smoke']:run(args,r,p,['confirmation_residual_training'])
            names=(*BASE,*EXTRA) if include or p['smoke'] else BASE
            run(args,r,p,['confirmation_eval'],names);run(args,r,p,['confirmation_timing'],names)
            quality=aggregate(args,r,p,'confirmation_eval',names);timing=aggregate(args,r,p,'confirmation_timing',names)
            outcomes=confirmation_analysis(args,r,p,quality,timing,include or p['smoke'])
            r.res.atomic_json(args.results_root/'status.json',dict(phase='SMOKE_COMPLETE' if p['smoke'] else 'CONFIRMATION_COMPLETE',completed=True,
                independent_confirmation=not p['smoke'],outcomes=outcomes,scientific_success_claim=not p['smoke'] and any(x['primary_passed'] for x in outcomes.values())))
        finally:stop.set();telemetry.join(timeout=5)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('pipeline','worker','protocol'))
    for k in PATH_ARGS:parser.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--authorize-confirmation',action='store_true')
    parser.add_argument('--development-only',action='store_true');parser.add_argument('--stop-after',type=int,default=0)
    for k in ('phase','method'):parser.add_argument('--'+k)
    for k in ('block','workload','episode'):parser.add_argument('--'+k,type=int,default=-1)
    args=parser.parse_args()
    for k in PATH_ARGS:setattr(args,k,getattr(args,k).expanduser().resolve())
    args.family='complete';args.seed=max(args.block,0)
    if sys.platform!='linux':raise RuntimeError('All numerical follow-up execution must run remotely')
    if args.mode=='worker':worker(args)
    elif args.mode=='pipeline':pipeline(args)
    else:print(json.dumps(make_protocol(args)))


if __name__=='__main__':main()
