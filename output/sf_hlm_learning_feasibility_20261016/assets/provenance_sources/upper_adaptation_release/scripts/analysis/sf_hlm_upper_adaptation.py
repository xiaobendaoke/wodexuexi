"""Registered update-engine and frozen-backbone pilots, on remote Linux only."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
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

FILES=('marl_models/sf_hlm_upper_adaptation.py','scripts/analysis/sf_hlm_upper_adaptation.py',
       'tests/test_sf_hlm_upper_adaptation.py')
R1=('original_update','current_update')
R2=('residual_feedback','residual_zero')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def runner(args):return load('adaptation_v2_runner',args.extension_root/'scripts/analysis/sf_hlm_v2_experiment.py')


def make_protocol(args):
    old=read(args.output/'protocol.json')
    blocks=old['historical_blocks'][:1] if args.smoke else old['historical_blocks']
    workloads=old['eval_workloads'][:1] if args.smoke else old['eval_workloads']
    tasks=[['reference_eval','frozen',b,w,e] for b in blocks for w in workloads for e in range(2)]
    for stage,methods in [('r1',R1),('r2',R2)]:
        tasks += [[stage+'_training',m,b,old['training_workloads'][str(b)],-1] for m in methods for b in blocks]
        tasks += [[phase,m,b,w,e] for phase in (stage+'_eval',stage+'_timing') for m in methods for b in blocks for w in workloads for e in range(2)]
    return dict(version='sf_hlm_upper_adaptation_registered_v1',original_protocol_sha256=sha(args.output/'protocol.json'),
                files={f:sha(args.adaptation_root/f) for f in FILES},historical_blocks=blocks,workloads=workloads,
                training_workloads=old['training_workloads'],episodes=2 if args.smoke else 50,steps=8 if args.smoke else 1000,
                epochs=1 if args.smoke else old['ppo_epochs'],matching_budget=old['matching_budget'],tasks=tasks,
                smoke=args.smoke,last_checkpoint_only=True,lower_fixed=True,reward_unchanged=True,
                optimizer_state='All arms start with fresh optimizers and the same historical actor/critic weights.',
                initialization='Matched branch weights; common per-block/per-episode seed excludes method and concurrency.',
                r1='Original actor/critic; original reference update versus frozen v2 joint update.',
                r2='Original reference update semantics; freeze original actor/critic, train only service residual modules; pressure versus zero.',
                transition='R2 follows completed R1 and validation. It tests a distinct registered mechanism irrespective of R1 effect direction.',
                confirmation_started=False,checkpoint_selection=False,independent_training_blocks=len(blocks),
                reference_workload_access='Development workloads only; confirmation namespaces untouched.')


def folder(args):
    p=args.results_root/args.phase/args.method/f'block_{args.block}_workload_{args.workload}'
    return p if args.episode<0 else p/f'episode_{args.episode:05d}'


def receipt(r,path,contract,**extra):
    files={str(f.relative_to(path)):sha(f) for f in path.rglob('*') if f.is_file() and f.name!='result.json' and not f.name.endswith('.tmp')}
    r.res.atomic_json(path/'result.json',dict(status='PASS',contract=contract,files=files,**extra))


def complete(path,contract):
    if not (path/'result.json').exists():return False
    d=read(path/'result.json')
    if d['contract']!=contract or d['status']!='PASS':raise ValueError('Result identity differs')
    for f,h in d['files'].items():
        if sha(path/f)!=h:raise ValueError('Result file differs')
    return True


def build_upper(args,r,p,models):
    import config
    from marl_models.sf_hlm_upper_adaptation import UpperAdaptation
    if args.method=='frozen':return None
    upper=UpperAdaptation(config.OBS_DIM_SINGLE,config.NUM_UAVS,kind=args.method)
    # Equal unused/new branch initialization within each block, independent of
    # method constructor or optimizer RNG consumption.
    upper.warm_start(models['upper'])
    return upper


def checkpoint_contract(contract):
    return dict(protocol_sha256=contract['adaptation_protocol_sha256'],block=contract['block'],method=contract['method'],
                lower_checkpoint_sha256=contract['lower_checkpoint_sha256'],source_hash=contract['source_hash'])


def train(args,r,p,models,contract):
    import config
    import numpy as np
    import torch
    from environment.env import Env
    from utils.canonical_experiment import digest,exogenous_frame
    path=folder(args);path.mkdir(parents=True,exist_ok=True)
    if complete(path,contract):return
    lower=r.read_lower(args,read(args.output/'protocol.json'))
    lower_fingerprint=r.base.policy_fingerprint([lower])
    torch.manual_seed(r.res.trajectory_seed(p['original_protocol_sha256'],args.block,'new_branch_init'))
    upper=build_upper(args,r,p,models);initial_actor=r.base.policy_fingerprint([upper])
    frozen={k:hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in getattr(upper,k).state_dict().values())).hexdigest() for k in ('actor','critic')}
    env=Env(scenario_seed=args.workload);start=0;cc=checkpoint_contract(contract)
    if (path/'latest.json').exists():
        d=read(path/'latest.json');checkpoint=path/d['checkpoint']
        for f,h in d['files'].items():
            if sha(checkpoint/f)!=h:raise ValueError('Recovery changed')
        upper.load(checkpoint,cc);state=torch.load(checkpoint/'state.pt',weights_only=False)
        if state['contract']!=cc:raise ValueError('Recovery contract differs')
        env=state['env'];start=state['episode'];torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng'])
    for ep in range(start,p['episodes']):
        seed=r.res.trajectory_seed(p['original_protocol_sha256'],'upper_adaptation',args.block,ep,'training')
        torch.manual_seed(seed);np.random.seed(seed)
        obs=np.asarray(env.reset(),np.float32);records=[];trace=hashlib.sha256();action_trace=hashlib.sha256()
        totals={k:0. for k in ('generated','admitted','success','potential','energy','fairness')}
        rng=np.random.default_rng(r.res.trajectory_seed(seed,'lower_search'))
        for step in range(p['steps']):
            trace.update(digest(exogenous_frame(env)).encode());potential=sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)
            motion,actions,info,entry,_,_=r.policy(env,obs,'full',models,family='complete',lower=lower,upper=upper,rng=rng,train=True,budget=p['matching_budget'])
            entry.update(old_mean=upper.last_mean.copy(),old_std=upper.last_std.copy())
            obs,rewards,metrics=env.step(motion,offloading_actions=actions);r.base.assert_step(metrics,info);obs=np.asarray(obs,np.float32)
            entry.update(reward=np.asarray(rewards,np.float32),done=step+1==p['steps']);records.append(entry)
            action_trace.update(motion.tobytes());action_trace.update(actions.tobytes());totals['potential']+=potential
            for k,n in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy'),('fairness','fairness')]:totals[k]+=float(metrics[n])
        loss=upper.update_rollout(records,epochs=p['epochs'],batch_size=config.PPO_BATCH_SIZE)
        if lower_fingerprint!=r.base.policy_fingerprint([lower]):raise ValueError('Lower parameters changed')
        if upper.frozen_backbone:
            for k,h in frozen.items():
                if h!=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in getattr(upper,k).state_dict().values())).hexdigest():raise ValueError('Frozen backbone changed')
        row=dict(episode=ep,seed=seed,**totals,upper_loss=loss,action_trace_hash=action_trace.hexdigest(),exogenous_trace_hash=trace.hexdigest())
        r.res.atomic_json(path/'episodes'/f'{ep:05d}.json',row)
        cp=path/'recovery'/f'episode_{ep:05d}';upper.save(cp,cc)
        torch.save(dict(env=env,episode=ep+1,contract=cc,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state()),cp/'state.pt')
        r.res.atomic_json(path/'latest.json',dict(checkpoint=str(cp.relative_to(path)),files={f.name:sha(f) for f in cp.iterdir() if f.is_file()}))
        r.res.atomic_json(path/'progress.json',dict(episode=ep+1,episodes=p['episodes'],pid=os.getpid(),dsr=totals['success']/max(totals['generated'],1)))
        if args.stop_after and ep+1>=args.stop_after:return
    upper.save(path/'checkpoint',cc)
    receipt(r,path,contract,episodes=p['episodes'],initial_fingerprint=initial_actor,lower_fixed=True,
            frozen_backbone=upper.frozen_backbone,checkpoint_contract=cc,
            checkpoint_sha256=sha(path/'checkpoint/upper_adaptation.pt'))


def evaluate(args,r,p,models,contract):
    import config
    import numpy as np
    import torch
    from utils.canonical_experiment import digest,exogenous_frame
    path=folder(args);path.mkdir(parents=True,exist_ok=True)
    if complete(path,contract):return
    lower=r.read_lower(args,read(args.output/'protocol.json'))
    upper=None
    if args.method!='frozen':
        torch.manual_seed(r.res.trajectory_seed(p['original_protocol_sha256'],args.block,'new_branch_init'))
        upper=build_upper(args,r,p,models)
        a=copy.copy(args);a.phase=('r1' if args.method in R1 else 'r2')+'_training';a.workload=p['training_workloads'][str(args.block)];a.episode=-1
        training=folder(a);doc=read(training/'result.json')
        if sha(training/'checkpoint/upper_adaptation.pt')!=doc['checkpoint_sha256']:raise ValueError('Training checkpoint changed')
        upper.load(training/'checkpoint',doc['checkpoint_contract'])
    frozen=r.base.policy_fingerprint([lower,upper,models['upper']])
    env,obs=r.episode_env(args.workload,args.episode)
    rng=np.random.default_rng(r.res.trajectory_seed(p['original_protocol_sha256'],args.block,args.workload,args.episode,'search'))
    trace=hashlib.sha256();rows=[];totals={k:0. for k in ('generated','admitted','success','potential','energy','mbs','fairness')}
    for step in range(p['steps']):
        trace.update(digest(exogenous_frame(env)).encode());potential=sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)
        motion,actions,info,_,_,_=r.policy(env,obs,'learned' if upper is None else 'full',models,family='complete',lower=lower,upper=upper,rng=rng,budget=p['matching_budget'])
        obs,_,metrics=env.step(motion,offloading_actions=actions);r.base.assert_step(metrics,info);obs=np.asarray(obs,np.float32)
        totals['potential']+=potential
        for k,n in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy'),('mbs','service_offloads_mbs'),('fairness','fairness')]:totals[k]+=float(metrics[n])
        rows.append(dict(step=step,metrics=metrics,potential=potential,**info))
    if frozen!=r.base.policy_fingerprint([lower,upper,models['upper']]):raise ValueError('Evaluation changed parameters')
    r.res.atomic_json(path/'episode.json',dict(summary=totals,exogenous_trace_hash=trace.hexdigest(),steps=rows))
    receipt(r,path,contract,model_fingerprint=frozen)


def worker(args,r):
    p=read(args.results_root/'protocol.json')
    if p['original_protocol_sha256']!=sha(args.output/'protocol.json'):raise ValueError('Original protocol changed')
    for f,h in p['files'].items():
        if sha(args.adaptation_root/f)!=h:raise ValueError('Adaptation release changed')
    if [args.phase,args.method,args.block,args.workload,args.episode] not in p['tasks']:raise ValueError('Unregistered task')
    resource=args.resources/'isolated' if args.phase.endswith('_timing') else args.resources
    with r.res.task_lock(args.results_root/'task_locks'/('_'.join(map(str,[args.phase,args.method,args.block,args.workload,args.episode]))+'.lock')):
        with r.res.CpuLease(resource,task=['upper_adaptation',args.phase,args.method,args.block,args.workload,args.episode],kind='environment'):
            old,_=r.bootstrap(args)
            load('marl_models.sf_hlm_upper_adaptation',args.adaptation_root/'marl_models/sf_hlm_upper_adaptation.py')
            models,source=r.base.configure(args,old,historical=True)
            contract=dict(**source,adaptation_protocol_sha256=sha(args.results_root/'protocol.json'),method=args.method,
                          lower_checkpoint_sha256=read(args.output/'lower_training/complete/learned'/f'block_{args.block}'/'result.json')['checkpoint_sha256'],
                          phase=args.phase,workload=args.workload,episode=args.episode)
            if args.phase.endswith('_training'):train(args,r,p,models,contract)
            else:evaluate(args,r,p,models,contract)


def run(args,r,p,phases):
    tasks=[t for t in p['tasks'] if t[0] in phases];failed=[];done=0
    def one(t):
        cmd=[sys.executable,str(args.adaptation_root/'scripts/analysis/sf_hlm_upper_adaptation.py'),'worker']
        for k in ('source_root','suite_root','extension_root','output','legacy_results','resources','adaptation_root','results_root'):
            cmd+=['--'+k.replace('_','-'),str(getattr(args,k))]
        for k,v in zip(('phase','method','block','workload','episode'),t):cmd+=['--'+k,str(v)]
        log=args.results_root/'job_logs'/('_'.join(map(str,t))+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=min(len(tasks),read(args.resources/'profile.json')['slots'])) as pool:
        fs={pool.submit(one,t):t for t in tasks}
        for f in as_completed(fs):
            try:f.result();done+=1
            except Exception as exc:failed.append(dict(task=fs[f],error=str(exc)))
            r.res.atomic_json(args.results_root/'status.json',dict(phases=phases,completed=done,expected=len(tasks),failures=failed,pid=os.getpid()))
    if failed:raise RuntimeError('Adaptation tasks failed: '+str(failed))


def aggregate(args,r,p,stage):
    import numpy as np
    r.base.np=np
    metrics={};raw={};traces={}
    names=R1 if stage=='r1' else R2
    for b in p['historical_blocks']:
        training=[]
        for m in names:
            a=copy.copy(args);a.phase=stage+'_training';a.method=m;a.block=b;a.workload=p['training_workloads'][str(b)];a.episode=-1
            path=folder(a);doc=read(path/'result.json')
            for f,h in doc['files'].items():
                if sha(path/f)!=h:raise ValueError('Training file changed')
            training.append((doc,read(path/'episodes/00000.json')))
        if training[0][0]['initial_fingerprint']!=training[1][0]['initial_fingerprint']:
            raise ValueError('Matched upper initialization differs')
        if stage=='r1' and training[0][1]['action_trace_hash']!=training[1][1]['action_trace_hash']:
            raise ValueError('First paired R1 rollout differs before any update')
    for phase,method,b,w,e in p['tasks']:
        if phase not in (stage+'_eval','reference_eval'):continue
        a=copy.copy(args);a.phase=phase;a.method=method;a.block=b;a.workload=w;a.episode=e;path=folder(a)
        doc=read(path/'result.json')
        for f,h in doc['files'].items():
            if sha(path/f)!=h:raise ValueError('Raw result changed')
        row=read(path/'episode.json');s=row['summary'];trace=row['exogenous_trace_hash'],s['potential']
        if (w,e) in traces and traces[w,e]!=trace:raise ValueError('Exogenous pairing failed')
        traces[w,e]=trace;raw[str(path/'episode.json')]=sha(path/'episode.json')
        u=metrics.setdefault((method,b),dict(totals={k:0. for k in s},times=[],steps=0))
        for k in s:u['totals'][k]+=s[k]
        for k,n in [('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('energy','energy')]:
            if not np.isclose(s[k],sum(x['metrics'][n] for x in row['steps']),rtol=1e-10):raise ValueError('Summary count mismatch')
        u['times'].extend(x['policy_seconds'] for x in row['steps']);u['steps']+=len(row['steps'])
        for x in row['steps']:
            if x['diagnostic']['matching_calls']>p['matching_budget'] or x['diagnostic']['chosen_predicted']['energy']>x['diagnostic']['common_energy_cap']+1e-7:raise ValueError('Matching constraint failed')
    out={}
    for (m,b),u in metrics.items():
        s=u['totals'];out[m,b]=dict(dsr=s['success']/s['generated'],potential_dsr=s['success']/s['potential'],admission=s['admitted']/s['generated'],
                                  conditional=s['success']/max(s['admitted'],1),energy=s['energy'],mbs_share=s['mbs']/max(s['admitted'],1),fairness=s['fairness']/u['steps'],p95=float(np.quantile(u['times'],.95)))
    groups={m:{k:float(np.mean([out[m,b][k] for b in p['historical_blocks']])) for k in out[m,p['historical_blocks'][0]]} for m in (*names,'frozen')}
    candidate=names[0];comparisons={m:{k:r.base.paired([out[candidate,b][k]-out[m,b][k] for b in p['historical_blocks']]) for k in ('dsr','admission','conditional','fairness')} for m in (names[1],'frozen')}
    energy={m:float(np.mean([out[candidate,b]['energy']/out[m,b]['energy'] for b in p['historical_blocks']])) for m in (names[1],'frozen')}
    result=dict(status='PASS',stage=stage,exploratory=True,confirmation=False,groups=groups,comparisons=comparisons,energy_ratios=energy,
                block_metrics={f'{m}:{b}':v for (m,b),v in out.items()},raw_files=raw,exogenous_pairing=True)
    r.res.atomic_json(args.results_root/'analysis'/f'{stage}.json',result)
    # Timings use reserved cores, and equality with ordinary evaluation is
    # checked rather than assuming concurrency cannot affect search outputs.
    times={}
    for phase,m,b,w,e in p['tasks']:
        if phase!=stage+'_timing':continue
        a=copy.copy(args);a.phase=phase;a.method=m;a.block=b;a.workload=w;a.episode=e;path=folder(a)
        d=read(path/'result.json')
        for f,h in d['files'].items():
            if sha(path/f)!=h:raise ValueError('Timing raw result changed')
        row=read(path/'episode.json');a.phase=stage+'_eval';quality=read(folder(a)/'episode.json')
        if row['summary']!=quality['summary'] or [s['assignment'] for s in row['steps']]!=[s['assignment'] for s in quality['steps']]:raise ValueError('Timing changes counters/actions')
        times.setdefault((m,b),[]).extend(x['policy_seconds'] for x in row['steps'])
    result['isolated_p95']={f'{m}:{b}':float(np.quantile(t,.95)) for (m,b),t in times.items()}
    r.res.atomic_json(args.results_root/'analysis'/f'{stage}.json',result)
    return result


def pipeline(args,r):
    p=make_protocol(args);file=args.results_root/'protocol.json'
    if file.exists() and read(file)!=p:raise ValueError('Registered adaptation protocol differs')
    r.res.atomic_json(file,p)
    with r.res.task_lock(args.results_root/'controller.lock'):
        stop=threading.Event();telemetry=threading.Thread(target=r.res.telemetry,args=(args.resources,stop),daemon=True);telemetry.start()
        try:
            run(args,r,p,['r1_training','reference_eval']);run(args,r,p,['r1_eval']);run(args,r,p,['r1_timing']);first=aggregate(args,r,p,'r1')
            r.res.atomic_json(args.results_root/'transition.json',dict(r1_completed=True,selected_engine='original_reference_locked_before_results',reason=p['transition']))
            run(args,r,p,['r2_training']);run(args,r,p,['r2_eval']);run(args,r,p,['r2_timing']);second=aggregate(args,r,p,'r2')
            promising=second['comparisons']['residual_zero']['dsr']['mean']>0 and sum(v>0 for v in second['comparisons']['residual_zero']['dsr']['values'])>=4
            promising &= second['comparisons']['frozen']['dsr']['mean']>=0 and second['energy_ratios']['frozen']<=1.05 and max(second['isolated_p95'].values())<1
            r.res.atomic_json(args.results_root/'status.json',dict(phase='SMOKE_COMPLETE' if args.smoke else 'UPPER_ADAPTATION_PILOT_COMPLETE',completed=True,
                screening_promising=bool(promising and not args.smoke),confirmation_started=False,smoke=args.smoke,scientific_success_claim=False))
        finally:stop.set();telemetry.join(timeout=5)


def main():
    if sys.platform!='linux':raise RuntimeError('All upper adaptation training and simulation runs on remote Linux')
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('pipeline','worker','protocol'))
    for name in ('source-root','suite-root','extension-root','output','legacy-results','resources','adaptation-root','results-root'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--phase');parser.add_argument('--method')
    for name in ('block','workload','episode','stop-after'):parser.add_argument('--'+name,type=int,default=-1 if name=='episode' else 0)
    parser.add_argument('--family',default='complete');parser.add_argument('--seed',type=int,default=690001)
    args=parser.parse_args()
    for k in ('source_root','suite_root','extension_root','output','legacy_results','resources','adaptation_root','results_root'):setattr(args,k,getattr(args,k).expanduser().resolve())
    r=runner(args)
    if args.mode=='worker':worker(args,r)
    elif args.mode=='pipeline':pipeline(args,r)
    else:print(json.dumps(make_protocol(args),indent=2))


if __name__=='__main__':main()
