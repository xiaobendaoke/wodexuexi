"""Frozen-model mechanism diagnostics. No optimizer or training is constructed."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK='chapter5_dynamic_service_20261028'
REVISION='revision_e2e_20261030'
DIAGNOSTIC='diagnostic_20261030'


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)


def locations(args):
    root=args.task/'results'/TASK/REVISION
    return root,root/DIAGNOSTIC,args.task/'output'/TASK/'validation'/REVISION/DIAGNOSTIC


def jobs():
    result=[]
    for method in ('trajectory_matching','joint','history_allocation'):
        for block in range(3):
            for workload in range(2):
                modes=[('replay',0)] if method!='history_allocation' else [('history_sample',0)]
                if method=='trajectory_matching':modes.extend([('sample_motion',r) for r in range(2)])
                elif method=='joint':
                    modes.extend((mode,r) for mode in ('sample_alloc','sample_motion','sample_both') for r in range(2))
                    modes.extend((mode,0) for mode in ('matching','history','zero_feedback','zero_summary'))
                for mode,rep in modes:
                    result.append(dict(method=method,block=block,workload=workload,mode=mode,rep=rep,
                        id=f'{method}_{block}_{workload}_{mode}_{rep}'))
    return result


def initialize(args):
    args.source=args.source.resolve();args.task=args.task.resolve()
    sys.path.insert(0,str(args.source))
    os.environ.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    root,out,val=locations(args)
    definition=json.loads((root/'protocol.json').read_text())
    for relative,d in definition['source'].items():
        if sha(args.source/relative)!=d:raise RuntimeError(f'frozen source changed: {relative}')
    args.contract=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
    args.diagnostic_sha256=sha(Path(__file__))
    return root,out,val


def worker(args):
    root,out,val=initialize(args)
    from scripts.analysis.dynamic_service_experiment import configure_runtime
    torch=configure_runtime()
    import numpy as np
    from environment.dynamic_service_v2 import DynamicServiceEnvV2,boundary_actions
    from environment.dynamic_service import evaluate,project_positions
    from marl_models.dynamic_service_joint import JointPolicy,JointCritic,allocation_summary,branch_gradients
    from marl_models.dynamic_service_hmarl import tensor
    from utils.dynamic_service_joint_rollout import causal_returns,slot_surrogate
    from utils.dynamic_service_baselines import HistoryController,matching_search
    from utils.dynamic_service_scenarios import Scenario
    ident=f'{args.method}_{args.block}_{args.workload}_{args.mode}_{args.rep}'
    destination=out/'cases'/(ident+'.json')
    if destination.exists():
        previous=json.loads(destination.read_text())
        if previous['diagnostic_sha256']!=args.diagnostic_sha256:raise RuntimeError('diagnostic code changed')
        return
    file=root/'models/pilot'/str(args.block)/args.method/'0040.pt'
    saved=torch.load(file,map_location='cpu',weights_only=False)
    assert saved['episode']==40 and saved['contract']==args.contract
    actor=JointPolicy(args.method);actor.load_state_dict(saved['actor']);actor.eval()
    critic=JointCritic(args.method);critic.load_state_dict(saved['critic']);critic.eval()
    expected_path=root/'evaluation/pilot'/str(args.block)/args.method/'40'/f'D_{args.workload}_0.json'
    expected=json.loads(expected_path.read_text());assert sha(file)==expected['checkpoint_sha256']
    seed=int.from_bytes(hashlib.sha256(f'joint_diagnostic/{ident}'.encode()).digest()[:4],'big')
    torch.manual_seed(seed)
    env=DynamicServiceEnvV2(Scenario(expected['workload_seed'],'D'))
    hidden=actor.initial_state();ch=critic.initial_state();controller=HistoryController()
    if args.mode=='zero_summary':actor.condition=False
    sample_allocation=args.mode in ('sample_alloc','sample_both','history_sample')
    sample_motion=args.mode in ('sample_motion','sample_both')
    use_matching=args.method=='trajectory_matching' or args.mode=='matching'
    use_history=args.method=='history_allocation' or args.mode=='history'
    collect=args.mode in ('replay','sample_both','history_sample') or (args.method=='trajectory_matching' and args.mode=='sample_motion')
    values=[];trace=[];probes=[];snapshots=[];motion=defaultdict(list);totals=defaultdict(int)
    replayed=0;began=time.perf_counter()
    for t in range(1000):
        before=env.positions.copy();obs=env.observe();batch=env.batch();controller.update(obs,before)
        policy_obs=dict(obs,feedback=np.zeros_like(obs['feedback'])) if args.mode=='zero_feedback' else obs
        hb=hidden.clone();cb=ch.clone()
        fixed=matching_search(batch) if use_matching else None
        sample=actor.act(policy_obs,batch,hidden,stochastic=sample_allocation,fixed_assignment=fixed)
        state=tensor(sample['hidden'],actor.device)
        dist=actor.motion_distribution(state,batch,sample['assignment'])
        raw=tensor(sample['raw'],actor.device) if sample_allocation and sample_motion else (dist.sample() if sample_motion else dist.mean)
        sample.update(raw=raw.detach().numpy(),motion_log=dist.log_prob(raw).sum(-1).detach().numpy(),
                      motion_mean=dist.mean.detach().numpy(),motion_std=dist.stddev.detach().numpy())
        action=boundary_actions(sample['raw'],before)
        if use_history:action=controller.actions(before)
        hidden=state.detach()
        if collect:
            if use_matching:sample['risk']=actor.allocation.risk_trajectory(batch,sample['assignment']).numpy() if len(batch) else np.empty((0,6,11),np.float32)
            prediction=critic.predict(policy_obs,batch,ch,sample['assignment'],t,risk=sample['risk'])
            ch=tensor(prediction['critic_hidden'],critic.device)
        fallback=bool(expected['rows'][t]['decision_timeout']) if args.mode=='replay' else False
        row=env.step(action,sample['assignment'],decision_timeout=fallback)
        for key in ('generated','admitted','success','uncovered','deadline_misses','mbs','boundary_uav_steps','hovering_uav_steps'):
            totals[key]+=row[key]
        if args.mode=='replay':
            for key in ('generated','admitted','success','uncovered','deadline_misses','mbs'):
                assert row[key]==expected['rows'][t][key],(ident,t,key)
            np.testing.assert_allclose(env.positions,expected['rows'][t]['positions'],atol=1e-6,rtol=0)
            replayed+=1
        bounded=np.clip(action,-1,1);norm=np.linalg.norm(bounded,axis=1,keepdims=True)
        intended=bounded/np.maximum(norm,1e-12)*np.minimum(norm,1)*15
        actual=env.positions-before;cn=np.linalg.norm(intended,axis=1);an=np.linalg.norm(actual,axis=1)
        residual=np.linalg.norm(actual-intended,axis=1)
        pair=np.linalg.norm(before[:,None]-before[None],axis=-1)+np.eye(5)*1000
        near_neighbor=pair.min(1)<201;near_wall=((before<65)|(before>635)).any(1)
        motion['commanded_m'].extend(cn);motion['actual_m'].extend(an);motion['residual_m'].extend(residual)
        motion['blocked'].extend((cn>1)&(an<.1));motion['wall'].extend(near_wall);motion['neighbor'].extend(near_neighbor)
        motion['blocked_wall'].extend((cn>1)&(an<.1)&near_wall)
        motion['blocked_neighbor'].extend((cn>1)&(an<.1)&near_neighbor)
        motion['raw_mean_norm'].extend(np.linalg.norm(sample['motion_mean'],axis=1))
        motion['latent_std'].extend(sample['motion_std'].ravel())
        h_action=controller.actions(before);h_after=project_positions(before,h_action)
        motion['history_same_geometry_m'].extend(np.linalg.norm(h_after-before,axis=1))
        trace.append(dict(t=t,generated=row['generated'],admitted=row['admitted'],success=row['success'],mbs=row['mbs'],
            commanded_m=float(cn.mean()),actual_m=float(an.mean()),blocked=int(((cn>1)&(an<.1)).sum())))
        if collect:
            values.append(dict(observation=policy_obs,batch=batch,hidden_before=hb.numpy(),critic_hidden_before=cb.numpy(),
                **sample,**prediction,reward=row['reward'],time=t))
        if t%125==0:
            # These are same-state input probes, not trained ablations.
            alternatives={}
            with torch.no_grad():
                base_state=actor.encoder(policy_obs,hb)
                base_dist=actor.motion_distribution(base_state,batch,sample['assignment'])
                zeros=dict(policy_obs,feedback=np.zeros_like(policy_obs['feedback']))
                no_demand=dict(policy_obs,ue=policy_obs['ue'].copy());no_demand['ue'][:,:,2:]=0
                for name,altered,h in [('no_previous_feedback',zeros,hb),('no_current_demand',no_demand,hb),('reset_memory',policy_obs,actor.initial_state())]:
                    s=actor.encoder(altered,h);d=actor.motion_distribution(s,batch,sample['assignment'])
                    a=boundary_actions(d.mean.numpy(),before);pos=project_positions(before,a)
                    alternatives[name]=dict(raw_mean_delta=float((d.mean-base_dist.mean).norm(dim=1).mean()),
                        mean_projected_m=float(np.linalg.norm(pos-before,axis=1).mean()))
                summary=np.zeros(18,np.float32)
                x=torch.cat((base_state,base_state.mean(0,keepdim=True).expand(5,-1),tensor(summary,actor.device).expand(5,-1)),-1)
                mean=actor.motion(x);pos=project_positions(before,boundary_actions(mean.numpy(),before))
                alternatives['no_current_plan']=dict(raw_mean_delta=float((mean-base_dist.mean).norm(dim=1).mean()),
                    mean_projected_m=float(np.linalg.norm(pos-before,axis=1).mean()))
            probes.append(dict(t=t,alternatives=alternatives,base_raw_norm=float(base_dist.mean.norm(dim=1).mean())))
            matched=matching_search(batch);best=evaluate(batch,matched)
            current=evaluate(batch,sample['assignment'])
            snapshot=dict(t=t,requests=len(batch),success=current['success'],matching_success=best['success'],
                mbs=current['mbs'],mbs_tx_alone_misses=int(((sample['assignment']==5)&(batch.tx[:,5]>batch.deadline)).sum()),
                recoverable_mbs=[])
            if not use_matching and len(batch):
                p=sample['allocation_probs'];sorted_p=np.sort(p,axis=1)
                snapshot.update(mean_mbs_probability=float(p[:,5].mean()),
                    mean_top_probability=float(p.max(1).mean()),mean_top_margin=float((sorted_p[:,-1]-sorted_p[:,-2]).mean()))
            for i in np.flatnonzero((sample['assignment']==5)&~current['satisfied']):
                gains=[]
                for node in np.flatnonzero(batch.legal[i,:5]):
                    alt=sample['assignment'].copy();alt[i]=node
                    gains.append(evaluate(batch,alt)['success']-current['success'])
                gain=max(gains,default=0)
                if gain>0:snapshot['recoverable_mbs'].append(dict(request_index=int(i),success_gain=int(gain)))
            snapshots.append(snapshot)
    calibration=None;gradient=[];credit=None
    if collect:
        rewards=np.array([r['reward'] for r in values]);full=np.cumsum(rewards[::-1])[::-1];future=full-rewards
        starts=np.array([r['prefix_values'][0] if len(r['prefix_values']) else r['motion_value'] for r in values])
        movement=np.array([r['motion_value'] for r in values])
        targets=causal_returns(values,allocation_only=args.method=='history_allocation')
        exact=causal_returns(values,lam=1.)
        for t,r in enumerate(values):
            assert abs(exact[t]['motion_target']-future[t])<1e-7
            if len(r['batch']):np.testing.assert_allclose(exact[t]['prefix_targets'],full[t],atol=2e-5,rtol=0)
        for t in (0,250,500,998,999):
            indices=np.arange(t+1,1000)
            expanded=float(np.sum((.99**(indices-t-1))*.01*starts[indices]+(.99**(indices-t))*rewards[indices]))
            if args.method!='history_allocation':assert abs(targets[t]['motion_target']-expanded)<1e-6
        def errors(pred,target):
            err=pred-target
            return dict(bias=float(err.mean()),rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.abs(err).mean()),
                explained_variance=float(1-err.var()/max(target.var(),1e-12)),pred_mean=float(pred.mean()),realized_target_mean=float(target.mean()))
        target_prefix=rewards if args.method=='history_allocation' else full
        calibration=dict(prefix_start=errors(starts,target_prefix),movement=errors(movement,future),
            immediate_reward_mean=float(rewards.mean()),immediate_reward_std=float(rewards.std()),
            temporal_bookkeeping_checks='PASS: direct gamma=1 returns and independently expanded physical-time lambda targets',
            caveat='One realized rollout per diagnostic sampling replicate; critic error includes Monte Carlo noise and horizon correlation.')
        first_adv=np.array([tar['prefix_targets'][0]-r['prefix_values'][0] if len(r['batch']) else np.nan for r,tar in zip(values,targets)])
        valid=np.isfinite(first_adv);credit=dict(advantage_std=float(first_adv[valid].std()),
            immediate_reward_std=float(rewards[valid].std()),
            correlation_first_adv_immediate_reward=float(np.corrcoef(first_adv[valid],rewards[valid])[0,1]) if valid.sum()>1 and first_adv[valid].std()>0 and rewards[valid].std()>0 else None)
        rescued=[]
        for snap in snapshots:
            t=snap['t'];r=values[t];target=targets[t]['prefix_targets']
            for item in snap['recoverable_mbs']:
                i=item['request_index'];adv=float(target[i]-r['prefix_values'][i])
                rescued.append(dict(t=t,**item,logged_prefix_advantage=adv,positive_advantage=adv>0))
        credit['recoverable_mbs_requests']=rescued
        if args.method=='joint' and args.mode=='sample_both' and args.rep==0:
            all_adv=[]
            for r,tar in zip(values,targets):
                all_adv.extend((tar['prefix_targets']-r['prefix_values']).tolist());all_adv.append(tar['motion_target']-r['motion_value'])
            scale=max(float(np.std(all_adv)),.05)
            for start in (0,256,512,768):
                h=tensor(values[start]['hidden_before'],actor.device);al=[];ml=[];replay_error=0.
                for t in range(start,start+64):
                    r=values[t];tar=targets[t];pred=actor.replay(r['observation'],r['batch'],h,r);h=pred['hidden']
                    if len(r['batch']):
                        old=tensor(r['allocation_log'],actor.device)
                        replay_error=max(replay_error,float((pred['allocation_log'].detach()-old).abs().max()))
                        al.append(slot_surrogate(pred['allocation_log'],old,tensor((tar['prefix_targets']-r['prefix_values'])/scale,actor.device)))
                    else:al.append(pred['motion_log'].sum()*0)
                    old=tensor(r['motion_log'],actor.device)
                    replay_error=max(replay_error,float((pred['motion_log'].detach()-old).abs().max()))
                    ml.append(slot_surrogate(pred['motion_log'],old,torch.full((5,),(tar['motion_target']-r['motion_value'])/scale)))
                assert replay_error<1e-4,('probability replay mismatch',ident,start,replay_error)
                diag=branch_gradients(torch.stack(al).mean(),torch.stack(ml).mean(),actor.encoder.parameters())
                gradient.append(dict(start=start,probability_replay_max_error=replay_error,**diag,
                    allocation_to_motion_norm_ratio=diag['allocation_encoder_grad']/max(diag['motion_encoder_grad'],1e-12)))
    motion_summary={key:float(np.mean(v)) for key,v in motion.items()}
    result=dict(id=ident,method=args.method,block=args.block,workload=args.workload,mode=args.mode,rep=args.rep,
        contract=args.contract,diagnostic_sha256=args.diagnostic_sha256,model_sha256=sha(file),
        workload_fingerprint=env.tape.fingerprint(),diagnostic_seed=seed,
        totals=dict(totals),dsr=totals['success']/totals['generated'],admission=totals['admitted']/totals['generated'],
        conditional=totals['success']/max(totals['admitted'],1),motion=motion_summary,
        replayed_steps=replayed,probes=probes,snapshots=snapshots,critic_calibration=calibration,
        allocation_credit=credit,shared_gradient=gradient,trace=trace,seconds=time.perf_counter()-began,
        no_parameter_updates=True,no_optimization_constructed=True,formal_performance_evidence=False,
        limitations='Closed-loop frozen-policy interventions ignore added diagnostic wall-clock overhead; no retraining ablation or deployment speed claim.')
    atomic(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trace','probes','snapshots')}),flush=True)


def summarize(args):
    root,out,val=initialize(args)
    import statistics as st
    groups=defaultdict(list);all_cases=[]
    for job in jobs():
        path=out/'cases'/(job['id']+'.json');d=json.loads(path.read_text())
        assert d['contract']==args.contract and d['diagnostic_sha256']==args.diagnostic_sha256
        assert d['totals']['generated']==50000 and len(d['trace'])==1000
        all_cases.append(d);groups[(d['method'],d['mode'])].append(d)
    summary={}
    for (method,mode),ds in groups.items():
        by_block={str(b):st.mean(d['dsr'] for d in ds if d['block']==b) for b in range(3)}
        summary[f'{method}/{mode}']=dict(cases=len(ds),mean_dsr=st.mean(by_block.values()),by_block=by_block,
            admission=st.mean(d['admission'] for d in ds),conditional=sum(d['totals']['success'] for d in ds)/sum(d['totals']['admitted'] for d in ds),
            **{k:st.mean(d['motion'][k] for d in ds) for k in ('commanded_m','actual_m','residual_m','blocked','wall','neighbor','history_same_geometry_m','raw_mean_norm','latent_std')})
    grads=[g for d in all_cases for g in d['shared_gradient']]
    credit=[x for d in all_cases if d['method']=='joint' and d['mode']=='sample_both' for x in d['allocation_credit']['recoverable_mbs_requests']]
    calibration=defaultdict(list)
    for d in all_cases:
        if d['critic_calibration']:calibration[d['method']+'/'+d['mode']].append(d['critic_calibration'])
    calibration_summary={k:dict(cases=len(v),mean_prefix_rmse=st.mean(x['prefix_start']['rmse'] for x in v),
        mean_movement_rmse=st.mean(x['movement']['rmse'] for x in v),
        mean_prefix_bias=st.mean(x['prefix_start']['bias'] for x in v),
        mean_movement_bias=st.mean(x['movement']['bias'] for x in v),
        mean_immediate_reward_std=st.mean(x['immediate_reward_std'] for x in v)) for k,v in calibration.items()}
    probes=defaultdict(list)
    for d in all_cases:
        if d['mode']=='replay':
            for probe in d['probes']:
                for k,v in probe['alternatives'].items():probes[d['method']+'/'+k].append(v['raw_mean_delta'])
    result=dict(status='COMPLETE_FROZEN_POLICY_DIAGNOSTIC',contract=args.contract,diagnostic_sha256=args.diagnostic_sha256,
        cases=len(all_cases),replayed_steps=sum(d['replayed_steps'] for d in all_cases),
        summary=summary,critic_calibration=calibration_summary,
        same_state_mean_action_sensitivity={k:st.mean(v) for k,v in probes.items()},
        shared_gradient=dict(samples=len(grads),median_allocation_to_motion_ratio=st.median(g['allocation_to_motion_norm_ratio'] for g in grads),
            mean_cosine=st.mean(g['encoder_grad_cosine'] for g in grads),negative_cosines=sum(g['encoder_grad_cosine']<0 for g in grads),
            max_probability_replay_error=max(g['probability_replay_max_error'] for g in grads)),
        recoverable_mbs_credit=dict(requests=len(credit),positive_prefix_advantages=sum(c['positive_advantage'] for c in credit),
            positive_fraction=sum(c['positive_advantage'] for c in credit)/max(len(credit),1)),
        limitations=['Two existing development workloads, three frozen training seeds; sampling repeats are not training repetitions',
            'Input zeroing is a distribution intervention, not a retrained ablation',
            'Critic calibration uses realized returns and does not by itself prove a training bug',
            'No trained weights, original evaluations, or formal protocol changed'])
    atomic(out/'analysis.json',result);print(json.dumps(result,indent=2),flush=True)


def run(args):
    root,out,val=initialize(args);val.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(val/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protected_path=val/'protected_before.json'
    if not protected_path.exists():
        atomic(protected_path,{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and DIAGNOSTIC not in p.relative_to(root).parts and p.suffix not in ('.lock','.tmp')})
    cpus=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topo=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((topo/'physical_package_id').read_text(),(topo/'core_id').read_text())
        if key not in seen:cpus.append(cpu);seen.add(key)
    cpus=cpus[8:8+args.workers];pending=jobs();done=set();failed={};active=[];started=time.monotonic()
    atomic(val/'manifest.json',dict(started=now(),pid=os.getpid(),jobs=pending,cpus=cpus,contract=args.contract,
        diagnostic_sha256=args.diagnostic_sha256,no_training=True))
    for job in pending:
        path=out/'cases'/(job['id']+'.json')
        if path.exists():
            if json.loads(path.read_text())['diagnostic_sha256']!=args.diagnostic_sha256:raise RuntimeError('mixed diagnostic versions')
            done.add(job['id'])
    while len(done)+len(failed)<len(pending):
        for item in active[:]:
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();active.remove(item)
            if code==0 and (out/'cases'/(item['job']['id']+'.json')).exists():done.add(item['job']['id'])
            else:failed[item['job']['id']]=code
        busy={i['cpu'] for i in active};free=[c for c in cpus if c not in busy];running={i['job']['id'] for i in active}
        for job in pending:
            if not free:break
            if job['id'] in done or job['id'] in failed or job['id'] in running:continue
            cpu=free.pop(0);log=(val/(job['id']+'.log')).open('a')
            command=['taskset','-c',str(cpu),sys.executable,str(Path(__file__).resolve()),'worker','--task',str(args.task),'--source',str(args.source)]
            for key in ('method','mode','block','workload','rep'):command.extend(['--'+key,str(job[key])])
            proc=subprocess.Popen(command,cwd=args.source,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(dict(process=proc,job=job,cpu=cpu,log=log))
        atomic(val/'status.json',dict(updated=now(),completed=len(done),total=len(pending),failed=failed,
            active=[dict(job=i['job']['id'],pid=i['process'].pid,cpu=i['cpu']) for i in active],elapsed_seconds=time.monotonic()-started))
        if active:time.sleep(3)
    if not failed:summarize(args)
    protected=json.loads(protected_path.read_text())
    changed=[name for name,d in protected.items() if not (root/name).exists() or sha(root/name)!=d]
    final=dict(status='PASS_DIAGNOSTIC_COMPLETED' if not failed and not changed else 'FAIL',updated=now(),completed=len(done),total=len(pending),
        failed=failed,changed_original_files=changed,protected_files=len(protected),no_parameter_updates=True,contract=args.contract)
    atomic(val/'completion.json',final);atomic(val/'status.json',final)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('worker','run','summarize'))
    p.add_argument('--source',type=Path,required=True);p.add_argument('--task',type=Path,required=True)
    p.add_argument('--method',choices=('joint','trajectory_matching','history_allocation'),default='joint')
    p.add_argument('--mode',default='replay');p.add_argument('--block',type=int,default=0);p.add_argument('--workload',type=int,default=0);p.add_argument('--rep',type=int,default=0)
    p.add_argument('--workers',type=int,default=96)
    args=p.parse_args()
    globals()[args.command](args)


if __name__=='__main__':main()
