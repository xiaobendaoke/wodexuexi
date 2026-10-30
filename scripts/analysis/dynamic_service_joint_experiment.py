"""Isolated D/1.0 joint-policy development; confirmation is deliberately absent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis.dynamic_service_experiment import atomic_json,source_contract,configure_runtime
from utils.dynamic_service_scenarios import Scenario,seed_for

TASK='chapter5_dynamic_service_20261028'
REVISION='revision_e2e_20261030'
METHODS=('trajectory_matching','history_allocation','joint','joint_no_condition','joint_mlp')
CONTROLS=('history','hover','mpc')


def protocol():
    return dict(revision=REVISION,source=source_contract(),development_only=True,confirmation_enabled=False,
        physical='v1 evaluator and v2 public-union/motion interface unchanged; serve before moving; no energy or queues',
        train_cell=['D',1.],evaluation_cells=[['D',1.],['A',1.]],methods=list(METHODS),controls=list(CONTROLS),
        budget=dict(blocks=3,episodes=40,steps=1000,checkpoints=[0,20,40],workloads=3,repeats=2),
        seeds=dict(training=[seed_for('dev/train',b) for b in range(3)],
            tape='intentional reuse of cleared dev/workload/{block},episode and dev/evaluation/{workload},repeat',
            confirmation='disabled; original confirmation collision remains unresolved'),
        actor='shared public-demand encoder; 128 GRU; autoregressive batch risk; latent boundary motion; single optimizer',
        mlp='shared per-UAV MLP, no attention or GRU; no request-node graph messages; sequential risk retained',
        conditioning='joint uses predicted final assignment loads; joint_no_condition zeroes this input only',
        critic='independent encoder and optimizer; remaining physical horizon; prefix and post-service stage values',
        credit='zero micro rewards; batch success/100 before movement; gamma=1; lambda=.99 only across physical slots',
        isolated_allocation='fixed history trajectory is independent of allocation; drop action-independent future return to reduce variance',
        ppo=dict(lr=3e-4,clip=.2,epochs=4,sequence=64,burn_in=16,gradient_clip=.5,
                 mean_cumulative_conditional_kl=.05,entropy_per_factor=.001,return_scale=100.,
                 surrogate='sum factors per physical slot then mean slots; conditional clipping approximation'),
        timeout='1 second full action path including observation/batch/controller; source-local assignment and hover fallback; proposal probabilities retained in training',
        timing='user-authorized maximum concurrency; calibrated training threads; loaded evaluation timing provisional; final serial 1-thread audit',
        decision='development learning/instability diagnostics only; no automatic 400-episode or confirmation launch')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_path(args,episode=None):
    base=args.output/'models'/args.split/str(args.block)/args.method
    return base/('latest.pt' if episode is None else f'{episode:04}.pt')


def save_checkpoint(path,state,torch):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.pt.tmp')
    torch.save(state,temporary);os.replace(temporary,path)


def identity(args):
    return dict(contract=args.contract,method=args.method,block=args.block,split=args.split,
                steps=args.steps,budget=args.episodes,seed=seed_for('dev/train',args.block))


def validate_checkpoint(saved,args):
    for key,value in identity(args).items():
        if saved.get(key)!=value:raise RuntimeError(f'checkpoint identity mismatch: {key}')


def run_episode(args,actor,critic,torch,episode=None):
    import numpy as np
    from environment.dynamic_service_v2 import DynamicServiceEnvV2
    from utils.dynamic_service_baselines import HistoryController,matching_search
    training=episode is not None
    seed=seed_for(f'dev/workload/{args.block}',episode) if training else seed_for(f'dev/evaluation/{args.workload}',args.repeat)
    env=DynamicServiceEnvV2(Scenario(seed,'D' if training else args.kind),1.)
    controller=HistoryController();ah=actor.initial_state() if actor else None
    ch=critic.initial_state() if critic else None
    rows=[];rollout=[];snapshots=[]
    keys=('generated','admitted','success','uncovered','deadline_misses','decision_timeout','mbs',
          'boundary_uav_steps','hovering_uav_steps')
    totals=dict.fromkeys(keys,0)
    for t in range(args.steps):
        started=time.perf_counter();obs=env.observe();batch=env.batch();controller.update(obs,env.positions)
        before=ah.numpy().copy() if actor else None
        if args.method in CONTROLS:
            if args.method=='history':actions=controller.actions(env.positions)
            elif args.method=='hover':actions=np.zeros((5,2))
            else:actions=controller.mpc(env,batch)
            assignment=matching_search(batch);sample=None
        else:
            fixed_assignment=matching_search(batch) if args.method=='trajectory_matching' else None
            fixed_actions=controller.actions(env.positions) if args.method=='history_allocation' else None
            sample=actor.act(obs,batch,ah,training,fixed_assignment,fixed_actions)
            actions=sample['actions'];assignment=sample['assignment'];ah=torch.tensor(sample['hidden'])
        elapsed=time.perf_counter()-started
        if training:
            if args.method=='trajectory_matching':
                sample['risk']=actor.allocation.risk_trajectory(batch,assignment).cpu().numpy() if len(batch) else np.empty((0,6,11),np.float32)
            cb=ch.numpy().copy()
            values=critic.predict(obs,batch,ch,assignment,t,risk=sample['risk']);ch=torch.tensor(values['critic_hidden'])
        row=env.step(actions,assignment,decision_timeout=elapsed>1.)
        row.update(t=t,seconds=elapsed,edge60_uav_steps=int(((env.positions<110)|(env.positions>590)).any(axis=1).sum()),
                   positions=env.positions.tolist())
        for key in keys:totals[key]+=row[key]
        rows.append(row)
        if training:
            rollout.append(dict(observation=obs,batch=batch,hidden_before=before,critic_hidden_before=cb,
                **sample,**values,reward=row['reward'],time=t,timeout=row['decision_timeout']))
        elif t%125==0:
            snapshots.append(dict(t=t,batch={k:getattr(batch,k).tolist() for k in batch.__dataclass_fields__},
                assignment=row['assignment'],success=row['success']))
    result=dict(totals=totals,dsr=totals['success']/max(totals['generated'],1),
        admission=totals['admitted']/max(totals['generated'],1),
        conditional=totals['success']/max(totals['admitted'],1),fairness=rows[-1]['fairness'],
        p95_seconds=float(np.percentile([r['seconds'] for r in rows],95)),
        workload_seed=seed,workload_fingerprint=env.tape.fingerprint(),rows=rows,snapshots=snapshots)
    return result,rollout


def train(args,torch):
    from marl_models.dynamic_service_joint import JointPolicy,JointCritic,update_joint
    seed=seed_for('dev/train',args.block);torch.manual_seed(seed)
    actor=JointPolicy(args.method);critic=JointCritic(args.method)
    aopt=torch.optim.Adam(actor.parameters(),lr=3e-4);copt=torch.optim.Adam(critic.parameters(),lr=3e-4)
    first=0;latest=model_path(args)
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False);validate_checkpoint(saved,args)
        if saved['training_threads']!=args.threads:raise RuntimeError('resume thread count changed')
        actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic'])
        aopt.load_state_dict(saved['aopt']);copt.load_state_dict(saved['copt']);torch.set_rng_state(saved['rng'])
        first=saved['episode']
        if saved.get('last_record') is not None:
            log=args.output/'training'/args.split/str(args.block)/args.method/f'{first:04}.json'
            if log.exists() and json.loads(log.read_text())!=saved['last_record']:raise RuntimeError('record/checkpoint mismatch')
            if not log.exists():atomic_json(log,saved['last_record'])
        if first in (0,20,args.episodes) and not model_path(args,first).exists():save_checkpoint(model_path(args,first),saved,torch)
    else:
        saved=dict(**identity(args),episode=0,training_threads=args.threads,actor=actor.state_dict(),critic=critic.state_dict(),
            aopt=aopt.state_dict(),copt=copt.state_dict(),rng=torch.get_rng_state(),last_record=None)
        save_checkpoint(latest,saved,torch);save_checkpoint(model_path(args,0),saved,torch)
    stop=min(args.stop_after if args.stop_after is not None else args.episodes,args.episodes)
    if first>stop:raise RuntimeError('stop precedes existing checkpoint')
    for episode in range(first,stop):
        started=time.perf_counter();result,rollout=run_episode(args,actor,critic,torch,episode)
        metrics=update_joint(actor,critic,aopt,copt,rollout)
        record=dict(**identity(args),episode=episode+1,training_cell=['D',1.],
            **{k:v for k,v in result.items() if k not in ('rows','snapshots')},
            update=metrics,seconds=time.perf_counter()-started,interaction_steps=(episode+1)*args.steps,
            actor_parameters=sum(p.numel() for p in actor.parameters()),critic_parameters=sum(p.numel() for p in critic.parameters()))
        saved=dict(**identity(args),episode=episode+1,training_threads=args.threads,actor=actor.state_dict(),critic=critic.state_dict(),
            aopt=aopt.state_dict(),copt=copt.state_dict(),rng=torch.get_rng_state(),last_record=record)
        save_checkpoint(latest,saved,torch)
        if episode+1 in (20,args.episodes):save_checkpoint(model_path(args,episode+1),saved,torch)
        atomic_json(args.output/'training'/args.split/str(args.block)/args.method/f'{episode+1:04}.json',record)
        print(json.dumps(record),flush=True)


def evaluation_path(args):
    return args.output/('timing_audit' if args.timing_audit else 'evaluation')/args.split/str(args.block)/args.method/str(args.checkpoint)/f'{args.kind}_{args.workload}_{args.repeat}.json'


def evaluate_job(args,torch):
    from marl_models.dynamic_service_joint import JointPolicy
    actor=None;checkpoint_hash=None
    if args.method in METHODS:
        path=model_path(args,args.checkpoint)
        saved=torch.load(path,map_location='cpu',weights_only=False);validate_checkpoint(saved,args)
        if saved['episode']!=args.checkpoint:raise RuntimeError('wrong evaluation checkpoint')
        actor=JointPolicy(args.method);actor.load_state_dict(saved['actor']);actor.eval();checkpoint_hash=digest(path)
        # Warm only the model. Environment/controller/history are reset below.
        from environment.dynamic_service_v2 import DynamicServiceEnvV2
        dummy=DynamicServiceEnvV2(Scenario(seed_for('dev/workload/0',0),'D'))
        for _ in range(3):actor.act(dummy.observe(),dummy.batch(),actor.initial_state())
    destination=evaluation_path(args)
    if destination.exists():
        saved=json.loads(destination.read_text())
        if saved['contract']!=args.contract or saved['checkpoint_sha256']!=checkpoint_hash:raise RuntimeError('evaluation reuse mismatch')
        return
    result,_=run_episode(args,actor,None,torch)
    result.update(**identity(args),checkpoint=args.checkpoint,checkpoint_sha256=checkpoint_hash,
        kind=args.kind,workload=args.workload,repeat=args.repeat,development_only=True,
        cpu_affinity=sorted(os.sched_getaffinity(0)),inference_threads=args.threads,timing_audit=args.timing_audit)
    atomic_json(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','snapshots')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('protocol','train','evaluate'))
    p.add_argument('--output',type=Path,default=REPO_ROOT/'results'/TASK/REVISION)
    p.add_argument('--split',choices=('smoke','pilot'),default='pilot')
    p.add_argument('--method',choices=METHODS+CONTROLS,default='joint')
    p.add_argument('--block',type=int,choices=range(3),default=0)
    p.add_argument('--episodes',type=int,default=40);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--stop-after',type=int);p.add_argument('--checkpoint',type=int,default=40)
    p.add_argument('--threads',type=int,choices=(1,2,4,8),default=1)
    p.add_argument('--timing-audit',action='store_true')
    p.add_argument('--kind',choices=('D','A'),default='D')
    p.add_argument('--workload',type=int,choices=range(3),default=0);p.add_argument('--repeat',type=int,choices=range(2),default=0)
    args=p.parse_args()
    if not 1<=args.steps<=1000 or args.episodes<1:p.error('invalid budget')
    if args.split=='pilot' and (args.steps!=1000 or args.episodes!=40):p.error('pilot is exactly 40 x 1000')
    if args.command=='train' and args.method not in METHODS:p.error('control is not trainable')
    definition=protocol();args.contract=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
    path=args.output/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=definition:raise RuntimeError('source/protocol is frozen')
    else:atomic_json(path,definition)
    if args.command=='protocol':print(args.contract);return
    torch=configure_runtime()
    torch.set_num_threads(args.threads)
    import fcntl
    lock=args.output/'locks'/f'{args.command}_{args.split}_{args.block}_{args.method}_{args.checkpoint}_{args.kind}_{args.workload}_{args.repeat}.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='train':train(args,torch)
        else:evaluate_job(args,torch)


if __name__=='__main__':main()
