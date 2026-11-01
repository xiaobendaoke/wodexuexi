"""New repair contract; never resumes or rewrites the old E2E experiment."""
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
from scripts.analysis.dynamic_service_joint_experiment import save_checkpoint,digest
from utils.dynamic_service_scenarios import Scenario,seed_for

TASK='chapter5_dynamic_service_20261028'
REVISION='revision_feasible_credit_20261030'
METHODS=('feasible_trajectory','local_allocation','repair_joint','repair_full_credit','repair_no_plan')
CONTROLS=('history','hover','mpc','library_greedy')
LEGACY='legacy_sampled'


def protocol():
    return dict(revision=REVISION,source=source_contract(),development_only=True,confirmation=False,
        physics='same serve-before-move, no queues/energy, 15m speed, 200m separation, 100m admission; original evaluator/projector unchanged',
        motion='categorical choice of 257 goal-independent joint direction/speed templates; analytic feasibility mask with 2mm observation-rounding margin',
        action_space_boundary='new discrete motion parameterization, not a continuous Gaussian/H-PPO claim; allowed physical region is unchanged',
        candidate_features='43 derived features from public geometry, current visible demand/count/cycle-deadline pressure; no future, hidden demand or teacher action',
        allocation='unchanged autoregressive graph/risk head; only physical reachability mask; no matching repair in learned allocator',
        sampling='same reproducible categorical sampling in training/main evaluation; separate slot-derived allocation/motion generators; greedy diagnostic separate',
        credit='current S/100 for routing; future physical-time lambda returns for motion; shared actor optimizer; explicitly not exact original joint-policy gradient',
        critic='identical current-prefix, total-prefix, future-motion heads in all new methods; both current/total heads trained, actor credit switches only',
        ppo=dict(actor_lr=1e-4,critic_lr=3e-4,epochs=4,sequence=64,burn_in=16,gamma=1,lam=.99,clip=.2,
            entropy=.001,grad_clip=.5,target_conditional_kl=.05,backtracking_attempts=6,
            backtracking='damped Adam parameter proposal, same-minibatch conditional KL; not full-trajectory trust-region guarantee'),
        methods=list(METHODS),controls=list(CONTROLS),legacy=LEGACY,
        budget=dict(train_cell=['D',1.],blocks=3,episodes=40,steps=1000,checkpoints=[0,20,40],
            evaluation_workloads=3,repeats=2,final_extra_cell=['A',1.]),
        seed_usage='cleared existing development train seeds/tapes intentionally reused; never independent confirmation',
        initial_seed=[seed_for('dev/train',b) for b in range(3)],
        legacy_contract='3911ce4e7fdf115aafff91d8eab33fcae7c5abcaacd6645c7cc5ef02df57c829',
        legacy_usage='old final/20/initial joint checkpoints re-evaluated with sampling, not re-trained or substituted for old scores',
        runtime='single-thread workers, performance governor retained; exclude physical CPUs 0..7; checkpoint-progress watchdog and one bounded retry',
        scope='40-episode repair development only; no auto 400-episode or confirmation expansion')


def checkpoint(args,episode=None):
    return args.output/'models'/args.split/str(args.block)/args.method/('latest.pt' if episode is None else f'{episode:04}.pt')


def identity(args):
    return dict(contract=args.contract,method=args.method,block=args.block,split=args.split,steps=args.steps,budget=args.episodes,
        seed=seed_for('dev/train',args.block),decoder='greedy' if args.greedy else 'sampled')


def check(saved,args):
    for key in ('contract','method','block','split','steps','budget','seed'):
        if saved.get(key)!=identity(args)[key]:raise RuntimeError(f'checkpoint identity mismatch: {key}')


def heartbeat(args,data):
    if args.heartbeat:atomic_json(args.heartbeat,dict(updated=time.time(),pid=os.getpid(),**data))


def episode(args,actor,critic,torch,number=None):
    import numpy as np
    from environment.dynamic_service_v2 import DynamicServiceEnvV2
    from utils.dynamic_service_baselines import HistoryController,matching_search
    from utils.dynamic_service_feasible_motion import motion_library
    training=number is not None
    tape_seed=seed_for(f'dev/workload/{args.block}',number) if training else seed_for(f'dev/evaluation/{args.workload}',args.repeat)
    env=DynamicServiceEnvV2(Scenario(tape_seed,'D' if training else args.kind));controller=HistoryController()
    hidden=actor.initial_state() if actor else None;ch=critic.initial_state() if critic else None
    rows=[];rollout=[];totals={k:0 for k in ('generated','admitted','success','uncovered','deadline_misses','mbs','decision_timeout','boundary_uav_steps','hovering_uav_steps')}
    for t in range(args.steps):
        begin=time.perf_counter();obs=env.observe();batch=env.batch();controller.update(obs,env.positions)
        before=env.positions.copy()
        aseed=seed_for(f'repair/actions/{args.block}/{number}' if training else f'repair/evaluation/actions/{args.block}/{args.workload}/{args.repeat}',t)
        if args.method in CONTROLS:
            assignment=matching_search(batch)
            if args.method=='hover':actions=np.zeros((5,2))
            elif args.method=='history':actions=controller.actions(before)
            elif args.method=='mpc':actions=controller.mpc(env,batch)
            else:
                bank=motion_library(obs);score=np.where(bank['mask'],bank['features'][:,40],-np.inf)
                actions=bank['actions'][int(np.argmax(score))]
        elif args.method==LEGACY:
            torch.manual_seed(aseed);sample=actor.act(obs,batch,hidden,stochastic=not args.greedy)
            assignment=sample['assignment'];actions=sample['actions'];hidden=sample['hidden']
            if not torch.is_tensor(hidden):hidden=torch.tensor(hidden)
        else:
            hb=hidden.numpy().copy();cb=ch.numpy().copy() if ch is not None else None
            fixed=matching_search(batch) if args.method=='feasible_trajectory' else None
            sample=actor.act(obs,batch,hidden,seed=aseed,greedy=args.greedy,fixed_assignment=fixed)
            hidden=torch.tensor(sample['hidden']);assignment=sample['assignment'];actions=sample['actions']
            if args.method=='local_allocation':actions=controller.actions(before)
        elapsed=time.perf_counter()-begin
        if training:
            prediction=critic.predict(obs,batch,ch,sample,t);ch=torch.tensor(prediction['critic_hidden'])
        row=env.step(actions,assignment,decision_timeout=elapsed>1)
        if args.method in METHODS and args.method!='local_allocation' and not row['decision_timeout']:
            if not np.allclose(env.positions-before,actions*15,atol=1e-3,rtol=0):
                raise RuntimeError('selected feasible plan suffered unexpected projection loss')
        row.update(t=t,seconds=elapsed,edge60_uav_steps=int(((env.positions<110)|(env.positions>590)).any(1).sum()),
                   positions=env.positions.tolist(),commanded_m=float(np.linalg.norm(actions,axis=1).mean()*15))
        rows.append(row)
        for key in totals:totals[key]+=row[key]
        if training:
            rollout.append(dict(observation=obs,batch=batch,hidden_before=hb,critic_hidden_before=cb,
                **sample,**prediction,reward=row['reward'],time=t))
        if t%100==0:heartbeat(args,dict(stage='collect',episode=number,step=t))
    return dict(totals=totals,dsr=totals['success']/max(totals['generated'],1),admission=totals['admitted']/max(totals['generated'],1),
        conditional=totals['success']/max(totals['admitted'],1),fairness=rows[-1]['fairness'],
        p95_seconds=float(np.percentile([r['seconds'] for r in rows],95)),workload_fingerprint=env.tape.fingerprint(),
        workload_seed=tape_seed,rows=rows),rollout


def train(args,torch):
    from marl_models.dynamic_service_repair import RepairedPolicy,RepairedCritic,update_repaired
    torch.manual_seed(seed_for('dev/train',args.block))
    actor=RepairedPolicy(args.method);critic=RepairedCritic()
    ao=torch.optim.Adam(actor.parameters(),lr=1e-4);co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    latest=checkpoint(args);first=0
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False);check(saved,args)
        actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic']);ao.load_state_dict(saved['aopt']);co.load_state_dict(saved['copt']);torch.set_rng_state(saved['rng'])
        first=saved['episode']
        if first:
            p=args.output/'training'/args.split/str(args.block)/args.method/f'{first:04}.json'
            if p.exists() and json.loads(p.read_text())!=saved['last_record']:raise RuntimeError('record/checkpoint mismatch')
            if not p.exists():atomic_json(p,saved['last_record'])
        if first in (0,20,args.episodes) and not checkpoint(args,first).exists():save_checkpoint(checkpoint(args,first),saved,torch)
    else:
        saved=dict(**identity(args),episode=0,actor=actor.state_dict(),critic=critic.state_dict(),aopt=ao.state_dict(),copt=co.state_dict(),rng=torch.get_rng_state(),last_record=None)
        save_checkpoint(latest,saved,torch);save_checkpoint(checkpoint(args,0),saved,torch)
    stop=min(args.stop_after if args.stop_after is not None else args.episodes,args.episodes)
    for number in range(first,stop):
        start=time.perf_counter();result,rollout=episode(args,actor,critic,torch,number)
        metrics=update_repaired(actor,critic,ao,co,rollout,progress=lambda data:heartbeat(args,dict(episode=number,**data)))
        record=dict(**identity(args),episode=number+1,interaction_steps=(number+1)*args.steps,
            **{k:v for k,v in result.items() if k!='rows'},update=metrics,seconds=time.perf_counter()-start)
        saved=dict(**identity(args),episode=number+1,actor=actor.state_dict(),critic=critic.state_dict(),aopt=ao.state_dict(),copt=co.state_dict(),rng=torch.get_rng_state(),last_record=record)
        save_checkpoint(latest,saved,torch)
        if number+1 in (20,args.episodes):save_checkpoint(checkpoint(args,number+1),saved,torch)
        atomic_json(args.output/'training'/args.split/str(args.block)/args.method/f'{number+1:04}.json',record)
        heartbeat(args,dict(stage='episode_complete',episode=number+1));print(json.dumps(record),flush=True)


def evaluation_path(args):
    folder='timing_audit' if args.timing_audit else ('greedy_diagnostic' if args.greedy else 'evaluation')
    return args.output/folder/args.split/str(args.block)/args.method/str(args.checkpoint)/f'{args.kind}_{args.workload}_{args.repeat}.json'


def evaluate(args,torch):
    actor=None;model=None
    if args.method in METHODS:
        from marl_models.dynamic_service_repair import RepairedPolicy
        model=checkpoint(args,args.checkpoint);saved=torch.load(model,map_location='cpu',weights_only=False);check(saved,args)
        if saved['episode']!=args.checkpoint:raise RuntimeError('wrong checkpoint')
        actor=RepairedPolicy(args.method);actor.load_state_dict(saved['actor']);actor.eval()
    elif args.method==LEGACY:
        from marl_models.dynamic_service_joint import JointPolicy
        if args.legacy_root is None:raise RuntimeError('legacy_root required')
        old=json.loads((args.legacy_root/'protocol.json').read_text())
        for relative in ('marl_models/dynamic_service_joint.py','marl_models/dynamic_service_hmarl_v2.py','environment/dynamic_service.py','environment/dynamic_service_v2.py','config.py','utils/dynamic_service_scenarios.py'):
            if digest(REPO_ROOT/relative)!=old['source'][relative]:raise RuntimeError('legacy base implementation changed')
        model=args.legacy_root/'models/pilot'/str(args.block)/'joint'/f'{args.checkpoint:04}.pt'
        saved=torch.load(model,map_location='cpu',weights_only=False)
        if saved['contract']!=protocol()['legacy_contract'] or saved['episode']!=args.checkpoint or saved['seed']!=seed_for('dev/train',args.block):raise RuntimeError('legacy model mismatch')
        actor=JointPolicy('joint');actor.load_state_dict(saved['actor']);actor.eval()
    path=evaluation_path(args);model_sha=digest(model) if model else None
    if path.exists():
        existing=json.loads(path.read_text())
        if existing['contract']!=args.contract or existing['checkpoint_sha256']!=model_sha:raise RuntimeError('evaluation identity changed')
        return
    if actor:
        from environment.dynamic_service_v2 import DynamicServiceEnvV2
        env=DynamicServiceEnvV2(Scenario(seed_for('dev/workload/0',0),'D'))
        for i in range(3):
            if args.method==LEGACY:actor.act(env.observe(),env.batch(),actor.initial_state())
            else:actor.act(env.observe(),env.batch(),actor.initial_state(),seed=i)
    result,_=episode(args,actor,None,torch)
    result.update(**identity(args),checkpoint=args.checkpoint,checkpoint_sha256=model_sha,kind=args.kind,workload=args.workload,repeat=args.repeat,
        timing_audit=args.timing_audit,cpu_affinity=sorted(os.sched_getaffinity(0)),legacy_reused=args.method==LEGACY,development_only=True)
    atomic_json(path,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('protocol','train','evaluate'))
    p.add_argument('--output',type=Path,default=REPO_ROOT/'results'/TASK/REVISION)
    p.add_argument('--split',choices=('pilot','smoke'),default='pilot');p.add_argument('--method',choices=METHODS+CONTROLS+(LEGACY,),default='repair_joint')
    p.add_argument('--block',type=int,choices=range(3),default=0);p.add_argument('--episodes',type=int,default=40);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--checkpoint',type=int,default=40);p.add_argument('--stop-after',type=int);p.add_argument('--kind',choices=('D','A'),default='D')
    p.add_argument('--workload',type=int,choices=range(3),default=0);p.add_argument('--repeat',type=int,choices=range(2),default=0)
    p.add_argument('--legacy-root',type=Path);p.add_argument('--heartbeat',type=Path);p.add_argument('--greedy',action='store_true');p.add_argument('--timing-audit',action='store_true')
    args=p.parse_args()
    if args.split=='pilot' and (args.episodes!=40 or args.steps!=1000):p.error('pilot is 40x1000')
    if args.episodes<1 or not 1<=args.steps<=1000:p.error('invalid budget')
    if args.command=='train' and (args.method not in METHODS or args.greedy):p.error('only sampled new methods train')
    definition=protocol();args.contract=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
    path=args.output/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=definition:raise RuntimeError('frozen output source mismatch')
    else:atomic_json(path,definition)
    if args.command=='protocol':print(args.contract);return
    torch=configure_runtime()
    import fcntl
    lock=args.output/'locks'/f'{args.command}_{args.split}_{args.method}_{args.block}_{args.checkpoint}_{args.kind}_{args.workload}_{args.repeat}_{args.greedy}_{args.timing_audit}.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='train':train(args,torch)
        else:evaluate(args,torch)


if __name__=='__main__':main()
