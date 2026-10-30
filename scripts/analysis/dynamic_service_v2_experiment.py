"""Separate second-release development runner; confirmation is disabled."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paths import REPO_ROOT
from scripts.analysis.dynamic_service_experiment import atomic_json,source_contract,configure_runtime
from utils.dynamic_service_scenarios import Scenario,seed_for

TASK='chapter5_dynamic_service_20261028'
REVISION='revision_v2_20261030'


def protocol():
    return dict(revision=REVISION,version=2,source=source_contract(),
        base_physical_contract='a4ffd7991c85f2a2887da47f5fac71b9219517cff496b46822b765f8f08b6639',
        observations='same public current-demand union for every method; previous physical motion feedback',
        actions='latent Normal -> tanh -> state-dependent boundary box -> inherited speed/collision projection',
        upper_reward='success/100',lower_reward='success/admitted',
        ppo=dict(lr=.0003,clip=.2,value_clip=.2,target_kl=.02,grad_clip=.5,max_epochs=4,
                 lower_minibatch_graphs=8,lower_advantage_scale_floor=.05),
        schedule=dict(warm=50,upper_until=200,alternating_block=5,total=400,horizon=1000),
        seed_usage='existing development seeds and tapes reused intentionally; never confirmation',
        confirmation_enabled=False)


def models():
    from marl_models.dynamic_service_hmarl_v2 import UpperPolicyV2,LowerPolicyV2
    return UpperPolicyV2(),LowerPolicyV2()


def checkpoint(args):
    return args.output/'models'/args.split/str(args.block)/(args.phase+'.pt')


def train(args,torch):
    import numpy as np
    from environment.dynamic_service_v2 import DynamicServiceEnvV2
    from utils.dynamic_service_baselines import HistoryController,matching_search
    from marl_models.dynamic_service_hmarl_v2 import update_upper_v2,update_lower_v2
    seed=seed_for('dev/train',args.block);torch.manual_seed(seed)
    upper,lower=models();uopt=torch.optim.Adam(upper.parameters(),lr=3e-4);lopt=torch.optim.Adam(lower.parameters(),lr=3e-4)
    file=checkpoint(args);first=0
    if file.exists():
        saved=torch.load(file,map_location='cpu',weights_only=False)
        if any(saved[k]!=getattr(args,k) for k in ('phase','split','block','contract','steps')):
            raise RuntimeError('checkpoint identity differs from this source/scope')
        if saved['seed']!=seed or saved['episode']>args.episodes:raise RuntimeError('checkpoint seed/budget mismatch')
        if args.split!='smoke' and saved['budget_episodes']!=args.episodes:raise RuntimeError('declared development budget changed')
        upper.load_state_dict(saved['upper']);lower.load_state_dict(saved['lower'])
        uopt.load_state_dict(saved['uopt']);lopt.load_state_dict(saved['lopt']);torch.set_rng_state(saved['rng']);first=saved['episode']
        if first:
            last=args.output/'training'/args.split/str(args.block)/args.phase/f'{first-1:04}.json'
            if last.exists():
                if json.loads(last.read_text())!=saved['last_record']:raise RuntimeError('committed record differs from checkpoint')
            else:atomic_json(last,saved['last_record'])
    for episode in range(first,args.episodes):
        selection=np.random.RandomState(seed_for(f'dev/cell/{args.block}',episode))
        kind='ABCD'[selection.randint(4)];scale=(.6,1.,1.4)[selection.randint(3)]
        env=DynamicServiceEnvV2(Scenario(seed_for(f'dev/workload/{args.block}',episode),kind),scale)
        warm=episode<50
        if args.phase=='upper_probe':upper_active,lower_active=True,False
        elif args.phase=='lower_probe':upper_active,lower_active=False,True
        else:
            lower_active=warm or (episode>=200 and ((episode-200)//5)%2==1)
            upper_active=not warm and not lower_active
        hidden=upper.initial_state();controller=HistoryController();ur=[];lr=[]
        total=dict(generated=0,admitted=0,success=0,boundary_uav_steps=0,hovering_uav_steps=0)
        started=time.perf_counter()
        for _ in range(args.steps):
            obs=env.observe();batch=env.batch();controller.update(obs,env.positions)
            before=hidden.numpy().copy()
            use_history=args.phase=='lower_probe' or (args.phase=='full' and warm)
            a=upper.act(obs,hidden,stochastic=upper_active);hidden=a['hidden']
            actions=controller.actions(env.positions) if use_history else a['actions']
            if args.phase=='upper_probe':b=None;allocation=matching_search(batch)
            else:b=lower.act(batch,stochastic=lower_active);allocation=b['assignment']
            row=env.step(actions,allocation)
            if upper_active:ur.append(dict(observation=obs,hidden_before=before,raw=a['raw'],logprob=a['logprob'],value=a['value'],reward=row['reward']))
            if lower_active:lr.append(dict(batch=batch,**b,reward=row['success']/max(len(batch),1)))
            for key in total:total[key]+=row[key]
        uloss=update_upper_v2(upper,uopt,ur) if upper_active else None
        lloss=update_lower_v2(lower,lopt,lr) if lower_active else None
        record=dict(**total,episode=episode+1,kind=kind,scale=scale,seconds=time.perf_counter()-started,
                    upper=uloss,lower=lloss,contract=args.contract,phase=args.phase,split=args.split,block=args.block,
                    workload_fingerprint=env.tape.fingerprint(),development_only=True)
        file.parent.mkdir(parents=True,exist_ok=True)
        state=dict(upper=upper.state_dict(),lower=lower.state_dict(),uopt=uopt.state_dict(),lopt=lopt.state_dict(),
                   rng=torch.get_rng_state(),episode=episode+1,seed=seed,phase=args.phase,split=args.split,block=args.block,
                   contract=args.contract,steps=args.steps,budget_episodes=args.episodes,last_record=record)
        temp=file.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,file)
        atomic_json(args.output/'training'/args.split/str(args.block)/args.phase/f'{episode:04}.json',record)
        print(json.dumps(record),flush=True)


def evaluate_job(args,torch):
    import numpy as np
    from environment.dynamic_service_v2 import DynamicServiceEnvV2
    from utils.dynamic_service_baselines import HistoryController,matching_search
    upper,lower=models();file=checkpoint(args)
    learned=args.control not in ('history','hover')
    if learned:
        saved=torch.load(file,map_location='cpu',weights_only=False)
        if any(saved[k]!=getattr(args,k) for k in ('phase','split','block','contract','steps')) or saved['episode']!=args.episodes:
            raise RuntimeError('evaluation requires the final checkpoint for the exact declared budget')
        upper.load_state_dict(saved['upper']);lower.load_state_dict(saved['lower'])
    upper.eval();lower.eval()
    destination=args.output/'evaluation'/args.split/str(args.block)/args.phase/f'{args.control}_{args.kind}_{args.scale}_{args.workload}_{args.repeat}.json'
    if destination.exists():
        d=json.loads(destination.read_text())
        if (d['contract']!=args.contract or d['budget_episodes']!=args.episodes or len(d['rows'])!=args.steps
                or (learned and d['checkpoint_sha256']!=hashlib.sha256(file.read_bytes()).hexdigest())):
            raise RuntimeError('existing result belongs to another source/budget/checkpoint')
        print('REUSED',destination);return
    env=DynamicServiceEnvV2(Scenario(seed_for(f'dev/evaluation/{args.workload}',args.repeat),args.kind),args.scale)
    hidden=upper.initial_state();controller=HistoryController();rows=[]
    keys=('generated','admitted','success','uncovered','deadline_misses','decision_timeout','mbs','boundary_uav_steps','hovering_uav_steps')
    total=dict.fromkeys(keys,0)
    for t in range(args.steps):
        started=time.perf_counter();obs=env.observe();batch=env.batch();controller.update(obs,env.positions)
        if args.control=='hover':actions=np.zeros((5,2))
        elif args.control=='history' or args.phase=='lower_probe':actions=controller.actions(env.positions)
        else:a=upper.act(obs,hidden);actions=a['actions'];hidden=a['hidden']
        allocation=matching_search(batch) if args.control in ('history','hover','matching') or args.phase=='upper_probe' else lower.act(batch)['assignment']
        elapsed=time.perf_counter()-started
        row=env.step(actions,allocation,decision_timeout=elapsed>1)
        if elapsed>1:hidden=upper.initial_state()
        rows.append(dict(t=t,seconds=elapsed,edge60_uav_steps=int(((env.positions<110)|(env.positions>590)).any(axis=1).sum()),**row))
        for key in keys:total[key]+=row[key]
    result=dict(contract=args.contract,phase=args.phase,split=args.split,block=args.block,control=args.control,
        kind=args.kind,scale=args.scale,workload=args.workload,repeat=args.repeat,budget_episodes=args.episodes,
        rows=rows,totals=total,dsr=total['success']/max(1,total['generated']),admission=total['admitted']/max(1,total['generated']),
        request_fingerprint=env.tape.fingerprint(),development_only=True,
        checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest() if learned else None)
    atomic_json(destination,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('protocol','train','evaluate'))
    parser.add_argument('--output',type=Path,default=REPO_ROOT/'results'/TASK/REVISION)
    parser.add_argument('--split',choices=('smoke','pilot','dev'),default='pilot')
    parser.add_argument('--phase',choices=('upper_probe','lower_probe','full'),default='full')
    parser.add_argument('--block',type=int,choices=range(5),default=0)
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--kind',choices=tuple('ABCD'),default='D')
    parser.add_argument('--scale',type=float,choices=(.6,1.,1.4),default=1.)
    parser.add_argument('--workload',type=int,choices=range(5),default=0)
    parser.add_argument('--repeat',type=int,choices=range(6),default=0)
    parser.add_argument('--control',choices=('learned','matching','history','hover'),default='learned')
    args=parser.parse_args()
    if args.episodes<1 or not 1<=args.steps<=1000:parser.error('positive episode budget and steps <=1000 required')
    if args.split=='dev' and (args.phase!='full' or args.episodes!=400 or args.steps!=1000):parser.error('dev requires full 400x1000; reduced runs are pilot/smoke only')
    if args.split=='pilot' and args.steps!=1000:parser.error('pilot uses full 1000-step episodes; short traces are smoke only')
    definition=protocol();args.contract=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
    p=args.output/'protocol.json'
    if p.exists():
        if json.loads(p.read_text())!=json.loads(json.dumps(definition)):raise RuntimeError('output frozen to a different source')
    else:atomic_json(p,definition)
    if args.command=='protocol':print(args.contract);return
    torch=configure_runtime()
    import fcntl
    lock=args.output/'locks'/f'{args.command}_{args.split}_{args.phase}_{args.block}_{args.control}_{args.workload}_{args.repeat}_{args.kind}_{args.scale}.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='train':train(args,torch)
        else:evaluate_job(args,torch)


if __name__=='__main__':main()
