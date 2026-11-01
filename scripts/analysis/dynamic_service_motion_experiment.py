"""Six-episode isolated motion trial; all numerical execution is server-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis.dynamic_service_experiment import source_contract,atomic_json,configure_runtime
from scripts.analysis.dynamic_service_joint_experiment import save_checkpoint
from utils.dynamic_service_scenarios import Scenario,seed_for

TASK='chapter5_dynamic_service_20261028';REVISION='revision_motion_short_20261031'
METHODS=('motion_ppo','motion_aux');CONTROLS=('uniform','hover','history','library_greedy','service_greedy')
def protocol():
    return dict(revision=REVISION,source=source_contract(),development_only=True,physics='unchanged serve-before-move, no energy/queue, fixed matching lower',
        motion='same feasible 257 plans; normalized legal-candidate features, nonlinear history/candidate fusion, no 0.01 score attenuation',
        credit='reward t+1 credits motion t, final movement excluded; GAE gamma=.99 lambda=.95, scalar unscaled value; finite complete episodes',
        objective_boundary='discounted short-horizon training approximates the undiscounted evaluation DSR; no equivalence claim',
        auxiliary='motion_aux only: weight .5 KL to softmax(current public matched-success counts /1 task) over <=11 plans every 32 slots',
        supervision_boundary='training-only model-based auxiliary, not pure PPO; reconstructed current public float32 data only, no future/tape/teacher at learner inference',
        controls=list(CONTROLS),methods=list(METHODS),training=dict(seeds=3,episodes=6,steps=1000,checkpoints=[0,3,6],cell=['D',1.]),
        optimizer=dict(actor_lr=3e-4,critic_lr=3e-4,epochs=2,sequence=64,burn_in=16,clip=.2,entropy=.001,grad_clip=.5,
            max_chunk_kl=.03,backtracking=6,skip='only excessive chunks skipped, no global actor freeze; end-rollout KL logged'),
        evaluation=dict(workloads=[0,1],repeat=0,decoder='sampled primary; final greedy diagnostic separate',reuse='existing development data, no independent confirmation'),
        gate='promising only if all 3 seed mean DSR gains over initialization >0, mean gain >=1pp, mean above uniform and hover; not a significance test',
        scope='fixed matching lower only, no joint retraining or automatic expansion')
def contract():return hashlib.sha256(json.dumps(protocol(),sort_keys=True).encode()).hexdigest()
def model_path(args,cp=None):return args.output/'models'/str(args.block)/args.method/('latest.pt' if cp is None else f'{cp:04}.pt')
def result_path(args):return args.output/('greedy_diagnostic' if args.greedy else 'evaluation')/str(args.block)/args.method/str(args.checkpoint)/f'D_{args.workload}.json'
def identity(args):return dict(contract=args.contract,method=args.method,block=args.block,episodes=args.episodes,steps=args.steps,seed=seed_for('dev/train',args.block))
def check(saved,args):
    for k,v in identity(args).items():
        if saved.get(k)!=v:raise RuntimeError(f'identity mismatch {k}')
def heartbeat(args,data):
    if args.heartbeat:atomic_json(args.heartbeat,dict(time=time.time(),pid=os.getpid(),**data))
def episode(args,actor,critic,torch,number=None):
    import numpy as np
    from environment.dynamic_service_v2 import DynamicServiceEnvV2
    from utils.dynamic_service_baselines import matching_search,HistoryController
    from utils.dynamic_service_feasible_motion import motion_library
    from utils.dynamic_service_motion_credit import public_service_scores,label_indices
    training=number is not None
    tape_seed=seed_for(f'dev/workload/{args.block}',number) if training else seed_for(f'dev/evaluation/{args.workload}',0)
    env=DynamicServiceEnvV2(Scenario(tape_seed,'D'));history=HistoryController()
    ah=actor.initial_state() if actor else None;ch=critic.initial_state() if critic else None
    rows=[];rollout=[];label_calls=0;label_seconds=0.
    totals={k:0 for k in ('generated','admitted','success','uncovered','deadline_misses','mbs','decision_timeout','boundary_uav_steps','hovering_uav_steps')}
    for t in range(args.steps):
        begin=time.perf_counter();obs=env.observe();batch=env.batch();before=env.positions.copy();history.update(obs,before)
        seed=seed_for(f'motion/train/actions/{args.block}/{number}' if training else f'motion/eval/actions/{args.block}/{args.workload}',t)
        assignment=matching_search(batch);bank=None
        if actor:
            hb=ah.numpy().copy();sample=actor.act(obs,ah,seed,greedy=args.greedy);ah=torch.tensor(sample['hidden']);actions=sample['actions'];bank=sample['bank']
        elif args.method=='history':actions=history.actions(before)
        elif args.method=='hover':actions=np.zeros((5,2))
        else:
            bank=motion_library(obs);legal=np.flatnonzero(bank['mask']);g=torch.Generator().manual_seed(seed)
            if args.method=='uniform':index=int(torch.multinomial(torch.tensor(bank['mask'],dtype=torch.float32),1,generator=g))
            elif args.method=='library_greedy':index=int(np.argmax(np.where(bank['mask'],bank['features'][:,40],-np.inf)))
            else:
                selected=int(torch.multinomial(torch.tensor(bank['mask'],dtype=torch.float32),1,generator=g))
                indices=label_indices(bank,selected,seed);scores=public_service_scores(obs,bank,indices);label_calls+=len(indices)
                index=int(indices[np.argmax(scores)])
            actions=bank['actions'][index]
        decision_seconds=time.perf_counter()-begin
        if training:
            cb=ch.numpy().copy();value,newch=critic.predict(obs,ch,t);ch=torch.tensor(newch)
            ix=scores=None
            if args.method=='motion_aux' and t%32==0 and t<args.steps-1:
                begin_label=time.perf_counter();ix=label_indices(bank,sample['index'],seed);scores=public_service_scores(obs,bank,ix)
                label_seconds+=time.perf_counter()-begin_label;label_calls+=len(ix)
        row=env.step(actions,assignment,decision_timeout=decision_seconds>1)
        if bank is not None and not row['decision_timeout']:
            np.testing.assert_allclose(env.positions-before,actions*15,atol=1e-3,rtol=0)
        row.update(t=t,seconds=decision_seconds,positions=env.positions.tolist(),commanded_m=float(np.linalg.norm(actions,axis=1).mean()*15))
        if actor:
            p=sample['probs'].astype(float);u=bank['mask'].astype(float)/bank['mask'].sum()
            row.update(entropy_fraction=float(-(p*np.log(np.maximum(p,1e-30))).sum()/np.log(bank['mask'].sum())) if bank['mask'].sum()>1 else 1.,
                tv_uniform=float(np.abs(p-u).sum()/2),expected_public_covered=float(p@bank['features'][:,40]*100))
        rows.append(row)
        for k in totals:totals[k]+=row[k]
        if training:rollout.append(dict(obs=obs,hidden_before=hb,critic_hidden_before=cb,**sample,value=value,reward=row['reward'],time=t,label_indices=ix,label_scores=scores))
        if t%100==0:heartbeat(args,dict(stage='collect',episode=number,t=t))
    return dict(totals=totals,dsr=totals['success']/totals['generated'],admission=totals['admitted']/totals['generated'],
        conditional=totals['success']/max(totals['admitted'],1),rows=rows,workload_seed=tape_seed,workload_fingerprint=env.tape.fingerprint(),
        label_calls=label_calls,label_seconds=label_seconds),rollout

def train(args,torch):
    from marl_models.dynamic_service_motion import MotionPolicy,MotionValue,update_motion
    torch.manual_seed(seed_for('dev/train',args.block));actor=MotionPolicy();critic=MotionValue()
    ao=torch.optim.Adam(actor.parameters(),lr=3e-4);co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    latest=model_path(args);first=0
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False);check(saved,args);first=saved['episode']
        actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic']);ao.load_state_dict(saved['aopt']);co.load_state_dict(saved['copt']);torch.set_rng_state(saved['rng'])
        if first:
            p=args.output/'training'/str(args.block)/args.method/f'{first:04}.json'
            if p.exists() and json.loads(p.read_text())!=saved['record']:raise RuntimeError('checkpoint record differs')
            if not p.exists():atomic_json(p,saved['record'])
        if first in (0,3,args.episodes) and not model_path(args,first).exists():save_checkpoint(model_path(args,first),saved,torch)
    else:
        saved=dict(**identity(args),episode=0,actor=actor.state_dict(),critic=critic.state_dict(),aopt=ao.state_dict(),copt=co.state_dict(),rng=torch.get_rng_state(),record=None)
        save_checkpoint(latest,saved,torch);save_checkpoint(model_path(args,0),saved,torch)
    for number in range(first,min(args.stop_after or args.episodes,args.episodes)):
        begin=time.perf_counter();result,rows=episode(args,actor,critic,torch,number)
        update=update_motion(actor,critic,ao,co,rows,aux_weight=.5 if args.method=='motion_aux' else 0.,
            progress=lambda d:heartbeat(args,dict(episode=number,**d)))
        record=dict(**identity(args),episode=number+1,interaction_steps=(number+1)*args.steps,**{k:v for k,v in result.items() if k!='rows'},
            update=update,seconds=time.perf_counter()-begin)
        saved=dict(**identity(args),episode=number+1,actor=actor.state_dict(),critic=critic.state_dict(),aopt=ao.state_dict(),copt=co.state_dict(),rng=torch.get_rng_state(),record=record)
        save_checkpoint(latest,saved,torch)
        if number+1 in (3,args.episodes):save_checkpoint(model_path(args,number+1),saved,torch)
        atomic_json(args.output/'training'/str(args.block)/args.method/f'{number+1:04}.json',record)
        heartbeat(args,dict(stage='episode_complete',episode=number+1));print(json.dumps(record),flush=True)

def evaluate(args,torch):
    actor=None;model=None
    if args.method in METHODS:
        from marl_models.dynamic_service_motion import MotionPolicy
        model=model_path(args,args.checkpoint);s=torch.load(model,map_location='cpu',weights_only=False);check(s,args)
        if s['episode']!=args.checkpoint:raise RuntimeError('wrong checkpoint')
        actor=MotionPolicy();actor.load_state_dict(s['actor']);actor.eval()
        from environment.dynamic_service_v2 import DynamicServiceEnvV2
        obs=DynamicServiceEnvV2(Scenario(seed_for('dev/workload/0',0),'D')).observe()
        for i in range(3):actor.act(obs,actor.initial_state(),i)
    path=result_path(args);model_sha=hashlib.sha256(model.read_bytes()).hexdigest() if model else None
    if path.exists():
        d=json.loads(path.read_text());assert d['contract']==args.contract and d['checkpoint_sha256']==model_sha;return
    d,_=episode(args,actor,None,torch)
    d.update(**identity(args),checkpoint=args.checkpoint,checkpoint_sha256=model_sha,workload=args.workload,
        decoder='greedy_diagnostic' if args.greedy else ('sampled' if actor or args.method=='uniform' else 'control'),cpu_affinity=sorted(os.sched_getaffinity(0)))
    atomic_json(path,d);print(json.dumps({k:v for k,v in d.items() if k!='rows'}),flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('train','evaluate','protocol'))
    p.add_argument('--output',type=Path,default=REPO_ROOT/'results'/TASK/REVISION);p.add_argument('--method',choices=METHODS+CONTROLS,default='motion_aux')
    p.add_argument('--block',type=int,choices=range(3),default=0);p.add_argument('--episodes',type=int,default=6);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--checkpoint',type=int,default=6);p.add_argument('--workload',type=int,choices=range(2),default=0)
    p.add_argument('--stop-after',type=int);p.add_argument('--greedy',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--heartbeat',type=Path)
    args=p.parse_args();args.contract=contract()
    if not args.smoke and (args.episodes!=6 or args.steps!=1000):p.error('short development budget is 6 x 1000')
    if args.steps<2 or args.steps>1000 or args.episodes<1:p.error('invalid finite budget')
    if args.command=='train' and (args.method not in METHODS or args.greedy):p.error('invalid training method')
    path=args.output/'protocol.json';definition=protocol()
    if path.exists():assert json.loads(path.read_text())==definition
    else:atomic_json(path,definition)
    if args.command=='protocol':print(args.contract);return
    import fcntl
    lock=args.output/'locks'/f'{args.command}_{args.method}_{args.block}_{args.checkpoint}_{args.workload}_{args.greedy}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);torch=configure_runtime()
        if args.command=='train':train(args,torch)
        else:evaluate(args,torch)
if __name__=='__main__':main()
