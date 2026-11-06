"""Frozen-allocation step-control repair and independent-future ranking test."""
import argparse
import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_long_value_unfreeze as legacy
from scripts.analysis import queued_long_value_experiment as study
from utils.queued_long_value import LongAdapter,remaining_future
from utils.queued_service_business_refinement import RecoveryScenario
from environment.queued_service import Engine,Protocol

base=study.base
REV='queued_motion_trust_20261102'
ARMS=('legacy','trust')
MODES=('initial','legacy','trust')
TEST=tuple(range(9200,9232))
RANK=tuple(range(9100,9104))
TIMES=(25,75,125,175)
FUTURES=tuple(range(100,108))
SMOKE=False


def out(task):return Path(task)/'results'/base.TASK/REV/('smoke' if SMOKE else '')


def checkpoint(task,b,arm,v):return out(task)/'models'/arm/str(b)/f'{v}.pt'


@lru_cache(maxsize=12)
def model_at(task,b,arm,v):
    import torch
    m=study.factory();m.load_state_dict(torch.load(checkpoint(task,b,arm,v),map_location='cpu',weights_only=False)['model']);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m


def configure():
    legacy.REV=REV;legacy.out=out;legacy.checkpoint=checkpoint;legacy.model_at=model_at


def run_case(job):configure();return legacy.run_case(job)


def initialize(task):
    import torch
    for b in range(3):
        saved=torch.load(study.checkpoint(task,b,2),map_location='cpu',weights_only=False)
        for arm in ARMS:
            p=checkpoint(task,b,arm,0);p.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(model=saved['model']),p)


def fit(job):
    task,b,arm,v,ws=job
    import torch
    from marl_models.queued_long_value import update_branches
    from marl_models.queued_motion_trust import update_motion,MotionBatch
    from utils.queued_service_diagnostics import mc_returns
    m=copy.deepcopy(model_at(task,b,arm,v-1));rows=[]
    for w in ws:
        name=f'{arm}_{v-1}_{b}_{w}_train';p=out(task)/'experience'/f'{name}.pkl.gz'
        ep=json.loads((out(task)/'episodes'/f'{name}.json').read_text());assert base.sha(p)==ep['experience_sha']
        selected=base.read(p);targets=mc_returns([r['reward'] for r in selected]);assert all(abs(r['returns']-x)<1e-9 for r,x in zip(selected,targets));rows+=selected
    saved=torch.load(checkpoint(task,b,arm,v-1),map_location='cpu',weights_only=False)
    torch.manual_seed(base.seed(REV,'fit',b,v))
    if arm=='trust':d=update_motion(m,rows,optimizer_states=saved.get('optimizer_states'))
    else:d=update_branches(m,rows,'motion',optimizer_states=saved.get('optimizer_states'))
    batch=MotionBatch(m,rows);p=batch.probabilities();d['measured_full_kl']=MotionBatch.kl(batch.old,p)
    d['measured_probability_l1']=float((batch.old-p).abs().sum(-1).mean())
    states=d.pop('_optimizer_states');dest=checkpoint(task,b,arm,v);dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=m.state_dict(),optimizer_states=states),dest)
    base.write(out(task)/'updates'/f'{arm}_{v}_{b}.json',dict(block=b,arm=arm,version=v,workloads=list(ws),rows=len(rows),episodes=len(ws),metrics=d,checkpoint_sha=base.sha(dest),cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def mode_model(task,b,mode):return model_at(task,b,'trust' if mode=='initial' else mode,0 if mode=='initial' else 2)


def collect(job):
    task,b,w=job;m=mode_model(task,b,'initial');tape=RecoveryScenario(w,'burst','normal',.6)
    e=Engine(Protocol(),base.START,tape.ue_positions);adapter=LongAdapter(True)
    import torch
    from marl_models.queued_service_joint import tensor
    while e.t<200 or e.tasks:
        t=int(e.t);e.add(tape.slot(t));e.admit();obs=e.observe();_,plan,move=m.act(obs,adapter,base.seed(REV,'rank_collect',b,w,t))
        if t in TIMES:
            full=dict(time=e.t,positions=e.positions,ues=e.ues,tasks=[asdict(x) for x in sorted(e.tasks.values(),key=lambda x:(x.deadline,x.uid))],protocol=e.p)
            compact=Engine.from_observation(full);ca=copy.deepcopy(adapter);ca.history.seen.intersection_update(compact.tasks)
            pre,post=study.banks(adapter,obs,plan);probs={}
            for mode in MODES:
                mm=mode_model(task,b,mode)
                with torch.no_grad():dist=mm.motion(tensor([pre['state']]),tensor([pre['features']]),tensor([pre['mask']],torch.bool),tensor([post['state']]),tensor([post['features']]),True)
                probs[mode]=dist.probs[0,:len(pre['actions'])].tolist()
            snap=dict(engine=compact,adapter=ca,plan=plan,pre=pre,post=post,probabilities=probs,block=b,workload=w,time=t)
            base.dump(out(task)/'states'/f'{b}_{w}_{t}.pkl.gz',snap)
        e.assign(plan);e.advance(move)
    e.audit();base.write(out(task)/'collection'/f'{b}_{w}.json',dict(block=b,workload=w,counts=dict(e.counts),fingerprint=tape.fingerprint,cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


@lru_cache(maxsize=8)
def state_at(task,b,w,t):return base.read(out(task)/'states'/f'{b}_{w}_{t}.pkl.gz')


def label(job):
    task,b,w,t,r,mode,chunk=job;snap=state_at(task,b,w,t);m=mode_model(task,b,mode)
    future=remaining_future(w,snap['engine'].ues,t,r);fingerprint=hashlib.sha256(pickle.dumps(future,5)).hexdigest()
    indexes=list(range(chunk*4,min(chunk*4+4,len(snap['pre']['actions']))));values=[];counts=[]
    for i in indexes:
        e=copy.deepcopy(snap['engine']);adapter=copy.deepcopy(snap['adapter']);initial=len(e.tasks)
        e.assign(snap['plan']);e.advance(snap['pre']['actions'][i])
        while e.t<200 or e.tasks:
            now=int(e.t);assert now<=204;e.add(copy.deepcopy(future.get(now,())));e.admit()
            _,plan,motion=m.act(e.observe(),adapter,base.seed(REV,'rank_tail',b,w,t,r,now));e.assign(plan);e.advance(motion)
        e.audit();c=dict(e.counts);total=initial+sum(len(xs) for at,xs in future.items() if t<at<200)
        assert total==c['generated']==c['success']+c['expired'];counts.append(c);values.append(c['success'])
    base.write(out(task)/'labels'/f'{b}_{w}_{t}_{r}_{mode}_{chunk}.json',dict(block=b,workload=w,time=t,replica=r,mode=mode,chunk=chunk,indexes=indexes,values=values,counts=counts,future_sha=fingerprint,cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def main():
    global SMOKE
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();task=a.task;SMOKE=a.smoke
    import torch,fcntl
    torch.set_num_threads(1);out(task).mkdir(parents=True,exist_ok=True)
    lock=open(out(task)/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    initialize(task)
    if SMOKE:
        run_case((str(task),0,'trust',0,8999,'train',True));fit((str(task),0,'trust',1,[8999]));print('SMOKE PASS',flush=True);return
    files={p.relative_to(REPO_ROOT).as_posix():base.sha(p) for d in ('environment','marl_models','utils','scripts/analysis','tests') for p in (REPO_ROOT/d).glob('*.py')}
    parents={str(b):base.sha(study.checkpoint(task,b,2)) for b in range(3)};cpus=sorted(os.sched_getaffinity(0))
    base.write(out(task)/'protocol.json',dict(revision=REV,source_sha=files,parent_sha=parents,train=list(range(9000,9096)),test=TEST,rank=RANK,times=TIMES,futures=FUTURES,
        arms=ARMS,rounds=2,episodes_per_round=48,allocation='frozen',full_kl_limit=.02,step_kl_limit=.0002,max_backtracks=12,base_lr=.0003,actor_epochs=2,critic_epochs=2,
        ranking_gate='Each trust lineage bootstrap lower bound of expected gain over uniform >0, split-half correlation >0 and crossfit selector gain >0; developmental only',
        dsr_gate='Each trust lineage >initial and >legacy, >16/32 positive workloads and all leave-one-out gains positive',cpu=cpus,independent_confirmation=False))
    original=base.out;base.out=out
    try:
        for v in (1,2):
            ws=list(range(9000+(v-1)*48,9000+v*48))
            base.parallel(task,f'train_{v}',run_case,[(str(task),b,arm,v-1,w,'train',True) for b in range(3) for arm in ARMS for w in ws],cpus)
            base.parallel(task,f'update_{v}',fit,[(str(task),b,arm,v,ws) for b in range(3) for arm in ARMS],cpus)
        base.parallel(task,'evaluation',run_case,[(str(task),b,'trust' if mode=='initial' else mode,0 if mode=='initial' else 2,w,mode,False) for b in range(3) for w in TEST for mode in MODES],cpus)
        base.parallel(task,'rank_collection',collect,[(str(task),b,w) for b in range(3) for w in RANK],cpus)
        jobs=[]
        for t in TIMES:
            for b in range(3):
                for w in RANK:
                    k=len(state_at(str(task),b,w,t)['pre']['actions'])
                    jobs.extend((str(task),b,w,t,r,mode,c) for mode in MODES for r in FUTURES for c in range(math.ceil(k/4)))
        base.parallel(task,'independent_futures',label,jobs,cpus)
    finally:base.out=original
    from scripts.analysis.analyze_queued_motion_trust import analyze
    analyze(task)
    assert all(base.sha(REPO_ROOT/name)==digest for name,digest in files.items())
    assert all(base.sha(study.checkpoint(task,b,2))==digest for b,digest in ((int(k),h) for k,h in parents.items()))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
