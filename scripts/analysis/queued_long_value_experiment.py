"""Full-remaining-return candidate learning, with a gated branch-unfreezing study."""
import argparse
import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_service_motion_ppo as parent
from scripts.analysis import queued_service_motion_ranking as base
from environment.queued_service import Engine,Protocol
from utils.queued_service_business_refinement import RecoveryScenario
from utils.queued_long_value import LongAdapter,remaining_future
from utils.queued_joint_opportunity import expanded_bank
from utils.queued_service_learning import VirtualQueue
from utils.queued_service_diagnostics import mc_returns

REV='queued_long_value_20261102'
TIMES=(25,75,125,175)
ITERATIONS=2
REPLICAS=2
TEST=tuple(range(8100,8124))
SMOKE=False


def out(task):return Path(task)/'results'/base.TASK/REV/('smoke' if SMOKE else '')


def checkpoint(task,block,version,bank='expanded'):
    return out(task)/'models'/bank/str(block)/f'{version}.pt'


def factory():
    from marl_models.queued_service_plan_residual import ResidualPolicy
    from marl_models.queued_long_value import wrap
    return wrap(ResidualPolicy('joint'))


def initialize(task):
    import torch
    from marl_models.queued_long_value import wrap
    for b in range(3):
        torch.manual_seed(base.seed(REV,'initialization',b))
        original=parent.model_at(task,b,3)
        for bank in ('old','expanded'):
            model=wrap(copy.deepcopy(original));p=checkpoint(task,b,0,bank);p.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(model=model.state_dict(),block=b,version=0,bank=bank),p)
            # The new zero head is deterministic; preserve exact same initialization.
            if bank=='old':saved=model.state_dict()
            else:model.load_state_dict(saved);torch.save(dict(model=model.state_dict(),block=b,version=0,bank=bank),p)


@lru_cache(maxsize=12)
def model_at(task,block,version,bank='expanded'):
    import torch
    saved=torch.load(checkpoint(task,block,version,bank),map_location='cpu',weights_only=False)
    model=factory();model.load_state_dict(saved['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def banks(adapter,obs,plan):
    prefix=VirtualQueue(obs)
    for uid,target in plan.items():prefix.assign(uid,target)
    pre=adapter.motion(obs);post=adapter.motion(prefix.obs)
    assert pre['actions']==post['actions']
    return pre,post


def workloads(iteration,split):
    start=8200+iteration*100+(0 if split=='train' else 40)
    return range(start,start+(8 if split=='train' else 4))


def collect(payload):
    task,b,iteration,w,split=payload;model=model_at(task,b,iteration-1)
    tape=RecoveryScenario(w,'burst','normal',.6);e=Engine(Protocol(),base.START,tape.ue_positions);adapter=LongAdapter(True)
    snapshots=[]
    while e.t<200 or e.tasks:
        t=int(e.t);e.add(tape.slot(t));e.admit();obs=e.observe()
        row,plan,move=model.act(obs,adapter,base.seed(REV,'collect',b,w,t))
        if t in TIMES:
            full=dict(time=e.t,positions=e.positions,ues=e.ues,tasks=[asdict(x) for x in sorted(e.tasks.values(),key=lambda x:(x.deadline,x.uid))],protocol=e.p)
            compact=Engine.from_observation(full);ca=copy.deepcopy(adapter);ca.history.seen.intersection_update(compact.tasks);ca.forecast_seconds=0.
            pre,post=banks(adapter,obs,plan)
            sample=dict(engine=compact,adapter=ca,plan=plan,pre=pre,post=post,old_count=adapter.old_count,
                        block=b,iteration=iteration,workload=w,time=t,split=split)
            p=out(task)/'states'/f'{iteration}_{b}_{w}_{t}.pkl.gz';base.dump(p,sample);snapshots.append(dict(path=p.name,sha=base.sha(p)))
        e.assign(plan);e.advance(move)
    e.audit();c=dict(e.counts);assert c['generated']==c['success']+c['expired']
    base.write(out(task)/'collection'/f'{iteration}_{b}_{w}.json',dict(counts=c,states=snapshots,split=split,cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


@lru_cache(maxsize=6)
def state_at(task,iteration,b,w,t):return base.read(out(task)/'states'/f'{iteration}_{b}_{w}_{t}.pkl.gz')


def long_rollout(snap,future,first,model,replica):
    e=copy.deepcopy(snap['engine']);adapter=copy.deepcopy(snap['adapter']);start=int(e.t)
    initial=len(e.tasks);e.assign(snap['plan']);e.advance(first)
    while e.t<200 or e.tasks:
        t=int(e.t);assert t<=204;e.add(copy.deepcopy(future.get(t,())));e.admit()
        _,plan,motion=model.act(e.observe(),adapter,base.seed(REV,'tail',snap['block'],snap['workload'],start,replica,t))
        e.assign(plan);e.advance(motion)
    e.audit();c=e.counts;total=initial+sum(len(row) for t,row in future.items() if start<t<200)
    assert c['generated']==total==c['success']+c['expired']
    return dict(success=c['success'],expired=c['expired'],generated=total,elapsed=int(e.t)-start)


def label(payload):
    task,iteration,b,w,t,replica,chunk=payload;snap=state_at(task,iteration,b,w,t);model=model_at(task,b,iteration-1)
    future=remaining_future(w,snap['engine'].ues,t,replica);fingerprint=hashlib.sha256(__import__('pickle').dumps(future,5)).hexdigest()
    indexes=list(range(chunk*4,min((chunk+1)*4,len(snap['pre']['actions']))));values=[];counts=[];begin=time.monotonic()
    for i in indexes:
        d=long_rollout(snap,future,snap['pre']['actions'][i],model,replica);values.append(d['success']);counts.append(d)
    assert fingerprint==hashlib.sha256(__import__('pickle').dumps(future,5)).hexdigest()
    record=dict(iteration=iteration,block=b,workload=w,time=t,replica=replica,chunk=chunk,indexes=indexes,values=values,counts=counts,
                future_sha=fingerprint,split=snap['split'],cpu=sorted(os.sched_getaffinity(0)),seconds=time.monotonic()-begin)
    base.write(out(task)/'labels'/f'{iteration}_{b}_{w}_{t}_{replica}_{chunk}.json',record)
    return dict(cpu=record['cpu'])


def dataset(task,b,iteration,split,bank):
    rows=[]
    for w in workloads(iteration,split):
        for t in TIMES:
            snap=state_at(task,iteration,b,w,t);k=len(snap['pre']['actions']);replicas=[]
            for r in range(REPLICAS):
                q=[]
                for chunk in range(math.ceil(k/4)):
                    d=json.loads((out(task)/'labels'/f'{iteration}_{b}_{w}_{t}_{r}_{chunk}.json').read_text());q+=d['values']
                assert len(q)==k;replicas.append(q)
            values=[statistics.mean(xs) for xs in zip(*replicas)];n=snap['old_count'] if bank=='old' else k
            pre=snap['pre'];post=snap['post'];mask=[i<n for i in range(16)]
            rows.append(dict(state=pre['state'],features=pre['features'],post_state=post['state'],post_features=post['features'],mask=mask,
                             values=values[:n],workload=w,time=t))
    return rows


def rank(model,data):
    import torch
    from marl_models.queued_long_value import value_batch
    result=[]
    with torch.no_grad():
        for s in data:
            q=value_batch(model,[s])[0,:len(s['values'])];p=torch.softmax(q/.5,0).tolist();target=s['values']
            result.append(dict(workload=s['workload'],time=s['time'],values=target,probabilities=p,
                gain=sum(a*b for a,b in zip(p,target))-statistics.mean(target),regret=max(target)-sum(a*b for a,b in zip(p,target))))
    return dict(gain=statistics.mean(r['gain'] for r in result),regret=statistics.mean(r['regret'] for r in result),rows=result)


def fit(payload):
    task,b,iteration,bank=payload
    import torch
    from marl_models.queued_long_value import fit_values
    m=copy.deepcopy(model_at(task,b,iteration-1,bank));torch.manual_seed(base.seed(REV,'fit',b,iteration));before={k:v.clone() for k,v in m.allocation.state_dict().items()}
    train=dataset(task,b,iteration,'train',bank);valid=dataset(task,b,iteration,'valid',bank)
    initial=rank(m,valid);metrics=fit_values(m,train);final=rank(m,valid)
    assert all(torch.equal(v,before[k]) for k,v in m.allocation.state_dict().items())
    p=checkpoint(task,b,iteration,bank);p.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=m.state_dict(),block=b,version=iteration,bank=bank),p)
    base.write(out(task)/'fit'/f'{iteration}_{b}_{bank}.json',dict(metrics=metrics,initial=initial,final=final,allocation_frozen=True,
               train_states=len(train),validation_states=len(valid),checkpoint_sha=base.sha(p),cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def evaluate(payload):
    task,b,w,mode=payload
    from scripts.analysis.diagnose_queued_service_conditioning import choose
    import torch
    bank='old' if mode.startswith('old') else 'expanded';version=0 if mode=='initial' else ITERATIONS
    m=model_at(task,b,version,bank);adapter=LongAdapter(bank=='expanded');tape=RecoveryScenario(w,'burst','normal',.6)
    e=Engine(Protocol(),base.START,tape.ue_positions);rows=[];durations=[]
    while e.t<200 or e.tasks:
        t=int(e.t);assert t<=204;e.add(tape.slot(t));e.admit();obs=e.observe();begin=time.monotonic();seed=base.seed(REV,'eval',b,w,t)
        if mode.endswith('model'):row,plan,motion=choose(m,obs,adapter,seed,'model_actual',torch)
        else:
            row,plan,motion=m.act(obs,adapter,seed)
            if mode.endswith('argmax'):motion=adapter.plans[max(range(len(adapter.plans)),key=lambda i:row['motion_probs'][i])]
        durations.append(time.monotonic()-begin);e.assign(plan);rows.append(e.advance(motion))
    e.audit();c=dict(e.counts);assert c['generated']==c['success']+c['expired'] and sum(r['success'] for r in rows)==c['success']
    base.write(out(task)/'evaluation'/f'{b}_{w}_{mode}.json',dict(block=b,workload=w,mode=mode,counts=c,rows=rows,
               fingerprint=tape.fingerprint,summary=e.summary(),mean_decision_seconds=statistics.mean(durations),cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def summarize(task):
    modes=('old_learned','expanded_learned','initial','old_model','expanded_model','expanded_argmax');summary={}
    records=[json.loads(p.read_text()) for p in (out(task)/'evaluation').glob('*.json')];assert len(records)==432
    for mode in modes:
        summary[mode]=[]
        for b in range(3):
            data=[r for r in records if r['mode']==mode and r['block']==b];assert len(data)==24
            c={k:sum(r['counts'][k] for r in data) for k in ('success','generated','admitted')}
            summary[mode].append(dict(block=b,**c,dsr=c['success']/c['generated'],admission=c['admitted']/c['generated'],conditional=c['success']/c['admitted']))
    delta=[summary['expanded_learned'][b]['dsr']-summary['initial'][b]['dsr'] for b in range(3)]
    bankdelta=[summary['expanded_learned'][b]['dsr']-summary['old_learned'][b]['dsr'] for b in range(3)]
    robustness=[]
    for b in range(3):
        def counts(mode,omit=None):return [r['counts'] for r in records if r['mode']==mode and r['block']==b and r['workload']!=omit]
        def dsr(ds):return sum(x['success'] for x in ds)/sum(x['generated'] for x in ds)
        loo=[dsr(counts('expanded_learned',w))-dsr(counts('initial',w)) for w in TEST]
        positive=sum(next(r for r in records if r['block']==b and r['workload']==w and r['mode']=='expanded_learned')['counts']['success']>
                     next(r for r in records if r['block']==b and r['workload']==w and r['mode']=='initial')['counts']['success'] for w in TEST)
        robustness.append(dict(block=b,positive_workloads=positive,min_leave_one_out=min(loo)))
    gate=all(x>0 for x in delta+bankdelta) and all(r['positive_workloads']>12 and r['min_leave_one_out']>0 for r in robustness)
    result=dict(status='COMPLETE_STAGE1',summary=summary,expanded_minus_initial=delta,expanded_minus_old=bankdelta,
                robustness=robustness,foundation_gate=gate,phase2_authorized_but_gated=True,independent_confirmation=False)
    base.write(out(task)/'analysis.json',result)
    lines=['# 扩展候选与完整剩余回报价值学习','', '|控制器|三谱系平均DSR|逐谱系DSR|','|---|---:|---|']
    for mode,data in summary.items():lines.append(f'|{mode}|{statistics.mean(x["dsr"] for x in data)*100:.4f}%|'+ '/'.join(f'{x["dsr"]*100:.4f}%' for x in data)+'|')
    lines+=['', '运动基础开发门槛：'+str(gate),'相对初始扩展评分器百分点差：'+str([100*x for x in delta]),
        '相对原候选价值学习百分点差：'+str([100*x for x in bankdelta]),str(robustness),
        '', '两轮完整剩余回合配对目标；第二轮重新采集更新后策略状态、重新生成未来和标签，不使用六秒排空代理或未经校准的尾部自举。',
        '原/扩展候选学习控制器使用相同状态和同一扩展参考继续策略的标签；old学习是共同参考策略的动作价值控制，不能称其自身精确Q。模型对照用于分离候选容量效应。',
        '119维固定尺度/候选差分特征；零初始化残差精确保留原评分，100轮末模型，不按验证/测试挑选。分配全程冻结。',
        '测试8100–8123从未参与训练或价值拟合；三既有谱系，开发结果非独立确认。所有额外分支/推理成本单列，不称相同总计算预算。',
        '若基础门槛未过，不把其称为可靠基础，阶段二不自动消耗训练预算。']
    dest=Path(task)/'output'/base.TASK/'deliverables'/'queued_long_value.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(lines),encoding='utf-8')
    return result


def main():
    global SMOKE
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();task=a.task;SMOKE=a.smoke
    import torch
    torch.set_num_threads(1);cpus=sorted(os.sched_getaffinity(0));initialize(task)
    if SMOKE:
        collect((str(task),0,1,list(workloads(1,'valid'))[0],'valid'))
        label((str(task),1,0,list(workloads(1,'valid'))[0],TIMES[-1],0,0));print('SMOKE PASS',flush=True);return
    files={str(p.relative_to(REPO_ROOT)):base.sha(p) for d in ('environment','marl_models','utils','scripts/analysis','tests') for p in (REPO_ROOT/d).glob('*.py')}
    parents={str(b):base.sha(parent.checkpoint(task,b,3)) for b in range(3)}
    base.write(out(task)/'protocol.json',dict(revision=REV,source_sha=files,parent_sha=parents,iterations=2,times=TIMES,replicas=REPLICAS,test=TEST,
        train={str(i):list(workloads(i,'train')) for i in (1,2)},validation={str(i):list(workloads(i,'valid')) for i in (1,2)},
        target='Completed tasks from intervention until t=200 plus drain, same conditional future and policy RNG across candidates',
        fitting=dict(epochs=100,lr=.001,batch=16,huber=True),max_candidates=16,features=119,cpu=cpus,
        gate='Each lineage expanded>initial and expanded>old; >12/24 test workloads positive; every leave-one-out gain positive',phase2='conditional on foundation gate'))
    original=base.out;base.out=out
    try:
        for iteration in (1,2):
            jobs=[(str(task),b,iteration,w,split) for b in range(3) for split in ('train','valid') for w in workloads(iteration,split)]
            base.parallel(task,f'collect_{iteration}',collect,jobs,cpus)
            labels=[]
            for _,b,_,w,split in jobs:
                for t in TIMES:
                    k=len(state_at(str(task),iteration,b,w,t)['pre']['actions'])
                    labels += [(str(task),iteration,b,w,t,r,c) for r in range(REPLICAS) for c in range(math.ceil(k/4))]
            labels.sort(key=lambda x:x[4])
            base.parallel(task,f'full_return_{iteration}',label,labels,cpus)
            base.parallel(task,f'fit_{iteration}',fit,[(str(task),b,iteration,bank) for b in range(3) for bank in ('old','expanded')],cpus)
        base.parallel(task,'evaluation',evaluate,[(str(task),b,w,mode) for b in range(3) for w in TEST for mode in ('old_learned','expanded_learned','initial','old_model','expanded_model','expanded_argmax')],cpus)
    finally:base.out=original
    summarize(task)
    assert all(base.sha(REPO_ROOT/p)==h for p,h in files.items())
    assert all(base.sha(parent.checkpoint(task,b,3))==parents[str(b)] for b in range(3))
    print('COMPLETE_STAGE1',flush=True)


if __name__=='__main__':main()
