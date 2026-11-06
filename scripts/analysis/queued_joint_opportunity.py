"""Frozen-policy, cross-fitted motion/allocation opportunity diagnosis."""
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
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_service_motion_ppo as policy
from scripts.analysis import queued_service_motion_ranking as base
from utils.queued_joint_opportunity import expanded_bank,motion_plan,conditional_suffix,crossfit
from utils.queued_service_controllers import assignments
from utils.queued_service_learning import VirtualQueue
from utils.queued_service_shared_budget import BudgetAdapter
from utils.queued_service_business_refinement import RecoveryScenario
from environment.queued_service import Engine,Protocol

REV='queued_joint_opportunity_20261102'
WORKLOADS=tuple(range(7200,7204))
TIMES=(10,35,60,85,110,135,160,185)
REPLICAS=8
SMOKE=False


def out(task):return Path(task)/'results'/base.TASK/REV/('smoke' if SMOKE else '')


def sample_plan(model,obs,weights,seed):
    import torch
    from marl_models.queued_service_joint import tensor
    prefix=VirtualQueue(obs);plan={};g=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for x in obs['tasks']:
            if x['source'] is None or x['target'] is not None:continue
            s,f,mask,prior=prefix.features(x['uid'],weights)
            dist=model.allocation(tensor([s]),tensor([f]),tensor([mask],torch.bool),tensor([prior]))
            a=int(torch.multinomial(dist.probs[0],1,generator=g));plan[x['uid']]=a;prefix.assign(x['uid'],a)
    return plan


def rollout(engine,adapter,plan,motion,future,model,decision_seed):
    e=copy.deepcopy(engine);a=copy.deepcopy(adapter)
    initial=len(e.tasks);success=e.counts['success'];expired=e.counts['expired'];cycles=list(e.served_cycles);busy=list(e.cpu_busy)
    e.assign(plan);e.advance(motion)
    for offset in range(1,9):
        e.add(copy.deepcopy(future[offset-1]) if offset<=5 else ());e.admit()
        _,route,move=model.act(e.observe(),a,base.seed(decision_seed,offset))
        e.assign(route);e.advance(move)
    assert not e.tasks;e.audit()
    completed=e.counts['success']-success;failed=e.counts['expired']-expired
    total=initial+sum(len(row) for row in future);assert completed+failed==total
    served=[x-y for x,y in zip(e.served_cycles,cycles)];used=[x-y for x,y in zip(e.cpu_busy,busy)]
    assert all(-1e-7<=x<=9+1e-7 for x in used)
    assert all(s<=c*t*(1+1e-9)+1e-5 for s,c,t in zip(served,e.p.capacity,used))
    return dict(completed=completed,expired=failed,at_risk=total,positions=e.positions,served_cycles=served,cpu_busy=used)


def collect(payload):
    task,block,w=payload;model=policy.model_at(task,block,3);adapter=BudgetAdapter()
    tape=RecoveryScenario(w,'burst','normal',.6);e=Engine(Protocol(),base.START,tape.ue_positions);saved=[];smoke_replay=None
    expected=json.loads((policy.out(task)/'updates'/f'{block}_3.json').read_text())['checkpoint_sha']
    assert base.sha(policy.checkpoint(task,block,3))==expected
    while e.t<200 or e.tasks:
        t=int(e.t);e.add(tape.slot(t));e.admit();obs=e.observe()
        _,plan,move=model.act(obs,adapter,base.seed(REV,'collect',block,w,t))
        if t in TIMES:
            full=dict(time=e.t,positions=e.positions,ues=e.ues,tasks=[asdict(x) for x in sorted(e.tasks.values(),key=lambda x:(x.deadline,x.uid))],protocol=e.p)
            compact=Engine.from_observation(full);compact.audit();ca=copy.deepcopy(adapter)
            ca.history.seen.intersection_update(compact.tasks);ca.forecast_seconds=0.
            bank=expanded_bank(obs,adapter.plans)
            plans=[plan,sample_plan(model,obs,adapter.history.weights,base.seed(REV,'sample_plan',block,w,t)),assignments(obs,'queue'),assignments(obs,'balance')]
            diagonal=[motion_plan(obs,u) for u in bank]
            for route in plans+diagonal:
                check=Engine.from_observation(obs);check.assign(route)
            snap=dict(engine=compact,adapter=ca,plans=plans,diagonal=diagonal,bank=bank,old_count=len(adapter.plans),
                      block=block,workload=w,time=t,new_tasks=len(plan),visible_tasks=len(obs['tasks']))
            p=out(task)/'states'/f'{block}_{w}_{t}.pkl.gz';base.dump(p,snap);saved.append(dict(file=p.name,sha256=base.sha(p),time=t,old_count=len(adapter.plans),new_count=len(bank)))
            if SMOKE and t==TIMES[0]:
                future=conditional_suffix(w,e.ues,t,0);seed=base.seed(REV,'replay')
                left=rollout(e,adapter,plan,bank[0],future,model,seed);right=rollout(compact,ca,plan,bank[0],future,model,seed)
                assert all(left[k]==right[k] for k in ('completed','expired','at_risk','positions'))
                assert all(math.isclose(x,y,rel_tol=1e-8,abs_tol=1e-3) for x,y in zip(left['served_cycles'],right['served_cycles']))
                smoke_replay=True
        e.assign(plan);e.advance(move)
    e.audit();c=dict(e.counts);assert c['generated']==c['success']+c['expired']
    base.write(out(task)/'collection'/f'{block}_{w}.json',dict(block=block,workload=w,counts=c,states=saved,
               parent_sha=expected,smoke_compact_replay=smoke_replay,cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


@lru_cache(maxsize=8)
def state_at(task,block,w,t):return base.read(out(task)/'states'/f'{block}_{w}_{t}.pkl.gz')


def case(payload):
    task,block,w,t,r=payload;started=time.monotonic();snap=state_at(task,block,w,t);model=policy.model_at(task,block,3)
    untouched=hashlib.sha256(pickle.dumps(snap,5)).hexdigest()
    future=conditional_suffix(w,snap['engine'].ues,t,r);future_sha=hashlib.sha256(pickle.dumps(future,5)).hexdigest()
    cache={};seed=base.seed(REV,'suffix',block,w,t,r)
    def value(plan,index):
        key=(tuple(sorted(plan.items())),index)
        if key not in cache:cache[key]=rollout(snap['engine'],snap['adapter'],plan,snap['bank'][index],future,model,seed)
        return cache[key]
    matrix=[[value(plan,i)['completed'] for i in range(len(snap['bank']))] for plan in snap['plans']]
    diagonal=[value(plan,i)['completed'] for i,plan in enumerate(snap['diagonal'])]
    totals={x['at_risk'] for x in cache.values()};assert len(totals)==1
    assert untouched==hashlib.sha256(pickle.dumps(snap,5)).hexdigest()
    assert future_sha==hashlib.sha256(pickle.dumps(future,5)).hexdigest()
    record=dict(block=block,workload=w,time=t,replica=r,values=matrix,diagonal=diagonal,old_count=snap['old_count'],
                count=len(snap['bank']),new_tasks=snap['new_tasks'],distinct_plans=len({tuple(sorted(p.items())) for p in snap['plans']}),
                distinct_motion_plans=len({tuple(sorted(p.items())) for p in snap['diagonal']}),
                at_risk=next(iter(totals)),initial_tasks=len(snap['engine'].tasks),future_counts=[len(row) for row in future],
                future_sha=future_sha,branch_runs=len(cache),all_branch_audits_passed=True,
                seconds=time.monotonic()-started,cpu=sorted(os.sched_getaffinity(0)))
    base.write(out(task)/'cases'/f'{block}_{w}_{t}_{r}.json',record);return dict(cpu=record['cpu'])


def diagnostic(records):
    futures=[x['values'] for x in records];n=records[0]['old_count'];k=records[0]['count']
    expanded=crossfit(futures,n);old=crossfit([[row[:n] for row in m] for m in futures],n)
    diag=[];h=len(records)//2
    for train,test in ((records[:h],records[h:]),(records[h:],records[:h])):
        d=[statistics.mean(r['diagonal'][i] for r in train) for i in range(k)]
        q=[statistics.mean(r['values'][2][i] for r in train) for i in range(k)]
        a=max(range(k),key=lambda i:d[i]);b=max(range(k),key=lambda i:q[i])
        diag.append(statistics.mean(r['diagonal'][a]-r['values'][2][b] for r in test))
    return dict(old=old,expanded=expanded,motion_aware_vs_queue=statistics.mean(diag))


def report(task):
    states=[]
    for block in range(3):
        for w in WORKLOADS:
            for t in TIMES:
                records=[json.loads((out(task)/'cases'/f'{block}_{w}_{t}_{r}.json').read_text()) for r in range(REPLICAS)]
                assert len({(x['old_count'],x['count']) for x in records})==1
                states.append(dict(block=block,workload=w,time=t,new_tasks=records[0]['new_tasks'],old_count=records[0]['old_count'],
                    count=records[0]['count'],distinct_motion_plans=records[0]['distinct_motion_plans'],**diagnostic(records)))
    names=('capacity_gain','joint_vs_native','motion_conditioning','allocation_conditioning');groups={}
    for b in range(3):
        for phase in ('all','before','peak','after'):
            rows=[r for r in states if r['block']==b and (phase=='all' or ('before' if r['time']<50 else 'peak' if r['time']<100 else 'after')==phase)]
            groups[f'{b}/{phase}']=dict(states=len(rows),new_tasks=sum(r['new_tasks']>0 for r in rows),
                added_motion_states=sum(r['count']>r['old_count'] for r in rows),
                old={n:statistics.mean(r['old'][n] for r in rows) for n in names},
                expanded={n:statistics.mean(r['expanded'][n] for r in rows) for n in names},
                motion_aware_vs_queue=statistics.mean(r['motion_aware_vs_queue'] for r in rows))
    result=dict(status='COMPLETE_DIAGNOSTIC',states=states,groups=groups,
                gates={n:all(groups[f'{b}/all']['expanded'][n]>0 for b in range(3)) for n in names},
                boundary='Finite proposed plans/candidates, frozen continuation and six arrival slots plus drain; cross-fitted finite-state diagnosis, not full-episode policy gain or a global upper bound')
    base.write(out(task)/'analysis.json',result)
    lines=['# 运动与分配的配对机会验证','', '业务、物理规则与PPO末策略冻结；未训练新策略。96个新状态、8份条件未来，每状态保留零机会及重复方案权重。', '',
      '|初始化|原候选：运动条件化收益|扩展后：运动条件化收益|原候选：分配条件化收益|扩展后：分配条件化收益|扩展候选容量增量|','|---|---:|---:|---:|---:|---:|']
    for b in range(3):
        g=groups[f'{b}/all'];lines.append(f'|{b}|{g["old"]["motion_conditioning"]:.6f}|{g["expanded"]["motion_conditioning"]:.6f}|{g["old"]["allocation_conditioning"]:.6f}|{g["expanded"]["allocation_conditioning"]:.6f}|{g["expanded"]["capacity_gain"]:.6f}|')
    lines+=['', '单位为任务/状态，全部指标采用四份未来选择、四份未来验证并交换的交叉拟合。负数保留，表示有限样本选择未能转化为留出收益。',
        '运动条件化：每种固定分配各选运动，相对所有分配共用一个运动。分配条件化：每种运动各选固定分配，相对所有运动共用一份分配；规则方案等权是诊断约定，不是策略分布。',
        '容量增量：扩展与原候选各自在选择折挑最佳固定方案/运动组合，在另一折比较。它包含不同候选数量造成的选择难度，不是最优长期收益差。',
        '原固定方案为实际PPO分配、额外同策略样本、EDF队列、负载均衡；重复方案保留但模拟去重。另逐候选生成一份运动感知EDF分配，单独比较，不混入矩形机会指标。',
        '未来在固定UE几何和原潜在热点轨迹下，按原150/s潜在Poisson及15/60稀疏化、原属性分布重新采样。八份到达相互独立；热点轨迹未重采样，不能声称覆盖所有未来不确定性。未来只用于结算，策略与方案生成不读取未来。',
        '新动作保留全部原历史候选，增加上传紧迫性联合/单机方向及两条最紧迫转发方向；安全过滤并去重，最多16个。分配合法性按当前几何，接纳/来源/目标锁定规则不变。',
        '只干预首步；随后逐秒恢复同一冻结PPO策略（原候选集）重规划。六批到达后排空，结果不是扩展策略闭环DSR，也不证明联合学习优于协调分解。',
        '三个既有初始化谱系、四个新工作负载7200–7203，时刻10/35/60/85/110/135/160/185；分阶段结果见analysis。状态/未来不是独立训练重复，不作确证检验。显式自审。']
    dest=Path(task)/'output'/base.TASK/'deliverables'/'queued_joint_opportunity.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(lines),encoding='utf-8')


def main():
    global SMOKE
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();SMOKE=a.smoke;task=a.task
    import torch
    torch.set_num_threads(1);cpus=sorted(os.sched_getaffinity(0))
    if SMOKE:
        collect((str(task),0,WORKLOADS[0]));case((str(task),0,WORKLOADS[0],TIMES[3],0))
        base.write(out(task)/'acceptance.json',dict(status='PASS',compact_vs_full=True,conditional_case=True));print('SMOKE PASS',flush=True);return
    files={str(p.relative_to(REPO_ROOT)):base.sha(p) for d in ('environment','marl_models','utils','scripts/analysis','tests') for p in (REPO_ROOT/d).glob('*.py')}
    parent={str(b):base.sha(policy.checkpoint(task,b,3)) for b in range(3)}
    base.write(out(task)/'protocol.json',dict(revision=REV,source_sha=files,parent_sha=parent,workloads=WORKLOADS,times=TIMES,replicas=REPLICAS,
        fixed_plan_names=['native','sample','queue','balance'],max_motion_candidates=16,arrival_horizon=6,crossfit='0:4 select / 4:8 evaluate and swap',
        gates='Exploratory: per-lineage held-out mean positive; report all negatives; no training or automatic expansion',cpu=cpus))
    original=base.out;base.out=out
    try:
        base.parallel(task,'collect',collect,[(str(task),b,w) for b in range(3) for w in WORKLOADS],cpus)
        base.parallel(task,'paired',case,[(str(task),b,w,t,r) for b in range(3) for w in WORKLOADS for t in TIMES for r in range(REPLICAS)],cpus)
    finally:base.out=original
    report(task)
    assert all(base.sha(REPO_ROOT/p)==h for p,h in files.items())
    assert all(base.sha(policy.checkpoint(task,b,3))==parent[str(b)] for b in range(3))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
