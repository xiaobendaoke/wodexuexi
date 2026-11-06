"""Common expanded-value initialization: one branch, both, or fresh alternation."""
import argparse
import copy
from functools import lru_cache
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_long_value_experiment as stage1
from utils.queued_long_value import LongAdapter
from utils.queued_service_business_refinement import RecoveryScenario
from utils.queued_service_diagnostics import mc_returns
from environment.queued_service import Engine,Protocol

base=stage1.base
REV='queued_long_unfreeze_20261102'
ARMS=('motion','allocation','joint','alternating')
TEST=tuple(range(8600,8624))
ROUNDS=2
PER_ROUND=48
SMOKE=False


def out(task):return Path(task)/'results'/base.TASK/REV/('smoke' if SMOKE else '')


def checkpoint(task,b,arm,version):return out(task)/'models'/arm/str(b)/f'{version}.pt'


@lru_cache(maxsize=16)
def model_at(task,b,arm,version):
    import torch
    m=stage1.factory();d=torch.load(checkpoint(task,b,arm,version),map_location='cpu',weights_only=False);m.load_state_dict(d['model']);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m


def run_case(job):
    task,b,arm,version,w,mode,training=job
    import torch
    from marl_models.queued_long_value import prefix_values
    from scripts.analysis.diagnose_queued_service_conditioning import choose
    m=model_at(task,b,arm,version);adapter=LongAdapter(True);tape=RecoveryScenario(w,'burst','normal',.6)
    e=Engine(Protocol(),base.START,tape.ue_positions);rows=[];experience=[];durations=[]
    while e.t<200 or e.tasks:
        t=int(e.t);assert t<=204;e.add(tape.slot(t));e.admit();obs=e.observe();begin=time.monotonic()
        seed=base.seed(REV,'train' if training else 'eval',b,w,t)
        if mode=='model':row,plan,move=choose(m,obs,adapter,seed,'model_actual',torch)
        else:
            row,plan,move=m.act(obs,adapter,seed)
            if mode.endswith('argmax'):move=adapter.plans[max(range(len(adapter.plans)),key=lambda i:row['motion_probs'][i])]
        durations.append(time.monotonic()-begin)
        if training:
            with torch.no_grad():values,_,tokens=prefix_values(m,[row])
            for token,value in zip(tokens,values.tolist()):token['prefix_value']=value
        before=e.counts['success'];e.assign(plan);rows.append(e.advance(move))
        if training:row.pop('history_weights',None);row.update(reward=(e.counts['success']-before)/20);experience.append(row)
    e.audit();c=dict(e.counts);assert c['generated']==c['success']+c['expired'] and sum(r['success'] for r in rows)==c['success']
    name=f'{arm}_{version}_{b}_{w}_{mode}';digest=None
    if training:
        targets=mc_returns([r['reward'] for r in experience])
        for row,target in zip(experience,targets):row['returns']=target
        p=out(task)/'experience'/f'{name}.pkl.gz';base.dump(p,experience);digest=base.sha(p)
    base.write(out(task)/'episodes'/f'{name}.json',dict(arm=arm,version=version,block=b,workload=w,mode=mode,training=training,
        counts=c,summary=e.summary(),rows=rows,fingerprint=tape.fingerprint,experience_sha=digest,
        checkpoint_sha=base.sha(checkpoint(task,b,arm,version)),mean_decision_seconds=statistics.mean(durations),over_one_second=sum(x>1 for x in durations),cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def fit(payload):
    task,b,arm,before,after,ws,active=payload
    import torch
    from marl_models.queued_long_value import update_branches
    m=copy.deepcopy(model_at(task,b,arm,before));torch.manual_seed(base.seed(REV,'fit',b,arm,str(after)));rows=[]
    for w in ws:
        name=f'{arm}_{before}_{b}_{w}_train';p=out(task)/'experience'/f'{name}.pkl.gz'
        d=json.loads((out(task)/'episodes'/f'{name}.json').read_text());assert base.sha(p)==d['experience_sha']
        ep=base.read(p);target=mc_returns([r['reward'] for r in ep]);assert all(abs(r['returns']-v)<1e-9 for r,v in zip(ep,target));rows+=ep
    previous=torch.load(checkpoint(task,b,arm,before),map_location='cpu',weights_only=False)
    metrics=update_branches(m,rows,active,actor_epochs=4 if arm=='alternating' else 2,critic_epochs=2,optimizer_states=previous.get('optimizer_states'))
    states=metrics.pop('_optimizer_states');p=checkpoint(task,b,arm,after);p.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=m.state_dict(),optimizer_states=states,block=b,arm=arm,version=after),p)
    base.write(out(task)/'updates'/f'{arm}_{after}_{b}.json',dict(arm=arm,block=b,before=before,after=after,active=active,workloads=list(ws),
        episodes=len(ws),rows=len(rows),metrics=metrics,checkpoint_sha=base.sha(p),cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def summary(task):
    modes=('frozen','frozen_argmax','model','motion','allocation','joint','alternating','motion_argmax','joint_argmax','alternating_argmax')
    records=[json.loads(p.read_text()) for p in (out(task)/'episodes').glob('*.json')];ev=[r for r in records if not r['training']]
    assert len(ev)==len(modes)*3*len(TEST)
    table={}
    for mode in modes:
        table[mode]=[]
        for b in range(3):
            rows=[r for r in ev if r['mode']==mode and r['block']==b];assert len(rows)==24
            c={k:sum(r['counts'][k] for r in rows) for k in ('generated','success','admitted')}
            table[mode].append(dict(block=b,**c,dsr=c['success']/c['generated'],admission=c['admitted']/c['generated'],conditional=c['success']/c['admitted']))
    for b in range(3):
        for w in TEST:
            matched=[r for r in ev if r['block']==b and r['workload']==w]
            assert len({r['fingerprint'] for r in matched})==1 and len({r['counts']['generated'] for r in matched})==1
    contrasts={reference:[table['joint'][b]['dsr']-table[reference][b]['dsr'] for b in range(3)] for reference in ('frozen','motion','allocation','alternating','model')}
    robustness={}
    for reference in ('motion','allocation','alternating'):
        robustness[reference]=[]
        for b in range(3):
            def aggregate(mode,omit=None):
                ds=[r['counts'] for r in ev if r['mode']==mode and r['block']==b and r['workload']!=omit]
                return sum(d['success'] for d in ds)/sum(d['generated'] for d in ds)
            deltas=[]
            for w in TEST:
                x=next(r for r in ev if r['mode']=='joint' and r['block']==b and r['workload']==w)['counts']
                y=next(r for r in ev if r['mode']==reference and r['block']==b and r['workload']==w)['counts']
                deltas.append((x['success']-y['success'])/x['generated'])
            loo=[aggregate('joint',w)-aggregate(reference,w) for w in TEST]
            robustness[reference].append(dict(block=b,positive=sum(x>0 for x in deltas),negative=sum(x<0 for x in deltas),median=statistics.median(deltas),min_leave_one_out=min(loo)))
    primary=all(x>0 for ref in ('motion','allocation','alternating') for x in contrasts[ref])
    reliable=primary and all(r['positive']>12 and r['min_leave_one_out']>0 for rows in robustness.values() for r in rows)
    protocol=json.loads((out(task)/'protocol.json').read_text())
    result=dict(status='COMPLETE',summary=table,joint_contrasts=contrasts,
        joint_gate=primary,robustness=robustness,reliable_development_gate=reliable,
        foundation_passed=protocol['foundation_passed'],evidence_mode=protocol['evidence_mode'],independent_confirmation=False)
    base.write(out(task)/'analysis.json',result)
    lines=['# 相同扩展候选下的分配解冻对照','', '|控制器|三谱系平均DSR|逐谱系DSR|','|---|---:|---|']
    for mode,rows in table.items():lines.append(f'|{mode}|{statistics.mean(r["dsr"] for r in rows)*100:.4f}%|'+ '/'.join(f'{r["dsr"]*100:.4f}%' for r in rows)+'|')
    lines+=['', '共同训练减各对照（百分点）：'+str({k:[100*x for x in v] for k,v in contrasts.items()}),
        '运动基础门槛：'+str(protocol['foundation_passed'])+'；本阶段证据模式：'+protocol['evidence_mode'],
        '联合均值开发门槛：'+str(result['joint_gate'])+'；工作负载稳健性门槛：'+str(reliable),str(robustness), '',
        '共同起点为第一阶段末扩展价值模型；所有学习对照相同119维特征、候选、前缀/运动价值和真实完整MC奖励。',
        '运动单训、分配单训、共同训练各轮48完整回合、2 actor/critic epochs。交替对照先24回合训练分配，再新采样24回合训练运动，各活跃actor4轮、critic2轮；总交互相同、每活跃分支优化次数按实际批大小核对。优化器状态持久化。',
        '分配基线在请求动作前的前缀状态计算并保存；不使用当前请求的采样动作作基线。动作分别clip0.2、KL0.02、lr.0003、entropy.001、梯度0.5。',
        '交替对照与共同训练均可看本轮分配并协调，不把强分解对照削弱为无信息策略；区别是新采样交替更新。',
        '第一阶段头部由长期动作价值拟合初始化；第二阶段接受PPO策略梯度，继续称为评分器，不把其输出当作仍然校准的Q。',
        '测试8600–8623，训练8700–8795，末检查点、无按测试选模型或追加训练。三个既有谱系开发研究，非独立确认。',
        'argmax是预定辅助解码，只对冻结、运动、共同和交替报告；主比较使用采样，不择优混用。']
    p=Path(task)/'output'/base.TASK/'deliverables'/'queued_long_unfreeze.md';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(lines),encoding='utf-8')


def main():
    global SMOKE
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--exploratory',action='store_true');a=p.parse_args();task=a.task;SMOKE=a.smoke
    import torch
    torch.set_num_threads(1)
    if SMOKE:
        source=stage1.checkpoint(task,0,2)
        if not source.exists():source=stage1.checkpoint(task,0,0)
        before=base.sha(source);saved=torch.load(source,map_location='cpu',weights_only=False)
        target=checkpoint(task,0,'joint',0);target.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(model=saved['model']),target)
        run_case((str(task),0,'joint',0,8599,'train',True))
        fit((str(task),0,'joint',0,1,[8599],'joint'))
        assert base.sha(source)==before
        base.write(out(task)/'acceptance.json',dict(status='PASS',full_mc=True,parents_unchanged=True));print('SMOKE PASS',flush=True);return
    foundation=json.loads((stage1.out(task)/'analysis.json').read_text())
    import fcntl
    validation=Path(task)/'output'/base.TASK/'validation'/REV;validation.mkdir(parents=True,exist_ok=True)
    controller_lock=(validation/'controller.lock').open('a+')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not foundation['foundation_gate'] and not a.exploratory:
        base.write(out(task)/'status.json',dict(status='NOT_RUN_FOUNDATION_GATE_FAILED',source_stage1=foundation['status']))
        dest=Path(task)/'output'/base.TASK/'deliverables'/'queued_long_unfreeze.md';dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text('# 分配解冻阶段：未启动\n\n运动基础未通过预定开发门槛，未消耗正式解冻训练预算。实现、分支冻结与前缀基线测试可单独验证；不能称两阶段优势已完成。\n',encoding='utf-8')
        print('NOT RUN: foundation gate failed',flush=True);return
    cpus=sorted(os.sched_getaffinity(0));files={str(p.relative_to(REPO_ROOT)):base.sha(p) for d in ('environment','marl_models','utils','scripts/analysis','tests') for p in (REPO_ROOT/d).glob('*.py')}
    parents={str(b):base.sha(stage1.checkpoint(task,b,2)) for b in range(3)}
    base.write(out(task)/'protocol.json',dict(revision=REV,source_sha=files,parent_sha=parents,arms=ARMS,rounds=2,episodes_per_arm_round=48,
        foundation_passed=foundation['foundation_gate'],evidence_mode='development_after_foundation' if foundation['foundation_gate'] else 'exploratory_without_validated_foundation',
        foundation_analysis_sha=base.sha(stage1.out(task)/'analysis.json'),
        test=TEST,train=list(range(8700,8796)),actor_epochs=2,alternating_actor_epochs=4,critic_epochs=2,batch=256,lr=.0003,clip=.2,kl=.02,
        gamma=1.,entropy=.001,gradient_clip=.5,critic_scale=100,cpu=cpus,
        joint_gate='Each joint-minus-motion/allocation/alternating lineage mean positive; reliability also >12/24 workloads positive and every leave-one-out gain positive; no automatic expansion'))
    for b in range(3):
        saved=torch.load(stage1.checkpoint(task,b,2),map_location='cpu',weights_only=False)
        for arm in ARMS:
            p=checkpoint(task,b,arm,0);p.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(model=saved['model']),p)
    original=base.out;base.out=out
    try:
        for r in (1,2):
            ws=list(range(8700+(r-1)*48,8700+r*48));first=[]
            for b in range(3):
                for arm in ARMS:
                    for w in (ws[:24] if arm=='alternating' else ws):first.append((str(task),b,arm,r-1,w,'train',True))
            base.parallel(task,f'train_{r}_first',run_case,first,cpus)
            fits=[(str(task),b,arm,r-1,f'{r}a' if arm=='alternating' else r,ws[:24] if arm=='alternating' else ws,'allocation' if arm=='alternating' else arm) for b in range(3) for arm in ARMS]
            base.parallel(task,f'update_{r}_first',fit,fits,cpus)
            second=[(str(task),b,'alternating',f'{r}a',w,'train',True) for b in range(3) for w in ws[24:]]
            if r==1:
                second += [(str(task),b,'motion',0,w,mode,False) for b in range(3) for w in TEST for mode in ('frozen','frozen_argmax','model')]
            else:
                second += [(str(task),b,arm,2,w,arm,False) for b in range(3) for w in TEST for arm in ('motion','allocation','joint')]
            base.parallel(task,f'train_{r}_second_and_eval',run_case,second,cpus)
            base.parallel(task,f'update_{r}_second',fit,[(str(task),b,'alternating',f'{r}a',r,ws[24:],'motion') for b in range(3)],cpus)
        final=[(str(task),b,'alternating',2,w,'alternating',False) for b in range(3) for w in TEST]
        final += [(str(task),b,arm,2,w,arm+'_argmax',False) for b in range(3) for w in TEST for arm in ('motion','joint','alternating')]
        base.parallel(task,'final_evaluation',run_case,final,cpus)
    finally:base.out=original
    summary(task)
    assert all(base.sha(REPO_ROOT/p)==h for p,h in files.items())
    assert all(base.sha(stage1.checkpoint(task,b,2))==parents[str(b)] for b in range(3))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
