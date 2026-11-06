"""True-return motion PPO after ranking warm start; allocation stays frozen."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_service_motion_ranking as base
from environment.queued_service import Engine,Protocol
from utils.queued_service_shared_budget import BudgetAdapter
from utils.queued_service_business_refinement import RecoveryScenario
from utils.queued_service_diagnostics import mc_returns

REV='queued_motion_ppo_20261101'
TEST=tuple(range(6200,6224))
ROUNDS=3
PER_ROUND=96
CACHE={}


def out(task):return Path(task)/'results'/base.TASK/REV


def checkpoint(task,block,round_):return out(task)/'models'/str(block)/f'{round_:04}.pt'


def complete_credit(rows):
    returns=mc_returns([r['reward'] for r in rows])
    for r,target in zip(rows,returns):r.update(returns=target,motion_advantage=target-r['motion_value'])


def replay_motion(model,rows):
    import torch
    from marl_models.queued_service_joint import tensor
    s=tensor([r['motion_state'] for r in rows]);f=tensor([r['motion_features'] for r in rows]);mask=tensor([r['motion_mask'] for r in rows],torch.bool)
    ps=tensor([r['post_state'] for r in rows]);pf=tensor([r['post_features'] for r in rows])
    dist=model.motion(s,f,mask,ps,pf,True);_,v=model.critic(s,f,mask,ps,pf)
    old=tensor([r['motion_probs'] for r in rows])
    return dict(log=dist.log_prob(torch.tensor([r['motion_action'] for r in rows])),value=v,entropy=dist.entropy(),
                kl=(old*(old.clamp_min(1e-30).log()-dist.probs.clamp_min(1e-30).log())).sum(-1))


def update(model,rows,epochs=2,batch=256,optimizer_state=None):
    import torch
    for p in model.allocation.parameters():p.requires_grad_(False)
    for module in (model.motion,model.critic):
        for p in module.parameters():p.requires_grad_(True)
    parameters=list(model.motion.parameters())+list(model.critic.parameters())
    optimizer=torch.optim.Adam(parameters,lr=3e-4)
    if optimizer_state is not None:optimizer.load_state_dict(optimizer_state)
    before={name:{k:v.clone() for k,v in getattr(model,name).state_dict().items()} for name in ('allocation','motion','critic')}
    target=torch.tensor([r['returns'] for r in rows]);old=torch.tensor([r['motion_log'] for r in rows])
    advantages=target-torch.tensor([r['motion_value'] for r in rows]);advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(.05)
    steps=0;accepted=0;losses=[];critic_losses=[];replay_error=0.
    with torch.no_grad():
        for start in range(0,len(rows),batch):
            z=replay_motion(model,rows[start:start+batch]);replay_error=max(replay_error,float((z['log']-old[start:start+batch]).abs().max()))
    assert replay_error<2e-5, replay_error
    for _ in range(epochs):
        for ids in torch.randperm(len(rows)).split(batch):
            selected=[rows[i] for i in ids.tolist()];z=replay_motion(model,selected)
            ratio=(z['log']-old[ids]).clamp(-20,20).exp();a=advantages[ids]
            actor=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()-.001*z['entropy'].mean()
            critic=((z['value']-target[ids])/100).square().mean()
            allow=float(z['kl'].mean().detach())<=.02
            loss=.5*critic+(actor if allow else actor*0)
            assert bool(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if not allow:
                for p in model.motion.parameters():p.grad=None
            for module in (model.motion,model.critic):torch.nn.utils.clip_grad_norm_(module.parameters(),.5,error_if_nonfinite=True)
            optimizer.step();steps+=1;accepted+=int(allow);losses.append(float(actor.detach()));critic_losses.append(float(critic.detach()))
    changes={name:sum(float((v-before[name][k]).square().sum()) for k,v in getattr(model,name).state_dict().items())**.5 for name in before}
    assert changes['allocation']==0
    values=[];kls=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch):
            z=replay_motion(model,rows[start:start+batch]);values+=z['value'].tolist();kls+=z['kl'].tolist()
    from utils.queued_service_diagnostics import explained_variance
    metrics=dict(optimizer_steps=steps,actor_steps=accepted,actor_skips=steps-accepted,actor_loss=statistics.mean(losses),
                 critic_loss=statistics.mean(critic_losses),replay_error=replay_error,kl=statistics.mean(kls),
                 value_ev=explained_variance(target.tolist(),values),**{name+'_change':v for name,v in changes.items()})
    metrics['_optimizer']=optimizer.state_dict()
    return metrics


def model_at(task,block,round_):
    import torch
    from marl_models.queued_service_plan_residual import ResidualPolicy
    key=(str(task),block,round_)
    if key not in CACHE:
        saved=torch.load(checkpoint(task,block,round_),map_location='cpu',weights_only=False)
        m=ResidualPolicy('joint');m.load_state_dict(saved['model']);m.eval()
        for p in m.parameters():p.requires_grad_(False)
        CACHE[key]=m
    return CACHE[key]


def episode(payload):
    task,block,round_,workload,mode,training=payload
    import torch
    from scripts.analysis.diagnose_queued_service_conditioning import choose
    m=model_at(task,block,round_);adapter=BudgetAdapter();tape=RecoveryScenario(workload,'burst','normal',.6)
    e=Engine(Protocol(),base.START,tape.ue_positions);rows=[];experience=[];durations=[];started=time.monotonic()
    while e.t<200 or e.tasks:
        t=int(e.t);assert t<=204;e.add(tape.slot(t));e.admit();obs=e.observe();begin=time.monotonic()
        # Evaluation streams do not depend on checkpoint or decode mode.
        action_seed=base.seed(REV,'train' if training else 'test',block,workload,t)
        if mode=='model':row,plan,motion=choose(m,obs,adapter,action_seed,'model_actual',torch)
        else:
            row,plan,motion=m.act(obs,adapter,action_seed)
            if mode.endswith('argmax'):motion=adapter.plans[max(range(len(adapter.plans)),key=lambda i:row['motion_probs'][i])]
        durations.append(time.monotonic()-begin);success=e.counts['success'];e.assign(plan);result=e.advance(motion);rows.append(result)
        if training:
            keys=('motion_state','motion_features','motion_mask','post_state','post_features','motion_action','motion_log','motion_probs','motion_value')
            experience.append(dict({k:row[k] for k in keys},reward=(e.counts['success']-success)/20,decision_time=t))
    e.audit();summary=e.summary();c=summary['counts'];assert c['generated']==c['success']+c['expired'] and not e.tasks
    assert sum(r['success'] for r in rows)==c['success']
    name=f'{block}_{round_}_{workload}_{mode}';experience_sha=None
    if training:
        complete_credit(experience);p=out(task)/'experience'/f'{name}.pkl.gz';base.dump(p,experience);experience_sha=base.sha(p)
    record=dict(block=block,round=round_,workload=workload,mode=mode,training=training,counts=dict(c),summary=summary,rows=rows,
                fingerprint=tape.fingerprint,checkpoint_sha=base.sha(checkpoint(task,block,round_)),experience_sha=experience_sha,
                mean_decision_seconds=statistics.mean(durations),over_one_second=sum(x>1 for x in durations),
                cpu=sorted(os.sched_getaffinity(0)),seconds=time.monotonic()-started)
    base.write(out(task)/'episodes'/f'{name}.json',record);return dict(cpu=record['cpu'])


def fit(payload):
    task,block,round_=payload
    import torch
    torch.manual_seed(base.seed(REV,'update',block,round_));m=copy.deepcopy(model_at(task,block,round_-1));rows=[]
    workloads=range(6300+(round_-1)*PER_ROUND,6300+round_*PER_ROUND)
    for w in workloads:
        p=out(task)/'experience'/f'{block}_{round_-1}_{w}_ppo.pkl.gz'
        record=json.loads((out(task)/'episodes'/f'{block}_{round_-1}_{w}_ppo.json').read_text())
        assert base.sha(p)==record['experience_sha']
        episode_rows=base.read(p);expected=mc_returns([r['reward'] for r in episode_rows])
        assert all(abs(r['returns']-x)<1e-9 for r,x in zip(episode_rows,expected));rows+=episode_rows
    previous=torch.load(checkpoint(task,block,round_-1),map_location='cpu',weights_only=False)
    metrics=update(m,rows,optimizer_state=previous.get('optimizer'));optimizer=metrics.pop('_optimizer')
    target=checkpoint(task,block,round_);target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=m.state_dict(),optimizer=optimizer,round=round_,block=block),target)
    base.write(out(task)/'updates'/f'{block}_{round_}.json',dict(block=block,round=round_,episodes=PER_ROUND,rows=len(rows),metrics=metrics,
               parent_sha=base.sha(checkpoint(task,block,round_-1)),checkpoint_sha=base.sha(target),cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def report(task):
    import torch
    modes=('frozen','ppo','frozen_argmax','ppo_argmax','model');summary={};paired={}
    for mode in modes:
        round_=0 if mode.startswith('frozen') or mode=='model' else ROUNDS;summary[mode]=[]
        for b in range(3):
            rows=[json.loads((out(task)/'episodes'/f'{b}_{round_}_{w}_{mode}.json').read_text()) for w in TEST]
            c={k:sum(r['counts'][k] for r in rows) for k in ('generated','success','admitted')}
            summary[mode].append(dict(block=b,**c,dsr=c['success']/c['generated'],admission=c['admitted']/c['generated'],
                         conditional=c['success']/c['admitted'],mean_decision_seconds=statistics.mean(r['mean_decision_seconds'] for r in rows)))
    for mode,reference in (('ppo','frozen'),('ppo_argmax','frozen_argmax')):
        paired[mode]=[summary[mode][b]['dsr']-summary[reference][b]['dsr'] for b in range(3)]
    updates=[json.loads(p.read_text()) for p in sorted((out(task)/'updates').glob('*.json'))];assert len(updates)==9
    for b in range(3):
        initial=torch.load(checkpoint(task,b,0),map_location='cpu',weights_only=False)['model']
        final=torch.load(checkpoint(task,b,ROUNDS),map_location='cpu',weights_only=False)['model']
        assert all(torch.equal(v,final[k]) for k,v in initial.items() if k.startswith('allocation.'))
    result=dict(status='COMPLETE',summary=summary,paired_gains=paired,updates=updates,
                learning_gate=all(x>0 for x in paired['ppo']),allocation_frozen=True,independent_confirmation=False)
    base.write(out(task)/'analysis.json',result)
    lines=['# 排序初始化后的真实回报运动PPO','', '开发实验：冻结业务、候选和分配，三初始化，各288完整训练回合、3次数据更新、每批2轮优化。', '',
           '|运动|三初始化平均DSR|逐初始化DSR|','|---|---:|---|']
    for mode,rows in summary.items():lines.append(f'|{mode}|{statistics.mean(r["dsr"] for r in rows)*100:.4f}%|'+ '/'.join(f'{r["dsr"]*100:.4f}%' for r in rows)+'|')
    lines+=['', 'PPO采样相对冻结采样的百分点差：'+str([100*x for x in paired['ppo']]),
            'PPO取最大相对冻结取最大的百分点差：'+str([100*x for x in paired['ppo_argmax']]),
            '', '测试6200–6223，训练6300–6587，均与先前排序数据分离。预定末检查点，未按测试择优；不扩训。',
            '实际完成奖励/20，gamma=1，完整回合及尾部排空MC；基线为采样运动前的计划条件价值。仅运动PPO和critic更新，分配参数逐项冻结。',
            'PPO clip0.2、lr3e-4、熵0.001、梯度裁剪0.5、minibatch256、每批2轮、运动KL门槛0.02。价值仅使用归一化总体MC误差，未继续使用100倍组内损失。',
            '本轮同时检验真实回报更新和价值总体损失方案的组合，不作单组件归因。冻结对照无优化成本，不声称等总计算预算。',
            '三批各288条独立训练轨迹采用256单线程CPU进程；三个拟合任务不复制填核。未来标签、教师损失和新规划评价不进入PPO更新。',
            '不代表解冻分配后的联合学习，不是独立确认；DSR跨不同工作负载不能与上一轮绝对值直接相减。显式自审。']
    dest=Path(task)/'output'/base.TASK/'deliverables'/'queued_motion_ppo.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=Path,required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args();task=args.task
    import torch
    torch.set_num_threads(1);cpus=sorted(os.sched_getaffinity(0));assert len(cpus)==256
    sources={str(p.relative_to(REPO_ROOT)):base.sha(p) for f in ('environment','marl_models','utils','scripts/analysis','tests') for p in (REPO_ROOT/f).glob('*.py')}
    parent={str(b):dict(base_checkpoint=base.sha(base.checkpoint(task,b)),ranking_checkpoint=base.sha(base.out(task)/'models'/f'{b}.pt')) for b in range(3)}
    base.write(out(task)/'protocol.json',dict(revision=REV,source_sha=sources,parent=parent,train=list(range(6300,6588)),test=TEST,
               rounds=ROUNDS,episodes_per_block_round=PER_ROUND,epochs=2,batch=256,lr=.0003,clip=.2,entropy=.001,kl=.02,
               gamma=1.,reward='new actual on-time completions /20; complete MC',acceptance='All three sampled PPO-minus-frozen gains positive; report failure, no model selection or automatic expansion',cpu=cpus))
    for b in range(3):
        m=base.model_at(task,b,True);p=checkpoint(task,b,0);p.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=m.state_dict(),round=0,block=b),p)
    if args.smoke:
        episode((str(task),0,0,6299,'smoke',True))
        rows=base.read(out(task)/'experience'/'0_0_6299_smoke.pkl.gz')
        metrics=update(copy.deepcopy(model_at(task,0,0)),rows);metrics.pop('_optimizer')
        base.write(out(task)/'smoke.json',dict(status='PASS',metrics=metrics,complete_episode=True))
        print('SMOKE PASS',flush=True);return
    # Reuse the audited all-CPU scheduler, writing execution metadata here.
    original_out=base.out;base.out=out
    try:
        for r in range(1,ROUNDS+1):
            jobs=[(str(task),b,r-1,w,'ppo',True) for b in range(3) for w in range(6300+(r-1)*PER_ROUND,6300+r*PER_ROUND)]
            base.parallel(task,f'train_{r}',episode,jobs,cpus)
            base.parallel(task,f'update_{r}',fit,[(str(task),b,r) for b in range(3)],cpus)
        jobs=[(str(task),b,0 if mode.startswith('frozen') or mode=='model' else ROUNDS,w,mode,False)
              for b in range(3) for w in TEST for mode in ('frozen','ppo','frozen_argmax','ppo_argmax','model')]
        base.parallel(task,'evaluation',episode,jobs,cpus)
    finally:base.out=original_out
    report(task)
    assert all(base.sha(REPO_ROOT/p)==h for p,h in sources.items())
    assert all(base.sha(base.checkpoint(task,b))==parent[str(b)]['base_checkpoint'] and base.sha(base.out(task)/'models'/f'{b}.pt')==parent[str(b)]['ranking_checkpoint'] for b in range(3))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
