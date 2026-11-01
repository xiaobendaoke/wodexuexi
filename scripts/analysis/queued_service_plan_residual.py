"""Same-rollout-budget pre bypass and sampled-plan value pilot, with diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import statistics
import sys
import time
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_service_shared_budget as parent
from scripts.analysis import diagnose_queued_service_conditioning as diagnosis
from scripts.analysis.queued_service_business import read,raw_write
from scripts.analysis.queued_service_validation import atomic,sha,now,START,core_order
from utils.queued_service_shared_budget import BudgetAdapter,mc_credit
from utils.queued_service_learning import VirtualQueue
from utils.queued_service_conditioning import value_fit_by_phase
from utils.queued_service_diagnostics import probability_metrics,mc_returns,commit_sequence
from environment.queued_service import Engine,Protocol
from utils.queued_service_business_refinement import RecoveryScenario

core=parent.core
REVISION='queued_plan_residual_20261101'
BASE=parent.protocol()
assert BASE['source_contract']=='15ea668d0efc613157c85b6b761b1741c7e35f4a371f918a5785586210c61257'
SOURCE_FILES=diagnosis.SOURCE_FILES+('utils/queued_service_plan_residual.py','marl_models/queued_service_plan_residual.py',
    'scripts/analysis/queued_service_plan_residual.py','tests/test_queued_service_plan_residual.py')


def contract():
    files={f:sha(REPO_ROOT/f) for f in SOURCE_FILES}
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files


def locations(task):return task/'results'/core.TASK/REVISION,task/'output'/core.TASK/'validation'/REVISION


def protocol():
    d=copy.deepcopy(BASE);digest,files=contract()
    d.update(revision=REVISION,source_contract=digest,source_files=files,parent_contract=BASE['source_contract'],
        difference='Common pre BudgetMotion plus joint-only zero-initialized gated plan residual; both motion critics condition on sampled plan, allocation baseline pre only',
        motion='Same safe bank, retained pre features; residual receives normalized pre, post-pre and state difference, tanh gate starts .1; no ranking teacher',
        credit=dict(gamma=1.,gae_lambda=1.,terminal_value=0.,allocation='MC minus pre critic',motion='MC minus sampled-plan critic; before motion sampling',
            value='pre state+legal mean/min/max raw candidate features; plan delta only in motion baseline; scale100 time base + scale10 spatial/plan residual',
            loss='mean of two raw squared errors /100 plus same-minibatch time-centered errors /10; coefficient .5; labels training only'),
        acceptance=dict(value='Both overall and same-block/time centered EV; target centered>0 and higher than frozen parent, report failures',
            ranking='Expected payoff minus uniform and best-candidate regret; predeclared one_step and six_steps; target gain over matched old probabilities positive at both scales',
            matched='Original frozen 192 states and actual plans; reuse fixed-continuation payoff matrices; counter-input distribution caveat',
            on_policy='48 final burst trajectories, four times each; same continuation diagnostic; no future used by actors',
            no_selection=True),
        budget_boundary='Same 288 train episodes and 156 optimizer steps per learner, 576 learned eval+96 controls. Extra critic capacity, double forecast and diagnostics separately reported',
        value_check='No separate pretraining; old offline fit protocol not rerun',combined_change=True)
    d.pop('model_features',None)
    return d


def activate():
    import marl_models.queued_service_joint as frozen
    from marl_models.queued_service_plan_residual import ResidualPolicy,residual_update
    core.REVISION=REVISION;core.SOURCES=SOURCE_FILES;core.protocol=protocol;core.LearningAdapter=BudgetAdapter
    core.finite_gae=mc_credit;core.action_seed=parent.action_seed;core.analyze=analyze
    frozen.JointPolicy=ResidualPolicy;frozen.update=residual_update


def validate(task):
    core.configure();os.environ['QUEUED_TORCH_TESTS']='1';stream=io.StringIO()
    names=['tests.test_queued_service_plan_residual','tests.test_queued_service_conditioning','tests.test_queued_service_shared_budget',
        'tests.test_queued_service_diagnostics','tests.test_queued_service_learning','tests.test_queued_service',
        'tests.test_queued_service_business','tests.test_queued_service_business_refinement']
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    _,v=locations(task);atomic(v/'tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        skipped=len(result.skipped),source_contract=contract()[0],time=now(),log=stream.getvalue(),independent_review=False))
    print(stream.getvalue(),flush=True);return result.wasSuccessful()


def lookup(summary,method,block,cp):
    return next(r for r in summary if r['method']==method and r['block']==block and r['checkpoint']==cp and r['profile']=='burst')


def analyze(task):
    out,val=locations(task);digest=contract()[0];groups={};hashes={};fingerprints={}
    jobs=[j for m in core.METHODS for b in range(3) for cp in (0,3,6) for j in core.evaluation_jobs(m,b,cp)]
    jobs += [j for m in core.CONTROLS for j in core.evaluation_jobs(m)]
    jobs += [j for m in core.METHODS for b in range(3) for r in range(1,7) for j in core.training_jobs(m,b,r)]
    assert len(jobs)==960
    parent_out,_=parent.locations(task);previous=read(parent_out/'analysis.json');updates={}
    for job in jobs:
        path=core.result_path(out,job);d=read(path);assert d['source_contract']==digest and d['job']==job and d['audit']['status']=='PASS'
        c=d['summary']['counts'];assert c['generated']==c.get('success',0)+c.get('expired',0)
        assert d['rows'][-1]['active']==0 and sum(r['success'] for r in d['rows'])==c.get('success',0)
        assert sum(d['summary']['expiration_stage'].values())==c.get('expired',0)
        key=(job['profile'],job['workload']);assert key not in fingerprints or fingerprints[key]==d['arrival_fingerprint'];fingerprints[key]=d['arrival_fingerprint']
        if job['method'] in core.METHODS:assert d['checkpoint_sha256']==sha(core.model_path(out,job['method'],job['block'],job['checkpoint']))
        else:
            old=read(core.result_path(parent_out,job));assert c==old['summary']['counts']
        hashes[path.relative_to(out).as_posix()]=sha(path)
        if job['kind']=='eval':groups.setdefault((job['profile'],job['method'],job['block'],job['checkpoint']),[]).append(d)
    summary=[dict(profile=p,method=m,block=b,checkpoint=cp,**core.aggregate(ds)) for (p,m,b,cp),ds in sorted(groups.items())]
    for m,b in itertools.product(core.METHODS,range(3)):
        ds=[read(out/'training'/str(b)/m/f'{r:04}.json') for r in range(1,7)]
        assert sum(d['episodes'] for d in ds)==48 and sum(d['update']['optimizer_steps'] for d in ds)==156
        assert all(d['source_contract']==digest for d in ds)
        updates[f'{m}/{b}']=[d['update'] for d in ds]
    gains={m:[lookup(summary,m,b,6)['dsr']-lookup(summary,m,b,0)['dsr'] for b in range(3)] for m in core.METHODS}
    delta=[lookup(summary,'joint',b,6)['dsr']-lookup(summary,'decomposed',b,6)['dsr'] for b in range(3)]
    result=dict(status='PASS_RAW_EXPERIMENT_AUDIT',source_contract=digest,parent_contract=BASE['source_contract'],summary=summary,updates=updates,
        gain_over_initialization=gains,joint_minus_decomposed=delta,learning_gate={m:all(x>0 for x in a) and statistics.mean(a)>=.01 for m,a in gains.items()},
        old_final_difference={m:[lookup(summary,m,b,6)['dsr']-lookup(previous['summary'],m,b,6)['dsr'] for b in range(3)] for m in core.METHODS},
        raw_sha256=hashes,completed=now(),independent_confirmation=False,automatic_expansion=False)
    atomic(out/'analysis.json',result)
    deliver=task/'output'/core.TASK/'deliverables';deliver.mkdir(parents=True,exist_ok=True)
    with (deliver/'queued_plan_residual_summary.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    return result


def load(task,m,b,torch,cp=6):
    from marl_models.queued_service_plan_residual import ResidualPolicy
    out,_=locations(task);path=core.model_path(out,m,b,cp);d=torch.load(path,map_location='cpu',weights_only=False)
    assert d['source_contract']==contract()[0];model=ResidualPolicy(m);model.load_state_dict(d['model']);model.eval();return model,sha(path)


def score(model,pre,post,torch):
    from marl_models.queued_service_plan_residual import tensor
    with torch.no_grad():
        s=tensor([pre['state']]);f=tensor([pre['features']]);mask=tensor([pre['mask']],torch.bool);ps=tensor([post['state']]);pf=tensor([post['features']])
        dist=model.motion(s,f,mask,ps,pf,model.method=='joint');v,mv=model.critic(s,f,mask,ps,pf)
        return dist.probs[0,:sum(pre['mask'])].tolist(),float(v[0]),float(mv[0])


def rank_record(matrix,prob,uniform=None):
    p=[x/sum(prob) for x in prob];q=matrix
    expected=sum(a*b for a,b in zip(p,q));baseline=statistics.mean(q)
    return dict(expected=expected,uniform=baseline,gain=expected-baseline,regret=max(q)-expected,
        informative=max(q)>min(q),best=max(q))


def actual_payoff(engine,tape,plan,actions):
    zero=[(0.,0.)]*5;result={}
    for scale in ('one_step','six_steps'):
        values=[]
        for first in actions:
            sequence=[first]+[zero]*5 if scale=='one_step' else commit_sequence(engine,first)
            f=copy.deepcopy(engine);offset=0
            while offset<6 or f.tasks:
                assert offset<=12
                if offset:
                    f.add(tape.slot(int(engine.t)+offset) if offset<6 else ());new=f.admit();route={uid:f.tasks[uid].source for uid in new}
                else:route=plan
                f.assign(route);f.advance(sequence[offset] if offset<6 else zero);offset+=1
            f.audit();values.append(f.counts['success']-engine.counts['success'])
        result[scale]=dict(values={'actual':values})
    return result


def diagnostic_case(payload):
    task_s,method,block,workload,kind=payload;task=Path(task_s);torch=core.configure();out,_=locations(task)
    path=out/'mechanism'/'cases'/f'{kind}_{method}_{block}_{workload}.json.gz'
    if path.exists():
        d=read(path);assert d['source_contract']==contract()[0];return d
    model,checkpoint=load(task,method,block,torch);adapter=BudgetAdapter();tape=RecoveryScenario(workload,'burst','normal',.6)
    job=dict(kind='eval',method=method,block=block,checkpoint=6,profile='burst',workload=workload)
    old_out,_=parent.locations(task);target=core.result_path(out if kind=='native' else old_out,job);raw=read(target)
    analysis=read((out if kind=='native' else old_out)/'analysis.json');assert sha(target)==analysis['raw_sha256'][target.relative_to(target.parents[1]).as_posix()]
    old_diag_out,_=diagnosis.locations(task)
    frozen=None
    if kind=='matched':
        spec=dict(kind='episode',method=method,block=block,workload=workload,mode='native')
        dpath=diagnosis.case_path(old_diag_out,spec);frozen=read(dpath)
        assert sha(dpath)==read(old_diag_out/'analysis.json')['raw_sha256'][dpath.relative_to(old_diag_out).as_posix()]
    e=Engine(Protocol(),START,tape.ue_positions);values=[];records=[];exact=0
    future=mc_returns([r['success']/20 for r in raw['rows']])
    while e.t<200 or e.tasks:
        t=int(e.t);e.add(tape.slot(t));e.admit();obs=e.observe();actual=raw['rows'][t]
        plan={int(uid):node for uid,node in actual['assignment'].items()};motion=actual['motion']
        if kind=='native':
            row,p,a=model.act(obs,adapter,parent.action_seed(job,t));assert p==plan and all(abs(x-y)<1e-9 for u,v in zip(a,motion) for x,y in zip(u,v))
            values.append(dict(time=t,target=future[t],pre=row['value'],plan=row['motion_value']))
        else:adapter.prepare(obs)
        if t in (40,75,100,150):
            prefix=VirtualQueue(obs)
            for uid,node in plan.items():prefix.assign(uid,node)
            pre=adapter.motion(obs);post=adapter.motion(prefix.obs);prob,_,_=score(model,pre,post,torch)
            if kind=='native':payoff=actual_payoff(e,tape,plan,adapter.plans)
            else:
                f=next(p for p in frozen['probes'] if p['time']==t);assert f['plans']['actual']=={str(k):v for k,v in plan.items()};payoff=f['payoff']
                oldprob=f['probabilities']['6']['actual' if method=='joint' else 'pre']
            record=dict(time=t,probability=prob,entropy=probability_metrics(prob,[1/len(prob)]*len(prob))['entropy_fraction'],payoff=payoff,rank={})
            for scale,v in payoff.items():
                q=v['values']['actual'];record['rank'][scale]=rank_record(q,prob)
                if kind=='matched':record['rank'][scale]['old']=rank_record(q,oldprob)
            records.append(record)
        e.assign(plan);got=e.advance(motion)
        assert all(got[k]==actual[k] for k in ('generated','completed','expired','admitted','active','success'));exact+=1
    d=dict(source_contract=contract()[0],kind=kind,method=method,block=block,workload=workload,checkpoint_sha256=checkpoint,
        parent_raw_sha256=sha(target),values=values,probes=records,replay_steps_exact=exact,counts=e.summary()['counts'],audit=e.audit(),cpu=sorted(os.sched_getaffinity(0)))
    raw_write(path,d);return d


def experience_check(task):
    torch=core.configure();out,val=locations(task);analysis=read(out/'analysis.json');records=[]
    for method,block in itertools.product(core.METHODS,range(3)):
        for round_ in range(1,7):
            model,_=load(task,method,block,torch,round_-1);maximum=0.;states=0;source=[]
            for job in core.training_jobs(method,block,round_):
                p=core.result_path(out,job);raw=read(p);assert sha(p)==analysis['raw_sha256'][p.relative_to(out).as_posix()]
                x=task/'.tmp'/core.TASK/REVISION/'experience'/(core.job_id(job)+'.pkl.gz');assert sha(x)==raw['experience_sha256']
                rows=pickle.loads(gzip.decompress(x.read_bytes()));ret=mc_returns([r['reward'] for r in rows])
                for r,target in zip(rows,ret):assert abs(r['returns']-target)<1e-9 and abs(r['advantage']-(target-r['value']))<1e-9
                with torch.no_grad():
                    z=model.replay(rows)
                    for key,field in (('value','value'),('motion_value','motion_value'),('motion_log','motion_log'),('allocation_log','allocation_log')):
                        err=float((z[key]-torch.tensor([r[field] for r in rows])).abs().max());maximum=max(maximum,err);assert err<.003,(method,block,round_,key,err)
                states+=len(rows);source.append(dict(raw=p.relative_to(out).as_posix(),raw_sha256=sha(p),experience_sha256=sha(x)))
            records.append(dict(method=method,block=block,round=round_,episodes=8,states=states,replay_max_error=maximum,source=source))
    d=dict(status='PASS_SEPARATE_BASELINE_MC_AND_REPLAY',source_contract=contract()[0],episodes=288,batches=records)
    atomic(val/'experience_audit.json',d);return d


def mechanism(task):
    out,val=locations(task);digest=contract()[0];cpus=core_order();jobs=list(itertools.product(('native','matched'),core.METHODS,range(3),core.EVALUATION));cases=[]
    atomic(val/'mechanism_status.json',dict(status='RUNNING',source_contract=digest,total=len(jobs),started=now()))
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(jobs)),mp_context=multiprocessing.get_context('fork'),initializer=core.worker_init,initargs=(cpus,)) as pool:
        pending=[pool.submit(diagnostic_case,(str(task),m,b,w,k)) for k,m,b,w in jobs]
        for f in as_completed(pending):
            cases.append(f.result());atomic(val/'mechanism_status.json',dict(status='RUNNING',source_contract=digest,completed=len(cases),total=96,updated=now()))
    value=[];ranking=[];old_diag_out,_=diagnosis.locations(task);old=read(old_diag_out/'analysis.json')
    for method in core.METHODS:
        ds=[d for d in cases if d['kind']=='native' and d['method']==method]
        target=[x['target'] for d in ds for x in d['values']];keys=[(d['block'],x['time']) for d in ds for x in d['values']]
        for baseline in ('pre','plan'):
            fit=value_fit_by_phase(target,[x[baseline] for d in ds for x in d['values']],keys)
            parent_fit=next(v for v in old['value'] if v['method']==method)
            value.append(dict(method=method,baseline=baseline,**fit,old_within_phase_ev=parent_fit['within_phase_ev'],
                gate=fit['within_phase_ev']>0 and fit['within_phase_ev']>parent_fit['within_phase_ev']))
        for kind,scale in itertools.product(('native','matched'),('one_step','six_steps')):
            rs=[p['rank'][scale] for d in cases if d['kind']==kind and d['method']==method for p in d['probes']]
            row=dict(method=method,kind=kind,scale=scale,states=len(rs),informative=sum(r['informative'] for r in rs),
                mean_gain=statistics.mean(r['gain'] for r in rs),mean_regret=statistics.mean(r['regret'] for r in rs))
            if kind=='matched':
                row.update(old_mean_gain=statistics.mean(r['old']['gain'] for r in rs),mean_gain_over_old=statistics.mean(r['gain']-r['old']['gain'] for r in rs))
                row['gate']=row['mean_gain_over_old']>0
            ranking.append(row)
    raw_hash={p.relative_to(out).as_posix():sha(p) for p in (out/'mechanism/cases').glob('*.json.gz')};assert len(raw_hash)==96
    result=dict(status='PASS_COMPLETE_MECHANISM_AUDIT',source_contract=digest,value=value,ranking=ranking,raw_sha256=raw_hash,
        cases=96,native_trajectories=48,native_states=192,matched_states=192,completed=now(),
        boundaries='Matched original states/plans are frozen counter-inputs; native ranks use new trajectory states. Fixed candidate/future/continuation evaluation is not a global policy bound; same 3 initialization units')
    atomic(out/'mechanism/analysis.json',result);atomic(val/'mechanism_status.json',dict(status='COMPLETED',source_contract=digest,cases=96,completed=now()))
    return result


def report(task):
    out,val=locations(task);d=read(out/'analysis.json');a=read(out/'mechanism/analysis.json');deliver=task/'output'/core.TASK/'deliverables'
    lines=['# 分配前旁路、计划残差与条件价值短试','',f"完成：{now()}；开发短试，3初始化种子，复用工作负载。",'',
        '运动保留原pre评分，联合增加零初始化门控计划残差；分配使用pre价值，运动使用采样计划条件价值。critic增加合法候选的空间服务池化摘要及同时间中心化训练损失，无排序教师或预训练。业务与期限不变。','',
        '| 方法 | 初始化DSR | 更新3 | 更新6 | 相对旧末模型 | 学习门槛 |','|---|---:|---:|---:|---:|---|']
    for m in core.METHODS:
        means=[statistics.mean(lookup(d['summary'],m,b,c)['dsr'] for b in range(3)) for c in (0,3,6)]
        lines.append(f"| {m} | {100*means[0]:.4f}% | {100*means[1]:.4f}% | {100*means[2]:.4f}% | {100*statistics.mean(d['old_final_difference'][m]):+.4f}个百分点 | {d['learning_gate'][m]} |")
    lines+=['',f"联合减分解三种子差值（百分点）：{[round(100*x,5) for x in d['joint_minus_decomposed']]}。",'',
        '## 同时间价值区分','', '| 方法 | 基线 | 总体EV | 同时间EV | 旧总体基线的同时间EV | 门槛 |','|---|---|---:|---:|---:|---|']
    for r in a['value']:lines.append(f"| {r['method']} | {r['baseline']} | {r['overall_ev']:.4f} | {r['within_phase_ev']:.4f} | {r['old_within_phase_ev']:.4f} | {r['gate']} |")
    lines+=['','同时间按同初始化种子/同一时刻跨8工作负载去均值；当前回报与预测均取完整排空事实轨迹。旧指标对应旧轨迹，不是同状态因果对照；预/计划基线在新轨迹上直接配对。','',
        '## 候选排序','', '| 方法 | 状态来源 | 尺度 | 非平状态/总数 | 期望完成−均匀 | 最佳候选遗憾 | 相对旧概率增益 |','|---|---|---|---:|---:|---:|---:|']
    for r in a['ranking']:lines.append(f"| {r['method']} | {r['kind']} | {r['scale']} | {r['informative']}/{r['states']} | {r['mean_gain']:+.6f} | {r['mean_regret']:.6f} | {r.get('mean_gain_over_old',float('nan')):+.6f} |")
    lines+=['','matched使用原冻结192状态、原实际分配和原固定动作完成矩阵，只比较运动概率；存在新策略对旧状态/计划输入的分布差异。native使用新48事实轨迹的192状态，结果不能直接当同状态旧新差值。one_step为首步后悬停，six_steps为六步安全方向保持；同未来5批只离线结算，之后排空，未来未进入策略。零差异状态保留，机会和遗憾都不是长期策略上界。','',
        '## 预算与限制','',
        '每方法/种子48完整训练回合、6更新、156优化步骤；总288训练/576学习评估/96控制，另2烟雾。机制验收另96重放回合、无梯度训练。两版网络实例容量相同、分解的运动残差关闭，两版critic均可看到采样计划；额外参数、两次模型预测和诊断成本不属于相同总计算预算。末检查点主结果，不挑最好模型。','',
        '结构、空间价值特征、分支基线和价值损失同时改变，本轮仅判断组合，不归因单组件。未改变业务、资源、截止期或奖励。三种子是训练重复，工作负载/状态/请求不扩充重复。显式自审，不冒称独立确认。','',
        '- [逐格CSV](queued_plan_residual_summary.csv)',
        f'- [完整结果](../../../results/{core.TASK}/{REVISION}/analysis.json)',
        f'- [机制验收](../../../results/{core.TASK}/{REVISION}/mechanism/analysis.json)',
        f'- [冻结协议](../../../results/{core.TASK}/{REVISION}/protocol.json)',
        f'- [服务器完成](../validation/{REVISION}/completion.json)',
        f'- [分支优势及重放验收](../validation/{REVISION}/experience_audit.json)',
        f'- [测试](../validation/{REVISION}/tests.json)','',f'源码合同：`{contract()[0]}`。']
    (deliver/'queued_plan_residual_short.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('package','validate','run','analyze','mechanism','experience'))
    p.add_argument('--task',type=Path,default=REPO_ROOT);p.add_argument('--archive',type=Path,default=REPO_ROOT/'.tmp/queued_plan_residual/release.zip');args=p.parse_args();task=args.task.resolve()
    if args.command=='package':
        args.archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in SOURCE_FILES:z.write(REPO_ROOT/f,f)
        print(json.dumps(dict(source_contract=contract()[0],archive_sha256=sha(args.archive),files=len(SOURCE_FILES))))
    elif args.command=='validate':raise SystemExit(0 if validate(task) else 1)
    else:
        core.configure();activate()
        if args.command=='run':
            core.run(task);experience_check(task);mechanism(task);report(task)
            out,val=locations(task);completion=read(val/'completion.json');completion.update(status='PASS_SHORT_TRIAL_AND_MECHANISM_COMPLETED',mechanism_cases=96,completed=now())
            old=read(val/'protected_before.json');changed=[f for f,h in old.items() if not (task/f).exists() or sha(task/f)!=h];assert not changed
            completion.update(protected_files=len(old),changed_old_files=[]);atomic(val/'completion.json',completion);atomic(val/'status.json',completion)
        elif args.command=='analyze':analyze(task)
        elif args.command=='experience':experience_check(task)
        else:mechanism(task);report(task)


if __name__=='__main__':main()
