"""Explicit process-local adaptation of the frozen short-pilot scheduler."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
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
from scripts.analysis import queued_service_joint_short as core
from scripts.analysis.queued_service_validation import atomic,sha,now,core_order
from scripts.analysis.queued_service_business import read
from utils.queued_service_diagnostics import mc_returns,explained_variance
from utils.queued_service_shared_budget import BudgetAdapter,mc_credit

FROZEN_REVISION='queued_joint_short_20261031'
REVISION='queued_shared_budget_20261101'
BASE_DEFINITION=core.protocol()
assert BASE_DEFINITION['source_contract']=='f55f9b56a5144f2de962ac6a42186297ecbb08a20e671ef7ab864d3e2960ebd0'
SOURCE_FILES=core.SOURCES+('utils/queued_service_diagnostics.py','tests/test_queued_service_diagnostics.py',
    'utils/queued_service_shared_budget.py','marl_models/queued_service_shared_budget.py',
    'scripts/analysis/queued_service_shared_budget.py','tests/test_queued_service_shared_budget.py')


def contract():
    files={name:sha(REPO_ROOT/name) for name in SOURCE_FILES}
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files


def locations(task):return task/'results'/core.TASK/REVISION,task/'output'/core.TASK/'validation'/REVISION


def protocol():
    result=copy.deepcopy(BASE_DEFINITION);digest,files=contract()
    result.update(revision=REVISION,source_contract=digest,source_files=files,parent_contract=BASE_DEFINITION['source_contract'],
        motion='Same safe bank; one-second known-task shared-airtime forecasts, deadline work and contact margins; legal-candidate normalization floor .02',
        model_features='Model-assisted public features; no teacher loss, service-logit prior or future arrivals; report forecast inference cost',
        difference='Joint forecasts sampled queue prefix; decomposed forecasts pre-allocation state with explicit source-processing defaults',
        reward='unchanged newly completed /20; full finite episode; gamma1 lambda1 MC minus rollout-time pre-action critic',
        credit=dict(gamma=1.,gae_lambda=1.,terminal_value=0.,critic='pre-action public state; raw prediction scale100; normalized target loss; SiLU and remaining-time polynomial skip'),
        value_check=dict(source='old joint training experience only',blocks=3,episodes_per_block=48,train_replicas=list(range(6)),
            validation_replicas=[6,7],steps_per_architecture_block=156,minibatch=128,lr=.0003,
            architectures=['old_raw','normalized_time'],criterion='new mean validation EV>=.5 and RMSE below old; diagnostic only, no tuning loop',
            no_warmstart=True,compute_separate=True),
        budget_boundary='Same 288 actual training episodes, 6 updates per learner, two epochs and minibatch128. Extra model forecasting and independent value-check compute are not equal-total-compute claims',
        action_seed_namespace=FROZEN_REVISION,combined_change=True)
    result['optimizer']['critic_weight']=.5
    result['optimizer']['critic_loss_units']='(raw_prediction/100 - MC_return/100)^2; replay value normalized only for critic update'
    return result


def action_seed(job,step):
    label=f'{FROZEN_REVISION}/{job["kind"]}/{job["block"]}/{job["workload"]}/{step}'
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'big')%(2**63-1)


def activate():
    """Configure only this interpreter; fork workers inherit these factories."""
    import marl_models.queued_service_joint as frozen_model
    from marl_models.queued_service_shared_budget import BudgetPolicy,budget_update
    core.REVISION=REVISION;core.SOURCES=SOURCE_FILES;core.protocol=protocol
    core.LearningAdapter=BudgetAdapter;core.finite_gae=mc_credit;core.action_seed=action_seed
    frozen_model.JointPolicy=BudgetPolicy;frozen_model.update=budget_update;core.analyze=analyze


def validate(task):
    core.configure();os.environ['QUEUED_TORCH_TESTS']='1';stream=io.StringIO()
    names=['tests.test_queued_service_shared_budget','tests.test_queued_service_diagnostics','tests.test_queued_service_learning',
        'tests.test_queued_service','tests.test_queued_service_business','tests.test_queued_service_business_refinement']
    suite=unittest.defaultTestLoader.loadTestsFromNames(names);result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    _,val=locations(task)
    atomic(val/'tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,skipped=len(result.skipped),
        source_contract=contract()[0],source=str(REPO_ROOT),time=now(),log=stream.getvalue(),independent_review=False))
    print(stream.getvalue(),flush=True);return result.wasSuccessful()


def fit_value(payload):
    task_string,architecture,block=payload;task=Path(task_string);torch=core.configure()
    from marl_models.queued_service_shared_budget import BudgetPolicy,FrozenPolicy,VALUE_SCALE,tensor
    original=task/'results'/core.TASK/FROZEN_REVISION;manifest=read(original/'analysis.json')
    train_x=[];train_y=[];valid_x=[];valid_y=[];sources=[]
    for round_ in range(1,7):
        for replica,job in enumerate(core.training_jobs('joint',block,round_)):
            raw_path=core.result_path(original,job);raw=read(raw_path)
            assert sha(raw_path)==manifest['raw_sha256'][raw_path.relative_to(original).as_posix()]
            assert raw['source_contract']==BASE_DEFINITION['source_contract']
            path=task/'.tmp'/core.TASK/FROZEN_REVISION/'experience'/(core.job_id(job)+'.pkl.gz')
            assert sha(path)==raw['experience_sha256'];rows=pickle.loads(gzip.decompress(path.read_bytes()))
            targets=mc_returns([r['reward'] for r in rows]);states=[r['state'] for r in rows]
            if replica<6:train_x+=states;train_y+=targets
            else:valid_x+=states;valid_y+=targets
            sources.append(dict(job=job,raw_sha256=sha(raw_path),experience_sha256=sha(path),split='train' if replica<6 else 'validation'))
    torch.manual_seed(17001+block)
    if architecture=='old_raw':model=FrozenPolicy('joint').critic;scale=1.
    else:model=BudgetPolicy('joint').critic;scale=VALUE_SCALE
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);generator=torch.Generator().manual_seed(912000+block)
    x=tensor(train_x);y=tensor(train_y);vx=tensor(valid_x);vy=tensor(valid_y)
    def metrics(xx,yy):
        with torch.no_grad():prediction=model(xx).squeeze(-1)
        return dict(rmse=float((prediction-yy).square().mean().sqrt()),
            ev=explained_variance(yy.tolist(),prediction.tolist()),prediction_mean=float(prediction.mean()),target_mean=float(yy.mean()))
    before=metrics(vx,vy);start=time.perf_counter()
    for _ in range(156):
        index=torch.randint(len(x),(128,),generator=generator)
        loss=.5*((model(x[index]).squeeze(-1)-y[index])/scale).square().mean()
        assert bool(torch.isfinite(loss));optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);optimizer.step()
    result=dict(source_contract=contract()[0],architecture=architecture,block=block,optimizer_steps=156,
        train_episodes=36,validation_episodes=12,train_rows=len(x),validation_rows=len(vx),before=before,
        train=metrics(x,y),validation=metrics(vx,vy),seconds=time.perf_counter()-start,source=sources,
        cpu=sorted(os.sched_getaffinity(0)),no_policy_warmstart=True)
    out,_=locations(task);atomic(out/'value_check'/f'{architecture}_{block}.json',result);return result


def value_check(task):
    out,val=locations(task);digest=contract()[0];records=[];pending=[]
    for architecture,block in itertools.product(('old_raw','normalized_time'),range(3)):
        path=out/'value_check'/f'{architecture}_{block}.json'
        if path.exists():
            d=read(path);assert d['source_contract']==digest;records.append(d)
        else:pending.append((str(task),architecture,block))
    if pending:
        with ProcessPoolExecutor(max_workers=len(pending),mp_context=multiprocessing.get_context('fork'),
                initializer=core.worker_init,initargs=(core_order(),)) as pool:records+=list(pool.map(fit_value,pending))
    summary={}
    for architecture in ('old_raw','normalized_time'):
        ds=[r for r in records if r['architecture']==architecture]
        summary[architecture]=dict(mean_validation_ev=statistics.mean(d['validation']['ev'] for d in ds),
            mean_validation_rmse=statistics.mean(d['validation']['rmse'] for d in ds),
            by_seed=[dict(block=d['block'],**d['validation']) for d in sorted(ds,key=lambda x:x['block'])])
    passed=summary['normalized_time']['mean_validation_ev']>=.5 and summary['normalized_time']['mean_validation_rmse']<summary['old_raw']['mean_validation_rmse']
    result=dict(status='PASS_VALUE_FIT_CHECK' if passed else 'VALUE_FIT_GATE_NOT_MET',source_contract=digest,summary=summary,
        cases=6,steps_per_case=156,new_interactions=0,no_policy_warmstart=True,automatic_tuning=False)
    atomic(out/'value_check/analysis.json',result);atomic(val/'value_check.json',result)
    print(json.dumps(dict(stage='value_check_complete',**result)),flush=True);return result


def analyze(task):
    out,val=locations(task);digest=contract()[0];groups={};hashes={};fingerprints={}
    jobs=[j for m in core.METHODS for b in range(3) for cp in (0,3,6) for j in core.evaluation_jobs(m,b,cp)]
    jobs += [j for m in core.CONTROLS for j in core.evaluation_jobs(m)]
    jobs += [j for m in core.METHODS for b in range(3) for r in range(1,7) for j in core.training_jobs(m,b,r)]
    assert len(jobs)==960
    for job in jobs:
        path=core.result_path(out,job);d=read(path);assert d['source_contract']==digest and d['job']==job and d['audit']['status']=='PASS'
        c=d['summary']['counts'];assert c['generated']==c.get('success',0)+c.get('expired',0)
        assert d['rows'][-1]['active']==0 and sum(r['success'] for r in d['rows'])==c.get('success',0)
        assert sum(d['summary']['expiration_stage'].values())==c.get('expired',0)
        key=(job['profile'],job['workload']);assert key not in fingerprints or fingerprints[key]==d['arrival_fingerprint'];fingerprints[key]=d['arrival_fingerprint']
        if job['method'] in core.METHODS:assert d['checkpoint_sha256']==sha(core.model_path(out,job['method'],job['block'],job['checkpoint']))
        hashes[path.relative_to(out).as_posix()]=sha(path)
        if job['kind']=='eval':groups.setdefault((job['profile'],job['method'],job['block'],job['checkpoint']),[]).append(d)
    summary=[dict(profile=p,method=m,block=b,checkpoint=cp,**core.aggregate(ds)) for (p,m,b,cp),ds in sorted(groups.items())]
    old_out=task/'results'/core.TASK/FROZEN_REVISION;old=read(old_out/'analysis.json');comparisons=[];updates={}
    for r in summary:
        assert math.isclose(r['dsr'],r['admission']*r['conditional'],abs_tol=1e-12)
        previous=next(x for x in old['summary'] if all(x[k]==r[k] for k in ('profile','method','block','checkpoint')))
        assert r['generated']==previous['generated']
        comparisons.append(dict(profile=r['profile'],method=r['method'],block=r['block'],checkpoint=r['checkpoint'],delta_dsr=r['dsr']-previous['dsr']))
    for m,b in itertools.product(core.METHODS,range(3)):
        records=[read(out/'training'/str(b)/m/f'{cp:04}.json') for cp in range(1,7)]
        assert all(r['source_contract']==digest and r['checkpoint']==i for i,r in enumerate(records,1))
        assert sum(r['episodes'] for r in records)==48 and sum(r['update']['optimizer_steps'] for r in records)==156
        updates[f'{m}/{b}']=dict(actor_steps=sum(r['update']['actor_steps'] for r in records),optimizer_steps=156,
            allocation_parameter_change=sum(r['update']['allocation_parameter_change'] for r in records),motion_parameter_change=sum(r['update']['motion_parameter_change'] for r in records),
            critic_ev_mc_by_update=[r['update']['critic_ev_mc_after'] for r in records],critic_rmse_raw_by_update=[r['update']['critic_rmse_raw_after'] for r in records])
    def lookup(method,b,cp):return next(x for x in summary if x['profile']=='burst' and x['method']==method and x['block']==b and x['checkpoint']==cp)
    gains={m:[lookup(m,b,6)['dsr']-lookup(m,b,0)['dsr'] for b in range(3)] for m in core.METHODS}
    joint_delta=[lookup('joint',b,6)['dsr']-lookup('decomposed',b,6)['dsr'] for b in range(3)]
    gate={m:all(x>0 for x in values) and statistics.mean(values)>=.01 for m,values in gains.items()}
    result=dict(status='PASS_RAW_EXPERIMENT_AUDIT',source_contract=digest,parent_contract=BASE_DEFINITION['source_contract'],
        summary=summary,updates=updates,gain_over_initialization=gains,joint_minus_decomposed=joint_delta,
        old_protocol_comparison=comparisons,learning_gate=gate,value_check=read(out/'value_check/analysis.json'),
        raw_sha256=hashes,completed=now(),independent_confirmation=False,automatic_expansion=False,
        limitations='Combined service features and normalized MC value fix; reused development data; same actual rollout/optimizer budget, extra model forecasting and offline fit compute separately reported')
    atomic(out/'analysis.json',result)
    import csv
    deliver=task/'output'/core.TASK/'deliverables';deliver.mkdir(parents=True,exist_ok=True)
    with (deliver/'queued_shared_budget_summary.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    lines=['# 共享带宽服务特征与价值修复短试','',f"完成：{result['completed']}。开发短试，无独立确认。",'',
        '真实环境和业务不变，安全轨迹银行和分配分支不变。运动使用一秒公开队列服务预测及接触余量，遵守共享带宽；候选内归一化。价值网络固定100倍输出尺度，完整回合MC优势，无额外策略预训练。', '',
        '## 同预算结果','', '| 方法 | 初始化DSR | 更新3 | 更新6 | 相对初始化 | 相对旧版末检查点 | 学习门槛 |', '|---|---:|---:|---:|---:|---:|---|']
    for m in core.METHODS:
        means=[statistics.mean(lookup(m,b,cp)['dsr'] for b in range(3)) for cp in (0,3,6)]
        delta=statistics.mean(x['delta_dsr'] for x in comparisons if x['profile']=='burst' and x['method']==m and x['checkpoint']==6)
        lines.append(f'| {m} | {100*means[0]:.3f}% | {100*means[1]:.3f}% | {100*means[2]:.3f}% | {100*statistics.mean(gains[m]):+.3f}个百分点 | {100*delta:+.3f}个百分点 | {gate[m]} |')
    lines+=['',f'联合减分解的三个种子差值（百分点）：{[round(100*v,4) for v in joint_delta]}。','',
        '## 离线价值拟合检查','', '| 价值网络 | 验证EV均值 | 验证RMSE均值 |', '|---|---:|---:|']
    for name,r in result['value_check']['summary'].items():lines.append(f"| {name} | {r['mean_validation_ev']:.4f} | {r['mean_validation_rmse']:.3f} |")
    lines+=['',f"预定拟合门槛：{result['value_check']['status']}。每架构/种子156步，仅旧joint训练经验，按整回合切分36训练/12验证；不使用评估数据，不把拟合权重转给正式策略。",'',
        '## 预算和证据范围','',
        '每个方法/种子48条完整回合、6次更新、156个优化步骤；总288训练、576学习评估和96控制评估，另2烟雾回合。与旧短试复用同一开发任务、种子、动作随机流和末检查点口径，不挑最好检查点。离线价值检查另有936个critic拟合步骤、零新交互；一秒服务预测增加推理计算，不能称总计算量相同。完整决策耗时见CSV。','',
        '同时更改特征、归一化及MC价值拟合，因此本轮只评估组合修复，不证明单组件因果或分布式MAPPO优势。模型生成服务特征，不使用未来到达或教师损失；分解方法在未分配新任务的预测中默认来源计算。统计单位为3个初始化种子，工作负载和请求不充作训练重复。显式自审。','',
        '- [逐格CSV](queued_shared_budget_summary.csv)',
        f'- [完整分析](../../../results/{core.TASK}/{REVISION}/analysis.json)',
        f'- [冻结协议](../../../results/{core.TASK}/{REVISION}/protocol.json)',
        f'- [服务器完成记录](../validation/{REVISION}/completion.json)',
        f'- [服务器测试](../validation/{REVISION}/tests.json)','',f'源码合同：`{digest}`。']
    (deliver/'queued_shared_budget_short.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('package','validate','run','analyze','value_check'))
    p.add_argument('--task',type=Path,default=REPO_ROOT);p.add_argument('--archive',type=Path,default=REPO_ROOT/'.tmp/queued_shared_budget/release.zip')
    args=p.parse_args();task=args.task.resolve()
    if args.command=='package':
        args.archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.archive,'w',zipfile.ZIP_DEFLATED) as z:
            for name in SOURCE_FILES:z.write(REPO_ROOT/name,name)
        print(json.dumps(dict(source_contract=contract()[0],archive_sha256=sha(args.archive),files=len(SOURCE_FILES))))
    elif args.command=='validate':raise SystemExit(0 if validate(task) else 1)
    else:
        core.configure();activate()
        assert multiprocessing.get_start_method()=='fork'
        if args.command=='run':value_check(task);core.run(task)
        elif args.command=='value_check':value_check(task)
        else:analyze(task)


if __name__=='__main__':main()
