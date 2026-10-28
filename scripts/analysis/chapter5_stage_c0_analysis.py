"""Offline C0 physical labels, raw audit, block summaries and decision report."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import chapter5_stage_b as b
import chapter5_stage_c0 as c


def effect(values):
    from scipy.stats import t
    values=list(map(float,values));n=len(values);mean=statistics.mean(values)
    se=statistics.stdev(values)/math.sqrt(n);radius=float(t.ppf(.975,n-1))*se
    return dict(mean=mean,ci95=[mean-radius,mean+radius],values=values,n=n,
        interpretation='descriptive paired interval across five historical upper blocks; no formal noninferiority')


def read_gzip(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def label_worker(a):
    import numpy as np
    p=c.verify(a);os.sched_setaffinity(0,{p['cpus'][p['blocks'].index(a.block)]})
    _,runtime,fast,core,prices=b.boot(a,a.block)
    from utils.snapshot_evaluator import capture,evaluate
    records=[]
    for workload in p['workloads']:
        path=a.output/'fixed_timing'/f'{a.block}_{workload}.json.gz'
        assert b.sha(path)==b.read(path.with_suffix('.receipt.json'))['sha256']
        for state in read_gzip(path):
            sd=Path(state['source']['folder']);pub=b.read(sd/'public.json')
            env,obs=pickle.loads((sd/'observed.bin').read_bytes())
            motion=pub['outcome']['motion'];snapshot=capture(env,motion);cache={};selected={}
            source_method='G' if state['identity'][2]=='G' else 'B10'
            assert state['runs'][source_method][0]['info']['assignment']==pub['outcome']['info']['assignment']
            for method,runs in state['runs'].items():
                run=runs[0];assignment=tuple(run['info']['assignment'])
                assert np.array_equal(np.asarray(run['motion']),np.asarray(motion))
                if assignment not in cache:cache[assignment]=evaluate(snapshot,assignment)
                out=cache[assignment];requests=out['requests']
                assert [q['ue_id'] for q in requests]==run['info']['request_ids']
                successes=[sum(q['latency']+r['info']['policy_seconds']<=q['deadline'] for q in requests) for r in runs]
                seconds=[r['info']['policy_seconds'] for r in runs]
                selected[method]=dict(success_repeats=successes,success=statistics.mean(successes),
                    raw_success=sum(q['latency']<=q['deadline'] for q in requests),
                    nominal10_success=sum(q['latency']+.010<=q['deadline'] for q in requests),
                    energy=out['metrics']['uav_fleet_total_energy'],mbs=sum(q['target']=='mbs' for q in requests),
                    admitted=len(requests),seconds=seconds,mean_seconds=statistics.mean(seconds),
                    matching_calls=0 if method=='G' else run['info']['diagnostic']['matching_calls'],
                    assignment=list(assignment),requests=requests)
            src=cache[tuple(state['runs'][source_method][0]['info']['assignment'])]['requests']
            for left,right in zip(src,pub['outcome']['requests']):
                assert left['ue_id']==right['ue_id']
                assert math.isclose(left['latency'],right['latency'],rel_tol=1e-10,abs_tol=1e-10)
            for method in c.METHODS:
                for run in state['runs'][method]:
                    assert run['info']['policy_seconds']<=(run['window_ns'][1]-run['window_ns'][0])/1e9+1e-6
            records.append(dict(identity=state['identity'],generated=pub['outcome']['metrics']['generated_service_count'],
                admitted=len(pub['rows']),weight=pub['design_weight'],methods=selected,
                private_truth_used_only_after_public_decisions=True))
    assert len(records)==240
    path=a.output/'fixed_labels'/f'{a.block}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Keep previous labels')
    with gzip.open(path,'wt') as f:json.dump(records,f,separators=(',',':'),allow_nan=False)
    b.write(path.with_suffix('.receipt.json'),dict(status='PASS',states=240,sha256=b.sha(path),source_sha256=b.sha(__file__)))


def labels(a):
    p=c.verify(a);assert b.read(a.output/'fixed_complete.json')['status']=='PASS'
    def execute(block):
        cmd=[sys.executable,str(Path(__file__).resolve()),'label-worker','--archive',str(a.archive),'--development',str(a.development),
             '--output',str(a.output),'--block',str(block)]
        with (a.output/'logs'/f'label_{block}.log').open('ab') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        for f in as_completed([pool.submit(execute,x) for x in p['blocks']]):f.result()
    b.write(a.output/'labels_complete.json',dict(status='PASS',states=1200,source_sha256=b.sha(__file__)))


def closed_summary(a,p):
    import numpy as np
    groups=defaultdict(lambda:dict(totals=Counter(),times=[],calls=[],stages=defaultdict(list)))
    audit=Counter();rows=[];fingerprints={}
    for block,workload in p['bundles']:
        external=set();fps=set()
        for method in c.METHODS:
            folder=a.output/'closed_loop'/f'{block}_{workload}'/method
            receipt=b.read(folder/'result.json');assert b.sha(folder/'steps.json.gz')==receipt['steps_sha256']
            assert receipt['protocol_sha256']==b.sha(a.output/'protocol.json')
            data=read_gzip(folder/'steps.json.gz');assert len(data)==1000
            total=Counter();group=groups[block,method]
            for step,z in enumerate(data):
                assert z['step']==step
                sec=z['info']['policy_seconds'];assert z['injected_seconds']==sec
                assert sec<=(z['window_ns'][1]-z['window_ns'][0])/1e9+1e-6
                assert sum(q['latency']+sec<=q['deadline'] for q in z['requests'])==z['metrics']['deadline_satisfied_count']
                assert len(z['requests'])==z['metrics']['admitted_service_count']==len(z['info']['assignment'])
                for key,metric in b.COUNTERS.items():total[key]+=float(z['metrics'][metric])
                trace=z['public_trace'];calls=0 if method=='G' else z['info']['diagnostic']['matching_calls']
                assert calls<=c.budget(method)
                if method.startswith('B'):
                    assert [v['quota'] for v in trace['visited']]==[v['quota'] for v in trace['quota_order'][:c.budget(method)-1]]
                    assert calls==trace['initialization_calls']+sum(v['matching_call'] for v in trace['visited'])
                    assert len(trace['computed_matrix_quotas'])<=c.budget(method)-1
                    expected=set(tuple(v['quota']) for v in trace['visited'])
                    assert set(map(tuple,trace['computed_matrix_quotas']))==expected
                    for key,value in trace['stage_seconds'].items():group['stages'][key].append(value)
                    group['stages']['other_complete_path'].append(sec-sum(trace['stage_seconds'].values()))
                if method!='G':
                    d=z['info']['diagnostic'];assert d['chosen_predicted']['energy']<=d['common_energy_cap']+1e-7
                group['times'].append(sec);group['calls'].append(calls)
                audit['steps']+=1;audit['requests']+=len(z['requests'])
            for key in total:assert math.isclose(total[key],receipt['totals'][key],rel_tol=1e-12,abs_tol=1e-8)
            group['totals'].update(total);external.add(receipt['exogenous_hash']);fps.add(receipt['upper_fingerprint'])
            rows.append(dict(block=block,workload=workload,method=method,**total));audit['episodes']+=1
        assert len(external)==len(fps)==1
        fingerprint=next(iter(fps));fingerprints.setdefault(str(block),fingerprint);assert fingerprints[str(block)]==fingerprint
    assert audit['episodes']==90 and audit['steps']==90000 and len(set(fingerprints.values()))==5
    blocks={}
    for (block,method),g in groups.items():
        t=g['totals'];blocks[block,method]=dict(block=block,method=method,**t,dsr=t['success']/t['generated'],
            admission=t['admitted']/t['generated'],quality=t['success']/t['admitted'],mbs_fraction=t['mbs']/t['admitted'],
            fairness_mean=t['fairness']/3000,mean_seconds=float(np.mean(g['times'])),p95_seconds=float(np.quantile(g['times'],.95)),
            maximum_seconds=max(g['times']),mean_matching_calls=statistics.mean(g['calls']),
            matching_call_distribution=dict(Counter(g['calls'])),stage_seconds={k:statistics.mean(v) for k,v in g['stages'].items()})
    b.write(a.output/'analysis/raw_audit.json',dict(status='PASS',counts=dict(audit),distinct_upper_fingerprints=fingerprints,
        sampled_physics_replays=1200,matching_and_matrix_prefixes='checked on all recorded prefix steps',source_sha256=b.sha(__file__)))
    with (a.output/'analysis/closed_episode_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return blocks


def fixed_summary(a,p):
    import numpy as np
    groups=defaultdict(lambda:dict(totals=Counter(),times=[],calls=[]));seen=set();same=Counter()
    for block in p['blocks']:
        path=a.output/'fixed_labels'/f'{block}.json.gz';assert b.sha(path)==b.read(path.with_suffix('.receipt.json'))['sha256']
        for state in read_gzip(path):
            identity=tuple(state['identity']);assert identity not in seen;seen.add(identity)
            assert state['weight']==25
            for method,z in state['methods'].items():
                g=groups[block,method]
                g['totals'].update(dict(generated=state['generated'],success=z['success'],raw_success=z['raw_success'],
                    nominal10_success=z['nominal10_success'],admitted=z['admitted'],energy=z['energy'],mbs=z['mbs']))
                g['times'].extend(z['seconds']);g['calls'].append(z['matching_calls'])
                same[method]+=z['assignment']==state['methods']['B10']['assignment']
    assert len(seen)==1200
    blocks={}
    for (block,method),g in groups.items():
        t=g['totals'];blocks[block,method]=dict(block=block,method=method,**t,dsr=t['success']/t['generated'],
            nominal10_dsr=t['nominal10_success']/t['generated'],raw_dsr=t['raw_success']/t['generated'],
            admission=t['admitted']/t['generated'],quality=t['success']/t['admitted'],mbs_fraction=t['mbs']/t['admitted'],
            mean_seconds=float(np.mean(g['times'])),p95_seconds=float(np.quantile(g['times'],.95)),
            mean_matching_calls=statistics.mean(g['calls']))
    return blocks,dict(same)


def comparisons(blocks,p):
    output={}
    for method in c.METHODS:
        values=lambda metric:[blocks[x,method][metric] for x in p['blocks']]
        output[method]=dict(budget=c.budget(method),metrics={metric:effect(values(metric)) for metric in ['dsr','admission','quality','mbs_fraction','mean_seconds','p95_seconds','mean_matching_calls']},
            dsr_difference_pp=effect([(blocks[x,method]['dsr']-blocks[x,'B10']['dsr'])*100 for x in p['blocks']]),
            full_time_ratio=effect([blocks[x,method]['mean_seconds']/blocks[x,'B10']['mean_seconds'] for x in p['blocks']]),
            energy_ratio=effect([blocks[x,method]['energy']/blocks[x,'B10']['energy'] for x in p['blocks']]),
            energy_ratio_G=effect([blocks[x,method]['energy']/blocks[x,'G']['energy'] for x in p['blocks']]))
        if 'nominal10_dsr' in blocks[p['blocks'][0],method]:
            output[method]['nominal10_difference_pp']=effect([(blocks[x,method]['nominal10_dsr']-blocks[x,'B10']['nominal10_dsr'])*100 for x in p['blocks']])
    return output


def analyze(a):
    p=c.verify(a);assert b.read(a.output/'labels_complete.json')['status']=='PASS'
    closed=closed_summary(a,p);fixed,same=fixed_summary(a,p)
    cc=comparisons(closed,p);ff=comparisons(fixed,p);screen={}
    gates=p['exploratory_screen']
    for method in ['B2','B4','B6']:
        z=cc[method];parts=dict(quality=z['dsr_difference_pp']['mean']>=-gates['max_dsr_loss_pp'],
            time=z['full_time_ratio']['mean']<=gates['max_full_time_ratio'],
            energy=z['energy_ratio']['mean']<=gates['max_energy_ratio'],
            energy_G=z['energy_ratio_G']['mean']<=gates['max_energy_ratio_G'])
        screen[method]=dict(**parts,all_point_estimates_pass=all(parts.values()))
    passed=[method for method,z in screen.items() if z['all_point_estimates_pass']]
    best=min(passed,key=lambda m:cc[m]['metrics']['mean_seconds']['mean']) if passed else None
    stages={k:effect([closed[x,'B10']['stage_seconds'][k] for x in p['blocks']]) for k in closed[p['blocks'][0],'B10']['stage_seconds']}
    result=dict(status='COMPLETE_C0_DEVELOPMENT',protocol_sha256=b.sha(a.output/'protocol.json'),
        independent_n=5,closed_loop=cc,fixed_snapshot=ff,fixed_assignments_equal_B10=same,
        closed_B10_stage_seconds=stages,
        candidate_stage_share=effect([(closed[x,'B10']['stage_seconds']['visited_matrix_scoring']+closed[x,'B10']['stage_seconds']['conditional_matching'])/closed[x,'B10']['mean_seconds'] for x in p['blocks']]),
        screen=screen,best_screened_fixed_budget=best,
        decision='Fixed-budget development opportunity exists; prefer simplest fixed budget before any adaptive claim' if passed else 'No fixed prefix meets the predeclared joint point-estimate screen; no automatic C1 advancement',
        noninferiority_established=False,C1_started=False,new_training=False)
    b.write(a.output/'analysis/summary.json',result)
    for name,values in [('closed_block_metrics.json',closed),('fixed_block_metrics.json',fixed)]:
        b.write(a.output/'analysis'/name,list(values.values()))
    csvrows=[dict(block=block,method=method,budget=c.budget(method),dsr=value['dsr'],mean_seconds=value['mean_seconds'],energy=value['energy'],mean_matching_calls=value['mean_matching_calls']) for (block,method),value in closed.items()]
    with (a.output/'analysis/closed_block_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    report(a,p,result)
    print(json.dumps(dict(status=result['status'],screen=screen,best_fixed=best,comparisons={m:{k:v[k] for k in ['dsr_difference_pp','full_time_ratio','energy_ratio']} for m,v in cc.items()})),flush=True)


def report(a,p,result):
    def ci(z,d=3):return f"{z['mean']:+.{d}f} [{z['ci95'][0]:+.{d}f}, {z['ci95'][1]:+.{d}f}]"
    lines=['# C0：减少匹配预算是否值得（2026-10-28）','',
        '**仅完成C0固定预算开发，不启动C1、自适应规则或新训练。**',
        '',('存在固定预算的开发取舍空间，但应先验证简单固定预算，不据此主张自适应必要。' if result['best_screened_fixed_budget'] else '**本轮不支持直接推进C1：没有减少预算的方案同时达到预定质量与20%完整时间门槛。**'),'',
        '## 设计与一致性','',
        '- 复用五个历史开发基座和B阶段三个已用开发负载19271001—19271003；未称新负载、留出验证或独立确认。六方法各15个1000步回合，共90回合、90000步。',
        '- 固定预算0/2/4/6/10；0直接G，其余包括一次原二值初始化及同序列前B−1个配额。初始化解不另塞入候选池，G始终保底，失败调用计数；没有预算1。',
        '- 1200个B阶段主状态全部通过B10候选、最终动作、预测量和反馈回归；前缀配额顺序、调用数与未访问矩阵不计算均通过。动作、身份和调用数严格相同，浮点容差预定rtol=10⁻¹²、atol=10⁻¹⁰。',
        '- 五项针对性验证覆盖空批次、预算0等于G、失败匹配计数、能耗筛选保底和完整B10动作/反馈/奖励/下一状态一致性。首次全状态验证缺少反馈函数所需的公开归一化参数，仅工程检查报错，原版本和日志保留；修正后在development_v2重新冻结，未运行或筛选性能结果。',
        '- 新实现逐个生成访问配额的矩阵，确实避免未访问矩阵计算；所有前缀共享该实现。原生批量B10另测，工程实现差异不归为停止策略收益。各方法完整调用计时包含记录阶段耗时的成本；时钟值只用于记录，不参与选择。',
        '- 新15核上限、单线程、模型加载屏障、共享服务器背景负载。所有方法在本轮重新计时；不使用B阶段或旧120核耗时计算加速比。磁盘记录与环境恢复在计时外。','',
        '## 计费闭环：主要判断','',
        '| 预算/实现 | DSR % | 相对同实现B10差 pp [95%区间] | 完整均值 ms | 完整时间比 | 机队能耗比 | 能耗/G | 实际平均匹配数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in c.METHODS:
        z=result['closed_loop'][method];m=z['metrics']
        lines.append(f"| {method} | {m['dsr']['mean']*100:.4f} | {ci(z['dsr_difference_pp'])} | {m['mean_seconds']['mean']*1000:.3f} | {z['full_time_ratio']['mean']:.4f} | {z['energy_ratio']['mean']:.4f} | {z['energy_ratio_G']['mean']:.4f} | {m['mean_matching_calls']['mean']:.3f} |")
    lines+=['','先按每块三个回合汇总成功/生成计数，再在五块上比较；完整时间和能耗比先作块内比值再跨块平均。区间仅为五历史块描述性配对t区间，未建立正式非劣效；实际匹配数可能少于上限，不能用预算比例代替实测速度。接纳率、接纳后质量、MBS、块p95、公平性与尾部记录在JSON和块表。','',
        '## 固定状态：排除轨迹分化的另一项描述','',
        '| 预算/实现 | 当次完整费用DSR差 pp [95%区间] | 共同10 ms差 pp | 完整时间比 | 同B10分配状态数/1200 |','|---|---:|---:|---:|---:|']
    for method in c.METHODS:
        z=result['fixed_snapshot'][method]
        lines.append(f"| {method} | {ci(z['dsr_difference_pp'])} | {z['nominal10_difference_pp']['mean']:+.4f} | {z['full_time_ratio']['mean']:.4f} | {result['fixed_assignments_equal_B10'][method]} |")
    lines+=['','固定状态含G与解析两种源轨迹，每状态每方法三次完整调用重计时，保留所有重复而不挑最快一次。原始分配与计时先封存，再由独立私有环境副本评分；同一物理分配仅评价一次，将三个原实测费用分别用于截止期判断。主状态均代表25步区间，额外step0未使用。该结果是固定快照单步费用评分，不是另一条闭环或系统最优上界。','',
        '## 效率空间与预定开发筛选','',
        '查看结果前沿用计划建议的点估计筛选：相对同实现B10，平均DSR损失≤0.1 pp、完整均值时间比≤0.80、机队能耗比≤1.05，并保留相对G能耗比≤1.05。五块筛选不是非劣效确认，不在结果揭示后放宽目标。','',
        '| 固定预算 | 质量 | 时间 | 能耗/B10 | 能耗/G | 四项点估计共同通过 |','|---|---|---|---|---|---|']
    for method,z in result['screen'].items():
        mark=lambda k:'通过' if z[k] else '未通过'
        lines.append(f"| {method} | {mark('quality')} | {mark('time')} | {mark('energy')} | {mark('energy_G')} | {mark('all_point_estimates_pass')} |")
    lines+=['','| 同实现B10完整路径分项 | 五块均值 ms |','|---|---:|']
    for key,z in result['closed_B10_stage_seconds'].items():lines.append(f"| {key} | {z['mean']*1000:.4f} |")
    lines+=['',f"访问配额矩阵评分与条件匹配的合计耗时，占同实现B10完整调用的平均块内比例约{result['candidate_stage_share']['mean']*100:.2f}%。这项分项比例为事后描述性解释，不是新增晋级端点。公共修复/初始化、提议构造及上层/映射/逻辑消息等路径占据其余大部。",'',
        '分项用于定位公共开销，只是本实现、硬件与开发分布中的测量，不是理论速度下界。完整调用始终是时间主端点，不能仅按匹配子函数节省宣称整体加速。','',
        '## 停止/继续决定','',
        f"预定联合点估计筛选选出的固定预算：{result['best_screened_fixed_budget'] or '无'}。",
        'B4的平均DSR损失略超过0.1 pp，但节时也仅约8%，因此不能靠放宽质量界将它变为通过。B6在固定快照的原生费用评分均值略正，闭环却为负；不以有利快照子指标替代闭环端点。原生B10略快于共同实现B10，其条件动作一致；该差异属于实现和时钟费用，不是新预算策略收益。','',
        '本次停在C0，不追加C1门控、学习器或新训练。该结论不证明一切状态自适应预算无效；它限定于当前提议与前缀实现、已用开发负载及本次硬件。若未来希望直接跳过提议构造或从入口回到G，应另立问题，重新核对公开信息、完整费用和强固定预算对照，不能借本轮阴性结果降低门槛。','',
        '## 证据位置','',
        '协议、1200状态兼容记录、全部闭环gzip、逐次公开候选、固定状态三次计时与离线标签分别保留。原B阶段数据和封存发布未覆盖。',
        '本机路径：`results/chapter5_followup_20261027/c0_20261028/development_v2/`；运行源码为`scripts/analysis/chapter5_budget_prefix.py`、`chapter5_stage_c0.py`及`chapter5_stage_c0_analysis.py`。所有科研计算在100.69.44.85执行。','']
    path=a.output/'analysis/report.md';path.write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['labels','label-worker','analyze'])
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--development',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--block',type=int)
    a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Scientific analysis executes on server')
    {'labels':labels,'label-worker':label_worker,'analyze':analyze}[a.mode](a)


if __name__=='__main__':main()
