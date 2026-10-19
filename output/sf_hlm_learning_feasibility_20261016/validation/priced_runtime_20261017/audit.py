"""Reconstruct both native timing rounds; logging_v2 is predeclared primary."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy import stats

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(*p):return int.from_bytes(hashlib.sha256(json.dumps(p,separators=(',',':')).encode()).digest()[:4],'big')
def close(x,y):
    if isinstance(x,dict):
        assert set(x)==set(y),(set(x),set(y))
        for k in x:close(x[k],y[k])
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        for a,b in zip(x,y):close(a,b)
    elif isinstance(x,(int,float,np.number)):assert np.isclose(x,y,rtol=1e-10,atol=1e-12),(x,y)
    else:assert x==y,(x,y)
def paired(values):
    mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(len(values));d=float(stats.t.ppf(.975,len(values)-1))*se
    return dict(mean=mean,ci95=[mean-d,mean+d],values=values,n=len(values),positive=sum(v>0 for v in values),exact_signflip_p=sum(abs(statistics.mean(v*s for v,s in zip(values,signs)))>=abs(mean)-1e-15 for signs in itertools.product((-1,1),repeat=len(values)))/2**len(values))
def quantile(values):
    x=sorted(values);where=(len(x)-1)*.95;lo=int(math.floor(where));hi=int(math.ceil(where));return x[lo]+(where-lo)*(x[hi]-x[lo])
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

ap=argparse.ArgumentParser()
for key in ('release','prior_release','results','prior_results','validation','report','feasibility_release'):
    ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
a=ap.parse_args();v=a.validation;profile=read(v/'resource_snapshot.json');cpus=set(profile['isolated']['cpus'])
assert not profile['shared_overlap_reserved']

def audit_round(root,release,strict):
    p=read(root/'protocol.json');assert not p['smoke'] and read(root/'status.json')['status']=='COMPLETED'
    for file,digest in p['files'].items():assert sha(release/file)==digest
    for file,digest in p['input_files'].items():assert sha(file)==digest
    assert sha(a.feasibility_release/'utils/sf_hlm_feasibility.py')==p['files']['utils/sf_hlm_feasibility.py']
    methods=p['methods'];base=methods[:-2];nonlearning=p['nonlearning'];states=[];intervals={c:[] for c in cpus};samples=0;instances=0;feature_checks=0
    for si,identity in enumerate(p['identities']):
        folder=root/'tasks'/('timed_'+'_'.join(map(str,identity)));receipt=read(folder/'result.json');b=read(folder/'benchmark.json')
        assert receipt['status']=='PASS' and receipt['identity']==identity and receipt['protocol_sha256']==sha(root/'protocol.json') and receipt['benchmark_sha256']==sha(folder/'benchmark.json')
        assert b['identity']==identity and b['timing_samples']==57 and len(b['results'])==19
        assert len(b['cpu'])==1 and b['cpu'][0] in cpus;cpu=b['cpu'][0]
        record=next(z for z in p['states'] if z['identity']==identity);assert b['state_sha256']==record['state_sha256']
        assert sha(b['reference_price_labels'])==b['reference_price_labels_sha256'];assert sha(b['reference_original'])==b['reference_original_sha256']
        labels=read(b['reference_price_labels']);original=read(b['reference_original']);pub=read(Path(record['price_folder'])/'public.json');matching=read(Path(record['matching_folder'])/'public.json')
        assert b['model_fingerprint']==pub['model_fingerprint'] and b['hgb_sha256']==matching['model_sha256']
        assert labels['generated']==b['generated'] and labels['admitted']==b['admitted']
        specs=[dict(method=name,repeat=0,kind='native_G' if name=='G' else name,cap=0 if name=='G' else 10,seed=seed('sf_hlm_feasibility_20261016',identity,0)) for name in base]
        specs += [dict(method=name,repeat=j,kind='random',cap=int(name[1:]),seed=seed('sf_hlm_feasibility_20261016',identity,j)) for name in ('R8','R16') for j in range(5)]
        indices={(z['method'],z['repeat']):j for j,z in enumerate(specs)};rows={};planned=[];observed=[]
        for iteration in range(3):planned += [[iteration,int(k)] for k in np.random.default_rng(seed('priced_native_timing_order',identity,iteration)).permutation(19)]
        if strict:assert b['timing_order']==planned and b['observation_timestamps_include_restore_and_validation'] is True
        assert [indices[z['method'],z['repeat']] for z in b['results']]==[k for it,k in planned if it==0]
        old_refs={(z['method'],z['repeat']):z for z in original['results']}
        for q in b['results']:
            key=q['method'],q['repeat'];index=indices[key];spec=specs[index]
            for name,value in spec.items():assert q[name]==value
            assert q['timing_iteration']==[0,1,2] and q['selected_order']==[index]*3 and len(q['software_seconds'])==3
            assert all(math.isfinite(t) and t>0 for t in q['software_seconds'])
            if q['method'] in base:
                ref=labels['outcomes'][labels['methods'][q['method']]];expected=pub['selected'][q['method']]
                close(q['diagnostic']['chosen_predicted'],expected['predicted'])
                if b['admitted']:assert q['diagnostic']['matching_calls']==expected['matching_calls']
                if q['method']!='G':assert q['diagnostic']['selected_action']==expected['action']
            else:
                ref=old_refs[key];close(q['diagnostic']['chosen_predicted'],ref['diagnostic']['chosen_predicted']);assert q['diagnostic']['matching_calls']==ref['diagnostic']['matching_calls']
            assert q['assignment']==ref['assignment'];close(q['metrics'],ref['metrics'])
            actual=[sum(z['latency']+t<=z['deadline'] for z in ref['requests']) for t in q['software_seconds']]
            close(q['fee_success'],statistics.mean(actual));assert q['raw_success']==sum(z['latency']<=z['deadline'] for z in ref['requests'])
            assert q['logical_upload_bytes']>=0 and q['logical_download_bytes']>=0
            if strict:
                assert len(q['execution_rank'])==3 and len(q['call_observation_ns'])==3
                for iteration,(rank,span,t) in enumerate(zip(q['execution_rank'],q['call_observation_ns'],q['software_seconds'])):
                    expected_order=[k for it,k in planned if it==iteration];assert rank==expected_order.index(index)
                    start,stop=span;assert isinstance(start,int) and isinstance(stop,int) and stop>start and t<=(stop-start)/1e9+1e-7
                    observed.append((iteration,rank,start,stop));intervals[cpu].append((start,stop,identity,q['method'],iteration))
            if key in rows:raise AssertionError('Duplicate timing branch')
            rows[key]=q;samples+=3;instances+=1
        assert set(rows)==set(indices)
        if strict:
            ordered=sorted(observed)
            assert all(left[3]<=right[2] for left,right in zip(ordered,ordered[1:]))
        check=b['compatibility'];assert check['initialization_equal'] and check['feature_quota_checks']==check['probability_quota_checks']==len(matching['quotas']);feature_checks+=check['feature_quota_checks']
        state_methods={}
        for name in methods:
            selected=[q for key,q in rows.items() if key[0]==name]
            state_methods[name]=dict(raw=statistics.mean(q['raw_success'] for q in selected),fee=statistics.mean(q['fee_success'] for q in selected),
                energy=statistics.mean(q['metrics']['uav_fleet_total_energy'] for q in selected),mbs=statistics.mean(q['metrics']['service_offloads_mbs'] for q in selected),fairness=statistics.mean(q['metrics']['fairness'] for q in selected),
                calls=statistics.mean(q['diagnostic']['matching_calls'] for q in selected),times=[t for q in selected for t in q['software_seconds']])
        states.append(dict(identity=identity,generated=b['generated'],admitted=b['admitted'],methods=state_methods))
        if si%40==0:write(v/'audit_progress.json',dict(round='primary_v2' if strict else 'technical_v1',states=si+1,total=400))
    if strict:
        for cpu,spans in intervals.items():
            spans.sort()
            assert all(left[1]<=right[0] for left,right in zip(spans,spans[1:])),('Same CPU call intervals overlap',cpu)
    close(states,read(root/'analysis/states.json'));s=read(root/'analysis/summary.json')
    for source in ('all','greedy','learned'):
        subset=[z for z in states if source=='all' or z['identity'][0]==source];blocks={}
        for block in sorted({z['identity'][1] for z in subset}):
            rr=[z for z in subset if z['identity'][1]==block];den=sum(z['generated'] for z in rr);admitted=sum(z['admitted'] for z in rr);methods_block={}
            for name in methods:
                ts=[t for z in rr for t in z['methods'][name]['times']]
                methods_block[name]=dict(raw_rate=sum(z['methods'][name]['raw'] for z in rr)/den,fee_rate=sum(z['methods'][name]['fee'] for z in rr)/den,energy=sum(z['methods'][name]['energy'] for z in rr),mbs_share=sum(z['methods'][name]['mbs'] for z in rr)/max(admitted,1),
                    fairness=statistics.mean(z['methods'][name]['fairness'] for z in rr),calls=statistics.mean(z['methods'][name]['calls'] for z in rr),mean_seconds=statistics.mean(ts),p95_seconds=quantile(ts),max_seconds=max(ts))
            blocks[str(block)]=dict(generated=den,admitted=admitted,methods=methods_block)
        saved=s['subsets'][source];close(blocks,saved['blocks'])
        for name,value in saved['comparisons'].items():
            left,right=name.split('-')
            for metric,statistic in value.items():close(paired([z['methods'][left][metric+'_rate']-z['methods'][right][metric+'_rate'] for z in blocks.values()]),statistic)
        groups={}
        for name in methods:
            ts=[t for z in subset for t in z['methods'][name]['times']]
            groups[name]={key:statistics.mean(z['methods'][name][key] for z in blocks.values()) for key in ('raw_rate','fee_rate','mbs_share','fairness','calls')}
            groups[name].update(energy_ratio_to_G=statistics.mean(z['methods'][name]['energy']/z['methods']['G']['energy'] for z in blocks.values()),pooled_mean_seconds=statistics.mean(ts),pooled_p95_seconds=quantile(ts),max_seconds=max(ts))
        close(groups,saved['groups'])
    main=s['subsets']['all'];best=max(nonlearning,key=lambda n:main['groups'][n]['fee_rate']);gain=main['comparisons']['probability_price-G']['fee'];inc=main['comparisons']['probability_price-'+best]['fee'] if best!='G' else gain
    energy=main['groups']['probability_price']['energy_ratio_to_G'];time_ok=all(z['methods']['probability_price']['p95_seconds']<1 for z in main['blocks'].values())
    assert s['decision']['native_development_signal']==(gain['mean']>=.005 and gain['positive']>=4 and inc['mean']>=.001 and energy<=1.05 and time_ok)
    assert samples==22800 and instances==7600 and s['timing_samples']==samples and s['branch_instances']==instances and s['feature_quota_checks']==feature_checks
    return s,dict(status='PASS',states=400,branch_instances=instances,timing_samples=samples,feature_quota_checks=feature_checks,
        protocol_sha256=sha(root/'protocol.json'),summary_sha256=sha(root/'analysis/summary.json'),full_order_and_clock_intervals_verified=strict,within_cpu_intervals_do_not_overlap=strict)

primary,primary_audit=audit_round(a.results,a.release,True);prior,prior_audit=audit_round(a.prior_results,a.prior_release,False)
for source in ('all','greedy','learned'):
    for name,group in primary['subsets'][source]['groups'].items():
        for key in ('raw_rate','energy_ratio_to_G','mbs_share','fairness','calls'):close(group[key],prior['subsets'][source]['groups'][name][key])
out=dict(status='PASS',primary=primary_audit,prior_technical_repeat=prior_audit,historical_training_blocks=5,crossfit_lower_training_overlaps=True,goal_complete=False,
    audit_type='primary-agent self-audit; raw per-call thresholding, sample population, observed order/clocks and isolation, alternate block statistics')
write(v/'raw_record_audit.json',out);main=primary['subsets']['all'];decision=primary['decision'];g=decision['gain_vs_G'];inc=decision['learning_increment'];old=prior['decision']
lines=['**价格匹配的原生完整软件费用：'+('通过开发门槛，继续闭环确认' if decision['native_development_signal'] else '未通过开发门槛')+'**','',
    '主轮次为效果揭示前固定的logging_development_v2：400个复用状态、7,600分支实例、22,800次实际完整软件时间。每次包括新观测、公共修复、动态特征、模型/匹配/价格、上层、反馈和JSON序列化；模型预加载、恢复快照和物理评分不计在线费用。实际网络信令仍未测量。','',
    f"学习计实测费用相对G {g['mean']*100:+.3f} pp（95%五训练块t区间[{g['ci95'][0]*100:+.3f},{g['ci95'][1]*100:+.3f}]，{g['positive']}/5块正向）；相对最好预定非学习{decision['best_nonlearning']} {inc['mean']*100:+.3f} pp（[{inc['ci95'][0]*100:+.3f},{inc['ci95'][1]*100:+.3f}]）。真实机队能耗比G {decision['energy_ratio_to_G']:.4f}。",'',
    '| 方法 | 原始成功率/% | 实测费用后/% | 平均/ms | 逐次池化p95/ms | 匹配调用 | 能耗比G |','|---|---:|---:|---:|---:|---:|---:|']
for name,z in main['groups'].items():lines.append(f"| {name} | {z['raw_rate']*100:.3f} | {z['fee_rate']*100:.3f} | {z['pooled_mean_seconds']*1000:.3f} | {z['pooled_p95_seconds']*1000:.3f} | {z['calls']:.3f} | {z['energy_ratio_to_G']:.4f} |")
lines+=['','| 方法 | MBS/已接纳请求/% | 快照公平性均值 |','|---|---:|---:|']
for name,z in main['groups'].items():lines.append(f"| {name} | {z['mbs_share']*100:.3f} | {z['fairness']:.4f} |")
lines+=['',
    'p95来自实际逐次时间分布，另在JSON保留每训练块p95/最大时间；不是先算块均值再取p95。每个时间先对逐请求截止期评分，随后平均三次；随机再平均五种子，没有挑最快时间或最好随机实现。状态及计时重复不增加独立训练n。','',
    '**首轮完整披露。** development_v1遗漏完整调用顺序的持久化，日志修复和缓存生命周期对齐在读取其完整费用效果前固定；其全部22,800时间保留，不作为主轮次选择依据。',
    f"首轮学习−G {old['gain_vs_G']['mean']*100:+.3f} pp，学习−首轮最好{old['best_nonlearning']} {old['learning_increment']['mean']*100:+.3f} pp；首轮门槛{'通过' if old['native_development_signal'] else '未过'}。修正版逐状态适配器持续存在，上一调用缓存清理在policy计时开头实际发生；所有方法同规则。模型、价格、全部原始分配/能耗与首轮一致。",'',
    '**复核。** 两轮全部45,600逐次时间、15,200分支实例与原始deadline重新评分，复核模型/原始分配、随机种子与次数、块级指标/区间/闸门。主轮完整打散序列、执行rank和monotonic观测区间一致，同CPU调用区间无重叠；隔离核心与共享池无交集。批量41维特征和概率逐位核对，所有正式分配均与冻结物理参考一致。主代理自审，不称独立代理审查。','',
    '**范围。** 当前仍是五历史训练块、400指定状态上的开发反事实；软件时间按每次调用加入deadline评分，未回写状态，未验证真实信令或部署队列。下层跨折模型共享训练数据，25折及五块区间不构成独立下层训练确认。五块全正的精确双侧符号翻转p仍为0.0625；t区间仅为探索描述，不作为正式确认的名义覆盖保证。即使通过，也仍须新负载完整闭环、独立训练确认及预定比较，不能称目标已完成或PPO收益。','',
    '可复核：[主协议](../../../results/sf_hlm_learning_feasibility_20261016/priced_runtime_20261017/logging_development_v2/protocol.json)、[主统计](../../../results/sf_hlm_learning_feasibility_20261016/priced_runtime_20261017/logging_development_v2/analysis/summary.json)、[首轮统计](../../../results/sf_hlm_learning_feasibility_20261016/priced_runtime_20261017/development_v1/analysis/summary.json)、[逐次复算验收](../validation/priced_runtime_20261017/raw_record_audit.json)。','',
    '原论文、旧确认和历史失败结果不改写；既存根目录`修改总结报告.md`白名单问题未处理。持续目标保持活跃。']
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')
receipt=dict(status='COMPLETED_AND_VALIDATED',goal_complete=False,decision=decision,files={str(path):sha(path) for path in (a.results/'protocol.json',a.results/'analysis/summary.json',a.prior_results/'analysis/summary.json',v/'raw_record_audit.json',a.report)})
write(v/'completion.json',receipt);print(json.dumps(out));print(json.dumps(decision))
