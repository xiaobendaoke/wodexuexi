"""Remote replay of public conditional optima and alternate raw-score audit.

Uses the original unaccelerated assessment, explicit cost matrices/Hungarian,
serialized OOF models, and a separate raw outcome/statistical reconstruction.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def close(x,y):
    if isinstance(x,dict):
        assert set(x)==set(y),(set(x),set(y))
        for k in x:close(x[k],y[k])
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        for a,b in zip(x,y):close(a,b)
    elif isinstance(x,(int,float,np.number)):assert np.isclose(x,y,rtol=1e-10,atol=1e-10),(x,y)
    else:assert x==y,(x,y)
def paired(x):
    x=np.asarray(x,dtype=float);mean=float(x.mean());d=float(stats.t.ppf(.975,len(x)-1)*stats.sem(x))
    return dict(mean=mean,ci95=[mean-d,mean+d],values=x.tolist(),n=len(x),positive=int((x>0).sum()),exact_signflip_p=sum(abs(np.mean(x*np.asarray(s)))>=abs(mean)-1e-15 for s in itertools.product((-1,1),repeat=len(x)))/2**len(x))
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

ap=argparse.ArgumentParser()
for key in ('release','training_release','training','risk_results','teacher_results','results','validation','report'):
    ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
ap.add_argument('--label',required=True);a=ap.parse_args();r=a.results;v=a.validation;p=read(r/'protocol.json');tp=read(a.training/'protocol.json')
sys.path.insert(0,str(a.release));cm=load('audited_calibrated_edges',a.release/'utils/sf_hlm_calibrated_matching.py');pool=load('audited_calibrated_pool',a.release/'utils/sf_hlm_request_risk.py');rank=load('audited_calibrated_proposals',a.release/'utils/sf_hlm_action_ranker.py')
from utils.deadline_repair import assess
from utils.sf_hlm_matching import allocation_counts
for file,digest in p['files'].items():assert sha(a.release/file)==digest
for file,digest in tp['files'].items():assert sha(a.training_release/file)==digest
assert p['training_protocol_sha256']==sha(a.training/'protocol.json') and p['training_completion_sha256']==sha(a.training/'completion.json')
for file,digest in read(a.training/'completion.json')['files'].items():assert sha(file)==digest
arc=np.load(a.training/'training_data.npz');data={k:arc[k] for k in arc.files};old=np.load(a.risk_results/'training_data.npz');old_data={k:old[k] for k in old.files}
assert np.array_equal(data['x'],old_data['x'][:,cm.LOCAL_COLUMNS]);assert np.array_equal(data['y'],old_data['y_actual'])
assert np.array_equal(data['weight'],old_data['weight']) and np.array_equal(data['state_index'],old_data['state_index'])
identities=tp['identities'];models={};checked_models=0;refitted=0
from sklearn.ensemble import HistGradientBoostingClassifier
for block in sorted({i[1] for i in identities}):
    for workload in sorted({i[2] for i in identities}):
        folder=a.training/'folds'/f'block_{block}_workload_{workload}';rec=read(folder/'result.json');bundle=pickle.loads((folder/'model.pkl').read_bytes());fold=bundle['fold']
        train=[i for i,z in enumerate(identities) if z[1]!=block and z[2]!=workload];test=[i for i,z in enumerate(identities) if z[1]==block and z[2]==workload]
        assert fold['train']==train and fold['test']==test and len(train)==256 and len(test)==16 and rec['fold']==fold
        assert bundle['protocol_sha256']==sha(a.training/'protocol.json') and bundle['schema']==cm.SCHEMA
        ix=np.flatnonzero(np.isin(data['state_index'],test));model=bundle['model'];proba=model.predict_proba(data['x'][ix])[:,list(model.classes_).index(1.)];saved=np.load(folder/'test_probabilities.npz')
        assert np.array_equal(ix,saved['row_indices']) and np.array_equal(proba,saved['probability'])
        if (block,workload)==(620001,780001):
            mask=np.isin(data['state_index'],train);again=HistGradientBoostingClassifier(**tp['hgb']).fit(data['x'][mask],data['y'][mask],sample_weight=data['weight'][mask])
            assert np.array_equal(proba,again.predict_proba(data['x'][ix])[:,list(again.classes_).index(1.)]);refitted+=1
        models[block,workload]=model;checked_models+=1

def independent_optimum(rows,q,cost,objective,model):
    xx,index,problem,q=cm.edge_features(rows,q,cost)
    if not rows:return dict(assignment=[],counts=list(q),predicted=assess([],()),expected_success=0.)
    load=np.maximum(q,1);load[-1]=1;lat=problem['base']+problem['work']*load[None,:];energy=problem['energy']+problem['compute_energy']/load[None,:]**2
    scale=max(float(energy[problem['legal']].max(initial=0.)),np.finfo(float).tiny)
    if objective=='probability':
        edge=model.predict_proba(xx)[:,list(model.classes_).index(1.)];prob=np.zeros_like(lat);prob[index[:,0],index[:,1]]=edge;costs=1-prob+1e-9*energy/scale
    elif objective=='binary':prob=(lat+cost/1000<=problem['deadline'][:,None]).astype(float);costs=(len(rows)+1)*(1-prob)+energy/scale
    else:
        z=np.clip(((lat+cost/1000)/problem['deadline'][:,None]-1)/.1,-60,60);prob=1/(1+np.exp(z));costs=1-prob+1e-9*energy/scale
    costs[~problem['legal']]=np.inf;slots=np.repeat(np.arange(6),q)
    try:ri,ci=linear_sum_assignment(costs[:,slots])
    except ValueError as error:
        if 'infeasible' in str(error):return None
        raise
    target=slots[ci];assignment=[int(k) for k in problem['action_ids'][ri,target]]
    assert np.array_equal(allocation_counts(rows,assignment,5),q)
    return dict(assignment=assignment,counts=list(q),predicted=assess(rows,assignment),expected_success=float(prob[ri,target].sum()))

methods=p['methods'];states=[];optima=0
for si,identity in enumerate(p['identities']):
    folder=r/'tasks'/('matched_'+'_'.join(map(str,identity)));receipt=read(folder/'result.json');pub=read(folder/'public.json');labels=read(folder/'labels.json')
    assert receipt['status']=='PASS' and receipt['identity']==identity and receipt['protocol_sha256']==sha(r/'protocol.json')
    for file,digest in receipt['files'].items():assert sha(folder/file)==digest
    assert labels['public_sha256']==sha(folder/'public.json') and labels['identity']==identity and pub['identity']==identity
    source=a.teacher_results/'tasks'/('oracle_'+'_'.join(map(str,identity)));original=read(source/'public.json');safe=rank.public_payload(original);rows=safe['rows'];ids,fx=rank.features(safe,10.)
    proposals=pool.proposal_actions(ids,fx);quotas=cm.quota_set(safe['initial_counts'],proposals);close(pub['proposals'],proposals);close(pub['quotas'],quotas)
    assert pub['public_parent_sha256']==sha(source/'public.json')
    model_path=a.training/'folds'/f'block_{identity[1]}_workload_{identity[2]}'/'model.pkl';assert pub['model_sha256']==sha(model_path);model=models[identity[1],identity[2]]
    gd=assess(rows,safe['greedy_assignment']);cap=min(safe['energy_cap'],gd['energy']*1.05) if p.get('g_anchored_energy',False) else safe['energy_cap'];close(pub['energy_cap'],cap)
    close(pub['selected']['G']['assignment'],safe['greedy_assignment']);close(pub['selected']['G']['predicted'],gd)
    original_pool=pool.unique_pool(original['candidates'],proposals);scores=[sum(t+.010<=row['deadline'] for t,row in zip(c['predicted']['latencies'],rows)) for c in original_pool]
    old_action=max(sorted(zip([c['action'] for c in original_pool],scores)),key=lambda q:q[1])[0];old_choice=next(c for c in original_pool if c['action']==old_action)
    assert pub['selected']['old_pool_fee']['action']==old_action;close(pub['selected']['old_pool_fee']['assignment'],old_choice['assignment'])
    for objective in ('binary','smooth','probability'):
        saved=pub['candidates'][objective];assert len(saved)==len(quotas)+1 and saved[0]['action']==-1
        score=cm.score_assignment(rows,safe['greedy_assignment'],10.,objective,model);close(saved[0]['expected_success'],score);close(saved[0]['predicted'],gd)
        expected=[dict(action=-1,assignment=list(safe['greedy_assignment']),predicted=gd,expected_success=score)]
        for (action,q),actual in zip(quotas,saved[1:]):
            result=independent_optimum(rows,q,10.,objective,model);optima+=1;assert actual['action']==action
            if result is None:assert actual['accepted'] is False and actual['reason']=='infeasible';continue
            for key,value in result.items():close(actual[key],value)
            accepted=result['predicted']['energy']<=cap+1e-7;assert actual['accepted']==accepted
            if accepted:expected.append(dict(action=action,**result))
        chosen=min(enumerate(expected),key=lambda pair:(-pair[1]['expected_success'],pair[1]['predicted']['energy'],pair[0]))[1]
        for key in ('action','assignment','predicted','expected_success'):close(pub['selected'][objective][key],chosen[key])
        assert pub['selected'][objective]['matching_calls']==1+len(quotas)<=10
    parent_labels=read(source/'labels.json');scores={}
    for name,key in labels['methods'].items():
        out=labels['outcomes'][key];assert out['assignment']==pub['selected'][name]['assignment'];assert len(out['requests'])==len(rows)
        for row,action,req in zip(rows,out['assignment'],out['requests']):assert row['ue_id']==req['ue_id'] and row['source']==req['source'] and action==req['action'] and row['deadline']==req['deadline']
        if name in ('G','old_pool_fee'):
            old_key=hashlib.sha256(json.dumps(out['assignment']).encode()).hexdigest();close(out['requests'],parent_labels['outcomes'][old_key]['requests']);close(out['metrics'],parent_labels['outcomes'][old_key]['metrics'])
    for cost in p['cost_ms']:scores[str(cost)]={name:sum(q['latency']+cost/1000<=q['deadline'] for q in labels['outcomes'][labels['methods'][name]]['requests']) for name in methods}
    states.append(dict(identity=identity,generated=labels['generated'],admitted=labels['admitted'],values=scores,
        metrics={name:labels['outcomes'][labels['methods'][name]]['metrics'] for name in methods},matching_calls={name:pub['selected'][name]['matching_calls'] for name in methods}))
    if si%40==0:write(v/(a.label+'_audit_progress.json'),dict(states=si+1,total=len(p['identities']),optima=optima))
close(states,read(r/'analysis/states.json'));summary=read(r/'analysis/summary.json')
for source in ('all','greedy','learned'):
    subset=[s for s in states if source=='all' or s['identity'][0]==source]
    for cost in p['cost_ms']:
        blocks={}
        for block in sorted({s['identity'][1] for s in subset}):
            rows=[s for s in subset if s['identity'][1]==block];den=sum(s['generated'] for s in rows);admitted=sum(s['admitted'] for s in rows)
            metrics={name:{key:sum(s['metrics'][name][key] for s in rows)/len(rows) for key in ('uav_fleet_total_energy','fairness','mbs_load_ratio')} for name in methods}
            blocks[str(block)]=dict(generated=den,rates={n:sum(s['values'][str(cost)][n] for s in rows)/den for n in methods},metrics=metrics,
                energy_ratio_to_G={n:metrics[n]['uav_fleet_total_energy']/metrics['G']['uav_fleet_total_energy'] for n in methods},mbs_admitted_fraction={n:sum(s['metrics'][n]['service_offloads_mbs'] for s in rows)/max(admitted,1) for n in methods})
        saved=summary['subsets'][source][str(cost)];close(blocks,saved['blocks'])
        for name,value in saved['comparisons'].items():
            left,right=name.split('-');close(paired([b['rates'][left]-b['rates'][right] for b in blocks.values()]),value)
        for name,value in saved['groups'].items():close(paired([b['rates'][name] for b in blocks.values()]),value)
main=summary['subsets']['all']['10.0'];best=max(('G','old_pool_fee','binary','smooth'),key=lambda n:main['groups'][n]['mean']);g=main['comparisons']['probability-G'];inc=main['comparisons']['probability-'+best] if best!='G' else g;energy=np.mean([b['energy_ratio_to_G']['probability'] for b in main['blocks'].values()])
assert summary['decision']['optimistic_signal']==(g['mean']>=.005 and g['positive']>=4 and inc['mean']>=.001 and energy<=1.05)
audit=dict(status='PASS',states=len(states),models_checked=checked_models,models_refitted=refitted,conditional_optima_reconstructed=optima,
    summary_sha256=sha(r/'analysis/summary.json'),protocol_sha256=sha(r/'protocol.json'),goal_complete=False,
    audit_type='primary-agent self-audit; explicit cost-matrix Hungarian replay, unaccelerated predicted assessment, alternate raw outcome statistics')
write(v/(a.label+'_raw_record_audit.json'),audit)
lines=['**'+('G锚定能耗下的概率匹配' if p.get('g_anchored_energy',False) else '概率直接参与条件匹配：旧能耗基准未满足验收')+'**','',
    f"{len(states)}个复用开发状态、25个按块/负载隔离模型、5个预定方法。所有分配先用公开输入封存，再执行原私有物理快照评分；没有原生完整费用计时，也没有独立确认。",'',
    '| 共同费用假设/ms | 原池规则−G/pp | 二值目标−G/pp | 平滑目标−G/pp | 学习概率目标−G/pp |','|---:|---:|---:|---:|---:|']
for cost,z in summary['subsets']['all'].items():lines.append(f"| {float(cost):g} | "+' | '.join(f"{z['comparisons'][n+'-G']['mean']*100:+.3f}" for n in ('old_pool_fee','binary','smooth','probability'))+' |')
lines+=['','| 主10 ms比较 | 差/pp | 95% t区间/pp | 正向块 |','|---|---:|---|---:|']
for name in ('probability-G','probability-old_pool_fee','probability-binary','probability-smooth'):
    z=main['comparisons'][name];lines.append(f"| {name} | {z['mean']*100:+.3f} | [{z['ci95'][0]*100:+.3f}, {z['ci95'][1]*100:+.3f}] | {z['positive']}/5 |")
lines+=['',f"学习方法的真实机队能耗块均比G为{energy:.4f}；预定乐观开发闸门：**{'通过，须继续原生完整费用比较' if summary['decision']['optimistic_signal'] else '未通过，不作为实用方案晋级'}**。该判断含相对G至少+0.5 pp/4正向块、相对最强非学习至少+0.1 pp及能耗比≤1.05；没有放宽原门槛。",'',
    '新概率/二值/平滑目标共享同一配额集合、求解器、观测和能耗约束。原池规则保留其旧匹配和最好保留，匹配次数上限9；三个新目标最多10次。学习的代价函数确实改变请求身份分配，不是给原配额搜索换名。','',
    ('新cap为min(旧cap,1.05×G预测能耗)，三新目标相同。原池规则保留旧guard，故约束不同；对照仍全部披露。' if p.get('g_anchored_energy',False) else '旧cap参照未经普通修复的贪心初值，不是当前G。能耗诊断发现它相对G平均可达1.7519倍，故“旧cap内”不等于“G能耗增加≤5%”。本轮失败保留；后续改基准必须用新协议。'),'',
    '**边界。** 五历史训练块为统计单位；状态、25个交叉折与请求不独立。五块全正的精确双侧符号翻转p仍为0.0625；t区间仅作开发描述。共同0/5/10/20 ms是假设，不是实测计费，也不是安全时间线。只是在指定冻结状态上的即时结果，不能外推新负载或闭环。','',
    f"**复核。** {optima:,}个条件匹配候选用显式代价矩阵/Hungarian和未加速原assessment重建；25个序列化模型全部留出推理核验，固定首折重拟合一致。原G与原池物理结果精确复现，全部逐请求费用及块级统计/闸门另行复算通过。主代理自审，不称独立代理评审。",'',
    '协议、完整统计、逐状态数据和验收均在本任务对应结果/validation目录。原论文、旧确认及历史负结果未改写。布局检查仍只有既存`修改总结报告.md`根目录白名单问题，本轮不处理。']
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')
completion=dict(status='COMPLETED_AND_VALIDATED',goal_complete=False,decision=summary['decision'],files={str(path):sha(path) for path in (r/'protocol.json',r/'analysis/summary.json',r/'analysis/states.json',v/(a.label+'_raw_record_audit.json'),a.report)})
write(v/(a.label+'_completion.json'),completion);print(json.dumps(audit));print(json.dumps(summary['decision']))
