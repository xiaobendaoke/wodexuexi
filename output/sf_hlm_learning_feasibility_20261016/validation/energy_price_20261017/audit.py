"""Remote audit: honest folds, public price constraints and held-out physics.

Every training probability/energy is reconstructed. First-fold conditional
optima are additionally replayed; all held-out selections/statistics are rebuilt.
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
def pick(candidates,price,normalizer):
    return min(range(len(candidates)),key=lambda i:(-(candidates[i]['score']-price*candidates[i]['energy']/normalizer),candidates[i]['energy'],i))
def check_price(cases,saved):
    mu=max(sum(c['candidates'][0]['energy'] for c in cases)/len(cases),1e-12);close(mu,saved['normalizer']);points={0.}
    for case in cases:
        for left,right in itertools.combinations(case['candidates'],2):
            if left['energy']==right['energy']:continue
            value=(left['score']-right['score'])*mu/(left['energy']-right['energy'])
            if value>=0 and math.isfinite(value):points.add(value)
    points=sorted(points);assert len(points)==saved['crossing_count']
    def check(price):
        groups={};indices=[]
        for c in cases:
            index=pick(c['candidates'],price,mu);indices.append(index);g=groups.setdefault(str(c['block']),[0.,0.]);g[0]+=c['candidates'][index]['energy'];g[1]+=c['candidates'][0]['energy']
        ratios={k:(x/y if y else 1. if x==0 else math.inf) for k,(x,y) in groups.items()}
        return all(value<=1.05+1e-12 for value in ratios.values()),ratios,indices
    feasible,ratios,indices=check(saved['price']);assert feasible
    close(ratios,saved['training_energy_ratio_by_block']);assert indices==saved['selected_candidate_indices']
    index=saved['first_feasible_crossing_index']
    if index is None:assert saved['price']==0 and check(0.)[0]
    else:
        close(saved['price'],points[index]+max(abs(points[index]),1.)*1e-12);assert not check(0.)[0]
        if index>0:assert not check(points[index-1]+max(abs(points[index-1]),1.)*1e-12)[0]

ap=argparse.ArgumentParser()
for key in ('release','base_results','gcap_results','teacher_results','training','calibration','results','validation','report'):
    ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
a=ap.parse_args();v=a.validation;r=a.results;cp=read(a.calibration/'protocol.json');p=read(r/'protocol.json');completion=read(a.calibration/'completion.json')
sys.path.insert(0,str(a.release));cm=load('audited_priced_matching',a.release/'utils/sf_hlm_calibrated_matching.py');rank=load('audited_price_observation',a.release/'utils/sf_hlm_action_ranker.py')
from utils.deadline_repair import assess
from utils.sf_hlm_matching import allocation_counts,problem_from_rows
for file,digest in cp['files'].items():assert sha(a.release/file)==digest
for file,digest in cp['input_files'].items():assert sha(file)==digest
for file,digest in completion['files'].items():assert sha(file)==digest
assert p['calibration_protocol_sha256']==sha(a.calibration/'protocol.json') and p['calibration_completion_sha256']==sha(a.calibration/'completion.json')
assert cp['target']==1.05 and cp['nominal_cost_ms']==10.
identities=cp['identities'];calibrations={};candidate_count=0;optima=0
for fold_number,folder in enumerate(sorted((a.calibration/'folds').iterdir()),1):
    rec=read(folder/'result.json');packet=read(folder/'prices.json');cases=read(folder/'cases.json');fold=packet['fold'];block=fold['block'];workload=fold['workload']
    train=[i for i,z in enumerate(identities) if z[1]!=block and z[2]!=workload];test=[i for i,z in enumerate(identities) if z[1]==block and z[2]==workload]
    assert fold['train']==train and fold['test']==test and len(train)==256 and len(test)==16 and rec['fold']==fold
    model_path=a.training/'folds'/folder.name/'model.pkl';bundle=pickle.loads(model_path.read_bytes());assert bundle['fold']==fold and sha(model_path)==packet['model_sha256'];model=bundle['model']
    assert packet['protocol_sha256']==sha(a.calibration/'protocol.json') and rec['protocol_sha256']==packet['protocol_sha256']
    for objective in ('binary','smooth','probability'):assert [c['state_index'] for c in cases[objective]]==train
    probability_x=[];offsets=[];offset=0
    for position,si in enumerate(train):
        identity=identities[si];pub=read(a.base_results/'tasks'/('matched_'+'_'.join(map(str,identity)))/'public.json')
        original=read(a.teacher_results/'tasks'/('oracle_'+'_'.join(map(str,identity)))/'public.json');safe=rank.public_payload(original);rows=safe['rows']
        for objective in ('binary','smooth'):
            expected=[dict(action=c['action'],assignment=c['assignment'],score=c['expected_success'],energy=c['predicted']['energy']) for c in pub['candidates'][objective] if c['accepted']]
            actual=cases[objective][position];assert actual['identity']==identity and actual['block']==identity[1];close(actual['candidates'],expected)
        case=cases['probability'][position];assert case['identity']==identity and case['block']==identity[1] and case['candidates'][0]['action']==-1
        allowed={action:counts for action,counts in pub['quotas']};features_by_quota={};problem=problem_from_rows(rows,5)
        for c in case['candidates']:
            assignment=c['assignment'];q=allocation_counts(rows,assignment,5);prediction=assess(rows,assignment);close(c['energy'],prediction['energy']);assert c['energy']<=pub['energy_cap']+1e-7
            if c['action']==-1:assert assignment==safe['greedy_assignment']
            else:assert list(q)==allowed[c['action']]
            key=tuple(q)
            if key not in features_by_quota:features_by_quota[key]=cm.edge_features(rows,q,10.,problem=problem)[:2]
            xx,index=features_by_quota[key];lookup={tuple(ij):value for ij,value in zip(index,xx)};selected=[]
            for i,(row,action) in enumerate(zip(rows,assignment)):
                target=row['options'][action]['target'];selected.append(lookup[i,5 if target=='mbs' else int(target)])
            if selected:probability_x.append(np.asarray(selected));offsets.append((c,offset,offset+len(selected)));offset+=len(selected)
            else:close(c['score'],0.)
            candidate_count+=1
        if (block,workload)==(620001,780001):
            expected=[]
            for action,q in pub['quotas']:
                optimum=cm.solve(rows,q,10.,'probability',model,problem=problem);optima+=1
                if optimum is not None and optimum['predicted']['energy']<=pub['energy_cap']+1e-7:expected.append((action,list(optimum['assignment'])))
            assert [(c['action'],c['assignment']) for c in case['candidates'][1:]]==expected
    if probability_x:
        xx=np.concatenate(probability_x);proba=model.predict_proba(xx)[:,list(model.classes_).index(1.)]
        for c,start,stop in offsets:close(float(proba[start:stop].sum()),c['score'])
    for objective in ('binary','smooth','probability'):check_price(cases[objective],packet['prices'][objective])
    calibrations[block,workload]=packet;write(v/'audit_progress.json',dict(stage='calibration',completed_folds=fold_number,total_folds=25,training_candidates=candidate_count))

states=[];controls=p['nonlearning'];methods=p['methods']
for si,identity in enumerate(p['identities']):
    folder=r/'tasks'/('priced_'+'_'.join(map(str,identity)));rec=read(folder/'result.json');pub=read(folder/'public.json');lab=read(folder/'labels.json')
    assert rec['status']=='PASS' and rec['identity']==identity and rec['protocol_sha256']==sha(r/'protocol.json')
    for file,digest in rec['files'].items():assert sha(folder/file)==digest
    assert lab['public_sha256']==sha(folder/'public.json') and lab['identity']==identity
    bd=a.base_results/'tasks'/('matched_'+'_'.join(map(str,identity)));gd=a.gcap_results/'tasks'/bd.name;base=read(bd/'public.json');gc=read(gd/'public.json');price=calibrations[identity[1],identity[2]]
    assert pub['base_public_sha256']==sha(bd/'public.json') and pub['gcap_public_sha256']==sha(gd/'public.json');assert pub['price_sha256']==sha(a.calibration/'folds'/f'block_{identity[1]}_workload_{identity[2]}'/'prices.json')
    selected={name:base['selected'][name] for name in ('G','old_pool_fee','binary','smooth')};selected.update(gcap_binary=gc['selected']['binary'],gcap_smooth=gc['selected']['smooth'])
    for objective in ('binary','smooth','probability'):
        candidates=[c for c in base['candidates'][objective] if c['accepted']];simple=[dict(action=c['action'],score=c['expected_success'],energy=c['predicted']['energy']) for c in candidates];parameters=price['prices'][objective]
        index=pick(simple,parameters['price'],parameters['normalizer']);got=pub['decision'][objective+'_price'];assert got['candidate_index']==index;close(got['price'],parameters['price']);close(got['normalizer'],parameters['normalizer'])
        c=candidates[index];selected[objective+'_price']=dict(action=c['action'],assignment=c['assignment'],predicted=c['predicted'],matching_calls=base['selected'][objective]['matching_calls'])
    close(selected,pub['selected'])
    for name,key in lab['methods'].items():assert lab['outcomes'][key]['assignment']==selected[name]['assignment']
    base_labels=read(bd/'labels.json');gcap_labels=read(gd/'labels.json')
    for name in ('G','old_pool_fee','binary','smooth','gcap_binary','gcap_smooth'):
        old=gcap_labels if name.startswith('gcap_') else base_labels;ref=name.removeprefix('gcap_');close(lab['outcomes'][lab['methods'][name]],old['outcomes'][old['methods'][ref]])
    values={str(cost):{name:sum(q['latency']+cost/1000<=q['deadline'] for q in lab['outcomes'][lab['methods'][name]]['requests']) for name in methods} for cost in p['cost_ms']}
    states.append(dict(identity=identity,generated=lab['generated'],admitted=lab['admitted'],values=values,metrics={name:lab['outcomes'][lab['methods'][name]]['metrics'] for name in methods},matching_calls={name:selected[name]['matching_calls'] for name in methods}))
    if si%40==0:write(v/'audit_progress.json',dict(stage='held_out_physics',states=si+1,total=400))
close(states,read(r/'analysis/states.json'));s=read(r/'analysis/summary.json')
for source in ('all','greedy','learned'):
    subset=[z for z in states if source=='all' or z['identity'][0]==source]
    for cost in p['cost_ms']:
        blocks={}
        for block in sorted({z['identity'][1] for z in subset}):
            rows=[z for z in subset if z['identity'][1]==block];den=sum(z['generated'] for z in rows);admitted=sum(z['admitted'] for z in rows)
            metrics={name:{key:sum(z['metrics'][name][key] for z in rows)/len(rows) for key in ('uav_fleet_total_energy','fairness','mbs_load_ratio')} for name in methods}
            blocks[str(block)]=dict(generated=den,rates={name:sum(z['values'][str(cost)][name] for z in rows)/den for name in methods},metrics=metrics,
                energy_ratio_to_G={name:metrics[name]['uav_fleet_total_energy']/metrics['G']['uav_fleet_total_energy'] for name in methods},mbs_admitted_fraction={name:sum(z['metrics'][name]['service_offloads_mbs'] for z in rows)/max(admitted,1) for name in methods})
        saved=s['subsets'][source][str(cost)];close(blocks,saved['blocks'])
        for name,value in saved['comparisons'].items():
            left,right=name.split('-');close(paired([b['rates'][left]-b['rates'][right] for b in blocks.values()]),value)
        for name,value in saved['groups'].items():close(paired([b['rates'][name] for b in blocks.values()]),value)
main=s['subsets']['all']['10.0'];best=max(controls,key=lambda n:main['groups'][n]['mean']);g=main['comparisons']['probability_price-G'];inc=main['comparisons']['probability_price-'+best] if best!='G' else g;energy=np.mean([b['energy_ratio_to_G']['probability_price'] for b in main['blocks'].values()])
assert s['decision']['optimistic_signal']==(g['mean']>=.005 and g['positive']>=4 and inc['mean']>=.001 and energy<=1.05)
audit=dict(status='PASS',states=400,calibration_folds=25,training_probability_candidates=candidate_count,conditional_optima_replayed_first_fold=optima,
    summary_sha256=sha(r/'analysis/summary.json'),protocol_sha256=sha(r/'protocol.json'),goal_complete=False,
    audit_type='primary-agent self-audit: all public training scores/energy/prices, first-fold optima, held-out choices and alternate raw statistics')
write(v/'raw_record_audit.json',audit)
lines=['**训练折公开能耗价格：通过乐观开发筛选，尚待实测全费用**','',
    '25折各自仅使用256个训练状态的公开候选、预测成功数与能耗标定价格；无新模型训练、无实际成功/真实缓存/实际能耗标签参与价格选择。每折价格需使四个训练块各自的预测分配能耗比G≤1.05。模型、价格与normalizer在16个留出状态上冻结。','',
    f"400个复用开发状态、9个预定方法完成。主10 ms条件下：学习相对G {g['mean']*100:+.3f} pp，95% t区间[{g['ci95'][0]*100:+.3f},{g['ci95'][1]*100:+.3f}]；相对最强非学习{best} {inc['mean']*100:+.3f} pp，区间[{inc['ci95'][0]*100:+.3f},{inc['ci95'][1]*100:+.3f}]。实际机队能耗块均比G {energy:.4f}。",'',
    '| 共同费用假设/ms | 学习价格−G/pp | 相对该条件最好非学习/pp |','|---:|---:|---:|']
for cost,z in s['subsets']['all'].items():
    ref=max(controls,key=lambda n:z['groups'][n]['mean']);gain=z['comparisons']['probability_price-G'];difference=z['comparisons']['probability_price-'+ref] if ref!='G' else gain
    lines.append(f"| {float(cost):g} | {gain['mean']*100:+.3f} | {difference['mean']*100:+.3f}（{ref}） |")
lines+=['','| 主10 ms方法 | 相对G/pp | 平均机队能耗比G |','|---|---:|---:|']
for name in methods:
    gain=0. if name=='G' else main['comparisons'][name+'-G']['mean']*100;ratio=np.mean([b['energy_ratio_to_G'][name] for b in main['blocks'].values()]);lines.append(f'| {name} | {gain:+.3f} | {ratio:.4f} |')
lines+=['',
    '**判定。** '+('预定+0.5 pp/4正向块、最好非学习+0.1 pp及实际能耗≤1.05三个乐观开发门槛均通过，进入原生完整费用比较。' if s['decision']['optimistic_signal'] else '至少一个预定门槛未过，不晋级为已解决方案。'),'',
    '**费用与证据边界。** 这里的0/5/10/20 ms是共同假设；离线复用公开匹配只节省分析计算，不是免费在线候选。新方法仍需支付最多10次匹配、41维特征、模型推理和价格选择；G及各非学习控制按最省必要路径计费。尚未比较新的原生实测费用和R8/R16，也未完成新负载闭环或独立确认。当前不能声称可部署、问题已解决或PPO贡献。','',
    '统计单位为五个历史训练块；25交叉折、状态、请求和训练候选不增加独立n。五块全正时精确双侧符号翻转p仍为0.0625；t区间仅作探索描述。MBS依赖、公平性和各来源均在完整统计中披露，不宣称所有指标最优。','',
    f"**自审。** 重构{candidate_count:,}个训练概率候选的公开期望成功数/预测能耗、25折训练身份与所有能耗价格；固定首折{optima:,}个条件最优重新求解。全部400状态留出选择、六个既有控制物理结果、逐请求假设计费和块级统计/闸门复算通过。主代理自审，不称独立代理评审。",'',
    '可复核：[协议](../../../results/sf_hlm_learning_feasibility_20261016/energy_price_20261017/development_v1/protocol.json)、[完整统计](../../../results/sf_hlm_learning_feasibility_20261016/energy_price_20261017/development_v1/analysis/summary.json)、[训练标定协议](../../../results/sf_hlm_learning_feasibility_20261016/energy_price_20261017/calibration_v1/protocol.json)、[原始验收](../validation/energy_price_20261017/raw_record_audit.json)。','',
    '原论文、旧确认和历史失败结果均未改写。布局检查保留既存`修改总结报告.md`根目录白名单问题；本轮不修改该文件或白名单。持续目标尚未完成。']
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')
receipt=dict(status='COMPLETED_AND_VALIDATED',goal_complete=False,decision=s['decision'],files={str(path):sha(path) for path in (a.calibration/'protocol.json',a.calibration/'completion.json',r/'protocol.json',r/'analysis/summary.json',v/'raw_record_audit.json',a.report)})
write(v/'completion.json',receipt);print(json.dumps(audit));print(json.dumps(s['decision']))
