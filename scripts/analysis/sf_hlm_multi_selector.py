"""Remote-only, fixed-candidate cross-fitting of the SF-HLM method selector.

Uses already completed development branches. Zero added selector cost is an
optimistic replay, never a measurement of the full selector deployment path.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

FILES=('paths.py','utils/sf_hlm_selector.py','tests/test_sf_hlm_selector.py',
       'scripts/analysis/sf_hlm_multi_selector.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
CANDIDATES=('ridge_all','ridge_nonlearning','tree_all','tree_nonlearning')
OVERHEAD_MS=(0.,.5,1.,2.,5.)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out);return out


def paired(values):
    from scipy.stats import t
    mean=statistics.mean(values);n=len(values)
    radius=float(t.ppf(.975,n-1))*statistics.stdev(values)/math.sqrt(n)
    exact=sum(abs(statistics.mean(s*v for s,v in zip(signs,values)))>=abs(mean)-1e-14
              for signs in itertools.product((-1,1),repeat=n))/(2**n)
    return dict(mean=mean,ci95=[mean-radius,mean+radius],values=values,n=n,
                positive=sum(v>0 for v in values),exact_signflip_p=exact)


def prepare(a,c):
    if a.results_root.exists():raise FileExistsError('New protocol directory must not exist')
    parent=read(a.parent_results/'protocol.json')
    protocol=dict(name='sf_hlm_multi_selector_20261017_release1',created_unix=time.time(),
        parent_results=str(a.parent_results),parent_protocol_sha256=sha(a.parent_results/'protocol.json'),
        parent_summary_sha256=sha(a.parent_results/'analysis/summary.json'),
        identities=parent['identities'],features=list(c.FEATURE_NAMES),methods=list(c.METHODS),
        candidates=list(CANDIDATES),ridge_alpha=10.,tree_depth=2,tree_min_leaf=16,
        random_state=20261017,overhead_ms=list(OVERHEAD_MS),
        gates=dict(gain=.005,positive_blocks=4,energy_ratio=1.05),
        files={f:sha(REPO_ROOT/f) for f in FILES},
        information='Observed rows only; no policy, assignment, private truth or identity features',
        folds='25 leave-one-block-and-workload; train 256, test 16, excluded 128',
        label='Expected fee-success count minus G; average all random and timing repeats',
        costs='Reused original complete branch times; additional selector cost assumed or swept, not measured',
        state_delay_applied=False,independent_confirmation=False,no_RL_training=True,
        no_automatic_next_stage=True)
    write(a.results_root/'protocol.json',protocol)
    write(a.results_root/'status.json',dict(status='PREPARED',updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',states=len(protocol['identities']),protocol_sha256=sha(a.results_root/'protocol.json'))),flush=True)


def load_population(a,p,c):
    rows=[];receipts=[];parent_sha=p['parent_protocol_sha256']
    if sha(a.parent_results/'protocol.json')!=parent_sha:raise ValueError('Parent protocol changed')
    if sha(a.parent_results/'analysis/summary.json')!=p['parent_summary_sha256']:raise ValueError('Parent summary changed')
    for identity in p['identities']:
        td=a.parent_results/'tasks'/('benchmark_'+'_'.join(map(str,identity)))
        rec=read(td/'result.json');bench_path=td/'benchmark.json';bench=read(bench_path)
        if rec['status']!='PASS' or rec['protocol_sha256']!=parent_sha:raise ValueError('Invalid parent receipt')
        if rec['files']!={'benchmark.json':sha(bench_path)}:raise ValueError('Benchmark hash mismatch')
        if rec['identity']!=identity or bench['identity']!=identity:raise ValueError('State identity mismatch')
        if len(rec['state_files'])!=1:raise ValueError('Ambiguous state metadata')
        state_path,next_hash=next(iter(rec['state_files'].items()))
        if sha(state_path)!=next_hash or bench['state_sha256']!=next_hash:raise ValueError('State metadata changed')
        state=read(state_path);inp_path=state['input']
        if sha(inp_path)!=state['files'][inp_path]:raise ValueError('Observed graph hash mismatch')
        inp=read(inp_path);observed=inp['rows']
        if state['identity']!=identity or len(observed)!=bench['admitted']:raise ValueError('Observed graph population')
        if bench['state_delay_applied']:raise ValueError('Unexpected waiting feedback')
        expected={(m,i) for m in c.METHODS for i in range(5 if m.startswith('R') else 1)}
        actual=[(q['method'],q['repeat']) for q in bench['results']]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Branch repetitions')
        trials={m:[q for q in bench['results'] if q['method']==m] for m in c.METHODS}
        for m,rs in trials.items():
            for q in rs:
                if len(q['software_seconds'])!=3 or len(q['requests'])!=len(observed):raise ValueError('Timing or request population')
                if [r['ue_id'] for r in q['requests']]!=[r['ue_id'] for r in observed]:raise ValueError('Request order')
                if q['generated']!=bench['generated'] or q['admitted']!=bench['admitted']:raise ValueError('Denominator differs')
                if not math.isclose(c.score([q]),q['fee_success'],abs_tol=1e-12):raise ValueError('Parent fee score')
                if sum(r['latency']<=r['deadline'] for r in q['requests'])!=q['raw_success']:raise ValueError('Raw score')
        means={m:dict(fee={str(h):c.score(rs,h/1000) for h in OVERHEAD_MS},
            raw=statistics.mean(q['raw_success'] for q in rs),
            energy=statistics.mean(q['metrics']['uav_fleet_total_energy'] for q in rs),
            mbs=statistics.mean(q['metrics']['service_offloads_mbs'] for q in rs),
            fairness=statistics.mean(q['metrics']['fairness'] for q in rs),
            calls=statistics.mean(q['diagnostic']['matching_calls'] for q in rs)) for m,rs in trials.items()}
        rows.append(dict(identity=identity,generated=bench['generated'],admitted=bench['admitted'],
                         features=c.features(observed),methods=means))
        receipts.append(dict(identity=identity,benchmark=str(bench_path),benchmark_sha256=sha(bench_path),
            state=state_path,state_sha256=next_hash,input=inp_path,input_sha256=sha(inp_path)))
    ids=[tuple(r['identity']) for r in rows]
    if len(ids)!=400 or len(set(ids))!=400 or set(ids)!=set(map(tuple,p['identities'])):raise ValueError('Population')
    return rows,receipts


def fit(kind,methods,x,y):
    import numpy as np
    if kind=='ridge':
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler().fit(x);reg=Ridge(alpha=10.).fit(scaler.transform(x),y)
        model=dict(kind=kind,methods=list(methods),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                   coef=reg.coef_.tolist(),intercept=reg.intercept_.tolist())
        native=lambda v:reg.predict(scaler.transform(np.asarray(v)))
    else:
        from sklearn.tree import DecisionTreeRegressor
        reg=DecisionTreeRegressor(max_depth=2,min_samples_leaf=16,random_state=20261017).fit(x,y)
        tree=reg.tree_
        model=dict(kind=kind,methods=list(methods),nodes=[dict(left=int(tree.children_left[i]),right=int(tree.children_right[i]),
            feature=int(tree.feature[i]),threshold=float(tree.threshold[i]),value=tree.value[i,:,0].tolist())
            for i in range(tree.node_count)])
        native=lambda v:reg.predict(np.asarray(v))
    return model,native


def probabilities(methods,chosen):
    count=Counter(chosen);return {m:count[m]/len(chosen) for m in methods}


def crossfit(a,rows,c):
    import numpy as np
    x=np.asarray([s['features'] for s in rows]);ids=[s['identity'] for s in rows];fold_records=[]
    for s in rows:s['policies']={m:{m:1.} for m in c.METHODS}
    for f in c.folds(ids):
        train,test=f['train'],f['test'];trained={};test_predictions={}
        # Equal training-block weights for the constant-arm baseline.
        block_ids=sorted({ids[i][1] for i in train})
        constant={m:statistics.mean(sum(rows[i]['methods'][m]['fee']['0.0'] for i in train if ids[i][1]==b)/
                  sum(rows[i]['generated'] for i in train if ids[i][1]==b) for b in block_ids) for m in c.METHODS}
        best=max(c.METHODS,key=lambda m:constant[m])
        for i in test:
            rows[i]['policies']['fold_best_constant']={best:1.}
            rows[i]['policies']['rule_R8_G']={'R8' if rows[i]['admitted']>=16 else 'G':1.}
        for name in CANDIDATES:
            kind=name.split('_')[0];methods=(c.METHODS if name.endswith('_all') else c.NONLEARNING)[1:]
            y=np.asarray([[s['methods'][m]['fee']['0.0']-s['methods']['G']['fee']['0.0'] for m in methods] for s in rows])
            model,native=fit(kind,methods,x[train],y[train]);native_preds=native(x)
            exported=np.asarray([[c.predict(model,v)[m] for m in methods] for v in x])
            if not np.allclose(native_preds,exported,rtol=1e-11,atol=1e-11):raise ValueError('Exported inference mismatch')
            selected=[c.choose(dict(zip(methods,pred)),empty=rows[i]['admitted']==0) for i,pred in enumerate(exported)]
            probs=probabilities(c.METHODS,[selected[i] for i in train])
            trained[name]=dict(model=model,train_proportions=probs,
                max_export_error=float(np.max(np.abs(native_preds-exported))))
            test_predictions[name]={str(i):dict(predicted=dict(zip(methods,exported[i].tolist())),chosen=selected[i]) for i in test}
            for i in test:
                rows[i]['policies'][name]={selected[i]:1.}
                rows[i]['policies'][name+'_random_mix']=probs.copy()
                if name.endswith('_all'):
                    chosen={'L8':'R8','L16':'R16'}.get(selected[i],selected[i])
                    rows[i]['policies'][name+'_replace_L_with_R']={chosen:1.}
        fold_records.append(dict(**f,constant_arm_scores=constant,best_constant=best,
                                 models=trained,predictions=test_predictions))
        print(json.dumps(dict(fold=len(fold_records),block=f['block'],workload=f['workload'],status='FITTED')),flush=True)
    write(a.results_root/'folds.json',fold_records)
    return fold_records


def summarize(rows,c):
    subsets={};policy_names=list(rows[0]['policies'])
    for subset in ('all','greedy','learned'):
        sr=[s for s in rows if subset=='all' or s['identity'][0]==subset];blocks=sorted({s['identity'][1] for s in sr})
        by_block={}
        for b in blocks:
            br=[s for s in sr if s['identity'][1]==b];den=sum(s['generated'] for s in br);ad=sum(s['admitted'] for s in br)
            values={}
            for name in policy_names:
                values[name]={}
                for h in OVERHEAD_MS:
                    fee_h='0.0' if name in c.METHODS else str(h)
                    successes=sum(sum(prob*s['methods'][m]['fee'][fee_h] for m,prob in s['policies'][name].items()) for s in br)
                    values[name][str(h)]=dict(fee_success=successes,fee_rate=successes/den,
                        raw_rate=sum(sum(prob*s['methods'][m]['raw'] for m,prob in s['policies'][name].items()) for s in br)/den,
                        energy=sum(sum(prob*s['methods'][m]['energy'] for m,prob in s['policies'][name].items()) for s in br),
                        mbs_share=sum(sum(prob*s['methods'][m]['mbs'] for m,prob in s['policies'][name].items()) for s in br)/ad if ad else 0.,
                        fairness=statistics.mean(sum(prob*s['methods'][m]['fairness'] for m,prob in s['policies'][name].items()) for s in br),
                        invocation=statistics.mean(1-s['policies'][name].get('G',0.) for s in br),
                        learning=statistics.mean(sum(v for m,v in s['policies'][name].items() if m.startswith('L')) for s in br),
                        oracle_loss=sum(max(s['methods'][m]['fee'][str(h)] for m in c.METHODS)-
                            sum(prob*s['methods'][m]['fee'][fee_h] for m,prob in s['policies'][name].items()) for s in br))
            for name,methods in [('oracle_all',c.METHODS),('oracle_nonlearning',c.NONLEARNING)]:
                values[name]={str(h):dict(fee_rate=sum(max(s['methods'][m]['fee'][str(h)] for m in methods) for s in br)/den)
                              for h in OVERHEAD_MS}
            by_block[str(b)]=dict(generated=den,admitted=ad,states=len(br),policies=values)
        groups={};comparisons={}
        names=policy_names+['oracle_all','oracle_nonlearning']
        for name in names:
            groups[name]={str(h):paired([q['policies'][name][str(h)]['fee_rate'] for q in by_block.values()]) for h in OVERHEAD_MS}
            if name!='G':
                comparisons[name+'-G']={str(h):paired([q['policies'][name][str(h)]['fee_rate']-q['policies']['G']['0.0']['fee_rate'] for q in by_block.values()]) for h in OVERHEAD_MS}
        pairs=[('oracle_all','oracle_nonlearning')]
        for kind in ('ridge','tree'):
            pairs += [(kind+'_all',kind+'_nonlearning'),(kind+'_all',kind+'_all_replace_L_with_R')]
        pairs += [(name,name+'_random_mix') for name in CANDIDATES]
        for first,second in pairs:
            comparisons[first+'-'+second]={str(h):paired([q['policies'][first][str(h)]['fee_rate']-
                q['policies'][second][str(h)]['fee_rate'] for q in by_block.values()]) for h in OVERHEAD_MS}
        auxiliary={name:dict(energy_ratio=statistics.mean(q['policies'][name]['0.0']['energy']/q['policies']['G']['0.0']['energy'] for q in by_block.values()),
            raw_rate=statistics.mean(q['policies'][name]['0.0']['raw_rate'] for q in by_block.values()),
            invocation=statistics.mean(q['policies'][name]['0.0']['invocation'] for q in by_block.values()),
            learning=statistics.mean(q['policies'][name]['0.0']['learning'] for q in by_block.values()),
            oracle_loss=sum(q['policies'][name]['0.0']['oracle_loss'] for q in by_block.values()),
            proportions={m:statistics.mean(s['policies'][name].get(m,0.) for s in sr) for m in c.METHODS}) for name in policy_names}
        subsets[subset]=dict(blocks=by_block,groups=groups,comparisons=comparisons,auxiliary=auxiliary)
    all_=subsets['all'];passes={}
    for name in CANDIDATES:
        d=all_['comparisons'][name+'-G']['0.0'];energy=all_['auxiliary'][name]['energy_ratio']
        passes[name]=dict(passes_optimistic_screen=d['mean']>=.005 and d['positive']>=4 and energy<=1.05,
                          delta_vs_G=d,energy_ratio=energy)
    decision=dict(candidates=passes,proceed_to_full_path_timing=any(v['passes_optimistic_screen'] for v in passes.values()),
        no_automatic_next_stage=True,interpretation='Optimistic development replay only; not actual selector latency or closed-loop DSR')
    return dict(states=len(rows),training_blocks=5,folds=25,models=100,subsets=subsets,decision=decision,
                state_delay_applied=False,independent_confirmation=False)


def report(a,result):
    all_=result['subsets']['all'];cmp=all_['comparisons'];aux=all_['auxiliary']
    lines=['**七方法选择器：开发留出筛选**','',
        '复用400个已用于开发的同状态分支；25折同时留一训练块和一工作负载，每折训练256、测试16，100个固定小模型。统计单位仍为五个历史训练块。本轮没有RL重训。','',
        '**以下均为零新增选择器费用的乐观重放，原分支完整软件费用已计入。不是完整选择器实测、整回合DSR或独立确认。**','',
        '| 方法 | 费用后成功率/% | 相对G/pp | 95% t区间/pp | 正向块 | 搜索调用/% | PPO调用/% | 能耗比G |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for name in ['G','fold_best_constant','rule_R8_G',*CANDIDATES,*[n+'_random_mix' for n in CANDIDATES],
                 'ridge_all_replace_L_with_R','tree_all_replace_L_with_R']:
        g=all_['groups'][name]['0.0'];d=cmp.get(name+'-G',{}).get('0.0');x=aux[name]
        lines.append(f"| {name} | {g['mean']*100:.3f} | {d['mean']*100:+.3f} | [{d['ci95'][0]*100:+.3f}, {d['ci95'][1]*100:+.3f}] | {d['positive']}/5 | {x['invocation']*100:.1f} | {x['learning']*100:.1f} | {x['energy_ratio']:.4f} |" if d else
            f"| G | {g['mean']*100:.3f} | 0 | — | — | 0 | 0 | 1 |")
    lines+=['','Ridge固定alpha=10，树固定深度2、叶节点至少16；all含七方法，nonlearning仅含G/R8/R16/H8/H16。随机混合的调用概率来自对应训练折；replace对照把已选择L8/L16替换成R8/R16。每折最好常开方法仅看训练折。固定规则为批次≥16则R8，否则G。','',
        '| 方法/经验预言机 | +0 ms/pp | +0.5 ms/pp | +1 ms/pp | +2 ms/pp | +5 ms/pp |','|---|---:|---:|---:|---:|---:|']
    for name in [*CANDIDATES,'oracle_all','oracle_nonlearning']:
        lines.append('| '+name+' | '+' | '.join(f"{cmp[name+'-G'][str(h)]['mean']*100:+.3f}" for h in OVERHEAD_MS)+' |')
    lines+=['','新增费用通过逐请求重新判定截止期得到；模型和动作在零新增费用训练后冻结，敏感性不重新调模型。经验预言机在各费用假设下重新事后择优。它不是可部署上界证明。','',
        '| 状态来源 | 方法 | 相对G/pp | 95% t区间/pp |','|---|---|---:|---|']
    for source in ('greedy','learned'):
        for name in CANDIDATES:
            d=result['subsets'][source]['comparisons'][name+'-G']['0.0']
            lines.append(f"| {source} | {name} | {d['mean']*100:+.3f} | [{d['ci95'][0]*100:+.3f}, {d['ci95'][1]*100:+.3f}] |")
    lines+=['','**决定。** '+('至少一候选通过乐观筛选；仍须冻结完整在线路径并实测后才能判断是否晋级。' if result['decision']['proceed_to_full_path_timing'] else
        '四候选均未达到预定平均+0.5 pp、至少4/5块正向及能耗约束的组合目标，本轮不进入完整路径计时或闭环。新增费用非负时，本轮已冻结动作的重放成功数不会增加。'),'',
        'OOF折共享训练数据且使用过的开发标签已参与先前路线判断，不能把25折或400状态充作独立训练重复。报告五块配对t区间及精确双侧符号翻转，所有候选同时披露；不据此作确认性多重检验结论。','',
        '候选的误选损失（相对同状态七方法经验择优的净成功数）、原始质量、调用比例及能耗保存在summary.json的auxiliary；逐状态输入特征/策略概率在states.json，25折模型与身份在folds.json。输入摘要在inputs.json。','',
        '边界：只检验本轮63个观测特征和四个固定选择器。没有检验多步教师、其他动作空间或跨时隙决策；本次失败不能宣布这些方向已被否定。费用仍是事后截止期评分，未回写环境。原确认与论文保持不变。']
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')


def run(a,c):
    import numpy,scipy,sklearn
    p=read(a.results_root/'protocol.json')
    if (a.results_root/'summary.json').exists():raise FileExistsError('Completed scientific result is immutable')
    if any(sha(REPO_ROOT/f)!=v for f,v in p['files'].items()):raise ValueError('Frozen source changed')
    write(a.results_root/'status.json',dict(status='RUNNING',updated_unix=time.time()))
    rows,receipts=load_population(a,p,c);write(a.results_root/'inputs.json',receipts)
    crossfit(a,rows,c);write(a.results_root/'states.json',rows)
    result=summarize(rows,c);result['protocol_sha256']=sha(a.results_root/'protocol.json')
    result['environment']=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        cpu=sorted(os.sched_getaffinity(0)),load=os.getloadavg(),blas_threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    write(a.results_root/'summary.json',result);report(a,result)
    write(a.results_root/'status.json',dict(status='COMPLETED',states=len(rows),models=100,updated_unix=time.time()))
    print(json.dumps(result['decision']),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run'])
    ap.add_argument('--parent-results',type=Path,required=True);ap.add_argument('--results-root',type=Path,required=True)
    ap.add_argument('--resource-module',type=Path,required=True);ap.add_argument('--resources',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Fit and scientific aggregation must run on remote Linux')
    c=module('sf_selector',REPO_ROOT/'utils/sf_hlm_selector.py')
    resources=module('sf_resources',a.resource_module)
    with resources.CpuLease(a.resources,task='sf_hlm_multi_selector_development'):
        try:
            if a.mode=='prepare':prepare(a,c)
            else:run(a,c)
        except Exception as e:
            if a.results_root.exists() and not (a.results_root/'summary.json').exists():
                write(a.results_root/'status.json',dict(status='FAILED',error=repr(e),updated_unix=time.time()))
            raise


if __name__=='__main__':main()
