"""Frozen multi-step teacher checkpoints: same-state quality and full-cost screen.

No training and no use of confirmation outcomes. Numerical work is Linux-only;
the archived feasibility runner supplies the unchanged environment/timer adapter.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

FILES=('paths.py','scripts/analysis/sf_hlm_teacher_feasibility.py',
       'tests/test_sf_hlm_teacher_feasibility.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
CANDIDATES=('baseline_bc','baseline_ppo','multistep_bc','multistep_ppo')
METHODS=('G','old_L16','R16',*CANDIDATES)
TEACHER_PAIRS=(('multistep_ppo','baseline_ppo'),('multistep_bc','baseline_bc'),
               ('baseline_ppo','baseline_bc'),('multistep_ppo','multistep_bc'))


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x) if isinstance(x,Path) else fail(x))+'\n')
    temp.replace(path)


def fail(x):raise TypeError(type(x).__name__)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).digest()[:4],'big')


def model_identity(method):
    if method not in CANDIDATES:raise ValueError('Unregistered teacher/model stage')
    variant,stage=method.rsplit('_',1)
    return variant,'bc' if stage=='bc' else 'checkpoint'


def branch_specs(identity):
    return [dict(method=m,repeat=i,kind='greedy' if m=='G' else 'random' if m=='R16' else 'learned',
        cap=0 if m=='G' else 16,seed=seed_for('sf_hlm_feasibility_20261016',identity,i))
        for m in METHODS for i in range(5 if m=='R16' else 1)]


def interaction(rates):
    return (rates['multistep_ppo']-rates['multistep_bc'])-(rates['baseline_ppo']-rates['baseline_bc'])


def holm(values):
    out=[0.]*len(values);last=0.
    for rank,i in enumerate(sorted(range(len(values)),key=lambda i:values[i])):
        last=max(last,min(1.,values[i]*(len(values)-rank)));out[i]=last
    return out


def classify(quality,cost,*,energy_vs_random,energy_vs_g,within_time):
    return dict(quality_candidate=quality['mean']>=.003 and quality['positive']>=4 and energy_vs_random<=1.05 and within_time,
        cost_candidate=cost['mean']>=.005 and cost['positive']>=4 and energy_vs_g<=1.05 and within_time)


def archived(a):return load('teacher_feasibility_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')


def runtime_paths(a,f):return f.parent_args(a)


def state_folder(a,f,identity):return f.state_dir(argparse.Namespace(results_root=a.state_root),identity)


def task_folder(a,identity):return a.results_root/'tasks'/('benchmark_'+'_'.join(map(str,identity)))


def inventory(a,f):
    parent=runtime_paths(a,f);old=read(parent.results_root/'protocol.json');states=read(a.state_root/'protocol.json')
    if old['smoke'] or states['smoke'] or states['blocks']!=old['blocks']:raise ValueError('Invalid development inputs')
    if old['matching_budget']!=16 or old['iterations']!=100 or old['graphs_per_update']!=32 or old['supervised_epochs']!=5:
        raise ValueError('Changed training opportunity contract')
    originals=load('teacher_original_inventory',parent.release_root/'scripts/analysis/sf_hlm_learning_improvement.py')
    inputs={str(a.parent_launch):sha(a.parent_launch),str(parent.results_root/'protocol.json'):sha(parent.results_root/'protocol.json'),
            str(a.state_root/'protocol.json'):sha(a.state_root/'protocol.json')}
    inputs.update({str(parent.release_root/rel):digest for rel,digest in old['files'].items()})
    inputs.update({str(a.feasibility_release/rel):digest for rel,digest in states['release_files'].items()})
    inputs[str(parent.extension_root/'utils/sf_hlm_resources.py')]=sha(parent.extension_root/'utils/sf_hlm_resources.py')
    models=[];teachers=[]
    for b in old['blocks']:
        schedules=[]
        dataset=parent.output/f'dataset/block_{b}/dataset.pt'
        if sha(dataset)!=old['inputs'][f'dataset_{b}']:raise ValueError('Training graph library changed')
        inputs[str(dataset)]=sha(dataset)
        for variant in ('baseline','multistep'):
            task=('train',variant,b,0,-1,-1)
            if not originals.verify_task(parent,old,task,full=True):raise ValueError('Missing complete training receipt')
            folder=originals.task_folder(parent,task);receipt=read(folder/'result.json')
            if receipt['dataset_sha256']!=sha(dataset):raise ValueError('Teacher variants used different training rows')
            losses=read(folder/'losses.json')
            schedule=[dict(iteration=z['iteration'],graph_indices=z['graph_indices'],trajectory_seeds=z['trajectory_seeds']) for z in losses if 'iteration' in z]
            if len(schedule)!=100 or any(len(z['graph_indices'])!=32 for z in schedule):raise ValueError('Incomplete paired training opportunity')
            schedules.append(schedule);inputs[str(folder/'result.json')]=sha(folder/'result.json')
            inputs[str(folder/'losses.json')]=sha(folder/'losses.json')
            for stage in ('bc','ppo'):
                method=variant+'_'+stage;sub='bc' if stage=='bc' else 'checkpoint';path=folder/sub/'quota_v2.pt'
                inputs[str(path)]=sha(path)
                models.append(dict(block=b,method=method,path=str(path),sha256=sha(path),
                    contract=receipt['checkpoint_contract'],supervised_samples=receipt['supervised_sample_count'],
                    unique_labels=receipt['unique_labels'],training_schedule_sha256=hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest()))
        if schedules[0]!=schedules[1]:raise ValueError('PPO graph/random opportunities unpaired')
        task=('teacher','multistep',b,0,-1,-1)
        if not originals.verify_task(parent,old,task,full=True):raise ValueError('Missing multi-step teacher receipt')
        folder=originals.task_folder(parent,task)
        for name in ('result.json','teacher.json','data.pt'):inputs[str(folder/name)]=sha(folder/name)
        td=read(folder/'teacher.json')
        if td['source_dataset_sha256']!=sha(dataset):raise ValueError('Offline teacher graph source changed')
        teachers.append(dict(block=b,**{k:td[k] for k in ('graphs','matching_calls','labels')},
            budget_per_graph=old['teacher_budget'],beam_width=old['beam_width'],branch_width=old['branch_width'],depth=old['teacher_depth']))
    state_records=[]
    for identity in states['identities']:
        sf=state_folder(a,f,identity);meta=read(sf/'state.json')
        td=f.task_dir(argparse.Namespace(results_root=a.state_root),['benchmark',*identity])
        if not f.completed(argparse.Namespace(results_root=a.state_root),['benchmark',*identity]):raise ValueError('Old state benchmark incomplete')
        if meta['identity']!=identity:raise ValueError('State identity changed')
        state_records.append(dict(identity=identity,state=str(sf/'state.json'),state_sha256=sha(sf/'state.json'),
                                  benchmark=str(td/'benchmark.json'),benchmark_sha256=sha(td/'benchmark.json')))
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Archived source/input changed: '+path)
    result=dict(status='PASS',states=state_records,models=models,teachers=teachers,input_files=inputs,
        no_training=True,confirmation_used=False,paired_training_opportunities=True,feasibility_release=str(a.feasibility_release))
    write(a.validation_root/'inventory.json',result)
    print(json.dumps(dict(status='PASS',states=len(state_records),checkpoints=len(models),teacher_records=len(teachers))),flush=True)
    return result


def init(a,f):
    inv=read(a.validation_root/'inventory.json')
    if inv['status']!='PASS':raise ValueError('Inventory not verified')
    ids=[s['identity'] for s in inv['states']]
    if a.smoke:ids=[i for i in ids if i[1] in (620001,620002) and i[2]==780001 and i[3]==0 and i[4] in (0,250)]
    p=dict(name='sf_hlm_teacher_same_state_20261017_v1',smoke=a.smoke,purpose='verification' if a.smoke else 'development',
        inventory_sha256=sha(a.validation_root/'inventory.json'),input_files=inv['input_files'],
        files={rel:sha(a.release_root/rel) for rel in FILES},identities=ids,methods=list(METHODS),candidates=list(CANDIDATES),
        states=[s for s in inv['states'] if s['identity'] in ids],models=inv['models'],offline_teacher=inv['teachers'],
        timing_repeats=3,random_repeats=5,matching_budget=16,base_repair_budget=128,
        jobs=[list(v) for v in sorted({tuple(i[:2]) for i in ids})],
        primary_attribution_pairs=[list(z) for z in TEACHER_PAIRS],
        gates=dict(quality_gain_vs_R16=.003,cost_gain_vs_G=.005,positive_blocks=4,energy_ratio=1.05,p95_seconds=1.),
        no_training=True,confirmation_used=False,state_delay_applied=False,no_automatic_next_stage=True,
        timing_contract='Full feasibility adapter: prepare+upper+repair+search+feedback+mapping+serialization; private restore/physics excluded',
        score_contract='Threshold every request at every measured cost then average timing/random repeats; denominator all generated',
        inference_unit='Five reused historical training blocks, not 400 states or timing repetitions')
    if (a.results_root/'protocol.json').exists():
        if read(a.results_root/'protocol.json')!=p:raise ValueError('Existing frozen protocol differs')
    else:write(a.results_root/'protocol.json',p)
    write(a.results_root/'status.json',dict(status='PREPARED',states=len(ids),updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',states=len(ids),protocol_sha256=sha(a.results_root/'protocol.json'))),flush=True)


def verify(a,p,external=False):
    for rel,digest in p['files'].items():
        if sha(a.release_root/rel)!=digest:raise ValueError('New frozen release changed: '+rel)
    if sha(a.validation_root/'inventory.json')!=p['inventory_sha256']:raise ValueError('Inventory changed')
    if external:
        for path,digest in p['input_files'].items():
            if sha(path)!=digest:raise ValueError('Archived input changed: '+path)


def complete(a,p,identity):
    folder=task_folder(a,identity);path=folder/'result.json'
    if not path.exists():return False
    receipt=read(path)
    if receipt['status']!='PASS' or receipt['identity']!=identity or receipt['protocol_sha256']!=sha(a.results_root/'protocol.json'):
        raise ValueError('Changed task receipt')
    if sha(folder/'benchmark.json')!=receipt['benchmark_sha256']:raise ValueError('Changed benchmark record')
    return True


def benchmark(a,p,f,c,identity,runtime):
    import config
    import numpy as np
    from utils.canonical_experiment import digest
    from utils.snapshot_evaluator import evaluate
    from utils.sf_hlm_learning_improvement import run_search
    _,_,runner,models,lowers,basis=runtime
    registered=next(z for z in p['states'] if z['identity']==identity)
    if sha(registered['state'])!=registered['state_sha256'] or sha(registered['benchmark'])!=registered['benchmark_sha256']:
        raise ValueError('State or parent benchmark changed')
    meta=read(registered['state']);inp=read(meta['input'])
    for path,digest_ in meta['files'].items():
        if sha(path)!=digest_:raise ValueError('Frozen state bytes changed')
    blob=Path(meta['observed']).read_bytes();snapshot=pickle.loads(Path(meta['snapshot']).read_bytes())
    expected=inp['rows']
    for row in expected:row['options']={int(k):v for k,v in row['options'].items()}
    env,obs=pickle.loads(blob);_,_,requests,rows=runner.base.prepare(env)
    if digest(rows)!=digest(expected) or [q[2] for q in requests]!=inp['request_ids'] or snapshot.snapshot_hash!=inp['snapshot_hash']:
        raise ValueError('Observed/private restoration mismatch')
    fingerprint=runner.base.policy_fingerprint([models['upper'],*[lowers[k] for k in ('old_L16',*CANDIDATES)]])
    equivalence=[]
    for method in CANDIDATES:
        x,d,records=c.truncated_search(rows,num_uavs=config.NUM_UAVS,kind='learned',model=lowers[method],cap=16)
        y,ref,ref_records=run_search(rows,num_uavs=config.NUM_UAVS,kind='learned',model=lowers[method],budget=16)
        if list(x)!=list(y) or [q['action'] for q in records]!=[q['action'] for q in ref_records]:raise ValueError('Archived candidate search differs')
        if any(d[key]!=ref[key] for key in ('matching_calls','base_evaluations','common_energy_cap','chosen_predicted','termination')):
            raise ValueError('Archived candidate guard differs')
        equivalence.append(dict(method=method,matching_calls=d['matching_calls'],actions=len(records)))
    specs=branch_specs(identity);original=read(registered['benchmark'])
    old_results={(q['method'],q['repeat']):q for q in original['results']};results={};crosschecks=0
    for spec in specs:
        env,obs=pickle.loads(blob);c.complete_policy(env,obs,spec,models,lowers.get(spec['method']),runner)
    for iteration in range(3):
        order=np.random.default_rng(seed_for('teacher_formal_order',identity,iteration)).permutation(len(specs))
        for idx in order:
            spec=specs[int(idx)];env,obs=pickle.loads(blob)
            motion,_,info,actual=c.complete_policy(env,obs,spec,models,lowers.get(spec['method']),runner)
            if digest(actual)!=digest(expected) or info['request_ids']!=inp['request_ids'] or not np.allclose(motion,snapshot.motion,rtol=0,atol=1e-7):
                raise ValueError('Formal state or upper action differs')
            d=info['diagnostic']
            if d['matching_calls']>spec['cap'] or d['chosen_predicted']['energy']>d['common_energy_cap']+1e-7:raise ValueError('Online feasibility guard violated')
            key=(spec['method'],spec['repeat']);result=results.get(key)
            if result is None:
                outcome=evaluate(snapshot,info['assignment']);metrics=outcome['metrics'];requests_out=outcome['requests']
                if int(metrics['generated_service_count'])!=meta['generated'] or len(requests_out)!=meta['admitted']:raise ValueError('Denominator changed')
                raw=c.fee_count(requests_out,0.)
                if raw!=int(metrics['deadline_satisfied_count']):raise ValueError('Raw success does not reproduce physical metrics')
                if spec['method'] in ('G','old_L16','R16'):
                    old=old_results[('L16' if spec['method']=='old_L16' else spec['method'],spec['repeat'])]
                    if list(info['assignment'])!=old['assignment'] or raw!=old['raw_success']:raise ValueError('Original control allocation changed')
                    if requests_out!=old['requests']:raise ValueError('Original control private outcome changed')
                    for field,value in old['metrics'].items():
                        if not math.isclose(float(metrics[field]),float(value),rel_tol=1e-10,abs_tol=1e-8):raise ValueError('Original control metric differs: '+field)
                    crosschecks+=1
                result=dict(**spec,assignment=list(info['assignment']),requests=requests_out,raw_success=raw,
                    generated=meta['generated'],admitted=meta['admitted'],metrics=metrics,diagnostic=d,software_seconds=[],timing_iteration=[],selected_order=[])
                results[key]=result
            elif result['assignment']!=list(info['assignment']):raise ValueError('Repeated timing changed assignment')
            result['software_seconds'].append(info['policy_seconds']);result['timing_iteration'].append(iteration);result['selected_order'].append(int(idx))
    final=runner.base.policy_fingerprint([models['upper'],*[lowers[k] for k in ('old_L16',*CANDIDATES)]])
    if fingerprint!=final or crosschecks!=7:raise ValueError('Frozen policy or reference controls not verified')
    for result in results.values():result['fee_success']=c.mean_fee_count(result['requests'],result['software_seconds'])
    folder=task_folder(a,identity);payload=dict(identity=identity,basis=basis,state_sha256=registered['state_sha256'],
        generated=meta['generated'],admitted=meta['admitted'],potential=meta['potential'],results=list(results.values()),
        model_fingerprint=fingerprint,candidate_equivalence=equivalence,control_crosschecks=crosschecks,
        timing_samples=33,cpu=sorted(os.sched_getaffinity(0)),load_average=os.getloadavg(),state_delay_applied=False)
    write(folder/'benchmark.json',payload)
    write(folder/'result.json',dict(status='PASS',identity=identity,protocol_sha256=sha(a.results_root/'protocol.json'),
        benchmark_sha256=sha(folder/'benchmark.json'),state=registered['state'],state_sha256=registered['state_sha256']))


def worker(a,f):
    p=read(a.results_root/'protocol.json');verify(a,p);job=json.loads(a.job)
    if job not in p['jobs']:raise ValueError('Unknown worker job')
    parent=runtime_paths(a,f);res=load('teacher_worker_resources',parent.extension_root/'utils/sf_hlm_resources.py')
    with res.task_lock(a.results_root/'locks'/('_'.join(map(str,job))+'.lock')):
        with res.CpuLease(parent.resources/'isolated',task=['teacher_same_state',*job],kind='environment'):
            old_args,old_runner,runner,models,old_lower,basis=f.boot(a,job[1])
            old_p=read(old_args.results_root/'protocol.json');lowers={'old_L16':old_lower}
            for method in CANDIDATES:lowers[method]=old_runner.load_candidate(old_args,old_p,method,job[1])
            runtime=old_args,old_runner,runner,models,lowers,basis;c=f.core(argparse.Namespace(release_root=a.feasibility_release))
            for i,identity in enumerate(z for z in p['identities'] if z[:2]==job):
                if not complete(a,p,identity):benchmark(a,p,f,c,identity,runtime)
                write(a.results_root/'progress'/('_'.join(map(str,job))+'.json'),dict(job=job,completed=i+1,last_identity=identity,updated_unix=time.time()))


def paired(values,f):
    from scipy.stats import t
    d=f.paired(values);sd=statistics.stdev(values)
    d['p']=float(2*t.sf(abs(statistics.mean(values))/(sd/math.sqrt(len(values))),len(values)-1)) if sd else (0. if statistics.mean(values) else 1.)
    return d


def analyze(a,p,f):
    import numpy as np
    c=f.core(argparse.Namespace(release_root=a.feasibility_release));rows=[]
    for identity in p['identities']:
        if not complete(a,p,identity):raise ValueError('Cannot aggregate incomplete population')
        state=read(task_folder(a,identity)/'benchmark.json');means={}
        c.require_population([(q['method'],q['repeat']) for q in state['results']],[(q['method'],q['repeat']) for q in branch_specs(identity)])
        for method in METHODS:
            trials=[q for q in state['results'] if q['method']==method]
            for trial in trials:
                if len(trial['software_seconds'])!=3 or not math.isclose(trial['fee_success'],c.mean_fee_count(trial['requests'],trial['software_seconds']),abs_tol=1e-12):raise ValueError('Incorrect fee score')
            means[method]=dict(raw=statistics.mean(q['raw_success'] for q in trials),fee=statistics.mean(q['fee_success'] for q in trials),
                energy=statistics.mean(q['metrics']['uav_fleet_total_energy'] for q in trials),
                mbs=statistics.mean(q['metrics']['service_offloads_mbs'] for q in trials),fairness=statistics.mean(q['metrics']['fairness'] for q in trials),
                calls=statistics.mean(q['diagnostic']['matching_calls'] for q in trials),times=[t for q in trials for t in q['software_seconds']])
        rows.append(dict(identity=identity,generated=state['generated'],admitted=state['admitted'],methods=means))
    subsets={}
    for source in ('all','greedy','learned'):
        sr=[s for s in rows if source=='all' or s['identity'][0]==source];blocks={}
        for b in sorted({s['identity'][1] for s in sr}):
            br=[s for s in sr if s['identity'][1]==b];den=sum(s['generated'] for s in br);ad=sum(s['admitted'] for s in br);methods={}
            for m in METHODS:
                ts=[v for s in br for v in s['methods'][m]['times']]
                methods[m]=dict(raw_rate=sum(s['methods'][m]['raw'] for s in br)/den,fee_rate=sum(s['methods'][m]['fee'] for s in br)/den,
                    energy=sum(s['methods'][m]['energy'] for s in br),mbs_share=sum(s['methods'][m]['mbs'] for s in br)/ad if ad else 0.,
                    fairness=statistics.mean(s['methods'][m]['fairness'] for s in br),calls=statistics.mean(s['methods'][m]['calls'] for s in br),
                    mean_seconds=statistics.mean(ts),p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
            blocks[str(b)]=dict(generated=den,admitted=ad,states=len(br),methods=methods)
        comparisons={};pairs=list(TEACHER_PAIRS)
        pairs += [(m,ref) for m in METHODS for ref in ('G','R16','old_L16') if m!=ref and (m,ref) not in pairs]
        for first,second in pairs:
            comparisons[first+'-'+second]={metric:paired([q['methods'][first][metric+'_rate']-q['methods'][second][metric+'_rate'] for q in blocks.values()],f) for metric in ('raw','fee')}
        interaction_stats={metric:paired([interaction({m:q['methods'][m][metric+'_rate'] for m in CANDIDATES}) for q in blocks.values()],f) for metric in ('raw','fee')}
        for metric in ('raw','fee'):
            adj=holm([comparisons[x+'-'+y][metric]['p'] for x,y in TEACHER_PAIRS])
            for (x,y),value in zip(TEACHER_PAIRS,adj):comparisons[x+'-'+y][metric]['holm_p_four_attribution_pairs']=value
        groups={m:{metric:statistics.mean(q['methods'][m][metric] for q in blocks.values())
            for metric in ('raw_rate','fee_rate','mean_seconds','p95_seconds','calls','mbs_share','fairness')} for m in METHODS}
        for m in METHODS:groups[m]['energy_ratio_G']=statistics.mean(q['methods'][m]['energy']/q['methods']['G']['energy'] for q in blocks.values())
        subsets[source]=dict(blocks=blocks,groups=groups,comparisons=comparisons,interaction=interaction_stats)
    all_=subsets['all'];m='multistep_ppo';blocks=all_['blocks'];comp=all_['comparisons']
    within=all(q['methods'][m]['p95_seconds']<1. for q in blocks.values())
    energy_random=statistics.mean(q['methods'][m]['energy']/q['methods']['R16']['energy'] for q in blocks.values())
    decision=classify(comp[m+'-R16']['raw'],comp[m+'-G']['fee'],energy_vs_random=energy_random,energy_vs_g=all_['groups'][m]['energy_ratio_G'],within_time=within)
    decision.update(quality_difference_vs_R16=comp[m+'-R16']['raw'],cost_difference_vs_G=comp[m+'-G']['fee'],
        energy_ratio_vs_R16=energy_random,energy_ratio_vs_G=all_['groups'][m]['energy_ratio_G'],within_time=within,
        all_four_cost_candidates={x:comp[x+'-G']['fee']['mean']>=.005 and comp[x+'-G']['fee']['positive']>=4 and
             all_['groups'][x]['energy_ratio_G']<=1.05 and all(q['methods'][x]['p95_seconds']<1. for q in blocks.values()) for x in CANDIDATES},
        no_automatic_next_stage=True,independent_confirmation=False)
    result=dict(status='PASS_SMOKE' if p['smoke'] else 'PASS',protocol_sha256=sha(a.results_root/'protocol.json'),states=len(rows),
        training_blocks=len(blocks),branch_instances=len(rows)*11,timing_samples=len(rows)*33,
        equivalence_checks=len(rows)*4,original_control_checks=len(rows)*7,subsets=subsets,decision=decision,
        no_training=True,state_delay_applied=False,independent_confirmation=False)
    write(a.results_root/'analysis/summary.json',result);write(a.results_root/'analysis/states.json',rows)
    report(a,result)
    write(a.results_root/'status.json',dict(status='COMPLETED_SMOKE' if p['smoke'] else 'COMPLETED',states=len(rows),updated_unix=time.time()))
    print(json.dumps(dict(status=result['status'],states=len(rows),decision=decision)),flush=True)


def report(a,result):
    s=result['subsets']['all'];lines=['**多步教师检查点：同状态、计全费用诊断**','',
        f"完成{result['states']}个复用开发状态、{result['branch_instances']}个分支实例、{result['timing_samples']}次正式计时；统计单位为{result['training_blocks']}个历史训练块。没有新训练，没有使用原确认结果。",'',
        '以下为采样状态的即时成功率，软件费用事后计入截止期；不是整回合DSR或等待回写后的闭环结果。','',
        '| 方法 | 原始成功率/% | 费用后成功率/% | 费用后相对G/pp | 平均/ms | 块p95均值/ms | 匹配次数 | 能耗比G |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,g in s['groups'].items():
        delta=s['comparisons'][m+'-G']['fee']['mean']*100 if m!='G' else 0.
        lines.append(f"| {m} | {g['raw_rate']*100:.3f} | {g['fee_rate']*100:.3f} | {delta:+.3f} | {g['mean_seconds']*1000:.2f} | {g['p95_seconds']*1000:.2f} | {g['calls']:.3f} | {g['energy_ratio_G']:.4f} |")
    lines+=['','BC是原监督初始化检查点，PPO是相同初始化后的100次原更新。多步教师使用额外离线搜索，梯度机会相同不代表总计算预算相同。G/旧L16/R16在本轮重新隔离计时，物理分配和结果须精确复现0b对照。','',
        '| 归因比较 | 原始差/pp及95% t区间 | 费用后差/pp及95% t区间 | 原始正向块 |','|---|---|---|---:|']
    for x,y in TEACHER_PAIRS:
        q=s['comparisons'][x+'-'+y];text=[]
        for metric in ('raw','fee'):
            d=q[metric];text.append(f"{d['mean']*100:+.3f} [{d['ci95'][0]*100:+.3f}, {d['ci95'][1]*100:+.3f}]")
        lines.append(f"| {x} − {y} | {text[0]} | {text[1]} | {q['raw']['positive']}/{result['training_blocks']} |")
    lines+=['','交互差直接比较两种初始化下PPO更新增量，不以“一组显著另一组不显著”代替交互。']
    for metric,d in s['interaction'].items():lines.append(f"- {metric}交互：{d['mean']*100:+.3f} pp，95% t区间[{d['ci95'][0]*100:+.3f}, {d['ci95'][1]*100:+.3f}]。")
    lines+=['','预定教师/PPO四比较的原始双侧t检验、各自Holm校正及精确双侧符号翻转均完整保存在summary.json。区间未按多重比较校正；五块小样本只作开发筛选。','',
        '**冻结闸门的执行结果：**','', '```json',json.dumps(result['decision'],ensure_ascii=False,indent=2),'```','',
        '质量侧门槛为多步PPO比R16原始成功率至少+0.3 pp且4/5块正向；费用侧为比G至少+0.5 pp且4/5块正向，并分别满足能耗/时间约束。两个目标分开判断，质量侧通过不能称费用反转解决。任何通过也不自动启动独立确认。','',
        '两种状态来源、逐块质量/费用/能耗、MBS占比、公平性、配对差值、原始时间及模型指纹随结果保存。完整摘要位于同任务teacher_20261017/release1/analysis/summary.json；逐任务记录位于tasks/。']
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')


def command(a,job):
    args=[sys.executable,str(a.release_root/FILES[1]),'worker']
    for key in ('release_root','feasibility_release','parent_launch','state_root','results_root','validation_root','report'):
        args += ['--'+key.replace('_','-'),str(getattr(a,key))]
    return args+['--job',json.dumps(job,separators=(',',':'))]


def pipeline(a,p,f):
    parent=runtime_paths(a,f);profile=read(parent.resources/'profile.json');os.sched_setaffinity(0,set(profile['cpus'][:profile['slots']]))
    res=load('teacher_controller_resources',parent.extension_root/'utils/sf_hlm_resources.py')
    with res.task_lock(a.results_root/'controller.lock'):
        verify(a,p,external=True)
        write(a.validation_root/('smoke_environment.json' if p['smoke'] else 'execution_environment.json'),dict(
            python=sys.version,uname=list(os.uname()),isolated_profile=read(parent.resources/'isolated/profile.json'),
            cpuinfo=Path('/proc/cpuinfo').read_text().split('\n\n')[0],threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},pid=os.getpid(),started_unix=time.time()))
        def run(job):
            path=a.results_root/'worker_logs'/('_'.join(map(str,job))+'.log');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as log:ret=subprocess.run(command(a,job),cwd=a.release_root,stdout=log,stderr=subprocess.STDOUT)
            if ret.returncode:raise RuntimeError('Worker failed: '+str(job))
            if not all(complete(a,p,i) for i in p['identities'] if i[:2]==job):raise ValueError('Incomplete source/block worker')
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(run,job) for job in p['jobs']]
                for future in as_completed(futures):
                    future.result();n=sum(complete(a,p,i) for i in p['identities'])
                    write(a.results_root/'status.json',dict(status='RUNNING',completed=n,total=len(p['identities']),updated_unix=time.time()))
            analyze(a,p,f)
        except BaseException as error:
            write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('inventory','init','pipeline','worker','analyze'))
    for key in ('release_root','feasibility_release','parent_launch','state_root','results_root','validation_root','report'):
        ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--job');a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Simulation, timing and scientific aggregation run only on remote Linux')
    f=archived(a)
    if a.mode=='inventory':inventory(a,f)
    elif a.mode=='init':init(a,f)
    elif a.mode=='worker':worker(a,f)
    else:
        p=read(a.results_root/'protocol.json')
        if a.mode=='pipeline':pipeline(a,p,f)
        else:verify(a,p);analyze(a,p,f)


if __name__=='__main__':main()
