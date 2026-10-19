"""Remote paired replay: archived, cached dense and legal-edge sparse runtimes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import importlib.util
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

FILES=('paths.py','scripts/analysis/sf_hlm_runtime_feasibility.py','utils/sf_hlm_fast_runtime.py',
       'tests/test_sf_hlm_fast_runtime.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
BASES=('G','L16','L8','R16','R8')
CANDIDATES=('fast_L16','fast_L8','sparse_L16','sparse_L8')
METHODS=tuple('legacy_'+m for m in BASES)+tuple('fast_'+m for m in BASES)+('sparse_L16','sparse_L8')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def normal(x):return json.loads(json.dumps(x,default=serial,allow_nan=False))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=serial,allow_nan=False)+'\n');tmp.replace(path)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def seed_for(*x):return int.from_bytes(hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).digest()[:4],'big')


def specs(identity):
    return [dict(method=impl+'_'+base,base_method=base,implementation=impl,
        kind='greedy' if base=='G' else 'random' if base.startswith('R') else 'learned',cap=0 if base=='G' else int(base[1:]),
        repeat=i,seed=seed_for('sf_hlm_feasibility_20261016',identity,i))
        for impl in ('legacy','fast','sparse') for base in (BASES if impl!='sparse' else ('L16','L8'))
        for i in range(5 if base.startswith('R') else 1)]


def decision(comparison,energy_ratio,within_time):
    return comparison['mean']>=.005 and comparison['positive']>=4 and energy_ratio<=1.05 and within_time


def archive(a):return load('runtime_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
def folder(a,identity):return a.results_root/'tasks'/('benchmark_'+'_'.join(map(str,identity)))


def init(a,f):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('New protocol namespace required')
    old=read(a.state_root/'protocol.json');identities=old['identities']
    if a.smoke:identities=[i for i in identities if i[1] in (620001,620002) and i[2]==780001 and i[3]==0 and i[4] in (0,250)]
    parent=f.parent_args(a);inputs={str(a.parent_launch):sha(a.parent_launch),str(a.state_root/'protocol.json'):sha(a.state_root/'protocol.json')}
    inputs.update(old['input_files']);inputs.update(old['parent_release_files'])
    inputs.update({str(a.feasibility_release/rel):digest for rel,digest in old['release_files'].items()})
    records=[]
    for identity in identities:
        old_args=argparse.Namespace(results_root=a.state_root);task=['benchmark',*identity]
        if not f.completed(old_args,task):raise ValueError('Missing verified original branch population')
        td=f.task_dir(old_args,task);sf=f.state_dir(old_args,identity)
        records.append(dict(identity=identity,state=str(sf/'state.json'),state_sha256=sha(sf/'state.json'),
            benchmark=str(td/'benchmark.json'),benchmark_sha256=sha(td/'benchmark.json')))
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Changed frozen input: '+path)
    p=dict(name='sf_hlm_runtime_20261017_v1',smoke=a.smoke,purpose='verification' if a.smoke else 'development',
        identities=identities,states=records,jobs=[list(z) for z in sorted({tuple(i[:2]) for i in identities})],
        methods=list(METHODS),candidates=list(CANDIDATES),random_repeats=5,timing_repeats=3,
        branch_instances_per_state=28,timing_samples_per_state=84,input_files=inputs,
        files={rel:sha(a.release_root/rel) for rel in FILES},gates=dict(practical_gain=.005,learning_increment=.001,positive_blocks=4,energy_ratio=1.05,p95_seconds=1.),
        input_budget=16,actual_caps=[8,16],repair_budget=128,sparse_gap_fallback=1e-4,sparse_logit_tolerance=1e-5,
        graph_parity='bitwise exact',allocation_parity='exact against archived branches at every timing sample',
        physical_outcomes='reuse hashed original per-request outcomes only after exact state and assignment parity',
        timing='full online path including fresh per-call matrices/caches, sparse fallback and unchanged serialization',
        no_training=True,confirmation_used=False,state_delay_applied=False,goal_continues_if_gate_fails=True,
        statistical_unit='five reused historical training blocks; repetitions/states are subsamples')
    write(a.results_root/'protocol.json',p);write(a.results_root/'status.json',dict(status='PREPARED',states=len(identities),updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',states=len(identities),protocol_sha256=sha(a.results_root/'protocol.json'))),flush=True)


def verify(a,p,external=False):
    for rel,digest in p['files'].items():
        if sha(a.release_root/rel)!=digest:raise ValueError('Changed new release: '+rel)
    if external:
        for path,digest in p['input_files'].items():
            if sha(path)!=digest:raise ValueError('Changed frozen input: '+path)


def completed(a,p,identity):
    path=folder(a,identity)/'result.json'
    if not path.exists():return False
    q=read(path)
    if q['identity']!=identity or q['status']!='PASS' or q['protocol_sha256']!=sha(a.results_root/'protocol.json'):
        raise ValueError('Receipt identity/protocol changed')
    if q['benchmark_sha256']!=sha(path.parent/'benchmark.json'):raise ValueError('Record hash changed')
    return True


def execute(fast,core,spec,blob,models,lower,runner):
    env,obs=pickle.loads(blob)
    if spec['implementation']=='legacy':return core.complete_policy(env,obs,spec,models,lower,runner)
    with fast.FastRuntime() as runtime:
        runtime.patch(core,'truncated_search',runtime.truncated_search)
        if spec['implementation']=='sparse':runtime.enable_sparse(lower)
        return runtime.complete_policy(core,env,obs,spec,models,lower,runner)


def compatibility(rows,identity,core,fast,lower):
    import numpy as np
    import torch
    originals={};checks=[]
    for spec in specs(identity):
        if spec['implementation']!='legacy' or spec['base_method']=='G':continue
        out=core.truncated_search(rows,num_uavs=5,kind=spec['kind'],model=lower,
            rng=np.random.default_rng(spec['seed']),cap=spec['cap'])
        originals[spec['base_method'],spec['repeat']]=out
    for spec in specs(identity):
        if spec['implementation']=='legacy' or spec['base_method']=='G':continue
        ref,rd,rr=originals[spec['base_method'],spec['repeat']];dense_forward=lower.forward_states
        with fast.FastRuntime() as runtime:
            if spec['implementation']=='sparse':runtime.enable_sparse(lower)
            actual,ad,ar=runtime.truncated_search(rows,num_uavs=5,kind=spec['kind'],model=lower,
                rng=np.random.default_rng(spec['seed']),cap=spec['cap'],trace=True)
            if list(ref)!=list(actual) or [z['action'] for z in rr]!=[z['action'] for z in ar]:raise ValueError('Search trajectory differs: '+spec['method'])
            for key in ('chosen_predicted','reference_predicted','matching_calls','base_evaluations','termination'):
                if normal(rd[key])!=normal(ad[key]):raise ValueError('Search diagnostic differs: '+key)
            graph_count=0;max_error=0.
            if spec['kind']=='learned':
                for old,new in zip(rr,ar):
                    for key in old['state']:
                        if not np.array_equal(old['state'][key],new['state'][key]):raise ValueError('Graph differs: '+key)
                    graph_count+=1
                    if spec['implementation']=='sparse':
                        with torch.no_grad():
                            ref_logits,ref_value=dense_forward([old['state']]);logits,value=lower.forward_states([new['state']])
                        error=float((ref_logits-logits).abs().max());max_error=max(max_error,error)
                        if error>1e-5 or int(ref_logits.argmax(-1)[0])!=int(logits.argmax(-1)[0]):raise ValueError('Sparse logits or argmax outside frozen guard')
            checks.append(dict(method=spec['method'],repeat=spec['repeat'],graph_checks=graph_count,max_logit_error=max_error,actions=len(ar)))
    return checks


def benchmark(a,p,f,fast,identity,runtime):
    import numpy as np
    from utils.canonical_experiment import digest
    _,_,runner,models,lower,basis=runtime;core=f.core(argparse.Namespace(release_root=a.feasibility_release))
    record=next(z for z in p['states'] if z['identity']==identity)
    if sha(record['state'])!=record['state_sha256'] or sha(record['benchmark'])!=record['benchmark_sha256']:raise ValueError('Source state record changed')
    meta=read(record['state']);original=read(record['benchmark']);inp=read(meta['input'])
    for path,checksum in meta['files'].items():
        if sha(path)!=checksum:raise ValueError('Frozen state bytes changed')
    blob=Path(meta['observed']).read_bytes();env,obs=pickle.loads(blob);_,_,requests,rows=runner.base.prepare(env)
    expected=inp['rows']
    for row in expected:row['options']={int(k):v for k,v in row['options'].items()}
    if digest(rows)!=digest(expected) or [q[2] for q in requests]!=inp['request_ids']:raise ValueError('Observed input restoration differs')
    fingerprint=runner.base.policy_fingerprint([models['upper'],lower])
    if fingerprint!=original['model_fingerprint']:raise ValueError('Original frozen weights differ')
    checks=compatibility(rows,identity,core,fast,lower)
    references={(q['method'],q['repeat']):q for q in original['results']};branches=specs(identity);results={}
    for spec in branches:
        motion,_,info,actual=execute(fast,core,spec,blob,models,lower,runner)
        ref=references[spec['base_method'],spec['repeat']]
        if list(info['assignment'])!=ref['assignment']:raise ValueError('Warmup allocation changed')
    for iteration in range(3):
        order=np.random.default_rng(seed_for('runtime_formal_order',identity,iteration)).permutation(len(branches))
        for idx in order:
            spec=branches[int(idx)];motion,_,info,actual=execute(fast,core,spec,blob,models,lower,runner)
            ref=references[spec['base_method'],spec['repeat']]
            if digest(actual)!=digest(expected) or list(info['assignment'])!=ref['assignment'] or not np.allclose(motion,inp['motion'],rtol=0,atol=1e-7):
                raise ValueError('Formal observed/allocation/upper parity failed')
            if info['request_ids']!=inp['request_ids']:raise ValueError('Request order changed')
            for key in ('chosen_predicted','reference_predicted','common_energy_cap','base_evaluations','matching_calls','matching_budget'):
                if normal(info['diagnostic'][key])!=normal(ref['diagnostic'][key]):raise ValueError('Formal diagnostic differs: '+key)
            key=spec['method'],spec['repeat'];q=results.get(key)
            if q is None:
                q=dict(**spec,assignment=list(info['assignment']),raw_success=ref['raw_success'],software_seconds=[],timing_iteration=[],selected_order=[],runtime_stats=[])
                results[key]=q
            q['software_seconds'].append(info['policy_seconds']);q['timing_iteration'].append(iteration);q['selected_order'].append(int(idx))
            q['runtime_stats'].append(info.get('runtime_stats',dict(sparse_forwards=0,dense_fallbacks=0)))
    if fingerprint!=runner.base.policy_fingerprint([models['upper'],lower]):raise ValueError('Frozen weights changed')
    for q in results.values():
        ref=references[q['base_method'],q['repeat']];q['fee_success']=core.mean_fee_count(ref['requests'],q['software_seconds'])
    payload=dict(identity=identity,state_sha256=record['state_sha256'],reference_benchmark=record['benchmark'],reference_benchmark_sha256=record['benchmark_sha256'],
        generated=original['generated'],admitted=original['admitted'],potential=original['potential'],basis=basis,model_fingerprint=fingerprint,
        compatibility=checks,results=list(results.values()),timing_samples=84,cpu=sorted(os.sched_getaffinity(0)),load_average=os.getloadavg(),state_delay_applied=False,
        physical_outcomes_reused_after_exact_assignment=True)
    td=folder(a,identity);write(td/'benchmark.json',payload);write(td/'result.json',dict(status='PASS',identity=identity,
        protocol_sha256=sha(a.results_root/'protocol.json'),benchmark_sha256=sha(td/'benchmark.json')))


def worker(a,p,f):
    verify(a,p);job=json.loads(a.job)
    if job not in p['jobs']:raise ValueError('Unregistered source/block worker')
    parent=f.parent_args(a);res=load('runtime_resources',parent.extension_root/'utils/sf_hlm_resources.py')
    with res.task_lock(a.results_root/'locks'/('_'.join(map(str,job))+'.lock')):
        with res.CpuLease(parent.resources/'isolated',task=['fast_runtime',*job],kind='environment'):
            runtime=f.boot(a,job[1]);fast=load('fast_runtime_core',a.release_root/'utils/sf_hlm_fast_runtime.py')
            for n,identity in enumerate(i for i in p['identities'] if i[:2]==job):
                if not completed(a,p,identity):benchmark(a,p,f,fast,identity,runtime)
                write(a.results_root/'progress'/('_'.join(map(str,job))+'.json'),dict(completed=n+1,last_identity=identity,updated_unix=time.time()))


def analyze(a,p,f):
    import numpy as np
    core=f.core(argparse.Namespace(release_root=a.feasibility_release));states=[];graph_checks=0;max_error=0.
    for identity in p['identities']:
        if not completed(a,p,identity):raise ValueError('Incomplete population')
        b=read(folder(a,identity)/'benchmark.json')
        if sha(b['reference_benchmark'])!=b['reference_benchmark_sha256']:raise ValueError('Physical reference changed')
        refs={(q['method'],q['repeat']):q for q in read(b['reference_benchmark'])['results']}
        core.require_population([(q['method'],q['repeat']) for q in b['results']],[(q['method'],q['repeat']) for q in specs(identity)])
        graph_checks+=sum(z['graph_checks'] for z in b['compatibility']);max_error=max(max_error,max(z['max_logit_error'] for z in b['compatibility']))
        means={}
        for m in METHODS:
            trials=[q for q in b['results'] if q['method']==m];references=[refs[q['base_method'],q['repeat']] for q in trials]
            for q,ref in zip(trials,references):
                if len(q['software_seconds'])!=3 or not math.isclose(q['fee_success'],core.mean_fee_count(ref['requests'],q['software_seconds']),abs_tol=1e-12):raise ValueError('Incomplete or incorrect cost score')
            means[m]=dict(raw=statistics.mean(q['raw_success'] for q in trials),fee=statistics.mean(q['fee_success'] for q in trials),
                energy=statistics.mean(q['metrics']['uav_fleet_total_energy'] for q in references),
                mbs=statistics.mean(q['metrics']['service_offloads_mbs'] for q in references),fairness=statistics.mean(q['metrics']['fairness'] for q in references),
                calls=statistics.mean(q['diagnostic']['matching_calls'] for q in references),times=[v for q in trials for v in q['software_seconds']],
                sparse_forwards=statistics.mean(z['sparse_forwards'] for q in trials for z in q['runtime_stats']),
                dense_fallbacks=statistics.mean(z['dense_fallbacks'] for q in trials for z in q['runtime_stats']))
        states.append(dict(identity=identity,generated=b['generated'],admitted=b['admitted'],methods=means))
    subsets={}
    pairs=[(m,'fast_G') for m in METHODS if m!='fast_G']
    pairs += [(m,ref) for m in CANDIDATES for ref in ('fast_R8','fast_R16')]
    pairs += [('fast_'+m,'legacy_'+m) for m in BASES]
    pairs += [('sparse_'+m,'fast_'+m) for m in ('L16','L8')]
    for source in ('all','greedy','learned'):
        sr=[s for s in states if source=='all' or s['identity'][0]==source];blocks={}
        for block in sorted({s['identity'][1] for s in sr}):
            br=[s for s in sr if s['identity'][1]==block];den=sum(s['generated'] for s in br);ad=sum(s['admitted'] for s in br);ms={}
            for m in METHODS:
                ts=[v for s in br for v in s['methods'][m]['times']]
                ms[m]=dict(raw_rate=sum(s['methods'][m]['raw'] for s in br)/den,fee_rate=sum(s['methods'][m]['fee'] for s in br)/den,
                    energy=sum(s['methods'][m]['energy'] for s in br),mbs_share=sum(s['methods'][m]['mbs'] for s in br)/ad if ad else 0.,
                    calls=statistics.mean(s['methods'][m]['calls'] for s in br),fairness=statistics.mean(s['methods'][m]['fairness'] for s in br),
                    mean_seconds=statistics.mean(ts),p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts),
                    sparse_forwards=statistics.mean(s['methods'][m]['sparse_forwards'] for s in br),dense_fallbacks=statistics.mean(s['methods'][m]['dense_fallbacks'] for s in br))
            blocks[str(block)]=dict(generated=den,admitted=ad,states=len(br),methods=ms)
        comparisons={x+'-'+y:{metric:f.paired([z['methods'][x][metric+'_rate']-z['methods'][y][metric+'_rate'] for z in blocks.values()]) for metric in ('raw','fee')} for x,y in pairs}
        groups={m:{key:statistics.mean(z['methods'][m][key] for z in blocks.values()) for key in ('raw_rate','fee_rate','mbs_share','calls','fairness','mean_seconds','p95_seconds','sparse_forwards','dense_fallbacks')} for m in METHODS}
        for m,g in groups.items():g['energy_ratio_fast_G']=statistics.mean(z['methods'][m]['energy']/z['methods']['fast_G']['energy'] for z in blocks.values())
        subsets[source]=dict(blocks=blocks,groups=groups,comparisons=comparisons)
    s=subsets['all'];best=max(('fast_G','fast_R8','fast_R16'),key=lambda m:s['groups'][m]['fee_rate']);gates={}
    for m in CANDIDATES:
        d=s['comparisons'][m+'-fast_G']['fee'];er=s['groups'][m]['energy_ratio_fast_G'];within=all(z['methods'][m]['p95_seconds']<1. for z in s['blocks'].values())
        increment=s['groups'][m]['fee_rate']-s['groups'][best]['fee_rate']
        gates[m]=dict(practical=decision(d,er,within),fee_difference_vs_fast_G=d,energy_ratio_fast_G=er,within_time=within,
            best_fast_nonlearning=best,learning_increment=increment,learning_increment_target_met=increment>=.001)
    result=dict(status='PASS_SMOKE' if p['smoke'] else 'PASS',protocol_sha256=sha(a.results_root/'protocol.json'),states=len(states),
        branch_instances=len(states)*28,timing_samples=len(states)*84,training_blocks=len(s['blocks']),graph_checks=graph_checks,max_sparse_logit_error=max_error,
        subsets=subsets,decision=dict(candidates=gates,best_fast_nonlearning=best,goal_complete=False,not_independent_confirmation=True),
        no_training=True,state_delay_applied=False)
    write(a.results_root/'analysis/summary.json',result);write(a.results_root/'analysis/states.json',states);report(a,result)
    write(a.results_root/'status.json',dict(status='COMPLETED_SMOKE' if p['smoke'] else 'COMPLETED',states=len(states),updated_unix=time.time()))
    print(json.dumps(dict(status=result['status'],states=len(states),decision=result['decision'])),flush=True)


def report(a,result):
    s=result['subsets']['all'];lines=['**SF-HLM运行实现优化：开发比较**','',
        f"{result['states']}状态、{result['branch_instances']}分支实例、{result['timing_samples']}次完整费用测量。图逐位核对{result['graph_checks']}次；稀疏logits最大差{result['max_sparse_logit_error']:.3g}。",'',
        'legacy为原实现，fast为公共计算加速且保留原密集网络，sparse再使用合法边/动作稀疏推理及近并列密集回退。G/R8/R16共享同一公共加速。原始分配与物理结果完全相同后才复用原逐请求时延。所有输入矩阵与特征构造计入逐次时间。','',
        '| 方法 | 原始成功率/% | 费用后成功率/% | 相对fast_G/pp | 均值/ms | 块p95均值/ms | 匹配次数 | 稀疏回退/调用 | 能耗比fast_G |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,g in s['groups'].items():
        d=(g['fee_rate']-s['groups']['fast_G']['fee_rate'])*100
        lines.append(f"| {m} | {g['raw_rate']*100:.3f} | {g['fee_rate']*100:.3f} | {d:+.3f} | {g['mean_seconds']*1000:.2f} | {g['p95_seconds']*1000:.2f} | {g['calls']:.3f} | {g['dense_fallbacks']:.3f} | {g['energy_ratio_fast_G']:.4f} |")
    lines+=['','冻结开发闸门：','', '```json',json.dumps(result['decision'],indent=2,ensure_ascii=False),'```','',
        '统计单位为复用历史训练块，报告块级配对t区间与精确符号翻转；不是独立确认。结果是指定状态集合的即时成功率，不是整回合DSR；软件等待未回写环境。cProfile时间未混入正式费用。','',
        '本轮完成不等于持续目标完成。后续根据已冻结门槛推进未用负载闭环或下一项成本感知学习实验，保留本轮全部结果，不放宽原门槛。']
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')


def command(a,job):
    out=[sys.executable,str(a.release_root/FILES[1]),'worker']
    for key in ('release_root','feasibility_release','parent_launch','state_root','results_root','validation_root','report'):
        out += ['--'+key.replace('_','-'),str(getattr(a,key))]
    return out+['--job',json.dumps(job,separators=(',',':'))]


def pipeline(a,p,f):
    parent=f.parent_args(a);profile=read(parent.resources/'profile.json');os.sched_setaffinity(0,set(profile['cpus'][:profile['slots']]))
    res=load('runtime_controller_resources',parent.extension_root/'utils/sf_hlm_resources.py')
    with res.task_lock(a.results_root/'controller.lock'):
        verify(a,p,external=True)
        write(a.validation_root/('smoke_environment.json' if p['smoke'] else 'execution_environment.json'),dict(python=sys.version,uname=list(os.uname()),
            isolated_profile=read(parent.resources/'isolated/profile.json'),cpuinfo=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},pid=os.getpid(),started_unix=time.time()))
        def run(job):
            path=a.results_root/'worker_logs'/('_'.join(map(str,job))+'.log');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as log:out=subprocess.run(command(a,job),cwd=a.release_root,stdout=log,stderr=subprocess.STDOUT)
            if out.returncode:raise RuntimeError('Worker failed: '+str(job))
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(run,j) for j in p['jobs']]
                for future in as_completed(futures):
                    future.result();n=sum(completed(a,p,i) for i in p['identities'])
                    write(a.results_root/'status.json',dict(status='RUNNING',completed=n,total=len(p['identities']),updated_unix=time.time()))
            analyze(a,p,f)
        except BaseException as error:
            write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('init','worker','pipeline','analyze'))
    for key in ('release_root','feasibility_release','parent_launch','state_root','results_root','validation_root','report'):
        ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--job');a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Remote simulation/timing/aggregation only')
    f=archive(a)
    if a.mode=='init':init(a,f)
    else:
        p=read(a.results_root/'protocol.json')
        if a.mode=='worker':worker(a,p,f)
        elif a.mode=='pipeline':pipeline(a,p,f)
        else:verify(a,p);analyze(a,p,f)


if __name__=='__main__':main()
