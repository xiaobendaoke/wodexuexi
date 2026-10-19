"""Native complete-path timing of the frozen energy-priced matching policy."""
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

BASES=('G','old_pool_fee','binary','smooth','gcap_binary','gcap_smooth','binary_price','smooth_price','probability_price')
METHODS=BASES+('R8','R16')
NONLEARNING=tuple(m for m in METHODS if m!='probability_price')
FILES=('paths.py','scripts/analysis/sf_hlm_priced_runtime.py','utils/sf_hlm_priced_runtime.py','utils/sf_hlm_fast_runtime.py',
    'utils/sf_hlm_calibrated_matching.py','utils/sf_hlm_request_risk.py','utils/sf_hlm_action_ranker.py','utils/sf_hlm_selector.py','utils/sf_hlm_energy_price.py',
    'utils/sf_hlm_feasibility.py','tests/test_sf_hlm_priced_runtime.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARGS=('release_root','price_results','price_calibration','price_validation','matching_results','training_root',
    'feasibility_release','parent_launch','results_root','validation_root','resources','resource_module')

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
def seed_for(*parts):return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).digest()[:4],'big')
def archive(a):return load('priced_runtime_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
def folder(a,identity):return a.results_root/'tasks'/('timed_'+'_'.join(map(str,identity)))
def specs(identity):
    return [dict(method=m,kind='native_G' if m=='G' else m,cap=0 if m=='G' else 10,repeat=0,seed=seed_for('sf_hlm_feasibility_20261016',identity,0)) for m in BASES]+[
        dict(method=m,kind='random',cap=int(m[1:]),repeat=i,seed=seed_for('sf_hlm_feasibility_20261016',identity,i)) for m in ('R8','R16') for i in range(5)]

def init(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable timing namespace')
    audit=read(a.price_validation/'raw_record_audit.json');price=read(a.price_results/'protocol.json');decision=read(a.price_results/'analysis/summary.json')['decision']
    if price['smoke'] or audit['status']!='PASS' or audit['summary_sha256']!=sha(a.price_results/'analysis/summary.json') or not decision['optimistic_signal']:raise ValueError('Price policy not verified and promoted')
    identities=price['identities']
    if a.smoke:identities=[i for i in identities if i[1] in (620001,620002) and i[2]==780001 and i[3]==0 and i[4] in (0,250)]
    inputs={str(a.price_results/'protocol.json'):sha(a.price_results/'protocol.json'),str(a.price_results/'analysis/summary.json'):sha(a.price_results/'analysis/summary.json'),
        str(a.price_validation/'raw_record_audit.json'):sha(a.price_validation/'raw_record_audit.json'),str(a.parent_launch):sha(a.parent_launch),
        str(a.price_calibration/'protocol.json'):sha(a.price_calibration/'protocol.json'),str(a.price_calibration/'completion.json'):sha(a.price_calibration/'completion.json')}
    inputs.update(read(a.price_calibration/'completion.json')['files']);inputs.update(read(a.training_root/'completion.json')['files'])
    states=[]
    for state in price['states']:
        identity=state['identity']
        if identity not in identities:continue
        record=dict(state)
        for key,root,prefix in (('price',a.price_results,'priced_'),('matching',a.matching_results,'matched_')):
            td=root/'tasks'/(prefix+'_'.join(map(str,identity)));rec=read(td/'result.json')
            for name,digest in rec['files'].items():
                if sha(td/name)!=digest:raise ValueError('Reference result changed')
                inputs[str(td/name)]=digest
            record[key+'_folder']=str(td)
        states.append(record)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Timing input changed')
    p=dict(name='sf_hlm_priced_native_full_path_20261017_logging_v2',smoke=a.smoke,identities=identities,states=states,
        jobs=[list(z) for z in sorted({tuple(i[:2]) for i in identities})],files={file:sha(a.release_root/file) for file in FILES},input_files=inputs,
        methods=list(METHODS),nonlearning=list(NONLEARNING),branch_instances=19,timing_repeats=3,random_repeats=5,timing_samples_per_state=57,
        nominal_target_cost_ms=10.,actual_cost='complete policy wall time for each call, no assumed added delay',
        costs_include='fresh prepare,repair,features,HGB/matching/price,feedback,upper,mapping,logical JSON serialization; per-call cache clear',
        costs_exclude='snapshot restore,model loading,private physics; actual network signal transmission not measured',
        graph_feature_parity='bitwise exact local41',probability_parity='bitwise exact scalar versus batched HGB',allocation_parity='exact all methods/repeats/calls',
        logging_schema='full_order_ranks_monotonic_observation_v2',cache_lifecycle='adapter persists per state; previous dynamic cache is cleared inside each policy timer',
        gates=dict(gain_vs_G=.005,positive_blocks=4,learning_increment=.001,energy_ratio=1.05,per_block_p95_seconds=1.),
        state_delay_applied=False,independent_confirmation=False,goal_complete=False)
    write(a.results_root/'protocol.json',p);write(a.results_root/'status.json',dict(status='PREPARED',states=len(states),updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',states=len(states),timing_samples=len(states)*57)),flush=True)

def verify(a,p,external=False):
    for file,digest in p['files'].items():
        if sha(a.release_root/file)!=digest:raise ValueError('Frozen native source changed')
    if external:
        for path,digest in p['input_files'].items():
            if sha(path)!=digest:raise ValueError('Frozen timing input changed')

def completed(a,p,identity):
    path=folder(a,identity)/'result.json'
    if not path.exists():return False
    rec=read(path)
    if rec['status']!='PASS' or rec['identity']!=identity or rec['protocol_sha256']!=sha(a.results_root/'protocol.json'):raise ValueError('Timing receipt changed')
    if sha(path.parent/'benchmark.json')!=rec['benchmark_sha256']:raise ValueError('Timing record changed')
    return True

def execute(core,fast,native,spec,blob,models,lower,runner,scorer,prices,prepared_runtime=None):
    env,obs=pickle.loads(blob)
    if prepared_runtime is not None:return prepared_runtime.complete_policy(core,env,obs,spec,models,lower,runner)
    with fast.FastRuntime() as runtime:
        policy=native.PricedPolicy(runtime,scorer,prices);runtime.patch(core,'truncated_search',policy.run)
        return runtime.complete_policy(core,env,obs,spec,models,lower,runner)

def compatibility(rows,fast,native,model,prices,public):
    import numpy as np
    from utils import sf_hlm_calibrated_matching as cm
    from utils.sf_hlm_action_ranker import features
    from utils.sf_hlm_request_risk import proposal_actions
    checks=0
    with fast.FastRuntime() as rt:
        policy=native.PricedPolicy(rt,model,prices);g,repair,base=policy.initialize(rows)
        old=rt.search_class(rows,num_uavs=5,matching_budget=16,base_repair_budget=128)
        for key in ('assignment','current_assignment','best','reference','cap','current'): 
            if normal(getattr(base,key))!=normal(getattr(old,key)):raise ValueError('Single-repair initialization differs: '+key)
        if not np.array_equal(base.action_mask(),old.action_mask()):raise ValueError('Initialization changed mask')
        legal=[int(k) for k in np.flatnonzero(base.action_mask()[1:])+1]
        safe=dict(rows=rows,greedy_assignment=g,initial_assignment=base.assignment,initial_counts=base.current.tolist(),energy_cap=base.cap,legal_nonstop_actions=legal)
        ids,x=features(safe,10.,problem=base.problem);proposed=proposal_actions(ids,x);quotas=cm.quota_set(base.current.tolist(),proposed)
        if normal(proposed)!=public['proposals'] or normal(quotas)!=public['quotas']:raise ValueError('Native quota proposals differ')
        kernel=native.LocalEdgeKernel(rows,base.current,10.,base.problem);matrices=kernel.probabilities([q for _,q in quotas],model)
        for _,q in quotas:
            expected,index,_,_=cm.edge_features(rows,q,10.,problem=base.problem)
            if not np.array_equal(index,kernel.index) or not np.array_equal(expected,kernel.features(q)):raise ValueError('Local41 batch features are not bitwise equal')
            if len(rows):
                pp=model.predict_proba(expected)[:,list(model.classes_).index(1.)];matrix=matrices[tuple(q)]
                if not np.array_equal(pp,matrix[index[:,0],index[:,1]]):raise ValueError('Batched HGB probability differs')
            checks+=1
    return dict(feature_quota_checks=checks,probability_quota_checks=checks,initialization_equal=True)

def benchmark(a,p,identity,runtime,f,fast,native):
    import numpy as np
    from utils.canonical_experiment import digest
    core=f.core(argparse.Namespace(release_root=a.feasibility_release));_,_,runner,models,lower,basis=runtime;record=next(s for s in p['states'] if s['identity']==identity)
    if sha(record['state'])!=record['state_sha256'] or sha(record['benchmark'])!=record['benchmark_sha256']:raise ValueError('Timing state changed')
    meta=read(record['state']);inp=read(meta['input']);original=read(record['benchmark']);blob=Path(meta['observed']).read_bytes()
    for path,digest0 in meta['files'].items():
        if sha(path)!=digest0:raise ValueError('Observed snapshot changed')
    price_folder=Path(record['price_folder']);price_public=read(price_folder/'public.json');price_labels=read(price_folder/'labels.json');matching=read(Path(record['matching_folder'])/'public.json')
    md=a.training_root/'folds'/f'block_{identity[1]}_workload_{identity[2]}'/'model.pkl';pd=a.price_calibration/'folds'/md.parent.name/'prices.json'
    if sha(md)!=matching['model_sha256']:raise ValueError('HGB model source changed')
    bundle=pickle.loads(md.read_bytes());scorer=bundle['model'];prices=read(pd)
    if prices['model_sha256']!=sha(md) or bundle['fold']!=prices['fold'] or identity[1]!=prices['fold']['block'] or identity[2]!=prices['fold']['workload']:raise ValueError('Wrong held-out runtime model or prices')
    env,obs=pickle.loads(blob);_,_,requests,rows=runner.base.prepare(env);expected=inp['rows']
    for row in expected:row['options']={int(k):v for k,v in row['options'].items()}
    if digest(rows)!=digest(expected) or [q[2] for q in requests]!=inp['request_ids']:raise ValueError('Runtime observed input differs')
    checks=compatibility(rows,fast,native,scorer,prices,matching)
    refs={}
    for method in BASES:
        value=price_labels['outcomes'][price_labels['methods'][method]]
        refs[method,0]=dict(assignment=value['assignment'],requests=value['requests'],metrics=value['metrics'],
            predicted=price_public['selected'][method]['predicted'],calls=price_public['selected'][method]['matching_calls'],
            action=price_public['selected'][method].get('action'))
    for row in original['results']:
        if row['method'] in ('R8','R16'):refs[row['method'],row['repeat']]=dict(assignment=row['assignment'],requests=row['requests'],metrics=row['metrics'],
            predicted=row['diagnostic']['chosen_predicted'],calls=row['diagnostic']['matching_calls'],action=None)
    population=specs(identity);fingerprint=runner.base.policy_fingerprint([models['upper'],lower]);model_fp=hashlib.sha256(pickle.dumps(scorer,protocol=5)).hexdigest()
    def check_output(output,spec):
        motion,_,info,observed_rows=output;ref=refs[spec['method'],spec['repeat']]
        if list(info['assignment'])!=ref['assignment'] or info['request_ids']!=inp['request_ids'] or digest(observed_rows)!=digest(rows):raise ValueError('Native assignment or request input changed')
        if not np.array_equal(motion,np.asarray(inp['motion'])):raise ValueError('Frozen upper motion changed')
        if normal(info['diagnostic']['chosen_predicted'])!=normal(ref['predicted']):raise ValueError('Native predicted objective changed')
        if rows and info['diagnostic']['matching_calls']!=ref['calls']:raise ValueError('Native matching call budget changed')
        if spec['method'] not in ('G','R8','R16') and info['diagnostic']['selected_action']!=ref['action']:raise ValueError('Native selected quota changed')
        return info
    results={};order=[]
    with fast.FastRuntime() as prepared:
        policy=native.PricedPolicy(prepared,scorer,prices);prepared.patch(core,'truncated_search',policy.run)
        for spec in population:check_output(execute(core,fast,native,spec,blob,models,lower,runner,scorer,prices,prepared),spec)
        for iteration in range(3):
            schedule=np.random.default_rng(seed_for('priced_native_timing_order',identity,iteration)).permutation(len(population))
            for execution_rank,position in enumerate(schedule):
                observed_start_ns=time.monotonic_ns()
                spec=population[int(position)];info=check_output(execute(core,fast,native,spec,blob,models,lower,runner,scorer,prices,prepared),spec)
                observed_stop_ns=time.monotonic_ns()
                key=spec['method'],spec['repeat'];ref=refs[key]
                if key not in results:results[key]=dict(**spec,assignment=list(info['assignment']),software_seconds=[],timing_iteration=[],selected_order=[],execution_rank=[],call_observation_ns=[],
                    diagnostic=normal(info['diagnostic']),raw_success=core.fee_count(ref['requests'],0.),metrics=ref['metrics'],logical_upload_bytes=info['logical_upload_bytes'],logical_download_bytes=info['logical_download_bytes'])
                results[key]['software_seconds'].append(info['policy_seconds']);results[key]['timing_iteration'].append(iteration);results[key]['selected_order'].append(int(position))
                results[key]['execution_rank'].append(execution_rank);results[key]['call_observation_ns'].append([observed_start_ns,observed_stop_ns]);order.append([iteration,int(position)])
    if fingerprint!=runner.base.policy_fingerprint([models['upper'],lower]) or model_fp!=hashlib.sha256(pickle.dumps(scorer,protocol=5)).hexdigest():raise ValueError('Runtime changed frozen model')
    for key,row in results.items():row['fee_success']=core.mean_fee_count(refs[key]['requests'],row['software_seconds'])
    out=dict(identity=identity,generated=price_labels['generated'],admitted=price_labels['admitted'],basis=basis,model_fingerprint=fingerprint,hgb_sha256=sha(md),price_sha256=sha(pd),
        state_sha256=record['state_sha256'],reference_price_labels=str(price_folder/'labels.json'),reference_price_labels_sha256=sha(price_folder/'labels.json'),
        reference_original=str(record['benchmark']),reference_original_sha256=record['benchmark_sha256'],compatibility=checks,results=list(results.values()),
        timing_samples=len(order),timing_order=order,observation_timestamps_include_restore_and_validation=True,
        cpu=sorted(os.sched_getaffinity(0)),load_average=os.getloadavg(),state_delay_applied=False,physical_results_reused_after_exact_state_and_assignment=True)
    td=folder(a,identity);write(td/'benchmark.json',out);write(td/'result.json',dict(status='PASS',identity=identity,protocol_sha256=sha(a.results_root/'protocol.json'),benchmark_sha256=sha(td/'benchmark.json')))

def worker(a,p):
    verify(a,p);job=json.loads(a.job)
    if job not in p['jobs']:raise ValueError('Unexpected timing source/block')
    f=archive(a);res=load('priced_worker_resources',a.resource_module)
    with res.task_lock(a.results_root/'locks'/('_'.join(map(str,job))+'.lock')):
        with res.CpuLease(a.resources/'isolated',task=['priced_runtime',*job],kind='environment'):
            runtime=f.boot(a,job[1]);fast=load('priced_fast_runtime',a.release_root/'utils/sf_hlm_fast_runtime.py');native=load('priced_policy_native',a.release_root/'utils/sf_hlm_priced_runtime.py')
            for n,identity in enumerate(i for i in p['identities'] if i[:2]==job):
                if not completed(a,p,identity):benchmark(a,p,identity,runtime,f,fast,native)
                write(a.results_root/'progress'/('_'.join(map(str,job))+'.json'),dict(completed=n+1,last_identity=identity,updated_unix=time.time()))

def analyze(a,p):
    import numpy as np
    f=archive(a);core=f.core(argparse.Namespace(release_root=a.feasibility_release));states=[];checks=0
    for identity in p['identities']:
        if not completed(a,p,identity):raise ValueError('Timing population incomplete')
        b=read(folder(a,identity)/'benchmark.json');checks+=b['compatibility']['feature_quota_checks'];values={}
        core.require_population([(z['method'],z['repeat']) for z in b['results']],[(z['method'],z['repeat']) for z in specs(identity)])
        for name in METHODS:
            rows=[z for z in b['results'] if z['method']==name];values[name]=dict(raw=statistics.mean(z['raw_success'] for z in rows),fee=statistics.mean(z['fee_success'] for z in rows),
                energy=statistics.mean(z['metrics']['uav_fleet_total_energy'] for z in rows),mbs=statistics.mean(z['metrics']['service_offloads_mbs'] for z in rows),fairness=statistics.mean(z['metrics']['fairness'] for z in rows),
                calls=statistics.mean(z['diagnostic']['matching_calls'] for z in rows),times=[t for z in rows for t in z['software_seconds']])
        states.append(dict(identity=identity,generated=b['generated'],admitted=b['admitted'],methods=values))
    subsets={}
    pairs=[(m,'G') for m in METHODS if m!='G']+[("probability_price",ref) for ref in NONLEARNING if ref!='G']
    for source_name in ('all','greedy','learned'):
        rows=[s for s in states if source_name=='all' or s['identity'][0]==source_name];blocks={}
        for block in sorted({s['identity'][1] for s in rows}):
            ss=[s for s in rows if s['identity'][1]==block];den=sum(s['generated'] for s in ss);admitted=sum(s['admitted'] for s in ss);methods={}
            for name in METHODS:
                ts=[t for s in ss for t in s['methods'][name]['times']]
                methods[name]=dict(raw_rate=sum(s['methods'][name]['raw'] for s in ss)/den,fee_rate=sum(s['methods'][name]['fee'] for s in ss)/den,
                    energy=sum(s['methods'][name]['energy'] for s in ss),mbs_share=sum(s['methods'][name]['mbs'] for s in ss)/max(admitted,1),
                    fairness=statistics.mean(s['methods'][name]['fairness'] for s in ss),calls=statistics.mean(s['methods'][name]['calls'] for s in ss),
                    mean_seconds=statistics.mean(ts),p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
            blocks[str(block)]=dict(generated=den,admitted=admitted,methods=methods)
        comparisons={left+'-'+right:{metric:f.paired([b['methods'][left][metric+'_rate']-b['methods'][right][metric+'_rate'] for b in blocks.values()]) for metric in ('raw','fee')} for left,right in pairs}
        groups={}
        for name in METHODS:
            ts=[t for s in rows for t in s['methods'][name]['times']]
            groups[name]={key:statistics.mean(b['methods'][name][key] for b in blocks.values()) for key in ('raw_rate','fee_rate','mbs_share','fairness','calls')}
            groups[name].update(energy_ratio_to_G=statistics.mean(b['methods'][name]['energy']/b['methods']['G']['energy'] for b in blocks.values()),
                pooled_mean_seconds=statistics.mean(ts),pooled_p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
        subsets[source_name]=dict(blocks=blocks,groups=groups,comparisons=comparisons)
    main=subsets['all'];best=max(NONLEARNING,key=lambda n:main['groups'][n]['fee_rate']);gain=main['comparisons']['probability_price-G']['fee'];increment=main['comparisons']['probability_price-'+best]['fee'] if best!='G' else gain
    energy=main['groups']['probability_price']['energy_ratio_to_G'];time_ok=all(b['methods']['probability_price']['p95_seconds']<1. for b in main['blocks'].values())
    passed=gain['mean']>=.005 and gain['positive']>=4 and increment['mean']>=.001 and energy<=1.05 and time_ok
    out=dict(status='PASS_SMOKE' if p['smoke'] else 'PASS_NATIVE_DEVELOPMENT',states=len(states),branch_instances=len(states)*19,timing_samples=len(states)*57,feature_quota_checks=checks,
        protocol_sha256=sha(a.results_root/'protocol.json'),subsets=subsets,decision=dict(gain_vs_G=gain,best_nonlearning=best,learning_increment=increment,
            energy_ratio_to_G=energy,all_block_p95_below_second=time_ok,native_development_signal=passed,goal_complete=False,independent_confirmation=False))
    write(a.results_root/'analysis/summary.json',out);write(a.results_root/'analysis/states.json',states);write(a.results_root/'status.json',dict(status='COMPLETED_SMOKE' if p['smoke'] else 'COMPLETED',states=len(states),updated_unix=time.time()));print(json.dumps(out['decision']),flush=True)

def command(a,job):
    cmd=[sys.executable,str(a.release_root/FILES[1]),'worker']
    for key in ARGS:cmd+=['--'+key.replace('_','-'),str(getattr(a,key))]
    return cmd+['--job',json.dumps(job,separators=(',',':'))]

def pipeline(a):
    p=read(a.results_root/'protocol.json');verify(a,p,external=True);res=load('priced_controller_resources',a.resource_module)
    def run(job):
        path=a.results_root/'worker_logs'/('_'.join(map(str,job))+'.log');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:out=subprocess.run(command(a,job),stdout=log,stderr=subprocess.STDOUT,cwd=a.release_root)
        if out.returncode:raise RuntimeError('Native timing failed '+str(job))
    with res.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                pending=[pool.submit(run,j) for j in p['jobs']]
                for future in as_completed(pending):future.result();write(a.results_root/'status.json',dict(status='RUNNING',completed=sum(completed(a,p,i) for i in p['identities']),total=len(p['identities']),updated_unix=time.time()))
            with res.CpuLease(a.resources,task='priced_runtime_aggregate'):analyze(a,p)
        except BaseException as error:write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('init','pipeline','worker','analyze'))
    for key in ARGS:ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--job');a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Native matching, timing and scientific analysis are remote only')
    if a.mode=='init':init(a)
    elif a.mode=='worker':worker(a,read(a.results_root/'protocol.json'))
    elif a.mode=='pipeline':pipeline(a)
    else:
        res=load('priced_analysis_resources',a.resource_module)
        with res.CpuLease(a.resources,task='priced_runtime_analysis'):analyze(a,read(a.results_root/'protocol.json'))

if __name__=='__main__':main()
