"""Prospective development with measured policy delay inside Env feedback.

All fitting, simulation, matching, timing and statistics run on the remote
Linux host. Historical upper blocks plus one shared calibrator are development
replication, not independent lower-policy confirmation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

METHODS=('G','R8','R16','binary_price','mixture','mixture_price','probability_price','L16')
NONLEARNING=METHODS[:6]
BLOCKS=list(range(620001,620006))
WORKLOADS=[2026101701,2026101702,2026101703]
MODES=('zero','measured')
FILES=('paths.py','scripts/analysis/sf_hlm_priced_closed_loop.py','utils/sf_hlm_cache_mixture.py',
    'utils/sf_hlm_software_delay.py','utils/sf_hlm_priced_runtime.py','utils/sf_hlm_fast_runtime.py',
    'utils/sf_hlm_calibrated_matching.py','utils/sf_hlm_action_ranker.py','utils/sf_hlm_request_risk.py',
    'utils/sf_hlm_selector.py','utils/sf_hlm_energy_price.py','utils/sf_hlm_feasibility.py',
    'tests/test_sf_hlm_cache_mixture.py','tests/test_sf_hlm_software_delay.py',
    'tests/test_sf_hlm_priced_runtime.py','tests/test_sf_hlm_calibrated_matching.py',
    'docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARGS=('release_root','feasibility_release','parent_launch','old_training','old_price_calibration',
    'native_results','native_validation','fit_root','smoke_results','results_root','validation_root','resource_module','resources')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def normal(x):return json.loads(json.dumps(x,default=serial,allow_nan=False))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=serial)+'\n');tmp.replace(p)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def archive(a):return load('closed_loop_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
def seed_for(*parts):return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).digest()[:4],'big')
def resources(a):return load('closed_loop_resources',a.resource_module)
def folder(a,task):return a.results_root/'tasks'/('_'.join(map(str,task)))
def public_modules(a):
    for name in ('sf_hlm_selector','sf_hlm_action_ranker','sf_hlm_request_risk','sf_hlm_calibrated_matching',
                 'sf_hlm_energy_price','sf_hlm_cache_mixture','sf_hlm_priced_runtime','sf_hlm_software_delay'):
        load('utils.'+name,a.release_root/'utils'/(name+'.py'))
    return load('closed_loop_fast_runtime',a.release_root/'utils/sf_hlm_fast_runtime.py')
def powers():
    import config
    return dict(backhaul_rx=float(config.UAV_BACKHAUL_RX_POWER),backhaul_tx=float(config.UAV_BACKHAUL_TX_POWER),
        uav_tx=float(config.UAV_COMM_TX_POWER),uav_rx=float(config.UAV_COMM_RX_POWER))
def verify(a,p):
    for file,digest in p['files'].items():
        if sha(a.release_root/file)!=digest:raise ValueError('Frozen closed-loop source changed: '+file)
def spec(method,block,workload,step):
    kind='native_G' if method=='G' else 'random' if method in ('R8','R16') else 'learned' if method=='L16' else method
    return dict(method=method,kind=kind,cap=0 if method=='G' else int(method[1:]) if method in ('R8','R16','L16') else 10,
        seed=seed_for('priced_closed_loop_search',block,workload,step))


def fit(a):
    if (a.fit_root/'protocol.json').exists():raise FileExistsError('Immutable final-fit namespace')
    audit=read(a.native_validation/'raw_record_audit.json')
    if audit['status']!='PASS' or audit['primary']['summary_sha256']!=sha(a.native_results/'analysis/summary.json'):raise ValueError('Native full-cost evidence not audited')
    native=read(a.native_results/'analysis/summary.json')
    if not native['decision']['native_development_signal']:raise ValueError('Native candidate not promoted')
    tp=read(a.old_training/'protocol.json');cp=read(a.old_price_calibration/'protocol.json')
    inputs={str(p):sha(p) for p in (a.old_training/'protocol.json',a.old_training/'prepared.json',a.old_training/'training_data.npz',
        a.old_price_calibration/'protocol.json',a.native_results/'analysis/summary.json',a.native_validation/'raw_record_audit.json',a.parent_launch)}
    for identity in tp['identities']:
        suffix='oracle_'+'_'.join(map(str,identity))+'/public.json'
        paths=[p for p in cp['input_files'] if p.endswith(suffix)]
        if len(paths)!=1:raise ValueError('Public calibration input not uniquely identified')
        path=paths[0]
        if sha(path)!=cp['input_files'][path]:raise ValueError('Old public training state changed')
        inputs[path]=cp['input_files'][path]
    if sha(a.old_training/'training_data.npz')!=read(a.old_training/'prepared.json')['data_sha256']:raise ValueError('Old label data changed')
    p=dict(name='sf_hlm_final_development_calibrator_20261017_v1',files={f:sha(a.release_root/f) for f in FILES},input_files=inputs,
        identities=tp['identities'],schema=tp['schema'],features=tp['features'],hgb=tp['hgb'],target=1.05,
        training_states=400,new_workloads_used=False,independent_confirmation=False)
    write(a.fit_root/'protocol.json',p);res=resources(a)
    with res.CpuLease(a.resources,task=['priced_closed_loop_final_fit'],kind='environment'):
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        f=archive(a);f.boot(a,BLOCKS[0]);fast=public_modules(a)
        from utils.sf_hlm_action_ranker import public_payload
        from utils.sf_hlm_priced_runtime import PricedPolicy
        from utils.sf_hlm_energy_price import calibrate
        data=np.load(a.old_training/'training_data.npz')
        model=HistGradientBoostingClassifier(**p['hgb']).fit(data['x'],data['y'],sample_weight=data['weight'])
        model_path=a.fit_root/'model.pkl';model_path.write_bytes(pickle.dumps(dict(model=model,schema=p['schema'],features=p['features'],
            protocol_sha256=sha(a.fit_root/'protocol.json'),training_state_indices=sorted(set(data['state_index'].tolist()))),protocol=5))
        if not np.array_equal(model.predict_proba(data['x']),pickle.loads(model_path.read_bytes())['model'].predict_proba(data['x'])):raise ValueError('Final model serialization differs')
        cases={name:[] for name in ('binary','mixture','probability')};pw=powers();zero=dict(prices={name:dict(price=0.,normalizer=1.) for name in cases})
        for i,identity in enumerate(p['identities']):
            suffix='oracle_'+'_'.join(map(str,identity))+'/public.json';path=next(path for path in inputs if path.endswith(suffix))
            rows=public_payload(read(path))['rows']
            with fast.FastRuntime() as rt:
                policy=PricedPolicy(rt,model,zero,pw)
                for objective in cases:
                    _,diagnostic,candidates=policy.run(rows,num_uavs=5,kind=objective+'_price',trace=True)
                    cases[objective].append(dict(state_index=i,identity=identity,block=identity[1],
                        candidates=[dict(action=c['action'],assignment=c['assignment'],score=c['expected_success'],energy=c['predicted']['energy']) for c in candidates],
                        mixture_fallback_counts=diagnostic['mixture_fallback_counts']))
            if (i+1)%50==0:write(a.fit_root/'status.json',dict(status='CALIBRATING',states=i+1,total=400,updated_unix=time.time()))
        prices={name:calibrate(values,p['target']) for name,values in cases.items()}
        write(a.fit_root/'cases.json',cases);write(a.fit_root/'prices.json',dict(prices=prices,powers=pw,model_sha256=sha(model_path),
            protocol_sha256=sha(a.fit_root/'protocol.json'),new_workload_data_used=False))
        write(a.fit_root/'completion.json',dict(status='COMPLETED',protocol_sha256=sha(a.fit_root/'protocol.json'),
            files={name:sha(a.fit_root/name) for name in ('model.pkl','cases.json','prices.json')},training_rows=len(data['x']),
            training_states=400,cpu=sorted(os.sched_getaffinity(0))))
    print(json.dumps(dict(status='FITTED',prices={k:v['price'] for k,v in prices.items()})),flush=True)


def init(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable closed-loop namespace')
    fp=read(a.fit_root/'protocol.json');verify(a,fp);fit_done=read(a.fit_root/'completion.json')
    if fit_done['status']!='COMPLETED' or fit_done['protocol_sha256']!=sha(a.fit_root/'protocol.json'):raise ValueError('Final fit incomplete')
    for name,digest in fit_done['files'].items():
        if sha(a.fit_root/name)!=digest:raise ValueError('Final fit changed')
    if not a.smoke:
        smoke=read(a.smoke_results/'completion.json');sp=read(a.smoke_results/'protocol.json')
        if smoke['status']!='COMPLETED' or not sp['smoke'] or sp['files']!=fp['files'] or smoke['summary_sha256']!=sha(a.smoke_results/'analysis/summary.json'):raise ValueError('Compatible old-load smoke not completed')
    blocks=BLOCKS[:2] if a.smoke else BLOCKS;loads=[780001] if a.smoke else WORKLOADS
    tasks=[[m,b,w,mode] for m in METHODS for b in blocks for w in loads for mode in MODES]
    random.Random(2026101704).shuffle(tasks)
    p=dict(name='sf_hlm_priced_measured_delay_closed_loop_development_20261017_v1',smoke=a.smoke,files=fp['files'],methods=list(METHODS),
        nonlearning=list(NONLEARNING),blocks=blocks,workloads=loads,modes=list(MODES),tasks=tasks,steps=16 if a.smoke else 1000,
        fit_protocol_sha256=sha(a.fit_root/'protocol.json'),fit_completion_sha256=sha(a.fit_root/'completion.json'),
        frozen_fit_files=fit_done['files'],parent_launch_sha256=sha(a.parent_launch),nominal_target_cost_ms=10.,
        real_delay_in_state=True,delay_point='after execute_request_batch, before coverage/reward/metrics/next_obs',
        fixed_time_slot_physics=True,continuous_queues_simulated=False,real_network_signalling_simulated=False,
        timer='same native complete-policy boundary; cache clear and all per-call preprocessing included; one measured sample/step',
        warmup='one non-advancing first-state call per episode, same stateless random seed as actual first call',
        independent_confirmation=False,historical_upper_training_blocks=True,shared_lower_calibrator=True,
        gates=dict(gain_vs_G=.005,positive_blocks=4,learning_increment=.001,energy_ratio=1.05,per_block_p95_seconds=1.))
    write(a.results_root/'protocol.json',p);write(a.results_root/'status.json',dict(status='PREPARED',episodes=len(tasks),steps=len(tasks)*p['steps'],updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',episodes=len(tasks),steps=len(tasks)*p['steps'])),flush=True)


def complete(a,task):
    td=folder(a,task)
    if not (td/'result.json').exists():return False
    r=read(td/'result.json')
    if r['status']!='PASS' or r['task']!=task or r['protocol_sha256']!=sha(a.results_root/'protocol.json') or sha(td/'steps.json.gz')!=r['steps_sha256']:raise ValueError('Closed-loop receipt changed')
    return True


def zero_equivalence(env,obs,motion,actions,request_ids):
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    blob=pickle.dumps((env,obs),protocol=5);plain,_=pickle.loads(blob);wrapped,_=pickle.loads(blob)
    left=plain.step(motion,offloading_actions=actions)
    with SoftwareServiceDelay(wrapped,'zero') as hook:
        hook.arm(request_ids,.031);right=wrapped.step(motion,offloading_actions=actions);hook.finish()
    from utils.canonical_experiment import digest
    if digest(left)!=digest(right) or pickle.dumps(plain,protocol=5)!=pickle.dumps(wrapped,protocol=5):raise ValueError('Zero delay changes original Env.step')
    return dict(status='PASS',outputs_equal=True,full_pickled_state_equal=True)


def trajectory(a,p,task,runtime,fast,model,prices,repeat=False):
    import config
    import numpy as np
    from utils.canonical_experiment import digest,exogenous_frame,config_snapshot
    from utils.sf_hlm_priced_runtime import PricedPolicy
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    f=archive(a);core=f.core(argparse.Namespace(release_root=a.feasibility_release));_,_,runner,models,lower,basis=runtime
    method,block,workload,mode=task;env,obs=runner.episode_env(workload,0);fp=runner.base.policy_fingerprint([models['upper'],lower]);steps=[]
    if powers()!=prices['powers']:raise ValueError('Public communication power changed')
    exogenous=hashlib.sha256();actions_hash=hashlib.sha256();state_hash=hashlib.sha256();totals={k:0. for k in ('generated','admitted','success','potential','fleet_energy','mbs','fairness','raw_success','shadow_fee_success')}
    equality=None
    with fast.FastRuntime() as rt:
        policy=PricedPolicy(rt,model,prices,prices['powers']);rt.patch(core,'truncated_search',policy.run)
        if method=='L16':rt.enable_sparse(lower)
        before=pickle.dumps((env,obs),protocol=5);warm_env,warm_obs=pickle.loads(before)
        warm=rt.complete_policy(core,warm_env,warm_obs,spec(method,block,workload,0),models,lower,runner)
        if pickle.dumps((env,obs),protocol=5)!=before:raise ValueError('Warmup changed live environment state')
        if p['smoke'] and not repeat:equality=zero_equivalence(warm_env,warm_obs,warm[0],warm[1],warm[2]['request_ids'])
        with SoftwareServiceDelay(env,mode) as hook:
            for step in range(p['steps']):
                started=time.monotonic_ns();frame=digest(exogenous_frame(env));exogenous.update(frame.encode())
                potential=sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)
                motion,actions,info,rows=rt.complete_policy(core,env,obs,spec(method,block,workload,step),models,lower,runner)
                hook.arm(info['request_ids'],info['policy_seconds']);next_obs,rewards,metrics=env.step(motion,offloading_actions=actions);outcome=hook.finish()
                runner.base.assert_step(metrics,info)
                if outcome['success']!=int(metrics['deadline_satisfied_count']) or len(outcome['requests'])!=int(metrics['admitted_service_count']):raise ValueError('Injected request count differs from Env metrics')
                totals['potential']+=potential;totals['raw_success']+=outcome['raw_success'];totals['shadow_fee_success']+=outcome['shadow_fee_success']
                for key,metric in (('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),
                                   ('fleet_energy','uav_fleet_total_energy'),('mbs','service_offloads_mbs'),('fairness','fairness')):totals[key]+=float(metrics[metric])
                action=digest([motion.tolist(),info['assignment']]);state=digest([normal(next_obs),normal(rewards),normal(metrics),outcome['requests']])
                actions_hash.update(action.encode());state_hash.update(state.encode());obs=np.asarray(next_obs,np.float32)
                steps.append(dict(step=step,exogenous=frame,potential=potential,action_hash=action,state_hash=state,metrics=normal(metrics),
                    requests=outcome['requests'],policy_seconds=info['policy_seconds'],injected_seconds=outcome['injected_seconds'],
                    diagnostic=normal(info['diagnostic']),request_ids=info['request_ids'],assignment=normal(info['assignment']),motion=motion.tolist(),
                    logical_upload_bytes=info['logical_upload_bytes'],logical_download_bytes=info['logical_download_bytes'],runtime_stats=info['runtime_stats'],
                    observation_ns=[started,time.monotonic_ns()]))
                if not repeat and (step+1)%100==0:write(folder(a,task)/'progress.json',dict(step=step+1,total=p['steps'],pid=os.getpid(),updated_unix=time.time()))
    if runner.base.policy_fingerprint([models['upper'],lower])!=fp:raise ValueError('Evaluation changed policy weights')
    return dict(task=task,summary=totals,steps=steps,exogenous_trace_hash=exogenous.hexdigest(),action_hash=actions_hash.hexdigest(),state_hash=state_hash.hexdigest(),
        model_fingerprint=fp,basis=basis,config_sha256=digest(config_snapshot()),cpu=sorted(os.sched_getaffinity(0)),zero_injection_equivalence=equality)


def worker(a,p):
    task=json.loads(a.task);verify(a,p)
    if task not in p['tasks']:raise ValueError('Unknown task')
    if complete(a,task):return
    res=resources(a)
    with res.task_lock(folder(a,task)/'worker.lock'):
        with res.CpuLease(a.resources/'isolated',task=['priced_closed_loop',*task],kind='environment'):
            runtime=archive(a).boot(a,task[1]);fast=public_modules(a)
            bundle=pickle.loads((a.fit_root/'model.pkl').read_bytes());prices=read(a.fit_root/'prices.json')
            for name,digest in p['frozen_fit_files'].items():
                if sha(a.fit_root/name)!=digest:raise ValueError('Final model/price changed')
            result=trajectory(a,p,task,runtime,fast,bundle['model'],prices)
            if p['smoke'] and task[3]=='zero':
                again=trajectory(a,p,task,runtime,fast,bundle['model'],prices,repeat=True)
                for key in ('exogenous_trace_hash','action_hash','state_hash','model_fingerprint'):
                    if normal(result[key])!=normal(again[key]):raise ValueError('Zero-mode deterministic replay differs: '+key)
                if {k:v for k,v in result['summary'].items() if k!='shadow_fee_success'}!={k:v for k,v in again['summary'].items() if k!='shadow_fee_success'}:raise ValueError('Zero-mode raw metric totals changed')
                result['zero_replay_equal']=True
            steps=result.pop('steps');td=folder(a,task);td.mkdir(parents=True,exist_ok=True)
            with gzip.open(td/'steps.json.gz','wt',encoding='utf8') as stream:json.dump(steps,stream,allow_nan=False,default=serial,separators=(',',':'))
            result.update(status='PASS',protocol_sha256=sha(a.results_root/'protocol.json'),steps_sha256=sha(td/'steps.json.gz'))
            write(td/'result.json',result)


def analyze(a,p):
    import numpy as np
    f=archive(a);episodes=[]
    for task in p['tasks']:
        if not complete(a,task):raise ValueError('Episode population incomplete')
        rec=read(folder(a,task)/'result.json')
        with gzip.open(folder(a,task)/'steps.json.gz','rt') as stream:steps=json.load(stream)
        if len(steps)!=p['steps']:raise ValueError('Wrong step population')
        times=[s['policy_seconds'] for s in steps]
        episodes.append(dict(**rec,times=times))
    exo={}
    for rec in episodes:
        key=tuple(rec['task'][1:3]);exo.setdefault(key,set()).add(rec['exogenous_trace_hash'])
    if any(len(z)!=1 for z in exo.values()):raise ValueError('Exogenous random streams are not paired')
    modes={}
    for mode in MODES:
        blocks={}
        for block in p['blocks']:
            methods={}
            for method in METHODS:
                es=[e for e in episodes if e['task'][0]==method and e['task'][1]==block and e['task'][3]==mode]
                values={k:sum(e['summary'][k] for e in es) for k in es[0]['summary']};ts=[t for e in es for t in e['times']]
                methods[method]=dict(**values,dsr=values['success']/values['generated'],potential_dsr=values['success']/values['potential'],
                    admission_rate=values['admitted']/values['generated'],accepted_quality=values['success']/values['admitted'],
                    raw_dsr=values['raw_success']/values['generated'],shadow_fee_dsr=values['shadow_fee_success']/values['generated'],
                    mbs_share=values['mbs']/values['admitted'],mean_fairness=values['fairness']/(len(es)*p['steps']),
                    mean_seconds=statistics.mean(ts),p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
            blocks[str(block)]=methods
        groups={m:{k:statistics.mean(b[m][k] for b in blocks.values()) for k in next(iter(blocks.values()))[m]} for m in METHODS}
        best=max(NONLEARNING,key=lambda m:groups[m]['dsr'])
        pairs={m:f.paired([b['probability_price']['dsr']-b[m]['dsr'] for b in blocks.values()]) for m in METHODS if m!='probability_price'}
        energy_ratio=statistics.mean(b['probability_price']['fleet_energy']/b['G']['fleet_energy'] for b in blocks.values())
        gate=p['gates'];go=(not p['smoke'] and pairs['G']['mean']>=gate['gain_vs_G'] and pairs['G']['positive']>=gate['positive_blocks']
            and pairs[best]['mean']>=gate['learning_increment'] and energy_ratio<=gate['energy_ratio']
            and all(b['probability_price']['p95_seconds']<gate['per_block_p95_seconds'] for b in blocks.values()))
        modes[mode]=dict(blocks=blocks,groups=groups,best_nonlearning=best,learning_differences=pairs,energy_ratio_vs_G=energy_ratio,development_gate=go)
    result=dict(protocol_sha256=sha(a.results_root/'protocol.json'),episodes=len(episodes),steps=len(episodes)*p['steps'],modes=modes,
        exogenous_streams_paired=True,independent_confirmation=False,shared_calibrator=True,goal_complete=False,
        decision=dict(closed_loop_development_signal=modes['measured']['development_gate']))
    write(a.results_root/'analysis/summary.json',result);print(json.dumps(dict(status='ANALYZED',decision=result['decision'])),flush=True)


def command(a,mode):
    cmd=[sys.executable,str(a.release_root/FILES[1]),mode]
    for name in ARGS:cmd+=['--'+name.replace('_','-'),str(getattr(a,name))]
    return cmd


def run(a,p):
    verify(a,p);res=resources(a)
    def execute(task):
        path=a.results_root/'worker_logs'/('_'.join(map(str,task))+'.log');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:out=subprocess.run(command(a,'worker')+['--task',json.dumps(task)],cwd=a.release_root,stdout=log,stderr=subprocess.STDOUT)
        if out.returncode:raise RuntimeError('Closed-loop task failed: '+str(task))
    with res.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                jobs=[pool.submit(execute,t) for t in p['tasks'] if not complete(a,t)]
                for future in as_completed(jobs):
                    future.result();n=sum(complete(a,t) for t in p['tasks']);write(a.results_root/'status.json',dict(status='RUNNING',completed=n,total=len(p['tasks']),updated_unix=time.time()))
            out=subprocess.run(command(a,'analysis'),cwd=a.release_root)
            if out.returncode:raise RuntimeError('Closed-loop analysis failed')
            write(a.results_root/'completion.json',dict(status='COMPLETED',episodes=len(p['tasks']),steps=len(p['tasks'])*p['steps'],
                protocol_sha256=sha(a.results_root/'protocol.json'),summary_sha256=sha(a.results_root/'analysis/summary.json')))
            write(a.results_root/'status.json',dict(status='COMPLETED',episodes=len(p['tasks']),updated_unix=time.time()))
        except BaseException as error:
            write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('fit','init','run','worker','analysis'))
    for name in ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--task');a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Scientific execution is remote Linux only')
    if a.mode=='fit':fit(a)
    elif a.mode=='init':init(a)
    else:
        p=read(a.results_root/'protocol.json');verify(a,p)
        if a.mode=='worker':worker(a,p)
        elif a.mode=='analysis':
            res=resources(a)
            with res.CpuLease(a.resources,task=['priced_closed_loop_analysis'],kind='environment'):analyze(a,p)
        else:run(a,p)


if __name__=='__main__':main()
