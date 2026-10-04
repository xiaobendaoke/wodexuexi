"""Frozen-checkpoint ideal-coordination closed-loop screening (remote only)."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

METHODS=['lower_mlp_upper_uncovered_obs','vanilla_lower_mlp_upper_uncovered_obs']
MODES=['baseline','ordinary','deadline','greedy','greedy_ordinary']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def job(args):
    sys.path.insert(0,str(args.source_root.resolve()))
    import numpy as np
    import torch
    import config
    from environment.env import Env
    from run_canonical_experiment import configure_method,policy_step
    from utils.canonical_experiment import config_snapshot,digest,exogenous_frame,source_manifest,json_value
    from utils.formal_protocol import verify_completed_job
    from utils.logger import refresh_derived_config_fields
    load_module('utils.deadline_repair',args.extension_root/'utils/deadline_repair.py')
    online=load_module('utils.deadline_repair_online',args.extension_root/'utils/deadline_repair_online.py')
    protocol=json.loads(args.protocol.read_text())
    if [args.method,args.mode,args.seed,args.workload] not in protocol['jobs']:raise ValueError('Undeclared job')
    for name,value in protocol['extension_files'].items():
        if sha(args.extension_root/name)!=value:raise ValueError('Extension changed')
    suite=json.loads((args.suite_root/'suite.json').read_text());plan=suite['plan']
    if sha(args.suite_root/'suite.json')!=protocol['suite_sha256']:raise ValueError('Suite changed')
    if source_manifest()['source_hash']!=suite['source_hash']:raise ValueError('Frozen source mismatch')
    v=plan['variants'][0]
    config.NUM_UES=v['ue_count'];config.UAV_COMPUTING_CAPACITY=config.UAV_COMPUTING_CAPACITY*v['compute_scale']
    refresh_derived_config_fields()
    config.PPO_EPOCHS=plan['ppo_epochs'];config.PPO_BATCH_SIZE=plan['batch_size'];config.PPO_ROLLOUT_LENGTH=plan['rollout']
    torch.set_num_threads(1);np.random.seed(args.seed);torch.manual_seed(args.seed)
    models=configure_method(args.method)
    base=args.suite_root/'main'/args.method/f'seed_{args.seed}'
    training=verify_completed_job(base/'train',suite['source_hash'])
    latest=json.loads((training/'latest.json').read_text())
    if latest['completed_episodes']!=200:raise ValueError('Not the final checkpoint')
    checkpoint=training/latest['checkpoint'];contract=json.loads((checkpoint/'contract.json').read_text())
    if (contract['method']!=args.method or contract['source_hash']!=suite['source_hash'] or contract['config_hash']!=digest(config_snapshot())):raise ValueError('Checkpoint contract mismatch')
    for name,value in contract['files'].items():
        if sha(checkpoint/name)!=value:raise ValueError('Checkpoint checksum mismatch')
    for name,model in models.items():model.load(str(checkpoint/name))
    if (args.output/'result.json').exists():raise FileExistsError(args.output/'result.json')
    reference=None;control_reference=None
    if protocol['smoke'] and args.mode=='baseline':
        original=verify_completed_job(base/f'workload_{args.workload}',suite['source_hash'])
        reference=json.loads((original/'episodes/00000.json').read_text())['steps']
    if protocol['smoke'] and protocol.get('study')=='greedy' and args.mode=='ordinary':
        if args.control_root is None:raise ValueError('Missing archived control')
        archived=args.control_root/args.method/'ordinary'/f'seed_{args.seed}_workload_{args.workload}'
        doc=json.loads((archived/'result.json').read_text())
        if doc['status']!='PASS' or doc['checkpoint_contract_sha256']!=sha(checkpoint/'contract.json'):raise ValueError('Archived control mismatch')
        for rel,h in doc['episode_files'].items():
            if sha(archived/rel)!=h:raise ValueError('Archived control checksum mismatch')
        control_reference=json.loads((archived/'episodes/00000.json').read_text())['steps']
    env=Env(scenario_seed=args.workload);episodes=[]
    for episode in range(protocol['episodes']):
        obs=np.asarray(env.reset(),dtype=np.float32)
        for model in models.values():model.reset()
        trace=hashlib.sha256();records=[]
        for step in range(protocol['steps']):
            frame=exogenous_frame(env);trace.update(digest(frame).encode())
            potential=sum(ue.latent_request[0]<config.NUM_SERVICES for ue in env.ues)
            started=time.perf_counter()
            is_greedy=args.mode.startswith('greedy')
            if is_greedy:
                _,masks=env.get_offloading_obs_and_masks()
                motion=models['upper'].select_actions(obs,exploration=False)
                original=np.zeros(masks.shape[:2],dtype=np.int64)
            else:
                motion,original,entries=policy_step(models,env,obs,exploration=False)
                masks=entries['lower'][2]
            network_seconds=time.perf_counter()-started
            t=time.perf_counter();requests,rows=online.prepare(env,masks);prepare_seconds=time.perf_counter()-t
            t=time.perf_counter();initialization_evaluations=0
            if is_greedy:
                initial,initialization_evaluations=online.greedy_assignment(rows)
                for request,action in zip(requests,initial):original[request[0],request[1]]=action
            initialization_seconds=time.perf_counter()-t
            repair_mode='baseline' if args.mode=='greedy' else 'ordinary' if args.mode=='greedy_ordinary' else args.mode
            t=time.perf_counter();chosen,diag=online.apply(requests,rows,original,repair_mode);repair_seconds=time.perf_counter()-t
            raw=[int(original[r[0],r[1]]) for r in requests];executed=[int(chosen[r[0],r[1]]) for r in requests]
            if any(not masks[r[0],r[1],a] for r,a in zip(requests,executed)):raise ValueError('Invalid mask action')
            if args.mode=='baseline' and not np.array_equal(chosen,original):raise ValueError('Baseline mutated')
            # Logical JSON schema size only; no physical link or energy implied.
            up=len(json.dumps({'rows':rows,'proposals':raw},separators=(',',':')).encode())
            down=len(json.dumps({'ue_ids':[r[2] for r in requests],'actions':executed},separators=(',',':')).encode())
            policy_seconds=time.perf_counter()-started
            next_obs,_,metrics=env.step(np.clip(motion,-1,1),offloading_actions=chosen)
            if metrics['service_fallback_count'] or metrics['forced_service_admissions']:raise ValueError('Fallback/forced admission')
            if metrics['admitted_service_count']!=len(requests) or metrics['service_offloads_mbs']!=executed.count(1):raise ValueError('Executed assignment mismatch')
            if reference is not None:
                for key in ('generated_service_count','admitted_service_count','deadline_satisfied_count','lower_assignment_energy','energy','fairness'):
                    if not np.isclose(metrics[key],reference[step]['metrics'][key],rtol=1e-10,atol=1e-10):raise ValueError('Historical baseline mismatch: '+key)
            if control_reference is not None:
                old=control_reference[step]
                if raw!=old['original_proposals'] or executed!=old['executed_actions']:raise ValueError('Control action regression')
                for key in ('generated_service_count','admitted_service_count','deadline_satisfied_count','lower_assignment_energy','energy','fairness'):
                    if not np.isclose(metrics[key],old['metrics'][key],rtol=1e-10,atol=1e-10):raise ValueError('Control metric regression')
            records.append({'step':step,'metrics':json_value(metrics),'potential_service_count':potential,
                            'request_ue_ids':[r[2] for r in requests],'original_proposals':raw,'executed_actions':executed,
                            'initialization_evaluations':initialization_evaluations,'initialization_seconds':initialization_seconds,
                            'lower_network_called':not is_greedy,'repair':diag,'policy_seconds':policy_seconds,'network_seconds':network_seconds,
                            'prepare_seconds':prepare_seconds,'repair_seconds':repair_seconds,
                            'logical_upload_bytes':up,'logical_download_bytes':down})
            obs=np.asarray(next_obs,dtype=np.float32)
            if step%250==0:print(json.dumps({'episode':episode,'step':step,'changed_actions':diag['changed_actions']}),flush=True)
        generated=sum(r['metrics']['generated_service_count'] for r in records)
        admitted=sum(r['metrics']['admitted_service_count'] for r in records)
        success=sum(r['metrics']['deadline_satisfied_count'] for r in records)
        summary={'episode':episode,'generated':generated,'admitted':admitted,'success':success,
                 'potential':sum(r['potential_service_count'] for r in records),
                 'dsr':success/generated if generated else 0.,'admission_rate':admitted/generated if generated else 0.,
                 'conditional_success':success/admitted if admitted else 0.,
                 'total_uav_energy':sum(r['metrics']['energy'] for r in records),
                 'assignment_energy':sum(r['metrics']['lower_assignment_energy'] for r in records),
                 'mbs_count':sum(r['metrics']['service_offloads_mbs'] for r in records),
                 'fairness_mean':statistics.mean(r['metrics']['fairness'] for r in records),
                 'changed_actions':sum(r['repair']['changed_actions'] for r in records),
                 'policy_p95_seconds':sorted(r['policy_seconds'] for r in records)[math.ceil(.95*len(records))-1],
                 'exogenous_trace_hash':trace.hexdigest()}
        if not math.isclose(summary['dsr'],summary['admission_rate']*summary['conditional_success'],abs_tol=1e-12):raise ValueError('DSR decomposition')
        write(args.output/'episodes'/f'{episode:05d}.json',json_value({'summary':summary,'steps':records}))
        episodes.append(summary)
    result={'status':'PASS','protocol_sha256':sha(args.protocol),'method':args.method,'mode':args.mode,
            'training_seed':args.seed,'workload_seed':args.workload,'source_hash':suite['source_hash'],
            'checkpoint_contract_sha256':sha(checkpoint/'contract.json'),'checkpoint':str(checkpoint),
            'episodes':episodes,'historical_baseline_prefix_verified':reference is not None,
            'ideal_coordination':True,'training_started':False,'learned_control_prefix_verified':control_reference is not None,
            'study':protocol.get('study','learned'),
            'episode_files':{str(p.relative_to(args.output)):sha(p) for p in sorted((args.output/'episodes').glob('*.json'))}}
    write(args.output/'result.json',json_value(result));print(json.dumps({'status':'PASS','job':[args.method,args.mode,args.seed,args.workload]}),flush=True)


def suite(args):
    old=json.loads((args.suite_root/'suite.json').read_text())
    seeds=[620001] if args.smoke else list(range(620001,620006))
    workloads=([740001] if args.smoke else list(range(750001,750006))) if args.study=='greedy' else ([720001] if args.smoke else list(range(740001,740006)))
    modes=['ordinary','greedy','greedy_ordinary'] if args.study=='greedy' else ['baseline','ordinary','deadline']
    jobs=[[m,v,s,w] for w in workloads for m in METHODS for v in modes for s in seeds]
    files=['utils/deadline_repair.py','utils/deadline_repair_online.py','scripts/analysis/deadline_repair_closed_loop.py']
    protocol={'version':'ideal_coordination_closed_loop_v2','study':args.study,'greedy_definition':'UE-id order; minimum own prefix-load latency, then own energy, then action id',
              'energy_guard':'1.05 times each initializer energy; absolute feasible sets can differ',
              'smoke':args.smoke,'purpose':'development_screening',
              'source_hash':old['source_hash'],'suite_sha256':sha(args.suite_root/'suite.json'),
              'extension_files':{p:sha(args.extension_root/p) for p in files},'jobs':jobs,
              'episodes':1 if args.smoke else 2,'steps':100 if args.smoke else 1000,'budget':128,'energy_ratio':1.05,
              'workers':args.workers,'threads_per_worker':1,'coordination':'ideal instantaneous lossless uncharged',
              'timing':'complete software policy path; concurrent, excludes physical signaling',
              'no_new_training':True,'no_automatic_next_stage':True}
    if args.workers not in range(1,6):raise ValueError('At most five workers')
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('Frozen protocol differs')
    else:write(protocol_path,protocol)
    completed=[];failures=[]
    def status(phase):write(args.output/'status.json',{'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),'expected_jobs':len(jobs),'completed_jobs':len(completed),'failures':failures,'training_started':False,'closed_loop':True})
    def run(item):
        m,v,s,w=item;out=args.output/m/v/f'seed_{s}_workload_{w}'
        if (out/'result.json').exists():
            doc=json.loads((out/'result.json').read_text())
            if doc['status']!='PASS' or doc['protocol_sha256']!=sha(protocol_path):raise ValueError('Existing result differs')
            for p,h in doc['episode_files'].items():
                if sha(out/p)!=h:raise ValueError('Existing episode differs')
            return out
        out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--job','--source-root',str(args.source_root),'--suite-root',str(args.suite_root),
             '--extension-root',str(args.extension_root),'--output',str(out),'--protocol',str(protocol_path),
             '--method',m,'--mode',v,'--seed',str(s),'--workload',str(w)]
        if args.control_root:cmd+=['--control-root',str(args.control_root)]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1')
        with (out/'run.log').open('w') as log:subprocess.run(cmd,cwd=args.source_root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return out
    status('RUNNING')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,item):item for item in jobs}
        for f in as_completed(futures):
            try:completed.append(f.result())
            except Exception as e:failures.append({'job':futures[f],'error':str(e)})
            status('RUNNING');print(json.dumps({'completed':len(completed),'failed':len(failures),'total':len(jobs)}),flush=True)
    if failures:status('FAILED');raise RuntimeError('Failed jobs retained')
    traces={};summaries=[]
    for out in completed:
        d=json.loads((out/'result.json').read_text());summaries.append(d)
        for ep in d['episodes']:
            key=(d['training_seed'],d['workload_seed'],ep['episode'])
            fingerprint=(ep['exogenous_trace_hash'],ep['potential'])
            if key in traces and traces[key]!=fingerprint:status('FAILED_PAIRING');raise ValueError('External streams differ')
            traces[key]=fingerprint
    write(args.output/'summary.json',{'status':'PASS','jobs':len(completed),'episodes':sum(len(d['episodes']) for d in summaries),
          'exogenous_pairing_verified':True,'ideal_coordination':True,'results':summaries,'independent_confirmation':False})
    status('COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source-root','suite-root','extension-root','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--study',choices=['learned','greedy'],default='learned');p.add_argument('--control-root',type=Path)
    p.add_argument('--workers',type=int,default=5);p.add_argument('--smoke',action='store_true');p.add_argument('--job',action='store_true')
    p.add_argument('--protocol',type=Path);p.add_argument('--method',choices=METHODS);p.add_argument('--mode',choices=MODES)
    p.add_argument('--seed',type=int);p.add_argument('--workload',type=int)
    args=p.parse_args()
    if platform.system()!='Linux':raise RuntimeError('Environment evaluation must run remotely')
    if args.job:job(args)
    else:suite(args)

if __name__=='__main__':main()
