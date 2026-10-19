"""Use up to 120 distinct physical cores for the prespecified confirmation.

The immutable legacy evaluator supplies every numerical policy, charge,
accounting and statistical function. This entry changes execution scheduling
and parallelizes verification after all measured jobs have ended.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

CAPACITY=120
METHODS=('G','R10','R16','binary_price','mixture','mixture_price','cold_prior_price','cold_half_proxy_price','HGB_probability_price')
EXTRA_FILES=('scripts/analysis/sf_hlm_framework_saturated.py','tests/test_sf_hlm_framework_saturated.py')
EXTRA_ARGS=('legacy_release','legacy_smoke_results','timing_resources')
_BACKEND=None
_AUDIT=None
_LEASE=None


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def design(fp,smoke):
    if bool(fp['smoke'])!=bool(smoke):raise ValueError('Foundation/smoke population differs')
    blocks=fp['confirmation_blocks'];loads=[fp['training_workloads'][str(blocks[0])]] if smoke else fp['confirmation_workloads']
    if smoke:
        tasks=[[m,blocks[0],loads[0],e,'measured'] for m in METHODS for e in range(2)];random.Random(2026101806).shuffle(tasks)
    else:
        tasks=fp['planned_evaluation_tasks']
        if len(tasks)!=2700 or len(blocks)!=20:raise ValueError('Full registered population required')
    bundles=list(dict.fromkeys(tuple(t[1:4]) for t in tasks));workers=min(CAPACITY,len(bundles))
    return dict(methods=list(METHODS),blocks=blocks,workloads=loads,tasks=tasks,block_order=list(dict.fromkeys(t[1] for t in tasks)),primary=fp['primary'],primary_comparisons=fp['primary_comparisons'],gates=fp['gates'],steps=16 if smoke else 1000,episodes_per_workload=2 if smoke else 3,runner_smoke=smoke,
        workers=workers,bundle_order=[list(x) for x in bundles],wave_sizes=[len(bundles[i:i+workers]) for i in range(0,len(bundles),workers)],audit_workers=2 if smoke else CAPACITY,saturated_server=True,hardware_max_workers=CAPACITY,
        physical_replay_tasks=tasks if smoke else [t for t in tasks if t[2]==loads[0] and t[3]==0])


def bundle_tasks(p,bundle):
    tasks=[t for t in p['tasks'] if t[1:4]==list(bundle)]
    if len(tasks)!=9 or {t[0] for t in tasks}!=set(METHODS):raise ValueError('Every paired bundle needs all nine methods exactly once')
    return tasks


def topology(root):
    profile=read(root/'profile.json');cpus=profile['cpus'][:profile['slots']];cores=[]
    if len(cpus)!=CAPACITY or len(set(cpus))!=CAPACITY:raise ValueError('Exactly 120 exclusive physical-core slots required')
    for cpu in cpus:
        base=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology';cores.append([int((base/'physical_package_id').read_text()),int((base/'core_id').read_text())])
    if len(set(map(tuple,cores)))!=CAPACITY:raise ValueError('Timing slots share physical cores')
    return dict(cpus=cpus,physical_cores=cores,profile_sha256=sha(root/'profile.json'))


def backend(a):
    global _BACKEND
    if _BACKEND is None:
        _BACKEND=load('framework_saturated_immutable_legacy',a.legacy_release/'scripts/analysis/sf_hlm_framework_evaluation.py');_BACKEND._original_files=_BACKEND.FILES;_BACKEND.FILES=(*_BACKEND.FILES,*EXTRA_FILES)
        _BACKEND.runner_design=design
    _BACKEND.isolated_topology=lambda unused:topology(a.timing_resources)
    return _BACKEND


def foundation(a):return backend(a).foundation(a)
def prerequisites(a):return backend(a).prerequisites(a)
def verified_episode(a,p,task):return backend(a).verified_episode(a,p,task)
def task_folder(a,task):return backend(a).task_folder(a,task)
def verify_runner(a,p):
    result=backend(a).verify_runner(a,p)
    if not p.get('saturated_server') or p['timing_resource_root']!=str(a.timing_resources):raise ValueError('Wrong hardware execution protocol')
    return result


def legacy_inputs(a):
    m=backend(a);p=read(a.legacy_smoke_results/'protocol.json');done=read(a.legacy_smoke_results/'completion.json');audit=read(a.legacy_smoke_results/'raw_audit.json')
    if done['status']!='COMPLETED' or not p['runner_smoke'] or done['raw_audit_sha256']!=sha(a.legacy_smoke_results/'raw_audit.json') or audit['status']!='PASS':raise ValueError('Immutable legacy numerical runtime not audited')
    inputs={str(path):sha(path) for path in (a.legacy_smoke_results/'protocol.json',a.legacy_smoke_results/'completion.json',a.legacy_smoke_results/'raw_audit.json')}
    for name,digest in p['files'].items():
        if sha(a.legacy_release/name)!=digest:raise ValueError('Legacy source archive changed')
        if name!='docs/plans/sf_hlm_learning_feasibility_20261016.md' and sha(a.release_root/name)!=digest:raise ValueError('Numerical runtime or original test changed: '+name)
        inputs[str(a.legacy_release/name)]=digest
    return inputs


def require_training_quiet(f,fa):
    old=f.parents(fa);res=f.resources(fa);active=[]
    for path in (old.resources/'slots').glob('*.lock'):
        try:
            value=read(path)
            if value.get('leased') and res.process_identity(value['pid'])==value.get('identity'):active.append(value)
        except (OSError,ValueError,KeyError):pass
    if active:raise RuntimeError('Other owned training/heavy work still holds general CPU leases: '+str([x['pid'] for x in active]))


def init(a):
    m=backend(a)
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable saturated-confirmation namespace')
    f,fa,fp,cp,prices,direct=prerequisites(a);checks=read(a.validation_root/'runner_selftests.json');files={name:sha(a.release_root/name) for name in m.FILES}
    if fp['smoke']!=a.runner_smoke or checks['status']!='PASS' or checks['files']!=files:raise ValueError('Exact-source tests and foundation population required')
    inputs=legacy_inputs(a)
    for path in (a.foundation_launch,fa.results_root/'protocol.json',fa.results_root/'completion.json',fa.results_root/'foundation_audit.json',a.direct_audit,a.adapter_smoke_results/'protocol.json',a.adapter_smoke_results/'adapter_audit.json',a.adapter_smoke_results/'completion.json',a.calibration_root/'protocol.json',a.calibration_root/'completion.json',a.calibration_root/'prices.json',a.price_audit,Path(cp['model_path']),a.validation_root/'runner_selftests.json'):inputs[str(path)]=sha(path)
    timing=None
    if not a.runner_smoke:
        sp=read(a.runner_smoke_results/'protocol.json');sd=read(a.runner_smoke_results/'completion.json');sa=read(a.runner_smoke_results/'raw_audit.json')
        if not sp['runner_smoke'] or not sp['saturated_server'] or sp['files']!=files or sd['status']!='COMPLETED' or sa['status']!='PASS' or sd['raw_audit_sha256']!=sha(a.runner_smoke_results/'raw_audit.json'):raise ValueError('Exact saturated-executor smoke is incomplete')
        for path in (a.runner_smoke_results/'protocol.json',a.runner_smoke_results/'completion.json',a.runner_smoke_results/'raw_audit.json'):inputs[str(path)]=sha(path)
        require_training_quiet(f,fa);timing=topology(a.timing_resources)
    p=dict(name='sf_hlm_fresh_framework_measured_saturated120_confirmation_20261018_v1',files=files,input_files=inputs,foundation_protocol_sha256=sha(fa.results_root/'protocol.json'),source_hash=fp['source_hash'],price_protocol_sha256=sha(a.calibration_root/'protocol.json'),upper_fingerprints={str(x['block']):x['upper_fingerprint'] for x in direct['blocks']},
        shared_fixed_development_hgb=True,quota_ppo_evaluated=False,isolated_topology=timing,timing_resource_root=str(a.timing_resources),original_confirmation_blocks_used=False,
        timing_boundary='Unchanged immutable FastRuntime.complete_policy; all dynamic preparation, method work, feedback, upper inference, mapping and logical messages included',warmup='One copied non-advancing first-state call per method episode, unchanged stateless first-call seed',
        scheduling='300 (block,workload,episode) bundles; all nine methods share a leased core within each bundle; frozen global shuffle projection retained; waves of up to 120 load models before release',
        hardware_regime='Up to 120 single-thread processes on distinct physical cores; loaded-server measurements, not interchangeable with old eight-core or unloaded costs',real_delay_in_state=True,delay_point='after physical execute_request_batch, before metrics/reward/next observation',continuous_queue_or_real_network_simulated=False,software_cpu_physical_energy_included=False,
        raw_audit='All raw threshold/clock accounting plus fixed 180-episode physical replay, parallel only after timed workers ended; clocks never remeasured',signflip_interpretation='Exact sensitivity under symmetric independent block paired effects, not a causal randomization test',**design(fp,a.runner_smoke))
    m.b.write(a.results_root/'protocol.json',p);m.b.write(a.results_root/'status.json',dict(status='PREPARED',episodes=len(p['tasks']),bundles=len(p['bundle_order']),workers=p['workers'],formal_confirmation_completed=False));print(json.dumps(dict(status='PREPARED',episodes=len(p['tasks']),workers=p['workers'],runner_smoke=a.runner_smoke)),flush=True)


def command(a,mode):
    m=backend(a);cmd=[sys.executable,str(a.release_root/EXTRA_FILES[0]),mode]
    for name in (*m.ARGS,*m.RUN_ARGS,*EXTRA_ARGS):cmd+=['--'+name.replace('_','-'),str(getattr(a,name))]
    if a.runner_smoke:cmd.append('--runner-smoke')
    return cmd


def bundle_name(bundle):return '_'.join(map(str,bundle))


def worker(a,p):
    m=backend(a);f,fa,fp=verify_runner(a,p);bundle=json.loads(a.bundle);tasks=bundle_tasks(p,bundle);block=bundle[0];old=f.original_args(fa,fp,block);r=f.followup(fa);res=f.resources(fa);lease_root=old.resources if a.runner_smoke else a.timing_resources;name=bundle_name(bundle)
    with res.task_lock(a.results_root/'bundle_locks'/(name+'.lock')):
        with res.CpuLease(lease_root,task=['framework_saturated_bundle',*bundle],kind='environment'):
            runner=r.runner(old);original,rollout=r.initialize(old,runner);models,source,checkpoint,state=r.basis(old,runner,original,fp);fast=m.public_modules(a);core=m.b.archive(fa).core(argparse.Namespace(release_root=fa.feasibility_release));cp,prices=m.verify_prices(a);hgb=pickle.loads(Path(cp['model_path']).read_bytes())['model']
            if source['source_hash']!=p['source_hash'] or runner.base.policy_fingerprint([models['upper']])!=p['upper_fingerprints'][str(block)]:raise ValueError('Actual saturated bundle basis differs')
            wave=a.results_root/'waves'/str(a.wave);m.b.write(wave/'ready'/(name+'.json'),dict(pid=os.getpid(),bundle=bundle,cpu=sorted(os.sched_getaffinity(0)),protocol_sha256=sha(a.results_root/'protocol.json')))
            while not (wave/'start.json').exists():
                if (wave/'abort.json').exists():raise RuntimeError('Wave aborted before measurements')
                time.sleep(.05)
            start=read(wave/'start.json')
            if start['protocol_sha256']!=sha(a.results_root/'protocol.json') or bundle not in start['bundles']:raise ValueError('Invalid wave release')
            for task in tasks:
                path=m.task_folder(a,task)
                if (path/'result.json').exists():m.verified_episode(a,p,task);continue
                if (path/'steps.json.gz').exists():raise FileExistsError('Keep partial timed data for explicit technical recovery')
                m.collect_episode(a,p,task,runner,models,core,fast,hgb,prices,checkpoint)
    m.b.write(a.results_root/'bundle_receipts'/(name+'.json'),dict(status='PASS',bundle=bundle,episodes=len(tasks),protocol_sha256=sha(a.results_root/'protocol.json'),completed_unix=time.time()))


def run(a,p):
    m=backend(a);f,fa,fp=verify_runner(a,p);res=f.resources(fa);stop=threading.Event();telemetry=threading.Thread(target=res.telemetry,args=(a.timing_resources,stop),daemon=True);telemetry.start()
    with res.task_lock(a.results_root/'controller.lock'):
        try:
            order=p['bundle_order'];finished=0
            for wave_id,offset in enumerate(range(0,len(order),p['workers'])):
                bundles=order[offset:offset+p['workers']];wave=a.results_root/'waves'/str(wave_id);children=[];streams=[]
                try:
                    for bundle in bundles:
                        name=bundle_name(bundle);log=a.results_root/'worker_logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('ab');streams.append(stream);cmd=command(a,'worker')+['--bundle',json.dumps(bundle),'--wave',str(wave_id)];child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=a.release_root);children.append((bundle,child));m.b.write(a.results_root/'workers'/(name+'.json'),dict(pid=child.pid,bundle=bundle,wave=wave_id,command=cmd,started_unix=time.time()))
                    while not all((wave/'ready'/(bundle_name(bundle)+'.json')).exists() for bundle in bundles):
                        if any(child.poll() is not None for bundle,child in children):raise RuntimeError('A bundle failed before the wave barrier')
                        time.sleep(.2)
                    ready=[read(wave/'ready'/(bundle_name(bundle)+'.json')) for bundle in bundles];cpus=[row['cpu'][0] for row in ready]
                    if any(len(row['cpu'])!=1 for row in ready) or len(set(cpus))!=len(cpus):raise ValueError('Wave workers do not hold distinct CPU leases')
                    m.b.write(wave/'start.json',dict(protocol_sha256=sha(a.results_root/'protocol.json'),bundles=bundles,released_unix=time.time(),cpus=cpus));codes=[child.wait() for bundle,child in children]
                    if any(codes):raise RuntimeError('Saturated bundle failed; retain all recorded samples')
                except BaseException:
                    if not (wave/'start.json').exists():m.b.write(wave/'abort.json',dict(error='Barrier failed',unix=time.time()))
                    for bundle,child in children:child.wait()
                    raise
                finally:
                    for stream in streams:stream.close()
                finished+=len(bundles);m.b.write(a.results_root/'status.json',dict(status='COLLECTING',completed_bundles=finished,total=len(order),updated_unix=time.time(),formal_confirmation_completed=False))
            for task in p['tasks']:m.verified_episode(a,p,task)
            m.b.write(a.results_root/'collection_complete.json',dict(status='COLLECTED',protocol_sha256=sha(a.results_root/'protocol.json'),episodes=len(p['tasks']),completed_unix=time.time(),all_own_timed_workers_ended=True))
        except BaseException as error:m.b.write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),formal_confirmation_completed=False));raise
        finally:stop.set();telemetry.join(timeout=2)
    for mode in ('analyze','raw-audit'):subprocess.run(command(a,mode),check=True,cwd=a.release_root)


def _release_lease():
    if _LEASE is not None:_LEASE.release()


def _audit_init(a,p,physics):
    global _AUDIT,_LEASE
    m=backend(a);f,fa,fp=m.foundation(a);old=f.parents(fa);res=f.resources(fa);root=old.resources if p['runner_smoke'] else a.timing_resources;_LEASE=res.CpuLease(root,task=['framework_saturated_post_timing_audit',physics],kind='environment').acquire();Finalize(None,_release_lease,exitpriority=10);_AUDIT=dict(a=a,p=p,m=m,f=f,fa=fa,fp=fp)
    if physics:
        old=f.original_args(fa,fp,p['blocks'][0]);r=f.followup(fa);runner=r.runner(old);original,rollout=r.initialize(old,runner);m.public_modules(a);_AUDIT.update(r=r,runner=runner,original=original,configured_block=None)


def _account(task):
    z=_AUDIT;a=z['a'];p=z['p'];m=z['m'];receipt,steps=m.checked_steps(a,p,task)
    if len(receipt['cpu'])!=1 or not p['runner_smoke'] and receipt['cpu'][0] not in p['isolated_topology']['cpus']:raise ValueError('Episode affinity escaped measured physical-core profile')
    return dict(task=task,totals=receipt['totals'],seconds=[s['policy_seconds'] for s in steps],exogenous_hash=receipt['exogenous_hash']),[receipt['cpu'][0],steps[0]['call_window_ns'][0],steps[-1]['call_window_ns'][1],task]


def all_episodes(a,p):
    m=backend(a);done=read(a.results_root/'collection_complete.json')
    if done['status']!='COLLECTED' or done['episodes']!=len(p['tasks']) or done['protocol_sha256']!=sha(a.results_root/'protocol.json') or not done['all_own_timed_workers_ended']:raise ValueError('No interim analysis before all measured workers end')
    with ProcessPoolExecutor(max_workers=p['audit_workers'],mp_context=multiprocessing.get_context('spawn'),initializer=_audit_init,initargs=(a,p,False)) as pool:rows=list(pool.map(_account,p['tasks'],chunksize=1))
    latest={}
    for cpu,start,end,task in sorted(row[1] for row in rows):
        if start<latest.get(cpu,0):raise ValueError('Own timed episodes overlap on the same physical CPU lease')
        latest[cpu]=end
    return [row[0] for row in rows]


def _replay(task):
    import numpy as np
    import config
    from utils.sf_hlm_matching import feedback
    from marl_models.sf_hlm_v2 import service_view
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    from utils.canonical_experiment import exogenous_frame,config_snapshot
    z=_AUDIT;a=z['a'];p=z['p'];m=z['m'];f=z['f'];fa=z['fa'];fp=z['fp'];r=z['r'];runner=z['runner'];receipt,steps=m.checked_steps(a,p,task)
    if z['configured_block']!=task[1]:
        old=f.original_args(fa,fp,task[1]);models,source,checkpoint,state=r.basis(old,runner,z['original'],fp);z['configured_block']=task[1]
        if source['source_hash']!=p['source_hash'] or runner.base.policy_fingerprint([models['upper']])!=p['upper_fingerprints'][str(task[1])]:raise ValueError('Replay actual basis differs')
    if m.payload_hash(config_snapshot())!=receipt['config_sha256']:raise ValueError('Replay configuration differs')
    env,obs=runner.episode_env(task[2],task[3]);requests_checked=0
    with SoftwareServiceDelay(env,'measured') as hook:
        for row in steps:
            if m.payload_hash(exogenous_frame(env))!=row['exogenous_hash'] or sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)!=row['potential']:raise ValueError('Replay external stream differs')
            _,masks,requests,rows=runner.base.prepare(env)
            if [q[2] for q in requests]!=row['request_ids'] or m.payload_hash(rows)!=row['public_rows_hash']:raise ValueError('Replay public predecision state differs')
            service_view(env,feedback(rows,row['assignment'],5,energy_reference=config.OFFLOAD_ENERGY_NORM_REF));actions=runner.base.actions_from_assignment(requests,row['assignment'],masks);motion=np.asarray(row['motion'],dtype=row['motion_dtype']);hook.arm(row['request_ids'],row['policy_seconds']);next_obs,rewards,metrics=env.step(motion,offloading_actions=actions);outcome=hook.finish();state=m.payload_hash([m.b.normal(next_obs),m.b.normal(rewards),m.b.normal(metrics),outcome['requests']])
            if m.b.normal(metrics)!=row['metrics'] or outcome['requests']!=row['requests'] or state!=row['state_hash']:raise ValueError('Parallel physical replay differs: '+str(task)+' step '+str(row['step']))
            obs=np.asarray(next_obs,np.float32);requests_checked+=len(row['requests'])
    return dict(task=task,steps_sha256=receipt['steps_sha256'],steps=len(steps),requests=requests_checked)


def analyze(a,p):
    m=backend(a);verify_runner(a,p)
    if (a.results_root/'analysis/summary.json').exists():raise FileExistsError('Immutable completed summary')
    stats=m.b.load('framework_saturated_frozen_statistics',a.legacy_release/'scripts/analysis/sf_hlm_framework_statistics.py');report=stats.summary(p,all_episodes(a,p));report.update(protocol_sha256=sha(a.results_root/'protocol.json'),collection_sha256=sha(a.results_root/'collection_complete.json'),signflip_assumption=p['signflip_interpretation']);m.b.write(a.results_root/'analysis/summary.json',report);m.b.write(a.results_root/'status.json',dict(status='ANALYZED_AWAITING_RAW_AUDIT',formal_confirmation_completed=False));print(json.dumps(dict(status='ANALYZED_AWAITING_RAW_AUDIT',episodes=report['episodes'])),flush=True)


def raw_audit(a,p):
    m=backend(a);verify_runner(a,p)
    if (a.results_root/'raw_audit.json').exists():raise FileExistsError('Immutable completed raw audit')
    stats=m.b.load('framework_saturated_frozen_statistics',a.legacy_release/'scripts/analysis/sf_hlm_framework_statistics.py');recomputed=stats.summary(p,all_episodes(a,p));prior=read(a.results_root/'analysis/summary.json')
    if any(recomputed[key]!=prior[key] for key in recomputed):raise ValueError('Full block statistics do not reproduce')
    with ProcessPoolExecutor(max_workers=p['audit_workers'],mp_context=multiprocessing.get_context('spawn'),initializer=_audit_init,initargs=(a,p,True)) as pool:records=list(pool.map(_replay,p['physical_replay_tasks'],chunksize=1))
    report=dict(status='PASS',runner_smoke=p['runner_smoke'],protocol_sha256=sha(a.results_root/'protocol.json'),summary_sha256=sha(a.results_root/'analysis/summary.json'),episodes=len(p['tasks']),all_steps_threshold_and_clock_checked=len(p['tasks'])*p['steps'],physical_replay_episodes=len(records),physical_replay_steps=sum(row['steps'] for row in records),physical_replay_requests=sum(row['requests'] for row in records),physical_replays=[{k:v for k,v in row.items() if k!='requests'} for row in records],all_block_statistics_reproduced=True,all_paired_streams_and_potential_denominators_checked=True,no_same_cpu_own_timing_overlap=True,replay_policy_clocks_remeasured=False,model_or_price_refit=False,parallel_audit_workers=p['audit_workers'],audit_type='primary agent self-audit; unchanged raw checks and full Env replay parallelized after all measured jobs ended')
    m.b.write(a.results_root/'raw_audit.json',report);m.b.write(a.results_root/'completion.json',dict(status='COMPLETED',runner_smoke=p['runner_smoke'],protocol_sha256=sha(a.results_root/'protocol.json'),summary_sha256=report['summary_sha256'],raw_audit_sha256=sha(a.results_root/'raw_audit.json'),formal_confirmation_completed=not p['runner_smoke'],unique_learning_increment_proven=False));m.b.write(a.results_root/'status.json',dict(status='COMPLETED',formal_confirmation_completed=not p['runner_smoke']));print(json.dumps({k:v for k,v in report.items() if k!='physical_replays'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('selftest','init','worker','run','analyze','raw-audit'))
    # Path argument names remain compatible with the completed evidence reporter.
    names=('release_root','foundation_release','foundation_launch','calibration_root','calibration_release','results_root','validation_root','adapter_release','adapter_smoke_results','direct_audit','price_audit','runner_smoke_results',*EXTRA_ARGS)
    for name in names:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--runner-smoke',action='store_true');parser.add_argument('--bundle');parser.add_argument('--wave',type=int,default=-1);a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('All scientific/saturated execution runs on the remote server')
    m=backend(a);f,fa,fp=m.foundation(a);old=f.parents(fa)
    if a.mode in ('selftest','init','analyze','raw-audit'):
        control=old.resources if a.runner_smoke else old.resources/'isolated'
        with f.resources(fa).CpuLease(control,task=['framework_saturated_control',a.mode],kind='environment'):
            if a.mode=='selftest':m.selftests(a)
            elif a.mode=='init':init(a)
            else:
                p=read(a.results_root/'protocol.json')
                if a.mode=='analyze':analyze(a,p)
                else:raw_audit(a,p)
    else:
        p=read(a.results_root/'protocol.json')
        if a.mode=='worker':worker(a,p)
        else:run(a,p)


if __name__=='__main__':main()
