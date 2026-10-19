"""Frozen-price evaluation behind direct training and adapter audit gates."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip
import hashlib
import json
import math
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
from scripts.analysis import sf_hlm_priced_closed_loop as b

METHODS=('G','R10','R16','binary_price','mixture','mixture_price','cold_prior_price','cold_half_proxy_price','HGB_probability_price')
ARGS=('release_root','foundation_release','foundation_launch','calibration_root','calibration_release','results_root','validation_root')
RUN_ARGS=('adapter_release','adapter_smoke_results','direct_audit','price_audit','runner_smoke_results')
MODULES=('sf_hlm_selector','sf_hlm_action_ranker','sf_hlm_request_risk','sf_hlm_calibrated_matching','sf_hlm_energy_price','sf_hlm_cache_mixture','sf_hlm_priced_runtime','sf_hlm_software_delay','sf_hlm_fast_runtime','sf_hlm_feasibility','sf_hlm_cache_calibration','sf_hlm_cache_context','sf_hlm_cache_matching')
FILES=('paths.py','scripts/analysis/sf_hlm_framework_evaluation.py','scripts/analysis/sf_hlm_framework_confirmation.py','scripts/analysis/sf_hlm_framework_foundation_audit.py','scripts/analysis/sf_hlm_framework_statistics.py','scripts/analysis/sf_hlm_priced_closed_loop.py',*('utils/'+x+'.py' for x in MODULES),'tests/test_sf_hlm_framework_evaluation.py','tests/test_sf_hlm_framework_statistics.py','tests/test_sf_hlm_framework_confirmation.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')


def foundation(a):
    f=b.load('framework_evaluation_foundations',a.foundation_release/'scripts/analysis/sf_hlm_framework_confirmation.py');data=b.read(a.foundation_launch)['args'];fa=argparse.Namespace(**{k:Path(v) for k,v in data.items()});p=b.read(fa.results_root/'protocol.json');fa.smoke=p['smoke'];fa.block=-1
    if fa.release_root!=a.foundation_release:raise ValueError('Wrong foundation implementation release')
    f.verify(fa,p,True)
    return f,fa,p


def spec(method,block,workload,episode,step):
    if method not in METHODS:raise ValueError('Unregistered framework method')
    kind='native_G' if method=='G' else 'random' if method in ('R10','R16') else 'probability_price' if method=='HGB_probability_price' else method
    return dict(method=method,kind=kind,cap=0 if method=='G' else 16 if method=='R16' else 10,seed=b.seed_for('fresh_framework_confirmation_search',block,workload,episode,step))


def comparison_prices(prices):
    """The two archived adapters name the same mixture price differently."""
    return dict(prices,prices={**prices['prices'],'prior':prices['prices']['mixture']})


def calibrate(a):
    if (a.calibration_root/'protocol.json').exists():raise FileExistsError('Immutable framework public-price namespace')
    f,fa,fp=foundation(a);old={k:Path(v) for k,v in b.read(fa.closed_launch)['args'].items()};fit=old['fit_root'];done=b.read(fit/'completion.json');original=b.read(fit/'prices.json');cold_done=b.read(fa.cold_calibration/'completion.json')
    if done['status']!='COMPLETED' or cold_done['status']!='COMPLETED':raise ValueError('Development price sources incomplete')
    inputs={str(p):b.sha(p) for p in (fa.closed_launch,fit/'protocol.json',fit/'prices.json',fit/'completion.json',fa.cold_calibration/'protocol.json',fa.cold_calibration/'completion.json')}
    for name,digest in done['files'].items():
        if b.sha(fit/name)!=digest:raise ValueError('Frozen development fit changed')
        inputs[str(fit/name)]=digest
    for path,digest in cold_done['files'].items():
        if b.sha(path)!=digest:raise ValueError('Frozen public control input changed')
        inputs[path]=digest
    pools={variant:{} for variant in ('cold_prior','cold_half')};duplicates=0
    for fold in (fa.cold_calibration/'folds').iterdir():
        cases=b.read(fold/'cases.json');powers=b.read(fold/'prices.json')['powers']
        if powers!=original['powers']:raise ValueError('Public communication powers changed')
        for variant in pools:
            for case in cases[variant]:
                si=case['state_index']
                if si in pools[variant]:
                    if pools[variant][si]!=case:raise ValueError('Public cold-control pool depends on unrelated fitted fold')
                    duplicates+=1
                else:pools[variant][si]=case
    if any(set(pool)!=set(range(400)) for pool in pools.values()):raise ValueError('Old public state population changed')
    price_module=b.load('framework_confirmation_price_recipe',a.release_root/'utils/sf_hlm_energy_price.py');cases={v:[pool[i] for i in range(400)] for v,pool in pools.items()};new={v:price_module.calibrate(rows,1.05) for v,rows in cases.items()}
    prices={**original['prices'],**new}
    for key in ('binary','mixture','probability'):
        if prices[key]!=original['prices'][key]:raise ValueError('Primary or original comparator price changed')
    p=dict(name='sf_hlm_framework_fixed_development_prices_20261018_v1',files={name:b.sha(a.release_root/name) for name in FILES},input_files=inputs,source_states=400,duplicate_control_cases_verified=duplicates,
        primary='mixture_price',primary_binary_hgb_prices_exactly_unchanged=True,model_path=str(fit/'model.pkl'),model_sha256=b.sha(fit/'model.pkl'),public_training_only=True,new_basis_or_confirmation_data_used=False,target=1.05)
    b.write(a.calibration_root/'protocol.json',p);b.write(a.calibration_root/'cases.json',cases);b.write(a.calibration_root/'prices.json',dict(prices=prices,powers=original['powers'],model_sha256=p['model_sha256'],source_fit_prices_sha256=b.sha(fit/'prices.json'),protocol_sha256=b.sha(a.calibration_root/'protocol.json')))
    for path,digest in inputs.items():
        if b.sha(path)!=digest:raise ValueError('Public price input changed during calibration')
    b.write(a.calibration_root/'completion.json',dict(status='COMPLETED',protocol_sha256=b.sha(a.calibration_root/'protocol.json'),files={name:b.sha(a.calibration_root/name) for name in ('cases.json','prices.json')},new_confirmation_data_used=False))
    print(json.dumps(dict(status='CALIBRATED',prices={k:v['price'] for k,v in prices.items()},duplicates=duplicates)),flush=True)


def verify_prices(a):
    p=b.read(a.calibration_root/'protocol.json');done=b.read(a.calibration_root/'completion.json')
    if done['status']!='COMPLETED' or done['protocol_sha256']!=b.sha(a.calibration_root/'protocol.json'):raise ValueError('Price calibration incomplete')
    for name,digest in p['files'].items():
        if b.sha(a.calibration_release/name)!=digest:raise ValueError('Immutable calibration source changed')
    for path,digest in p['input_files'].items():
        if b.sha(path)!=digest:raise ValueError('Price source changed')
    for name,digest in done['files'].items():
        if b.sha(a.calibration_root/name)!=digest:raise ValueError('Calibrated public price changed')
    if b.sha(p['model_path'])!=p['model_sha256']:raise ValueError('Frozen shared HGB changed')
    return p,b.read(a.calibration_root/'prices.json')


class ForbiddenQuotaActor:
    def act(self,*args,**kwargs):raise AssertionError('Analytical/random/HGB adapter attempted quota actor inference')


def public_modules(a):
    fast=b.public_modules(a)
    for name in ('sf_hlm_cache_calibration','sf_hlm_cache_context','sf_hlm_cache_matching'):b.load('utils.'+name,a.release_root/'utils'/(name+'.py'))
    return fast


def adapter_smoke(a):
    import numpy as np
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable framework adapter-smoke namespace')
    f,fa,fp=foundation(a)
    if not fp['smoke']:raise ValueError('Adapter smoke must use the independent smoke basis')
    foundation_done=b.read(fa.results_root/'completion.json')
    if foundation_done['status']!='COMPLETED' or foundation_done['protocol_sha256']!=b.sha(fa.results_root/'protocol.json'):raise ValueError('Smoke foundation incomplete')
    cp,prices=verify_prices(a);old=f.original_args(fa,fp,fp['confirmation_blocks'][0]);r=f.followup(fa);runner=r.runner(old);original,rollout=r.initialize(old,runner);models,source,checkpoint,state=r.basis(old,runner,original,fp);fast=public_modules(a)
    from utils.sf_hlm_priced_runtime import PricedPolicy
    from utils.sf_hlm_cache_matching import CacheMatchingPolicy
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    from utils.canonical_experiment import digest
    core=b.archive(fa).core(argparse.Namespace(release_root=fa.feasibility_release))
    hgb=pickle.loads(Path(cp['model_path']).read_bytes())['model'];model_fp=hashlib.sha256(pickle.dumps(hgb,protocol=5)).hexdigest();upper_fp=runner.base.policy_fingerprint([models['upper']])
    inputs={str(path):b.sha(path) for path in (a.foundation_launch,fa.results_root/'protocol.json',fa.results_root/'completion.json',fa.results_root/'foundation_audit.json',checkpoint/'contract.json',a.calibration_root/'protocol.json',a.calibration_root/'completion.json',a.calibration_root/'prices.json')}
    p=dict(name='sf_hlm_fresh_framework_nine_method_adapter_smoke_20261018_v1',files={name:b.sha(a.release_root/name) for name in FILES},input_files=inputs,methods=list(METHODS),steps=8,smoke=True,formal_confirmation_data_used=False,checks='None versus forbidden quota actor; unchanged archived paid policy; zero Env parity; measured threshold counts; budgets and shared hard guard',formal_evaluation_started=False)
    b.write(a.results_root/'protocol.json',p);results=[];physics_checks=0;zero_checks=0;actor_checks=0;ordinary_checks=0;random_prefix_checks=0
    for method in METHODS:
        env,obs=runner.episode_env(old.workload,0);records=[]
        with fast.FastRuntime() as rt:
            policy=CacheMatchingPolicy(rt,hgb,None,prices,prices['powers']) if method in ('cold_prior_price','cold_half_proxy_price') else PricedPolicy(rt,hgb,prices,prices['powers']);rt.patch(core,'truncated_search',policy.run)
            for step in range(8):
                blob=pickle.dumps((env,obs),protocol=5);left_env,left_obs=pickle.loads(blob);right_env,right_obs=pickle.loads(blob);job=spec(method,old.block,old.workload,0,step)
                left=rt.complete_policy(core,left_env,left_obs,job,models,None,runner);right=rt.complete_policy(core,right_env,right_obs,job,models,ForbiddenQuotaActor(),runner)
                for z in (left[2],right[2]):z.pop('policy_seconds',None);z['diagnostic'].pop('search_seconds',None)
                if digest(b.normal(left))!=digest(b.normal(right)):raise ValueError('Quota actor placeholder affects a non-PPO method')
                actor_checks+=1
                motion,actions,info,rows=rt.complete_policy(core,env,obs,job,models,None,runner);d=info['diagnostic']
                if d['matching_calls']>job['cap'] and method!='G' or d['chosen_predicted']['energy']>d['common_energy_cap']+1e-7:raise ValueError('Adapter violates matching budget or hard energy guard')
                if method in ('mixture','mixture_price'):
                    alternate=CacheMatchingPolicy(rt,hgb,None,comparison_prices(prices),prices['powers']);rt.clear();assignment,diagnostic,_=alternate.run(rows,num_uavs=5,kind=method)
                    if list(assignment)!=list(info['assignment']) or not math.isclose(diagnostic['expected_success'],d['expected_success'],abs_tol=1e-10,rel_tol=0):raise ValueError('Archived cache and original mixture numerical paths differ')
                    ordinary_checks+=1
                if method=='R10':
                    rt.clear();ten=policy.run(rows,num_uavs=5,kind='random',rng=np.random.default_rng(job['seed']),cap=10,trace=True)
                    rt.clear();sixteen=policy.run(rows,num_uavs=5,kind='random',rng=np.random.default_rng(job['seed']),cap=16,trace=True)
                    if ten[2]!=sixteen[2][:len(ten[2])] or list(ten[0])!=list(info['assignment']) or not ten[1]['matching_calls']<=min(10,sixteen[1]['matching_calls']):raise ValueError('R10 is not the registered R16 random-search prefix')
                    random_prefix_checks+=1
                if step==0:b.zero_equivalence(env,obs,motion,actions,info['request_ids']);zero_checks+=1
                with SoftwareServiceDelay(env,'measured') as hook:
                    hook.arm(info['request_ids'],info['policy_seconds']);next_obs,rewards,metrics=env.step(motion,offloading_actions=actions);outcome=hook.finish();runner.base.assert_step(metrics,info)
                    if outcome['success']!=int(metrics['deadline_satisfied_count']) or outcome['success']!=sum(x['latency']+outcome['injected_seconds']<=x['deadline'] for x in outcome['requests']):raise ValueError('Actual injected delay threshold count differs')
                    physics_checks+=1;obs=np.asarray(next_obs,np.float32);records.append(dict(step=step,assignment=list(info['assignment']),diagnostic=b.normal(d),policy_seconds=info['policy_seconds'],injected_seconds=outcome['injected_seconds'],requests=outcome['requests'],metrics=b.normal(metrics)))
        b.write(a.results_root/'methods'/method/'steps.json',records);results.append(dict(method=method,steps=len(records)))
    if upper_fp!=runner.base.policy_fingerprint([models['upper']]) or model_fp!=hashlib.sha256(pickle.dumps(hgb,protocol=5)).hexdigest() or cp['model_sha256']!=b.sha(cp['model_path']):raise ValueError('Frozen upper or HGB changed')
    for path,digest0 in inputs.items():
        if b.sha(path)!=digest0:raise ValueError('Adapter source changed')
    report=dict(status='PASS',smoke=True,methods=results,quota_actor_independence_checks=actor_checks,ordinary_mixture_equivalence_checks=ordinary_checks,random_ten_matching_prefix_checks=random_prefix_checks,zero_env_equivalence_checks=zero_checks,actual_injected_delay_count_checks=physics_checks,source_config_and_checkpoint_verified=True,protocol_sha256=b.sha(a.results_root/'protocol.json'),formal_confirmation_completed=False,performance_selection_performed=False)
    b.write(a.results_root/'adapter_audit.json',report);b.write(a.results_root/'completion.json',dict(status='COMPLETED',protocol_sha256=b.sha(a.results_root/'protocol.json'),adapter_audit_sha256=b.sha(a.results_root/'adapter_audit.json'),formal_confirmation_completed=False));print(json.dumps(report),flush=True)


def payload_hash(value):
    return hashlib.sha256(json.dumps(b.normal(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def runner_design(fp,smoke):
    if bool(fp['smoke'])!=bool(smoke):raise ValueError('Runner/foundation smoke populations differ')
    if smoke:
        blocks=fp['confirmation_blocks'];loads=[fp['training_workloads'][str(blocks[0])]]
        tasks=[[m,blocks[0],loads[0],e,'measured'] for m in METHODS for e in range(2)];random.Random(2026101806).shuffle(tasks)
    else:
        blocks=fp['confirmation_blocks'];loads=fp['confirmation_workloads'];tasks=fp['planned_evaluation_tasks']
        if len(tasks)!=2700 or len(blocks)!=20:raise ValueError('Frozen full confirmation population changed')
    block_order=list(dict.fromkeys(task[1] for task in tasks))
    return dict(methods=list(METHODS),blocks=blocks,workloads=loads,tasks=tasks,block_order=block_order,primary=fp['primary'],primary_comparisons=fp['primary_comparisons'],gates=fp['gates'],
        steps=16 if smoke else 1000,episodes_per_workload=2 if smoke else 3,runner_smoke=smoke,workers=1 if smoke else 8,
        physical_replay_tasks=tasks if smoke else [t for t in tasks if t[2]==loads[0] and t[3]==0])


def prerequisites(a):
    f,fa,fp=foundation(a);done=b.read(fa.results_root/'completion.json');direct=b.read(a.direct_audit);ap=b.read(a.adapter_smoke_results/'protocol.json');ad=b.read(a.adapter_smoke_results/'adapter_audit.json');ac=b.read(a.adapter_smoke_results/'completion.json');cp,prices=verify_prices(a);pa=b.read(a.price_audit)
    if done['status']!='COMPLETED' or done['protocol_sha256']!=b.sha(fa.results_root/'protocol.json'):raise ValueError('All registered foundation training must complete first')
    if direct['status']!='PASS' or direct['smoke']!=fp['smoke'] or direct['protocol_sha256']!=b.sha(fa.results_root/'protocol.json') or direct['foundation_completion_sha256']!=b.sha(fa.results_root/'completion.json'):raise ValueError('Direct canonical foundation audit missing or stale')
    if direct['audit_script_sha256']!=b.sha(a.release_root/'scripts/analysis/sf_hlm_framework_foundation_audit.py'):raise ValueError('Direct audit implementation differs from frozen runner')
    if [row['block'] for row in direct['blocks']]!=fp['confirmation_blocks'] or direct['distinct_upper_weight_fingerprints']!=len(fp['confirmation_blocks']) or not direct['upper_state_perturb_reload_checked'] or not direct['actual_seed_and_recipe_manifests_checked'] or not direct['all_steps_natural_admission_and_finite_rewards']:raise ValueError('Independent training/load proof incomplete')
    if direct['training_episodes_checked']!=len(fp['confirmation_blocks'])*fp['foundation_episodes'] or direct['training_steps_checked']!=len(fp['confirmation_blocks'])*fp['foundation_episodes']*fp['steps']:raise ValueError('Direct audit training population differs')
    if ac['status']!='COMPLETED' or ad['status']!='PASS' or not ap['smoke'] or ac['adapter_audit_sha256']!=b.sha(a.adapter_smoke_results/'adapter_audit.json') or ac['protocol_sha256']!=b.sha(a.adapter_smoke_results/'protocol.json'):raise ValueError('Nine-method adapter not audited')
    if ad['quota_actor_independence_checks']!=72 or ad['ordinary_mixture_equivalence_checks']!=16 or ad['random_ten_matching_prefix_checks']!=8 or ad['actual_injected_delay_count_checks']!=72 or ad['zero_env_equivalence_checks']!=9:raise ValueError('Adapter smoke population changed')
    for name,digest in ap['files'].items():
        if b.sha(a.adapter_release/name)!=digest:raise ValueError('Archived passed adapter source changed')
    for path,digest in ap['input_files'].items():
        if b.sha(path)!=digest:raise ValueError('Adapter evidence input changed')
    for name in ('paths.py','scripts/analysis/sf_hlm_framework_confirmation.py','scripts/analysis/sf_hlm_priced_closed_loop.py',*('utils/'+x+'.py' for x in MODULES)):
        if b.sha(a.release_root/name)!=b.sha(a.adapter_release/name):raise ValueError('Passed deployed numerical adapter changed: '+name)
    if pa['status']!='PASS' or pa['protocol_sha256']!=b.sha(a.calibration_root/'protocol.json') or pa['completion_sha256']!=b.sha(a.calibration_root/'completion.json') or pa['price_sha256']!=b.sha(a.calibration_root/'prices.json'):raise ValueError('Frozen public prices not independently audited')
    return f,fa,fp,cp,prices,direct


def isolated_topology(root):
    profile=b.read(root/'profile.json');cpus=profile['cpus'][:profile['slots']];cores=[]
    if len(cpus)!=8 or len(set(cpus))!=8:raise ValueError('Formal timing requires exactly eight isolated CPU leases')
    for cpu in cpus:
        p=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology';cores.append((int((p/'physical_package_id').read_text()),int((p/'core_id').read_text())))
    if len(set(cores))!=8:raise ValueError('Formal timing leases share a physical core')
    return dict(cpus=cpus,physical_cores=cores,profile_sha256=b.sha(root/'profile.json'))


def init_runner(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable framework runner namespace')
    f,fa,fp,cp,prices,direct=prerequisites(a)
    if fp['smoke']!=a.runner_smoke:raise ValueError('Runner and actual foundation populations differ')
    checks=b.read(a.validation_root/'runner_selftests.json')
    if checks['status']!='PASS' or checks['files']!={name:b.sha(a.release_root/name) for name in FILES}:raise ValueError('Exact frozen runner tests must pass before collection')
    inputs={str(path):b.sha(path) for path in (a.foundation_launch,fa.results_root/'protocol.json',fa.results_root/'completion.json',fa.results_root/'foundation_audit.json',a.direct_audit,a.adapter_smoke_results/'protocol.json',a.adapter_smoke_results/'adapter_audit.json',a.adapter_smoke_results/'completion.json',a.calibration_root/'protocol.json',a.calibration_root/'completion.json',a.calibration_root/'prices.json',a.price_audit,Path(cp['model_path']),a.validation_root/'runner_selftests.json')}
    topology=None
    if not a.runner_smoke:
        sd=b.read(a.runner_smoke_results/'completion.json');sp=b.read(a.runner_smoke_results/'protocol.json');sa=b.read(a.runner_smoke_results/'raw_audit.json')
        if sd['status']!='COMPLETED' or not sp['runner_smoke'] or sp['files']!={name:b.sha(a.release_root/name) for name in FILES} or sa['status']!='PASS' or sd['raw_audit_sha256']!=b.sha(a.runner_smoke_results/'raw_audit.json'):raise ValueError('Exact full runner smoke with raw physical replay must pass first')
        inputs.update({str(path):b.sha(path) for path in (a.runner_smoke_results/'protocol.json',a.runner_smoke_results/'completion.json',a.runner_smoke_results/'raw_audit.json')});topology=isolated_topology(f.parents(fa).resources/'isolated')
    p=dict(name='sf_hlm_fresh_framework_measured_closed_loop_confirmation_20261018_v1',files={name:b.sha(a.release_root/name) for name in FILES},input_files=inputs,
        foundation_protocol_sha256=b.sha(fa.results_root/'protocol.json'),source_hash=fp['source_hash'],price_protocol_sha256=b.sha(a.calibration_root/'protocol.json'),upper_fingerprints={str(row['block']):row['upper_fingerprint'] for row in direct['blocks']},
        shared_fixed_development_hgb=True,quota_ppo_evaluated=False,isolated_topology=topology,original_confirmation_blocks_used=False,
        timing_boundary='Frozen FastRuntime.complete_policy including cache clear, request preparation, method work, feedback, upper inference, mapping and logical message serialization; one actual sample per step',
        warmup='One copied first-state call per episode, no state advance, same stateless search seed as first recorded call',
        scheduling='Group by block in first-appearance order; within each block preserve frozen global shuffle projection; record actual asynchronous calls',
        real_delay_in_state=True,delay_point='after physical execute_request_batch, before metrics/reward/next observation',continuous_queue_or_real_network_simulated=False,software_cpu_physical_energy_included=False,
        raw_audit='All step/request threshold accounting plus prespecified full physical Env replay; no policy-clock remeasurement',
        signflip_interpretation='Exact sensitivity under symmetric independent block paired effects, not a causal randomization test',**runner_design(fp,a.runner_smoke))
    b.write(a.results_root/'protocol.json',p);b.write(a.results_root/'status.json',dict(status='PREPARED',episodes=len(p['tasks']),steps=len(p['tasks'])*p['steps'],formal_confirmation_completed=False));print(json.dumps(dict(status='PREPARED',episodes=len(p['tasks']),runner_smoke=a.runner_smoke)),flush=True)


def verify_runner(a,p):
    for name,digest in p['files'].items():
        if b.sha(a.release_root/name)!=digest:raise ValueError('Frozen full runner source changed: '+name)
    for path,digest in p['input_files'].items():
        if b.sha(path)!=digest:raise ValueError('Frozen full runner evidence changed: '+path)
    if p['runner_smoke']!=a.runner_smoke:raise ValueError('Wrong smoke/formal runner flag')
    f,fa,fp=foundation(a)
    if runner_design(fp,a.runner_smoke)!={key:p[key] for key in runner_design(fp,a.runner_smoke)}:raise ValueError('Registered task population or thresholds changed')
    if not a.runner_smoke and p['isolated_topology']!=b.normal(isolated_topology(f.parents(fa).resources/'isolated')):raise ValueError('Isolated timing CPU topology changed')
    return f,fa,fp


def task_folder(a,task):return a.results_root/'tasks'/('_'.join(map(str,task)))


def verified_episode(a,p,task):
    path=task_folder(a,task);receipt=b.read(path/'result.json')
    if receipt['status']!='PASS' or receipt['task']!=task or receipt['protocol_sha256']!=b.sha(a.results_root/'protocol.json') or receipt['steps_sha256']!=b.sha(path/'steps.json.gz'):raise ValueError('Episode receipt/source/step file mismatch')
    return receipt


def collect_episode(a,p,task,runner,models,core,fast,hgb,prices,checkpoint):
    import config
    import numpy as np
    from utils.canonical_experiment import exogenous_frame,config_snapshot
    from utils.sf_hlm_priced_runtime import PricedPolicy
    from utils.sf_hlm_cache_matching import CacheMatchingPolicy
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    from scripts.analysis.sf_hlm_framework_statistics import COUNTERS
    method,block,workload,episode,mode=task;env,obs=runner.episode_env(workload,episode);upper_fp=runner.base.policy_fingerprint([models['upper']]);hgb_fp=hashlib.sha256(pickle.dumps(hgb,protocol=5)).hexdigest();steps=[];totals={k:0. for k in COUNTERS};chains={k:hashlib.sha256() for k in ('exogenous','action','state')}
    if upper_fp!=p['upper_fingerprints'][str(block)] or b.powers()!=prices['powers']:raise ValueError('Loaded upper or public physical power differs')
    started_unix=time.time();cpu=sorted(os.sched_getaffinity(0))
    if len(cpu)!=1:raise ValueError('Each timed worker must have one CPU lease')
    with fast.FastRuntime() as rt:
        policy=CacheMatchingPolicy(rt,hgb,None,prices,prices['powers']) if method in ('cold_prior_price','cold_half_proxy_price') else PricedPolicy(rt,hgb,prices,prices['powers']);rt.patch(core,'truncated_search',policy.run)
        blob=pickle.dumps((env,obs),protocol=5);warm_env,warm_obs=pickle.loads(blob);rt.complete_policy(core,warm_env,warm_obs,spec(method,block,workload,episode,0),models,None,runner)
        if pickle.dumps((env,obs),protocol=5)!=blob:raise ValueError('Warmup advanced or mutated the live environment')
        with SoftwareServiceDelay(env,mode) as hook:
            for step in range(p['steps']):
                frame=payload_hash(exogenous_frame(env));potential=sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues);job=spec(method,block,workload,episode,step);call_start=time.monotonic_ns();motion,actions,info,rows=rt.complete_policy(core,env,obs,job,models,None,runner);call_end=time.monotonic_ns()
                hook.arm(info['request_ids'],info['policy_seconds']);next_obs,rewards,metrics=env.step(motion,offloading_actions=actions);outcome=hook.finish();runner.base.assert_step(metrics,info)
                if outcome['success']!=int(metrics['deadline_satisfied_count']) or len(outcome['requests'])!=int(metrics['admitted_service_count']):raise ValueError('Measured software delay and environment request count disagree')
                totals['potential']+=potential;totals['raw_success']+=outcome['raw_success'];totals['shadow_fee_success']+=outcome['shadow_fee_success']
                for key,metric in METRIC_KEYS:totals[key]+=float(metrics[metric])
                action=payload_hash([motion.tolist(),info['assignment']]);state=payload_hash([b.normal(next_obs),b.normal(rewards),b.normal(metrics),outcome['requests']])
                for key,value in (('exogenous',frame),('action',action),('state',state)):chains[key].update(value.encode())
                steps.append(dict(step=step,exogenous_hash=frame,potential=potential,action_hash=action,state_hash=state,metrics=b.normal(metrics),requests=outcome['requests'],policy_seconds=info['policy_seconds'],injected_seconds=outcome['injected_seconds'],
                    diagnostic=b.normal(info['diagnostic']),request_ids=info['request_ids'],assignment=b.normal(info['assignment']),motion=motion.tolist(),motion_dtype=str(np.asarray(motion).dtype),public_rows_hash=payload_hash(rows),
                    logical_upload_bytes=info['logical_upload_bytes'],logical_download_bytes=info['logical_download_bytes'],runtime_stats=info['runtime_stats'],call_window_ns=[call_start,call_end]))
                obs=np.asarray(next_obs,np.float32)
                if (step+1)%100==0:b.write(task_folder(a,task)/'progress.json',dict(step=step+1,total=p['steps'],pid=os.getpid(),updated_unix=time.time()))
    if upper_fp!=runner.base.policy_fingerprint([models['upper']]) or hgb_fp!=hashlib.sha256(pickle.dumps(hgb,protocol=5)).hexdigest():raise ValueError('Evaluation changed frozen upper or fixed HGB weights')
    path=task_folder(a,task);path.mkdir(parents=True,exist_ok=True)
    with gzip.open(path/'steps.json.gz','wt',encoding='utf8') as stream:json.dump(b.normal(steps),stream,separators=(',',':'),allow_nan=False)
    receipt=dict(status='PASS',task=task,protocol_sha256=b.sha(a.results_root/'protocol.json'),steps_sha256=b.sha(path/'steps.json.gz'),totals=totals,**{key+'_hash':value.hexdigest() for key,value in chains.items()},upper_fingerprint=upper_fp,hgb_fingerprint=hgb_fp,checkpoint=str(checkpoint),config_sha256=payload_hash(config_snapshot()),cpu=cpu,started_unix=started_unix,ended_unix=time.time())
    b.write(path/'result.json',receipt)


METRIC_KEYS=(('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('fleet_energy','uav_fleet_total_energy'),('mbs','service_offloads_mbs'),('fairness','fairness'))


def block_worker(a,p):
    f,fa,fp=verify_runner(a,p)
    if a.block not in p['blocks']:raise ValueError('Unregistered new block')
    old=f.original_args(fa,fp,a.block);r=f.followup(fa);res=f.resources(fa);lease_root=old.resources if a.runner_smoke else old.resources/'isolated'
    with res.task_lock(a.results_root/'block_locks'/f'{a.block}.lock'):
        with res.CpuLease(lease_root,task=['framework_timed_confirmation',a.block,a.runner_smoke],kind='environment'):
            runner=r.runner(old);original,rollout=r.initialize(old,runner);models,source,checkpoint,state=r.basis(old,runner,original,fp);fast=public_modules(a);core=b.archive(fa).core(argparse.Namespace(release_root=fa.feasibility_release));cp,prices=verify_prices(a);hgb=pickle.loads(Path(cp['model_path']).read_bytes())['model']
            if source['source_hash']!=p['source_hash']:raise ValueError('Actual loaded environment source changed')
            for task in [t for t in p['tasks'] if t[1]==a.block]:
                if (task_folder(a,task)/'result.json').exists():verified_episode(a,p,task);continue
                if (task_folder(a,task)/'steps.json.gz').exists():raise FileExistsError('Preserve partial timed data before a separately logged technical recovery')
                collect_episode(a,p,task,runner,models,core,fast,hgb,prices,checkpoint)
    b.write(a.results_root/'block_receipts'/f'{a.block}.json',dict(status='PASS',block=a.block,episodes=sum(t[1]==a.block for t in p['tasks']),protocol_sha256=b.sha(a.results_root/'protocol.json'),completed_unix=time.time()))


def runner_command(a,mode):
    cmd=[sys.executable,str(a.release_root/'scripts/analysis/sf_hlm_framework_evaluation.py'),mode]
    for name in (*ARGS,*RUN_ARGS):cmd+=['--'+name.replace('_','-'),str(getattr(a,name))]
    if a.runner_smoke:cmd.append('--runner-smoke')
    return cmd


def run_collection(a,p):
    f,fa,fp=verify_runner(a,p);res=f.resources(fa)
    def execute(block):
        log=a.results_root/'worker_logs'/f'{block}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            child=subprocess.Popen(runner_command(a,'worker')+['--block',str(block)],stdout=stream,stderr=subprocess.STDOUT,cwd=a.release_root)
            b.write(a.results_root/'workers'/f'{block}.json',dict(pid=child.pid,block=block,started_unix=time.time(),command=child.args));code=child.wait()
        if code:raise RuntimeError('Framework timed worker failed: '+str(block))
    with res.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=p['workers']) as pool:
                futures=[pool.submit(execute,block) for block in p['block_order']]
                for count,future in enumerate(as_completed(futures),1):future.result();b.write(a.results_root/'status.json',dict(status='COLLECTING',completed_blocks=count,total=len(futures),updated_unix=time.time(),formal_confirmation_completed=False))
            for task in p['tasks']:verified_episode(a,p,task)
            b.write(a.results_root/'collection_complete.json',dict(status='COLLECTED',protocol_sha256=b.sha(a.results_root/'protocol.json'),episodes=len(p['tasks']),completed_unix=time.time(),all_own_timed_workers_ended=True))
        except BaseException as error:b.write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time(),formal_confirmation_completed=False));raise
    # All timed processes ended before numerical analysis or replay begins.
    for mode in ('analyze','raw-audit'):subprocess.run(runner_command(a,mode),check=True,cwd=a.release_root)


def checked_steps(a,p,task):
    from scripts.analysis.sf_hlm_framework_statistics import COUNTERS
    receipt=verified_episode(a,p,task)
    with gzip.open(task_folder(a,task)/'steps.json.gz','rt') as stream:steps=json.load(stream)
    if len(steps)!=p['steps'] or [s['step'] for s in steps]!=list(range(p['steps'])):raise ValueError('Recorded step population differs')
    totals={k:0. for k in COUNTERS};chains={k:hashlib.sha256() for k in ('exogenous','action','state')};previous_end=0
    for row in steps:
        seconds=row['policy_seconds'];injected=row['injected_seconds'];metrics=row['metrics'];requests=row['requests'];ids=[int(r['ue_id']) for r in requests]
        if not math.isfinite(seconds) or seconds<0 or seconds!=injected or len(set(ids))!=len(ids) or ids!=sorted(row['request_ids']) or len(row['assignment'])!=len(ids):raise ValueError('Per-call fee or request identities differ')
        if any(not math.isfinite(r['latency']) or r['latency']<0 or not math.isfinite(r['deadline']) or r['deadline']<=0 for r in requests):raise ValueError('Nonfinite raw physical deadline accounting')
        success=sum(r['latency']+seconds<=r['deadline'] for r in requests);raw=sum(r['latency']<=r['deadline'] for r in requests)
        if success!=metrics['deadline_satisfied_count'] or len(ids)!=metrics['admitted_service_count'] or metrics['forced_service_admissions']!=0 or not 0<=success<=raw<=len(ids)<=metrics['generated_service_count']<=row['potential']:raise ValueError('Actual fee threshold or natural admission accounting failed')
        start,end=row['call_window_ns']
        if not previous_end<=start<end or seconds>(end-start)/1e9+1e-6:raise ValueError('Recorded policy clock is outside its actual call window')
        previous_end=end;diagnostic=row['diagnostic'];cap=spec(*task[:4],row['step'])['cap']
        if diagnostic['matching_calls']>cap or diagnostic['chosen_predicted']['energy']>diagnostic['common_energy_cap']+1e-7:raise ValueError('Budget or common hard energy guard violated')
        if row['logical_upload_bytes']<0 or row['logical_download_bytes']<0:raise ValueError('Invalid logical communication size')
        if payload_hash([row['motion'],row['assignment']])!=row['action_hash']:raise ValueError('Recorded action hash differs')
        for key in chains:chains[key].update(row[key+'_hash'].encode())
        totals['potential']+=row['potential'];totals['raw_success']+=raw;totals['shadow_fee_success']+=success
        for key,metric in METRIC_KEYS:totals[key]+=float(metrics[metric])
    if totals!=receipt['totals'] or any(chains[key].hexdigest()!=receipt[key+'_hash'] for key in chains):raise ValueError('Raw records do not reproduce episode receipt')
    if receipt['upper_fingerprint']!=p['upper_fingerprints'][str(task[1])]:raise ValueError('Episode evaluated the wrong upper checkpoint')
    return receipt,steps


def all_episodes(a,p):
    collection=b.read(a.results_root/'collection_complete.json')
    if collection['status']!='COLLECTED' or collection['episodes']!=len(p['tasks']) or collection['protocol_sha256']!=b.sha(a.results_root/'protocol.json') or not collection['all_own_timed_workers_ended']:raise ValueError('No interim analysis before the complete timed population')
    out=[];windows=[]
    for task in p['tasks']:
        receipt,steps=checked_steps(a,p,task);out.append(dict(task=task,totals=receipt['totals'],seconds=[s['policy_seconds'] for s in steps],exogenous_hash=receipt['exogenous_hash']))
        if len(receipt['cpu'])!=1:raise ValueError('Episode affinity changed')
        cpu=receipt['cpu'][0]
        if not p['runner_smoke'] and cpu not in p['isolated_topology']['cpus']:raise ValueError('Episode escaped isolated CPU pool')
        windows.append((cpu,steps[0]['call_window_ns'][0],steps[-1]['call_window_ns'][1],tuple(task)))
    by_cpu={}
    for cpu,start,end,task in sorted(windows):
        if start<by_cpu.get(cpu,0):raise ValueError('Two own timed episodes overlap on one leased CPU')
        by_cpu[cpu]=end
    return out


def analyze_runner(a,p):
    verify_runner(a,p)
    if (a.results_root/'analysis/summary.json').exists():raise FileExistsError('Completed full confirmation analysis is immutable')
    from scripts.analysis.sf_hlm_framework_statistics import summary
    report=summary(p,all_episodes(a,p));report.update(protocol_sha256=b.sha(a.results_root/'protocol.json'),collection_sha256=b.sha(a.results_root/'collection_complete.json'),signflip_assumption=p['signflip_interpretation']);b.write(a.results_root/'analysis/summary.json',report)
    b.write(a.results_root/'status.json',dict(status='ANALYZED_AWAITING_RAW_AUDIT',episodes=report['episodes'],formal_confirmation_completed=False));print(json.dumps(dict(status='ANALYZED_AWAITING_RAW_AUDIT',episodes=report['episodes'],runner_smoke=p['runner_smoke'])),flush=True)


def raw_audit(a,p):
    f,fa,fp=verify_runner(a,p)
    if (a.results_root/'raw_audit.json').exists():raise FileExistsError('Completed raw audit is immutable')
    import numpy as np
    from scripts.analysis.sf_hlm_framework_statistics import summary
    episodes=all_episodes(a,p);recomputed=summary(p,episodes);prior=b.read(a.results_root/'analysis/summary.json')
    for key in recomputed:
        if recomputed[key]!=prior[key]:raise ValueError('Full block statistics do not reproduce')
    old=f.original_args(fa,fp,p['blocks'][0]);r=f.followup(fa);runner=r.runner(old);original,rollout=r.initialize(old,runner);public_modules(a)
    import config
    from utils.sf_hlm_matching import feedback
    from marl_models.sf_hlm_v2 import service_view
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    from utils.canonical_experiment import exogenous_frame,config_snapshot
    replay_steps=0;replay_requests=0;replay_records=[];configured_block=None
    for task in p['physical_replay_tasks']:
        receipt,steps=checked_steps(a,p,task)
        if configured_block!=task[1]:
            actual_args=f.original_args(fa,fp,task[1]);models,actual_source,checkpoint,state=r.basis(actual_args,runner,original,fp);configured_block=task[1]
            if actual_source['source_hash']!=p['source_hash'] or runner.base.policy_fingerprint([models['upper']])!=p['upper_fingerprints'][str(task[1])]:raise ValueError('Physical replay basis differs')
        if payload_hash(config_snapshot())!=receipt['config_sha256']:raise ValueError('Physical replay configuration or observation mode differs')
        env,obs=runner.episode_env(task[2],task[3])
        with SoftwareServiceDelay(env,'measured') as hook:
            for row in steps:
                if payload_hash(exogenous_frame(env))!=row['exogenous_hash'] or sum(u.latent_request[0]<config.NUM_SERVICES for u in env.ues)!=row['potential']:raise ValueError('Physical replay external stream differs')
                _,masks,requests,rows=runner.base.prepare(env)
                if [q[2] for q in requests]!=row['request_ids'] or payload_hash(rows)!=row['public_rows_hash']:raise ValueError('Physical replay public predecision state differs')
                service_view(env,feedback(rows,row['assignment'],5,energy_reference=config.OFFLOAD_ENERGY_NORM_REF));actions=runner.base.actions_from_assignment(requests,row['assignment'],masks);motion=np.asarray(row['motion'],dtype=row['motion_dtype']);hook.arm(row['request_ids'],row['policy_seconds']);next_obs,rewards,metrics=env.step(motion,offloading_actions=actions);outcome=hook.finish()
                state=payload_hash([b.normal(next_obs),b.normal(rewards),b.normal(metrics),outcome['requests']])
                if b.normal(metrics)!=row['metrics'] or outcome['requests']!=row['requests'] or state!=row['state_hash']:raise ValueError(json.dumps(dict(error='Full physical replay diverges from sealed timed trajectory',task=task,step=row['step'],metrics_equal=b.normal(metrics)==row['metrics'],requests_equal=outcome['requests']==row['requests'],state_equal=state==row['state_hash'])))
                obs=np.asarray(next_obs,np.float32);replay_steps+=1;replay_requests+=len(row['requests'])
        replay_records.append(dict(task=task,steps_sha256=receipt['steps_sha256'],steps=len(steps)))
        b.write(a.results_root/'raw_audit_progress.json',dict(status='AUDITING',replayed_episodes=len(replay_records),expected=len(p['physical_replay_tasks']),replayed_steps=replay_steps))
    report=dict(status='PASS',runner_smoke=p['runner_smoke'],protocol_sha256=b.sha(a.results_root/'protocol.json'),summary_sha256=b.sha(a.results_root/'analysis/summary.json'),episodes=len(p['tasks']),all_steps_threshold_and_clock_checked=len(p['tasks'])*p['steps'],physical_replay_episodes=len(replay_records),physical_replay_steps=replay_steps,physical_replay_requests=replay_requests,physical_replays=replay_records,
        all_block_statistics_reproduced=True,all_paired_streams_and_potential_denominators_checked=True,no_same_cpu_own_timing_overlap=True,replay_policy_clocks_remeasured=False,model_or_price_refit=False,audit_type='primary agent self-audit; exhaustive raw accounting and prespecified full Env trajectory replay, not an independent reviewer')
    b.write(a.results_root/'raw_audit.json',report);b.write(a.results_root/'completion.json',dict(status='COMPLETED',runner_smoke=p['runner_smoke'],protocol_sha256=b.sha(a.results_root/'protocol.json'),summary_sha256=report['summary_sha256'],raw_audit_sha256=b.sha(a.results_root/'raw_audit.json'),formal_confirmation_completed=not p['runner_smoke'],unique_learning_increment_proven=False));b.write(a.results_root/'status.json',dict(status='COMPLETED',runner_smoke=p['runner_smoke'],formal_confirmation_completed=not p['runner_smoke']));print(json.dumps({k:v for k,v in report.items() if k!='physical_replays'}),flush=True)


def selftests(a):
    import unittest
    suite=unittest.TestLoader().discover(str(a.release_root/'tests'),pattern='test_sf_hlm_framework*.py');result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('Frozen framework runner tests failed')
    b.write(a.validation_root/'runner_selftests.json',dict(status='PASS',tests=result.testsRun,files={name:b.sha(a.release_root/name) for name in FILES},scientific_computation_remote=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('calibrate','adapter-smoke','init','worker','run','analyze','raw-audit','selftest'))
    for name in ARGS:p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    for name in RUN_ARGS:p.add_argument('--'+name.replace('_','-'),type=Path)
    p.add_argument('--runner-smoke',action='store_true');p.add_argument('--block',type=int,default=-1);a=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('All price calculation, model inference and simulation run remotely')
    f,fa,fp=foundation(a);old=f.parents(fa)
    if a.mode in ('calibrate','adapter-smoke','selftest','init','analyze','raw-audit'):
        with f.resources(fa).CpuLease(old.resources,task=['framework_confirmation_runner',a.mode],kind='environment'):
            if a.mode=='calibrate':calibrate(a)
            elif a.mode=='adapter-smoke':adapter_smoke(a)
            elif a.mode=='selftest':selftests(a)
            else:
                if any(getattr(a,name) is None for name in RUN_ARGS):raise ValueError('Complete passed-runner evidence paths required')
                if a.mode=='init':init_runner(a)
                else:
                    protocol=b.read(a.results_root/'protocol.json')
                    if a.mode=='analyze':analyze_runner(a,protocol)
                    else:raw_audit(a,protocol)
    else:
        protocol=b.read(a.results_root/'protocol.json');verify_runner(a,protocol)
        if a.mode=='worker':block_worker(a,protocol)
        else:run_collection(a,protocol)


if __name__=='__main__':main()
