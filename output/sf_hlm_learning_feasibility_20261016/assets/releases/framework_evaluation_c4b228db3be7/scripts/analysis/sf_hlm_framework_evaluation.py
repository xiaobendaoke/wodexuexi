"""Frozen development prices and nine-method smoke for fresh-basis confirmation.

This release exposes calibration and adapter smoke only. Formal evaluation is
deliberately unavailable until every foundation and adapter audit is complete.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import sf_hlm_priced_closed_loop as b

METHODS=('G','R10','R16','binary_price','mixture','mixture_price','cold_prior_price','cold_half_proxy_price','HGB_probability_price')
ARGS=('release_root','foundation_release','foundation_launch','calibration_root','calibration_release','results_root','validation_root')
MODULES=('sf_hlm_selector','sf_hlm_action_ranker','sf_hlm_request_risk','sf_hlm_calibrated_matching','sf_hlm_energy_price','sf_hlm_cache_mixture','sf_hlm_priced_runtime','sf_hlm_software_delay','sf_hlm_fast_runtime','sf_hlm_feasibility','sf_hlm_cache_calibration','sf_hlm_cache_context','sf_hlm_cache_matching')
FILES=('paths.py','scripts/analysis/sf_hlm_framework_evaluation.py','scripts/analysis/sf_hlm_framework_confirmation.py','scripts/analysis/sf_hlm_priced_closed_loop.py',*('utils/'+x+'.py' for x in MODULES),'tests/test_sf_hlm_framework_evaluation.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')


def foundation(a):
    f=b.load('framework_evaluation_foundations',a.foundation_release/'scripts/analysis/sf_hlm_framework_confirmation.py');data=b.read(a.foundation_launch)['args'];fa=argparse.Namespace(**{k:Path(v) for k,v in data.items()});p=b.read(fa.results_root/'protocol.json');fa.smoke=p['smoke'];fa.block=-1
    if fa.release_root!=a.foundation_release:raise ValueError('Wrong foundation implementation release')
    f.verify(fa,p,True)
    return f,fa,p


def spec(method,block,workload,episode,step):
    if method not in METHODS:raise ValueError('Unregistered framework method')
    kind='native_G' if method=='G' else 'random' if method in ('R10','R16') else 'probability_price' if method=='HGB_probability_price' else method
    return dict(method=method,kind=kind,cap=0 if method=='G' else 16 if method=='R16' else 10,seed=b.seed_for('fresh_framework_confirmation_search',block,workload,episode,step))


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
    b.write(a.results_root/'protocol.json',p);results=[];physics_checks=0;zero_checks=0;actor_checks=0;ordinary_checks=0
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
                    alternate=CacheMatchingPolicy(rt,hgb,None,prices,prices['powers']);rt.clear();assignment,diagnostic,_=alternate.run(rows,num_uavs=5,kind=method)
                    if list(assignment)!=list(info['assignment']) or not math.isclose(diagnostic['expected_success'],d['expected_success'],abs_tol=1e-10,rel_tol=0):raise ValueError('Archived cache and original mixture numerical paths differ')
                    ordinary_checks+=1
                if step==0:b.zero_equivalence(env,obs,motion,actions,info['request_ids']);zero_checks+=1
                with SoftwareServiceDelay(env,'measured') as hook:
                    hook.arm(info['request_ids'],info['policy_seconds']);next_obs,rewards,metrics=env.step(motion,offloading_actions=actions);outcome=hook.finish();runner.base.assert_step(metrics,info)
                    if outcome['success']!=int(metrics['deadline_satisfied_count']) or outcome['success']!=sum(x['latency']+outcome['injected_seconds']<=x['deadline'] for x in outcome['requests']):raise ValueError('Actual injected delay threshold count differs')
                    physics_checks+=1;obs=np.asarray(next_obs,np.float32);records.append(dict(step=step,assignment=list(info['assignment']),diagnostic=b.normal(d),policy_seconds=info['policy_seconds'],injected_seconds=outcome['injected_seconds'],requests=outcome['requests'],metrics=b.normal(metrics)))
        b.write(a.results_root/'methods'/method/'steps.json',records);results.append(dict(method=method,steps=len(records)))
    if upper_fp!=runner.base.policy_fingerprint([models['upper']]) or model_fp!=hashlib.sha256(pickle.dumps(hgb,protocol=5)).hexdigest() or cp['model_sha256']!=b.sha(cp['model_path']):raise ValueError('Frozen upper or HGB changed')
    for path,digest0 in inputs.items():
        if b.sha(path)!=digest0:raise ValueError('Adapter source changed')
    report=dict(status='PASS',smoke=True,methods=results,quota_actor_independence_checks=actor_checks,ordinary_mixture_equivalence_checks=ordinary_checks,zero_env_equivalence_checks=zero_checks,actual_injected_delay_count_checks=physics_checks,source_config_and_checkpoint_verified=True,protocol_sha256=b.sha(a.results_root/'protocol.json'),formal_confirmation_completed=False,performance_selection_performed=False)
    b.write(a.results_root/'adapter_audit.json',report);b.write(a.results_root/'completion.json',dict(status='COMPLETED',protocol_sha256=b.sha(a.results_root/'protocol.json'),adapter_audit_sha256=b.sha(a.results_root/'adapter_audit.json'),formal_confirmation_completed=False));print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('calibrate','adapter-smoke'))
    for name in ARGS:p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('All price calculation, model inference and simulation run remotely')
    f,fa,fp=foundation(a);old=f.parents(fa)
    with f.resources(fa).CpuLease(old.resources,task=['framework_confirmation_adapter',a.mode],kind='environment'):
        if a.mode=='calibrate':calibrate(a)
        else:adapter_smoke(a)


if __name__=='__main__':main()
