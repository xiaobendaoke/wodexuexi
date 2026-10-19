"""Cross-fit cache-weight matching, public energy prices and native full fees.

Frozen-state development only; fresh closed loop and confirmation still follow.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

METHODS=('G','R8','R16','binary_price','mixture','mixture_price','own_proxy_price','half_proxy_price',
    'cache_probability_price','HGB_probability_price','L16')
NONLEARNING=METHODS[:8]
PRIMARY='cache_probability_price'
VARIANT_METHODS=dict(prior='mixture_price',own_proxy='own_proxy_price',half_proxy='half_proxy_price',learned=PRIMARY)
FILES=('paths.py','scripts/analysis/sf_hlm_cache_matching.py','utils/sf_hlm_cache_matching.py',
    'utils/sf_hlm_cache_calibration.py','utils/sf_hlm_cache_context.py','utils/sf_hlm_cache_mixture.py','utils/sf_hlm_priced_runtime.py',
    'utils/sf_hlm_fast_runtime.py','utils/sf_hlm_calibrated_matching.py','utils/sf_hlm_action_ranker.py',
    'utils/sf_hlm_request_risk.py','utils/sf_hlm_selector.py','utils/sf_hlm_energy_price.py',
    'utils/sf_hlm_feasibility.py','tests/test_sf_hlm_cache_matching.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARGS=('release_root','cache_training','cache_validation','old_price_calibration','old_hgb_training','old_native_validation',
    'feasibility_release','parent_launch','calibration_root','calibration_release','smoke_root','results_root','validation_root','resources','resource_module')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def normal(x):return json.loads(json.dumps(x,default=serial,allow_nan=False))
def decision_diagnostic(value):
    """Search wall time is an observation; all decision fields stay exact."""
    return normal({key:item for key,item in value.items() if key!='search_seconds'})
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=serial)+'\n');temp.replace(p)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def archive(a):return load('cache_matching_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py')
def resource(a):return load('cache_matching_resources',a.resource_module)
def seed_for(*parts):return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).digest()[:4],'big')
def folder(a,identity):return a.results_root/'tasks'/('cache_'+'_'.join(map(str,identity)))
def fold_name(b,w):return f'block_{b}_workload_{w}'
def splits(a,p):return load('cache_matching_splits',a.release_root/'utils/sf_hlm_selector.py').folds(p['identities'])
def modules(a):
    for name in ('sf_hlm_selector','sf_hlm_action_ranker','sf_hlm_request_risk','sf_hlm_calibrated_matching',
                 'sf_hlm_energy_price','sf_hlm_cache_calibration','sf_hlm_cache_mixture','sf_hlm_priced_runtime','sf_hlm_cache_matching'):
        load('utils.'+name,a.release_root/'utils'/(name+'.py'))
    return load('utils.sf_hlm_fast_runtime',a.release_root/'utils/sf_hlm_fast_runtime.py')
def powers():
    import config
    return dict(backhaul_rx=float(config.UAV_BACKHAUL_RX_POWER),backhaul_tx=float(config.UAV_BACKHAUL_TX_POWER),uav_tx=float(config.UAV_COMM_TX_POWER),uav_rx=float(config.UAV_COMM_RX_POWER))
def verify(a,p,external=False):
    for name,digest in p['files'].items():
        if sha(a.release_root/name)!=digest:raise ValueError('Cache matching source changed')
    if external:
        for name,digest in p['input_files'].items():
            if sha(name)!=digest:raise ValueError('Cache matching input changed')
def specs(identity):
    out=[]
    for method in METHODS:
        kind='native_G' if method=='G' else 'random' if method in ('R8','R16') else 'learned' if method=='L16' else 'probability_price' if method=='HGB_probability_price' else method
        for repeat in range(5 if kind=='random' else 1):
            out.append(dict(method=method,kind=kind,cap=0 if method=='G' else int(method[1:]) if method in ('R8','R16','L16') else 10,
                repeat=repeat,seed=seed_for('sf_hlm_feasibility_20261016',identity,repeat)))
    return out

def branch_instances(identity):return len(specs(identity))


def prepare(a):
    if (a.calibration_root/'protocol.json').exists():raise FileExistsError('Immutable cache-price calibration namespace')
    audit=read(a.cache_validation/'raw_record_audit.json');tp=read(a.cache_training/'protocol.json');old=read(a.old_price_calibration/'protocol.json')
    if audit['status']!='PASS' or audit['summary_sha256']!=sha(a.cache_training/'analysis/summary.json') or not read(a.cache_training/'analysis/summary.json')['decision']['cache_hit_target_promising_development_only']:raise ValueError('Cache target not audited and promoted')
    if read(a.old_native_validation/'raw_record_audit.json')['status']!='PASS':raise ValueError('Old native baseline not audited')
    if tp['identities']!=old['identities'] or read(a.old_hgb_training/'protocol.json')['identities']!=tp['identities']:raise ValueError('Calibration populations differ')
    if tp.get('context_rule','none') not in ('none','all_visible_miss_zero_v1'):raise ValueError('Unsupported declared context rule')
    inputs={str(p):sha(p) for p in (a.cache_validation/'raw_record_audit.json',a.cache_training/'protocol.json',a.cache_training/'completion.json',
        a.cache_training/'analysis/summary.json',a.old_price_calibration/'protocol.json',a.old_price_calibration/'completion.json',a.old_hgb_training/'protocol.json',
        a.old_hgb_training/'completion.json',a.old_native_validation/'raw_record_audit.json',a.parent_launch,a.resource_module,a.resources/'isolated/profile.json')}
    inputs.update(read(a.old_price_calibration/'completion.json')['files']);inputs.update(read(a.old_hgb_training/'completion.json')['files'])
    for fold in splits(a,tp):
        td=a.cache_training/'folds'/fold_name(fold['block'],fold['workload']);rec=read(td/'result.json')
        if rec['status']!='PASS' or rec['fold']!=fold:raise ValueError('Cache training fold missing')
        inputs[str(td/'result.json')]=sha(td/'result.json')
        for name,digest in rec['files'].items():
            if sha(td/name)!=digest:raise ValueError('Cache model changed')
            inputs[str(td/name)]=digest
    states=[]
    for entry in old['states']:
        identity=entry['identity'];state=read(entry['state'])
        if sha(entry['state'])!=entry['state_sha256']:raise ValueError('Old state changed')
        inputs[entry['state']]=entry['state_sha256']
        for name,digest in state['files'].items():
            if sha(name)!=digest:raise ValueError('Old observed/private state changed')
            inputs[name]=digest
        suffix='oracle_'+'_'.join(map(str,identity))+'/public.json';pub=next(x for x in old['input_files'] if x.endswith(suffix))
        if sha(pub)!=old['input_files'][pub]:raise ValueError('Original public rows changed')
        inputs[pub]=old['input_files'][pub];states.append(dict(entry,public_rows=pub))
    p=dict(name='sf_hlm_cache_weight_public_price_20261018_v1',files={f:sha(a.release_root/f) for f in FILES},input_files=inputs,
        identities=tp['identities'],states=states,variants=list(VARIANT_METHODS),target=1.05,nominal_cost_ms=10.,
        split='25 leave-block-and-workload-out folds, 256 price-training states, 16 test states, 128 excluded',
        private_labels_used_for_price=False,public_inverse_approximation_retained=True,source_model_training_protocol=sha(a.cache_training/'protocol.json'),
        source_feature_mode=tp.get('feature_mode','clipped'),
        source_context_rule=tp.get('context_rule','none'),
        methods=list(METHODS),nonlearning=list(NONLEARNING),primary=PRIMARY,independent_confirmation=False,goal_complete=False)
    write(a.calibration_root/'protocol.json',p);write(a.calibration_root/'status.json',dict(status='PREPARED',folds=25,updated_unix=time.time()))


def calibrate_fold(a,p):
    verify(a,p);fold=next(f for f in splits(a,p) if f['block']==a.block and f['workload']==a.workload);name=fold_name(a.block,a.workload)
    out=a.calibration_root/'folds'/name
    if (out/'result.json').exists():raise FileExistsError('Immutable cache price fold')
    archive(a).boot(a,a.block);fast=modules(a)
    from utils.sf_hlm_action_ranker import public_payload
    from utils.sf_hlm_cache_matching import CacheMatchingPolicy
    from utils.sf_hlm_energy_price import calibrate
    model_path=a.cache_training/'folds'/name/'model.pkl'
    if sha(model_path)!=p['input_files'][str(model_path)]:raise ValueError('Cache model bytes changed')
    bundle=pickle.loads(model_path.read_bytes())
    from utils.sf_hlm_cache_calibration import validate_bundle
    validate_bundle(bundle,supported_context_rules=('none',p.get('source_context_rule','none')))
    if bundle['fold']!=fold or bundle['protocol_sha256']!=p['source_model_training_protocol']:raise ValueError('Wrong cache training fold')
    cases={variant:[] for variant in VARIANT_METHODS};zero=dict(prices={variant:dict(price=0.,normalizer=1.) for variant in cases});pw=powers()
    with fast.FastRuntime() as rt:
        policy=CacheMatchingPolicy(rt,None,bundle['model'],zero,pw)
        for si in fold['train']:
            state=p['states'][si];path=state['public_rows']
            if sha(path)!=p['input_files'][path]:raise ValueError('Calibration rows changed')
            safe=public_payload(read(path));rows=safe['rows'];rt.clear()
            for variant,method in VARIANT_METHODS.items():
                _,diagnostic,cc=policy.run(rows,num_uavs=5,kind=method,trace=True)
                cases[variant].append(dict(state_index=si,identity=state['identity'],block=state['identity'][1],
                    candidates=[dict(action=c['action'],assignment=c['assignment'],score=c['expected_success'],energy=c['predicted']['energy']) for c in cc]))
    prices={variant:calibrate(values,1.05) for variant,values in cases.items()}
    write(out/'cases.json',cases);write(out/'prices.json',dict(prices=prices,powers=pw,fold=fold,cache_model_sha256=sha(model_path),protocol_sha256=sha(a.calibration_root/'protocol.json')))
    write(out/'result.json',dict(status='PASS',fold=fold,protocol_sha256=sha(a.calibration_root/'protocol.json'),files={name:sha(out/name) for name in ('cases.json','prices.json')},cpu=sorted(os.sched_getaffinity(0))))


def command(a,mode):
    cmd=[sys.executable,str(a.release_root/FILES[1]),mode]
    for name in ARGS:cmd+=['--'+name.replace('_','-'),str(getattr(a,name))]
    return cmd


def calibrate_pipeline(a):
    p=read(a.calibration_root/'protocol.json');verify(a,p,True);res=resource(a)
    def run(fold):
        log=a.calibration_root/'worker_logs'/(fold_name(fold['block'],fold['workload'])+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:out=subprocess.run(command(a,'fold')+['--block',str(fold['block']),'--workload',str(fold['workload'])],cwd=a.release_root,stdout=stream,stderr=subprocess.STDOUT)
        if out.returncode:raise RuntimeError('Cache price worker failed '+str(fold))
    with res.task_lock(a.calibration_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                jobs=[pool.submit(run,f) for f in splits(a,p)]
                for n,future in enumerate(as_completed(jobs),1):future.result();write(a.calibration_root/'status.json',dict(status='RUNNING',completed=n,total=25,updated_unix=time.time()))
            files={}
            for fold in splits(a,p):
                td=a.calibration_root/'folds'/fold_name(fold['block'],fold['workload']);rec=read(td/'result.json')
                if rec['status']!='PASS' or rec['fold']!=fold:raise ValueError('Price completion receipt changed')
                files[str(td/'result.json')]=sha(td/'result.json')
                for name,digest in rec['files'].items():
                    if sha(td/name)!=digest:raise ValueError('Price artifact changed')
                    files[str(td/name)]=digest
            write(a.calibration_root/'completion.json',dict(status='COMPLETED',folds=25,protocol_sha256=sha(a.calibration_root/'protocol.json'),files=files))
            write(a.calibration_root/'status.json',dict(status='COMPLETED',folds=25,updated_unix=time.time()))
        except BaseException as error:write(a.calibration_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


def init(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable cache matching namespace')
    cp=read(a.calibration_root/'protocol.json');done=read(a.calibration_root/'completion.json')
    for name,digest in cp['files'].items():
        if sha(a.calibration_release/name)!=digest:raise ValueError('Frozen price source changed')
        if (name.startswith('utils/') or name=='paths.py') and sha(a.release_root/name)!=digest:raise ValueError('Numerical policy differs from price calibration source')
    for path,digest in cp['input_files'].items():
        if sha(path)!=digest:raise ValueError('Frozen price input changed')
    native_files={name:sha(a.release_root/name) for name in FILES}
    if done['status']!='COMPLETED' or done['protocol_sha256']!=sha(a.calibration_root/'protocol.json'):raise ValueError('Prices not completed')
    for path,digest in done['files'].items():
        if sha(path)!=digest:raise ValueError('Price completion changed')
    if not a.smoke:
        sr=read(a.smoke_root/'completion.json');sp=read(a.smoke_root/'protocol.json')
        if sr['status']!='COMPLETED' or not sp['smoke'] or sp['files']!=native_files:raise ValueError('Same-source eight-state smoke not completed')
    states=[s for s in cp['states'] if not a.smoke or (s['identity'][1] in (620001,620002) and s['identity'][2]==780001 and s['identity'][3]==0 and s['identity'][4] in (0,250))]
    p=dict(name='sf_hlm_cache_matching_native_20261018_decision_clock_v2',smoke=a.smoke,files=native_files,input_files=cp['input_files'],states=states,
        identities=[s['identity'] for s in states],jobs=[list(k) for k in sorted({tuple(s['identity'][:2]) for s in states})],methods=list(METHODS),nonlearning=list(NONLEARNING),primary=PRIMARY,
        calibration_protocol_sha256=sha(a.calibration_root/'protocol.json'),calibration_completion_sha256=sha(a.calibration_root/'completion.json'),calibration_files=done['files'],
        calibration_release=str(a.calibration_release),calibration_source_files=cp['files'],numerical_policy_matches_calibration_source=True,
        source_feature_mode=cp.get('source_feature_mode','clipped'),source_model_training_protocol=cp['source_model_training_protocol'],source_context_rule=cp.get('source_context_rule','none'),
        branch_instances=branch_instances(states[0]['identity']),timing_repeats=3,timing_samples_per_state=3*branch_instances(states[0]['identity']),nominal_cost_ms=10.,
        cost_boundary='per-call cache clear,prepare,repair128,all features/models/matches/prices,feedback,upper,mapping,logical JSON',
        cost_excludes='restore,model preload,private physics; real networking not simulated',public_decisions_sealed_before_physics=True,public_inverse_approximation_retained=True,
        gates=dict(gain_vs_G=.005,positive_blocks=4,learning_increment=.001,energy_ratio=1.05,p95_seconds=1.),state_delay_applied=False,independent_confirmation=False,goal_complete=False)
    write(a.results_root/'protocol.json',p);write(a.results_root/'status.json',dict(status='PREPARED',states=len(states),updated_unix=time.time()))
    print(json.dumps(dict(status='PREPARED',states=len(states),timing_samples=p['timing_samples_per_state']*len(states))),flush=True)


def completed(a,p,identity):
    td=folder(a,identity)
    if not (td/'result.json').exists():return False
    rec=read(td/'result.json')
    if rec['status']!='PASS' or rec['identity']!=identity or rec['protocol_sha256']!=sha(a.results_root/'protocol.json'):raise ValueError('Native cache receipt changed')
    for name,digest in rec['files'].items():
        if sha(td/name)!=digest:raise ValueError('Native cache record changed')
    return True


def models_and_prices(a,p,identity):
    name=fold_name(identity[1],identity[2]);cache=a.cache_training/'folds'/name/'model.pkl';hgb=a.old_hgb_training/'folds'/name/'model.pkl';price=a.calibration_root/'folds'/name/'prices.json';oldprice=a.old_price_calibration/'folds'/name/'prices.json'
    for path in (cache,hgb,oldprice):
        if sha(path)!=p['input_files'][str(path)]:raise ValueError('Frozen model/old price changed')
    if sha(price)!=p['calibration_files'][str(price)]:raise ValueError('New price changed')
    cache_bundle=pickle.loads(cache.read_bytes());hgb_bundle=pickle.loads(hgb.read_bytes());prices=read(price);old=read(oldprice)
    from utils.sf_hlm_cache_calibration import validate_bundle
    validate_bundle(cache_bundle,supported_context_rules=('none',p.get('source_context_rule','none')))
    if cache_bundle['fold']!=hgb_bundle['fold'] or cache_bundle['fold']!=prices['fold'] or prices['fold']!=old['fold']:raise ValueError('Cross-fit models/prices misaligned')
    fold=prices['fold'];si=read(a.calibration_root/'protocol.json')['identities'].index(identity)
    if si not in fold['test'] or si in fold['train']:raise ValueError('Test state seen by cache model')
    if prices['cache_model_sha256']!=sha(cache) or old['model_sha256']!=sha(hgb):raise ValueError('Wrong price scorer')
    merged=dict(prices={**old['prices'],**prices['prices']},powers=prices['powers'])
    return cache_bundle['model'],hgb_bundle['model'],merged,dict(cache_model_sha256=sha(cache),hgb_model_sha256=sha(hgb),cache_price_sha256=sha(price),old_price_sha256=sha(oldprice),fold=fold)


def compatibility(rows,pw,model):
    import numpy as np
    from utils.sf_hlm_cache_matching import CacheWeightKernel,COLD_VARIANTS,CONTEXT_RULE
    from utils.sf_hlm_cache_mixture import CacheMixtureKernel,probability
    from utils.sf_hlm_cache_calibration import eligible,cache_features,controls
    original=CacheMixtureKernel(rows,pw);all_indices=[(i,action) for i,parts in enumerate(original.parts) for action,part in parts.items() if eligible(part)]
    mode=getattr(model,'public_cache_feature_mode','clipped')
    from utils.sf_hlm_cache_context import public_context
    guard=public_context(rows)['global']['all_visible_miss']
    model_guard=getattr(model,'public_cache_context_rule','none')==CONTEXT_RULE and guard
    expected=np.zeros(len(all_indices)) if model_guard else model.predict_proba(np.asarray([cache_features(rows[i],a,mode=mode) for i,a in all_indices],dtype=np.float64))[:,1] if all_indices else []
    checks=0
    for variant in VARIANT_METHODS:
        actual=CacheWeightKernel(rows,pw,variant,model)
        if actual.eligible_edges!=len(all_indices):raise ValueError('Deployment eligibility changed')
        index={pair:k for k,pair in enumerate(all_indices)}
        for i,parts in enumerate(original.parts):
            for action,part in parts.items():
                observed=actual.parts[i][action]
                for key in part:
                    if key!='p' and part[key]!=observed[key]:raise ValueError('Cache weight changed physical branch')
                ordinary=COLD_VARIANTS.get(variant,variant)
                guarded=guard and (variant in COLD_VARIANTS or model_guard and variant=='learned')
                wanted=part['p'] if (i,action) not in index else 0. if guarded else part['p'] if ordinary=='prior' else float(expected[index[i,action]]) if ordinary=='learned' else controls(rows[i],action)[ordinary]
                if observed['p']!=wanted:raise ValueError('Deployment cache weights differ from frozen scorer')
                checks+=1
    return dict(public_edge_weight_checks=checks,eligible_edges=len(all_indices),all_frozen_predictions_equal=True,private_components_unavailable=True)


def benchmark(a,p,entry,runtime,fast,*,measure=True,reference_root=None):
    import numpy as np
    from utils.sf_hlm_cache_matching import CacheMatchingPolicy
    from utils.canonical_experiment import digest
    from utils.snapshot_evaluator import evaluate
    identity=entry['identity'];meta=read(entry['state']);inp=read(meta['input']);blob=Path(meta['observed']).read_bytes();snapshot=pickle.loads(Path(meta['snapshot']).read_bytes())
    for path,digest0 in meta['files'].items():
        if sha(path)!=digest0:raise ValueError('State bytes changed')
    if sha(entry['state'])!=entry['state_sha256']:raise ValueError('State metadata changed')
    _,_,runner,models,lower,basis=runtime;f=archive(a);core=f.core(argparse.Namespace(release_root=a.feasibility_release))
    env,obs=pickle.loads(blob);_,_,requests,rows=runner.base.prepare(env);wanted=read(entry['public_rows'])['rows']
    for row in wanted:row['options']={int(k):v for k,v in row['options'].items()}
    if digest(rows)!=digest(wanted) or [q[2] for q in requests]!=inp['request_ids']:raise ValueError('Native proposal observations changed')
    if not np.array_equal(np.asarray(snapshot.motion),np.asarray(inp['motion'])):raise ValueError('Physical snapshot upper motion differs')
    cache_model,hgb_model,prices,locks=models_and_prices(a,p,identity);pw=powers()
    if pw!=prices['powers']:raise ValueError('Public power changed')
    checks=compatibility(rows,pw,cache_model);population=specs(identity);fp=runner.base.policy_fingerprint([models['upper'],lower]);cache_fp=hashlib.sha256(pickle.dumps(cache_model,protocol=5)).hexdigest();hgb_fp=hashlib.sha256(pickle.dumps(hgb_model,protocol=5)).hexdigest()
    refs={};timed={};order=[];td=folder(a,identity)
    with fast.FastRuntime() as rt:
        policy=CacheMatchingPolicy(rt,hgb_model,cache_model,prices,pw);rt.patch(core,'truncated_search',policy.run);rt.enable_sparse(lower)
        def call(spec):
            env,obs=pickle.loads(blob);motion,actions,info,observed_rows=rt.complete_policy(core,env,obs,spec,models,lower,runner)
            if digest(rows)!=digest(observed_rows) or info['request_ids']!=inp['request_ids'] or not np.array_equal(motion,np.asarray(inp['motion'])):raise ValueError('Native call changed observed state/motion')
            d=info['diagnostic']
            if d['chosen_predicted']['energy']>d['common_energy_cap']+1e-7 or d['matching_calls']>d['matching_budget']:raise ValueError('Native guard/budget violation')
            return info
        for spec in population:
            info=call(spec);key=(spec['method'],spec['repeat']);refs[key]=dict(**spec,assignment=list(info['assignment']),diagnostic=normal(info['diagnostic']),
                logical_upload_bytes=info['logical_upload_bytes'],logical_download_bytes=info['logical_download_bytes'])
        public=dict(identity=identity,rows_sha256=digest(rows),request_ids=inp['request_ids'],motion=inp['motion'],locks=locks,basis=basis,
            model_fingerprint=fp,compatibility=checks,selections=list(refs.values()),public_inverse_approximation_retained=True)
        if reference_root is not None:
            rd=Path(reference_root)/'tasks'/('cache_'+'_'.join(map(str,identity)));old_public=read(rd/'public.json');receipt=read(rd/'result.json')
            for name,h in receipt['files'].items():
                if sha(rd/name)!=h:raise ValueError('Audited reference changed')
            if old_public['request_ids']!=inp['request_ids'] or old_public['motion']!=inp['motion'] or old_public['rows_sha256']!=digest(rows):raise ValueError('Reference state differs')
            legacy_refs={(r['method'],r['repeat']):r for r in old_public['selections']};replayed=0
            for key,ref in refs.items():
                if key not in legacy_refs:continue
                previous=legacy_refs[key]
                if ref['assignment']!=previous['assignment'] or decision_diagnostic(ref['diagnostic'])!=decision_diagnostic(previous['diagnostic']):raise ValueError('Legacy public decision changed')
                replayed+=1
            public.update(reference_public_sha256=sha(rd/'public.json'),legacy_public_branches_replayed=replayed,common_delay_only=True)
        write(td/'public.json',public)
        for iteration in (range(3) if measure else ()):
            schedule=np.random.default_rng(seed_for('cache_matching_timing_order',identity,iteration)).permutation(len(population))
            for rank,position in enumerate(schedule):
                start=time.monotonic_ns();spec=population[int(position)];info=call(spec);stop=time.monotonic_ns();key=spec['method'],spec['repeat'];ref=refs[key]
                if list(info['assignment'])!=ref['assignment'] or decision_diagnostic(info['diagnostic'])!=decision_diagnostic(ref['diagnostic']):raise ValueError('Formal call differs from sealed public decision')
                rec=timed.setdefault(key,dict(**spec,software_seconds=[],search_seconds=[],timing_iteration=[],selected_order=[],execution_rank=[],observation_ns=[]))
                rec['software_seconds'].append(info['policy_seconds']);rec['timing_iteration'].append(iteration);rec['selected_order'].append(int(position));rec['execution_rank'].append(rank);rec['observation_ns'].append([start,stop]);order.append([iteration,int(position)])
                rec['search_seconds'].append(info['diagnostic'].get('search_seconds'))
    # All public decisions and times are sealed before private physics is read.
    write(td/'timing.json',dict(identity=identity,records=list(timed.values()),timing_order=order,cpu=sorted(os.sched_getaffinity(0))))
    if sha(entry['benchmark'])!=entry['benchmark_sha256']:raise ValueError('Original branch benchmark changed')
    original=read(entry['benchmark']);legacy={(x['method'],x['repeat']):x for x in original['results'] if x['method'] in ('G','R8','R16','L16')}
    outcomes={};mapping={};reused_outcomes=0;previous_outcomes={}
    if reference_root is not None:
        previous_labels=read(rd/'labels.json')
        if previous_labels['generated']!=meta['generated'] or previous_labels['admitted']!=meta['admitted']:raise ValueError('Reference physics denominator differs')
        previous_outcomes={digest(x['assignment']):x for x in previous_labels['outcomes'].values()}
    for ref in refs.values():
        assignment=ref['assignment'];key=digest(assignment);mapping[ref['method']+':'+str(ref['repeat'])]=key
        if key not in outcomes:
            if key in previous_outcomes:
                actual=previous_outcomes[key];reused_outcomes+=1
                if actual['assignment']!=assignment:raise ValueError('Reference allocation digest collision')
            else:actual=evaluate(snapshot,assignment)
            if 'assignment' in actual and list(actual['assignment'])!=assignment:raise ValueError('Evaluator assignment differs')
            if actual['metrics']['generated_service_count']!=meta['generated'] or actual['metrics']['admitted_service_count']!=meta['admitted']:raise ValueError('Private physics denominator changed')
            runner.base.assert_step(actual['metrics'],dict(request_ids=inp['request_ids'],assignment=tuple(assignment),diagnostic=ref['diagnostic']))
            outcomes[key]=dict(assignment=assignment,requests=actual['requests'],metrics=actual['metrics'])
        legacy_ref=legacy.get((ref['method'],ref['repeat']))
        if legacy_ref is not None:
            if assignment!=legacy_ref['assignment'] or normal(outcomes[key]['requests'])!=normal(legacy_ref['requests']) or normal(outcomes[key]['metrics'])!=normal(legacy_ref['metrics']):raise ValueError('Native legacy branch differs from frozen original')
    if runner.base.policy_fingerprint([models['upper'],lower])!=fp or hashlib.sha256(pickle.dumps(cache_model,protocol=5)).hexdigest()!=cache_fp or hashlib.sha256(pickle.dumps(hgb_model,protocol=5)).hexdigest()!=hgb_fp:raise ValueError('Evaluation changed model bytes')
    write(td/'labels.json',dict(identity=identity,generated=meta['generated'],admitted=meta['admitted'],outcomes=outcomes,mapping=mapping,
        public_sha256=sha(td/'public.json'),timing_sha256=sha(td/'timing.json'),private_truth_for_scoring_only=True,
        reference_physical_outcomes_reused=reused_outcomes))
    write(td/'result.json',dict(status='PASS',identity=identity,protocol_sha256=sha(a.results_root/'protocol.json'),files={name:sha(td/name) for name in ('public.json','timing.json','labels.json')},timing_samples=len(order),unique_physical_outcomes=len(outcomes)))


def worker(a,p):
    verify(a,p);job=json.loads(a.job)
    if job not in p['jobs']:raise ValueError('Unknown cache native job')
    res=resource(a)
    with res.task_lock(a.results_root/'locks'/('_'.join(map(str,job))+'.lock')):
        with res.CpuLease(a.resources/'isolated',task=['cache_matching',*job],kind='environment'):
            runtime=archive(a).boot(a,job[1]);fast=modules(a)
            for n,entry in enumerate(s for s in p['states'] if s['identity'][:2]==job):
                if not completed(a,p,entry['identity']):benchmark(a,p,entry,runtime,fast)
                write(a.results_root/'progress'/('_'.join(map(str,job))+'.json'),dict(completed=n+1,last_identity=entry['identity'],updated_unix=time.time()))


def analyze(a,p):
    import numpy as np
    f=archive(a);core=f.core(argparse.Namespace(release_root=a.feasibility_release));states=[]
    for identity in p['identities']:
        if not completed(a,p,identity):raise ValueError('Native cache population incomplete')
        td=folder(a,identity);pub=read(td/'public.json');timing=read(td/'timing.json');labels=read(td/'labels.json');methods={}
        core.require_population([(s['method'],s['repeat']) for s in pub['selections']],[(s['method'],s['repeat']) for s in specs(identity)])
        core.require_population([(s['method'],s['repeat']) for s in timing['records']],[(s['method'],s['repeat']) for s in specs(identity)])
        for method in METHODS:
            records=[]
            for selection in (s for s in pub['selections'] if s['method']==method):
                repeat=selection['repeat'];t=next(x for x in timing['records'] if x['method']==method and x['repeat']==repeat);truth=labels['outcomes'][labels['mapping'][method+':'+str(repeat)]]
                records.append(dict(raw=core.fee_count(truth['requests'],0.),fee=core.mean_fee_count(truth['requests'],t['software_seconds']),
                    energy=truth['metrics']['uav_fleet_total_energy'],mbs=truth['metrics']['service_offloads_mbs'],fairness=truth['metrics']['fairness'],
                    calls=selection['diagnostic']['matching_calls'],times=t['software_seconds']))
            methods[method]={k:statistics.mean(x[k] for x in records) for k in ('raw','fee','energy','mbs','fairness','calls')};methods[method]['times']=[v for x in records for v in x['times']]
        states.append(dict(identity=identity,generated=labels['generated'],admitted=labels['admitted'],methods=methods))
    subsets={}
    for source in ('all','greedy','learned'):
        rows=[s for s in states if source=='all' or s['identity'][0]==source];blocks={}
        for block in sorted({s['identity'][1] for s in rows}):
            rr=[s for s in rows if s['identity'][1]==block];den=sum(s['generated'] for s in rr);admitted=sum(s['admitted'] for s in rr);mm={}
            for method in METHODS:
                ts=[t for s in rr for t in s['methods'][method]['times']]
                mm[method]=dict(raw_rate=sum(s['methods'][method]['raw'] for s in rr)/den,fee_rate=sum(s['methods'][method]['fee'] for s in rr)/den,
                    energy=sum(s['methods'][method]['energy'] for s in rr),mbs_share=sum(s['methods'][method]['mbs'] for s in rr)/max(admitted,1),
                    fairness=statistics.mean(s['methods'][method]['fairness'] for s in rr),calls=statistics.mean(s['methods'][method]['calls'] for s in rr),
                    mean_seconds=statistics.mean(ts),p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
            blocks[str(block)]=dict(generated=den,admitted=admitted,methods=mm)
        groups={}
        for method in METHODS:
            ts=[t for s in rows for t in s['methods'][method]['times']]
            groups[method]={k:statistics.mean(b['methods'][method][k] for b in blocks.values()) for k in ('raw_rate','fee_rate','mbs_share','fairness','calls')}
            groups[method].update(energy_ratio_to_G=statistics.mean(b['methods'][method]['energy']/b['methods']['G']['energy'] for b in blocks.values()),
                pooled_mean_seconds=statistics.mean(ts),pooled_p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
        comparisons={method:{metric:f.paired([b['methods'][PRIMARY][metric+'_rate']-b['methods'][method][metric+'_rate'] for b in blocks.values()]) for metric in ('raw','fee')} for method in METHODS if method!=PRIMARY}
        subsets[source]=dict(blocks=blocks,groups=groups,comparisons=comparisons)
    main=subsets['all'];best=max(NONLEARNING,key=lambda m:main['groups'][m]['fee_rate']);gain=main['comparisons']['G']['fee'];increment=main['comparisons'][best]['fee'];energy=main['groups'][PRIMARY]['energy_ratio_to_G'];time_ok=all(b['methods'][PRIMARY]['p95_seconds']<1 for b in main['blocks'].values())
    passed=gain['mean']>=.005 and gain['positive']>=4 and increment['mean']>=.001 and energy<=1.05 and time_ok
    out=dict(status='COMPLETED_SMOKE' if p['smoke'] else 'COMPLETED_DEVELOPMENT',protocol_sha256=sha(a.results_root/'protocol.json'),states=len(states),branch_instances=len(states)*p['branch_instances'],timing_samples=len(states)*p['timing_samples_per_state'],subsets=subsets,
        decision=dict(gain_vs_G=gain,best_nonlearning=best,learning_increment=increment,energy_ratio_to_G=energy,all_block_p95_below_second=time_ok,native_development_signal=passed,independent_confirmation=False,goal_complete=False))
    write(a.results_root/'analysis/summary.json',out);write(a.results_root/'analysis/states.json',states);print(json.dumps(out['decision']),flush=True)


def pipeline(a,*,worker_mode='worker',analyze_fn=None):
    p=read(a.results_root/'protocol.json');verify(a,p,True);res=resource(a)
    def run(job):
        path=a.results_root/'worker_logs'/('_'.join(map(str,job))+'.log');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as stream:out=subprocess.run(command(a,worker_mode)+['--job',json.dumps(job)],cwd=a.release_root,stdout=stream,stderr=subprocess.STDOUT)
        if out.returncode:raise RuntimeError('Cache native worker failed '+str(job))
    with res.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                jobs=[pool.submit(run,j) for j in p['jobs']]
                for future in as_completed(jobs):future.result();write(a.results_root/'status.json',dict(status='RUNNING',completed=sum(completed(a,p,i) for i in p['identities']),total=len(p['identities']),updated_unix=time.time()))
            with res.CpuLease(a.resources,task=['cache_matching_analysis'],kind='environment'):(analyze if analyze_fn is None else analyze_fn)(a,p)
            write(a.results_root/'completion.json',dict(status='COMPLETED',states=len(p['identities']),timing_samples=len(p['identities'])*p['timing_samples_per_state'],
                protocol_sha256=sha(a.results_root/'protocol.json'),summary_sha256=sha(a.results_root/'analysis/summary.json')))
            write(a.results_root/'status.json',dict(status='COMPLETED',states=len(p['identities']),updated_unix=time.time()))
        except BaseException as error:write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','calibrate','fold','init','pipeline','worker','analyze'))
    for name in ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--block',type=int);parser.add_argument('--workload',type=int);parser.add_argument('--job');parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Scientific matching, calibration and statistics run remotely')
    if a.mode=='prepare':prepare(a)
    elif a.mode=='calibrate':calibrate_pipeline(a)
    elif a.mode=='init':init(a)
    elif a.mode=='pipeline':pipeline(a)
    elif a.mode=='worker':worker(a,read(a.results_root/'protocol.json'))
    else:
        res=resource(a)
        with res.CpuLease(a.resources,task=['cache_matching',a.mode,a.block,a.workload],kind='environment'):
            if a.mode=='fold':calibrate_fold(a,read(a.calibration_root/'protocol.json'))
            else:analyze(a,read(a.results_root/'protocol.json'))


if __name__=='__main__':main()
