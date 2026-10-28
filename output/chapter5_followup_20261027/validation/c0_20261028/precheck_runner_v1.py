"""Prospective C0 fixed-budget development; all numerical modes are server-only."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import chapter5_stage_b as b

ROOT=Path(__file__).resolve().parents[2]
METHODS=['G','B2','B4','B6','B10','native_B10']
FILES=['scripts/analysis/chapter5_stage_c0.py','scripts/analysis/chapter5_budget_prefix.py',
       'scripts/analysis/chapter5_stage_b.py','tests/test_chapter5_budget_prefix.py']


def budget(method):return 0 if method=='G' else 10 if method=='native_B10' else int(method[1:])


def policy(rt,method,prices):
    from chapter5_budget_prefix import PrefixPolicy,NativeRecordedPolicy
    return NativeRecordedPolicy(rt,None,prices,prices['powers']) if method=='native_B10' else PrefixPolicy(rt,prices,prices['powers'],budget(method))


def job(method,block,workload,step):
    return dict(kind='mixture_price',cap=budget(method),seed=int(b.digest(['stage_c0',block,workload,step])[:8],16))


def verify(a):
    p=b.read(a.output/'protocol.json')
    for rel,h in p['files'].items():
        if b.sha(ROOT/rel)!=h:raise ValueError('Frozen C0 source changed: '+rel)
    for path,h in p['inputs'].items():
        if b.sha(path)!=h:raise ValueError('Frozen input changed: '+path)
    return p


def init(a):
    if (a.output/'protocol.json').exists():raise FileExistsError('Immutable C0 namespace')
    bp=b.read(a.development/'protocol.json')
    assert b.read(a.development/'collection_complete.json')['status']=='PASS'
    assert b.read(a.development/'fixed_complete.json')['status']=='PASS'
    inputs=dict(bp['inputs'])
    for name in ['protocol.json','collection_complete.json','fixed_complete.json']:
        path=a.development/name;inputs[str(path)]=b.sha(path)
    states=[]
    for block in bp['blocks']:
        for workload in bp['workloads']:
            for source in ['G','mixture_price']:
                for sd in sorted((a.development/'tasks'/f'{block}_{workload}_center'/source/'samples').glob('*')):
                    pub=b.read(sd/'public.json')
                    if pub['sample_role']!='main':continue
                    states.append(dict(identity=[block,workload,source,pub['step']],folder=str(sd),
                        public_sha256=b.sha(sd/'public.json'),observed_sha256=b.sha(sd/'observed.bin')))
    assert len(states)==1200
    cpus=bp['cpus'][:15]
    if len(cpus)!=15:raise ValueError('15 distinct physical cores required')
    physical=[]
    for cpu in cpus:
        root=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.append([(root/'physical_package_id').read_text().strip(),(root/'core_id').read_text().strip()])
    assert len({tuple(z) for z in physical})==15
    bundles=[[x,y] for x in bp['blocks'] for y in bp['workloads']]
    random.Random(2026102801).shuffle(bundles)
    orders={f'{x}_{y}':random.Random(f'c0_{x}_{y}').sample(METHODS,len(METHODS)) for x,y in bundles}
    p=dict(name='chapter5_stage_c0_fixed_prefix_20261028_v1',files={r:b.sha(ROOT/r) for r in FILES},inputs=inputs,
        archive=str(a.archive),development=str(a.development),blocks=bp['blocks'],workloads=bp['workloads'],
        seed_purpose='reuse declared development workloads; not new or held out; no fitting in C0',
        budgets=[0,2,4,6,10],methods=METHODS,steps=1000,episodes=90,bundles=bundles,method_orders=orders,
        states=states,snapshot_repeats=3,workers=15,cpus=cpus,physical_cores=physical,
        budget_semantics='0=G; positive includes 1 binary initialization then first B-1 of unchanged proposal quota list; G retained; failed matching counts; no budget 1',
        implementation='lazy matrix scoring only for visited quotas; all prefixes share implementation; original batched native B10 separately remeasured',
        timing='complete FastRuntime policy call; all preparation/repair/proposals/scoring/matching/selection/feedback/upper/mapping/logical serialization; disk and env restore outside; source model loading barrier',
        hardware='new up-to-15 single-thread physical-core regime; shared server background load; no comparison to old B-stage clocks',
        floating_tolerance=dict(rtol=1e-12,atol=1e-10,integer_identity_action_call_count='exact'),
        primary='paired block S/G difference versus common-implementation B10; full-time mean ratio and fleet-energy ratio',
        exploratory_screen=dict(max_dsr_loss_pp=.1,max_full_time_ratio=.80,max_energy_ratio=1.05,max_energy_ratio_G=1.05),
        screen_status='fixed-budget development screen only; no formal noninferiority or adaptive policy claim',
        statistical_unit='five historical upper blocks; pool counts within blocks; state/time repetitions do not enlarge n',
        failure='retain all outcomes; technical failure reruns require explicit new attempt; no outcome-dependent seed replacement',
        scope='C0 only; no C1 rules, no training, no C3 confirmation, no Git operations',
        smoke=dict(block=bp['blocks'][0],workload=bp['workloads'][0],steps=24,methods=METHODS))
    b.write(a.output/'protocol.json',p)
    print(json.dumps(dict(status='PREPARED',episodes=90,states=1200,methods=METHODS)),flush=True)


def close(left,right,tolerance):
    if isinstance(left,dict):
        assert set(left)==set(right)
        for k in left:close(left[k],right[k],tolerance)
    elif isinstance(left,list):
        assert len(left)==len(right)
        for x,y in zip(left,right):close(x,y,tolerance)
    elif isinstance(left,float):
        assert math.isclose(left,right,rel_tol=tolerance['rtol'],abs_tol=tolerance['atol']),(left,right)
    else:assert left==right,(left,right)


def compatibility(a,p,block,fast,prices):
    from utils import sf_hlm_calibrated_matching as cm
    from utils.sf_hlm_matching import feedback
    from utils.sf_hlm_energy_price import choose
    rows_checked=[]
    for state in p['states']:
        if state['identity'][0]!=block:continue
        sd=Path(state['folder']);assert b.sha(sd/'public.json')==state['public_sha256']
        rows=b.read(sd/'public.json')['rows']
        for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
        with fast.FastRuntime() as rt:
            calls=[];original=cm.solve
            def logged(*args,**kwargs):
                calls.append([int(v) for v in args[1]])
                return original(*args,**kwargs)
            cm.solve=logged
            try:
                native=policy(rt,'native_B10',prices)
                expected=native.run(rows,num_uavs=5,kind='mixture_price',cap=10,trace=True)
            finally:cm.solve=original
            controls={}
            for method in ['B2','B4','B6','B10']:
                impl=policy(rt,method,prices)
                got=impl.run(rows,num_uavs=5,kind='mixture_price',cap=budget(method),trace=True)
                trace=impl.last_trace
                assert [z['quota'] for z in trace['visited']]==calls[:budget(method)-1]
                assert trace['initialization_calls']+sum(z['matching_call'] for z in trace['visited'])==got[1]['matching_calls']
                assert got[1]['matching_calls']<=budget(method)
                assert {tuple(q) for q in trace['computed_matrix_quotas']}=={tuple(q) for q in calls[:budget(method)-1]}
                assert len(trace['computed_matrix_quotas'])<=budget(method)-1
                allowed={-1,*[z['action'] for z in trace['visited']]}
                pool=[c for c in expected[2] if c['action'] in allowed]
                close(b.normal(got[2]),b.normal(pool),p['floating_tolerance'])
                params=prices['prices']['mixture']
                chosen=choose([dict(action=c['action'],score=c['expected_success'],energy=c['predicted']['energy']) for c in pool],params['price'],params['normalizer'])
                assert list(got[0])==list(pool[chosen]['assignment'])
                if method=='B10':
                    close(b.normal(got),b.normal(expected),p['floating_tolerance'])
                    assert b.normal(feedback(rows,got[0],5))==b.normal(feedback(rows,expected[0],5))
                controls[method]=dict(calls=got[1]['matching_calls'],visited=len(trace['visited']),matrices=len(trace['computed_matrix_quotas']))
        rows_checked.append(dict(identity=state['identity'],status='PASS',controls=controls))
    b.write(a.output/'compatibility'/f'{block}.json',dict(status='PASS',states=len(rows_checked),checks=rows_checked))


def episode(a,p,block,workload,method,runtime,fast,core,prices,smoke):
    import numpy as np
    from utils.canonical_experiment import exogenous_frame
    from utils.sf_hlm_software_delay import SoftwareServiceDelay
    _,_,runner,models,_,basis=runtime
    folder=a.output/('smoke' if smoke else 'closed_loop')/f'{block}_{workload}'/method
    if folder.exists():raise FileExistsError('Keep previous raw attempts: '+str(folder))
    folder.mkdir(parents=True);env,obs=runner.episode_env(workload,0)
    fp=runner.base.policy_fingerprint([models['upper']]);records=[];totals={k:0. for k in b.COUNTERS};external=[]
    with fast.FastRuntime() as rt:
        impl=policy(rt,method,prices);rt.patch(core,'truncated_search',impl.run)
        warm_env,warm_obs=pickle.loads(pickle.dumps((env,obs),protocol=5))
        rt.complete_policy(core,warm_env,warm_obs,job(method,block,workload,0),models,None,runner)
        with SoftwareServiceDelay(env,'measured') as hook:
            for step in range(p['smoke']['steps'] if smoke else p['steps']):
                frame=b.digest(exogenous_frame(env));external.append(frame)
                start=time.monotonic_ns();motion,actions,info,rows=rt.complete_policy(core,env,obs,job(method,block,workload,step),models,None,runner);end=time.monotonic_ns()
                hook.arm(info['request_ids'],info['policy_seconds']);nxt,reward,metrics=env.step(motion,offloading_actions=actions);outcome=hook.finish()
                runner.base.assert_step(metrics,info)
                assert outcome['success']==metrics['deadline_satisfied_count']
                assert len(outcome['requests'])==len(rows)==len(info['assignment'])
                if method!='G':
                    assert info['diagnostic']['matching_calls']<=budget(method)
                    assert info['diagnostic']['chosen_predicted']['energy']<=info['diagnostic']['common_energy_cap']+1e-7
                for k,v in b.COUNTERS.items():totals[k]+=float(metrics[v])
                records.append(dict(step=step,exogenous_hash=frame,public_rows_hash=b.digest(rows),metrics=b.normal(metrics),
                    info=b.normal(info),public_trace=b.normal(impl.last_trace),requests=outcome['requests'],motion=motion.tolist(),
                    injected_seconds=outcome['injected_seconds'],window_ns=[start,end]))
                obs=np.asarray(nxt,np.float32)
                if (step+1)%100==0:b.write(folder/'progress.json',dict(step=step+1,unix=time.time()))
    assert fp==runner.base.policy_fingerprint([models['upper']])
    with gzip.open(folder/'steps.json.gz','wt') as f:json.dump(b.normal(records),f,separators=(',',':'),allow_nan=False)
    receipt=dict(status='PASS',block=block,workload=workload,method=method,steps=len(records),totals=totals,
        exogenous_hash=b.digest(external),upper_fingerprint=fp,basis=basis,steps_sha256=b.sha(folder/'steps.json.gz'),
        protocol_sha256=b.sha(a.output/'protocol.json'),cpu=sorted(os.sched_getaffinity(0)))
    b.write(folder/'result.json',receipt);return receipt


def fixed_timing(a,p,block,workload,runtime,fast,core,prices):
    _,_,runner,models,_,_=runtime
    records=[]
    with fast.FastRuntime() as rt:
        for state in p['states']:
            if state['identity'][:2]!=[block,workload]:continue
            sd=Path(state['folder']);assert b.sha(sd/'observed.bin')==state['observed_sha256']
            blob=(sd/'observed.bin').read_bytes();runs={m:[] for m in METHODS}
            for method in METHODS:
                impl=policy(rt,method,prices)
                # Per-state unmeasured warmup on an isolated non-advancing clone.
                old=core.truncated_search;core.truncated_search=impl.run
                env,obs=pickle.loads(blob);rt.complete_policy(core,env,obs,job(method,block,workload,state['identity'][3]),models,None,runner)
                core.truncated_search=old
            for repeat in range(p['snapshot_repeats']):
                order=random.Random(f"c0_fixed_{state['identity']}_{repeat}").sample(METHODS,len(METHODS))
                for method in order:
                    impl=policy(rt,method,prices);old=core.truncated_search;core.truncated_search=impl.run
                    try:
                        env,obs=pickle.loads(blob);start=time.monotonic_ns()
                        motion,actions,info,rows=rt.complete_policy(core,env,obs,job(method,block,workload,state['identity'][3]),models,None,runner)
                        end=time.monotonic_ns()
                    finally:core.truncated_search=old
                    runs[method].append(dict(repeat=repeat,info=b.normal(info),motion=motion.tolist(),
                        trace=b.normal(impl.last_trace),window_ns=[start,end]))
            assert all(z['info']['assignment']==runs['B10'][0]['info']['assignment'] for z in runs['native_B10'])
            for method in METHODS:
                assert all(z['info']['assignment']==runs[method][0]['info']['assignment'] for z in runs[method])
            records.append(dict(identity=state['identity'],source=state,runs=runs))
            if len(records)%10==0:b.write(a.output/'fixed_progress'/f'{block}_{workload}.json',dict(states=len(records)))
    assert len(records)==80
    path=a.output/'fixed_timing'/f'{block}_{workload}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    with gzip.open(path,'wt') as f:json.dump(records,f,separators=(',',':'),allow_nan=False)
    b.write(path.with_suffix('.receipt.json'),dict(status='PASS',states=80,sha256=b.sha(path)))


def worker(a):
    p=verify(a);block,workload=json.loads(a.bundle);os.sched_setaffinity(0,{a.cpu})
    f,runtime,fast,core,prices=b.boot(a,block)
    if a.phase=='compatibility':return compatibility(a,p,block,fast,prices)
    barrier=a.output/'barriers'/a.phase
    b.write(barrier/'ready'/f'{block}_{workload}.json',dict(pid=os.getpid(),cpu=a.cpu))
    while not (barrier/'start.json').exists():
        if (barrier/'abort.json').exists():raise RuntimeError('Barrier aborted')
        time.sleep(.05)
    if a.phase=='fixed':return fixed_timing(a,p,block,workload,runtime,fast,core,prices)
    receipts=[episode(a,p,block,workload,m,runtime,fast,core,prices,a.phase=='smoke') for m in p['method_orders'][f'{block}_{workload}']]
    assert len({z['exogenous_hash'] for z in receipts})==len({z['upper_fingerprint'] for z in receipts})==1
    b.write(a.output/('smoke_bundles' if a.phase=='smoke' else 'closed_bundles')/f'{block}_{workload}.json',dict(status='PASS',episodes=len(receipts)))


def controller(a,phase):
    p=verify(a)
    if phase in ('closed','fixed'):assert b.read(a.output/'smoke_complete.json')['status']=='PASS'
    if phase=='fixed':assert b.read(a.output/'closed_complete.json')['status']=='PASS'
    bundles=[[x,p['workloads'][0]] for x in p['blocks']] if phase=='compatibility' else [p['bundles'][0]] if phase=='smoke' else p['bundles']
    if phase=='smoke':
        assert all(b.read(a.output/'compatibility'/f'{x}.json')['status']=='PASS' for x in p['blocks'])
    logs=a.output/'logs'/phase;logs.mkdir(parents=True,exist_ok=True)
    if phase!='compatibility' and (a.output/'barriers'/phase).exists():raise FileExistsError('Keep prior barrier/attempt')
    children=[]
    for i,pair in enumerate(bundles):
        cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--archive',str(a.archive),'--development',str(a.development),
            '--output',str(a.output),'--bundle',json.dumps(pair),'--cpu',str(p['cpus'][i]),'--phase',phase]
        stream=(logs/('_'.join(map(str,pair))+'.log')).open('ab')
        children.append((pair,subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT),stream))
    if phase!='compatibility':
        barrier=a.output/'barriers'/phase
        while len(list((barrier/'ready').glob('*.json')))<len(bundles):
            if any(child.poll() is not None for pair,child,stream in children):
                b.write(barrier/'abort.json',dict(status='ABORT'));raise RuntimeError('Model loading failed')
            time.sleep(.2)
        b.write(barrier/'start.json',dict(status='START',unix=time.time(),bundles=bundles))
    done=0
    while children:
        for entry in children[:]:
            pair,child,stream=entry;code=child.poll()
            if code is None:continue
            stream.close();children.remove(entry)
            if code:raise RuntimeError(f'{phase} bundle failed: {pair}; preserve its data')
            done+=1;b.write(a.output/(phase+'_status.json'),dict(status='RUNNING',completed=done,total=len(bundles),unix=time.time()))
        if children:time.sleep(.2)
    b.write(a.output/(phase+'_complete.json'),dict(status='PASS',bundles=len(bundles),protocol_sha256=b.sha(a.output/'protocol.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['init','compatibility','smoke','closed','fixed','worker'])
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--development',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle');parser.add_argument('--cpu',type=int);parser.add_argument('--phase')
    a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Scientific work executes on the experiment server')
    if a.mode=='init':init(a)
    elif a.mode=='worker':worker(a)
    else:controller(a,a.mode)


if __name__=='__main__':main()
