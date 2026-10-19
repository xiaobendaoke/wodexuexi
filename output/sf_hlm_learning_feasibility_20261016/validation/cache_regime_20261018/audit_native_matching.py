"""Raw native-fee self-audit after isolated timing workers have finished."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

METHODS=('G','R8','R16','binary_price','mixture','mixture_price','own_proxy_price','half_proxy_price','cold_prior_price','cold_own_proxy_price','cold_half_proxy_price','cache_regime_probability_price','HGB_probability_price','L16')
PRIMARY='cache_regime_probability_price'
NONLEARNING=METHODS[:11]
RULE='all_visible_miss_zero_v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def seed_for(*parts):return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).digest()[:4],'big')
def close(x,y):
    if isinstance(x,dict):
        assert x.keys()==y.keys(),(x.keys(),y.keys())
        for key in x:close(x[key],y[key])
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        for a,b in zip(x,y):close(a,b)
    elif isinstance(x,(float,int)) and not isinstance(x,bool):assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12),(x,y)
    else:assert x==y,(x,y)
def effect(values):
    import itertools
    from scipy.stats import t
    n=len(values);mean=statistics.mean(values);radius=float(t.ppf(.975,n-1))*statistics.stdev(values)/math.sqrt(n)
    flip=sum(abs(statistics.mean(v*s for v,s in zip(values,signs)))>=abs(mean)-1e-14 for signs in itertools.product((-1,1),repeat=n))/(2**n)
    return dict(mean=mean,ci95=[mean-radius,mean+radius],values=values,n=n,positive=sum(v>0 for v in values),exact_signflip_p=flip)

def audit(a):
    import numpy as np
    root=a.results_root;p=read(root/'protocol.json');done=read(root/'completion.json');summary=read(root/'analysis/summary.json');cp=read(a.calibration_root/'protocol.json')
    assert done['status']=='COMPLETED' and done['summary_sha256']==sha(root/'analysis/summary.json') and done['protocol_sha256']==sha(root/'protocol.json')
    assert len(p['states'])==(8 if p['smoke'] else 400) and len(set(tuple(s['identity']) for s in p['states']))==len(p['states'])
    assert p['methods']==list(METHODS) and p['nonlearning']==list(NONLEARNING) and p['source_context_rule']==RULE and p['branch_instances']==22 and p['timing_samples_per_state']==66
    for file,h in p['files'].items():assert sha(a.release_root/file)==h
    for file,h in p['calibration_source_files'].items():
        assert sha(a.calibration_release/file)==h
        if file.startswith('utils/') or file=='paths.py':assert sha(a.release_root/file)==h
    for file,h in p['input_files'].items():assert sha(file)==h
    for file,h in p['calibration_files'].items():assert sha(file)==h
    assert p['calibration_protocol_sha256']==sha(a.calibration_root/'protocol.json') and p['calibration_completion_sha256']==sha(a.calibration_root/'completion.json')
    price_audit=read(a.validation_root/'price_calibration_audit.json');assert price_audit['status']=='PASS' and price_audit['calibration_completion_sha256']==p['calibration_completion_sha256']
    cpus=set(read(a.resources/'isolated/profile.json')['cpus']);intervals={cpu:[] for cpu in cpus};states=[];counts=dict(instances=0,times=0,physical_requests=0,legacy_instances=0);weight_checks=0
    for entry in p['states']:
        identity=entry['identity'];td=root/'tasks'/('cache_'+'_'.join(map(str,identity)));receipt=read(td/'result.json');public=read(td/'public.json');timing=read(td/'timing.json');labels=read(td/'labels.json')
        assert receipt['status']=='PASS' and receipt['identity']==identity and receipt['protocol_sha256']==sha(root/'protocol.json')
        for file,h in receipt['files'].items():assert sha(td/file)==h
        assert labels['public_sha256']==sha(td/'public.json') and labels['timing_sha256']==sha(td/'timing.json')
        assert len(timing['cpu'])==1 and timing['cpu'][0] in cpus;cpu=timing['cpu'][0]
        expected=[(m,rep) for m in METHODS for rep in range(5 if m in ('R8','R16') else 1)]
        assert [(s['method'],s['repeat']) for s in public['selections']]==expected
        actual={(t['method'],t['repeat']):t for t in timing['records']};assert set(actual)==set(expected) and len(actual)==22
        wanted_order=[]
        for iteration in range(3):
            wanted_order.extend([[iteration,int(i)] for i in np.random.default_rng(seed_for('cache_matching_timing_order',identity,iteration)).permutation(22)])
        assert timing['timing_order']==wanted_order
        name=f'block_{identity[1]}_workload_{identity[2]}';locks=public['locks'];price_path=a.calibration_root/'folds'/name/'prices.json';prices=read(price_path)
        assert locks['cache_model_sha256']==sha(a.cache_training/'folds'/name/'model.pkl') and locks['hgb_model_sha256']==sha(a.old_hgb_training/'folds'/name/'model.pkl')
        assert locks['cache_price_sha256']==sha(price_path) and locks['old_price_sha256']==sha(a.old_price_calibration/'folds'/name/'prices.json') and locks['fold']==prices['fold']
        si=cp['identities'].index(identity);assert si in locks['fold']['test'] and si not in locks['fold']['train']
        assert all(cp['identities'][i][1]!=identity[1] and cp['identities'][i][2]!=identity[2] for i in locks['fold']['train'])
        assert public['compatibility']['all_frozen_predictions_equal'];weight_checks+=public['compatibility']['public_edge_weight_checks']
        assert sha(entry['benchmark'])==entry['benchmark_sha256'];original=read(entry['benchmark']);legacy={(s['method'],s['repeat']):s for s in original['results']}
        used=set()
        for key,truth in labels['outcomes'].items():
            assert key==digest(truth['assignment']);requests=truth['requests'];metrics=truth['metrics'];counts['physical_requests']+=len(requests)
            assert [r['ue_id'] for r in requests]==public['request_ids'] and len(requests)==labels['admitted']==metrics['admitted_service_count']
            assert metrics['generated_service_count']==labels['generated']
            assert sum(r['latency']<=r['deadline'] for r in requests)==metrics['deadline_satisfied_count']
            assert sum(int(x)==1 for x in truth['assignment'])==metrics['service_offloads_mbs']
            assert all(math.isfinite(r['latency']) and r['latency']>=0 and r['deadline']>0 for r in requests)
            assert not metrics['forced_service_admissions'] and not metrics['service_fallback_count'] and not metrics['service_predict_exception_fallback_count']
        records={m:[] for m in METHODS}
        for position,selection in enumerate(public['selections']):
            method=selection['method'];rep=selection['repeat'];t=actual[method,rep];key=labels['mapping'][method+':'+str(rep)];truth=labels['outcomes'][key];used.add(key)
            assert selection['assignment']==truth['assignment'];d=selection['diagnostic'];assert d['matching_calls']<=d['matching_budget'] and d['chosen_predicted']['energy']<=d['common_energy_cap']+1e-7
            if method in ('cold_prior_price','cold_own_proxy_price','cold_half_proxy_price',PRIMARY):
                from utils.sf_hlm_cache_context import public_context
                guard=public_context(read(entry['public_rows'])['rows'])['global']['all_visible_miss']
                assert d['public_cache_context_rule']==RULE and d['public_cache_guard_triggered']==guard
                if guard or method!=PRIMARY:assert d['cache_prediction_batches']==0
                elif public['compatibility']['eligible_edges']:assert d['cache_prediction_batches']==1
            if method in ('R8','R16','L16'):assert d['feature_budget']==16 and selection['cap']==int(method[1:])
            if method in ('mixture','mixture_price','own_proxy_price','half_proxy_price','cold_prior_price','cold_own_proxy_price','cold_half_proxy_price',PRIMARY):assert d['cache_eligible_edges']==public['compatibility']['eligible_edges']
            assert t['seed']==selection['seed']==seed_for('sf_hlm_feasibility_20261016',identity,rep) and t['kind']==selection['kind'] and t['cap']==selection['cap']
            assert t['timing_iteration']==[0,1,2] and t['selected_order']==[position]*3 and len(t['software_seconds'])==len(t['search_seconds'])==3
            fees=[]
            for iteration,(seconds,inner,rank,span) in enumerate(zip(t['software_seconds'],t['search_seconds'],t['execution_rank'],t['observation_ns'])):
                assert math.isfinite(seconds) and seconds>=0 and span[1]>=span[0] and (span[1]-span[0])/1e9>=seconds
                assert inner is None or (math.isfinite(inner) and 0<=inner<=seconds)
                assert wanted_order[iteration*22+rank]==[iteration,position]
                intervals[cpu].append((span[0],span[1],identity,method,rep,iteration))
                fees.append(sum(r['latency']+seconds<=r['deadline'] for r in truth['requests']));counts['times']+=1
            if method in ('G','R8','R16','L16'):
                ref=legacy[method,rep];assert ref['assignment']==selection['assignment'] and ref['requests']==truth['requests'] and ref['metrics']==truth['metrics'];counts['legacy_instances']+=1
            records[method].append(dict(raw=sum(r['latency']<=r['deadline'] for r in truth['requests']),fee=sum(fees)/3,energy=truth['metrics']['uav_fleet_total_energy'],mbs=truth['metrics']['service_offloads_mbs'],fairness=truth['metrics']['fairness'],calls=d['matching_calls'],times=t['software_seconds']));counts['instances']+=1
        assert used==set(labels['outcomes'])
        methods={}
        for method,rr in records.items():
            methods[method]={k:sum(r[k] for r in rr)/len(rr) for k in ('raw','fee','energy','mbs','fairness','calls')};methods[method]['times']=[t for r in rr for t in r['times']]
        states.append(dict(identity=identity,generated=labels['generated'],admitted=labels['admitted'],methods=methods))
    for cpu,rr in intervals.items():
        rr.sort()
        for left,right in zip(rr,rr[1:]):assert left[1]<=right[0],(cpu,left[2:],right[2:])
    assert counts['times']==66*len(states) and counts['instances']==22*len(states)
    subsets={}
    for source in ('all','greedy','learned'):
        ss=[s for s in states if source=='all' or s['identity'][0]==source];blocks={}
        for block in sorted({s['identity'][1] for s in ss}):
            rr=[s for s in ss if s['identity'][1]==block];den=sum(s['generated'] for s in rr);admitted=sum(s['admitted'] for s in rr);methods={}
            for method in METHODS:
                ts=[t for s in rr for t in s['methods'][method]['times']]
                methods[method]=dict(raw_rate=sum(s['methods'][method]['raw'] for s in rr)/den,fee_rate=sum(s['methods'][method]['fee'] for s in rr)/den,energy=sum(s['methods'][method]['energy'] for s in rr),mbs_share=sum(s['methods'][method]['mbs'] for s in rr)/max(admitted,1),fairness=statistics.mean(s['methods'][method]['fairness'] for s in rr),calls=statistics.mean(s['methods'][method]['calls'] for s in rr),mean_seconds=statistics.mean(ts),p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
            blocks[str(block)]=dict(generated=den,admitted=admitted,methods=methods)
        groups={}
        for method in METHODS:
            ts=[t for s in ss for t in s['methods'][method]['times']];groups[method]={key:statistics.mean(b['methods'][method][key] for b in blocks.values()) for key in ('raw_rate','fee_rate','mbs_share','fairness','calls')}
            groups[method].update(energy_ratio_to_G=statistics.mean(b['methods'][method]['energy']/b['methods']['G']['energy'] for b in blocks.values()),pooled_mean_seconds=statistics.mean(ts),pooled_p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
        comparisons={m:{metric:effect([b['methods'][PRIMARY][metric+'_rate']-b['methods'][m][metric+'_rate'] for b in blocks.values()]) for metric in ('raw','fee')} for m in METHODS if m!=PRIMARY}
        subsets[source]=dict(blocks=blocks,groups=groups,comparisons=comparisons);close(subsets[source],summary['subsets'][source])
    main=subsets['all'];best=max(NONLEARNING,key=lambda m:main['groups'][m]['fee_rate']);gain=main['comparisons']['G']['fee'];inc=main['comparisons'][best]['fee'];energy=main['groups'][PRIMARY]['energy_ratio_to_G'];time_ok=all(b['methods'][PRIMARY]['p95_seconds']<1 for b in main['blocks'].values())
    passed=gain['mean']>=.005 and gain['positive']>=4 and inc['mean']>=.001 and energy<=1.05 and time_ok
    close(summary['decision'],dict(gain_vs_G=gain,best_nonlearning=best,learning_increment=inc,energy_ratio_to_G=energy,all_block_p95_below_second=time_ok,native_development_signal=passed,independent_confirmation=False,goal_complete=False))
    report=dict(status='PASS',smoke=p['smoke'],states=len(states),counts=counts,public_edge_weight_checks=weight_checks,full_timing_order_verified=True,within_cpu_intervals_do_not_overlap=True,raw_deadline_thresholds_and_statistics_recomputed=True,models_prices_sources_verified=True,numerical_policy_matches_original_calibration=True,legacy_frozen_physics_equal=True,search_clock_retained_separate_from_decision=True,protocol_sha256=sha(root/'protocol.json'),summary_sha256=sha(root/'analysis/summary.json'),independent_confirmation=False,audit_type='primary agent self-audit; raw outcomes and individual fees, observed order/intervals, legacy branches and separate statistical reconstruction')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Scientific audit must run remotely')
    a=argparse.Namespace(**{k:Path(v) for k,v in read(args.launch)['args'].items()},output=args.output)
    spec=importlib.util.spec_from_file_location('native_audit_resources',a.resource_module);res=importlib.util.module_from_spec(spec);spec.loader.exec_module(res)
    with res.CpuLease(a.resources,task=['cache_native_raw_audit'],kind='environment'):audit(a)
