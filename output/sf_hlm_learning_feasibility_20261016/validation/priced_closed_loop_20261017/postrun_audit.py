"""Raw-record postrun self-audit; run only after all isolated timers finish."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(x,y):
    if isinstance(x,dict):
        assert x.keys()==y.keys(),(x.keys(),y.keys())
        for k in x:close(x[k],y[k])
    elif isinstance(x,(tuple,list)):
        assert len(x)==len(y)
        for a,b in zip(x,y):close(a,b)
    elif isinstance(x,(float,int)) and not isinstance(x,bool):assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12),(x,y)
    else:assert x==y,(x,y)
def effect(values):
    import itertools
    from scipy.stats import t
    n=len(values);mean=statistics.mean(values);radius=float(t.ppf(.975,n-1))*statistics.stdev(values)/math.sqrt(n)
    flip=sum(abs(statistics.mean(v*s for v,s in zip(values,signs)))>=abs(mean)-1e-14 for signs in itertools.product((-1,1),repeat=n))/(2**n)
    return dict(n=n,mean=mean,ci95=[mean-radius,mean+radius],values=values,positive=sum(v>0 for v in values),exact_signflip_p=flip)


def audit(a):
    import numpy as np
    r=a.results_root;p=read(r/'protocol.json');summary=read(r/'analysis/summary.json');done=read(r/'completion.json')
    assert done['status']=='COMPLETED' and done['summary_sha256']==sha(r/'analysis/summary.json') and done['protocol_sha256']==sha(r/'protocol.json')
    assert len(p['tasks'])==240 and len(set(tuple(t) for t in p['tasks']))==240 and p['steps']==1000
    assert not p['smoke'] and p['workloads']==[2026101701,2026101702,2026101703]
    assert p['nonlearning']==['G','R8','R16','binary_price','mixture','mixture_price']
    for name,digest0 in p['files'].items():assert sha(a.release_root/name)==digest0
    for name,digest0 in p['frozen_fit_files'].items():assert sha(a.fit_root/name)==digest0
    fit=read(a.fit_root/'protocol.json')
    for path,digest0 in fit['input_files'].items():assert sha(path)==digest0
    assert sha(a.fit_root/'protocol.json')==p['fit_protocol_sha256']
    launch=read(a.validation_root/'development_launch_v1.json')
    assert launch['preflight_sha256']==sha(a.validation_root/'preflight_audit.json')
    assert read(a.validation_root/'preflight_audit.json')['status']=='PASS'
    first=read(a.validation_root/'runtime_isolation_first_observation.json')
    for path,digest0 in first['files'].items():assert sha(path)==digest0
    profile=first['isolated_profile'];cpus=set(profile['cpus'])
    intervals={cpu:[] for cpu in cpus};episodes=[];counts=dict(steps=0,requests=0,measured_steps=0,zero_steps=0,over_second=0);fallback={}
    for task in p['tasks']:
        td=r/'tasks'/('_'.join(map(str,task)));receipt=read(td/'result.json')
        assert receipt['status']=='PASS' and receipt['task']==task and receipt['protocol_sha256']==sha(r/'protocol.json')
        assert receipt['steps_sha256']==sha(td/'steps.json.gz') and len(receipt['cpu'])==1 and receipt['cpu'][0] in cpus
        cpu=receipt['cpu'][0];totals={k:0. for k in ('generated','admitted','success','potential','fleet_energy','mbs','fairness','raw_success','shadow_fee_success')}
        exo=hashlib.sha256();actions=hashlib.sha256();states=hashlib.sha256();times=[]
        with gzip.open(td/'steps.json.gz','rt') as stream:steps=json.load(stream)
        assert len(steps)==1000
        for index,s in enumerate(steps):
            assert s['step']==index
            m=s['metrics'];requests=s['requests'];time0=s['policy_seconds'];injected=s['injected_seconds']
            assert math.isfinite(time0) and time0>=0 and injected==(time0 if task[3]=='measured' else 0.)
            assert sorted(s['request_ids'])==[x['ue_id'] for x in requests]
            assert len(set(s['request_ids']))==len(requests)==len(s['assignment'])==m['admitted_service_count']
            assert all(math.isfinite(x['latency']) and x['latency']>=0 and math.isfinite(x['deadline']) and x['deadline']>0 for x in requests)
            success=sum(x['latency']+injected<=x['deadline'] for x in requests)
            assert success==m['deadline_satisfied_count']
            assert sum(action==1 for action in s['assignment'])==m['service_offloads_mbs']
            assert 0<=success<=m['admitted_service_count']<=m['generated_service_count']<=s['potential']
            assert 0<=m['fairness']<=1+1e-12
            assert s['observation_ns'][1]>=s['observation_ns'][0] and (s['observation_ns'][1]-s['observation_ns'][0])/1e9>=time0
            intervals[cpu].append((s['observation_ns'][0],s['observation_ns'][1],task,index))
            totals['potential']+=s['potential'];totals['raw_success']+=sum(x['latency']<=x['deadline'] for x in requests)
            totals['shadow_fee_success']+=sum(x['latency']+time0<=x['deadline'] for x in requests)
            for key,metric in (('generated','generated_service_count'),('admitted','admitted_service_count'),('success','deadline_satisfied_count'),('fleet_energy','uav_fleet_total_energy'),('mbs','service_offloads_mbs'),('fairness','fairness')):totals[key]+=m[metric]
            assert math.isclose(m['deadline_satisfaction_rate'],success/max(m['generated_service_count'],1),abs_tol=1e-12)
            close(s['action_hash'],digest([s['motion'],s['assignment']]))
            exo.update(s['exogenous'].encode());actions.update(s['action_hash'].encode());states.update(s['state_hash'].encode());times.append(time0)
            d=s['diagnostic'];assert d['matching_calls']<=d['matching_budget']
            if task[0] in ('R8','R16','L16'):assert d['feature_budget']==16
            for key,value in d.get('mixture_fallback_counts',{}).items():fallback[key]=fallback.get(key,0)+value
            counts['steps']+=1;counts['requests']+=len(requests);counts['measured_steps' if task[3]=='measured' else 'zero_steps']+=1;counts['over_second']+=int(time0>1)
        close(totals,receipt['summary'])
        for key,value in (('exogenous_trace_hash',exo.hexdigest()),('action_hash',actions.hexdigest()),('state_hash',states.hexdigest())):assert receipt[key]==value
        episodes.append(dict(task=task,totals=totals,times=times,exo=exo.hexdigest(),basis=receipt['basis'],fingerprint=receipt['model_fingerprint'],config=receipt['config_sha256']))
    for cpu,rows in intervals.items():
        rows.sort()
        for left,right in zip(rows,rows[1:]):assert left[1]<=right[0],(cpu,left[2:],right[2:])
    for block in p['blocks']:
        group=[e for e in episodes if e['task'][1]==block]
        assert len(set(e['config'] for e in group))==1 and len(set(e['fingerprint'] for e in group))==1 and len(set(digest(e['basis']) for e in group))==1
        for workload in p['workloads']:
            es=[e for e in group if e['task'][2]==workload]
            assert len(set(e['exo'] for e in es))==1 and len(set(e['totals']['potential'] for e in es))==1
    modes={}
    for mode in p['modes']:
        blocks={}
        for block in p['blocks']:
            methods={}
            for method in p['methods']:
                es=[e for e in episodes if e['task'][0]==method and e['task'][1]==block and e['task'][3]==mode];assert len(es)==3
                values={k:sum(e['totals'][k] for e in es) for k in es[0]['totals']};ts=[t for e in es for t in e['times']]
                methods[method]=dict(**values,dsr=values['success']/values['generated'],potential_dsr=values['success']/values['potential'],admission_rate=values['admitted']/values['generated'],accepted_quality=values['success']/values['admitted'],raw_dsr=values['raw_success']/values['generated'],shadow_fee_dsr=values['shadow_fee_success']/values['generated'],mbs_share=values['mbs']/values['admitted'],mean_fairness=values['fairness']/3000,mean_seconds=statistics.mean(ts),p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
            blocks[str(block)]=methods
        groups={method:{key:statistics.mean(b[method][key] for b in blocks.values()) for key in next(iter(blocks.values()))[method]} for method in p['methods']}
        best=max(p['nonlearning'],key=lambda method:groups[method]['dsr'])
        diffs={m:effect([b['probability_price']['dsr']-b[m]['dsr'] for b in blocks.values()]) for m in p['methods'] if m!='probability_price'}
        er=statistics.mean(b['probability_price']['fleet_energy']/b['G']['fleet_energy'] for b in blocks.values())
        g=p['gates'];go=diffs['G']['mean']>=g['gain_vs_G'] and diffs['G']['positive']>=g['positive_blocks'] and diffs[best]['mean']>=g['learning_increment'] and er<=g['energy_ratio'] and all(b['probability_price']['p95_seconds']<g['per_block_p95_seconds'] for b in blocks.values())
        modes[mode]=dict(blocks=blocks,groups=groups,best_nonlearning=best,learning_differences=diffs,energy_ratio_vs_G=er,development_gate=go)
        close(modes[mode],summary['modes'][mode])
    pooled={}
    for mode in p['modes']:
        pooled[mode]={}
        for method in p['methods']:
            ts=[t for e in episodes if e['task'][0]==method and e['task'][3]==mode for t in e['times']]
            pooled[mode][method]=dict(samples=len(ts),mean_seconds=statistics.mean(ts),pooled_p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts))
    assert summary['decision']['closed_loop_development_signal']==modes['measured']['development_gate']
    result=dict(status='PASS',protocol_sha256=sha(r/'protocol.json'),summary_sha256=sha(r/'analysis/summary.json'),episodes=240,counts=counts,request_delay_thresholds_recomputed=True,block_statistics_recomputed=True,exogenous_streams_paired=True,fixed_potential_denominator_paired=True,same_cpu_observed_intervals_do_not_overlap=True,source_model_price_hashes_verified=True,mixture_fallback_counts=fallback,pooled_timing=pooled,independent_confirmation=False,shared_lower_calibrator=True,historical_upper_blocks=5,audit_type='primary agent self-audit; raw per-step outcomes, counters, paired streams, time intervals and separate statistical formulas')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='pooled_timing'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Numerical audit must run remotely')
    a=argparse.Namespace(**{k:Path(v) for k,v in read(args.launch)['args'].items()},output=args.output)
    spec=importlib.util.spec_from_file_location('postrun_resources',a.resource_module);res=importlib.util.module_from_spec(spec);spec.loader.exec_module(res)
    with res.CpuLease(a.resources,task=['priced_closed_loop_raw_audit'],kind='environment'):audit(a)
