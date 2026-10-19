"""Remote second-implementation audit of the teacher screen (main-agent review)."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11),(a,b)


def audit(a):
    import numpy as np
    from scipy.stats import t
    p=read(a.results_root/'protocol.json');out=read(a.results_root/'analysis/summary.json')
    assert not p['smoke'] and len(p['identities'])==400
    assert out['protocol_sha256']==sha(a.results_root/'protocol.json')
    launch=read(a.output.parent/'formal_launch.json')['command']
    release=Path(launch[launch.index('--release-root')+1])
    for rel,digest in p['files'].items():assert sha(release/rel)==digest,rel
    methods=p['methods'];data=[];fingerprints=defaultdict(set);physical_files=set()
    branch_count=time_count=0
    for path,digest in p['input_files'].items():assert sha(path)==digest,path
    for registered in p['states']:
        identity=registered['identity'];folder=a.results_root/'tasks'/('benchmark_'+'_'.join(map(str,identity)))
        rec=read(folder/'result.json');b=read(folder/'benchmark.json')
        assert rec['status']=='PASS' and rec['identity']==identity and b['identity']==identity
        assert rec['protocol_sha256']==sha(a.results_root/'protocol.json')
        assert rec['benchmark_sha256']==sha(folder/'benchmark.json')
        assert sha(registered['state'])==registered['state_sha256']==b['state_sha256']
        assert sha(registered['benchmark'])==registered['benchmark_sha256']
        meta=read(registered['state'])
        for path,digest in meta['files'].items():
            assert sha(path)==digest,path;physical_files.add(path)
        obs=read(meta['input']);ids=obs['request_ids']
        old={(q['method'],q['repeat']):q for q in read(registered['benchmark'])['results']}
        assert len(b['candidate_equivalence'])==4 and b['control_crosschecks']==7
        assert {q['method'] for q in b['candidate_equivalence']}==set(p['candidates'])
        assert b['generated']==meta['generated'] and b['admitted']==meta['admitted'] and not b['state_delay_applied']
        fingerprints[identity[1]].add(b['model_fingerprint'])
        assert len(b['cpu'])==1
        observed_keys=[(q['method'],q['repeat']) for q in b['results']]
        expected={(m,i) for m in methods for i in range(5 if m=='R16' else 1)}
        assert len(observed_keys)==11 and set(observed_keys)==expected
        order={i:[] for i in range(3)};state=dict(identity=identity,generated=b['generated'],admitted=b['admitted'],means={})
        for q in b['results']:
            assert q['generated']==b['generated'] and q['admitted']==len(q['requests'])==len(ids)
            assert [z['ue_id'] for z in q['requests']]==ids
            assert len(q['software_seconds'])==3 and q['timing_iteration']==[0,1,2]
            assert all(math.isfinite(v) and v>=0. for v in q['software_seconds'])
            for i,selected in enumerate(q['selected_order']):order[i].append(selected)
            branch_count+=1;time_count+=3
            raw=sum(z['latency']<=z['deadline'] for z in q['requests'])
            fee=sum(sum(z['latency']+cost<=z['deadline'] for z in q['requests']) for cost in q['software_seconds'])/3
            close(raw,q['raw_success']);close(raw,q['metrics']['deadline_satisfied_count']);close(fee,q['fee_success'])
            if q['method'] in ('G','old_L16','R16'):
                reference=old['L16' if q['method']=='old_L16' else q['method'],q['repeat']]
                assert reference['assignment']==q['assignment'] and reference['requests']==q['requests']
                assert reference['seed']==q['seed']
                for key,value in reference['metrics'].items():close(value,q['metrics'][key])
            cap=0 if q['method']=='G' else 16
            assert q['cap']==cap and 0<=q['diagnostic']['matching_calls']<=cap
            assert q['diagnostic']['chosen_predicted']['energy']<=q['diagnostic']['common_energy_cap']+1e-7
        assert all(sorted(x)==list(range(11)) for x in order.values())
        for method in methods:
            trials=[q for q in b['results'] if q['method']==method];n=len(trials)
            state['means'][method]=dict(raw=sum(q['raw_success'] for q in trials)/n,fee=sum(
                sum(sum(z['latency']+cost<=z['deadline'] for z in q['requests']) for cost in q['software_seconds'])/3 for q in trials)/n,
                energy=sum(q['metrics']['uav_fleet_total_energy'] for q in trials)/n,
                mbs=sum(q['metrics']['service_offloads_mbs'] for q in trials)/n,
                fairness=sum(q['metrics']['fairness'] for q in trials)/n,
                calls=sum(q['diagnostic']['matching_calls'] for q in trials)/n,
                times=[cost for q in trials for cost in q['software_seconds']])
        data.append(state)
    assert len(fingerprints)==5 and all(len(v)==1 for v in fingerprints.values())
    assert len(set(next(iter(v)) for v in fingerprints.values()))==5
    comparisons=blocks_checked=0

    def verify_stats(values,stat):
        nonlocal comparisons
        mean=sum(values)/len(values);sd=statistics.stdev(values);radius=float(t.ppf(.975,4))*sd/math.sqrt(5)
        for x,y in zip(values,stat['values']):close(x,y)
        close(mean,stat['mean']);close(mean-radius,stat['ci95'][0]);close(mean+radius,stat['ci95'][1])
        exact=sum(abs(sum(s*v for s,v in zip(signs,values))/5)>=abs(mean)-1e-14 for signs in itertools.product([-1,1],repeat=5))/32
        close(exact,stat['exact_signflip_p']);assert stat['positive']==sum(v>0 for v in values) and stat['n']==5
        pv=float(2*t.sf(abs(mean)/(sd/math.sqrt(5)),4)) if sd else (0. if mean else 1.)
        close(pv,stat['p']);comparisons+=1

    for subset,record in out['subsets'].items():
        sr=[s for s in data if subset=='all' or s['identity'][0]==subset];rates={}
        for block,reported in record['blocks'].items():
            br=[s for s in sr if s['identity'][1]==int(block)];den=sum(s['generated'] for s in br);ad=sum(s['admitted'] for s in br)
            assert den==reported['generated'] and ad==reported['admitted'];rates[block]={}
            for method in methods:
                ms=[s['means'][method] for s in br];times=[x for m in ms for x in m['times']]
                values=dict(raw_rate=sum(m['raw'] for m in ms)/den,fee_rate=sum(m['fee'] for m in ms)/den,
                    energy=sum(m['energy'] for m in ms),mbs_share=sum(m['mbs'] for m in ms)/ad if ad else 0.,
                    fairness=sum(m['fairness'] for m in ms)/len(ms),calls=sum(m['calls'] for m in ms)/len(ms),
                    mean_seconds=sum(times)/len(times),p95_seconds=float(np.quantile(times,.95)),max_seconds=max(times))
                for k,v in values.items():close(v,reported['methods'][method][k])
                rates[block][method]=values;blocks_checked+=1
        for pair,reported in record['comparisons'].items():
            first,second=pair.split('-')
            for metric,stat in reported.items():
                values=[b[first][metric+'_rate']-b[second][metric+'_rate'] for b in rates.values()]
                verify_stats(values,stat)
        for metric,stat in record['interaction'].items():
            values=[(b['multistep_ppo'][metric+'_rate']-b['baseline_ppo'][metric+'_rate'])-
                    (b['multistep_bc'][metric+'_rate']-b['baseline_bc'][metric+'_rate']) for b in rates.values()]
            verify_stats(values,stat)
        for metric in ('raw','fee'):
            names=[x+'-'+y for x,y in p['primary_attribution_pairs']];pvs=[record['comparisons'][n][metric]['p'] for n in names]
            running=0.
            for rank,i in enumerate(sorted(range(4),key=lambda i:pvs[i])):
                running=max(running,min(1.,pvs[i]*(4-rank)))
                close(running,record['comparisons'][names[i]][metric]['holm_p_four_attribution_pairs'])
        for method,reported in record['groups'].items():
            for key,value in reported.items():
                expected=sum(b[method]['energy']/b['G']['energy'] for b in rates.values())/5 if key=='energy_ratio_G' else sum(b[method][key] for b in rates.values())/5
                close(expected,value)
    s=out['subsets']['all'];d=out['decision'];q=s['comparisons']['multistep_ppo-R16']['raw'];c=s['comparisons']['multistep_ppo-G']['fee']
    er=sum(b['methods']['multistep_ppo']['energy']/b['methods']['R16']['energy'] for b in s['blocks'].values())/5
    eg=s['groups']['multistep_ppo']['energy_ratio_G'];within=all(b['methods']['multistep_ppo']['p95_seconds']<1. for b in s['blocks'].values())
    assert d['quality_candidate']==(q['mean']>=.003 and q['positive']>=4 and er<=1.05 and within)
    assert d['cost_candidate']==(c['mean']>=.005 and c['positive']>=4 and eg<=1.05 and within)
    for m,flag in d['all_four_cost_candidates'].items():
        c=s['comparisons'][m+'-G']['fee'];g=s['groups'][m]
        assert flag==(c['mean']>=.005 and c['positive']>=4 and g['energy_ratio_G']<=1.05 and all(b['methods'][m]['p95_seconds']<1. for b in s['blocks'].values()))
    verdict=dict(status='PASS',states=400,branch_instances=branch_count,timing_samples=time_count,
        checkpoints=len(p['models']),equivalence_checks=1600,control_checks=2800,state_files_verified=len(physical_files),
        block_methods_verified=blocks_checked,statistical_comparisons_verified=comparisons,
        model_fingerprints={str(k):list(v) for k,v in fingerprints.items()},
        independent_agent_review=False,independent_confirmation=False,summary_sha256=sha(a.results_root/'analysis/summary.json'))
    a.output.write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resource-module',type=Path,required=True);ap.add_argument('--resources',type=Path,required=True);a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Remote audit only')
    spec=importlib.util.spec_from_file_location('resources',a.resource_module);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    with r.CpuLease(a.resources,task='teacher_raw_record_audit'):audit(a)
