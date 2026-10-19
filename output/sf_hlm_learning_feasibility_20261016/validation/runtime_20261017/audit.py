"""Second implementation of raw-score, pairing and gate audit; remote only."""
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
def close(x,y):assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-11),(x,y)


def audit(a):
    import numpy as np
    from scipy.stats import t
    root=a.results_root;p=read(root/'protocol.json');out=read(root/'analysis/summary.json')
    assert not p['smoke'] and len(p['identities'])==400
    assert out['protocol_sha256']==sha(root/'protocol.json')
    launch=read(a.output.parent/'development_launch.json')['command'];release=Path(launch[launch.index('--release-root')+1])
    for rel,digest in p['files'].items():assert sha(release/rel)==digest,rel
    for path,digest in p['input_files'].items():assert sha(path)==digest,path
    methods=p['methods'];data=[];fingerprints=defaultdict(set);branch_count=time_count=graph_checks=0;max_error=0.;state_files=set()
    for registered in p['states']:
        identity=registered['identity'];folder=root/'tasks'/('benchmark_'+'_'.join(map(str,identity)))
        rec=read(folder/'result.json');b=read(folder/'benchmark.json')
        assert rec['status']=='PASS' and rec['identity']==identity and b['identity']==identity
        assert rec['protocol_sha256']==sha(root/'protocol.json') and rec['benchmark_sha256']==sha(folder/'benchmark.json')
        assert sha(registered['state'])==registered['state_sha256']==b['state_sha256']
        assert sha(registered['benchmark'])==registered['benchmark_sha256']==b['reference_benchmark_sha256']
        meta=read(registered['state']);old=read(registered['benchmark'])
        for path,digest in meta['files'].items():assert sha(path)==digest,path;state_files.add(path)
        refs={(q['method'],q['repeat']):q for q in old['results']}
        assert b['model_fingerprint']==old['model_fingerprint'];fingerprints[identity[1]].add(b['model_fingerprint'])
        assert len(b['cpu'])==1 and not b['state_delay_applied'] and b['physical_outcomes_reused_after_exact_assignment']
        assert b['generated']==old['generated']==meta['generated'] and b['admitted']==old['admitted']==meta['admitted']
        expected={(m,i) for m in methods for i in range(5 if m.endswith(('R8','R16')) else 1)}
        observed=[(q['method'],q['repeat']) for q in b['results']]
        assert len(observed)==28 and set(observed)==expected
        assert len(b['compatibility'])==14
        for c in b['compatibility']:
            graph_checks+=c['graph_checks'];max_error=max(max_error,c['max_logit_error']);assert c['max_logit_error']<=1e-5
        order={i:[] for i in range(3)};means={}
        for q in b['results']:
            ref=refs[q['base_method'],q['repeat']]
            assert q['assignment']==ref['assignment'] and q['seed']==ref['seed'] and q['cap']==ref['cap']
            assert len(q['software_seconds'])==len(q['runtime_stats'])==3 and q['timing_iteration']==[0,1,2]
            assert all(math.isfinite(c) and c>=0. for c in q['software_seconds'])
            for i,x in enumerate(q['selected_order']):order[i].append(x)
            for item in q['runtime_stats']:
                assert 0<=item['dense_fallbacks']<=item['sparse_forwards']
                if q['implementation']=='sparse':
                    expected_forwards=max(0,ref['diagnostic']['matching_calls']-1)+int(ref['diagnostic'].get('termination')=='policy_stop')
                    assert item['sparse_forwards']==expected_forwards
                else:assert item['sparse_forwards']==item['dense_fallbacks']==0
            branch_count+=1;time_count+=3
            raw=sum(z['latency']<=z['deadline'] for z in ref['requests'])
            fee=sum(sum(z['latency']+c<=z['deadline'] for z in ref['requests']) for c in q['software_seconds'])/3
            close(raw,q['raw_success']);close(fee,q['fee_success'])
        assert all(sorted(values)==list(range(28)) for values in order.values())
        for method in methods:
            trials=[q for q in b['results'] if q['method']==method];n=len(trials);rr=[refs[q['base_method'],q['repeat']] for q in trials]
            means[method]=dict(raw=sum(z['raw_success'] for z in rr)/n,
                fee=sum(sum(sum(z['latency']+c<=z['deadline'] for z in ref['requests']) for c in q['software_seconds'])/3 for q,ref in zip(trials,rr))/n,
                energy=sum(z['metrics']['uav_fleet_total_energy'] for z in rr)/n,mbs=sum(z['metrics']['service_offloads_mbs'] for z in rr)/n,
                fairness=sum(z['metrics']['fairness'] for z in rr)/n,calls=sum(z['diagnostic']['matching_calls'] for z in rr)/n,
                times=[c for q in trials for c in q['software_seconds']],
                sparse_forwards=sum(x['sparse_forwards'] for q in trials for x in q['runtime_stats'])/(3*n),
                dense_fallbacks=sum(x['dense_fallbacks'] for q in trials for x in q['runtime_stats'])/(3*n))
        data.append(dict(identity=identity,generated=b['generated'],admitted=b['admitted'],means=means))
    assert len(fingerprints)==5 and all(len(x)==1 for x in fingerprints.values())
    assert len({next(iter(x)) for x in fingerprints.values()})==5
    close(max_error,out['max_sparse_logit_error']);assert graph_checks==out['graph_checks']
    checked=stats_checked=0
    for subset,reported in out['subsets'].items():
        sr=[s for s in data if subset=='all' or s['identity'][0]==subset];blocks={}
        for block,rec in reported['blocks'].items():
            br=[s for s in sr if s['identity'][1]==int(block)];den=sum(s['generated'] for s in br);ad=sum(s['admitted'] for s in br)
            assert den==rec['generated'] and ad==rec['admitted'];blocks[block]={}
            for m in methods:
                ms=[s['means'][m] for s in br];times=[t for v in ms for t in v['times']]
                values=dict(raw_rate=sum(v['raw'] for v in ms)/den,fee_rate=sum(v['fee'] for v in ms)/den,
                    energy=sum(v['energy'] for v in ms),mbs_share=sum(v['mbs'] for v in ms)/ad if ad else 0.,
                    fairness=sum(v['fairness'] for v in ms)/len(ms),calls=sum(v['calls'] for v in ms)/len(ms),
                    mean_seconds=sum(times)/len(times),p95_seconds=float(np.quantile(times,.95)),max_seconds=max(times),
                    sparse_forwards=sum(v['sparse_forwards'] for v in ms)/len(ms),dense_fallbacks=sum(v['dense_fallbacks'] for v in ms)/len(ms))
                for key,value in values.items():close(value,rec['methods'][m][key])
                blocks[block][m]=values;checked+=1
        for pair,record in reported['comparisons'].items():
            first,second=pair.split('-')
            for metric,stat in record.items():
                values=[b[first][metric+'_rate']-b[second][metric+'_rate'] for b in blocks.values()]
                mean=sum(values)/5;radius=float(t.ppf(.975,4))*statistics.stdev(values)/math.sqrt(5)
                for x,y in zip(values,stat['values']):close(x,y)
                close(mean,stat['mean']);close(mean-radius,stat['ci95'][0]);close(mean+radius,stat['ci95'][1])
                exact=sum(abs(sum(s*x for s,x in zip(signs,values))/5)>=abs(mean)-1e-14 for signs in itertools.product([-1,1],repeat=5))/32
                close(exact,stat['exact_signflip_p']);assert stat['positive']==sum(v>0 for v in values);stats_checked+=1
        for m,record in reported['groups'].items():
            for key,value in record.items():
                expected=sum(b[m]['energy']/b['fast_G']['energy'] for b in blocks.values())/5 if key=='energy_ratio_fast_G' else sum(b[m][key] for b in blocks.values())/5
                close(expected,value)
    s=out['subsets']['all'];best=max(('fast_G','fast_R8','fast_R16'),key=lambda m:s['groups'][m]['fee_rate'])
    assert best==out['decision']['best_fast_nonlearning']
    for m,d in out['decision']['candidates'].items():
        delta=s['comparisons'][m+'-fast_G']['fee'];g=s['groups'][m];within=all(b['methods'][m]['p95_seconds']<1. for b in s['blocks'].values())
        expected=delta['mean']>=.005 and delta['positive']>=4 and g['energy_ratio_fast_G']<=1.05 and within
        assert d['practical']==expected
        close(d['learning_increment'],g['fee_rate']-s['groups'][best]['fee_rate'])
        assert d['learning_increment_target_met']==(d['learning_increment']>=.001)
    verdict=dict(status='PASS',states=400,branch_instances=branch_count,timing_samples=time_count,state_files_verified=len(state_files),
        graph_checks=graph_checks,max_sparse_logit_error=max_error,block_methods_checked=checked,statistical_comparisons_checked=stats_checked,
        exact_physical_reference_pairing=True,independent_agent_review=False,independent_confirmation=False,
        summary_sha256=sha(root/'analysis/summary.json'))
    a.output.write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ('results_root','output','resource_module','resources'):ap.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=ap.parse_args()
    if sys.platform!='linux':raise RuntimeError('Remote scientific audit only')
    spec=importlib.util.spec_from_file_location('resources',a.resource_module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with m.CpuLease(a.resources,task='runtime_raw_audit'):audit(a)
