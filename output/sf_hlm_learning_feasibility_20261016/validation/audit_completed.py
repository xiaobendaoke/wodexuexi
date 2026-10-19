"""Task-only independent record audit; run on server after the population closes."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for q in iter(lambda:f.read(1024*1024),b''):h.update(q)
    return h.hexdigest()


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11):raise AssertionError((a,b))


def quantile(v,p):
    v=sorted(v);x=(len(v)-1)*p;i=math.floor(x);j=math.ceil(x)
    return v[i]*(1-(x-i))+v[j]*(x-i)


def main():
    args=argparse.ArgumentParser();args.add_argument('--results-root',type=Path,required=True)
    args.add_argument('--validation-root',type=Path,required=True);args.add_argument('--allow-smoke',action='store_true')
    a=args.parse_args();os.sched_setaffinity(0,{4});r=a.results_root
    p=read(r/'protocol.json');summary=read(r/'analysis/summary.json')
    assert not p['smoke'] or a.allow_smoke
    if not p['smoke']:assert len(p['identities'])==400 and len(p['blocks'])==5
    from scipy.stats import t
    methods=['G','L16','L8','R16','R8','H16','H8'];nonlearning=['G','R16','R8','H16','H8']
    expected={tuple(v) for v in p['identities']};seen=set();task_receipts=0;instances=times=prefixes=0
    aggregate={};fingerprints=defaultdict(set);source_files={};bench_hashes={}
    for task in p['tasks']:
        folder=r/'tasks'/'_'.join(map(str,task));receipt=read(folder/'result.json')
        assert receipt['status']=='PASS' and receipt['task']==task and receipt['protocol_sha256']==sha(r/'protocol.json')
        for name,h in receipt['files'].items():assert sha(folder/name)==h
        source_files.update(receipt.get('state_files',{}));task_receipts+=1
        if task[0]!='benchmark':continue
        d=read(folder/'benchmark.json');identity=tuple(d['identity']);assert identity in expected and identity not in seen;seen.add(identity)
        meta_path=r/'states'/identity[0]/f'block_{identity[1]}_workload_{identity[2]}'/f'episode_{identity[3]:05d}'/f'step_{identity[4]:05d}'/'state.json'
        assert sha(meta_path)==d['state_sha256'];meta=read(meta_path);source_files.update(meta['files'])
        public=read(meta['input']);ids=public['request_ids'];assert len(ids)==d['admitted']
        assert d['original_reference_exact'] and len(d['prefix_checks'])==3
        assert len(d['cpu'])==1 and d['cpu'][0] in [0,64,1,65,2,66,3,67]
        fingerprints[identity[1]].add(d['model_fingerprint']);prefixes+=3
        by_method=defaultdict(list);trial_ids=set()
        for trial in d['results']:
            m,rep=trial['method'],trial['repeat'];assert (m,rep) not in trial_ids;trial_ids.add((m,rep))
            assert m in methods and trial['cap']==(0 if m=='G' else int(m[1:]))
            assert len(trial['software_seconds'])==3
            assert all(math.isfinite(v) and v>=0 for v in trial['software_seconds'])
            assert [q['ue_id'] for q in trial['requests']]==ids
            assert trial['generated']==d['generated']==int(trial['metrics']['generated_service_count'])
            assert trial['admitted']==d['admitted']==int(trial['metrics']['admitted_service_count'])
            assert trial['metrics']['forced_service_admissions']==0
            assert trial['diagnostic']['matching_calls']<=trial['cap']
            if m!='G':assert trial['diagnostic']['feature_budget']==16
            assert trial['diagnostic']['chosen_predicted']['energy']<=trial['diagnostic']['common_energy_cap']+1e-7
            raw=sum(q['deadline']>=q['latency'] for q in trial['requests'])
            assert raw==trial['raw_success']==int(trial['metrics']['deadline_satisfied_count'])
            fee=sum(sum(q['deadline']>=q['latency']+cost for q in trial['requests']) for cost in trial['software_seconds'])/3
            close(fee,trial['fee_success']);by_method[m].append((trial,fee));instances+=1;times+=3
        assert trial_ids=={(m,k) for m in methods for k in range(5 if m.startswith('R') else 1)}
        for rep in range(5):
            pair=[q for q in d['results'] if q['method'] in ['R8','R16'] and q['repeat']==rep]
            assert pair[0]['seed']==pair[1]['seed']
        avg={m:sum(x[1] for x in v)/len(v) for m,v in by_method.items()}
        avg['oracle_all']=max(avg[m] for m in methods)
        avg['oracle_nonlearning']=max(avg[m] for m in nonlearning)
        avg['oracle_L8_G']=max(avg['L8'],avg['G']);avg['oracle_L16_G']=max(avg['L16'],avg['G'])
        avg['rule_L8_G']=avg['L8' if d['admitted']>=16 else 'G']
        for ms in p['software_overhead_ms']:
            extras={m:sum(sum(sum(q['deadline']>=q['latency']+(cost+ms/1000) for q in tr['requests'])
                for cost in tr['software_seconds'])/3 for tr,_ in v)/len(v) for m,v in by_method.items()}
            for m in ['L8','L16']:avg[f'oracle_{m}_G_overhead_{ms}ms']=max(extras[m],extras['G'])
        for subset in ['all',identity[0]]:
            block=aggregate.setdefault((subset,str(identity[1])),{'generated':0,'states':0,'fee':defaultdict(float),'raw':defaultdict(float),'times':defaultdict(list),'energy':defaultdict(float)})
            block['generated']+=d['generated'];block['states']+=1
            for m,value in avg.items():block['fee'][m]+=value
            for m,v in by_method.items():
                block['raw'][m]+=sum(tr['raw_success'] for tr,_ in v)/len(v)
                block['energy'][m]+=sum(tr['metrics']['uav_fleet_total_energy'] for tr,_ in v)/len(v)
                block['times'][m].extend(cost for tr,_ in v for cost in tr['software_seconds'])
        bench_hashes[str(folder/'benchmark.json')]=sha(folder/'benchmark.json')
    assert seen==expected
    for name,h in source_files.items():assert sha(name)==h
    assert all(len(v)==1 for v in fingerprints.values())
    for subset,out in summary['subsets'].items():
        for b,reported in out['blocks'].items():
            block=aggregate[subset,b];assert reported['generated']==block['generated'] and reported['states']==block['states']
            for m,v in reported['methods'].items():
                close(v['fee_rate'],block['fee'][m]/block['generated'])
                if m in methods:
                    close(v['raw_rate'],block['raw'][m]/block['generated']);close(v['energy'],block['energy'][m])
                    close(v['mean_seconds'],statistics.mean(block['times'][m]));close(v['p95_seconds'],quantile(block['times'][m],.95))
        for m,v in out['groups'].items():
            close(v['mean'],statistics.mean(aggregate[subset,str(b)]['fee'][m]/aggregate[subset,str(b)]['generated'] for b in p['blocks']))
        for name,v in out['comparisons'].items():
            raw=name.startswith('raw_');first,second=(name[4:] if raw else name).split('-');values=[]
            for b in p['blocks']:
                block=aggregate[subset,str(b)];counts=block['raw'] if raw else block['fee']
                values.append((counts[first]-counts[second])/block['generated'])
            mean=statistics.mean(values);close(mean,v['mean']);assert v['n']==len(p['blocks'])
            radius=float(t.ppf(.975,len(values)-1))*statistics.stdev(values)/math.sqrt(len(values))
            close(v['ci95'][0],mean-radius);close(v['ci95'][1],mean+radius)
            exact=sum(abs(sum(s*x for s,x in zip(signs,values))/len(values))>=abs(mean)-1e-14
                for signs in itertools.product([-1,1],repeat=len(values)))/(2**len(values))
            close(v['exact_signflip_p'],exact)
    assert instances==summary['branch_instances'] and times==summary['timing_samples']
    if not p['smoke']:assert instances==6000 and times==18000 and prefixes==1200 and task_receipts==500
    out={'status':'PASS','states':len(seen),'task_receipts':task_receipts,'branch_instances':instances,'timing_samples':times,
         'prefix_cases':prefixes,'frozen_model_fingerprints':{str(b):list(v) for b,v in fingerprints.items()},
         'request_scores':'recomputed each actual fee before averaging','block_rates_ci_and_signflip':'reproduced independently',
         'source_hashes_verified':len(source_files),'summary_sha256':sha(r/'analysis/summary.json'),
         'audit_source_sha256':sha(__file__),'raw_benchmark_sha256':bench_hashes,'completed_unix':time.time(),
         'scope':'independent recomputation of records by primary agent; not independent agent review'}
    a.validation_root.mkdir(parents=True,exist_ok=True);(a.validation_root/'raw_record_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['raw_benchmark_sha256','frozen_model_fingerprints']},ensure_ascii=False))


if __name__=='__main__':main()
