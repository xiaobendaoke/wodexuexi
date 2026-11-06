"""Metadata/statistical self-review of transported candidate reliability results.

No local Torch inference, physical rollout, policy fitting, or original-analysis
rewrite. Supplementary signal moments are descriptive, not a new success gate.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_candidate_reliability as a


def signal_moments(y):
    y=np.asarray(y,float)
    if y.ndim!=2 or y.shape[1]<2 or not np.isfinite(y).all():raise ValueError('finite paired samples required')
    c=y-y.mean(axis=0,keepdims=True)
    observed=float(np.mean(c.mean(1)**2));noise=float(np.mean(c.var(1,ddof=1)/y.shape[1]))
    return dict(observed_candidate_mean_second_moment=observed,estimated_mean_noise=noise,
                unbiased_signal_second_moment=observed-noise,
                note='Signed finite-sample moment estimate; a negative value is retained, not clipped to zero.')


def normalize(d):
    d=json.loads(json.dumps(d))
    if 'continuation_difference' in d:
        d['continuation_difference']['rows']=sorted(d['continuation_difference']['rows'],key=lambda x:json.dumps(x,sort_keys=True))
    return d



def assert_nested_close(actual,expected,path='analysis'):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and set(actual)==set(expected),path
        for key in expected:assert_nested_close(actual[key],expected[key],path+'/'+str(key))
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected),path
        for i,(x,y) in enumerate(zip(actual,expected)):assert_nested_close(x,y,path+'/'+str(i))
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert isinstance(actual,(int,float)) and np.isclose(actual,expected,atol=1e-9,rtol=1e-10),path
    else:assert actual==expected,path


def review(root):
    root=Path(root).resolve();r=a.result_dir(root);v=a.validation_dir(root)
    manifest=a.load_json(v/'transport_manifest.json')
    for name,sha in manifest.items():assert a.digest(root/name)==sha,name
    protocol=a.load_json(r/'protocol.json');assert protocol['replicas']==list(a.REPLICAS) and protocol['branches']==6144
    for name,sha in protocol['source'].items():assert a.digest(root/name)==sha,name
    records=[a.load_json(p) for p in (r/'labels').glob('*.json')]
    assert len(records)==6144
    seen=set();counts=defaultdict(set);grouped=defaultdict(list)
    for x in records:
        key=tuple(x[k] for k in ('b','w','time','i','j','replica','mode'))
        assert key not in seen;seen.add(key)
        assert x['mode'] in a.MODES and x['replica'] in a.REPLICAS
        assert x['generated']==x['success']+x['expired'] and x['success']==x['actual_prefix']+x['actual_tail']
        assert x['root_success']+x['future_success']==x['success'] and x['audit']['status']=='PASS'
        counts[x['b'],x['w'],x['time'],x['replica']].add(x['generated'])
        grouped[x['mode'],x['b'],x['w'],x['time']].append(x)
    assert len(counts)==24*32 and all(len(x)==1 for x in counts.values())
    assert len(grouped)==48
    server=a.load_json(v/'analysis_server_saved.json') if (v/'analysis_server_saved.json').exists() else a.load_json(r/'analysis.json')
    captured=[];old_write=a.write
    a.write=lambda path,d:captured.append((str(path),d))
    try:computed=a.analyze(root)
    finally:a.write=old_write
    assert_nested_close(normalize(computed),normalize(server))
    assert a.load_json(v/'protection_before.json')==a.load_json(v/'protection_after.json')
    resource=a.load_json(v/'resources.json')
    entries=resource['records'];assert len(entries)==resource['tasks']==6168
    assert all(x['ok'] and x['threads']==1 and len(x['affinity'])==1 for x in entries)
    assert len({x['file'] for x in entries})==6168
    assert abs(sum(x['cpu_seconds'] for x in entries)-resource['worker_cpu_seconds'])<1e-6
    cpu=resource['worker_cpu_seconds']+resource['coordinator_cpu_seconds']
    assert abs(resource['equivalent_cpus']-cpu/resource['wall_seconds'])<1e-8
    assert abs(resource['utilization']-resource['equivalent_cpus']/len(resource['allowed_cpus']))<1e-10
    events=sorted([(x['start'],1) for x in entries]+[(x['end'],-1) for x in entries]);active=peak=0
    for _,change in events:active+=change;peak=max(peak,active)
    assert peak==resource['peak_active'] and active==0
    extra=[]
    for (mode,b,w,now),rs in sorted(grouped.items()):
        y,choices,_=a.matrix(rs,'success');moments=signal_moments(y)
        p=np.array([next(x for x in rs if (x['i'],x['j'])==ij)['predicted_prefix']+next(x for x in rs if (x['i'],x['j'])==ij)['predicted_tail'] for ij in choices])
        stat=a.paired_statistics(y,p)
        extra.append(dict(mode=mode,b=b,workload=w,time=now,prediction_gain=stat['prediction_gain'],
            mean_pair_se=float(np.mean([p['se'] for p in stat['pairs']])),**{k:z for k,z in moments.items() if k!='note'}))
    summary={mode:{key:float(np.mean([x[key] for x in extra if x['mode']==mode])) for key in
        ('prediction_gain','mean_pair_se','observed_candidate_mean_second_moment','estimated_mean_noise','unbiased_signal_second_moment')} for mode in a.MODES}
    shifts=[];prefix_details=[]
    for b,w,now in sorted({key[1:] for key in grouped}):
        ys={m:a.matrix(grouped[m,b,w,now],'success')[0] for m in a.MODES}
        d=ys['a2']-ys['matched'];moments=signal_moments(d)
        shifts.append(dict(b=b,w=w,time=now,mean_success_change=float(d.mean()),**{k:z for k,z in moments.items() if k!='note'}))
        rs=grouped['matched',b,w,now]
        arr={f:a.matrix(rs,f)[0] for f in ('success','actual_prefix','oracle_public_prefix','predicted_prefix','oracle_public_endpoint_value')}
        center=lambda z:z-z.mean(axis=0,keepdims=True)
        prefix_details.append(dict(b=b,w=w,time=now,
            root_visibility_mean_contrast_mse=float(np.mean(center(arr['actual_prefix']-arr['oracle_public_prefix']).mean(1)**2)),
            arrival_forecast_mean_contrast_mse=float(np.mean(center(arr['oracle_public_prefix']-arr['predicted_prefix']).mean(1)**2)),
            oracle_total_mean_contrast_mse=float(np.mean(center(arr['oracle_public_prefix']+arr['oracle_public_endpoint_value']-arr['success']).mean(1)**2))))
    continuation_summary={key:float(np.mean([x[key] for x in shifts])) for key in
        ('mean_success_change','observed_candidate_mean_second_moment','estimated_mean_noise','unbiased_signal_second_moment')}

    controls={}
    for mode in a.MODES:
        gains={name:[] for name in ('prefix_single','prefix_ensemble16','tail_single','total_single_tie_uniform')}
        ties={name:[] for name in gains}
        for b,w,now in sorted({key[1:] for key in grouped}):
            rs=grouped[mode,b,w,now];y,choices,_=a.matrix(rs,'success')
            single={key:a.matrix(rs,key)[0][:,0] for key in ('predicted_prefix','predicted_tail')}
            f=a.load_json(r/'forecasts'/f'{b}_{w}_{now}.json')
            lookup={(x['i'],x['j']):x for x in f['forecasts']}
            scores=dict(prefix_single=single['predicted_prefix'],tail_single=single['predicted_tail'],
                total_single_tie_uniform=single['predicted_prefix']+single['predicted_tail'],
                prefix_ensemble16=np.array([np.mean([d['prefix'] for d in lookup[ij]['draws']]) for ij in choices]))
            for name,score in scores.items():
                top=np.isclose(score,score.max(),rtol=0,atol=1e-12)
                gains[name].append(y[top].mean(0)-y.mean(0));ties[name].append(int(top.sum()))
        controls[mode]={}
        for name,values in gains.items():
            g=np.array(values);rng=np.random.default_rng(20261103)
            boot=[float(g[:,rng.integers(0,32,32)].mean()) for _ in range(2000)]
            controls[mode][name]=dict(mean_gain=float(g.mean()),conditional_replica_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                tied_top_states=sum(x>1 for x in ties[name]),tie_rule='Uniform over maximum-score ties; no candidate-index preference')
    a.write(r/'ranking_controls.json',dict(status='COMPLETE',controls=controls,
        scope='Post-hoc descriptive short-only/tail-only score controls on the same fixed labels, not trained policies or independent confirmation.'))

    a.write(r/'label_reliability_summary.json',dict(status='COMPLETE',descriptive_signal_moments=summary,rows=extra,
        continuation_shift=continuation_summary,continuation_rows=shifts,prefix_rows=prefix_details,
        scope='Post-hoc descriptive signed signal/noise moments on fixed roots; no new method selection or performance gate.'))
    csv_path=root/'output'/a.TASK/'deliverables'/'queued_candidate_reliability_states.csv'
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(extra[0]));writer.writeheader();writer.writerows(extra)
    result=dict(status='PASS',raw_labels=6144,forecasts=24,paired_root_future_blocks=768,common_generated_counts=True,
        saved_analysis_recomputed=True,source_and_transport_hashes=True,protected_manifests_equal=True,
        resource_arithmetic=True,resource_equivalent_cpus=resource['equivalent_cpus'],resource_utilization=resource['utilization'],
        supplementary_moments=summary,continuation_shift=continuation_summary,local_torch_or_physical_rerun=False,independent_review=False,
        scope='Explicit self-review of transported raw counters, statistical computation, source/transport hashes, protection records and CPU arithmetic.')
    a.write(v/'local_review.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);args=p.parse_args();review(args.task_root)

if __name__=='__main__':main()
