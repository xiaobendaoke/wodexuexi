"""Remote post-run audit and bounded interpretation; never rewrites frozen evidence."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if platform.system()!='Linux':raise RuntimeError('Scientific statistics belong on the experiment server')
    base=a.base;root=base/'results_v2';source=base/'release_v2';validation=base/'validation_v2'
    sys.path.insert(0,str(source))
    import numpy as np
    from scipy.stats import ttest_1samp
    from ch3_analysis import effect,unit
    from utils.canonical_experiment import source_manifest
    def load(path):return json.loads(path.read_text())
    state=load(root/'status.json');lock=load(validation/'release.json');suite=load(root/'suite.json');plan=suite['plan'];report=load(root/'analysis.json')
    assert state['status']=='COMPLETE' and lock['status']=='PASS'
    assert state['source_hash']==lock['source_hash']==suite['source_hash']==report['source_hash']==source_manifest()['source_hash']
    protocol=base/'incoming/ch3_evidence_20261103_v2.json'
    assert hashlib.sha256(protocol.read_bytes()).hexdigest()==lock['protocol_sha256']==suite['protocol_sha256']
    assert load(protocol)==plan
    with (root/'evaluation_units.csv').open() as f:rows=list(csv.DictReader(f))
    lookup={(r['cell'],int(r['training_seed']),int(r['workload_seed'])):{k:float(v) for k,v in r.items() if k not in ('cell','method','training_seed','workload_seed')} for r in rows}
    expected={(c,s,w) for c in plan['cells'] for s in plan['training_seeds'] for w in plan['workload_seeds']}
    assert len(rows)==200 and len(lookup)==200 and set(lookup)==expected
    for v in lookup.values():
        assert 0<=v['successful']<=v['admitted']<=v['generated'] and v['generated']>0 and v['admitted']>0
        assert abs(v['DSR']-v['successful']/v['generated'])<1e-12
        assert abs(v['DSR']-v['alpha']*v['eta'])<1e-12
    def matrix(cell,key):return np.array([[lookup[cell,s,w][key] for w in plan['workload_seeds']] for s in plan['training_seeds']])
    for c in plan['cells']:
        for k,v in report['means'][c].items():assert abs(v-matrix(c,k).mean())<1e-12
    comparisons={}
    for name,c,b in [('primary','B','D'),('covered_architecture','A','C'),('attention_information','B','A'),('mlp_information','D','C')]:
        metrics={}
        for metric in ('DSR','alpha','eta'):
            delta=matrix(c,metric)-matrix(b,metric);v=effect(delta,plan);old=report['comparisons'][name][metric]
            assert abs(v['mean_difference']-old['mean_difference'])<1e-12
            assert np.allclose(v['t_ci95'],old['t_ci95'],rtol=0,atol=1e-12)
            assert np.allclose(v['crossed_bootstrap_ci95'],old['crossed_bootstrap_ci95'],rtol=0,atol=1e-12)
            v['two_sided_p_unadjusted']=float(ttest_1samp(delta.mean(axis=1),0).pvalue)
            metrics[metric]=v
        metrics['algebraic_decomposition']=report['comparisons'][name]['decomposition']
        comparisons[name]=metrics
    interaction=report['comparisons']['interaction']
    assert np.allclose(interaction['seed_mean_differences'],((matrix('B','DSR')-matrix('D','DSR'))-(matrix('A','DSR')-matrix('C','DSR'))).mean(axis=1),rtol=0,atol=1e-12)
    jobs=[]
    for cell,method in plan['cells'].items():
        for seed in plan['training_seeds']:
            train=root/'main'/method/f'seed_{seed}'/'train';jobs.append((train,200,False))
            assert (train/'checkpoints'/plan['final_checkpoint']/'contract.json').exists()
            for workload in plan['workload_seeds']:jobs.append((train.parent/f'workload_{workload}',6,True))
    boundary=load(root/'boundary_analysis.json');assert len(boundary['units'])==216
    invalid=[r for r in boundary['units'] if r['status']!='PASS']
    boundary_lookup={}
    for r in boundary['units']:
        q=Path(r['path']);setting,method,seed,workload=q.parts[-4:]
        boundary_lookup[setting,method,int(seed[5:]),int(workload[9:])]=r
        if r['status']=='PASS':jobs.append((q,2,True))
    for path,episodes,request_records in jobs:
        done=load(path/'completed.json');assert done['status']=='PASS' and done['source_hash']==state['source_hash']
        manifest=load(path/'manifest.json');assert manifest['source']['source_hash']==state['source_hash']
        assert len(list((path/'episodes').glob('*.json')))==episodes
        if request_records:assert len(list((path/'requests').glob('episode_*.jsonl.gz')))==episodes
    assert not list((root/'main').rglob('failure.json')) and not list((root/'boundary').rglob('failure.json'))
    assert not report['primary_criterion_met'] and (root/'capacity_not_triggered.json').exists()
    assert not (root/'main'/plan['capacity_method']).exists()
    # Check checksums and raw count aggregation in one full main unit per cell and
    # one boundary unit per method and condition. Remaining jobs get metadata/count checks.
    samples=[]
    for cell,method in plan['cells'].items():
        path=root/'main'/method/f"seed_{plan['training_seeds'][0]}"/f"workload_{plan['workload_seeds'][0]}"
        metrics,_=unit(path,state['source_hash'],6,1000)
        assert abs(metrics['DSR']-lookup[cell,plan['training_seeds'][0],plan['workload_seeds'][0]]['DSR'])<1e-12
        samples.append(str(path))
    for setting in plan['boundary']['settings']:
        for cell in ('A','B'):
            method=plan['cells'][cell];s=plan['training_seeds'][0];w=plan['boundary_workload_seeds'][0]
            rec=boundary_lookup[setting['name'],method,s,w]
            if rec['status']!='PASS':continue
            metrics,_=unit(Path(rec['path']),state['source_hash'],2,1000)
            assert abs(metrics['DSR']-rec['DSR'])<1e-12;samples.append(rec['path'])
    diagnostics=load(root/'request_diagnostics.json');assert len(diagnostics['units'])==50
    pooled=defaultdict(lambda:defaultdict(float));unit_diagnostics=[]
    for rec in diagnostics['units']:
        seed,workload=rec['training_seed'],rec['workload_seed']
        for side,c in [('left','A'),('right','B')]:
            for countname,col in [('generated','generated'),('admitted','admitted'),('successful','successful')]:
                n=sum(g.get(side+'_'+countname,0.) for g in rec['groups'].values())
                assert n==lookup[c,seed,workload][col],(c,seed,workload,countname,n)
        groups={}
        for g,counts in rec['groups'].items():
            for key,value in counts.items():pooled[g][key]+=value
            group={}
            for side in ('left','right'):
                gen=counts.get(side+'_generated',0);ad=counts.get(side+'_admitted',0);su=counts.get(side+'_successful',0)
                group[side]=dict(generated=gen,admitted=ad,successful=su,eta=su/ad if ad else None)
                for k in ('size','cycles','deadline','cache_hit','associated_users','final_node_tasks','upload_latency','forward_latency','fetch_latency','compute_latency','latency'):
                    n=counts.get(side+'_'+k+'_n',0)
                    group[side][k]=counts.get(side+'_'+k+'_sum',0)/n if n else None
            groups[g]=group
        unit_diagnostics.append(dict(training_seed=seed,workload_seed=workload,groups=groups))
    diag_pooled={}
    for g,counts in pooled.items():
        result={'count':counts['count']}
        for side in ('left','right'):
            gen=counts.get(side+'_generated',0);ad=counts.get(side+'_admitted',0);su=counts.get(side+'_successful',0)
            result[side]=dict(generated=gen,admitted=ad,successful=su,eta=su/ad if ad else None)
            for k in ('size','cycles','deadline','cache_hit','associated_users','final_node_tasks','upload_latency','forward_latency','fetch_latency','compute_latency','latency'):
                n=counts.get(side+'_'+k+'_n',0);result[side][k]=counts.get(side+'_'+k+'_sum',0)/n if n else None
        diag_pooled[g]=result
    common=[]
    for seed in plan['training_seeds']:
        samples_seed=[r for r in unit_diagnostics if r['training_seed']==seed]
        common.append([r['groups']['both_admitted']['right']['eta']-r['groups']['both_admitted']['left']['eta'] for r in sorted(samples_seed,key=lambda r:r['workload_seed'])])
    common_eta=effect(common,plan)
    boundary_results={}
    for setting in plan['boundary']['settings']:
        name=setting['name'];entry={};mat={}
        for cell in ('A','B'):
            mat[cell]={key:np.array([[boundary_lookup[name,plan['cells'][cell],s,w][key] for w in plan['boundary_workload_seeds']] for s in plan['training_seeds'][:3]]) for key in ('DSR','alpha','eta','energy')}
            entry[cell]={key:float(x.mean()) for key,x in mat[cell].items()}
        entry['B_minus_A']={key:effect(mat['B'][key]-mat['A'][key],plan) for key in ('DSR','alpha','eta')}
        boundary_results[name]=entry
    # Same frozen B policy vs its ideal-telemetry evaluation, not a new training sample.
    for name in ('radius100','radius230','delay1','delay5','drop20','drop50'):
        x=np.array([[boundary_lookup[name,plan['cells']['B'],s,w]['DSR']-boundary_lookup['main',plan['cells']['B'],s,w]['DSR'] for w in plan['boundary_workload_seeds']] for s in plan['training_seeds'][:3]])
        boundary_results[name]['B_vs_B_ideal']=effect(x,plan)
    summary=dict(status='PASS',protocol=plan['protocol'],source_hash=state['source_hash'],completion=state,
        audit=dict(metadata_checked_jobs=len(jobs),training_jobs=40,main_evaluation_jobs=200,boundary_jobs=216,main_episodes=1200,boundary_episodes=432,checksum_and_raw_count_sample_jobs=len(samples),sample_jobs=samples,invalid_boundary_units=invalid,scope='All job metadata, episode counts and main CSV statistics checked; raw hashes/counts sampled, not a second exhaustive audit.'),
        means=report['means'],comparisons=comparisons,interaction=interaction,capacity='Not triggered by the predeclared primary gate',
        diagnostic_pooled_request_weighted=diag_pooled,diagnostic_units=unit_diagnostics,common_admitted_eta_change=common_eta,
        boundary=boundary_results,notes=['Secondary comparisons and diagnostic intervals are descriptive and not multiplicity-adjusted.','Boundary n=3 training seeds; mean across 3 independent workloads within seed.','Pooled request diagnostic means use request weights, not the main equal-seed estimator.','No causal mediation or equivalence/no-effect claim.'])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','means','comparisons','interaction','diagnostic_pooled_request_weighted','common_admitted_eta_change','boundary')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
