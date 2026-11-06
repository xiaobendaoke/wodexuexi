"""Paired checks for arrival accuracy, corrected scoring and coverage tails."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_arrival_adaptation as experiment


def paired(a,da,b,db,draws=4000):
    a,da,b,db=(np.asarray(x,float) for x in (a,da,b,db))
    assert a.shape==da.shape==b.shape==db.shape==(3,2,8,2)
    assert all(np.isfinite(x).all() for x in (a,da,b,db)) and np.all(da>0) and np.all(db>0)
    rng=np.random.default_rng(2026110432);ws=rng.integers(0,8,(draws,8));rs=rng.integers(0,2,(draws,8,2));boot=np.zeros(draws)
    effects=a.sum((2,3))/da.sum((2,3))-b.sum((2,3))/db.sum((2,3))
    for parent in range(3):
        ss=rng.integers(0,2,(draws,2))
        sums=[x[parent][ss[:,:,None,None],ws[:,None,:,None],rs[:,None,:,:]].sum((2,3)) for x in (a,da,b,db)]
        boot+=(sums[0]/sums[1]-sums[2]/sums[3]).mean(1)/3
    return dict(mean=float(effects.mean()),by_parent=effects.mean(1).tolist(),draws=draws,
        ci95=np.quantile(boot,[.025,.975]).tolist(),simultaneous97_5=np.quantile(boot,[.0125,.9875]).tolist(),
        simultaneous98_75=np.quantile(boot,[.00625,.99375]).tolist(),
        scope='Fixed three parents, repeats within parent and paired workload/action blocks.')


def prediction_review(root):
    records=[experiment.load(p) for p in sorted((experiment.out(root)/'prediction_test').glob('*.json'))]
    assert len(records)==96
    summaries=defaultdict(list);matrices={method:{phase:{key:np.zeros((3,2,8,2)) for key in ('absolute_error','poisson_deviance','error')}
        for phase in ('pre','peak','recovery','post','all')} for method in ('slow',)+experiment.FAST_METHODS}
    for row in records:
        index=(row['run']//2,row['run']%2,row['workload']-949000,row['replica'])
        for method in matrices:
            summary=experiment.prediction_summary(row['predictions'],method)
            for phase in summary:
                for key in matrices[method][phase]:matrices[method][phase][key][index]=summary[phase][key]
            for key in matrices[method]['all']:
                values=[]
                for record in row['predictions']:
                    if record['time']>=198:continue
                    values.extend(experiment.count_losses(record['rates'][method],n)[key] for n in record['actual'])
                matrices[method]['all'][key][index]=np.mean(values)
    selected=experiment.selected(root);comparisons={}
    for phase,key in [('recovery','absolute_error'),('all','poisson_deviance')]:
        x,y=matrices[selected][phase][key],matrices['slow'][phase][key]
        comparisons[phase+'/'+key]=paired(x,np.ones_like(x),y,np.ones_like(y))
    primary=[dict(metric=key,result=value,passed=value['simultaneous97_5'][1]<0) for key,value in comparisons.items()]
    summary={method:{phase:{key:float(values.mean()) for key,values in fields.items()}
        for phase,fields in phases.items()} for method,phases in matrices.items()}
    curves={}
    for method in matrices:
        curves[method]={str(time):dict(estimate=float(np.mean([row['predictions'][time]['rates'][method] for row in records])),
            actual=float(np.mean([row['predictions'][time]['actual'] for row in records])))
            for time in (49,50,51,52,55,60,99,100,101,102,105,110,120,150)}
    return dict(selection=experiment.load(experiment.out(root)/'predictor_selection.json'),summary=summary,
        comparisons=comparisons,primary=primary,curves=curves)


def performance_review(root):
    rows=[experiment.load(p) for p in sorted((experiment.out(root)/'episodes').glob('*/*.json'))]
    assert len(rows)==1776
    lookup={(r['arm'],r['run'],r['workload'],r['replica']):r for r in rows};assert len(lookup)==1776
    matrices={};summary={};fingerprints={}
    for arm in experiment.ARMS+('initial',):
        own=[r for r in rows if r['arm']==arm];baseline=arm=='initial'
        assert len(own)==(48 if baseline else 96)
        arrays={phase:{key:np.zeros((3,2,8,2)) for key in ('generated','admitted','success')}
                for phase in ('all','pre','peak','post','recovery')}
        for run in range(6):
            for wi,w in enumerate(experiment.TEST_WORKLOADS):
                for rep in (0,1):
                    record=lookup[arm,run//2*2 if baseline else run,w,rep];counts=record['counts']
                    assert record['audit']['status']=='PASS' and counts['generated']==counts.get('success',0)+counts.get('expired',0)
                    assert sum(record['successes'])==counts.get('success',0)
                    pair=(w,rep);fp=(record['fingerprint'],counts['generated'])
                    assert pair not in fingerprints or fingerprints[pair]==fp;fingerprints[pair]=fp
                    for phase in arrays:
                        source=counts if phase=='all' else record['recovery'] if phase=='recovery' else record['cohorts']['arrival'][phase]
                        for key in arrays[phase]:arrays[phase][key][run//2,run%2,wi,rep]=source.get(key,0)
        arrays['nonpeak']={key:arrays['pre'][key]+arrays['post'][key] for key in arrays['pre']};matrices[arm]=arrays
        result={}
        for phase in arrays:
            parts=('pre','post') if phase=='nonpeak' else (phase,)
            totals={key:sum((r['counts'] if p=='all' else r['recovery'] if p=='recovery' else r['cohorts']['arrival'][p]).get(key,0)
                for r in own for p in parts) for key in ('generated','admitted','success')}
            g,a,s=(totals[key] for key in ('generated','admitted','success'))
            result[phase]=dict(counts=totals,dsr=s/g,admission=a/g,conditional=s/a)
        result['physical_episodes']=len(own)
        result['decision_mean']=float(np.mean([v for r in own for _,v in r['decisions']]))
        result['decision_over_second']=float(np.mean([v>1 for r in own for _,v in r['decisions']]))
        result['peak_wasted_cycle_fraction']=sum(x['wasted_cycles'] for r in own for x in r['node_quality']['peak'])/max(1.,sum(x['served_cycles'] for r in own for x in r['node_quality']['peak']))
        summary[arm]=result
    pairs=set()
    for mode in experiment.MODES:
        for config in ('slow_rates','fast_base','fast_rates'):
            pairs.add((config+'_'+mode,'slow_base_'+mode))
        pairs.add(('fast_rates_'+mode,'slow_rates_'+mode));pairs.add(('fast_rates_'+mode,'fast_base_'+mode))
        for tail in ('full','queue','shuffled'):pairs.add(('tail_'+tail+'_'+mode,'fast_rates_'+mode))
        pairs.add(('tail_full_'+mode,'tail_queue_'+mode));pairs.add(('tail_full_'+mode,'tail_shuffled_'+mode))
        pairs.add(('fast_rates_'+mode,'old_reg_'+mode));pairs.add(('tail_full_'+mode,'old_hard_'+mode))
    for config in experiment.BASE_CONFIGS+experiment.TAIL_CONFIGS:
        pairs.add((config+'_joint',config+'_decomposed'))
    comparisons={}
    for a,b in sorted(pairs):
        result={}
        for phase in ('all','peak','recovery','nonpeak'):
            x,y=matrices[a][phase],matrices[b][phase]
            assert np.array_equal(x['generated'],y['generated'])
            result[phase]={metric:paired(x['success'],x[den],y['success'],y[den]) for metric,den in [('dsr','generated'),('conditional','admitted')]}
        comparisons[a+'--'+b]=result
    stage2_specs=[('fast_rates_joint--slow_rates_joint','all','dsr'),('fast_rates_joint--slow_rates_joint','recovery','conditional')]
    stage2=[dict(pair=pair,phase=phase,metric=metric,result=comparisons[pair][phase][metric],
        passed=comparisons[pair][phase][metric]['simultaneous97_5'][0]>0) for pair,phase,metric in stage2_specs]
    stage3_specs=[('tail_full_joint--fast_rates_joint','all',0.),('tail_full_joint--tail_queue_joint','all',0.),
                  ('tail_full_joint--tail_shuffled_joint','all',0.),('tail_full_joint--fast_rates_joint','nonpeak',-.0025)]
    stage3=[dict(pair=pair,phase=phase,metric='dsr',threshold=threshold,result=comparisons[pair][phase]['dsr'],
        passed=comparisons[pair][phase]['dsr']['simultaneous98_75'][0]>threshold) for pair,phase,threshold in stage3_specs]
    return dict(summary=summary,comparisons=comparisons,stage2_primary=stage2,stage3_primary=stage3)


def review(root):
    root=Path(root);folder=experiment.out(root)
    arrival=prediction_review(root);performance=performance_review(root)
    fits={f'{arm}/{run}':experiment.load(folder/'fits'/arm/f'{run}.json') for arm in experiment.NEW_ARMS for run in range(6)}
    tail_fits=[experiment.load(p) for p in sorted((folder/'tail_fits').glob('*.json'))];assert len(tail_fits)==6
    tail_summary={}
    for method in ('h3','full','queue','shuffled','oracle_endpoint_full','oracle_endpoint_queue','oracle_endpoint_shuffled'):
        records=[x['methods'][method] for fit in tail_fits for x in fit['diagnostics']]
        tail_summary[method]={key:float(np.mean([r[key] for r in records])) for key in records[0]}
    diagnostics=defaultdict(list)
    for p in sorted((folder/'recovery_diagnostics').glob('*/*.pkl.gz')):
        row=experiment.saved(p)
        for arm,record in row['methods'].items():diagnostics[arm+'/'+str(row['time'])].append(record)
    assert sum(len(values) for values in diagnostics.values())==96*18
    recovery={name:dict(roots=len(rows),gains={key:float(np.mean([r['gains'][key] for r in rows])) for key in rows[0]['gains']},
        mean_realized_regret=float(np.mean([r['realized']['score_regret'] for r in rows]))) for name,rows in diagnostics.items()}
    result=dict(status='COMPLETE',arrival=arrival,performance=performance,fits=fits,tail_heldout=tail_summary,
        tail_fits=tail_fits,recovery_diagnostics=recovery,formal_test_access=False,independent_review=False,
        scope='Ordered development tests, fixed inherited parents. Arrival test949000 separate from closed-loop950000. Public inputs at control; realized futures only in training labels or evaluation.')
    experiment.write(folder/'analysis.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,required=True)
    review(parser.parse_args().task_root)
