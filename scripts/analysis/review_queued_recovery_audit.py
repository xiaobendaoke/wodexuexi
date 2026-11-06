"""Paired replay, estimator layers and first-action factorial effect review."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_recovery_audit as experiment
from utils.queued_recovery_audit import BRANCHES, TIMES, factorial_effects, gain_layers


def paired_summary(rows,key,time,draws=4000):
    selected=[r for r in rows if r['time']==time]
    cells=np.zeros((3,2,2));counts=np.zeros_like(cells)
    for row in selected:
        index=(row['run']//2,row['run']%2,experiment.ROOT_WORKLOADS.index(row['workload']))
        cells[index]+=row[key];counts[index]+=1
    assert np.all(counts==2)
    cells/=counts
    rng=np.random.default_rng(2026110428);ws=rng.integers(0,2,(draws,2));boot=np.zeros(draws)
    for parent in range(3):
        ss=rng.integers(0,2,(draws,2))
        boot+=cells[parent][ss[:,:,None],ws[:,None,:]].mean((1,2))/3
    return dict(mean=float(cells.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        by_parent=cells.mean((1,2)).tolist(),draws=draws,
        scope='Two behavior roots and four action draws averaged within fixed-parent repeat/workload cells; two workload blocks. Exploratory.')


def geometry_difference(branch,anchor,start):
    first={x['time']:x for x in branch['trace']};second={x['time']:x for x in anchor['trace']}
    metrics={}
    for label,begin in [('late',start+3),('post',max(100,start+3))]:
        slots=list(range(begin,200))
        coverage=[];distances=[];demand=[]
        for now in slots:
            a,b=first[now],second[now]
            coverage.append(len(a['before']['covered_ues'])-len(b['before']['covered_ues']))
            demand.append(a['new_in_coverage']-b['new_in_coverage'])
            distances.append(float(np.sqrt(np.mean(np.sum((np.array(a['before']['positions'])-np.array(b['before']['positions']))**2,axis=1)))))
        metrics[label+'_covered_ue_slots']=float(sum(coverage))
        metrics[label+'_covered_new_requests']=float(sum(demand))
        metrics[label+'_mean_position_rms_m']=float(np.mean(distances)) if distances else 0.
    return metrics


def review(root):
    root=Path(root);folder=experiment.out(root)
    traces=[experiment.load(p) for p in sorted((folder/'traces').glob('*/*.json'))]
    assert len(traces)==384
    trace_index={(x['arm'],x['run'],x['workload'],x['replica']):x for x in traces}
    assert len(trace_index)==384
    for record in traces:
        assert record['factual_match']
        counts=record['counts'];trajectory=record['trace']
        for key,tracekey in [('generated','generated'),('admitted','admitted'),('success','completed'),('expired','expired')]:
            assert sum(x[tracekey] for x in trajectory)==counts[key]
        assert counts['generated']==counts['success']+counts['expired']
    states=[];factual_matches=0;unique_count=0
    for path in sorted((folder/'roots').glob('*/*.json')):
        descriptor=experiment.load(path)
        key=path.relative_to(folder/'roots').with_suffix('')
        labels=experiment.saved(folder/'labels'/key.with_suffix('.pkl.gz'))
        assert labels['factual_one_step'] and labels['root_settled']
        anchor=np.array(descriptor['anchor']);proposed=np.array(descriptor['proposed']);predicted=np.array(descriptor['predicted'])
        public=labels['forecasts'].mean(0);oracle=labels['oracle_public'];real=labels['realized']
        difference=proposed-anchor
        exact=gain_layers(*(float(np.sum(difference*q)) for q in (predicted,public,oracle,real)))
        assert np.isclose(exact['predicted']-exact['realized'],sum(exact[k] for k in ('approximation_error','prediction_error','visibility_error')))
        branches={}
        for i,j in descriptor['pairs']:
            record=experiment.load(folder/'branches'/key/f'{i}_{j}.json')
            assert record['audit']['status']=='PASS'
            assert record['counts']['generated']==record['counts']['success']+record['counts']['expired']
            assert record['cohorts']['root']['generated']+record['cohorts']['future']['generated']==record['counts']['generated']
            branches[i,j]=record;unique_count+=1;factual_matches+=record['factual_hard_match']
        effects=defaultdict(list)
        for draw in descriptor['draws']:
            selected={name:branches[tuple(pair)] for name,pair in draw.items()}
            for cohort in ('root','future','future_late','future_post'):
                generated=[x['cohorts'][cohort]['generated'] for x in selected.values()]
                assert len(set(generated))==1
            metrics={'hard_3':{name:record['prefix']['3'] for name,record in selected.items()},
                     'hard_10':{name:record['prefix']['10'] for name,record in selected.items()},
                     'hard_30':{name:record['prefix']['30'] for name,record in selected.items()},
                     'hard_end':{name:record['success'] for name,record in selected.items()}}
            for label,q in [('predicted',predicted),('public',public),('oracle_public',oracle),('edf_3',real)]:
                metrics[label]={name:float(q[tuple(pair)]) for name,pair in draw.items()}
            for cohort in ('root','future','future_late','future_post'):
                for count in ('admitted','success'):
                    metrics[cohort+'_'+count]={name:record['cohorts'][cohort][count] for name,record in selected.items()}
            for name,values in metrics.items():
                factor=factorial_effects(values)
                assert np.isclose(factor['total'],factor['allocation']+factor['motion']+factor['interaction'])
                for component,value in factor.items():effects[name+'/'+component].append(value)
            for branch in ('allocation','motion','full'):
                geometry=geometry_difference(selected[branch],selected['anchor'],descriptor['time'])
                for metric,value in geometry.items():effects['geometry/'+branch+'/'+metric].append(value)
        averaged={key:float(np.mean(value)) for key,value in effects.items()}
        for component in ('total','allocation','motion','interaction'):
            averaged['continuation/'+component]=averaged['edf_3/'+component]-averaged['hard_3/'+component]
            averaged['future_effect/'+component]=averaged['hard_end/'+component]-averaged['hard_3/'+component]
        states.append(dict(behavior=descriptor['behavior'],run=descriptor['run'],parent=descriptor['parent'],
            workload=descriptor['workload'],time=descriptor['time'],active=descriptor['control']['alpha']>0,
            public_tasks=descriptor['public_tasks'],private_tasks=descriptor['private_tasks'],
            forecast_arrivals=float(np.mean(labels['forecast_arrivals'])),actual_arrivals=float(np.mean(labels['actual_arrivals'])),
            **{'exact/'+key:value for key,value in exact.items()},**averaged))
    assert len(states)==96 and factual_matches==48
    assert unique_count==experiment.load(folder/'branch_budget.json')['unique_branches']
    groups={}
    numeric=[key for key in states[0] if '/' in key]+['forecast_arrivals','actual_arrivals']
    for time in TIMES:
        rows=[r for r in states if r['time']==time]
        groups[str(time)]=dict(roots=len(rows),active_roots=sum(x['active'] for x in rows),
            hidden_tasks_mean=float(np.mean([x['private_tasks']-x['public_tasks'] for x in rows])),
            statistics={key:paired_summary(states,key,time) for key in numeric})
    analysis=dict(status='COMPLETE',replay_episodes=384,roots=96,unique_branches=unique_count,
        mapped_branches=1536,factual_hard_remainders=factual_matches,groups=groups,states=states,
        new_optimizer_steps=0,formal_test_access=False,independent_review=False,
        scope='Old workloads replayed, fixed policies and two workload blocks. Paired first-action factorial interventions followed by frozen hard policy. No new independent performance test.')
    experiment.write(folder/'analysis.json',analysis)
    return analysis


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,required=True)
    review(parser.parse_args().task_root)
