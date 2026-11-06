"""Frozen-state candidate-label reliability diagnostics; heavy work is server-only.

All new Monte Carlo labels are offline conditional simulator evaluations. No
actor, tail-value fit, environment, candidate selection, or old result is changed.
"""
import argparse
import copy
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
import numpy as np

TASK='chapter5_dynamic_service_20261028'
REV='queued_candidate_reliability_20261103'
REPLICAS=tuple(range(5000,5032))
MODES=('a2','matched')


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,p)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def result_dir(root):return Path(root)/'results'/TASK/REV
def validation_dir(root):return Path(root)/'output'/TASK/'validation'/REV
def load_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>1e-12 and np.std(b)>1e-12 else None

def paired_statistics(y,pred):
    y=np.asarray(y,float);pred=np.asarray(pred,float)
    if y.ndim!=2 or y.shape[1]<2 or pred.shape!=(y.shape[0],) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('finite candidate-by-paired-replica matrix required')
    pairs=[];concordance=[]
    for i,j in itertools.combinations(range(len(y)),2):
        diff=y[i]-y[j];delta=float(diff.mean());estimate=float(pred[i]-pred[j])
        if abs(delta)>1e-12 and abs(estimate)>1e-12:concordance.append(float(delta*estimate>0))
        pairs.append(dict(i=i,j=j,mean=delta,se=float(diff.std(ddof=1)/math.sqrt(len(diff))),
            unpaired_se=float(math.sqrt((y[i].var(ddof=1)+y[j].var(ddof=1))/len(diff))),prediction_difference=estimate))
    chosen=int(pred.argmax());gain=y[chosen]-y.mean(0)
    return dict(pairs=pairs,mean_success=y.mean(1).tolist(),chosen=chosen,prediction_gain=float(gain.mean()),
                prediction_gain_se=float(gain.std(ddof=1)/math.sqrt(len(gain))),
                pair_concordance=float(np.mean(concordance)) if concordance else None,
                exactly_flat=bool(np.ptp(y.mean(1))<1e-12))

def error_components(total,prefix,endpoint_value,pred_prefix,pred_value):
    total,prefix,endpoint_value=[np.asarray(x,float) for x in (total,prefix,endpoint_value)]
    p=np.asarray(pred_prefix,float)[:,None];v=np.asarray(pred_value,float)[:,None]
    return dict(error=total-p-v,components=[prefix-p,total-prefix-endpoint_value,endpoint_value-v])

def crossfit_selection(y,n,seed=20261103,draws=100):
    y=np.asarray(y,float)
    if y.ndim!=2 or n<1 or 2*n>y.shape[1]:raise ValueError('two disjoint nonempty replica sets required')
    rng=np.random.default_rng(seed);gains=[];agree=[];a=[];b=[]
    for _ in range(draws):
        ids=rng.permutation(y.shape[1]);left=y[:,ids[:n]].mean(1);right=y[:,ids[n:2*n]].mean(1)
        gains.append(float(right[int(left.argmax())]-right.mean()))
        agree.append(float(left.argmax()==right.argmax()));a.extend(left-left.mean());b.extend(right-right.mean())
    return dict(n_per_half=n,partitions=draws,heldout_selection_gain_mean=float(np.mean(gains)),
                half_choice_agreement=float(np.mean(agree)),centered_half_correlation=corr(a,b),
                note='Repeated partitions of the same data, not independent experiments.')

def snapshot_path(root,b,w,now):
    from scripts.analysis import queued_joint_mpo_experiment as t
    return t.validation(root)/'candidate_states'/f'{b}_{w}_{now}.pkl.gz'

def states(root):
    from scripts.analysis import queued_joint_mpo_experiment as t
    rows=load_json(t.out(root)/'candidate_state_index.json')
    assert len(rows)==24
    return sorted(rows,key=lambda r:(r['time'],r['b'],r['w']))

def label_name(b,w,now,i,j,rep,mode):return f'{b}_{w}_{now}_{i}_{j}_{rep}_{mode}.json'

def rollout(root,b,w,now,i,j,rep,mode):
    from scripts.analysis import queued_joint_mpo_experiment as t
    from utils.queued_long_value import remaining_future
    from utils.queued_joint_mpo import forecast_window
    from utils.queued_service_controllers import assignments
    snap=t.read(str(snapshot_path(root,b,w,now)))
    e=copy.deepcopy(snap['engine']);adapter=copy.deepcopy(snap['adapter']);row=snap['row']
    root_ids=set(e.tasks);model=t.parent(root,b);future=remaining_future(w,e.ues,now,rep)
    tail=t.read(str(t.out(root)/'tail'/f'{b}.pkl.gz'))
    e.assign(row['plans'][i]);e.advance(row['actions'][j])
    prefix=endpoint=None;root_active_at_h=None
    while e.t<200 or e.tasks:
        tick=int(e.t)
        if tick==now+3:
            prefix=int(e.counts['success']);root_active_at_h=len(root_ids & set(e.tasks))
            endpoint=t.tail_predict(tail,t.value_features(e.observe(),adapter.history),tick,not e.tasks and tick>=200)
        e.add(copy.deepcopy(future.get(tick,())));e.admit();obs=e.observe()
        if mode=='matched' and tick<now+3:
            # Same public-history timing as the forecast; no future exposed.
            adapter.history._update_history(obs);adapter.last_time=obs['time']
            plan=assignments(obs);motion=[(0.,0.)]*len(e.positions)
        else:
            _,plan,motion=model.act(obs,adapter,t.seed(t.REV,'candidate_tail',b,w,now,rep,tick))
        e.assign(plan);e.advance(motion)
    audit=e.audit();c=e.counts
    assert prefix is not None and c['generated']==c['success']+c['expired']
    root_success=sum(z.uid in root_ids and z.stage=='completed' for z in e.finished.values())
    rec=dict(b=b,w=w,time=now,i=i,j=j,replica=rep,mode=mode,success=c['success'],generated=c['generated'],expired=c['expired'],
        actual_prefix=prefix,actual_tail=c['success']-prefix,actual_endpoint_value=endpoint,
        root_success=root_success,future_success=c['success']-root_success,root_tasks=len(root_ids),root_active_at_h=root_active_at_h,
        predicted_prefix=float(row['short'][i,j]),predicted_tail=float(row['tail'][i,j]),audit=audit)
    if mode=='matched':
        oracle=forecast_window(snap['engine'].observe(),snap['adapter'].history,row['plans'][i],row['actions'][j],
            [future.get(now+k,[]) for k in (1,2)],3)
        rec['oracle_public_prefix']=oracle['success']
        rec['oracle_public_endpoint_value']=t.tail_predict(tail,oracle['terminal'],oracle['end_time'],oracle['terminal_empty'])
    return rec

def forecasts(root,b,w,now):
    from scripts.analysis import queued_joint_mpo_experiment as t
    from utils.queued_joint_mpo import arrival_prediction,forecast_window
    snap=t.read(str(snapshot_path(root,b,w,now)));row=snap['row'];obs=snap['engine'].observe();h=snap['adapter'].history
    tail=t.read(str(t.out(root)/'tail'/f'{b}.pkl.gz'));arrivals=arrival_prediction(obs,h,t.seed(t.REV,'forecast',b,w,now),3)
    result=[]
    for i,j in snap['choices']:
        saved=forecast_window(obs,h,row['plans'][i],row['actions'][j],arrivals,3)
        value=t.tail_predict(tail,saved['terminal'],saved['end_time'],saved['terminal_empty'])
        assert saved['success']==row['short'][i,j] and abs(value-row['tail'][i,j])<1e-9
        draws=[]
        for k in range(16):
            public=arrival_prediction(obs,h,t.seed(REV,'public_forecast',b,w,now,k),3)
            f=forecast_window(obs,h,row['plans'][i],row['actions'][j],public,3)
            draws.append(dict(prefix=f['success'],tail=t.tail_predict(tail,f['terminal'],f['end_time'],f['terminal_empty'])))
        result.append(dict(i=i,j=j,saved_prefix=saved['success'],saved_tail=value,draws=draws))
    return dict(b=b,w=w,time=now,forecasts=result,saved_teacher_reconstruction='PASS')

def protection(root):
    from scripts.analysis import queued_joint_mpo_experiment as t
    original=t.protect(root);paths=[]
    for folder in (t.out(root)/'models',t.out(root)/'tail',t.out(root)/'candidate_validation',t.validation(root)/'candidate_states',Path(root)/'source_joint_mpo_run_20261102'):
        paths += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [t.out(root)/'analysis.json',t.out(root)/'protocol.json']
    original.update({str(p.relative_to(root)):digest(p) for p in paths})
    return original

def smoke(root):
    from scripts.analysis import queued_joint_mpo_experiment as t
    rec=rollout(root,0,11100,175,0,0,400,'a2')
    old=load_json(t.out(root)/'candidate_validation'/'0_11100_175_0_0_400.json')
    keys=('success','generated','expired','actual_prefix','actual_tail','actual_endpoint_value','predicted_prefix','predicted_tail')
    assert all(abs(rec[k]-old[k])<1e-9 for k in keys)
    prediction=forecasts(root,0,11100,175)
    matched=rollout(root,0,11100,175,0,0,400,'matched')
    return dict(status='PASS',old_label_exact_replay=True,old_record=old,new_replay=rec,
                forecast=prediction,matched_smoke=matched)

def initialize(cpus):
    os.sched_setaffinity(0,{cpus[(mp.current_process()._identity[-1]-1)%len(cpus)]})

def timed(payload):
    kind,args=payload;start=time.time();cpu=time.process_time()
    try:
        rec=rollout(*args) if kind=='rollout' else forecasts(*args)
        root=args[0]
        name=label_name(*args[1:]) if kind=='rollout' else f'{args[1]}_{args[2]}_{args[3]}.json'
        dest=result_dir(root)/('labels' if kind=='rollout' else 'forecasts')/name
        write(dest,rec)
        return dict(ok=True,kind=kind,file=str(dest.relative_to(root)),start=start,end=time.time(),cpu_seconds=time.process_time()-cpu,
                    pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),threads=1)
    except Exception:
        import traceback
        return dict(ok=False,kind=kind,args=list(args[1:]),error=traceback.format_exc(),start=start,end=time.time(),cpu_seconds=time.process_time()-cpu)

def run(root,resume=False):
    if sys.platform!='linux':raise RuntimeError('Full physical/Torch work is server-only')
    r=result_dir(root);v=validation_dir(root);cpus=sorted(os.sched_getaffinity(0))
    if r.exists() and not resume:raise FileExistsError(r)
    r.mkdir(parents=True,exist_ok=True);v.mkdir(parents=True,exist_ok=True)
    lock=v/'run.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        index=states(root)
        for s in index:assert digest(snapshot_path(root,s['b'],s['w'],s['time']))==s['sha']
        source={str(p.relative_to(REPO_ROOT)):digest(p) for p in (REPO_ROOT/'scripts/analysis/queued_candidate_reliability.py',REPO_ROOT/'tests/test_queued_candidate_reliability.py')}
        protocol=dict(revision=REV,source=source,replicas=list(REPLICAS),modes=list(MODES),states=index,conditional_states=24,
            branches=6144,public_forecasts_per_state=16,training=False,source_root=str(REPO_ROOT),allowed_cpus=cpus)
        if resume:assert load_json(r/'protocol.json')==protocol
        else:write(r/'protocol.json',protocol)
        before=protection(root)
        if resume:assert load_json(v/'protection_before.json')==before
        else:write(v/'protection_before.json',before)
        if not (v/'smoke.json').exists():
            wall=time.perf_counter();cpu=time.process_time();d=smoke(root)
            d.update(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
            write(v/'smoke.json',d)
        forecast_jobs=[('forecast',(str(root),s['b'],s['w'],s['time'])) for s in index
                       if not (r/'forecasts'/f'{s["b"]}_{s["w"]}_{s["time"]}.json').exists()]
        rollout_jobs=[]
        # Early/high-load states first; modes/candidates/roots share the queue.
        for now in (75,175):
            for rep in REPLICAS:
                for s in [x for x in index if x['time']==now]:
                    for i,j in s['choices']:
                        for mode in MODES:
                            args=(str(root),s['b'],s['w'],now,i,j,rep,mode)
                            if not (r/'labels'/label_name(*args[1:])).exists():rollout_jobs.append(('rollout',args))
        jobs=[]
        for n,job in enumerate(rollout_jobs):
            if n<len(forecast_jobs):jobs.append(forecast_jobs[n])
            jobs.append(job)
        if not rollout_jobs:jobs.extend(forecast_jobs)
        start=time.perf_counter();own=time.process_time();timings=[]
        with ProcessPoolExecutor(max_workers=len(cpus),mp_context=mp.get_context('fork'),initializer=initialize,initargs=(cpus,)) as pool:
            futures=[pool.submit(timed,job) for job in jobs]
            for future in as_completed(futures):
                rec=future.result();timings.append(rec)
                if not rec['ok']:
                    write(v/'failure.json',rec)
                    raise RuntimeError(rec['error'])
                if len(timings)%128==0:print('completed',len(timings),'/',len(jobs),'wall',round(time.perf_counter()-start,1),flush=True)
        wall=time.perf_counter()-start;own=time.process_time()-own;cpu=sum(x['cpu_seconds'] for x in timings)
        active=peak=0
        for _,change in sorted([(x['start'],1) for x in timings]+[(x['end'],-1) for x in timings]):active+=change;peak=max(active,peak)
        resource=dict(tasks=len(jobs),allowed_cpus=cpus,workers=len(cpus),threads=1,wall_seconds=wall,worker_cpu_seconds=cpu,coordinator_cpu_seconds=own,
            equivalent_cpus=(cpu+own)/wall,utilization=(cpu+own)/wall/len(cpus),peak_active=peak,failures=0,retries=0,records=timings)
        name='resources.json' if not resume else f'resources_resume_{int(time.time())}.json'
        write(v/name,resource)
        assert protection(root)==before
        write(v/'protection_after.json',before)
        analyze(root)
        write(v/'server_review.json',dict(status='PASS',branches=6144,states=24,protected_files=len(before),source_unchanged=all(digest(REPO_ROOT/p)==d for p,d in source.items()),
            full_old_label_replayed=True,all_saved_teacher_predictions_reconstructed=True,training=False,independent_review=False))
        print('COMPLETE',json.dumps({k:resource[k] for k in ('tasks','wall_seconds','equivalent_cpus','utilization','peak_active')}),flush=True)
    finally:lock.unlink(missing_ok=True)

def matrix(records,field):
    choices=sorted({(x['i'],x['j']) for x in records});reps=sorted({x['replica'] for x in records})
    lookup={(x['i'],x['j'],x['replica']):x[field] for x in records}
    assert len(lookup)==len(records)==len(choices)*len(reps)
    return np.array([[lookup[i,j,rep] for rep in reps] for i,j in choices],float),choices,reps

def aggregate(groups,predictions,bootstrap=2000):
    # Shared replica resampling preserves correlations across fixed roots.
    centered_truth=[];centered_pred=[];gain=[];stats=[];cross={n:[] for n in (1,2,4,8,16)}
    for key,rs in sorted(groups.items()):
        y,choices,reps=matrix(rs,'success');p=np.array([predictions[key][a] for a in choices])
        d=paired_statistics(y,p);d.update(b=key[0],w=key[1],time=key[2],choices=choices)
        stats.append(d);centered_truth.extend(y.mean(1)-y.mean());centered_pred.extend(p-p.mean())
        gain.append(y[int(p.argmax())]-y.mean(0))
        for n in cross:
            if 2*n<=len(reps):cross[n].append(crossfit_selection(y,n))
    truth=np.asarray(centered_truth);p=np.asarray(centered_pred);g=np.asarray(gain);rng=np.random.default_rng(20261103)
    samples=[float(g[:,rng.integers(0,g.shape[1],g.shape[1])].mean()) for _ in range(bootstrap)]
    pair=[p for s in stats for p in s['pairs']]
    return dict(states=len(groups),replicas_per_state=g.shape[1],prediction_gain=float(g.mean()),
        conditional_replica_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),
        within_state_ev=float(1-np.mean((truth-p)**2)/np.mean(truth**2)) if np.mean(truth**2)>1e-12 else None,
        within_state_correlation=corr(truth,p),mean_pair_se=float(np.mean([x['se'] for x in pair])),
        mean_unpaired_se=float(np.mean([x['unpaired_se'] for x in pair])),
        descriptive_pairs_over_2se=sum(abs(x['mean'])>2*x['se'] for x in pair),pairs=len(pair),
        crossfit={n:{field:float(np.mean([x[field] for x in ds])) for field in ('heldout_selection_gain_mean','half_choice_agreement')}
                  for n,ds in cross.items() if ds},rows=stats,
        uncertainty_scope=f'Conditional on these {len(groups)} fixed roots/3 inherited lineages/4 geometries. No new training replication or confirmation.')

def analyze(root,legacy=False):
    from collections import defaultdict
    from scripts.analysis import queued_joint_mpo_experiment as t
    folder=t.out(root)/'candidate_validation' if legacy else result_dir(root)/'labels'
    rows=[load_json(p) for p in folder.glob('*.json')]
    if legacy:
        assert len(rows)==384
        for x in rows:x['mode']='a2'
    else:assert len(rows)==6144,len(rows)
    for x in rows:
        assert x['generated']==x['success']+x['expired'] and x['success']==x['actual_prefix']+x['actual_tail']
        if not legacy:assert x['root_success']+x['future_success']==x['success'] and x['audit']['status']=='PASS'
    output=dict(status='COMPLETE',branches=len(rows),legacy=legacy,by_mode={})
    for mode in sorted({x['mode'] for x in rows}):
        groups=defaultdict(list)
        for x in rows:
            if x['mode']==mode:groups[x['b'],x['w'],x['time']].append(x)
        assert len(groups)==24
        predictions={k:{(x['i'],x['j']):x['predicted_prefix']+x['predicted_tail'] for x in rs} for k,rs in groups.items()}
        current=aggregate(groups,predictions)
        if not legacy:
            ensemble={}
            for k in groups:
                f=load_json(result_dir(root)/'forecasts'/f'{k[0]}_{k[1]}_{k[2]}.json')
                ensemble[k]={(x['i'],x['j']):float(np.mean([d['prefix']+d['tail'] for d in x['draws']])) for x in f['forecasts']}
            current['ensemble16']=aggregate(groups,ensemble)
        current['by_time']={now:aggregate({k:v for k,v in groups.items() if k[2]==now},predictions,bootstrap=400) for now in (75,175)}
        if mode=='matched':
            terms=[];mean_terms=[];errors=[];mean_errors=[];label_noise=[];prefix_visibility=[];prefix_forecast=[];oracle=[]
            for k,rs in sorted(groups.items()):
                arrays={f:matrix(rs,f)[0] for f in ('success','actual_prefix','actual_endpoint_value','predicted_prefix','predicted_tail','oracle_public_prefix','oracle_public_endpoint_value')}
                comp=error_components(arrays['success'],arrays['actual_prefix'],arrays['actual_endpoint_value'],arrays['predicted_prefix'][:,0],arrays['predicted_tail'][:,0])
                assert np.max(np.abs(comp['error']-sum(comp['components'])))<1e-8
                center=lambda a:a-a.mean(axis=0,keepdims=True)
                terms.append([center(a).ravel() for a in comp['components']]);errors.extend(center(comp['error']).ravel())
                mean_terms.append([center(a).mean(1) for a in comp['components']]);mean_errors.extend(center(comp['error']).mean(1))
                label_noise.extend(np.var(center(arrays['success']),axis=1,ddof=1)/arrays['success'].shape[1])
                prefix_visibility.extend(center(arrays['actual_prefix']-arrays['oracle_public_prefix']).ravel())
                prefix_forecast.extend(center(arrays['oracle_public_prefix']-arrays['predicted_prefix']).ravel())
                oracle.extend(center(arrays['oracle_public_endpoint_value']+arrays['oracle_public_prefix']-arrays['success']).ravel())
            components=np.concatenate(terms,axis=1);cov=components@components.T/components.shape[1]
            means=np.concatenate(mean_terms,axis=1);mean_cov=means@means.T/means.shape[1]
            current['error_decomposition']=dict(component_names=['prefix_error','actual_endpoint_value_error','predicted_endpoint_value_shift'],
                centered_component_mse=np.diag(cov).tolist(),cross_second_moments=cov.tolist(),centered_total_mse=float(np.mean(np.asarray(errors)**2)),
                reconstruction_mse=float(cov.sum()),centered_mean_component_mse=np.diag(mean_cov).tolist(),
                centered_mean_cross_second_moments=mean_cov.tolist(),centered_mean_total_mse=float(np.mean(np.square(mean_errors))),
                estimated_mean_label_noise=float(np.mean(label_noise)),prefix_hidden_root_mse=float(np.mean(np.square(prefix_visibility))),
                prefix_forecast_arrival_mse=float(np.mean(np.square(prefix_forecast))),offline_oracle_public_total_mse=float(np.mean(np.square(oracle))))
        output['by_mode'][mode]=current
    if not legacy:
        paired=defaultdict(dict)
        for x in rows:paired[(x['b'],x['w'],x['time'],x['i'],x['j'],x['replica'])][x['mode']]=x
        differences=[]
        for key,rs in paired.items():
            assert set(rs)==set(MODES)
            assert rs['a2']['generated']==rs['matched']['generated']
            differences.append(dict(b=key[0],w=key[1],time=key[2],success_difference=rs['a2']['success']-rs['matched']['success'],
                prefix_difference=rs['a2']['actual_prefix']-rs['matched']['actual_prefix'],root_success_difference=rs['a2']['root_success']-rs['matched']['root_success']))
        output['continuation_difference']=dict(direction='a2 minus matched',mean=float(np.mean([x['success_difference'] for x in differences])),
            rms=float(np.sqrt(np.mean([x['success_difference']**2 for x in differences]))),
            mean_prefix=float(np.mean([x['prefix_difference'] for x in differences])),
            root_active_at_h=sum(x['root_active_at_h'] for x in rows),rows=differences)
    path=validation_dir(root)/'legacy_analysis.json' if legacy else result_dir(root)/'analysis.json'
    write(path,output)
    print(json.dumps(dict(status=output['status'],branches=len(rows),legacy=legacy,
        metrics={k:{f:v[f] for f in ('prediction_gain','conditional_replica_bootstrap95','within_state_ev','mean_pair_se','mean_unpaired_se')} for k,v in output['by_mode'].items()}),indent=2))
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--run',action='store_true');mode.add_argument('--analyze',action='store_true');mode.add_argument('--legacy',action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.task_root.resolve()
    if a.run:run(root,a.resume)
    else:analyze(root,a.legacy)

if __name__=='__main__':main()
