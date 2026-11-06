"""One bounded follow-up for the registered 32-task diagnostic truncation."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
if __name__=='__main__':
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_learning_validation as e
from utils.queued_taskset_validation import taskset_features
from utils.queued_learning_validation import augment, train_curve, phase
from marl_models.queued_planner_distillation import initialize, policy
from utils.queued_planner_distillation import check_rows, imitation_metrics


def out(root):return e.out(root)/'stage2_taskset_followup'


def extract(root, file):
    states=e.read(Path(root)/file);items=[]
    stem=Path(file).stem.replace('.pkl','')
    for index,state in enumerate(states):
        bank=state['bank'];extra=[];count=None
        for plan in bank['plans']:
            values=[]
            for motion in bank['actions']:
                feature,n=taskset_features(state['obs'],plan,motion);values.append(feature);count=n
            extra.append(values)
        items.append(dict(label=stem+f'_{index}.pkl.gz',extra=np.asarray(extra),tasks=count))
    e.dump(out(root)/'features'/(stem+'.pkl.gz'),items)
    return len(items)


def fitting(root,run,label,mode):
    feature_items={item['label']:item for p in sorted((out(root)/'features').glob(f'{run}_*.pkl.gz')) for item in e.read(p)}
    records=[];train=[];hold=[]
    for name,item in sorted(feature_items.items()):
        record=e.read(e.out(root)/'stage2/labels'/name);row=augment(record['row'],item['extra'])
        row['q']=record['forecasts'][0] if label=='single' else record['forecasts'].mean(0)
        records.append((record,item));(train if row['split']=='training' else hold).append(row)
    check_rows(train,hold);assert len(train)==144 and len(hold)==48
    actor=initialize(train[0],hidden=32,seed=e.t.seed(e.REVISION,'feature_actor',run))
    fitted,curve=train_curve(train,hold,actor,mode,64,(4,16,32,64),e.t.seed(e.REVISION,'feature_fit',run))
    evaluation=[]
    for record,item in records:
        if record['row']['split']!='holdout':continue
        row=augment(record['row'],item['extra']);q=record['forecasts'].mean(0)
        z=imitation_metrics(q,policy(row,fitted,mode),row['prior']);z.update(time=row['time'],workload=row['workload'])
        evaluation.append(z)
    name=f'{run}_{label}_{mode}'
    e.dump(out(root)/'models'/(name+'.pkl.gz'),fitted)
    e.write(out(root)/'fits'/(name+'.json'),dict(run=run,feature='taskset',label=label,mode=mode,curve=curve,
        heldout_common_reference=evaluation,parameter_count=sum(v.size for b in fitted['parameters'].values() for v in b.values()),
        no_holdout_selection=True,no_truncation=True))
    return name


def timed(job):
    kind,args=job;start=time.time();cpu=time.process_time()
    result=extract(*args) if kind=='extract' else fitting(*args)
    return dict(kind=kind,start=start,end=time.time(),cpu_seconds=time.process_time()-cpu,result=result,
                affinity=sorted(os.sched_getaffinity(0)))


def dispatch(kind,entries):
    cpus=sorted(os.sched_getaffinity(0));records=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(entries)),mp_context=mp.get_context('spawn'),
            initializer=e.init_worker,initargs=(cpus,)) as pool:
        futures=[pool.submit(timed,(kind,args)) for args in entries]
        for f in as_completed(futures):records.append(f.result())
    wall=time.perf_counter()-start;used=sum(r['cpu_seconds'] for r in records)
    return dict(tasks=len(records),wall_seconds=wall,utilization=used/wall/len(cpus),records=records)


def aggregate(records):
    groups={}
    for feature in ('summary','task','taskset'):
        for label in ('single','mean16'):
            for mode in ('joint','decomposed'):
                rs=sorted([r for r in records if (r['feature'],r['label'],r['mode'])==(feature,label,mode)],key=lambda r:r['run'])
                assert len(rs)==6
                per_run=[]
                for r in rs:
                    result={}
                    for p in ('all','peak','nonpeak'):
                        zs=[z for z in r['heldout_common_reference'] if p=='all' or phase(z['time'])==p]
                        available=sum(z['available_planning_gain'] for z in zs)
                        result[p]=dict(score_regret=float(np.mean([z['score_regret'] for z in zs])),
                            retained_planning_gain=sum(z['expected_score_gain'] for z in zs)/available if available>1e-12 else None)
                    per_run.append(result)
                groups[f'{feature}/{label}/{mode}']=dict(parameter_count=rs[0]['parameter_count'],by_run=per_run,
                    means={p:{k:float(np.mean([r[p][k] for r in per_run])) for k in ('score_regret','retained_planning_gain')} for p in ('all','peak','nonpeak')})
    comparisons={}
    for label in ('single','mean16'):
        for mode in ('joint','decomposed'):
            for ref in ('summary','task'):
                a=groups[f'{ref}/{label}/{mode}']['by_run'];b=groups[f'taskset/{label}/{mode}']['by_run']
                comparisons[f'taskset_vs_{ref}/{label}/{mode}']={p:dict(mean=float(np.mean([x[p]['score_regret']-y[p]['score_regret'] for x,y in zip(a,b)])),
                    by_run=[x[p]['score_regret']-y[p]['score_regret'] for x,y in zip(a,b)]) for p in ('all','peak','nonpeak')}
    return dict(groups=groups,paired_regret_reduction=comparisons)


def run(root):
    e.original.require_server();root=Path(root).resolve()
    assert (e.val(root)/'feature_fit_complete.json').exists()
    before={str(p.relative_to(root)):e.t.sha(p) for p in (e.out(root)/'stage2').rglob('*') if p.is_file()}
    e.write(out(root)/'protocol.json',dict(reason='Registered 32-task test truncated 100% of peak states; one all-task follow-up',
        preserved_stage2=before,feature_dimension=64,hidden=32,epochs=64,fits=24,new_physical_episodes=0,new_forecasts=0,
        posthoc_exploratory=True,source={str(p.relative_to(REPO_ROOT)):e.t.sha(p) for p in
            [REPO_ROOT/'utils/queued_taskset_validation.py',REPO_ROOT/'scripts/analysis/queued_taskset_validation.py']}))
    resources={}
    entries=[(str(root),str(p.relative_to(root))) for p in sorted((e.out(root)/'stage2/states').glob('*.pkl.gz'))]
    assert len(entries)==96;resources['extract']=dispatch('extract',entries)
    resources['fit']=dispatch('fit',[(str(root),r,l,m) for r in range(6) for l in ('single','mean16') for m in ('joint','decomposed')])
    records=[e.load(p) for p in (e.out(root)/'stage2/fits').glob('*.json')]+[e.load(p) for p in (out(root)/'fits').glob('*.json')]
    result=aggregate(records);result.update(status='COMPLETE',resources=resources,original_stage2_unchanged=True,posthoc_exploratory=True)
    assert all(e.t.sha(root/n)==digest for n,digest in before.items())
    e.write(out(root)/'analysis.json',result)
    e.write(e.val(root)/'taskset_followup_complete.json',dict(status='COMPLETE',fit_tasks=24,no_new_physics=True,no_new_labels=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,default=REPO_ROOT);args=p.parse_args();run(args.task_root)
