"""Frozen all-public-task actors in closed loop; post-hoc ranking audit only."""
import argparse
import copy
from dataclasses import asdict
from functools import lru_cache
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
if __name__=='__main__':
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_learning_validation as v
from scripts.analysis import queued_planner_distillation as old
from scripts.analysis import queued_joint_mpo_experiment as t
from marl_models.queued_planner_distillation import policy
from utils.queued_learning_validation import augment,REVISION as SOURCE_REV
from utils.queued_taskset_validation import taskset_features
from utils.queued_taskset_closed_loop import (REVISION,ARMS,BASELINES,TIMES,design,cohort_statistics,
    summary_extra,calibration_record)
from utils.queued_planner_distillation import planner_target
from utils.queued_service_controllers import assignments

TASK=old.TASK


def out(root):return Path(root)/'results'/TASK/REVISION
def val(root):return Path(root)/'output'/TASK/'validation'/REVISION
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read(p):return pickle.loads(gzip.decompress(Path(p).read_bytes()))


def write(p,data):
    if Path(p).exists():raise FileExistsError(str(p))
    t.write(p,data)


def dump(p,data):
    if Path(p).exists():raise FileExistsError(str(p))
    t.dump(p,data)


def source_model(root,arm,run):
    if arm.startswith('original_'):
        return old.model_path(root,'mlp_'+arm.split('_')[1],run,2)
    feature,label,mode=arm.split('_')
    base=Path(root)/'results'/TASK/SOURCE_REV
    return (base/'stage2_taskset_followup/models'/f'{run}_{label}_{mode}.pkl.gz' if feature=='taskset'
            else base/'stage2/models'/f'{run}_summary_{label}_{mode}.pkl.gz')


@lru_cache(maxsize=128)
def actor(root,arm,run):return read(source_model(root,arm,run))


def make_extra(obs,bank):
    extra=[]
    for plan in bank['plans']:
        line=[]
        for motion in bank['actions']:
            features,count=taskset_features(obs,plan,motion)
            assert count==len(obs['tasks']),'task truncation'
            line.append(features)
        extra.append(line)
    return np.asarray(extra)


def probabilities(root,arm,run,row,obs,bank,extras=None):
    if arm=='initial':return np.broadcast_to(row['prior'][:,None]/row['m'].shape[1],row['m'].shape[:2]).copy()
    if arm.startswith('taskset_'):
        extra=extras['taskset'] if extras is not None else make_extra(obs,bank)
        row=augment(row,extra)
    elif arm.startswith('summary_'):
        extra=extras['summary'] if extras is not None else summary_extra(row)
        row=augment(row,extra)
    return policy(row,actor(str(root),arm,run),'decomposed' if arm.endswith('decomposed') else 'joint')


def private_snapshot(engine):
    return dict(time=engine.t,positions=list(engine.positions),ues=dict(engine.ues),
                tasks=[asdict(z) for z in engine.tasks.values()],protocol=engine.p)


def episode(root,arm,run,parent,workload,replica):
    old.require_server();spec=design()
    if arm not in ARMS+BASELINES or workload not in spec['workloads'] or replica not in spec['replicas']:
        raise ValueError('outside locked development design')
    if (arm in ARMS and (run not in range(6) or parent!=run//2)) or (arm in BASELINES and (run!=-1 or parent not in range(3))):
        raise ValueError('invalid inherited policy identity')
    stem=f'{arm}/{run}_{parent}_{workload}_{replica}'
    path=out(root)/'episodes'/(stem+'.json')
    if path.exists():raise FileExistsError('episode exists before physical work')
    proposer=t.parent(root,parent);tape=t.RecoveryScenario(workload,'burst','normal',.6)
    engine=t.Engine(t.Protocol(),t.START,tape.ue_positions);adapter=t.LearningAdapter()
    states=[];successes=[];decisions=[];flight=0.;initial=None
    collect=workload in spec['diagnostic_workloads'] and replica==0
    while engine.t<200 or engine.tasks:
        now=int(engine.t)
        if now>204:raise AssertionError('undrained canonical tail')
        engine.add(tape.slot(now));engine.admit();obs=engine.observe()
        if initial is None:initial=old.value_fingerprint(obs) if hasattr(old,'value_fingerprint') else hashlib.sha256(pickle.dumps((obs['time'],obs['positions'],obs['ues'],obs['tasks']),5)).hexdigest()
        start=time.perf_counter()
        bank=t.bank(proposer,obs,adapter,t.seed(REVISION,'bank',parent,workload,replica,now))
        row=old.public_row(bank,obs,adapter.history)
        if arm=='plan_short':
            scores=old.short_scores(bank,obs,adapter.history,t.seed(REVISION,'online_forecast',parent,workload,replica,now))
            p=planner_target(scores,row['prior'])
        else:p=probabilities(root,arm,run,row,obs,bank)
        rng=np.random.default_rng(t.seed(REVISION,'action',parent,workload,replica,now))
        i,j=np.unravel_index(int(rng.choice(p.size,p=p.ravel())),p.shape)
        decisions.append(time.perf_counter()-start)
        if collect and now in TIMES:
            # The private packet has no path into the actor or public predictor.
            state=dict(arm=arm,run=run,parent=parent,workload=workload,replica=replica,time=now,
                row=row,obs=obs,bank=bank,history=copy.deepcopy(adapter.history),probabilities=p,
                chosen=[int(i),int(j)],private=dict(full=private_snapshot(engine),
                    future=[copy.deepcopy(tape.slot(now+offset)) for offset in (1,2)]))
        else:state=None
        motion=bank['actions'][j];flight+=sum(float(np.hypot(*d)) for d in motion)
        engine.assign(bank['plans'][i]);reward=engine.advance(motion)['success'];successes.append(reward)
        if state is not None:state['factual_success_slot']=reward;states.append(state)
    counts=dict(engine.counts);audit=engine.audit()
    assert not engine.tasks and counts['generated']==counts.get('success',0)+counts.get('expired',0)
    assert sum(successes)==counts.get('success',0) and audit['status']=='PASS'
    cohorts=cohort_statistics(engine.finished.values())
    assert sum(z['generated'] for z in cohorts['arrival'].values())==counts['generated']
    assert sum(z['admitted'] for z in cohorts['arrival'].values())==counts['admitted']
    assert sum(z['success'] for z in cohorts['arrival'].values())==counts.get('success',0)
    record=dict(arm=arm,run=run,parent=parent,workload=workload,replica=replica,counts=counts,
        metrics=engine.summary(),cohorts=cohorts,audit=audit,successes=successes,
        fingerprint=tape.fingerprint,initial_fingerprint=initial,decision_mean=float(np.mean(decisions)),
        decision_max=max(decisions),over_second=sum(x>1 for x in decisions),slots=len(decisions),
        flight_distance_m=flight,joule_energy=None,public_task_truncation=0,diagnostic_states=len(states))
    write(path,record)
    if states:
        assert len(states)==6;dump(out(root)/'states'/(stem+'.pkl.gz'),states)
    return dict(arm=arm,workload=workload,states=len(states))


def realized_scores(state):
    private=state['private'];bank=state['bank'];k=len(bank['plans']);m=len(bank['actions'])
    total=np.zeros((k,m));accepted=np.zeros((k,m));root_ids={x['uid'] for x in private['full']['tasks']}
    admitted_ids={x['uid'] for x in private['full']['tasks'] if x['source'] is not None}
    matched=False
    for i,plan in enumerate(bank['plans']):
        for j,motion in enumerate(bank['actions']):
            engine=t.Engine.from_observation(private['full']);engine.assign(plan)
            first=engine.advance(motion)['success']
            if [i,j]==state['chosen']:
                assert first==state['factual_success_slot'],'private residual one-step differs from fact'
                matched=True
            for tasks in private['future']:
                engine.add(tasks);engine.admit();engine.assign(assignments(engine.observe()))
                engine.advance([(0.,0.)]*len(engine.positions))
            assert engine.audit()['status']=='PASS' and not(root_ids & engine.tasks.keys())
            total[i,j]=engine.counts.get('success',0)
            accepted[i,j]=sum(uid in admitted_ids and x.stage=='completed' for uid,x in engine.finished.items())
    assert matched
    return total,accepted


def diagnostic(root,file,index):
    state=read(Path(root)/file)[index];row=state['row'];prior=row['prior']
    forecasts=np.array([v.predict_scores(state,t.seed(REVISION,'diagnostic_forecast',state['parent'],state['workload'],state['time'],rep))
                        for rep in range(16)])
    reference=np.mean([planner_target(q,prior) for q in forecasts],axis=0)
    real,accepted=realized_scores(state);real_ref=planner_target(real,prior)
    own=dict(public=calibration_record(state['probabilities'],reference,forecasts.mean(0)),
             realized=calibration_record(state['probabilities'],real_ref,real),
             root_admitted=calibration_record(state['probabilities'],planner_target(accepted,prior),accepted))
    cross={}
    if state['run']>=0:
        extras=dict(summary=summary_extra(row),taskset=make_extra(state['obs'],state['bank']))
        for arm in ARMS+('initial',):
            p=probabilities(root,arm,state['run'],row,state['obs'],state['bank'],extras)
            cross[arm]=dict(public=calibration_record(p,reference,forecasts.mean(0)),
                realized=calibration_record(p,real_ref,real),
                root_admitted=calibration_record(p,planner_target(accepted,prior),accepted))
    stem=f'{state["arm"]}/{state["run"]}_{state["parent"]}_{state["workload"]}_{state["time"]}'
    write(out(root)/'diagnostics'/(stem+'.json'),dict(arm=state['arm'],run=state['run'],parent=state['parent'],
        workload=state['workload'],time=state['time'],own=own,cross=cross,
        public_tasks=len(state['obs']['tasks']),private_tasks=len(state['private']['full']['tasks']),
        factual_one_step_match=True,root_tasks_settled=True,private_future_online_access=False))
    dump(out(root)/'audit_matrices'/(stem+'.pkl.gz'),dict(forecasts=forecasts,reference=reference,
        realized=real,root_admitted=accepted,prior=prior,own_probability=state['probabilities']))
    return stem


def init_worker(cpus):os.sched_setaffinity(0,{cpus[(mp.current_process()._identity[-1]-1)%len(cpus)]})


def timed(payload):
    kind,key,args=payload;start=time.time();cpu=time.process_time()
    result=episode(*args) if kind=='episode' else diagnostic(*args)
    return dict(kind=kind,key=key,start=start,end=time.time(),cpu_seconds=time.process_time()-cpu,
        pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),threads=len(os.listdir('/proc/self/task')),result=result)


def dispatch(root,kind,jobs):
    cpus=sorted(os.sched_getaffinity(0));start=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=len(cpus),mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(cpus,)) as pool:
        futures=[pool.submit(timed,(kind,key,(str(root),)+args)) for key,args in jobs]
        for f in as_completed(futures):
            records.append(f.result())
            if len(records)%12==0:print(kind,len(records),'/',len(jobs),flush=True)
    elapsed=time.perf_counter()-start;used=sum(x['cpu_seconds'] for x in records)
    write(val(root)/(kind+'_complete.json'),dict(status='COMPLETE',tasks=len(records),wall_seconds=elapsed,
        worker_cpu_seconds=used,equivalent_cpus=used/elapsed,utilization=used/elapsed/len(cpus),records=records))


def protection(root):
    return {**v.source_protection(root),**{str(p.relative_to(root)):t.sha(p) for p in
        (Path(root)/'results'/TASK/SOURCE_REV).rglob('*') if p.is_file()}}


def run(root):
    old.require_server();root=Path(root).resolve()
    if (out(root)/'protocol.json').exists():raise FileExistsError('immutable run; no restart')
    before=protection(root);models={str(source_model(root,a,r).relative_to(root)):t.sha(source_model(root,a,r)) for a in ARMS for r in range(6)}
    text=(REPO_ROOT/'docs/plans/chapter5_dynamic_service_20261028.md').read_text(encoding='utf-8')
    write(out(root)/'protocol.json',dict(specification=design(),sealed_models=models,protected=before,
        protocol_text=text.split('## 2026-11-04 学习瓶颈按序验证')[0],
        source={str(p.relative_to(REPO_ROOT)):t.sha(p) for folder in ('utils','marl_models','scripts/analysis','tests') for p in (REPO_ROOT/folder).rglob('*.py')},
        formal_test_access=False,new_optimizer_steps=0))
    try:
        jobs=[]
        # Long planning jobs start early so short model calls can fill their tail.
        for arm in ('plan_short',)+ARMS+('initial',):
            identities=[(r,r//2) for r in range(6)] if arm in ARMS else [(-1,b) for b in range(3)]
            for r,b in identities:
                for w in design()['workloads']:
                    for rep in (0,1):jobs.append((f'{arm}:{r}:{b}:{w}:{rep}',(arm,r,b,w,rep)))
        assert len(jobs)==1056;dispatch(root,'episode',jobs)
        jobs=[(f'{p}:{i}',(str(p.relative_to(root)),i)) for p in sorted((out(root)/'states').glob('*/*.pkl.gz')) for i in range(6)]
        assert len(jobs)==792;dispatch(root,'diagnostic',jobs)
        assert protection(root)==before,'historical artifacts changed'
        from scripts.analysis.review_queued_taskset_closed_loop import review
        review(root)
        write(val(root)/'run_complete.json',dict(status='COMPLETE',episodes=1056,diagnostic_roots=792,
            old_artifacts_unchanged=True,new_optimizer_steps=0,formal_test_access=False,independent_review=False))
    except Exception:
        write(val(root)/'run_failed.json',dict(error=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,default=REPO_ROOT);a=p.parse_args();run(a.task_root)
