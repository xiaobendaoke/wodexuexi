"""Frozen-policy replay and paired recovery-state counterfactual diagnostics."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

if __name__=='__main__':
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_score_control as source
from utils.queued_learning_validation import augment
from utils.queued_joint_mpo import arrival_prediction, forecast_window
from utils.queued_service_controllers import assignments
from utils.queued_recovery_audit import REVISION, TIMES, action_factors, telemetry, remainder_cohorts

TASK=source.TASK
ARMS=('hard_joint','reg_gated_joint','reg_alloc_joint','reg_motion_joint')
ROOT_ARMS=ARMS[:2]
ROOT_WORKLOADS=(948000,948004)
t=source.previous.engine_tools


def out(root):return Path(root)/'results'/TASK/REVISION
def val(root):return Path(root)/'output'/TASK/'validation'/REVISION
def load(path):return source.load(path)
def saved(path):return source.saved(path)
def write(path,value):source.write(path,value)
def dump(path,value):source.dump(path,value)


def root_key(state):return f'{state["behavior"]}/{state["run"]}_{state["workload"]}_{state["replica"]}_{state["time"]}'


def decision(root,arm,run,workload,replica,engine,adapter,proposer):
    now=int(engine.t);obs=engine.observe();parent=run//2
    bank=t.bank(proposer,obs,adapter,t.seed(source.REVISION,'bank',parent,workload,replica,now))
    row=source.previous.previous.original.public_row(bank,obs,adapter.history)
    row=augment(row,source.previous.extra(obs,bank,'pool'))
    probability,control=source.distribution(root,arm,run,row,obs,bank)
    seed=t.seed(source.REVISION,'action',parent,workload,replica,now)
    chosen=tuple(map(int,np.unravel_index(np.random.default_rng(seed).choice(probability.size,p=probability.ravel()),probability.shape)))
    return obs,bank,row,probability,control,chosen


def advance_trace(engine,bank,chosen,generated,admitted,before_counts):
    before=telemetry(engine);i,j=chosen
    engine.assign(bank['plans'][i]);reward=engine.advance(bank['actions'][j])['success']
    after=telemetry(engine)
    return reward,dict(time=int(before['time']),before=before,after=after,chosen=list(chosen),
        motion=bank['actions'][j],generated=generated,admitted=admitted,
        completed=engine.counts.get('success',0)-before_counts.get('success',0),
        expired=engine.counts.get('expired',0)-before_counts.get('expired',0))


def replay(root,arm,run,workload,replica):
    source.previous.previous.original.require_server()
    historical=load(source.out(root)/'episodes'/arm/f'{run}_{workload}_{replica}.json')
    tape=t.RecoveryScenario(workload,'burst','normal',.6)
    engine=t.Engine(t.Protocol(),t.START,tape.ue_positions);adapter=t.LearningAdapter();proposer=t.parent(root,run//2)
    trace=[];successes=[];packets=[]
    while engine.t<200 or engine.tasks:
        now=int(engine.t);assert now<=204
        before_counts=dict(engine.counts);arrivals=tape.slot(now)
        engine.add(arrivals);admitted=engine.admit()
        obs,bank,row,probability,control,chosen=decision(root,arm,run,workload,replica,engine,adapter,proposer)
        expected=historical['controls'][now]
        for key in ('alpha','pressure','tv','predicted_gain'):
            if expected[key] is None:assert control[key] is None
            else:assert np.isclose(control[key],expected[key],rtol=1e-12,atol=1e-12),(now,key)
        capture=arm in ROOT_ARMS and workload in ROOT_WORKLOADS and replica==0 and now in TIMES
        state=None
        if capture:
            state=dict(behavior=arm,run=run,parent=run//2,workload=workload,replica=replica,time=now,
                obs=obs,bank=bank,row=row,adapter=copy.deepcopy(adapter),chosen=chosen,
                private=dict(full=source.closed.private_snapshot(engine),
                             future=[copy.deepcopy(tape.slot(now+offset)) for offset in (1,2)]),
                trace_file=f'traces/{arm}/{run}_{workload}_{replica}.json')
        reward,record=advance_trace(engine,bank,chosen,len(arrivals),len(admitted),before_counts)
        record['new_in_coverage']=sum(x.ue in record['before']['covered_ues'] for x in arrivals)
        record['control']=control;trace.append(record);successes.append(reward)
        assert reward==historical['successes'][now],('factual reward',arm,run,workload,replica,now)
        if state is not None:state['factual_first_success']=reward;packets.append(state)
    assert dict(engine.counts)==historical['counts'] and tape.fingerprint==historical['fingerprint']
    assert engine.audit()['status']=='PASS'
    assert sum(x['generated'] for x in trace)==engine.counts['generated']
    assert sum(x['admitted'] for x in trace)==engine.counts['admitted']
    assert sum(x['completed'] for x in trace)==engine.counts['success']
    assert sum(x['expired'] for x in trace)==engine.counts['expired']
    write(out(root)/'traces'/arm/f'{run}_{workload}_{replica}.json',dict(arm=arm,run=run,parent=run//2,
        workload=workload,replica=replica,counts=dict(engine.counts),fingerprint=tape.fingerprint,
        ue_positions=tape.ue_positions,trace=trace,factual_match=True))
    for state in packets:dump(out(root)/'states'/(root_key(state)+'.pkl.gz'),state)
    return dict(arm=arm,roots=len(packets),factual_match=True)


def prepare(root,file):
    state=saved(Path(root)/file);row=state['row'];run=state['run']
    anchor,_=source.distribution(root,'hard_joint',run,row,state['obs'],state['bank'])
    proposed,control=source.distribution(root,'reg_gated_joint',run,row,state['obs'],state['bank'])
    predicted=source.row_scores(row,source.scorer(str(root),'reg','joint',run),'joint')
    seeds=[t.seed(source.REVISION,'action',run//2,state['workload'],state['replica'],state['time'])]
    seeds.extend(t.seed(REVISION,'paired_action',run//2,state['workload'],state['replica'],state['time'],draw) for draw in (1,2,3))
    mappings,unique=action_factors(anchor,proposed,seeds)
    factual='anchor' if state['behavior']=='hard_joint' else 'full'
    assert mappings[0][factual]==tuple(state['chosen'])
    value=dict(state_file=file,behavior=state['behavior'],run=run,parent=run//2,workload=state['workload'],
        replica=state['replica'],time=state['time'],draws=mappings,pairs=unique,control=control,
        anchor=anchor.tolist(),proposed=proposed.tolist(),predicted=predicted.tolist(),
        public_tasks=len(state['obs']['tasks']),private_tasks=len(state['private']['full']['tasks']))
    write(out(root)/'roots'/(root_key(state)+'.json'),value)
    return len(unique)


def clone_audit(engine,start):
    audit=engine.audit()
    for node,served in enumerate(engine.served_cycles):
        assert served<=engine.p.capacity[node]*engine.cpu_busy[node]*(1+1e-9)+1e-5
    assert all(x<=engine.t-start+1e-7 for x in engine.cpu_busy)
    assert all(x<=engine.t-start+1e-7 for x in engine.airtime.values())
    return audit


def realized_bank(state,public_root=False):
    full=copy.deepcopy(state['private']['full'])
    if public_root:full['tasks']=copy.deepcopy(state['obs']['tasks'])
    bank=state['bank'];result=np.zeros((len(bank['plans']),len(bank['actions'])))
    root_ids={x['uid'] for x in full['tasks']}
    for i,plan in enumerate(bank['plans']):
        for j,motion in enumerate(bank['actions']):
            engine=t.Engine.from_observation(full);engine.assign(plan);first=engine.advance(motion)['success']
            if not public_root and (i,j)==tuple(state['chosen']):assert first==state['factual_first_success']
            for arrivals in state['private']['future']:
                engine.add(arrivals);engine.admit();engine.assign(assignments(engine.observe()))
                engine.advance([(0.,0.)]*len(engine.positions))
            assert not(root_ids & engine.tasks.keys())
            clone_audit(engine,state['time']);result[i,j]=engine.counts.get('success',0)
    return result


def labels(root,file):
    state=saved(Path(root)/file);forecasts=[];arrivals=[]
    for replica in range(16):
        future=arrival_prediction(state['obs'],state['adapter'].history,
            t.seed(REVISION,'public_prediction',state['parent'],state['workload'],state['time'],replica),3)
        arrivals.append([len(x) for x in future])
        forecasts.append([[forecast_window(state['obs'],state['adapter'].history,plan,motion,future,3)['success']
                           for motion in state['bank']['actions']] for plan in state['bank']['plans']])
    real=realized_bank(state);oracle=realized_bank(state,True)
    dump(out(root)/'labels'/(root_key(state)+'.pkl.gz'),dict(forecasts=np.asarray(forecasts,float),
        oracle_public=oracle,realized=real,forecast_arrivals=arrivals,
        actual_arrivals=[len(x) for x in state['private']['future']],factual_one_step=True,root_settled=True))
    return 1


def counterfactual(root,file,i,j):
    state=saved(Path(root)/file);start=state['time'];root_ids={x['uid'] for x in state['private']['full']['tasks']}
    tape=t.RecoveryScenario(state['workload'],'burst','normal',.6)
    engine=t.Engine.from_observation(state['private']['full']);adapter=copy.deepcopy(state['adapter'])
    proposer=t.parent(root,state['parent']);trace=[];successes=[];prefix={}
    before_counts=dict(engine.counts)
    reward,record=advance_trace(engine,state['bank'],(i,j),0,0,before_counts)
    record['new_in_coverage']=0
    trace.append(record);successes.append(reward)
    if (i,j)==tuple(state['chosen']):assert reward==state['factual_first_success']
    factual=state['behavior']=='hard_joint' and (i,j)==tuple(state['chosen'])
    historical=load(out(root)/state['trace_file']) if factual else None
    while engine.t<200 or engine.tasks:
        now=int(engine.t);assert now<=204
        before_counts=dict(engine.counts);arrivals=tape.slot(now)
        engine.add(arrivals);admitted=engine.admit()
        obs,bank,row,p,control,chosen=decision(root,'hard_joint',state['run'],state['workload'],state['replica'],engine,adapter,proposer)
        reward,record=advance_trace(engine,bank,chosen,len(arrivals),len(admitted),before_counts)
        record['new_in_coverage']=sum(x.ue in record['before']['covered_ues'] for x in arrivals)
        trace.append(record);successes.append(reward)
    clone_audit(engine,start)
    assert not engine.tasks and sum(successes)==engine.counts.get('success',0)
    if factual:
        expected=historical['trace'][start:]
        assert len(expected)==len(trace)
        for actual,old in zip(trace,expected):
            assert actual['chosen']==old['chosen'] and actual['completed']==old['completed']
            assert actual['after']['positions']==old['after']['positions']
            assert actual['after']['covered_ues']==old['after']['covered_ues']
    for horizon in (3,10,30):prefix[str(horizon)]=sum(successes[:horizon])
    cohorts=remainder_cohorts(engine.finished.values(),root_ids,start)
    assert cohorts['root']['generated']+cohorts['future']['generated']==engine.counts['generated']
    write(out(root)/'branches'/root_key(state)/f'{i}_{j}.json',dict(behavior=state['behavior'],run=state['run'],
        parent=state['parent'],workload=state['workload'],replica=state['replica'],time=start,pair=[i,j],
        prefix=prefix,success=engine.counts.get('success',0),counts=dict(engine.counts),cohorts=cohorts,
        trace=trace,factual_hard_match=factual,audit=engine.audit()))
    return dict(pair=[i,j],factual_hard_match=factual)


def worker(payload):
    kind,args=payload;started=time.perf_counter();cpu=time.process_time()
    result=dict(replay=replay,prepare=prepare,labels=labels,counterfactual=counterfactual)[kind](*args)
    return dict(cpu_seconds=time.process_time()-cpu,seconds=time.perf_counter()-started,result=result)


def dispatch(root,stage,kind,jobs):
    assert not (val(root)/(stage+'_complete.json')).exists()
    cpus=sorted(os.sched_getaffinity(0));started=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(jobs)),mp_context=mp.get_context('spawn'),
        initializer=source.previous.previous.init_worker,initargs=(cpus,)) as pool:
        futures=[pool.submit(worker,(kind,(str(root),)+tuple(args))) for args in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%12==0:print(stage,len(records),'/',len(jobs),flush=True)
    elapsed=time.perf_counter()-started;cpu=sum(x['cpu_seconds'] for x in records)
    write(val(root)/(stage+'_complete.json'),dict(status='COMPLETE',tasks=len(jobs),wall_seconds=elapsed,
        worker_cpu_seconds=cpu,equivalent_cpus=cpu/elapsed,utilization=cpu/elapsed/len(cpus),records=records))


def protected(root):
    return source.protected(root) | {str(p.relative_to(root)):t.sha(p) for p in source.out(root).rglob('*') if p.is_file()}


def preflight(root):
    source.previous.previous.original.require_server()
    packet=saved(next((source.closed.out(root)/'states/taskset_single_joint').glob('*.pkl.gz')))[0]
    engine=t.Engine.from_observation(packet['private']['full']);i,j=packet['chosen']
    engine.assign(packet['bank']['plans'][i]);first=engine.advance(packet['bank']['actions'][j])['success']
    assert first==packet['factual_success_slot'];clone_audit(engine,packet['time'])
    print('private residual one-step factual preflight passed',flush=True)


def run(root):
    source.previous.previous.original.require_server();root=Path(root).resolve()
    assert not (out(root)/'protocol.json').exists(),'immutable audit already started'
    before=protected(root)
    write(out(root)/'protocol.json',dict(revision=REVISION,protected=before,new_optimizer_steps=0,
        replay_episodes=384,roots=96,action_draws=4,maximum_remainder_branches=1536,
        protocol=(REPO_ROOT/'docs/plans/chapter5_dynamic_service_20261028.md').read_text(encoding='utf-8').split('## 2026-11-04 收益评分与拥塞条件决策分离')[0],
        source={str(p.relative_to(REPO_ROOT)):t.sha(p) for folder in ('utils','marl_models','scripts/analysis','tests') for p in (REPO_ROOT/folder).rglob('*.py')},
        formal_test_access=False,independent_review=False))
    try:
        jobs=[(arm,run,w,rep) for arm in ARMS for run in range(6) for w in source.WORKLOADS for rep in (0,1)]
        assert len(jobs)==384;dispatch(root,'factual_replay','replay',jobs)
        files=sorted((out(root)/'states').glob('*/*.pkl.gz'));assert len(files)==96
        dispatch(root,'root_preparation','prepare',[(str(p.relative_to(root)),) for p in files])
        dispatch(root,'label_banks','labels',[(str(p.relative_to(root)),) for p in files])
        jobs=[]
        for file in files:
            state=saved(file);mapping=load(out(root)/'roots'/(root_key(state)+'.json'))
            jobs.extend((str(file.relative_to(root)),i,j) for i,j in mapping['pairs'])
        assert 96<=len(jobs)<=1536
        write(out(root)/'branch_budget.json',dict(mapped_branches=1536,unique_branches=len(jobs),deduplicated=1536-len(jobs)))
        dispatch(root,'counterfactuals','counterfactual',jobs)
        assert protected(root)==before,'historical data changed'
        from scripts.analysis.review_queued_recovery_audit import review
        review(root)
        write(val(root)/'run_complete.json',dict(status='COMPLETE',replay_episodes=384,roots=96,
            unique_remainder_branches=len(jobs),mapped_branches=1536,public_forecasts=1536,
            new_optimizer_steps=0,historical_artifacts_unchanged=True,formal_test_access=False,independent_review=False))
    except Exception:
        write(val(root)/'run_failed.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,default=REPO_ROOT)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    preflight(args.task_root) if args.preflight else run(args.task_root)
