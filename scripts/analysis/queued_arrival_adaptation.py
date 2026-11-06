"""Ordered arrival adaptation, corrected-score and learned-tail experiment."""
import argparse
import copy
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
from pathlib import Path
import sys
import time
import traceback
if __name__=='__main__':
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_score_control as old
from scripts.analysis import queued_recovery_audit as recovery
from scripts.analysis import queued_learning_validation as validation
from marl_models.queued_planner_distillation import initialize,policy
from utils.queued_learning_validation import augment,ROOT_TIMES
from utils.queued_score_control import fit_scores,row_scores,score_data,score_metrics,congestion_pressure,controlled_policy
from utils.queued_arrival_adaptation import (REVISION,FAST_METHODS,RateHistory,rate,rate_features,with_rates,
    future_arrivals,forecast_endpoint,endpoint_features,fit_tail,tail_prediction,count_losses)
from utils.queued_planner_distillation import specification
from utils.queued_taskset_closed_loop import cohort_statistics,calibration_record
from utils.queued_service_controllers import assignments
from utils.queued_learning_causes import quality_details

TASK=old.TASK;t=old.previous.engine_tools
TAIL_TIMES=(80,100,150,190)
MODES=('joint','decomposed')
BASE_CONFIGS=('slow_base','slow_rates','fast_base','fast_rates')
TAIL_CONFIGS=('tail_full','tail_queue','tail_shuffled')
NEW_ARMS=tuple(f'{name}_{mode}' for name in BASE_CONFIGS+TAIL_CONFIGS for mode in MODES)
OLD_ARMS=tuple(f'old_{name}_{mode}' for name in ('hard','reg') for mode in MODES)
ARMS=NEW_ARMS+OLD_ARMS
TEST_WORKLOADS=tuple(range(950000,950008))


def out(root):return Path(root)/'results'/TASK/REVISION
def val(root):return Path(root)/'output'/TASK/'validation'/REVISION
def load(path):return old.load(path)
def saved(path):return old.saved(path)
def write(path,value):old.write(path,value)
def dump(path,value):old.dump(path,value)
def model_path(root,arm,run):return out(root)/'models'/arm/f'{run}.pkl.gz'


def forecast_records(obs,history,temporal,tape):
    now=int(obs['time'])
    estimates={method:rate(method,obs,history,temporal) for method in ('slow',)+FAST_METHODS}
    # Truth is attached only after all causal predictions have been computed.
    return dict(time=now,rates=estimates,actual=[len(tape.slot(now+offset)) for offset in (1,2)],
        observed_current=temporal.counts[-1],generated_current=len(tape.slot(now)))


def history_replay(root,run,version,split,workload):
    original=validation.original;parent=run//2
    historical=load(original.out(root)/'episodes'/split/'mlp_joint'/f'{run}_{parent}_{workload}_0_{version}.json')
    old_states=saved(next((validation.out(root)/'stage2/states').glob(f'{run}_{version}_{split}_{workload}.pkl.gz')))
    by_time={state['row']['time']:state for state in old_states}
    proposer=t.parent(root,parent);tape=t.RecoveryScenario(workload,'burst','normal',.6)
    engine=t.Engine(t.Protocol(),t.START,tape.ue_positions);adapter=t.LearningAdapter();temporal=RateHistory()
    actor=None if version==0 else saved(original.model_path(root,'mlp_joint',run,version))
    states=[];predictions=[];successes=[]
    while engine.t<200 or engine.tasks:
        now=int(engine.t);assert now<=204
        engine.add(tape.slot(now));engine.admit();obs=engine.observe();temporal.update(obs)
        bank=t.bank(proposer,obs,adapter,t.seed(original.REVISION,'bank',parent,workload,0,now))
        row=original.public_row(bank,obs,adapter.history)
        if actor is None:actor=original.initial_model(row,'mlp_joint',run)
        p=policy(row,actor,'joint')
        if now<200:predictions.append(forecast_records(obs,adapter.history,temporal,tape))
        if now in by_time:
            state=by_time[now]
            for key in ('a','m','context','prior'):np.testing.assert_allclose(row[key],state['row'][key],atol=1e-12,rtol=1e-12)
            item=dict(row=copy.deepcopy(state['row']),obs=obs,bank=bank,history=copy.deepcopy(adapter.history),
                temporal=copy.deepcopy(temporal),adapter=copy.deepcopy(adapter),
                rates=rate_features(obs,adapter.history,temporal))
            if now in TAIL_TIMES:item['private']=old.closed.private_snapshot(engine)
            states.append(item)
        rng=np.random.default_rng(t.seed(original.REVISION,'action',parent,workload,0,now))
        i,j=np.unravel_index(rng.choice(p.size,p=p.ravel()),p.shape)
        engine.assign(bank['plans'][i]);successes.append(engine.advance(bank['actions'][j])['success'])
    assert successes==historical['successes'] and dict(engine.counts)==historical['counts'] and len(states)==12
    assert engine.audit()['status']=='PASS'
    stem=f'{run}_{version}_{split}_{workload}'
    dump(out(root)/'history_states'/(stem+'.pkl.gz'),states)
    write(out(root)/'history'/(stem+'.json'),dict(run=run,split=split,workload=workload,version=version,
        predictions=predictions,factual_match=True,counts=dict(engine.counts)))
    return stem


def prediction_summary(records,method):
    groups={phase:[] for phase in ('pre','peak','recovery','post')}
    for record in records:
        now=record['time']
        if now>=198:continue
        phase='pre' if now<50 else 'peak' if now<100 else 'recovery' if now<120 else 'post'
        for actual in record['actual']:groups[phase].append(count_losses(record['rates'][method],actual))
    return {phase:{key:float(np.mean([x[key] for x in rows])) for key in rows[0]} for phase,rows in groups.items()}


def select_predictor(root):
    files=[load(p) for p in sorted((out(root)/'history').glob('*.json'))]
    held=[r for r in files if r['split']=='holdout'];assert len(held)==24
    scores={method:float(np.mean([np.mean([x['absolute_error'] for x in prediction_summary(row['predictions'],method).values()]) for row in held]))
            for method in FAST_METHODS}
    selected=min(scores,key=lambda key:(scores[key],key))
    write(out(root)/'predictor_selection.json',dict(selected=selected,scores=scores,
        criterion='Four temporal strata equally weighted; mean two-step count MAE on 24 old heldout trajectories only.',
        selected_before_prediction_test=True,prediction_workloads=list(range(949000,949008))))
    return selected


def selected(root):return load(out(root)/'predictor_selection.json')['selected']


def prediction_test(root,run,workload,replica):
    assert workload in range(949000,949008)
    tape=t.RecoveryScenario(workload,'burst','normal',.6)
    engine=t.Engine(t.Protocol(),t.START,tape.ue_positions)
    adapter=t.LearningAdapter();temporal=RateHistory();parent=run//2;proposer=t.parent(root,parent);records=[]
    while engine.t<200 or engine.tasks:
        now=int(engine.t);assert now<=204
        engine.add(tape.slot(now));engine.admit();obs=engine.observe();temporal.update(obs)
        bank=t.bank(proposer,obs,adapter,t.seed(REVISION,'bank',parent,workload,replica,now))
        raw=validation.original.public_row(bank,obs,adapter.history)
        row=augment(raw,old.previous.extra(obs,bank,'pool'))
        p,_=old.distribution(root,'hard_joint',run,row,obs,bank)
        if now<200:records.append(forecast_records(obs,adapter.history,temporal,tape))
        rng=np.random.default_rng(t.seed(REVISION,'action',parent,workload,replica,now))
        i,j=np.unravel_index(rng.choice(p.size,p=p.ravel()),p.shape)
        engine.assign(bank['plans'][i]);engine.advance(bank['actions'][j])
    assert engine.audit()['status']=='PASS' and not engine.tasks
    write(out(root)/'prediction_test'/f'{run}_{workload}_{replica}.json',dict(run=run,parent=parent,
        workload=workload,replica=replica,predictions=records,counts=dict(engine.counts)))
    return workload


def label_bank(root,file,index,diagnostic=False):
    if diagnostic:
        state=saved(Path(root)/file);row=state['row'];now=state['time'];run=state['run']
        factual=load(recovery.out(root)/state['trace_file']);tape=t.RecoveryScenario(state['workload'],'burst','normal',.6)
        temporal=RateHistory();ues=np.asarray(tape.ue_positions)
        for frame in factual['trace'][:now+1]:
            positions=np.asarray(frame['before']['positions']);time_now=frame['time']
            visible=np.any(np.linalg.norm(ues[:,None,:]-positions[None,:,:],axis=2)<=460+1e-9,axis=1)
            temporal.update_count(time_now,sum(visible[x.ue] for x in tape.slot(time_now)))
        assert temporal.counts[-1]==sum(abs(x['arrival']-now)<1e-9 for x in state['obs']['tasks'])
        history=state['adapter'].history;obs=state['obs'];bank=state['bank']
        old_label=saved(recovery.out(root)/'labels'/(recovery.root_key(state)+'.pkl.gz'))
        name=recovery.root_key(state);stem=name
    else:
        state=saved(Path(root)/file)[index];row=state['row'];now=row['time'];run=row['run']
        obs,bank,history,temporal=(state[key] for key in ('obs','bank','history','temporal'))
        stem=Path(file).name.removesuffix('.pkl.gz')+f'_{index}'
        old_label=saved(validation.out(root)/'stage2/labels'/(stem+'.pkl.gz'))
    forecasts=[];features=[];empty=[];counts=[]
    for replica in range(16):
        if diagnostic:seed=t.seed(recovery.REVISION,'public_prediction',run//2,state['workload'],now,replica)
        elif replica==0:seed=t.seed(validation.original.REVISION,'forecast',run//2,row['workload'],0,now)
        else:seed=t.seed(validation.REVISION,'forecast',run//2,row['workload'],now,replica)
        arrivals=future_arrivals(obs,history,temporal,selected(root),seed)
        counts.append([len(x) for x in arrivals]);mat=[];x=[];terminal=[]
        for plan in bank['plans']:
            values=[forecast_endpoint(obs,history,temporal,plan,motion,arrivals) for motion in bank['actions']]
            mat.append([z['success'] for z in values]);x.append([z['features'] for z in values]);terminal.append([z['terminal_empty'] for z in values])
        forecasts.append(mat);features.append(x);empty.append(terminal)
    payload=dict(run=run,time=now,row=row,rates=rate_features(obs,history,temporal),
        fast=np.asarray(forecasts,float),slow=old_label['forecasts'],features=np.asarray(features),
        terminal=np.asarray(empty,bool),forecast_arrivals=counts)
    if diagnostic:
        payload.update(behavior=state['behavior'],workload=state['workload'],oracle_public=old_label['oracle_public'],realized=old_label['realized'])
    dump(out(root)/('diagnostic_labels' if diagnostic else 'labels')/(stem+'.pkl.gz'),payload)
    return stem


def fitting_rows(root,run,label,context,tail=None):
    base_train,base_hold=old.previous.fixed_data(root,run,'pool')
    lookup={(row['workload'],row['time']):row for row,q in base_train+base_hold}
    pairs=[];tail_model=saved(out(root)/'tail_models'/f'{run}.pkl.gz') if tail else None
    for path in sorted((out(root)/'labels').glob(f'{run}_*.pkl.gz')):
        record=saved(path);original=lookup[record['row']['workload'],record['time']]
        row=with_rates(original,record['rates'],context=='rates')
        qs=np.array(record[label],copy=True)
        if tail:qs+=tail_prediction(tail_model,tail,record['features'],record['terminal'])
        row['q']=qs.mean(0);pairs.append((row,qs,record['fast']))
    train=[x for x in pairs if x[0]['split']=='training'];held=[x for x in pairs if x[0]['split']=='holdout']
    assert len(train)==144 and len(held)==48
    return train,held


def fit_score(root,config,mode,run):
    if config.startswith('tail_'):label,context,tail='fast','rates',config.removeprefix('tail_')
    else:label,context=config.split('_');tail=None
    training,held=fitting_rows(root,run,label,context,tail)
    rows,qs,_=zip(*training);hs,hq,fastq=zip(*held)
    initial=initialize(rows[0],hidden=32,seed=t.seed(validation.REVISION,'feature_actor',run))
    model,curve=fit_scores(rows,qs,hs,hq,initial,mode,'reg',t.seed(validation.REVISION,'feature_fit',run))
    common=[dict(row,q=q.mean(0)) for row,q in zip(hs,fastq)]
    common_metrics=score_metrics(score_data(common,fastq,mode),model,'reg')
    arm=config+'_'+mode;dump(model_path(root,arm,run),model)
    write(out(root)/'fits'/arm/f'{run}.json',dict(arm=arm,run=run,curve=curve,common_fast_heldout=common_metrics,
        train_states=144,holdout_states=48,parameter_count=sum(v.size for b in model['parameters'].values() for v in b.values()),
        label=label,causal_context=context,tail=tail,optimizer_steps=1152))
    return arm


@lru_cache(maxsize=128)
def actor(root,arm,run):return saved(model_path(root,arm,run))


def tail_prepare(root,file,index):
    state=saved(Path(root)/file)[index];row=state['row'];assert row['time'] in TAIL_TIMES
    base=augment(row,old.previous.extra(state['obs'],state['bank'],'pool'))
    anchor,_=old.distribution(root,'hard_joint',row['run'],base,state['obs'],state['bank'])
    rich=with_rates(base,state['rates'],True);q=row_scores(rich,actor(str(root),'fast_rates_joint',row['run']),'joint')
    rng=np.random.default_rng(t.seed(REVISION,'tail_anchor',row['run'],row['workload'],row['time']))
    a=tuple(map(int,np.unravel_index(rng.choice(anchor.size,p=anchor.ravel()),anchor.shape)))
    b=tuple(map(int,np.unravel_index(np.argmax(np.where(anchor>0,q,-np.inf)),q.shape)))
    pairs=sorted({a,b,(a[0],b[1]),(b[0],a[1])})
    key=Path(file).name.removesuffix('.pkl.gz')+f'_{index}'
    write(out(root)/'tail_roots'/(key+'.json'),dict(state_file=file,index=index,pairs=pairs,
        run=row['run'],workload=row['workload'],time=row['time'],split=row['split']))
    return len(pairs)


def tail_branch(root,descriptor_file,i,j):
    descriptor=load(Path(root)/descriptor_file);state=saved(Path(root)/descriptor['state_file'])[descriptor['index']]
    row=state['row'];now=row['time'];run=row['run'];tape=t.RecoveryScenario(row['workload'],'burst','normal',.6)
    engine=t.Engine.from_observation(state['private']);adapter=copy.deepcopy(state['adapter']);temporal=copy.deepcopy(state['temporal'])
    root_ids=set(engine.tasks);engine.assign(state['bank']['plans'][i]);engine.advance(state['bank']['actions'][j])
    for offset in (1,2):
        engine.add(tape.slot(now+offset));engine.admit();obs=engine.observe()
        adapter.history._update_history(obs);temporal.update(obs);engine.assign(assignments(obs));engine.advance([(0.,0.)]*5)
    assert not(root_ids & engine.tasks.keys())
    prefix3=engine.counts.get('success',0);x=endpoint_features(engine.observe(),adapter.history,temporal)
    proposer=t.parent(root,run//2);score20=None;prefix={3:prefix3};trace=[]
    while engine.t<now+20 or (row['split']=='holdout' and (engine.t<200 or engine.tasks)):
        time_now=int(engine.t)
        if time_now>=200 and not engine.tasks:break
        assert time_now<=210
        arrivals=tape.slot(time_now);engine.add(arrivals);admitted=engine.admit();obs=engine.observe();temporal.update(obs)
        bank=t.bank(proposer,obs,adapter,t.seed(REVISION,'tail_bank',run//2,row['workload'],time_now))
        raw=validation.original.public_row(bank,obs,adapter.history);base=augment(raw,old.previous.extra(obs,bank,'pool'))
        p,_=old.distribution(root,'hard_joint',run,base,obs,bank)
        rng=np.random.default_rng(t.seed(REVISION,'tail_action',run//2,row['workload'],time_now))
        a,b=np.unravel_index(rng.choice(p.size,p=p.ravel()),p.shape)
        before=recovery.telemetry(engine);engine.assign(bank['plans'][a]);engine.advance(bank['actions'][b])
        trace.append(dict(time=time_now,positions=before['positions'],covered_ues=before['covered_ues'],
            covered_new=sum(z.ue in before['covered_ues'] for z in arrivals),admitted=len(admitted)))
        for horizon in (10,20,30):
            if engine.t==now+horizon:prefix[horizon]=engine.counts.get('success',0)
    if 20 not in prefix:prefix[20]=engine.counts.get('success',0)
    if row['split']=='holdout':assert not engine.tasks
    recovery.clone_audit(engine,now)
    write(out(root)/'tail_branches'/Path(descriptor_file).stem/f'{i}_{j}.json',dict(run=run,workload=row['workload'],
        time=now,split=row['split'],pair=[i,j],features=x.tolist(),h3=prefix3,h20=prefix[20],
        tail20=prefix[20]-prefix3,final_success=engine.counts.get('success',0) if row['split']=='holdout' else None,
        active=len(engine.tasks),trace=trace,audit=engine.audit()))
    return descriptor['split']


def fit_tail_model(root,run):
    rows=[];held=[]
    for p in sorted((out(root)/'tail_roots').glob(f'{run}_*.json')):
        d=load(p);branches=[load(out(root)/'tail_branches'/p.stem/f'{i}_{j}.json') for i,j in d['pairs']]
        row=dict(x=np.array([x['features'] for x in branches]),y=np.array([x['tail20'] for x in branches]),
                 time=d['time'],split=d['split'],workload=d['workload'])
        (rows if d['split']=='training' else held).append((row,p,branches))
    assert len(rows)==48 and len(held)==16
    fitted=fit_tail([x[0] for x in rows]);dump(out(root)/'tail_models'/f'{run}.pkl.gz',fitted)
    diagnostics=[]
    for row,p,branches in held:
        descriptor=load(p);label=saved(out(root)/'labels'/(p.stem+'.pkl.gz'))
        pairs=descriptor['pairs'];truth20=np.array([x['h20'] for x in branches]);truth_end=np.array([x['final_success'] for x in branches])
        h3=np.array([x['h3'] for x in branches]);public=label['fast'].mean(0)
        predictions={'h3':np.array([public[i,j] for i,j in pairs])}
        for kind in ('full','queue','shuffled'):
            q=label['fast']+tail_prediction(fitted,kind,label['features'],label['terminal'])
            predictions[kind]=np.array([q.mean(0)[i,j] for i,j in pairs])
            predictions['oracle_endpoint_'+kind]=h3+tail_prediction(fitted,kind,row['x'])
        records={}
        for kind,q in predictions.items():
            best=q>=q.max()-1e-12
            records[kind]=dict(regret20=float(truth20.max()-truth20[best].mean()),
                regret_end=float(truth_end.max()-truth_end[best].mean()),score_range=float(np.ptp(q)))
        diagnostics.append(dict(workload=row['workload'],time=row['time'],methods=records))
    write(out(root)/'tail_fits'/f'{run}.json',dict(run=run,train_roots=48,holdout_roots=16,
        feature_dimension=105,alpha=1.,diagnostics=diagnostics))
    return run


def distribution(root,arm,run,base,features,obs,bank):
    mode=arm.split('_')[-1]
    if arm.startswith('old_'):
        name='hard' if arm.startswith('old_hard') else 'reg_gated'
        return old.distribution(root,name+'_'+mode,run,base,obs,bank)
    enabled='_rates_' in arm or arm.startswith('tail_')
    row=with_rates(base,features,enabled)
    q=row_scores(row,actor(str(root),arm,run),mode)
    anchor,_=old.distribution(root,'hard_'+mode,run,base,obs,bank)
    pressure=congestion_pressure(obs,bank['plans'],anchor.sum(1))
    return controlled_policy(anchor,q,pressure,'gated')


def final_episode(root,arm,run,workload,replica):
    assert workload in TEST_WORKLOADS and (arm in ARMS or arm=='initial')
    assert arm!='initial' or run in (0,2,4)
    tape=t.RecoveryScenario(workload,'burst','normal',.6);engine=t.Engine(t.Protocol(),t.START,tape.ue_positions)
    adapter=t.LearningAdapter();temporal=RateHistory();proposer=t.parent(root,run//2)
    successes=[];decisions=[];controls=[];positions=[]
    while engine.t<200 or engine.tasks:
        now=int(engine.t);assert now<=204
        engine.add(tape.slot(now));engine.admit();obs=engine.observe();temporal.update(obs);started=time.perf_counter()
        bank=t.bank(proposer,obs,adapter,t.seed(REVISION,'bank',run//2,workload,replica,now))
        raw=validation.original.public_row(bank,obs,adapter.history)
        if arm=='initial':p=np.broadcast_to(raw['prior'][:,None]/raw['m'].shape[1],raw['m'].shape[:2]);control=dict(alpha=0.,pressure=None,tv=0.,predicted_gain=None)
        else:
            base=augment(raw,old.previous.extra(obs,bank,'pool'))
            p,control=distribution(root,arm,run,base,rate_features(obs,adapter.history,temporal),obs,bank)
        rng=np.random.default_rng(t.seed(REVISION,'action',run//2,workload,replica,now))
        i,j=np.unravel_index(rng.choice(p.size,p=p.ravel()),p.shape)
        decisions.append((now,time.perf_counter()-started));controls.append(dict(time=now,**control));positions.append(list(engine.positions))
        engine.assign(bank['plans'][i]);successes.append(engine.advance(bank['actions'][j])['success'])
    assert not engine.tasks and engine.audit()['status']=='PASS'
    assert sum(successes)==engine.counts.get('success',0)
    write(out(root)/'episodes'/arm/f'{run}_{workload}_{replica}.json',dict(arm=arm,run=run,parent=run//2,
        workload=workload,replica=replica,counts=dict(engine.counts),cohorts=cohort_statistics(engine.finished.values()),
        recovery=cohort_statistics([x for x in engine.finished.values() if 100<=x.arrival<120])['arrival']['post'],
        node_quality=quality_details(list(engine.finished.values())),successes=successes,decisions=decisions,
        controls=controls,positions=positions,fingerprint=tape.fingerprint,audit=engine.audit()))
    return arm


def recovery_diagnostics(root,file):
    record=saved(Path(root)/file);behavior=record['behavior'];run=record['run'];now=record['time']
    state=saved(recovery.out(root)/'states'/f'{behavior}/{run}_{record["workload"]}_0_{now}.pkl.gz')
    output={}
    for arm in NEW_ARMS+OLD_ARMS:
        p,control=distribution(root,arm,run,state['row'],record['rates'],state['obs'],state['bank'])
        mode=arm.split('_')[-1]
        if arm.startswith('old_'):
            q=None
        else:
            rich=with_rates(state['row'],record['rates'],'_rates_' in arm or arm.startswith('tail_'))
            q=row_scores(rich,actor(str(root),arm,run),mode)
        anchor,_=old.distribution(root,'hard_'+mode,run,state['row'],state['obs'],state['bank'])
        gain={kind:float(np.sum((p-anchor)*value)) for kind,value in [('slow',record['slow'].mean(0)),('fast',record['fast'].mean(0)),('realized',record['realized'])]}
        if q is not None:gain['predicted']=float(np.sum((p-anchor)*q))
        output[arm]=dict(gains=gain,control=control,realized=calibration_record(p,
            old.planner_target(record['realized'],state['row']['prior']),record['realized']))
    dump(out(root)/'recovery_diagnostics'/Path(file).parent.name/Path(file).name,
        dict(behavior=behavior,run=run,workload=record['workload'],time=now,methods=output))
    return 1


def worker(payload):
    kind,args=payload;started=time.perf_counter();cpu=time.process_time()
    functions=dict(history=history_replay,prediction=prediction_test,label=label_bank,fit=fit_score,
        tail_prepare=tail_prepare,tail_branch=tail_branch,tail_fit=fit_tail_model,episode=final_episode,diagnostic=recovery_diagnostics)
    result=functions[kind](*args)
    return dict(cpu_seconds=time.process_time()-cpu,seconds=time.perf_counter()-started,result=result)


def dispatch(root,stage,kind,jobs):
    assert jobs and not (val(root)/(stage+'_complete.json')).exists()
    cpus=sorted(os.sched_getaffinity(0));started=time.perf_counter();records=[]
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(jobs)),mp_context=mp.get_context('spawn'),
        initializer=validation.init_worker,initargs=(cpus,)) as pool:
        futures=[pool.submit(worker,(kind,(str(root),)+tuple(args))) for args in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%12==0:print(stage,len(records),'/',len(jobs),flush=True)
    elapsed=time.perf_counter()-started;cpu=sum(x['cpu_seconds'] for x in records)
    write(val(root)/(stage+'_complete.json'),dict(status='COMPLETE',tasks=len(jobs),wall_seconds=elapsed,
        worker_cpu_seconds=cpu,equivalent_cpus=cpu/elapsed,utilization=cpu/elapsed/len(cpus),records=records))


def protection(root):
    return recovery.protected(root)|{str(p.relative_to(root)):t.sha(p) for p in recovery.out(root).rglob('*') if p.is_file()}


def preflight(root):
    validation.original.require_server()
    state=saved(next((validation.out(root)/'stage2/states').glob('*.pkl.gz')))[0]
    temporal=RateHistory();temporal.update(state['obs'])
    slow=future_arrivals(state['obs'],state['history'],temporal,'slow',5)
    expected=old.previous.engine_tools.arrival_prediction(state['obs'],state['history'],5,3)
    assert slow==expected
    plan=state['bank']['plans'][0];motion=state['bank']['actions'][0]
    value=forecast_endpoint(state['obs'],state['history'],temporal,plan,motion,slow)
    assert value['features'].shape==(105,)
    print('causal arrival and endpoint preflight passed',flush=True)


def run(root):
    validation.original.require_server();root=Path(root).resolve();assert not (out(root)/'protocol.json').exists()
    before=protection(root)
    write(out(root)/'protocol.json',dict(revision=REVISION,protected=before,
        protocol=(REPO_ROOT/'docs/plans/chapter5_dynamic_service_20261028.md').read_text(encoding='utf-8').split('## 2026-11-04 峰后恢复误判')[0],
        source={str(p.relative_to(REPO_ROOT)):t.sha(p) for folder in ('utils','marl_models','scripts/analysis','tests') for p in (REPO_ROOT/folder).rglob('*.py')},
        formal_test_access=False,independent_review=False,new_parent_training=False))
    try:
        jobs=[]
        for run in specification()['runs']:
            for version in (0,1):
                for split in ('training','holdout'):
                    for workload in run[split][version]:jobs.append((run['id'],version,split,workload))
        assert len(jobs)==96;dispatch(root,'history_replay','history',jobs)
        select_predictor(root)
        dispatch(root,'prediction_test','prediction',[(r,w,rep) for r in range(6) for w in range(949000,949008) for rep in (0,1)])
        files=sorted((out(root)/'history_states').glob('*.pkl.gz'))
        dispatch(root,'corrected_labels','label',[(str(p.relative_to(root)),i) for p in files for i in range(12)])
        dispatch(root,'base_score_fits','fit',[(config,mode,r) for config in BASE_CONFIGS for mode in MODES for r in range(6)])
        tail_jobs=[(str(p.relative_to(root)),i) for p in files for i,state in enumerate(saved(p)) if state['row']['time'] in TAIL_TIMES]
        assert len(tail_jobs)==384;dispatch(root,'tail_preparation','tail_prepare',tail_jobs)
        branches=[]
        for p in sorted((out(root)/'tail_roots').glob('*.json')):
            d=load(p);branches.extend((str(p.relative_to(root)),i,j) for i,j in d['pairs'])
        assert 384<=len(branches)<=1536;dispatch(root,'tail_branches','tail_branch',branches)
        dispatch(root,'tail_value_fits','tail_fit',[(r,) for r in range(6)])
        dispatch(root,'tail_score_fits','fit',[(config,mode,r) for config in TAIL_CONFIGS for mode in MODES for r in range(6)])
        diagnostic_files=sorted((recovery.out(root)/'states').glob('*/*.pkl.gz'));assert len(diagnostic_files)==96
        dispatch(root,'recovery_labels','label',[(str(p.relative_to(root)),0,True) for p in diagnostic_files])
        models=list((out(root)/'models').glob('*/*.pkl.gz'));assert len(models)==84
        sealed=models+list((out(root)/'tail_models').glob('*.pkl.gz'))
        write(out(root)/'sealed_models.json',dict(files={str(p.relative_to(root)):t.sha(p) for p in sealed},sealed_before_evaluation=True))
        jobs=[(arm,r,w,rep) for arm in ARMS for r in range(6) for w in TEST_WORKLOADS for rep in (0,1)]
        jobs+=[('initial',r,w,rep) for r in (0,2,4) for w in TEST_WORKLOADS for rep in (0,1)]
        assert len(jobs)==1776;dispatch(root,'final_evaluation','episode',jobs)
        dispatch(root,'recovery_diagnostics','diagnostic',[(str(p.relative_to(root)),) for p in sorted((out(root)/'diagnostic_labels').glob('*/*.pkl.gz'))])
        assert protection(root)==before
        assert all(t.sha(root/p)==digest for p,digest in load(out(root)/'sealed_models.json')['files'].items())
        from scripts.analysis.review_queued_arrival_adaptation import review
        review(root)
        write(val(root)/'run_complete.json',dict(status='COMPLETE',history_replays=96,prediction_episodes=96,
            corrected_roots=1152,recovery_roots=96,tail_roots=384,tail_branches=len(branches),
            score_models=84,tail_model_groups=6,final_episodes=1776,historical_artifacts_unchanged=True,
            formal_test_access=False,independent_review=False))
    except Exception:
        write(val(root)/'run_failed.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,default=REPO_ROOT)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    preflight(args.task_root) if args.preflight else run(args.task_root)
