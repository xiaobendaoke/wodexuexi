"""Server experiment separating learned score differences from policy control."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_learning_causes as previous
from scripts.analysis import queued_taskset_closed_loop as closed
from marl_models.queued_planner_distillation import initialize, policy
from utils.queued_score_control import REVISION, row_scores, fit_scores, congestion_pressure, controlled_policy
from utils.queued_learning_validation import augment
from utils.queued_taskset_closed_loop import cohort_statistics, calibration_record
from utils.queued_planner_distillation import planner_target
from utils.queued_learning_causes import quality_details

TASK = previous.TASK
MODES = ('joint','decomposed')
ARMS = (tuple(f'{name}_{mode}' for name in ('hard','vote','directrank') for mode in MODES)
        +tuple(f'{objective}_{decoder}_{mode}' for objective in ('reg','rawrank') for decoder in ('uniform','gated') for mode in MODES)
        +tuple(f'prob_gated_{mode}' for mode in MODES)+('reg_alloc_joint','reg_motion_joint'))
BASELINES = ('initial','plan_short')
WORKLOADS = tuple(range(948000,948008))


def out(root):return Path(root)/'results'/TASK/REVISION
def val(root):return Path(root)/'output'/TASK/'validation'/REVISION
def load(path):return previous.load(path)
def saved(path):return previous.saved(path)
def write(path, value):previous.write(path, value)
def dump(path, value):previous.dump(path, value)
def score_path(root, objective, mode, run):return out(root)/'models'/f'{objective}_{mode}'/f'{run}.pkl.gz'


@lru_cache(maxsize=64)
def frozen_actor(root, name, mode, run):
    objective = 'rank' if name == 'directrank' else name
    return saved(previous.model_path(root, f'pool_{objective}_{mode}', run))


@lru_cache(maxsize=32)
def scorer(root, objective, mode, run):return saved(score_path(root, objective, mode, run))


def distribution(root, arm, run, row, obs, bank):
    mode = arm.split('_')[-1]
    name = arm.split('_')[0]
    if name in ('hard','vote','directrank'):
        p = policy(row, frozen_actor(str(root), name, mode, run), mode)
        return p, dict(alpha=0., pressure=None, tv=0., predicted_gain=None)
    anchor = policy(row, frozen_actor(str(root), 'hard', mode, run), mode)
    pressure = congestion_pressure(obs, bank['plans'], anchor.sum(1))
    branch = 'allocation' if arm == 'reg_alloc_joint' else 'motion' if arm == 'reg_motion_joint' else 'both'
    decoder = 'uniform' if '_uniform_' in arm else 'gated'
    if name == 'prob':
        target = policy(row, frozen_actor(str(root), 'directrank', mode, run), mode)
        return controlled_policy(anchor, np.log(target.clip(1e-300)), pressure, decoder, probability_target=target)
    predicted = row_scores(row, scorer(str(root), name, mode, run), mode)
    return controlled_policy(anchor, predicted, pressure, decoder, branch)


def fitting(root, objective, mode, run):
    training, heldout = previous.fixed_data(root, run, 'pool')
    rows, forecasts = zip(*training);hs,hq = zip(*heldout)
    initial = initialize(rows[0], hidden=32, seed=previous.engine_tools.seed(previous.previous.REVISION, 'feature_actor', run))
    model,curve = fit_scores(rows, forecasts, hs, hq, initial, mode, objective,
        previous.engine_tools.seed(previous.previous.REVISION, 'feature_fit', run))
    dump(score_path(root, objective, mode, run), model)
    write(out(root)/'fits'/f'{objective}_{mode}'/f'{run}.json', dict(objective=objective, mode=mode, run=run,
        curve=curve, train_states=len(rows), heldout_states=len(hs), optimizer_steps=64*18,
        parameter_count=sum(v.size for b in model['parameters'].values() for v in b.values()),
        score_training_uses_policy_softmax=False, heldout_selection=False))
    return f'{objective}_{mode}/{run}'


def episode(root, arm, run, workload, replica):
    previous.previous.original.require_server()
    assert arm in ARMS+BASELINES and workload in WORKLOADS and run in range(6) and replica in (0,1)
    assert arm in ARMS or run in (0,2,4)
    parent = run//2;t = previous.engine_tools
    proposer = t.parent(root, parent)
    tape = t.RecoveryScenario(workload,'burst','normal',.6)
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions);adapter = t.LearningAdapter()
    successes = [];decisions = [];controls = [];flight = 0.;peak_load = []
    while engine.t < 200 or engine.tasks:
        now = int(engine.t);assert now <= 204
        engine.add(tape.slot(now));engine.admit();obs = engine.observe()
        started = time.perf_counter()
        bank = t.bank(proposer, obs, adapter, t.seed(REVISION,'bank',parent,workload,replica,now))
        raw = previous.previous.original.public_row(bank, obs, adapter.history)
        if arm in BASELINES:
            p = (planner_target(previous.previous.original.short_scores(bank,obs,adapter.history,
                t.seed(REVISION,'forecast',parent,workload,replica,now)), raw['prior']) if arm == 'plan_short'
                else np.broadcast_to(raw['prior'][:,None]/raw['m'].shape[1], raw['m'].shape[:2]))
            control = dict(alpha=0.,pressure=None,tv=0.,predicted_gain=None)
        else:
            row = augment(raw, previous.extra(obs,bank,'pool'))
            p,control = distribution(root, arm, run, row, obs, bank)
        rng = np.random.default_rng(t.seed(REVISION,'action',parent,workload,replica,now))
        i,j = np.unravel_index(rng.choice(p.size,p=p.ravel()), p.shape)
        decisions.append((now,time.perf_counter()-started));controls.append(dict(time=now,**control))
        if 50 <= now < 100:
            peak_load.append([sum(x.remaining_cycles for x in engine.tasks.values() if x.target == node)/engine.p.capacity[node] for node in range(6)])
        motion = bank['actions'][j];flight += sum(float(np.hypot(*d)) for d in motion)
        engine.assign(bank['plans'][i]);successes.append(engine.advance(motion)['success'])
    counts = dict(engine.counts)
    assert counts['generated'] == counts.get('success',0)+counts.get('expired',0)
    assert sum(successes) == counts.get('success',0) and engine.audit()['status'] == 'PASS'
    write(out(root)/'episodes'/arm/f'{run}_{workload}_{replica}.json', dict(arm=arm,run=run,parent=parent,
        workload=workload,replica=replica,counts=counts,cohorts=cohort_statistics(engine.finished.values()),
        metrics=engine.summary(),node_quality=quality_details(list(engine.finished.values())),
        peak_node_load_mean=np.mean(peak_load,0).tolist(),decisions=decisions,controls=controls,
        flight_distance_m=flight,fingerprint=tape.fingerprint,successes=successes,audit=engine.audit()))
    return arm


def score_ranking(prediction, actual):
    prediction,actual = np.asarray(prediction),np.asarray(actual)
    i,j = np.triu_indices(actual.size,1)
    delta = actual.ravel()[i]-actual.ravel()[j]
    dscore = prediction.ravel()[i]-prediction.ravel()[j]
    valid = np.abs(delta)>1e-12
    weights = np.abs(delta[valid])
    correct = np.where(np.abs(dscore[valid])<=1e-12,.5,np.sign(dscore[valid])==np.sign(delta[valid]))
    best = prediction >= prediction.max()-1e-12
    return dict(weighted_pairwise_accuracy=float(np.sum(weights*correct)/weights.sum()) if weights.size else None,
        greedy_regret=float(actual.max()-actual[best].mean()),score_range=float(np.ptp(prediction)))


def audit(root, file):
    output = []
    for state in saved(Path(root)/file):
        assert state['run'] >= 0
        name = f'{state["arm"]}/{state["run"]}_{state["parent"]}_{state["workload"]}_{state["time"]}.pkl.gz'
        matrices = saved(closed.out(root)/'audit_matrices'/name)
        row = augment(state['row'], previous.extra(state['obs'],state['bank'],'pool'))
        records = {};ranking = {}
        for arm in ARMS:
            probability,control = distribution(root,arm,state['run'],row,state['obs'],state['bank'])
            records[arm] = dict(control=control, **{kind:calibration_record(probability,reference,q) for kind,reference,q in (
                ('public',matrices['reference'],matrices['forecasts'].mean(0)),
                ('realized',planner_target(matrices['realized'],matrices['prior']),matrices['realized']))})
        for objective in ('reg','rawrank'):
            for mode in MODES:
                prediction = row_scores(row,scorer(str(root),objective,mode,state['run']),mode)
                ranking[f'{objective}_{mode}'] = {kind:score_ranking(prediction,q) for kind,q in (
                    ('public',matrices['forecasts'].mean(0)),('realized',matrices['realized']))}
        output.append(dict(source_arm=state['arm'],run=state['run'],parent=state['parent'],workload=state['workload'],
                           time=state['time'],methods=records,score_ranking=ranking))
    dump(out(root)/'cross_audit'/Path(file).parent.name/Path(file).name,output)
    return len(output)


def worker(payload):
    kind,args = payload;started = time.perf_counter();cpu = time.process_time()
    function = dict(fit=fitting,episode=episode,audit=audit)[kind]
    result = function(*args)
    return dict(kind=kind,cpu_seconds=time.process_time()-cpu,seconds=time.perf_counter()-started,result=result)


def dispatch(root, stage, kind, jobs):
    assert not (val(root)/(stage+'_complete.json')).exists()
    cpus = sorted(os.sched_getaffinity(0));started = time.perf_counter();records = []
    with ProcessPoolExecutor(max_workers=min(len(cpus),len(jobs)),mp_context=mp.get_context('spawn'),
                             initializer=previous.previous.init_worker,initargs=(cpus,)) as pool:
        futures = [pool.submit(worker,(kind,(str(root),)+tuple(args))) for args in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%12 == 0:print(stage,len(records),'/',len(jobs),flush=True)
    elapsed = time.perf_counter()-started;cpu = sum(r['cpu_seconds'] for r in records)
    write(val(root)/(stage+'_complete.json'),dict(status='COMPLETE',tasks=len(jobs),wall_seconds=elapsed,
        worker_cpu_seconds=cpu,equivalent_cpus=cpu/elapsed,utilization=cpu/elapsed/len(cpus),records=records))


def protected(root):
    return previous.protected(root) | {str(p.relative_to(root)):previous.engine_tools.sha(p) for p in previous.out(root).rglob('*') if p.is_file()}


def preflight(root):
    previous.previous.original.require_server()
    state = saved(next((closed.out(root)/'states/taskset_single_joint').glob('*.pkl.gz')))
    first = state[0]
    row = augment(first['row'],previous.extra(first['obs'],first['bank'],'pool'))
    model = initialize(row,hidden=32,seed=0)
    prediction = row_scores(row,model,'joint')
    anchor = policy(row,frozen_actor(str(root),'hard','joint',first['run']),'joint')
    pressure = congestion_pressure(first['obs'],first['bank']['plans'],anchor.sum(1))
    probability,control = controlled_policy(anchor,prediction,pressure,'gated')
    assert np.isfinite(probability).all() and control['tv'] <= .25
    print('public-state score/control preflight passed',flush=True)


def run(root):
    previous.previous.original.require_server();root = Path(root).resolve()
    assert not (out(root)/'protocol.json').exists(), 'immutable experiment already started'
    before = protected(root)
    write(out(root)/'protocol.json',dict(revision=REVISION,protected=before,workloads=list(WORKLOADS),arms=list(ARMS),
        protocol=(REPO_ROOT/'docs/plans/chapter5_dynamic_service_20261028.md').read_text(encoding='utf-8').split('## 2026-11-04 高峰学习原因')[0],
        source={str(p.relative_to(REPO_ROOT)):previous.engine_tools.sha(p) for folder in ('utils','marl_models','scripts/analysis','tests') for p in (REPO_ROOT/folder).rglob('*.py')},
        alpha=.25,pressure_threshold=1.,fits=24,episodes=1824,formal_test_access=False,independent_review=False))
    try:
        dispatch(root,'score_fits','fit',[(objective,mode,run) for objective in ('reg','rawrank') for mode in MODES for run in range(6)])
        models = list((out(root)/'models').glob('*/*.pkl.gz'));assert len(models) == 24
        write(out(root)/'sealed_models.json',dict(files={str(p.relative_to(root)):previous.engine_tools.sha(p) for p in models},
                                               sealed_before_evaluation=True))
        jobs = [('plan_short',r,w,rep) for r in (0,2,4) for w in WORKLOADS for rep in (0,1)]
        jobs += [(arm,r,w,rep) for arm in ARMS for r in range(6) for w in WORKLOADS for rep in (0,1)]
        jobs += [('initial',r,w,rep) for r in (0,2,4) for w in WORKLOADS for rep in (0,1)]
        assert len(jobs) == 1824;dispatch(root,'evaluation','episode',jobs)
        files = [p for p in sorted((closed.out(root)/'states').glob('*/*.pkl.gz')) if p.parent.name not in BASELINES]
        assert len(files) == 120;dispatch(root,'cross_audit','audit',[(str(p.relative_to(root)),) for p in files])
        assert protected(root) == before, 'historical artifacts changed'
        assert all(previous.engine_tools.sha(root/p) == digest for p,digest in load(out(root)/'sealed_models.json')['files'].items())
        from scripts.analysis.review_queued_score_control import review
        review(root)
        write(val(root)/'run_complete.json',dict(status='COMPLETE',episodes=1824,fits=24,cross_roots=720,
            historical_artifacts_unchanged=True,models_unchanged_after_seal=True,independent_review=False,formal_test_access=False))
    except Exception:
        write(val(root)/'run_failed.json',dict(error=traceback.format_exc()));raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,default=REPO_ROOT)
    parser.add_argument('--preflight',action='store_true');args = parser.parse_args()
    preflight(args.task_root) if args.preflight else run(args.task_root)
