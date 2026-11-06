"""Opt-in, server-only public-planner distillation development runner.

Prepare/inspect code locally; run physics and fitting only on the server.
No formal phase, old-run mutation, best-checkpoint selection or automatic retry.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_joint_mpo_experiment as t
from utils.queued_joint_mpo import value_features
from utils.queued_closed_loop import paired_summary
from utils.queued_planner_distillation import (
    REVISION, ARMS, BASELINES, specification, validate_workload, validate_episode, planner_target,
    summarize_imitation, check_rows,
)
from marl_models.queued_planner_distillation import initialize, policy, fit

TASK = t.TASK


def out(root):
    return Path(root)/'results'/TASK/REVISION


def val(root):
    return Path(root)/'output'/TASK/'validation'/REVISION


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def model_path(root, arm, run, version):
    return out(root)/'models'/arm/str(run)/f'{version}.pkl.gz'


def actor_mode(arm):
    return 'decomposed' if arm == 'mlp_decomposed' else 'joint'


def initial_model(row, arm, run):
    return initialize(row, hidden=0 if arm == 'linear_joint' else specification()['hidden'],
                      seed=t.seed(REVISION, 'actor', run))


def require_server():
    if os.name != 'posix' or not hasattr(os, 'sched_getaffinity') or not Path('/proc/self/task').is_dir():
        raise RuntimeError('physical work and policy fitting require the Linux server')


def protected(root):
    """Hash old scientific records, without reading reserved formal outcomes."""
    names = ['results/'+TASK+'/queued_closed_loop_formal_20261103/frozen_config.json',
             'results/'+TASK+'/queued_closed_loop_pilot_20261103/analysis.json',
             'utils/queued_closed_loop.py', 'scripts/analysis/queued_closed_loop_experiment.py',
             'marl_models/queued_joint_mpo.py']
    names += [str(t.parent_path(root, b).relative_to(root)) for b in range(3)]
    return {name: t.sha(Path(root)/name) for name in names}


def prepare(root):
    spec = specification()
    path = out(root)/'reservation.json'
    if path.exists():
        if load(path) != spec:
            raise ValueError('existing development reservation differs')
    else:
        t.write(path, spec)
    return spec


def public_row(bank, obs, history):
    # Explicit allowlist: labels, terminal features and private simulation state
    # cannot accidentally become actor inputs by adding fields to the bank.
    return dict(a=bank['a'], m=bank['m'], prior=bank['prior'],
                context=np.arcsinh(value_features(obs, history)))


def short_scores(bank, obs, history, forecast_seed):
    arrivals = t.arrival_prediction(obs, history, forecast_seed, 3)
    return np.array([[t.forecast_window(obs, history, plan, motion, arrivals, 3)['success']
                      for motion in bank['actions']] for plan in bank['plans']], dtype=float)


def episode(root, arm, run_id, parent, workload, replica, version, split):
    """Collect only the current student's visited states; teacher never acts."""
    require_server()
    spec = specification()
    validate_episode(spec, arm, run_id, parent, workload, replica, version, split)
    stem = f'{arm}/{run_id}_{parent}_{workload}_{replica}_{version}'
    path = out(root)/'episodes'/split/(stem+'.json')
    experience_path = out(root)/'experience'/split/(stem+'.pkl.gz')
    if path.exists() or experience_path.exists():
        raise FileExistsError('refuse episode overwrite before any physics')
    proposer = t.parent(root, parent)
    tape = t.RecoveryScenario(workload, 'burst', 'normal', .6)
    engine = t.Engine(t.Protocol(), t.START, tape.ue_positions)
    adapter = t.LearningAdapter()
    model = None if version == 0 or arm in BASELINES else t.read(str(model_path(root, arm, run_id, version)))
    samples, successes, decisions, label_times = [], [], [], []
    initial_fingerprint = None
    distance_m = 0.
    while engine.t < 200 or engine.tasks:
        now = int(engine.t)
        if now > 204:
            raise AssertionError('queue tail failed to drain')
        engine.add(tape.slot(now))
        engine.admit()
        obs = engine.observe()
        if initial_fingerprint is None:
            # Same canonical encoding for all arms; does not enter features.
            import pickle
            initial_fingerprint = hashlib.sha256(pickle.dumps(
                (obs['time'], obs['positions'], obs['ues'], obs['tasks']), 5)).hexdigest()
        begin = time.perf_counter()
        bank = t.bank(proposer, obs, adapter, t.seed(REVISION, 'bank', parent, workload, replica, now))
        row = public_row(bank, obs, adapter.history)
        if model is None:
            model = initial_model(row, arm, run_id)
        probabilities = policy(row, model, actor_mode(arm))
        needs_label = split in ('training', 'holdout') and now in spec['label_times']
        if needs_label or arm == 'plan_short':
            start = time.perf_counter()
            scores = short_scores(bank, obs, adapter.history,
                                  t.seed(REVISION, 'forecast', parent, workload, replica, now))
            label_times.append(time.perf_counter()-start)
            if arm == 'plan_short':
                probabilities = planner_target(scores, row['prior'])
            if needs_label:
                row.update(q=scores, workload=workload, split=split, time=now,
                           parent=parent, run=run_id, version=version)
                samples.append(row)
        rng = np.random.default_rng(t.seed(REVISION, 'action', parent, workload, replica, now))
        choice = int(rng.choice(probabilities.size, p=probabilities.ravel()))
        i, j = np.unravel_index(choice, probabilities.shape)
        plan, motion = bank['plans'][i], bank['actions'][j]
        decisions.append(time.perf_counter()-begin)
        distance_m += sum(float(np.hypot(*v)) for v in motion)
        engine.assign(plan)
        successes.append(engine.advance(motion)['success'])
    counts = dict(engine.counts)
    audit = engine.audit()
    if (engine.tasks or counts['generated'] != counts.get('success', 0)+counts.get('expired', 0)
            or sum(successes) != counts.get('success', 0)):
        raise AssertionError('unsettled/conservation failure')
    if split != 'evaluation' and len(samples) != len(spec['label_times']):
        raise AssertionError('missing uniformly scheduled labels')
    record = dict(arm=arm, run=run_id, parent=parent, workload=workload, replica=replica,
                  version=version, split=split, counts=counts, metrics=engine.summary(), audit=audit,
                  fingerprint=tape.fingerprint, initial_fingerprint=initial_fingerprint,
                  parent_sha=t.sha(t.parent_path(root, parent)), successes=successes,
                  slots=len(decisions), decision_mean=float(np.mean(decisions)), decision_max=max(decisions),
                  over_second=sum(x > 1 for x in decisions), flight_distance_m=distance_m,
                  teacher_states=len(samples), teacher_seconds=sum(label_times), joule_energy=None)
    t.write(path, record)
    if samples:
        t.dump(experience_path, samples)
    return str(path.relative_to(root))


def saved_rows(root, arm, run_id, split, through_version):
    rows = []
    for path in sorted((out(root)/'experience'/split/arm).glob(f'{run_id}_*.pkl.gz')):
        group = t.read(str(path))
        if group[0]['version'] < through_version:
            rows.extend(group)
    return rows


def fit_job(root, arm, run_id, version):
    require_server()
    path = model_path(root, arm, run_id, version)
    if path.exists() or (out(root)/'updates'/arm/f'{run_id}_{version}.json').exists():
        raise FileExistsError('refuse fitted-model overwrite before optimization')
    training = saved_rows(root, arm, run_id, 'training', version)
    heldout = saved_rows(root, arm, run_id, 'holdout', version)
    check_rows(training, heldout)
    spec = specification()
    if arm not in ARMS or version not in range(1, spec['rounds']+1):
        raise ValueError('unsupported fit request')
    expected_run = next(r for r in spec['runs'] if r['id'] == run_id)
    for split, rows in [('training', training), ('holdout', heldout)]:
        expected_workloads = {w for group in expected_run[split][:version] for w in group}
        if {r['workload'] for r in rows} != expected_workloads:
            raise ValueError('missing or foreign aggregation workloads')
        for r in rows:
            validate_episode(spec, arm, run_id, r['parent'], r['workload'], 0, r['version'], split)
        keys = {(r['workload'], r['time']) for r in rows}
        expected_keys = {(w, now) for w in expected_workloads for now in spec['label_times']}
        if len(keys) != len(rows) or keys != expected_keys:
            raise ValueError('duplicated or missing uniformly sampled states')
    previous = (initial_model(training[0], arm, run_id) if version == 1
                else t.read(str(model_path(root, arm, run_id, version-1))))
    mode = actor_mode(arm)
    begin = time.perf_counter()
    model, optimization = fit(training, previous, mode, spec['epochs'], spec['batch_size'],
                              spec['learning_rate'], seed=t.seed(REVISION, 'fit', run_id, version))
    fitting_seconds = time.perf_counter()-begin
    # Seal the fixed final iterate BEFORE computing holdout diagnostics.
    t.dump(path, model)
    digest = t.sha(path)
    report = dict(arm=arm, run=run_id, version=version, optimization=optimization,
                  fitting_seconds=fitting_seconds, model_sha256=digest,
                  training_before=summarize_imitation(training, previous, mode),
                  training_after=summarize_imitation(training, model, mode),
                  holdout_before=summarize_imitation(heldout, previous, mode),
                  holdout_after=summarize_imitation(heldout, model, mode),
                  holdout_used_for_selection=False, checkpoint_rule='Fixed final iterate, no best-checkpoint selection')
    report['parameter_changes'] = {
        branch: {key: float(np.linalg.norm(value-previous['parameters'][branch][key]))
                 for key, value in parameters.items()}
        for branch, parameters in model['parameters'].items()}
    if t.sha(path) != digest:
        raise AssertionError('holdout mutated sealed actor')
    t.write(out(root)/'updates'/arm/f'{run_id}_{version}.json', report)
    return dict(model=str(path.relative_to(root)), sha256=digest)


def initialize_worker(cpus):
    os.sched_setaffinity(0, {cpus[(mp.current_process()._identity[-1]-1) % len(cpus)]})


def timed(job):
    key, kind, args = job
    start, cpu = time.time(), time.process_time()
    result = episode(*args) if kind == 'episode' else fit_job(*args)
    return dict(key=key, kind=kind, result=result, start=start, end=time.time(),
                cpu_seconds=time.process_time()-cpu, pid=os.getpid(),
                affinity=sorted(os.sched_getaffinity(0)), threads=len(os.listdir('/proc/self/task')))


def dispatch(pool, jobs, records):
    futures = {pool.submit(timed, j): j[0] for j in jobs}
    for future in as_completed(futures):
        result = future.result()
        records.append(result)
        if len(records) % 12 == 0:
            print('completed', len(records), 'last', result['key'], flush=True)


def analyze(root):
    spec = specification()
    records = [load(p) for p in (out(root)/'episodes'/'evaluation').glob('*/*.json')]
    if len(records) != spec['maximum_evaluation_episodes']:
        raise ValueError('incomplete development evaluation')
    lookup = {(r['arm'], r['run'], r['parent'], r['workload'], r['replica']): r for r in records}
    if len(lookup) != len(records):
        raise ValueError('duplicate evaluation records')
    success, generated, summary = {}, {}, {}
    fingerprints = {}
    for arm in ARMS+BASELINES:
        a = np.zeros((3, 2, 8, 2))
        g = np.zeros_like(a)
        for run in spec['runs']:
            rid = run['id'] if arm in ARMS else -1
            for wi, w in enumerate(spec['evaluation_workloads']):
                for rep in spec['action_replicas']:
                    r = lookup[arm, rid, run['parent'], w, rep]
                    validate_episode(spec, arm, rid, r['parent'], r['workload'], r['replica'], r['version'], r['split'])
                    c = r['counts']
                    if (r['audit']['status'] != 'PASS' or c['generated'] != c.get('success', 0)+c.get('expired', 0)
                            or sum(r['successes']) != c.get('success', 0)):
                        raise ValueError('invalid development counts')
                    index = (run['parent'], run['repeat'], wi, rep)
                    a[index], g[index] = c.get('success', 0), c['generated']
                    pair = (w, rep)
                    signature = (r['fingerprint'], r['initial_fingerprint'], c['generated'])
                    if pair in fingerprints and signature != fingerprints[pair]:
                        raise ValueError('unpaired development inputs')
                    fingerprints[pair] = signature
        success[arm], generated[arm] = a, g
        own = [r for r in records if r['arm'] == arm]
        summary[arm] = dict(mean_dsr=float(np.mean(a.sum((2, 3))/g.sum((2, 3)))),
                            decision_mean=float(np.mean([r['decision_mean'] for r in own])),
                            flight_distance_mean_m=float(np.mean([r['flight_distance_m'] for r in own])),
                            physical_evaluation_episodes=len(own), joule_energy=None)
    if not all(np.array_equal(generated[a], generated['initial']) for a in ARMS):
        raise ValueError('different evaluation denominators')
    comparisons = {f'{a}--{b}': paired_summary(success[a], success[b], generated[a])
                   for a in ARMS for b in ('initial', 'plan_short')}
    for other in ('mlp_decomposed', 'linear_joint'):
        comparisons['mlp_joint--'+other] = paired_summary(success['mlp_joint'], success[other], generated['mlp_joint'])
    result = dict(status='COMPLETE', revision=REVISION, summary=summary, comparisons=comparisons,
                  formal_test_access=False, formal_go_decision=None,
                  scope='Exploratory development; fixed inherited parents and already-read evaluation workloads. Imitation diagnostics do not establish DSR gains.')
    t.write(out(root)/'analysis.json', result)
    return result


def run(root):
    require_server()
    root = Path(root).resolve()
    spec = prepare(root)
    lock = val(root)/'run.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        _run(root, spec)
    except Exception:
        import traceback
        t.write(val(root)/'run_failed.json', dict(error=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        lock.unlink(missing_ok=True)


def _run(root, spec):
    folder = out(root)
    if (folder/'protocol.json').exists():
        raise FileExistsError('existing scientific run; no restart or overwrite')
    cpus = sorted(os.sched_getaffinity(0))
    before = protected(root)
    from scripts.analysis.queued_closed_loop_experiment import source_hashes
    t.write(folder/'protocol.json', dict(specification=spec, source=source_hashes(),
                                        protected=before, allowed_cpus=cpus, formal_access=False))
    jobs, records = [], []
    start, cpu = time.perf_counter(), time.process_time()
    # Spawn inherits the CLI's explicit single-thread BLAS environment.
    with ProcessPoolExecutor(max_workers=len(cpus), mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(cpus,)) as pool:
        for version in range(1, spec['rounds']+1):
            jobs = []
            for arm in ARMS:
                for r in spec['runs']:
                    for split in ('training', 'holdout'):
                        for w in r[split][version-1]:
                            key = f'{split}:{arm}:{r["id"]}:{w}'
                            jobs.append((key, 'episode', (str(root), arm, r['id'], r['parent'], w, 0, version-1, split)))
            dispatch(pool, jobs, records)
            fits = [(f'fit:{a}:{r["id"]}:{version}', 'fit', (str(root), a, r['id'], version))
                    for a in ARMS for r in spec['runs']]
            dispatch(pool, fits, records)
            t.write(val(root)/f'round_{version}_complete.json', dict(status='COMPLETE'))
        sealed = {str(model_path(root, a, r['id'], spec['rounds']).relative_to(root)):
                  t.sha(model_path(root, a, r['id'], spec['rounds'])) for a in ARMS for r in spec['runs']}
        t.write(val(root)/'final_models_sealed.json', dict(models=sealed, evaluation_started=False))
        jobs = []
        for arm in ARMS+BASELINES:
            runs = spec['runs'] if arm in ARMS else [dict(id=-1, parent=b) for b in range(3)]
            for r in runs:
                for w in spec['evaluation_workloads']:
                    for rep in spec['action_replicas']:
                        key = f'evaluation:{arm}:{r["id"]}:{r["parent"]}:{w}:{rep}'
                        jobs.append((key, 'episode', (str(root), arm, r['id'], r['parent'], w, rep,
                                                     spec['rounds'] if arm in ARMS else 0, 'evaluation')))
        dispatch(pool, jobs, records)
    elapsed = time.perf_counter()-start
    used = sum(r['cpu_seconds'] for r in records)+time.process_time()-cpu
    if sum(r['kind'] == 'episode' for r in records) != spec['maximum_physical_episodes']:
        raise AssertionError('unexpected physical budget')
    if protected(root) != before:
        raise AssertionError('old scientific artifacts changed')
    active = peak = 0
    for _, delta in sorted([(r['start'], 1) for r in records]+[(r['end'], -1) for r in records]):
        active += delta
        peak = max(peak, active)
    t.write(val(root)/'resources.json', dict(records=records, wall_seconds=elapsed,
                                           equivalent_cpus=used/elapsed, utilization=used/elapsed/len(cpus),
                                           allowed_cpus=cpus, peak_active=peak,
                                           note='Only genuine jobs; fitting has 18 parallel actors, no padding work.'))
    analyze(root)
    t.write(val(root)/'run_complete.json', dict(status='COMPLETE', protected=before,
                                               physical_episodes=spec['maximum_physical_episodes'],
                                               formal_test_access=False, independent_review=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--mode', choices=('prepare', 'run', 'analyze'), required=True)
    args = parser.parse_args()
    result = prepare(args.task_root) if args.mode == 'prepare' else analyze(args.task_root) if args.mode == 'analyze' else run(args.task_root)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
