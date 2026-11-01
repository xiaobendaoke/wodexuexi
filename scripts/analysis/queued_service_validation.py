"""Validate and calibrate cross-slot queues on the experiment server, without RL."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone, timedelta
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from environment.queued_service import Engine, Protocol, Task
from utils.queued_service_controllers import Controller, METHODS, assignments
from utils.queued_service_scenarios import QueueScenario, NAMESPACE, PROFILES

TASK = 'chapter5_dynamic_service_20261028'
REVISION = 'queued_protocol_20261031'
SOURCES = ('paths.py', 'tests/__init__.py', 'environment/queued_service.py', 'utils/queued_service_controllers.py',
           'utils/queued_service_scenarios.py', 'utils/dynamic_service_scenarios.py',
           'tests/test_queued_service.py', 'scripts/analysis/queued_service_validation.py')
START = [(100., 100.), (600., 100.), (350., 350.), (100., 600.), (600., 600.)]


def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def source_contract():
    files = {name: sha(REPO_ROOT/name) for name in SOURCES}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return digest, files


def locations(task): return task/'results'/TASK/REVISION, task/'output'/TASK/'validation'/REVISION


def spec_id(spec):
    return f"{spec['profile']}_c{spec['compute']}_b{spec['bandwidth']}_{spec['method']}_w{spec['workload']}_dt{spec['dt']}"


def main_specs():
    return [dict(profile=p, compute=c, bandwidth=b, method=m, workload=w, dt=.1, horizon=200, kind='main')
            for p, c, b, m, w in itertools.product(PROFILES, (.6, 1., 1.4), (.25, .5, 1.), METHODS, range(6))]


def refinement_specs():
    return [dict(profile=p, compute=c, bandwidth=b, method=m, workload=w, dt=dt, horizon=200, kind='fixed_action_refinement')
            for p, (c, b), m, w, dt in itertools.product(('burst', 'overload'), ((.6, .25), (1., 1.)), METHODS, range(2), (.05, .025))]


def frozen_protocol():
    return dict(version=NAMESPACE, physical=asdict(Protocol()), horizon=200, deadlines=[2., 5.],
                profiles=dict(light=10, steady=50, burst='20; 75 during [50,100); then 20', overload=100),
                compute_scales=[.6, 1., 1.4], bandwidth_scales=[.25, .5, 1.], workload_replicates=6,
                methods=list(METHODS), main_jobs=len(main_specs()), fixed_action_refinement_jobs=len(refinement_specs()),
                rolling=dict(candidates=4, route_plans=2, known_task_horizon=2, future_arrivals=False),
                admission='macro boundary, nearest covered, source and target immutable',
                service='orthogonal flow-fair airtime pools; EDF within link and CPU; full-path safety',
                screen=dict(dsr_between=[.05, .95], mean_compute_wait_seconds_above=.01,
                            admitted_crossed_fraction_above=.1, transfer_nonempty_fraction_above=.05,
                            burst_late_backlog_at_most_baseline_multiple=1.2),
                no_rl=True, independent_confirmation=False, automatic_training=False)


def validate(task):
    stream = io.StringIO(); suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_queued_service')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    _, val = locations(task)
    atomic(val/'engine_tests.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
           source_contract=source_contract()[0], log=stream.getvalue(), independent_review=False, time=now()))
    print(stream.getvalue(), flush=True)
    return result.wasSuccessful()


def demonstration(task):
    """Scaled deterministic mechanism example, not representative network data."""
    p = Protocol(capacity=(1., 10., 20.), altitude=0., mbs=(350., 350., 0.), boundary=(0., 700.),
                 separation=200., coverage=100., inter_range=500., speed=15., dt=.1,
                 uplink_bps=10., inter_bps=10., backhaul_bps=10.)
    cases = {}
    for moving, remote in itertools.product((False, True), repeat=2):
        e = Engine(p, [(100., 100.), (550., 100.)], [(100., 100.), (660., 100.)], trace=True)
        e.add([Task(1, 0, 0., 2.7, 5., 16.), Task(2, 1, 0., 2.8, 0., 4.), Task(3, 1, 0., 2.8, 0., 4.)])
        rows = []
        for t in range(3):
            new = e.admit(); e.assign({u: (1 if remote else 0) if u == 1 else 1 for u in new})
            rows.append(e.advance([(0., 0.), (15. if moving else 0., 0.)]))
        cases[f"{'move' if moving else 'hold'}_{'remote' if remote else 'local'}"] = dict(
            summary=e.summary(), rows=rows, events=e.events, audit=e.audit())
    j = {k: v['summary']['counts'].get('success', 0) for k, v in cases.items()}
    interaction = (j['hold_remote']-j['hold_local'])-(j['move_remote']-j['move_local'])
    reversal = (j['hold_remote']-j['hold_local'])*(j['move_remote']-j['move_local']) < 0
    assert reversal and interaction != 0, j
    out, _ = locations(task)
    result = dict(status='PASS_SCALED_MECHANISM_EXAMPLE', source_contract=source_contract()[0], cases=cases,
                  completions=j, interaction=interaction, assignment_ranking_reversal=reversal,
                  limitations='Deliberately scaled fixed rates/cycles, 2 UAVs, 3 tasks; not a realistic gain estimate')
    atomic(out/'mechanism_example.json', result)
    return result


def counterfactual(e, scenario, motion, obs, slots=6):
    """Same full state and exogenous replay; only initial routing/path vary."""
    base = e.counts['success']; result = {}; t0 = int(e.t)
    source_plan = {x['uid']: x['source'] for x in obs['tasks'] if x['source'] is not None and x['target'] is None}
    queue_plan = assignments(obs, 'queue'); zero = [(0., 0.)]*e.n
    for moving, routing in itertools.product((False, True), ('source', 'queue')):
        f = copy.deepcopy(e); rows = []
        for offset in range(slots):
            if offset:
                f.add(scenario.slot(t0+offset)); f.admit()
                plan = {z.uid: z.source for z in f.tasks.values() if z.source is not None and z.target is None}
            else: plan = source_plan if routing == 'source' else queue_plan
            f.assign(plan); rows.append(f.advance(motion if moving and offset == 0 else zero))
        f.audit()
        result[f"{'move' if moving else 'hold'}_{routing}"] = dict(completions=f.counts['success']-base,
            next_state=rows[0], tail_state=rows[-1])
    j = {k: v['completions'] for k, v in result.items()}
    d_hold = j['hold_queue']-j['hold_source']; d_move = j['move_queue']-j['move_source']
    state = dict(time=e.t, positions=e.positions, tasks=[asdict(z) for z in e.tasks.values()])
    return dict(time=t0, cases=result, interaction=d_hold-d_move, reversal=d_hold*d_move < 0,
                assignments_differ=source_plan != queue_plan, motion_nonzero=any(math.hypot(*d) > 1e-8 for d in motion),
                state_sha256=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                replay_fingerprint=scenario.fingerprint, limitations='6-second local intervention, future decisions fixed to source/hover; not optimized joint gain')


def run_episode(spec, task, pilot=False):
    started = time.monotonic(); out, val = locations(task); name = spec_id(spec)
    hb = val/('pilot_progress' if pilot else 'progress')/(name+'.json')
    atomic(hb, dict(stage='running', pid=os.getpid(), cpu=next(iter(os.sched_getaffinity(0))), step=0, updated=now()))
    scenario = QueueScenario(spec['workload'], spec['profile'], spec['horizon'])
    p = Protocol(); p = replace(p, capacity=tuple(c*spec['compute'] for c in p.capacity), dt=spec['dt'],
        uplink_bandwidth=p.uplink_bandwidth*spec['bandwidth'], inter_bandwidth=p.inter_bandwidth*spec['bandwidth'],
        backhaul_bandwidth=p.backhaul_bandwidth*spec['bandwidth'])
    e = Engine(p, START, scenario.ue_positions); controller = Controller(spec['method'])
    replay = None
    if spec['kind'] == 'fixed_action_refinement':
        parent = dict(spec, dt=.1, kind='main'); parent.pop('parent_sha', None)
        parent_path = out/'main'/(spec_id(parent)+'.json'); old = json.loads(parent_path.read_text())
        assert old['source_contract'] == source_contract()[0]
        replay = old['rows']; parent_sha = sha(parent_path)
    rows = []; timings = []; probes = []
    while int(e.t) < scenario.horizon or e.tasks:
        t = int(e.t)
        if t >= scenario.horizon+6: raise RuntimeError('tail did not drain within deadline bound')
        e.add(scenario.slot(t)); e.admit(); obs = e.observe()
        decision_start = time.monotonic()
        if replay is None: route, motion = controller.act(obs)
        else:
            reference = replay[t] if t < len(replay) else dict(assignment={}, motion=[(0., 0.)]*e.n)
            route = {int(k): v for k, v in reference['assignment'].items()}; motion = reference['motion']
            expected = {z.uid for z in e.tasks.values() if z.source is not None and z.target is None}
            assert set(route) == expected, ('fixed replay admissions changed', t)
        timing = time.monotonic()-decision_start; timings.append(timing)
        if (not pilot and replay is None and spec['method'] == 'history_queue' and spec['profile'] == 'burst'
                and spec['workload'] < 2 and (spec['compute'], spec['bandwidth']) in ((.6, .25), (1., 1.)) and t in (75, 100)):
            candidate = next((m for m in controller.candidates(obs) if any(math.hypot(*d) > 1e-8 for d in m)), motion)
            probes.append(counterfactual(e, scenario, candidate, obs))
        e.assign(route); row = e.advance(motion)
        row.update(assignment=route, motion=motion, decision_seconds=timing,
                   admitted_crossed=sum(z.crossed and z.source is not None for z in e.tasks.values()))
        rows.append(row)
        if t % 10 == 0: atomic(hb, dict(stage='running', pid=os.getpid(), cpu=next(iter(os.sched_getaffinity(0))), step=t+1, updated=now()))
    audit = e.audit(); summary = e.summary()
    summary['admitted_crossed_fraction'] = sum(z.crossed and z.source is not None for z in e.finished.values())/max(e.counts['admitted'], 1)
    def backlog(r): return sum(r['stages'][s] for s in ('upload', 'transfer', 'compute'))
    summary['burst_backlog'] = {label: statistics.mean([backlog(r) for r in rows[a:b]])
        for label, a, b in (('before', 30, 50), ('during', 80, 100), ('late', max(100, scenario.horizon-20), scenario.horizon))}
    summary['tail_seconds'] = e.t-scenario.horizon
    result = dict(spec=spec, source_contract=source_contract()[0], fingerprint=scenario.fingerprint,
                  generated_window=scenario.horizon, summary=summary, rows=rows, audit=audit, probes=probes,
                  decision=dict(mean=statistics.mean(timings), p95=sorted(timings)[int(.95*(len(timings)-1))],
                                maximum=max(timings), over_one_second=sum(x>1 for x in timings),
                                forecast_calls=controller.forecast_calls, runtime='full-load diagnostic, no real-time guarantee'),
                  seconds=time.monotonic()-started, pid=os.getpid(), cpu=next(iter(os.sched_getaffinity(0))))
    if replay is not None: result['fixed_action_parent_sha256'] = parent_sha
    directory = 'pilot' if pilot else 'refinement' if replay is not None else 'main'
    path = out/directory/(name+'.json')
    if path.exists(): raise FileExistsError(path)
    atomic(path, result); atomic(hb, dict(stage='completed', pid=os.getpid(), cpu=result['cpu'], step=len(rows), updated=now()))
    return dict(id=name, file=str(path.relative_to(out)), seconds=result['seconds'], cpu=result['cpu'], pid=os.getpid(),
                dsr=summary['dsr'], sha256=sha(path))


def core_order():
    first = []; siblings = []; seen = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((base/'physical_package_id').read_text(), (base/'core_id').read_text())
        (siblings if key in seen else first).append(cpu); seen.add(key)
    # Physical cores first. Delay historically troublesome CPU0 without omitting it.
    return first[8:]+first[:8]+siblings[8:]+siblings[:8]


def init_worker(cpus, val):
    index = multiprocessing.current_process()._identity[-1]-1
    cpu = cpus[index % len(cpus)]; os.sched_setaffinity(0, {cpu})
    atomic(Path(val)/'workers'/f'{os.getpid()}.json', dict(pid=os.getpid(), cpu=cpu, affinity=sorted(os.sched_getaffinity(0)), started=now()))


def cpu_sample():
    result = {}
    for line in Path('/proc/stat').read_text().splitlines():
        parts = line.split()
        if parts[0].startswith('cpu') and parts[0] != 'cpu':
            values = [int(x) for x in parts[1:]]
            result[parts[0]] = [sum(values[:8]), values[3]+values[4]]
    return result


def batch(specs, task, workers, pilot=False):
    out, val = locations(task); cpus = core_order()[:workers]; pending = []
    digest = source_contract()[0]; completed = []; failures = []
    for spec in specs:
        folder = 'pilot' if pilot else 'refinement' if spec['kind'] == 'fixed_action_refinement' else 'main'
        path = out/folder/(spec_id(spec)+'.json')
        if path.exists():
            d = json.loads(path.read_text()); assert d['source_contract'] == digest and d['spec'] == spec
            completed.append(dict(id=spec_id(spec), file=str(path.relative_to(out)), resumed=True, sha256=sha(path)))
        else: pending.append(spec)
    if not pending: return completed
    cpus = cpus[:min(len(cpus), len(pending))]
    active_path = val/('pilot_status.json' if pilot else 'status.json'); last = cpu_sample(); peak = 0
    with ProcessPoolExecutor(max_workers=len(cpus), initializer=init_worker, initargs=(cpus, val)) as pool:
        futures = {pool.submit(run_episode, spec, task, pilot): spec for spec in pending}
        while futures:
            done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                spec = futures.pop(future)
                try: completed.append(future.result())
                except Exception as exc: failures.append(dict(spec=spec, error=repr(exc)))
            active = []
            folder = val/('pilot_progress' if pilot else 'progress')
            for path in folder.glob('*.json'):
                d = json.loads(path.read_text())
                if d['stage'] == 'running': active.append(dict(job=path.stem, **d))
            peak = max(peak, len(active)); current = cpu_sample()
            utilization = {cpu: 1-(v[1]-last[cpu][1])/max(v[0]-last[cpu][0], 1) for cpu, v in current.items()}
            with (val/'resource_samples.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(time=now(), active=len(active), cpu_busy=utilization))+'\n')
            last = current
            atomic(active_path, dict(updated=now(), controller_pid=os.getpid(), source_contract=digest,
                   completed=len(completed), total=len(specs), failures=failures, active=active,
                   requested_workers=workers, worker_cpus=cpus, peak_active=peak, complete=not futures))
    if failures:
        atomic(val/'failures.json', failures); raise RuntimeError(f'{len(failures)} queue jobs failed')
    return completed


def protect(task, val):
    path = val/'protected_before.json'
    if path.exists(): return json.loads(path.read_text())
    files = {}
    for base in (task/'results'/TASK, task/'output'/TASK):
        for p in base.rglob('*'):
            if p.is_file() and REVISION not in p.relative_to(base).parts and p.suffix not in ('.tmp', '.lock'):
                files[str(p.relative_to(task))] = sha(p)
    atomic(path, files); return files


def analyze(task):
    out, val = locations(task); digest = source_contract()[0]; groups = {}; fingerprints = {}; inventory = {}
    probes = []; refinement = []; executed_cpus = set()
    for spec in main_specs()+refinement_specs():
        folder = 'main' if spec['kind'] == 'main' else 'refinement'; path = out/folder/(spec_id(spec)+'.json')
        d = json.loads(path.read_text()); assert d['spec'] == spec and d['source_contract'] == digest
        assert d['audit']['status'] == 'PASS' and d['rows'][-1]['active'] == 0
        c = d['summary']['counts']; assert c['generated'] == c.get('success', 0)+c.get('expired', 0)
        assert c.get('expired', 0) == sum(d['summary']['expiration_stage'].values())
        assert sum(r['success'] for r in d['rows']) == c.get('success', 0)
        assert len(d['rows']) == d['summary']['elapsed']
        key = (spec['profile'], spec['workload'])
        assert key not in fingerprints or fingerprints[key] == d['fingerprint']; fingerprints[key] = d['fingerprint']
        assert math.isclose(d['summary']['dsr'], d['summary']['admission']*d['summary']['conditional'], abs_tol=1e-12)
        inventory[str(path.relative_to(out))] = sha(path)
        if folder == 'main':
            executed_cpus.add(d['cpu'])
            key = (spec['profile'], spec['compute'], spec['bandwidth'], spec['method'])
            groups.setdefault(key, []).append(d); probes.extend(d['probes'])
        else:
            parent = dict(spec, dt=.1, kind='main'); parent_path = out/'main'/(spec_id(parent)+'.json')
            assert d['fixed_action_parent_sha256'] == sha(parent_path)
            original = json.loads(parent_path.read_text())
            refinement.append(dict(spec=spec, dsr_change=d['summary']['dsr']-original['summary']['dsr'],
                success_change=c.get('success', 0)-original['summary']['counts'].get('success', 0)))
    summary = []
    for key, ds in sorted(groups.items()):
        assert len(ds) == 6
        p, c, b, m = key
        result = dict(profile=p, compute=c, bandwidth=b, method=m, dsr=statistics.mean(d['summary']['dsr'] for d in ds),
            admission=statistics.mean(d['summary']['admission'] for d in ds),
            conditional=statistics.mean(d['summary']['conditional'] for d in ds),
            compute_wait=statistics.mean(d['summary']['mean_stage_time']['compute_wait'] for d in ds),
            crossed=statistics.mean(d['summary']['service_crossed_fraction'] for d in ds),
            transfer_nonempty=statistics.mean(d['summary']['nonempty_fraction']['transfer'] for d in ds),
            cpu_nonempty=statistics.mean(d['summary']['nonempty_fraction']['compute'] for d in ds),
            backlog={s: statistics.mean(d['summary']['burst_backlog'][s] for d in ds) for s in ('before', 'during', 'late')},
            mbs_fraction=statistics.mean(d['summary']['mbs_fraction'] for d in ds),
            decision_mean=statistics.mean(d['decision']['mean'] for d in ds),
            decision_over_one_second=sum(d['decision']['over_one_second'] for d in ds),
            dsr_by_workload=[d['summary']['dsr'] for d in sorted(ds, key=lambda d: d['spec']['workload'])])
        result['queue_screen'] = (.05 < result['dsr'] < .95 and result['compute_wait'] > .01 and
                                  result['crossed'] > .1 and result['transfer_nonempty'] > .05)
        if p == 'burst': result['recovery_screen'] = result['backlog']['late'] <= 1.2*max(result['backlog']['before'], 1.)
        summary.append(result)
    workers = [json.loads(p.read_text()) for p in (val/'workers').glob('*.json')]
    old = json.loads((val/'protected_before.json').read_text()); changed = [name for name, h in old.items() if not (task/name).exists() or sha(task/name) != h]
    result = dict(status='PASS_COMPLETED_ENGINE_AND_CALIBRATION' if not changed else 'FAIL_OLD_FILES_CHANGED',
                  source_contract=digest, summary=summary, raw_sha256=inventory, main_jobs=len(main_specs()),
                  refinement_jobs=len(refinement_specs()), representative_probes=probes,
                  representative_interactions_nonzero=sum(p['interaction'] != 0 for p in probes),
                  representative_rank_reversals=sum(p['reversal'] for p in probes),
                  refinement=refinement, max_abs_fixed_action_dsr_change=max(abs(x['dsr_change']) for x in refinement),
                  protected_old_files=len(old), changed_old_files=changed,
                  actual_cpus=sorted({w['cpu'] for w in workers}), worker_processes=len(workers),
                  executed_main_cpus=sorted(executed_cpus),
                  scientific_limitations='Development calibration, 6 workload repetitions, no RL or independent training repetitions; local probes are not joint optimization gains; fixed-action convergence checks do not test retrained policies',
                  automatic_training=False, completed=now())
    atomic(out/'analysis.json', result)
    atomic(val/'completion.json', {k: v for k, v in result.items() if k not in ('summary', 'raw_sha256', 'representative_probes', 'refinement')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'demo', 'pilot', 'run', 'analyze', 'package'))
    parser.add_argument('--task', type=Path, default=REPO_ROOT)
    parser.add_argument('--workers', type=int, default=256)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); task = args.task.resolve(); out, val = locations(task)
    if args.command == 'package':
        if args.output is None: parser.error('--output required')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED) as z:
            for name in SOURCES: z.write(REPO_ROOT/name, name)
        print(json.dumps(dict(source_contract=source_contract()[0], zip_sha256=sha(args.output)), indent=2)); return
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if args.command == 'validate': raise SystemExit(0 if validate(task) else 1)
    if args.command == 'demo': demonstration(task); return
    if args.command == 'analyze': analyze(task); return
    import fcntl
    val.mkdir(parents=True, exist_ok=True)
    lock = (val/'controller.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tests = json.loads((val/'engine_tests.json').read_text()); assert tests['status'] == 'PASS' and tests['source_contract'] == source_contract()[0]
    protect(task, val)
    if args.command == 'pilot':
        specs = [dict(profile='burst', compute=c, bandwidth=b, method=m, workload=0, dt=.1, horizon=120, kind='pilot')
                 for (c, b), m in itertools.product(((1., 1.), (.6, .25)), METHODS)]
        result = batch(specs, task, min(args.workers, len(specs)), pilot=True)
        atomic(val/'pilot_summary.json', dict(jobs=result, completed=now(), scientific_gate=False)); return
    digest, files = source_contract(); proto = dict(frozen_protocol(), source_contract=digest, source_files=files)
    if (out/'protocol.json').exists(): assert json.loads((out/'protocol.json').read_text()) == proto
    else: atomic(out/'protocol.json', proto)
    atomic(val/'launch.json', dict(started=now(), pid=os.getpid(), cpus=core_order(), workers=args.workers,
        main_jobs=len(main_specs()), refinement_jobs=len(refinement_specs()), source_contract=digest, source=str(REPO_ROOT), automatic_training=False))
    demonstration(task)
    batch(main_specs(), task, args.workers)
    batch(refinement_specs(), task, args.workers)
    analyze(task)


if __name__ == '__main__': main()
