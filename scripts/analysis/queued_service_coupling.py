"""Post-calibration, explicitly distinct routing/path interventions; no RL.

Replays frozen history_queue trajectories to representative states. Future
arrivals are simulation input shared by arms, never input to action selection.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import replace, asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from environment.queued_service import Engine, distance, safe_path
from utils.queued_service_scenarios import QueueScenario, PROFILES
from scripts.analysis.queued_service_validation import (
    START, TASK, REVISION, atomic, core_order, init_worker, locations, now, sha, source_contract, spec_id)


def specs():
    return [dict(profile=p, compute=c, bandwidth=b, workload=w, time=t)
            for p, (c, b), w, t in itertools.product(PROFILES, ((1., 1.), (.6, .25), (.6, 1.)), range(6), (40, 75, 100, 150))]


def name(s): return f"{s['profile']}_c{s['compute']}_b{s['bandwidth']}_w{s['workload']}_t{s['time']}"


def cooperative_plan(obs):
    p = obs['protocol']; n = len(obs['positions']); load = [0.]*(n+1); plan = {}
    for x in obs['tasks']:
        if x['target'] is not None: load[x['target']] += x['remaining_cycles']
    for x in obs['tasks']:
        if x['source'] is None or x['target'] is not None: continue
        source = x['source']
        nodes = [i for i in range(n) if i != source and distance(obs['positions'][source], obs['positions'][i]) <= p.inter_range]
        node = min(nodes, key=lambda i: ((load[i]+x['cycles'])/p.capacity[i], i)) if nodes else source
        plan[x['uid']] = node; load[node] += x['cycles']
    return plan


def motion_sequence(e, recorded):
    """Commit to six feasible path segments using only the initial state."""
    zero = [(0., 0.)]*e.n; first = recorded
    if not any(math.hypot(*d) > 1e-6 for d in first):
        alternatives = []
        for i in range(e.n):
            for dx, dy in ((15., 0.), (-15., 0.), (0., 15.), (0., -15.)):
                d = list(zero); d[i] = (dx, dy)
                if safe_path(e.positions, d, e.p): alternatives.append(d)
        first = alternatives[0] if alternatives else zero
    sequence = []; positions = e.positions
    for _ in range(6):
        motion = zero
        for scale in (1., .5, .25):
            candidate = [(dx*scale, dy*scale) for dx, dy in first]
            if safe_path(positions, candidate, e.p): motion = candidate; break
        sequence.append(motion)
        positions = [(a[0]+d[0], a[1]+d[1]) for a, d in zip(positions, motion)]
    return sequence


def run_case(s, task):
    started = time.monotonic(); out, val = locations(task)
    base_spec = dict(s, method='history_queue', dt=.1); base_spec.pop('time')
    path = out/'main'/(spec_id(base_spec)+'.json'); base = json.loads(path.read_text())
    assert base['source_contract'] == source_contract()[0]
    scenario = QueueScenario(s['workload'], s['profile'])
    from environment.queued_service import Protocol
    p = Protocol(); p = replace(p, capacity=tuple(x*s['compute'] for x in p.capacity),
        uplink_bandwidth=p.uplink_bandwidth*s['bandwidth'], inter_bandwidth=p.inter_bandwidth*s['bandwidth'],
        backhaul_bandwidth=p.backhaul_bandwidth*s['bandwidth'])
    e = Engine(p, START, scenario.ue_positions)
    for t in range(s['time']):
        e.add(scenario.slot(t)); e.admit(); row = base['rows'][t]
        e.assign({int(k): v for k, v in row['assignment'].items()}); got = e.advance(row['motion'])
        assert all(got[k] == row[k] for k in ('generated', 'completed', 'expired', 'admitted', 'active'))
    e.add(scenario.slot(s['time'])); e.admit(); e.audit(); obs = e.observe()
    local = {x['uid']: x['source'] for x in obs['tasks'] if x['source'] is not None and x['target'] is None}
    coop = cooperative_plan(obs); sequence = motion_sequence(e, base['rows'][s['time']]['motion'])
    changed = sum(local[u] != coop[u] for u in local); zero = [(0., 0.)]*e.n
    initial = e.counts['success']; initial_tasks = len(e.tasks); cases = {}
    for moving, route in itertools.product((False, True), ('local', 'cooperative')):
        f = copy.deepcopy(e); offset = 0; future_tasks = 0; first_state = None
        while offset < 6 or f.tasks:
            if offset > 12: raise RuntimeError('intervention tail failed to drain')
            if offset:
                arrivals = scenario.slot(s['time']+offset) if offset < 6 else ()
                future_tasks += len(arrivals); f.add(arrivals); new = f.admit()
                assignment = {u: f.tasks[u].source for u in new}
            else: assignment = local if route == 'local' else coop
            f.assign(assignment); motion = sequence[offset] if moving and offset < 6 else zero
            state = f.advance(motion)
            if offset == 0: first_state = state
            offset += 1
        f.audit(); completed = f.counts['success']-initial
        denominator = initial_tasks+future_tasks
        cases[f"{'move' if moving else 'hold'}_{route}"] = dict(completed=completed, at_risk=denominator,
            dsr=completed/max(denominator, 1), next_state=first_state, drain_seconds=offset,
            total_served_cycles=f.served_cycles, end_positions=f.positions)
    values = {k: x['completed'] for k, x in cases.items()}
    d_hold = values['hold_cooperative']-values['hold_local']
    d_move = values['move_cooperative']-values['move_local']
    assert len({x['at_risk'] for x in cases.values()}) == 1
    result = dict(spec=s, source_contract=source_contract()[0], diagnostic_script_sha256=sha(Path(__file__)),
        parent_raw_sha256=sha(path), replay_counts_exact=True,
        initial_state_sha256=hashlib.sha256(json.dumps(dict(time=e.t, positions=e.positions,
            tasks=[asdict(z) for z in e.tasks.values()]), sort_keys=True).encode()).hexdigest(),
        initial_changed_assignments=changed, motion_length=sum(math.hypot(*d) for step in sequence for d in step),
        initial_new_tasks=len(local), cases=cases, interaction=d_hold-d_move, reversal=d_hold*d_move < 0,
        eligible=changed > 0 and any(math.hypot(*d) > 1e-6 for step in sequence for d in step),
        seconds=time.monotonic()-started, cpu=next(iter(os.sched_getaffinity(0))), pid=os.getpid())
    destination = out/'coupling_diagnostic'/(name(s)+'.json')
    if destination.exists(): raise FileExistsError(destination)
    atomic(destination, result)
    return result


def run(task, workers):
    out, val = locations(task); val = val/'coupling_diagnostic'; val.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (val/'controller.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    digest = sha(Path(__file__)); schedule = specs()
    manifest = dict(source_contract=source_contract()[0], diagnostic_script_sha256=digest, cases=schedule,
        design='Post-calibration diagnosis, all 4 profiles x 3 resource pairs x 6 workloads x 4 preset times; no selection by outcome',
        interventions='Initial new tasks local vs all feasible cooperative; hold vs six-segment committed trajectory; next five arrival batches common, source routing thereafter; drain to deadlines',
        full_system_performance=False, automatic_training=False)
    manifest_path = val/'manifest.json'
    if manifest_path.exists(): assert json.loads(manifest_path.read_text()) == manifest
    else: atomic(manifest_path, manifest)
    done = []; pending = []
    for s in schedule:
        path = out/'coupling_diagnostic'/(name(s)+'.json')
        if path.exists():
            r = json.loads(path.read_text()); assert r['diagnostic_script_sha256'] == digest; done.append(r)
        else: pending.append(s)
    cpus = core_order()[:min(workers, len(pending))]
    if pending:
        with ProcessPoolExecutor(max_workers=len(cpus), initializer=init_worker, initargs=(cpus, val)) as pool:
            futures = {pool.submit(run_case, s, task): s for s in pending}
            for future in as_completed(futures):
                done.append(future.result())
                atomic(val/'status.json', dict(updated=now(), completed=len(done), total=len(schedule), workers=len(cpus),
                    source_contract=source_contract()[0], diagnostic_script_sha256=digest))
    eligible = [x for x in done if x['eligible']]
    summary = dict(status='PASS_COMPLETED_DISTINCT_INTERVENTIONS', cases=len(done), eligible=len(eligible),
        nonzero_interactions=sum(x['interaction'] != 0 for x in eligible), rank_reversals=sum(x['reversal'] for x in eligible),
        max_absolute_interaction=max(abs(x['interaction']) for x in eligible),
        actual_cpus=sorted({x['cpu'] for x in done}), completed=now(), source_contract=source_contract()[0],
        diagnostic_script_sha256=digest, raw_sha256={name(x['spec'])+'.json': sha(out/'coupling_diagnostic'/(name(x['spec'])+'.json')) for x in done},
        limitations='Post-calibration local intervention on reused development workloads; deliberately different routing, committed path, finite arrival window and drain; not optimized joint-policy or RL performance')
    atomic(out/'coupling_diagnostic/analysis.json', summary); atomic(val/'completion.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('raw_sha256', 'actual_cpus')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=256); args = parser.parse_args(); run(args.task.resolve(), args.workers)
