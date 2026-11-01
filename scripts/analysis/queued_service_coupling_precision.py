"""Fixed-state, fixed-intervention refinement of all observed rank reversals."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from environment.queued_service import Engine, Protocol
from scripts.analysis.queued_service_coupling import cooperative_plan, motion_sequence, name
from scripts.analysis.queued_service_validation import START, atomic, locations, spec_id, source_contract, sha
from utils.queued_service_scenarios import QueueScenario


def replay(args):
    original, dt, task = args; spec = original['spec']; out, _ = locations(task)
    parent_spec = dict(spec, method='history_queue', dt=.1); parent_spec.pop('time')
    parent = json.loads((out/'main'/(spec_id(parent_spec)+'.json')).read_text())
    tape = QueueScenario(spec['workload'], spec['profile']); p = Protocol()
    p = replace(p, capacity=tuple(x*spec['compute'] for x in p.capacity),
                uplink_bandwidth=p.uplink_bandwidth*spec['bandwidth'], inter_bandwidth=p.inter_bandwidth*spec['bandwidth'],
                backhaul_bandwidth=p.backhaul_bandwidth*spec['bandwidth'])
    e = Engine(p, START, tape.ue_positions)
    for t in range(spec['time']):
        e.add(tape.slot(t)); e.admit(); row = parent['rows'][t]
        e.assign({int(k): v for k, v in row['assignment'].items()}); got = e.advance(row['motion'])
        assert all(got[k] == row[k] for k in ('generated', 'completed', 'expired', 'admitted', 'active'))
    e.add(tape.slot(spec['time'])); e.admit(); obs = e.observe()
    local = {x['uid']: x['source'] for x in obs['tasks'] if x['source'] is not None and x['target'] is None}
    coop = cooperative_plan(obs); sequence = motion_sequence(e, parent['rows'][spec['time']]['motion'])
    initial = e.counts['success']; cases = {}; zero = [(0., 0.)]*e.n
    # Preserve the identical replayed state; refine only the intervention dynamics.
    e.p = replace(e.p, dt=dt)
    for moving, routing in itertools.product((False, True), ('local', 'cooperative')):
        f = copy.deepcopy(e); offset = 0
        while offset < 6 or f.tasks:
            assert offset <= 12
            if offset:
                f.add(tape.slot(spec['time']+offset) if offset < 6 else ()); new = f.admit()
                plan = {uid: f.tasks[uid].source for uid in new}
            else: plan = local if routing == 'local' else coop
            f.assign(plan); f.advance(sequence[offset] if moving and offset < 6 else zero); offset += 1
        f.audit(); cases[f"{'move' if moving else 'hold'}_{routing}"] = f.counts['success']-initial
    baseline = {k: v['completed'] for k, v in original['cases'].items()}
    result = dict(spec=spec, dt=dt, source_contract=source_contract()[0], original_counts=baseline, counts=cases,
                  identical=cases == baseline, script_sha256=sha(Path(__file__)))
    atomic(out/'coupling_precision'/(name(spec)+f'_dt{dt}.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--task', type=Path, required=True)
    task = parser.parse_args().task.resolve(); out, val = locations(task)
    originals = [json.loads(p.read_text()) for p in (out/'coupling_diagnostic').glob('*.json') if p.name != 'analysis.json']
    reversals = [x for x in originals if x['reversal']]
    jobs = [(x, dt, task) for x in reversals for dt in (.1, .05, .025)]
    manifest = dict(source_contract=source_contract()[0], script_sha256=sha(Path(__file__)),
                    selection='All observed ranking-reversal cases; numerical verification, not independent confirmation',
                    cases=[x['spec'] for x in reversals], steps=[.1, .05, .025])
    atomic(val/'coupling_precision/manifest.json', manifest)
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool: results = list(pool.map(replay, jobs))
    summary = dict(status='PASS_FIXED_INTERVENTION_PRECISION' if all(x['identical'] for x in results) else 'NUMERICAL_SENSITIVITY_FOUND',
                   cases=len(reversals), runs=len(results), all_counts_identical=all(x['identical'] for x in results),
                   results=results, source_contract=source_contract()[0], script_sha256=sha(Path(__file__)))
    atomic(out/'coupling_precision/analysis.json', summary); atomic(val/'coupling_precision/completion.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}), flush=True)


if __name__ == '__main__': main()
