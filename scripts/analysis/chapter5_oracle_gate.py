"""Server-only, zero-overhead two-branch oracle on sealed C0 snapshot labels.

Binary routing is shared by the three timing repetitions of each state. The
continuous relaxation is a more optimistic bound, never an online policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    rows = list(rows)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paid_success(method, extra_seconds=0.):
    return statistics.mean(sum(q['latency'] + t + extra_seconds <= q['deadline']
                               for q in method['requests']) for t in method['seconds'])


def aggregate(states, full, route, extra_seconds=0.):
    """Preserve block aggregation; route=1 returns G, route=0 returns full."""
    blocks = sorted({s['identity'][0] for s in states})
    result = []
    for block in blocks:
        totals = Counter()
        for state, x in zip(states, route):
            if state['identity'][0] != block:
                continue
            w = state['weight']
            f, g = state['methods'][full], state['methods']['G']
            totals.update(generated=w * state['generated'], admitted=w * state['admitted'],
                          success=w * ((1-x)*paid_success(f, extra_seconds)+x*paid_success(g, extra_seconds)),
                          full_success=w*f['success'], greedy_success=w*g['success'],
                          seconds=w*((1-x)*f['mean_seconds']+x*g['mean_seconds']+extra_seconds),
                          full_seconds=w*f['mean_seconds'],
                          energy=w*((1-x)*f['energy']+x*g['energy']),
                          full_energy=w*f['energy'], greedy_energy=w*g['energy'],
                          mbs=w*((1-x)*f['mbs']+x*g['mbs']), full_mbs=w*f['mbs'], weight=w)
        result.append(dict(block=block, **totals,
                           dsr=totals['success']/totals['generated'],
                           dsr_difference_pp=100*(totals['success']-totals['full_success'])/totals['generated'],
                           time_ratio=totals['seconds']/totals['full_seconds'],
                           energy_ratio=totals['energy']/totals['full_energy'],
                           energy_ratio_G=totals['energy']/totals['greedy_energy'],
                           mbs_difference_pp=100*(totals['mbs']-totals['full_mbs'])/totals['admitted'],
                           mean_seconds=totals['seconds']/totals['weight']))
    keys = ('dsr', 'dsr_difference_pp', 'time_ratio', 'energy_ratio', 'energy_ratio_G',
            'mbs_difference_pp', 'mean_seconds')
    return dict(blocks=result, **{k: statistics.mean(r[k] for r in result) for k in keys},
                pooled_dsr=sum(r['success'] for r in result)/sum(r['generated'] for r in result),
                greedy_state_fraction=statistics.mean(route))


def coefficients(states, full, extra_seconds=0.):
    """Linear coefficients of the original mean-of-block-rates objectives."""
    import numpy as np
    baseline = aggregate(states, full, [0.] * len(states), extra_seconds)
    by_block = {r['block']: r for r in baseline['blocks']}
    nblocks = len(by_block)
    values = {k: [] for k in ('dsr', 'time', 'energy', 'energy_G')}
    for s in states:
        f, g = s['methods'][full], s['methods']['G']
        b = by_block[s['identity'][0]]
        w = s['weight']/nblocks
        values['dsr'].append(w*100*(paid_success(g, extra_seconds)-paid_success(f, extra_seconds))/b['generated'])
        values['time'].append(w*(g['mean_seconds']-f['mean_seconds'])/b['full_seconds'])
        values['energy'].append(w*(g['energy']-f['energy'])/b['full_energy'])
        values['energy_G'].append(w*(g['energy']-f['energy'])/b['greedy_energy'])
    return baseline, {k: np.asarray(v) for k, v in values.items()}


def solve(states, full, loss_pp, relaxed=False, extra_seconds=0., limit=20.):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    base, co = coefficients(states, full, extra_seconds)
    matrix = np.vstack([-co['dsr'], co['energy'], co['energy_G']])
    upper = np.asarray([loss_pp+base['dsr_difference_pp'],
                        1.05-base['energy_ratio'], 1.05-base['energy_ratio_G']])
    if relaxed:
        opt = linprog(co['time'], A_ub=matrix, b_ub=upper, bounds=(0., 1.), method='highs',
                      options={'time_limit': limit, 'primal_feasibility_tolerance': 1e-9,
                               'dual_feasibility_tolerance': 1e-9})
    else:
        opt = milp(co['time'], integrality=np.ones(len(states)), bounds=Bounds(0., 1.),
                   constraints=LinearConstraint(matrix, -np.inf, upper),
                   options={'time_limit': limit, 'mip_rel_gap': 1e-7})
    record = dict(full=full, allowed_loss_pp=loss_pp, relaxed=relaxed,
                  extra_seconds=extra_seconds, solver_status=int(opt.status),
                  optimal=bool(opt.status == 0), message=str(opt.message))
    if getattr(opt, 'x', None) is None:
        record['feasible_route'] = False
        return record, None
    x = np.asarray(opt.x) if relaxed else np.rint(opt.x)
    residual = float(np.max(matrix @ x-upper))
    metrics = aggregate(states, full, x.tolist(), extra_seconds)
    assert abs(metrics['time_ratio']-(base['time_ratio']+co['time']@x)) < 1e-9
    assert abs(metrics['dsr_difference_pp']-(base['dsr_difference_pp']+co['dsr']@x)) < 1e-9
    assert abs(metrics['energy_ratio']-(base['energy_ratio']+co['energy']@x)) < 1e-9
    record.update(feasible_route=bool(residual <= 1e-7), max_constraint_residual=residual,
                  objective_time_ratio=metrics['time_ratio'], metrics=metrics,
                  joint_target_pass=bool(residual <= 1e-7 and metrics['time_ratio'] <= .8+1e-8))
    if not relaxed:
        record['mip_gap'] = float(opt.mip_gap)
        record['mip_dual_time_bound'] = float(base['time_ratio']+opt.mip_dual_bound)
    return record, x.tolist()


def load_states(root):
    protocol = read(root/'protocol.json')
    manifest = {str(root/'protocol.json'): sha(root/'protocol.json')}
    for name in ('labels_complete.json', 'fixed_complete.json'):
        assert read(root/name)['status'] == 'PASS'
        manifest[str(root/name)] = sha(root/name)
    records, seen = [], set()
    for block in protocol['blocks']:
        path = root/'fixed_labels'/f'{block}.json.gz'
        receipt = path.with_suffix('.receipt.json')
        digest = sha(path)
        assert digest == read(receipt)['sha256']
        manifest[str(path)] = digest
        manifest[str(receipt)] = sha(receipt)
        entries = read(path)
        assert len(entries) == 240
        for s in entries:
            identity = tuple(s['identity'])
            assert identity not in seen and identity[0] == block
            seen.add(identity)
            assert s['weight'] == 25 and s['generated'] >= s['admitted'] >= 0
            assert s['private_truth_used_only_after_public_decisions'] is True
            for method, z in s['methods'].items():
                assert len(z['seconds']) == len(z['success_repeats']) == 3
                assert all(math.isfinite(t) and t >= 0 for t in z['seconds'])
                assert len(z['requests']) == len(z['assignment']) == s['admitted'] == z['admitted']
                assert math.isfinite(z['energy']) and z['energy'] > 0
                successes = [sum(q['latency']+t <= q['deadline'] for q in z['requests']) for t in z['seconds']]
                assert successes == z['success_repeats']
                assert math.isclose(statistics.mean(successes), z['success'], abs_tol=1e-12)
                assert math.isclose(statistics.mean(z['seconds']), z['mean_seconds'], abs_tol=1e-12)
                assert sum(q['target'] == 'mbs' for q in z['requests']) == z['mbs']
            records.append(s)
    assert len(records) == 1200
    assert Counter(s['identity'][2] for s in records) == {'G': 600, 'mixture_price': 600}
    summary = read(root/'analysis/summary.json')
    manifest[str(root/'analysis/summary.json')] = sha(root/'analysis/summary.json')
    checks = []
    for method in ('G', 'B10', 'native_B10'):
        actual = aggregate(records, method, [0.] * len(records))
        for key, mapped in (('dsr', 'dsr'), ('mean_seconds', 'mean_seconds')):
            expected = summary['fixed_snapshot'][method]['metrics'][mapped]['values']
            observed = [z[key] for z in actual['blocks']]
            assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) for a, b in zip(expected, observed))
            checks.append(dict(method=method, metric=key, expected=expected, observed=observed, status='PASS'))
    return protocol, records, manifest, checks


def analyze(args):
    import scipy
    if args.output.exists():
        raise FileExistsError('Preserve prior attempts: '+str(args.output))
    args.output.mkdir(parents=True)
    protocol, states, manifest, checks = load_states(args.input)
    write(args.output/'input_audit.json', dict(status='PASS', n_blocks=5, n_states=1200,
                                              checks=checks, inputs=manifest))
    write(args.output/'execution_protocol.json', dict(
        source_sha256=sha(__file__), plan_sha256=sha(args.plan), inputs=manifest,
        source_protocol=protocol['name'], python=platform.python_version(), scipy=scipy.__version__,
        primary_full='native_B10', sensitivity_full='B10', independent_n=5,
        primary_quality_loss_pp=.10, primary_time_ratio=.80,
        max_energy_ratio=1.05, max_energy_ratio_G=1.05,
        loss_grid_pp=[i/100 for i in range(21)], additional_overhead_seconds=[.0001, .0005, .001],
        interpretation='observed-label oracle with zero gate cost; not an online or closed-loop bound'))
    curves, routes, sensitivities, deltas = [], [], [], []
    for full in ('native_B10', 'B10'):
        for k in range(21):
            for relaxed in (True, False):
                result, route = solve(states, full, k/100, relaxed, limit=args.time_limit)
                curves.append(result)
                if k == 10 and route is not None:
                    for s, x in zip(states, route):
                        routes.append(dict(identity=s['identity'], full=full, relaxed=relaxed, return_G=x))
                print(json.dumps({k: result.get(k) for k in ('full', 'allowed_loss_pp', 'relaxed', 'optimal',
                                                            'objective_time_ratio', 'joint_target_pass')}), flush=True)
        for source in ('G', 'mixture_price'):
            subset = [s for s in states if s['identity'][2] == source]
            for relaxed in (True, False):
                result, _ = solve(subset, full, .10, relaxed, limit=args.time_limit)
                sensitivities.append(dict(source=source, **result))
        for overhead in (.0001, .0005, .001):
            result, _ = solve(states, full, .10, False, extra_seconds=overhead, limit=args.time_limit)
            sensitivities.append(dict(source='balanced', **result))
        for s in states:
            f, g = s['methods'][full], s['methods']['G']
            deltas.append(dict(block=s['identity'][0], workload=s['identity'][1], source=s['identity'][2],
                               step=s['identity'][3], full=full, weight=s['weight'], generated=s['generated'],
                               delta_success=f['success']-g['success'],
                               delta_seconds=f['mean_seconds']-g['mean_seconds'],
                               same_assignment=f['assignment'] == g['assignment']))
    baseline = {full: {label: aggregate(states, full, [x]*len(states)) for label, x in (('all_full', 0.), ('all_G', 1.))}
                for full in ('native_B10', 'B10')}
    distributions = {}
    for full in ('native_B10', 'B10'):
        rows = [r for r in deltas if r['full'] == full]
        distributions[full] = dict(positive=sum(r['delta_success'] > 1e-10 for r in rows),
                                   zero=sum(abs(r['delta_success']) <= 1e-10 for r in rows),
                                   negative=sum(r['delta_success'] < -1e-10 for r in rows),
                                   same_assignment=sum(r['same_assignment'] for r in rows))
    primary = next(r for r in curves if r['full'] == 'native_B10' and r['allowed_loss_pp'] == .1 and not r['relaxed'])
    result = dict(status='COMPLETE_FIXED_STATE_ORACLE', independent_n=5, states=1200,
                  curves=curves, sensitivity=sensitivities, baselines=baseline, distributions=distributions,
                  proceed_to_public_feature_diagnosis=primary.get('joint_target_pass', False),
                  online_policy_validated=False, closed_loop_validated=False, training_started=False)
    write(args.output/'summary.json', result)
    write(args.output/'routes.json', routes)
    csv_write(args.output/'state_deltas.csv', deltas)
    csv_write(args.output/'curve.csv', [dict(full=r['full'], loss_pp=r['allowed_loss_pp'], relaxed=r['relaxed'],
                                           optimal=r['optimal'], time_ratio=r.get('objective_time_ratio'),
                                           dsr_difference_pp=r.get('metrics', {}).get('dsr_difference_pp'),
                                           energy_ratio=r.get('metrics', {}).get('energy_ratio'),
                                           energy_ratio_G=r.get('metrics', {}).get('energy_ratio_G')) for r in curves])
    write(args.output/'completion.json', dict(status='PASS', source_sha256=sha(__file__),
                                             inputs_unchanged=all(sha(p) == h for p, h in manifest.items())))
    assert read(args.output/'completion.json')['inputs_unchanged']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--time-limit', type=float, default=20.)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Scientific analysis executes only on the remote server')
    analyze(args)


if __name__ == '__main__':
    main()
