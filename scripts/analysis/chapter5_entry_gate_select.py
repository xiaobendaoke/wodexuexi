"""Block-held-out threshold development using existing labels only."""
import argparse
import csv
from pathlib import Path
import statistics
import sys

import numpy as np

from chapter5_oracle_gate import load_states, sha, write, csv_write

N_GRID = (0, 8, 12, 16, 20, 24, 28, 32, 40, 64)
Q_GRID = (0., .25, .4, .5, .6, .7, .8, .9, 1., 1.01)


def candidates(mode):
    for n0 in N_GRID if mode in ('A', 'C') else (0,):
        for q0 in Q_GRID if mode in ('B', 'C') else (1.01,):
            yield dict(mode=mode, n0=n0, q0=q0, fallback_full=False)


def mask(rule, n, q):
    if rule.get('fallback_full'):
        return np.ones(len(n), dtype=bool)
    if rule['mode'] == 'A':
        return n >= rule['n0']
    if rule['mode'] == 'B':
        return q < rule['q0']
    return (n >= rule['n0']) & (q < rule['q0'])


def arrays(states, feature_rows):
    features = {(int(r['block']), int(r['workload']), r['source'], int(r['step'])): r for r in feature_rows}
    assert len(features) == len(states)
    out = {k: [] for k in ('block', 'generated', 'admitted', 'Sf', 'Sg', 'Tf', 'Tg', 'Ef', 'Eg', 'Mf', 'Mg', 'n', 'q')}
    for s in states:
        f, g = s['methods']['native_B10'], s['methods']['G']
        z = features[tuple(s['identity'])]
        values = dict(block=s['identity'][0], generated=s['generated'], admitted=s['admitted'],
                      Sf=f['success'], Sg=g['success'], Tf=f['mean_seconds'], Tg=g['mean_seconds'],
                      Ef=f['energy'], Eg=g['energy'], Mf=f['mbs'], Mg=g['mbs'],
                      n=float(z['batch_size']), q=float(z['G_pred_success_fraction']))
        for k, v in values.items():
            out[k].append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def metrics(a, rule, blocks):
    entered = mask(rule, a['n'], a['q'])
    rows = []
    for block in blocks:
        idx = a['block'] == block
        f = entered[idx]
        selected = lambda full, greedy: np.where(f, a[full][idx], a[greedy][idx]).sum()
        rows.append(dict(block=int(block),
            dsr_difference_pp=100*(selected('Sf', 'Sg')-a['Sf'][idx].sum())/a['generated'][idx].sum(),
            time_ratio=selected('Tf', 'Tg')/a['Tf'][idx].sum(),
            energy_ratio=selected('Ef', 'Eg')/a['Ef'][idx].sum(),
            energy_ratio_G=selected('Ef', 'Eg')/a['Eg'][idx].sum(),
            mbs_difference_pp=100*(selected('Mf', 'Mg')-a['Mf'][idx].sum())/a['admitted'][idx].sum(),
            entered_fraction=float(np.mean(f))))
    keys = ('dsr_difference_pp', 'time_ratio', 'energy_ratio', 'energy_ratio_G', 'mbs_difference_pp', 'entered_fraction')
    return dict(blocks=rows, **{k: statistics.mean(z[k] for z in rows) for k in keys})


def choose(a, mode, blocks):
    grid = []
    for rule in candidates(mode):
        m = metrics(a, rule, blocks)
        parts = dict(quality=m['dsr_difference_pp'] >= -.05,
                     block_quality=all(z['dsr_difference_pp'] >= -.10 for z in m['blocks']),
                     time=m['time_ratio'] <= .80, energy=m['energy_ratio'] <= 1.05,
                     energy_G=m['energy_ratio_G'] <= 1.05, mbs=m['mbs_difference_pp'] <= .30)
        parts = {k: bool(v) for k, v in parts.items()}
        grid.append(dict(rule=rule, metrics=m, screen=parts, eligible=all(parts.values())))
    eligible = [z for z in grid if z['eligible']]
    if not eligible:
        return dict(mode=mode, fallback_full=True, reason='No threshold meets frozen training-fold screen'), grid
    best = min(eligible, key=lambda z: (-z['metrics']['dsr_difference_pp'], z['metrics']['time_ratio'],
                                      z['rule']['n0'], z['rule']['q0']))
    return best['rule'], grid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--c0', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Threshold development executes on server')
    if a.output.exists():
        raise FileExistsError('Preserve previous threshold attempt')
    a.output.mkdir(parents=True)
    protocol, states, inputs, checks = load_states(a.c0)
    with a.features.open() as f:
        data = arrays(states, list(csv.DictReader(f)))
    inputs[str(a.features)] = sha(a.features)
    folds, all_grid, heldout_rows = {}, {}, []
    for heldout in protocol['blocks']:
        training = [b for b in protocol['blocks'] if b != heldout]
        fold = dict(training_blocks=training, heldout_block=heldout, rules={}, evaluations={})
        assert heldout not in training
        for mode in ('A', 'B', 'C'):
            rule, grid = choose(data, mode, training)
            fold['rules'][mode] = rule
            fold['evaluations'][mode] = metrics(data, rule, [heldout])
            all_grid[f'{heldout}_{mode}'] = grid
            heldout_rows.append(dict(heldout_block=heldout, **rule,
                                     **{k: v for k, v in fold['evaluations'][mode].items() if k != 'blocks'}))
        folds[str(heldout)] = fold
    pooled = {}
    for mode in ('A', 'B', 'C'):
        pooled[mode], all_grid[f'pooled_{mode}'] = choose(data, mode, protocol['blocks'])
    result = dict(status='FROZEN_THRESHOLD_DEVELOPMENT', folds=folds, pooled_development_rules=pooled,
                  all_folds_fallback=all(r['fallback_full'] for f in folds.values() for r in f['rules'].values()),
                  heldout_is_development_only=True, source_sha256=sha(__file__), plan_sha256=sha(a.plan),
                  inputs=inputs, initial_label_reconstruction_checks=checks,
                  feature_hypotheses_previously_informed_by_all_five_blocks=True)
    write(a.output/'rules.json', result)
    write(a.output/'threshold_grid.json', all_grid)
    csv_write(a.output/'heldout_proxy_metrics.csv', heldout_rows)
    print(__import__('json').dumps(dict(all_folds_fallback=result['all_folds_fallback'],
        rules={k: v['rules'] for k, v in folds.items()}, pooled=pooled)), flush=True)


if __name__ == '__main__':
    main()
