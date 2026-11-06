"""Recompute the ordered experiment from immutable counts and audit records."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_learning_causes as experiment
from utils.queued_taskset_closed_loop import calibration_summary


def paired(a, da, b, db, draws=4000):
    a,da,b,db = (np.asarray(x, float) for x in (a,da,b,db))
    assert a.shape == da.shape == b.shape == db.shape == (3, 2, 8, 2)
    assert all(np.isfinite(x).all() for x in (a,da,b,db)) and (da > 0).all() and (db > 0).all()
    rng = np.random.default_rng(2026110418)
    ws = rng.integers(0, 8, (draws, 8))
    rs = rng.integers(0, 2, (draws, 8, 2))
    boot = np.zeros(draws)
    effects = a.sum((2,3))/da.sum((2,3))-b.sum((2,3))/db.sum((2,3))
    for parent in range(3):
        ss = rng.integers(0, 2, (draws, 2))
        sums = [x[parent][ss[:, :, None, None], ws[:, None, :, None], rs[:, None, :, :]].sum((2,3)) for x in (a,da,b,db)]
        boot += (sums[0]/sums[1]-sums[2]/sums[3]).mean(1)/3
    return dict(mean=float(effects.mean()), by_parent=effects.mean(1).tolist(),
        ci95=np.quantile(boot, [.025, .975]).tolist(),
        simultaneous97_5=np.quantile(boot, [.0125, .9875]).tolist(), draws=draws,
        scope='Paired workload/action blocks, repeats resampled within three fixed inherited parents.')


def review(root):
    root = Path(root);out = experiment.out(root)
    records = [experiment.load(p) for p in sorted((out/'episodes').glob('*/*.json'))]
    assert len(records) == 1632
    lookup = {(x['arm'],x['run'],x['workload'],x['replica']):x for x in records}
    assert len(lookup) == 1632
    matrices = {};summary = {};fingerprints = {}
    arms = experiment.BASE_ARMS+experiment.ADDED_ARMS+('initial','plan_short')
    for arm in arms:
        own = [r for r in records if r['arm'] == arm]
        baseline = arm in ('initial','plan_short')
        assert len(own) == (48 if baseline else 96)
        matrices[arm] = {phase:{k:np.zeros((3,2,8,2)) for k in ('generated','admitted','success')}
                         for phase in ('all','pre','peak','post')}
        for run in range(6):
            for wi,w in enumerate(experiment.TEST_WORKLOADS):
                for replica in (0,1):
                    row = lookup[arm, run//2*2 if baseline else run, w, replica]
                    assert row['audit']['status'] == 'PASS'
                    assert row['counts']['generated'] == row['counts'].get('success',0)+row['counts'].get('expired',0)
                    assert sum(row['successes']) == row['counts'].get('success',0)
                    pair = (w, replica)
                    fp = (row['fingerprint'], row['counts']['generated'])
                    assert pair not in fingerprints or fingerprints[pair] == fp
                    fingerprints[pair] = fp
                    for phase in matrices[arm]:
                        counts = row['counts'] if phase == 'all' else row['cohorts']['arrival'][phase]
                        for k in matrices[arm][phase]:matrices[arm][phase][k][run//2,run%2,wi,replica] = counts.get(k,0)
        result = {}
        for phase in ('all','pre','peak','post'):
            totals = {k:sum((r['counts'] if phase == 'all' else r['cohorts']['arrival'][phase]).get(k,0) for r in own)
                      for k in ('generated','admitted','success')}
            g,a,s = (totals[k] for k in ('generated','admitted','success'))
            result[phase] = dict(counts=totals, dsr=s/g, admission=a/g, conditional=s/a)
        result['decision_mean'] = sum(sum(t for _,t in r['decisions']) for r in own)/sum(len(r['decisions']) for r in own)
        result['decision_over_second'] = sum(sum(t > 1 for _,t in r['decisions']) for r in own)/sum(len(r['decisions']) for r in own)
        result['peak_decision_mean'] = np.mean([t for r in own for now,t in r['decisions'] if 50 <= now < 100]).item()
        result['peak_node_load_mean'] = np.mean([r['peak_node_load_mean'] for r in own],0).tolist()
        result['peak_node_quality'] = []
        for node in range(6):
            xs = [r['node_quality']['peak'][node] for r in own]
            totals = {k:sum(x[k] for x in xs) for k in xs[0] if k != 'node'}
            totals['node'] = node
            totals['compute_entry_slack_mean'] = totals['entry_slack_sum']/max(totals['compute_entries'],1)
            totals['wasted_cycle_fraction'] = totals['wasted_cycles']/max(totals['served_cycles'],1)
            result['peak_node_quality'].append(totals)
        result['peak_expiration_stage'] = {k:sum(r['cohorts']['arrival']['peak']['expiration_stage'][k] for r in own) for k in ('waiting','upload','transfer','compute')}
        result['physical_episodes'] = len(own)
        summary[arm] = result
    selected = experiment.load(out/'selection.json')['selected']
    pairs = set()
    for feature in experiment.FEATURES:
        for objective in experiment.OBJECTIVES:
            for mode in experiment.MODES:
                arm = f'{feature}_{objective}_{mode}'
                for other in (f'pool_{objective}_{mode}', f'{feature}_hard_{mode}', f'pool_hard_{mode}'):
                    if arm != other:pairs.add((arm,other))
            pairs.add((f'{feature}_{objective}_joint',f'{feature}_{objective}_decomposed'))
    for mode in experiment.MODES:
        base = selected+'_'+mode
        pairs.update([(f'mix_{mode}',base),(f'replay_{mode}',base),(f'mix_{mode}',f'replay_{mode}'),
                      (base,'plan_short'),(base,'initial')])
    pairs.add(('mix_joint','mix_decomposed'))
    comparisons = {}
    for a,b in sorted(pairs):
        result = {}
        for phase in ('all','peak'):
            x,y = matrices[a][phase],matrices[b][phase]
            assert np.array_equal(x['generated'],y['generated'])
            result[phase] = dict(dsr=paired(x['success'],x['generated'],y['success'],y['generated']),
                conditional=paired(x['success'],x['admitted'],y['success'],y['admitted']))
        comparisons[a+'--'+b] = result
    cross = defaultdict(list)
    roots = 0
    for file in sorted((out/'cross_audit').glob('*/*.pkl.gz')):
        for row in experiment.saved(file):
            roots += 1
            phase = 'peak' if 50 <= row['time'] < 100 else 'nonpeak'
            for arm,metrics in row['methods'].items():
                for kind,record in metrics.items():cross[(arm,phase,kind)].append(record)
    assert roots == 720
    diagnostic = {f'{arm}/{phase}/{kind}':calibration_summary(rows) for (arm,phase,kind),rows in cross.items()}
    fits = {f'{arm}/{r}':experiment.load(out/'fits'/arm/f'{r}.json') for arm in experiment.BASE_ARMS+experiment.ADDED_ARMS for r in range(6)}
    analysis = dict(status='COMPLETE', selection=experiment.load(out/'selection.json'), performance=summary,
        comparisons=comparisons, diagnostics=diagnostic, fits=fits,
        primary_pairs=['mix_joint--'+selected+'_joint','mix_joint--mix_decomposed'],
        primary_interval='simultaneous97_5, two predeclared all-episode DSR comparisons only',
        roots=roots, formal_test_access=False, independent_review=False,
        scope='Exploratory ordered cause tests, fixed inherited parents, selected configuration chosen with old public heldout data.')
    experiment.write(out/'analysis.json', analysis)
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--task-root', type=Path, required=True)
    review(parser.parse_args().task_root)
