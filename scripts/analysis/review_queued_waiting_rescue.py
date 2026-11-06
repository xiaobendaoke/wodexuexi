"""Count-based paired analyses for the staged waiting-request prototype."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.analysis import queued_waiting_rescue as experiment
from scripts.analysis.review_queued_recovery_audit import paired_summary


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paired(a, da, b, db):
    a, da, b, db = (np.asarray(value, float) for value in (a, da, b, db))
    assert a.shape == da.shape == b.shape == db.shape
    assert a.ndim == 3 and a.shape[0] == 3 and np.all(da > 0) and np.all(db > 0)
    _, workloads, replicas = a.shape
    rng = np.random.default_rng(2026110440)
    indices = rng.integers(0, workloads, (4000, workloads))
    streams = rng.integers(0, replicas, (4000, workloads, replicas))
    samples = np.zeros(4000)
    effects = a.sum((1, 2))/da.sum((1, 2))-b.sum((1, 2))/db.sum((1, 2))
    for parent in range(3):
        sums = [value[parent][indices[:, :, None], streams].sum((1, 2)) for value in (a, da, b, db)]
        samples += (sums[0]/sums[1]-sums[2]/sums[3])/3
    return dict(mean=float(effects.mean()), by_parent=effects.tolist(),
                ci95=np.quantile(samples, [.025, .975]).tolist(),
                simultaneous97_5=np.quantile(samples, [.0125, .9875]).tolist(),
                scope='Fixed three parents, paired workload blocks and action streams. No new training repeats.')


def offline_review(root):
    rows = [load(path) for path in sorted((experiment.out(root)/'offline').glob('*/*.json'))]
    assert len(rows) == 96
    for row in rows:
        old = np.array(row['actual_old'])
        new = np.array(row['actual_new'])
        assert np.array_equal(old, new[:, :old.shape[1]])
        assert row['new_actions']-row['old_actions'] == row['metadata']['added']
        assert np.isclose(row['capacity_gain'], new.max()-old.max())
        prior = np.array(row['prior'])
        old_p = experiment.planner_target(np.array(row['public_old']), prior)
        new_p = experiment.planner_target(np.array(row['public_new']), prior)
        expected = np.sum(new_p*new)-np.sum(old_p*old)
        assert np.isclose(expected, row['selected_gain'])
    changed = sum(row['metadata']['added'] > 0 for row in rows)
    capacity = float(np.mean([row['capacity_gain'] for row in rows]))
    selected = float(np.mean([row['selected_gain'] for row in rows]))
    return dict(roots=96, changed_roots=changed,
                mean_additions=float(np.mean([row['metadata']['added'] for row in rows])),
                mean_capacity_gain=capacity, mean_selected_gain=selected,
                gate_passed=bool(changed >= 12 and capacity >= .05 and selected >= 0),
                gate=dict(changed_roots_minimum=12, capacity_minimum=.05, selected_gain_minimum=0),
                groups={str(now):{key:dict(paired_summary(rows, key, now),
                    scope='Two behavior roots averaged per repeat/workload cell. Four public forecasts and exact best-set expectations. Exploratory.')
                    for key in ('capacity_gain', 'selected_gain')}
                        for now in experiment.previous.recovery.TIMES})


def stage_review(root, stage, folder=None, candidate_arm='rescue_plan'):
    workloads = experiment.PILOT_WORKLOADS if stage == 'pilot' else experiment.CONFIRMATION_WORKLOADS
    replicas = 1 if stage == 'pilot' else 2
    folder = experiment.out(root) if folder is None else Path(folder)
    rows = [load(path) for path in (folder/stage).glob('*/*.json')]
    assert len(rows) == len(workloads)*replicas*3*3
    matrices = {}
    summary = {}
    fingerprints = {}
    for arm in ('old_hard', 'old_plan', candidate_arm):
        own = [row for row in rows if row['arm'] == arm]
        arrays = {phase:{key:np.zeros((3, len(workloads), replicas)) for key in ('generated', 'admitted', 'success')}
                  for phase in ('all', 'pre', 'peak', 'post')}
        for row in own:
            counts = row['counts']
            assert row['audit']['status'] == 'PASS'
            assert counts['generated'] == counts.get('success', 0)+counts.get('expired', 0)
            assert sum(row['successes']) == counts.get('success', 0)
            index = (row['parent'], workloads.index(row['workload']), row['replica'])
            exogenous = (row['workload'], row['replica'])
            fingerprint = (row['fingerprint'], counts['generated'])
            assert exogenous not in fingerprints or fingerprints[exogenous] == fingerprint
            fingerprints[exogenous] = fingerprint
            for phase in arrays:
                source = counts if phase == 'all' else row['cohorts']['arrival'][phase]
                for key in arrays[phase]:
                    arrays[phase][key][index] = source.get(key, 0)
        arrays['nonpeak'] = {key:arrays['pre'][key]+arrays['post'][key] for key in arrays['pre']}
        matrices[arm] = arrays
        phase_summary = {}
        for phase, values in arrays.items():
            g, a, s = (values[key].sum() for key in ('generated', 'admitted', 'success'))
            phase_summary[phase] = dict(generated=int(g), admitted=int(a), success=int(s),
                                        dsr=float(s/g), admission=float(a/g), conditional=float(s/a))
        frames = [frame for row in own for frame in row['trajectory']]
        phase_summary.update(episodes=len(own),
            added_decision_fraction=float(np.mean([frame['metadata']['added'] > 0 for frame in frames])),
            selected_new_fraction=float(np.mean([frame['selected_new'] for frame in frames])),
            decision_mean=float(np.mean([frame['decision_seconds'] for frame in frames])),
            decision_p95=float(np.quantile([frame['decision_seconds'] for frame in frames], .95)),
            over_second=float(np.mean([frame['decision_seconds'] > 1 for frame in frames])),
            flight_m_per_episode=float(np.mean([sum(np.linalg.norm(delta) for frame in row['trajectory'] for delta in frame['motion']) for row in own])),
            forecast_calls=sum(frame['forecast_calls'] for frame in frames))
        summary[arm] = phase_summary
    comparisons = {}
    for first, second in ((candidate_arm, 'old_plan'), (candidate_arm, 'old_hard'), ('old_plan', 'old_hard')):
        result = {}
        for phase in matrices[first]:
            a, b = matrices[first][phase], matrices[second][phase]
            assert np.array_equal(a['generated'], b['generated'])
            result[phase] = {
                metric:paired(a['success'] if metric != 'admission' else a['admitted'], a[denominator],
                              b['success'] if metric != 'admission' else b['admitted'], b[denominator])
                for metric, denominator in (('dsr', 'generated'), ('admission', 'generated'), ('conditional', 'admitted'))
            }
        comparisons[first+'--'+second] = result
    primary = comparisons[candidate_arm+'--old_plan']
    if stage == 'pilot':
        passed = primary['all']['dsr']['mean'] >= .005 and primary['nonpeak']['dsr']['mean'] >= -.0025
        passed = passed and primary['peak']['conditional']['mean'] >= -.01
    else:
        passed = primary['all']['dsr']['mean'] >= .005 and primary['all']['dsr']['simultaneous97_5'][0] > 0
        passed = passed and primary['nonpeak']['dsr']['simultaneous97_5'][0] > -.0025
    return dict(stage=stage, episodes=len(rows), summary=summary, comparisons=comparisons, gate_passed=bool(passed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(offline_review(args.task_root), indent=2))
