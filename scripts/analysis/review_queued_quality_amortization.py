"""Count-based paired analyses for cascade-safe planning and analytical queue amortization."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.analysis import queued_waiting_rescue as base_experiment
from scripts.analysis.review_queued_waiting_rescue import load, paired

ARMS = ('old_hard', 'old_plan', 'quality_plan', 'quality_amortized')


def stage_review(root, stage, folder, primary_arm='quality_plan'):
    workloads = base_experiment.PILOT_WORKLOADS if stage == 'pilot' else base_experiment.CONFIRMATION_WORKLOADS
    replicas = 1 if stage == 'pilot' else 2
    folder = Path(folder)
    rows = [load(path) for path in (folder / stage).glob('*/*.json')]
    assert len(rows) == len(workloads) * replicas * 3 * len(ARMS)
    matrices = {}
    summary = {}
    fingerprints = {}
    for arm in ARMS:
        own = [row for row in rows if row['arm'] == arm]
        arrays = {
            phase: {key: np.zeros((3, len(workloads), replicas)) for key in ('generated', 'admitted', 'success')}
            for phase in ('all', 'pre', 'peak', 'post')
        }
        for row in own:
            counts = row['counts']
            assert row['audit']['status'] == 'PASS'
            assert counts['generated'] == counts.get('success', 0) + counts.get('expired', 0)
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
        arrays['nonpeak'] = {key: arrays['pre'][key] + arrays['post'][key] for key in arrays['pre']}
        matrices[arm] = arrays
        phase_summary = {}
        for phase, values in arrays.items():
            g, a, s = (values[key].sum() for key in ('generated', 'admitted', 'success'))
            phase_summary[phase] = dict(
                generated=int(g), admitted=int(a), success=int(s),
                dsr=float(s / g), admission=float(a / g), conditional=float(s / a),
            )
        frames = [frame for row in own for frame in row['trajectory']]
        phase_summary.update(
            episodes=len(own),
            added_decision_fraction=float(np.mean([frame['metadata']['added'] > 0 for frame in frames])),
            selected_new_fraction=float(np.mean([frame['selected_new'] for frame in frames])),
            selected_safe_plan_fraction=float(np.mean([frame['metadata'].get('selected_safe_plan', False) for frame in frames])),
            decision_mean=float(np.mean([frame['decision_seconds'] for frame in frames])),
            decision_p95=float(np.quantile([frame['decision_seconds'] for frame in frames], .95)),
            over_second=float(np.mean([frame['decision_seconds'] > 1 for frame in frames])),
            flight_m_per_episode=float(np.mean([
                sum(np.linalg.norm(delta) for frame in row['trajectory'] for delta in frame['motion'])
                for row in own
            ])),
            forecast_calls=sum(frame['forecast_calls'] for frame in frames),
        )
        summary[arm] = phase_summary

    pairs = (
        ('quality_plan', 'old_plan'),
        ('quality_plan', 'old_hard'),
        ('quality_amortized', 'old_plan'),
        ('quality_amortized', 'old_hard'),
        ('quality_plan', 'quality_amortized'),
        ('old_plan', 'old_hard'),
    )
    comparisons = {}
    for first, second in pairs:
        result = {}
        for phase in matrices[first]:
            a, b = matrices[first][phase], matrices[second][phase]
            assert np.array_equal(a['generated'], b['generated'])
            result[phase] = {
                metric: paired(
                    a['success'] if metric != 'admission' else a['admitted'], a[denominator],
                    b['success'] if metric != 'admission' else b['admitted'], b[denominator],
                )
                for metric, denominator in (('dsr', 'generated'), ('admission', 'generated'), ('conditional', 'admitted'))
            }
        comparisons[first + '--' + second] = result

    def check_gate(arm_name):
        comp = comparisons[arm_name + '--old_plan']
        if stage == 'pilot':
            ok = comp['all']['dsr']['mean'] >= .005 and comp['nonpeak']['dsr']['mean'] >= -.0025
            ok = ok and comp['peak']['conditional']['mean'] >= -.01
            return bool(ok)
        ok = comp['all']['dsr']['mean'] >= .005 and comp['all']['dsr']['simultaneous97_5'][0] > 0
        ok = ok and comp['nonpeak']['dsr']['simultaneous97_5'][0] > -.0025
        ok = ok and comp['peak']['conditional']['mean'] >= -.01
        return bool(ok)

    gates = {
        'quality_plan': check_gate('quality_plan'),
        'quality_amortized': check_gate('quality_amortized'),
        'amortized_vs_hard_passed': bool(
            comparisons['quality_amortized--old_hard']['all']['dsr']['mean'] >= .02
            and comparisons['quality_amortized--old_hard']['peak']['conditional']['mean'] >= .05
        ),
    }
    return dict(
        stage=stage,
        episodes=len(rows),
        summary=summary,
        comparisons=comparisons,
        gates=gates,
        gate_passed=bool(gates[primary_arm]),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--stage', choices=('pilot', 'confirmation'), default='pilot')
    args = parser.parse_args()
    print(json.dumps(stage_review(args.folder.parents[2], args.stage, args.folder), indent=2))
