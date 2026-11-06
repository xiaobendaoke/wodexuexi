"""Arithmetic and complete-grid review of the ordered diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from utils.queued_learning_validation import REVISION, case_grid, phase
from utils.queued_closed_loop import paired_summary

TASK = 'chapter5_dynamic_service_20261028'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def means(records, keys):
    return {k: float(np.mean([r[k] for r in records if r[k] is not None]))
            if any(r[k] is not None for r in records) else None for k in keys}


def fit_summary(records):
    groups = defaultdict(list)
    for r in records:groups[(r['arm'], r['small'])].append(r)
    result = {}
    for (arm, small), rows in groups.items():
        assert len(rows) == 6
        epochs = [v['epoch'] for v in rows[0]['curve']]
        assert all([v['epoch'] for v in r['curve']] == epochs for r in rows)
        curve = []
        for i, epoch in enumerate(epochs):
            row = dict(epoch=epoch, steps=rows[0]['curve'][i]['steps'])
            for split in ('train', 'holdout'):
                if rows[0]['curve'][i][split] is None:continue
                row[split] = {p: means([r['curve'][i][split][p] for r in rows],
                    ('teacher_kl', 'score_regret', 'retained_planning_gain', 'optimal_probability')) for p in ('all', 'peak', 'nonpeak')}
            curve.append(row)
        change = []
        sufficient = []
        for r in rows:
            last = r['curve'][-1];earlier = next(v for v in r['curve'] if v['epoch'] == (128 if small else 32))
            a, b = earlier['train']['all']['teacher_kl'], last['train']['all']['teacher_kl']
            change.append((a-b)/max(a, 1e-12))
            sufficient.append(all(last['train'][p]['teacher_kl'] <= .1 and
                (last['train'][p]['retained_planning_gain'] is None or last['train'][p]['retained_planning_gain'] >= .9)
                for p in ('peak', 'nonpeak')))
        result[arm+('_tiny' if small else '_full')] = dict(curve=curve,
            late_relative_training_kl_reduction_by_run=change, sufficient_fit_by_run=sufficient,
            interpretation=['still_improving' if x > .05 else 'near_plateau_in_fixed_budget' if x <= .01 else 'uncertain' for x in change])
    return result


def feature_summary(records):
    assert len(records) == 48
    groups = defaultdict(list)
    for r in records:
        assert r['no_holdout_selection'] and r['curve'][-1]['epoch'] == 64
        groups[(r['feature'], r['label'], r['mode'])].append(r)
    result = {};values = {}
    for key, group in groups.items():
        assert len(group) == 6 and len({r['parameter_count'] for r in group}) == 1
        name = '/'.join(key);per_run = []
        for r in sorted(group, key=lambda z: z['run']):
            selected = r['heldout_common_reference'];assert len(selected) == 48
            def summarize(xs):
                available = sum(z['available_planning_gain'] for z in xs)
                return dict(score_regret=float(np.mean([z['score_regret'] for z in xs])),
                    retained_planning_gain=sum(z['expected_score_gain'] for z in xs)/available if available > 1e-12 else None)
            per_run.append({p: summarize([z for z in selected if p == 'all' or phase(z['time']) == p])
                            for p in ('all', 'peak', 'nonpeak')})
        result[name] = dict(parameter_count=group[0]['parameter_count'], by_run=per_run,
            means={p: means([r[p] for r in per_run], ('score_regret', 'retained_planning_gain')) for p in ('all', 'peak', 'nonpeak')})
        values[name] = np.array([r['all']['score_regret'] for r in per_run])
    contrasts = {}
    for label in ('single', 'mean16'):
        for mode in ('joint', 'decomposed'):
            contrast = values[f'summary/{label}/{mode}']-values[f'task/{label}/{mode}']
            contrasts[f'task_gain/{label}/{mode}'] = dict(mean=float(contrast.mean()), by_run=contrast.tolist())
    for feature in ('summary', 'task'):
        for mode in ('joint', 'decomposed'):
            contrast = values[f'{feature}/single/{mode}']-values[f'{feature}/mean16/{mode}']
            contrasts[f'averaging_gain/{feature}/{mode}'] = dict(mean=float(contrast.mean()), by_run=contrast.tolist())
    return dict(groups=result, paired_regret_reductions=contrasts,
                scope='Paired saved heldout roots and final fixed models; forecast scores, not closed-loop DSR; no checkpoint selection.')


def case_summary(episodes, roots):
    assert len(episodes) == 1296 and len(roots) == 1944
    result = []
    for case in case_grid():
        records = [r for r in episodes if r['case'] == case]
        selected = [r for r in roots if r['case'] == case]
        assert len(records) == 48 and len(selected) == 72
        summary = {};counts = {};fingerprints = {}
        for arm in ('mlp_joint', 'mlp_decomposed'):
            own = [r for r in records if r['arm'] == arm]
            success = np.zeros((3, 2, 4, 1));generated = np.zeros_like(success)
            admitted = 0
            for r in own:
                c = r['counts'];assert sum(r['successes']) == c.get('success', 0)
                assert c['generated'] == c.get('success', 0)+c.get('expired', 0)
                index = (r['parent'], r['run'] % 2, r['workload']-942000, 0)
                success[index] = c.get('success', 0);generated[index] = c['generated'];admitted += c['admitted']
                pair = r['workload'];f = (r['fingerprint'], c['generated'])
                assert pair not in fingerprints or fingerprints[pair] == f;fingerprints[pair] = f
            assert len(own) == 24 and np.all(generated > 0)
            counts[arm] = (success, generated)
            summary[arm] = dict(dsr=float(np.mean(success.sum((2, 3))/generated.sum((2, 3)))),
                admission=admitted/float(generated.sum()), conditional=float(success.sum())/admitted,
                service_crossed=float(np.mean([r['metrics']['service_crossed_fraction'] for r in own])))
        a, g = counts['mlp_joint'];b, gb = counts['mlp_decomposed'];assert np.array_equal(g, gb)
        contrast = paired_summary(a, b, g, draws=2000, seed=2026110401)
        margins = [m for r in selected for m in r['new_link_margins_seconds']]
        mechanism = {}
        for p in ('all', 'peak', 'nonpeak'):
            rs = [r for r in selected if p == 'all' or phase(r['time']) == p]
            mechanism[p] = dict(states=len(rs),
                conditional_opportunity=float(np.mean([r['opportunity']['opportunity'] for r in rs])),
                positive_opportunity_fraction=float(np.mean([r['opportunity']['opportunity'] > 1e-10 for r in rs])),
                interaction_variance_fraction=float(np.mean([r['opportunity']['interaction_variance_fraction'] for r in rs])),
                projection_regret=float(np.mean([r['projection']['independent_projection_score_regret'] for r in rs])),
                split_selection_regret=float(np.mean([r['label_stability']['split_selection_regret'] for r in rs])),
                admitted_tasks=float(np.mean([r['admitted_tasks'] for r in rs])))
        result.append(dict(case=case, summary=summary, joint_minus_decomposed=contrast, mechanism=mechanism,
            candidate_new_links=len(margins), new_link_margin_lt_3_fraction=float(np.mean(np.array(margins) < 3)) if margins else None))
    return result


def review(root):
    root = Path(root).resolve();r = root/'results'/TASK/REVISION;v = root/'output'/TASK/'validation'/REVISION
    resources = {kind: load(v/(kind+'_complete.json')) for kind in ('fit', 'replay', 'label', 'feature_fit', 'condition', 'condition_label')}
    assert {k: z['tasks'] for k, z in resources.items()} == dict(fit=36, replay=96, label=1152, feature_fit=48, condition=1296, condition_label=1944)
    fits = [load(p) for p in (r/'stage1/fits').glob('*.json')]
    assert len(fits) == 36
    replay = [load(p) for p in (r/'stage2/replay').glob('*.json')]
    assert len(replay) == 96 and all(z['status'] == 'PASS' and z['states'] == 12 and z['saved_feature_states_checked'] == 100 for z in replay)
    feature = [load(p) for p in (r/'stage2/fits').glob('*.json')]
    import gzip,pickle
    labels = [pickle.loads(gzip.decompress(p.read_bytes())) for p in (r/'stage2/labels').glob('*.pkl.gz')]
    assert len(labels) == 1152 and all(z['forecasts'].shape[0] == 16 for z in labels)
    stability = {}
    for p in ('all', 'peak', 'nonpeak'):
        own = [z for z in labels if p == 'all' or phase(z['row']['time']) == p]
        stability[p] = dict(states=len(own), **means([z['stability'] for z in own],
            ('single_prediction_variance', 'split_selection_overlap', 'split_selection_regret', 'mean_prediction_range')),
            truncated_task_states=sum(z['truncated'] > 0 for z in own), mean_truncated_tasks=float(np.mean([z['truncated'] for z in own])))
    episodes = [load(p) for p in (r/'stage3/episodes').glob('*.json')]
    roots = [load(p) for p in (r/'stage3/labels').glob('*.json')]
    result = dict(status='COMPLETE', stage1=fit_summary(fits), stage2=dict(stability=stability, fitting=feature_summary(feature)),
                  stage3=case_summary(episodes, roots),
                  resources={k: {key: z[key] for key in ('tasks', 'wall_seconds', 'worker_cpu_seconds', 'equivalent_cpus', 'utilization')} for k, z in resources.items()},
                  scope='Fixed-data fitting, public-teacher diagnostics and frozen-policy zero-shot transfer. No new independent end-to-end training or formal test access.',
                  formal_test_access=False, independent_review=False)
    path = r/'analysis.json'
    if path.exists():raise FileExistsError('refuse analysis overwrite')
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--task-root', type=Path, default=REPO_ROOT)
    print(json.dumps(review(p.parse_args().task_root)['status']))
