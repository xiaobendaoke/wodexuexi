"""Read-only saved-state diagnostics; no physics, Torch, training or tuning.

Conditional opportunity is a public-teacher, fixed-bank quantity, not a bound
on the value of joint control in the physical environment or on long-run DSR.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from marl_models.queued_planner_distillation import policy
from utils.queued_planner_distillation import ARMS, BASELINES, REVISION, specification, imitation_metrics, planner_target

TASK = 'chapter5_dynamic_service_20261028'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def saved(path):
    return pickle.loads(gzip.decompress(Path(path).read_bytes()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditional_opportunity(q, weights):
    q, w = np.asarray(q, float), np.asarray(weights, float)
    if (q.ndim != 2 or not q.size or w.shape != (len(q),)
            or not np.isfinite(q).all() or not np.isfinite(w).all()
            or np.any(w < 0) or not np.isclose(w.sum(), 1.)):
        raise ValueError('finite score bank and normalized fixed plan weights required')
    conditional = float(w@q.max(1))
    common = float(np.max(w@q))
    centered = q-q.mean()
    interaction = q-q.mean(0, keepdims=True)-q.mean(1, keepdims=True)+q.mean()
    total = float(np.sum(centered**2))
    return dict(opportunity=conditional-common, conditional_best=conditional, common_motion_best=common,
                interaction_variance_fraction=float(np.sum(interaction**2))/total if total > 1e-15 else 0.,
                plan_range=float(np.ptp(q.mean(1))), motion_range=float(np.ptp(q.mean(0))),
                q_range=float(np.ptp(q)), plans=q.shape[0], motions=q.shape[1])


def teacher_projection(q, prior):
    """Correlation cost of projecting the tie-aware teacher onto its marginals.

    This characterizes imitation of the specified randomized teacher, not the
    best deterministic factorized controller: one consistent optimal pair can
    always be represented by two point masses in an unrestricted fixed bank.
    """
    q = np.asarray(q, float)
    target = planner_target(q, prior)
    independent = target.sum(1)[:, None]*target.sum(0)[None, :]
    active = target > 0
    mutual_information = float(np.sum(target[active]*np.log(target[active]/independent[active])))
    return dict(teacher_mutual_information=mutual_information,
                independent_projection_score_regret=float(q.max()-np.sum(independent*q)),
                correlated_teacher=bool(mutual_information > 1e-10))


def row_diagnostic(row, actor, mode):
    p = policy(row, actor, mode)
    q = np.asarray(row['q'], float)
    pa = p.sum(1)
    pm = np.divide(p, pa[:, None], out=np.zeros_like(p), where=pa[:, None] > 0)
    common = p.sum(0)
    teacher = conditional_opportunity(q, row['prior'])
    actual = conditional_opportunity(q, pa)
    best = p >= p.max()-1e-12
    greedy_score = float(q[best].mean())
    output = imitation_metrics(q, p, row['prior'])
    output.update(prior_conditioning_opportunity=teacher['opportunity'],
                  student_conditioning_opportunity=actual['opportunity'],
                  interaction_variance_fraction=teacher['interaction_variance_fraction'],
                  plan_score_range=teacher['plan_range'], motion_score_range=teacher['motion_range'],
                  q_range=teacher['q_range'], plans=teacher['plans'], motions=teacher['motions'],
                  conditional_motion_tv=float(np.sum(pa[:, None]*np.abs(pm-common[None, :]))/2),
                  greedy_score_regret=float(q.max()-greedy_score),
                  motion_features_plan_invariant=bool(np.max(np.abs(row['m']-row['m'].mean(0, keepdims=True))) < 1e-10),
                  allocation_feature_plan_range=float(np.max(np.ptp(row['a'], axis=0))),
                  phase='pre' if row['time'] < 50 else 'peak' if row['time'] < 100 else 'post')
    return output


def aggregate_rows(rows):
    out = dict(states=len(rows))
    for k in ('teacher_kl', 'score_regret', 'optimal_probability', 'greedy_score_regret',
              'prior_conditioning_opportunity', 'student_conditioning_opportunity',
              'conditional_motion_tv', 'interaction_variance_fraction', 'plan_score_range',
              'motion_score_range', 'q_range', 'plans', 'motions', 'allocation_feature_plan_range'):
        out[k] = float(np.mean([r[k] for r in rows]))
    for k in ('all_tied', 'motion_features_plan_invariant'):
        out[k+'_fraction'] = float(np.mean([r[k] for r in rows]))
    out['positive_prior_conditioning_fraction'] = float(np.mean([r['prior_conditioning_opportunity'] > 1e-10 for r in rows]))
    out['multiplan_fraction'] = float(np.mean([r['plans'] > 1 for r in rows]))
    available = sum(r['available_planning_gain'] for r in rows)
    out['available_planning_gain'] = available/len(rows)
    out['retained_planning_gain'] = sum(r['expected_score_gain'] for r in rows)/available if available > 1e-12 else None
    return out


def analyze(root):
    root = Path(root).resolve()
    folder = root/'results'/TASK/REVISION
    validation = root/'output'/TASK/'validation'/REVISION
    completion = load(validation/'run_complete.json')
    assert completion['status'] == 'COMPLETE' and not completion['formal_test_access']
    protocol = load(folder/'protocol.json')
    assert protocol['specification'] == specification()
    hashes = {str(p.relative_to(root)): sha(p) for p in folder.rglob('*') if p.is_file()}
    all_episodes = [load(p) for p in (folder/'episodes').glob('*/*/*.json')]
    split_counts = {split: sum(r['split'] == split for r in all_episodes) for split in ('training', 'holdout', 'evaluation')}
    assert split_counts == dict(training=216, holdout=72, evaluation=384)
    for r in all_episodes:
        c = r['counts']
        assert r['audit']['status'] == 'PASS'
        assert c['generated'] == c.get('success', 0)+c.get('expired', 0)
        assert sum(r['successes']) == c.get('success', 0)
        assert r['metrics']['expiration_stage'].get('waiting', 0) == c['generated']-c['admitted']
    episode_summary = {}
    for arm in ARMS+BASELINES:
        records = [r for r in all_episodes if r['split'] == 'evaluation' and r['arm'] == arm]
        totals = {k: sum(r['counts'].get(k, 0) for r in records) for k in ('generated', 'admitted', 'success', 'expired', 'service_crossed', 'disconnect_events')}
        g, a, s = (totals[k] for k in ('generated', 'admitted', 'success'))
        metrics = [r['metrics'] for r in records]
        stages = {k: sum(m['expiration_stage'].get(k, 0) for m in metrics) for k in ('waiting', 'upload', 'transfer', 'compute')}
        episode_summary[arm] = dict(counts=totals, dsr=s/g, admission=a/g, conditional=s/a,
            failure_waiting_fraction=stages['waiting']/(g-s), expiration_stage=stages,
            service_crossed_fraction=totals['service_crossed']/a,
            mean_stage_time={k: float(np.mean([m['mean_stage_time'][k] for m in metrics])) for k in metrics[0]['mean_stage_time']},
            cpu_utilization=np.mean([m['cpu_utilization'] for m in metrics], 0).tolist(),
            mbs_fraction_mean=float(np.mean([m['mbs_fraction'] for m in metrics])),
            successful_completions_by_clock_phase={k: float(np.mean([sum(r['successes'][lo:hi]) for r in records]))
                for k, lo, hi in [('pre', 0, 50), ('peak', 50, 100), ('post_and_drain', 100, 205)]})
    updates = [load(p) for p in (folder/'updates').glob('*/*.json')]
    assert len(updates) == 36
    diagnostics, update_summary = {}, {}
    for arm in ARMS:
        mode = 'decomposed' if arm == 'mlp_decomposed' else 'joint'
        rows_out = []
        for run in specification()['runs']:
            actor = saved(folder/'models'/arm/str(run['id'])/'2.pkl.gz')
            for p in sorted((folder/'experience/holdout'/arm).glob(f'{run["id"]}_*.pkl.gz')):
                rows_out.extend(row_diagnostic(r, actor, mode) for r in saved(p))
        assert len(rows_out) == 2400
        diagnostics[arm] = dict(all=aggregate_rows(rows_out),
            by_phase={phase: aggregate_rows([r for r in rows_out if r['phase'] == phase]) for phase in ('pre', 'peak', 'post')})
        update_summary[arm] = {}
        for version in (1, 2):
            selected = [r for r in updates if r['arm'] == arm and r['version'] == version]
            update_summary[arm][version] = {
                split: {k: float(np.mean([r[split][k] for r in selected])) for k in ('teacher_kl', 'score_regret', 'optimal_probability', 'retained_planning_gain')}
                for split in ('training_before', 'training_after', 'holdout_before', 'holdout_after')}
    resources = load(validation/'resources.json')
    groups = defaultdict(list)
    for r in resources['records']:
        groups[r['key'].split(':')[0]].append(r)
    costs = {k: dict(tasks=len(v), cpu_seconds=sum(r['cpu_seconds'] for r in v),
                    task_wall_mean=float(np.mean([r['end']-r['start'] for r in v])),
                    task_wall_max=max(r['end']-r['start'] for r in v)) for k, v in groups.items()}
    assert all(sha(root/name) == digest for name, digest in hashes.items()), 'saved scientific result changed'
    assert all(sha(root/name) == digest for name, digest in completion['protected'].items()), 'old protected input changed'
    result = dict(status='PASS', analysis_kind='Saved-data post-hoc diagnosis; no new physics/training',
                  split_counts=split_counts, episode_summary=episode_summary, updates=update_summary,
                  holdout_diagnostics=diagnostics, costs=costs,
                  resources={k: resources[k] for k in ('wall_seconds', 'equivalent_cpus', 'utilization', 'peak_active')},
                  original_analysis=load(folder/'analysis.json'),
                  invariants=dict(scientific_results_unchanged=True, protected_inputs_unchanged=True,
                                  formal_test_access=False, new_physical_episodes=0, new_optimizer_steps=0),
                  scope='2400 saved heldout states per arm are correlated observations from 6 residual policies and 24 workloads; not independent training repeats. Conditional opportunities concern fixed public teacher banks only.')
    return result, hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-root', type=Path, default=REPO_ROOT)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('refuse diagnostic overwrite')
    result, hashes = analyze(args.task_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    receipt = args.output.with_name(args.output.stem+'_source_hashes.json')
    receipt.write_text(json.dumps(hashes, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], output=str(args.output), raw_files_checked=len(hashes))))


if __name__ == '__main__':
    main()
