"""Frozen batch-coordination protocol. Plan generation never launches training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from paths import REPO_ROOT

A = 'lower_mlp_upper_uncovered_obs'
B = 'lower_batch_upper_uncovered_obs'
C = 'lower_sequential_upper_uncovered_obs'
D = 'vanilla_lower_mlp_upper_uncovered_obs'
PARTNERS = {B: 'vanilla_lower_batch_upper_uncovered_obs', C: 'vanilla_lower_sequential_upper_uncovered_obs'}
GATES = {'minimum_dsr_difference': .01, 'minimum_positive_seeds': 4,
         'maximum_energy_ratio': 1.05, 'maximum_policy_p95_seconds': 1., 'simplicity_tie': .005}


def make_plan(stage='pilot', selected=None):
    if stage not in ('pilot', 'confirmation', 'smoke'):
        raise ValueError('Unknown stage')
    if stage == 'confirmation' and selected not in PARTNERS:
        raise ValueError('Confirmation requires a promoted B or C')
    confirm = stage == 'confirmation'
    seeds = list(range(630001, 630011)) if confirm else list(range(620001, 620006))
    workloads = list(range(730001, 730006)) if confirm else list(range(720001, 720006))
    methods = [A, D, selected, PARTNERS[selected]] if confirm else [A, B, C, D]
    plan = {'protocol': 'batch_offload_' + stage + '_20260928_v1', 'stage': stage,
            'purpose': 'verification' if stage == 'smoke' else 'pilot',
            'training_seeds': seeds, 'workload_seeds': workloads,
            'training_workload_seeds': {str(s): (830001 if confirm else 820001)+i for i, s in enumerate(seeds)},
            'train_episodes': 200, 'eval_episodes': 6, 'steps': 1000, 'rollout': 1000,
            'ppo_epochs': 8, 'batch_size': 256, 'checkpoint_every': 20,
            'statistical_unit': 'training seed after averaging paired workloads',
            'candidate_method': selected if confirm else A,
            'primary_comparators': [A, PARTNERS[selected]] if confirm else [],
            'promotion_gates': dict(GATES), 'selected_lower_method': selected if confirm else None,
            'variants': [{'name': 'main', 'ue_count': 100, 'compute_scale': 1., 'methods': methods}],
            'timing_contract': 'policy_step_wall_cuda_synchronized_v1',
            'checkpoint_selection': 'final prespecified episode only',
            'information_contract': 'covered_plus_uncovered; unchanged radius, fields, mask and rewards'}
    if stage == 'smoke':
        plan.update(training_seeds=[610001], workload_seeds=[710001],
                    training_workload_seeds={'610001': 810001}, train_episodes=2, eval_episodes=1,
                    steps=8, rollout=4, ppo_epochs=1, batch_size=4, checkpoint_every=1)
    return plan


def validate_plan(plan):
    expected = make_plan(plan['stage'], plan.get('selected_lower_method'))
    # The only extra confirmation fields are provenance, never protocol overrides.
    if {k: plan.get(k) for k in expected} != expected:
        raise ValueError('Frozen protocol fields changed')
    if set(plan) - set(expected) - {'pilot_provenance'}:
        raise ValueError('Unknown protocol fields')
    if plan['stage'] == 'confirmation' and 'pilot_provenance' not in plan:
        raise ValueError('Confirmation needs validated pilot provenance')
    all_seeds = plan['training_seeds'] + plan['workload_seeds'] + list(plan['training_workload_seeds'].values())
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('Overlapping seed roles')


def select_candidate(comparisons, timing):
    eligible = []
    for method in (B, C):
        metrics = comparisons[method]['metrics']; dsr = metrics['DSR']; energy = metrics['UAV Fleet Total Energy']
        ok = (dsr['mean_difference'] >= GATES['minimum_dsr_difference'] and
              dsr['positive_training_seeds'] >= GATES['minimum_positive_seeds'] and
              energy['candidate_mean'] <= GATES['maximum_energy_ratio'] * energy['reference_mean'] and
              timing[method]['p95_seconds'] < GATES['maximum_policy_p95_seconds'])
        comparisons[method]['promotion_criteria_met'] = bool(ok)
        if ok:
            eligible.append(method)
    if len(eligible) == 2:
        delta = comparisons[C]['metrics']['DSR']['mean_difference'] - comparisons[B]['metrics']['DSR']['mean_difference']
        return C if delta >= GATES['simplicity_tie'] else B
    return eligible[0] if eligible else None


def write_new(path, value):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f'Refusing to replace a different frozen file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke', 'pilot', 'confirmation'])
    parser.add_argument('--pilot-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    selected = provenance = None
    if args.stage == 'confirmation':
        if not args.pilot_root:
            parser.error('Confirmation requires --pilot-root; no manual winner override')
        from scripts.analysis.batch_offload_analysis import analyze
        from utils.canonical_experiment import digest
        report = analyze(args.pilot_root)
        if report['stage'] != 'pilot' or not report['selected_for_confirmation']:
            raise ValueError('Pilot has no promoted candidate; retain current method')
        selected = report['selected_for_confirmation']
        provenance = {'pilot_report_hash': digest(report), 'pilot_source_hash': report['source_hash']}
    plan = make_plan(args.stage, selected)
    if provenance:
        plan['pilot_provenance'] = provenance
    validate_plan(plan)
    path = args.output or REPO_ROOT/'docs/plans'/f'batch_offload_{args.stage}_20260928.json'
    write_new(path, plan)
    print(path)


if __name__ == '__main__':
    main()
