"""Recompute whole-system DSR with a policy-independent latent service denominator.

Run from the frozen experiment source directory. Request RNG streams are keyed by
scenario seed, episode index, and UE; mobility uses a separate stream. This
analysis replays only request draws, never changes a checkpoint or evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

import config
from environment.env import Env
from run_canonical_experiment import source_manifest
from scripts.analysis.whole_system_comparison import crossed_ci, holm_adjust
from utils.formal_protocol import NONLEARNED


def latent_counts(workload_seed: int, episodes: int, steps: int) -> list[int]:
    """Count the same latent service draws that policy-dependent battery can mask."""
    env = Env(scenario_seed=workload_seed)
    counts = []
    for _ in range(episodes):
        env.reset()  # Generates the first request for each UE.
        count = sum(ue.latent_request[0] < config.NUM_SERVICES for ue in env.ues)
        for _ in range(steps - 1):
            for ue in env.ues:
                ue.generate_request()
                count += ue.latent_request[0] < config.NUM_SERVICES
        counts.append(int(count))
    return counts


def checked_summary(job: Path, expected_source: str, expected_episodes: int) -> list[dict]:
    completed_path = job / 'completed.json'
    completed = json.loads(completed_path.read_text())
    if completed.get('status') != 'PASS':
        raise ValueError(f'Incomplete job: {job}')
    attempt = job / completed['attempt']
    manifest_path = attempt / 'manifest.json'
    summary_path = attempt / 'last_invocation.json'
    for path in (manifest_path, summary_path):
        relative = str(path.relative_to(job))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if completed['hashes'].get(relative) != digest:
            raise ValueError(f'Hash mismatch: {path}')
    manifest = json.loads(manifest_path.read_text())
    if manifest['source']['source_hash'] != expected_source:
        raise ValueError(f'Source mismatch: {job}')
    summary = json.loads(summary_path.read_text())
    episodes = summary['completed']
    if summary.get('status') != 'PASS' or not summary.get('evaluation') or len(episodes) != expected_episodes:
        raise ValueError(f'Invalid evaluation summary: {job}')
    if [int(row['episode']) for row in episodes] != list(range(expected_episodes)):
        raise ValueError(f'Unexpected episode indices: {job}')
    return episodes


def analyze(suite_root: Path) -> dict:
    suite = json.loads((suite_root / 'suite.json').read_text())
    plan = suite['plan']
    original = json.loads((suite_root / 'whole_system_paired_comparison.json').read_text())
    if original['status'] != 'PASS' or suite['source_hash'] != source_manifest()['source_hash']:
        raise ValueError('Frozen source and original comparison do not match')
    if original['source_hash'] != suite['source_hash']:
        raise ValueError('Original comparison source mismatch')
    if len(plan['variants']) != 1 or plan['variants'][0]['name'] != 'main':
        raise ValueError('Expected one prespecified main setting')
    if plan['variants'][0]['ue_count'] != config.NUM_UES:
        raise ValueError('Environment UE count differs from the experiment plan')
    candidate = plan['candidate_method']
    methods = plan['variants'][0]['methods']
    seeds = plan['training_seeds']
    workloads = plan['workload_seeds']
    episodes = plan['eval_episodes']
    steps = plan['steps']
    if not all(isinstance(value, int) and value > 0 for value in (episodes, steps)):
        raise ValueError('Invalid episode or step count')

    with (suite_root / 'evaluation_units.csv').open(newline='') as file:
        original_units = {
            (row['method'], None if row['training_seed'] in ('', 'None') else int(row['training_seed']),
             int(row['workload_seed'])): float(row['DSR'])
            for row in csv.DictReader(file)
        }

    by_workload = {workload: latent_counts(workload, episodes, steps) for workload in workloads}
    units = []
    lookup = {}
    max_original_error = 0.0
    for method in methods:
        training = [None] if method in NONLEARNED else seeds
        for seed in training:
            for workload in workloads:
                job = suite_root / 'main' / method
                if seed is not None:
                    job /= f'seed_{seed}'
                job /= f'workload_{workload}'
                rows = checked_summary(job, suite['source_hash'], episodes)
                potential_by_episode = by_workload[workload]
                actual = sum(int(row['generated_service_count']) for row in rows)
                successful = sum(int(row['deadline_satisfied_count']) for row in rows)
                potential = sum(potential_by_episode)
                for row, latent in zip(rows, potential_by_episode):
                    generated = int(row['generated_service_count'])
                    success = int(row['deadline_satisfied_count'])
                    if not (0 <= success <= generated <= latent):
                        raise ValueError(f'Invalid request counts: {method}, {seed}, {workload}, episode {row["episode"]}')
                original_dsr = successful / actual if actual else 0.0
                key = (method, seed, workload)
                max_original_error = max(max_original_error, abs(original_dsr - original_units[key]))
                unit = {
                    'method': method, 'training_seed': seed, 'workload_seed': workload,
                    'actual_generated_service_count': actual,
                    'potential_service_count': potential,
                    'suppressed_potential_service_count': potential - actual,
                    'deadline_satisfied_count': successful,
                    'original_dsr_reproduced': original_dsr,
                    'potential_denominator_dsr': successful / potential if potential else 0.0,
                }
                units.append(unit)
                lookup[key] = unit
    if max_original_error > 1e-12 or len(lookup) != len(original_units):
        raise ValueError(f'Original DSR not reproduced: max error {max_original_error}')

    def grid(method: str) -> np.ndarray:
        training = [None] if method in NONLEARNED else seeds
        return np.array([[lookup[(method, seed, workload)]['potential_denominator_dsr']
                          for workload in workloads] for seed in training], dtype=float)

    target = grid(candidate)
    comparisons = {}
    raw_p = []
    for method in methods:
        if method == candidate:
            continue
        reference = grid(method)
        difference = target - reference
        entry = {
            'kind': 'nonlearned' if method in NONLEARNED else 'learned',
            'candidate_mean': float(target.mean()),
            'comparator_mean': float(reference.mean()),
            'mean_difference': float(difference.mean()),
            'paired_crossed_bootstrap_ci95': crossed_ci(difference),
        }
        if method in plan['primary_comparators']:
            seed_means = difference.mean(axis=1)
            sem = stats.sem(seed_means)
            crit = stats.t.ppf(.975, len(seed_means) - 1)
            entry['training_seed_t_ci95'] = [float(seed_means.mean() - crit * sem),
                                             float(seed_means.mean() + crit * sem)]
            entry['two_sided_p'] = float(stats.ttest_1samp(seed_means, 0).pvalue)
            entry['positive_training_seeds'] = int(np.sum(seed_means > 0))
            raw_p.append((method, entry['two_sided_p']))
        comparisons[method] = entry
    adjusted = holm_adjust([p for _, p in raw_p])
    for (method, _), p in zip(raw_p, adjusted):
        comparisons[method]['holm_adjusted_p'] = p
    decision = all(comparisons[method]['mean_difference'] > 0 and
                   comparisons[method]['holm_adjusted_p'] < .05
                   for method in plan['primary_comparators'])
    report = {
        'status': 'PASS',
        'study': 'fixed_potential_service_denominator_robustness',
        'source_hash': suite['source_hash'],
        'original_protocol': plan['protocol'],
        'denominator': 'all latent service draws, including those converted to energy requests by battery state',
        'numerator': 'archived deadline-satisfied actual service requests; no policy rerun or retraining',
        'training_seeds': seeds, 'workload_seeds': workloads,
        'latent_service_count_by_workload_episode': {str(k): v for k, v in by_workload.items()},
        'original_dsr_max_absolute_reproduction_error': max_original_error,
        'statistical_unit': 'training seed after averaging the five paired workloads',
        'units': units, 'comparisons': comparisons, 'decision_rule_met': decision,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.suite_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, args.output)
    print(json.dumps({'status': report['status'], 'decision_rule_met': report['decision_rule_met'],
                      'original_dsr_max_absolute_reproduction_error':
                      report['original_dsr_max_absolute_reproduction_error'],
                      'strongest_comparison': report['comparisons']['lower_no_attention']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
