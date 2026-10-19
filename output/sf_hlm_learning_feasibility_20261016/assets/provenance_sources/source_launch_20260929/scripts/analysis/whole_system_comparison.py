"""Prespecified paired comparison of a complete policy against existing methods.

Run from the repository root with ``python -m scripts.analysis.whole_system_comparison``.
The suite analyzer must have passed first; this script never reads training reward.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from utils.canonical_experiment import write_json
from utils.formal_protocol import NONLEARNED


METRICS = (
    'DSR', 'Admission fraction', 'Deadline satisfaction among admitted',
    'UAV Fleet Total Energy', 'Non-flight assignment energy', 'Fairness',
    'Effective tasks per MJ', 'MBS offload fraction',
)


def crossed_ci(values: np.ndarray, *, draws: int = 20000, seed: int = 20260928) -> list[float]:
    """Resample training seeds and paired workloads, not episodes or requests."""
    rng = np.random.default_rng(seed)
    ti = rng.integers(0, values.shape[0], size=(draws, values.shape[0]))
    wi = rng.integers(0, values.shape[1], size=(draws, values.shape[1]))
    means = values[ti[:, :, None], wi[:, None, :]].mean(axis=(1, 2))
    return [float(v) for v in np.quantile(means, [.025, .975])]


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[int(index)] = running
    return adjusted


def analyze(directory: Path) -> dict:
    directory = Path(directory)
    suite = json.loads((directory / 'suite.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    analysis = json.loads((directory / 'analysis.json').read_text())
    if status.get('status') != 'PASS' or analysis.get('status') != 'PASS':
        raise ValueError('The complete suite has not passed')
    if suite['source_hash'] != analysis['source_hash']:
        raise ValueError('Analysis source differs from suite source')
    plan = suite['plan']
    if len(plan['variants']) != 1 or plan['variants'][0]['name'] != 'main':
        raise ValueError('Expected the prespecified single main variant')
    candidate = plan['candidate_method']
    learned_comparators = plan['primary_comparators']
    methods = plan['variants'][0]['methods']
    if candidate not in methods or any(m not in methods or m in NONLEARNED for m in learned_comparators):
        raise ValueError('Invalid candidate or primary comparison in plan')
    seeds, workloads = plan['training_seeds'], plan['workload_seeds']
    expected = len(seeds) * len(workloads) * sum(m not in NONLEARNED for m in methods)
    expected += len(workloads) * sum(m in NONLEARNED for m in methods)
    units = analysis['units']
    if len(units) != expected:
        raise ValueError(f'Expected {expected} units, found {len(units)}')
    lookup = {(u['method'], u['training_seed'], u['workload_seed']): u for u in units}
    if len(lookup) != len(units):
        raise ValueError('Duplicate evaluation unit')

    def grid(method: str, metric: str) -> np.ndarray:
        training = [None] if method in NONLEARNED else seeds
        try:
            return np.asarray([[lookup[(method, s, w)][metric] for w in workloads] for s in training], dtype=float)
        except KeyError as exc:
            raise ValueError(f'Missing unit or metric: {exc}') from exc

    report = {'status': 'PASS', 'protocol': plan['protocol'], 'source_hash': suite['source_hash'],
              'candidate': candidate, 'training_seeds': seeds, 'workload_seeds': workloads,
              'statistical_unit': 'paired training-seed mean over workloads for learned comparisons',
              'primary_metric': 'DSR', 'primary_comparators': learned_comparators,
              'comparisons': {}}
    raw_p = []
    for method in methods:
        if method == candidate:
            continue
        entry = {'method': method, 'kind': 'nonlearned' if method in NONLEARNED else 'learned', 'metrics': {}}
        for metric in METRICS:
            target, reference = grid(candidate, metric), grid(method, metric)
            difference = target - reference
            details = {'candidate_mean': float(target.mean()), 'comparator_mean': float(reference.mean()),
                       'mean_difference': float(difference.mean()),
                       'paired_crossed_bootstrap_ci95': crossed_ci(difference)}
            if metric == 'DSR':
                details['positive_training_seeds'] = int(np.sum(difference.mean(axis=1) > 0))
                details['positive_seed_workload_units'] = int(np.sum(difference > 0))
                if method in learned_comparators:
                    seed_means = difference.mean(axis=1)
                    sem = stats.sem(seed_means)
                    crit = stats.t.ppf(.975, len(seed_means) - 1)
                    details['training_seed_t_ci95'] = [float(seed_means.mean() - crit * sem),
                                                       float(seed_means.mean() + crit * sem)]
                    pvalue = float(stats.ttest_1samp(seed_means, 0).pvalue)
                    details['two_sided_p'] = pvalue
                    raw_p.append((method, pvalue))
            entry['metrics'][metric] = details
        report['comparisons'][method] = entry
    adjusted = holm_adjust([value for _, value in raw_p])
    for (method, _), value in zip(raw_p, adjusted):
        report['comparisons'][method]['metrics']['DSR']['holm_adjusted_p'] = value
    report['decision_rule_met'] = all(
        report['comparisons'][method]['metrics']['DSR']['mean_difference'] > 0
        and report['comparisons'][method]['metrics']['DSR']['holm_adjusted_p'] < .05
        for method in learned_comparators
    )
    report['decision_scope'] = ('DSR relative to all prespecified learned comparators in this '
                                '100-UE simulation; secondary metrics and nonlearned baselines '
                                'do not support an unqualified all-metrics superiority claim')
    write_json(directory / 'whole_system_paired_comparison.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps({'status': analyze(args.directory)['status']}, ensure_ascii=False))
