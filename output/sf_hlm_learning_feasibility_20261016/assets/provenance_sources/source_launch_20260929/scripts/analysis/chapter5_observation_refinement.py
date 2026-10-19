"""Paired exploratory screen for Chapter 5 observation variants.

This analysis is separate from the frozen training source. Its own SHA-256 is
recorded in the output so postprocessing cannot be confused with the source
hash used by the training and evaluation jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.analysis.whole_system_comparison import crossed_ci
from utils.canonical_experiment import write_json


METRICS = (
    'DSR', 'Admission fraction', 'Deadline satisfaction among admitted',
    'UAV Fleet Total Energy', 'Non-flight assignment energy', 'Fairness',
    'Effective tasks per MJ', 'MBS offload fraction',
)


def analyze(root: Path) -> dict:
    root = Path(root)
    suite = json.loads((root / 'suite.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    analysis = json.loads((root / 'analysis.json').read_text())
    plan = suite['plan']
    if status.get('status') != 'PASS' or analysis.get('status') != 'PASS':
        raise ValueError('Pilot suite or analysis did not pass')
    if plan['protocol'] != 'chapter5_observation_refinement_pilot_20260928_v1':
        raise ValueError('Unexpected protocol')
    if analysis['source_hash'] != suite['source_hash']:
        raise ValueError('Analysis source hash differs from suite')
    if len(plan['variants']) != 1 or plan['variants'][0]['name'] != 'main':
        raise ValueError('Expected one main scenario')
    methods = plan['variants'][0]['methods']
    reference = plan['candidate_method']
    seeds, workloads = plan['training_seeds'], plan['workload_seeds']
    if reference not in methods or len(methods) != 3:
        raise ValueError('Expected the original candidate and two observation variants')
    units = analysis['units']
    expected = len(methods) * len(seeds) * len(workloads)
    lookup = {(u['method'], u['training_seed'], u['workload_seed']): u for u in units}
    if len(units) != expected or len(lookup) != expected:
        raise ValueError('Missing or duplicate evaluation units')

    def grid(method: str, metric: str) -> np.ndarray:
        try:
            values = np.asarray([[lookup[(method, seed, workload)][metric]
                                  for workload in workloads] for seed in seeds], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite metric: {method}, {metric}')
            return values
        except KeyError as exc:
            raise ValueError(f'Missing paired observation: {exc}') from exc

    comparisons = {}
    for method in methods:
        if method == reference:
            continue
        metrics = {}
        for metric in METRICS:
            candidate = grid(method, metric)
            baseline = grid(reference, metric)
            difference = candidate - baseline
            row = {'candidate_mean': float(candidate.mean()),
                   'reference_mean': float(baseline.mean()),
                   'mean_difference': float(difference.mean()),
                   'paired_crossed_bootstrap_ci95': crossed_ci(difference)}
            if metric == 'DSR':
                seed_means = difference.mean(axis=1)
                critical = stats.t.ppf(.975, len(seeds) - 1)
                sem = stats.sem(seed_means)
                row.update({'training_seed_differences': seed_means.tolist(),
                            'positive_training_seeds': int(np.sum(seed_means > 0)),
                            'training_seed_t_ci95': [float(seed_means.mean() - critical * sem),
                                                     float(seed_means.mean() + critical * sem)]})
            metrics[metric] = row
        dsr = metrics['DSR']
        energy = metrics['UAV Fleet Total Energy']
        qualifies = (dsr['mean_difference'] > 0 and dsr['positive_training_seeds'] >= 4
                     and energy['candidate_mean'] <= 1.05 * energy['reference_mean'])
        comparisons[method] = {'metrics': metrics, 'pilot_promotion_criteria_met': bool(qualifies)}
    promoted = [m for m, row in comparisons.items() if row['pilot_promotion_criteria_met']]
    selected = max(promoted, key=lambda m: comparisons[m]['metrics']['DSR']['mean_difference']) if promoted else None
    report = {'status': 'PASS', 'protocol': plan['protocol'], 'training_source_hash': suite['source_hash'],
              'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference_method': reference, 'training_seeds': seeds, 'workload_seeds': workloads,
              'statistical_unit': 'training seed after averaging five paired workloads',
              'comparisons': comparisons, 'pilot_selected_for_independent_confirmation': selected,
              'interpretation_boundary': 'Exploratory screen only; selected variant requires new-seed confirmation.'}
    write_json(root / 'observation_refinement_paired_comparison.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps({'status': analyze(parser.parse_args().directory)['status']}))
