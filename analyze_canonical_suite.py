"""Complete-matrix aggregation; paired training-seed effects, never episode pseudoreplication."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from utils.canonical_experiment import write_json
from utils.formal_protocol import NONLEARNED


def read_unit(directory, episodes, source_hash):
    done = json.loads((directory / 'completed.json').read_text())
    for rel, expected in done['hashes'].items():
        if hashlib.sha256((directory / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Altered result: {directory / rel}')
    attempt = directory / done['attempt']
    manifest = json.loads((attempt / 'manifest.json').read_text())
    if manifest['source']['source_hash'] != source_hash:
        raise ValueError('Mixed source versions')
    files = sorted((attempt / 'episodes').glob('*.json'))
    if len(files) != episodes:
        raise ValueError('Missing or extra evaluation episodes')
    raw = [json.loads(p.read_text()) for p in files]
    rows = [r['metrics'] for episode in raw for r in episode['steps']]
    generated = sum(r['generated_service_count'] for r in rows)
    success = sum(r['deadline_satisfied_count'] for r in rows)
    energy = sum(r['uav_fleet_total_energy'] for r in rows)
    admitted = sum(r['admitted_service_count'] for r in rows)
    return {'DSR': success / generated if generated else 0.,
            'UAV Fleet Total Energy': energy / episodes,
            'Critical-UE Ratio': float(np.mean([r['offline_rate'] for r in rows])),
            'Latency per generated request': sum(r['latency'] for r in rows) / generated if generated else 0.,
            'Fairness': float(np.mean([r['fairness'] for r in rows])),
            'Effective tasks per MJ': success * 1e6 / energy if energy else 0.,
            'MBS offload fraction': sum(r['service_offloads_mbs'] for r in rows) / admitted if admitted else 0.}, [r['exogenous_trace_hash'] for r in raw]


def paired_effect(differences, draws=5000):
    """Input [training seeds, paired workloads]; CI resamples both axes together."""
    values = np.asarray(differences, dtype=float)
    seed_means = values.mean(axis=1)
    result = {'mean_difference': float(values.mean()), 'n_training_seeds': len(values),
              'n_workloads': values.shape[1], 'ci95': None, 'paired_seed_dz': None,
              'ci_method': 'paired crossed bootstrap over training seeds and workload seeds'}
    if len(values) < 2 or values.shape[1] < 2:
        result['inference'] = 'insufficient independent repeats'
        return result
    rng = np.random.default_rng(20260920)
    ti = rng.integers(0, len(values), size=(draws, len(values)))
    wi = rng.integers(0, values.shape[1], size=(draws, values.shape[1]))
    means = values[ti[:, :, None], wi[:, None, :]].mean(axis=(1, 2))
    result['ci95'] = np.quantile(means, [.025, .975]).tolist()
    sd = seed_means.std(ddof=1)
    if sd > 0:
        result['paired_seed_dz'] = float(seed_means.mean() / sd)
    result['inference'] = 'CI and effect size primary; only five training replicates under default formal protocol'
    return result


def analyze(output):
    output = Path(output)
    suite = json.loads((output / 'suite.json').read_text())
    plan = suite['plan']
    units, comparisons, curves = [], [], []
    for variant in plan['variants']:
        data = {}
        expected_traces = {}
        for method in variant['methods']:
            data[method] = []
            seeds = [None] if method in NONLEARNED else plan['training_seeds']
            for seed in seeds:
                row = []
                base = output / variant['name'] / method
                if seed is not None:
                    base /= f'seed_{seed}'
                    done = json.loads((base / 'train/completed.json').read_text())
                    for path in sorted((base / 'train' / done['attempt'] / 'episodes').glob('*.json')):
                        ep = json.loads(path.read_text())
                        curves.append({'variant': variant['name'], 'method': method, 'training_seed': seed,
                                       'episode': ep['episode'], 'training_DSR': ep['request_weighted_dsr'],
                                       'mean_training_reward': float(np.mean([r['team_reward'] for r in ep['steps']]))})
                for workload in plan['workload_seeds']:
                    metrics, trace = read_unit(base / f'workload_{workload}', plan['eval_episodes'], suite['source_hash'])
                    if workload in expected_traces and trace != expected_traces[workload]:
                        raise ValueError('Unpaired exogenous evaluation scenarios')
                    expected_traces[workload] = trace
                    row.append(metrics)
                    units.append({'variant': variant['name'], 'method': method, 'training_seed': seed,
                                  'workload_seed': workload, **metrics})
                data[method].append(row)
        if 'hierarchical' in data:
            for method in variant['methods']:
                if method == 'hierarchical':
                    continue
                for metric in data['hierarchical'][0][0]:
                    target = np.array([[u[metric] for u in row] for row in data['hierarchical']])
                    baseline = np.array([[u[metric] for u in row] for row in data[method]])
                    comparisons.append({'variant': variant['name'], 'reference': 'hierarchical',
                                        'comparison': method, 'metric': metric,
                                        'sign': 'hierarchical minus comparison', **paired_effect(target - baseline)})
    report = {'status': 'PASS', 'purpose': plan['purpose'], 'units': units, 'comparisons': comparisons,
              'statistical_unit': plan['statistical_unit'], 'source_hash': suite['source_hash']}
    write_json(output / 'analysis.json', report)
    for name, rows in [('evaluation_units.csv', units), ('training_curves.csv', curves)]:
        with (output / name).open('w', newline='') as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    print(analyze(parser.parse_args().output)['status'])
