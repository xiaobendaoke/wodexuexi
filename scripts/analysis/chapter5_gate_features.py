"""Descriptive, pre-proposal public feature audit; no fitted routing model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from chapter5_oracle_gate import csv_write, load_states, read, sha, write

FEATURES = ('batch_size', 'deadline_q25', 'G_pred_success_fraction', 'G_margin_q25',
            'G_max_uav_load_fraction', 'mean_legal_options')


def public_features(rows, greedy_assignment):
    """Only public pre-proposal rows and the already-computed G assignment."""
    n = len(rows)
    assert n == len(greedy_assignment)
    if not n:
        return {k: 0. for k in FEATURES}
    options = [r['options'][str(a)] if str(a) in r['options'] else r['options'][a]
               for r, a in zip(rows, greedy_assignment)]
    targets = [5 if o['target'] == 'mbs' else int(o['target']) for o in options]
    counts = Counter(targets)
    deadlines = np.asarray([float(r['deadline']) for r in rows])
    assert np.all(deadlines > 0)
    latency = np.asarray([float(o['base'])+float(o['work'])*(1 if j == 5 else counts[j])
                          for o, j in zip(options, targets)])
    return dict(batch_size=float(n), deadline_q25=float(np.quantile(deadlines, .25)),
                G_pred_success_fraction=float(np.mean(latency+.010 <= deadlines)),
                G_margin_q25=float(np.quantile((deadlines-latency-.010)/deadlines, .25)),
                G_max_uav_load_fraction=max(counts[j] for j in range(5))/n,
                mean_legal_options=statistics.mean(len(r['options']) for r in rows))


def analyze(a):
    if a.output.exists():
        raise FileExistsError('Preserve prior feature attempts')
    a.output.mkdir(parents=True)
    assert read(a.oracle/'summary.json')['proceed_to_public_feature_diagnosis']
    protocol, states, manifest, _ = load_states(a.input)
    locations = {tuple(s['identity']): s for s in protocol['states']}
    write(a.output/'protocol.json', dict(features=FEATURES, additional_timing_repeats=10,
        plan_sha256=sha(a.plan), source_sha256=sha(__file__), oracle_summary_sha256=sha(a.oracle/'summary.json'),
        pooling='quantile bins over development states; no held-out prediction claim',
        timing='feature function only, public rows and G assignment already available; not full gate cost'))
    records, timings, public_inputs = [], [], {}
    for s in states:
        loc = locations[tuple(s['identity'])]
        path = Path(loc['folder'])/'public.json'
        assert sha(path) == loc['public_sha256']
        public_inputs[str(path)] = loc['public_sha256']
        rows = read(path)['rows']
        g, f = s['methods']['G'], s['methods']['native_B10']
        start = time.perf_counter()
        features = public_features(rows, g['assignment'])
        first = time.perf_counter()-start
        runs = []
        for _ in range(10):
            start = time.perf_counter()
            repeated = public_features(rows, g['assignment'])
            runs.append(time.perf_counter()-start)
            assert repeated == features
        assert all(np.isfinite(v) for v in features.values())
        records.append(dict(block=s['identity'][0], workload=s['identity'][1], source=s['identity'][2],
            step=s['identity'][3], generated=s['generated'], weight=s['weight'],
            delta_success=f['success']-g['success'], **features))
        timings.append(dict(block=s['identity'][0], first_seconds=first,
                            repeat_mean_seconds=statistics.mean(runs)))
    grouped, contrast = [], []
    for feature in FEATURES:
        values = np.asarray([r[feature] for r in records])
        inner = sorted(set(float(v) for v in np.quantile(values, [.2, .4, .6, .8])
                           if values.min() < v < values.max()))
        ids = np.searchsorted(inner, values, side='right')
        assert len(ids) == 1200
        entries = []
        for group in range(len(inner)+1):
            subset = [r for r, j in zip(records, ids) if j == group]
            if not subset:
                continue
            for block in protocol['blocks']:
                selected = [r for r in subset if r['block'] == block]
                denominator = sum(r['weight']*r['generated'] for r in selected)
                entries.append(dict(feature=feature, bin=group+1, block=block, states=len(selected),
                    value_min=min(r[feature] for r in subset), value_max=max(r[feature] for r in subset),
                    dsr_gain_pp=(100*sum(r['weight']*r['delta_success'] for r in selected)/denominator
                                 if denominator else None)))
        grouped.extend(entries)
        block_contrasts = []
        for block in protocol['blocks']:
            b = [r for r in entries if r['block'] == block and r['dsr_gain_pp'] is not None]
            if len(b) >= 2:
                block_contrasts.append(dict(block=block,
                    high_minus_low_pp=b[-1]['dsr_gain_pp']-b[0]['dsr_gain_pp']))
        contrast.append(dict(feature=feature, bins=len(inner)+1, cutpoints=inner, blocks=block_contrasts))
    block_timing = [dict(block=block,
        first_mean_seconds=statistics.mean(r['first_seconds'] for r in timings if r['block'] == block),
        repeat_mean_seconds=statistics.mean(r['repeat_mean_seconds'] for r in timings if r['block'] == block))
        for block in protocol['blocks']]
    csv_write(a.output/'public_features_with_offline_labels.csv', records)
    csv_write(a.output/'feature_bins.csv', grouped)
    csv_write(a.output/'feature_timing.csv', timings)
    write(a.output/'summary.json', dict(status='COMPLETE_DESCRIPTIVE_FEATURE_DIAGNOSIS',
        independent_n=5, states=1200, features=FEATURES, contrasts=contrast,
        block_timing=block_timing,
        first_mean_seconds=statistics.mean(r['first_mean_seconds'] for r in block_timing),
        repeat_mean_seconds=statistics.mean(r['repeat_mean_seconds'] for r in block_timing),
        feature_inputs='public rows and G assignment only; outcomes joined after feature extraction',
        model_trained=False, online_policy_validated=False, held_out_prediction_established=False))
    write(a.output/'input_audit.json', dict(status='PASS', public_inputs=public_inputs,
        sealed_inputs=manifest, all_public_inputs_unchanged=all(sha(p) == h for p, h in public_inputs.items()),
        all_sealed_inputs_unchanged=all(sha(p) == h for p, h in manifest.items())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Scientific analysis executes on server')
    analyze(a)


if __name__ == '__main__':
    main()
