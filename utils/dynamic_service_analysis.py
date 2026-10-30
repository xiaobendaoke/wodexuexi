"""Complete-case, training-block analysis; never treats requests as replicates."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats

METHODS = ('full', 'no_memory', 'no_feedback', 'parallel', 'independent',
           'greedy_lower', 'matching_lower', 'history', 'mpc')
MAIN = ('history', 'mpc', 'independent')
MECHANISMS = ('no_memory', 'no_feedback', 'parallel', 'greedy_lower', 'matching_lower')


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, previous = [0.]*len(pvalues), 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order)-rank)*pvalues[index]))
        adjusted[index] = previous
    return adjusted


def contrast(values):
    values = np.asarray(values, float)
    mean, sem = float(values.mean()), float(stats.sem(values))
    margin = float(stats.t.ppf(.975, len(values)-1))*sem
    if sem == 0:
        p = 1. if mean == 0 else 0.
    else:
        p = float(stats.ttest_1samp(values, 0).pvalue)
    return dict(mean=mean, ci95=[mean-margin, mean+margin], p=p,
                positive_blocks=int((values > 0).sum()), differences=values.tolist())


def analyze(root, split='dev', primary_only=False):
    root = Path(root)
    n = 5 if split == 'dev' else 10
    cells = [('D', 1.)] if primary_only else [(k, s) for k in 'ABCD' for s in (.6, 1., 1.4)]
    results, hashes, missing = {}, {}, []
    contract = None
    for kind, scale in cells:
        cell = f'{kind}_{scale}'
        results[cell] = {}
        for method in METHODS:
            blocks, details = [], []
            for b in range(n):
                total = {k: 0 for k in ('generated', 'admitted', 'success', 'mbs', 'decision_timeout')}
                timing, fairness = [], []
                source_block = 0 if method in ('history', 'mpc') else b
                for w in range(5):
                    for e in range(6):
                        path = root/'evaluation'/split/str(source_block)/method/f'{kind}_{scale}_{w}_{e}.json'
                        if not path.exists():
                            missing.append(str(path)); continue
                        raw = path.read_bytes()
                        data = json.loads(raw)
                        if data['smoke'] or len(data['rows']) != 1000:
                            raise ValueError(f'non-formal evaluation: {path}')
                        identity = (data['split'], data['block'], data['method'], data['kind'], data['scale'], data['workload'], data['repeat'])
                        if identity != (split, source_block, method, kind, scale, w, e):
                            raise ValueError(f'evaluation identity mismatch: {path}')
                        if contract is not None and data['protocol'] != contract:
                            raise ValueError('mixed source/protocol results')
                        contract = data['protocol']
                        for key in total:
                            counted = sum(r[key] for r in data['rows'])
                            if counted != data['totals'][key]:
                                raise ValueError('raw counter mismatch')
                            total[key] += counted
                        if data['totals']['generated'] != 50000:
                            raise ValueError('external denominator changed')
                        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
                        timing.extend(r['seconds'] for r in data['rows'])
                        fairness.append(data['fairness'])
                if total['generated']:
                    blocks.append(total['success']/total['generated'])
                    details.append(dict(**total, admission=total['admitted']/total['generated'],
                                        conditional=total['success']/max(total['admitted'], 1),
                                        p95=float(np.percentile(timing, 95)), fairness=float(np.mean(fairness))))
            results[cell][method] = dict(blocks=blocks, details=details)
    if missing:
        return dict(status='INCOMPLETE', missing=sorted(set(missing)), completed_files=len(hashes))
    comparisons = {}
    for cell, methods in results.items():
        comparisons[cell] = {}
        for family, controls in (('main', MAIN), ('mechanisms', MECHANISMS)):
            rows = {control: contrast(np.array(methods['full']['blocks'])-methods[control]['blocks']) for control in controls}
            for control, adjusted in zip(controls, holm([rows[c]['p'] for c in controls])):
                rows[control]['holm_p'] = adjusted
                rows[control]['supports_positive_effect'] = rows[control]['ci95'][0] > 0 and adjusted < .05
                rows[control]['passes_main_margin'] = rows[control]['supports_positive_effect'] and rows[control]['mean'] >= .01
            comparisons[cell][family] = rows
    interactions = {}
    if not primary_only:
        for scale in (.6, 1., 1.4):
            for control in MAIN:
                def delta(kind):
                    methods = results[f'{kind}_{scale}']
                    return np.array(methods['full']['blocks'])-methods[control]['blocks']
                interactions[f'{control}_{scale}'] = contrast(delta('D')-delta('A'))
        keys = list(interactions)
        for key, p in zip(keys, holm([interactions[k]['p'] for k in keys])):
            interactions[key]['holm_p'] = p
    gate_controls = ('history', 'independent', 'no_memory', 'no_feedback', 'parallel')
    primary = results['D_1.0']
    evidence = {c: contrast(np.array(primary['full']['blocks'])-primary[c]['blocks']) for c in gate_controls}
    passed = all(r['mean'] > 0 and r['positive_blocks'] >= 4 for r in evidence.values())
    return dict(status='COMPLETE', split=split, protocol=contract, results=results, comparisons=comparisons,
                interactions=interactions, file_hashes=hashes,
                development_gate=dict(status='PASS' if passed else 'STOPPED_DEVELOPMENT_GATE',
                                      exploratory=True, comparisons=evidence),
                primary_performance_pass=all(r['passes_main_margin'] for r in comparisons['D_1.0']['main'].values()))
