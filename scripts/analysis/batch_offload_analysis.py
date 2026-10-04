"""Raw-verified promotion and prespecified confirmation; never launch jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats
from analyze_canonical_suite import analyze as aggregate
from scripts.analysis.whole_system_comparison import crossed_ci, holm_adjust, METRICS
from scripts.analysis.batch_offload_protocol import A, B, C, D, PARTNERS, validate_plan, select_candidate, write_new
from utils.formal_protocol import verify_completed_job


def effect(candidate, reference):
    delta = candidate-reference
    means = delta.mean(1); average = float(means.mean())
    sem = float(stats.sem(means))
    critical = float(stats.t.ppf(.975, len(means)-1))
    pvalue = float(stats.ttest_1samp(means, 0).pvalue) if sem > 0 else (1. if average == 0 else 0.)
    return {'candidate_mean': float(candidate.mean()), 'reference_mean': float(reference.mean()),
            'mean_difference': average, 'training_seed_differences': means.tolist(),
            'positive_training_seeds': int((means > 0).sum()),
            'training_seed_t_ci95': [average-critical*sem, average+critical*sem],
            'paired_crossed_bootstrap_ci95': crossed_ci(delta), 'two_sided_p': pvalue}


def analyze(root):
    root = Path(root)
    suite = json.loads((root/'suite.json').read_text()); plan = suite['plan']
    validate_plan(plan)
    if plan['stage'] == 'smoke':
        raise ValueError('Smoke data cannot promote a candidate')
    if json.loads((root/'status.json').read_text()).get('status') != 'PASS':
        raise ValueError('Incomplete suite')
    # Rebuild metrics from checksum-verified raw episodes, not a supplied summary.
    analysis = aggregate(root)
    seeds, workloads = plan['training_seeds'], plan['workload_seeds']
    methods = plan['variants'][0]['methods']
    lookup = {(u['method'], u['training_seed'], u['workload_seed']): u for u in analysis['units']}
    if len(lookup) != len(methods)*len(seeds)*len(workloads):
        raise ValueError('Incomplete or duplicate grid')
    def grid(method, metric):
        values = np.array([[lookup[method,s,w][metric] for w in workloads] for s in seeds])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite metric')
        return values
    timing, totals, hardware = {}, {}, set()
    for method in methods:
        times = []; generated = admitted = successful = 0
        for seed in seeds:
            base = root/'main'/method/f'seed_{seed}'
            trained = verify_completed_job(base/'train', suite['source_hash'])
            latest = json.loads((trained/'latest.json').read_text())
            if latest['completed_episodes'] != plan['train_episodes'] or latest['checkpoint'] != f'checkpoints/episode_{plan["train_episodes"]-1:05d}':
                raise ValueError('Not the prespecified final checkpoint')
            for workload in workloads:
                attempt = verify_completed_job(base/f'workload_{workload}', suite['source_hash'])
                manifest = json.loads((attempt/'manifest.json').read_text())
                if manifest.get('evaluation_checkpoint_contract_sha256') != hashlib.sha256((trained/latest['checkpoint']/'contract.json').read_bytes()).hexdigest():
                    raise ValueError('Evaluation did not identify the final training checkpoint')
                runtime = manifest['runtime']
                keys = ('host','machine','threads','device','timing','torch','python')
                if any(k not in runtime for k in keys) or runtime['timing'] != plan['timing_contract']:
                    raise ValueError('Missing comparable timing metadata')
                hardware.add(tuple(str(runtime[k]) for k in keys))
                for path in sorted((attempt/'episodes').glob('*.json')):
                    ep = json.loads(path.read_text())
                    if len(ep['steps']) != plan['steps']:
                        raise ValueError('Incomplete episode')
                    for row in ep['steps']:
                        t = float(row['policy_inference_seconds'])
                        if not np.isfinite(t) or t < 0:
                            raise ValueError('Invalid timing')
                        times.append(t); m = row['metrics']
                        g,a,s = (m[k] for k in ('generated_service_count','admitted_service_count','deadline_satisfied_count'))
                        if not 0 <= s <= a <= g:
                            raise ValueError('Request counts do not close')
                        generated += g; admitted += a; successful += s
        timing[method] = {'p95_seconds': float(np.quantile(times,.95)), 'mean_seconds': float(np.mean(times)), 'n_steps': len(times)}
        totals[method] = {'generated': generated, 'admitted': admitted, 'successful': successful,
                          'DSR': successful/generated if generated else 0.,
                          'admission': admitted/generated if generated else 0.,
                          'quality': successful/admitted if admitted else 0.}
    if len(hardware) != 1:
        raise ValueError('Inference timings use different hardware/runtime settings')
    comparisons = {}
    if plan['stage'] == 'pilot':
        pairs = [(B,A), (C,A), (D,A)]
    else:
        chosen = plan['candidate_method']; pairs = [(chosen,A), (chosen,PARTNERS[chosen]), (PARTNERS[chosen],D)]
    for candidate, reference in pairs:
        key = candidate if plan['stage'] == 'pilot' else candidate+' minus '+reference
        comparisons[key] = {'candidate': candidate, 'reference': reference,
                            'metrics': {m: effect(grid(candidate,m),grid(reference,m)) for m in METRICS}}
    report = {'status': 'PASS', 'stage': plan['stage'], 'protocol': plan['protocol'], 'source_hash': suite['source_hash'],
              'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'comparisons': comparisons, 'timing': timing, 'count_weighted_decomposition': totals,
              'statistical_unit': plan['statistical_unit'], 'selected_for_confirmation': None}
    if plan['stage'] == 'pilot':
        report['selected_for_confirmation'] = select_candidate(comparisons,timing)
        report['interpretation'] = 'Exploratory selection only; no confirmed gain.'
    else:
        primary = list(comparisons)[:2]
        corrected = holm_adjust([comparisons[k]['metrics']['DSR']['two_sided_p'] for k in primary])
        for key, p in zip(primary,corrected):
            row = comparisons[key]['metrics']['DSR']; row['holm_adjusted_p'] = p
            row['confirmed_positive'] = bool(row['mean_difference'] > 0 and p < .05)
        report['replace_current_method_supported'] = comparisons[primary[0]]['metrics']['DSR']['confirmed_positive']
        report['same_information_upper_advantage_supported'] = comparisons[primary[1]]['metrics']['DSR']['confirmed_positive']
        interaction = grid(chosen,'DSR') - grid(A,'DSR') - grid(PARTNERS[chosen],'DSR') + grid(D,'DSR')
        report['secondary_interaction'] = effect(interaction,np.zeros_like(interaction))
        report['interpretation'] = 'Only prespecified paired main-scenario comparisons; no all-baselines claim.'
    write_new(root/'batch_offload_analysis.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path)
    print(json.dumps(analyze(parser.parse_args().root),ensure_ascii=False))
