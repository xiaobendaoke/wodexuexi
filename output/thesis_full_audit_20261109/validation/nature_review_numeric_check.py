"""Read-only arithmetic checks for the 2026-11-09 independent thesis review.

No training or simulation is run. Outputs stay beside this task validation script.
"""
from pathlib import Path
from collections import defaultdict
import ast
import csv
import hashlib
import json
import re
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / '.tmp/nature_review_20261109/packet'
DATA = PACKET / 'docs/figures/data'
OUT = Path(__file__).resolve().parent


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    return list(csv.DictReader(p.open(encoding='utf-8-sig', newline='')))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mean_ci(values):
    a = np.asarray(values, float)
    m = a.mean()
    h = t.ppf(.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(m), [float(m-h), float(m+h)]


result = {'scope': 'Static recomputation, not independent simulation replication', 'checks': {}}
packet_manifest = read_json(PACKET / 'REVIEW_PACKET.json')
result['checks']['packet_integrity'] = {
    'packet_mismatch': [x['path'] for x in packet_manifest['manifest'] if digest(PACKET/x['path']) != x['sha256']],
    'workspace_changed_since_snapshot': [x['path'] for x in packet_manifest['manifest'] if not (ROOT/x['path']).exists() or digest(ROOT/x['path']) != x['sha256']],
}
expected_reports = {
    1: 'cd0825b61456f3d86fe282b68e6c7ff70dec342810471e26b9bf0af2ca33f53d',
    2: '485326e2359e45dbe2d41ab4d62f8b2baa35f3445fc5cf0c56893c866fc02e28',
    3: '026c2190510cce18bd6f571e81ebae4e05f5965d06f9898b039fd258a7e01dc2',
}
result['checks']['frozen_reports'] = {str(i): {'sha256': digest(OUT.parent/'deliverables'/f'nature_reviewer_{i}.md'), 'matches_frozen': digest(OUT.parent/'deliverables'/f'nature_reviewer_{i}.md') == h} for i, h in expected_reports.items()}

c3 = DATA/'chapter34_20261004'
u3 = rows(c3/'confirmation_units.csv')
j3 = read_json(c3/'confirmation_paired_comparison.json')
groups3 = defaultdict(dict)
for row in u3:
    groups3[(row['training_seed'], row['workload_seed'])][row['method']] = row
metrics3 = {}
for metric, ref in j3['metrics'].items():
    differences = defaultdict(list)
    bs, cs = [], []
    for (seed, workload), methods in groups3.items():
        b = float(methods['lower_no_attention'][metric])
        c = float(methods['lower_mlp_upper_uncovered_obs'][metric])
        bs.append(b); cs.append(c); differences[seed].append(c-b)
    seed_diffs = [np.mean(v) for seed, v in sorted(differences.items(), key=lambda kv: int(kv[0]))]
    m, ci = mean_ci(seed_diffs)
    metrics3[metric] = {'baseline_mean': float(np.mean(bs)), 'candidate_mean': float(np.mean(cs)), 'difference': m, 'seed_t_ci95': ci, 'tolerance': 1e-9, 'matches_json': bool(np.allclose([np.mean(bs), np.mean(cs), m, *ci], [ref['baseline_mean'], ref['candidate_mean'], ref['mean_difference'], *ref['training_seed_t_ci95']], atol=1e-9, rtol=0))}
result['checks']['chapter3'] = {'rows': len(u3), 'pairs': len(groups3), 'metrics': metrics3, 'pooled_count_identities': {k: bool(v['deadline_satisfied_count'] <= v['admitted_service_count'] <= v['generated_service_count'] and abs(v['pooled_dsr']-v['deadline_satisfied_count']/v['generated_service_count']) < 1e-12) for k,v in j3['pooled_totals'].items()}}

u4 = rows(c3/'chapter4_units.csv')
j4 = read_json(c3/'chapter4_analysis.json')
indexed4 = {(x['variant'], int(x['workload_seed']), x['method']): x for x in u4}
comp4 = []
for comp in j4['comparisons']:
    diffs = []
    for w in sorted({int(x['workload_seed']) for x in u4 if x['variant'] == comp['variant']}):
        a, b = indexed4[(comp['variant'], w, 'mc_itr')], indexed4[(comp['variant'], w, comp['reference'])]
        diffs.append([float(a[m])-float(b[m]) for m in j4['metrics']])
    comp4.append(bool(np.allclose(np.mean(diffs, axis=0), comp['mean_difference'], atol=1e-12, rtol=0)))
result['checks']['chapter4'] = {
    'rows': len(u4), 'unique_snapshot_count': sum(int(v['snapshots']) for v in {(x['variant'], x['workload_seed']): x for x in u4}.values()),
    'main_means': {method: {m: float(np.mean([float(x[m]) for x in u4 if x['variant']=='main' and x['method']==method])) for m in j4['metrics']} for method in sorted({x['method'] for x in u4 if x['variant']=='main'})},
    'comparison_point_estimates_match_json': all(comp4),
    'all_mcitr_units_local_optimality_verified': all(float(x['local_optimality_verified_fraction']) == 1 for x in u4 if x['method']=='mc_itr'),
}
bridge = read_json(c3/'bridge_analysis.json')
bd = defaultdict(list)
for x in bridge['units']:
    bd[x['training_seed']].append(x['mc_itr']['deadline_misses']-x['online_lower']['deadline_misses'])
bm, bci = mean_ci([np.mean(v) for v in bd.values()])
result['checks']['bridge'] = {'snapshots': sum(x['snapshots'] for x in bridge['units']), 'training_seeds': bridge['training_seeds'], 'workload_seeds': bridge['workload_seeds'], 'online_misses': float(np.mean([x['online_lower']['deadline_misses'] for x in bridge['units']])), 'mcitr_misses': float(np.mean([x['mc_itr']['deadline_misses'] for x in bridge['units']])), 'paired_difference': bm, 'seed_t_ci95': bci}

c5 = DATA/'queued_joint_rl_20261106'
u5, s5, e5 = (rows(c5/n) for n in ['confirmation_units.csv','confirmation_summary.csv','paired_dsr_effects.csv'])
summaries, arrays = {}, {}
for arm in sorted({x['arm'] for x in u5}):
    rr = [x for x in u5 if x['arm']==arm]
    g,a,s = [sum(int(x[k]) for x in rr) for k in ['generated','admitted','success']]
    parents = {str(p): {k: sum(int(x[k]) for x in rr if int(x['parent'])==p) for k in ['generated','admitted','success']} for p in range(3)}
    summaries[arm] = dict(episodes=len(rr),generated=g,admitted=a,success=s,dsr=s/g,admission=a/g,conditional=s/a,by_parent=parents)
    aa = {k: np.zeros((3,8,2)) for k in ['generated','admitted','success']}
    for x in rr:
        idx = (int(x['parent']),int(x['workload'])-955000,int(x['replica']))
        for k in aa: aa[k][idx]=int(x[k])
    arrays[arm]=aa
summary_matches = []
for x in s5:
    s = summaries[x['arm']]
    summary_matches.append(all(abs(float(x[k])-s[k])<1e-12 for k in ['generated','admitted','success','dsr','admission','conditional']))
src = (PACKET/'scripts/analysis/review_queued_joint_rl.py').read_text(encoding='utf-8')
func = next(x for x in ast.parse(src).body if isinstance(x, ast.FunctionDef) and x.name=='paired_bootstrap')
ns = dict(np=np, BOOTSTRAP_DRAWS=10000, PRIMARY_FAMILY_SIZE=8, COMPARISON_PAIRS=list(range(31)))
exec(compile(ast.Module(body=[func], type_ignores=[]), '<frozen paired_bootstrap only>', 'exec'),ns)
effects=[]
for row in e5:
    a,b=row['pair'].split('--'); aa,bb=arrays[a],arrays[b]
    z=ns['paired_bootstrap'](aa['success'],aa['generated'],bb['success'],bb['generated'])
    calculated=[z['mean']*100,*[v*100 for v in z['ci95']],*[v*100 for v in z['bonferroni_all_comparisons95']]]
    recorded=[float(row[k]) for k in ['effect_pp','ci95_low_pp','ci95_high_pp','bonf31_low_pp','bonf31_high_pp']]
    effects.append({'pair':row['pair'],'calculated_pp':calculated,'matches_csv':bool(np.allclose(calculated,recorded,atol=1e-10,rtol=0))})
ref_g=next(iter(arrays.values()))['generated']
result['checks']['chapter5'] = {'rows':len(u5),'unique_keys':len({(x['arm'],x['parent'],x['workload'],x['replica']) for x in u5}), 'all_counts_bounded':all(0<=int(x['success'])<=int(x['admitted'])<=int(x['generated']) for x in u5), 'same_generated_per_paired_episode':all(np.array_equal(x['generated'],ref_g) for x in arrays.values()), 'all_17_summary_rows_match':all(summary_matches),'summaries':summaries,'all_31_effects_and_intervals_match':all(x['matches_csv'] for x in effects),'effects':effects,'wall_time_reduction_percent':(1-float(next(x for x in s5 if x['arm']=='joint_rl_mpc')['decision_mean_s'])/float(next(x for x in s5 if x['arm']=='quality_amortized')['decision_mean_s']))*100}

freeze=read_json(ROOT/'results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106/freeze_manifest.json')
source_check={f: {'frozen_sha256':h,'snapshot_sha256':digest(PACKET/f) if (PACKET/f).exists() else None} for f,h in freeze['source_sha256'].items()}
result['checks']['chapter5_frozen_source_hashes'] = source_check
manifest5=read_json(c5/'source_manifest.json')
result['checks']['chapter5_data_source_hashes']={'checked':len(manifest5['sources']),'mismatches':[f for f,h in manifest5['sources'].items() if not (ROOT/f).exists() or digest(ROOT/f)!=h]}

latex_files=list((PACKET/'latex/docs').rglob('*.tex')) + [PACKET/'latex/main.tex',PACKET/'latex/sysusetup.tex']
joined='\n'.join(p.read_text(encoding='utf-8') for p in latex_files)
labels=re.findall(r'\\label\{([^}]+)\}',joined)
refs=re.findall(r'\\(?:eqref|ref|autoref)\{([^}]+)\}',joined)
bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',(PACKET/'latex/reference.bib').read_text(encoding='utf-8')))
cites=[k.strip() for grp in re.findall(r'\\cite\w*\s*(?:\[[^\]]*\])?\{([^}]+)\}',joined) for k in grp.split(',')]
result['checks']['local_reference_resolution']={'unresolved_labels':sorted(set(refs)-set(labels)), 'duplicate_labels':sorted({x for x in labels if labels.count(x)>1}), 'missing_citation_keys':sorted(set(cites)-bibkeys), 'distinct_cited_keys':len(set(cites)), 'boundary':'Local key resolution only. External bibliography facts not verified.'}
(OUT/'nature_review_numeric_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'chapter3_metrics_match':all(x['matches_json'] for x in metrics3.values()),'chapter4_comparison_points_match':all(comp4),'chapter5_summaries_match':all(summary_matches),'chapter5_all31_intervals_match':all(x['matches_csv'] for x in effects),'packet_integrity':result['checks']['packet_integrity'],'frozen_reports':result['checks']['frozen_reports'],'local_reference_resolution':result['checks']['local_reference_resolution']},ensure_ascii=False,indent=2))
