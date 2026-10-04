"""Audit archived repair diagnostics without importing or running the simulator.

Exploratory paired effects use five training-seed means (40 snapshots each).
The four deadline-versus-ordinary miss contrasts form one Holm family.
"""
from pathlib import Path
import argparse, csv, hashlib, itertools, json, math, platform, statistics, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(x): return statistics.mean(x)
def p95(x): return sorted(x)[max(0, math.ceil(.95*len(x))-1)]
def close(a,b): return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)
def paired(values):
    assert len(values)==5
    m=mean(values); se=statistics.stdev(values)/math.sqrt(5)
    t=m/se if se else None
    if t is not None:
        x=abs(t)/math.sqrt(t*t+4)
        p=max(0.,min(1.,1-1.5*x+.5*x**3))  # exact two-sided Student t, df=4
    else: p=1. if m==0 else 0.
    flips=[abs(mean([s*v for s,v in zip(signs,values)])) for signs in itertools.product((-1,1),repeat=5)]
    return {'seed_values':values,'mean':m,'ci95':[m-2.7764451051977987*se,m+2.7764451051977987*se],
            'n':5,'df':4,'t':t,'p_two_sided':p,'exact_sign_flip_p':sum(v>=abs(m)-1e-12 for v in flips)/32,
            'negative_seeds':sum(v<0 for v in values),'positive_seeds':sum(v>0 for v in values)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=REPO_ROOT/'results/deadline_repair_20260929/full')
    p.add_argument('--output',type=Path,default=REPO_ROOT/'results/deadline_repair_20260929/review')
    p.add_argument('--validation',type=Path,default=REPO_ROOT/'output/deadline_repair_20260929/validation')
    args=p.parse_args(); root=args.input
    protocol=json.loads((root/'protocol.json').read_text())
    status=json.loads((root/'status.json').read_text())
    assert status['phase']=='COMPLETE' and status['completed_jobs']==50 and not status['failed_jobs']
    manifest=json.loads((args.validation/'remote_result_manifest.json').read_text())
    assert {str(f.relative_to(root)):sha(f) for f in root.rglob('*.json')}==manifest
    expected={tuple(j) for j in protocol['jobs']}; found=set(); rows=[]; prediction_by_action={}
    methods=sorted({j[0] for j in expected}); variants=['baseline','ordinary_32','deadline_32','ordinary_128','deadline_128']
    seeds=sorted({j[1] for j in expected})
    for path in sorted(root.glob('*/*.json')):
        d=json.loads(path.read_text()); key=(d['reference_method'],d['training_seed'],d['workload_seed'])
        assert key in expected and key not in found; found.add(key)
        assert d['status']=='PASS' and d['verification_scope']=='full_episodes'
        for a,b in [('source_hash','source_hash'),('script_sha256','driver_sha256'),('repair_module_sha256','repair_sha256')]: assert d[a]==protocol[b]
        assert len(d['replay'])==2 and len(d['snapshots'])==8
        assert {(s['episode'],s['step']) for s in d['snapshots']}==set(itertools.product(range(2),[0,250,500,750]))
        for snap in d['snapshots']:
            n=snap['request_count']; requests=snap['requests']; deadlines=[r['deadline'] for r in requests]
            assert n==len(requests) and len({r['ue_id'] for r in requests})==n
            r=snap['bridge']['deadline_repair']; base=r['baseline_actual']; bp=r['baseline_predicted']
            assert r['information_contract']=='centralized_proposal_pool_with_cache_beliefs_v1'
            assert r['candidate_contract']=='unchanged_quality_mask'
            assert r['initial']==[q['action'] for q in requests]
            assert base['misses']==sum(not q['success'] for q in requests)
            assert close(base['assignment_energy'],sum(q['assignment_energy'] for q in requests))
            assert base['latencies']==[q['latency'] for q in requests]
            assert set(r['comparisons'])==set(variants[1:])
            for variant in variants:
                item=r['comparisons'].get(variant)
                a=item['actual'] if item else base; pred=item['predicted'] if item else bp
                assignment=item['assignment'] if item else r['initial']
                assert len(assignment)==len(a['latencies'])==len(pred['latencies'])==n
                assert all(type(v)is int and 0<=v<=6 for v in assignment)
                assert a['misses']==sum(t>dl for t,dl in zip(a['latencies'],deadlines))
                assert pred['misses']==sum(t>dl for t,dl in zip(pred['latencies'],deadlines))
                assert a['mbs_count']==assignment.count(1)
                assert all(math.isfinite(v) and v>=0 for v in a['latencies']+pred['latencies'])
                assert close(pred['latency_ratio'],sum(t/dl for t,dl in zip(pred['latencies'],deadlines))/max(n,1))
                if item:
                    assert item['evaluations']<=item['budget']==int(variant.split('_')[1])
                    assert item['baseline_predicted']==bp
                    assert pred['energy']<=bp['energy']*1.05+1e-8
                    assert tuple(pred['objective'])<=tuple(bp['objective'])
                    assert close(item['proposal_path_seconds'],r['preparation_seconds']+item['search_seconds'])
                    assert item['actual_miss_difference']==a['misses']-base['misses']
                errors=[x-y for x,y in zip(a['latencies'],pred['latencies'])]
                for action,error,pt,at,dl in zip(assignment,errors,pred['latencies'],a['latencies'],deadlines):
                    role='local' if action==0 else 'mbs' if action==1 else 'cooperative'
                    g=prediction_by_action.setdefault(key[0]+'/'+variant+'/'+role,{'requests':0,'error_sum':0.,'absolute_error_sum':0.,'optimistic_success_errors':0})
                    g['requests']+=1;g['error_sum']+=error;g['absolute_error_sum']+=abs(error)
                    g['optimistic_success_errors']+=int(pt<=dl<at)
                row={'method':key[0],'seed':key[1],'workload':key[2],'episode':snap['episode'],'step':snap['step'],
                     'snapshot_hash':snap['snapshot_hash'],'variant':variant,'requests':n,'misses':a['misses'],
                     'energy_j':a['assignment_energy'],'mbs':a['mbs_count'],'miss_delta':a['misses']-base['misses'],
                     'energy_delta_j':a['assignment_energy']-base['assignment_energy'],'mbs_delta':a['mbs_count']-base['mbs_count'],
                     'energy_guard_violation':int(a['assignment_energy']>base['assignment_energy']*1.05+1e-8),
                     'energy_ratio':a['assignment_energy']/base['assignment_energy'] if base['assignment_energy'] else 1.,
                     'pred_misses':pred['misses'],'pred_energy_j':pred['energy'],
                     'miss_prediction_error':a['misses']-pred['misses'],'latency_error_sum':sum(errors),
                     'latency_abs_error_sum':sum(abs(x) for x in errors),
                     'pred_success_actual_miss':sum(p<=dl<t for p,t,dl in zip(pred['latencies'],a['latencies'],deadlines)),
                     'pred_miss_actual_success':sum(t<=dl<p for p,t,dl in zip(pred['latencies'],a['latencies'],deadlines)),
                     'pred_improve_actual_worse':int(pred['misses']<bp['misses'] and a['misses']>base['misses']),
                     'changed_actions':sum(x!=y for x,y in zip(assignment,r['initial'])),
                     'preparation_seconds':r['preparation_seconds'],
                     'search_seconds':item['search_seconds'] if item else 0.,
                     'path_seconds':item['proposal_path_seconds'] if item else 0.,
                     'evaluations':item['evaluations'] if item else 0,'screens':item['screen_operations'] if item else 0}
                rows.append(row)
    assert found==expected and len(rows)==2000
    aggregates={}; contrasts={}
    for method in methods:
        groups={v:[r for r in rows if r['method']==method and r['variant']==v] for v in variants}
        b=groups['baseline']; total_requests=sum(r['requests'] for r in b); base_misses=sum(r['misses'] for r in b)
        base_energy=sum(r['energy_j'] for r in b); base_mbs=sum(r['mbs'] for r in b)
        for v,group in groups.items():
            assert len(group)==200
            energy=sum(r['energy_j'] for r in group); misses=sum(r['misses'] for r in group)
            item={'snapshots':len(group),'requests':total_requests,'misses':misses,'mean_misses':misses/200,
                  'miss_rate':misses/total_requests,'relative_miss_reduction':(base_misses-misses)/base_misses,
                  'success_rate_gain_pp':100*(base_misses-misses)/total_requests,
                  'mean_assignment_energy_j':energy/200,'aggregate_energy_change_pct':100*(energy/base_energy-1),
                  'mbs_share':sum(r['mbs'] for r in group)/total_requests,
                  'mbs_share_delta_pp':100*(sum(r['mbs'] for r in group)-base_mbs)/total_requests,
                  'improved_snapshots':sum(r['miss_delta']<0 for r in group),'worse_snapshots':sum(r['miss_delta']>0 for r in group),
                  'tied_snapshots':sum(r['miss_delta']==0 for r in group),
                  'energy_guard_violations':sum(r['energy_guard_violation'] for r in group),'max_energy_ratio':max(r['energy_ratio'] for r in group),
                  'pred_improve_actual_worse':sum(r['pred_improve_actual_worse'] for r in group),
                  'prediction':{'mean_actual_minus_predicted_misses':mean([r['miss_prediction_error'] for r in group]),
                                'latency_bias_s':sum(r['latency_error_sum'] for r in group)/total_requests,
                                'latency_mae_s':sum(r['latency_abs_error_sum'] for r in group)/total_requests,
                                'optimistic_success_errors':sum(r['pred_success_actual_miss'] for r in group),
                                'pessimistic_success_errors':sum(r['pred_miss_actual_success'] for r in group),
                                'energy_actual_minus_predicted_pct':100*(energy/sum(r['pred_energy_j'] for r in group)-1)},
                  'timing':{metric:{'mean':mean([r[metric] for r in group]),'p95':p95([r[metric] for r in group]),'max':max(r[metric] for r in group)} for metric in ['preparation_seconds','search_seconds','path_seconds']},
                  'mean_evaluations':mean([r['evaluations'] for r in group]),'mean_screens':mean([r['screens'] for r in group]),
                  'mean_changed_actions':mean([r['changed_actions'] for r in group])}
            item['seed_effects']={metric:paired([mean([r[metric] for r in group if r['seed']==s]) for s in seeds]) for metric in ['miss_delta','energy_delta_j','mbs_delta']}
            item['seed_energy_change_pct']=paired([100*(sum(r['energy_j'] for r in group if r['seed']==s)/sum(r['energy_j'] for r in b if r['seed']==s)-1) for s in seeds])
            aggregates[method+'/'+v]=item
        for budget in [32,128]:
            o=groups['ordinary_'+str(budget)];d=groups['deadline_'+str(budget)]
            assert [(r['seed'],r['workload'],r['episode'],r['step']) for r in o]==[(r['seed'],r['workload'],r['episode'],r['step']) for r in d]
            key=method+'/'+str(budget)
            contrasts[key]={metric:paired([mean([y[metric]-x[metric] for x,y in zip(o,d) if x['seed']==s]) for s in seeds]) for metric in ['misses','energy_j','mbs','search_seconds','path_seconds']}
            contrasts[key]['snapshot_wins_ties_losses']=[sum(y['misses']<x['misses'] for x,y in zip(o,d)),sum(y['misses']==x['misses'] for x,y in zip(o,d)),sum(y['misses']>x['misses'] for x,y in zip(o,d))]
    ranked=sorted(contrasts,key=lambda k:contrasts[k]['misses']['p_two_sided']);running=0.
    for rank,k in enumerate(ranked):
        running=max(running,min(1.,(4-rank)*contrasts[k]['misses']['p_two_sided']))
        contrasts[k]['misses']['holm_p_four_exploratory_contrasts']=running
    scipy_check=json.loads((args.validation/'independent_statistics_check.json').read_text())
    for k,v in scipy_check['direct_raw_recalculation'].items():
        actual=contrasts[k]['misses']
        assert close(v['p'],actual['p_two_sided']) and close(v['t'],actual['t'])
        assert all(close(x,y) for x,y in zip(v['ci95'],actual['ci95']))
        assert all(close(x,y) for x,y in zip(v['seed_values'],actual['seed_values']))
    old=json.loads((root/'summary.json').read_text())
    for k,v in old['comparisons'].items():
        assert close(v['mean_miss_difference_per_snapshot'],aggregates[k]['seed_effects']['miss_delta']['mean'])
        assert close(v['diagnostic_proposal_p95_seconds'],aggregates[k]['timing']['path_seconds']['p95'])
    result={'status':'PASS','analysis_type':'exploratory archived-data reanalysis, not independent confirmation',
            'unit':'training seed n=5 per architecture; 5 workloads x 8 snapshots averaged within each seed',
            'statistics':'Paired seed-mean t intervals df=4; normality not established at n=5; exact two-sided sign-flip sensitivity; Holm family = four deadline-minus-ordinary miss contrasts. Other intervals descriptive/unadjusted.',
            'software':{'python':platform.python_version(),'stdlib_only':True,'script_sha256':sha(Path(__file__))},
            'source_hash':protocol['source_hash'],'raw_manifest':manifest,'aggregates':aggregates,'deadline_minus_ordinary':contrasts,
            'prediction_by_action':prediction_by_action,'scipy_crosscheck_version':scipy_check['scipy_version'],
            'limitations':['Assignment-causal UAV energy only: not full UAV energy or MBS energy, no signaling cost.',
                           'Timing covers snapshot deserialization/mask reconstruction and search, NOT complete policy inference or communication.',
                           'Search variants always ran ordinary32/deadline32/ordinary128/deadline128; timing order not randomized.',
                           'A/D snapshots arise from different trajectories; no same-state architecture superiority inference.',
                           'Masks and checkpoint/replay validity were verified by the archived driver; this analysis checks stored records without rerunning environment.',
                           'No request-level fairness or full counterfactual online history recorded; online fairness/DSR cannot be inferred.']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (args.output/'snapshot_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    audit={'status':'PASS','jobs':len(found),'snapshots':len(rows)//5,'variant_records':len(rows),
           'scipy_direct_raw_recalculation_matches':True,'remote_sha256_match':True,'all_jobs_and_snapshot_slots_present':True,'all_stored_metric_identities_pass':True,
           'all_predicted_energy_guards_pass':True,'original_summary_reproduced':True,'excluded_snapshots':0,
           'empty_snapshots':sum(r['requests']==0 for r in rows if r['variant']=='baseline'),
           'new_environment_runs':0,'new_training_runs':0,'analysis_sha256':sha(args.output/'analysis.json'),
           'script_sha256':sha(Path(__file__))}
    (args.validation/'completion_review.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))

if __name__=='__main__':main()
