"""Recover aggregation from sealed records, auditing choices and physical truth."""
import argparse,json,math,pickle,statistics,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);o=p.parse_args()
a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(o.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_cache_regime_matching as parent
b=parent.base
with b.resource(a).CpuLease(a.resources,task=['guarded_rerank_scalar_audit_and_aggregation'],kind='environment'):
 b.archive(a).boot(a,620001);b.modules(a)
 from utils.sf_hlm_cache_matching import CacheWeightKernel
 from utils.sf_hlm_matching import allocation_counts
 from utils.snapshot_evaluator import evaluate
 root=o.results;v=o.validation;protocol=b.read(root/'protocol.json');assert not (root/'completion.json').exists()
 helper=b.load('guarded_rerank_audit_helpers',a.source_validation/'audit_native_matching.py');close=helper.close
 for path,digest in protocol['input_files'].items():assert b.sha(path)==digest
 assert protocol['models_retrained'] is False and protocol['new_matching_called_in_screen'] is False and protocol['online_requires_fresh_pool'] and protocol['matching_budget']==10
 primary=protocol['primary'];methods=protocol['methods'];new=(primary,'cold_prior_half_pool','cold_own_half_pool','cold_half_pool_zero');variants=('learned','cold_prior','cold_own','cold_half');nonlearning=protocol['nonlearning'];assert len(methods)==18 and len(nonlearning)==14
 pools={};duplicates=0
 for fold in (a.calibration_root/'folds').iterdir():
  for case in b.read(fold/'cases.json')['cold_half']:
   si=case['state_index'];pool=case['candidates']
   if si in pools:assert pools[si]==pool;duplicates+=1
   else:pools[si]=pool
 assert len(pools)==400
 models={};states=[];coverage=[];fresh=0;reused=0;requests_checked=0;changed=0;choices=0;lower_count=0;lower_absent=0
 # A metadata-only launch supplies the original immutable zero-pool path.
 launch=b.read(v/'launch_v1.json');cmd=launch['command'];zero=Path(cmd[cmd.index('--zero-results')+1]);diag=Path(cmd[cmd.index('--diagnostic')+1]);diag_states={tuple(x['identity']):x for x in b.read(diag/'states.json')}
 for si,entry in enumerate(protocol['states']):
  identity=entry['identity'];suffix='_'.join(map(str,identity));td=root/'tasks'/('pool_'+suffix);receipt=b.read(td/'result.json');pub=b.read(td/'public.json');labels=b.read(td/'labels.json');assert receipt['status']=='PASS' and receipt['protocol_sha256']==b.sha(root/'protocol.json')
  for name,digest in receipt['files'].items():assert b.sha(td/name)==digest
  assert labels['public_sha256']==b.sha(td/'public.json') and pub['sealed_before_private_scoring'] and pub['pool']==pools[si]
  source=a.source_results/'tasks'/('cache_'+suffix);source_pub=b.read(source/'public.json');source_labels=b.read(source/'labels.json');rows=b.read(entry['public_rows'])['rows']
  for row in rows:row['options']={int(k):x for k,x in row['options'].items()}
  name=b.fold_name(identity[1],identity[2]);model_path=a.cache_training/'folds'/name/'model.pkl';price_path=a.calibration_root/'folds'/name/'prices.json';prices=b.read(price_path);assert pub['cache_model_sha256']==b.sha(model_path) and pub['price_sha256']==b.sha(price_path)
  if name not in models:
   bundle=pickle.loads(model_path.read_bytes());assert bundle['context_rule']==parent.RULE and si in bundle['fold']['test'];models[name]=bundle['model']
  quotas=[allocation_counts(rows,c['assignment'],5).tolist() for c in pub['pool']];assert quotas==pub['quotas'];scores={};cap=next(x for x in source_pub['selections'] if x['method']=='cold_half_proxy_price')['diagnostic']['common_energy_cap']
  for variant in variants:
   kernel=CacheWeightKernel(rows,prices['powers'],variant,models[name]);matrices=kernel.probabilities(quotas);scores[variant]=[]
   for c,q in zip(pub['pool'],quotas):
    targets=[5 if r['options'][ac]['target']=='mbs' else r['options'][ac]['target'] for r,ac in zip(rows,c['assignment'])];matrix=matrices[tuple(q)];value=math.fsum(float(matrix[i,j]) for i,j in enumerate(targets));scores[variant].append(dict(action=c['action'],score=value,energy=c['energy']))
    assert c['energy']<=cap+1e-7
   close(scores[variant],pub['scores'][variant])
  selected={m:min(range(len(pub['pool'])),key=lambda i:(-scores[variant][i]['score'],scores[variant][i]['energy'],i)) for m,variant in zip(new,variants)}
  # Use the original summation's exact scores for any machine-precision tie.
  for m,variant in zip(new,variants):
   i=pub['selected_indices'][m];best=max(x['score'] for x in scores[variant]);assert scores[variant][i]['score']>=best-1e-12
   selected[m]=min(range(len(pub['pool'])),key=lambda j:(-pub['scores'][variant][j]['score'],pub['scores'][variant][j]['energy'],j))
  assert selected==pub['selected_indices'];choices+=len(selected)
  reference=next(x for x in source_pub['selections'] if x['method']=='cold_half_proxy_price');parameter=prices['prices']['cold_half'];index=min(range(len(pub['pool'])),key=lambda i:(-(pub['scores']['cold_half'][i]['score']-parameter['price']*pub['pool'][i]['energy']/parameter['normalizer']),pub['pool'][i]['energy'],i))
  assert pub['pool'][index]['assignment']==reference['assignment'] and pub['pool'][index]['action']==reference['diagnostic']['selected_action'];changed+=int(pub['pool'][selected[primary]]['assignment']!=reference['assignment'])
  outcomes=dict(source_labels['outcomes']);outcomes.update(labels['new_outcomes'])
  for key,truth in labels['new_outcomes'].items():
   assert not any(x['assignment']==truth['assignment'] for x in source_labels['outcomes'].values());snapshot=pickle.loads(Path(b.read(entry['state'])['snapshot']).read_bytes());actual=evaluate(snapshot,truth['assignment']);close(truth['requests'],actual['requests']);close(truth['metrics'],actual['metrics']);fresh+=1;requests_checked+=len(truth['requests'])
  seen_mapping=set(source_labels['outcomes'])
  for m,i in selected.items():
   truth=outcomes[labels['mapping'][m]];assert truth['assignment']==pub['pool'][i]['assignment'];assert truth['metrics']['generated_service_count']==source_labels['generated'] and truth['metrics']['admitted_service_count']==source_labels['admitted']
   if labels['mapping'][m] in seen_mapping:reused+=1
   else:seen_mapping.add(labels['mapping'][m])
  direct=b.read(zero/'tasks'/('rerank_'+suffix)/'public.json');direct_actions={c['action'] for c in direct['candidates']};control_action=reference['diagnostic']['selected_action'];ds=diag_states[tuple(identity)];lower=ds['cross']['learned']['cold_half_proxy_price']['priced_score']>ds['cross']['learned'][parent.PRIMARY]['priced_score']+1e-10;lower_count+=lower;lower_absent+=lower and control_action not in direct_actions
  coverage.append(dict(identity=identity,control_selected_action=control_action,direct_pool_actions=sorted(direct_actions),control_action_present=control_action in direct_actions,learning_own_priced_score_below_control=lower));metrics={}
  for m in methods:
   truth=[outcomes[labels['mapping'][m]]] if m in new else [source_labels['outcomes'][source_labels['mapping'][m+':'+str(x['repeat'])]] for x in source_pub['selections'] if x['method']==m]
   metrics[m]=dict(success={str(ms):statistics.mean(sum(r['latency']+ms/1000<=r['deadline'] for r in x['requests']) for x in truth) for ms in (0,10,20,30)},energy=statistics.mean(float(x['metrics']['uav_fleet_total_energy']) for x in truth),mbs=statistics.mean(float(x['metrics']['service_offloads_mbs']) for x in truth),fairness=statistics.mean(float(x['metrics']['fairness']) for x in truth))
  states.append(dict(identity=identity,generated=source_labels['generated'],admitted=source_labels['admitted'],methods=metrics))
 blocks={}
 for block in map(str,range(620001,620006)):
  rr=[x for x in states if str(x['identity'][1])==block];den=sum(x['generated'] for x in rr);admitted=sum(x['admitted'] for x in rr)
  blocks[block]=dict(generated=den,admitted=admitted,methods={m:dict(rate={str(ms):sum(x['methods'][m]['success'][str(ms)] for x in rr)/den for ms in (0,10,20,30)},energy=sum(x['methods'][m]['energy'] for x in rr),mbs_share=sum(x['methods'][m]['mbs'] for x in rr)/admitted,fairness=statistics.mean(x['methods'][m]['fairness'] for x in rr)) for m in methods})
 groups={m:dict(rate={str(ms):statistics.mean(x['methods'][m]['rate'][str(ms)] for x in blocks.values()) for ms in (0,10,20,30)},energy_ratio_to_G=statistics.mean(x['methods'][m]['energy']/x['methods']['G']['energy'] for x in blocks.values()),mbs_share=statistics.mean(x['methods'][m]['mbs_share'] for x in blocks.values()),fairness=statistics.mean(x['methods'][m]['fairness'] for x in blocks.values())) for m in methods}
 old=b.read(a.source_results/'analysis/summary.json')['subsets']['all'];close({k:groups[k] for k in parent.METHODS},old['groups'])
 contrasts={}
 for ms in (0,10,20,30):
  k=str(ms);best=max(nonlearning,key=lambda m:groups[m]['rate'][k]);contrasts[k]=dict(best_nonlearning=best,vs_best=helper.effect([x['methods'][primary]['rate'][k]-x['methods'][best]['rate'][k] for x in blocks.values()]),vs_G=helper.effect([x['methods'][primary]['rate'][k]-x['methods']['G']['rate'][k] for x in blocks.values()]))
 passed=all(contrasts[k]['vs_best']['mean']>=.001 and contrasts[k]['vs_best']['positive']>=4 for k in ('10','20')) and contrasts['20']['vs_G']['mean']>=.005 and groups[primary]['energy_ratio_to_G']<=1.05
 summary=dict(status='GUARDED_HALF_POOL_CONDITIONAL_RERANK_COMMON_DELAY_SCREEN',groups=groups,blocks=blocks,contrasts=contrasts,decision=dict(common_delay_screen_pass=passed,native_software_cost_tested=False,closed_loop_improvement_tested=False,independent_confirmation=False,goal_complete=False),pool_duplicate_checks=duplicates,original_cold_half_choices_reproduced=400,learning_choices_differ_from_cold_half=changed,fresh_physical_outcomes=fresh,reused_exact_assignments=reused,lower_priced_score_states=int(lower_count),lower_priced_score_control_action_absent=int(lower_absent),statistics_recovered_from_sealed_records=True,aggregation_repair='Convert all metric scalars to builtin float; no selection or physical outcome rerun',aggregation_script_sha256=b.sha(__file__))
 b.write(root/'analysis/states.json',states);b.write(root/'analysis/pool_coverage.json',coverage);b.write(root/'analysis/summary.json',summary)
 for path,digest in protocol['input_files'].items():assert b.sha(path)==digest
 report=dict(status='PASS',states=400,zero_price_choices_checked=choices,fresh_physical_outcomes_replayed=fresh,physical_requests_replayed=requests_checked,old_fourteen_method_statistics_exactly_reproduced=True,all_public_scores_recomputed=True,summary_sha256=b.sha(root/'analysis/summary.json'),protocol_sha256=b.sha(root/'protocol.json'),audit_script_sha256=b.sha(__file__),audit_type='primary agent self-audit; scalar choices, new physical replays, raw-record aggregation with numeric type normalization',independent_confirmation=False)
 b.write(v/'raw_record_audit.json',report);b.write(root/'completion.json',dict(status='COMPLETED',states=400,protocol_sha256=b.sha(root/'protocol.json'),summary_sha256=b.sha(root/'analysis/summary.json'),recovered_aggregation=True));b.write(root/'status.json',dict(status='COMPLETED',states=400));print(json.dumps(dict(audit=report,contrasts=contrasts,decision=summary['decision'],changed=changed,lower_count=int(lower_count),lower_absent=int(lower_absent))),flush=True)
