"""Raw check of a single zero-price intervention; no new model fitting."""
import argparse,json,math,pickle,statistics,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);opt=parser.parse_args();a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(opt.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_cache_zero_price as c
b=c.base
with b.resource(a).CpuLease(a.resources,task=['cache_zero_price_raw_audit'],kind='environment'):
 b.archive(a).boot(a,620001);b.modules(a)
 from utils.sf_hlm_cache_matching import CacheWeightKernel
 from utils.sf_hlm_cache_calibration import validate_bundle
 from utils.sf_hlm_matching import allocation_counts
 from utils.sf_hlm_energy_price import choose
 from utils.snapshot_evaluator import evaluate
 helper=b.load('zero_price_audit_helpers',a.source_validation/'audit_native_matching.py');close=helper.close
 p=b.read(a.results_root/'protocol.json');done=b.read(a.results_root/'completion.json');summary=b.read(a.results_root/'analysis/summary.json');b.verify(a,p,True)
 assert done['status']=='COMPLETED' and done['summary_sha256']==b.sha(a.results_root/'analysis/summary.json') and done['protocol_sha256']==b.sha(a.results_root/'protocol.json') and p['prescribed_primary_price']==0.
 assert p['methods']==list(c.METHODS) and p['nonlearning']==list(c.NONLEARNING) and not p['models_retrained'] and not p['calibrated_soft_energy_target_applies_to_primary']
 states=[];models={};changed=0;fresh=0;reused=0;requests=0
 for si,entry in enumerate(p['states']):
  identity=entry['identity'];td=a.results_root/'tasks'/('rerank_'+'_'.join(map(str,identity)));receipt=b.read(td/'result.json');pub=b.read(td/'public.json');labels=b.read(td/'labels.json');assert receipt['status']=='PASS' and receipt['protocol_sha256']==b.sha(a.results_root/'protocol.json')
  for name,digest in receipt['files'].items():assert b.sha(td/name)==digest
  source=a.source_results/'tasks'/('cache_'+'_'.join(map(str,identity)));source_pub=b.read(source/'public.json');source_labels=b.read(source/'labels.json');parent=next(r for r in source_pub['selections'] if r['method']==c.parent.PRIMARY)
  assert pub['original_selection_reproduced'] and pub['sealed_before_private_scoring'] and pub['parent_public_sha256']==b.sha(source/'public.json') and labels['public_sha256']==b.sha(td/'public.json')
  name=b.fold_name(identity[1],identity[2]);price=b.read(a.calibration_root/'folds'/name/'prices.json');model_path=a.cache_training/'folds'/name/'model.pkl';assert pub['cache_model_sha256']==b.sha(model_path) and pub['price_sha256']==b.sha(a.calibration_root/'folds'/name/'prices.json')
  if name not in models:
   bundle=pickle.loads(model_path.read_bytes());assert validate_bundle(bundle,supported_context_rules=('none',c.parent.RULE))=='full_logit_context' and bundle['fold']==price['fold'] and si in bundle['fold']['test'];models[name]=bundle['model']
  rows=b.read(entry['public_rows'])['rows']
  for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
  candidates=pub['candidates'];quotas=[allocation_counts(rows,x['assignment'],5) for x in candidates];kernel=CacheWeightKernel(rows,price['powers'],'learned',models[name]);matrices=kernel.probabilities(quotas);scores=[]
  for candidate,q in zip(candidates,quotas):
   matrix=matrices[tuple(q)];targets=[5 if row['options'][int(action)]['target']=='mbs' else row['options'][int(action)]['target'] for row,action in zip(rows,candidate['assignment'])];value=float(sum(matrix[i,j] for i,j in enumerate(targets)));assert math.isclose(value,candidate['expected_success'],abs_tol=1e-10,rel_tol=0)
   scores.append(dict(action=candidate['action'],score=candidate['expected_success'],energy=candidate['predicted']['energy']))
  old=price['prices']['learned'];old_index=choose(scores,old['price'],old['normalizer']);new_index=choose(scores,0.,old['normalizer']);assert candidates[old_index]['assignment']==parent['assignment'] and candidates[old_index]['action']==parent['diagnostic']['selected_action'] and candidates[new_index]['assignment']==pub['assignment'] and pub['parent_assignment']==parent['assignment'] and pub['parent_selected_index']==old_index and pub['zero_selected_index']==new_index
  assert scores[new_index]['score']>=scores[old_index]['score']-1e-12 and pub['diagnostic']['matching_calls']<=10 and pub['diagnostic']['chosen_predicted']['energy']<=pub['diagnostic']['common_energy_cap']+1e-7
  if old['price']==0:assert pub['assignment']==parent['assignment']
  changed+=int(pub['assignment']!=parent['assignment']);truth=labels['truth'];assert truth['assignment']==pub['assignment'] and [r['ue_id'] for r in truth['requests']]==source_pub['request_ids'] and len(truth['requests'])==source_labels['admitted']==truth['metrics']['admitted_service_count'];requests+=len(truth['requests'])
  if labels['physical_reused_after_exact_assignment_match']:
   expected=next(x for x in source_labels['outcomes'].values() if x['assignment']==pub['assignment']);close(truth,expected);reused+=1
  else:
   snapshot=pickle.loads(Path(b.read(entry['state'])['snapshot']).read_bytes());actual=evaluate(snapshot,pub['assignment']);close(truth['requests'],actual['requests']);close(truth['metrics'],actual['metrics']);fresh+=1
  methods={}
  for method in c.METHODS:
   outcomes=[truth] if method==c.PRIMARY else [source_labels['outcomes'][source_labels['mapping'][method+':'+str(s['repeat'])]] for s in source_pub['selections'] if s['method']==method]
   methods[method]=dict(counts={str(int(d*1000)):statistics.mean(sum(r['latency']+d<=r['deadline'] for r in t['requests']) for t in outcomes) for d in c.replay.DELAYS},energy=statistics.mean(t['metrics']['uav_fleet_total_energy'] for t in outcomes),mbs=statistics.mean(t['metrics']['service_offloads_mbs'] for t in outcomes),fairness=statistics.mean(t['metrics']['fairness'] for t in outcomes))
  states.append(dict(block=str(identity[1]),generated=source_labels['generated'],admitted=source_labels['admitted'],methods=methods))
 blocks={}
 for block in map(str,range(620001,620006)):
  rr=[s for s in states if s['block']==block];den=sum(s['generated'] for s in rr);admitted=sum(s['admitted'] for s in rr);blocks[block]=dict(generated=den,admitted=admitted,methods={m:dict(rate={str(int(d*1000)):sum(s['methods'][m]['counts'][str(int(d*1000))] for s in rr)/den for d in c.replay.DELAYS},energy=sum(s['methods'][m]['energy'] for s in rr),mbs_share=sum(s['methods'][m]['mbs'] for s in rr)/admitted,fairness=statistics.mean(s['methods'][m]['fairness'] for s in rr)) for m in c.METHODS})
 close(blocks,summary['blocks']);groups={m:dict(rate={str(int(d*1000)):statistics.mean(x['methods'][m]['rate'][str(int(d*1000))] for x in blocks.values()) for d in c.replay.DELAYS},energy_ratio_to_G=statistics.mean(x['methods'][m]['energy']/x['methods']['G']['energy'] for x in blocks.values()),mbs_share=statistics.mean(x['methods'][m]['mbs_share'] for x in blocks.values()),fairness=statistics.mean(x['methods'][m]['fairness'] for x in blocks.values())) for m in c.METHODS};close(groups,summary['groups']);contrasts={};vs_parent={}
 for delay in c.replay.DELAYS:
  key=str(int(delay*1000));best=max(c.NONLEARNING,key=lambda m:groups[m]['rate'][key]);contrasts[key]=dict(best_nonlearning=best,vs_best=helper.effect([x['methods'][c.PRIMARY]['rate'][key]-x['methods'][best]['rate'][key] for x in blocks.values()]),vs_G=helper.effect([x['methods'][c.PRIMARY]['rate'][key]-x['methods']['G']['rate'][key] for x in blocks.values()]));vs_parent[key]=helper.effect([x['methods'][c.PRIMARY]['rate'][key]-x['methods'][c.parent.PRIMARY]['rate'][key] for x in blocks.values()])
 close(contrasts,summary['contrasts']);passed=all(contrasts[k]['vs_best']['mean']>=.001 and contrasts[k]['vs_best']['positive']>=4 for k in ('10','20')) and contrasts['20']['vs_G']['mean']>=.005 and groups[c.PRIMARY]['energy_ratio_to_G']<=1.05
 assert passed==summary['decision']['common_delay_screen_pass'] and changed==summary['parent_choices_changed'] and summary['original_priced_choices_reproduced']==400 and not summary['decision']['native_software_cost_tested']
 report=dict(status='PASS',states=400,original_priced_choices_reproduced=400,parent_choices_changed=changed,physical_states_replayed=fresh,physical_states_reused_after_exact_match=reused,selected_physical_requests=requests,all_same_pool_scores_and_choices_recomputed=True,all_common_delay_statistics_recomputed=True,zero_vs_priced_parent=vs_parent,screen_pass=passed,protocol_sha256=b.sha(a.results_root/'protocol.json'),summary_sha256=b.sha(a.results_root/'analysis/summary.json'),audit_type='primary agent self-audit; same-pool original/zero choice, immutable models and code, actual outcomes and separate statistics',independent_confirmation=False)
 b.write(a.validation_root/'raw_record_audit.json',report);print(json.dumps(report),flush=True)
