"""Remote raw audit; common-delay screening is not native cost confirmation."""
import argparse,json,math,os,pickle,statistics,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);arg=parser.parse_args()
a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(arg.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_cache_rerank as c
b=c.base
with b.resource(a).CpuLease(a.resources,task=['cache_rerank_raw_audit'],kind='environment'):
 b.archive(a).boot(a,620001);b.modules(a)
 from utils.sf_hlm_cache_matching import CacheWeightKernel
 from utils.sf_hlm_cache_calibration import validate_bundle
 from utils.sf_hlm_energy_price import calibrate,choose
 from utils.sf_hlm_matching import allocation_counts
 from utils.sf_hlm_action_ranker import public_payload
 from utils.deadline_repair import assess
 from utils.snapshot_evaluator import evaluate
 helper=b.load('rerank_statistical_audit_helpers',a.source_validation/'audit_native.py');close=helper.close
 p=b.read(a.results_root/'protocol.json');summary=b.read(a.results_root/'analysis/summary.json');done=b.read(a.results_root/'completion.json');b.verify(a,p,True)
 assert done['status']=='COMPLETED' and done['summary_sha256']==b.sha(a.results_root/'analysis/summary.json') and done['protocol_sha256']==b.sha(a.results_root/'protocol.json')
 assert p['methods']==list(c.METHODS) and p['nonlearning']==list(b.NONLEARNING) and p['delays_seconds']==list(c.DELAYS) and not p['models_retrained']
 rows=[public_payload(b.read(e['public_rows']))['rows'] for e in p['states']];models={};prices={};pool={};price_rows=0
 for fold in b.splits(a,p):
  name=b.fold_name(fold['block'],fold['workload']);td=a.results_root/'folds'/name;parameters=b.read(td/'prices.json');cases=b.read(td/'cases.json');original=b.read(a.calibration_root/'folds'/name/'cases.json')['half_proxy'];path=a.cache_training/'folds'/name/'model.pkl';bundle=pickle.loads(path.read_bytes())
  assert validate_bundle(bundle)=='full_logit_context' and bundle['fold']==fold and parameters['fold']==fold and parameters['cache_model_sha256']==b.sha(path)
  assert len(fold['train'])==256 and len(fold['test'])==16 and not set(fold['train'])&set(fold['test']) and [x['state_index'] for x in cases]==fold['train']
  for case,source in zip(cases,original):
   si=case['state_index'];assert case['identity']==p['identities'][si] and case['identity'][1]!=fold['block'] and case['identity'][2]!=fold['workload']
   assert c.pool_projection(case['candidates'])==c.pool_projection(source['candidates'])
   if si in pool:assert pool[si]==source['candidates']
   pool[si]=source['candidates'];rr=rows[si];kernel=CacheWeightKernel(rr,parameters['powers'],'learned',bundle['model']);quotas=[allocation_counts(rr,x['assignment'],5) for x in case['candidates']];matrices=kernel.probabilities(quotas)
   for candidate,q in zip(case['candidates'],quotas):
    targets=[5 if row['options'][int(action)]['target']=='mbs' else row['options'][int(action)]['target'] for row,action in zip(rr,candidate['assignment'])];matrix=matrices[tuple(q)];score=float(sum(matrix[i,j] for i,j in enumerate(targets)))
    assert math.isclose(score,candidate['score'],abs_tol=1e-10,rel_tol=0) and math.isclose(assess(rr,tuple(candidate['assignment']))['energy'],candidate['energy'],abs_tol=1e-9,rel_tol=1e-12);price_rows+=1
  close(calibrate(cases,1.05),parameters['prices']['rerank']);models[name]=bundle['model'];prices[name]=parameters
 assert len(pool)==400
 states=[];new_physics=0;reused=0;physical_requests=0
 for si,entry in enumerate(p['states']):
  identity=entry['identity'];td=a.results_root/'tasks'/('rerank_'+'_'.join(map(str,identity)));receipt=b.read(td/'result.json');pub=b.read(td/'public.json');labels=b.read(td/'labels.json');assert receipt['status']=='PASS' and receipt['protocol_sha256']==b.sha(a.results_root/'protocol.json')
  for name,digest in receipt['files'].items():assert b.sha(td/name)==digest
  assert labels['public_sha256']==b.sha(td/'public.json') and pub['sealed_before_private_scoring'] and pub['source_half_pool_exactly_equal']
  name=b.fold_name(identity[1],identity[2]);parameters=prices[name];cc=pub['candidates'];rr=rows[si];assert c.pool_projection(cc)==c.pool_projection(pool[si]);assert pub['cache_model_sha256']==b.sha(a.cache_training/'folds'/name/'model.pkl') and pub['price_sha256']==b.sha(a.results_root/'folds'/name/'prices.json')
  quotas=[allocation_counts(rr,x['assignment'],5) for x in cc];kernel=CacheWeightKernel(rr,parameters['powers'],'learned',models[name]);matrices=kernel.probabilities(quotas);reconstructed=[]
  for candidate,q in zip(cc,quotas):
   targets=[5 if row['options'][int(action)]['target']=='mbs' else row['options'][int(action)]['target'] for row,action in zip(rr,candidate['assignment'])];matrix=matrices[tuple(q)];score=float(sum(matrix[i,j] for i,j in enumerate(targets)));energy=assess(rr,tuple(candidate['assignment']))['energy']
   assert math.isclose(score,candidate['expected_success'],abs_tol=1e-10,rel_tol=0) and math.isclose(energy,candidate['predicted']['energy'],abs_tol=1e-9,rel_tol=1e-12)
   reconstructed.append(dict(action=candidate['action'],score=candidate['expected_success'],energy=candidate['predicted']['energy']))
  params=parameters['prices']['rerank'];index=choose(reconstructed,params['price'],params['normalizer']);assert pub['assignment']==list(cc[index]['assignment']) and pub['diagnostic']['matching_calls']<=10 and pub['diagnostic']['fixed_half_assignment_pool']
  source=a.source_results/'tasks'/('cache_'+'_'.join(map(str,identity)));source_labels=b.read(source/'labels.json');source_public=b.read(source/'public.json');truth=labels['truth'];assert truth['assignment']==pub['assignment'] and [r['ue_id'] for r in truth['requests']]==source_public['request_ids']
  assert len(truth['requests'])==source_labels['admitted']==truth['metrics']['admitted_service_count'] and truth['metrics']['generated_service_count']==source_labels['generated']
  assert sum(r['latency']<=r['deadline'] for r in truth['requests'])==truth['metrics']['deadline_satisfied_count']
  physical_requests+=len(truth['requests'])
  if labels['physical_reused_after_exact_assignment_match']:
   expected=next(z for z in source_labels['outcomes'].values() if z['assignment']==pub['assignment']);close(truth,expected);reused+=1
  else:
   meta=b.read(entry['state']);snapshot=pickle.loads(Path(meta['snapshot']).read_bytes());actual=evaluate(snapshot,pub['assignment']);close(truth['requests'],actual['requests']);close(truth['metrics'],actual['metrics']);new_physics+=1
  methods={}
  for method in c.METHODS:
   outcomes=[truth] if method==c.PRIMARY else [source_labels['outcomes'][source_labels['mapping'][method+':'+str(s['repeat'])]] for s in source_public['selections'] if s['method']==method]
   methods[method]=dict(counts={str(int(d*1000)):statistics.mean(sum(r['latency']+d<=r['deadline'] for r in x['requests']) for x in outcomes) for d in c.DELAYS},energy=statistics.mean(x['metrics']['uav_fleet_total_energy'] for x in outcomes),mbs=statistics.mean(x['metrics']['service_offloads_mbs'] for x in outcomes),fairness=statistics.mean(x['metrics']['fairness'] for x in outcomes))
  states.append(dict(block=str(identity[1]),generated=source_labels['generated'],admitted=source_labels['admitted'],methods=methods))
 blocks={}
 for block in map(str,range(620001,620006)):
  ss=[x for x in states if x['block']==block];den=sum(x['generated'] for x in ss);admitted=sum(x['admitted'] for x in ss);methods={}
  for method in c.METHODS:methods[method]=dict(rate={str(int(d*1000)):sum(x['methods'][method]['counts'][str(int(d*1000))] for x in ss)/den for d in c.DELAYS},energy=sum(x['methods'][method]['energy'] for x in ss),mbs_share=sum(x['methods'][method]['mbs'] for x in ss)/admitted,fairness=statistics.mean(x['methods'][method]['fairness'] for x in ss))
  blocks[block]=dict(generated=den,admitted=admitted,methods=methods)
 close(blocks,summary['blocks']);groups={method:dict(rate={str(int(d*1000)):statistics.mean(x['methods'][method]['rate'][str(int(d*1000))] for x in blocks.values()) for d in c.DELAYS},energy_ratio_to_G=statistics.mean(x['methods'][method]['energy']/x['methods']['G']['energy'] for x in blocks.values()),mbs_share=statistics.mean(x['methods'][method]['mbs_share'] for x in blocks.values()),fairness=statistics.mean(x['methods'][method]['fairness'] for x in blocks.values())) for method in c.METHODS};close(groups,summary['groups']);contrasts={}
 for d in c.DELAYS:
  key=str(int(d*1000));best=max(b.NONLEARNING,key=lambda m:groups[m]['rate'][key]);contrasts[key]=dict(best_nonlearning=best,vs_best=helper.effect([x['methods'][c.PRIMARY]['rate'][key]-x['methods'][best]['rate'][key] for x in blocks.values()]),vs_G=helper.effect([x['methods'][c.PRIMARY]['rate'][key]-x['methods']['G']['rate'][key] for x in blocks.values()]))
 close(contrasts,summary['contrasts']);passed=all(contrasts[k]['vs_best']['mean']>=.001 and contrasts[k]['vs_best']['positive']>=4 for k in ('10','20')) and contrasts['20']['vs_G']['mean']>=.005 and groups[c.PRIMARY]['energy_ratio_to_G']<=1.05
 assert passed==summary['decision']['common_delay_screen_pass'] and not summary['decision']['native_software_cost_tested']
 report=dict(status='PASS',states=400,folds=25,public_price_candidates_recomputed=price_rows,all_training_splits_verified=True,half_assignment_pools_exactly_equal=True,live_selection_recomputed_states=400,physical_states_replayed=new_physics,physical_states_reused_after_exact_match=reused,selected_physical_requests=physical_requests,all_common_delay_counts_and_statistics_recomputed=True,models_retrained=False,native_software_cost_tested=False,screen_pass=passed,protocol_sha256=b.sha(a.results_root/'protocol.json'),summary_sha256=b.sha(a.results_root/'analysis/summary.json'),audit_type='primary agent self-audit; all public price scores and choices, exact half pools, fresh physical replay or audited same-assignment reuse and separate raw statistics',independent_confirmation=False,cpu=sorted(os.sched_getaffinity(0)))
 b.write(a.validation_root/'raw_record_audit.json',report);print(json.dumps(report),flush=True)
