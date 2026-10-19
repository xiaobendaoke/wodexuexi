"""Raw self-audit of the frozen fourteen-method common-delay screen."""
import argparse,json,math,pickle,statistics,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);opt=parser.parse_args();a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(opt.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_cache_regime_matching as c
c.install();b=c.base
with b.resource(a).CpuLease(a.resources,task=['conditional_cache_common_audit'],kind='environment'):
 b.archive(a).boot(a,620001);b.modules(a)
 from utils.sf_hlm_cache_matching import CacheWeightKernel,CACHE_METHODS
 from utils.sf_hlm_cache_context import public_context
 from utils.sf_hlm_cache_calibration import validate_bundle
 from utils.sf_hlm_matching import allocation_counts
 from utils.snapshot_evaluator import evaluate
 helper=b.load('regime_audit_helpers',a.validation_root/'audit_native_matching.py');close=helper.close
 p=b.read(a.results_root/'protocol.json');done=b.read(a.results_root/'completion.json');summary=b.read(a.results_root/'analysis/summary.json');b.verify(a,p,True)
 assert done['status']=='COMPLETED' and done['summary_sha256']==b.sha(a.results_root/'analysis/summary.json') and done['protocol_sha256']==b.sha(a.results_root/'protocol.json')
 assert p['methods']==list(c.METHODS) and p['nonlearning']==list(c.NONLEARNING) and p['source_context_rule']==c.RULE and p['branch_instances']==22 and p['timing_samples_per_state']==0
 for path,digest in p['calibration_files'].items():assert b.sha(path)==digest
 assert b.read(a.validation_root/'price_calibration_audit.json')['status']=='PASS'
 states=[];models={};counts=dict(instances=0,legacy_public_instances=0,physical_requests=0,fresh_physical_replays=0,reused_physical_outcomes=0);guarded_states=0
 for entry in p['states']:
  identity=entry['identity'];td=b.folder(a,identity);receipt=b.read(td/'result.json');pub=b.read(td/'public.json');labels=b.read(td/'labels.json');timing=b.read(td/'timing.json')
  assert receipt['status']=='PASS' and receipt['protocol_sha256']==b.sha(a.results_root/'protocol.json') and receipt['timing_samples']==0
  for name,digest in receipt['files'].items():assert b.sha(td/name)==digest
  assert labels['public_sha256']==b.sha(td/'public.json') and labels['timing_sha256']==b.sha(td/'timing.json') and timing['records']==[] and timing['timing_order']==[]
  expected=[(m,r) for m in c.METHODS for r in range(5 if m in ('R8','R16') else 1)];assert [(x['method'],x['repeat']) for x in pub['selections']]==expected
  original_dir=Path(p['reference_results'])/'tasks'/td.name;old_pub=b.read(original_dir/'public.json');old_labels=b.read(original_dir/'labels.json');old_refs={(r['method'],r['repeat']):r for r in old_pub['selections']}
  assert b.sha(original_dir/'public.json')==pub['reference_public_sha256'] and pub['legacy_public_branches_replayed']==18 and pub['request_ids']==old_pub['request_ids'] and pub['motion']==old_pub['motion'] and pub['rows_sha256']==old_pub['rows_sha256']
  rows=b.read(entry['public_rows'])['rows']
  for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
  guard=public_context(rows)['global']['all_visible_miss'];guarded_states+=int(guard);name=b.fold_name(identity[1],identity[2]);prices=b.read(a.calibration_root/'folds'/name/'prices.json')
  if name not in models:
   path=a.cache_training/'folds'/name/'model.pkl';bundle=pickle.loads(path.read_bytes());assert b.sha(path)==pub['locks']['cache_model_sha256'] and validate_bundle(bundle,supported_context_rules=('none',c.RULE))=='full_logit_context' and bundle['context_rule']==c.RULE and len(bundle['training_state_indices'])==192;models[name]=bundle['model']
  assert prices['fold']==pub['locks']['fold'] and b.sha(a.calibration_root/'folds'/name/'prices.json')==pub['locks']['cache_price_sha256']
  si=p['identities'].index(identity);fold=prices['fold'];assert si in fold['test'] and si not in fold['train'] and all(p['identities'][i][1]!=identity[1] and p['identities'][i][2]!=identity[2] for i in fold['train'])
  old_outcomes={helper.digest(x['assignment']):x for x in old_labels['outcomes'].values()};snapshot=None
  for key,truth in labels['outcomes'].items():
   assert key==helper.digest(truth['assignment']) and [r['ue_id'] for r in truth['requests']]==pub['request_ids'] and len(truth['requests'])==labels['admitted']==truth['metrics']['admitted_service_count'] and labels['generated']==truth['metrics']['generated_service_count']
   assert sum(r['latency']<=r['deadline'] for r in truth['requests'])==truth['metrics']['deadline_satisfied_count'];counts['physical_requests']+=len(truth['requests'])
   if key in old_outcomes:close(truth,old_outcomes[key]);counts['reused_physical_outcomes']+=1
   else:
    if snapshot is None:snapshot=pickle.loads(Path(b.read(entry['state'])['snapshot']).read_bytes())
    replay=evaluate(snapshot,truth['assignment']);close(truth['requests'],replay['requests']);close(truth['metrics'],replay['metrics']);counts['fresh_physical_replays']+=1
  methods={m:[] for m in c.METHODS};used=set()
  for spec in pub['selections']:
   method=spec['method'];repeat=spec['repeat'];truth=labels['outcomes'][labels['mapping'][method+':'+str(repeat)]];used.add(labels['mapping'][method+':'+str(repeat)]);d=spec['diagnostic'];assert spec['assignment']==truth['assignment'] and d['matching_calls']<=d['matching_budget'] and d['chosen_predicted']['energy']<=d['common_energy_cap']+1e-7
   if (method,repeat) in old_refs:
    ref=old_refs[method,repeat];assert spec['assignment']==ref['assignment'] and b.decision_diagnostic(d)==b.decision_diagnostic(ref['diagnostic']);counts['legacy_public_instances']+=1
   if method in CACHE_METHODS:
    variant=CACHE_METHODS[method];kernel=CacheWeightKernel(rows,prices['powers'],variant,models[name]);q=allocation_counts(rows,spec['assignment'],5);matrix=kernel.probabilities([q])[tuple(q)];targets=[5 if r['options'][int(action)]['target']=='mbs' else int(r['options'][int(action)]['target']) for r,action in zip(rows,spec['assignment'])];value=float(sum(matrix[i,j] for i,j in enumerate(targets)));assert math.isclose(value,d['expected_success'],abs_tol=1e-10,rel_tol=0)
   if method in ('cold_prior_price','cold_own_proxy_price','cold_half_proxy_price',c.PRIMARY):
    assert d['public_cache_context_rule']==c.RULE and d['public_cache_guard_triggered']==guard
    assert d['cache_prediction_batches']==(int(bool(pub['compatibility']['eligible_edges'])) if method==c.PRIMARY and not guard else 0)
   methods[method].append(dict(success={str(int(delay*1000)):sum(r['latency']+delay<=r['deadline'] for r in truth['requests']) for delay in c.DELAYS},energy=truth['metrics']['uav_fleet_total_energy'],mbs=truth['metrics']['service_offloads_mbs'],fairness=truth['metrics']['fairness'],calls=d['matching_calls']));counts['instances']+=1
  assert used==set(labels['outcomes']);averaged={m:dict(success={str(int(d*1000)):statistics.mean(x['success'][str(int(d*1000))] for x in rr) for d in c.DELAYS},**{k:statistics.mean(x[k] for x in rr) for k in ('energy','mbs','fairness','calls')}) for m,rr in methods.items()};states.append(dict(identity=identity,generated=labels['generated'],admitted=labels['admitted'],methods=averaged))
 assert len(states)==400 and counts['instances']==8800 and counts['legacy_public_instances']==7200
 subsets={}
 for source in ('all','greedy','learned'):
  rr=[s for s in states if source=='all' or s['identity'][0]==source];blocks={}
  for block in sorted({s['identity'][1] for s in rr}):
   ss=[s for s in rr if s['identity'][1]==block];den=sum(s['generated'] for s in ss);admitted=sum(s['admitted'] for s in ss);methods={}
   for method in c.METHODS:methods[method]=dict(rate={str(int(d*1000)):sum(s['methods'][method]['success'][str(int(d*1000))] for s in ss)/den for d in c.DELAYS},energy=sum(s['methods'][method]['energy'] for s in ss),mbs_share=sum(s['methods'][method]['mbs'] for s in ss)/admitted,fairness=statistics.mean(s['methods'][method]['fairness'] for s in ss),calls=statistics.mean(s['methods'][method]['calls'] for s in ss))
   blocks[str(block)]=dict(generated=den,admitted=admitted,methods=methods)
  groups={m:dict(rate={str(int(d*1000)):statistics.mean(x['methods'][m]['rate'][str(int(d*1000))] for x in blocks.values()) for d in c.DELAYS},energy_ratio_to_G=statistics.mean(x['methods'][m]['energy']/x['methods']['G']['energy'] for x in blocks.values()),mbs_share=statistics.mean(x['methods'][m]['mbs_share'] for x in blocks.values()),fairness=statistics.mean(x['methods'][m]['fairness'] for x in blocks.values()),calls=statistics.mean(x['methods'][m]['calls'] for x in blocks.values())) for m in c.METHODS};contrasts={}
  for delay in c.DELAYS:
   key=str(int(delay*1000));best=max(c.NONLEARNING,key=lambda m:groups[m]['rate'][key]);contrasts[key]=dict(best_nonlearning=best,vs_best=helper.effect([x['methods'][c.PRIMARY]['rate'][key]-x['methods'][best]['rate'][key] for x in blocks.values()]),vs_G=helper.effect([x['methods'][c.PRIMARY]['rate'][key]-x['methods']['G']['rate'][key] for x in blocks.values()]))
  subsets[source]=dict(blocks=blocks,groups=groups,contrasts=contrasts);close(subsets[source],summary['subsets'][source])
 main=subsets['all'];contrasts=main['contrasts'];energy=main['groups'][c.PRIMARY]['energy_ratio_to_G'];passed=all(contrasts[k]['vs_best']['mean']>=.001 and contrasts[k]['vs_best']['positive']>=4 for k in ('10','20')) and contrasts['20']['vs_G']['mean']>=.005 and energy<=1.05
 assert passed==summary['decision']['common_delay_screen_pass'] and not summary['decision']['native_software_cost_tested']
 report=dict(status='PASS',states=400,counts=counts,guarded_states=guarded_states,public_guard_and_model_metadata_verified=True,all_common_delay_thresholds_and_statistics_recomputed=True,no_formal_timing_samples=True,legacy_public_branches_exactly_replayed=7200,screen_pass=passed,protocol_sha256=b.sha(a.results_root/'protocol.json'),summary_sha256=b.sha(a.results_root/'analysis/summary.json'),audit_type='primary agent self-audit; live legacy parity, public probability checks, fresh/reused physical outcomes and separate weighted statistics',independent_confirmation=False)
 b.write(a.validation_root/'common_delay_raw_audit.json',report);print(json.dumps(report),flush=True)
