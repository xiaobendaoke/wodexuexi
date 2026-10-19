import argparse,json,math,pickle,statistics,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);opt=parser.parse_args();a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(opt.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_cache_regime as c
b=c.base
with b.res(a).CpuLease(a.resources,task=['cache_regime_raw_audit'],kind='environment'):
 import numpy as np
 from scipy.stats import t
 from utils import sf_hlm_cache_calibration as cc
 from utils.sf_hlm_cache_context import public_context
 p=b.read(a.results_root/'protocol.json');done=b.read(a.results_root/'completion.json');summary=b.read(a.results_root/'analysis/summary.json');prepared=b.read(a.results_root/'prepared.json');b.verify(a,p)
 for path,digest in p['input_files'].items():assert b.sha(path)==digest
 assert done['summary_sha256']==b.sha(a.results_root/'analysis/summary.json') and done['protocol_sha256']==b.sha(a.results_root/'protocol.json') and prepared['data_sha256']==b.sha(a.results_root/'training_data.npz')
 archive=np.load(a.results_root/'training_data.npz');data={k:archive[k] for k in archive.files};source=np.load(a.source_training/'training_data.npz');old={k:source[k] for k in source.files}
 for key in old:assert np.array_equal(old[key],data[key])
 guarded=[]
 for si,state in enumerate(p['states']):
  flag=public_context(b.read(state['public'])['rows'])['global']['all_visible_miss'];guarded.append(flag);assert np.all(data['guarded'][data['state_index']==si]==flag)
 for target,origin in (('cold_guard_prior','prior'),('cold_guard_own','own_proxy'),('cold_guard_half','half_proxy')):
  expected=data[origin].copy();expected[data['guarded']]=0.;assert np.array_equal(expected,data[target])
 probability=np.full(len(data['y']),np.nan);legacy=probability.copy();seen=np.zeros(len(probability),int);refits=0
 for fold in b.splits(a,p):
  path=b.folded(a,fold['block'],fold['workload']);bundle=pickle.loads((path/'model.pkl').read_bytes());receipt=b.read(path/'result.json');prediction=np.load(path/'test_predictions.npz');previous=np.load(a.source_training/'folds'/path.name/'test_predictions.npz')
  for name,digest in receipt['files'].items():assert b.sha(path/name)==digest
  assert bundle['fold']==fold and bundle['context_rule']==c.RULE and bundle['protocol_sha256']==b.sha(a.results_root/'protocol.json') and cc.validate_bundle(bundle,supported_context_rules=('none',c.RULE))=='full_logit_context'
  try:cc.validate_bundle(bundle)
  except ValueError:pass
  else:raise AssertionError('Legacy adapter must reject a conditional model')
  train=np.isin(data['state_index'],fold['train'])&~data['guarded'];test=np.isin(data['state_index'],fold['test']);warm_test=test&~data['guarded'];assert len(set(data['state_index'][train].tolist()))==192 and bundle['training_state_indices']==sorted(set(data['state_index'][train].tolist()))
  assert not np.any(data['guarded'][train]) and not set(fold['train'])&set(fold['test']) and all(p['identities'][int(i)][1]!=fold['block'] and p['identities'][int(i)][2]!=fold['workload'] for i in data['state_index'][train])
  assert np.array_equal(prediction['row_indices'],np.flatnonzero(test)) and np.array_equal(prediction['row_indices'],previous['row_indices']) and np.array_equal(prediction['guarded'],data['guarded'][test])
  model,diagnostic=cc.fit_model(data['x'][train],data['y'][train],data['weight'][train]);expected=np.zeros(int(test.sum()));expected[~data['guarded'][test]]=model.predict_proba(data['x'][warm_test])[:,1];assert np.allclose(expected,prediction['probability'],rtol=0,atol=1e-12)
  assert np.array_equal(bundle['model'].predict_proba(data['x'][warm_test])[:,1],prediction['probability'][~data['guarded'][test]])
  if diagnostic['kind']=='weighted_standardized_logistic':
   scaler=bundle['model'].named_steps['standardize'];mean=np.average(data['x'][train],axis=0,weights=data['weight'][train]);var=np.average((data['x'][train]-mean)**2,axis=0,weights=data['weight'][train]);assert np.allclose(mean,scaler.mean_,rtol=1e-12,atol=1e-12) and np.allclose(var,scaler.var_,rtol=1e-12,atol=1e-12)
  probability[test]=prediction['probability'];legacy[test]=previous['probability'];seen[test]+=1;refits+=1
 assert np.all(seen==1) and np.all(probability[data['guarded']]==0)
 predictions=dict(learned=probability,legacy_mixed=legacy,**{k:data[k] for k in c.METHODS[1:]})
 def block_metrics(mask,block):
  take=mask&np.asarray([p['identities'][int(i)][1]==block for i in data['state_index']]);y=data['y'][take];w=data['weight'][take];metrics={}
  for name,pp in predictions.items():
   pr=pp[take];clip=np.clip(pr,1e-12,1-1e-12);metrics[name]=dict(brier=float(np.average((pr-y)**2,weights=w)),log_loss=float(np.average(-y*np.log(clip)-(1-y)*np.log1p(-clip),weights=w)))
  return metrics
 def close(left,right):
  if isinstance(left,dict):
   assert left.keys()==right.keys()
   for k in left:close(left[k],right[k])
  else:assert math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12),(left,right)
 blocks={str(k):block_metrics(np.ones(len(probability),bool),k) for k in range(620001,620006)}
 for block,metrics in blocks.items():close({k:metrics[k] for k in c.METHODS},summary['blocks'][block])
 groups={name:{metric:statistics.mean(x[name][metric] for x in blocks.values()) for metric in ('brier','log_loss')} for name in c.METHODS};close(groups,summary['groups']);best=min(c.METHODS[1:],key=lambda k:groups[k]['brier'])
 def effect(values):
  mean=statistics.mean(values);radius=float(t.ppf(.975,4))*statistics.stdev(values)/5**.5
  return dict(mean_improvement=mean,ci95=[mean-radius,mean+radius],values=values,positive=sum(v>0 for v in values))
 for method in c.METHODS[1:]:
  e=effect([x[method]['brier']-x['learned']['brier'] for x in blocks.values()]);r=summary['brier_improvements'][method]
  assert math.isclose(e['mean_improvement'],r['mean_improvement'],abs_tol=1e-12) and np.allclose(e['ci95'],r['ci95'],rtol=0,atol=1e-12) and e['positive']==r['positive']
 masks=dict(guarded=data['guarded'],non_guarded=~data['guarded'],step0=np.asarray([p['identities'][int(i)][4]==0 for i in data['state_index']]),later=np.asarray([p['identities'][int(i)][4]!=0 for i in data['state_index']]))
 for name,mask in masks.items():
  bb={str(k):block_metrics(mask,k) for k in range(620001,620006)};close(bb,summary['subsets'][name]['blocks']);gg={m:{v:statistics.mean(x[m][v] for x in bb.values()) for v in ('brier','log_loss')} for m in predictions};close(gg,summary['subsets'][name]['groups'])
 warm=summary['subsets']['non_guarded']['blocks'];e=effect([x['prior']['brier']-x['learned']['brier'] for x in warm.values()]);r=summary['non_guarded_vs_prior'];assert np.allclose(e['ci95'],r['ci95'],rtol=0,atol=1e-12) and e['positive']==r['positive'] and math.isclose(e['mean_improvement'],r['mean_improvement'],abs_tol=1e-12)
 global_go=summary['brier_improvements'][best]['mean_improvement']>0 and summary['brier_improvements'][best]['positive']>=4;go=global_go and e['mean_improvement']>0 and e['positive']>=4;assert go==summary['decision']['conditional_target_promising_development_only']
 report=dict(status='PASS',states=400,eligible_edges=len(probability),guarded_states=sum(guarded),all_original_features_labels_weights_identical=True,public_guard_reconstructed=True,folds=refits,effective_training_states_per_fold=192,all_guarded_rows_excluded_from_fit_and_scaler=True,all_held_out_rows_once=True,all_static_context_metadata_verified=True,legacy_adapter_rejects_new_context_models=True,all_group_metrics_recomputed=True,model_recipes_reproduced=refits,prediction_screen=go,protocol_sha256=b.sha(a.results_root/'protocol.json'),summary_sha256=b.sha(a.results_root/'analysis/summary.json'),audit_type='primary agent self-audit; same arrays, public guard, 25 filtered recipe refits, warm-only scaler and paired group statistics',independent_confirmation=False)
 b.write(a.validation_root/'raw_record_audit.json',report);print(json.dumps(report),flush=True)
