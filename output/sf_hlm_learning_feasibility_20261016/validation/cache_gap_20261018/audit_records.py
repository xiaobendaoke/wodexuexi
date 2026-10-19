"""Raw-array/feature/refit audit of the single public fetch-gap experiment."""
import argparse,json,math,pickle,statistics,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);o=p.parse_args();a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(o.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_cache_gap as c
b=c.b
with b.res(a).CpuLease(a.resources,task=['cache_gap19_raw_audit'],kind='environment'):
 import numpy as np
 from scipy.stats import t
 from utils.sf_hlm_cache_gap import validate_bundle,MODE,RULE,FEATURES
 from utils.sf_hlm_cache_calibration import cache_features,fit_model,eligible,validate_bundle as legacy_validate
 from utils.sf_hlm_cache_mixture import decode_row
 from utils.sf_hlm_cache_context import public_context
 protocol=b.read(a.results_root/'protocol.json');prepared=b.read(a.results_root/'prepared.json');done=b.read(a.results_root/'completion.json');s=b.read(a.results_root/'analysis/summary.json');b.verify(a,protocol)
 for path,digest in protocol['input_files'].items():assert b.sha(path)==digest
 assert done['status']=='COMPLETED' and done['protocol_sha256']==b.sha(a.results_root/'protocol.json') and done['summary_sha256']==b.sha(a.results_root/'analysis/summary.json') and prepared['data_sha256']==b.sha(a.results_root/'training_data.npz')
 z=np.load(a.results_root/'training_data.npz');d={k:z[k] for k in z.files};z=np.load(a.source_training/'training_data.npz');old={k:z[k] for k in z.files}
 for key in old:
  if key=='x':assert np.array_equal(old[key],d[key][:,:18])
  else:assert np.array_equal(old[key],d[key])
 checks=0
 for si,state in enumerate(protocol['states']):
  rows=b.read(state['public'])['rows'];flag=public_context(rows)['global']['all_visible_miss'];indices=np.flatnonzero(d['state_index']==si);expected=[]
  assert np.all(d['guarded'][indices]==flag)
  for row in rows:
   row['options']={int(k):v for k,v in row['options'].items()}
   for ac,part in decode_row(row,protocol['public_powers']).items():
    if eligible(part):expected.append(cache_features(row,ac,mode='full_logit_context')+[math.log1p(part['miss_base']-part['hit_base'])]);checks+=1
  assert np.array_equal(np.asarray(expected),d['x'][indices])
 pred=np.full(len(d['y']),np.nan);previous=pred.copy();seen=np.zeros(len(pred),int);refits=0
 for fold in b.splits(a,protocol):
  path=b.folded(a,fold['block'],fold['workload']);bundle=pickle.loads((path/'model.pkl').read_bytes());receipt=b.read(path/'result.json');z=np.load(path/'test_predictions.npz');legacy=np.load(a.source_training/'folds'/path.name/'test_predictions.npz')
  for name,digest in receipt['files'].items():assert b.sha(path/name)==digest
  assert validate_bundle(bundle)==MODE and tuple(bundle['features'])==FEATURES and bundle['fold']==fold and bundle['context_rule']==RULE
  try:legacy_validate(bundle,supported_context_rules=('none',RULE))
  except ValueError:pass
  else:raise AssertionError('Legacy eighteen-dimensional adapter must reject gap19')
  train=np.isin(d['state_index'],fold['train'])&~d['guarded'];test=np.isin(d['state_index'],fold['test']);warm=test&~d['guarded'];assert len(set(d['state_index'][train].tolist()))==192 and not set(fold['train'])&set(fold['test'])
  assert all(protocol['identities'][int(i)][1]!=fold['block'] and protocol['identities'][int(i)][2]!=fold['workload'] for i in d['state_index'][train]);assert bundle['training_state_indices']==sorted(set(d['state_index'][train].tolist()))
  assert np.array_equal(z['row_indices'],np.flatnonzero(test)) and np.array_equal(z['row_indices'],legacy['row_indices']) and np.array_equal(z['guarded'],d['guarded'][test])
  model,diagnostic=fit_model(d['x'][train],d['y'][train],d['weight'][train]);expected=np.zeros(int(test.sum()));expected[~d['guarded'][test]]=model.predict_proba(d['x'][warm])[:,1];assert np.allclose(expected,z['probability'],rtol=0,atol=1e-12)
  assert np.array_equal(bundle['model'].predict_proba(d['x'][warm])[:,1],z['probability'][~d['guarded'][test]])
  if diagnostic['kind']=='weighted_standardized_logistic':
   scaler=bundle['model'].named_steps['standardize'];mean=np.average(d['x'][train],weights=d['weight'][train],axis=0);var=np.average((d['x'][train]-mean)**2,weights=d['weight'][train],axis=0);assert np.allclose(mean,scaler.mean_,rtol=1e-12,atol=1e-12) and np.allclose(var,scaler.var_,rtol=1e-12,atol=1e-12)
  pred[test]=z['probability'];previous[test]=legacy['probability'];seen[test]+=1;refits+=1
 assert np.all(seen==1) and np.all(pred[d['guarded']]==0) and np.array_equal(previous,d['legacy_learned'])
 probs=dict(learned=pred,**{m:d[m] for m in c.METHODS[1:]});edge=np.load(a.decision_results/'edge_context.npz');assert np.array_equal(edge['state_index'],d['state_index'])
 masks=dict(all=np.ones(len(pred),bool),guarded=d['guarded'],non_guarded=~d['guarded'],**{k:edge[k] for k in ('sensitive_union_10','sensitive_union_20')})
 def close(x,y):
  if isinstance(x,dict):
   assert x.keys()==y.keys()
   for k in x:close(x[k],y[k])
  else:assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12),(x,y)
 blocks_by_mask={}
 for name,mask in masks.items():
  blocks={}
  for block in range(620001,620006):
   take=mask&np.asarray([protocol['identities'][int(i)][1]==block for i in d['state_index']]);y=d['y'][take];w=d['weight'][take];metrics={}
   for m,prob in probs.items():
    pp=prob[take];clip=np.clip(pp,1e-12,1-1e-12);metrics[m]=dict(brier=float(np.average((pp-y)**2,weights=w)),log_loss=float(np.average(-y*np.log(clip)-(1-y)*np.log1p(-clip),weights=w)))
   blocks[str(block)]=metrics
  groups={m:{k:statistics.mean(v[m][k] for v in blocks.values()) for k in ('brier','log_loss')} for m in c.METHODS};wanted=s if name=='all' else s['subsets'][name];close(blocks,wanted['blocks']);close(groups,wanted['groups']);blocks_by_mask[name]=blocks
 best=min(c.METHODS[2:],key=lambda m:s['groups'][m]['brier']);assert best==s['best_label_free_control']
 for control in c.METHODS[1:]:
  values=[v[control]['brier']-v['learned']['brier'] for v in blocks_by_mask['all'].values()];mean=statistics.mean(values);radius=float(t.ppf(.975,4))*statistics.stdev(values)/5**.5;wanted=s['brier_improvements'][control];assert math.isclose(mean,wanted['mean_improvement'],abs_tol=1e-12) and np.allclose([mean-radius,mean+radius],wanted['ci95'],rtol=0,atol=1e-12) and sum(x>0 for x in values)==wanted['positive']
 legacy=s['brier_improvements']['legacy_learned'];control=s['brier_improvements'][best];go=legacy['mean_improvement']>=.001 and legacy['positive']>=4 and control['mean_improvement']>0 and control['positive']>=4;assert go==s['decision']['public_gap_feature_promising_development_only']
 report=dict(status='PASS',states=400,eligible_edges=len(pred),public_fetch_features_reconstructed=checks,original_eighteen_fields_labels_weights_and_splits_unchanged=True,model_recipes_reproduced=refits,non_guarded_training_states_per_fold=192,old_adapter_rejects_new_schema=True,all_oof_rows_once=True,fixed_decision_masks_unchanged=True,all_group_metrics_recomputed=True,prediction_screen=go,protocol_sha256=b.sha(a.results_root/'protocol.json'),summary_sha256=b.sha(a.results_root/'analysis/summary.json'),audit_script_sha256=b.sha(__file__),audit_type='primary agent self-audit; public scalar features, original arrays, twenty-five refits and fixed-mask group metrics',independent_confirmation=False)
 b.write(a.validation_root/'raw_record_audit.json',report);print(json.dumps(report),flush=True)
