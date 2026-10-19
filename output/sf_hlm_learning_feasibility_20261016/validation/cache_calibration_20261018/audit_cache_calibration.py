import sys,json,pickle,math,statistics,os
from pathlib import Path
from argparse import Namespace
sys.path.insert(0,"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/cache_calibration_6f284a564884")
from scripts.analysis import sf_hlm_cache_calibration as c
a=Namespace(**{k:Path(v) for k,v in {"release_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/cache_calibration_6f284a564884","feasibility_release":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3","parent_launch":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3/parent_launch.json","old_calibration":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/energy_price_20261017/calibration_v1","closed_loop":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/development_v1","closed_validation":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/priced_closed_loop_20261017","results_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/cache_calibration_20261018/training_v3","validation_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/cache_calibration_20261018","resources":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/train_logs/resources","resource_module":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/extension_release_v2_20261001/utils/sf_hlm_resources.py","precision_diagnostic":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/cache_calibration_20261018/physical_precision_diagnostic.json"}.items()})
r=c.res(a)
with r.CpuLease(a.resources,task=['cache_calibration_raw_audit'],kind='environment'):
 f=c.load('cache_audit_archive',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py');f.boot(a,620001)
 cc,mix=c.helpers(a)
 import numpy as np
 from scipy.special import expit
 p=c.read(a.results_root/'protocol.json');prepared=c.read(a.results_root/'prepared.json');done=c.read(a.results_root/'completion.json');summary=c.read(a.results_root/'analysis/summary.json')
 assert done['summary_sha256']==c.sha(a.results_root/'analysis/summary.json') and done['protocol_sha256']==c.sha(a.results_root/'protocol.json')
 c.verify(a,p)
 for path,digest in p['input_files'].items():assert c.sha(path)==digest
 assert c.sha(a.results_root/'training_data.npz')==prepared['data_sha256']
 data=np.load(a.results_root/'training_data.npz');layout=c.read(a.results_root/'layout.json');assert len(data['y'])==13459 and len(layout)==400
 pw=c.read(a.results_root/'preparation_audit.json')['powers'];expected_x=[];expected_y=[];expected_si=[];expected_controls={name:[] for name in c.METHODS[1:]}
 for si,state in enumerate(p['states']):
  public=c.read(state['public']);env,obs=pickle.loads(Path(state['observed']).read_bytes());ues={u.id:u for u in env.ues};uavs={u.id:u for u in env.uavs}
  for row in public['rows']:
   row['options']={int(k):v for k,v in row['options'].items()};parts=mix.decode_row(row,pw)
   for action,part in parts.items():
    if not cc.eligible(part):continue
    expected_x.append(cc.cache_features(row,action));expected_y.append(float(bool(uavs[row['options'][action]['target']].cache[ues[row['ue_id']].current_request.req_id])));expected_si.append(si)
    for name,value in cc.controls(row,action).items():expected_controls[name].append(value)
 assert np.array_equal(data['x'],np.asarray(expected_x)) and np.array_equal(data['y'],np.asarray(expected_y)) and np.array_equal(data['state_index'],np.asarray(expected_si))
 for name,values in expected_controls.items():assert np.array_equal(data[name],np.asarray(values))
 for si in range(400):assert abs(data['weight'][data['state_index']==si].sum()-1)<1e-12
 probability=np.full(len(data['y']),np.nan);seen=np.zeros(len(probability),int);recipes=0
 for fold in c.splits(a,p):
  assert len(fold['train'])==256 and len(fold['test'])==16 and not set(fold['train'])&set(fold['test'])
  assert all(p['identities'][i][1]!=fold['block'] and p['identities'][i][2]!=fold['workload'] for i in fold['train'])
  path=c.folded(a,fold['block'],fold['workload']);receipt=c.read(path/'result.json');bundle=pickle.loads((path/'model.pkl').read_bytes());pred=np.load(path/'test_predictions.npz')
  for name,digest in receipt['files'].items():assert c.sha(path/name)==digest
  assert bundle['fold']==fold and bundle['protocol_sha256']==c.sha(a.results_root/'protocol.json')
  train=np.isin(data['state_index'],fold['train']);test=np.isin(data['state_index'],fold['test']);assert np.array_equal(pred['row_indices'],np.flatnonzero(test))
  replay,diag=cc.fit_model(data['x'][train],data['y'][train],data['weight'][train]);pp=replay.predict_proba(data['x'][test])[:,1]
  assert np.allclose(pp,pred['probability'],rtol=0,atol=1e-12)
  if diag['kind']=='weighted_standardized_logistic':
   scaler=bundle['model'].named_steps['standardize'];lr=bundle['model'].named_steps['logistic']
   mean=np.average(data['x'][train],axis=0,weights=data['weight'][train]);var=np.average((data['x'][train]-mean)**2,axis=0,weights=data['weight'][train])
   assert np.allclose(mean,scaler.mean_,rtol=1e-12,atol=1e-12) and np.allclose(var,scaler.var_,rtol=1e-12,atol=1e-12)
   manual=expit(scaler.transform(data['x'][test])@lr.coef_.T+lr.intercept_).ravel()
   assert np.allclose(manual,pp,rtol=0,atol=1e-12)
  probability[test]=pred['probability'];seen[test]+=1;recipes+=1
 assert np.all(seen==1)
 allp=dict(learned=probability,**{name:data[name] for name in c.METHODS[1:]});blocks={}
 for block in range(620001,620006):
  states=[i for i,z in enumerate(p['identities']) if z[1]==block];take=np.isin(data['state_index'],states);yy=data['y'][take];w=data['weight'][take];methods={}
  for name,pp in allp.items():
   pr=pp[take];clipped=np.clip(pr,1e-12,1-1e-12)
   methods[name]=dict(brier=float(np.average((pr-yy)**2,weights=w)),log_loss=float(np.average(-yy*np.log(clipped)-(1-yy)*np.log(1-clipped),weights=w)))
   for metric,value in methods[name].items():assert math.isclose(value,summary['blocks'][str(block)][name][metric],rel_tol=1e-12,abs_tol=1e-12)
  blocks[str(block)]=methods
 groups={name:{metric:statistics.mean(b[name][metric] for b in blocks.values()) for metric in ['brier','log_loss']} for name in c.METHODS}
 best=min(c.METHODS[1:],key=lambda name:groups[name]['brier']);values=[b[best]['brier']-b['learned']['brier'] for b in blocks.values()]
 assert best==summary['best_label_free_control'] and (statistics.mean(values)>0 and sum(v>0 for v in values)>=4)==summary['decision']['cache_hit_target_promising_development_only']
 physical=c.read(a.results_root/'preparation_audit.json');assert physical['public_inverse_parity_passed'] is False and physical['counts']['public_inverse_parity_failures']==540 and physical['max_physical_residual']<1e-12
 assert c.sha(a.results_root/'physical_residuals.npz')==physical['all_residuals_sha256']
 report=dict(status='PASS',source_states=400,eligible_edges=len(data['y']),labels_and_features_reconstructed=len(expected_y),per_state_weight_verified=True,fold_training_recipes_reproduced=recipes,all_test_rows_exactly_once=True,all_state_block_metrics_recomputed=True,public_inverse_precision_failure_retained=True,summary_sha256=c.sha(a.results_root/'analysis/summary.json'),protocol_sha256=c.sha(a.results_root/'protocol.json'),best_control=best,brier_improvement_values=values,best_control_improved_blocks=sum(v>0 for v in values),independent_confirmation=False,audit_type='primary agent self-audit; original cache labels/public inputs, training-only scaler and 25 model refits, raw weighted statistics',cpu=sorted(os.sched_getaffinity(0)))
 c.write(a.validation_root/'raw_record_audit.json',report);print(json.dumps(report),flush=True)
