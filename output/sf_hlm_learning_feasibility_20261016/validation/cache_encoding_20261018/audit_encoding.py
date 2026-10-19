"""Remote self-audit of a frozen encoding run; no fitting outside a CPU lease."""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys

parser=argparse.ArgumentParser();parser.add_argument('--launch',type=Path,required=True);options=parser.parse_args()
launch=json.loads(options.launch.read_text());a=argparse.Namespace(**{k:Path(v) for k,v in launch['args'].items()})
sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_cache_encoding as e
c=e.base
with c.res(a).CpuLease(a.resources,task=['cache_encoding_raw_audit'],kind='environment'):
 import numpy as np
 from scipy.special import expit
 from scipy.stats import t
 from utils import sf_hlm_cache_calibration as cc
 from utils.sf_hlm_cache_mixture import decode_row
 p=c.read(a.results_root/'protocol.json');prepared=c.read(a.results_root/'prepared.json');done=c.read(a.results_root/'completion.json');summary=c.read(a.results_root/'analysis/summary.json')
 c.verify(a,p)
 for path,digest in p['input_files'].items():assert c.sha(path)==digest
 assert done['summary_sha256']==c.sha(a.results_root/'analysis/summary.json') and done['protocol_sha256']==c.sha(a.results_root/'protocol.json')
 assert c.sha(a.results_root/'training_data.npz')==prepared['data_sha256'] and c.sha(a.results_root/'layout.json')==prepared['layout_sha256']
 old=np.load(a.source_training/'training_data.npz');data=np.load(a.results_root/'training_data.npz')
 assert data.files==old.files and len(data['y'])==13459 and p['changed_columns']==[0,4] and p['feature_mode']==e.MODE
 for key in data.files:
  if key!='x':assert np.array_equal(data[key],old[key])
 keep=[j for j in range(18) if j not in (0,4)]
 assert np.array_equal(data['x'][:,keep],old['x'][:,keep])
 assert np.array_equal(data['x'][:,0],np.asarray([(math.log(float(v))-math.log1p(-float(v)))/12 for v in data['prior']]))
 assert np.array_equal(data['x'][:,4],data['x'][:,0]*data['x'][:,2])
 powers=c.read(a.source_training/'preparation_audit.json')['powers'];expected=[]
 for si,state in enumerate(p['states']):
  rows=c.read(state['public'])['rows'];xx=[]
  for row in rows:
   row['options']={int(k):v for k,v in row['options'].items()}
   for action,part in decode_row(row,powers).items():
    if cc.eligible(part):xx.append(cc.cache_features(row,action,mode=e.MODE))
  assert np.array_equal(np.asarray(xx),data['x'][data['state_index']==si]);expected.extend(xx)
  assert math.isclose(data['weight'][data['state_index']==si].sum(),1.,abs_tol=1e-12)
 probability=np.full(len(data['y']),np.nan);previous=probability.copy();seen=np.zeros(len(probability),int);recipes=0
 for fold in c.splits(a,p):
  assert len(fold['train'])==256 and len(fold['test'])==16 and not set(fold['train'])&set(fold['test'])
  assert all(p['identities'][i][1]!=fold['block'] and p['identities'][i][2]!=fold['workload'] for i in fold['train'])
  path=c.folded(a,fold['block'],fold['workload']);receipt=c.read(path/'result.json');bundle=pickle.loads((path/'model.pkl').read_bytes());pred=np.load(path/'test_predictions.npz');older=np.load(a.source_training/'folds'/path.name/'test_predictions.npz')
  for name,digest in receipt['files'].items():assert c.sha(path/name)==digest
  assert bundle['fold']==fold and bundle['protocol_sha256']==c.sha(a.results_root/'protocol.json') and cc.validate_bundle(bundle)==e.MODE
  train=np.isin(data['state_index'],fold['train']);test=np.isin(data['state_index'],fold['test'])
  assert np.array_equal(pred['row_indices'],np.flatnonzero(test)) and np.array_equal(pred['row_indices'],older['row_indices'])
  replay,diag=cc.fit_model(data['x'][train],data['y'][train],data['weight'][train]);pp=replay.predict_proba(data['x'][test])[:,1]
  assert np.allclose(pp,pred['probability'],rtol=0,atol=1e-12)
  assert np.array_equal(pred['probability'],bundle['model'].predict_proba(data['x'][test])[:,1])
  if diag['kind']=='weighted_standardized_logistic':
   scaler=bundle['model'].named_steps['standardize'];lr=bundle['model'].named_steps['logistic'];mean=np.average(data['x'][train],axis=0,weights=data['weight'][train]);var=np.average((data['x'][train]-mean)**2,axis=0,weights=data['weight'][train])
   assert np.allclose(mean,scaler.mean_,rtol=1e-12,atol=1e-12) and np.allclose(var,scaler.var_,rtol=1e-12,atol=1e-12)
   assert np.allclose(expit(scaler.transform(data['x'][test])@lr.coef_.T+lr.intercept_).ravel(),pp,rtol=0,atol=1e-12)
  probability[test]=pred['probability'];previous[test]=older['probability'];seen[test]+=1;recipes+=1
 assert np.all(seen==1)
 allp=dict(learned=probability,**{name:data[name] for name in c.METHODS[1:]});blocks={};encoding=[]
 for block in range(620001,620006):
  states=[i for i,z in enumerate(p['identities']) if z[1]==block];mask=np.isin(data['state_index'],states);yy=data['y'][mask];ww=data['weight'][mask];methods={}
  for name,pp in allp.items():
   pr=pp[mask];clip=np.clip(pr,1e-12,1-1e-12);methods[name]=dict(brier=float(np.average((pr-yy)**2,weights=ww)),log_loss=float(np.average(-yy*np.log(clip)-(1-yy)*np.log(1-clip),weights=ww)))
   for metric,value in methods[name].items():assert math.isclose(value,summary['blocks'][str(block)][name][metric],rel_tol=1e-12,abs_tol=1e-12)
  blocks[str(block)]=methods;encoding.append(float(np.average((previous[mask]-yy)**2-(probability[mask]-yy)**2,weights=ww)))
 groups={name:{metric:statistics.mean(b[name][metric] for b in blocks.values()) for metric in ('brier','log_loss')} for name in c.METHODS}
 for name,metrics in groups.items():
  for metric,value in metrics.items():assert math.isclose(value,summary['groups'][name][metric],abs_tol=1e-12)
 best=min(c.METHODS[1:],key=lambda name:groups[name]['brier']);values=[b[best]['brier']-b['learned']['brier'] for b in blocks.values()];old_mean=statistics.mean(encoding);radius=float(t.ppf(.975,4))*statistics.stdev(encoding)/5**.5
 assert np.allclose(encoding,summary['encoding_vs_clipped']['values'],rtol=0,atol=1e-12) and np.allclose([old_mean-radius,old_mean+radius],summary['encoding_vs_clipped']['ci95'],rtol=0,atol=1e-12)
 gate=statistics.mean(values)>0 and sum(v>0 for v in values)>=4 and old_mean>0 and sum(v>0 for v in encoding)>=4
 assert gate==summary['decision']['encoding_promising_development_only']
 old_audit=c.read(a.source_validation/'raw_record_audit.json');physical=c.read(a.source_training/'preparation_audit.json')
 assert old_audit['status']=='PASS' and physical['counts']['public_inverse_parity_failures']==540 and physical['max_physical_residual']<1e-12 and c.sha(a.source_training/'physical_residuals.npz')==physical['all_residuals_sha256']
 report=dict(status='PASS',source_states=400,eligible_edges=len(data['y']),feature_rows_reconstructed=len(expected),only_columns_changed=[0,4],labels_weights_controls_splits_exactly_unchanged=True,fold_training_recipes_reproduced=recipes,all_test_rows_exactly_once=True,static_encoding_metadata_verified=True,summary_sha256=c.sha(a.results_root/'analysis/summary.json'),protocol_sha256=c.sha(a.results_root/'protocol.json'),source_label_audit_sha256=c.sha(a.source_validation/'raw_record_audit.json'),public_inverse_precision_failure_retained=True,encoding_screen=gate,audit_type='primary agent self-audit; public feature reconstruction, inherited audited identical labels, 25 exact recipe refits and weighted statistics',independent_confirmation=False,cpu=sorted(os.sched_getaffinity(0)))
 c.write(a.validation_root/'raw_record_audit.json',report);print(json.dumps(report),flush=True)
