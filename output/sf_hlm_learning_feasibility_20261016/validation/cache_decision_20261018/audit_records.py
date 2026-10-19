"""Separate scalar audit of read-only decision diagnostics; run remotely."""
import argparse,json,math,pickle,statistics,sys
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);o=p.parse_args()
a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(o.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_cache_matching as b
with b.resource(a).CpuLease(a.resources,task=['cache_decision_raw_audit'],kind='environment'):
 import numpy as np
 b.archive(a).boot(a,620001);b.modules(a)
 from utils.sf_hlm_cache_mixture import decode_row
 from utils.sf_hlm_cache_calibration import eligible
 root=o.results;protocol=b.read(root/'protocol.json');result=b.read(root/'analysis.json');done=b.read(root/'completion.json');states=b.read(root/'states.json');sp=b.read(a.source_results/'protocol.json')
 assert done['status']=='COMPLETED_READ_ONLY' and done['analysis_sha256']==b.sha(root/'analysis.json') and done['states_sha256']==b.sha(root/'states.json')
 for path,digest in protocol['input_files'].items():assert b.sha(path)==digest
 assert result['protocol_sha256']==b.sha(root/'protocol.json') and result['source_inventory_sha256']==b.sha(root/'source_inventory.json') and result['edge_context_sha256']==b.sha(root/'edge_context.npz')
 z=np.load(a.cache_training/'training_data.npz');d={k:z[k] for k in z.files};z=np.load(root/'edge_context.npz');edges={k:z[k] for k in z.files};assert np.array_equal(edges['state_index'],d['state_index'])
 pred=np.full(len(d['y']),np.nan);seen=np.zeros(len(pred),int)
 for fold in (a.cache_training/'folds').iterdir():
  z=np.load(fold/'test_predictions.npz');pred[z['row_indices']]=z['probability'];seen[z['row_indices']]+=1
 assert np.all(seen==1)
 methods=protocol['methods'];variants=['learned','cold_half','cold_prior'];probs=dict(learned=pred,**{k:d[k] for k in ('prior','own_proxy','half_proxy','cold_guard_prior','cold_guard_own','cold_guard_half')});checks=0
 for si,entry in enumerate(sp['states']):
  identity=entry['identity'];td=a.source_results/'tasks'/('cache_'+'_'.join(map(str,identity)));pub=b.read(td/'public.json');labels=b.read(td/'labels.json');rows=b.read(entry['public_rows'])['rows'];actual=states[si]
  for r in rows:r['options']={int(k):v for k,v in r['options'].items()}
  refs={m:next(x for x in pub['selections'] if x['method']==m) for m in methods};counts={m:Counter(r['options'][ac]['target'] for r,ac in zip(rows,refs[m]['assignment'])) for m in methods};prices=b.read(a.calibration_root/'folds'/b.fold_name(identity[1],identity[2])/'prices.json');parts=[decode_row(r,prices['powers']) for r in rows];edge_lookup={}
  indices=np.flatnonzero(d['state_index']==si);expected=[(ri,ac) for ri,pp in enumerate(parts) for ac,part in pp.items() if eligible(part)];assert len(expected)==len(indices)
  for j,(ri,ac) in zip(indices,expected):
   assert edges['row'][j]==ri and edges['action'][j]==ac;edge_lookup[ri,ac]=int(j);part=parts[ri][ac];row=rows[ri];gap=part['miss_base']-part['hit_base'];assert math.isclose(gap,edges['public_fetch_gap'][j],rel_tol=0,abs_tol=1e-12)
   union={10:False,20:False}
   for m in methods:
    selected=refs[m]['assignment'][ri]==ac;assert bool(edges['selected_'+m][j])==selected
    q=counts[m][row['options'][ac]['target']]
    for ms in (10,20):
     hit=part['hit_base']+part['work']*q+ms/1000;miss=part['miss_base']+part['work']*q+ms/1000;critical=q>0 and hit<=row['deadline'] and miss>row['deadline'];assert bool(edges['sensitive_'+str(ms)+'_'+m][j])==critical;union[ms]|=critical;checks+=1
   for ms in (10,20):assert bool(edges['sensitive_union_'+str(ms)][j])==union[ms]
  for m in methods:
   requests=labels['outcomes'][labels['mapping'][m+':0']]['requests']
   for ms in (0,10,20,30):assert actual['actual'][m][str(ms)]==sum(r['latency']+ms/1000<=r['deadline'] for r in requests)
   for variant in variants:
    scores=[]
    for ri,(row,ac) in enumerate(zip(rows,refs[m]['assignment'])):
     part=parts[ri][ac];target=row['options'][ac]['target'];load=1 if target=='mbs' else max(counts[m][target],1);compute=part['work']*load
     j=edge_lookup.get((ri,ac));prob=part['p'] if j is None else probs[{'learned':'learned','cold_half':'cold_guard_half','cold_prior':'cold_guard_prior'}[variant]][j]
     score=float(part['base']+compute+.010<=row['deadline']) if part['deterministic'] or part['fallback'] is not None else prob*float(part['hit_base']+compute+.010<=row['deadline'])+(1-prob)*float(part['miss_base']+compute+.010<=row['deadline']);scores.append(score)
    expected=math.fsum(scores);sealed=actual['cross'][variant][m];assert math.isclose(expected,sealed['expected_success_10ms'],abs_tol=1e-10,rel_tol=0)
    param=prices['prices'][variant];priced=expected-param['price']*refs[m]['diagnostic']['chosen_predicted']['energy']/param['normalizer'];assert math.isclose(priced,sealed['priced_score'],abs_tol=1e-10,rel_tol=0)
  for m in methods[1:]:
   category='same_assignment' if refs[m]['assignment']==refs[methods[0]]['assignment'] else 'same_quota_different_assignment' if counts[m]==counts[methods[0]] else 'different_quota';assert actual['categories'][m]==category
 n=len(pred);masks={'all':np.ones(n,bool),'non_guarded':~d['guarded'],'guarded':d['guarded']}
 for m in methods:
  masks['selected_'+m]=edges['selected_'+m]
  for ms in (10,20):masks['sensitive_'+str(ms)+'_'+m]=edges['sensitive_'+str(ms)+'_'+m];masks['selected_sensitive_'+str(ms)+'_'+m]=edges['selected_'+m]&edges['sensitive_'+str(ms)+'_'+m]
 for ms in (10,20):masks['sensitive_union_'+str(ms)]=edges['sensitive_union_'+str(ms)]
 for name,mask in {'own_missing':d['x'][:,3]==0,'own_miss':(d['x'][:,3]==1)&(d['x'][:,2]==0),'own_hit':(d['x'][:,3]==1)&(d['x'][:,2]==1)}.items():
  masks[name]=mask
  for ms in (10,20):masks[name+'_sensitive_'+str(ms)]=mask&edges['sensitive_union_'+str(ms)]
 for lo,hi in ((0.,.01),(.01,.05),(.05,.1),(.1,.5),(.5,np.inf)):
  mask=(edges['public_fetch_gap']>lo)&(edges['public_fetch_gap']<=hi);key=f'fetch_gap_({lo},{hi}]';masks[key]=mask
  for ms in (10,20):masks[key+'_sensitive_'+str(ms)]=mask&edges['sensitive_union_'+str(ms)]
 def check(mask,wanted):
  ii=np.flatnonzero(mask);assert len(ii)==wanted['rows'] and len(set(d['state_index'][ii].tolist()))==wanted['states']
  if not len(ii):assert wanted['methods']=={};return
  w=[float(d['weight'][i]) for i in ii];total=math.fsum(w);assert math.isclose(total,wanted['original_weight'],rel_tol=1e-12,abs_tol=1e-12)
  for method,prob in probs.items():
   brier=math.fsum(ww*(float(prob[i])-float(d['y'][i]))**2 for i,ww in zip(ii,w))/total;loss=[]
   for i,ww in zip(ii,w):
    value=min(1-1e-12,max(1e-12,float(prob[i])));y=float(d['y'][i]);loss.append(ww*(-y*math.log(value)-(1-y)*math.log1p(-value)))
   for key,value in (('brier',brier),('log_loss',math.fsum(loss)/total)):assert math.isclose(value,wanted['methods'][method][key],rel_tol=1e-12,abs_tol=1e-12)
 assert set(masks)==set(result['groups'])
 for name,mask in masks.items():
  check(mask,result['groups'][name])
  for block,wanted in result['blocks'].items():check(mask&np.asarray([str(sp['identities'][int(i)][1])==block for i in d['state_index']]),wanted[name])
 report=dict(status='PASS',states=400,edges=n,sensitivity_scalar_checks=checks,metric_groups=len(masks),scalar_losses_and_cross_scores_recomputed=True,source_hashes_verified=True,analysis_sha256=b.sha(root/'analysis.json'),protocol_sha256=b.sha(root/'protocol.json'),audit_type='primary agent self-audit, separate scalar formulas and source records',independent_confirmation=False)
 b.write(o.out,report);print(json.dumps(report),flush=True)
