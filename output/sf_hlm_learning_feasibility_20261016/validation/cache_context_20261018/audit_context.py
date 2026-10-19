import argparse,json,math,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);arg=parser.parse_args();m=json.loads(arg.manifest.read_text());launch=json.loads(Path(m['source_launch']).read_text());a=argparse.Namespace(**{k:Path(v) for k,v in launch['args'].items()});sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_cache_matching as b
with b.resource(a).CpuLease(a.resources,task=['cache_context_raw_audit'],kind='environment'):
 import numpy as np
 root=Path(m['out']);p=b.read(root/'protocol.json');done=b.read(root/'completion.json');result=b.read(root/'analysis.json');contexts=b.read(root/'states.json');tp=b.read(a.cache_training/'protocol.json');archive=np.load(a.cache_training/'training_data.npz');data={name:archive[name] for name in archive.files}
 for name,digest in p['files'].items():assert b.sha(Path(m['release'])/name)==digest
 for name,digest in p['input_files'].items():assert b.sha(name)==digest
 assert done['analysis_sha256']==b.sha(root/'analysis.json') and result['protocol_sha256']==b.sha(root/'protocol.json') and result['source_inventory_sha256']==b.sha(root/'source_inventory.json')
 inventory=b.read(root/'source_inventory.json')
 for name,digest in inventory['files'].items():assert b.sha(name)==digest
 predictions=np.full(len(data['y']),np.nan);seen=np.zeros(len(predictions),int)
 for file in (a.cache_training/'folds').glob('*/test_predictions.npz'):
  pp=np.load(file);ii=pp['row_indices'];predictions[ii]=pp['probability'];seen[ii]+=1
 assert np.all(seen==1)
 fractions={};source_fractions={};cold=set();cold_late=set()
 for si,state in enumerate(tp['states']):
  rr=b.read(state['public'])['rows'];groups={str(j):dict(requests=0,visible=0,hits=0) for j in range(5)};global_group=dict(requests=0,visible=0,hits=0)
  for row in rr:
   options=[o for o in row['options'].values() if o['target']==row['source']];assert len(options)<=1;visible=int(bool(options));value=int(options[0]['cache_belief']) if visible else 0
   if visible:assert options[0]['cache_belief'] in (0.,1.)
   for group in (global_group,groups[str(row['source'])]):group['requests']+=1;group['visible']+=visible;group['hits']+=value
  for group in (global_group,*groups.values()):group['fraction']=group['hits']/group['visible'] if group['visible'] else None;group['all_visible_miss']=bool(group['visible'] and group['hits']==0)
  expected=dict(global_context_contract='observed_local_options_only_v1',**{'global':global_group},sources=groups);assert contexts[si]['context']==expected and contexts[si]['identity']==state['identity'];fractions[si]=global_group['fraction'];source_fractions[si]={j:g['fraction'] for j,g in groups.items()}
  if global_group['all_visible_miss']:
   cold.add(si)
   if state['identity'][4]!=0:cold_late.add(si)
 guarded=np.asarray([0. if int(si) in cold else pp for si,pp in zip(data['state_index'],data['half_proxy'])]);allp=dict(learned_full=predictions,prior=data['prior'],own_proxy=data['own_proxy'],half_proxy=data['half_proxy'],cold_guard_half=guarded)
 def check(indices,reported):
  indices=list(indices);weight=math.fsum(float(data['weight'][i]) for i in indices);assert reported['rows']==len(indices) and reported['states']==len({int(data['state_index'][i]) for i in indices}) and math.isclose(reported['original_weight'],weight,abs_tol=1e-12)
  if not indices:return
  hit=math.fsum(float(data['weight'][i])*float(data['y'][i]) for i in indices)/weight;assert math.isclose(hit,reported['hit_rate'],abs_tol=1e-12)
  for name,pr in allp.items():
   score=math.fsum(float(data['weight'][i])*(float(pr[i])-float(data['y'][i]))**2 for i in indices)/weight
   def loss_term(i):
    pp=max(1e-12,min(1-1e-12,float(pr[i])));yy=float(data['y'][i]);ww=float(data['weight'][i])
    return ww*(-yy*math.log(pp)-(1-yy)*math.log1p(-pp))
   loss=math.fsum(loss_term(i) for i in indices)/weight
   assert math.isclose(score,reported['methods'][name]['brier'],abs_tol=1e-12,rel_tol=1e-12) and math.isclose(loss,reported['methods'][name]['log_loss'],abs_tol=1e-12,rel_tol=1e-12)
 ii=range(len(data['y']));check(ii,result['all']);check((i for i in ii if tp['identities'][int(data['state_index'][i])][4]==0),result['step0']);check((i for i in ii if tp['identities'][int(data['state_index'][i])][4]!=0),result['later'])
 for block in range(620001,620006):check((i for i in ii if tp['identities'][int(data['state_index'][i])][1]==block),result['blocks'][str(block)])
 def box(value):
  if value is None:return 'none_visible'
  if value==0:return 'all_visible_miss'
  for lo,hi in ((0.,.25),(.25,.5),(.5,.75),(.75,1.)):
   if lo<value<=hi:return f'({lo},{hi}]'
  raise ValueError(value)
 for field,get in [('global_bins',lambda i:fractions[int(data['state_index'][i])]),('source_bins',lambda i:source_fractions[int(data['state_index'][i])][str(int(np.argmax(data['x'][i,8:13])))])]:
  for name,reported in result[field].items():check((i for i in ii if box(get(i))==name),reported)
 assert len(cold)==result['guarded_states'] and len(cold_late)==result['guarded_later_states'] and sum(int(data['y'][i]) for i in ii if int(data['state_index'][i]) in cold)==result['guarded_edges_with_true_neighbor_hit']
 helper=b.load('context_statistical_audit_helpers',a.source_validation/'audit_native.py')
 for name in allp:
  if name=='cold_guard_half':continue
  helper.close(helper.effect([x['methods'][name]['brier']-x['methods']['cold_guard_half']['brier'] for x in result['blocks'].values()]),result['cold_guard_brier_improvement'][name])
 assert not result['models_retrained'] and not result['matching_called'] and not result['native_timing_called']
 report=dict(status='PASS',source_states=400,eligible_edges=len(data['y']),all_public_contexts_independently_reconstructed=True,all_groups_brier_log_loss_recomputed_with_scalar_sums=True,held_out_rows_once=True,source_inventory_hashes_verified=True,guarded_states=len(cold),guarded_later_states=len(cold_late),analysis_sha256=b.sha(root/'analysis.json'),protocol_sha256=b.sha(root/'protocol.json'),audit_type='primary agent self-audit; separate public-option counting and scalar weighted losses',independent_confirmation=False)
 b.write(Path(m['validation'])/'raw_record_audit.json',report);print(json.dumps(report),flush=True)
