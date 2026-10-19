import sys,json,pickle,collections,math,os
from pathlib import Path
from argparse import Namespace
sys.path.insert(0,"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/priced_closed_loop_a3e5f7128a06")
from scripts.analysis import sf_hlm_priced_closed_loop as c
a=Namespace(**{k:Path(v) for k,v in {"release_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/priced_closed_loop_a3e5f7128a06","feasibility_release":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3","parent_launch":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3/parent_launch.json","old_training":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/calibrated_matching_20261017/training_v1","old_price_calibration":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/energy_price_20261017/calibration_v1","native_results":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_runtime_20261017/logging_development_v2","native_validation":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/priced_runtime_20261017","fit_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/final_fit_v1","smoke_results":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/smoke_v1","results_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/priced_closed_loop_20261017/smoke_v1","validation_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/priced_closed_loop_20261017","resource_module":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/extension_release_v2_20261001/utils/sf_hlm_resources.py","resources":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/train_logs/resources"}.items()})
res=c.resources(a)
with res.CpuLease(a.resources,task=['priced_closed_loop_preflight_audit'],kind='environment'):
 c.archive(a).boot(a,620001);fast=c.public_modules(a)
 from utils.sf_hlm_action_ranker import public_payload
 from utils.sf_hlm_priced_runtime import PricedPolicy,LocalEdgeKernel
 from utils.sf_hlm_energy_price import calibrate
 from utils.sf_hlm_calibrated_matching import edge_features
 from utils.sf_hlm_cache_mixture import decode_row
 import numpy as np
 p=c.read(a.fit_root/'protocol.json');done=c.read(a.fit_root/'completion.json')
 bundle=pickle.loads((a.fit_root/'model.pkl').read_bytes());prices=c.read(a.fit_root/'prices.json');cases=c.read(a.fit_root/'cases.json')
 assert bundle['model'].get_params()=={**bundle['model'].get_params(),**p['hgb']}
 assert bundle['training_state_indices']==list(range(400))
 assert len(p['identities'])==400 and done['training_rows']==98260
 for path,digest in p['input_files'].items():assert c.sha(path)==digest
 for name,digest in done['files'].items():assert c.sha(a.fit_root/name)==digest
 fallback=collections.Counter();edges=0;parity=0;candidate_sets=0
 zero=dict(prices={obj:dict(price=0.,normalizer=1.) for obj in cases})
 for si,identity in enumerate(p['identities']):
  suffix='oracle_'+'_'.join(map(str,identity))+'/public.json';path=next(path for path in p['input_files'] if path.endswith(suffix))
  rows=public_payload(c.read(path))['rows']
  for row in rows:
   for part in decode_row(row,prices['powers']).values():
    edges+=1
    if part['fallback']:fallback[part['fallback']]+=1
    elif not part['deterministic']:
     mean=part['p']*part['hit_base']+(1-part['p'])*part['miss_base']
     assert abs(mean-part['base'])<=1e-8*max(1.,abs(part['base']))
  with fast.FastRuntime() as rt:
   policy=PricedPolicy(rt,bundle['model'],zero,prices['powers'])
   for obj,values in cases.items():
    _,diag,cc=policy.run(rows,num_uavs=5,kind=obj+'_price',trace=True)
    observed=[dict(action=x['action'],assignment=x['assignment'],score=x['expected_success'],energy=x['predicted']['energy']) for x in cc]
    assert c.normal(observed)==values[si]['candidates']
    assert values[si]['identity']==identity and values[si]['state_index']==si
    candidate_sets+=1
   # Native local features/probabilities versus the original scalar scorer.
   g,repair,base=policy.initialize(rows)
   from utils.sf_hlm_action_ranker import features
   from utils.sf_hlm_request_risk import proposal_actions
   from utils.sf_hlm_calibrated_matching import quota_set
   ids,x=features(dict(rows=rows,greedy_assignment=g,initial_assignment=base.assignment,initial_counts=base.current.tolist(),energy_cap=base.cap,legal_nonstop_actions=[int(k)+1 for k in np.flatnonzero(base.action_mask()[1:])]),10.,problem=base.problem)
   qs=[q for _,q in quota_set(base.current.tolist(),proposal_actions(ids,x))]
   kernel=LocalEdgeKernel(rows,base.current,10.,base.problem);matrices=kernel.probabilities(qs,bundle['model'])
   for q in qs:
    xx,ii,_,_=edge_features(rows,q,10.,problem=base.problem)
    assert np.array_equal(xx,kernel.features(q))
    if len(xx):
     probs=bundle['model'].predict_proba(xx)[:,list(bundle['model'].classes_).index(1.)]
     assert np.array_equal(probs,matrices[tuple(q)][ii[:,0],ii[:,1]])
    parity+=1
 for obj,values in cases.items():
  assert c.normal(calibrate(values,1.05))==prices['prices'][obj]
 smoke=c.read(a.smoke_results/'protocol.json')
 assert len(smoke['tasks'])==32 and smoke['steps']==16
 for task in smoke['tasks']:
  rr=c.read(a.smoke_results/'tasks'/('_'.join(map(str,task)))/'result.json')
  assert rr['zero_injection_equivalence']['full_pickled_state_equal']
  if task[3]=='zero':assert rr['zero_replay_equal']
 report=dict(status='PASS',training_states=400,rows=done['training_rows'],public_candidate_sets_recomputed=candidate_sets,real_quota_feature_probability_parity=parity,public_edges_checked=edges,mixture_fallback_counts=dict(fallback),prices_recomputed=True,old_load_episodes=32,zero_replay_episodes=16,zero_full_state_equivalence=32,fit_protocol_sha256=c.sha(a.fit_root/'protocol.json'),fit_completion_sha256=c.sha(a.fit_root/'completion.json'),smoke_completion_sha256=c.sha(a.smoke_results/'completion.json'),audit_type='primary agent self-audit; all public training candidate sets and original scalar feature checks',cpu=sorted(os.sched_getaffinity(0)))
 c.write(a.validation_root/'preflight_audit.json',report)
 print(json.dumps(report),flush=True)
