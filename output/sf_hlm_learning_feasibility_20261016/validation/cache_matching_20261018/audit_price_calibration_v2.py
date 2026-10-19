import sys,pickle,json,math,os
from pathlib import Path
from argparse import Namespace
sys.path.insert(0,"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/cache_matching_f1377426c604")
from scripts.analysis import sf_hlm_cache_matching as c
a=Namespace(**{k:Path(v) for k,v in {"release_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/cache_matching_f1377426c604","cache_training":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/cache_calibration_20261018/training_v3","cache_validation":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/cache_calibration_20261018","old_price_calibration":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/energy_price_20261017/calibration_v1","old_hgb_training":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/calibrated_matching_20261017/training_v1","old_native_validation":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/priced_runtime_20261017","feasibility_release":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3","parent_launch":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/assets/releases/d448ad95f9a3/parent_launch.json","calibration_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/cache_matching_20261018/calibration_v1","smoke_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/cache_matching_20261018/smoke_v1","results_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/results/sf_hlm_learning_feasibility_20261016/cache_matching_20261018/smoke_v1","validation_root":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/output/sf_hlm_learning_feasibility_20261016/validation/cache_matching_20261018","resources":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/train_logs/resources","resource_module":"/home/PengYanghan/Lunwen/sf-hlm-v2-20261001/extension_release_v2_20261001/utils/sf_hlm_resources.py"}.items()})
res=c.resource(a)
with res.CpuLease(a.resources,task=['cache_matching_price_audit'],kind='environment'):
 c.archive(a).boot(a,620001);c.modules(a)
 from utils.sf_hlm_action_ranker import public_payload
 from utils.sf_hlm_cache_matching import CacheWeightKernel
 from utils.sf_hlm_energy_price import calibrate
 from utils.sf_hlm_matching import allocation_counts
 from utils.sf_hlm_calibrated_matching import solve
 from utils.deadline_repair import assess
 p=c.read(a.calibration_root/'protocol.json');done=c.read(a.calibration_root/'completion.json');c.verify(a,p,True)
 assert done['status']=='COMPLETED' and done['protocol_sha256']==c.sha(a.calibration_root/'protocol.json')
 rows=[public_payload(c.read(s['public_rows']))['rows'] for s in p['states']]
 checked=0;optima=0;prices_out={};folds=list(c.splits(a,p))
 for fi,fold in enumerate(folds):
  name=c.fold_name(fold['block'],fold['workload']);td=a.calibration_root/'folds'/name;prices=c.read(td/'prices.json');cases=c.read(td/'cases.json');receipt=c.read(td/'result.json')
  for file,digest in receipt['files'].items():assert c.sha(td/file)==digest
  assert prices['fold']==fold and receipt['fold']==fold and prices['protocol_sha256']==c.sha(a.calibration_root/'protocol.json')
  model_path=a.cache_training/'folds'/name/'model.pkl';bundle=pickle.loads(model_path.read_bytes())
  assert bundle['fold']==fold and prices['cache_model_sha256']==c.sha(model_path) and c.sha(model_path)==p['input_files'][str(model_path)]
  assert not set(fold['train'])&set(fold['test']) and len(fold['train'])==256 and len(fold['test'])==16
  for variant,values in cases.items():
   assert [x['state_index'] for x in values]==fold['train']
   rebuilt=calibrate(values,1.05);assert c.normal(rebuilt)==prices['prices'][variant]
   for ci,case in enumerate(values):
    si=case['state_index'];identity=p['identities'][si]
    assert case['identity']==identity and identity[1]!=fold['block'] and identity[2]!=fold['workload']
    rr=rows[si];kernel=CacheWeightKernel(rr,prices['powers'],variant,bundle['model']);counts=[allocation_counts(rr,cc['assignment'],5) for cc in case['candidates']];matrices=kernel.probabilities(counts)
    assert case['candidates'][0]['action']==-1
    for cc,q in zip(case['candidates'],counts):
     matrix=matrices[tuple(q)];targets=[5 if row['options'][action]['target']=='mbs' else int(row['options'][action]['target']) for row,action in zip(rr,cc['assignment'])]
     score=sum(matrix[i,j] for i,j in enumerate(targets));energy=assess(rr,tuple(cc['assignment']))['energy']
     assert math.isclose(score,cc['score'],rel_tol=0,abs_tol=1e-10) and math.isclose(energy,cc['energy'],rel_tol=1e-12,abs_tol=1e-9)
     checked+=1
     if fi==0 and ci<8 and cc['action']!=-1:
      solved=solve(rr,q,10.,'probability',probability_matrix=matrix)
      assert list(solved['assignment'])==cc['assignment'];optima+=1
   prices_out.setdefault(variant,[]).append(prices['prices'][variant]['price'])
 report=dict(status='PASS',folds=len(folds),training_states_per_fold=256,held_out_states_per_fold=16,public_candidates_recomputed=checked,conditional_optima_replayed=optima,all_minimum_prices_recomputed=True,
  held_out_blocks_and_workloads_excluded=True,price_ranges={k:[min(v),max(v)] for k,v in prices_out.items()},calibration_protocol_sha256=c.sha(a.calibration_root/'protocol.json'),calibration_completion_sha256=c.sha(a.calibration_root/'completion.json'),
  audit_type='primary agent self-audit; all public candidate scores/energy and prices, original conditional solver on a fixed first-fold subset',private_labels_read_for_price_audit=False,cpu=sorted(os.sched_getaffinity(0)))
 c.write(a.validation_root/'price_calibration_audit.json',report);print(json.dumps(report),flush=True)
