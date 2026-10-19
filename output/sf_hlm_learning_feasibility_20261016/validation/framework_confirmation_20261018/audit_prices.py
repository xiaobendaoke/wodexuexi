"""Scalar public-score audit of fixed confirmation prices, no model fitting."""
import argparse,json,math,statistics,sys
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);o=p.parse_args();a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(o.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
from scripts.analysis import sf_hlm_framework_evaluation as c
b=c.b
f,fa,fp=c.foundation(a);old=f.parents(fa)
with f.resources(fa).CpuLease(old.resources,task=['framework_fixed_price_raw_audit'],kind='environment'):
 from utils.sf_hlm_cache_mixture import decode_row
 from utils.sf_hlm_cache_calibration import eligible,controls
 from utils.sf_hlm_cache_context import public_context
 cp,prices=c.verify_prices(a);cases=b.read(a.calibration_root/'cases.json');source=b.read(fa.cold_calibration/'protocol.json');inputs={};pools={v:{} for v in cases};duplicates=0
 for fold in (fa.cold_calibration/'folds').iterdir():
  content=b.read(fold/'cases.json')
  for variant in cases:
   for case in content[variant]:
    si=case['state_index']
    if si in pools[variant]:assert pools[variant][si]==case;duplicates+=1
    else:pools[variant][si]=case
 assert duplicates==12000==cp['duplicate_control_cases_verified']
 for v in cases:assert cases[v]==[pools[v][i] for i in range(400)]
 score_checks=0;request_checks=0;zero_ratios={}
 for variant,rows0 in cases.items():
  totals={};selected=[];g_energy=[]
  for si,case in enumerate(rows0):
   entry=source['states'][si];path=Path(entry['public_rows']);assert b.sha(path)==source['input_files'][str(path)];inputs[str(path)]=b.sha(path);rows=b.read(path)['rows']
   for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
   parts=[decode_row(r,prices['powers']) for r in rows];guard=public_context(rows)['global']['all_visible_miss'];scores=[]
   assert case['state_index']==si and case['identity']==entry['identity']
   for candidate in case['candidates']:
    counts=Counter(r['options'][ac]['target'] for r,ac in zip(rows,candidate['assignment']));prob=[];energy=[]
    for ri,(row,ac) in enumerate(zip(rows,candidate['assignment'])):
     part=parts[ri][ac];opt=row['options'][ac];target=opt['target'];q=1 if target=='mbs' else counts[target];assert q>=1
     pp=part['p'] if not eligible(part) else 0. if guard else controls(row,ac)['prior' if variant=='cold_prior' else 'half_proxy']
     value=float(part['base']+part['work']*q+.010<=row['deadline']) if part['deterministic'] or part['fallback'] is not None else pp*float(part['hit_base']+part['work']*q+.010<=row['deadline'])+(1-pp)*float(part['miss_base']+part['work']*q+.010<=row['deadline'])
     prob.append(value);energy.append(float(opt['energy'])+float(opt['compute_energy'])/q**2);request_checks+=1
    value=math.fsum(prob);en=math.fsum(energy);assert math.isclose(value,candidate['score'],abs_tol=1e-10,rel_tol=0) and math.isclose(en,candidate['energy'],abs_tol=1e-10,rel_tol=1e-12);scores.append((candidate['score'],candidate['energy']));score_checks+=1
   index=min(range(len(scores)),key=lambda j:(-scores[j][0],scores[j][1],j));selected.append(index);g_energy.append(scores[0][1]);group=totals.setdefault(str(case['block']),[0.,0.]);group[0]+=scores[index][1];group[1]+=scores[0][1]
  param=prices['prices'][variant];ratios={k:v[0]/v[1] for k,v in totals.items()};assert param['price']==0. and param['target']==1.05 and param['states']==400 and param['first_feasible_crossing_index'] is None
  assert all(v<=1.05+1e-12 for v in ratios.values()) and math.isclose(param['normalizer'],sum(g_energy)/400,rel_tol=1e-12,abs_tol=1e-12) and param['selected_candidate_indices']==selected
  for block,value in ratios.items():assert math.isclose(value,param['training_energy_ratio_by_block'][block],rel_tol=1e-12,abs_tol=1e-12)
  zero_ratios[variant]=ratios
 old_args={k:Path(v) for k,v in b.read(fa.closed_launch)['args'].items()};original=b.read(old_args['fit_root']/'prices.json')
 for key in ('binary','mixture','probability'):assert original['prices'][key]==prices['prices'][key]
 for path,digest in inputs.items():assert b.sha(path)==digest
 report=dict(status='PASS',old_public_states=400,control_targets=list(cases),duplicate_cases_verified=duplicates,public_candidate_scores_recomputed=score_checks,public_candidate_requests_recomputed=request_checks,
  both_new_control_prices_zero_and_minimum_feasible=True,zero_price_public_energy_ratios=zero_ratios,primary_binary_hgb_prices_exactly_unchanged=True,private_truth_read=False,new_confirmation_data_used=False,
  protocol_sha256=b.sha(a.calibration_root/'protocol.json'),completion_sha256=b.sha(a.calibration_root/'completion.json'),price_sha256=b.sha(a.calibration_root/'prices.json'),audit_script_sha256=b.sha(__file__),audit_type='primary agent self-audit; scalar public branch success/energy, zero-price argmax and feasibility; inherited prices byte-equivalent as JSON values')
 b.write(a.validation_root/'price_audit.json',report);print(json.dumps(report),flush=True)
