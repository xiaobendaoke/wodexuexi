"""Single conditional learned reranker and label-free controls on one safe pool."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import pickle
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

PRIMARY='cache_regime_half_pool'
NEW_METHODS=(PRIMARY,'cold_prior_half_pool','cold_own_half_pool','cold_half_pool_zero')
VARIANTS=('learned','cold_prior','cold_own','cold_half')
FILES=('paths.py','scripts/analysis/sf_hlm_cache_guarded_rerank.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')


def run(a,out,diag,diag_validation,zero):
    import numpy as np
    from scripts.analysis import sf_hlm_cache_regime_matching as parent
    b=parent.base
    runtime=b.archive(a).boot(a,620001);b.modules(a)
    from utils.sf_hlm_cache_calibration import validate_bundle
    from utils.sf_hlm_cache_matching import CacheWeightKernel
    from utils.sf_hlm_energy_price import choose
    from utils.sf_hlm_matching import allocation_counts
    from utils.snapshot_evaluator import evaluate
    if (out/'protocol.json').exists():raise FileExistsError('Immutable guarded rerank namespace')
    sp=b.read(a.source_results/'protocol.json');cp=b.read(a.calibration_root/'protocol.json');audit=b.read(a.source_validation/'common_delay_raw_audit.json');da=b.read(diag_validation/'raw_record_audit.json');zp=b.read(zero/'protocol.json');zd=b.read(zero/'completion.json')
    if audit['status']!='PASS' or audit['summary_sha256']!=b.sha(a.source_results/'analysis/summary.json') or b.read(a.source_results/'analysis/summary.json')['decision']['common_delay_screen_pass']:raise ValueError('Expected audited failed source')
    if da['status']!='PASS' or da['analysis_sha256']!=b.sha(diag/'analysis.json'):raise ValueError('Diagnostic not audited')
    if zd['status']!='COMPLETED' or zd['summary_sha256']!=b.sha(zero/'analysis/summary.json') or zp['identities']!=sp['identities']:raise ValueError('Wrong direct-learning pool source')
    methods=tuple(sp['methods'])+NEW_METHODS;nonlearning=tuple(sp['nonlearning'])+NEW_METHODS[1:];own=Path(__file__).resolve().parents[2]
    inputs={**sp['input_files'],**sp['calibration_files']}
    for path in (a.source_results/'protocol.json',a.source_results/'analysis/summary.json',a.source_validation/'common_delay_raw_audit.json',diag/'protocol.json',diag/'analysis.json',diag_validation/'raw_record_audit.json',zero/'protocol.json',zero/'completion.json',zero/'analysis/summary.json'):inputs[str(path)]=b.sha(path)
    pools={};pool_duplicates=0
    for fold in (a.calibration_root/'folds').iterdir():
        path=fold/'cases.json'
        if b.sha(path)!=b.read(a.calibration_root/'completion.json')['files'][str(path)]:raise ValueError('Calibrated pool source changed')
        inputs[str(path)]=b.sha(path)
        for case in b.read(path)['cold_half']:
            si=case['state_index'];pool=case['candidates']
            if si in pools:
                if pool!=pools[si]:raise ValueError('Label-free pool depends on training fold')
                pool_duplicates+=1
            else:pools[si]=pool
    if set(pools)!=set(range(400)):raise ValueError('Frozen public pool population differs')
    for entry in sp['states']:
        suffix='_'.join(map(str,entry['identity']));td=a.source_results/'tasks'/('cache_'+suffix);zt=zero/'tasks'/('rerank_'+suffix)
        for folder,names in ((td,('result.json','public.json','labels.json')),(zt,('result.json','public.json'))):
            receipt=b.read(folder/'result.json')
            for name in names:
                if name in receipt['files'] and b.sha(folder/name)!=receipt['files'][name]:raise ValueError('Sealed source changed')
                inputs[str(folder/name)]=b.sha(folder/name)
    for path,digest in inputs.items():
        if b.sha(path)!=digest:raise ValueError('Frozen source changed')
    protocol=dict(name='sf_hlm_conditional_guarded_half_pool_rerank_20261018_v1',files={f:b.sha(own/f) for f in FILES},input_files=inputs,states=sp['states'],identities=sp['identities'],methods=methods,nonlearning=nonlearning,primary=PRIMARY,matching_budget=10,nominal_cost_ms=10.,prescribed_prices={m:0. for m in NEW_METHODS},models_retrained=False,new_matching_called_in_screen=False,online_requires_fresh_pool=True,common_delay_not_native_software_cost=True,independent_confirmation=False,goal_complete=False)
    b.write(out/'protocol.json',protocol);b.write(out/'status.json',dict(status='PREPARED',states=400))
    models={};results=[];choices=0;fresh=0;reuse=0;coverage=[];diagnostic_states={tuple(x['identity']):x for x in b.read(diag/'states.json')}
    for si,entry in enumerate(sp['states']):
        identity=entry['identity'];name=b.fold_name(identity[1],identity[2]);model_path=a.cache_training/'folds'/name/'model.pkl'
        if name not in models:
            bundle=pickle.loads(model_path.read_bytes());validate_bundle(bundle,supported_context_rules=('none',parent.RULE))
            if bundle['context_rule']!=parent.RULE or si not in bundle['fold']['test']:raise ValueError('Wrong held-out conditional model')
            models[name]=bundle['model']
        rows=b.read(entry['public_rows'])['rows']
        for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
        prices=b.read(a.calibration_root/'folds'/name/'prices.json');pool=pools[si];quotas=[allocation_counts(rows,c['assignment'],5) for c in pool];scores={}
        for variant in VARIANTS:
            kernel=CacheWeightKernel(rows,prices['powers'],variant,models[name]);matrices=kernel.probabilities(quotas);scores[variant]=[]
            for c,q in zip(pool,quotas):
                target=[5 if r['options'][int(ac)]['target']=='mbs' else r['options'][int(ac)]['target'] for r,ac in zip(rows,c['assignment'])];matrix=matrices[tuple(q)];value=float(sum(matrix[i,j] for i,j in enumerate(target)))
                if variant=='cold_half' and not math.isclose(value,c['score'],abs_tol=1e-10,rel_tol=0):raise ValueError('Frozen pool score differs')
                scores[variant].append(dict(action=c['action'],score=value,energy=c['energy']))
        source=a.source_results/'tasks'/('cache_'+'_'.join(map(str,identity)));pub=b.read(source/'public.json');ref=next(x for x in pub['selections'] if x['method']=='cold_half_proxy_price');param=prices['prices']['cold_half'];old_index=choose(scores['cold_half'],param['price'],param['normalizer'])
        if pool[old_index]['assignment']!=ref['assignment'] or pool[old_index]['action']!=ref['diagnostic']['selected_action']:raise ValueError('Frozen pool does not reproduce original control choice')
        selected={m:choose(scores[v],0.,1.) for m,v in zip(NEW_METHODS,VARIANTS)};choices+=int(pool[selected[PRIMARY]]['assignment']!=ref['assignment'])
        td=out/'tasks'/('pool_'+'_'.join(map(str,identity)));public=dict(identity=identity,pool=pool,quotas=[q.tolist() for q in quotas],scores=scores,selected_indices=selected,original_cold_half_reproduced=True,cache_model_sha256=b.sha(model_path),price_sha256=b.sha(a.calibration_root/'folds'/name/'prices.json'),sealed_before_private_scoring=True,all_new_soft_prices=0.,online_matching_budget=10)
        b.write(td/'public.json',public)
        direct=b.read(zero/'tasks'/('rerank_'+'_'.join(map(str,identity)))/'public.json');direct_ref=next(x for x in pub['selections'] if x['method']==parent.PRIMARY);direct_actions={c['action'] for c in direct['candidates']};control_action=ref['diagnostic']['selected_action'];lower=False
        if ref['assignment']!=direct_ref['assignment']:
            dr=diagnostic_states[tuple(identity)];lower=dr['cross']['learned']['cold_half_proxy_price']['priced_score']>dr['cross']['learned'][parent.PRIMARY]['priced_score']+1e-10
        coverage.append(dict(identity=identity,control_selected_action=control_action,direct_pool_actions=sorted(direct_actions),control_action_present=control_action in direct_actions,learning_own_priced_score_below_control=lower))
        old=b.read(source/'labels.json');physical=dict(old['outcomes']);new_truth={};mapping={}
        for m,index in selected.items():
            assignment=pool[index]['assignment'];found=next((key for key,value in physical.items() if value['assignment']==assignment),None)
            if found is None:
                meta=b.read(entry['state']);snapshot=pickle.loads(Path(meta['snapshot']).read_bytes());actual=evaluate(snapshot,assignment);truth=dict(assignment=assignment,requests=actual['requests'],metrics=actual['metrics']);found='new_'+str(len(new_truth));physical[found]=truth;new_truth[found]=truth;fresh+=1
            else:reuse+=1
            mapping[m]=found
            if physical[found]['metrics']['generated_service_count']!=old['generated'] or physical[found]['metrics']['admitted_service_count']!=old['admitted']:raise ValueError('Fixed denominators changed')
        b.write(td/'labels.json',dict(identity=identity,new_outcomes=new_truth,mapping=mapping,public_sha256=b.sha(td/'public.json')))
        b.write(td/'result.json',dict(status='PASS',identity=identity,protocol_sha256=b.sha(out/'protocol.json'),files={n:b.sha(td/n) for n in ('public.json','labels.json')}))
        metrics={}
        for m in methods:
            truth=[physical[mapping[m]]] if m in NEW_METHODS else [old['outcomes'][old['mapping'][m+':'+str(x['repeat'])]] for x in pub['selections'] if x['method']==m]
            metrics[m]=dict(success={str(ms):statistics.mean(sum(r['latency']+ms/1000<=r['deadline'] for r in z['requests']) for z in truth) for ms in (0,10,20,30)},energy=statistics.mean(float(z['metrics']['uav_fleet_total_energy']) for z in truth),mbs=statistics.mean(float(z['metrics']['service_offloads_mbs']) for z in truth),fairness=statistics.mean(float(z['metrics']['fairness']) for z in truth))
        results.append(dict(identity=identity,generated=old['generated'],admitted=old['admitted'],methods=metrics));b.write(out/'status.json',dict(status='SCORING',states=si+1,total=400))
    blocks={}
    for block in map(str,range(620001,620006)):
        rr=[s for s in results if str(s['identity'][1])==block];den=sum(s['generated'] for s in rr);admitted=sum(s['admitted'] for s in rr)
        blocks[block]=dict(generated=den,admitted=admitted,methods={m:dict(rate={str(ms):sum(s['methods'][m]['success'][str(ms)] for s in rr)/den for ms in (0,10,20,30)},energy=sum(s['methods'][m]['energy'] for s in rr),mbs_share=sum(s['methods'][m]['mbs'] for s in rr)/admitted,fairness=statistics.mean(s['methods'][m]['fairness'] for s in rr)) for m in methods})
    groups={m:dict(rate={str(ms):statistics.mean(x['methods'][m]['rate'][str(ms)] for x in blocks.values()) for ms in (0,10,20,30)},energy_ratio_to_G=statistics.mean(x['methods'][m]['energy']/x['methods']['G']['energy'] for x in blocks.values()),mbs_share=statistics.mean(x['methods'][m]['mbs_share'] for x in blocks.values()),fairness=statistics.mean(x['methods'][m]['fairness'] for x in blocks.values())) for m in methods}
    f=b.archive(a);contrasts={}
    for ms in (0,10,20,30):
        k=str(ms);best=max(nonlearning,key=lambda m:groups[m]['rate'][k]);contrasts[k]=dict(best_nonlearning=best,vs_best=f.paired([x['methods'][PRIMARY]['rate'][k]-x['methods'][best]['rate'][k] for x in blocks.values()]),vs_G=f.paired([x['methods'][PRIMARY]['rate'][k]-x['methods']['G']['rate'][k] for x in blocks.values()]))
    passed=all(contrasts[k]['vs_best']['mean']>=.001 and contrasts[k]['vs_best']['positive']>=4 for k in ('10','20')) and contrasts['20']['vs_G']['mean']>=.005 and groups[PRIMARY]['energy_ratio_to_G']<=1.05
    summary=dict(status='GUARDED_HALF_POOL_CONDITIONAL_RERANK_COMMON_DELAY_SCREEN',groups=groups,blocks=blocks,contrasts=contrasts,decision=dict(common_delay_screen_pass=passed,native_software_cost_tested=False,closed_loop_improvement_tested=False,independent_confirmation=False,goal_complete=False),pool_duplicate_checks=pool_duplicates,original_cold_half_choices_reproduced=400,learning_choices_differ_from_cold_half=choices,fresh_physical_outcomes=fresh,reused_exact_assignments=reuse,lower_priced_score_states=sum(x['learning_own_priced_score_below_control'] for x in coverage),lower_priced_score_control_action_absent=sum(x['learning_own_priced_score_below_control'] and not x['control_action_present'] for x in coverage))
    b.write(out/'analysis/states.json',results);b.write(out/'analysis/pool_coverage.json',coverage);b.write(out/'analysis/summary.json',summary)
    for path,digest in inputs.items():
        if b.sha(path)!=digest:raise ValueError('Frozen input changed during screen')
    b.write(out/'completion.json',dict(status='COMPLETED',states=400,protocol_sha256=b.sha(out/'protocol.json'),summary_sha256=b.sha(out/'analysis/summary.json')));b.write(out/'status.json',dict(status='COMPLETED',states=400));print(json.dumps(dict(contrasts=contrasts,decision=summary['decision'],choices_changed=choices,lower_priced_score_states=summary['lower_priced_score_states'],control_action_absent=summary['lower_priced_score_control_action_absent'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--diagnostic-validation',type=Path,required=True);p.add_argument('--zero-results',type=Path,required=True);o=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('All scoring, simulation and statistics run on the experiment server')
    a=argparse.Namespace(**{k:Path(v) for k,v in json.loads(o.launch.read_text())['args'].items()});sys.path.insert(0,str(a.release_root))
    from scripts.analysis import sf_hlm_cache_matching as b
    with b.resource(a).CpuLease(a.resources,task=['cache_guarded_rerank'],kind='environment'):run(a,o.out,o.diagnostic,o.diagnostic_validation,o.zero_results)


if __name__=='__main__':main()
