"""Single prescribed-zero-price intervention on frozen conditional models."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import sf_hlm_cache_regime_matching as parent
from scripts.analysis import sf_hlm_cache_rerank as replay
base=parent.base
PRIMARY='cache_regime_zero_price'
METHODS=parent.METHODS+(PRIMARY,)
NONLEARNING=parent.NONLEARNING
FILES=('paths.py','scripts/analysis/sf_hlm_cache_zero_price.py','scripts/analysis/sf_hlm_cache_rerank.py',*parent.FILES[1:],'tests/test_sf_hlm_cache_zero_price.py')
ARGS=parent.ARGS+('source_results','source_validation')


def zero_prices(prices):
    old=prices['prices']['learned'];value=float(old['price']);normalizer=float(old['normalizer'])
    if not math.isfinite(value) or value<0 or not math.isfinite(normalizer) or normalizer<=0:raise ValueError('Invalid original learned price')
    return dict(prices,prices={**prices['prices'],**{'learned':dict(price=0.,normalizer=normalizer,prescribed_zero=True)}})


def prepare(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable zero-price intervention namespace')
    source=base.read(a.source_results/'protocol.json');summary=base.read(a.source_results/'analysis/summary.json');audit=base.read(a.source_validation/'common_delay_raw_audit.json');done=base.read(a.calibration_root/'completion.json');cp=base.read(a.calibration_root/'protocol.json')
    if audit['status']!='PASS' or audit['summary_sha256']!=base.sha(a.source_results/'analysis/summary.json') or summary['decision']['common_delay_screen_pass']:raise ValueError('Expected audited failed priced source')
    if source['methods']!=list(parent.METHODS) or source['source_context_rule']!=parent.RULE:raise ValueError('Wrong parent method/guard contract')
    inputs=dict(source['input_files']);inputs.update(done['files'])
    for path in (a.source_results/'protocol.json',a.source_results/'completion.json',a.source_results/'analysis/summary.json',a.source_validation/'common_delay_raw_audit.json',a.calibration_root/'protocol.json',a.calibration_root/'completion.json'):inputs[str(path)]=base.sha(path)
    for name,digest in source['files'].items():
        if name.endswith('.py') and base.sha(a.release_root/name)!=digest:raise ValueError('Parent code changed outside the prescribed intervention')
    for entry in source['states']:
        td=a.source_results/'tasks'/('cache_'+'_'.join(map(str,entry['identity'])));receipt=base.read(td/'result.json');inputs[str(td/'result.json')]=base.sha(td/'result.json')
        for name,digest in receipt['files'].items():
            if base.sha(td/name)!=digest:raise ValueError('Parent outcome changed')
            inputs[str(td/name)]=digest
    p=dict(name='sf_hlm_conditional_cache_fixed_zero_price_20261018_v1',files={f:base.sha(a.release_root/f) for f in FILES},input_files=inputs,states=source['states'],identities=source['identities'],methods=METHODS,nonlearning=NONLEARNING,primary=PRIMARY,source_context_rule=parent.RULE,source_feature_mode='full_logit_context',prescribed_primary_price=0.,source_protocol_sha256=base.sha(a.source_results/'protocol.json'),unchanged_parent_code_verified=True,nominal_cost_ms=10.,matching_budget=10,models_retrained=False,calibrated_soft_energy_target_applies_to_primary=False,common_delay_not_actual_software_cost=True,independent_confirmation=False,goal_complete=False)
    base.write(a.results_root/'protocol.json',p);base.write(a.results_root/'status.json',dict(status='PREPARED',states=400))


def run(a,p):
    base.verify(a,p,True);runtime=base.archive(a).boot(a,620001);fast=base.modules(a)
    from utils.sf_hlm_cache_matching import CacheMatchingPolicy
    from utils.sf_hlm_cache_calibration import validate_bundle
    from utils.sf_hlm_action_ranker import public_payload
    from utils.sf_hlm_energy_price import choose
    from utils.snapshot_evaluator import evaluate
    models={};parent_choices_changed=0;fresh_physics=0
    with fast.FastRuntime() as rt:
        for si,entry in enumerate(p['states']):
            identity=entry['identity'];fold_name=base.fold_name(identity[1],identity[2]);model_path=a.cache_training/'folds'/fold_name/'model.pkl';price_path=a.calibration_root/'folds'/fold_name/'prices.json';price=base.read(price_path)
            if base.sha(model_path)!=p['input_files'][str(model_path)] or base.sha(price_path)!=p['input_files'][str(price_path)] or si not in price['fold']['test']:raise ValueError('Frozen test-fold provenance differs')
            if fold_name not in models:
                bundle=pickle.loads(model_path.read_bytes());validate_bundle(bundle,supported_context_rules=('none',parent.RULE))
                if bundle['fold']!=price['fold'] or bundle['context_rule']!=parent.RULE:raise ValueError('Wrong frozen conditional model')
                models[fold_name]=bundle['model']
            rows=public_payload(base.read(entry['public_rows']))['rows'];rt.clear();fixed=zero_prices(price);policy=CacheMatchingPolicy(rt,None,models[fold_name],fixed,price['powers'])
            assignment,diagnostic,candidates=policy.run(rows,num_uavs=5,kind=parent.PRIMARY,trace=True)
            scores=[dict(action=c['action'],score=c['expected_success'],energy=c['predicted']['energy']) for c in candidates];parameters=price['prices']['learned'];old_index=choose(scores,parameters['price'],parameters['normalizer']);zero_index=choose(scores,0.,parameters['normalizer'])
            source_dir=a.source_results/'tasks'/('cache_'+'_'.join(map(str,identity)));old_pub=base.read(source_dir/'public.json');old_ref=next(s for s in old_pub['selections'] if s['method']==parent.PRIMARY)
            if list(candidates[old_index]['assignment'])!=old_ref['assignment'] or candidates[old_index]['action']!=old_ref['diagnostic']['selected_action'] or list(assignment)!=list(candidates[zero_index]['assignment']):raise ValueError('Same-pool original/zero choice does not reproduce')
            if diagnostic['matching_calls']>10 or diagnostic['chosen_predicted']['energy']>diagnostic['common_energy_cap']+1e-7:raise ValueError('Hard guard/budget violation')
            changed=list(assignment)!=old_ref['assignment'];parent_choices_changed+=int(changed);td=a.results_root/'tasks'/('rerank_'+'_'.join(map(str,identity)))
            public=dict(identity=identity,assignment=list(assignment),diagnostic=diagnostic,candidates=candidates,prescribed_price=0.,parent_price=parameters,original_selection_reproduced=True,parent_assignment=old_ref['assignment'],parent_selected_index=old_index,zero_selected_index=zero_index,parent_public_sha256=base.sha(source_dir/'public.json'),cache_model_sha256=base.sha(model_path),price_sha256=base.sha(price_path),sealed_before_private_scoring=True)
            base.write(td/'public.json',public)
            old_labels=base.read(source_dir/'labels.json');truth=next((x for x in old_labels['outcomes'].values() if x['assignment']==list(assignment)),None);reused=truth is not None
            if truth is None:
                meta=base.read(entry['state']);snapshot=pickle.loads(Path(meta['snapshot']).read_bytes());actual=evaluate(snapshot,assignment);truth=dict(assignment=list(assignment),requests=actual['requests'],metrics=actual['metrics']);fresh_physics+=1
            if truth['metrics']['generated_service_count']!=old_labels['generated'] or truth['metrics']['admitted_service_count']!=old_labels['admitted']:raise ValueError('Fixed state denominators differ')
            runtime[2].base.assert_step(truth['metrics'],dict(request_ids=old_pub['request_ids'],assignment=assignment,diagnostic=diagnostic))
            base.write(td/'labels.json',dict(identity=identity,truth=truth,physical_reused_after_exact_assignment_match=reused,public_sha256=base.sha(td/'public.json')))
            base.write(td/'result.json',dict(status='PASS',identity=identity,protocol_sha256=base.sha(a.results_root/'protocol.json'),files={n:base.sha(td/n) for n in ('public.json','labels.json')}))
            base.write(a.results_root/'status.json',dict(status='SCORING',states=si+1,total=400))
    replay.METHODS=METHODS;replay.PRIMARY=PRIMARY;base.NONLEARNING=NONLEARNING
    summary,states=replay.analyze(a,p,write_result=False);summary.update(parent_protocol_sha256=p['source_protocol_sha256'],prescribed_primary_price=0.,parent_choices_changed=parent_choices_changed,fresh_physical_outcomes=fresh_physics,original_priced_choices_reproduced=400,new_policy_calls=400,frozen_reference_branch_instances=8800)
    base.write(a.results_root/'analysis/summary.json',summary);base.write(a.results_root/'analysis/states.json',states)
    base.write(a.results_root/'completion.json',dict(status='COMPLETED',states=400,protocol_sha256=base.sha(a.results_root/'protocol.json'),summary_sha256=base.sha(a.results_root/'analysis/summary.json')));base.write(a.results_root/'status.json',dict(status='COMPLETED',states=400))
    print(json.dumps(dict(contrasts=summary['contrasts'],decision=summary['decision'],parent_choices_changed=parent_choices_changed)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    for name in ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('All matching, physics and statistics run remotely')
    if a.mode=='prepare':prepare(a)
    else:
        if (a.results_root/'completion.json').exists():raise FileExistsError('Experiment already completed')
        with base.resource(a).task_lock(a.results_root/'controller.lock'):
            try:
                with base.resource(a).CpuLease(a.resources,task=['cache_zero_price_intervention'],kind='environment'):run(a,base.read(a.results_root/'protocol.json'))
            except BaseException as error:base.write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


if __name__=='__main__':main()
