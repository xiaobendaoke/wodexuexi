"""Frozen-half assignment pools, learned selection: remote common-delay screen."""
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
from scripts.analysis import sf_hlm_cache_matching as base

PRIMARY='cache_rerank_price'
METHODS=base.METHODS+(PRIMARY,)
DELAYS=(0.,.010,.020,.030)
FILES=('scripts/analysis/sf_hlm_cache_rerank.py','tests/test_sf_hlm_cache_calibration.py')+base.FILES
ARGS=base.ARGS+('source_results','source_validation','diagnostic_results')


def prepare(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable reranking namespace')
    cp=base.read(a.calibration_root/'protocol.json');tp=base.read(a.cache_training/'protocol.json');summary=base.read(a.source_results/'analysis/summary.json');audit=base.read(a.source_validation/'native_raw_audit.json');diag=base.read(a.diagnostic_results)
    if audit['status']!='PASS' or audit['summary_sha256']!=base.sha(a.source_results/'analysis/summary.json') or summary['decision']['native_development_signal']:raise ValueError('Expected audited failed direct matching source')
    if tp['feature_mode']!='full_logit_context' or diag['source_summary_sha256']!=audit['summary_sha256'] or diag['models_retrained']:raise ValueError('Wrong frozen diagnostic/model')
    if tp['identities']!=cp['identities'] or cp['identities']!=base.read(a.source_results/'protocol.json')['identities']:raise ValueError('Source populations differ')
    inputs=dict(cp['input_files']);inputs.update(base.read(a.calibration_root/'completion.json')['files'])
    for name,digest in cp['files'].items():
        path=a.calibration_release/name
        if base.sha(path)!=digest:raise ValueError('Original price source changed')
        inputs[str(path)]=digest
    for path in (a.calibration_root/'protocol.json',a.calibration_root/'completion.json',a.source_results/'protocol.json',a.source_results/'completion.json',a.source_results/'analysis/summary.json',a.source_validation/'native_raw_audit.json',a.diagnostic_results):inputs[str(path)]=base.sha(path)
    for entry in cp['states']:
        td=a.source_results/'tasks'/('cache_'+'_'.join(map(str,entry['identity'])));receipt=base.read(td/'result.json');inputs[str(td/'result.json')]=base.sha(td/'result.json')
        for name,digest in receipt['files'].items():
            if base.sha(td/name)!=digest:raise ValueError('Audited source state changed')
            inputs[str(td/name)]=digest
    protocol=dict(name='sf_hlm_fixed_half_pool_learned_selection_20261018_v1',files={f:base.sha(a.release_root/f) for f in FILES},input_files=inputs,
        states=cp['states'],identities=cp['identities'],methods=METHODS,nonlearning=base.NONLEARNING,primary=PRIMARY,delays_seconds=DELAYS,
        source_feature_mode=tp['feature_mode'],target=1.05,nominal_cost_ms=10.,matching_budget=10,models_retrained=False,
        public_training_price_only=True,common_delay_not_native_software_cost=True,independent_confirmation=False,goal_complete=False)
    base.write(a.results_root/'protocol.json',protocol);base.write(a.results_root/'status.json',dict(status='PREPARED',states=400))
    print(json.dumps(dict(status='PREPARED',states=400,models_retrained=False)),flush=True)


def pool_projection(pool):
    return [(int(c['action']),tuple(int(x) for x in c['assignment']),float(c.get('energy',c.get('predicted',{}).get('energy')))) for c in pool]


def run(a,p):
    import numpy as np
    base.verify(a,p,True);runtime=base.archive(a).boot(a,620001);fast=base.modules(a)
    from utils.sf_hlm_cache_calibration import validate_bundle
    from utils.sf_hlm_cache_matching import CacheMatchingPolicy,CacheWeightKernel
    from utils.sf_hlm_action_ranker import public_payload
    from utils.sf_hlm_energy_price import calibrate
    from utils.sf_hlm_matching import allocation_counts
    from utils.snapshot_evaluator import evaluate
    rows=[public_payload(base.read(s['public_rows']))['rows'] for s in p['states']];folds=list(base.splits(a,p));pools={};models={};prices={};pw=base.powers();price_checks=0
    for fold in folds:
        name=base.fold_name(fold['block'],fold['workload']);old=base.read(a.calibration_root/'folds'/name/'cases.json')['half_proxy']
        if [case['state_index'] for case in old]!=fold['train']:raise ValueError('Public training pool differs from split')
        for case in old:
            si=case['state_index'];pool=case['candidates']
            if si in pools and pools[si]!=pool:raise ValueError('Half matching pool depends on unrelated fitted model')
            pools[si]=pool
    if len(pools)!=400:raise ValueError('Public half pool population incomplete')
    for fi,fold in enumerate(folds):
        name=base.fold_name(fold['block'],fold['workload']);path=a.cache_training/'folds'/name/'model.pkl';bundle=pickle.loads(path.read_bytes())
        if base.sha(path)!=p['input_files'][str(path)] or bundle['fold']!=fold or validate_bundle(bundle)!=p['source_feature_mode']:raise ValueError('Frozen cache model differs')
        models[name]=bundle['model'];cases=[]
        for si in fold['train']:
            rr=rows[si];kernel=CacheWeightKernel(rr,pw,'learned',bundle['model']);cc=pools[si];quotas=[allocation_counts(rr,c['assignment'],5) for c in cc];matrices=kernel.probabilities(quotas);candidates=[]
            for old,q in zip(cc,quotas):
                targets=[5 if row['options'][int(action)]['target']=='mbs' else int(row['options'][int(action)]['target']) for row,action in zip(rr,old['assignment'])];matrix=matrices[tuple(q)];score=float(sum(matrix[i,j] for i,j in enumerate(targets)))
                candidates.append(dict(action=old['action'],assignment=old['assignment'],score=score,energy=old['energy']));price_checks+=1
            cases.append(dict(state_index=si,identity=p['identities'][si],block=p['identities'][si][1],candidates=candidates))
        parameters=calibrate(cases,p['target']);prices[name]=parameters;td=a.results_root/'folds'/name
        base.write(td/'cases.json',cases);base.write(td/'prices.json',dict(prices=dict(rerank=parameters),fold=fold,cache_model_sha256=base.sha(path),powers=pw,protocol_sha256=base.sha(a.results_root/'protocol.json')))
        base.write(a.results_root/'status.json',dict(status='CALIBRATING',completed_folds=fi+1,total=25,public_candidates=price_checks))
    base.write(a.results_root/'price_completion.json',dict(status='COMPLETED',folds=25,public_candidates=price_checks,private_labels_used=False,models_retrained=False))
    with fast.FastRuntime() as rt:
        for si,entry in enumerate(p['states']):
            identity=entry['identity'];name=base.fold_name(identity[1],identity[2]);td=a.results_root/'tasks'/('rerank_'+'_'.join(map(str,identity)));rt.clear()
            policy=CacheMatchingPolicy(rt,None,models[name],dict(prices=dict(rerank=prices[name])),pw)
            assignment,diagnostic,candidates=policy.run(rows[si],num_uavs=5,kind=PRIMARY,trace=True)
            if pool_projection(candidates)!=pool_projection(pools[si]):raise ValueError('Live matching did not reproduce the fixed half pool')
            if diagnostic['matching_calls']>10 or diagnostic['chosen_predicted']['energy']>diagnostic['common_energy_cap']+1e-7:raise ValueError('Guard/budget violation')
            source=a.source_results/'tasks'/('cache_'+'_'.join(map(str,identity)));source_public=base.read(source/'public.json')
            public=dict(identity=identity,assignment=list(assignment),diagnostic=diagnostic,candidates=candidates,source_half_pool_exactly_equal=True,cache_model_sha256=base.sha(a.cache_training/'folds'/name/'model.pkl'),price_sha256=base.sha(a.results_root/'folds'/name/'prices.json'),sealed_before_private_scoring=True)
            base.write(td/'public.json',public)
            labels=base.read(source/'labels.json');truth=next((x for x in labels['outcomes'].values() if x['assignment']==list(assignment)),None);reused=truth is not None
            if truth is None:
                meta=base.read(entry['state']);snapshot=pickle.loads(Path(meta['snapshot']).read_bytes());actual=evaluate(snapshot,assignment);truth=dict(assignment=list(assignment),requests=actual['requests'],metrics=actual['metrics'])
            if truth['metrics']['generated_service_count']!=labels['generated'] or truth['metrics']['admitted_service_count']!=labels['admitted']:raise ValueError('New physics changed fixed denominators')
            runtime[2].base.assert_step(truth['metrics'],dict(request_ids=source_public['request_ids'],assignment=assignment,diagnostic=diagnostic))
            base.write(td/'labels.json',dict(identity=identity,truth=truth,physical_reused_after_exact_assignment_match=reused,public_sha256=base.sha(td/'public.json')))
            base.write(td/'result.json',dict(status='PASS',identity=identity,protocol_sha256=base.sha(a.results_root/'protocol.json'),files={f:base.sha(td/f) for f in ('public.json','labels.json')}))
            base.write(a.results_root/'status.json',dict(status='SCORING',states=si+1,total=400))
    analyze(a,p)
    base.write(a.results_root/'completion.json',dict(status='COMPLETED',states=400,models_retrained=False,protocol_sha256=base.sha(a.results_root/'protocol.json'),summary_sha256=base.sha(a.results_root/'analysis/summary.json')))
    base.write(a.results_root/'status.json',dict(status='COMPLETED',states=400))


def analyze(a,p,*,write_result=True):
    f=base.archive(a);states=[]
    for entry in p['states']:
        identity=entry['identity'];name='rerank_'+'_'.join(map(str,identity));td=a.results_root/'tasks'/name;rec=base.read(td/'result.json')
        if rec['status']!='PASS' or rec['protocol_sha256']!=base.sha(a.results_root/'protocol.json'):raise ValueError('Bad rerank receipt')
        for name,digest in rec['files'].items():
            if base.sha(td/name)!=digest:raise ValueError('Rerank record changed')
        source=a.source_results/'tasks'/('cache_'+'_'.join(map(str,identity)));old=base.read(source/'labels.json');pub=base.read(source/'public.json');new=base.read(td/'labels.json');methods={}
        if new['public_sha256']!=base.sha(td/'public.json'):raise ValueError('Public choice not sealed')
        for method in METHODS:
            truth=[new['truth']] if method==PRIMARY else [old['outcomes'][old['mapping'][method+':'+str(s['repeat'])]] for s in pub['selections'] if s['method']==method]
            methods[method]=dict(success={str(int(delay*1000)):statistics.mean(sum(r['latency']+delay<=r['deadline'] for r in x['requests']) for x in truth) for delay in DELAYS},energy=statistics.mean(x['metrics']['uav_fleet_total_energy'] for x in truth),mbs=statistics.mean(x['metrics']['service_offloads_mbs'] for x in truth),fairness=statistics.mean(x['metrics']['fairness'] for x in truth))
        states.append(dict(identity=identity,generated=old['generated'],admitted=old['admitted'],methods=methods))
    blocks={}
    for block in range(620001,620006):
        rr=[s for s in states if s['identity'][1]==block];den=sum(s['generated'] for s in rr);admitted=sum(s['admitted'] for s in rr);methods={}
        for method in METHODS:methods[method]=dict(rate={str(int(delay*1000)):sum(s['methods'][method]['success'][str(int(delay*1000))] for s in rr)/den for delay in DELAYS},energy=sum(s['methods'][method]['energy'] for s in rr),mbs_share=sum(s['methods'][method]['mbs'] for s in rr)/admitted,fairness=statistics.mean(s['methods'][method]['fairness'] for s in rr))
        blocks[str(block)]=dict(generated=den,admitted=admitted,methods=methods)
    groups={method:dict(rate={str(int(delay*1000)):statistics.mean(b['methods'][method]['rate'][str(int(delay*1000))] for b in blocks.values()) for delay in DELAYS},energy_ratio_to_G=statistics.mean(b['methods'][method]['energy']/b['methods']['G']['energy'] for b in blocks.values()),mbs_share=statistics.mean(b['methods'][method]['mbs_share'] for b in blocks.values()),fairness=statistics.mean(b['methods'][method]['fairness'] for b in blocks.values())) for method in METHODS}
    contrasts={}
    for delay in DELAYS:
        key=str(int(delay*1000));best=max(base.NONLEARNING,key=lambda m:groups[m]['rate'][key]);contrasts[key]=dict(best_nonlearning=best,vs_best=f.paired([b['methods'][PRIMARY]['rate'][key]-b['methods'][best]['rate'][key] for b in blocks.values()]),vs_G=f.paired([b['methods'][PRIMARY]['rate'][key]-b['methods']['G']['rate'][key] for b in blocks.values()]))
    go=all(contrasts[k]['vs_best']['mean']>=.001 and contrasts[k]['vs_best']['positive']>=4 for k in ('10','20')) and contrasts['20']['vs_G']['mean']>=.005 and groups[PRIMARY]['energy_ratio_to_G']<=1.05
    summary=dict(status='COMPLETED_COMMON_DELAY_DEVELOPMENT_SCREEN',states=400,methods=METHODS,blocks=blocks,groups=groups,contrasts=contrasts,decision=dict(common_delay_screen_pass=go,energy_ratio_to_G=groups[PRIMARY]['energy_ratio_to_G'],native_software_cost_tested=False,closed_loop_tested=False,independent_confirmation=False,goal_complete=False),protocol_sha256=base.sha(a.results_root/'protocol.json'),models_retrained=False,common_delay_is_not_actual_per_call_software_cost=True)
    if write_result:
        base.write(a.results_root/'analysis/summary.json',summary);base.write(a.results_root/'analysis/states.json',states);print(json.dumps(dict(contrasts=contrasts,decision=summary['decision'])),flush=True)
    return summary,states


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    for name in ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('All matching, calibration, physics and statistics run remotely')
    if a.mode=='prepare':prepare(a)
    else:
        if (a.results_root/'completion.json').exists():raise FileExistsError('Run already completed')
        with base.resource(a).task_lock(a.results_root/'controller.lock'):
            try:
                with base.resource(a).CpuLease(a.resources,task=['cache_rerank_screen'],kind='environment'):run(a,base.read(a.results_root/'protocol.json'))
            except BaseException as error:base.write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time()));raise


if __name__=='__main__':main()
