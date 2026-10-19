"""Prespecified conditional-cache matching, strong controls and full costs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import sf_hlm_cache_matching as base

RULE='all_visible_miss_zero_v1'
PRIMARY='cache_regime_probability_price'
METHODS=('G','R8','R16','binary_price','mixture','mixture_price','own_proxy_price','half_proxy_price',
    'cold_prior_price','cold_own_proxy_price','cold_half_proxy_price',PRIMARY,'HGB_probability_price','L16')
NONLEARNING=METHODS[:11]
VARIANT_METHODS=dict(prior='mixture_price',own_proxy='own_proxy_price',half_proxy='half_proxy_price',
    cold_prior='cold_prior_price',cold_own='cold_own_proxy_price',cold_half='cold_half_proxy_price',learned=PRIMARY)
FILES=('paths.py','scripts/analysis/sf_hlm_cache_regime_matching.py',*base.FILES[1:],
    'tests/test_sf_hlm_cache_calibration.py','tests/test_sf_hlm_cache_context.py','tests/test_sf_hlm_cache_regime_matching.py')
ARGS=base.ARGS+('reference_launch',)
DELAYS=(0.,.010,.020,.030)


def install():
    base.METHODS=METHODS;base.NONLEARNING=NONLEARNING;base.PRIMARY=PRIMARY
    base.VARIANT_METHODS=VARIANT_METHODS;base.FILES=FILES;base.ARGS=ARGS


def reference(a):return argparse.Namespace(**{k:Path(v) for k,v in base.read(a.reference_launch)['args'].items()})


def prepare(a):
    tp=base.read(a.cache_training/'protocol.json');audit=base.read(a.cache_validation/'raw_record_audit.json')
    if tp['context_rule']!=RULE or tp['feature_mode']!='full_logit_context' or not audit['prediction_screen']:raise ValueError('Conditional model not audited/promoted')
    ra=reference(a);old=base.read(ra.results_root/'protocol.json');native_audit=base.read(ra.validation_root/'native_raw_audit.json')
    if native_audit['status']!='PASS' or native_audit['summary_sha256']!=base.sha(ra.results_root/'analysis/summary.json') or old['identities']!=tp['identities']:raise ValueError('Historical reference not an audited identical population')
    base.prepare(a);p=base.read(a.calibration_root/'protocol.json')
    paths=(a.reference_launch,ra.results_root/'protocol.json',ra.results_root/'completion.json',ra.results_root/'analysis/summary.json',ra.validation_root/'native_raw_audit.json',ra.calibration_root/'protocol.json',ra.calibration_root/'completion.json')
    for path in paths:p['input_files'][str(path)]=base.sha(path)
    p['input_files'].update(base.read(ra.calibration_root/'completion.json')['files'])
    for state in old['states']:
        td=ra.results_root/'tasks'/('cache_'+'_'.join(map(str,state['identity'])));receipt=base.read(td/'result.json');p['input_files'][str(td/'result.json')]=base.sha(td/'result.json')
        for name,digest in receipt['files'].items():
            if base.sha(td/name)!=digest:raise ValueError('Reference state record changed')
            p['input_files'][str(td/name)]=digest
    p.update(name='sf_hlm_conditional_cache_public_prices_20261018_v1',reference_calibration=str(ra.calibration_root),reference_results=str(ra.results_root),context_rule=RULE)
    base.write(a.calibration_root/'protocol.json',p)


def require_prices(a):
    p=base.read(a.calibration_root/'protocol.json');done=base.read(a.calibration_root/'completion.json');audit=base.read(a.validation_root/'price_calibration_audit.json')
    if done['status']!='COMPLETED' or done['protocol_sha256']!=base.sha(a.calibration_root/'protocol.json') or audit['status']!='PASS' or not audit['all_three_control_prices_exactly_unchanged'] or audit['calibration_completion_sha256']!=base.sha(a.calibration_root/'completion.json'):raise ValueError('Prices/legacy control parity not audited')
    base.verify(a,p,True)
    for path,digest in done['files'].items():
        if base.sha(path)!=digest:raise ValueError('Price artifact changed')
    return p,done


def screen_init(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable common-delay namespace')
    cp,done=require_prices(a)
    p=dict(name='sf_hlm_conditional_cache_common_delay_screen_20261018_v1',files={f:base.sha(a.release_root/f) for f in FILES},input_files=cp['input_files'],states=cp['states'],identities=cp['identities'],jobs=[list(k) for k in sorted({tuple(s['identity'][:2]) for s in cp['states']})],methods=list(METHODS),nonlearning=list(NONLEARNING),primary=PRIMARY,
        calibration_protocol_sha256=base.sha(a.calibration_root/'protocol.json'),calibration_completion_sha256=base.sha(a.calibration_root/'completion.json'),calibration_files=done['files'],calibration_release=str(a.calibration_release),source_feature_mode=cp['source_feature_mode'],source_context_rule=RULE,
        branch_instances=22,timing_samples_per_state=0,timing_repeats=0,delays_seconds=DELAYS,common_delay_only=True,native_software_cost_tested=False,reference_results=cp['reference_results'],independent_confirmation=False,goal_complete=False)
    base.write(a.results_root/'protocol.json',p);base.write(a.results_root/'status.json',dict(status='PREPARED',states=400,branch_instances=8800))


def screen_worker(a,p):
    base.verify(a,p);job=json.loads(a.job)
    if job not in p['jobs']:raise ValueError('Unknown common-delay job')
    r=base.resource(a)
    with r.task_lock(a.results_root/'locks'/('_'.join(map(str,job))+'.lock')):
        with r.CpuLease(a.resources,task=['conditional_cache_common',*job],kind='environment'):
            runtime=base.archive(a).boot(a,job[1]);fast=base.modules(a)
            for n,entry in enumerate(s for s in p['states'] if s['identity'][:2]==job):
                if not base.completed(a,p,entry['identity']):base.benchmark(a,p,entry,runtime,fast,measure=False,reference_root=p['reference_results'])
                base.write(a.results_root/'progress'/('_'.join(map(str,job))+'.json'),dict(completed=n+1,last_identity=entry['identity'],updated_unix=time.time()))


def screen_analyze(a,p):
    states=[];f=base.archive(a)
    for identity in p['identities']:
        if not base.completed(a,p,identity):raise ValueError('Common-delay population incomplete')
        td=base.folder(a,identity);pub=base.read(td/'public.json');labels=base.read(td/'labels.json');timing=base.read(td/'timing.json');methods={}
        if timing['records'] or timing['timing_order'] or pub['legacy_public_branches_replayed']!=18:raise ValueError('Incorrect common-delay/legacy protocol')
        for method in METHODS:
            records=[labels['outcomes'][labels['mapping'][method+':'+str(s['repeat'])]] for s in pub['selections'] if s['method']==method]
            if len(records)!=(5 if method in ('R8','R16') else 1):raise ValueError('Wrong branch repetitions')
            methods[method]=dict(success={str(int(d*1000)):statistics.mean(sum(r['latency']+d<=r['deadline'] for r in t['requests']) for t in records) for d in DELAYS},energy=statistics.mean(t['metrics']['uav_fleet_total_energy'] for t in records),mbs=statistics.mean(t['metrics']['service_offloads_mbs'] for t in records),fairness=statistics.mean(t['metrics']['fairness'] for t in records),calls=statistics.mean(s['diagnostic']['matching_calls'] for s in pub['selections'] if s['method']==method))
        states.append(dict(identity=identity,generated=labels['generated'],admitted=labels['admitted'],methods=methods))
    subsets={}
    for source in ('all','greedy','learned'):
        rows=[s for s in states if source=='all' or s['identity'][0]==source];blocks={}
        for block in sorted({s['identity'][1] for s in rows}):
            rr=[s for s in rows if s['identity'][1]==block];den=sum(s['generated'] for s in rr);admitted=sum(s['admitted'] for s in rr)
            blocks[str(block)]=dict(generated=den,admitted=admitted,methods={m:dict(rate={str(int(d*1000)):sum(s['methods'][m]['success'][str(int(d*1000))] for s in rr)/den for d in DELAYS},energy=sum(s['methods'][m]['energy'] for s in rr),mbs_share=sum(s['methods'][m]['mbs'] for s in rr)/admitted,fairness=statistics.mean(s['methods'][m]['fairness'] for s in rr),calls=statistics.mean(s['methods'][m]['calls'] for s in rr)) for m in METHODS})
        groups={m:dict(rate={str(int(d*1000)):statistics.mean(b['methods'][m]['rate'][str(int(d*1000))] for b in blocks.values()) for d in DELAYS},energy_ratio_to_G=statistics.mean(b['methods'][m]['energy']/b['methods']['G']['energy'] for b in blocks.values()),mbs_share=statistics.mean(b['methods'][m]['mbs_share'] for b in blocks.values()),fairness=statistics.mean(b['methods'][m]['fairness'] for b in blocks.values()),calls=statistics.mean(b['methods'][m]['calls'] for b in blocks.values())) for m in METHODS}
        contrasts={}
        for d in DELAYS:
            key=str(int(d*1000));best=max(NONLEARNING,key=lambda m:groups[m]['rate'][key]);contrasts[key]=dict(best_nonlearning=best,vs_best=f.paired([b['methods'][PRIMARY]['rate'][key]-b['methods'][best]['rate'][key] for b in blocks.values()]),vs_G=f.paired([b['methods'][PRIMARY]['rate'][key]-b['methods']['G']['rate'][key] for b in blocks.values()]))
        subsets[source]=dict(blocks=blocks,groups=groups,contrasts=contrasts)
    main=subsets['all'];contrasts=main['contrasts'];energy=main['groups'][PRIMARY]['energy_ratio_to_G'];go=all(contrasts[k]['vs_best']['mean']>=.001 and contrasts[k]['vs_best']['positive']>=4 for k in ('10','20')) and contrasts['20']['vs_G']['mean']>=.005 and energy<=1.05
    summary=dict(status='COMPLETED_COMMON_DELAY_DEVELOPMENT_SCREEN',states=400,branch_instances=8800,methods=list(METHODS),subsets=subsets,decision=dict(common_delay_screen_pass=go,energy_ratio_to_G=energy,native_software_cost_tested=False,closed_loop_tested=False,independent_confirmation=False,goal_complete=False),protocol_sha256=base.sha(a.results_root/'protocol.json'),common_delay_is_not_actual_per_call_software_cost=True)
    base.write(a.results_root/'analysis/summary.json',summary);base.write(a.results_root/'analysis/states.json',states);print(json.dumps(dict(contrasts=contrasts,decision=summary['decision'])),flush=True)


def require_screen(a):
    root=a.calibration_root.parent/'common_delay_v1';done=base.read(root/'completion.json');summary=base.read(root/'analysis/summary.json');audit=base.read(a.validation_root/'common_delay_raw_audit.json')
    if done['status']!='COMPLETED' or done['summary_sha256']!=base.sha(root/'analysis/summary.json') or audit['status']!='PASS' or audit['summary_sha256']!=done['summary_sha256'] or not summary['decision']['common_delay_screen_pass']:raise ValueError('Common-delay candidate not audited/promoted')


def main():
    install();parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','calibrate','fold','screen-init','screen','screen-worker','screen-analyze','init','pipeline','worker','analyze'))
    for name in ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--block',type=int);parser.add_argument('--workload',type=int);parser.add_argument('--job');parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('All calibration, matching, physics, timing and statistics run remotely')
    if a.mode=='prepare':prepare(a)
    elif a.mode=='calibrate':base.calibrate_pipeline(a)
    elif a.mode=='screen-init':screen_init(a)
    elif a.mode=='screen':base.pipeline(a,worker_mode='screen-worker',analyze_fn=screen_analyze)
    elif a.mode=='screen-worker':screen_worker(a,base.read(a.results_root/'protocol.json'))
    elif a.mode=='init':require_prices(a);require_screen(a);base.init(a)
    elif a.mode=='pipeline':require_screen(a);base.pipeline(a)
    elif a.mode=='worker':base.worker(a,base.read(a.results_root/'protocol.json'))
    else:
        with base.resource(a).CpuLease(a.resources,task=['conditional_cache',a.mode,a.block,a.workload],kind='environment'):
            if a.mode=='fold':base.calibrate_fold(a,base.read(a.calibration_root/'protocol.json'))
            elif a.mode=='screen-analyze':screen_analyze(a,base.read(a.results_root/'protocol.json'))
            else:base.analyze(a,base.read(a.results_root/'protocol.json'))


if __name__=='__main__':main()
