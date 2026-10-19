"""Block-level statistics for the frozen fresh-framework confirmation."""
from __future__ import annotations
import math
import statistics
import sys

COUNTERS=('generated','admitted','success','potential','fleet_energy','mbs','fairness','raw_success','shadow_fee_success')


def effect(values,*,exact=False):
    if sys.platform!='linux':raise RuntimeError('Scientific statistics run on the experiment server')
    from scipy.stats import t
    v=[float(x) for x in values];n=len(v)
    if n<2 or not all(math.isfinite(x) for x in v):raise ValueError('Paired effects need finite independent-block values')
    mean=statistics.mean(v);sd=statistics.stdev(v);se=sd/math.sqrt(n);radius=float(t.ppf(.975,n-1))*se
    p=2*float(t.sf(abs(mean/se),n-1)) if se else 1. if mean==0 else 0.
    result=dict(mean=mean,ci95=[mean-radius,mean+radius],values=v,n=n,positive=sum(x>0 for x in v),sd=sd,t_statistic=mean/se if se else None,t_p_two_sided=p,constant_pair_differences=sd==0,exact_signflip_p=None)
    if exact:
        import numpy as np
        if n>24:raise ValueError('Exact sign enumeration is registered for at most 24 blocks')
        vector=np.asarray(v,dtype=float);observed=abs(math.fsum(v));tol=1e-14*max(1.,sum(abs(x) for x in v));count=0;total=1<<n;bits=np.arange(n,dtype=np.uint64)
        for start in range(0,total,4096):
            ids=np.arange(start,min(start+4096,total),dtype=np.uint64);signs=1.-2.*(((ids[:,None]>>bits[None,:])&1).astype(float));count+=int(np.count_nonzero(np.abs(signs@vector)>=observed-tol))
        result['exact_signflip_p']=count/total
    return result


def holm(pvalues):
    if not pvalues or any(not math.isfinite(v) or not 0<=v<=1 for v in pvalues.values()):raise ValueError('Invalid prespecified test family')
    result={};previous=0.;ordered=sorted(pvalues,key=lambda key:pvalues[key])
    for i,key in enumerate(ordered):previous=min(1.,max(previous,(len(ordered)-i)*pvalues[key]));result[key]=previous
    return result


def summary(protocol,episodes):
    if sys.platform!='linux':raise RuntimeError('Scientific statistics run on the experiment server')
    import numpy as np
    methods=protocol['methods'];blocks=protocol['blocks'];primary=protocol['primary'];smoke=protocol['runner_smoke'];expected={tuple(t) for t in protocol['tasks']};observed=[tuple(x['task']) for x in episodes]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:raise ValueError('Confirmation population missing, duplicated or changed')
    if len(methods)!=9 or primary!='mixture_price' or protocol['primary_comparisons']!=[['mixture_price','G'],['mixture_price','binary_price']]:raise ValueError('Prespecified framework family changed')
    paired={}
    for ep in episodes:
        key=tuple(ep['task'][1:4]);value=(ep['exogenous_hash'],ep['totals']['potential'])
        if key in paired and paired[key]!=value:raise ValueError('External streams or fixed potential denominator are unpaired')
        paired[key]=value
        if len(ep['seconds'])!=protocol['steps'] or any(not math.isfinite(x) or x<0 for x in ep['seconds']):raise ValueError('Per-call timing population changed')
        z=ep['totals']
        if not all(math.isfinite(float(z[k])) for k in COUNTERS):raise ValueError('Nonfinite episode counts or physical metrics')
        if not 0<=z['success']<=z['raw_success']<=z['admitted']<=z['generated']<=z['potential'] or z['success']!=z['shadow_fee_success']:raise ValueError('Request accounting or measured delay invariant failed')
    by_block={};steps_per_method=len(protocol['workloads'])*protocol['episodes_per_workload']*protocol['steps']
    for block in blocks:
        row={}
        for method in methods:
            selected=[ep for ep in episodes if ep['task'][0]==method and ep['task'][1]==block]
            if len(selected)!=len(protocol['workloads'])*protocol['episodes_per_workload']:raise ValueError('Block repetition count changed')
            totals={k:sum(ep['totals'][k] for ep in selected) for k in COUNTERS};times=[t for ep in selected for t in ep['seconds']]
            if len(times)!=steps_per_method or totals['generated']<=0 or totals['potential']<=0:raise ValueError('Undefined main block estimand or incomplete calls')
            quality=totals['success']/totals['admitted'] if totals['admitted'] else None
            row[method]=dict(**totals,dsr=totals['success']/totals['generated'],potential_dsr=totals['success']/totals['potential'],raw_dsr=totals['raw_success']/totals['generated'],admission_rate=totals['admitted']/totals['generated'],accepted_quality=quality,
                mbs_share=totals['mbs']/totals['admitted'] if totals['admitted'] else None,mean_fairness=totals['fairness']/len(times),mean_seconds=statistics.mean(times),p95_seconds=float(np.percentile(times,95)),max_seconds=max(times),timing_samples=len(times),episodes=len(selected))
            if quality is not None and not math.isclose(row[method]['dsr'],row[method]['admission_rate']*quality,abs_tol=1e-14,rel_tol=0):raise ValueError('Block DSR decomposition does not hold')
        by_block[str(block)]=row
    rate_metrics=('dsr','potential_dsr','raw_dsr','admission_rate','accepted_quality','mbs_share','mean_fairness');groups={}
    for method in methods:
        times=[t for ep in episodes if ep['task'][0]==method for t in ep['seconds']];rows=[b[method] for b in by_block.values()]
        groups[method]={metric:None if any(row[metric] is None for row in rows) else statistics.mean(row[metric] for row in rows) for metric in rate_metrics}
        groups[method].update(mean_seconds=statistics.mean(times),pooled_p95_seconds=float(np.percentile(times,95)),mean_block_p95_seconds=statistics.mean(row['p95_seconds'] for row in rows),max_seconds=max(times),timing_samples=len(times),pooled_counts={k:sum(row[k] for row in rows) for k in COUNTERS})
    result=dict(status='SMOKE_DATA_CHECKED_NO_INFERENCE' if smoke else 'INDEPENDENT_FRAMEWORK_CONFIRMATION_ANALYSIS',blocks=by_block,groups=groups,episodes=len(episodes),steps=sum(len(ep['seconds']) for ep in episodes),exogenous_streams_paired=True,fixed_potential_denominator_paired=True,
        dsr_decomposition_verified_per_block=True,statistical_unit='independent new upper foundation training block',fixed_development_hgb_model=True,quota_ppo_models_evaluated=0,runner_smoke=smoke,comparisons={},primary_family={},decision=dict(confirmation_gate_evaluated=False,framework_confirmation_signal=False,unique_learning_increment_proven=False,old_sf_hlm_contract_satisfied=False,goal_complete=False))
    if smoke:return result
    limits=protocol['gates']
    if len(blocks)!=20 or len(blocks)!=limits['total_blocks']:raise ValueError('All twenty independent blocks are required')
    for control in methods:
        if control==primary:continue
        comparisons={}
        for metric in rate_metrics:
            values=[row[primary][metric]-row[control][metric] for row in by_block.values()] if all(row[primary][metric] is not None and row[control][metric] is not None for row in by_block.values()) else None
            comparisons[metric]=None if values is None else effect(values,exact=control in ('G','binary_price') and metric in ('dsr','potential_dsr'))
        if any(row[control]['fleet_energy']<=0 for row in by_block.values()):raise ValueError('Energy ratio comparator denominator is zero')
        ratios=[row[primary]['fleet_energy']/row[control]['fleet_energy'] for row in by_block.values()];center=effect([v-1 for v in ratios]);comparisons['fleet_energy_ratio']=dict(mean=1+center['mean'],ci95=[1+v for v in center['ci95']],values=ratios,n=len(ratios));result['comparisons'][control]=comparisons
    family={control:result['comparisons'][control]['dsr'] for control in ('G','binary_price')};adjusted=holm({key:value['t_p_two_sided'] for key,value in family.items()})
    result['primary_family']={key:dict(**value,holm_p=adjusted[key]) for key,value in family.items()}
    mean_ok=family['G']['mean']>=limits['mean_gain_G'] and family['binary_price']['mean']>=limits['mean_gain_binary'];directions=all(value['positive']>=limits['positive_blocks'] for value in family.values());tests=all(value<limits['holm_alpha'] for value in adjusted.values());energy=result['comparisons']['G']['fleet_energy_ratio']['mean']<=limits['energy_ratio_G'];time_ok=all(row[primary]['p95_seconds']<limits['per_block_p95_seconds'] for row in by_block.values())
    result['decision'].update(confirmation_gate_evaluated=True,framework_confirmation_signal=mean_ok and directions and tests and energy and time_ok,mean_thresholds_pass=mean_ok,positive_block_thresholds_pass=directions,primary_holm_family_pass=tests,energy_mean_ratio_pass=energy,all_block_p95_pass=time_ok,
        main_positive_intervals=all(v['ci95'][0]>0 for v in family.values()),t_and_signflip_nominal_decisions_agree=all((v['t_p_two_sided']<.05)==(v['exact_signflip_p']<.05) for v in family.values()))
    return result
