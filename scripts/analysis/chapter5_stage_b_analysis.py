"""Server-only analysis and private-label audit for Chapter 5 Stage B.

Private environments enter this offline evaluator only after public decisions
have been sealed. Fixed snapshots use common 10ms accounting; closed-loop
results always retain their original measured full-call charges.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0,str(Path(__file__).resolve().parent))
import chapter5_stage_b as run


def effect(v):
    from scipy.stats import t
    v=list(map(float,v));n=len(v);mean=statistics.mean(v)
    se=statistics.stdev(v)/math.sqrt(n) if n>1 else 0.
    radius=float(t.ppf(.975,n-1))*se if n>1 else 0.
    return dict(mean=mean,ci95=[mean-radius,mean+radius],values=v,n=n,
                interpretation='descriptive marginal interval across historical upper blocks; not simultaneous or confirmatory')


def quota(rows,assignment):
    out=[0]*6
    for row,action in zip(rows,assignment):
        target=row['options'][int(action)]['target'];out[5 if target=='mbs' else int(target)]+=1
    return out


def fixed_state(a,p,sd,fast,prices):
    import numpy as np
    from utils import sf_hlm_calibrated_matching as cm
    from utils.sf_hlm_cache_mixture import CacheMixtureKernel,decode_row
    from utils.sf_hlm_cache_matching import CacheWeightKernel
    from utils.sf_hlm_cache_context import public_context
    from utils.snapshot_evaluator import capture,evaluate
    pub=run.read(sd/'public.json'); rows=pub['rows']
    for row in rows:row['options']={int(k):v for k,v in row['options'].items()}
    decisions={};attempts={}
    # All matching/scoring is public-only. Private unpickling happens below.
    with fast.FastRuntime() as rt:
        for method in ['G',*run.COMPONENTS]:
            policy,kind=run.make_policy(rt,method,prices);calls=[];original=cm.solve
            def logged(*args,**kwargs):
                result=original(*args,**kwargs)
                calls.append(dict(quota=list(map(int,args[1])),feasible=result is not None,
                    assignment=list(result['assignment']) if result else None,
                    predicted=run.normal(result['predicted']) if result else None))
                return result
            cm.solve=logged
            try:assignment,diagnostic,pool=policy.run(rows,num_uavs=5,kind=kind,cap=10,trace=True)
            finally:cm.solve=original
            decisions[method]=dict(assignment=list(assignment),quota=quota(rows,assignment),
                diagnostic=run.normal(diagnostic),candidates=run.normal(pool))
            accepted={tuple(c['assignment']) for c in pool}
            for call in calls:
                call['accepted']=call['feasible'] and tuple(call['assignment']) in accepted
                call['reason']='infeasible' if not call['feasible'] else 'accepted' if call['accepted'] else 'energy_guard'
            attempts[method]=calls
        source='Mlambda' if pub['source']=='mixture_price' else 'G'
        assert decisions[source]['assignment']==pub['outcome']['info']['assignment'],'Observed policy replay differs'
        # Re-score a shared union of accepted legal assignments. This is a fixed
        # pool ranking diagnostic; no physical labels are used for selection.
        pool=[];seen=set()
        for decision in decisions.values():
            candidates=decision['candidates'] or [dict(assignment=decision['assignment'])]
            for c in candidates:
                key=tuple(c['assignment'])
                if key not in seen:seen.add(key);pool.append(key)
        qs=[quota(rows,x) for x in pool];kernel=CacheMixtureKernel(rows,prices['powers']);matrices=kernel.probabilities(qs)
        values=[]
        for assignment,q in zip(pool,qs):
            targets=rt.kernel(rows).targets(assignment)
            mixture=float(sum(matrices[tuple(q)][i,j] for i,j in enumerate(targets)))
            binary=cm.score_assignment(rows,assignment,10.,'binary')
            energy=rt.assess(rows,assignment)['energy']
            values.append(dict(assignment=list(assignment),quota=q,binary=binary,mixture=mixture,energy=energy))
        ranked={}
        for method in run.COMPONENTS:
            field='binary' if method.startswith('B') else 'mixture'
            price=0. if method.endswith('0') else p['common_price']['price']
            index=min(range(len(values)),key=lambda i:(-(values[i][field]-price*values[i]['energy']/p['common_price']['normalizer']),values[i]['energy'],i))
            ranked[method]=index
    # Private truth is a separate object and is never passed to policy.run.
    env,obs=pickle.loads((sd/'observed.bin').read_bytes())
    snapshot=capture(env,pub['outcome']['motion'])
    assert [r[2] for r in snapshot.requests]==[r['ue_id'] for r in rows]
    private={};physical={}
    for method,decision in decisions.items():
        assignment=tuple(decision['assignment'])
        if assignment not in physical:physical[assignment]=evaluate(snapshot,assignment)
        out=physical[assignment]
        private[method]=dict(success10=sum(r['latency']+.010<=r['deadline'] for r in out['requests']),
            raw_success=sum(r['latency']<=r['deadline'] for r in out['requests']),
            fleet_energy=out['metrics']['uav_fleet_total_energy'],
            mbs=sum(r['target']=='mbs' for r in out['requests']),requests=out['requests'])
    src=private[source];original=pub['outcome']
    assert len(src['requests'])==len(original['requests'])
    for lhs,rhs in zip(src['requests'],original['requests']):
        assert lhs['ue_id']==rhs['ue_id']
        assert math.isclose(lhs['latency'],rhs['latency'],rel_tol=1e-10,abs_tol=1e-10),'Private source physics replay differs'
    charged=sum(r['latency']+original['info']['policy_seconds']<=r['deadline'] for r in src['requests'])
    assert charged==original['metrics']['deadline_satisfied_count']
    labels=[];uavs={u.id:u for u in env.uavs};ues={u.id:u for u in env.ues}
    parts=[decode_row(row,prices['powers']) for row in rows]
    cold=public_context(rows)['global']['all_visible_miss']
    cold_prior=CacheWeightKernel(rows,prices['powers'],'cold_prior').parts
    cold_half=CacheWeightKernel(rows,prices['powers'],'cold_half').parts
    allqs={tuple(d['quota']) for d in decisions.values()}
    allqs.update(tuple(c['quota']) for calls in attempts.values() for c in calls)
    for i,(row,decoded) in enumerate(zip(rows,parts)):
        for action,part in decoded.items():
            option=row['options'][action];target=option['target']
            if part['deterministic']:continue
            hit=int(bool(uavs[int(target)].cache[ues[row['ue_id']].current_request.req_id]))
            def sensitive(q):
                n=q[int(target)]
                return bool(n>0 and part['fallback'] is None and part['hit_base']+part['work']*n+.010<=row['deadline']<part['miss_base']+part['work']*n+.010)
            gq=decisions['G']['quota'];mq=decisions['Mlambda']['quota']
            source_selected=decisions[source]['assignment'][i]==action
            inverse_residual=None
            if source_selected and part['fallback'] is None:
                qsource=decisions[source]['quota'][int(target)]
                inverse=(part['hit_base'] if hit else part['miss_base'])+part['work']*qsource
                inverse_residual=abs(inverse-src['requests'][i]['latency'])
            labels.append(dict(request_index=i,ue_id=row['ue_id'],action=action,target=int(target),truth=hit,p=part['p'],
                cold_prior=cold_prior[i][action]['p'],cold_half=cold_half[i][action]['p'],
                fallback=part['fallback'],hit_base=part['hit_base'],miss_base=part['miss_base'],work=part['work'],
                sensitive_g=sensitive(gq),active_g=gq[int(target)]>0,sensitive_m=sensitive(mq),active_m=mq[int(target)]>0,
                sensitive_union=any(sensitive(q) for q in allqs),active_union=any(q[int(target)]>0 for q in allqs),
                quota_edge_count=sum(q[int(target)]>0 for q in allqs),sensitive_quota_edge_count=sum(sensitive(q) for q in allqs),
                selected_g=decisions['G']['assignment'][i]==action,selected_m=decisions['Mlambda']['assignment'][i]==action,
                source_selected=source_selected,public_inverse_residual=inverse_residual))
    category='same_assignment' if decisions['Mlambda']['assignment']==decisions['B0']['assignment'] else 'same_quota_different_assignment' if decisions['Mlambda']['quota']==decisions['B0']['quota'] else 'different_quota'
    analysis=dict(identity=[pub['block'],pub['workload'],pub['source'],pub['step']],role=pub['sample_role'],stratum=pub['stratum'],
        weight=pub['design_weight'],generated=original['metrics']['generated_service_count'],cold=cold,
        source_replay_passed=True,category=category,cooperative_edges=len(labels),
        fallback_edges=sum(e['fallback'] is not None for e in labels),near_one_edges=sum(e['p']>=1-1e-10 for e in labels),
        selected={k:{q:v[q] for q in ('success10','raw_success','fleet_energy','mbs')} for k,v in private.items()},
        fixed_pool_ranking=ranked,fixed_pool_size=len(values),distinct_quotas=len(allqs))
    out=a.output/'fixed_states'/f"{pub['block']}_{pub['workload']}_{pub['source']}_{pub['step']:04d}"
    run.write(out/'public_decisions.json',dict(decisions=decisions,attempts=attempts,fixed_pool=values,ranking=ranked,
        public_input_sha256=run.sha(sd/'public.json')))
    run.write(out/'private_labels.json',dict(edges=labels,outcomes=private,policy_input_includes_private_labels=False))
    run.write(out/'analysis.json',analysis)
    run.write(out/'receipt.json',dict(status='PASS',files={n:run.sha(out/n) for n in ['public_decisions.json','private_labels.json','analysis.json']},
        source_observed_sha256=run.sha(sd/'observed.bin'),analysis_source_sha256=run.sha(__file__)))
    return analysis


def fixed_worker(a):
    p=run.protocol(a); b=a.block;os.sched_setaffinity(0,{a.cpu})
    f,runtime,fast,core,prices=run.boot(a,b)
    count=0
    for w in p['workloads']:
        for m in ['G','mixture_price']:
            for sd in sorted((run.taskdir(a,b,w,'center',m)/'samples').glob('*')):
                pub=run.read(sd/'public.json');out=a.output/'fixed_states'/f"{b}_{w}_{m}_{pub['step']:04d}"
                if (out/'receipt.json').exists():
                    for name,h in run.read(out/'receipt.json')['files'].items():assert run.sha(out/name)==h
                else:fixed_state(a,p,sd,fast,prices)
                count+=1
                if count%25==0:run.write(a.output/'fixed_progress'/f'{b}.json',dict(completed=count,block=b))
    run.write(a.output/'fixed_blocks'/f'{b}.json',dict(status='PASS',states=count,block=b))


def fixed(a):
    p=run.protocol(a);assert run.read(a.output/'collection_complete.json')['status']=='PASS'
    def execute(pair):
        i,b=pair;cmd=[sys.executable,str(Path(__file__).resolve()),'fixed-worker','--archive',str(a.archive),'--output',str(a.output),'--block',str(b),'--cpu',str(p['cpus'][i])]
        log=a.output/'logs'/f'fixed_{b}.log'
        with log.open('ab') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        for f in as_completed([pool.submit(execute,x) for x in enumerate(p['blocks'])]):f.result()
    run.write(a.output/'fixed_complete.json',dict(status='PASS',analysis_source_sha256=run.sha(__file__)))


def raw_audit(a,p):
    import numpy as np
    grouped=defaultdict(lambda:{'totals':Counter(),'times':[],'episodes':0,'fallback':Counter(),'edges':0})
    audit=Counter();episode_rows=[]
    for b,w,s in p['bundles']:
        exog=set();fingerprints=set()
        for m in run.methods(s):
            td=run.taskdir(a,b,w,s,m);receipt=run.read(td/'result.json')
            assert receipt['protocol_sha256']==run.sha(a.output/'protocol.json')
            assert receipt['steps_sha256']==run.sha(td/'steps.json.gz')
            with gzip.open(td/'steps.json.gz','rt') as f:records=json.load(f)
            assert len(records)==1000
            totals=Counter();times=[]
            for step,r in enumerate(records):
                assert r['step']==step
                seconds=r['info']['policy_seconds'];assert r['injected_seconds']==seconds
                assert seconds>=0 and math.isfinite(seconds)
                assert (r['window_ns'][1]-r['window_ns'][0])/1e9+1e-6>=seconds
                expected=sum(q['latency']+seconds<=q['deadline'] for q in r['requests'])
                assert expected==r['metrics']['deadline_satisfied_count']
                assert len(r['requests'])==r['metrics']['admitted_service_count']==len(r['info']['assignment'])
                assert r['metrics']['admitted_service_count']<=r['metrics']['generated_service_count']
                if m!='G':assert r['info']['diagnostic']['matching_calls']<=10
                for k,metric in run.COUNTERS.items():totals[k]+=float(r['metrics'][metric])
                times.append(seconds);audit['steps']+=1;audit['requests']+=len(r['requests'])
            for k in totals:assert math.isclose(totals[k],receipt['totals'][k],rel_tol=1e-12,abs_tol=1e-8)
            assert math.isclose(float(np.mean(times)),receipt['mean_seconds'],rel_tol=1e-12)
            group=grouped[b,s,m];group['totals'].update(totals);group['times'].extend(times);group['episodes']+=1
            for r in records:
                group['fallback'].update(r['info']['diagnostic'].get('mixture_fallback_counts',{}))
            exog.add(receipt['exogenous_hash']);fingerprints.add(receipt['upper_fingerprint'])
            episode_rows.append(dict(block=b,workload=w,scenario=s,method=m,**totals,mean_seconds=receipt['mean_seconds'],p95_seconds=receipt['p95_seconds']))
            audit['episodes']+=1
        assert len(exog)==len(fingerprints)==1
        audit['paired_bundles']+=1
    assert audit['episodes']==345 and audit['steps']==345000 and audit['paired_bundles']==75
    result={}
    for (b,s,m),g in grouped.items():
        t=g['totals'];result[b,s,m]=dict(block=b,scenario=s,method=m,**t,
            dsr=t['success']/t['generated'],admission=t['admitted']/t['generated'],
            quality=t['success']/t['admitted'] if t['admitted'] else 0.,mbs_fraction=t['mbs']/t['admitted'] if t['admitted'] else 0.,
            mean_seconds=float(np.mean(g['times'])),p95_seconds=float(np.quantile(g['times'],.95)),
            fallback_counts=dict(g['fallback']),episodes=g['episodes'])
        assert math.isclose(result[b,s,m]['dsr'],result[b,s,m]['admission']*result[b,s,m]['quality'],abs_tol=1e-14)
    run.write(a.output/'analysis/raw_audit.json',dict(status='PASS',counts=dict(audit),source_sha256=run.sha(__file__),
        coverage='all raw steps: charge, threshold, counts, clock, budget, receipt, model and exogenous pairing; sampled state physical replay reported separately'))
    for name,rows in [('episode_metrics.csv',episode_rows),('block_metrics.csv',list(result.values()))]:
        with (a.output/'analysis'/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def analyze(a):
    import numpy as np
    p=run.protocol(a);assert run.read(a.output/'collection_complete.json')['status']=='PASS'
    blocks=raw_audit(a,p);transfer={};components={}
    def compare(s,left,right):
        return dict(dsr_pp=effect([(blocks[b,s,left]['dsr']-blocks[b,s,right]['dsr'])*100 for b in p['blocks']]),
            admission_pp=effect([(blocks[b,s,left]['admission']-blocks[b,s,right]['admission'])*100 for b in p['blocks']]),
            quality_pp=effect([(blocks[b,s,left]['quality']-blocks[b,s,right]['quality'])*100 for b in p['blocks']]),
            energy_ratio=effect([blocks[b,s,left]['energy']/blocks[b,s,right]['energy'] for b in p['blocks']]),
            mbs_pp=effect([(blocks[b,s,left]['mbs_fraction']-blocks[b,s,right]['mbs_fraction'])*100 for b in p['blocks']]))
    for s in p['scenarios']:
        summary={}
        for m in run.TRANSFER:
            summary[m]={metric:effect([blocks[b,s,m][metric] for b in p['blocks']]) for metric in ['dsr','admission','quality','mbs_fraction','mean_seconds','p95_seconds']}
            summary[m]['fallback_counts']=dict(sum((Counter(blocks[b,s,m]['fallback_counts']) for b in p['blocks']),Counter()))
        transfer[s]=dict(methods=summary,comparisons={m+'-G':compare(s,m,'G') for m in run.TRANSFER if m!='G'},
            mixture_minus_binary=compare(s,'mixture_price','binary_price'),mixture_minus_cold_half=compare(s,'mixture_price','cold_half_proxy_price'))
    alias={'Mlambda':'mixture_price'}
    for left,right in [('M0','B0'),('Mlambda','Blambda'),('Blambda','B0'),('Mlambda','M0')]:
        components[left+'-'+right]=compare('center',alias.get(left,left),alias.get(right,right))
    components['interaction_pp']=effect([((blocks[b,'center','mixture_price']['dsr']-blocks[b,'center','Blambda']['dsr'])-(blocks[b,'center','M0']['dsr']-blocks[b,'center','B0']['dsr']))*100 for b in p['blocks']])
    result=dict(status='COMPLETE_CLOSED_LOOP',protocol_sha256=run.sha(a.output/'protocol.json'),independent_n=5,
        evidence='historical upper blocks, new development workloads, new 32-core regime; no confirmation claim',transfer=transfer,components=components)
    run.write(a.output/'analysis/closed_loop_summary.json',result)
    print(json.dumps({'status':result['status'],'transfer':{s:z['comparisons']['mixture_price-G']['dsr_pp'] for s,z in transfer.items()},'components':components},ensure_ascii=False),flush=True)


def calibration(edges,field):
    if not edges:return None
    pp=[float(e[field]) for e in edges]; yy=[e['truth'] for e in edges];n=len(edges)
    brier=sum((v-y)**2 for v,y in zip(pp,yy))/n
    loss=-sum(y*math.log(max(1e-12,min(1-1e-12,v)))+(1-y)*math.log1p(-max(1e-12,min(1-1e-12,v))) for v,y in zip(pp,yy))/n
    bins=[]
    for i in range(10):
        selected=[k for k,v in enumerate(pp) if min(int(v*10),9)==i]
        bins.append(dict(bin=i,n=len(selected),predicted=statistics.mean(pp[k] for k in selected) if selected else None,
            observed=statistics.mean(yy[k] for k in selected) if selected else None))
    return dict(n=n,brier=brier,log_loss=loss,hit_rate=statistics.mean(yy),mean_probability=statistics.mean(pp),bins=bins)


def mechanism(a):
    p=run.protocol(a);assert run.read(a.output/'fixed_complete.json')['status']=='PASS'
    states=[];edge_records=[]
    for folder in sorted((a.output/'fixed_states').glob('*')):
        receipt=run.read(folder/'receipt.json')
        for name,h in receipt['files'].items():assert run.sha(folder/name)==h
        z=run.read(folder/'analysis.json');labels=run.read(folder/'private_labels.json')['edges'];states.append(z)
        edge_records.extend(dict(**e,identity=z['identity'],role=z['role'],cold=z['cold']) for e in labels)
    main=[z for z in states if z['role']=='main'];cold_only=[z for z in states if z['role']=='cold_only']
    assert len(main)==1200
    identities=[tuple(z['identity']) for z in main];assert len(set(identities))==1200
    for b in p['blocks']:
        for w in p['workloads']:
            for m in ['G','mixture_price']:
                sample=[z for z in main if z['identity'][:3]==[b,w,m]]
                assert sorted(z['stratum'] for z in sample)==list(range(40))
    summaries={}; fixed_effects={};categories={}
    for b in p['blocks']:
        summaries[str(b)]={}
        for source in ['G','mixture_price','both']:
            es=[e for e in edge_records if e['role']=='main' and e['identity'][0]==b and (source=='both' or e['identity'][2]==source)]
            ss=[z for z in main if z['identity'][0]==b and (source=='both' or z['identity'][2]==source)]
            den=sum(z['generated'] for z in ss);n=len(es)
            summaries[str(b)][source]=dict(states=len(ss),edges=n,fallback=sum(e['fallback'] is not None for e in es),
                near_one=sum(e['p']>=1-1e-10 for e in es),cold_states=sum(z['cold'] for z in ss),
                sensitivity={key:dict(numerator=sum(e['sensitive_'+key] for e in es),denominator=sum(e['active_'+key] for e in es),
                    all_legal_denominator=n,ratio=sum(e['sensitive_'+key] for e in es)/max(sum(e['active_'+key] for e in es),1)) for key in ['g','m','union']},
                selected={key:dict(edges=sum(e['selected_'+key] for e in es),sensitive=sum(e['selected_'+key] and e['sensitive_'+key] for e in es)) for key in ['g','m']},
                calibration={stratum:{field:calibration([e for e in es if stratum=='all' or e['cold']==(stratum=='cold')],field) for field in ['p','cold_prior','cold_half']} for stratum in ['all','cold','warm']})
            for left,right in [('M0','B0'),('Mlambda','Blambda'),('Blambda','B0'),('Mlambda','M0')]:
                value=sum(z['selected'][left]['success10']-z['selected'][right]['success10'] for z in ss)/den*100
                fixed_effects.setdefault(source,{}).setdefault(left+'-'+right,[]).append(value)
            contributions={k:sum(z['selected']['Mlambda']['success10']-z['selected']['B0']['success10'] for z in ss if z['category']==k)/den*100 for k in ['same_assignment','same_quota_different_assignment','different_quota']}
            total=sum(z['selected']['Mlambda']['success10']-z['selected']['B0']['success10'] for z in ss)/den*100
            assert math.isclose(sum(contributions.values()),total,abs_tol=1e-12)
            categories.setdefault(source,{}).setdefault(str(b),dict(contributions=contributions,total_pp=total,counts=dict(Counter(z['category'] for z in ss))))
    aggregated={source:{k:effect(v) for k,v in comps.items()} for source,comps in fixed_effects.items()}
    for source,comparisons in fixed_effects.items():
        aggregated[source]['interaction_pp']=effect([left-right for left,right in zip(comparisons['Mlambda-Blambda'],comparisons['M0-B0'])])
    # State-balanced losses are additionally reported; edge-weighted losses above
    # retain the population-of-legal-edges interpretation and block hierarchy.
    calibration_state={}
    for field in ['p','cold_prior','cold_half']:
        values=[]
        for b in p['blocks']:
            losses=[]
            for z in main:
                if z['identity'][0]!=b:continue
                es=[e for e in edge_records if e['role']=='main' and e['identity']==z['identity']]
                if es:losses.append(calibration(es,field)['brier'])
            values.append(statistics.mean(losses))
        calibration_state[field]=effect(values)
    strata={}
    for source in ['G','mixture_price']:
        for workload in p['workloads']:
            ss=[z for z in main if z['identity'][1:3]==[workload,source]]
            es=[e for e in edge_records if e['role']=='main' and e['identity'][1:3]==[workload,source]]
            strata[f'{source}_{workload}']=dict(states=len(ss),cold_states=sum(z['cold'] for z in ss),edges=len(es),
                fallback=sum(e['fallback'] is not None for e in es),sensitive_union=sum(e['sensitive_union'] for e in es),
                active_union=sum(e['active_union'] for e in es))
    stability={}
    for name,condition in [('all',lambda e:True),('near_one',lambda e:e['p']>=1-1e-10)]:
        residuals=[e['public_inverse_residual'] for e in edge_records if e['role']=='main' and condition(e) and e['public_inverse_residual'] is not None]
        stability[name]=dict(selected_edges_checked=len(residuals),max_residual=max(residuals,default=None),
            above_1e6_absolute=sum(v>1e-6 for v in residuals),median_residual=statistics.median(residuals) if residuals else None,
            note='Approximate public inverse retained without correction; private outcomes never replace public branches')
    result=dict(status='PASS',main_states=len(main),cold_only_states=len(cold_only),physical_source_replays=len(states),independent_n=5,
        weighting='each main state represents one equal 25-step stratum; cold-only states excluded; edge ratios use summed counts within each block',
        per_block=summaries,fixed_snapshot_common_10ms=aggregated,category_contributions=categories,state_balanced_brier=calibration_state,
        fixed_pool_changes=dict(Counter('same_ranking' if z['fixed_pool_ranking']['B0']==z['fixed_pool_ranking']['M0'] else 'changed_ranking' for z in main)),
        cold_extra=dict(states=len(cold_only),edges=sum(e['role']=='cold_only' for e in edge_records)),
        source_workload_strata=strata,public_inverse_stability=stability,
        online_probabilities_and_branches_unchanged=True,policy_truth_leakage=False,
        decisions='No fitting or new learning needed; report long-episode distribution, common-price module effects and transfer boundaries')
    run.write(a.output/'analysis/mechanism_summary.json',result)
    run.write(a.output/'analysis/state_metrics.json',states)
    print(json.dumps(dict(status='MECHANISM_PASS',states=len(main),replays=len(states),fixed_effects=aggregated,state_brier=calibration_state)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['closed-loop','fixed','fixed-worker','mechanism'])
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--block',type=int);parser.add_argument('--cpu',type=int)
    a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Scientific analysis only on the experiment server')
    {'closed-loop':analyze,'fixed':fixed,'fixed-worker':fixed_worker,'mechanism':mechanism}[a.mode](a)


if __name__=='__main__':main()
