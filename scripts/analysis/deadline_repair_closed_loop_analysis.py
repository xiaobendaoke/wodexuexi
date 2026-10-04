"""Audit all online evaluation records before exploratory seed-paired analysis.

No simulation or training. Required CLI paths identify frozen external results.
"""
from pathlib import Path
import argparse, collections, csv, hashlib, itertools, json, math, statistics, sys


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(v):return statistics.mean(v)
def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-7)
def paired(v):
    assert len(v)==5
    m=mean(v);se=statistics.stdev(v)/math.sqrt(5);t=m/se if se else None
    if t is None:p=1. if m==0 else 0.
    else:
        x=abs(t)/math.sqrt(t*t+4);p=max(0.,min(1.,1-1.5*x+.5*x**3))
    flips=[abs(mean([s*x for s,x in zip(signs,v)])) for signs in itertools.product((-1,1),repeat=5)]
    return {'seed_differences':v,'mean':m,'ci95':[m-2.7764451051977987*se,m+2.7764451051977987*se],
            't':t,'df':4,'p_two_sided':p,'exact_sign_flip_p':sum(x>=abs(m)-1e-12 for x in flips)/32,
            'positive_seeds':sum(x>0 for x in v),'negative_seeds':sum(x<0 for x in v)}
def holm(items):
    running=0.
    for i,item in enumerate(sorted(items,key=lambda x:x['p_two_sided'])):
        running=max(running,min(1.,(len(items)-i)*item['p_two_sided']));item['holm_p']=running

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=args.input;protocol=json.loads((r/'protocol.json').read_text())
    status=json.loads((r/'status.json').read_text());assert status['phase']=='COMPLETE' and not status['failures']
    assert not protocol['smoke'] and protocol['episodes']==2 and protocol['steps']==1000
    expected={tuple(j) for j in protocol['jobs']};assert len(expected)==150
    manifests={'protocol.json':sha(r/'protocol.json'),'status.json':sha(r/'status.json')}
    results={};episodes=[];stream={};checkpoints={};times=collections.defaultdict(list)
    primary='lower_mlp_upper_uncovered_obs';vanilla='vanilla_lower_mlp_upper_uncovered_obs'
    seeds=range(620001,620006);modes=['baseline','ordinary','deadline']
    for path in sorted(r.glob('*/*/*/result.json')):
        d=json.loads(path.read_text());key=(d['method'],d['mode'],d['training_seed'],d['workload_seed'])
        assert key in expected and key not in results and d['status']=='PASS'
        assert d['protocol_sha256']==sha(r/'protocol.json') and d['source_hash']==protocol['source_hash']
        ck=(d['method'],d['training_seed']);h=d['checkpoint_contract_sha256']
        assert ck not in checkpoints or checkpoints[ck]==h;checkpoints[ck]=h
        assert len(d['episodes'])==2 and len(d['episode_files'])==2
        manifests[str(path.relative_to(r))]=sha(path);results[key]=d
        for e in d['episodes']:
            epfile=path.parent/'episodes'/f"{e['episode']:05d}.json"
            assert sha(epfile)==d['episode_files'][str(epfile.relative_to(path.parent))]
            manifests[str(epfile.relative_to(r))]=sha(epfile)
            doc=json.loads(epfile.read_text());assert doc['summary']==e
            records=doc['steps'];assert len(records)==1000 and [q['step'] for q in records]==list(range(1000))
            sums=collections.Counter();changed=0;prediction_error=0.;absolute_prediction_error=0.;up=down=0
            for q in records:
                m=q['metrics'];n=int(m['admitted_service_count'])
                assert all(math.isfinite(v) for v in m.values())
                assert not m['service_fallback_count'] and not m['forced_service_admissions']
                assert m['generated_service_count']==m['admitted_service_count']+m['unadmitted_service_count']
                assert m['admitted_service_count']==m['processed_service_count']==len(q['request_ue_ids'])
                assert len(set(q['request_ue_ids']))==n
                assert len(q['original_proposals'])==len(q['executed_actions'])==n
                assert all(type(a)is int and 0<=a<=6 for a in q['executed_actions'])
                assert m['service_offloads_mbs']==q['executed_actions'].count(1)
                assert 0<=m['deadline_satisfied_count']<=n<=m['generated_service_count']<=q['potential_service_count']
                if d['mode']=='baseline':assert q['original_proposals']==q['executed_actions']
                diag=q['repair'];delta=sum(a!=b for a,b in zip(q['original_proposals'],q['executed_actions']))
                assert delta==diag['changed_actions'];changed+=delta
                assert diag['evaluations']<=128
                assert diag['chosen_predicted_energy']<=1.05*diag['original_predicted_energy']+1e-7
                assert diag['chosen_predicted_misses']<=diag['original_predicted_misses']
                prediction_error+=n-m['deadline_satisfied_count']-diag['chosen_predicted_misses']
                absolute_prediction_error+=abs(n-m['deadline_satisfied_count']-diag['chosen_predicted_misses'])
                for field in ['network_seconds','prepare_seconds','repair_seconds','policy_seconds']:
                    assert math.isfinite(q[field]) and q[field]>=0
                assert q['policy_seconds']+1e-9>=sum(q[f] for f in ['network_seconds','prepare_seconds','repair_seconds'])
                times[(d['method'],d['mode'])].append(q['policy_seconds'])
                for field in ['generated_service_count','admitted_service_count','deadline_satisfied_count','energy','lower_assignment_energy','service_offloads_mbs','fairness']:
                    sums[field]+=m[field]
                sums['potential']+=q['potential_service_count'];up+=q['logical_upload_bytes'];down+=q['logical_download_bytes']
            checks={'generated':'generated_service_count','admitted':'admitted_service_count','success':'deadline_satisfied_count',
                    'total_uav_energy':'energy','assignment_energy':'lower_assignment_energy','mbs_count':'service_offloads_mbs','potential':'potential'}
            for k,f in checks.items():assert close(e[k],sums[f])
            assert close(e['fairness_mean'],sums['fairness']/1000)
            assert changed==e['changed_actions']
            assert close(e['dsr'],e['success']/e['generated'])
            assert close(e['dsr'],e['admission_rate']*e['conditional_success'])
            sk=(d['workload_seed'],e['episode']);fingerprint=(e['exogenous_trace_hash'],e['potential'])
            assert sk not in stream or stream[sk]==fingerprint;stream[sk]=fingerprint
            episodes.append(dict(method=d['method'],mode=d['mode'],seed=d['training_seed'],workload=d['workload_seed'],**e,
                                 actual_minus_predicted_misses_per_step=prediction_error/1000,
                                 miss_prediction_mae_per_step=absolute_prediction_error/1000,
                                 logical_upload_bytes_mean=up/1000,logical_download_bytes_mean=down/1000))
    assert set(results)==expected and len(episodes)==300
    units={};groups={}
    for method in [primary,vanilla]:
        for mode in modes:
            for seed in seeds:
                es=[e for e in episodes if (e['method'],e['mode'],e['seed'])==(method,mode,seed)];assert len(es)==10
                totals={f:sum(e[f] for e in es) for f in ['generated','admitted','success','potential','total_uav_energy','assignment_energy','mbs_count','changed_actions']}
                u={'method':method,'mode':mode,'seed':seed,**totals}
                u.update(dsr=totals['success']/totals['generated'],admission_rate=totals['admitted']/totals['generated'],
                         conditional_success=totals['success']/totals['admitted'],potential_dsr=totals['success']/totals['potential'],
                         mean_uav_energy_per_episode=totals['total_uav_energy']/10,
                         mean_assignment_energy_per_episode=totals['assignment_energy']/10,
                         mbs_share=totals['mbs_count']/totals['admitted'],fairness=mean([e['fairness_mean'] for e in es]),
                         successful_tasks_per_MJ=1e6*totals['success']/totals['total_uav_energy'])
                assert close(u['dsr'],u['admission_rate']*u['conditional_success']);units[(method,mode,seed)]=u
            us=[units[method,mode,s] for s in seeds];ts=sorted(times[method,mode]);es=[e for e in episodes if (e['method'],e['mode'])==(method,mode)]
            fields=['dsr','admission_rate','conditional_success','potential_dsr','mean_uav_energy_per_episode','mean_assignment_energy_per_episode','mbs_share','fairness','successful_tasks_per_MJ']
            groups[method+'/'+mode]={f:mean([u[f] for u in us]) for f in fields}
            groups[method+'/'+mode].update(policy_mean_seconds=mean(ts),policy_p95_seconds=ts[math.ceil(.95*len(ts))-1],policy_max_seconds=max(ts),
                changed_actions=sum(u['changed_actions'] for u in us),
                actual_minus_predicted_misses_per_step=mean([e['actual_minus_predicted_misses_per_step'] for e in es]),
                logical_upload_bytes_mean=mean([e['logical_upload_bytes_mean'] for e in es]),logical_download_bytes_mean=mean([e['logical_download_bytes_mean'] for e in es]))
    comparisons={}
    fields=['dsr','admission_rate','conditional_success','potential_dsr','mean_uav_energy_per_episode','mean_assignment_energy_per_episode','mbs_share','fairness','successful_tasks_per_MJ']
    definitions=[]
    for method in [primary,vanilla]:
        for new,old in [('ordinary','baseline'),('deadline','baseline'),('deadline','ordinary')]:definitions.append((method+'/'+new+'-'+old,(method,new),(method,old)))
    for mode in modes:definitions.append(('attention-minus-vanilla/'+mode,(primary,mode),(vanilla,mode)))
    for name,new,old in definitions:
        differences={f:paired([units[new+(s,)][f]-units[old+(s,)][f] for s in seeds]) for f in fields}
        differences['energy_relative_pct']=paired([100*(units[new+(s,)]['total_uav_energy']/units[old+(s,)]['total_uav_energy']-1) for s in seeds])
        comparisons[name]=differences
    for method in [primary,vanilla]:holm([v['dsr'] for k,v in comparisons.items() if k.startswith(method+'/')])
    holm([v['dsr'] for k,v in comparisons.items() if k.startswith('attention-minus-vanilla/')])
    output={'status':'PASS','design':'existing 5 training seeds; new 5 workloads; development screening under ideal uncharged coordination',
            'independent_n':5,'excluded_jobs':0,'excluded_episodes':0,'jobs':150,'episodes':300,'steps':300000,
            'statistics':'Each seed pools counts across ten episodes before ratios; paired t df4 intervals, exact sign-flip sensitivity. Holm on three A DSR contrasts; separate secondary families: three D contrasts and three architecture contrasts. Other endpoints descriptive/unadjusted.',
            'groups':groups,'seed_units':list(units.values()),'comparisons':comparisons,
            'audit':{'all_episode_sha256_verified':True,'all_metric_identities_verified':True,'all_modes_seeds_same_workload_external_trace':True,
                     'checkpoint_contract_consistent':True,'mask_scope':'archived runner checked quality masks; this audit verifies saved action ranges/counts, cannot reconstruct masks'},
            'provenance':{'source_hash':protocol['source_hash'],'analysis_script_sha256':sha(Path(__file__)),'python':sys.version,'raw_files':manifests},
            'limitations':['No new independent training; no confirmation claim.','Coordination latency/energy/loss not modeled.','Five parallel CPU workers, no isolated realtime guarantee.','Baseline timing includes the shared ideal-coordination adapter.','Seed-average DSR decomposition must not multiply group mean ratios.','Only original configuration; no cross-scenario superiority inference.']}
    args.output.mkdir(parents=True,exist_ok=True)
    write=args.output/'analysis.json';write.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (args.output/'seed_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(next(iter(units.values()))));writer.writeheader();writer.writerows(units.values())
    print(json.dumps({'status':'PASS','jobs':150,'episodes':300,'analysis_sha256':sha(write)}))

if __name__=='__main__':main()
