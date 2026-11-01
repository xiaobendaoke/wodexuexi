"""Full local raw-count and opportunity-value review; no Torch or simulation."""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import diagnose_queued_service_conditioning as diagnostic
from scripts.analysis.queued_service_business import read
from scripts.analysis.queued_service_validation import atomic,sha
from scripts.analysis.review_queued_service_diagnosis import same
from utils.queued_service_conditioning import conditioning_value,value_fit_by_phase
from utils.queued_service_diagnostics import probability_metrics,mc_returns


def expectation(q,p):return sum(a*b for a,b in zip(q,p))/sum(p)


def review(task):
    out,val=diagnostic.locations(task);trial=diagnostic.trial;core=trial.core
    data=read(out/'analysis.json');proto=read(out/'protocol.json');completion=read(val/'completion.json');tests=read(val/'tests.json')
    digest,files=diagnostic.contract()
    assert digest=='bbcfbe1018af9ddca789dd3eed272aefc25d0ad5f33757ce72277a0853de471c'
    assert all(d['diagnostic_contract']==digest for d in (data,proto,completion,tests)) and proto['source_files']==files
    assert data['parent_contract']==trial.contract()[0]==completion['parent_contract']
    assert completion['status']=='PASS_FROZEN_CONDITIONING_DIAGNOSTIC' and completion['jobs']==276
    assert not completion['changed_old_files'] and completion['no_training'] and len(completion['used_cpus'])==256
    assert tests['status']=='PASS' and tests['tests']==74 and tests['skipped']==0
    assert read(val/'smoke.json')['status']=='PASS_TWO_EXACT_COMPLETE_REPLAYS'
    assert proto['jobs']==diagnostic.jobs()
    parent_out,_=trial.locations(task);parent=read(parent_out/'analysis.json');parent_protocol=read(parent_out/'protocol.json')
    rows=[];groups=defaultdict(list);probes=[];credit=[];fingerprints={}
    assert set(data['raw_sha256'])=={diagnostic.case_path(out,s).relative_to(out).as_posix() for s in diagnostic.jobs()}
    for spec in diagnostic.jobs():
        path=diagnostic.case_path(out,spec);d=read(path)
        assert sha(path)==data['raw_sha256'][path.relative_to(out).as_posix()] and d['diagnostic_contract']==digest and d['spec']==spec
        assert d['parent_contract']==trial.contract()[0]
        if spec['kind']=='episode':
            c=d['summary']['counts'];assert c['generated']==c.get('success',0)+c.get('expired',0)
            assert sum(r['success'] for r in d['rows'])==c.get('success',0) and d['rows'][-1]['active']==0 and d['audit']['status']=='PASS'
            assert sum(d['summary']['expiration_stage'].values())==c.get('expired',0)
            w=spec['workload'];assert w not in fingerprints or fingerprints[w]==d['fingerprint'];fingerprints[w]=d['fingerprint']
            original=dict(kind='eval',method=spec['method'],block=spec['block'],checkpoint=6,profile='burst',workload=w)
            name=core.result_path(parent_out,original).relative_to(parent_out).as_posix()
            assert d['parent_raw_sha256']==parent['raw_sha256'][name]
            groups[(spec['method'],spec['block'],spec['mode'])].append(d);rows.append(d)
            if spec['mode']=='native':
                assert d['replay_steps_exact']==len(d['rows'])==len(d['values'])
                for item,target in zip(d['values'],mc_returns([r['success']/20 for r in d['rows']])):same(item['target'],target)
                assert len(d['probes'])==4
            else:assert not d['probes']
            probes += [dict(method=spec['method'],block=spec['block'],workload=w,**p) for p in d['probes']]
        else:
            assert d['episodes']==8 and d['max_sample_log_error']<.002 and len(d['source'])==8
            expected={core.result_path(parent_out,j).relative_to(parent_out).as_posix() for j in core.training_jobs(spec['method'],spec['block'],spec['round'])}
            assert {r['raw'] for r in d['source']}==expected
            for r in d['source']:
                assert r['raw_sha256']==parent['raw_sha256'][r['raw']] and re.fullmatch('[0-9a-f]{64}',r['experience_sha256'])
            assert d['steps']==read(parent_out/'training'/str(spec['block'])/spec['method']/f"{spec['round']:04}.json")['physical_steps']
            if spec['method']=='joint':assert d['feature_reward_prediction_max']<1e-7
            assert 0<=d['candidate_short_reward_tie_fraction']<=1 and 0<=d['motion_gradient']['episode_coherence']<=1.00001
            credit.append(d)
    assert len(rows)==240 and len(credit)==36 and len(probes)==192
    same(credit,data['credit']);same(probes,data['probes'])
    for r in data['summary']:
        ds=groups[(r['method'],r['block'],r['mode'])];assert len(ds)==r['episodes']==8
        for key in ('generated','admitted','success'):assert r[key]==sum(d['summary']['counts'].get(key,0) for d in ds)
        same(r['dsr'],r['success']/r['generated']);same(r['admission'],r['admitted']/r['generated']);same(r['conditional'],r['success']/max(r['admitted'],1))
        if r['mode']=='native':
            old=next(x for x in parent['summary'] if x['profile']=='burst' and x['method']==r['method'] and x['block']==r['block'] and x['checkpoint']==6)
            for key in ('generated','admitted','success','dsr'):same(r[key],old[key])
    derived=[];link_distances=[]
    range_=parent_protocol['physics']['inter_range'];speed=parent_protocol['physics']['speed'];max_deadline=max(parent_protocol['deadline'])
    for p in probes:
        k=p['candidates'];assert 1<=k<=16
        assert len(p['pre_state'])==len(p['post_state'])==80
        same(p['pre_new_count'],p['pre_state'][74]*40);same(p['post_new_count'],p['post_state'][74]*40)
        assert p['post_new_count']==0 and abs(p['pre_new_count']-p['new_tasks'])<1e-8
        plans=p['plans'];assert all(route.keys()==plans['actual'].keys() for route in plans.values())
        distinct=len({tuple(sorted(plans[name].items())) for name in ('actual','sample1','sample2')})
        assert distinct==p['sampled_plan_distinct'] and p['eligible']==(p['new_tasks']>0 and k>1 and distinct>1)
        for cp,dist in p['probabilities'].items():
            assert all(len(v)==k for v in dist.values())
            for v in dist.values():probability_metrics(v,[1/k]*k)
        probability=p['probabilities']['6'];native='actual' if p['method']=='joint' else 'pre'
        same(p['pre_vs_post'],probability_metrics(probability['pre'],probability['actual']))
        same(p['native_vs_uniform'],probability_metrics(probability[native],[1/k]*k))
        same(p['sampled_plan_tv'],statistics.mean(probability_metrics(probability['actual'],probability[n])['tv'] for n in ('sample1','sample2')))
        same(p['initial_final_tv'],probability_metrics(p['probabilities']['0'][native],probability[native]))
        for scale,matrix in p['payoff'].items():
            values=matrix['values'];assert set(values)==set(plans) and all(len(v)==k for v in values.values())
            assert all(0<=v<=matrix['at_risk'] for vector in values.values() for v in vector)
            same(matrix['on_policy'],conditioning_value([values[n] for n in ('actual','sample1','sample2')]))
            same(matrix['extreme'],conditioning_value([values[n] for n in ('local','cooperative')]))
            for label in ('on_policy','extreme'):
                q=matrix[label];same(q['total_variance'],q['routing_variance']+q['motion_variance']+q['interaction_variance'])
            q=values['actual'];prob=probability[native]
            conditional=statistics.mean(expectation(values[n],probability[n]) for n in ('actual','sample1','sample2'))
            common=statistics.mean(expectation(values[n],probability['pre']) for n in ('actual','sample1','sample2'))
            derived.append(dict(method=p['method'],block=p['block'],workload=p['workload'],time=p['time'],scale=scale,
                opportunity=matrix['on_policy']['information_gain'],policy_expected_regret=max(q)-expectation(q,prob),
                uniform_expected_regret=max(q)-statistics.mean(q),greedy_regret=max(q)-q[max(range(k),key=prob.__getitem__)],
                frozen_conditional_response_tasks=conditional-common,
                boundary='Finite-horizon fixed-continuation payoffs. Conditional response uses frozen counter-inputs, not a retrained policy comparison'))
        xy=[(p['pre_state'][i*12]*700,p['pre_state'][i*12+1]*700) for i in range(5)]
        for uid,target in plans['actual'].items():
            source=plans['local'][uid]
            if target not in (source,5):
                distance=math.hypot(xy[source][0]-xy[target][0],xy[source][1]-xy[target][1])
                assert distance<=range_+1e-6;link_distances.append(distance)
    for r in data['value']:
        ds=[d for d in rows if d['spec']['method']==r['method'] and d['spec']['mode']=='native']
        target=[x['target'] for d in ds for x in d['values']];prediction=[x['prediction'] for d in ds for x in d['values']]
        keys=[(d['spec']['block'],x['time']) for d in ds for x in d['values']]
        same({k:v for k,v in r.items() if k!='method'},value_fit_by_phase(target,prediction,keys))
    deltas=parent['joint_minus_decomposed'];mean=statistics.mean(deltas);se=statistics.stdev(deltas)/math.sqrt(3)
    t975=math.sqrt(2)*.95/math.sqrt(1-.95**2)
    extras=dict(diagnostic_contract=digest,paired_seed_difference=dict(n=3,mean=mean,by_seed=deltas,
        exploratory_normal_seed_t95=[mean-t975*se,mean+t975*se],not_equivalence_test=True),
        geometry=dict(new_cooperative_assignment_references=len(link_distances),mean_distance=statistics.mean(link_distances) if link_distances else None,
            within_one_slot_contact_guarantee=sum(d+2*speed<=range_+1e-8 for d in link_distances),
            within_max_deadline_contact_guarantee=sum(d+2*speed*max_deadline<=range_+1e-8 for d in link_distances),
            range_m=range_,speed_mps=speed,deadline_max_s=max_deadline,
            boundary='Triangle-inequality contact guarantee for referenced initial links at maximum speed; not a completion guarantee. Repeated states/tasks are not independent samples.'),
        payoff_policy_diagnostics=derived)
    atomic(out/'interpretation.json',extras)
    deliver=task/'output'/core.TASK/'deliverables'
    with (deliver/'queued_conditioning_summary.csv').open(encoding='utf-8-sig',newline='') as h:csvrows=list(csv.DictReader(h))
    assert len(csvrows)==30
    for actual,target in zip(csvrows,data['summary']):
        for key,value in target.items():
            if isinstance(value,(float,int)):same(float(actual[key]),value)
            else:same(actual[key],value)
    report=deliver/'queued_conditioning_diagnosis.md';links=[]
    for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        if '://' not in link and not link.startswith('#'):
            assert (report.parent/link.split('#')[0]).resolve().exists(),link;links.append(link)
    layout=subprocess.run([sys.executable,str(REPO_ROOT/'scripts/maintenance/check_layout.py')],cwd=REPO_ROOT,capture_output=True,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    log=layout.stdout.decode('utf-8');assert layout.returncode==1 and len([s for s in log.splitlines() if 'unexpected root files' in s])==2
    result=dict(status='PASS_COMPLETE_LOCAL_CONDITIONING_RAW_AND_METADATA_REVIEW',client_date='2026-11-01',diagnostic_contract=digest,
        source_files=files,parent_contract=trial.contract()[0],raw_files=276,native_trajectories=48,closed_loop_episodes=240,
        sampled_plan_states=192,credit_batches=36,credit_experience_references=288,server_tests=74,local_raw_complete=True,
        logical_cpus_used=len(completion['used_cpus']),peak_active_recorded=completion['peak_active'],protected_files=completion['protected_files'],
        changed_old_files=[],no_training=True,layout_exit_code=layout.returncode,layout_log=log,report_links_checked=len(links),
        review_script_sha256=sha(Path(__file__)),independent_review=False,
        boundary='Local raw checks recompute counts, probability and payoff arithmetic and value residuals; frozen network inference/gradient calculations were executed on server, not independently recomputed here')
    atomic(val/'local_review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','layout_log')},ensure_ascii=False));return result,extras


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,default=REPO_ROOT)
    review(p.parse_args().task.resolve())
