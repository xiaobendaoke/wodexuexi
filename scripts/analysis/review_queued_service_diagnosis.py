"""Reconcile frozen-diagnostic metadata locally, with optional complete raw audit."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import diagnose_queued_service_joint as diagnostic
from scripts.analysis import queued_service_joint_short as parent
from scripts.analysis.queued_service_business import read
from scripts.analysis.queued_service_validation import atomic,sha
from utils.queued_service_diagnostics import probability_metrics,payoff_metrics


def same(actual,expected):
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:same(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):same(a,b)
    elif isinstance(expected,float):assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10),(actual,expected)
    else:assert actual==expected,(actual,expected)


def review(task):
    out,val=diagnostic.locations(task);digest,files=diagnostic.contract()
    analysis=read(out/'analysis.json');protocol=read(out/'protocol.json');completion=read(val/'completion.json');tests=read(val/'tests.json')
    assert digest=='e42509d86a58f5c8a36aec3f31722007c57f674ee34c7bc63f78b389c765b6c6'
    assert protocol['source_files']==files and protocol['diagnostic_contract']==digest
    assert all(d['diagnostic_contract']==digest for d in (analysis,completion,tests))
    assert parent.contract()[0]==analysis['parent_contract']==completion['parent_contract']==protocol['parent_contract']
    assert tests['status']=='PASS' and tests['tests']==55 and tests['skipped']==0
    assert completion['status']=='PASS_COMPLETED_FROZEN_DIAGNOSTICS' and completion['jobs']==332
    assert len(completion['used_cpus'])==256 and completion['peak_active']==256
    assert not completion['changed_old_files'] and completion['no_training']
    smoke=read(val/'smoke.json');assert smoke['status']=='PASS_TWO_FULL_FACTUAL_REPRODUCTIONS'
    assert protocol['jobs']==diagnostic.jobs()
    expected={diagnostic.case_path(out,s).relative_to(out).as_posix() for s in diagnostic.jobs()}
    assert len(expected)==332 and expected==set(analysis['raw_sha256'])
    assert all(re.fullmatch('[0-9a-f]{64}',v) for v in analysis['raw_sha256'].values())
    old_out,_=parent.locations(task);old=read(old_out/'analysis.json')
    summary=analysis['episode_summary'];assert len(summary)==37
    keys=set()
    for r in summary:
        key=(r['method'],r['block'],r['mode']);assert key not in keys;keys.add(key)
        assert r['episodes']==8 and 0<=r['success']<=r['admitted']<=r['generated']
        same(r['dsr'],r['success']/r['generated']);same(r['admission'],r['admitted']/r['generated'])
        same(r['conditional'],r['success']/max(r['admitted'],1));same(r['dsr'],r['admission']*r['conditional'])
        assert r['success']+sum(r['expired_'+s] for s in ('waiting','upload','transfer','compute'))==r['generated']
        if r['mode'] in ('factual','queue_history'):
            prior=next(x for x in old['summary'] if x['profile']=='burst' and x['method']==r['method'] and x['block']==r['block'] and x['checkpoint']==6)
            for k in ('generated','admitted','success','dsr','admission','conditional'):same(r[k],prior[k])
    probes=analysis['probes'];assert len(probes)==192
    probe_keys={(p['method'],p['block'],p['workload'],p['time']) for p in probes};assert len(probe_keys)==192
    assert probe_keys==set((m,b,w,t) for m in parent.METHODS for b in range(3) for w in parent.EVALUATION for t in (40,75,100,150))
    for p in probes:
        k=p['candidates'];assert 1<=k<=16 and 0<=p['changed_assignments']<=p['new_tasks']
        assert p['eligible']==(p['changed_assignments']>0 and k>1)
        for cp in ('0','6'):
            dist=p['probability'][cp]
            assert all(len(v)==k for v in dist.values())
            expected_metrics=dict(local_vs_cooperative=probability_metrics(dist['local'],dist['cooperative']),
                factual_vs_no_prefix=probability_metrics(dist['factual'],dist['no_prefix']),
                factual_vs_uniform=probability_metrics(dist['factual'],[1/k]*k))
            same(p['metrics'][cp],expected_metrics)
            if p['method']=='decomposed':assert all(v==dist['no_prefix'] for v in dist.values())
        same(p['initial_vs_final'],probability_metrics(p['probability']['0']['factual'],p['probability']['6']['factual']))
        for q in p['payoff'].values():
            assert len(q['values']['local'])==len(q['values']['cooperative'])==k
            assert all(0<=x<=q['at_risk'] for values in q['values'].values() for x in values)
            same({k:q[k] for k in ('local_range','coop_range','interaction_range','strict_reversal')},payoff_metrics(q['values']['local'],q['values']['cooperative']))
        assert 0<=p['progress_saturation'].get('cap_hit',0)<=p['progress_saturation'].get('connected',0)
    credits=analysis['credit'];assert len(credits)==36;referenced=set()
    base=PurePosixPath('results')/parent.TASK/parent.REVISION
    for c in credits:
        s=c['spec'];assert c['episodes']==8 and c['parent_contract']==parent.contract()[0] and c['diagnostic_contract']==digest
        assert c['max_sample_replay_log_error']<.002
        train=read(old_out/'training'/str(s['block'])/s['method']/f"{s['round']:04}.json")
        assert c['steps']==train['physical_steps'] and len(c['source'])==8
        for item in c['source']:
            relative=PurePosixPath(item['raw']).relative_to(base).as_posix()
            assert item['raw_sha256']==old['raw_sha256'][relative]
            assert re.fullmatch('[0-9a-f]{64}',item['experience_sha256'])
            referenced.add(relative)
        if s['round']>1:
            previous=next(x for x in credits if x['spec']['method']==s['method'] and x['spec']['block']==s['block'] and x['spec']['round']==s['round']-1)
            assert previous['checkpoints']['after']==c['checkpoints']['before']
        for branch in ('allocation','motion'):
            g=c['gradient'][branch]
            assert 0<=g['episode_gradient_coherence']<=1.00001 and len(g['leave_one_episode_out_cosines'])==8
            assert g['reward_gradient_norm']>0 and g['entropy_gradient_norm']>=0
            assert -1.00001<=g['gae_vs_mc_gradient_cosine']<=1.00001
            assert all(v is None or -1.00001<=v<=1.00001 for v in g['leave_one_episode_out_cosines'])
        assert all(math.isfinite(v) for v in c['critic'].values() if v is not None)
    assert len(referenced)==288
    for method in parent.METHODS:
        for block in range(3):
            xs=[x for x in analysis['task_outcome_changes'] if x['spec']['method']==method and x['spec']['block']==block]
            assert len(xs)==8
            lost=sum(sum(x['lost_success_by_terminal_stage'].values()) for x in xs)
            gained=sum(sum(x['gained_success_from_reference_stage'].values()) for x in xs)
            factual=next(r for r in summary if r['method']==method and r['block']==block and r['mode']=='factual')
            reference=next(r for r in summary if r['mode']=='queue_history')
            assert gained-lost==factual['success']-reference['success']
    # A full local raw check is optional and reported separately from metadata.
    available=[name for name in expected if (out/name).exists()]
    if len(available)==332:
        groups=defaultdict(list);raw_probes=[];raw_credits=[]
        for spec in diagnostic.jobs():
            path=diagnostic.case_path(out,spec);name=path.relative_to(out).as_posix()
            assert sha(path)==analysis['raw_sha256'][name];d=read(path)
            assert d['spec']==spec and d['diagnostic_contract']==digest
            if spec['kind']=='episode':
                counts=d['summary']['counts'];assert counts['generated']==len(d['terminal'])==len({z[0] for z in d['terminal']})
                assert sum(z[1]=='success' for z in d['terminal'])==counts.get('success',0)
                assert sum(r['success'] for r in d['rows'])==counts.get('success',0) and d['rows'][-1]['active']==0
                assert d['audit']['status']=='PASS'
                groups[(spec['method'],spec['block'],spec['mode'])].append(d)
                raw_probes += [dict(method=spec['method'],block=spec['block'],workload=spec['workload'],**p) for p in d['probes']]
            else:raw_credits.append(d)
        same(raw_probes,probes);same(raw_credits,credits)
        for r in summary:
            ds=groups[(r['method'],r['block'],r['mode'])]
            for key in ('generated','admitted','success'):assert r[key]==sum(d['summary']['counts'].get(key,0) for d in ds)
    deliver=task/'output'/parent.TASK/'deliverables'
    with (deliver/'queued_joint_diagnosis_summary.csv').open(encoding='utf-8-sig',newline='') as h:csvrows=list(csv.DictReader(h))
    assert len(csvrows)==37
    for actual,expected_row in zip(csvrows,summary):
        for key,value in expected_row.items():
            if isinstance(value,(int,float)):same(float(actual[key]),value)
            else:same(actual[key],value)
    layout=subprocess.run([sys.executable,str(REPO_ROOT/'scripts/maintenance/check_layout.py')],cwd=REPO_ROOT,capture_output=True,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    layout_log=layout.stdout.decode('utf-8')
    assert layout.returncode==1 and len([s for s in layout_log.splitlines() if 'unexpected root files' in s])==2
    result=dict(status='PASS_LOCAL_DIAGNOSTIC_RAW_AND_METADATA_AUDIT' if len(available)==332 else 'PASS_LOCAL_DIAGNOSTIC_METADATA_AUDIT_RAW_SYNC_INCOMPLETE',
        client_review_date='2026-11-01',diagnostic_contract=digest,parent_contract=parent.contract()[0],source_files=files,
        checked_probability_and_payoff_states=192,checked_credit_batches=36,referenced_training_episodes=288,
        checked_csv_cells=37,server_completed=completion['completed'],server_audited_diagnostic_cases=332,
        local_raw_diagnostic_cases=len(available),local_raw_audit_complete=len(available)==332,
        server_tests=55,logical_cpus=256,peak_active=256,protected_files=completion['protected_files'],changed_old_files=[],
        layout_exit_code=layout.returncode,layout_log=layout_log,review_script_sha256=sha(Path(__file__)),independent_review=False,
        boundary='Server exact replay and gradient calculations are validated by frozen source and transferred records; local metadata arithmetic is not an independent simulation/gradient rerun')
    atomic(val/'local_review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','layout_log')},ensure_ascii=False));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,default=REPO_ROOT)
    review(p.parse_args().task.resolve())
