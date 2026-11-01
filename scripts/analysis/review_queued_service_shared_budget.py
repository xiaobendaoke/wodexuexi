"""Audit the shared-budget pilot's source, value split, budget and result tables."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_service_shared_budget as trial
from scripts.analysis.queued_service_validation import atomic,sha,core_order,init_worker
from scripts.analysis.queued_service_business import read
from utils.queued_service_diagnostics import mc_returns,probability_metrics


def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)


def audit_episode(payload):
    task_string,job=payload;task=Path(task_string);out,_=trial.locations(task);core=trial.core
    raw=read(core.result_path(out,job))
    experience=task/'.tmp'/core.TASK/trial.REVISION/'experience'/(core.job_id(job)+'.pkl.gz')
    digest=sha(experience);assert digest==raw['experience_sha256']
    rows=pickle.loads(gzip.decompress(experience.read_bytes()));assert len(rows)==len(raw['rows'])
    returns=mc_returns([r['reward'] for r in rows]);entropy=[];tv=[];contact_vary=0;cooperative=0;progress_vary=0;valid_steps=0
    for row,value,physical in zip(rows,returns,raw['rows']):
        close(row['returns'],value);close(row['advantage'],value-row['value']);close(row['reward'],physical['success']/20)
        k=sum(row['motion_mask']);assert len(row['motion_features'])==16 and len(row['motion_features'][0])==86
        p=[p for p,legal in zip(row['motion_probs'],row['motion_mask']) if legal]
        statistics_=probability_metrics(p,[1/k]*k);entropy.append(statistics_['entropy_fraction']);tv.append(statistics_['tv'])
        if k>1:
            valid_steps+=1;features=row['motion_features'][:k]
            inter=[r[-1] for r in features]
            if any(abs(x)>1e-12 for x in inter):
                cooperative+=1;contact_vary+=max(inter)-min(inter)>1e-9
            progress_vary+=any(max(r[i] for r in features)-min(r[i] for r in features)>1e-9 for i in (6,14,22,30,38))
    close(sum(r['reward'] for r in rows)*20,raw['summary']['counts']['success'])
    return dict(job=job,experience_sha256=digest,steps=len(rows),entropy_fraction=statistics.mean(entropy),
        tv_uniform=statistics.mean(tv),valid_motion_steps=valid_steps,cooperative_contact_steps=cooperative,
        cooperative_contact_feature_varies=contact_vary,actual_transfer_progress_varies=progress_vary,
        cpu=sorted(os.sched_getaffinity(0)))


def audit_experience(task):
    core=trial.core;out,val=trial.locations(task)
    jobs=[j for m in core.METHODS for b in range(3) for r in range(1,7) for j in core.training_jobs(m,b,r)]
    cpus=core_order()
    with ProcessPoolExecutor(max_workers=len(cpus),mp_context=multiprocessing.get_context('fork'),
            initializer=init_worker,initargs=(cpus,val/'experience_audit')) as pool:
        records=list(pool.map(audit_episode,[(str(task),j) for j in jobs]))
    groups=[]
    for method in core.METHODS:
        for cp in range(6):
            xs=[d for d in records if d['job']['method']==method and d['job']['checkpoint']==cp]
            assert len(xs)==24
            groups.append(dict(method=method,rollout_checkpoint=cp,episodes=24,
                entropy_fraction=statistics.mean(d['entropy_fraction'] for d in xs),tv_uniform=statistics.mean(d['tv_uniform'] for d in xs),
                valid_motion_steps=sum(d['valid_motion_steps'] for d in xs),cooperative_contact_steps=sum(d['cooperative_contact_steps'] for d in xs),
                cooperative_contact_feature_varies=sum(d['cooperative_contact_feature_varies'] for d in xs),
                actual_transfer_progress_varies=sum(d['actual_transfer_progress_varies'] for d in xs)))
    result=dict(status='PASS_COMPLETE_MC_EXPERIENCE_AUDIT',source_contract=trial.contract()[0],episodes=288,
        groups=groups,records=records,used_cpus=sorted({c for r in records for c in r['cpu']}),
        boundary='Read-only replay of saved training experience, not extra policy training or new environment interaction')
    atomic(val/'experience_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False));return result


def review(task):
    core=trial.core;out,val=trial.locations(task);digest,files=trial.contract()
    definition=read(out/'protocol.json');data=read(out/'analysis.json');complete=read(val/'completion.json');tests=read(val/'tests.json')
    assert definition==json.loads(json.dumps(trial.protocol()))
    assert all(d['source_contract']==digest for d in (data,complete,tests))
    assert complete['status']=='PASS_SHORT_TRIAL_COMPLETED' and complete['main_jobs']==960 and not complete['changed_old_files']
    assert len(complete['used_cpus'])==256 and complete['peak_active']==256
    assert tests['status']=='PASS' and tests['tests']==65 and tests['skipped']==0
    old_out=task/'results'/core.TASK/trial.FROZEN_REVISION;old=read(old_out/'analysis.json')
    expected_jobs=[j for m in core.METHODS for b in range(3) for cp in (0,3,6) for j in core.evaluation_jobs(m,b,cp)]
    expected_jobs += [j for m in core.CONTROLS for j in core.evaluation_jobs(m)]
    expected_jobs += [j for m in core.METHODS for b in range(3) for r in range(1,7) for j in core.training_jobs(m,b,r)]
    names={core.result_path(out,j).relative_to(out).as_posix() for j in expected_jobs}
    assert len(names)==960 and names==set(data['raw_sha256'])
    assert all(re.fullmatch('[0-9a-f]{64}',h) for h in data['raw_sha256'].values())
    summary=data['summary'];assert len(summary)==84
    for row in summary:
        assert row['episodes']==8 and 0<=row['success']<=row['admitted']<=row['generated']
        close(row['dsr'],row['success']/row['generated']);close(row['admission'],row['admitted']/row['generated'])
        close(row['conditional'],row['success']/max(row['admitted'],1));close(row['dsr'],row['admission']*row['conditional'])
        previous=next(x for x in old['summary'] if all(x[k]==row[k] for k in ('profile','method','block','checkpoint')))
        assert previous['generated']==row['generated']
        comparison=next(x for x in data['old_protocol_comparison'] if all(x[k]==row[k] for k in ('profile','method','block','checkpoint')))
        close(comparison['delta_dsr'],row['dsr']-previous['dsr'])
        if row['method'] in core.CONTROLS:
            assert previous['success']==row['success'] and previous['admitted']==row['admitted']
    records=[]
    for method in core.METHODS:
        for block in range(3):
            group=[]
            for cp in range(1,7):
                d=read(out/'training'/str(block)/method/f'{cp:04}.json')
                assert d['source_contract']==digest and d['checkpoint']==cp and d['episodes']==8
                expected={core.result_path(out,j).relative_to(out).as_posix() for j in core.training_jobs(method,block,cp)}
                assert set(d['experience_files'])==expected
                assert all(math.isfinite(v) for v in d['update'].values())
                assert d['update']['allocation_parameter_change']>0 and d['update']['motion_parameter_change']>0
                assert d['update']['critic_rmse_raw_after']>=0
                group.append(d)
            assert sum(d['update']['optimizer_steps'] for d in group)==156
            assert sum(d['episodes'] for d in group)==48
            records+=group
    fit=[];referenced=set()
    for architecture in ('old_raw','normalized_time'):
        for block in range(3):
            d=read(out/'value_check'/f'{architecture}_{block}.json')
            assert d['source_contract']==digest and d['optimizer_steps']==156 and d['train_episodes']==36 and d['validation_episodes']==12
            assert d['no_policy_warmstart'] and len(d['source'])==48
            train_ids=set();validation_ids=set()
            for index,source in enumerate(d['source']):
                job=source['job'];assert job['method']=='joint' and job['block']==block
                relative=core.result_path(old_out,job).relative_to(old_out).as_posix()
                assert source['raw_sha256']==old['raw_sha256'][relative]
                assert source['split']==('train' if index%8<6 else 'validation')
                (train_ids if source['split']=='train' else validation_ids).add(job['workload']);referenced.add(relative)
            assert len(train_ids)==36 and len(validation_ids)==12 and not (train_ids & validation_ids)
            fit.append(d)
    assert len(referenced)==144
    for architecture in ('old_raw','normalized_time'):
        xs=[d for d in fit if d['architecture']==architecture]
        target=data['value_check']['summary'][architecture]
        close(target['mean_validation_ev'],statistics.mean(x['validation']['ev'] for x in xs))
        close(target['mean_validation_rmse'],statistics.mean(x['validation']['rmse'] for x in xs))
    new_fit=data['value_check']['summary']['normalized_time'];old_fit=data['value_check']['summary']['old_raw']
    passed=new_fit['mean_validation_ev']>=.5 and new_fit['mean_validation_rmse']<old_fit['mean_validation_rmse']
    assert (data['value_check']['status']=='PASS_VALUE_FIT_CHECK')==passed
    def lookup(m,b,cp):return next(r for r in summary if r['profile']=='burst' and r['method']==m and r['block']==b and r['checkpoint']==cp)
    for method in core.METHODS:
        gains=[lookup(method,b,6)['dsr']-lookup(method,b,0)['dsr'] for b in range(3)]
        for actual,expected in zip(data['gain_over_initialization'][method],gains):close(actual,expected)
        assert data['learning_gate'][method]==(all(v>0 for v in gains) and statistics.mean(gains)>=.01)
    assert data['learning_gate']==complete['learning_gate']
    for b in range(3):close(data['joint_minus_decomposed'][b],lookup('joint',b,6)['dsr']-lookup('decomposed',b,6)['dsr'])
    experience_status=None
    if (val/'experience_audit.json').exists():
        audit=read(val/'experience_audit.json');experience_status=audit['status']
        assert experience_status=='PASS_COMPLETE_MC_EXPERIENCE_AUDIT' and audit['source_contract']==digest and audit['episodes']==288
        train_jobs=[j for m in core.METHODS for b in range(3) for r in range(1,7) for j in core.training_jobs(m,b,r)]
        assert len(audit['records'])==288 and {core.job_id(d['job']) for d in audit['records']}=={core.job_id(j) for j in train_jobs}
        for d in audit['records']:
            assert re.fullmatch('[0-9a-f]{64}',d['experience_sha256']) and 200<=d['steps']<=205
            assert 0<=d['cooperative_contact_feature_varies']<=d['cooperative_contact_steps']<=d['valid_motion_steps']<=d['steps']
            assert 0<=d['actual_transfer_progress_varies']<=d['valid_motion_steps']
            assert 0<=d['entropy_fraction']<=1.00001 and 0<=d['tv_uniform']<=1.00001
        for g in audit['groups']:
            xs=[d for d in audit['records'] if d['job']['method']==g['method'] and d['job']['checkpoint']==g['rollout_checkpoint']]
            assert len(xs)==24 and g['episodes']==24
            close(g['entropy_fraction'],statistics.mean(d['entropy_fraction'] for d in xs))
            close(g['tv_uniform'],statistics.mean(d['tv_uniform'] for d in xs))
            for k in ('valid_motion_steps','cooperative_contact_steps','cooperative_contact_feature_varies','actual_transfer_progress_varies'):
                assert g[k]==sum(d[k] for d in xs)
        for m in core.METHODS:
            for b in range(3):
                for cp in range(6):
                    xs=[d for d in audit['records'] if d['job']['method']==m and d['job']['block']==b and d['job']['checkpoint']==cp]
                    assert len(xs)==8
                    assert sum(d['steps'] for d in xs)==read(out/'training'/str(b)/m/f'{cp+1:04}.json')['physical_steps']
    raw_available=sum((out/name).exists() for name in names)
    if raw_available==960:
        for job in expected_jobs:
            path=core.result_path(out,job);d=read(path)
            assert sha(path)==data['raw_sha256'][path.relative_to(out).as_posix()]
            c=d['summary']['counts'];assert d['source_contract']==digest and d['job']==job and d['audit']['status']=='PASS'
            assert c['generated']==c.get('success',0)+c.get('expired',0) and d['rows'][-1]['active']==0
            assert sum(r['success'] for r in d['rows'])==c.get('success',0)
    deliver=task/'output'/core.TASK/'deliverables';report=deliver/'queued_shared_budget_short.md'
    with (deliver/'queued_shared_budget_summary.csv').open(encoding='utf-8-sig',newline='') as h:csvrows=list(csv.DictReader(h))
    assert len(csvrows)==84
    for actual,target in zip(csvrows,summary):
        for key,value in target.items():
            if value is None:assert actual[key]==''
            elif isinstance(value,(float,int)):close(float(actual[key]),value)
            else:assert actual[key]==value
    links=[]
    for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        if '://' not in link and not link.startswith('#'):
            assert (report.parent/link.split('#')[0]).resolve().exists(),link;links.append(link)
    checker=REPO_ROOT/'scripts/maintenance/check_layout.py'
    if checker.exists():
        layout=subprocess.run([sys.executable,str(checker)],cwd=REPO_ROOT,capture_output=True,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
        log=layout.stdout.decode('utf-8');layout_exit=layout.returncode
        assert layout_exit==1 and len([s for s in log.splitlines() if 'unexpected root files' in s])==2
    else:
        layout_exit=None
        log='Not run in the isolated server source release: repository maintenance checker is not packaged. Full local workspace must run its layout check.'
    result=dict(status='PASS_LOCAL_SHARED_BUDGET_RAW_AND_METADATA_REVIEW' if raw_available==960 else 'PASS_LOCAL_SHARED_BUDGET_METADATA_REVIEW',
        client_date='2026-11-01',source_contract=digest,source_files=files,server_tests=65,training_update_records=36,
        actual_training_episodes=288,optimizer_steps_per_learner=156,evaluation_episodes=672,csv_rows=84,
        value_fit_cases=6,value_fit_steps_per_case=156,value_fit_source_episodes=144,value_fit_no_warmstart=True,
        value_fit_gate=data['value_check']['status'],learning_gate=data['learning_gate'],local_raw_episode_files=raw_available,
        server_mc_experience_audit=experience_status,mc_experience_records_checked=288 if experience_status else 0,
        local_raw_audit_complete=raw_available==960,server_raw_audit=True,logical_cpus=256,peak_active=256,
        protected_files=complete['protected_files'],changed_old_files=[],layout_exit_code=layout_exit,layout_log=log,
        report_links_checked=len(links),review_script_sha256=sha(Path(__file__)),independent_review=False,
        budget_boundary='Equal actual rollout and policy optimizer-step budgets; separate offline critic fitting and model-prediction inference compute')
    atomic(val/'local_review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','layout_log')},ensure_ascii=False));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,default=REPO_ROOT)
    p.add_argument('--experience-audit',action='store_true');args=p.parse_args();task=args.task.resolve()
    if args.experience_audit:audit_experience(task)
    else:review(task)
