"""Read-only stdlib audit of the full plan-residual pilot delivery."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_service_plan_residual as trial
from scripts.analysis.queued_service_business import read
from scripts.analysis.queued_service_validation import sha,atomic
from utils.queued_service_conditioning import value_fit_by_phase
from utils.queued_service_diagnostics import mc_returns


def same(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
    else:assert a==b,(a,b)


def review(task,metadata_only=False):
    out,val=trial.locations(task);core=trial.core;digest,files=trial.contract()
    definition=read(out/'protocol.json');data=read(out/'analysis.json');mechanism=read(out/'mechanism/analysis.json')
    complete=read(val/'completion.json');tests=read(val/'tests.json');experience=read(val/'experience_audit.json')
    same(definition,json.loads(json.dumps(trial.protocol())))
    assert all(d['source_contract']==digest for d in (data,mechanism,complete,tests,experience))
    assert complete['status']=='PASS_SHORT_TRIAL_AND_MECHANISM_COMPLETED' and complete['main_jobs']==960 and complete['mechanism_cases']==96
    assert not complete['changed_old_files'] and len(complete['used_cpus'])==256
    assert tests['status']=='PASS' and tests['tests']==81 and tests['skipped']==0
    assert experience['status']=='PASS_SEPARATE_BASELINE_MC_AND_REPLAY' and experience['episodes']==288
    parent_out,_=trial.parent.locations(task);old=read(parent_out/'analysis.json');old_diag_out,_=trial.diagnosis.locations(task);old_diag=read(old_diag_out/'analysis.json')
    jobs=[j for m in core.METHODS for b in range(3) for cp in (0,3,6) for j in core.evaluation_jobs(m,b,cp)]
    jobs += [j for m in core.CONTROLS for j in core.evaluation_jobs(m)]
    jobs += [j for m in core.METHODS for b in range(3) for r in range(1,7) for j in core.training_jobs(m,b,r)]
    assert len(jobs)==960 and set(data['raw_sha256'])=={core.result_path(out,j).relative_to(out).as_posix() for j in jobs}
    raw_by_name={};groups={}
    for job in ([] if metadata_only else jobs):
        p=core.result_path(out,job);assert sha(p)==data['raw_sha256'][p.relative_to(out).as_posix()]
        d=read(p);assert d['job']==job and d['source_contract']==digest and d['audit']['status']=='PASS'
        c=d['summary']['counts'];assert c['generated']==c.get('success',0)+c.get('expired',0)
        assert d['rows'][-1]['active']==0 and sum(r['success'] for r in d['rows'])==c.get('success',0)
        assert sum(d['summary']['expiration_stage'].values())==c.get('expired',0)
        assert len(d['rows'])>=200
        raw_by_name[p.relative_to(out).as_posix()]=d
        if job['kind']=='eval':groups.setdefault((job['profile'],job['method'],job['block'],job['checkpoint']),[]).append(d)
    for row in data['summary']:
        if not metadata_only:
            actual=groups[(row['profile'],row['method'],row['block'],row['checkpoint'])];same(core.aggregate(actual),{k:v for k,v in row.items() if k not in ('profile','method','block','checkpoint')})
        assert row['episodes']==8 and row['success']<=row['admitted']<=row['generated']
        same(row['dsr'],row['success']/row['generated'])
        same(row['admission'],row['admitted']/row['generated']);same(row['conditional'],row['success']/row['admitted'])
        same(row['dsr'],row['admission']*row['conditional'])
        previous=next(x for x in old['summary'] if all(x[k]==row[k] for k in ('profile','method','block','checkpoint')))
        assert row['generated']==previous['generated']
        if row['method'] in core.CONTROLS:
            for key in ('generated','admitted','success'):assert row[key]==previous[key]
    for m,b in __import__('itertools').product(core.METHODS,range(3)):
        records=[read(out/'training'/str(b)/m/f'{cp:04}.json') for cp in range(1,7)]
        assert sum(r['episodes'] for r in records)==48 and sum(r['update']['optimizer_steps'] for r in records)==156
        same([r['update'] for r in records],data['updates'][f'{m}/{b}'])
    assert len(experience['batches'])==36
    for r in experience['batches']:
        assert r['episodes']==8 and r['replay_max_error']<.003 and len(r['source'])==8
        for source in r['source']:
            assert source['raw_sha256']==data['raw_sha256'][source['raw']]
            if not metadata_only:assert source['experience_sha256']==raw_by_name[source['raw']]['experience_sha256']
            assert re.fullmatch('[0-9a-f]{64}',source['experience_sha256'])
    cases=[]
    assert len(mechanism['raw_sha256'])==96 and all(re.fullmatch('[0-9a-f]{64}',h) for h in mechanism['raw_sha256'].values())
    assert all(re.fullmatch('[0-9a-f]{64}',h) for h in data['raw_sha256'].values())
    for name,h in ([] if metadata_only else mechanism['raw_sha256'].items()):
        p=out/name;assert sha(p)==h;d=read(p);assert d['source_contract']==digest and d['audit']['status']=='PASS'
        assert d['replay_steps_exact']>=200 and len(d['probes'])==4;cases.append(d)
        original_job=dict(kind='eval',method=d['method'],block=d['block'],checkpoint=6,profile='burst',workload=d['workload'])
        original_path=core.result_path(out if d['kind']=='native' else parent_out,original_job)
        source_manifest=data if d['kind']=='native' else old
        assert d['parent_raw_sha256']==source_manifest['raw_sha256'][original_path.relative_to(original_path.parents[1]).as_posix()]
        if d['kind']=='native':
            raw=raw_by_name[original_path.relative_to(out).as_posix()];same(d['counts'],raw['summary']['counts'])
            same([v['target'] for v in d['values']],mc_returns([r['success']/20 for r in raw['rows']]))
        for probe in d['probes']:
            for scale,mat in probe['payoff'].items():
                rank=trial.rank_record(mat['values']['actual'],probe['probability']);measured=probe['rank'][scale]
                same(rank,{k:v for k,v in measured.items() if k!='old'})
                if d['kind']=='matched':
                    old_probe=next(p for p in old_diag['probes'] if all(p[k]==d[k] for k in ('method','block','workload')) and p['time']==probe['time'])
                    same(mat,old_probe['payoff'][scale]);oldprob=old_probe['probabilities']['6']['actual' if d['method']=='joint' else 'pre']
                    same(measured['old'],trial.rank_record(mat['values']['actual'],oldprob))
    assert len(cases)==(0 if metadata_only else 96)
    for r in mechanism['value']:
        if not metadata_only:
            ds=[d for d in cases if d['kind']=='native' and d['method']==r['method']]
            target=[x['target'] for d in ds for x in d['values']];pred=[x[r['baseline']] for d in ds for x in d['values']]
            keys=[(d['block'],x['time']) for d in ds for x in d['values']]
            fit=value_fit_by_phase(target,pred,keys)
            for k,v in fit.items():same(r[k],v)
        assert r['total_states']==r['within_phase_states']==4848
        assert r['gate']==(r['within_phase_ev']>0 and r['within_phase_ev']>r['old_within_phase_ev'])
    for r in mechanism['ranking']:
        assert r['states']==96 and 0<=r['informative']<=96
        if not metadata_only:
            ranks=[p['rank'][r['scale']] for d in cases if d['method']==r['method'] and d['kind']==r['kind'] for p in d['probes']]
            assert len(ranks)==96 and r['informative']==sum(x['informative'] for x in ranks)
            same(r['mean_gain'],statistics.mean(x['gain'] for x in ranks));same(r['mean_regret'],statistics.mean(x['regret'] for x in ranks))
        if r['kind']=='matched':
            same(r['mean_gain_over_old'],r['mean_gain']-r['old_mean_gain'])
            if not metadata_only:same(r['mean_gain_over_old'],statistics.mean(x['gain']-x['old']['gain'] for x in ranks))
            assert r['gate']==(r['mean_gain_over_old']>0)
    for method in core.METHODS:
        gains=[trial.lookup(data['summary'],method,b,6)['dsr']-trial.lookup(data['summary'],method,b,0)['dsr'] for b in range(3)]
        same(gains,data['gain_over_initialization'][method]);assert data['learning_gate'][method]==(all(x>0 for x in gains) and statistics.mean(gains)>=.01)
    same(data['joint_minus_decomposed'],[trial.lookup(data['summary'],'joint',b,6)['dsr']-trial.lookup(data['summary'],'decomposed',b,6)['dsr'] for b in range(3)])
    deliver=task/'output'/core.TASK/'deliverables';report=deliver/'queued_plan_residual_short.md';links=[]
    for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        if '://' not in link and not link.startswith('#'):assert (report.parent/link).resolve().exists(),link;links.append(link)
    with (deliver/'queued_plan_residual_summary.csv').open(encoding='utf-8-sig',newline='') as h:cells=list(csv.DictReader(h))
    assert len(cells)==84
    for cell,record in zip(cells,data['summary']):
        for key,value in record.items():
            if isinstance(value,(float,int)):same(float(cell[key]),value)
            elif value is None:assert cell[key]==''
            else:same(cell[key],value)
    layout=subprocess.run([sys.executable,str(REPO_ROOT/'scripts/maintenance/check_layout.py')],capture_output=True,cwd=REPO_ROOT)
    log=layout.stdout.decode('utf-8',errors='replace')
    assert layout.returncode==1 and log.count('unexpected root files')==2
    postcheck=(val/'postcheck.log').read_text(encoding='utf-8');assert 'Ran 3 tests' in postcheck and 'OK' in postcheck
    result=dict(status='PASS_LOCAL_PLAN_RESIDUAL_METADATA_REVIEW' if metadata_only else 'PASS_COMPLETE_LOCAL_PLAN_RESIDUAL_RAW_REVIEW',source_contract=digest,client_date='2026-11-01',source_files=files,
        raw_main_episodes=0 if metadata_only else 960,raw_mechanism_cases=0 if metadata_only else 96,main_episode_manifest=960,mechanism_manifest=96,
        native_states=192,matched_states=192,training_episodes=288,optimizer_steps_per_learner=156,
        training_updates=36,experience_references=288,server_tests=81,post_freeze_adversarial_tests=3,logical_cpus=len(complete['used_cpus']),peak_active=complete['peak_active'],
        protected_files=complete['protected_files'],changed_old_files=[],learning_gate=data['learning_gate'],value_gate={r['method']+'/'+r['baseline']:r['gate'] for r in mechanism['value']},
        matched_ranking_gate={r['method']+'/'+r['scale']:r['gate'] for r in mechanism['ranking'] if r['kind']=='matched'},
        local_raw_complete=not metadata_only,server_raw_audit=True,layout_exit_code=layout.returncode,layout_log=log,report_links_checked=len(links),independent_review=False,
        review_script_sha256=sha(Path(__file__)),boundary='Metadata mode verifies source, budgets, seed deltas, summary counts and gate arithmetic; raw files remain on server. Full mode also recomputes raw arithmetic. Torch inference/gradients and experience replay executed on server, not independently repeated here')
    atomic(val/'local_review.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','layout_log')},ensure_ascii=False));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,default=REPO_ROOT);p.add_argument('--metadata-only',action='store_true')
    args=p.parse_args();review(args.task.resolve(),args.metadata_only)
