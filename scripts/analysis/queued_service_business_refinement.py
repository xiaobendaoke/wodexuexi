"""One bounded recovery revision, reusing frozen execution via explicit inputs.

Activation changes only this process's scenario factory, output namespace and
source manifest. No earlier source file or earlier result is edited. Pools must
use Linux fork so workers inherit this declared execution configuration.
"""
import argparse
from dataclasses import asdict
import io
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_service_business as core
from utils.queued_service_business_refinement import RecoveryScenario,NAMESPACE

REVISION='business_recovery_20261031'
SOURCES=tuple(dict.fromkeys(tuple(core.SOURCES)+('utils/queued_service_business_refinement.py',
    'scripts/analysis/queued_service_business_refinement.py','tests/test_queued_service_business_refinement.py',
    'scripts/analysis/queued_service_business_precision.py')))


def activate():
    core.REVISION=REVISION;core.SOURCES=SOURCES;core.BusinessScenario=RecoveryScenario


def specs(phase,peak=None):
    result=[]
    workloads=range(12,36) if phase=='screen' else range(36,48) if phase=='holdout' else (36,37)
    for profile in core.PROFILES:
        peaks=(45,60,75) if phase=='screen' and profile=='burst' else (peak,) if profile=='burst' else (None,)
        for p,m,w,dt in itertools.product(peaks,core.METHODS,workloads,(.05,.025) if phase=='precision' else (.1,)):
            result.append(dict(phase=phase,profile=profile,sla='normal',scale=p/100 if profile=='burst' else .75,
                               method=m,workload=w,dt=dt,horizon=200,burst_peak=p))
    return result


def validate(task):
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromNames(
        ['tests.test_queued_service_business_refinement','tests.test_queued_service_business','tests.test_queued_service'])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);_,val=core.locations(task)
    core.atomic(val/'tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        source_contract=core.contract()[0],log=stream.getvalue(),independent_review=False,time=core.now()))
    print(stream.getvalue(),flush=True);return result.wasSuccessful()


def choose(screen,task):
    candidates=[]
    for peak in (45,60,75):
        rows=[r for r in screen['summary'] if r['profile']=='burst' and r['scale']==peak/100]
        assert len(rows)==3
        candidates.append(dict(peak=peak,passed=all(r['gate'] for r in rows),
            failures={r['method']:[k for k,v in r['criteria'].items() if not v] for r in rows},reference_only=peak==75))
    eligible=[x for x in candidates if x['passed'] and not x['reference_only']]
    selected=dict(sla='normal',scale=(eligible[0]['peak'] if eligible else 45)/100,
        burst_peak=eligible[0]['peak'] if eligible else 45,passed=bool(eligible))
    result=dict(status='FROZEN_RECOVERY_CANDIDATE' if eligible else 'NO_SETTING_PASSED_DIAGNOSTIC_ONLY',
        selected=selected,candidates=candidates,source_contract=core.contract()[0],uses_controller_ranking=False)
    out,val=core.locations(task)
    if (out/'selection.json').exists():assert core.read(out/'selection.json')==result
    else:core.atomic(out/'selection.json',result)
    core.atomic(val/'selection_frozen.json',dict(**result,selected_before_new_workloads=True))
    return selected


def analyze(task):
    out,val=core.locations(task);selected=core.read(out/'selection.json')['selected'];peak=selected['burst_peak']
    screen=core.aggregate(specs('screen'),task);hold=core.aggregate(specs('holdout',peak),task)
    precision=[];hashes={}
    for s in specs('precision',peak):
        path=core.path_for(out,s);d=core.read(path);parent_path=core.path_for(out,dict(s,phase='holdout',dt=.1));parent=core.read(parent_path)
        assert d['fixed_action_parent_sha256']==core.sha(parent_path)
        assert d['source_contract']==core.contract()[0] and d['arrival_fingerprint']==parent['arrival_fingerprint']
        precision.append(dict(spec=s,dsr_change=d['summary']['dsr']-parent['summary']['dsr'],
            success_change=d['summary']['counts'].get('success',0)-parent['summary']['counts'].get('success',0)))
        hashes[str(path.relative_to(out))]=core.sha(path)
    old=core.read(val/'protected_before.json');changed=[p for p,h in old.items() if not (task/p).exists() or core.sha(task/p)!=h]
    records=[core.read(core.path_for(out,s)) for s in specs('screen')];probes=hold['interventions'];valid=[p for p in probes if p['eligible']]
    assert len(probes)==192
    burst=[r for r in hold['summary'] if r['profile']=='burst'];hold_gate=all(r['gate'] for r in burst)
    result=dict(status='PASS_COMPLETED_RECOVERY_REVISION' if not changed else 'FAIL_OLD_FILES_CHANGED',
        scientific_gate='PASS_NEW_WORKLOAD_RECOVERY' if selected['passed'] and hold_gate else 'FAILED_NEW_WORKLOAD_RECOVERY',
        source_contract=core.contract()[0],selection=core.read(out/'selection.json'),screen=screen,holdout=hold,
        precision=precision,precision_raw_sha256=hashes,max_abs_fixed_action_dsr_change=max(abs(x['dsr_change']) for x in precision),
        interventions=dict(total=len(probes),eligible=len(valid),nonzero=sum(p['interaction']!=0 for p in valid),reversals=sum(p['reversal'] for p in valid)),
        actual_screen_cpus=sorted({d['cpu'] for d in records}),main_generated_tasks=sum(d['summary']['counts']['generated'] for d in records),
        protected_files=len(old),changed_old_files=changed,completed=core.now(),no_rl=True,
        limitations='Second and last workload revision after a failed first holdout; synthetic image-event business, not real SLA calibration; no RL superiority')
    core.atomic(out/'analysis.json',result)
    core.atomic(val/'completion.json',{k:v for k,v in result.items() if k not in ('screen','holdout','precision','precision_raw_sha256','actual_screen_cpus')})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('validate','package','run','analyze','reversal_precision'))
    p.add_argument('--task',type=Path,default=REPO_ROOT);p.add_argument('--workers',type=int,default=256);p.add_argument('--output',type=Path)
    a=p.parse_args();activate();task=a.task.resolve();out,val=core.locations(task)
    if a.command=='package':
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(a.output,'x',zipfile.ZIP_DEFLATED) as z:
            for name in SOURCES:z.write(REPO_ROOT/name,name)
        print(json.dumps(dict(source_contract=core.contract()[0],zip_sha256=core.sha(a.output))));return
    if a.command=='validate':raise SystemExit(0 if validate(task) else 1)
    if a.command=='analyze':analyze(task);return
    if sys.platform!='linux':raise RuntimeError('Experiments require the Linux server')
    if multiprocessing.get_start_method(allow_none=True) is None:multiprocessing.set_start_method('fork')
    if multiprocessing.get_start_method()!='fork':raise RuntimeError('Explicit scenario configuration requires fork workers')
    if a.command=='reversal_precision':
        from scripts.analysis import queued_service_business_precision as precision
        precision.BusinessScenario=RecoveryScenario
        sys.argv=[str(precision.__file__),'--task',str(task)]
        precision.main();return
    import fcntl
    val.mkdir(parents=True,exist_ok=True);lock=(val/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tests=core.read(val/'tests.json');assert tests['status']=='PASS' and tests['source_contract']==core.contract()[0]
    core.protect(task,val);digest,files=core.contract()
    proto=json.loads(json.dumps(dict(namespace=NAMESPACE,physics=asdict(core.Protocol()),
        input_bytes=[1500000,3000000],cycles_per_bit=[500,1000],sla_seconds=[2,3],baseline_burst_rate=15,
        candidate_peaks=[45,60],reference_peak=75,common_rates=dict(light=7.5,steady=30,overload=90),
        prior_failed_revision='business_calibration_20261031',development_workloads=list(range(12,36)),
        holdout_workloads=list(range(36,48)),selection='Prefer45 if all physical gates pass, otherwise60;75 reference only',
        screen_criteria=core.protocol()['screen'],screen_jobs=432,holdout_jobs=144,precision_jobs=48,
        no_rl=True,automatic_further_revision=False,source_contract=digest,source_files=files)))
    if (out/'protocol.json').exists():assert core.read(out/'protocol.json')==proto
    else:core.atomic(out/'protocol.json',proto)
    core.atomic(val/'launch.json',dict(started=core.now(),pid=os.getpid(),source=str(REPO_ROOT),source_contract=digest,
        cpus=core.core_order(),workers=a.workers,source_disk_unchanged=True,start_method='fork',no_rl=True))
    core.batch(specs('screen'),task,a.workers);screen=core.aggregate(specs('screen'),task);core.atomic(out/'screen_analysis.json',screen)
    selected=choose(screen,task);peak=selected['burst_peak']
    core.batch(specs('holdout',peak),task,a.workers);core.batch(specs('precision',peak),task,a.workers)
    result=analyze(task);print(json.dumps({k:v for k,v in result.items() if k not in ('screen','holdout','precision','precision_raw_sha256','actual_screen_cpus')}),flush=True)


if __name__=='__main__':main()
