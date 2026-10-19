"""Run the existing direct audit after this exact foundation job terminates.

No training, model, protocol or evaluation changes; no retries on failed jobs.
This is an immediate experiment dependency, not a recurring scheduler.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

FILES=('paths.py','scripts/analysis/sf_hlm_framework_dependency.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def process_state(pid):
    root=Path('/proc')/str(pid)
    try:
        fields=(root/'stat').read_text().rsplit(')',1)[1].split();command=(root/'cmdline').read_bytes()
        return dict(pid=pid,state=fields[0],start_ticks=fields[19],command_sha256=hashlib.sha256(command).hexdigest())
    except FileNotFoundError:return dict(pid=pid,state='missing')


def same_live_process(record):
    actual=process_state(record['pid'])
    return actual['state'] not in ('missing','Z','X') and all(actual.get(k)==record.get(k) for k in ('start_ticks','command_sha256'))


def run(a):
    data=read(a.run_manifest);root=Path(data.get('release',data.get('frozen_release','')));m=load('framework_dependency_frozen_evaluator',root/'scripts/analysis/sf_hlm_framework_evaluation.py');args=argparse.Namespace(**{k:Path(v) for k,v in data['args'].items()});f,fa,p=m.foundation(args)
    if root!=args.release_root or p['smoke']!=a.smoke:raise ValueError('Foundation/source/smoke identity differs')
    if not a.smoke and p['confirmation_blocks']!=list(f.BLOCKS):raise ValueError('Registered twenty-block foundation population differs')
    with f.resources(fa).task_lock(a.output/'controller.lock'):
        if (a.output/'protocol.json').exists():raise FileExistsError('Immutable dependency-run namespace')
        identities=[process_state(pid) for pid in a.training_pid]
        if any(x['state'] in ('missing','Z','X') for x in identities) and not (fa.results_root/'completion.json').exists():raise RuntimeError('Missing training handle without completed foundation evidence; no automatic restart')
        protocol=dict(name='sf_hlm_direct_audit_after_exact_foundation_job_20261018_v1',files={name:sha(a.release_root/name) for name in FILES},run_manifest=str(a.run_manifest),run_manifest_sha256=sha(a.run_manifest),foundation_protocol_sha256=sha(fa.results_root/'protocol.json'),training_processes=identities,smoke=a.smoke,formal_evaluation_dispatched=False,no_training_restarts=True)
        write(a.output/'protocol.json',protocol)
        while any(same_live_process(record) for record in identities):
            write(a.output/'status.json',dict(status='WAITING_FOR_EXACT_TRAINING_JOB',updated_unix=time.time(),training_processes=[process_state(x['pid']) for x in identities],formal_evaluation_dispatched=False));time.sleep(30)
        # A terminal handle alone is insufficient. Require every original receipt
        # and the unchanged source's own full completion audit before continuing.
        m.foundation(args);done=read(fa.results_root/'completion.json');prior=read(fa.results_root/'foundation_audit.json')
        if done['status']!='COMPLETED' or done['protocol_sha256']!=protocol['foundation_protocol_sha256'] or done['foundation_audit_sha256']!=sha(fa.results_root/'foundation_audit.json') or prior['status']!='PASS' or prior['foundations']!=len(p['confirmation_blocks']):raise RuntimeError('Foundation job ended without its full verified completion; no audit dispatch')
        auditor=args.release_root/'scripts/analysis/sf_hlm_framework_foundation_audit.py';cmd=[sys.executable,str(auditor),'--foundation-release',str(args.foundation_release),'--foundation-launch',str(args.foundation_launch),'--out',str(args.direct_audit)]
        if a.smoke:cmd.append('--smoke')
        if not args.direct_audit.exists():
            with (a.output/'direct_audit.log').open('ab') as stream:
                child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=args.release_root);write(a.output/'audit_launch.json',dict(pid=child.pid,command=cmd,started_unix=time.time(),auditor_sha256=sha(auditor)));write(a.output/'status.json',dict(status='DIRECT_AUDIT_RUNNING',pid=child.pid,formal_evaluation_dispatched=False));code=child.wait()
            write(a.output/'audit_exit.json',dict(returncode=code,ended_unix=time.time()))
            if code:raise RuntimeError('Direct audit failed; preserve logs and do not retry')
        report=read(args.direct_audit)
        if report['status']!='PASS' or report['smoke']!=a.smoke or report['protocol_sha256']!=protocol['foundation_protocol_sha256'] or report['foundation_completion_sha256']!=sha(fa.results_root/'completion.json') or report['audit_script_sha256']!=sha(auditor) or [row['block'] for row in report['blocks']]!=p['confirmation_blocks'] or report['distinct_upper_weight_fingerprints']!=len(p['confirmation_blocks']) or report['training_steps_checked']!=len(p['confirmation_blocks'])*p['foundation_episodes']*p['steps']:raise ValueError('Direct audit output does not prove the exact registered foundation population')
        write(a.output/'completion.json',dict(status='COMPLETED_AUDIT_ONLY',smoke=a.smoke,protocol_sha256=sha(a.output/'protocol.json'),direct_audit=str(args.direct_audit),direct_audit_sha256=sha(args.direct_audit),existing_audit_reused=not (a.output/'audit_launch.json').exists(),formal_evaluation_dispatched=False,formal_confirmation_completed=False));write(a.output/'status.json',dict(status='COMPLETED_AUDIT_ONLY',formal_evaluation_dispatched=False));print(json.dumps(dict(status='COMPLETED_AUDIT_ONLY',smoke=a.smoke,foundations=len(p['confirmation_blocks']))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release-root',type=Path,default=REPO_ROOT);parser.add_argument('--run-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--training-pid',type=int,action='append',default=[]);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('This experiment dependency runs on the server')
    try:run(a)
    except BaseException as error:
        if not (a.output/'failure.json').exists():write(a.output/'failure.json',dict(status='FAILED',error=repr(error),updated_unix=time.time(),formal_evaluation_dispatched=False,no_training_restarted=True))
        raise


if __name__=='__main__':main()
