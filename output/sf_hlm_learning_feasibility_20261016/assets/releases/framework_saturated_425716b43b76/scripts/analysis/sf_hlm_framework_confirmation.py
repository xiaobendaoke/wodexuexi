"""Fresh foundation training for a prespecified analytical-framework confirmation.

This entry never dispatches the historical follow-up pipeline or its ten
confirmation blocks. Formal timed evaluations require a separate audited gate.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT

BLOCKS=tuple(range(1520001,1520021))
TRAIN_LOADS=tuple(range(1540001,1540021))
EVAL_LOADS=tuple(range(1560001,1560006))
METHODS=('G','R10','R16','binary_price','mixture','mixture_price','cold_prior_price','cold_half_proxy_price','HGB_probability_price')
PRIMARY='mixture_price'
SMOKE_BLOCK=21520001
SMOKE_LOAD=21540001
FILES=('paths.py','scripts/analysis/sf_hlm_framework_confirmation.py','tests/test_sf_hlm_framework_confirmation.py','docs/plans/sf_hlm_learning_feasibility_20261016.md')
ARGS=('release_root','feasibility_release','parent_launch','closed_launch','cold_calibration','readiness_results','readiness_validation','results_root','validation_root','seed_preflight')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def parents(a):return load('framework_foundation_feasibility',a.feasibility_release/'scripts/analysis/sf_hlm_learning_feasibility.py').parent_args(a)
def followup(a):return load('framework_foundation_followup',parents(a).followup_root/'scripts/analysis/sf_hlm_followup.py')
def resources(a):
    old=parents(a);return load('framework_foundation_resources',old.extension_root/'utils/sf_hlm_resources.py')


def design(smoke=False):
    blocks=[SMOKE_BLOCK] if smoke else list(BLOCKS);training={str(b):w for b,w in zip(blocks,[SMOKE_LOAD] if smoke else TRAIN_LOADS)}
    evaluation=[[m,b,w,e,'measured'] for m in METHODS for b in BLOCKS for w in EVAL_LOADS for e in range(3)];random.Random(2026101806).shuffle(evaluation)
    return dict(confirmation_blocks=blocks,training_workloads=training,confirmation_workloads=list(EVAL_LOADS),
        tasks=[['foundation_training','shared',b,training[str(b)],-1] for b in blocks],planned_evaluation_tasks=evaluation,
        foundation_episodes=2 if smoke else 200,steps=8 if smoke else 1000,confirmation_episodes_per_workload=3,
        methods=list(METHODS),primary=PRIMARY,primary_comparisons=[['mixture_price','G'],['mixture_price','binary_price']],
        gates=dict(mean_gain_G=.005,mean_gain_binary=.001,positive_blocks=16,total_blocks=20,energy_ratio_G=1.05,per_block_p95_seconds=1.,holm_alpha=.05),
        independent_unit='new foundation training block; shared fixed development calibrator is not independently retrained',
        no_original_ten_blocks=True,unique_learning_increment_resolved=False)


def prepare(a):
    if (a.results_root/'protocol.json').exists():raise FileExistsError('Immutable fresh-foundation namespace')
    old=parents(a);f=followup(a);original=read(old.output/'protocol.json');audit=read(a.readiness_validation/'raw_record_audit.json');ready=read(a.readiness_results/'analysis.json');seeds=read(a.seed_preflight)
    if a.results_root==old.results_root or a.results_root==old.parent_results or old.output in a.results_root.parents:raise ValueError('Original result namespace is protected')
    if audit['status']!='PASS' or audit['analysis_sha256']!=sha(a.readiness_results/'analysis.json') or not ready['readiness']['practical_development_signal_vs_G']:raise ValueError('Framework readiness not audited')
    expected=set(BLOCKS)|set(TRAIN_LOADS)|set(EVAL_LOADS)|{SMOKE_BLOCK,SMOKE_LOAD}
    if seeds['status']!='PASS' or set(seeds['target_seeds'])!=expected or seeds['hits']:raise ValueError('Fresh seed preflight is incomplete')
    closed_args={k:Path(v) for k,v in read(a.closed_launch)['args'].items()};closed_protocol=read(closed_args['results_root']/'protocol.json');closed_audit=read(closed_args['validation_root']/'raw_record_audit.json');fit_done=read(closed_args['fit_root']/'completion.json')
    if closed_audit['status']!='PASS' or closed_audit['summary_sha256']!=sha(closed_args['results_root']/'analysis/summary.json'):raise ValueError('Development full-cost source not audited')
    inputs={str(p):sha(p) for p in (a.parent_launch,a.closed_launch,a.seed_preflight,a.readiness_results/'protocol.json',a.readiness_results/'analysis.json',a.readiness_validation/'raw_record_audit.json',old.output/'protocol.json',old.suite_root/'suite.json',old.output/'upper_adaptation/protocol.json',closed_args['results_root']/'protocol.json',closed_args['results_root']/'analysis/summary.json',closed_args['validation_root']/'raw_record_audit.json',closed_args['fit_root']/'protocol.json',closed_args['fit_root']/'completion.json',a.cold_calibration/'protocol.json',a.cold_calibration/'completion.json')}
    for name,digest in fit_done['files'].items():
        path=closed_args['fit_root']/name
        if sha(path)!=digest:raise ValueError('Frozen full-development weights or prices changed')
        inputs[str(path)]=digest
    cold_done=read(a.cold_calibration/'completion.json')
    for path,digest in cold_done['files'].items():
        if sha(path)!=digest:raise ValueError('Public cold-control calibration input changed')
        inputs[path]=digest
    adaptation=read(old.output/'upper_adaptation/protocol.json')['files'];followup_files={name:sha(old.followup_root/name) for name in f.FILES}
    for files,root in ((original['extension_files'],old.extension_root),(adaptation,old.adaptation_root),(followup_files,old.followup_root)):
        for name,digest in files.items():
            if sha(root/name)!=digest:raise ValueError('Archived source changed')
            inputs[str(root/name)]=digest
    suite=read(old.suite_root/'suite.json')['plan'];recipe={k:suite[k] for k in ('rollout','ppo_epochs','batch_size')}
    if recipe!=dict(rollout=1000,ppo_epochs=8,batch_size=256):raise ValueError('Training recipe differs from registered ancestor')
    p=dict(name='sf_hlm_fresh_analytical_framework_confirmation_foundations_20261018_v1',smoke=a.smoke,
        original_protocol_sha256=sha(old.output/'protocol.json'),source_hash=original['source_hash'],extension_files=original['extension_files'],adaptation_files=adaptation,files=followup_files,
        own_files={name:sha(a.release_root/name) for name in FILES},input_files=inputs,training_recipe=recipe,checkpoint_selection='final prespecified episode only',
        original_closed_protocol_sha256=sha(closed_args['results_root']/'protocol.json'),frozen_development_fit=str(closed_args['fit_root']),frozen_cold_control_cases=str(a.cold_calibration),
        confirmation_evaluation_root=str(a.results_root.parent/'confirmation_v1'),canonical_child_purpose_label='verification for smoke, pilot for formal; unchanged archived numerical recipe',
        formal_evaluation_enabled=False,formal_evaluation_requires_audited_adapter_price_and_all_foundations=True,shared_lower_calibrator=True,original_confirmation_outcomes_used=False,
        authority='User continuous objective and approved learning-feasibility plan section 16; fresh-foundation scope registered in section 41',**design(a.smoke))
    write(a.results_root/'protocol.json',p);write(a.results_root/'confirmation_authorization.json',dict(authorized=True,protocol_sha256=sha(a.results_root/'protocol.json'),scope=['foundation_training'],authority=p['authority'],not_a_new_human_approval=True))
    write(a.results_root/'status.json',dict(status='PREPARED',foundations=len(p['tasks']),formal_evaluation_started=False));print(json.dumps(dict(status='PREPARED',foundations=len(p['tasks']),smoke=p['smoke'])),flush=True)


def verify(a,p,full=False):
    for name,digest in p['own_files'].items():
        if sha(a.release_root/name)!=digest:raise ValueError('Frozen framework training implementation changed')
    if full:
        for path,digest in p['input_files'].items():
            if sha(path)!=digest:raise ValueError('Framework training input changed')


def original_args(a,p,block):
    old=parents(a);old.results_root=a.results_root;old.phase='foundation_training';old.method='shared';old.block=block;old.workload=p['training_workloads'][str(block)];old.episode=-1;old.family='complete';old.seed=block;old.smoke=a.smoke;old.authorize_confirmation=True;old.development_only=False;old.stop_after=0
    return old


def worker(a,p):
    if a.block not in p['confirmation_blocks']:raise ValueError('Unregistered new training block')
    verify(a,p,True);followup(a).worker(original_args(a,p,a.block))


def command(a,mode):
    cmd=[sys.executable,str(a.release_root/'scripts/analysis/sf_hlm_framework_confirmation.py'),mode]
    for name in ARGS:cmd+=['--'+name.replace('_','-'),str(getattr(a,name))]
    if a.smoke:cmd.append('--smoke')
    return cmd


def audit_foundations(a,p):
    verify(a,p,True);f=followup(a);r=resources(a);old=parents(a);out=[]
    with r.CpuLease(old.resources,task=['framework_foundations_audit',p['smoke']],kind='environment'):
        for block in p['confirmation_blocks']:
            args=original_args(a,p,block);path=f.folder(args);receipt=f.verified(path,full=True);expected=dict(followup_protocol_sha256=sha(a.results_root/'protocol.json'),source_hash=p['source_hash'],phase='foundation_training',method='shared',block=block,workload=p['training_workloads'][str(block)],episode=-1)
            if receipt['contract']!=expected or receipt['episodes']!=p['foundation_episodes']:raise ValueError('Foundation identity or episode count changed')
            latest=read(path/'canonical/latest.json');checkpoint=path/'canonical'/latest['checkpoint'];lock=read(checkpoint/'contract.json')
            if latest['completed_episodes']!=p['foundation_episodes'] or lock['source_hash']!=p['source_hash']:raise ValueError('Wrong final foundation checkpoint')
            if set(x.name for x in (path/'canonical/episodes').glob('*.json'))!={f'{i:05d}.json' for i in range(p['foundation_episodes'])}:raise ValueError('Training episode population differs')
            out.append(dict(block=block,workload=args.workload,path=str(path),receipt_sha256=sha(path/'result.json'),checkpoint=str(checkpoint),checkpoint_contract_sha256=sha(checkpoint/'contract.json'),source_hash=lock['source_hash'],config_hash=lock['config_hash']))
    report=dict(status='PASS',foundations=len(out),smoke=p['smoke'],source_and_all_checkpoint_files_verified=True,final_checkpoint_only=True,original_confirmation_blocks_untouched=True,protocol_sha256=sha(a.results_root/'protocol.json'),blocks=out,audit_type='primary agent self-audit of archived canonical training contracts and full receipts',formal_evaluation_started=False)
    write(a.results_root/'foundation_audit.json',report);write(a.results_root/'completion.json',dict(status='COMPLETED',foundations=len(out),protocol_sha256=sha(a.results_root/'protocol.json'),foundation_audit_sha256=sha(a.results_root/'foundation_audit.json'),formal_confirmation_completed=False));write(a.results_root/'status.json',dict(status='FOUNDATIONS_COMPLETED',foundations=len(out),formal_evaluation_started=False));print(json.dumps(dict(status='FOUNDATIONS_COMPLETED',foundations=len(out),smoke=p['smoke'])),flush=True)


def train(a,p):
    verify(a,p,True);r=resources(a)
    if not a.smoke:
        smoke=a.results_root.parent/'smoke_v1';done=read(smoke/'completion.json');sp=read(smoke/'protocol.json');sa=read(smoke/'foundation_audit.json')
        if done['status']!='COMPLETED' or not sp['smoke'] or sp['own_files']!=p['own_files'] or sa['status']!='PASS' or done['foundation_audit_sha256']!=sha(smoke/'foundation_audit.json'):raise ValueError('Compatible independent foundation smoke not audited')
    def execute(block):
        path=a.results_root/'worker_logs'/f'block_{block}.log';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as log:
            process=subprocess.Popen(command(a,'worker')+['--block',str(block)],stdout=log,stderr=subprocess.STDOUT,cwd=a.release_root)
            write(a.results_root/'workers'/f'{block}.json',dict(pid=process.pid,block=block,started_unix=time.time(),command=process.args))
            code=process.wait()
        if code:raise RuntimeError('Fresh foundation worker failed: '+str(block))
    with r.task_lock(a.results_root/'controller.lock'):
        try:
            with ThreadPoolExecutor(max_workers=1 if a.smoke else 6) as pool:
                jobs=[pool.submit(execute,block) for block in p['confirmation_blocks']]
                for count,job in enumerate(as_completed(jobs),1):job.result();write(a.results_root/'status.json',dict(status='TRAINING',completed_foundations=count,total=len(jobs),updated_unix=time.time(),formal_evaluation_started=False))
            audit_foundations(a,p)
        except BaseException as error:write(a.results_root/'status.json',dict(status='FAILED',error=repr(error),updated_unix=time.time(),formal_evaluation_started=False));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','worker','train','audit'))
    for name in ARGS:p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--block',type=int,default=-1);a=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('All numerical framework confirmation work runs on the experiment server')
    if a.mode=='prepare':prepare(a)
    else:
        protocol=read(a.results_root/'protocol.json');verify(a,protocol)
        if protocol['smoke']!=a.smoke:raise ValueError('Smoke/formal namespace mismatch')
        if a.mode=='worker':worker(a,protocol)
        elif a.mode=='train':train(a,protocol)
        else:audit_foundations(a,protocol)


if __name__=='__main__':main()
