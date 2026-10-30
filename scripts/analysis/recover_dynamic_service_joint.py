"""Recover a stalled, never-updated joint-policy seed without rewriting completed work.

The scientific source stays frozen. A single-thread retry starts from the exact
initial parameters, optimizer states and RNG; thread metadata and provenance are
explicitly different. Publishing refuses to overwrite any completed record.
"""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK='chapter5_dynamic_service_20261028'
REVISION='revision_e2e_20261030'
METHOD='trajectory_matching'


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)


def copy_new(source,target):
    """Atomic, no-clobber publication. A differing existing file is an error."""
    if target.exists():
        if sha(source)!=sha(target):raise RuntimeError(f'refusing to replace existing data: {target}')
        return
    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.publish')
    shutil.copy2(source,temp)
    # The recovery controller is the sole writer after the old controller stops.
    if target.exists():raise RuntimeError(f'concurrent writer: {target}')
    os.replace(temp,target)


def stop_owned(pid,expected):
    """Never signal a process whose full command line has changed."""
    cmd=Path(f'/proc/{pid}/cmdline')
    if not cmd.exists():return False
    before=cmd.read_bytes().replace(b'\0',b' ').decode()
    if any(part not in before for part in expected):raise RuntimeError(f'PID {pid} identity mismatch: {before}')
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not cmd.exists():return True
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        if stat=='Z':return True
        time.sleep(.1)
    after=cmd.read_bytes().replace(b'\0',b' ').decode()
    if after!=before:raise RuntimeError('PID identity changed before forced termination')
    os.kill(pid,signal.SIGKILL)
    return True


def paths(args):
    out=args.task/'results'/TASK/REVISION
    validation=args.task/'output'/TASK/'validation'/REVISION
    evidence=validation/'recovery_20261030'
    retry=out/'recovery_single_thread'
    return out,validation,evidence,retry


def setup(args):
    args.source=args.source.resolve();args.task=args.task.resolve()
    sys.path.insert(0,str(args.source))
    os.environ.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parent.parent/'lib'))
    out,validation,evidence,retry=paths(args)
    definition=json.loads((out/'protocol.json').read_text())
    for relative,expected in definition['source'].items():
        if sha(args.source/relative)!=expected:raise RuntimeError(f'frozen source changed: {relative}')
    args.contract=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
    return out,validation,evidence,retry


def probe(args):
    out,validation,evidence,retry=setup(args)
    import faulthandler
    faulthandler.enable();faulthandler.dump_traceback_later(25,repeat=True)
    from scripts.analysis.dynamic_service_experiment import configure_runtime
    from scripts.analysis.dynamic_service_joint_experiment import run_episode
    from utils.dynamic_service_scenarios import seed_for
    torch=configure_runtime();torch.set_num_threads(args.threads)
    from marl_models.dynamic_service_joint import JointPolicy,JointCritic,update_joint
    import marl_models.dynamic_service_joint as joint
    initial=out/'models/pilot/0'/METHOD/'0000.pt'
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    if saved['episode']!=0 or saved['contract']!=args.contract or saved['seed']!=seed_for('dev/train',0):
        raise RuntimeError('probe initial state mismatch')
    actor=JointPolicy(METHOD);critic=JointCritic(METHOD)
    actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic'])
    ao=torch.optim.Adam(actor.parameters(),lr=3e-4);co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    ao.load_state_dict(saved['aopt']);co.load_state_dict(saved['copt']);torch.set_rng_state(saved['rng'])
    from types import SimpleNamespace
    scoped=SimpleNamespace(method=METHOD,block=0,steps=1000)
    progress=evidence/f'probe_{args.threads}_progress.json';started=time.monotonic()
    from environment.dynamic_service_v2 import DynamicServiceEnvV2
    original=DynamicServiceEnvV2.step
    def step(env,*a,**kw):
        result=original(env,*a,**kw)
        if env.t%100==0:atomic(progress,dict(updated=now(),stage='collection',step=env.t,seconds=time.monotonic()-started))
        return result
    DynamicServiceEnvV2.step=step
    original_grad=joint.branch_gradients
    def gradients(*a,**kw):
        atomic(progress,dict(updated=now(),stage='branch_gradients_enter',seconds=time.monotonic()-started))
        result=original_grad(*a,**kw)
        atomic(progress,dict(updated=now(),stage='branch_gradients_exit',seconds=time.monotonic()-started))
        return result
    joint.branch_gradients=gradients
    atomic(progress,dict(updated=now(),stage='starting',threads=args.threads))
    result,rollout=run_episode(scoped,actor,critic,torch,episode=0)
    atomic(progress,dict(updated=now(),stage='update',step=1000,seconds=time.monotonic()-started))
    metrics=update_joint(actor,critic,ao,co,rollout)
    faulthandler.cancel_dump_traceback_later()
    atomic(evidence/f'probe_{args.threads}_result.json',dict(status='PASS',threads=args.threads,
        initial_sha256=sha(initial),contract=args.contract,seconds=time.monotonic()-started,
        totals=result['totals'],update=metrics,formal_performance_evidence=False,
        purpose='instrumented full-episode runtime diagnostic, not scientific training'))
    print(json.dumps(dict(status='PASS',threads=args.threads,seconds=time.monotonic()-started)),flush=True)


def start(args):
    out,validation,evidence,retry=setup(args)
    checked=json.loads((evidence/'probe_1_result.json').read_text())
    if checked['status']!='PASS' or checked['contract']!=args.contract:raise RuntimeError('single-thread full-episode probe required')
    status=json.loads((validation/'status.json').read_text())
    launch=json.loads((validation/'process.json').read_text())
    if status.get('failed') or len(status.get('active',[]))!=1 or status['active'][0]['job']!='train_trajectory_matching_0':
        raise RuntimeError('recovery scope changed; inspect current jobs')
    training=out/'training/pilot/0'/METHOD
    if list(training.glob('*.json')):raise RuntimeError('this recovery is only for an uncommitted first episode')
    from scripts.analysis.dynamic_service_experiment import configure_runtime
    torch=configure_runtime();latest=out/'models/pilot/0'/METHOD/'latest.pt'
    saved=torch.load(latest,map_location='cpu',weights_only=False)
    if saved['episode']!=0 or saved['training_threads']!=8 or saved['contract']!=args.contract:
        raise RuntimeError('unexpected retry checkpoint')
    initial=out/'models/pilot/0'/METHOD/'0000.pt'
    if checked['initial_sha256']!=sha(initial):raise RuntimeError('probe/retry initial state changed')
    original=torch.load(initial,map_location='cpu',weights_only=False)
    def same(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    if not all(same(saved[k],original[k]) for k in ('actor','critic','aopt','copt','rng')):
        raise RuntimeError('episode-0 checkpoint payload differs from probed initial state')
    evidence.mkdir(parents=True,exist_ok=True)
    manifest=evidence/'recovery.json'
    if manifest.exists():raise RuntimeError('recovery already exists; inspect its status')
    existing={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'recovery_single_thread' not in p.relative_to(out).parts
              and p!=latest and p.suffix not in ('.lock','.tmp')}
    atomic(evidence/'completed_before.json',existing)
    atomic(evidence/'original_status.json',status);atomic(evidence/'original_process.json',launch)
    shutil.copy2(latest,evidence/'original_latest_episode0.pt')
    shutil.copy2(validation/'train_trajectory_matching_0.log',evidence/'original_training.log')
    record=dict(started=now(),original_controller_pid=launch['pid'],original_training_pid=status['active'][0]['pid'],
        source=str(args.source),contract=args.contract,initial_sha256=sha(initial),old_threads=8,new_threads=1,
        reason='no committed episode for over 80 minutes; libgomp spin at TensorIterator/cat; independent 1-thread episode passed',
        scientific_algorithm_changed=False,completed_runs_retrained=False,protected_results=len(existing),
        initial_parameters_optimizers_rng_equal=True,environment_overrides=dict(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0'),
        recovery_script_sha256=sha(Path(__file__)),status='PREPARING')
    other=evidence/'probe_8_result.json'
    if other.exists():
        eight=json.loads(other.read_text())
        record['probe_comparison']=dict(single_thread_seconds=checked['seconds'],eight_thread_seconds=eight['seconds'],
            same_request_counters=checked['totals']==eight['totals'],
            conclusion='both independent fresh processes passed; original OpenMP stall is not a deterministic data-loop failure')
    atomic(manifest,record)
    stop_owned(launch['pid'],[str(args.source/'scripts/analysis/dynamic_service_joint_pipeline.py'),' run ',str(args.task)])
    stop_owned(status['active'][0]['pid'],[str(args.source/'scripts/analysis/dynamic_service_joint_experiment.py'),
        'train --method trajectory_matching --block 0 --threads 8'])
    # Initial tensor/optimizer/RNG payload is preserved. Only execution metadata
    # changes; no trained weights or past rewards are transformed.
    saved['training_threads']=1
    target=retry/'models/pilot/0'/METHOD
    target.mkdir(parents=True,exist_ok=True)
    torch.save(saved,target/'latest.pt');torch.save(saved,target/'0000.pt')
    command=[sys.executable,str(Path(__file__).resolve()),'execute','--source',str(args.source),'--task',str(args.task)]
    with (evidence/'controller.log').open('a') as log:
        proc=subprocess.Popen(command,env=os.environ.copy(),cwd=args.source,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(status='RECOVERING',recovery_controller_pid=proc.pid,command=command)
    atomic(manifest,record)
    atomic(validation/'status.json',dict(updated=now(),status='RECOVERING',completed=status['completed'],total=429,
        failed={},blocked=[],active=[dict(job='single_thread_recovery',pid=proc.pid,cpus=list(range(32,40)))],
        experiment_complete=False,confirmation=False,recovery_manifest=str(manifest)))
    print(json.dumps(record),flush=True)


def execute(args):
    out,validation,evidence,retry=setup(args)
    import fcntl
    lock=(evidence/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    entry=args.source/'scripts/analysis/dynamic_service_joint_experiment.py'
    command=['taskset','-c','32',sys.executable,str(entry),'train','--method',METHOD,'--block','0',
             '--threads','1','--output',str(retry)]
    with (evidence/'single_thread_training.log').open('a') as log:
        process=subprocess.Popen(command,cwd=args.source,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(evidence/'training_process.json',dict(pid=process.pid,command=command,started=now()))
        last_progress=time.monotonic();last_episode=0
        while process.poll() is None:
            files=sorted((retry/'training/pilot/0'/METHOD).glob('*.json'))
            if len(files)>last_episode:last_episode=len(files);last_progress=time.monotonic()
            atomic(evidence/'status.json',dict(updated=now(),status='TRAINING_SINGLE_THREAD',episode=last_episode,
                total_episodes=40,pid=process.pid,no_progress_seconds=time.monotonic()-last_progress))
            atomic(validation/'status.json',dict(updated=now(),status='RECOVERING',completed=392,total=429,
                failed={},blocked=[],active=[dict(job='train_trajectory_matching_0_single_thread',pid=process.pid,cpus=[32])],
                recovery_episode=last_episode,recovery_total_episodes=40,
                recovery_controller_pid=os.getpid(),experiment_complete=False,confirmation=False))
            if time.monotonic()-last_progress>600:
                stop_owned(process.pid,[str(entry),'train --method trajectory_matching --block 0 --threads 1'])
                raise RuntimeError('single-thread recovery made no episode progress for 600 seconds')
            time.sleep(5)
    if process.returncode:raise RuntimeError(f'single-thread training failed: {process.returncode}')
    from scripts.analysis.dynamic_service_experiment import configure_runtime
    torch=configure_runtime();model=retry/'models/pilot/0'/METHOD
    state=torch.load(model/'0040.pt',map_location='cpu',weights_only=False)
    if state['episode']!=40 or state['training_threads']!=1 or state['contract']!=args.contract:raise RuntimeError('incomplete recovery model')
    records=sorted((retry/'training/pilot/0'/METHOD).glob('*.json'))
    if len(records)!=40:raise RuntimeError('missing recovery records')
    published={}
    for name in ('0020.pt','0040.pt'):
        target=out/'models/pilot/0'/METHOD/name;copy_new(model/name,target);published[str(target.relative_to(out))]=sha(target)
    for path in records:
        data=json.loads(path.read_text())
        if data['contract']!=args.contract or data['interaction_steps']!=data['episode']*1000:raise RuntimeError('invalid recovery record')
        target=out/'training/pilot/0'/METHOD/path.name;copy_new(path,target);published[str(target.relative_to(out))]=sha(target)
    protected=json.loads((evidence/'completed_before.json').read_text())
    changed=[name for name,digest in protected.items() if not (out/name).exists() or sha(out/name)!=digest]
    if changed:raise RuntimeError(f'completed data changed: {changed}')
    atomic(evidence/'publication.json',dict(status='PASS',updated=now(),published=published,
        protected_results=len(protected),changed_results=changed,original_initial_checkpoint_preserved=True,
        runtime_change='only the missing seed trained with 1 thread; other seeds retain their original runtime',
        latest_note='original latest.pt retained as episode-0 failure evidence; final models are 0020/0040.pt'))
    command=[sys.executable,str(args.source/'scripts/analysis/dynamic_service_joint_pipeline.py'),
             'run','--task',str(args.task),'--workers','256']
    with (evidence/'resumed_pipeline.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=args.source,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(evidence/'resumed_pipeline.json',dict(pid=proc.pid,command=command,started=now(),
        reason='40 recovered episodes published; original scheduler skips all completed jobs and resumes remaining evaluations'))
    old=json.loads((evidence/'original_process.json').read_text());old.update(pid=proc.pid,command=command,recovery_manifest=str(evidence/'recovery.json'))
    atomic(validation/'process.json',old)
    atomic(evidence/'status.json',dict(updated=now(),status='RESUMED_EVALUATIONS',episode=40,pipeline_pid=proc.pid))
    print(json.dumps(dict(status='RESUMED_EVALUATIONS',pid=proc.pid)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('probe','start','execute'))
    p.add_argument('--source',type=Path,required=True);p.add_argument('--task',type=Path,required=True)
    p.add_argument('--threads',type=int,choices=(1,8),default=1)
    args=p.parse_args()
    try:globals()[args.command](args)
    except Exception as exc:
        _,_,evidence,_=paths(args)
        atomic(evidence/f'{args.command}_error.json',dict(updated=now(),status='FAIL',error=repr(exc)))
        raise


if __name__=='__main__':main()
