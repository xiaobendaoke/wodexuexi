"""Bounded six-episode motion trial: validation, progress watchdog and final gate."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis.dynamic_service_motion_experiment import TASK,REVISION,METHODS,CONTROLS,protocol,contract
from scripts.analysis.dynamic_service_experiment import atomic_json,configure_runtime

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def locations(task):return task/'results'/TASK/REVISION,task/'output'/TASK/'validation'/REVISION
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def jobs(out):
    result=[]
    for m in METHODS:
        for b in range(3):
            result.append(dict(id=f'train_{m}_{b}',training=True,method=m,block=b,expected=out/'training'/str(b)/m/'0006.json',
                args=['train','--method',m,'--block',str(b)],dependency=None))
            for cp in (0,3,6):
                for w in range(2):
                    result.append(dict(id=f'eval_{m}_{b}_{cp}_{w}',training=False,method=m,block=b,
                        expected=out/'evaluation'/str(b)/m/str(cp)/f'D_{w}.json',dependency=out/'models'/str(b)/m/f'{cp:04}.pt',
                        args=['evaluate','--method',m,'--block',str(b),'--checkpoint',str(cp),'--workload',str(w)]))
            for w in range(2):
                result.append(dict(id=f'greedy_{m}_{b}_{w}',training=False,method=m,block=b,
                    expected=out/'greedy_diagnostic'/str(b)/m/'6'/f'D_{w}.json',dependency=out/'models'/str(b)/m/'0006.pt',
                    args=['evaluate','--greedy','--method',m,'--block',str(b),'--workload',str(w)]))
    for m in CONTROLS:
        for b in range(3) if m in ('uniform','service_greedy') else (0,):
            for w in range(2):
                result.append(dict(id=f'control_{m}_{b}_{w}',training=False,method=m,block=b,dependency=None,
                    expected=out/'evaluation'/str(b)/m/'6'/f'D_{w}.json',args=['evaluate','--method',m,'--block',str(b),'--workload',str(w)]))
    # Train first so all six runs start immediately, while evaluations consume spare cores.
    return sorted(result,key=lambda x:not x['training'])

def validate(task):
    import unittest
    configure_runtime();_,val=locations(task);stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_dynamic_service_motion','tests.test_dynamic_service_repair',
        'tests.test_dynamic_service_joint','tests.test_dynamic_service_v2','tests.test_dynamic_service_runtime','tests.test_dynamic_service_scenarios'])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    atomic_json(val/'release_tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        contract=contract(),source=str(REPO_ROOT),log=stream.getvalue(),independent_review=False))
    print(stream.getvalue(),flush=True);return result.wasSuccessful()

def analyze(task):
    import numpy as np
    out,val=locations(task);groups={};tapes={};hashes={};training={}
    for job in jobs(out):
        p=job['expected'];d=json.loads(p.read_text());assert d['contract']==contract()
        if job['training']:
            files=sorted((out/'training'/str(job['block'])/job['method']).glob('*.json'));assert len(files)==6
            ds=[json.loads(p.read_text()) for p in files]
            assert all(x['episode']==i and x['interaction_steps']==i*1000 for i,x in enumerate(ds,1))
            training[job['id']]=dict(actor_steps=sum(x['update']['actor_steps'] for x in ds),critic_steps=sum(x['update']['critic_steps'] for x in ds),
                label_calls=sum(x['label_calls'] for x in ds),label_seconds=sum(x['label_seconds'] for x in ds),seconds=sum(x['seconds'] for x in ds),
                decision_timeout=sum(x['totals']['decision_timeout'] for x in ds),skipped_chunks=sum(x['update']['skipped_chunks'] for x in ds))
            continue
        rows=d['rows'];total=d['totals'];assert len(rows)==1000 and total['generated']==50000
        for k,v in total.items():assert sum(r[k] for r in rows)==v
        assert total['generated']==total['admitted']+total['uncovered'] and total['admitted']==total['success']+total['deadline_misses']
        if d['method'] in METHODS:assert d['checkpoint_sha256']==sha(out/'models'/str(d['block'])/d['method']/f"{d['checkpoint']:04}.pt")
        w=d['workload'];assert w not in tapes or tapes[w]==d['workload_fingerprint'];tapes[w]=d['workload_fingerprint']
        key=f"{p.relative_to(out).parts[0]}/{d['method']}/{d['checkpoint']}";groups.setdefault(key,[]).append(d);hashes[str(p.relative_to(out))]=sha(p)
    summary={}
    for key,ds in groups.items():
        blocks={}
        for b in sorted({d['block'] for d in ds}):
            xs=[d for d in ds if d['block']==b];r=[r for d in xs for r in d['rows']];total={k:sum(d['totals'][k] for d in xs) for k in xs[0]['totals']}
            blocks[str(b)]=dict(**total,dsr=total['success']/total['generated'],admission=total['admitted']/total['generated'],
                conditional=total['success']/max(total['admitted'],1),actual_m=float(np.mean([x['actual_step_m'] for x in r])),
                p95_seconds=float(np.percentile([x['seconds'] for x in r],95)),
                entropy_fraction=float(np.mean([x['entropy_fraction'] for x in r])) if 'entropy_fraction' in r[0] else None,
                tv_uniform=float(np.mean([x['tv_uniform'] for x in r])) if 'tv_uniform' in r[0] else None)
        summary[key]=dict(mean_dsr=float(np.mean([x['dsr'] for x in blocks.values()])),by_block=blocks)
    comparisons={}
    for method in METHODS:
        end=summary[f'evaluation/{method}/6']['by_block'];initial=summary[f'evaluation/{method}/0']['by_block']
        comparisons[method]={}
        for reference in ('initial','uniform','hover','library_greedy','history','service_greedy','motion_ppo'):
            control=initial if reference=='initial' else summary[f'evaluation/{reference}/6']['by_block']
            delta=[end[str(b)]['dsr']-control[str(b) if str(b) in control else '0']['dsr'] for b in range(3)]
            comparisons[method][reference]=dict(mean=float(np.mean(delta)),by_seed=delta,positive_seeds=sum(x>0 for x in delta))
    aux=comparisons['motion_aux'];promising=(aux['initial']['positive_seeds']==3 and aux['initial']['mean']>=.01 and aux['uniform']['mean']>0 and aux['hover']['mean']>0)
    result=dict(status='PASS_SHORT_LEARNING_SIGNAL' if promising else 'SHORT_GATE_NOT_MET',contract=contract(),summary=summary,comparisons=comparisons,
        training=training,raw_sha256=hashes,confirmation=False,automatic_expansion=False,
        limitations='6 episodes, 3 reused development seeds x 2 workloads; model-assisted auxiliary, not pure RL superiority; greedy diagnostic separate; no joint result')
    atomic_json(out/'analysis.json',result);return result

def run(args):
    task=args.task.resolve();out,val=locations(task);val.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(val/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tests=json.loads((val/'release_tests.json').read_text());assert tests['status']=='PASS' and tests['source']==str(REPO_ROOT) and tests['contract']==contract()
    clearance=json.loads((val.parent/'development_seed_clearance.json').read_text());assert clearance['status']=='PASS_DEVELOPMENT_ONLY' and not clearance['development_collisions']
    protect=val/'protected_before.json'
    if not protect.exists():atomic_json(protect,{str(p.relative_to(task)):sha(p) for p in (task/'results'/TASK).rglob('*') if p.is_file() and REVISION not in p.relative_to(task/'results'/TASK).parts and p.suffix not in ('.lock','.tmp')})
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol()
    else:atomic_json(out/'protocol.json',protocol())
    schedule=jobs(out);active=[];done=set();failed={};blocked=set();attempts={}
    for job in schedule:
        if job['expected'].exists():assert json.loads(job['expected'].read_text())['contract']==contract();done.add(job['id'])
    cpus=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology');key=((p/'core_id').read_text(),(p/'physical_package_id').read_text())
        if key not in seen:cpus.append(cpu);seen.add(key)
    cpus=cpus[8:8+min(args.workers,120)]
    atomic_json(val/'launch.json',dict(started=now(),pid=os.getpid(),contract=contract(),source=str(REPO_ROOT),cpus=cpus,total_jobs=len(schedule),
        training_jobs=6,training_episodes=36,steps_per_episode=1000,main_evaluations=54,greedy_diagnostics=12,automatic_expansion=False))
    base=[sys.executable,str(REPO_ROOT/'scripts/analysis/dynamic_service_motion_experiment.py'),'--output',str(out)]
    while len(done)+len(failed)+len(blocked)<len(schedule):
        for x in active[:]:
            rc=x['proc'].poll();last=x['hb'].stat().st_mtime
            if rc is None and time.time()-last>600:
                x['proc'].terminate()
                try:x['proc'].wait(timeout=5)
                except subprocess.TimeoutExpired:x['proc'].kill();x['proc'].wait()
                rc=x['proc'].returncode
            if rc is None:continue
            active.remove(x);x['log'].close();job=x['job']
            if rc==0 and job['expected'].exists():assert json.loads(job['expected'].read_text())['contract']==contract();done.add(job['id'])
            elif attempts[job['id']]>=2:failed[job['id']]=rc
        busy={x['cpu'] for x in active};free=[c for c in cpus if c not in busy];running={x['job']['id'] for x in active}
        for job in schedule:
            if job['id'] in done or job['id'] in failed or job['id'] in blocked or job['id'] in running:continue
            dependency=job['dependency']
            if dependency is not None and not dependency.exists():
                if f"train_{job['method']}_{job['block']}" in failed:blocked.add(job['id'])
                continue
            if not free:continue
            cpu=free.pop(0);attempts[job['id']]=attempts.get(job['id'],0)+1;hb=val/'progress'/(job['id']+'.json')
            atomic_json(hb,dict(stage='starting',time=time.time()));log=(val/(job['id']+'.log')).open('a')
            p=subprocess.Popen(['taskset','-c',str(cpu)]+base+job['args']+['--heartbeat',str(hb)],env=os.environ,cwd=REPO_ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(dict(job=job,proc=p,cpu=cpu,hb=hb,log=log))
        atomic_json(val/'status.json',dict(updated=now(),completed=len(done),total=len(schedule),failed=failed,blocked=sorted(blocked),
            active=[dict(job=x['job']['id'],pid=x['proc'].pid,cpu=x['cpu']) for x in active],experiment_complete=False))
        if not active and len(done)+len(failed)+len(blocked)<len(schedule):raise RuntimeError('unresolved dependencies')
        if active:time.sleep(3)
    result=None
    if not failed and not blocked:configure_runtime();result=analyze(task)
    prior=json.loads(protect.read_text());changed=[name for name,d in prior.items() if sha(task/name)!=d]
    completion=dict(status='PASS_SHORT_EXPERIMENT_COMPLETED' if not failed and not blocked and not changed else 'FAIL',
        completed=len(done),total=len(schedule),failed=failed,blocked=sorted(blocked),updated=now(),protected_files=len(prior),changed_old_files=changed,
        learning_gate=result['status'] if result else None,experiment_complete=not failed and not blocked,automatic_expansion=False)
    atomic_json(val/'completion.json',completion);atomic_json(val/'status.json',completion)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('validate','run','analyze'));p.add_argument('--task',type=Path,required=True)
    p.add_argument('--workers',type=int,default=48);args=p.parse_args()
    os.environ.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    if args.command=='validate':raise SystemExit(0 if validate(args.task) else 1)
    elif args.command=='run':run(args)
    else:configure_runtime();analyze(args.task)
if __name__=='__main__':main()
