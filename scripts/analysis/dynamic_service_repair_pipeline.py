"""Bounded repair development with progress watchdog, immutable-source checks and controls."""
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
from scripts.analysis.dynamic_service_experiment import atomic_json,configure_runtime
from scripts.analysis.dynamic_service_repair_experiment import TASK,REVISION,METHODS,CONTROLS,LEGACY,protocol,digest


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def contract():return hashlib.sha256(json.dumps(protocol(),sort_keys=True).encode()).hexdigest()
def locations(task):return task/'results'/TASK/REVISION,task/'output'/TASK/'validation'/REVISION


def validate(task):
    import unittest
    configure_runtime();out,val=locations(task);stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_dynamic_service_repair','tests.test_dynamic_service_joint',
        'tests.test_dynamic_service_v2','tests.test_dynamic_service_runtime','tests.test_dynamic_service_scenarios'])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    atomic_json(val/'release_tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        source=str(REPO_ROOT),contract=contract(),log=stream.getvalue(),scientific_performance_claim=False))
    print(stream.getvalue(),flush=True);return 0 if result.wasSuccessful() else 1


def jobs(out):
    result=[]
    for method in METHODS:
        for b in range(3):result.append(dict(id=f'train_{method}_{b}',train=True,method=method,block=b,prerequisite=None,
            expected=out/'training/pilot'/str(b)/method/'0040.json',args=['train','--method',method,'--block',str(b)]))
    for method in METHODS+(LEGACY,):
        for b in range(3):
            for checkpoint in (0,20,40):
                for kind in ('D','A') if checkpoint==40 else ('D',):
                    for w in range(3):
                        for r in range(2):
                            result.append(dict(id=f'eval_{method}_{b}_{checkpoint}_{kind}_{w}_{r}',train=False,method=method,block=b,
                                prerequisite=out/'models/pilot'/str(b)/method/f'{checkpoint:04}.pt' if method in METHODS else None,
                                expected=out/'evaluation/pilot'/str(b)/method/str(checkpoint)/f'{kind}_{w}_{r}.json',
                                args=['evaluate','--method',method,'--block',str(b),'--checkpoint',str(checkpoint),'--kind',kind,'--workload',str(w),'--repeat',str(r)]))
    for method in CONTROLS:
        for kind in ('D','A'):
            for w in range(3):
                for r in range(2):result.append(dict(id=f'eval_{method}_{kind}_{w}_{r}',train=False,method=method,block=0,prerequisite=None,
                    expected=out/'evaluation/pilot/0'/method/'40'/f'{kind}_{w}_{r}.json',args=['evaluate','--method',method,'--kind',kind,'--workload',str(w),'--repeat',str(r)]))
    return result


def post_jobs(out):
    result=[]
    for method in METHODS+(LEGACY,)+CONTROLS:
        for b in range(3) if method not in CONTROLS else (0,):
            result.append(dict(id=f'timing_{method}_{b}',train=False,serial=True,method=method,block=b,prerequisite=None,
                expected=out/'timing_audit/pilot'/str(b)/method/'40/D_0_0.json',
                args=['evaluate','--timing-audit','--method',method,'--block',str(b)]))
    # The decoder diagnostic is predeclared and never used to pick the more
    # favorable primary score after evaluation.
    for b in range(3):
        for w in range(3):
            for r in range(2):result.append(dict(id=f'greedy_repair_joint_{b}_{w}_{r}',train=False,serial=False,method='repair_joint',block=b,prerequisite=None,
                expected=out/'greedy_diagnostic/pilot'/str(b)/'repair_joint/40'/f'D_{w}_{r}.json',
                args=['evaluate','--greedy','--method','repair_joint','--block',str(b),'--workload',str(w),'--repeat',str(r)]))
    return result


def analyze(task):
    import numpy as np
    out,val=locations(task);groups={};hashes={};tapes={}
    for job in jobs(out):
        if job['train']:continue
        path=job['expected'];d=json.loads(path.read_text());total=d['totals'];rows=d['rows']
        assert d['contract']==contract() and len(rows)==1000 and total['generated']==50000
        for key,value in total.items():assert sum(r[key] for r in rows)==value
        assert total['generated']==total['admitted']+total['uncovered'] and total['admitted']==total['success']+total['deadline_misses']
        if d['method'] in METHODS:
            model=out/'models/pilot'/str(d['block'])/d['method']/f"{d['checkpoint']:04}.pt"
            assert digest(model)==d['checkpoint_sha256']
        key=(d['kind'],d['workload'],d['repeat'])
        if key in tapes:assert tapes[key]==d['workload_fingerprint']
        tapes[key]=d['workload_fingerprint'];hashes[str(path.relative_to(out))]=digest(path)
        group=f"{d['kind']}/{d['method']}/{d['checkpoint']}";groups.setdefault(group,[]).append(d)
    summary={}
    for key,ds in groups.items():
        blocks={}
        for b in sorted({d['block'] for d in ds}):
            subset=[d for d in ds if d['block']==b];counts={k:sum(d['totals'][k] for d in subset) for k in subset[0]['totals']}
            rows=[r for d in subset for r in d['rows']]
            blocks[str(b)]=dict(**counts,dsr=counts['success']/counts['generated'],admission=counts['admitted']/counts['generated'],
                conditional=counts['success']/max(counts['admitted'],1),actual_m=float(np.mean([r['actual_step_m'] for r in rows])),
                commanded_m=float(np.mean([r['commanded_m'] for r in rows])),projection_residual=float(np.mean([r['projection_residual'] for r in rows])),
                loaded_p95_seconds=float(np.percentile([r['seconds'] for r in rows],95)))
        summary[key]=dict(mean_dsr=float(np.mean([d['dsr'] for d in blocks.values()])),by_block=blocks)
    comparisons={}
    for control in ('history','mpc','library_greedy',LEGACY,'repair_full_credit','repair_no_plan'):
        a=summary['D/repair_joint/40']['by_block'];b=summary[f'D/{control}/40']['by_block']
        delta=[a[str(i)]['dsr']-b['0' if control in CONTROLS else str(i)]['dsr'] for i in range(3)]
        comparisons[control]=dict(mean=float(np.mean(delta)),by_seed=delta,positive_seeds=sum(x>0 for x in delta),development_only=True)
    training={}
    for method in METHODS:
        for b in range(3):
            files=sorted((out/'training/pilot'/str(b)/method).glob('*.json'));assert len(files)==40
            ds=[json.loads(p.read_text()) for p in files]
            assert all(d['contract']==contract() and d['interaction_steps']==d['episode']*1000 for d in ds)
            training[f'{method}/{b}']=dict(actor_steps=sum(d['update']['actor_steps'] for d in ds),
                critic_steps=sum(d['update']['critic_steps'] for d in ds),kl_stop_episodes=sum(d['update']['stopped_kl'] for d in ds),
                backtracking_steps=sum(d['update']['backtracking_steps'] for d in ds),zero_updates=sum(d['update']['actor_steps']==0 for d in ds),
                seconds=sum(d['seconds'] for d in ds))
    atomic_json(out/'analysis.json',dict(status='COMPLETE_REPAIR_DEVELOPMENT',contract=contract(),summary=summary,comparisons=comparisons,
        training=training,evaluation_sha256=hashes,automatic_expansion=False,confirmation=False,
        limitations='Three training seeds, 40 episodes, reused development tapes; no architectural superiority or unbiased-original-joint-gradient claim'))


def run(args):
    task=args.task.resolve();out,val=locations(task);val.mkdir(parents=True,exist_ok=True)
    import fcntl
    handle=(val/'controller.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tested=json.loads((val/'release_tests.json').read_text())
    if tested['status']!='PASS' or tested['contract']!=contract() or tested['source']!=str(REPO_ROOT):raise RuntimeError('untested source')
    clearance=json.loads((val.parent/'development_seed_clearance.json').read_text())
    if clearance['development_collisions'] or clearance['status']!='PASS_DEVELOPMENT_ONLY':raise RuntimeError('uncleared development seeds')
    protected=val/'protected_before.json'
    if not protected.exists():
        paths=[p for p in (task/'results'/TASK).rglob('*') if p.is_file() and REVISION not in p.relative_to(task/'results'/TASK).parts and p.suffix not in ('.lock','.tmp')]
        atomic_json(protected,{str(p.relative_to(task)):digest(p) for p in paths})
    definition=protocol()
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=definition:raise RuntimeError('changed output contract')
    else:atomic_json(out/'protocol.json',definition)
    cpus=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topo=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology');key=((topo/'physical_package_id').read_text(),(topo/'core_id').read_text())
        if key not in seen:cpus.append(cpu);seen.add(key)
    cpus=cpus[8:8+min(args.workers,120)]
    all_jobs=jobs(out)+post_jobs(out);done=set();failed={};blocked=set();active=[];attempts={};began=time.monotonic();main_ids={j['id'] for j in jobs(out)}
    env=dict(os.environ,MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    legacy=task/'results'/TASK/'revision_e2e_20261030'
    base=[sys.executable,str(REPO_ROOT/'scripts/analysis/dynamic_service_repair_experiment.py'),'--output',str(out),'--legacy-root',str(legacy)]
    for job in all_jobs:
        if job['expected'].exists():
            assert json.loads(job['expected'].read_text())['contract']==contract();done.add(job['id'])
    atomic_json(val/'launch.json',dict(started=now(),pid=os.getpid(),contract=contract(),source=str(REPO_ROOT),cpus=cpus,
        training_jobs=15,training_episodes=600,main_evaluation_jobs=480,post_jobs=len(post_jobs(out)),
        total_jobs=len(all_jobs),watchdog_seconds=600,max_retries=1,confirmation=False))
    while len(done)+len(failed)+len(blocked)<len(all_jobs):
        for item in active[:]:
            code=item['process'].poll();heartbeat=item['heartbeat']
            last=heartbeat.stat().st_mtime if heartbeat.exists() else item['started']
            if code is None and time.time()-last>600:
                item['process'].terminate()
                try:item['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:item['process'].kill();item['process'].wait()
                code=item['process'].returncode
                atomic_json(val/(item['job']['id']+'_watchdog.json'),dict(time=now(),pid=item['process'].pid,reason='600 seconds without progress',attempt=item['attempt']))
            if code is None:continue
            item['log'].close();active.remove(item);job=item['job']
            if code==0 and job['expected'].exists() and json.loads(job['expected'].read_text())['contract']==contract():done.add(job['id'])
            elif item['attempt']>=2:failed[job['id']]=dict(code=code,attempts=item['attempt'])
        busy={x['cpu'] for x in active};free=[c for c in cpus if c not in busy];running={x['job']['id'] for x in active}
        for job in all_jobs:
            if job['id'] in done or job['id'] in failed or job['id'] in blocked or job['id'] in running:continue
            if job['id'] not in main_ids and not main_ids.issubset(done|set(failed)|blocked):continue
            if job.get('serial') and active:continue
            if any(x['job'].get('serial') for x in active):continue
            prerequisite=job['prerequisite']
            if prerequisite is not None and not prerequisite.exists():
                if f"train_{job['method']}_{job['block']}" in failed:blocked.add(job['id'])
                continue
            if job['id'] not in main_ids and job['method'] in METHODS and f"train_{job['method']}_{job['block']}" in failed:
                blocked.add(job['id']);continue
            if not free:continue
            cpu=free.pop(0);attempts[job['id']]=attempts.get(job['id'],0)+1
            hb=val/'progress'/(job['id']+'.json')
            atomic_json(hb,dict(updated=time.time(),stage='starting',attempt=attempts[job['id']]))
            log=(val/(job['id']+'.log')).open('a')
            command=['taskset','-c',str(cpu)]+base+job['args']+['--heartbeat',str(hb)]
            proc=subprocess.Popen(command,cwd=REPO_ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(dict(process=proc,job=job,cpu=cpu,log=log,heartbeat=hb,started=time.time(),attempt=attempts[job['id']]))
        atomic_json(val/'status.json',dict(updated=now(),status='RUNNING',completed=len(done),total=len(all_jobs),failed=failed,blocked=sorted(blocked),
            elapsed_seconds=time.monotonic()-began,active=[dict(job=x['job']['id'],pid=x['process'].pid,cpu=x['cpu']) for x in active],experiment_complete=False))
        if not active and len(done)+len(failed)+len(blocked)<len(all_jobs):raise RuntimeError('unresolved dependencies')
        if active:time.sleep(3)
    if not failed and not blocked:configure_runtime();analyze(task)
    prior=json.loads(protected.read_text());changed=[name for name,d in prior.items() if not (task/name).exists() or digest(task/name)!=d]
    result=dict(status='PASS_REPAIR_DEVELOPMENT_COMPLETED' if not failed and not blocked and not changed else 'FAIL',
        updated=now(),completed=len(done),total=len(all_jobs),failed=failed,blocked=sorted(blocked),changed_old_results=changed,
        protected_files=len(prior),experiment_complete=not failed and not blocked,confirmation=False,automatic_expansion=False)
    atomic_json(val/'completion.json',result);atomic_json(val/'status.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('validate','run','analyze'))
    p.add_argument('--task',type=Path,required=True);p.add_argument('--workers',type=int,default=96);args=p.parse_args()
    os.environ.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    if args.command=='validate':raise SystemExit(validate(args.task))
    if args.command=='run':run(args)
    else:configure_runtime();analyze(args.task)


if __name__=='__main__':main()
