"""Test and dispatch only the bounded, main-scene joint development experiment."""
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
from scripts.analysis.dynamic_service_joint_experiment import TASK,REVISION,METHODS,CONTROLS,protocol,digest
from scripts.analysis.dynamic_service_experiment import atomic_json,configure_runtime


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()


def contract():return hashlib.sha256(json.dumps(protocol(),sort_keys=True).encode()).hexdigest()


def locations(task):
    return task/'results'/TASK/REVISION,task/'output'/TASK/'validation'/REVISION


def validate(task):
    import unittest
    configure_runtime();out,validation=locations(task)
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_dynamic_service_joint',
        'tests.test_dynamic_service_v2','tests.test_dynamic_service_runtime','tests.test_dynamic_service_scenarios'])
    stream=io.StringIO();started=time.time();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    evidence=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        elapsed_seconds=time.time()-started,source=str(REPO_ROOT),contract=contract(),log=stream.getvalue(),
        review='explicit self-review, not independent agent review',scientific_claims=False)
    atomic_json(validation/'release_tests.json',evidence);print(stream.getvalue(),flush=True)
    return 0 if result.wasSuccessful() else 1


def protection(task):
    files=[]
    for folder in (task/'results'/TASK,task/'output'/TASK/'validation'):
        files.extend(p for p in folder.rglob('*') if p.is_file() and REVISION not in p.relative_to(folder).parts)
    return {str(p.relative_to(task)):digest(p) for p in sorted(files)}


def jobs(out):
    result=[]
    for method in METHODS:
        for block in range(3):
            train_id=f'train_{method}_{block}'
            result.append(dict(id=train_id,train=True,method=method,block=block,
                expected=out/'training/pilot'/str(block)/method/'0040.json',
                args=['train','--method',method,'--block',str(block)],prerequisite=None,parent=None))
            for checkpoint in (0,20,40):
                for kind in ('D','A') if checkpoint==40 else ('D',):
                    for workload in range(3):
                        for repeat in range(2):
                            result.append(dict(id=f'eval_{method}_{block}_{checkpoint}_{kind}_{workload}_{repeat}',
                                train=False,method=method,block=block,
                                expected=out/'evaluation/pilot'/str(block)/method/str(checkpoint)/f'{kind}_{workload}_{repeat}.json',
                                prerequisite=out/'models/pilot'/str(block)/method/f'{checkpoint:04}.pt',parent=train_id,
                                args=['evaluate','--method',method,'--block',str(block),'--checkpoint',str(checkpoint),
                                      '--kind',kind,'--workload',str(workload),'--repeat',str(repeat)]))
    for method in CONTROLS:
        for kind in ('D','A'):
            for workload in range(3):
                for repeat in range(2):
                    result.append(dict(id=f'eval_{method}_{kind}_{workload}_{repeat}',train=False,method=method,block=0,
                        expected=out/'evaluation/pilot/0'/method/'40'/f'{kind}_{workload}_{repeat}.json',
                        prerequisite=None,parent=None,args=['evaluate','--method',method,'--kind',kind,
                            '--workload',str(workload),'--repeat',str(repeat)]))
    return result


def timing_jobs(out):
    result=[]
    for method in METHODS+CONTROLS:
        for block in range(3) if method in METHODS else (0,):
            result.append(dict(id=f'timing_{method}_{block}',train=False,timing=True,method=method,block=block,
                expected=out/'timing_audit/pilot'/str(block)/method/'40/D_0_0.json',
                prerequisite=out/'models/pilot'/str(block)/method/'0040.pt' if method in METHODS else None,
                parent=f'train_{method}_{block}' if method in METHODS else None,
                args=['evaluate','--timing-audit','--method',method,'--block',str(block)]))
    return result


def analyze(task):
    import numpy as np
    out,validation=locations(task);expected=[j for j in jobs(out) if not j['train']]
    missing=[j['id'] for j in expected if not j['expected'].exists()]
    if missing:raise RuntimeError(f'{len(missing)} evaluations missing')
    groups={};hashes={};fingerprints={}
    for job in expected:
        path=job['expected'];data=json.loads(path.read_text());rows=data['rows'];total=data['totals']
        if data['contract']!=contract() or len(rows)!=1000 or total['generated']!=50000:raise RuntimeError('evaluation identity/count')
        if data['method'] in METHODS:
            model=out/'models/pilot'/str(data['block'])/data['method']/f"{data['checkpoint']:04}.pt"
            if digest(model)!=data['checkpoint_sha256']:raise RuntimeError('checkpoint digest mismatch')
        for k,v in total.items():
            if sum(r[k] for r in rows)!=v:raise RuntimeError('raw/summed counter mismatch')
        if total['generated']!=total['uncovered']+total['admitted'] or total['admitted']!=total['success']+total['deadline_misses']:
            raise RuntimeError('request conservation')
        key=(data['kind'],data['workload'],data['repeat'])
        if key in fingerprints and fingerprints[key]!=data['workload_fingerprint']:raise RuntimeError('unpaired tape')
        fingerprints[key]=data['workload_fingerprint'];hashes[str(path.relative_to(out))]=digest(path)
        group=f"{data['kind']}/{data['method']}/{data['checkpoint']}"
        groups.setdefault(group,[]).append(data)
    summary={}
    for key,ds in groups.items():
        blocks={}
        for b in sorted({d['block'] for d in ds}):
            subset=[d for d in ds if d['block']==b]
            counts={k:sum(d['totals'][k] for d in subset) for k in subset[0]['totals']}
            rows=[r for d in subset for r in d['rows']]
            blocks[str(b)]=dict(**counts,dsr=counts['success']/counts['generated'],
                admission=counts['admitted']/counts['generated'],conditional=counts['success']/max(counts['admitted'],1),
                edge60=sum(r['edge60_uav_steps'] for r in rows)/(5*len(rows)),
                hover=counts['hovering_uav_steps']/(5*len(rows)),
                actual_step_m=float(np.mean([r['actual_step_m'] for r in rows])),
                p95_seconds=float(np.percentile([r['seconds'] for r in rows],95)))
        summary[key]=dict(cases=len(ds),by_block=blocks,mean_dsr=float(np.mean([v['dsr'] for v in blocks.values()])))
    differences={}
    for kind in ('D','A'):
        for control in ('history','mpc','hover','trajectory_matching','joint_no_condition','joint_mlp'):
            candidate=summary[f'{kind}/joint/40']['by_block'];ref=summary[f'{kind}/{control}/40']['by_block']
            diffs=[candidate[str(b)]['dsr']-ref[str(b) if control in METHODS else '0']['dsr'] for b in range(3)]
            differences[f'{kind}/joint-{control}']=dict(mean=float(np.mean(diffs)),by_block=diffs,
                positive_blocks=sum(x>0 for x in diffs),unit='three training blocks; nonlearning control evaluated once')
    training={}
    for method in METHODS:
        for b in range(3):
            files=sorted((out/'training/pilot'/str(b)/method).glob('*.json'))
            if len(files)!=40:raise RuntimeError('incomplete training')
            ds=[json.loads(f.read_text()) for f in files]
            if any(d['contract']!=contract() or d['interaction_steps']!=d['episode']*1000 for d in ds):raise RuntimeError('training identity')
            training[f'{method}/{b}']=dict(episodes=40,actor_steps=sum(d['update']['actor_steps'] for d in ds),
                critic_steps=sum(d['update']['critic_steps'] for d in ds),stopped_kl_episodes=sum(d['update']['stopped_kl'] for d in ds),
                seconds=sum(d['seconds'] for d in ds),first10_mean_dsr=float(np.mean([d['dsr'] for d in ds[:10]])),
                last10_mean_dsr=float(np.mean([d['dsr'] for d in ds[-10:]])),
                mean_allocation_encoder_grad=float(np.mean([d['update']['allocation_encoder_grad'] for d in ds])),
                mean_motion_encoder_grad=float(np.mean([d['update']['motion_encoder_grad'] for d in ds])),
                mean_gradient_cosine=float(np.mean([d['update']['encoder_grad_cosine'] for d in ds])),
                max_post_batch_kl=max(d['update']['max_post_batch_kl'] for d in ds))
    atomic_json(out/'analysis.json',dict(status='COMPLETE_DEVELOPMENT_DIAGNOSTIC',contract=contract(),
        summary=summary,differences=differences,training=training,evaluation_sha256=hashes,
        interpretation='40-episode development only; loaded timings provisional; serial audit saved separately; no automatic expansion',confirmation=False))
    timing={}
    for job in timing_jobs(out):
        data=json.loads(job['expected'].read_text())
        if data['contract']!=contract() or not data['timing_audit'] or len(data['rows'])!=1000:raise RuntimeError('incomplete timing audit')
        loaded=out/'evaluation/pilot'/str(data['block'])/data['method']/'40/D_0_0.json'
        old=json.loads(loaded.read_text())
        timing[job['id']]=dict(p95_seconds=data['p95_seconds'],totals=data['totals'],
            loaded_p95_seconds=old['p95_seconds'],loaded_totals=old['totals'],
            cpu_affinity=data['cpu_affinity'],source_sha256=digest(job['expected']),
            note='same workload/model; any changed timeout fallback is retained, never substituted into primary evaluation')
    atomic_json(out/'timing_audit.json',timing)
    return summary


def run(task,workers):
    if sys.platform!='linux' or not 1<=workers<=256:raise RuntimeError('server only; workers 1..256 logical CPUs')
    out,validation=locations(task);validation.mkdir(parents=True,exist_ok=True)
    import fcntl
    handle=(validation/'controller.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tested=json.loads((validation/'release_tests.json').read_text())
    if tested['status']!='PASS' or tested['contract']!=contract() or tested['source']!=str(REPO_ROOT):raise RuntimeError('untested release')
    clearance=json.loads((validation.parent/'development_seed_clearance.json').read_text())
    if clearance['status']!='PASS_DEVELOPMENT_ONLY' or clearance['development_collisions']:raise RuntimeError('development seed clearance')
    protected_file=validation/'protected_before.json'
    if not protected_file.exists():atomic_json(protected_file,protection(task))
    protected=json.loads(protected_file.read_text())
    definition=protocol()
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=definition:raise RuntimeError('output contract changed')
    else:atomic_json(out/'protocol.json',definition)
    cpus=[];siblings=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topo=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((topo/'physical_package_id').read_text(),(topo/'core_id').read_text())
        if key not in seen:cpus.append(cpu);seen.add(key)
        else:siblings.append(cpu)
    cpus=(cpus+siblings)[:workers]
    if not cpus:raise RuntimeError('no dedicated physical cores')
    env=dict(os.environ,MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    base=[sys.executable,str(REPO_ROOT/'scripts/analysis/dynamic_service_joint_experiment.py'),'--output',str(out)]
    calibration=json.loads((validation/'runtime_calibration.json').read_text())
    thread_map=calibration['threads_by_method']
    if any(thread_map.get(m) not in (1,2,4,8) for m in METHODS):raise RuntimeError('invalid calibrated threads')
    main_jobs=jobs(out);all_jobs=main_jobs+timing_jobs(out);done=set();failed={};blocked=set();active=[];started=time.time()
    launch=dict(started=now(),controller_pid=os.getpid(),source=str(REPO_ROOT),contract=contract(),
        cpus=cpus,workers=len(cpus),training_threads=thread_map,training_jobs=15,evaluation_jobs=396,timing_jobs=18,training_episodes=600,
        steps_per_episode=1000,jobs=[j['id'] for j in all_jobs],confirmation=False,expansion=False,
        seed_clearance_sha256=digest(validation.parent/'development_seed_clearance.json'),
        runtime_calibration_sha256=digest(validation/'runtime_calibration.json'),frequency_mode='performance requested by user')
    path=validation/'launch.json'
    if path.exists():
        if json.loads(path.read_text())['contract']!=contract():raise RuntimeError('controller contract changed')
    else:atomic_json(path,launch)
    for job in all_jobs:
        if job['expected'].exists():
            if json.loads(job['expected'].read_text())['contract']!=contract():raise RuntimeError('mixed job result')
            done.add(job['id'])
    while len(done)+len(failed)+len(blocked)<len(all_jobs):
        for item in active[:]:
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();active.remove(item);job=item['job']
            if code==0 and job['expected'].exists() and json.loads(job['expected'].read_text())['contract']==contract():done.add(job['id'])
            else:failed[job['id']]=dict(returncode=code,log=str(item['log_path']))
        active_ids={i['job']['id'] for i in active};busy={cpu for i in active for cpu in i['cpus']}
        free=[c for c in cpus if c not in busy]
        # All train jobs are listed first; remaining physical cores evaluate
        # immutable checkpoints as soon as they appear.
        for job in all_jobs:
            if job['id'] in done or job['id'] in failed or job['id'] in blocked or job['id'] in active_ids:continue
            if job.get('timing'):
                if any(j['id'] not in done and j['id'] not in failed and j['id'] not in blocked for j in main_jobs) or active:continue
            prerequisite=job['prerequisite']
            if prerequisite is not None and not prerequisite.exists():
                if job['parent'] in failed:blocked.add(job['id'])
                continue
            threads=thread_map[job['method']] if job['train'] else 1
            if len(free)<threads:continue
            assigned=free[:threads];free=free[threads:];log_path=validation/(job['id']+'.log');log=log_path.open('a')
            process=subprocess.Popen(['taskset','-c',','.join(map(str,assigned))]+base+job['args']+['--threads',str(threads)],cwd=REPO_ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(dict(process=process,job=job,cpus=assigned,log=log,log_path=log_path))
        atomic_json(validation/'status.json',dict(updated=now(),elapsed_seconds=time.time()-started,
            completed=len(done),total=len(all_jobs),failed=failed,blocked=sorted(blocked),
            active=[dict(job=i['job']['id'],pid=i['process'].pid,cpus=i['cpus']) for i in active],
            status='RUNNING',experiment_complete=False,confirmation=False))
        if not active and len(done)+len(failed)+len(blocked)<len(all_jobs):raise RuntimeError('unresolved job dependency')
        if active:time.sleep(3)
    if not failed and not blocked:analyze(task)
    changed=[name for name,sha in protected.items() if not (task/name).exists() or digest(task/name)!=sha]
    final=dict(updated=now(),status='PASS_DEVELOPMENT_COMPLETED' if not failed and not blocked and not changed else 'FAIL',
        completed=len(done),total=len(all_jobs),failed=failed,blocked=sorted(blocked),protected_files=len(protected),
        changed_protected_files=changed,contract=contract(),experiment_complete=not failed and not blocked,
        confirmation=False,automatic_expansion=False)
    atomic_json(validation/'completion.json',final);atomic_json(validation/'status.json',final)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('validate','run','analyze'))
    p.add_argument('--task',type=Path,required=True);p.add_argument('--workers',type=int,default=256)
    args=p.parse_args();task=args.task.resolve()
    os.environ.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if args.command=='validate':raise SystemExit(validate(task))
    if args.command=='run':run(task,args.workers)
    else:configure_runtime();analyze(task)


if __name__=='__main__':main()
