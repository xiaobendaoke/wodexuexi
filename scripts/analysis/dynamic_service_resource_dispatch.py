"""NUMA-aware high-throughput development DAG, with bounded CPU ownership."""
import argparse
import atexit
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def build_jobs(output, methods, primary_only=False):
    """Build the fixed development DAG, including evaluation-time model reads."""
    jobs = []
    for block in range(5):
        for method in ('full', 'independent', 'no_memory', 'no_feedback', 'parallel'):
            jobs.append(dict(id=f'train_dev_{block}_{method}', command='train', block=block, method=method,
                             args=['train', '--split', 'dev', '--block', str(block), '--method', method],
                             dependencies=[] if method in ('full', 'independent') else [(block, 'full')]))
    cells = [('D', 1.)] if primary_only else [(k, s) for k in 'ABCD' for s in (.6, 1., 1.4)]
    for kind, scale in cells:
        for method in methods:
            for block in range(1 if method in ('history', 'mpc') else 5):
                deps = [] if method in ('history', 'mpc') else [
                    (block, 'full' if method in ('greedy_lower', 'matching_lower') else method)]
                if method == 'full':
                    # evaluate_job loads both controls before its first step,
                    # even on repeats without snapshot capture. Never expose
                    # a partially trained control checkpoint to that reader.
                    deps += [(block, 'independent'), (block, 'parallel')]
                for workload in range(5):
                    for repeat in range(6):
                        jobs.append(dict(id=f'eval_dev_{block}_{method}_{kind}_{scale}_{workload}_{repeat}',
                            command='evaluate', block=block, method=method, kind=kind, scale=scale,
                            workload=workload, repeat=repeat,
                            args=['evaluate', '--split', 'dev', '--block', str(block), '--method', method,
                                  '--kind', kind, '--scale', str(scale), '--workload', str(workload),
                                  '--repeat', str(repeat)], dependencies=list(deps),
                            expected=output/'evaluation/dev'/str(block)/method/f'{kind}_{scale}_{workload}_{repeat}.json'))
    jobs.sort(key=lambda j: (0 if j['command'] == 'train' else 1,
                            0 if j.get('kind') == 'D' and j.get('scale') == 1. else 1,
                            j['method'] == 'mpc', j['block']))
    expected = 1135 if primary_only else 13345
    if len({job['id'] for job in jobs}) != len(jobs) or len(jobs) != expected:
        raise RuntimeError('duplicate or missing jobs in the fixed development DAG')
    return jobs


def topology():
    groups = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        topo = base/'topology'
        key = (int((topo/'physical_package_id').read_text()), int((topo/'core_id').read_text()))
        nodes = list(base.glob('node[0-9]*'))
        node = int(nodes[0].name[4:]) if nodes else key[0]
        groups.setdefault(key, dict(node=node, cpus=[]))['cpus'].append(cpu)
    domains = sorted({g['node'] for g in groups.values()})
    by_node = {n: sorted((g for g in groups.values() if g['node'] == n), key=lambda g:g['cpus'][0]) for n in domains}
    ordered = []
    for index in range(max(map(len, by_node.values()))):
        for node in domains:
            if index < len(by_node[node]):
                ordered.append(by_node[node][index])
    reserved = ordered[:8]
    timing = ordered[8:16]
    mixed = ordered[16:]
    cpu_info = {cpu: dict(node=g['node'], primary=cpu == g['cpus'][0], siblings=g['cpus']) for g in ordered for cpu in g['cpus']}
    return reserved, timing, mixed, cpu_info


def main():
    # Keep the pure DAG importable for lightweight checks on the writing host.
    import fcntl
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--equivalence', type=Path, required=True)
    parser.add_argument('--cpu-budget', type=int, default=240)
    parser.add_argument('--max-trainers', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--adopt-status', type=Path)
    parser.add_argument('--primary-only', action='store_true',
                        help='Complete D/1.0 development evaluations before resuming the wider grid')
    args = parser.parse_args()
    source, extension, task = (p.resolve() for p in (args.source, args.extension, args.task))
    if source.name != 'source_b4b20295' or not source.is_relative_to(task) or not extension.is_relative_to(task):
        raise RuntimeError('unexpected scientific source/extension')
    sys.path.insert(0, str(source))
    from scripts.analysis.dynamic_service_experiment import protocol, source_contract, atomic_json, METHODS
    from utils.dynamic_service_scenarios import seed_for
    validation = task/'output/chapter5_dynamic_service_20261028/validation'
    output = task/'results/chapter5_dynamic_service_20261028'
    checks = json.loads(args.equivalence.read_text())
    if checks['status'] != 'PASS' or checks['environment_cache']['status'] != 'PASS_EXACT' or len(checks['comparisons']) != 3:
        raise RuntimeError('execution equivalence/calibration incomplete')
    workers = checks['selected_workers']
    if workers not in (0, 1, 2, 4, 8):
        raise RuntimeError('invalid calibrated worker count')
    tests = json.loads((validation/'runtime_tests_b4b20295.json').read_text())
    if tests['status'] != 'PASS' or tests['source'] != source_contract():
        raise RuntimeError('scientific source differs from tested frozen release')
    clearance = json.loads((validation/'development_seed_clearance.json').read_text())
    raw_audit = (validation/'seed_audit.json').read_bytes()
    if (clearance['status'] != 'PASS_DEVELOPMENT_ONLY' or clearance['original_audit_sha256'] != hashlib.sha256(raw_audit).hexdigest()
            or clearance['development_collisions']):
        raise RuntimeError('development seed clearance invalid')
    definition = protocol()
    contract = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    if clearance['protocol'] != contract:
        raise RuntimeError('clearance applies to another protocol')
    reserved, timing, mixed, info = topology()
    mixed_cpus = [c for g in mixed for c in g['cpus']]
    timing_cpus = [g['cpus'][0] for g in timing]
    if args.cpu_budget > 240 or args.cpu_budget < 16 or args.max_trainers < 1:
        parser.error('budget must be 16..240 and trainers positive')
    total_capacity = min(args.cpu_budget, len(mixed_cpus)+len(timing_cpus))
    runtime_sources = {str(p.relative_to(extension)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        extension/'utils/dynamic_service_parallel.py', extension/'scripts/analysis/dynamic_service_accelerated.py')}
    attestation = json.loads((validation/'execution_attestation.json').read_text())
    if (attestation['equivalence_sha256'] != hashlib.sha256(args.equivalence.read_bytes()).hexdigest()
            or attestation['runtime_sources'] != runtime_sources):
        raise RuntimeError('execution source differs from calibrated runtime')

    def model_done(block, method):
        checkpoint = output/'models/dev'/str(block)/(method+'.pt')
        last = output/'training/dev'/str(block)/method/'0399.json'
        if not checkpoint.exists() or not last.exists():
            return False
        row = json.loads(last.read_text())
        return row['episode'] == 400 and row['protocol'] == contract

    jobs = build_jobs(output, METHODS, args.primary_only)
    if args.dry_run:
        print(json.dumps(dict(status='READY', scientific_contract=contract, cpu_budget=args.cpu_budget,
            effective_logical_capacity=total_capacity, reserved_physical_cores=len(reserved),
            dedicated_mpc_physical_cores=len(timing), gradient_workers=workers,
            initially_ready_training=10, training_tasks=25, evaluation_tasks=len(jobs)-25,
            primary_only=args.primary_only,
            confirmation_dispatch=False, runtime_sources=runtime_sources)))
        return
    lock = (validation/'execution/controller.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft,min(hard,4096)), hard))
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='GNU')
    subprocess.run([sys.executable,str(source/'scripts/analysis/dynamic_service_experiment.py'),
                    'protocol','--output',str(output)],cwd=source,check=True)
    atomic_json(validation/'resource_launch.json',dict(pid=os.getpid(), started=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        source=str(source), extension=str(extension), protocol=contract, cpu_budget=args.cpu_budget,
        primary_only=args.primary_only,
        controller_source=str(Path(__file__).resolve()),controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        effective_logical_capacity=total_capacity, max_trainers=args.max_trainers, gradient_workers=workers,
        runtime_sources=runtime_sources, confirmation_dispatch=False,
        timing_cores=timing_cpus, reserved_cores=[g['cpus'] for g in reserved], uid=os.geteuid()))
    done, active = set(), []
    def release_paused_parents():
        for item in active:
            if not item.get('paused_for_loans'): continue
            path=Path('/proc')/str(item['process'].pid)/'cmdline'
            if path.exists():
                command=path.read_bytes()
                if b'dynamic_service_accelerated.py' in command and str(source).encode() in command:
                    try: os.kill(item['process'].pid,signal.SIGCONT)
                    except ProcessLookupError: pass
    atexit.register(release_paused_parents)
    def terminate_controller(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate_controller)
    for job in jobs:
        if job['command']=='train':
            if model_done(job['block'],job['method']): done.add(job['id'])
        elif job['expected'].exists():
            data=json.loads(job['expected'].read_text())
            if data['protocol']!=contract or data['smoke'] or len(data['rows'])!=1000:
                raise RuntimeError('existing evaluation invalid')
            done.add(job['id'])
    job_lookup={job['id']:job for job in jobs}
    def job_complete(job):
        if job['command']=='train': return model_done(job['block'],job['method'])
        if not job['expected'].exists(): return False
        row=json.loads(job['expected'].read_text())
        return row['protocol']==contract and not row['smoke'] and len(row['rows'])==1000
    class AdoptedProcess:
        def __init__(self,pid,cmdline,job): self.pid,self.cmdline,self.job=pid,cmdline,job
        def poll(self):
            path=Path('/proc')/str(self.pid)/'cmdline'
            if path.exists() and path.read_bytes()==self.cmdline: return None
            return 0 if job_complete(self.job) else 1
    if args.adopt_status:
        adopted=json.loads(args.adopt_status.read_text())
        for row in adopted['active']:
            job=job_lookup[row['job']]
            path=Path('/proc')/str(row['pid'])/'cmdline'
            if not path.exists() or not path.read_bytes():
                if job_complete(job): done.add(job['id'])
                continue
            command=path.read_bytes()
            required=[b'dynamic_service_accelerated.py',str(source).encode(),b'--split\x00dev',
                      f'--block\x00{job["block"]}'.encode(),f'--method\x00{job["method"]}'.encode()]
            if not all(token in command for token in required): raise RuntimeError('adopted process identity mismatch')
            if not set(row['cpus']).issubset(set(mixed_cpus)|set(timing_cpus)):
                raise RuntimeError('adopted process has unexpected CPU group')
            active.append(dict(job=job,process=AdoptedProcess(row['pid'],command,job),handle=None,
                               cpus=row['cpus'],loan_parent=row.get('loan_parent')))

    def latest_episode(job):
        folder=output/'training/dev'/str(job['block'])/job['method']
        return max((int(p.stem)+1 for p in folder.glob('[0-9][0-9][0-9][0-9].json')),default=0)
    def lendable(item):
        job=item['job']
        if job['command']!='train': return False
        if job['method'] in ('no_memory','no_feedback'): return True
        # At least 25 full episodes of warning before the first possible lower
        # update at 205. The controller pauses the parent if a loan persists.
        return 51<=latest_episode(job)<=180
    started=time.time()
    failed=None
    while len(done)<len(jobs):
        for item in active[:]:
            code=item['process'].poll()
            if code is not None:
                if item['handle']: item['handle'].close()
                active.remove(item)
                if code:
                    failed=item['job']['id']; break
                job=item['job']
                if job['command']=='train' and not model_done(job['block'],job['method']):
                    failed=job['id']; break
                if job['command']=='evaluate' and not job['expected'].exists():
                    failed=job['id']; break
                done.add(job['id'])
        if failed:
            atomic_json(validation/'execution/resource_failure.json',dict(failed_job=failed,
                preserved_processes=[x['process'].pid for x in active], confirmation_dispatch=False))
            raise RuntimeError(f'worker failed: {failed}; other workers preserved')
        active_ids={x['job']['id'] for x in active}
        lend_flags={x['job']['id']:lendable(x) for x in active if x['job']['command']=='train'}
        for item in active:
            if item['job']['command']!='train': continue
            loans=any(x.get('loan_parent')==item['job']['id'] for x in active)
            if loans and not lend_flags[item['job']['id']] and not item.get('paused_for_loans'):
                os.kill(item['process'].pid,signal.SIGSTOP); item['paused_for_loans']=True
            elif item.get('paused_for_loans') and not loans:
                os.kill(item['process'].pid,signal.SIGCONT); item['paused_for_loans']=False
        occupied={c for x in active for c in x['cpus']}
        free=set(mixed_cpus)-occupied
        timing_free=set(timing_cpus)-occupied
        capacity=total_capacity-len(occupied)
        complete_models={(b,m) for b in range(5) for m in ('full','independent','no_memory','no_feedback','parallel') if model_done(b,m)}
        trainer_count=sum(x['job']['command']=='train' for x in active)
        waiting_train=False
        for job in jobs:
            if job['id'] in done or job['id'] in active_ids or not all(dep in complete_models for dep in job['dependencies']):
                continue
            group=[]; loan_parent=None
            if job['command']=='train':
                need=1+workers
                if trainer_count>=args.max_trainers:
                    continue
                domains=sorted({info[c]['node'] for c in free},key=lambda n:-sum(info[c]['node']==n for c in free))
                for domain in domains:
                    local=sorted((c for c in free if info[c]['node']==domain),key=lambda c:(not info[c]['primary'],c))
                    if len(local)>=need and capacity>=need:
                        group=local[:need]; break
                if not group:
                    waiting_train=True; continue
            elif job['method']=='mpc':
                if timing_free and capacity:
                    group=[min(timing_free)]
                elif capacity>=2:
                    # Use genuinely free physical cores first, with both SMT
                    # siblings reserved against any concurrent evaluator.
                    for cpu in sorted(free):
                        siblings=info[cpu]['siblings']
                        if info[cpu]['primary'] and set(siblings).issubset(free):
                            group=list(siblings); break
                    if not group:
                        for parent in active:
                            if not lend_flags.get(parent['job']['id'],False): continue
                            for cpu in parent['cpus'][1:]:
                                siblings=set(info[cpu]['siblings'])
                                if not info[cpu]['primary']: continue
                                users=[x for x in active if set(x['cpus'])&siblings]
                                if len(users)==1 and users[0] is parent and (siblings-{cpu}).issubset(free):
                                    group=[cpu,*sorted(siblings-{cpu})]
                                    loan_parent=parent['job']['id']; break
                            if group: break
            elif not waiting_train and free and capacity:
                group=[min(free,key=lambda c:(not info[c]['primary'],c))]
            if not group:
                continue
            # Leave memory headroom for existing services and OS caches.
            memory={row.split(':')[0]:int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith(('MemAvailable:','MemTotal:'))}
            if memory['MemAvailable']<max(32*1024**3,.08*memory['MemTotal']):
                break
            record=validation/'runtime'/f'{job["id"]}.json'
            log=validation/'execution'/f'{job["id"]}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            handle=log.open('a')
            handle.write('\nRESOURCE_ACCELERATED_RESUME '+datetime.now(timezone.utc).isoformat()+'\n'); handle.flush()
            command=['taskset','-c',str(group[0]),sys.executable,str(extension/'scripts/analysis/dynamic_service_accelerated.py'),
                     '--source',str(source),'--runtime-record',str(record)]
            if job['command']=='train' and workers:
                command+=['--gradient-cpus',','.join(map(str,group[1:]))]
            command+=job['args']+['--output',str(output)]
            process=subprocess.Popen(command,cwd=source,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                                     start_new_session=True)
            active.append(dict(job=job,process=process,handle=handle,cpus=group,loan_parent=loan_parent))
            active_ids.add(job['id']); occupied.update(group)
            free.difference_update(group); timing_free.difference_update(group); capacity-=len(group)
            trainer_count+=job['command']=='train'
        atomic_json(validation/'execution/resource_status.json',dict(
            controller_pid=os.getpid(), updated=datetime.now(timezone(timedelta(hours=8))).isoformat(),
            primary_only=args.primary_only,
            stage='DEVELOPMENT_HIGH_THROUGHPUT', elapsed_seconds=time.time()-started,
            completed_jobs=len(done), total_jobs=len(jobs), reserved_logical_slots=len(occupied),
            active_trainers=sum(x['job']['command']=='train' for x in active),
            active_evaluations=sum(x['job']['command']=='evaluate' for x in active),
            active=[dict(pid=x['process'].pid,job=x['job']['id'],cpus=x['cpus'],loan_parent=x.get('loan_parent'),
                         paused_for_loans=x.get('paused_for_loans',False)) for x in active],
            confirmation_dispatch=False))
        time.sleep(2)
    analyze=[sys.executable,str(source/'scripts/analysis/dynamic_service_experiment.py'),'analyze','--split','dev','--output',str(output)]
    if args.primary_only:
        analyze.append('--primary-only')
    subprocess.run(analyze,cwd=source,check=True)
    atomic_json(validation/'execution/final_status.json',dict(
                status='DEVELOPMENT_PRIMARY_COMPLETE_CONFIRMATION_HELD' if args.primary_only else 'DEVELOPMENT_COMPLETE_CONFIRMATION_HELD',
                controller_pid=os.getpid(), primary_only=args.primary_only,
                confirmation_dispatch=False,reason='unresolved confirmation seed collision'))


if __name__=='__main__':
    main()
