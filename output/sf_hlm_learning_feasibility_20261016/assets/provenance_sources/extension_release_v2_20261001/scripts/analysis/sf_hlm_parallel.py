"""Accelerate a frozen SF-HLM run without changing source or protocol bytes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('sf_resources', ROOT/'utils/sf_hlm_resources.py')
resources = importlib.util.module_from_spec(spec); spec.loader.exec_module(resources)


def active_workers(extension, output):
    found = {}
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try: words = p.read_bytes().decode().split('\0')[:-1]
        except (OSError, UnicodeError): continue
        if len(words)<3 or words[1] != str(extension/'scripts/analysis/sf_hlm_experiment.py') or words[2]!='worker': continue
        if '--output' not in words or words[words.index('--output')+1] != str(output): continue
        flags = dict(zip(words[3::2], words[4::2]))
        key = (flags['--phase'], flags['--method'], int(flags['--block']), int(flags['--workload']), flags.get('--variant'))
        if key in found: raise RuntimeError('Duplicate existing worker')
        found[key] = int(p.parent.name)
    return found


def old_pipeline(args):
    if sys.platform!='linux': raise RuntimeError('Remote Linux only')
    resources.initialize(args.resources)
    with resources.task_lock(args.resources/'legacy_controller.lock'):
        identities = active_workers(args.extension_root, args.output)
        old_args = Path(f'/proc/{args.controller_pid}/cmdline')
        if old_args.exists():
            words = old_args.read_bytes().decode().split('\0')
            entry = (Path(f'/proc/{args.controller_pid}/cwd').resolve()/words[1]).resolve() if len(words)>1 else None
            original=len(words)>=3 and entry==args.extension_root/'scripts/analysis/sf_hlm_experiment.py' and words[2]=='pipeline'
            previous=len(words)>=3 and entry==Path(__file__).resolve() and words[2]=='legacy' and '--output' in words and words[words.index('--output')+1]==str(args.output)
            if not (original or previous):
                raise RuntimeError('Refusing to signal an unrelated controller')
            os.kill(args.controller_pid, signal.SIGTERM)
            for _ in range(100):
                if resources.process_identity(args.controller_pid) is None: break
                time.sleep(.05)
            else: raise RuntimeError('Original controller did not stop dispatching')
        identities = active_workers(args.extension_root, args.output)
        adopted = {key: resources.CpuLease(args.resources, owner=pid).acquire() for key,pid in identities.items()}
        resources.atomic_json(args.resources/'legacy_takeover.json', dict(unix=time.time(), old_controller=args.controller_pid,
            adopted=[dict(job=list(k),pid=identities[k],cpu=v.cpu) for k,v in adopted.items()], source_and_protocol_unchanged=True))
        sys.path.insert(0,str(args.source_root))
        import numpy as np
        spec = importlib.util.spec_from_file_location('frozen_sf_runner', args.extension_root/'scripts/analysis/sf_hlm_experiment.py')
        base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base); base.np=np
        original_hash = base.sha(args.output/'protocol.json')
        stop = threading.Event(); thread=threading.Thread(target=resources.telemetry,args=(args.resources,stop),daemon=True); thread.start()
        def run_jobs(old, protocol, phases, blocks=None, variant=None):
            selected=[j for j in protocol['jobs'] if j[0] in phases and (blocks is None or j[2] in blocks)]
            completed=[]; failures=[]
            def one(job):
                phase,method,block,workload=job; key=(*job,variant)
                if key in adopted:
                    lease=adopted.pop(key); pid=identities[key]
                    while resources.process_identity(pid) is not None: time.sleep(1)
                    lease.release()
                    root=old.output/'robustness_variants'/variant if variant else old.output
                    if not (base.result_path(root,phase,method,block,workload)/'result.json').exists():
                        raise RuntimeError('Adopted worker exited without its result')
                # Always invoke the frozen entrypoint: it verifies completed
                # outputs or restores unfinished work with the original contract.
                cmd=[sys.executable,str(old.extension_root/'scripts/analysis/sf_hlm_experiment.py'),'worker']
                for flag,value in [('source-root',old.source_root),('suite-root',old.suite_root),('extension-root',old.extension_root),
                    ('output',old.output),('phase',phase),('method',method),('block',block),('workload',workload),('seed',protocol['lower_seeds'][str(block)])]:
                    cmd += ['--'+flag,str(value)]
                if variant: cmd += ['--variant',variant]
                log=args.resources/'legacy_logs'/f'{phase}_{method}_{block}_{workload}_{variant or "main"}.log'; log.parent.mkdir(parents=True,exist_ok=True)
                with resources.CpuLease(args.resources,pin=False,task=list(key)) as lease,log.open('a') as f:
                    cmd=[sys.executable,str(Path(__file__).resolve()),'exec','--cpu',str(lease.cpu),'--',*cmd]
                    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=old.source_root,
                        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                    lease.adopt_owner(child.pid)
                    if child.wait():raise RuntimeError('Frozen worker failed: '+str(job))
                return job
            with ThreadPoolExecutor(max_workers=240) as pool:
                fs={pool.submit(one,j):j for j in selected}
                for f in as_completed(fs):
                    try: completed.append(f.result())
                    except Exception as exc: failures.append(dict(job=fs[f],error=str(exc)))
                    base.write(old.output/'status.json',dict(phase=','.join(phases),variant=variant,pid=os.getpid(),
                        completed=len(completed),expected=len(selected),failures=failures,updated_unix=time.time(),execution_manifest=str(args.resources)))
            if failures: raise RuntimeError(str(failures))
        base.run_jobs=run_jobs
        try:
            # workers=5 is scientific provenance in the frozen protocol. Actual
            # resource concurrency is deliberately recorded separately.
            args.workers=5; args.smoke=False; args.variant=None
            base.pipeline(args)
            if base.sha(args.output/'protocol.json')!=original_hash: raise RuntimeError('Frozen protocol changed')
        finally:
            stop.set(); thread.join(timeout=5)
            for lease in adopted.values(): lease.release()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='exec':
        boundary=sys.argv.index('--'); cpu=int(sys.argv[sys.argv.index('--cpu')+1]); cmd=sys.argv[boundary+1:]
        os.sched_setaffinity(0,{cpu}); os.execv(cmd[0],cmd)
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['legacy'])
    for name in ('source-root','suite-root','extension-root','output','resources'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--controller-pid',type=int,required=True); args=p.parse_args()
    for key in ('source_root','suite_root','extension_root','output','resources'):setattr(args,key,getattr(args,key).expanduser().resolve())
    old_pipeline(args)


if __name__=='__main__':main()
