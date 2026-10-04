"""External scheduler: frozen canonical training in parallel, then serial evaluation.

Does not alter source, seeds, per-process thread count or checkpoint contracts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--plan',default='docs/plans/batch_offload_pilot_20260928.json')
    parser.add_argument('--workers',type=int,default=20)
    args=parser.parse_args()
    source=args.source.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source))
    from utils.canonical_experiment import source_manifest,write_json
    from utils.formal_protocol import verify_completed_job
    from scripts.analysis.batch_offload_protocol import validate_plan
    from run_formal_suite import groups
    manifest=source_manifest();plan=json.loads((source/args.plan).read_text());validate_plan(plan)
    if not 1<=args.workers<=min(20,os.cpu_count() or 1):raise ValueError('Invalid worker count')
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if args.workers*8*1024**2>memory['MemAvailable']:raise RuntimeError('Reserve 8 GiB per worker')
    identity={'plan':plan,'source_hash':manifest['source_hash'],'threads':1,
              'git_sha':manifest.get('git_sha'),'git_clean':manifest.get('git_clean')}
    if (output/'suite.json').exists():
        if json.loads((output/'suite.json').read_text())!=identity:raise ValueError('Existing suite identity changed')
    else:write_json(output/'suite.json',identity)
    def record(phase,**extra):
        write_json(args.state,{'phase':phase,'pid':os.getpid(),'source_hash':manifest['source_hash'],
                              'output':str(output),'time_unix':time.time(),'training_workers':args.workers,
                              'threads_per_training':1,'evaluation_workers':1,'confirmation_started':False,**extra})
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def complete(directory,attempt):
        files=sorted((attempt/'episodes').glob('*.json'))
        if len(files)!=plan['train_episodes']:raise ValueError('Incomplete training episode set')
        for index,path in enumerate(files):
            episode=json.loads(path.read_text())
            if episode['episode']!=index or len(episode['steps'])!=plan['steps']:raise ValueError('Incomplete training episode')
        latest=json.loads((attempt/'latest.json').read_text())
        if latest['completed_episodes']!=plan['train_episodes']:raise ValueError('Incomplete checkpoint')
        checkpoint=attempt/latest['checkpoint'];contract=json.loads((checkpoint/'contract.json').read_text())
        if contract['source_hash']!=manifest['source_hash']:raise ValueError('Checkpoint source changed')
        for relative,expected in contract['files'].items():
            if hashlib.sha256((checkpoint/relative).read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint checksum mismatch')
        hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in attempt.rglob('*') if p.is_file()}
        write_json(directory/'completed.json',{'status':'PASS','attempt':attempt.name,'hashes':hashes})
        write_json(directory/'state.json',{'status':'PASS'})
    def train(item):
        variant,method,seed=item
        directory=output/variant['name']/method/f'seed_{seed}'/'train';directory.mkdir(parents=True,exist_ok=True)
        if (directory/'completed.json').exists():
            verify_completed_job(directory,manifest['source_hash']);return
        attempts=sorted(directory.glob('attempt_*'));resume=False
        if attempts and (attempts[-1]/'latest.json').exists():
            attempt=attempts[-1];resume=True
            if json.loads((attempt/'latest.json').read_text())['completed_episodes']>=plan['train_episodes']:
                complete(directory,attempt);return
        else:attempt=directory/f'attempt_{len(attempts):04d}'
        workload=plan['training_workload_seeds'][str(seed)]
        command=[sys.executable,str(source/'run_canonical_experiment.py'),'--method',method,'--training-seed',str(seed),
                 '--workload-seed',str(workload),'--episodes',str(plan['train_episodes']),'--steps',str(plan['steps']),
                 '--rollout',str(plan['rollout']),'--ppo-epochs',str(plan['ppo_epochs']),'--batch-size',str(plan['batch_size']),
                 '--threads','1','--checkpoint-every',str(plan['checkpoint_every']),'--ue-count',str(variant['ue_count']),
                 '--compute-scale',str(variant['compute_scale']),'--purpose',plan['purpose'],'--output',str(attempt)]
        if resume:command.append('--resume')
        write_json(directory/'state.json',{'status':'RUNNING','command':command,'started':time.time(),'scheduler':'parallel_training_serial_evaluation_v1'})
        with (directory/'worker.log').open('a') as log:
            result=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(directory/'state.json',{'status':'FAILED','returncode':result.returncode,'command':command})
            raise RuntimeError('Training failed: '+str(directory))
        complete(directory,attempt)
    with (output/'.parallel-controller.lock').open('a') as ownership:
        fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with (output/'.controller.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                record('PILOT_PARALLEL_TRAINING')
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(train,groups(plan)))
            record('PILOT_SERIAL_EVALUATION')
            subprocess.run([sys.executable,str(source/'run_formal_suite.py'),'--plan-file',str(source/args.plan),
                            '--output',str(output),'--workers','1','--threads','1'],cwd=source,env=env,check=True)
            if plan['stage']=='smoke':
                record('VERIFICATION_COMPLETE');return
            record('PILOT_ANALYZING')
            subprocess.run([sys.executable,'-m','scripts.analysis.batch_offload_analysis',str(output)],cwd=source,env=env,check=True)
            report=json.loads((output/'batch_offload_analysis.json').read_text())
            record('PILOT_COMPLETE',selected_for_confirmation=report['selected_for_confirmation'])
        except Exception as error:
            record('FAILED',error=str(error));raise


if __name__=='__main__':main()
