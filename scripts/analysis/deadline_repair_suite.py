"""Launch only fixed-checkpoint snapshot diagnostics against immutable source."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--suite-root',type=Path,required=True)
    p.add_argument('--extension-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--smoke',action='store_true');p.add_argument('--run',action='store_true')
    args=p.parse_args()
    if not 1<=args.workers<=5:raise ValueError('Use at most five independent workers')
    suite=json.loads((args.suite_root/'suite.json').read_text());plan=suite['plan']
    methods=['lower_mlp_upper_uncovered_obs','vanilla_lower_mlp_upper_uncovered_obs']
    jobs=[(m,s,w) for m in methods for s in plan['training_seeds'] for w in plan['workload_seeds']]
    if args.smoke:jobs=[(m,plan['training_seeds'][0],plan['workload_seeds'][0]) for m in methods]
    script=args.extension_root/'scripts/analysis/batch_offload_diagnostic.py'
    module=args.extension_root/'utils/deadline_repair.py'
    protocol={'version':'deadline_repair_snapshot_v1','purpose':'verification' if args.smoke else 'development_diagnostic',
              'source_hash':suite['source_hash'],'suite_sha256':sha(args.suite_root/'suite.json'),
              'driver_sha256':sha(script),'repair_sha256':sha(module),'controller_sha256':sha(__file__),
              'jobs':jobs,'episodes':1 if args.smoke else 2,'steps':100 if args.smoke else plan['steps'],
              'snapshot_steps':[0] if args.smoke else [0,250,500,750], 'budgets':[32,128],
              'energy_ratio':1.05,'threads':1,'workers':args.workers,
              'quality_mask':'unchanged','information':'centralized current proposals, predecision cache beliefs',
              'timing':'includes preparation and ordering; concurrent diagnostic, NOT isolated latency benchmark',
              'selection':'no promotion to training; all results retained; no automatic closed-loop or training launch'}
    if not args.run:print(json.dumps(protocol));return
    if platform.system()!='Linux':raise RuntimeError('Environment evaluation is remote Linux only')
    args.output.mkdir(parents=True,exist_ok=True)
    identity=args.output/'protocol.json'
    if identity.exists():
        if json.loads(identity.read_text())!=json.loads(json.dumps(protocol)):raise ValueError('Protocol differs')
    else:write(identity,protocol)
    completed=[];failed=[]
    def state(phase):
        write(args.output/'status.json',{'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),
              'expected_jobs':len(jobs),'completed_jobs':len(completed),'failed_jobs':failed,
              'training_started':False,'closed_loop_started':False,'source_hash':suite['source_hash']})
    def run(job):
        method,seed,workload=job;out=args.output/method/f'seed_{seed}_workload_{workload}.json'
        if out.exists():
            old=json.loads(out.read_text())
            expected={'status':'PASS','source_hash':suite['source_hash'],'script_sha256':sha(script),
                      'repair_module_sha256':sha(module),'reference_method':method,
                      'training_seed':seed,'workload_seed':workload,
                      'verification_scope':'prefix_smoke' if args.smoke else 'full_episodes'}
            if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('Existing job differs')
            if len(old['snapshots'])!=protocol['episodes']*len(protocol['snapshot_steps']):raise ValueError('Incomplete snapshots')
            return out
        out.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(script),'--source-root',str(args.source_root),'--suite-root',str(args.suite_root),
             '--output',str(out),'--training-seed',str(seed),'--workload-seed',str(workload),
             '--reference-method',method,'--episodes',str(protocol['episodes']),'--threads','1',
             '--repair-module',str(module)]
        if args.smoke:cmd+=['--smoke-steps','100']
        child_env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        with out.with_suffix('.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=args.source_root,env=child_env)
        return out
    state('RUNNING')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,j):j for j in jobs}
        for future in as_completed(futures):
            try:completed.append(future.result())
            except Exception as exc:failed.append({'job':futures[future],'error':str(exc)})
            state('RUNNING');print(json.dumps({'completed':len(completed),'failed':len(failed),'total':len(jobs)}),flush=True)
    if failed:state('FAILED');raise RuntimeError('Some jobs failed; logs retained')
    groups=defaultdict(lambda:defaultdict(list));timing=defaultdict(list);total=empty=0
    for path in sorted(completed):
        doc=json.loads(path.read_text())
        for snap in doc['snapshots']:
            total+=1;empty+=int(snap['request_count']==0)
            result=snap['bridge']['deadline_repair']
            for name,item in result['comparisons'].items():
                key=doc['reference_method']+'/'+name
                groups[key][doc['training_seed']].append(item['actual_miss_difference'])
                timing[key].append(item['proposal_path_seconds'])
    comparisons={}
    for key,seeds in groups.items():
        values=[statistics.mean(v) for _,v in sorted(seeds.items())]
        mean=statistics.mean(values)
        half=2.7764451051977987*statistics.stdev(values)/math.sqrt(5) if len(values)==5 else None
        times=sorted(timing[key])
        comparisons[key]={'mean_miss_difference_per_snapshot':mean,'training_seed_means':values,
                          'improved_seeds':sum(v<0 for v in values),
                          'training_seed_t_ci95':[mean-half,mean+half] if half is not None else None,
                          'diagnostic_proposal_p95_seconds':times[max(0,math.ceil(.95*len(times))-1)]}
    write(args.output/'summary.json',{'status':'PASS','snapshots':total,'empty_snapshots':empty,
          'completed_jobs':len(completed),'comparisons':comparisons,
          'limitations':['Fixed-snapshot development evidence, not online DSR or independent confirmation.',
                        'n=5 training seeds; snapshots/workloads are not independent replicates.',
                        'Concurrent timing excludes signaling and is not a realtime deployment guarantee.',
                        'Energy, MBS and prediction errors remain in each raw snapshot for complete review.']})
    state('COMPLETE')


if __name__=='__main__':main()
