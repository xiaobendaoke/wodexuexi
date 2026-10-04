"""Bounded frozen-source replay controller and descriptive summary, no training."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from paths import REPO_ROOT
from scripts.analysis.batch_offload_protocol import write_new


def summarize(paths):
    groups = defaultdict(lambda: {'requests':0,'misses':0,'target_load_sum':0,'latency_error_sum':0.,'unclipped_noncoop_estimates':0})
    truncation = Counter(); jobs=[]; tails={}; seen=set(); source_hashes=set(); script_hashes=set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        if data['status'] != 'PASS': raise ValueError('Unverified diagnostic job')
        key = data['training_seed'],data['workload_seed']
        if key in seen: raise ValueError('Repeated diagnostic job')
        seen.add(key); source_hashes.add(data['source_hash']); script_hashes.add(data['script_sha256'])
        truncation.update(data['observation_truncation']); tails[data['training_seed']]=data['training_tail']
        empty=0
        for snap in data['snapshots']:
            empty += int(snap['request_count']==0)
            for row in snap['requests']:
                n=row['source_request_count']; band='1-5' if n<=5 else '6-10' if n<=10 else '11-20' if n<=20 else '21+'
                group=groups[band];group['requests']+=1;group['misses']+=int(not row['success']);group['target_load_sum']+=row['final_target_count']
                if row['estimate_role']!='best_coop_proxy' and not row['estimate_clipped']:
                    group['latency_error_sum']+=row['actual_minus_estimate']; group['unclipped_noncoop_estimates']+=1
                group['same_source_shared_requests']=group.get('same_source_shared_requests',0)+int(row['same_source_target_count']>1)
                group['cross_source_shared_requests']=group.get('cross_source_shared_requests',0)+int(row['target_source_count']>1)
        jobs.append({'training_seed':key[0],'workload_seed':key[1],'snapshots':len(data['snapshots']),'empty_snapshots':empty})
    if len(source_hashes)!=1 or len(script_hashes)!=1: raise ValueError('Mixed diagnostic protocols')
    for group in groups.values():
        group['miss_fraction']=group['misses']/group['requests']
        group['mean_target_count']=group['target_load_sum']/group['requests']
        group['mean_actual_minus_estimate']=group['latency_error_sum']/group['unclipped_noncoop_estimates'] if group['unclipped_noncoop_estimates'] else None
    return {'status':'PASS','source_hash':next(iter(source_hashes)),'diagnostic_script_hash':next(iter(script_hashes)),
            'jobs':jobs,'predecision_source_load_groups':dict(groups),'observation_truncation':dict(truncation),
            'training_tail_by_seed':tails,'inference':'Descriptive only; request groups are not independent training repeats.',
            'sharing_definition':'Request experiences multiple tasks at target; same-source and cross-source categories can overlap.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--suite-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--run',action='store_true');args=p.parse_args()
    suite=json.loads((args.suite_root/'suite.json').read_text());plan=suite['plan']
    jobs=[(s,w) for s in plan['training_seeds'] for w in plan['workload_seeds']]
    script=REPO_ROOT/'scripts/analysis/batch_offload_diagnostic.py'
    identity={'source_hash':suite['source_hash'],'diagnostic_script_hash':hashlib.sha256(script.read_bytes()).hexdigest(),
              'episodes':2,'snapshot_steps':[0,250,500,750],'jobs':jobs,'threads':1,'workers':args.workers}
    if not args.run:
        print(json.dumps(identity));return
    if platform.system()!='Linux' or not 1<=args.workers<=4:
        raise ValueError('Run remotely with 1 to 4 workers')
    args.output.mkdir(parents=True,exist_ok=True);write_new(args.output/'protocol.json',identity)
    def run(job):
        seed,workload=job;path=args.output/f'seed_{seed}_workload_{workload}.json'
        if path.exists():
            existing=json.loads(path.read_text())
            if (existing.get('status')!='PASS' or existing.get('source_hash')!=suite['source_hash'] or
                existing.get('script_sha256')!=identity['diagnostic_script_hash'] or len(existing.get('replay',[]))!=2 or
                existing.get('reference_method')!='lower_mlp_upper_uncovered_obs' or existing.get('compare_method') is not None or existing.get('mc_itr')):
                raise ValueError('Existing diagnostic identity differs')
            return path
        command=[sys.executable,str(script),'--source-root',str(args.source_root.resolve()),'--suite-root',str(args.suite_root.resolve()),
                 '--output',str(path.resolve()),'--training-seed',str(seed),'--workload-seed',str(workload),'--threads','1']
        with path.with_suffix('.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=args.source_root)
        return path
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths=list(pool.map(run,jobs))
    report=summarize(paths);write_new(args.output/'summary.json',report)
    print(json.dumps({'status':'PASS','jobs':len(paths),'output':str(args.output)}))


if __name__=='__main__': main()
