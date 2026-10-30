"""Run bounded isolated-layer development probes for a tested frozen release."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK='chapter5_dynamic_service_20261028'
REVISION='revision_v2_20261030'


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(tmp,path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--workers',type=int,default=20)
    args=parser.parse_args();source=args.source.resolve();task=args.task.resolve()
    if sys.platform!='linux' or not 1<=args.workers<=24 or not 1<=args.episodes<=24:parser.error('bounded server pilot only')
    sys.path.insert(0,str(source))
    from scripts.analysis.dynamic_service_v2_experiment import protocol
    validation=task/'output'/TASK/'validation'/REVISION;output=task/'results'/TASK/REVISION
    validation.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(validation/'pilot_controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tests=json.loads((validation/'release_tests.json').read_text());contract=hashlib.sha256(json.dumps(protocol(),sort_keys=True).encode()).hexdigest()
    if tests['status']!='PASS' or tests['protocol']!=contract or Path(tests['release'])!=source:
        raise RuntimeError('pilot release differs from tested source')
    clearance=json.loads((validation.parent/'development_seed_clearance.json').read_text())
    if clearance['status']!='PASS_DEVELOPMENT_ONLY' or clearance['development_collisions']:
        raise RuntimeError('development tapes not cleared')
    protected_file=validation/'protected_before.json'
    if protected_file.exists():protected=json.loads(protected_file.read_text())
    else:
        base=task/'results'/TASK
        files=[*base.glob('models/dev/*/*.pt'),*base.glob('evaluation/dev/*/*/*.json'),base/'analysis/dev.json',base/'development_gate.json']
        protected={str(p.relative_to(task)):sha(p) for p in files};atomic(protected_file,protected)
    seen=set();cpus=[]
    for cpu in sorted(os.sched_getaffinity(0)):
        path=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((path/'physical_package_id').read_text(),(path/'core_id').read_text())
        if key not in seen:seen.add(key);cpus.append(cpu)
    cpus=cpus[8:8+args.workers]
    entry=source/'scripts/analysis/dynamic_service_v2_experiment.py'
    base=[sys.executable,str(entry),'--output',str(output),'--split','pilot','--episodes',str(args.episodes)]
    # argparse accepts options before the positional subcommand.
    jobs=[]
    for phase in ('upper_probe','lower_probe'):
        for block in range(5):
            ident=f'train_{phase}_{block}'
            jobs.append(dict(id=ident,command='train',phase=phase,block=block,dependencies=[],
                args=['train','--phase',phase,'--block',str(block)],
                expected=output/'training/pilot'/str(block)/phase/f'{args.episodes-1:04}.json'))
            for workload in range(3):
                jobs.append(dict(id=f'eval_{phase}_{block}_{workload}',command='evaluate',phase=phase,block=block,
                    dependencies=[ident],args=['evaluate','--phase',phase,'--block',str(block),'--workload',str(workload)],
                    expected=output/'evaluation/pilot'/str(block)/phase/f'learned_D_1.0_{workload}_0.json'))
    for control in ('history','hover'):
        for workload in range(3):
            jobs.append(dict(id=f'eval_{control}_{workload}',command='evaluate',phase='upper_probe',block=0,dependencies=[],
                args=['evaluate','--phase','upper_probe','--control',control,'--workload',str(workload)],
                expected=output/'evaluation/pilot/0/upper_probe'/f'{control}_D_1.0_{workload}_0.json'))
    manifest=dict(started=datetime.now(timezone(timedelta(hours=8))).isoformat(),controller_pid=os.getpid(),
        source=str(source),contract=contract,controller_sha256=sha(Path(__file__)),episodes=args.episodes,horizon=1000,
        jobs=[j['id'] for j in jobs],cpus=cpus,development_only=True,confirmation=False,
        purpose='isolated upper with fixed matching; isolated lower with fixed public-history trajectory; not full-system evidence')
    atomic(validation/'pilot_launch.json',manifest)
    subprocess.run(base+['protocol'],cwd=source,check=True)
    done=set();active=[];started=time.time()
    for j in jobs:
        if j['expected'].exists():
            d=json.loads(j['expected'].read_text())
            if d['contract']!=contract:raise RuntimeError('existing pilot output belongs to another release')
            done.add(j['id'])
    while len(done)<len(jobs):
        for item in active[:]:
            code=item['process'].poll()
            if code is not None:
                item['log'].close();active.remove(item)
                if code or not item['job']['expected'].exists():raise RuntimeError(f'pilot job failed: {item["job"]["id"]}')
                done.add(item['job']['id'])
        active_ids={x['job']['id'] for x in active};free=[c for c in cpus if c not in {x['cpu'] for x in active}]
        for job in jobs:
            if not free:break
            if job['id'] in done or job['id'] in active_ids or not all(d in done for d in job['dependencies']):continue
            cpu=free.pop(0);log=(validation/(job['id']+'.log')).open('a')
            process=subprocess.Popen(['taskset','-c',str(cpu)]+base+job['args'],cwd=source,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(dict(process=process,log=log,job=job,cpu=cpu))
        atomic(validation/'pilot_status.json',dict(updated=datetime.now(timezone(timedelta(hours=8))).isoformat(),
            elapsed_seconds=time.time()-started,completed=len(done),total=len(jobs),active=[dict(pid=x['process'].pid,job=x['job']['id']) for x in active],
            confirmation=False,experiment_complete=False))
        time.sleep(2)
    # Fixed-snapshot lower check, kept distinct from closed-loop evaluation.
    import torch
    import numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from environment.dynamic_service import Batch,evaluate
    from marl_models.dynamic_service_hmarl_v2 import LowerPolicyV2
    frozen=[]
    for block in range(5):
        saved=torch.load(output/'models/pilot'/str(block)/'lower_probe.pt',map_location='cpu',weights_only=False)
        policy=LowerPolicyV2();policy.load_state_dict(saved['lower']);policy.eval()
        details=[]
        for f in sorted((task/'results'/TASK/'evaluation/dev'/str(block)/'full').glob('D_1.0_*.json')):
            for snap in json.loads(f.read_text()).get('snapshots',[]):
                batch=Batch(**{k:np.asarray(v) for k,v in snap['batch'].items()})
                assignment=policy.act(batch)['assignment'];result=evaluate(batch,assignment)
                details.append(dict(requests=len(batch),v1_success=snap['controls']['full']['success'],
                    v2_success=result['success'],matching_success=snap['controls']['matching']['success'],
                    mbs=result['mbs'],mbs_failed=int(((assignment==5)&~result['satisfied']).sum())))
        frozen.append(dict(block=block,snapshots=len(details),requests=sum(x['requests'] for x in details),
            **{k:sum(x[k] for x in details) for k in ('v1_success','v2_success','matching_success','mbs','mbs_failed')},details=details))
    groups=defaultdict(list);files={}
    for j in jobs:
        if j['command']!='evaluate':continue
        data=json.loads(j['expected'].read_text());rows=data['rows'];t=data['totals']
        if len(rows)!=1000 or t['generated']!=50000 or data['contract']!=contract:raise RuntimeError('incomplete pilot evaluation')
        for k,v in t.items():
            if sum(x[k] for x in rows)!=v:raise RuntimeError('pilot counter mismatch')
        key=data['phase'] if data['control']=='learned' else data['control']
        groups[key].append(dict(block=data['block'],dsr=data['dsr'],admission=data['admission'],
            boundary=t['boundary_uav_steps']/5000,edge60=sum(x['edge60_uav_steps'] for x in rows)/5000,
            hover=t['hovering_uav_steps']/5000,decision_timeout=t['decision_timeout']))
        files[str(j['expected'].relative_to(output))]=sha(j['expected'])
    averaged={key:dict(cases=len(ds),**{k:float(np.mean([d[k] for d in ds])) for k in ('dsr','admission','boundary','edge60','hover')},
        decision_timeouts=sum(x['decision_timeout'] for x in ds),by_block={str(b):float(np.mean([x['dsr'] for x in ds if x['block']==b])) for b in sorted({x['block'] for x in ds})}) for key,ds in groups.items()}
    changed=[name for name,h in protected.items() if sha(task/name)!=h]
    if changed:raise RuntimeError(f'original scientific artifacts changed: {changed[:3]}')
    atomic(output/'pilot_analysis.json',dict(status='COMPLETE_BOUNDED_PILOT',contract=contract,manifest=manifest,
        groups=averaged,frozen_snapshots=frozen,file_hashes=files,protected_artifacts=len(protected),protected_changed=changed,
        scientific_performance_claim=False,confirmation=False,limitations=['12-episode development probe, not a matched 400-episode comparison',
        'same development tapes reused; no independent confirmation','upper/lower probes are isolated, not an end-to-end trained system',
        'boundary metric also accompanied by 60m edge band and service rates to avoid treating tiny inward moves as success']))
    atomic(validation/'pilot_status.json',dict(status='COMPLETE_BOUNDED_PILOT',completed=len(jobs),total=len(jobs),active=[],confirmation=False,
        analysis_sha256=sha(output/'pilot_analysis.json'),full_experiment_complete=False))


if __name__=='__main__':main()
