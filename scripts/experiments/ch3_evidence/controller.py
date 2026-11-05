"""Fixed-matrix server controller with historical gate, frozen release and sequential testing."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from utils.canonical_experiment import source_manifest, write_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capacity_widths():
    import torch
    torch.set_num_threads(1)
    from marl_models.attention_mappo.agents import ActorNetwork,CriticNetwork
    from marl_models.vanilla_mappo.agents import VanillaActorNetwork,VanillaCriticNetwork
    import config
    counts={'actor':sum(p.numel() for p in ActorNetwork(325,2).parameters()),
            'critic':sum(p.numel() for p in CriticNetwork(325).parameters())}
    result={}
    for kind,input_size,output_size in [('actor',325,2),('critic',1625,5)]:
        def count(w):return input_size*w+w*w+(6+output_size)*w+output_size+(2 if kind=='actor' else 0)
        width=min(range(1,4097),key=lambda w:(abs(count(w)-counts[kind]),w))
        setattr(config,'UPPER_MLP_'+kind.upper()+'_WIDTH',width)
        model=VanillaActorNetwork(325,2) if kind=='actor' else VanillaCriticNetwork(1625,5)
        actual=sum(p.numel() for p in model.parameters())
        if actual!=count(width):raise AssertionError('Parameter count formula mismatch')
        result[kind]=dict(width=width,attention_parameters=counts[kind],matched_parameters=actual,
                          relative_mismatch=abs(actual-counts[kind])/counts[kind])
    return result


def command(plan,method,seed,workload,output,checkpoint=None,smoke=False,setting=None,widths=None):
    cmd=[sys.executable,'run_canonical_experiment.py','--method',method,'--training-seed',str(seed),
         '--workload-seed',str(workload),'--episodes',str(2 if smoke else plan['eval_episodes'] if checkpoint else plan['train_episodes']),
         '--steps',str(8 if smoke else plan['steps']),'--rollout',str(8 if smoke else plan['rollout']),
         '--ppo-epochs',str(1 if smoke else plan['ppo_epochs']),'--batch-size',str(8 if smoke else plan['batch_size']),
         '--threads',str(plan['threads']),'--checkpoint-every',str(plan['checkpoint_every']),
         '--purpose','verification' if smoke else 'pilot','--output',str(output)]
    if checkpoint:
        cmd+=['--evaluate-checkpoint',str(checkpoint),'--ch3-record-requests']
    if widths:
        cmd+=['--ch3-actor-width',str(widths['actor']['width']),'--ch3-critic-width',str(widths['critic']['width'])]
    if setting:
        cmd+=['--ch3-boundary','--episodes',str(plan['boundary']['eval_episodes'])]
        for key,flag in [('ue_count','--ue-count'),('compute_scale','--compute-scale'),('demand_radius','--ch3-demand-radius'),('delay','--ch3-delay'),('drop','--ch3-drop')]:
            if key in setting:cmd += [flag,str(setting[key])]
        if setting.get('uniform'):cmd+=['--ch3-uniform']
    return cmd


def run_job(command,output,lock):
    output=Path(output)
    if source_manifest()['source_hash']!=lock['source_hash']:raise RuntimeError('Frozen source changed')
    if (output/'completed.json').exists():
        from ch3_analysis import verify_job
        episode_count=int(command[command.index('--episodes')+1]);step_count=int(command[command.index('--steps')+1])
        # Boundary overrides are last; argparse uses the last occurrence.
        episode_count=int(command[len(command)-1-command[::-1].index('--episodes')+1])
        verify_job(output,lock['source_hash'],episode_count,step_count)
        return str(output)
    if output.exists() and any(output.iterdir()):raise ValueError(f'Partial job requires explicit diagnosis: {output}')
    output.mkdir(parents=True,exist_ok=True)
    log=output.parent/(output.name+'.log')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with log.open('w') as handle:
        process=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=env)
    if process.returncode:
        write_json(output/'failure.json',dict(command=command,returncode=process.returncode,log=str(log)))
        if 'INTERFACE_NOT_APPLICABLE' in log.read_text():return {'not_applicable':str(output)}
        raise RuntimeError(f'Job failed: {output}; {log.read_text()[-1800:]}')
    hashes={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'completed.json',dict(status='PASS',source_hash=lock['source_hash'],command=command,hashes=hashes))
    return str(output)


def batch(jobs,plan,root,lock,stage):
    state=dict(stage=stage,status='RUNNING',expected=len(jobs),completed=0,started=time.time(),failures=[])
    write_json(root/'status.json',state)
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures={pool.submit(run_job,cmd,out,lock):str(out) for cmd,out in jobs}
        try:
            for future in as_completed(futures):
                value=future.result();state['completed']+=1
                if isinstance(value,dict):state.setdefault('not_applicable',[]).append(value)
                state['updated']=time.time();write_json(root/'status.json',state)
                print(json.dumps(dict(stage=stage,completed=state['completed'],expected=len(jobs),job=value)),flush=True)
        except Exception as exc:
            for f in futures:f.cancel()
            state['status']='FAILED';state['failures'].append(str(exc));write_json(root/'status.json',state)
            raise
    state['status']='PASS';write_json(root/'status.json',state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--mode',choices=['preflight','run'],required=True);args=parser.parse_args()
    if platform.system()!='Linux':raise RuntimeError('Scientific execution belongs on the remote Linux server')
    plan=json.loads(args.protocol.read_text());root=args.root.resolve();validation=args.validation.resolve()
    root.mkdir(parents=True,exist_ok=True);validation.mkdir(parents=True,exist_ok=True)
    if args.mode=='preflight':
        if json.loads((validation/'historical_gate.json').read_text())['status']!='PASS':raise ValueError('Historical gate failed')
        if json.loads((validation/'seed_audit.json').read_text())['status']!='PASS':raise ValueError('Seed audit failed')
        if (validation/'release.json').exists():raise ValueError('Refuse to overwrite frozen release')
        result=subprocess.run([sys.executable,'-m','unittest','ch3_remote_tests','-v'],capture_output=True,text=True)
        (validation/'remote_tests.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        widths=capacity_widths();write_json(validation/'capacity_widths.json',widths)
        lock=dict(source_hash=source_manifest()['source_hash'],source_files=source_manifest()['files'],
                  protocol_sha256=sha(args.protocol),widths=widths,status='PREFLIGHT',runtime_python=sys.version)
        jobs=[]
        for index,method in enumerate(list(plan['cells'].values())+[plan['capacity_method']]):
            seed=plan['smoke_training_seeds'][index];out=root/'smoke'/method/'train'
            jobs.append((command(plan,method,seed,plan['smoke_workload_seed'],out,smoke=True,widths=widths if index==4 else None),out))
        batch(jobs,plan,root,lock,'smoke_training')
        jobs=[]
        for index,method in enumerate(list(plan['cells'].values())+[plan['capacity_method']]):
            out=root/'smoke'/method/'evaluation';checkpoint=root/'smoke'/method/'train/checkpoints/episode_00001'
            jobs.append((command(plan,method,plan['smoke_training_seeds'][index],plan['smoke_workload_seed'],out,checkpoint,smoke=True,widths=widths if index==4 else None),out))
        batch(jobs,plan,root,lock,'smoke_evaluation')
        jobs=[]
        for setting in plan['boundary']['settings']:
            for cell in ('A','B'):
                method=plan['cells'][cell];index=list(plan['cells']).index(cell)
                checkpoint=root/'smoke'/method/'train/checkpoints/episode_00001'
                out=root/'smoke_boundary'/setting['name']/method
                jobs.append((command(plan,method,plan['smoke_training_seeds'][index],plan['smoke_workload_seed'],out,checkpoint,smoke=True,setting=setting),out))
        batch(jobs,plan,root,lock,'smoke_boundary')
        lock['status']='PASS';lock['preflight_completed']=time.time();write_json(validation/'release.json',lock)
        print('PREFLIGHT PASS',flush=True);return
    lock=json.loads((validation/'release.json').read_text())
    if lock['status']!='PASS' or sha(args.protocol)!=lock['protocol_sha256'] or source_manifest()['source_hash']!=lock['source_hash']:
        raise ValueError('Release or protocol changed after preflight')
    write_json(root/'suite.json',dict(plan=plan,source_hash=lock['source_hash'],protocol_sha256=lock['protocol_sha256']))
    methods=list(plan['cells'].values())
    def training_jobs(items):
        return [(command(plan,m,s,plan['training_workload_seed'],root/'main'/m/f'seed_{s}'/'train',widths=lock['widths'] if m==plan['capacity_method'] else None),root/'main'/m/f'seed_{s}'/'train') for m in items for s in plan['training_seeds']]
    def evaluation_jobs(items):
        jobs=[]
        for m in items:
            for s in plan['training_seeds']:
                base=root/'main'/m/f'seed_{s}';checkpoint=base/'train/checkpoints'/plan['final_checkpoint']
                for w in plan['workload_seeds']:
                    out=base/f'workload_{w}';jobs.append((command(plan,m,s,w,out,checkpoint,widths=lock['widths'] if m==plan['capacity_method'] else None),out))
        return jobs
    batch(training_jobs(methods),plan,root,lock,'main_training')
    batch(evaluation_jobs(methods),plan,root,lock,'main_evaluation')
    from ch3_analysis import analyze,diagnostic_pair,unit
    report=analyze(root,plan,lock['source_hash'])
    if report['primary_criterion_met']:
        batch(training_jobs([plan['capacity_method']]),plan,root,lock,'capacity_training')
        batch(evaluation_jobs([plan['capacity_method']]),plan,root,lock,'capacity_evaluation')
        analyze(root,plan,lock['source_hash'],capacity=True)
    else:write_json(root/'capacity_not_triggered.json',dict(primary_criterion_met=False,reason='Predeclared gate failed; no capacity training or optional sample extension'))
    diagnostics=[]
    for s in plan['training_seeds']:
        for w in plan['workload_seeds']:
            left=root/'main'/plan['cells']['A']/f'seed_{s}'/f'workload_{w}'
            right=root/'main'/plan['cells']['B']/f'seed_{s}'/f'workload_{w}'
            diagnostics.append(dict(training_seed=s,workload_seed=w,groups=diagnostic_pair(left,right)))
    write_json(root/'request_diagnostics.json',dict(status='PASS',units=diagnostics,interpretation='conditional descriptive comparisons; not causal mediation'))
    jobs=[]
    for setting in plan['boundary']['settings']:
        for cell in ('A','B'):
            m=plan['cells'][cell]
            for s in plan['training_seeds'][:3]:
                checkpoint=root/'main'/m/f'seed_{s}'/'train/checkpoints'/plan['final_checkpoint']
                for w in plan['boundary_workload_seeds']:
                    out=root/'boundary'/setting['name']/m/f'seed_{s}'/f'workload_{w}'
                    jobs.append((command(plan,m,s,w,out,checkpoint,setting=setting),out))
    batch(jobs,plan,root,lock,'boundary_evaluation')
    rows=[];traces={}
    for cmd,out in jobs:
        if (out/'failure.json').exists():rows.append(dict(path=str(out),status='INTERFACE_NOT_APPLICABLE'));continue
        values,trace=unit(out,lock['source_hash'],2,plan['steps'])
        # Pair only inside the same physical setting, never across different interventions.
        setting=out.parents[2].name;workload=out.name
        key=(setting,workload)
        if key in traces and traces[key]!=trace:raise ValueError('Unpaired boundary stream')
        traces[key]=trace;rows.append(dict(path=str(out),status='PASS',**values))
    write_json(root/'boundary_analysis.json',dict(status='PASS',classification=plan['boundary']['classification'],units=rows))
    write_json(root/'status.json',dict(status='COMPLETE',stage='all_evidence',primary_criterion_met=report['primary_criterion_met'],source_hash=lock['source_hash'],completed=time.time()))


if __name__=='__main__':main()
