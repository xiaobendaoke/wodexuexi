"""Training-only steady-state throughput calibration sharing global CPU leases.

Imports/warm-up are excluded. Existing workers keep their leases. Requested and
effective concurrency are distinct when old experiments occupy the global cap.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('resources',ROOT/'utils/sf_hlm_resources.py')
res=importlib.util.module_from_spec(spec);spec.loader.exec_module(res)
runtime=None


def initialize(source,extension,resources,dataset):
    with res.CpuLease(resources,timeout=180):_initialize(source,extension,resources,dataset)


def _initialize(source,extension,resources,dataset):
    global runtime
    sys.path.insert(0,str(extension/'utils'))
    import sf_hlm_rollouts
    sf_hlm_rollouts.initialize_worker(str(source),str(extension),str(resources))
    import torch
    from marl_models.sf_hlm_v2 import CompleteQuotaPPO,ArchivePolicy,ServiceMAPPO
    from environment.env import Env
    from run_canonical_experiment import configure_method
    import config
    models=configure_method('lower_mlp_upper_uncovered_obs')
    sys.path.insert(0,str(extension/'scripts/analysis'))
    import sf_hlm_v2_experiment
    rows=torch.load(dataset,weights_only=False)['rows'];nonempty=[r for r in rows if r]
    if not nonempty:raise ValueError('Calibration requires real nonempty training graphs')
    upper=ServiceMAPPO(config.OBS_DIM_SINGLE,config.NUM_UAVS);upper.warm_start(models['upper'])
    runtime=dict(rows=nonempty,model=CompleteQuotaPPO(),archive=ArchivePolicy(),upper=upper,models=models,
        policy=sf_hlm_v2_experiment.policy,env=Env(scenario_seed=890001),index=0)
    runtime['obs']=np_obs=runtime['env'].reset()


def measure(task):
    folder,kind,ordinal,resource=task
    from utils.sf_hlm_v2 import search,teacher
    import numpy as np
    with res.CpuLease(resource,timeout=120):
        rows=runtime['rows'][ordinal%len(runtime['rows'])]
        rng=np.random.default_rng(ordinal);count=0
        def unit():
            if kind=='teacher':teacher(rows,5,128)
            elif kind.startswith('rollout'):
                family='critical' if kind.endswith('critical') else 'complete'
                search(rows,num_uavs=5,family=family,kind='learned',model=runtime['archive'] if family=='critical' else runtime['model'],exploration=True,rng=rng)
            else:
                env=runtime['env'];motion,actions,*_=runtime['policy'](env,np.asarray(runtime['obs'],np.float32),'full',runtime['models'],
                    family='complete',lower=runtime['model'],upper=runtime['upper'],rng=rng)
                runtime['obs'],_,_=env.step(motion,offloading_actions=actions)
                runtime['index']+=1
                if runtime['index']%1000==0:runtime['obs']=env.reset()
        unit() # warm-up outside the timed interval
        res.atomic_json(folder/'ready'/f'{ordinal}.json',dict(pid=os.getpid(),cpu=next(iter(os.sched_getaffinity(0)))))
        started=folder/'start.json';waiting=time.monotonic()
        while not started.exists():
            if time.monotonic()-waiting>180:raise TimeoutError('Calibration start barrier timed out')
            time.sleep(.01)
        timing=json.loads(started.read_text())
        while time.time()<timing['start']:time.sleep(.002)
        elapsed_start=time.perf_counter()
        while time.time()<timing['end']:unit();count+=1
        return dict(ordinal=ordinal,completed_units=count,elapsed=time.perf_counter()-elapsed_start,cpu=next(iter(os.sched_getaffinity(0))))


def active(root):
    values=[]
    for file in (root/'slots').glob('*.lock'):
        try:
            d=json.loads(file.read_text())
            if d.get('leased') and res.process_identity(d['pid'])==d.get('identity'):values.append(d)
        except (OSError,ValueError,KeyError):pass
    return values


def main():
    p=argparse.ArgumentParser()
    for key in ('source','extension','resources','dataset','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=20);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--concurrency',type=int,nargs='+',default=[32,64,96,120,192,240]);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with res.task_lock(args.output/'calibration.lock'):
        profile=res.initialize(args.resources);profile['slots']=profile['maximum_slots']
        profile['calibration_running']=True;res.atomic_json(args.resources/'profile.json',profile)
        records=[];selected={}
        with ProcessPoolExecutor(max_workers=max(args.concurrency),mp_context=mp.get_context('spawn'),
             initializer=initialize,initargs=(args.source,args.extension,args.resources,args.dataset)) as pool:
            for kind in ('environment','teacher','rollout_complete','rollout_critical'):
                for requested in args.concurrency:
                    for repeat in range(args.repeats):
                        # Keep sixteen spare shared slots for concurrently running
                        # v2 parents. Never stop existing scientific workers.
                        live=active(args.resources);effective=min(requested,max(1,profile['maximum_slots']-len(live)-16))
                        folder=args.output/kind/f'concurrency_{requested}_round_{repeat}';folder.mkdir(parents=True,exist_ok=True)
                        futures=[pool.submit(measure,(folder,kind,i,args.resources)) for i in range(effective)]
                        waiting=time.monotonic()
                        while len(list((folder/'ready').glob('*.json')))<effective:
                            for f in futures:
                                if f.done() and f.exception():raise f.exception()
                            if time.monotonic()-waiting>150:raise TimeoutError('Calibration warm-up barrier failed')
                            time.sleep(.1)
                        start=time.time()+.2;res.atomic_json(folder/'start.json',dict(start=start,end=start+args.seconds))
                        measurements=[f.result() for f in futures]
                        row=dict(kind=kind,requested=requested,effective=effective,repeat=repeat,seconds=args.seconds,
                            occupied_by_existing=len(live),throughput=sum(x['completed_units'] for x in measurements)/args.seconds,
                            distinct_cpus=len({x['cpu'] for x in measurements}),units=measurements)
                        records.append(row);res.atomic_json(args.output/'measurements.json',records)
                        res.atomic_json(args.output/'status.json',dict(phase='RUNNING',**{k:v for k,v in row.items() if k!='units'}))
                means={n:sum(r['throughput'] for r in records if r['kind']==kind and r['requested']==n)/args.repeats for n in args.concurrency}
                best=max(means.values());selected[kind]=min(n for n in args.concurrency if means[n]>=.95*best)
                profile.update(concurrency_by_kind=selected);res.atomic_json(args.resources/'profile.json',profile)
            profile.update(slots=min(profile['maximum_slots'],max(selected.values())),calibrated=True,calibration_running=False)
            res.atomic_json(args.resources/'profile.json',profile)
            res.atomic_json(args.output/'status.json',dict(phase='COMPLETE',concurrency_by_kind=selected,global_slots=profile['slots'],
                measurements=len(records),steady_state=True,existing_workers_preserved=True))


if __name__=='__main__':main()
