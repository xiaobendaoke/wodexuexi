"""Barrier-synchronized CPU rollout collection with schedule-independent seeds."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import sys
import contextlib

_runtime=None


def initialize_worker(source,extension,resource_root):
    global _runtime
    sys.path.insert(0,source)
    for name in ('utils.deadline_repair','utils.deadline_repair_online','utils.node_offload_observation',
                 'utils.sf_hlm_matching','utils.sf_hlm_search','marl_models.sf_hlm','utils.sf_hlm_resources',
                 'utils.sf_hlm_v2','marl_models.sf_hlm_v2'):
        path=Path(extension)/(name.replace('.','/')+'.py');spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    _runtime=dict(resource_root=resource_root,models={})


def collect_task(task):
    import numpy as np
    import torch
    from utils.sf_hlm_resources import CpuLease
    from utils.sf_hlm_v2 import search,teacher
    from marl_models.sf_hlm_v2 import CompleteQuotaPPO,ArchivePolicy
    kind='teacher' if task['kind']=='teacher' else 'rollout_'+task['family']
    with CpuLease(_runtime['resource_root'],task=['rollout',task['kind'],task['id'],task.get('version')],kind=kind):
        if task['kind']=='teacher':return dict(id=task['id'],teacher=teacher(task['rows'],task['num_uavs'],task['budget']))
        rng=np.random.default_rng(task['seed'])
        model=None
        if task['kind']=='learned':
            key=(task['family'],task['version']);models=_runtime['models']
            if key not in models:
                model=(CompleteQuotaPPO if task['family']=='complete' else ArchivePolicy)(task['num_uavs'])
                model.network.load_state_dict(task['weights']);model.network.eval();models.clear();models[key]=model
            model=models[key]
        # Model construction uses Torch RNG. Reset only after loading so cache
        # hits and first use define identical trajectories.
        torch.manual_seed(task['seed'])
        a,d,r=search(task['rows'],num_uavs=task['num_uavs'],kind=task['kind'],family=task['family'],model=model,rng=rng,
                     exploration=True,budget=task['budget'],breakpoints=task.get('breakpoints',True),multiple=task.get('multiple',True))
        return dict(id=task['id'],assignment=a,diagnostic=d,records=r,**{'return':d.get('terminal_return',sum(x['reward'] for x in r))})


class ParallelCollector:
    def __init__(self,args,workers=32):
        self.args=args;self.workers=workers;self.pool=None

    def map(self,tasks,lease=None):
        if self.workers==0:
            with lease.suspended() if lease is not None else contextlib.nullcontext():
                if _runtime is None:initialize_worker(str(self.args.source_root),str(self.args.extension_root),str(self.args.resources))
                values=[collect_task(t) for t in tasks]
        else:
            if self.pool is None:self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context('spawn'),
                initializer=initialize_worker,initargs=(str(self.args.source_root),str(self.args.extension_root),str(self.args.resources)))
            if lease is None:values=list(self.pool.map(collect_task,tasks,chunksize=1))
            else:
                with lease.suspended():values=list(self.pool.map(collect_task,tasks,chunksize=1))
        return sorted(values,key=lambda t:t['id'])

    def close(self):
        if self.pool is not None:self.pool.shutdown(wait=True);self.pool=None


def rollout_tasks(model,rows_list,indices,*,protocol_hash,seed,iteration,budget,breakpoints=True,multiple=True):
    from utils.sf_hlm_resources import trajectory_seed
    import hashlib
    weights={k:v.detach().clone().share_memory_() for k,v in model.network.state_dict().items()}
    version=hashlib.sha256(b''.join(v.numpy().tobytes() for v in weights.values())).hexdigest()
    copies=1 if model.family=='complete' else 8;tasks=[]
    for group,index in enumerate(indices):
        for trajectory in range(copies):
            tasks.append(dict(id=(group,trajectory),rows=rows_list[int(index)],kind='learned',family=model.family,
                version=version,weights=weights,num_uavs=model.num_uavs,budget=budget,
                seed=trajectory_seed(protocol_hash,seed,iteration,group,trajectory),breakpoints=breakpoints,multiple=multiple))
    return tasks
