"""Posthoc no-training comparison of counterfactual random-number coupling."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_long_value_experiment as study
from scripts.analysis import queued_service_motion_ranking as base
from utils.queued_long_value import remaining_future
from utils.queued_sampling_coupling import act_coupled,semantic_labels

REV='queued_long_coupling_20261102'
REPLICAS=(2,3,4,5)


def out(task):return Path(task)/'results'/base.TASK/REV


def branch(payload):
    task,b,w,t,r,mode,chunk=payload
    snap=study.state_at(task,2,b,w,t);model=study.model_at(task,b,1)
    future=remaining_future(w,snap['engine'].ues,t,r);fingerprint=hashlib.sha256(pickle.dumps(future,5)).hexdigest()
    indexes=list(range(4*chunk,min(4*chunk+4,len(snap['pre']['actions']))));values=[];counts=[]
    for i in indexes:
        e=copy.deepcopy(snap['engine']);adapter=copy.deepcopy(snap['adapter']);initial=len(e.tasks)
        e.assign(snap['plan']);e.advance(snap['pre']['actions'][i])
        while e.t<200 or e.tasks:
            now=int(e.t);assert now<=204;e.add(copy.deepcopy(future.get(now,())));e.admit();obs=e.observe()
            if mode=='original':_,plan,motion=model.act(obs,adapter,base.seed(study.REV,'tail',b,w,t,r,now))
            else:plan,motion=act_coupled(model,obs,adapter,(REV,b,w,t,r))
            e.assign(plan);e.advance(motion)
        e.audit();c=dict(e.counts);total=initial+sum(len(xs) for at,xs in future.items() if t<at<200)
        assert total==c['generated']==c['success']+c['expired'];values.append(c['success']);counts.append(c)
    base.write(out(task)/'branches'/f'{b}_{w}_{t}_{r}_{mode}_{chunk}.json',dict(block=b,workload=w,time=t,replica=r,mode=mode,
        indexes=indexes,values=values,counts=counts,future_sha=fingerprint,cpu=sorted(os.sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def correlation(a,b):
    a=[x-statistics.mean(a) for x in a];b=[x-statistics.mean(b) for x in b];den=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/den if den else None


def summarize(task,states):
    result={};rows=[]
    for mode in ('original','semantic'):
        for b,w,t,k in states:
            q=[]
            for r in REPLICAS:
                values=[]
                for chunk in range(math.ceil(k/4)):
                    d=json.loads((out(task)/'branches'/f'{b}_{w}_{t}_{r}_{mode}_{chunk}.json').read_text());values+=d['values']
                    other=json.loads((out(task)/'branches'/f'{b}_{w}_{t}_{r}_{"semantic" if mode=="original" else "original"}_{chunk}.json').read_text())
                    assert d['future_sha']==other['future_sha']
                assert len(values)==k;q.append(values)
            a=[statistics.mean(x) for x in zip(*q[:2])];z=[statistics.mean(x) for x in zip(*q[2:])]
            gain=((z[max(range(k),key=a.__getitem__)]-statistics.mean(z))+(a[max(range(k),key=z.__getitem__)]-statistics.mean(a)))/2
            centered=[[x-statistics.mean(xs) for x in xs] for xs in q]
            variance=statistics.mean(statistics.variance(xs) for xs in zip(*centered))
            rows.append(dict(mode=mode,block=b,workload=w,time=t,values=q,correlation=correlation(a,z),crossfit_gain=gain,centered_variance=variance))
        selected=[x for x in rows if x['mode']==mode];corr=[x['correlation'] for x in selected if x['correlation'] is not None]
        result[mode]=dict(mean_correlation=statistics.mean(corr),mean_crossfit_gain=statistics.mean(x['crossfit_gain'] for x in selected),
            mean_centered_variance=statistics.mean(x['centered_variance'] for x in selected))
    base.write(out(task)/'analysis.json',dict(status='COMPLETE',summary=result,rows=rows,training=False,
        scope='Posthoc 12 selected validation states, four fresh futures, split-half diagnostics. Marginal categorical policies unchanged; coupling differs. No DSR claim or independent training replication.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);a=p.parse_args();task=a.task
    import torch
    torch.set_num_threads(1);states=[];jobs=[]
    for b in range(3):
        for w in (8440,8441):
            for t in (75,175):
                snap=study.state_at(task,2,b,w,t);k=len(snap['pre']['actions']);states.append((b,w,t,k))
                assert len(semantic_labels(snap['engine'].observe(),snap['adapter'].history.weights,snap['pre']['actions']))==k
                for r in REPLICAS:
                    for mode in ('original','semantic'):
                        jobs.extend((str(task),b,w,t,r,mode,c) for c in range(math.ceil(k/4)))
    base.write(out(task)/'protocol.json',dict(revision=REV,states=states,replicas=REPLICAS,jobs=len(jobs),training=False,
        parent_sha={str(b):base.sha(study.checkpoint(task,b,1)) for b in range(3)},cpu=sorted(os.sched_getaffinity(0))))
    original=base.out;base.out=out
    try:base.parallel(task,'coupling',branch,jobs,sorted(os.sched_getaffinity(0)))
    finally:base.out=original
    summarize(task,states);print('COMPLETE',flush=True)


if __name__=='__main__':main()
