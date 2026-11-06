"""Post-run two-future candidate-signal agreement; never selects a model."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys


def diagnose(task,source):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_long_value_experiment as trial
    out=trial.out(task);base=trial.base;load=lambda p:json.loads(p.read_text())
    protocol=load(out/'protocol.json');states=[]
    for iteration in (1,2):
        for b in range(3):
            for field in ('train','validation'):
                for w in protocol[field][str(iteration)]:
                    for t in protocol['times']:
                        snap=base.read(out/'states'/f'{iteration}_{b}_{w}_{t}.pkl.gz');k=len(snap['pre']['actions']);values=[]
                        for r in range(2):
                            q=[]
                            for c in range(math.ceil(k/4)):q+=load(out/'labels'/f'{iteration}_{b}_{w}_{t}_{r}_{c}.json')['values']
                            values.append(q)
                        a,z=values;ma=statistics.mean(a);mz=statistics.mean(z)
                        ca=[x-ma for x in a];cz=[x-mz for x in z]
                        norm=math.sqrt(sum(x*x for x in ca)*sum(x*x for x in cz))
                        correlation=sum(x*y for x,y in zip(ca,cz))/norm if norm else None
                        ia=max(range(k),key=lambda i:a[i]);iz=max(range(k),key=lambda i:z[i])
                        heldout=((z[ia]-mz)+(a[iz]-ma))/2
                        mean=[(x+y)/2 for x,y in zip(a,z)]
                        states.append(dict(iteration=iteration,block=b,split=field,workload=w,time=t,correlation=correlation,
                            one_future_select_other_gain=heldout,naive_best_gain=max(mean)-statistics.mean(mean),
                            mean_target_spread=max(mean)-min(mean)))
    groups={}
    for iteration in (1,2):
        for field in ('train','validation'):
            rows=[r for r in states if r['iteration']==iteration and r['split']==field]
            corr=[r['correlation'] for r in rows if r['correlation'] is not None]
            groups[f'{iteration}/{field}']=dict(states=len(rows),nonflat_both=len(corr),
                mean_correlation=statistics.mean(corr) if corr else None,negative_correlation=sum(x<0 for x in corr),
                heldout_selector_gain=statistics.mean(r['one_future_select_other_gain'] for r in rows),
                naive_best_gain=statistics.mean(r['naive_best_gain'] for r in rows))
    base.write(out/'signal_diagnosis.json',dict(groups=groups,states=states,
        boundary='Descriptive two-future agreement and one-future selection on the other; low agreement does not identify a unique cause, and states/candidates are not independent training repeats. Post hoc, not a model-selection rule.'))
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();diagnose(a.task,a.source)
