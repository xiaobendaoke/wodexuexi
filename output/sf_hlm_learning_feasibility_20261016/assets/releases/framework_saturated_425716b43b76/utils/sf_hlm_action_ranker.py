"""Observed-only 128-dimensional one-transfer ranking inputs (no simulator)."""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path

SCHEMA='observed_global63_cost_node_action128_v1'
PUBLIC_FIELDS=('rows','initial_assignment','greedy_assignment','initial_counts','energy_cap','legal_nonstop_actions')
_STATS=None


def public_payload(record):
    value={key:record[key] for key in PUBLIC_FIELDS}
    row_fields=('source','ue_id','deadline','size','cycles')
    option_fields=('target','base','work','energy','compute_energy','cache_belief','capacity')
    cleaned=[]
    for row in value['rows']:
        selected={key:row[key] for key in row_fields};options={}
        for action,option in row['options'].items():
            if isinstance(action,bool):raise ValueError('Boolean action identifier')
            options[int(action)]={key:option[key] for key in option_fields}
        selected['options']=options;cleaned.append(selected)
    value['rows']=cleaned;return value


def statistics_module():
    global _STATS
    if _STATS is None:
        spec=importlib.util.spec_from_file_location('ranker_public_statistics',Path(__file__).with_name('sf_hlm_selector.py'))
        _STATS=importlib.util.module_from_spec(spec);spec.loader.exec_module(_STATS)
    return _STATS


def actions():return (None,)+tuple((i,j,k) for i in range(6) for j in range(6) if i!=j for k in (1,2,4))


def training_weight(n_actions,n_costs):
    if n_actions<1 or n_costs<1:raise ValueError('Empty training state')
    return 1./(n_actions*n_costs)


def choose(action_ids,predicted):
    best=-1;score=0.
    for action,value in sorted(zip(action_ids,predicted)):
        if action==-1:continue
        if not math.isfinite(float(value)):raise ValueError('Nonfinite utility')
        if value>score:best,score=int(action),float(value)
    return best


def features(record,cost_ms,*,problem=None):
    import numpy as np
    from utils.sf_hlm_matching import problem_from_rows
    p=public_payload(record);rows=p['rows'];n=len(rows);r=max(n,1);delay=float(cost_ms)/1000
    if not math.isfinite(delay) or delay<0:raise ValueError('Invalid cost condition')
    problem=problem_from_rows(rows,5) if problem is None else problem
    ids=[-1,0,*p['legal_nonstop_actions']];counts=np.asarray(p['initial_counts'],dtype=np.int64)
    if counts.shape!=(6,) or (counts<0).any() or counts.sum()!=n:raise ValueError('Initial quota does not conserve requests')
    cap=max(float(p['energy_cap']),1e-12);index=np.arange(n);deadline=problem['deadline']

    def feedback(assignment):
        targets=np.asarray([5 if row['options'][a]['target']=='mbs' else row['options'][a]['target'] for row,a in zip(rows,assignment)],dtype=np.int64)
        allocation=np.bincount(targets,minlength=6);load=allocation.copy();load[-1]=1
        times=problem['base'][index,targets]+problem['work'][index,targets]*load[targets]
        energy=problem['energy'][index,targets]+problem['compute_energy'][index,targets]/load[targets]**2
        margin=1-(times+delay)/deadline;out=np.zeros((6,6),np.float64)
        for j in range(6):
            selected=targets==j
            if not selected.any():continue
            release=problem['base'][index,targets]+problem['work'][index,targets]*np.maximum(load[targets]-1,1)
            rescued=(targets!=5)&(release+delay<=deadline)&(times+delay>deadline)
            out[j]=[allocation[j]/r,float((margin[selected]<0).mean()),np.clip(margin[selected].mean(),-10,1),
                    np.clip(margin[selected].min(),-10,1),rescued[selected].sum()/r,math.fsum(energy[selected])/cap]
        return targets,times,energy,out
    targets,times,energy,node=feedback(p['initial_assignment']);_,gt,ge,_=feedback(p['greedy_assignment'])
    prefix=np.asarray([*statistics_module().features(rows),float(np.count_nonzero(times+delay>deadline))/r,
        math.fsum(energy)/cap,float(np.count_nonzero(gt+delay>deadline))/r,math.fsum(ge)/cap,*node.ravel()],dtype=np.float64)
    table=[];tokens=actions();load=np.maximum(counts,1);load[-1]=1
    for action in ids:
        local=np.zeros(12);effect=np.zeros(8);flags=[float(action==-1),float(action==0),0.,0.]
        if action>0:
            source,dest,amount=tokens[action];local=np.r_[node[source],node[dest]]
            new_source=1 if source==5 else max(int(counts[source])-amount,1)
            new_dest=1 if dest==5 else max(int(counts[dest])+amount,1)
            before_source=problem['base'][:,source]+problem['work'][:,source]*load[source]
            after_source=problem['base'][:,source]+problem['work'][:,source]*new_source
            before_dest=problem['base'][:,dest]+problem['work'][:,dest]*load[dest]
            after_dest=problem['base'][:,dest]+problem['work'][:,dest]*new_dest
            rescue=np.count_nonzero((targets==source)&(before_source+delay>deadline)&(after_source+delay<=deadline))
            loss=np.count_nonzero((targets==dest)&(before_dest+delay<=deadline)&(after_dest+delay>deadline))
            safe=np.count_nonzero(problem['legal'][:,dest]&(after_dest+delay<=deadline))
            effect=np.asarray([amount/r,(counts[source]-amount)/r,(counts[dest]+amount)/r,
                rescue/r,loss/r,safe/r,(problem['legal'][:,dest].sum()-counts[dest]-amount)/r,float(dest==5)])
            flags[2:]=[float(source==5),float(dest==5)]
        table.append(np.r_[prefix,local,effect,flags,cost_ms/100])
    x=np.asarray(table,dtype=np.float64)
    if x.shape!=(len(ids),128) or not np.isfinite(x).all():raise ValueError('Feature shape/finiteness')
    return ids,x


def analytic_choice(action_ids,x,n):
    # Columns 63/65 are initial/G miss fractions; action effects start at 115.
    scores=(x[:,65]-x[:,63])*max(n,1)+(x[:,118]-x[:,119])*max(n,1)
    return choose(action_ids,scores)
