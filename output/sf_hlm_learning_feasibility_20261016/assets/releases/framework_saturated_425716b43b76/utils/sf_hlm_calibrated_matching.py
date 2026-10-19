"""Request-success edge costs at an observable, fixed execution quota.

The objective differs from the archived predictor's integer deadline objective.
All inputs are public; actual cache and execution truth are not accepted.
"""
from __future__ import annotations
import math

LOCAL_COLUMNS=tuple(range(35))+(37,)+tuple(range(48,53))
SCHEMA='quota_local_request41_v1'
OBJECTIVES=('probability','binary','smooth')

def quota_set(initial,proposals):
    from utils.sf_hlm_action_ranker import actions
    tokens=actions();result=[(0,list(initial))]
    for action in proposals:
        source,dest,amount=tokens[action];q=list(initial);q[source]-=amount;q[dest]+=amount
        if min(q)<0 or sum(q)!=sum(initial):raise ValueError('Invalid proposed quota')
        result.append((int(action),q))
    return result

def select(scores,energies):
    if len(scores)!=len(energies) or not scores:raise ValueError('No selection candidates')
    return min(range(len(scores)),key=lambda i:(-scores[i],energies[i],i))

def edge_features(rows,counts,cost_ms,*,problem=None):
    """Flatten all legal (request,target) edges; exact local41 representation."""
    import numpy as np
    from utils.sf_hlm_matching import problem_from_rows,validated_counts
    problem=problem_from_rows(rows,5) if problem is None else problem;counts=validated_counts(counts,len(rows),5)
    loads=np.maximum(counts,1);loads[-1]=1;n=len(rows)
    if not math.isfinite(cost_ms) or cost_ms<0:raise ValueError('Invalid nominal cost')
    times=problem['base']+problem['work']*loads[None,:];deadline=problem['deadline'];source_count=[sum(r['source']==j for r in rows) for j in range(5)]
    table=[];indices=[]
    for i,row in enumerate(rows):
        options=list(row['options'].items());target=lambda o:5 if o['target']=='mbs' else int(o['target'])
        alts=[float(v['base'])+float(v['work'])*loads[target(v)] for _,v in options]
        cache=[float(v['cache_belief']) for _,v in options];best=min(range(len(alts)),key=lambda k:alts[k]);s=int(row['source']);dline=float(row['deadline'])
        ratio=lambda value:max(-10.,min(10.,value/dline))
        for action,option in options:
            j=target(option);base=float(option['base']);work=float(option['work']);t=base+work*loads[j]
            margin=max(-10.,min(1.,1-(t+cost_ms/1000)/dline))
            values=[n/100,math.log1p(row['size']),math.log1p(row['cycles']),dline,math.log1p(dline),cost_ms/100,
                base,work,work*loads[j],t,margin,ratio(t+cost_ms/1000),ratio(base),ratio(work),
                math.log1p(option['energy']),math.log1p(option['compute_energy']),option['cache_belief'],math.log1p(option['capacity']),counts[j]/100,counts[j]/n,
                *[float(s==k) for k in range(5)],*[float(j==k) for k in range(6)],float(s==j),float(j!=5 and s!=j),float(j==5),counts[5]/n,
                source_count[s]/n,ratio(alts[best]+cost_ms/1000),cache[best],len(options)/6,min(cache),max(cache)]
            table.append(values);indices.append((i,j))
    x=np.asarray(table,dtype=np.float64).reshape(-1,len(LOCAL_COLUMNS));indices=np.asarray(indices,dtype=np.int64).reshape(-1,2)
    if not np.isfinite(x).all():raise ValueError('Nonfinite local edge features')
    return x,indices,problem,counts

def solve(rows,counts,cost_ms,objective,model=None,*,assess=None,problem=None,probability_matrix=None):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from utils.deadline_repair import assess as original_assess
    from utils.sf_hlm_matching import allocation_counts,problem_from_rows,validated_counts
    if objective not in OBJECTIVES:raise ValueError('Unknown conditional objective')
    if objective=='probability' and model is None and probability_matrix is None:raise ValueError('Edge probabilities absent')
    if objective=='probability' and probability_matrix is None:x,index,p,q=edge_features(rows,counts,cost_ms,problem=problem)
    else:
        p=problem_from_rows(rows,5) if problem is None else problem;q=validated_counts(counts,len(rows),5)
    if not rows:return dict(assignment=(),counts=q,expected_success=0.,predicted=(assess or original_assess)([],()))
    loads=np.maximum(q,1);loads[-1]=1
    times=p['base']+p['work']*loads[None,:];energy=p['energy']+p['compute_energy']/loads[None,:]**2
    scale=max(float(energy[p['legal']].max(initial=0.)),np.finfo(float).tiny)
    if objective=='probability':
        if probability_matrix is None:
            flat=model.predict_proba(x)[:,list(model.classes_).index(1.)];prob=np.zeros_like(times);prob[index[:,0],index[:,1]]=flat
        else:
            prob=np.asarray(probability_matrix,dtype=np.float64)
            if prob.shape!=times.shape or not np.isfinite(prob).all() or (prob<0).any() or (prob>1).any():raise ValueError('Invalid prepaid edge probabilities')
        costs=1-prob+1e-9*energy/scale
    elif objective=='smooth':
        z=np.clip(((times+cost_ms/1000)/p['deadline'][:,None]-1)/.1,-60,60)
        prob=1/(1+np.exp(z));costs=1-prob+1e-9*energy/scale
    else:
        prob=(times+cost_ms/1000<=p['deadline'][:,None]).astype(float);costs=(len(rows)+1)*(1-prob)+energy/scale
    costs[~p['legal']]=np.inf;slots=np.repeat(np.arange(6),q)
    try:rr,cc=linear_sum_assignment(costs[:,slots])
    except ValueError as exc:
        if 'infeasible' in str(exc):return None
        raise
    targets=slots[cc]
    if not np.array_equal(rr,np.arange(len(rows))) or not p['legal'][rr,targets].all():raise ValueError('Illegal conditional matching')
    assignment=tuple(int(v) for v in p['action_ids'][rr,targets]);actual=allocation_counts(rows,assignment,5)
    if not np.array_equal(actual,q):raise ValueError('Conditional matching changed quota')
    return dict(assignment=assignment,counts=q,expected_success=float(prob[rr,targets].sum()),predicted=(assess or original_assess)(rows,assignment))

def score_assignment(rows,assignment,cost_ms,objective,model=None,*,problem=None):
    """Use the same edge scorer for G fallback and conditional candidates."""
    import numpy as np
    from utils.sf_hlm_matching import allocation_counts,problem_from_rows
    if not rows:return 0.
    counts=allocation_counts(rows,assignment,5);targets=[]
    for row,action in zip(rows,assignment):
        option=row['options'][action];targets.append(5 if option['target']=='mbs' else int(option['target']))
    if objective=='probability':
        x,index,p,q=edge_features(rows,counts,cost_ms,problem=problem)
        proba=model.predict_proba(x)[:,list(model.classes_).index(1.)];lookup={tuple(ij):prob for ij,prob in zip(index,proba)}
        return float(sum(lookup[i,j] for i,j in enumerate(targets)))
    p=problem_from_rows(rows,5) if problem is None else problem
    load=np.maximum(counts,1);load[-1]=1;t=p['base'][np.arange(len(rows)),targets]+p['work'][np.arange(len(rows)),targets]*load[targets]
    if objective=='binary':return float(np.count_nonzero(t+cost_ms/1000<=p['deadline']))
    if objective!='smooth':raise ValueError('Unknown objective')
    z=np.clip(((t+cost_ms/1000)/p['deadline']-1)/.1,-60,60);return float(np.sum(1/(1+np.exp(z))))
