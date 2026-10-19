"""Public, paid-matching request features and a fixed eight-action proposal pool.

No simulator or truth is accepted by these interfaces. Candidate assignments
must already have been obtained by paying the conditional matching cost.
"""
from __future__ import annotations
import math
import random

SCHEMA='paid_candidate_request_risk54_v1'
FEATURES=('batch_n','log_size','log_cycles','deadline','log_deadline','cost',
    'base','work','compute_time','pred_time','margin_ratio','pred_ratio','base_ratio','work_ratio',
    'log_energy','log_compute_energy','cache_belief','log_capacity','target_count','target_fraction',
    *('source_'+str(i) for i in range(5)),*('target_'+str(i) for i in range(6)),
    'local','cooperative','mbs','batch_mbs','batch_cooperative','batch_local','source_fraction',
    'source_to_target','source_mbs','target_source_diversity','target_mean_margin','target_min_margin','target_miss',
    'source_mean_margin','source_miss','target_total_work','source_total_base',
    'alternative_pred_ratio','best_cache','legal_fraction','min_cache','max_cache','target_energy_fraction')
assert len(FEATURES)==54

def proposal_actions(action_ids,x):
    """Four distinct source/destination proposals, then four fixed random ones."""
    from utils.sf_hlm_action_ranker import actions
    tokens=actions();indices={a:i for i,a in enumerate(action_ids)}
    positive=sorted(a for a in action_ids if a>0)
    order=sorted(positive,key=lambda a:(-(x[indices[a]][118]-x[indices[a]][119]),
        -x[indices[a]][120],tokens[a][2],a))
    chosen=[];pairs=set()
    for action in order:
        pair=tokens[action][:2]
        if pair in pairs:continue
        pairs.add(pair);chosen.append(action)
        if len(chosen)==4:break
    rest=[a for a in positive if a not in chosen]
    chosen+=random.Random(20261017).sample(rest,min(4,len(rest)))
    return chosen

def unique_pool(candidates,proposals):
    """Called on already computed candidates; never influences proposal creation."""
    by_action={c['action']:c for c in candidates};seen=set();out=[]
    for action in [-1,0,*sorted(proposals)]:
        c=by_action[action];key=tuple(c['assignment'])
        if key not in seen:out.append(c);seen.add(key)
    return out

def choose(action_ids,scores):
    if len(action_ids)!=len(scores) or not action_ids:raise ValueError('Empty or mismatched candidate scores')
    if not all(math.isfinite(float(v)) for v in scores):raise ValueError('Nonfinite risk score')
    return max(sorted(zip(action_ids,scores)),key=lambda v:v[1])[0]

def candidate_features(rows,assignment,cost_ms,energy_cap):
    """One row per request, no request identity or private result in features."""
    import numpy as np
    if len(rows)!=len(assignment):raise ValueError('Candidate assignment length')
    delay=float(cost_ms)/1000
    if not math.isfinite(delay) or delay<0:raise ValueError('Invalid cost')
    n=len(rows)
    if not n:return np.empty((0,len(FEATURES)),dtype=np.float64),np.empty(0,dtype=np.float64)
    sources=[int(r['source']) for r in rows];opts=[r['options'][str(a)] if str(a) in r['options'] else r['options'][a] for r,a in zip(rows,assignment)]
    dest=[5 if o['target']=='mbs' else int(o['target']) for o in opts]
    if not all(0<=s<5 for s in sources) or not all(0<=d<6 for d in dest):raise ValueError('Node index')
    counts=[dest.count(j) for j in range(6)];loads=[max(k,1) for k in counts];loads[5]=1
    base=[float(o['base']) for o in opts];work=[float(o['work']) for o in opts]
    times=[b+w*loads[d] for b,w,d in zip(base,work,dest)]
    deadlines=[float(r['deadline']) for r in rows]
    if any(d<=0 for d in deadlines):raise ValueError('Nonpositive deadline')
    margin=[max(-10.,min(1.,1-(t+delay)/d)) for t,d in zip(times,deadlines)]
    miss=[t+delay>d for t,d in zip(times,deadlines)]
    energies=[float(o['energy'])+float(o['compute_energy'])/loads[d]**2 for o,d in zip(opts,dest)]
    local=sum(s==d for s,d in zip(sources,dest))/n;mbs=counts[5]/n;coop=1-local-mbs;cap=max(float(energy_cap),1e-12)
    table=[]
    for i,(row,o,s,d) in enumerate(zip(rows,opts,sources,dest)):
        node=[k for k in range(n) if dest[k]==d];source=[k for k in range(n) if sources[k]==s]
        options=list(row['options'].values());deadline=deadlines[i]
        alts=[float(v['base'])+float(v['work'])*loads[5 if v['target']=='mbs' else int(v['target'])] for v in options]
        cache=[float(v['cache_belief']) for v in options];best=min(range(len(alts)),key=lambda k:alts[k])
        ratio=lambda v:max(-10.,min(10.,v/deadline))
        values=[n/100,math.log1p(row['size']),math.log1p(row['cycles']),deadline,math.log1p(deadline),cost_ms/100,
            base[i],work[i],work[i]*loads[d],times[i],margin[i],ratio(times[i]+delay),ratio(base[i]),ratio(work[i]),
            math.log1p(o['energy']),math.log1p(o['compute_energy']),o['cache_belief'],math.log1p(o['capacity']),counts[d]/100,counts[d]/n,
            *[float(s==j) for j in range(5)],*[float(d==j) for j in range(6)],float(s==d),float(d!=5 and s!=d),float(d==5),
            mbs,coop,local,len(source)/n,sum(dest[k]==d for k in source)/n,sum(dest[k]==5 for k in source)/n,
            len({sources[k] for k in node})/5,sum(margin[k] for k in node)/len(node),min(margin[k] for k in node),sum(miss[k] for k in node)/len(node),
            sum(margin[k] for k in source)/len(source),sum(miss[k] for k in source)/len(source),sum(work[k] for k in node),sum(base[k] for k in source),
            ratio(alts[best]+delay),cache[best],len(options)/6,min(cache),max(cache),sum(energies[k] for k in node)/cap]
        table.append(values)
    x=np.asarray(table,dtype=np.float64);p0=np.asarray([not v for v in miss],dtype=np.float64)
    if x.shape!=(n,len(FEATURES)) or not np.isfinite(x).all():raise ValueError('Request feature shape or finiteness')
    return x,p0

def state_weight(candidates,requests,costs):
    if min(candidates,requests,costs)<1:raise ValueError('Empty training unit')
    return 1./(candidates*requests*costs)
