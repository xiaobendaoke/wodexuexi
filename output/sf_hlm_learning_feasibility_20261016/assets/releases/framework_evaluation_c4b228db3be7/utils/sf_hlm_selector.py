"""Observed-only, dependency-free inference for the fixed development screen.

This module never receives private environment snapshots or outcome labels.
Scientific fitting and aggregation live in the remote-only analysis entry point.
"""
from __future__ import annotations
import math
import statistics
import struct

METHODS = ('G','R8','R16','H8','H16','L8','L16')
NONLEARNING = METHODS[:5]
SUMMARY_FIELDS = ('deadline','log_size','log_cycles','legal_count','best_ratio',
                  'best_slack','local_ratio','mbs_ratio','cache_mean','log_capacity')
FEATURE_NAMES = ('n',) + tuple(f'{k}_{s}' for k in SUMMARY_FIELDS
    for s in ('min','q25','mean','q75','max')) + (
    'source_max_share','source_hhi','source_nonempty_fraction','option_max_share','option_hhi',
    'best_ratio_ge_075','best_ratio_ge_090','best_ratio_ge_100',
    'best_slack_le_002','best_slack_le_005','local_legal_fraction','mbs_legal_fraction')


def quantile(values, q):
    a=sorted(values); x=(len(a)-1)*q; lo=math.floor(x); hi=math.ceil(x)
    return a[lo]+(a[hi]-a[lo])*(x-lo)


def _number(value, minimum=None):
    value=float(value)
    if not math.isfinite(value) or (minimum is not None and value<minimum):
        raise ValueError('Invalid observed numeric feature')
    return value


def features(rows, num_uavs=5):
    if num_uavs != 5: raise ValueError('Frozen feature contract requires five UAVs')
    if not rows: return [0.]*len(FEATURE_NAMES)
    values={k:[] for k in SUMMARY_FIELDS};source=[0]*num_uavs;targets=[0]*(num_uavs+1)
    local_legal=mbs_legal=0
    for row in rows:
        src=int(row['source'])
        if src != row['source'] or not 0<=src<num_uavs: raise ValueError('Unknown observed source')
        source[src]+=1
        deadline=_number(row['deadline'],minimum=0.)
        if deadline==0:raise ValueError('Deadline must be positive')
        options=row['options']
        if not options:raise ValueError('No legal options')
        latency={};cache=[];capacity=[]
        for key,opt in options.items():
            latency[str(key)]=_number(opt['base'],0.)+_number(opt['work'],0.)
            cache.append(_number(opt['cache_belief'],0.))
            if cache[-1]>1:raise ValueError('Invalid cache belief')
            capacity.append(math.log1p(_number(opt['capacity'],0.)))
            target=opt['target'];idx=num_uavs if target=='mbs' else int(target)
            if not 0<=idx<=num_uavs:raise ValueError('Unknown target')
            targets[idx]+=1
        best=min(latency.values())
        local_legal+=('0' in latency);mbs_legal+=('1' in latency)
        numbers=dict(deadline=deadline,log_size=math.log1p(_number(row['size'],0.)),
            log_cycles=math.log1p(_number(row['cycles'],0.)),legal_count=len(options),
            best_ratio=min(10.,best/deadline),best_slack=max(-10.,min(10.,deadline-best)),
            local_ratio=min(10.,latency['0']/deadline) if '0' in latency else 10.,
            mbs_ratio=min(10.,latency['1']/deadline) if '1' in latency else 10.,
            cache_mean=statistics.mean(cache),log_capacity=statistics.mean(capacity))
        for key,value in numbers.items():values[key].append(value)
    n=len(rows);out=[float(n)]
    for key in SUMMARY_FIELDS:
        a=values[key];out += [min(a),quantile(a,.25),statistics.mean(a),quantile(a,.75),max(a)]
    ss=[v/n for v in source];ts=[v/sum(targets) for v in targets]
    out += [max(ss),sum(v*v for v in ss),sum(v>0 for v in source)/num_uavs,
            max(ts),sum(v*v for v in ts)]
    out += [sum(v>=t for v in values['best_ratio'])/n for t in (.75,.9,1.)]
    out += [sum(v<=t for v in values['best_slack'])/n for t in (.02,.05)]
    out += [local_legal/n,mbs_legal/n]
    if len(out)!=len(FEATURE_NAMES) or not all(math.isfinite(v) for v in out):
        raise ValueError('Feature dimension or finiteness')
    return out


def folds(identities):
    ids=[tuple(v) for v in identities]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate state identity')
    blocks=sorted({v[1] for v in ids});workloads=sorted({v[2] for v in ids});seen=[]
    if len(blocks)!=5 or len(workloads)!=5:raise ValueError('Expected five blocks and workloads')
    for b in blocks:
        for w in workloads:
            train=[i for i,v in enumerate(ids) if v[1]!=b and v[2]!=w]
            test=[i for i,v in enumerate(ids) if v[1]==b and v[2]==w]
            if len(train)!=256 or len(test)!=16:raise ValueError('Incomplete frozen population')
            seen+=test
            yield dict(block=b,workload=w,train=train,test=test)
    if sorted(seen)!=list(range(len(ids))):raise ValueError('Not exactly one held-out prediction')


def choose(predicted, *, empty=False):
    if empty:return 'G'
    method='G';best=0.
    for key in METHODS[1:]:
        value=float(predicted.get(key,0.))
        if not math.isfinite(value):raise ValueError('Invalid predicted benefit')
        if value>best:method,best=key,value
    return method


def predict(model, vector):
    if model['kind']=='ridge':
        if not len(vector)==len(model['mean'])==len(model['scale']):raise ValueError('Feature dimension')
        z=[(x-m)/s for x,m,s in zip(vector,model['mean'],model['scale'])]
        values=[bias+sum(w*x for w,x in zip(weights,z))
                for bias,weights in zip(model['intercept'],model['coef'])]
    elif model['kind']=='tree':
        # sklearn trees convert inputs to float32 before comparing thresholds.
        x=[struct.unpack('f',struct.pack('f',v))[0] for v in vector];idx=0
        while model['nodes'][idx]['left']!=-1:
            node=model['nodes'][idx]
            idx=node['left'] if x[node['feature']]<=node['threshold'] else node['right']
        values=model['nodes'][idx]['value']
    else:raise ValueError('Unknown selector kind')
    return dict(zip(model['methods'],values))


def score(trials, overhead_seconds=0.):
    overhead_seconds=_number(overhead_seconds,0.)
    if not trials:raise ValueError('No branch repetitions')
    means=[]
    for trial in trials:
        times=trial['software_seconds']
        if not times:raise ValueError('No timing repeats')
        counts=[]
        for t in times:
            fee=_number(t,0.)+overhead_seconds;count=0
            for r in trial['requests']:
                count+=(_number(r['latency'],0.)+fee<=_number(r['deadline'],0.))
            counts.append(count)
        means.append(statistics.mean(counts))
    return statistics.mean(means)
