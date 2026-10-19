"""Analytical cache-hit mixture from public proposal rows only.

Decomposition uses the predictor's own latency/energy equations. An invalid
edge falls back to that predictor's mean-time indicator and is counted.
"""
from __future__ import annotations
import collections
import math


def decode_row(row,powers):
    r=float(powers['backhaul_rx']);tx=float(powers['backhaul_tx'])
    h=float(powers['uav_tx'])+float(powers['uav_rx'])
    if any(not math.isfinite(x) or x<=0 for x in (r,tx,h)) or abs(h-r)<=1e-12:
        raise ValueError('Public powers do not identify cache/hop decomposition')
    anchors=[]
    for option in row['options'].values():
        if option['target']==row['source']:anchors.append(float(option['base'])-float(option['energy'])/r)
        elif option['target']=='mbs':
            anchors.append(float(option['base'])-float(option['energy'])/tx-float(row['cycles'])/float(option['capacity']))
    reason=None
    if not anchors:reason='missing_upload_anchor';upload=0.
    else:
        upload=anchors[0];tol=1e-8*max(1.,*(abs(x) for x in anchors))
        if not all(math.isfinite(x) and abs(x-upload)<=tol for x in anchors):reason='inconsistent_upload_anchor'
        elif upload < -tol:reason='negative_upload_anchor'
        upload=max(upload,0.)
    result={}
    for action,option in row['options'].items():
        base=float(option['base']);work=float(option['work']);target=option['target'];p=float(option['cache_belief'])
        part=dict(base=base,work=work,p=p,hit_base=base,miss_base=base,fallback=None)
        if target=='mbs' or target==row['source']:part['deterministic']=True
        else:
            part['deterministic']=False;part['fallback']=reason
            if reason is None:
                b=base-upload;hop=(float(option['energy'])-r*b)/(h-r);fetch_mean=b-hop
                tol=1e-8*max(1.,abs(base),abs(b),abs(hop),abs(fetch_mean))
                if not 0<=p<=1:part['fallback']='invalid_cache_probability'
                elif hop < -tol or fetch_mean < -tol:part['fallback']='negative_decomposition'
                elif p==1 and abs(fetch_mean)>tol:part['fallback']='unit_probability_nonzero_fetch'
                else:
                    hit=upload+max(hop,0.);part['hit_base']=hit
                    part['miss_base']=hit+max(fetch_mean,0.)/(1-p) if p<1 else hit
        result[int(action)]=part
    return result


def probability(part,count,deadline,delay):
    compute=part['work']*max(int(count),1)
    if part['deterministic'] or part['fallback'] is not None:
        return float(part['base']+compute+delay<=deadline)
    p=part['p']
    return p*float(part['hit_base']+compute+delay<=deadline)+(1-p)*float(part['miss_base']+compute+delay<=deadline)


class CacheMixtureKernel:
    def __init__(self,rows,powers,cost_ms=10.):
        self.rows=rows;self.parts=[decode_row(row,powers) for row in rows];self.delay=cost_ms/1000
        self.fallback_counts=dict(collections.Counter(p['fallback'] for parts in self.parts for p in parts.values() if p['fallback']))

    def probabilities(self,quotas):
        import numpy as np
        matrices={}
        for q in quotas:
            key=tuple(int(x) for x in q)
            if key in matrices:continue
            matrix=np.zeros((len(self.rows),6),dtype=np.float64)
            for i,(row,parts) in enumerate(zip(self.rows,self.parts)):
                for action,part in parts.items():
                    target=row['options'][action]['target'];j=5 if target=='mbs' else int(target)
                    matrix[i,j]=probability(part,1 if j==5 else q[j],row['deadline'],self.delay)
            matrices[key]=matrix
        return matrices
