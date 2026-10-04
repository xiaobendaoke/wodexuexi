"""Action-aligned observations from source-local predecision knowledge.

No final loads, neighbor cache truth, future requests or other UAV proposals
enter these actor features. The fleet repair still uses the separate protocol.
"""
from __future__ import annotations
import math
import numpy as np

NODE_FEATURE_NAMES=(
    'legal','local','mbs','cooperative','transmission_ratio','single_compute_ratio',
    'single_total_ratio','single_deadline_margin','log_capacity_ratio','cache_belief',
    'log_noncompute_energy','log_compute_energy',
)
FEATURE_CONTRACT='source_local_action_aligned_12_v1'


def describe_candidates(env,requests,rows):
    """Use predictor inputs plus publicly configured capacities and beliefs."""
    import config
    if len(requests)!=len(rows):raise ValueError('Request/row count mismatch')
    ues={ue.id:ue for ue in env.ues};uavs={u.id:u for u in env.uavs}
    descriptions=[]
    for request,row in zip(requests,rows):
        owner,slot,ue_id,source,deadline=request
        if row['source']!=source or row['ue_id']!=ue_id:raise ValueError('Row misalignment')
        uav=env.uavs[owner];req=ues[ue_id].current_request
        cycles=float(config.CPU_CYCLES_PER_BYTE[req.req_id])*req.req_size
        options={}
        for action,o in row['options'].items():
            if action==1:
                capacity=float(config.MBS_COMPUTING_CAPACITY)
                compute=cycles/capacity;tx=max(0.,o['base']-compute);belief=1.
            else:
                target=uav if action==0 else uavs[action-2]
                capacity=float(config.UAV_COMPUTING_CAPACITY[target.id])
                compute=o['work'];tx=o['base']
                belief=float(uav.cache[req.req_id]) if action==0 else float(uav._get_service_neighbor_cache_belief(req.req_id,target))
            options[action]=dict(tx=tx,compute=compute,capacity=capacity,belief=belief,
                                 energy=o['energy'],compute_energy=o['compute_energy'])
        descriptions.append(options)
    return descriptions


def pack_observation(lower_obs,masks,requests,descriptions,*,capacity_reference,energy_reference):
    """Pack identical node inputs for flat and shared actors; retain old critic prefix."""
    obs=np.asarray(lower_obs,dtype=np.float32);mask=np.asarray(masks)
    if obs.ndim!=2 or mask.ndim!=3 or mask.shape[0]!=len(obs):raise ValueError('Observation shape')
    if not np.isfinite(obs).all() or not np.isfinite(mask).all():raise ValueError('Nonfinite observation')
    if capacity_reference<=0 or energy_reference<=0:raise ValueError('Normalization scales')
    if len(requests)!=len(descriptions):raise ValueError('Lost descriptions')
    nodes=np.zeros((*mask.shape,len(NODE_FEATURE_NAMES)),np.float32)
    seen=set()
    for request,options in zip(requests,descriptions):
        owner,slot,ue_id,source,deadline=request
        if not (0<=owner<mask.shape[0] and 0<=slot<mask.shape[1]):raise ValueError('Invalid request slot')
        if (owner,slot) in seen or not math.isfinite(deadline) or deadline<=0:raise ValueError('Duplicate slot/invalid deadline')
        seen.add((owner,slot))
        legal=set(np.flatnonzero(mask[owner,slot]>0).tolist())
        if not legal or set(options)!=legal:raise ValueError('Description/action mask mismatch')
        for action,o in options.items():
            if not all(math.isfinite(float(v)) for v in o.values()):raise ValueError('Nonfinite node')
            if any(o[k]<0 for k in ['tx','compute','energy','compute_energy']) or o['capacity']<=0 or not 0<=o['belief']<=1:
                raise ValueError('Invalid node quantities')
            total=(o['tx']+o['compute'])/deadline
            nodes[owner,slot,action]=[
                1.,float(action==0),float(action==1),float(action>=2),
                min(o['tx']/deadline,10.),min(o['compute']/deadline,10.),min(total,10.),max(-10.,1.-total),
                min(math.log1p(o['capacity']/capacity_reference),10.),o['belief'],
                min(math.log1p(o['energy']/energy_reference),10.),min(math.log1p(o['compute_energy']/energy_reference),10.),
            ]
    expected=set(zip(*np.where((mask>0).any(-1))))
    if seen!=expected:raise ValueError('Lost or extra active request')
    return np.concatenate([obs,nodes.reshape(len(obs),-1)],axis=-1)


def prepare_observation(env,lower_obs,masks,requests,rows):
    import config
    return pack_observation(lower_obs,masks,requests,describe_candidates(env,requests,rows),
        capacity_reference=float(np.mean(config.UAV_COMPUTING_CAPACITY)),
        energy_reference=float(config.OFFLOAD_ENERGY_NORM_REF))
