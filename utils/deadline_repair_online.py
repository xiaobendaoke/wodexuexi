"""Ideal instantaneous proposal coordination; no unmodeled costs implied.

prepare reads predecision source-local knowledge. apply only sees explicit
proposal rows/actions, never the environment or post-execution truth.
"""
import math
from utils.deadline_repair import proposal_inputs, repair, assess


def prepare(env,masks):
    import numpy as np
    requests=[];candidates=[]
    for owner,uav in enumerate(env.uavs):
        service=sorted((ue for ue in uav.current_covered_ues if ue.current_request.is_service),
                       key=lambda ue:float(np.linalg.norm(uav.pos[:2]-ue.pos[:2])))
        if len(service)>masks.shape[1]:raise ValueError('Request capacity exceeded')
        for slot,ue in enumerate(service):
            requests.append((owner,slot,ue.id,uav.id,float(ue.current_request.deadline)))
            candidates.append(tuple(int(a) for a in np.flatnonzero(masks[owner,slot])))
    ordered=sorted(zip(requests,candidates),key=lambda item:item[0][2])
    requests=tuple(r for r,_ in ordered);candidates=tuple(c for _,c in ordered)
    return requests,proposal_inputs(env,requests,candidates,masks)


def apply(requests,rows,actions,mode,budget=128):
    import numpy as np
    if mode not in ('baseline','ordinary','deadline'):raise ValueError('Unknown mode')
    if len(rows)!=len(requests):raise ValueError('Lost requests')
    raw=[actions[r[0],r[1]] for r in requests]
    if any(isinstance(a,(bool,np.bool_)) or not isinstance(a,(int,np.integer)) for a in raw):raise ValueError('Noninteger action')
    initial=tuple(int(a) for a in raw)
    predicted=assess(rows,initial)
    proposal=None if mode=='baseline' else repair(rows,initial,mode,budget)
    revised=actions.copy()
    chosen=initial if proposal is None else proposal['assignment']
    for request,row,a in zip(requests,rows,chosen):
        if a not in row['options']:raise ValueError('Quality mask violation')
        revised[request[0],request[1]]=a
    return revised,{'changed_actions':sum(a!=b for a,b in zip(initial,chosen)),
                    'evaluations':proposal['evaluations'] if proposal else 0,
                    'screen_operations':proposal['screen_operations'] if proposal else 0,
                    'original_predicted_misses':predicted['misses'],
                    'chosen_predicted_misses':proposal['predicted']['misses'] if proposal else predicted['misses'],
                    'original_predicted_energy':predicted['energy'],
                    'chosen_predicted_energy':proposal['predicted']['energy'] if proposal else predicted['energy']}


def greedy_assignment(rows):
    """One own-latency greedy pass using only already assigned target loads."""
    from collections import Counter
    counts=Counter();actions=[];evaluations=0
    for row in rows:
        if not row['options']:raise ValueError('No quality-legal action')
        def key(action):
            o=row['options'][action];n=1 if o['target']=='mbs' else counts[o['target']]+1
            return (o['base']+o['work']*n,o['energy']+o['compute_energy']/n**2,action)
        action=min(row['options'],key=key);evaluations+=len(row['options'])
        actions.append(action);counts[row['options'][action]['target']]+=1
    return tuple(actions),evaluations
