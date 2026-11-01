"""Delayed motion credit and training-only supervision from current public data."""
from types import SimpleNamespace
import numpy as np
import config
from environment.dynamic_service import DynamicServiceEnv
from utils.dynamic_service_scenarios import Request
from utils.dynamic_service_baselines import matching_search


def motion_gae(rewards,values,gamma=.99,lam=.95):
    """Action t receives service reward t+1; the last movement is terminal.

    Finite complete episodes only. This deliberately discounts future service,
    unlike the previous gamma=1 repair. Evaluation still uses total DSR.
    """
    r=np.asarray(rewards,float);v=np.asarray(values,float)
    if r.shape!=v.shape or r.ndim!=1 or not 0<=gamma<=1 or not 0<=lam<=1:raise ValueError('invalid credit input')
    n=max(len(r)-1,0);adv=np.zeros(n);tail=0.
    for t in reversed(range(n)):
        nxt=v[t+1] if t+1<n else 0.
        tail=r[t+1]+gamma*nxt-v[t]+gamma*lam*tail;adv[t]=tail
    return adv.astype(np.float32),(adv+v[:n]).astype(np.float32)


def label_indices(bank,selected,seed):
    legal=np.flatnonzero(bank['mask']);rng=np.random.default_rng(int(seed))
    best=int(np.argmax(np.where(bank['mask'],bank['features'][:,40],-np.inf)))
    indices=np.r_[selected,0,best,rng.choice(legal,min(8,len(legal)),replace=False)]
    return np.asarray(sorted(set(map(int,indices))),np.int64)


def public_service_scores(obs,bank,indices):
    """Current-demand, hypothetical-position surrogate, NOT a future label.

    Only the public union in obs is reconstructed, at its transmitted float32
    precision. Unknown request backing storage is never used. Solver calls are
    training overhead for the auxiliary method, and explicit runtime overhead
    for its myopic-service-greedy reference.
    """
    indices=np.asarray(indices,np.int64)
    if not bank['mask'][indices].all():raise ValueError('infeasible label action')
    nodes=np.asarray(obs['nodes']);ue=np.asarray(obs['ue'])[0];visible=np.asarray(obs['visible'],bool)[0]
    active=visible & (ue[:,2]>.5)
    public_positions=(ue[:,:2]+nodes[0,:2])*700
    requests=[Request(int(i),int(i),int(round(float(ue[i,3])*5e6)),int(round(float(ue[i,4])*14e9)),float(ue[i,5])) for i in np.flatnonzero(active)]
    positions=(nodes[:,:2]*700).astype(np.float64)
    scales=nodes[:,2].astype(np.float64)*2e11/config.UAV_COMPUTING_CAPACITY
    if not np.allclose(scales,scales[0],rtol=1e-5):raise ValueError('unregistered nonuniform capacity scaling')
    capacity=np.r_[nodes[:,2].astype(np.float64)*2e11,config.MBS_COMPUTING_CAPACITY*float(scales[0])]
    public=SimpleNamespace(ue_positions=public_positions,capacity=capacity)
    scores=[]
    from environment.dynamic_service import evaluate
    for i in indices:
        batch=DynamicServiceEnv.batch(public,positions=positions+bank['actions'][i]*15,requests=requests)
        scores.append(evaluate(batch,matching_search(batch))['success'])
    return np.asarray(scores,np.float32)
