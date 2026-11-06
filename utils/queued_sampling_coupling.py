"""Categorical Gumbel coupling keyed by request and motion semantics.

    This changes coupling between simulator forks, not each categorical law.
"""
import hashlib
import math
from collections import defaultdict
from environment.queued_service import distance,safe_path


def gumbel_choice(probabilities,labels,key):
    if len(probabilities)!=len(labels) or len(labels)!=len(set(labels)):raise ValueError('unique categorical labels required')
    values=[]
    for p,label in zip(probabilities,labels):
        raw=int.from_bytes(hashlib.sha256(repr((key,label)).encode()).digest()[:8],'big')>>11
        u=min(1-2**-53,max(2**-54,(raw+.5)/2**53))
        values.append(math.log(p)-math.log(-math.log(u)) if p>0 else -math.inf)
    if max(values)==-math.inf:raise ValueError('positive categorical mass required')
    return max(range(len(values)),key=values.__getitem__)


def semantic_labels(obs,weights,bank):
    p=obs['protocol'];pos=obs['positions'];n=len(pos);grouped=defaultdict(list);named={}
    def remember(move,name):named.setdefault(tuple(move),name)
    zero=[(0.,0.)]*n;remember(zero,'hover')
    for u,xy in obs['ues'].items():
        w=weights.get(u,0.)
        if w>1e-8:grouped[min(range(n),key=lambda i:distance(xy,pos[i]))].append((xy,w))
    desired=[]
    for i,a in enumerate(pos):
        xs=grouped[i];total=sum(w for _,w in xs)
        target=tuple(sum(x[k]*w for x,w in xs)/total for k in range(2)) if total else a
        delta=(target[0]-a[0],target[1]-a[1]);scale=min(1.,p.speed/max(math.hypot(*delta),1e-12));desired.append(tuple(x*scale for x in delta))
    for fraction in (1.,.5):
        move=[(dx*fraction,dy*fraction) for dx,dy in desired]
        if safe_path(pos,move,p):remember(move,f'history_team_{fraction}')
    for i,d in enumerate(desired):
        move=list(zero);move[i]=d
        if safe_path(pos,move,p):remember(move,f'history_single_{i}')
    def toward(i,targets):
        total=sum(w for _,w in targets)
        if not total:return (0.,0.)
        target=tuple(sum(x[k]*w for x,w in targets)/total for k in range(2));delta=tuple(target[k]-pos[i][k] for k in range(2))
        scale=min(1.,p.speed/max(math.hypot(*delta),1e-12));return tuple(x*scale for x in delta)
    upload=[];forward=[]
    for i in range(n):
        tasks=[x for x in obs['tasks'] if x['source']==i and x['remaining_upload']>0]
        upload.append(toward(i,[(obs['ues'][x['ue']],x['remaining_upload']/max(x['deadline']-obs['time'],.1)) for x in tasks]))
        tasks=[x for x in obs['tasks'] if x['source']==i and x['target'] not in (None,i) and x['remaining_transfer']>0]
        targets=[(p.mbs if x['target']==n else pos[x['target']],x['remaining_transfer']/max(x['deadline']-obs['time'],.1)) for x in tasks]
        forward.append((sum(w for _,w in targets),i,toward(i,targets)))
    proposals=[(upload,'upload_team')]
    for i,d in enumerate(upload):move=list(zero);move[i]=d;proposals.append((move,f'upload_single_{i}'))
    for _,i,d in sorted(forward,reverse=True)[:2]:move=list(zero);move[i]=d;proposals.append((move,f'forward_{i}'))
    for move,name in proposals:
        for fraction in (1.,.5,.25):
            action=[(dx*fraction,dy*fraction) for dx,dy in move]
            if safe_path(pos,action,p):remember(action,f'{name}_{fraction}');break
    labels=[named[tuple(a)] for a in bank]
    assert len(labels)==len(set(labels));return labels


def act_coupled(model,obs,adapter,key):
    import torch
    from marl_models.queued_service_joint import tensor
    from utils.queued_service_learning import VirtualQueue
    with torch.no_grad():
        adapter.prepare(obs);prefix=VirtualQueue(obs);plan={};weights=adapter.history.weights
        for x in obs['tasks']:
            if x['source'] is None or x['target'] is not None:continue
            state,features,mask,prior=prefix.features(x['uid'],weights)
            dist=model.allocation(tensor([state]),tensor([features]),tensor([mask],torch.bool),tensor([prior]))
            a=gumbel_choice(dist.probs[0].tolist(),tuple(range(6)),(key,'request',x['uid']))
            plan[x['uid']]=a;prefix.assign(x['uid'],a)
        pre=adapter.motion(obs);post=adapter.motion(prefix.obs)
        dist=model.motion(tensor([pre['state']]),tensor([pre['features']]),tensor([pre['mask']],torch.bool),tensor([post['state']]),tensor([post['features']]),True)
        labels=semantic_labels(obs,weights,pre['actions'])
        action=gumbel_choice(dist.probs[0,:len(labels)].tolist(),labels,(key,'motion',obs['time']))
        return plan,pre['actions'][action]
