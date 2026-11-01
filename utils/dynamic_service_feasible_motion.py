"""Goal-independent joint motion templates and public-demand candidate features."""
import itertools
import numpy as np

FEATURE_DIM=43
SAFETY_MARGIN_M=.002


def _templates():
    directions=np.c_[np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)]
    plans=[np.zeros((5,2),np.float64)]
    groups=[tuple(range(5))]+[(i,) for i in range(5)]+list(itertools.combinations(range(5),2))
    for group in groups:
        for speed in (.5,1.):
            for direction in directions:
                action=np.zeros((5,2),np.float64);action[list(group)]=speed*direction;plans.append(action)
    return np.asarray(plans)


TEMPLATES=_templates()


def motion_library(obs):
    """257 fixed plans; no matching, reward search, hotspot or future access.

    A margin absorbs float32 public-coordinate reconstruction. Pairs with the
    same displacement preserve their original separation exactly, including
    near-contact hovering/translation. The original environment remains the
    final authority for speed, boundary, and separation constraints.
    """
    nodes=np.asarray(obs['nodes']);positions=np.asarray(nodes[:,:2]*700,dtype=np.float64)
    actions=TEMPLATES.copy();delta=actions*15;after=positions[None]+delta
    moved=np.abs(delta)>1e-8
    safe_axes=np.where(moved,(after>=50+SAFETY_MARGIN_M)&(after<=650-SAFETY_MARGIN_M),True)
    legal=safe_axes.all((1,2))
    distance=np.linalg.norm(after[:,:,None]-after[:,None,:],axis=-1)
    for i in range(5):
        for j in range(i+1,5):
            identical=np.linalg.norm(delta[:,i]-delta[:,j],axis=1)<1e-10
            legal&=identical | (distance[:,i,j]>=200+SAFETY_MARGIN_M)
    legal[0]=True
    # Public union only. Unknown records are excluded even if their backing
    # storage contains arbitrary values.
    visible=np.asarray(obs['visible'],dtype=bool)[0]
    ue=np.asarray(obs['ue'])[0];active=visible & (ue[:,2]>.5)
    public_ue=(ue[:,:2]+nodes[0,:2])*700
    if active.any():
        d=np.linalg.norm(after[:,:,None]-public_ue[active][None,None],axis=-1)
        cover=d<=100
        count=cover.sum(-1)/100
        cycles=ue[active,4]*14e9;deadline=np.maximum(ue[active,5],.05)
        capacity=np.asarray(nodes[:,2],np.float64)*2e11
        pressure=np.clip((cover*(cycles/deadline)[None,None]).sum(-1)/capacity[None]/10,0,1)
        nearest=np.clip(d.min(-1)/700,0,1)
        union=cover.any(1).sum(-1)/100
        old=np.linalg.norm(positions[:,None]-public_ue[active][None],axis=-1)
        current=(old<=100).any(0).sum()/100
    else:
        count=pressure=np.zeros((len(actions),5));nearest=np.ones((len(actions),5))
        union=np.zeros(len(actions));current=0.
    margin=np.minimum(after-50,650-after).min(-1)/300
    pairs=distance+np.eye(5)[None]*1e6
    min_pair=np.clip((pairs.min((1,2))-200)/200,-1,1)
    features=np.c_[actions.reshape(len(actions),-1),after.reshape(len(actions),-1)/700,
        count,pressure,nearest,np.clip(margin,0,1),union,union-current,min_pair].astype(np.float32)
    assert features.shape==(257,FEATURE_DIM)
    return dict(actions=actions,features=features,mask=legal)
