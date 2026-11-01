"""Normalized service scoring and fixed-scale finite-horizon value fitting."""
import torch
from torch import nn
from marl_models.queued_service_joint import JointPolicy as FrozenPolicy,CandidateScore,tensor,update as frozen_update
from utils.queued_service_shared_budget import MOTION_DIM

VALUE_SCALE=100.


def normalized_candidates(features,mask):
    weights=mask.unsqueeze(-1).to(features.dtype);count=weights.sum(1,keepdim=True)
    mean=(features*weights).sum(1,keepdim=True)/count
    variance=((features-mean).square()*weights).sum(1,keepdim=True)/count
    return (features-mean)/variance.sqrt().clamp_min(.02)*weights


class BudgetMotion(CandidateScore):
    def __init__(self):super().__init__(MOTION_DIM)

    def forward(self,state,features,mask,prior=None):
        return super().forward(state,normalized_candidates(features,mask),mask,prior)


class ScaledValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(83,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        self.time_skip=nn.Linear(4,1)
        nn.init.orthogonal_(self.network[-1].weight,gain=.01);nn.init.zeros_(self.network[-1].bias)
        nn.init.zeros_(self.time_skip.weight);nn.init.zeros_(self.time_skip.bias)

    def forward(self,state):
        remaining=state[:,72:73]
        basis=torch.cat((torch.ones_like(remaining),remaining,remaining.square(),remaining.pow(3)),1)
        features=torch.cat((state,remaining.square(),remaining.pow(3),remaining.pow(4)),1)
        return VALUE_SCALE*(self.network(features)+self.time_skip(basis))


class BudgetPolicy(FrozenPolicy):
    def __init__(self,method):
        super().__init__(method);self.motion=BudgetMotion();self.critic=ScaledValue()

    def replay(self,rows):
        result=super().replay(rows);result['raw_value']=result['value'];result['value']=result['value']/VALUE_SCALE
        return result


def budget_update(model,optimizer,rows,epochs=2,minibatch=128):
    states=tensor([r['state'] for r in rows]);target=tensor([r['returns'] for r in rows])
    with torch.no_grad():before=model.critic(states).squeeze(-1)
    normalized=[dict(r,returns=r['returns']/VALUE_SCALE) for r in rows]
    metrics=frozen_update(model,optimizer,normalized,epochs,minibatch)
    with torch.no_grad():
        after=model.critic(states).squeeze(-1);variance=target.var(unbiased=False)
        metrics['critic_rmse_raw_before']=float((before-target).square().mean().sqrt())
        metrics['critic_rmse_raw_after']=float((after-target).square().mean().sqrt())
        metrics['critic_ev_mc_after']=float(1-(after-target).var(unbiased=False)/variance) if float(variance)>1e-12 else 0.
    return metrics
