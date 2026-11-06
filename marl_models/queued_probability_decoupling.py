"""Frozen task-count value scorer and independent dimensionless policy residual."""
import torch
from torch import nn
from torch.distributions import Categorical
from marl_models.queued_service_shared_budget import normalized_candidates


def normalized_scores(q,mask):
    w=mask.to(q.dtype);n=w.sum(-1,keepdim=True);center=(q-(q*w).sum(-1,keepdim=True)/n)*w
    scale=(center.square().sum(-1,keepdim=True)/n).sqrt()
    return center/scale.clamp_min(torch.finfo(q.dtype).tiny)


class DecoupledMotion(nn.Module):
    def __init__(self,value_scorer,normalize,temperature):
        super().__init__();self.value_scorer=value_scorer;self.normalize=normalize
        self.register_buffer('policy_temperature',torch.tensor(float(temperature)))
        self.policy_net=nn.Sequential(nn.Linear(80+119*3,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.policy_net[-1].weight);nn.init.zeros_(self.policy_net[-1].bias)

    @torch.no_grad()
    def values(self,state,features,mask,post_state,post_features):
        # update_motion enables motion parameters globally; this explicit graph
        # boundary ensures the value scorer receives no PPO gradients or Adam steps.
        return self.value_scorer.values(state,features,mask,post_state,post_features)

    def forward(self,state,features,mask,post_state=None,post_features=None,enabled=True):
        ps=state if post_state is None else post_state;pf=features if post_features is None else post_features
        q=self.values(state,features,mask,ps,pf)
        prior=(normalized_scores(q,mask) if self.normalize else q)/self.policy_temperature
        s=state.unsqueeze(1).expand(-1,features.shape[1],-1)
        x=torch.cat((s,torch.asinh(pf),normalized_candidates(pf,mask),torch.asinh(pf-features)),-1)
        residual=self.policy_net(x).squeeze(-1);w=mask.to(q.dtype)
        residual=residual-(residual*w).sum(-1,keepdim=True)/w.sum(-1,keepdim=True)
        return Categorical(logits=(prior+residual).masked_fill(~mask,-1e9))
