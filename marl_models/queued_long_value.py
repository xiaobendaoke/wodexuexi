"""Centered long-return candidate values plus separate legal prefix baselines."""
import copy
import math
import torch
from torch import nn
from torch.distributions import Categorical
from marl_models.queued_service_joint import tensor
from marl_models.queued_service_shared_budget import normalized_candidates


class LongMotion(nn.Module):
    temperature=.5
    def __init__(self,legacy):
        super().__init__();self.legacy=legacy
        self.net=nn.Sequential(nn.Linear(80+119*3,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def values(self,state,features,mask,post_state,post_features):
        old=self.legacy(state,features[:,:,:86],mask,post_state,post_features[:,:,:86],True)
        w=mask.to(features.dtype);n=w.sum(1,keepdim=True)
        logits=old.logits.masked_fill(~mask,0);legacy=logits-(logits*w).sum(1,keepdim=True)/n
        raw=torch.asinh(post_features);relative=normalized_candidates(post_features,mask)
        delta=torch.asinh(post_features-features);s=state.unsqueeze(1).expand(-1,features.shape[1],-1)
        residual=5*self.net(torch.cat((s,raw,relative,delta),-1)).squeeze(-1)
        residual=residual-(residual*w).sum(1,keepdim=True)/n
        return (self.temperature*legacy+residual)*w

    def forward(self,state,features,mask,post_state=None,post_features=None,enabled=True):
        ps=state if post_state is None else post_state;pf=features if post_features is None else post_features
        q=self.values(state,features,mask,ps,pf)
        return Categorical(logits=(q/self.temperature).masked_fill(~mask,-1e9))


class SliceValue(nn.Module):
    def __init__(self,legacy):super().__init__();self.legacy=legacy
    def forward(self,state,features,mask,post_state=None,post_features=None):
        return self.legacy(state,features[:,:,:86],mask,post_state,None if post_features is None else post_features[:,:,:86])


class PrefixValue(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=copy.deepcopy(base)
        self.net=nn.Sequential(nn.Linear(110,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,state,features,mask):
        w=mask.unsqueeze(-1);mean=(features*w).sum(1)/w.sum(1)
        return self.base(state).squeeze(-1)+10*self.net(torch.cat((state,torch.asinh(mean)),1)).squeeze(-1)


def wrap(model):
    model.motion=LongMotion(model.motion);model.prefix_value=PrefixValue(model.critic.base)
    model.critic=SliceValue(model.critic);return model


def value_batch(model,samples):
    return model.motion.values(tensor([s['state'] for s in samples]),tensor([s['features'] for s in samples]),
        tensor([s['mask'] for s in samples],torch.bool),tensor([s['post_state'] for s in samples]),tensor([s['post_features'] for s in samples]))


def fit_values(model,samples,epochs=100):
    for p in model.parameters():p.requires_grad_(False)
    for p in model.motion.parameters():p.requires_grad_(True)
    optimizer=torch.optim.Adam(model.motion.parameters(),lr=.001);losses=[]
    for _ in range(epochs):
        for ids in torch.randperm(len(samples)).split(16):
            data=[samples[i] for i in ids.tolist()];pred=value_batch(model,data)
            target=torch.zeros_like(pred);mask=tensor([s['mask'] for s in data],torch.bool)
            for i,s in enumerate(data):
                q=tensor(s['values']);target[i,:len(q)]=q-q.mean()
            loss=nn.functional.smooth_l1_loss(pred[mask],target[mask]);assert bool(torch.isfinite(loss))
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.motion.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
    return dict(epochs=epochs,steps=len(losses),first_loss=losses[0],final_loss=losses[-1])


def prefix_values(model,rows):
    tokens=[t for r in rows for t in r['allocations']]
    if not tokens:return torch.empty(0),[],tokens
    owners=[i for i,r in enumerate(rows) for _ in r['allocations']]
    v=model.prefix_value(tensor([t['state'] for t in tokens]),tensor([t['features'] for t in tokens]),tensor([t['mask'] for t in tokens],torch.bool))
    return v,owners,tokens


def update_branches(model,rows,arm,actor_epochs=2,critic_epochs=2,optimizer_states=None):
    """Motion/individual-allocation PPO; no post-action allocation baseline."""
    active=('motion',) if arm=='motion' else ('allocation',) if arm=='allocation' else ('motion','allocation')
    for name in ('motion','allocation','critic','prefix_value'):
        for p in getattr(model,name).parameters():p.requires_grad_(name in active or name in ('critic','prefix_value'))
    before={name:{k:v.clone() for k,v in getattr(model,name).state_dict().items()} for name in ('motion','allocation')}
    optim={name:torch.optim.Adam(getattr(model,name).parameters(),lr=.0003) for name in active}
    critic_optim=torch.optim.Adam(list(model.critic.parameters())+list(model.prefix_value.parameters()),lr=.0003)
    optimizer_states={} if optimizer_states is None else dict(optimizer_states)
    for name,optimizer in optim.items():
        if name in optimizer_states:optimizer.load_state_dict(optimizer_states[name])
    if 'critic' in optimizer_states:critic_optim.load_state_dict(optimizer_states['critic'])
    target=tensor([r['returns'] for r in rows]);ma=target-tensor([r['motion_value'] for r in rows]);ma=(ma-ma.mean())/ma.std(unbiased=False).clamp_min(.05)
    oldmotion=tensor([r['motion_log'] for r in rows]);steps={name:0 for name in active};skips={name:0 for name in active};csteps=0
    replay_error=0.
    with torch.no_grad():
        for start in range(0,len(rows),256):
            selected=rows[start:start+256];z=model.replay(selected)
            replay_error=max(replay_error,float((z['motion_log']-oldmotion[start:start+256]).abs().max()),
                float((z['allocation_log']-tensor([r['allocation_log'] for r in selected])).abs().max()))
        assert replay_error<2e-4,replay_error
        tokens=[t for r in rows for t in r['allocations']];owners=[i for i,r in enumerate(rows) for _ in r['allocations']]
        # Saved prefix values cannot use this token's sampled allocation.
        aa=tensor([rows[o]['returns']-t['prefix_value'] for o,t in zip(owners,tokens)]) if tokens else torch.empty(0)
        aa_mean=aa.mean() if tokens else torch.tensor(0.)
        aa_scale=aa.std(unbiased=False).clamp_min(.05) if tokens else torch.tensor(1.)
    for epoch in range(max(actor_epochs,critic_epochs)):
        for ids in torch.randperm(len(rows)).split(256):
            selected=[rows[i] for i in ids.tolist()];z=model.replay(selected)
            losses={}
            if 'motion' in active:
                ratio=(z['motion_log']-oldmotion[ids]).clamp(-20,20).exp();a=ma[ids]
                losses['motion']=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()-.001*z['motion_entropy'].mean()
            pv,owner,local=prefix_values(model,selected)
            if 'allocation' in active and local:
                dist=model.allocation(tensor([t['state'] for t in local]),tensor([t['features'] for t in local]),tensor([t['mask'] for t in local],torch.bool),tensor([t['prior'] for t in local]))
                old=tensor([math.log(max(t['probs'][t['action']],1e-30)) for t in local]);log=dist.log_prob(torch.tensor([t['action'] for t in local]))
                # Prefix baselines are stored per token at rollout time.
                a=tensor([selected[o]['returns']-t['prefix_value'] for o,t in zip(owner,local)])
                a=(a-aa_mean)/aa_scale
                ratio=(log-old).clamp(-20,20).exp();terms=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)
                sums=torch.zeros(len(selected)).index_add(0,torch.tensor(owner),terms)
                losses['allocation']=sums.mean()-.001*dist.entropy().mean()
            if epoch<actor_epochs:
                for name,loss in losses.items():
                    kl=z['motion_kl'].mean() if name=='motion' else z['allocation_kl'].mean()
                    if float(kl.detach())>.02:skips[name]+=1;continue
                    optim[name].zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(getattr(model,name).parameters(),.5,error_if_nonfinite=True);optim[name].step();steps[name]+=1
            if epoch<critic_epochs:
                critic=((z['motion_value']-target[ids])/100).square().mean()
                if local:critic=critic+((pv-target[ids][torch.tensor(owner)])/100).square().mean()
                critic_optim.zero_grad();critic.backward();nn.utils.clip_grad_norm_(list(model.critic.parameters())+list(model.prefix_value.parameters()),.5,error_if_nonfinite=True);critic_optim.step();csteps+=1
    changes={name:sum(float((v-before[name][k]).square().sum()) for k,v in getattr(model,name).state_dict().items())**.5 for name in before}
    for name in ('motion','allocation'):
        if name not in active:assert changes[name]==0
    for name,optimizer in optim.items():optimizer_states[name]=optimizer.state_dict()
    optimizer_states['critic']=critic_optim.state_dict()
    return dict(actor_steps=steps,actor_skips=skips,critic_steps=csteps,parameter_changes=changes,replay_error=replay_error,_optimizer_states=optimizer_states)
