"""Explicit pre-motion bypass and action-independent sampled-plan baseline."""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from marl_models.queued_service_shared_budget import BudgetPolicy,BudgetMotion,ScaledValue,normalized_candidates,tensor
from marl_models.queued_service_joint import CandidateScore
from utils.queued_service_learning import VirtualQueue,state_features
from utils.queued_service_plan_residual import phase_keys


class PlanMotion(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.residual=CandidateScore(86*2+80)
        nn.init.zeros_(self.residual.score[-1].weight);nn.init.zeros_(self.residual.score[-1].bias)
        self.gate=nn.Parameter(torch.tensor(math.atanh(.1)))

    def forward(self,state,features,mask,post_state=None,post_features=None,enabled=True):
        base=self.base(state,features,mask)
        if not enabled or post_state is None:return base
        pre=normalized_candidates(features,mask);post=normalized_candidates(post_features,mask)
        delta=(post_state-state).unsqueeze(1).expand(-1,features.shape[1],-1)
        extra=self.residual(state,torch.cat((pre,post-pre,delta),-1),mask)
        return Categorical(logits=(base.logits+self.gate.tanh()*extra.logits).masked_fill(~mask,-1e9))


def pool(features,mask):
    w=mask.unsqueeze(-1);mean=(features*w).sum(1)/w.sum(1)
    low=features.masked_fill(~w,float('inf')).amin(1);high=features.masked_fill(~w,-float('inf')).amax(1)
    return torch.cat((mean,low,high),-1)


class PlanValue(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        def network(d):
            z=nn.Sequential(nn.Linear(d,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
            nn.init.zeros_(z[-1].weight);nn.init.zeros_(z[-1].bias);return z
        self.spatial=network(80+258);self.plan=network(80+258+80+258)

    def forward(self,state,features,mask,post_state=None,post_features=None):
        pre=torch.cat((state,pool(features,mask)),1)
        v=self.base(state).squeeze(-1)+10*self.spatial(pre).squeeze(-1)
        if post_state is None:return v
        delta=torch.cat((post_state-state,pool(post_features,mask)-pool(features,mask)),1)
        return v,v+10*self.plan(torch.cat((pre,delta),1)).squeeze(-1)


class ResidualPolicy(BudgetPolicy):
    def __init__(self,method):
        super().__init__(method);self.motion=PlanMotion(self.motion);self.critic=PlanValue(self.critic)

    @torch.no_grad()
    def act(self,obs,adapter,seed):
        adapter.prepare(obs);weights=adapter.history.weights;before=state_features(obs,weights)
        prefix=VirtualQueue(obs);plan={};tokens=[];generator=torch.Generator().manual_seed(int(seed));allocation_log=0.
        for task in obs['tasks']:
            if task['source'] is None or task['target'] is not None:continue
            state,features,mask,prior=prefix.features(task['uid'],weights)
            dist=self.allocation(tensor([state]),tensor([features]),tensor([mask],torch.bool),tensor([prior]))
            action=int(torch.multinomial(dist.probs[0],1,generator=generator));allocation_log+=float(dist.log_prob(torch.tensor([action]))[0])
            tokens.append(dict(state=state,features=features,mask=mask,prior=prior,action=action,probs=dist.probs[0].tolist()))
            plan[task['uid']]=action;prefix.assign(task['uid'],action)
        pre=adapter.motion(obs);post=adapter.motion(prefix.obs)
        assert pre['actions']==post['actions'] and pre['mask']==post['mask']
        s=tensor([pre['state']]);f=tensor([pre['features']]);mask=tensor([pre['mask']],torch.bool)
        ps=tensor([post['state']]);pf=tensor([post['features']])
        value,motion_value=self.critic(s,f,mask,ps,pf)
        dist=self.motion(s,f,mask,ps,pf,self.method=='joint')
        action=int(torch.multinomial(dist.probs[0],1,generator=generator))
        row=dict(state=before,decision_time=obs['time'],allocations=tokens,allocation_log=allocation_log,motion_state=pre['state'],motion_features=pre['features'],
            post_state=post['state'],post_features=post['features'],motion_mask=pre['mask'],motion_action=action,
            motion_probs=dist.probs[0].tolist(),motion_log=float(dist.log_prob(torch.tensor([action]))[0]),
            value=float(value[0]),motion_value=float(motion_value[0]),history_weights=dict(weights))
        return row,plan,pre['actions'][action]

    def replay(self,rows):
        n=len(rows);owners=[];tokens=[]
        for i,r in enumerate(rows):owners += [i]*len(r['allocations']);tokens+=r['allocations']
        logs=torch.zeros(n);entropy=torch.zeros(n);kl=torch.zeros(n);counts=torch.zeros(n)
        if tokens:
            owner=torch.tensor(owners)
            dist=self.allocation(tensor([x['state'] for x in tokens]),tensor([x['features'] for x in tokens]),tensor([x['mask'] for x in tokens],torch.bool),tensor([x['prior'] for x in tokens]))
            old=tensor([x['probs'] for x in tokens]);div=(old*(old.clamp_min(1e-30).log()-dist.probs.clamp_min(1e-30).log())).sum(-1)
            logs=logs.index_add(0,owner,dist.log_prob(torch.tensor([x['action'] for x in tokens])))
            entropy=entropy.index_add(0,owner,dist.entropy());kl=kl.index_add(0,owner,div);counts=counts.index_add(0,owner,torch.ones(len(owner)))
        state=tensor([r['motion_state'] for r in rows]);features=tensor([r['motion_features'] for r in rows]);mask=tensor([r['motion_mask'] for r in rows],torch.bool)
        ps=tensor([r['post_state'] for r in rows]);pf=tensor([r['post_features'] for r in rows])
        motion=self.motion(state,features,mask,ps,pf,self.method=='joint');old=tensor([r['motion_probs'] for r in rows])
        v,mv=self.critic(state,features,mask,ps,pf)
        return dict(allocation_log=logs,allocation_entropy=entropy/counts.clamp_min(1),allocation_count=counts,allocation_kl=kl,
            motion_log=motion.log_prob(torch.tensor([r['motion_action'] for r in rows])),motion_entropy=motion.entropy(),
            motion_kl=(old*(old.clamp_min(1e-30).log()-motion.probs.clamp_min(1e-30).log())).sum(-1),value=v,motion_value=mv)


def centered_loss(prediction,target,keys):
    ids=torch.tensor(keys);residual=prediction-target;loss=residual.sum()*0;count=0
    for key in ids.unique():
        selected=residual[ids==key]
        if len(selected)>1:loss=loss+(selected-selected.mean()).square().sum();count+=len(selected)
    return loss/max(count,1)


def residual_update(model,optimizer,rows,epochs=2,minibatch=128):
    from utils.queued_service_diagnostics import explained_variance
    advantages={name:tensor([r['returns']-r[key] for r in rows]) for name,key in (('allocation','value'),('motion','motion_value'))}
    for name,a in advantages.items():advantages[name]=(a-a.mean())/a.std(unbiased=False).clamp_min(.05)
    old={name:tensor([r[name+'_log'] for r in rows]) for name in ('allocation','motion')}
    before={name:{k:v.clone() for k,v in getattr(model,name).state_dict().items()} for name in ('allocation','motion','critic')}
    metrics={k:0. for k in ('allocation_grad','motion_grad','critic_loss','phase_loss','allocation_loss','motion_loss','allocation_entropy','motion_entropy')}
    metrics.update(optimizer_steps=0,actor_steps=0,actor_kl_skips=0)
    for _ in range(epochs):
        for ids in torch.randperm(len(rows)).split(minibatch):
            selected=[rows[i] for i in ids.tolist()];z=model.replay(selected);target=tensor([r['returns'] for r in selected]);keys=phase_keys(selected)
            losses={}
            for name in ('allocation','motion'):
                ratio=(z[name+'_log']-old[name][ids]).clamp(-20,20).exp();adv=advantages[name][ids]
                loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
                valid=z['allocation_count']>0 if name=='allocation' else torch.ones_like(loss,dtype=torch.bool)
                losses[name]=loss[valid].mean() if bool(valid.any()) else loss.sum()*0
            raw=sum(((z[key]-target)/100).square().mean() for key in ('value','motion_value'))/2
            phase=sum(centered_loss(z[key]/10,target/10,keys) for key in ('value','motion_value'))/2
            critic=raw+phase;accepted=float((z['allocation_kl']+z['motion_kl']).mean().detach())<=.05
            actor=sum(losses.values())-.001*(z['allocation_entropy'].mean()+z['motion_entropy'].mean())
            loss=.5*critic+(actor if accepted else actor*0)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite plan residual update')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if not accepted:
                for module in (model.allocation,model.motion):
                    for parameter in module.parameters():parameter.grad=None
            norms={name:float(nn.utils.clip_grad_norm_(getattr(model,name).parameters(),.5,error_if_nonfinite=True)) for name in ('allocation','motion','critic')}
            optimizer.step();metrics['optimizer_steps']+=1;metrics['actor_steps']+=int(accepted);metrics['actor_kl_skips']+=int(not accepted)
            values=dict(allocation_grad=norms['allocation'],motion_grad=norms['motion'],critic_loss=critic,phase_loss=phase,
                allocation_loss=losses['allocation'],motion_loss=losses['motion'],allocation_entropy=z['allocation_entropy'].mean(),motion_entropy=z['motion_entropy'].mean())
            for key,value in values.items():metrics[key]+=float(value)
    for key in values:metrics[key]/=metrics['optimizer_steps']
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(rows),minibatch):outputs.append(model.replay(rows[start:start+minibatch]))
    metrics['joint_conditional_kl']=float(torch.cat([x['allocation_kl']+x['motion_kl'] for x in outputs]).mean())
    target=[r['returns'] for r in rows]
    for key in ('value','motion_value'):
        prediction=torch.cat([x[key] for x in outputs]).tolist();metrics[key+'_ev_mc_after']=explained_variance(target,prediction)
    metrics['critic_ev_mc_after']=metrics['value_ev_mc_after']
    metrics['critic_rmse_raw_after']=math.sqrt(np.mean((np.array(target)-np.array(torch.cat([x['value'] for x in outputs]).tolist()))**2))
    for name in before:metrics[name+'_parameter_change']=math.sqrt(sum(float((v-before[name][k]).square().sum()) for k,v in getattr(model,name).state_dict().items()))
    metrics['plan_parameter_change']=math.sqrt(sum(float((v-before['motion']['residual.'+k]).square().sum()) for k,v in model.motion.residual.state_dict().items()))
    metrics['gate']=float(model.motion.gate.tanh().detach())
    if not all(math.isfinite(x) for x in metrics.values()):raise FloatingPointError('nonfinite plan residual metrics')
    return metrics
