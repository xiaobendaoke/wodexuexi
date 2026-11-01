"""Feasible categorical motion, consistent sampling and decomposed service credit."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from marl_models.dynamic_service_joint import DemandEncoder,AllocationHead,allocation_summary
from marl_models.dynamic_service_hmarl import tensor
from utils.dynamic_service_feasible_motion import motion_library,FEATURE_DIM
from utils.dynamic_service_joint_rollout import causal_returns,slot_surrogate

METHODS=('feasible_trajectory','local_allocation','repair_joint','repair_full_credit','repair_no_plan')


def draw(dist,generator,greedy=False):
    return dist.logits.argmax(-1) if greedy else torch.multinomial(dist.probs,1,generator=generator).squeeze(-1)


def normalized_plan(batch,assignment):
    x=allocation_summary(batch,assignment).reshape(6,3).copy()
    x[:,1]=np.clip(x[:,1]/5,0,1)
    return x.ravel()


class RepairedPolicy(nn.Module):
    def __init__(self,method='repair_joint'):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method;self.learn_motion=method!='local_allocation';self.learn_allocation=method!='feasible_trajectory'
        self.use_plan=method not in ('repair_no_plan','feasible_trajectory')
        self.full_credit=method=='repair_full_credit'
        self.encoder=DemandEncoder();self.allocation=AllocationHead()
        self.candidate=nn.Sequential(nn.Linear(FEATURE_DIM,128),nn.Tanh())
        self.context=nn.Sequential(nn.Linear(658,128),nn.Tanh())
        self.bias=nn.Sequential(nn.Linear(FEATURE_DIM,64),nn.Tanh(),nn.Linear(64,1))
        self.score_scale=nn.Parameter(torch.tensor(.01))
        nn.init.zeros_(self.bias[-1].weight);nn.init.zeros_(self.bias[-1].bias)

    @property
    def device(self):return next(self.parameters()).device
    def initial_state(self):return torch.zeros(5,128,device=self.device)

    def motion_distribution(self,state,batch,assignment,bank):
        plan=normalized_plan(batch,assignment) if self.use_plan else np.zeros(18,np.float32)
        context=self.context(torch.cat((state.flatten(),tensor(plan,self.device))))
        features=tensor(bank['features'],self.device)
        logits=self.score_scale*(self.candidate(features)*context).sum(-1)/math.sqrt(128)+self.bias(features).squeeze(-1)
        mask=torch.as_tensor(bank['mask'],dtype=torch.bool,device=self.device)
        return Categorical(logits=logits.masked_fill(~mask,-torch.inf))

    @torch.no_grad()
    def act(self,obs,batch,hidden,seed,greedy=False,fixed_assignment=None):
        ga=torch.Generator().manual_seed(int(seed));gm=torch.Generator().manual_seed((int(seed)+1000003)%(2**63-1))
        state=self.encoder(obs,hidden);chosen=[];risks=[];logs=[];probs=[]
        encoded=self.allocation.encode_context(batch,state) if len(batch) and fixed_assignment is None else None
        for i in range(len(batch) if fixed_assignment is None else 0):
            risk=self.allocation.dynamic_features(batch,chosen,i);r,n,e=encoded
            logits=self.allocation.score(torch.cat((r[i].expand(6,-1),n,e[i],risk),-1)).squeeze(-1)+self.allocation.risk_score(risk).squeeze(-1)
            dist=Categorical(logits=logits.masked_fill(~torch.as_tensor(batch.legal[i],device=self.device),-torch.inf))
            action=draw(dist,ga,greedy);chosen.append(int(action));risks.append(risk.numpy());logs.append(float(dist.log_prob(action)));probs.append(dist.probs.numpy())
        assignment=np.asarray(chosen if fixed_assignment is None else fixed_assignment,np.int64)
        if fixed_assignment is not None:
            risks=self.allocation.risk_trajectory(batch,assignment).numpy() if len(batch) else np.empty((0,6,11),np.float32)
        bank=motion_library(obs);dist=self.motion_distribution(state,batch,assignment,bank);index=draw(dist,gm,greedy)
        return dict(assignment=assignment,risk=np.asarray(risks,np.float32).reshape(-1,6,11),
            allocation_log=np.asarray(logs,np.float32),allocation_probs=np.asarray(probs,np.float32).reshape(-1,6),
            motion_index=int(index),motion_log=float(dist.log_prob(index)),motion_probs=dist.probs.numpy(),
            bank=bank,actions=bank['actions'][int(index)].copy(),hidden=state.numpy())

    def replay(self,obs,batch,hidden,sample):
        state=self.encoder(obs,hidden)
        if len(batch) and self.learn_allocation:
            dist=self.allocation.distribution_context(batch,self.allocation.encode_context(batch,state),tensor(sample['risk'],self.device))
            allocation_log=dist.log_prob(torch.as_tensor(sample['assignment'],device=self.device))
            allocation_entropy=dist.entropy();allocation_probs=dist.probs
        else:
            allocation_log=allocation_entropy=torch.empty(0,device=self.device);allocation_probs=torch.empty(0,6,device=self.device)
        dist=self.motion_distribution(state,batch,sample['assignment'],sample['bank'])
        return dict(hidden=state,allocation_log=allocation_log,allocation_entropy=allocation_entropy,allocation_probs=allocation_probs,
            motion_log=dist.log_prob(torch.tensor(sample['motion_index'],device=self.device)),motion_probs=dist.probs,motion_entropy=dist.entropy())


class RepairedCritic(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=DemandEncoder();self.batch_encoder=AllocationHead()
        self.prefix=nn.Sequential(nn.Linear(453,128),nn.Tanh(),nn.Linear(128,2))
        self.movement=nn.Sequential(nn.Linear(147,128),nn.Tanh(),nn.Linear(128,1))

    @property
    def device(self):return next(self.parameters()).device
    def initial_state(self):return torch.zeros(5,128,device=self.device)

    def forward(self,obs,batch,hidden,sample,time):
        state=self.encoder(obs,hidden);pooled=state.mean(0);n=len(batch)
        remaining=tensor([(1000-time)/1000],self.device)
        if n:
            r,nodes,_=self.batch_encoder.encode_context(batch,state)
            rest=r.flip(0).cumsum(0).flip(0)/torch.arange(n,0,-1,device=self.device)[:,None]
            progress=tensor(np.c_[np.arange(n)/n,np.full(n,n/100)],self.device)
            x=torch.cat((pooled.expand(n,-1),rest,nodes.mean(0).expand(n,-1),tensor(sample['risk'],self.device).flatten(1),progress,remaining.expand(n,1)),-1)
            raw=self.prefix(x);current=raw[:,0];total=raw[:,1]*100
        else:current=total=torch.empty(0,device=self.device)
        move=self.movement(torch.cat((pooled,tensor(normalized_plan(batch,sample['assignment']),self.device),remaining))).squeeze(-1)*100
        return current,total,move,state

    @torch.no_grad()
    def predict(self,obs,batch,hidden,sample,time):
        current,total,move,state=self(obs,batch,hidden,sample,time)
        return dict(batch_values=current.numpy(),prefix_values=total.numpy(),motion_value=float(move),critic_hidden=state.numpy())


def conditional_kl(actor,rows,start,end,burn_in=16):
    burn=max(0,start-burn_in);hidden=tensor(rows[burn]['hidden_before'],actor.device);values=[]
    with torch.no_grad():
        for t in range(burn,start):hidden=actor.encoder(rows[t]['observation'],hidden)
        for row in rows[start:end]:
            pred=actor.replay(row['observation'],row['batch'],hidden,row);hidden=pred['hidden'];kl=0.
            if actor.learn_allocation and len(row['batch']):
                old=tensor(row['allocation_probs'],actor.device)
                kl+=float((old*(old.clamp_min(1e-30).log()-pred['allocation_probs'].clamp_min(1e-30).log())).sum())
            if actor.learn_motion:
                old=tensor(row['motion_probs'],actor.device)
                kl+=float((old*(old.clamp_min(1e-30).log()-pred['motion_probs'].clamp_min(1e-30).log())).sum())
            values.append(max(kl,0.))
    return float(np.mean(values))


def update_repaired(actor,critic,aopt,copt,rows,sequence=64,epochs=4,target_kl=.05,progress=None):
    targets=causal_returns(rows)
    metrics=dict(actor_steps=0,critic_steps=0,actor_loss=0.,critic_loss=0.,backtracking_steps=0,rejected_updates=0,
        max_accepted_conditional_kl=0.,stopped_kl=False,allocation_credit='long_term' if actor.full_credit else 'current_batch',
        allocation_advantage_std=0.,motion_advantage_std=0.)
    # A common fixed scale preserves the relative service objective. No token
    # averaging or separate per-branch standardization is used.
    all_a=[];all_m=[]
    for row,target in zip(rows,targets):
        av=target['prefix_targets']-row['prefix_values'] if actor.full_credit else row['reward']-row['batch_values']
        if actor.learn_allocation:all_a.extend(np.asarray(av).tolist())
        if actor.learn_motion:all_m.append(target['motion_target']-row['motion_value'])
    scale=max(float(np.std(all_a+all_m)),.05)
    metrics['allocation_advantage_std']=float(np.std(all_a)) if all_a else 0.
    metrics['motion_advantage_std']=float(np.std(all_m)) if all_m else 0.
    starts=list(range(0,len(rows),sequence));stopped=False
    for epoch in range(epochs):
        for index in torch.randperm(len(starts)).tolist():
            start=starts[index];end=min(start+sequence,len(rows));burn=max(0,start-16)
            ah=tensor(rows[burn]['hidden_before'],actor.device);ch=tensor(rows[burn]['critic_hidden_before'],critic.device)
            with torch.no_grad():
                for t in range(burn,start):
                    ah=actor.encoder(rows[t]['observation'],ah);ch=critic.encoder(rows[t]['observation'],ch)
            actor_terms=[];critic_terms=[]
            for t in range(start,end):
                row=rows[t];target=targets[t]
                pred=actor.replay(row['observation'],row['batch'],ah,row);ah=pred['hidden']
                cur,total,move,ch=critic(row['observation'],row['batch'],ch,row,row['time'])
                zero=pred['motion_log']*0;loss=zero;entropy=zero;v=[]
                if len(row['batch']):
                    v.extend([(cur-row['reward']).square().mean(),((total-tensor(target['prefix_targets'],critic.device))/100).square().mean()])
                v.append(((move-target['motion_target'])/100).square())
                if v:critic_terms.append(torch.stack(v).mean())
                if actor.learn_allocation and len(row['batch']):
                    adv=(target['prefix_targets']-row['prefix_values']) if actor.full_credit else row['reward']-row['batch_values']
                    loss=loss+slot_surrogate(pred['allocation_log'],tensor(row['allocation_log'],actor.device),tensor(adv/scale,actor.device))
                    entropy=entropy+pred['allocation_entropy'].sum()
                if actor.learn_motion and t<len(rows)-1:
                    loss=loss+slot_surrogate(pred['motion_log'],tensor(row['motion_log'],actor.device),
                        tensor((target['motion_target']-row['motion_value'])/scale,actor.device))
                    entropy=entropy+pred['motion_entropy']
                actor_terms.append(loss-.001*entropy)
            if critic_terms:
                loss_c=torch.stack(critic_terms).mean()
                if not torch.isfinite(loss_c):raise FloatingPointError('critic loss')
                copt.zero_grad(set_to_none=True);loss_c.backward()
                nn.utils.clip_grad_norm_(critic.parameters(),.5,error_if_nonfinite=True);copt.step()
                metrics['critic_steps']+=1;metrics['critic_loss']+=float(loss_c.detach())
            before_kl=conditional_kl(actor,rows,start,end)
            if before_kl>target_kl or not np.isfinite(before_kl):stopped=True;metrics['stopped_kl']=True
            if not stopped:
                loss_a=torch.stack(actor_terms).mean()
                if not torch.isfinite(loss_a):raise FloatingPointError('actor loss')
                aopt.zero_grad(set_to_none=True);loss_a.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),.5,error_if_nonfinite=True)
                old={k:v.clone() for k,v in actor.state_dict().items()};optimizer_before=copy.deepcopy(aopt.state_dict())
                aopt.step();proposed={k:v.clone() for k,v in actor.state_dict().items()};accepted=False
                for attempt in range(6):
                    if attempt:
                        alpha=.5**attempt
                        actor.load_state_dict({k:old[k]+alpha*(proposed[k]-old[k]) for k in old})
                    kl=conditional_kl(actor,rows,start,end)
                    if np.isfinite(kl) and kl<=target_kl+1e-8:
                        accepted=True;metrics['max_accepted_conditional_kl']=max(metrics['max_accepted_conditional_kl'],kl)
                        metrics['backtracking_steps']+=attempt;break
                if accepted:metrics['actor_steps']+=1;metrics['actor_loss']+=float(loss_a.detach())
                else:
                    actor.load_state_dict(old);aopt.load_state_dict(optimizer_before)
                    metrics['rejected_updates']+=1;metrics['stopped_kl']=True;stopped=True
            if progress:progress(dict(stage='update',epoch=epoch,minibatch=start,actor_steps=metrics['actor_steps']))
    metrics['actor_loss']/=max(metrics['actor_steps'],1);metrics['critic_loss']/=max(metrics['critic_steps'],1)
    metrics['advantage_scale']=scale
    metrics['full_rollout_conditional_kl_mean']=conditional_kl(actor,rows,0,len(rows),burn_in=0)
    return metrics
