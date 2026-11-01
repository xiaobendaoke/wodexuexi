"""Isolated calibrated motion PPO, with an explicit optional service auxiliary."""
import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from marl_models.dynamic_service_joint import DemandEncoder
from marl_models.dynamic_service_hmarl import tensor
from utils.dynamic_service_feasible_motion import motion_library,FEATURE_DIM
from utils.dynamic_service_motion_credit import motion_gae
from utils.dynamic_service_joint_rollout import slot_surrogate


class MotionPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=DemandEncoder()
        self.context=nn.Sequential(nn.Linear(640,64),nn.LayerNorm(64),nn.Tanh())
        self.candidate=nn.Sequential(nn.Linear(FEATURE_DIM,64),nn.Tanh())
        self.score=nn.Sequential(nn.Linear(192,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.orthogonal_(self.score[-1].weight,gain=1.);nn.init.zeros_(self.score[-1].bias)
    @property
    def device(self):return next(self.parameters()).device
    def initial_state(self):return torch.zeros(5,128,device=self.device)
    def distribution(self,state,bank):
        x=tensor(bank['features'],self.device);mask=torch.as_tensor(bank['mask'],device=self.device)
        legal=x[mask];x=(x-legal.mean(0))/legal.std(0,unbiased=False).clamp_min(.01)
        candidate=self.candidate(x);context=self.context(state.flatten()).expand(len(x),-1)
        logits=self.score(torch.cat((candidate,context,candidate*context),-1)).squeeze(-1).clamp(-12,12)
        return Categorical(logits=logits.masked_fill(~mask,-torch.inf))
    @torch.no_grad()
    def act(self,obs,hidden,seed,greedy=False):
        state=self.encoder(obs,hidden);bank=motion_library(obs);dist=self.distribution(state,bank)
        g=torch.Generator().manual_seed(int(seed))
        index=dist.logits.argmax() if greedy else torch.multinomial(dist.probs,1,generator=g).squeeze()
        return dict(index=int(index),log=float(dist.log_prob(index)),probs=dist.probs.numpy(),actions=bank['actions'][int(index)].copy(),
            hidden=state.numpy(),bank=bank)
    def replay(self,obs,hidden,row):
        state=self.encoder(obs,hidden);dist=self.distribution(state,row['bank'])
        return dict(hidden=state,log=dist.log_prob(torch.tensor(row['index'],device=self.device)),probs=dist.probs,logits=dist.logits,entropy=dist.entropy())


class MotionValue(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=DemandEncoder();self.value=nn.Sequential(nn.Linear(641,128),nn.Tanh(),nn.Linear(128,1))
    @property
    def device(self):return next(self.parameters()).device
    def initial_state(self):return torch.zeros(5,128,device=self.device)
    def forward(self,obs,hidden,time):
        state=self.encoder(obs,hidden)
        return self.value(torch.cat((state.flatten(),tensor([(999-time)/1000],self.device)))).squeeze(),state
    @torch.no_grad()
    def predict(self,obs,hidden,time):
        v,h=self(obs,hidden,time);return float(v),h.numpy()


def chunk_kl(actor,rows,start,end):
    burn=max(0,start-16);h=tensor(rows[burn]['hidden_before'],actor.device);values=[]
    with torch.no_grad():
        for i in range(burn,start):h=actor.encoder(rows[i]['obs'],h)
        for row in rows[start:end]:
            z=actor.replay(row['obs'],h,row);h=z['hidden'];p=tensor(row['probs'],actor.device)
            values.append(float((p*(p.clamp_min(1e-30).log()-z['probs'].clamp_min(1e-30).log())).sum()))
    return float(np.mean(values))


def update_motion(actor,critic,aopt,copt,rows,aux_weight=0.,epochs=2,sequence=64,progress=None):
    advantages,returns=motion_gae([r['reward'] for r in rows],[r['value'] for r in rows])
    n=len(advantages)
    if n==0:raise ValueError('at least two complete physical steps needed')
    scale=max(float(advantages.std()),.05);advantages=(advantages-advantages.mean())/scale
    metrics=dict(actor_steps=0,critic_steps=0,skipped_chunks=0,backtracking_steps=0,actor_loss=0.,critic_loss=0.,aux_loss=0.,
        max_accepted_chunk_kl=0.,advantage_std=scale,aux_weight=aux_weight)
    starts=list(range(0,n,sequence))
    for epoch in range(epochs):
        for j in torch.randperm(len(starts)).tolist():
            start=starts[j];end=min(start+sequence,n);burn=max(0,start-16)
            h=tensor(rows[burn]['hidden_before'],actor.device);ch=tensor(rows[burn]['critic_hidden_before'],critic.device)
            with torch.no_grad():
                for i in range(burn,start):
                    h=actor.encoder(rows[i]['obs'],h);ch=critic.encoder(rows[i]['obs'],ch)
            policy=[];value=[];aux=[]
            for i in range(start,end):
                row=rows[i];z=actor.replay(row['obs'],h,row);h=z['hidden'];v,ch=critic(row['obs'],ch,row['time'])
                policy.append(slot_surrogate(z['log'],tensor(row['log'],actor.device),tensor(advantages[i],actor.device))-.001*z['entropy'])
                value.append((v-float(returns[i])).square())
                if aux_weight and row.get('label_indices') is not None:
                    ix=torch.as_tensor(row['label_indices'],device=actor.device);scores=tensor(row['label_scores'],actor.device)
                    target=torch.softmax(scores-scores.max(),0).detach();logp=torch.log_softmax(z['logits'][ix],0)
                    aux.append((target*(target.clamp_min(1e-30).log()-logp)).sum())
            loss_c=torch.stack(value).mean();copt.zero_grad(set_to_none=True);loss_c.backward()
            nn.utils.clip_grad_norm_(critic.parameters(),.5,error_if_nonfinite=True);copt.step()
            metrics['critic_steps']+=1;metrics['critic_loss']+=float(loss_c.detach())
            before=chunk_kl(actor,rows,start,end)
            if not np.isfinite(before) or before>.03:
                metrics['skipped_chunks']+=1
            else:
                auxiliary=torch.stack(aux).mean() if aux else torch.tensor(0.,device=actor.device)
                loss_a=torch.stack(policy).mean()+aux_weight*auxiliary
                if not torch.isfinite(loss_a):raise FloatingPointError('actor loss')
                aopt.zero_grad(set_to_none=True);loss_a.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5,error_if_nonfinite=True)
                old={k:v.clone() for k,v in actor.state_dict().items()};oldopt=copy.deepcopy(aopt.state_dict())
                aopt.step();proposal={k:v.clone() for k,v in actor.state_dict().items()};accepted=False
                for attempt in range(6):
                    if attempt:actor.load_state_dict({k:old[k]+(.5**attempt)*(proposal[k]-old[k]) for k in old})
                    kl=chunk_kl(actor,rows,start,end)
                    if np.isfinite(kl) and kl<=.03+1e-8:
                        accepted=True;metrics['max_accepted_chunk_kl']=max(metrics['max_accepted_chunk_kl'],kl)
                        metrics['backtracking_steps']+=attempt;break
                if accepted:
                    metrics['actor_steps']+=1;metrics['actor_loss']+=float(loss_a.detach());metrics['aux_loss']+=float(auxiliary.detach())
                else:actor.load_state_dict(old);aopt.load_state_dict(oldopt);metrics['skipped_chunks']+=1
            if progress:progress(dict(stage='update',epoch=epoch,start=start,actor_steps=metrics['actor_steps']))
    for key in ('actor_loss','aux_loss'):metrics[key]/=max(metrics['actor_steps'],1)
    metrics['critic_loss']/=max(metrics['critic_steps'],1);metrics['full_rollout_kl']=chunk_kl(actor,rows,0,n)
    return metrics
