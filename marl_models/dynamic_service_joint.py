"""Central joint PPO candidate. Physics and historical releases are untouched."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Categorical, kl_divergence
from environment.dynamic_service_v2 import boundary_actions
from marl_models.dynamic_service_hmarl import tensor
from marl_models.dynamic_service_hmarl_v2 import LowerPolicyV2
from utils.dynamic_service_joint_rollout import causal_returns, slot_surrogate

METHODS=('trajectory_matching','history_allocation','joint','joint_no_condition','joint_mlp')


class DemandEncoder(nn.Module):
    def __init__(self,mlp=False):
        super().__init__();self.mlp=mlp
        if mlp:
            self.flat=nn.Sequential(nn.Linear(716,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
        else:
            self.query=nn.Sequential(nn.Linear(16,128),nn.Tanh())
            self.token=nn.Sequential(nn.Linear(6,128),nn.Tanh())
            self.attention=nn.MultiheadAttention(128,4,batch_first=True)
            self.context=nn.Sequential(nn.Linear(256,128),nn.Tanh())
            self.neighbors=nn.MultiheadAttention(128,4,batch_first=True)
            self.fuse=nn.Sequential(nn.Linear(384,128),nn.Tanh())
            self.gru=nn.GRUCell(128,128)

    def forward(self,obs,hidden):
        device=next(self.parameters()).device
        ue=tensor(obs['ue'],device);visible=torch.as_tensor(obs['visible'],dtype=torch.bool,device=device)
        local=torch.cat([tensor(obs[k],device) for k in ('nodes','feedback','motion','walls')],-1)
        if self.mlp:
            return self.flat(torch.cat((ue.flatten(1),visible.float(),local),-1))
        query=self.query(local);tokens=self.token(ue)
        tokens=torch.cat((tokens,torch.zeros(5,1,128,device=device)),1)
        padding=torch.cat((~visible,torch.zeros(5,1,dtype=torch.bool,device=device)),1)
        demand=self.attention(query[:,None],tokens,tokens,key_padding_mask=padding,need_weights=False)[0][:,0]
        context=self.context(torch.cat((query,demand),-1))
        # Central coordinator already has the public union: no extra hidden records.
        neighbors=self.neighbors(context[None],context[None],context[None],need_weights=False)[0][0]
        return self.gru(self.fuse(torch.cat((query,demand,neighbors),-1)),hidden)


def allocation_summary(batch,assignment):
    assignment=np.asarray(assignment,dtype=np.int64)
    out=np.zeros((6,3),np.float32)
    counts=np.bincount(assignment,minlength=6)
    out[:,0]=counts/100
    for node in range(6):
        members=np.flatnonzero(assignment==node)
        out[node,1]=batch.cycles[members].sum()/batch.capacity[node]
        if len(members):
            predicted=batch.tx[members,node]+batch.cycles[members]*counts[node]/batch.capacity[node]
            out[node,2]=(predicted>batch.deadline[members]).mean()
    return out.ravel()


class AllocationHead(LowerPolicyV2):
    def __init__(self,mlp=False):
        super().__init__('independent' if mlp else 'autoregressive')
        # MLP comparator keeps sequential load/risk feedback, but removes graph
        # messages and history. It tests the whole extra encoding complexity.
        self.mode='autoregressive';self.mlp=mlp
        if mlp:
            self.messages=nn.ModuleList();self.node_updates=nn.ModuleList();self.request_updates=nn.ModuleList()
        del self.value
        self.shared_request=nn.Linear(128,128)
        self.shared_node=nn.Linear(128,128)

    def encode_context(self,batch,state):
        r,n,e=super().encode(batch)
        r=r+self.shared_request(state.mean(0))
        node_state=torch.cat((state,state.mean(0,keepdim=True)),0)
        return r,n+self.shared_node(node_state),e

    def distribution_context(self,batch,encoded,risk):
        r,n,e=encoded
        logits=self.score(torch.cat((r[:,None].expand(-1,6,-1),n[None].expand(len(batch),-1,-1),e,risk),-1)).squeeze(-1)
        logits=logits+self.risk_score(risk).squeeze(-1)
        legal=torch.as_tensor(batch.legal,dtype=torch.bool,device=self.device)
        return Categorical(logits=logits.masked_fill(~legal,-torch.inf))


class JointPolicy(nn.Module):
    def __init__(self,method='joint'):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method;self.learn_motion=method!='history_allocation'
        self.learn_allocation=method!='trajectory_matching'
        self.condition=method not in ('joint_no_condition','trajectory_matching')
        self.encoder=DemandEncoder(method=='joint_mlp')
        self.allocation=AllocationHead(method=='joint_mlp')
        self.motion=nn.Sequential(nn.Linear(274,128),nn.Tanh(),nn.Linear(128,2))
        self.log_std=nn.Parameter(torch.full((2,),-.7))
        nn.init.orthogonal_(self.motion[-1].weight,gain=.01);nn.init.zeros_(self.motion[-1].bias)

    @property
    def device(self):return next(self.parameters()).device

    def initial_state(self):return torch.zeros(5,128,device=self.device)

    def motion_distribution(self,state,batch,assignment):
        summary=allocation_summary(batch,assignment) if self.condition else np.zeros(18,np.float32)
        x=torch.cat((state,state.mean(0,keepdim=True).expand(5,-1),tensor(summary,self.device).expand(5,-1)),-1)
        return Normal(self.motion(x),self.log_std.clamp(-3,1).exp())

    @torch.no_grad()
    def act(self,obs,batch,hidden,stochastic=False,fixed_assignment=None,fixed_actions=None):
        state=self.encoder(obs,hidden);chosen=[];risks=[];logs=[];probs=[]
        encoded=self.allocation.encode_context(batch,state) if len(batch) and fixed_assignment is None else None
        for i in range(len(batch) if fixed_assignment is None else 0):
            risk=self.allocation.dynamic_features(batch,chosen,i)
            # Only the current request is decoded; the risk uses its true prefix.
            r,n,e=encoded
            logits=self.allocation.score(torch.cat((r[i].expand(6,-1),n,e[i],risk),-1)).squeeze(-1)
            logits=logits+self.allocation.risk_score(risk).squeeze(-1)
            dist=Categorical(logits=logits.masked_fill(~torch.as_tensor(batch.legal[i],device=self.device),-torch.inf))
            a=(dist.sample() if stochastic else dist.logits.argmax()) if fixed_assignment is None else torch.tensor(fixed_assignment[i],device=self.device)
            chosen.append(int(a));risks.append(risk.cpu().numpy());logs.append(float(dist.log_prob(a)));probs.append(dist.probs.cpu().numpy())
        assignment=np.asarray(chosen if fixed_assignment is None else fixed_assignment,np.int64)
        dist=self.motion_distribution(state,batch,assignment)
        raw=dist.sample() if stochastic else dist.mean
        actions=boundary_actions(raw.cpu().numpy(),np.asarray(obs['nodes'][:,:2])*700)
        if fixed_actions is not None:actions=np.asarray(fixed_actions).copy()
        return dict(assignment=assignment,risk=np.asarray(risks,np.float32).reshape(-1,6,11),
            allocation_log=np.asarray(logs,np.float32),allocation_probs=np.asarray(probs,np.float32).reshape(-1,6),
            raw=raw.cpu().numpy(),actions=actions,motion_log=dist.log_prob(raw).sum(-1).cpu().numpy(),
            motion_mean=dist.mean.cpu().numpy(),motion_std=dist.stddev.cpu().numpy(),hidden=state.cpu().numpy())

    def replay(self,obs,batch,hidden,sample):
        state=self.encoder(obs,hidden)
        if len(batch) and self.learn_allocation:
            dist=self.allocation.distribution_context(batch,self.allocation.encode_context(batch,state),tensor(sample['risk'],self.device))
            log=dist.log_prob(torch.as_tensor(sample['assignment'],device=self.device))
            entropy=dist.entropy();probs=dist.probs
        else:
            log=entropy=torch.empty(0,device=self.device);probs=torch.empty(0,6,device=self.device)
        motion=self.motion_distribution(state,batch,sample['assignment'])
        return dict(allocation_log=log,allocation_entropy=entropy,allocation_probs=probs,
            motion_log=motion.log_prob(tensor(sample['raw'],self.device)).sum(-1),
            motion_entropy=motion.entropy().sum(-1),motion_dist=motion,hidden=state)


class JointCritic(nn.Module):
    """Independent encoder/optimizer; unbounded values with fixed return scale."""
    def __init__(self,method='joint'):
        super().__init__();self.method=method;self.scale=1. if method=='history_allocation' else 100.
        self.encoder=DemandEncoder(method=='joint_mlp')
        self.batch_encoder=AllocationHead(method=='joint_mlp')
        self.prefix=nn.Sequential(nn.Linear(453,128),nn.Tanh(),nn.Linear(128,1))
        self.movement=nn.Sequential(nn.Linear(147,128),nn.Tanh(),nn.Linear(128,1))

    @property
    def device(self):return next(self.parameters()).device

    def initial_state(self):return torch.zeros(5,128,device=self.device)

    def forward(self,obs,batch,hidden,assignment,risk,time):
        state=self.encoder(obs,hidden);pooled=state.mean(0);n=len(batch)
        # Physical clock is allowed. Never expose hotspot phase or future tape.
        remaining=tensor([(1000-time)/1000],self.device)
        if n:
            r,nodes,_=self.batch_encoder.encode_context(batch,state)
            # Reverse cumulative mean explicitly describes the remaining batch.
            rest=r.flip(0).cumsum(0).flip(0)/torch.arange(n,0,-1,device=self.device)[:,None]
            progress=tensor(np.c_[np.arange(n)/n,np.full(n,n/100)],self.device)
            x=torch.cat((pooled.expand(n,-1),rest,nodes.mean(0).expand(n,-1),tensor(risk,self.device).flatten(1),progress,remaining.expand(n,1)),-1)
            values=self.prefix(x).squeeze(-1)*self.scale
        else:values=torch.empty(0,device=self.device)
        summary=tensor(allocation_summary(batch,assignment),self.device)
        movement=self.movement(torch.cat((pooled,summary,remaining))).squeeze(-1)*self.scale
        return values,movement,state

    @torch.no_grad()
    def predict(self,obs,batch,hidden,assignment,time,risk=None):
        if risk is None:risk=self.batch_encoder.risk_trajectory(batch,assignment).cpu().numpy() if len(batch) else np.empty((0,6,11),np.float32)
        p,m,h=self(obs,batch,hidden,assignment,risk,time)
        return dict(prefix_values=p.cpu().numpy(),motion_value=float(m),critic_hidden=h.cpu().numpy())


def branch_gradients(loss_a,loss_m,parameters):
    params=list(parameters)
    def flat(loss):
        if not loss.requires_grad:return torch.zeros(sum(p.numel() for p in params),device=params[0].device)
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(params,grads)])
    a,m=flat(loss_a),flat(loss_m);an=a.norm();mn=m.norm()
    return dict(allocation_encoder_grad=float(an),motion_encoder_grad=float(mn),
                encoder_grad_cosine=float(torch.dot(a,m)/(an*mn).clamp_min(1e-12)))


def update_joint(actor,critic,aopt,copt,rows,sequence=64,burn_in=16,epochs=4,target_kl=.05):
    targets=causal_returns(rows,allocation_only=actor.method=='history_allocation')
    raw=[]
    for row,target in zip(rows,targets):
        if actor.learn_allocation:raw.extend((target['prefix_targets']-row['prefix_values']).tolist())
        if actor.learn_motion:raw.append(target['motion_target']-row['motion_value'])
    scale=max(float(np.std(raw)),.05)
    metrics=dict(actor_steps=0,critic_steps=0,actor_loss=0.,critic_loss=0.,max_batch_kl=0.,
        max_post_batch_kl=0.,max_allocation_factor_kl=0.,max_motion_kl=0.,clip_fraction=0.,
        allocation_entropy=0.,motion_entropy=0.,stopped_kl=False,advantage_scale=scale,
        allocation_encoder_grad=0.,motion_encoder_grad=0.,encoder_grad_cosine=0.,gradient_samples=0)
    starts=list(range(0,len(rows),sequence))
    actor_stopped=False
    for epoch in range(epochs):
        for index in torch.randperm(len(starts)).tolist():
            start=starts[index];end=min(start+sequence,len(rows));burn=max(0,start-burn_in)
            ah=tensor(rows[burn]['hidden_before'],actor.device);ch=tensor(rows[burn]['critic_hidden_before'],critic.device)
            with torch.no_grad():
                for t in range(burn,start):
                    ah=actor.encoder(rows[t]['observation'],ah)
                    ch=critic.encoder(rows[t]['observation'],ch)
            al=[];ml=[];cl=[];ent=[];kls=[];clips=[];alloc_ent=[];motion_ent=[]
            for t in range(start,end):
                row=rows[t];target=targets[t];n=len(row['batch'])
                pred=actor.replay(row['observation'],row['batch'],ah,row);ah=pred['hidden']
                p,m,ch=critic(row['observation'],row['batch'],ch,row['assignment'],row['risk'],row['time'])
                zero=pred['motion_log'].sum()*0
                la=lm=zero;entropy=zero;ka=km=0.
                value_terms=[]
                if n:
                    value_terms.append(((p-tensor(target['prefix_targets'],critic.device))/critic.scale).square().mean())
                if actor.learn_motion:value_terms.append(((m-target['motion_target'])/critic.scale).square())
                if value_terms:cl.append(torch.stack(value_terms).mean())
                if actor.learn_allocation and n:
                    old=tensor(row['allocation_log'],actor.device)
                    adv=tensor((target['prefix_targets']-row['prefix_values'])/scale,actor.device)
                    la=slot_surrogate(pred['allocation_log'],old,adv)
                    oldp=tensor(row['allocation_probs'],actor.device)
                    kd=(oldp*(oldp.clamp_min(1e-30).log()-pred['allocation_probs'].clamp_min(1e-30).log())).sum(-1)
                    ka=float(kd.detach().sum());metrics['max_allocation_factor_kl']=max(metrics['max_allocation_factor_kl'],float(kd.detach().max()))
                    entropy=entropy+pred['allocation_entropy'].sum();alloc_ent.append(float(pred['allocation_entropy'].detach().mean()))
                    clips.extend(((pred['allocation_log'].detach()-old).exp().sub(1).abs()>.2).tolist())
                if actor.learn_motion:
                    adv=torch.full((5,),(target['motion_target']-row['motion_value'])/scale,device=actor.device)
                    # Final physical movement cannot affect any scored service.
                    if t==len(rows)-1:adv=adv*0
                    old=tensor(row['motion_log'],actor.device)
                    lm=slot_surrogate(pred['motion_log'],old,adv)
                    kd=kl_divergence(Normal(tensor(row['motion_mean'],actor.device),tensor(row['motion_std'],actor.device)),pred['motion_dist']).sum()
                    km=float(kd.detach());metrics['max_motion_kl']=max(metrics['max_motion_kl'],km)
                    if t<len(rows)-1:entropy=entropy+pred['motion_entropy'].sum()
                    motion_ent.append(float(pred['motion_entropy'].detach().mean()))
                    clips.extend(((pred['motion_log'].detach()-old).exp().sub(1).abs()>.2).tolist())
                al.append(la);ml.append(lm);ent.append(entropy);kls.append(max(0.,ka+km))
            if cl:
                loss_c=torch.stack(cl).mean()
                if not torch.isfinite(loss_c):raise FloatingPointError('critic loss')
                copt.zero_grad(set_to_none=True);loss_c.backward()
                nn.utils.clip_grad_norm_(critic.parameters(),.5,error_if_nonfinite=True);copt.step()
                metrics['critic_steps']+=1;metrics['critic_loss']+=float(loss_c.detach())
            batch_kl=float(np.mean(kls));metrics['max_batch_kl']=max(metrics['max_batch_kl'],batch_kl)
            if batch_kl>target_kl or not np.isfinite(batch_kl):actor_stopped=True;metrics['stopped_kl']=True
            if not actor_stopped:
                la=torch.stack(al).mean();lm=torch.stack(ml).mean()
                loss=la+lm-.001*torch.stack(ent).mean()
                if not torch.isfinite(loss):raise FloatingPointError('actor loss')
                if index==0:
                    diag=branch_gradients(la,lm,actor.encoder.parameters())
                    for key,value in diag.items():metrics[key]+=value
                    metrics['gradient_samples']+=1
                aopt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),.5,error_if_nonfinite=True);aopt.step()
                metrics['actor_steps']+=1;metrics['actor_loss']+=float(loss.detach())
                metrics['clip_fraction']+=float(np.mean(clips)) if clips else 0.
                metrics['allocation_entropy']+=float(np.mean(alloc_ent)) if alloc_ent else 0.
                metrics['motion_entropy']+=float(np.mean(motion_ent)) if motion_ent else 0.
                # Next minibatch/epoch checks old-policy conditional KL before
                # updating. Post-update whole-rollout KL is audited below.
    for key in ('actor_loss','clip_fraction','allocation_entropy','motion_entropy'):metrics[key]/=max(metrics['actor_steps'],1)
    metrics['critic_loss']/=max(metrics['critic_steps'],1)
    for key in ('allocation_encoder_grad','motion_encoder_grad','encoder_grad_cosine'):metrics[key]/=max(metrics['gradient_samples'],1)
    # Exact recurrent replay from the actual reset, with current actor weights.
    with torch.no_grad():
        hidden=actor.initial_state();post=[]
        for row in rows:
            pred=actor.replay(row['observation'],row['batch'],hidden,row);hidden=pred['hidden'];kl=0.
            if actor.learn_allocation and len(row['batch']):
                old=tensor(row['allocation_probs'],actor.device)
                kl+=float((old*(old.clamp_min(1e-30).log()-pred['allocation_probs'].clamp_min(1e-30).log())).sum())
            if actor.learn_motion:
                kl+=float(kl_divergence(Normal(tensor(row['motion_mean'],actor.device),tensor(row['motion_std'],actor.device)),pred['motion_dist']).sum())
            post.append(max(0.,kl))
        metrics['max_post_batch_kl']=max(post,default=0.);metrics['mean_post_batch_kl']=float(np.mean(post)) if post else 0.
    return metrics
