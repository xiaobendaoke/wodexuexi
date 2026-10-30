"""Second-release constraint-aware recurrent actor and batch-risk graph policy."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Categorical
from environment.dynamic_service_v2 import boundary_actions
from marl_models.dynamic_service_hmarl import UpperPolicy, LowerPolicy, tensor


class UpperPolicyV2(UpperPolicy):
    def __init__(self,memory=True,feedback=True):
        super().__init__(memory=memory,feedback=feedback)
        self.self_embed=nn.Sequential(nn.Linear(16,128),nn.Tanh())
        self.neighbor_context=nn.Sequential(nn.Linear(256,128),nn.Tanh())

    def forward(self,observation,hidden):
        ue=tensor(observation['ue'],self.device)
        visible=torch.as_tensor(observation['visible'],dtype=torch.bool,device=self.device)
        nodes=tensor(observation['nodes'],self.device)
        feedback=tensor(observation['feedback'],self.device)
        if not self.feedback_enabled:feedback=feedback*0
        extra=torch.cat((tensor(observation['motion'],self.device),tensor(observation['walls'],self.device)),-1)
        query=self.self_embed(torch.cat((nodes,feedback,extra),-1))
        tokens=self.ue_embed(ue)
        tokens=torch.cat((tokens,torch.zeros(5,1,128,device=self.device)),1)
        padding=torch.cat((~visible,torch.zeros(5,1,dtype=torch.bool,device=self.device)),1)
        demand=self.ue_attention(query[:,None],tokens,tokens,key_padding_mask=padding,need_weights=False)[0][:,0]
        context=self.neighbor_context(torch.cat((query,demand),-1))
        mask=torch.cdist(nodes[:,:2],nodes[:,:2])>460/700
        neighbors=self.neighbor_attention(context[None],context[None],context[None],attn_mask=mask,need_weights=False)[0][0]
        x=self.fuse(torch.cat((query,demand,neighbors),-1))
        state=self.gru(x,hidden) if self.memory else x
        dist=Normal(self.actor(state),self.log_std.clamp(-3,1).exp())
        global_state=state.mean(0,keepdim=True).expand(5,-1)
        return dist,self.critic(torch.cat((state,global_state),-1)).squeeze(-1),state

    @torch.no_grad()
    def act(self,observation,hidden,stochastic=False):
        dist,value,state=self(observation,hidden)
        raw=dist.sample() if stochastic else dist.mean
        # nodes are public normalized coordinates. No geometry/tape access.
        positions=np.asarray(observation['nodes'][:,:2],dtype=np.float64)*700
        actions=boundary_actions(raw.cpu().numpy(),positions)
        return dict(raw=raw.cpu().numpy(),actions=actions,logprob=dist.log_prob(raw).sum(-1).cpu().numpy(),
                    value=value.cpu().numpy(),hidden=state)


class LowerPolicyV2(LowerPolicy):
    dynamic_dim=11

    def __init__(self,mode='autoregressive'):
        super().__init__(mode=mode)
        self.score=nn.Sequential(nn.Linear(128*3+self.dynamic_dim,128),nn.Tanh(),nn.Linear(128,1))
        self.risk_score=nn.Linear(self.dynamic_dim,1,bias=False)
        nn.init.zeros_(self.risk_score.weight)
        # Terminal batch reward is a fraction in [0,1]. Preserve node-specific
        # risk instead of averaging away which node is overloaded.
        self.value=nn.Sequential(nn.Linear(128*2+6*self.dynamic_dim,128),nn.Tanh(),nn.Linear(128,1),nn.Sigmoid())

    def encode(self,batch):
        r=tensor(np.c_[batch.size/5e6,batch.cycles/14e9,batch.deadline,batch.source/5],self.device)
        n=tensor(np.c_[batch.node_positions/700,batch.capacity/2e11,np.arange(6)==5],self.device)
        deadline=batch.deadline[:,None]
        single=batch.cycles[:,None]/batch.capacity
        edge=tensor(np.stack([np.clip(batch.tx/deadline,0,20),np.clip(single/deadline,0,20),
                   np.clip(1-(batch.tx+single)/deadline,-19,1),batch.legal],-1),self.device)
        request,node,edge=self.request_embed(r),self.node_embed(n),self.edge_embed(edge)
        mask=tensor(batch.legal,self.device)[...,None]
        if self.mode!='independent':
            for message,nupdate,rupdate in zip(self.messages,self.node_updates,self.request_updates):
                expanded=request[:,None].expand(-1,6,-1)
                msg=torch.tanh(message(torch.cat((expanded,edge),-1)))*mask
                node=torch.tanh(nupdate(torch.cat((node,msg.sum(0)/mask.sum(0).clamp_min(1)),-1)))
                pooled=(node[None]*mask).sum(1)/mask.sum(1).clamp_min(1)
                request=torch.tanh(rupdate(torch.cat((request,pooled),-1)))
        return request,node,edge

    def dynamic_features(self,batch,chosen,i):
        features=np.zeros((6,self.dynamic_dim),np.float32)
        # Parallel/independent controls see singleton deadline risks but never
        # see the decoder's evolving assignment state.
        assignment=np.asarray(chosen if self.mode=='autoregressive' else [],dtype=int)
        counts=np.bincount(assignment,minlength=6)
        for node in range(6):
            members=np.flatnonzero(assignment==node);count=counts[node]
            tx=batch.tx[i,node];comp=batch.cycles[i]*(count+1)/batch.capacity[node]
            ratio=(tx+comp)/batch.deadline[i]
            features[node,:6]=[count/max(len(batch),1),batch.cycles[members].sum()/batch.capacity[node]/batch.deadline.mean(),
                                    min(20,tx/batch.deadline[i]),min(20,comp/batch.deadline[i]),
                                    min(20,ratio),np.clip(1-ratio,-19,1)]
            old=batch.tx[members,node]+batch.cycles[members]*count/batch.capacity[node]
            new=batch.tx[members,node]+batch.cycles[members]*(count+1)/batch.capacity[node]
            additional=((new>batch.deadline[members])&(old<=batch.deadline[members])).sum()
            features[node,6]=additional/max(i,1)
            features[node,7]=((new>batch.deadline[members]).sum()+int(ratio>1))/(count+1)
            margins=np.r_[1-new/batch.deadline[members],1-ratio]
            features[node,8]=np.clip(margins.min(),-19,1)
            if self.mode=='autoregressive':
                remaining=np.flatnonzero(batch.legal[i+1:,node])+i+1
                if len(remaining):
                    later=batch.tx[remaining,node]+batch.cycles[remaining]*(count+2)/batch.capacity[node]
                    features[node,9]=(later<=batch.deadline[remaining]).mean()
            features[node,10]=ratio>1
        return tensor(features,self.device)

    def distribution(self,batch,encoded,chosen,i):
        request,nodes,edges=encoded;dynamic=self.dynamic_features(batch,chosen,i)
        r=request[i].expand(6,-1)
        logits=self.score(torch.cat((r,nodes,edges[i],dynamic),-1)).squeeze(-1)+self.risk_score(dynamic).squeeze(-1)
        mask=torch.as_tensor(batch.legal[i],device=self.device,dtype=torch.bool)
        dist=Categorical(logits=logits.masked_fill(~mask,-torch.inf))
        value=self.value(torch.cat((request.mean(0),nodes.mean(0),dynamic.flatten()))).squeeze(-1)
        return dist,value

    def risk_trajectory(self,batch,assignment):
        return torch.stack([self.dynamic_features(batch,assignment[:i],i) for i in range(len(batch))])

    def replay(self,batch,assignment,risk=None):
        # Teacher-forced prefixes are known during PPO. Vectorize the decoder
        # MLP without changing their order, masks, or load-state calculations.
        request,nodes,edges=self.encode(batch)
        dynamic=self.risk_trajectory(batch,assignment) if risk is None else tensor(risk,self.device)
        n=len(batch)
        values_input=torch.cat((request.mean(0).expand(n,-1),nodes.mean(0).expand(n,-1),dynamic.flatten(1)),-1)
        logits=self.score(torch.cat((request[:,None].expand(-1,6,-1),nodes[None].expand(n,-1,-1),edges,dynamic),-1)).squeeze(-1)
        logits=logits+self.risk_score(dynamic).squeeze(-1)
        legal=torch.as_tensor(batch.legal,device=self.device,dtype=torch.bool)
        dist=Categorical(logits=logits.masked_fill(~legal,-torch.inf))
        action=torch.as_tensor(assignment,device=self.device,dtype=torch.long)
        return dist.log_prob(action),self.value(values_input).squeeze(-1),dist.entropy()


def approximate_kl(new_log,old_log):
    with torch.no_grad():
        delta=new_log-old_log
        return float((delta.exp()-1-delta).mean())


def clipped_loss(log,old_log,advantage,values,old_values,targets,entropy):
    ratio=(log-old_log).exp()
    actor=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
    clipped=old_values+(values-old_values).clamp(-.2,.2)
    critic=torch.maximum((values-targets).square(),(clipped-targets).square()).mean()
    loss=actor+.5*critic-.005*entropy.mean()
    return loss,float(actor.detach()),float(critic.detach()),float(((ratio-1).abs()>.2).float().mean())


def _metrics():
    return dict(loss=0.,actor_loss=0.,value_loss=0.,clip_fraction=0.,max_kl=0.,accepted_kl_max=0.,
                optimizer_steps=0,stopped_kl=False,epochs_started=0)


def _accept_kl(metrics,kl,target):
    metrics['max_kl']=max(metrics['max_kl'],min(kl,1e6) if np.isfinite(kl) else 1e6)
    if not np.isfinite(kl) or kl>target:
        metrics['stopped_kl']=True
        return False
    metrics['accepted_kl_max']=max(metrics['accepted_kl_max'],kl)
    return True


def _step(policy,optimizer,loss,parts,metrics):
    if not torch.isfinite(loss):raise FloatingPointError('non-finite PPO loss')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
    optimizer.step()
    metrics['optimizer_steps']+=1
    metrics['loss']+=float(loss.detach())
    for key,value in zip(('actor_loss','value_loss','clip_fraction'),parts):metrics[key]+=value


def _finish(metrics):
    n=max(metrics['optimizer_steps'],1)
    for key in ('loss','actor_loss','value_loss','clip_fraction'):metrics[key]/=n
    return metrics


def update_lower_v2(policy,optimizer,transitions,target_kl=.02):
    valid=[dict(x,risk=policy.risk_trajectory(x['batch'],x['assignment']).cpu().numpy())
           for x in transitions if len(x['batch'])];metrics=_metrics()
    if not valid:return metrics
    advantages=np.concatenate([x['reward']-np.asarray(x['values']) for x in valid])
    mean=float(advantages.mean());scale=max(float(advantages.std()),.05)
    metrics.update(advantage_mean=mean,advantage_scale=scale,graphs=len(valid))
    for epoch in range(4):
        metrics['epochs_started']=epoch+1
        order=torch.randperm(len(valid)).tolist()
        for start in range(0,len(order),8):
            losses=[];kls=[];parts=[]
            for index in order[start:start+8]:
                item=valid[index];log,value,entropy=policy.replay(item['batch'],item['assignment'],item['risk'])
                old=tensor(item['logprob'],policy.device);old_value=tensor(item['values'],policy.device)
                target=torch.full_like(value,item['reward'])
                adv=(target-old_value-mean)/scale
                kls.append(approximate_kl(log,old))
                # A KL rejection happens before exponentiation in the loss.
                if np.isfinite(kls[-1]) and kls[-1]<=target_kl:
                    result=clipped_loss(log,old,adv,value,old_value,target,entropy)
                    losses.append(result[0]);parts.append(result[1:])
            batch_kl=max(kls) if all(np.isfinite(x) for x in kls) else float('inf')
            if not _accept_kl(metrics,batch_kl,target_kl):return _finish(metrics)
            loss=torch.stack(losses).mean();averaged=np.mean(parts,axis=0)
            _step(policy,optimizer,loss,averaged,metrics)
    return _finish(metrics)


def update_upper_v2(policy,optimizer,rollout,target_kl=.02):
    metrics=_metrics()
    if not rollout:return metrics
    rewards=np.array([x['reward'] for x in rollout])[:,None]
    old_values=np.stack([x['value'] for x in rollout]);advantages=np.zeros_like(old_values)
    running=np.zeros(5);following=np.zeros(5)
    for t in reversed(range(len(rollout))):
        delta=rewards[t]+.99*following-old_values[t]
        running=delta+.99*.95*running;advantages[t]=running;following=old_values[t]
    targets=advantages+old_values
    advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    starts=list(range(0,len(rollout),32))
    for epoch in range(4):
        metrics['epochs_started']=epoch+1
        for index in torch.randperm(len(starts)).tolist():
            start=starts[index];burn=max(0,start-16);end=min(start+32,len(rollout))
            hidden=tensor(rollout[burn]['hidden_before'],policy.device)
            with torch.no_grad():
                for t in range(burn,start):hidden=policy(rollout[t]['observation'],hidden)[2]
            logs=[];values=[];entropies=[]
            for t in range(start,end):
                dist,value,hidden=policy(rollout[t]['observation'],hidden)
                logs.append(dist.log_prob(tensor(rollout[t]['raw'],policy.device)).sum(-1))
                values.append(value);entropies.append(dist.entropy().sum(-1))
            log=torch.stack(logs);old=tensor(np.stack([x['logprob'] for x in rollout[start:end]]),policy.device)
            if not _accept_kl(metrics,approximate_kl(log,old),target_kl):return _finish(metrics)
            result=clipped_loss(log,old,tensor(advantages[start:end],policy.device),torch.stack(values),
                       tensor(old_values[start:end],policy.device),tensor(targets[start:end],policy.device),torch.stack(entropies))
            _step(policy,optimizer,result[0],result[1:],metrics)
    return _finish(metrics)
