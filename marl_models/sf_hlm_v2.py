"""Versioned quota PPO, grouped policy gradients and service-conditioned MAPPO."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical,Normal
from marl_models.sf_hlm import QuotaNetwork,QuotaPPO,FeedbackMAPPO,mlp
from utils.sf_hlm_v2 import NODE_DIM,EDGE_DIM,INPUT_CONTRACT,SEARCH_CONTRACT,ARCHIVE_CONTRACT


def collate(states):
    r=max(1,max(len(s['requests']) for s in states));j=len(states[0]['nodes']);b=len(states)
    arrays=dict(requests=np.zeros((b,r,4),np.float32),edges=np.zeros((b,r,j,EDGE_DIM),np.float32),
                legal=np.zeros((b,r,j),bool),nodes=np.zeros((b,j,NODE_DIM),np.float32),mask=np.zeros((b,len(states[0]['mask'])),bool))
    for i,s in enumerate(states):
        n=len(s['requests'])
        for k in ('requests','edges','legal'):arrays[k][i,:n]=s[k]
        arrays['nodes'][i]=s['nodes'];arrays['mask'][i]=s['mask']
    if any(not np.isfinite(x).all() for x in arrays.values()):raise ValueError('Nonfinite v2 graph')
    return {k:torch.from_numpy(v) for k,v in arrays.items()}


def embeddings(network,g):
    legal=g['legal'];weight=legal[...,None].float();valid=legal.any(-1)[...,None].float()
    req=network.request(g['requests'])*valid;node=network.node(g['nodes']);edge=network.edge(g['edges'])*weight
    b,r,j,h=edge.shape
    for msg,ru,nu in zip(network.messages,network.req_updates,network.node_updates):
        pair=torch.cat([req[:,:,None].expand(-1,-1,j,-1),node[:,None].expand(-1,r,-1,-1),edge],-1)
        m=msg(pair)*weight;req=ru(torch.cat([req,m.sum(2)/weight.sum(2).clamp_min(1)],-1))*valid
        node=nu(torch.cat([node,m.sum(1)/weight.sum(1).clamp_min(1)],-1))
    pooled=torch.cat([req.sum(1)/valid.sum(1).clamp_min(1),node.mean(1)],-1)
    return node,pooled


class CompleteQuotaPPO(QuotaPPO):
    family='complete'
    def __init__(self,num_uavs=5,hidden=128,learning_rate=3e-4):
        super().__init__(num_uavs,hidden,learning_rate);self.network.node=mlp(NODE_DIM,hidden,hidden);self.network.edge=mlp(EDGE_DIM,hidden,hidden)
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=learning_rate)

    def forward_states(self,states):return self.network(collate(states))

    def save(self,folder,contract):
        path=Path(folder);path.mkdir(parents=True,exist_ok=True)
        torch.save(dict(network=self.network.state_dict(),optimizer=self.optimizer.state_dict(),contract=contract,
                        input=INPUT_CONTRACT,search=SEARCH_CONTRACT,family=self.family,num_uavs=self.num_uavs,hidden=self.hidden),path/'quota_v2.pt')

    def load(self,folder,contract):
        d=torch.load(Path(folder)/'quota_v2.pt',weights_only=False,map_location='cpu')
        if d['contract']!=contract or d['input']!=INPUT_CONTRACT or d['search']!=SEARCH_CONTRACT or d['family']!=self.family or d['num_uavs']!=self.num_uavs:
            raise ValueError('Incompatible v2 quota checkpoint')
        self.network.load_state_dict(d['network']);self.optimizer.load_state_dict(d['optimizer'])


class ArchiveNetwork(QuotaNetwork):
    def __init__(self,num_uavs=5,hidden=128):
        super().__init__(num_uavs,hidden);self.node=mlp(NODE_DIM,hidden,hidden);self.edge=mlp(EDGE_DIM,hidden,hidden)
        self.rank=mlp(hidden*4+3,hidden,1)

    def forward_archives(self,states):
        graphs=[p for s in states for p in s['parents']];nodes,pooled=embeddings(self,collate(graphs))
        logits=[];offset=0;maximum=max(len(s['tokens']) for s in states)
        for s in states:
            tokens=torch.as_tensor(s['tokens'],dtype=torch.long);valid=tokens[:,0]>=0
            if not valid.any():scores=pooled[offset,:1]*0
            else:
                parent=tokens[:,0]+offset;source=tokens[:,1];dest=tokens[:,2]
                r=max(len(s['parents'][0]['requests']),1)
                features=torch.cat([nodes[parent,source],nodes[parent,dest],pooled[parent],tokens[:,3:].float()/r],-1)
                scores=self.rank(features).squeeze(-1)
            scores=scores.masked_fill(~torch.as_tensor(s['mask']),-1e9)
            logits.append(torch.nn.functional.pad(scores,(0,maximum-len(scores)),value=-1e9));offset+=len(s['parents'])
        return torch.stack(logits)


class ArchivePolicy:
    family='critical'
    def __init__(self,num_uavs=5,hidden=128,learning_rate=3e-4):
        self.num_uavs=num_uavs;self.hidden=hidden;self.network=ArchiveNetwork(num_uavs,hidden)
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=learning_rate)

    @torch.no_grad()
    def act(self,state,exploration=False):
        logits=self.network.forward_archives([state]);dist=Categorical(logits=logits)
        a=dist.sample() if exploration else logits.argmax(-1)
        return int(a[0]),float(dist.log_prob(a)[0]),0.

    def evaluate(self,states,actions):
        dist=Categorical(logits=self.network.forward_archives(states));return dist.log_prob(actions),dist.entropy()

    def update_groups(self,groups):
        if not groups:return dict(pg_loss=0.,groups=0,initial_ratio_mean=1.)
        losses=[];ratios=[];entropies=[]
        for group in groups:
            returns=np.asarray([t['return'] for t in group],np.float64)
            if len(group)!=8:raise ValueError('Expected eight independent same-instance trajectories')
            baseline=(returns.sum()-returns)/7.
            for index,t in enumerate(group):
                records=t['records']
                if not records:continue
                # Chunk graph tensors to keep per-update memory independent of
                # request count. Sum log probabilities is the trajectory score.
                lps=[];ens=[]
                for start in range(0,len(records),16):
                    chunk=records[start:start+16];actions=torch.tensor([x['action'] for x in chunk])
                    lp,en=self.evaluate([x['state'] for x in chunk],actions)
                    old=torch.tensor([x['log_prob'] for x in chunk]);ratios.extend(torch.exp(lp.detach()-old).tolist())
                    lps.append(lp.sum());ens.append(en.mean())
                losses.append(-(float(returns[index]-baseline[index]))*torch.stack(lps).sum())
                entropies.append(torch.stack(ens).mean())
        if not np.allclose(ratios,1.,atol=2e-5):raise ValueError('Grouped trajectories are not from the frozen policy version')
        if not losses:return dict(pg_loss=0.,groups=len(groups),initial_ratio_mean=1.)
        loss=torch.stack(losses).sum()/(len(groups)*8)-.005*torch.stack(entropies).mean()
        self.optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.network.parameters(),.5);self.optimizer.step()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite grouped policy gradient')
        return dict(pg_loss=float(loss.detach()),groups=len(groups),initial_ratio_mean=float(np.mean(ratios)))

    def save(self,folder,contract):
        path=Path(folder);path.mkdir(parents=True,exist_ok=True)
        torch.save(dict(network=self.network.state_dict(),optimizer=self.optimizer.state_dict(),contract=contract,
            input=INPUT_CONTRACT,search=ARCHIVE_CONTRACT,family=self.family,num_uavs=self.num_uavs,hidden=self.hidden),path/'quota_v2.pt')

    def load(self,folder,contract):
        d=torch.load(Path(folder)/'quota_v2.pt',weights_only=False,map_location='cpu')
        if d['contract']!=contract or d['input']!=INPUT_CONTRACT or d['search']!=ARCHIVE_CONTRACT or d['num_uavs']!=self.num_uavs or d['family']!=self.family:
            raise ValueError('Incompatible archive policy checkpoint')
        self.network.load_state_dict(d['network']);self.optimizer.load_state_dict(d['optimizer'])


UPPER_CONTRACT='visible_node_id_aligned_service_ue_attention_v2'


def service_view(env,summary):
    """Read already-generated predecision data, without calling _get_obs()."""
    import config
    output=np.zeros((len(env.uavs),len(env.uavs)+1,10),np.float32)
    scale=np.array([config.AREA_WIDTH,config.AREA_HEIGHT])
    for i,u in enumerate(env.uavs):
        visible=[u]+sorted(u.neighbors,key=lambda n:float(np.linalg.norm(u.pos-n.pos)))[:config.MAX_UAV_NEIGHBORS]
        for target in visible:
            output[i,target.id,:6]=summary[target.id];output[i,target.id,6:8]=(target.pos[:2]-u.pos[:2])/scale
            output[i,target.id,9]=1.
        output[i,-1,:6]=summary[-1];output[i,-1,6:8]=(np.asarray(config.MBS_POS)[:2]-u.pos[:2])/scale
        output[i,-1,8:]=1.
    return output


class ServiceBranch(nn.Module):
    def __init__(self,h=64):
        super().__init__();self.node=mlp(9,h,h);self.relative=mlp(2,h,8)
        self.attention=nn.MultiheadAttention(h,8,batch_first=True);self.residual=nn.Linear(h,h)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)

    def forward(self,ues,ue_raw,view):
        nodes=self.node(view[...,:9]);relative=ue_raw[:,:,:2,None].transpose(-1,-2)-view[:,None,:,6:8]
        bias=self.relative(relative).permute(0,3,1,2).masked_fill(~view[:,None,None,:,9].bool(),float('-inf'))
        context,_=self.attention(ues,nodes,nodes,attn_mask=bias.flatten(0,1),need_weights=False)
        return ues+self.residual(context)


class ServiceMAPPO(FeedbackMAPPO):
    def __init__(self,obs_dim,num_uavs=5,enabled=True):
        super().__init__(obs_dim,num_uavs,enabled);self.actor_feedback=ServiceBranch(self.actor.hidden_dim)
        self.critic_feedback=mlp(num_uavs*(num_uavs+1)*10,64,num_uavs)
        nn.init.zeros_(self.critic_feedback[-1].weight);nn.init.zeros_(self.critic_feedback[-1].bias)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=3e-4)

    def warm_start(self,base):
        self.actor.load_state_dict(getattr(base.actor,'_orig_mod',base.actor).state_dict())
        self.critic.load_state_dict(getattr(base.critic,'_orig_mod',base.critic).state_dict())

    def distribution(self,obs,summary):
        a=self.actor;b=obs.shape[0];flat=obs.reshape(-1,obs.shape[-1]);view=summary.reshape(-1,self.num_uavs+1,10).clone()
        if not self.enabled:view[...,:6]=0.
        own=flat[:,:a.own_dim];neighbors=flat[:,a.own_dim:a.own_dim+a.neighbor_block_size].reshape(-1,a.num_neighbors,a.neighbor_obs_dim)
        ues=flat[:,a.own_dim+a.neighbor_block_size:].reshape(-1,a.num_ues,a.ue_obs_dim)
        se=a.self_encoder(own);ne=a.neighbor_encoder(neighbors);ue=a.ue_encoder(ues)
        ue=self.actor_feedback(ue,ues,view)
        nc=a.neighbor_attn(se,ne,mask=(neighbors.abs().sum(-1)>1e-5).float())
        uc=a.ue_attn(se,ue,mask=(ues.abs().sum(-1)>1e-5).float())
        x=torch.relu(a.ln1(a.fc1(torch.cat([se,nc,uc],-1))));x=torch.relu(a.ln2(a.fc2(x)))
        mean=torch.tanh(a.mean(x));std=a.log_std.clamp(__import__('config').LOG_STD_MIN,__import__('config').LOG_STD_MAX).exp().expand_as(mean)
        critic_input=summary.clone()
        if not self.enabled:critic_input[...,:6]=0.
        values=self.critic(obs)+self.critic_feedback(critic_input.reshape(b,-1))
        return Normal(mean.reshape(b,self.num_uavs,2),std.reshape(b,self.num_uavs,2)),values

    def save(self,folder,contract):
        path=Path(folder);path.mkdir(parents=True,exist_ok=True)
        torch.save(dict(**{k:getattr(self,k).state_dict() for k in ('actor','critic','actor_feedback','critic_feedback')},
            optimizer=self.optimizer.state_dict(),enabled=self.enabled,num_uavs=self.num_uavs,obs_dim=self.obs_dim,
            upper=UPPER_CONTRACT,contract=contract),path/'feedback_v2.pt')

    def load(self,folder,contract):
        d=torch.load(Path(folder)/'feedback_v2.pt',weights_only=False,map_location='cpu')
        if d['contract']!=contract or d['enabled']!=self.enabled or d['num_uavs']!=self.num_uavs or d['obs_dim']!=self.obs_dim or d['upper']!=UPPER_CONTRACT:raise ValueError('Incompatible service attention checkpoint')
        for k in ('actor','critic','actor_feedback','critic_feedback'):getattr(self,k).load_state_dict(d[k])
        self.optimizer.load_state_dict(d['optimizer'])


class PlainMAPPO(ServiceMAPPO):
    """Original attention architecture for greedy/original-lower controls."""
    def distribution(self,obs,summary):
        b=obs.shape[0];raw=self.actor(obs.reshape(-1,obs.shape[-1]))
        return Normal(raw.mean.reshape(b,self.num_uavs,2),raw.stddev.reshape(b,self.num_uavs,2)),self.critic(obs)
