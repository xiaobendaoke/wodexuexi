"""Matched node information, two actor structures, unchanged lower MAPPO update."""
from __future__ import annotations
import torch
import torch.nn as nn
import config
from marl_models.offload_mappo.offload_mappo import OffloadMAPPO
from marl_models.buffer_and_helpers import layer_init
from utils.node_offload_observation import NODE_FEATURE_NAMES,FEATURE_CONTRACT


class NodeActor(nn.Module):
    def __init__(self,kind,original_obs_dim,max_requests,num_actions):
        super().__init__()
        if kind not in ('flat','shared'):raise ValueError('Unknown node actor')
        self.kind=kind;self.original_obs_dim=original_obs_dim
        self.max_requests=max_requests;self.num_actions=num_actions
        self.request_dim=config.OFFLOAD_REQUEST_FEATURE_DIM;self.node_dim=len(NODE_FEATURE_NAMES)
        width=5+self.request_dim+(num_actions*self.node_dim if kind=='flat' else self.node_dim)
        hidden=config.MLP_HIDDEN_DIM
        self.net=nn.Sequential(layer_init(nn.Linear(width,hidden)),nn.LayerNorm(hidden),nn.ReLU(),
            layer_init(nn.Linear(hidden,hidden)),nn.LayerNorm(hidden),nn.ReLU(),
            layer_init(nn.Linear(hidden,num_actions if kind=='flat' else 1),std=.01))

    def forward(self,obs,masks=None):
        expected=self.original_obs_dim+self.max_requests*self.num_actions*self.node_dim
        if obs.shape[-1]!=expected or masks is None:raise ValueError('Node input shape/mask required')
        leading=obs.shape[:-1];flat=obs.reshape(-1,expected)
        mask=masks.reshape(-1,self.max_requests,self.num_actions)>0
        valid=mask.any(-1)
        req=flat[:,5:self.original_obs_dim].reshape(-1,self.max_requests,self.request_dim)
        nodes=flat[:,self.original_obs_dim:].reshape(-1,self.max_requests,self.num_actions,self.node_dim)
        req=torch.where(valid[...,None],req,torch.zeros_like(req))
        own=flat[:,:5,None].transpose(1,2).expand(-1,self.max_requests,-1)
        own=torch.where(valid[...,None],own,torch.zeros_like(own))
        nodes=torch.where(mask[...,None],nodes,torch.zeros_like(nodes))
        context=torch.cat([own,req],-1)
        if self.kind=='flat':
            logits=self.net(torch.cat([context,nodes.flatten(-2)],-1))
        else:
            context=context[:,:,None,:].expand(-1,-1,self.num_actions,-1)
            logits=self.net(torch.cat([context,nodes],-1)).squeeze(-1)
        return logits.masked_fill(~mask,-1e9).reshape(*leading,self.max_requests,self.num_actions)


class OriginalObservationCritic(nn.Module):
    def __init__(self,critic,dim):
        super().__init__();self.critic=critic;self.original_obs_dim=dim
    def forward(self,obs):return self.critic(obs[...,:self.original_obs_dim])


class NodeOffloadMAPPO(OffloadMAPPO):
    def __init__(self,kind,num_agents,original_obs_dim,action_dim,device,max_requests=None,num_actions=None):
        super().__init__('no_attention_offload_mappo',num_agents,original_obs_dim,action_dim,device,
                         max_requests=max_requests,num_actions=num_actions)
        self.model_name='node_'+kind+'_offload_mappo'
        self.obs_dim=original_obs_dim+self.max_requests*self.num_actions*len(NODE_FEATURE_NAMES)
        self.probability_contract='node_'+kind+'_joint_sum_'+FEATURE_CONTRACT
        self.actor=NodeActor(kind,original_obs_dim,self.max_requests,self.num_actions).to(device)
        self.critic=OriginalObservationCritic(self.critic,original_obs_dim).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=config.ACTOR_LR)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=config.CRITIC_LR)
