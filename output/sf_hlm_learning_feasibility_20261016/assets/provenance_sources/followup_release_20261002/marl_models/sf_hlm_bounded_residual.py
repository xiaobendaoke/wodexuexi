"""Latent-action PPO for the registered, frozen-backbone service residual pilot."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from marl_models.sf_hlm_upper_adaptation import UpperAdaptation
from marl_models.sf_hlm_v2 import ServiceMAPPO, PlainMAPPO

KINDS=('bounded_feedback','bounded_zero','unbounded_feedback','bounded_heuristic','bounded_random')
CONTRACT='sf_hlm_latent_residual_015_std05_anchor01_v1'


class BoundedResidualMAPPO(UpperAdaptation):
    delta=.15
    latent_std=.5
    anchor_coefficient=.1
    clip_epsilon=.1

    def __init__(self, obs_dim, num_uavs=5, *, kind):
        if kind not in KINDS:raise ValueError('Unknown latent residual kind')
        super().__init__(obs_dim,num_uavs,kind='residual_zero' if kind=='bounded_zero' else 'residual_feedback')
        self.kind=kind;self.bounded=kind!='unbounded_feedback'
        self.actor_optimizer=torch.optim.Adam(self.actor_parameters,lr=1e-4)
        self.critic_optimizer=torch.optim.Adam(self.critic_parameters,lr=3e-4)

    def reference_distribution(self, obs):
        return PlainMAPPO.distribution(self,obs,None)[0]

    def distribution(self, obs, summary):
        service,values=ServiceMAPPO.distribution(self,obs,summary)
        reference=self.reference_distribution(obs)
        mean=(service.mean-reference.mean)/self.delta
        return Normal(mean,torch.full_like(mean,self.latent_std)),values

    def execute_latent(self, obs, z):
        reference=self.reference_distribution(obs).mean
        shift=self.delta*(torch.tanh(z) if self.bounded else z)
        return (reference+shift).clamp(-1,1)

    @torch.no_grad()
    def act(self, obs, summary, exploration=False):
        tensor=torch.as_tensor(obs,dtype=torch.float32)[None]
        view=torch.as_tensor(summary,dtype=torch.float32)[None]
        dist,value=self.distribution(tensor,view)
        z=dist.sample() if exploration else dist.mean
        action=self.execute_latent(tensor,z)
        self.last_mean=dist.mean[0].numpy().copy();self.last_std=dist.stddev[0].numpy().copy()
        self.last_latent=z[0].numpy().copy();self.last_reference=self.reference_distribution(tensor).mean[0].numpy().copy()
        if self.bounded and np.abs(action[0].numpy()-self.last_reference).max() > self.delta+1e-6:
            raise RuntimeError('Executed residual exceeds the registered bound')
        return action[0].numpy(),dist.log_prob(z).sum(-1)[0].numpy(),value[0].numpy()

    def update_batch(self, batch):
        obs,view=batch['obs'],batch['feedback'];values=self.distribution(obs,view)[1]
        clipped=batch['old_values']+(values-batch['old_values']).clamp(-self.clip_epsilon,self.clip_epsilon)
        critic=.5*torch.maximum((values-batch['returns']).square(),(clipped-batch['returns']).square()).mean()
        self.critic_optimizer.zero_grad(set_to_none=True);critic.backward()
        nn.utils.clip_grad_norm_(self.critic_parameters,.5);self.critic_optimizer.step()
        dist,_=self.distribution(obs,view);lp=dist.log_prob(batch['actions']).sum(-1)
        ratio=(lp-batch['old_log_probs']).exp()
        actor=-torch.minimum(ratio*batch['advantages'],ratio.clamp(1-self.clip_epsilon,1+self.clip_epsilon)*batch['advantages']).mean()
        prior=Normal(torch.zeros_like(dist.mean),torch.full_like(dist.mean,self.latent_std))
        anchor=kl_divergence(dist,prior).sum(-1).mean()
        entropy=dist.entropy().mean()
        # The fixed latent variance makes this entropy term constant; retain
        # its original coefficient and report it without claiming exploration learning.
        loss=actor+self.anchor_coefficient*anchor-.005*entropy
        self.actor_optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(self.actor_parameters,.5);self.actor_optimizer.step()
        return dict(actor=float(actor.detach()),critic=float(critic.detach()),entropy=float(entropy.detach()),anchor_kl=float(anchor.detach()))

    def update_rollout(self, records, *, epochs=4, batch_size=256):
        return super().update_rollout(records,epochs=epochs,batch_size=batch_size)

    @torch.no_grad()
    def diagnostics(self, records):
        result=super().diagnostics(records)
        ratios=[]
        for start in range(0,len(records),256):
            rows=records[start:start+256]
            dist,_=self.distribution(torch.tensor(np.stack([r['obs'] for r in rows])),
                                     torch.tensor(np.stack([r['feedback'] for r in rows])))
            lp=dist.log_prob(torch.tensor(np.stack([r['action'] for r in rows]))).sum(-1)
            ratios.extend((lp-torch.tensor(np.stack([r['log_prob'] for r in rows]))).exp().flatten().tolist())
        result['clip_fraction']=float(np.mean(np.abs(np.asarray(ratios)-1)>self.clip_epsilon))
        return result

    def save(self, folder, contract):
        path=Path(folder);path.mkdir(parents=True,exist_ok=True)
        torch.save(dict(version=CONTRACT,kind=self.kind,contract=contract,obs_dim=self.obs_dim,num_uavs=self.num_uavs,
            constants=(self.delta,self.latent_std,self.anchor_coefficient,self.clip_epsilon),
            modules={k:getattr(self,k).state_dict() for k in ('actor','critic','actor_feedback','critic_feedback')},
            actor_optimizer=self.actor_optimizer.state_dict(),critic_optimizer=self.critic_optimizer.state_dict()),path/'latent_residual.pt')

    def load(self, folder, contract):
        d=torch.load(Path(folder)/'latent_residual.pt',map_location='cpu',weights_only=False)
        if (d['version'],d['kind'],d['contract'],d['obs_dim'],d['num_uavs'],d['constants']) != (
            CONTRACT,self.kind,contract,self.obs_dim,self.num_uavs,(self.delta,self.latent_std,self.anchor_coefficient,self.clip_epsilon)):
            raise ValueError('Incompatible latent residual checkpoint')
        for k,v in d['modules'].items():getattr(self,k).load_state_dict(v)
        self.actor_optimizer.load_state_dict(d['actor_optimizer']);self.critic_optimizer.load_state_dict(d['critic_optimizer'])
