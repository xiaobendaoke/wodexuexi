"""Isolated upper-training controls; frozen SF-HLM v2 files stay unchanged."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence, Normal
from marl_models.sf_hlm import FeedbackMAPPO
from marl_models.sf_hlm_v2 import ServiceMAPPO, PlainMAPPO

KINDS = ('original_update', 'current_update', 'residual_feedback', 'residual_zero')
CONTRACT = 'sf_hlm_upper_update_and_frozen_residual_v1'


class UpperAdaptation(ServiceMAPPO):
    def __init__(self, obs_dim, num_uavs=5, *, kind):
        if kind not in KINDS:
            raise ValueError('Unknown upper adaptation kind')
        super().__init__(obs_dim, num_uavs, enabled=kind=='residual_feedback')
        import config
        self.kind = kind
        self.frozen_backbone = kind.startswith('residual_')
        if self.frozen_backbone:
            for module in (self.actor, self.critic):
                for parameter in module.parameters():parameter.requires_grad_(False)
        self.actor_parameters = list(self.actor_feedback.parameters()) if self.frozen_backbone else list(self.actor.parameters())
        self.critic_parameters = list(self.critic_feedback.parameters()) if self.frozen_backbone else list(self.critic.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=config.ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=config.CRITIC_LR)
        # R1 current_update uses the frozen wrapper's joint optimizer verbatim.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def distribution(self, obs, summary):
        if self.kind in ('original_update', 'current_update'):
            return PlainMAPPO.distribution(self, obs, summary)
        return ServiceMAPPO.distribution(self, obs, summary)

    @torch.no_grad()
    def act(self, obs, summary, exploration=False):
        dist, value = self.distribution(torch.as_tensor(obs,dtype=torch.float32)[None],
                                        torch.as_tensor(summary,dtype=torch.float32)[None])
        self.last_mean = dist.mean[0].numpy().copy()
        self.last_std = dist.stddev[0].numpy().copy()
        action = dist.sample() if exploration else dist.mean
        return action[0].numpy(),dist.log_prob(action).sum(-1)[0].numpy(),value[0].numpy()

    def rollout_batches(self, records, *, epochs, batch_size):
        import config
        from marl_models.buffer_and_helpers import AttentionRolloutBuffer
        buffer = AttentionRolloutBuffer(self.num_uavs,self.obs_dim,2,len(records),'cpu')
        for row in records:
            buffer.add(row['obs'].reshape(-1),row['obs'],row['action'],row['log_prob'],
                       row['reward'],row['done'],row['value'])
        buffer.compute_returns_and_advantages(np.zeros(self.num_uavs,np.float32),config.DISCOUNT_FACTOR,config.PPO_GAE_LAMBDA)
        # Identical temporal mini-batch indexing to AttentionRolloutBuffer, with
        # the service view carried alongside the existing observation contract.
        data = {k:torch.from_numpy(v) for k,v in dict(obs=buffer.observations,actions=buffer.actions,
                old_log_probs=buffer.log_probs,advantages=buffer.advantages,returns=buffer.returns,
                old_values=buffer.values,feedback=np.stack([r['feedback'] for r in records])).items()}
        for _ in range(epochs):
            for ids in torch.randperm(len(records)).split(batch_size):
                yield {k:v[ids] for k,v in data.items()}

    def update_batch(self, batch):
        if self.kind=='original_update':
            # Execute the frozen original class method, rather than reimplement
            # its reference actor/critic update in the primary R1 comparison.
            from marl_models.attention_mappo.attention_mappo import AttentionMAPPO
            return AttentionMAPPO.update(self,batch)
        if not self.frozen_backbone:
            raise ValueError('Only reference or residual models use separated updates')
        import config
        obs,summary=batch['obs'],batch['feedback']
        values=self.distribution(obs,summary)[1]
        clipped=batch['old_values']+(values-batch['old_values']).clamp(-config.PPO_CLIP_EPS,config.PPO_CLIP_EPS)
        critic=.5*torch.maximum((values-batch['returns']).square(),(clipped-batch['returns']).square()).mean()
        self.critic_optimizer.zero_grad(set_to_none=True);critic.backward()
        nn.utils.clip_grad_norm_(self.critic_parameters,config.MAX_GRAD_NORM);self.critic_optimizer.step()
        dist,_=self.distribution(obs,summary)
        lp=dist.log_prob(batch['actions']).sum(-1);ratio=(lp-batch['old_log_probs']).exp()
        actor=-torch.minimum(ratio*batch['advantages'],ratio.clamp(1-config.PPO_CLIP_EPS,1+config.PPO_CLIP_EPS)*batch['advantages']).mean()
        entropy=dist.entropy().mean()
        actor=actor-config.PPO_ENTROPY_COEF*entropy
        self.actor_optimizer.zero_grad(set_to_none=True);actor.backward()
        nn.utils.clip_grad_norm_(self.actor_parameters,config.MAX_GRAD_NORM);self.actor_optimizer.step()
        return dict(actor=float(actor.detach()),critic=float(critic.detach()),entropy=float(entropy.detach()))

    @torch.no_grad()
    def diagnostics(self,records):
        import config
        ratios=[];kls=[];entropies=[]
        for start in range(0,len(records),256):
            rows=records[start:start+256]
            obs=torch.tensor(np.stack([r['obs'] for r in rows]))
            view=torch.tensor(np.stack([r['feedback'] for r in rows]))
            actions=torch.tensor(np.stack([r['action'] for r in rows]))
            dist,_=self.distribution(obs,view)
            old=Normal(torch.tensor(np.stack([r['old_mean'] for r in rows])),torch.tensor(np.stack([r['old_std'] for r in rows])))
            ratio=(dist.log_prob(actions).sum(-1)-torch.tensor(np.stack([r['log_prob'] for r in rows]))).exp()
            ratios.extend(ratio.flatten().tolist());kls.extend(kl_divergence(old,dist).sum(-1).flatten().tolist())
            entropies.extend(dist.entropy().flatten().tolist())
        return dict(ratio_mean=float(np.mean(ratios)),ratio_max=float(np.max(ratios)),
                    clip_fraction=float(np.mean(np.abs(np.asarray(ratios)-1)>config.PPO_CLIP_EPS)),
                    kl_mean=float(np.mean(kls)),kl_p95=float(np.quantile(kls,.95)),entropy=float(np.mean(entropies)))

    def update_rollout(self,records,*,epochs=8,batch_size=256):
        before=self.diagnostics(records)
        if not np.isclose(before['ratio_mean'],1.,atol=2e-5) or before['kl_p95']>2e-5:
            raise ValueError('Upper rollout does not match its fixed sample-time policy')
        if self.kind=='current_update':
            loss=FeedbackMAPPO.update(self,records,epochs=epochs,batch_size=batch_size)
        else:
            rows=[self.update_batch(b) for b in self.rollout_batches(records,epochs=epochs,batch_size=batch_size)]
            loss={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
        after=self.diagnostics(records)
        if not all(np.isfinite(x) for x in list(loss.values())+list(after.values())):
            raise FloatingPointError('Nonfinite upper update')
        loss.update(initial_ratio_mean=before['ratio_mean'],post_update=after)
        return loss

    def save(self,folder,contract):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(version=CONTRACT,kind=self.kind,contract=contract,obs_dim=self.obs_dim,num_uavs=self.num_uavs,
                   modules={k:getattr(self,k).state_dict() for k in ('actor','critic','actor_feedback','critic_feedback')},
                   optimizers={k:getattr(self,k).state_dict() for k in ('optimizer','actor_optimizer','critic_optimizer')}),folder/'upper_adaptation.pt')

    def load(self,folder,contract):
        data=torch.load(Path(folder)/'upper_adaptation.pt',map_location='cpu',weights_only=False)
        if (data['version'],data['kind'],data['contract'],data['obs_dim'],data['num_uavs'])!=(CONTRACT,self.kind,contract,self.obs_dim,self.num_uavs):
            raise ValueError('Incompatible upper adaptation checkpoint')
        for k,v in data['modules'].items():getattr(self,k).load_state_dict(v)
        for k,v in data['optimizers'].items():getattr(self,k).load_state_dict(v)
