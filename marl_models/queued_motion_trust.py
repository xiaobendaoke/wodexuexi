"""Motion-only PPO with post-proposal full-rollout KL and Adam rollback.

Frozen scientific modules from earlier experiments remain unchanged.
"""
import copy
import math
import torch
from marl_models.queued_service_joint import tensor
from marl_models.queued_long_value import prefix_values


def guarded_step(module,optimizer,accept,max_backtracks=12,initial_scale=1.):
    parameters=copy.deepcopy(module.state_dict());state=copy.deepcopy(optimizer.state_dict())
    rates=[g['lr'] for g in optimizer.param_groups]
    for backtracks in range(max_backtracks+1):
        module.load_state_dict(parameters);optimizer.load_state_dict(copy.deepcopy(state))
        scale=initial_scale*2.**(-backtracks)
        for g,rate in zip(optimizer.param_groups,rates):g['lr']=rate*scale
        optimizer.step()
        if accept():
            for g,rate in zip(optimizer.param_groups,rates):g['lr']=rate
            return dict(accepted=True,backtracks=backtracks,scale=scale)
    module.load_state_dict(parameters);optimizer.load_state_dict(state)
    return dict(accepted=False,backtracks=max_backtracks,scale=0.)


class MotionBatch:
    def __init__(self,model,rows):
        self.model=model;self.rows=rows
        self.x=[tensor([r[key] for r in rows],torch.bool if key=='motion_mask' else torch.float32)
                for key in ('motion_state','motion_features','motion_mask','post_state','post_features')]
        self.old=tensor([r['motion_probs'] for r in rows]);self.actions=torch.tensor([r['motion_action'] for r in rows])
        self.oldlog=tensor([r['motion_log'] for r in rows])

    def dist(self,ids):return self.model.motion(*(x[ids] for x in self.x),True)

    @torch.no_grad()
    def probabilities(self):return torch.cat([self.dist(slice(i,i+512)).probs for i in range(0,len(self.rows),512)])

    @staticmethod
    def kl(p,q):return float((p*(p.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum(-1).mean())


def update_motion(model,rows,actor_epochs=2,critic_epochs=2,optimizer_states=None,limit=.02,step_limit=.0002):
    for name in ('motion','allocation','critic','prefix_value'):
        for p in getattr(model,name).parameters():p.requires_grad_(name!='allocation')
    frozen=copy.deepcopy(model.allocation.state_dict());start=copy.deepcopy(model.motion.state_dict())
    state={} if optimizer_states is None else dict(optimizer_states)
    actor=torch.optim.Adam(model.motion.parameters(),lr=.0003)
    critic=torch.optim.Adam(list(model.critic.parameters())+list(model.prefix_value.parameters()),lr=.0003)
    if 'motion' in state:actor.load_state_dict(state['motion'])
    if 'critic' in state:critic.load_state_dict(state['critic'])
    batch=MotionBatch(model,rows);current=batch.probabilities()
    with torch.no_grad():
        replay=max(float((batch.dist(slice(i,i+512)).log_prob(batch.actions[i:i+512])-batch.oldlog[i:i+512]).abs().max()) for i in range(0,len(rows),512))
    assert replay<2e-4,replay
    target=tensor([r['returns'] for r in rows]);a=target-tensor([r['motion_value'] for r in rows]);a=(a-a.mean())/a.std(unbiased=False).clamp_min(.05)
    attempts=[];trace=[];csteps=0;scale=1.;no_progress=0;stopped=False
    for epoch in range(max(actor_epochs,critic_epochs)):
        for ids in torch.randperm(len(rows)).split(256):
            if epoch<actor_epochs and not stopped:
                dist=batch.dist(ids);ratio=(dist.log_prob(batch.actions[ids])-batch.oldlog[ids]).clamp(-20,20).exp()
                loss=-torch.minimum(ratio*a[ids],ratio.clamp(.8,1.2)*a[ids]).mean()-.001*dist.entropy().mean()
                actor.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.motion.parameters(),.5,error_if_nonfinite=True)
                proposal={}
                def accept():
                    q=batch.probabilities();full=MotionBatch.kl(batch.old,q);increment=MotionBatch.kl(current,q)
                    proposal.update(probabilities=q,full_kl=full,step_kl=increment)
                    return math.isfinite(full) and math.isfinite(increment) and full<=limit and increment<=step_limit
                d=guarded_step(model.motion,actor,accept,initial_scale=scale)
                attempts.append(dict(**d,last_trial_full_kl=proposal['full_kl'],last_trial_step_kl=proposal['step_kl']))
                if d['accepted']:
                    current=proposal['probabilities'];no_progress=0
                    trace.append(dict(full_kl=proposal['full_kl'],step_kl=proposal['step_kl'],scale=d['scale'],backtracks=d['backtracks']))
                    scale=min(1.,d['scale']*(1.25 if proposal['step_kl']<step_limit/4 else 1.))
                else:
                    no_progress+=1
                    if no_progress>=2:stopped=True
            if epoch<critic_epochs:
                selected=[rows[i] for i in ids.tolist()];z=model.replay(selected);pv,owners,tokens=prefix_values(model,selected)
                loss=((z['motion_value']-target[ids])/100).square().mean()
                if tokens:loss=loss+((pv-target[ids][torch.tensor(owners)])/100).square().mean()
                critic.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(model.critic.parameters())+list(model.prefix_value.parameters()),.5,error_if_nonfinite=True);critic.step();csteps+=1
    assert all(torch.equal(v,model.allocation.state_dict()[k]) for k,v in frozen.items())
    final=batch.probabilities();kl=MotionBatch.kl(batch.old,final);assert kl<=limit+1e-7
    state['motion']=actor.state_dict();state['critic']=critic.state_dict()
    planned=actor_epochs*math.ceil(len(rows)/256)
    return dict(actor_steps=len(trace),actor_skips=planned-len(trace),planned_steps=planned,critic_steps=csteps,
        final_full_kl=kl,full_probability_l1=float((batch.old-final).abs().sum(-1).mean()),replay_error=replay,
        parameter_change=sum(float((v-start[k]).square().sum()) for k,v in model.motion.state_dict().items())**.5,
        attempts=attempts,accepted_trace=trace,stopped=stopped,allocation_frozen=True,_optimizer_states=state)
