"""Posthoc policy response measurements on saved states, no extra training."""
import argparse,json,math,statistics,sys
from pathlib import Path


def proposal_job(job):
    import copy,os
    import torch
    from scripts.analysis import queued_probability_decoupling as trial
    from marl_models.queued_motion_trust import MotionBatch,guarded_step
    from marl_models.queued_service_joint import tensor
    task,b,arm=job;out=trial.out(task);m=copy.deepcopy(trial.model_at(task,b,arm,0));rows=[];sources={}
    for w in range(9500,9508):
        path=out/'experience'/f'{arm}_0_{b}_{w}_train.pkl.gz';sources[path.relative_to(out).as_posix()]=trial.base.sha(path);rows+=trial.base.read(path)
    for p in m.motion.parameters():p.requires_grad_(True)
    before=copy.deepcopy(m.motion.state_dict());batch=MotionBatch(m,rows);old=batch.probabilities();target=tensor([r['returns']-r['motion_value'] for r in rows]);target=(target-target.mean())/target.std(unbiased=False).clamp_min(.05)
    torch.manual_seed(trial.base.seed(trial.REV,'proposal_probe',b));ids=torch.randperm(len(rows))[:256];dist=batch.dist(ids);ratio=(dist.log_prob(batch.actions[ids])-batch.oldlog[ids]).clamp(-20,20).exp()
    loss=-torch.minimum(ratio*target[ids],ratio.clamp(.8,1.2)*target[ids]).mean()-.001*dist.entropy().mean()
    optimizer=torch.optim.Adam(m.motion.parameters(),lr=.0003);optimizer.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(m.motion.parameters(),.5,error_if_nonfinite=True));curves=[]
    for scale in (1.,4.,16.,64.):
        measured={}
        def reject():
            p=batch.probabilities();measured.update(kl=MotionBatch.kl(old,p),l1=float((old-p).abs().sum(-1).mean()));return False
        result=guarded_step(m.motion,optimizer,reject,max_backtracks=0,initial_scale=scale)
        assert not result['accepted'] and all(torch.equal(p,m.motion.state_dict()[name]) for name,p in before.items())
        assert not optimizer.state
        curves.append(dict(scale=scale,learning_rate=.0003*scale,**measured))
    trial.base.write(out/'proposal_response'/f'{b}_{arm}.json',dict(block=b,arm=arm,rows=len(rows),raw_grad_norm=grad,curves=curves,input_sha=sources,
        all_proposals_rolled_back=True,checkpoint_saved=False,scope='Posthoc first-eight-training-workload gradient response at version0, fresh Adam, fixed 256-row sample, own behavior states. Not a learning-rate selection or a trained performance result.',cpu=sorted(__import__('os').sched_getaffinity(0))))
    return dict(cpu=sorted(os.sched_getaffinity(0)))


def proposals(task,source):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_probability_decoupling as trial
    import os
    base=trial.base;original=base.out;base.out=trial.out
    try:base.parallel(task,'proposal_response_probe',proposal_job,[(str(task),b,arm) for b in range(3) for arm in trial.ARMS],sorted(os.sched_getaffinity(0)))
    finally:base.out=original
    print('PROPOSAL RESPONSE COMPLETE: all proposals rolled back, no checkpoints',flush=True)


def diagnose(task,source):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_probability_decoupling as trial
    out=trial.out(task);load=lambda p:json.loads(p.read_text(encoding='utf-8'));data=load(out/'analysis.json');updates=[load(p) for p in (out/'updates').glob('*.json')];summary={};changes={}
    for arm in trial.ARMS:
        us=[d for d in updates if d['arm']==arm];traces=[t for d in us for t in d['metrics']['accepted_trace']]
        summary[arm]=dict(accepted=sum(d['metrics']['actor_steps'] for d in us),planned=sum(d['metrics']['planned_steps'] for d in us),
            full_kl=[dict(block=d['block'],version=d['version'],value=d['metrics']['final_full_kl'],l1=d['metrics']['full_probability_l1']) for d in us],
            median_step_kl=statistics.median(t['step_kl'] for t in traces),median_scale=statistics.median(t['scale'] for t in traces),backtracks=sum(t['backtracks'] for t in traces))
    for mode in trial.MODES:
        rs=data['probability'][mode];changes[mode]={}
        reference='calibrated_initial' if mode=='decoupled' else 'raw_initial'
        for t in trial.TIMES:
            selected=[r for r in rs if r['time']==t];flips=0
            for r in selected:
                ref=next(x for x in data['probability'][reference] if x['block']==r['block'] and x['workload']==r['workload'] and x['time']==t)
                flips+=r['argmax']!=ref['argmax']
            changes[mode][str(t)]=dict(states=len(selected),mean_entropy=statistics.mean(r['entropy'] for r in selected),saturated=sum(r['pmax']>=.999 for r in selected),
                changed_argmax=flips,reference=reference,mean_tv_from_calibrated=statistics.mean(r['tv_from_calibrated'] for r in selected))
    result=dict(status='COMPLETE',updates=summary,probabilities=changes,training=False,extra_rollouts=0,
        scope='Posthoc saved-state entropy/first-choice changes and actual conditional KL response. Small KL alone cannot identify whether gradients, learning rate, initialization or target noise limit learning.')
    trial.base.write(out/'response_diagnosis.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--proposals',action='store_true');a=p.parse_args()
    (proposals if a.proposals else diagnose)(a.task,a.source)
