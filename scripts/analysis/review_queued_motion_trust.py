"""Posthoc full raw review, separate from the frozen scientific snapshot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys


def review(task,source):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_motion_trust_experiment as trial
    from utils.queued_service_diagnostics import mc_returns
    from marl_models.queued_motion_trust import MotionBatch
    from marl_models.queued_service_joint import tensor
    import torch
    torch.set_num_threads(1);base=trial.base;out=trial.out(task);load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=load(out/'protocol.json');analysis=load(out/'analysis.json')
    continuation=load(out/'continuation.json') if (out/'continuation.json').exists() else None
    original_source=task/continuation['original_source'] if continuation else source
    for name,digest in protocol['source_sha'].items():assert base.sha(original_source/name)==digest,name
    if continuation:
        for name,digest in continuation['source_sha'].items():assert base.sha(source/name)==digest,name
        for name,digest in continuation['protected'].items():assert base.sha(task/name)==digest,name
        assert base.sha(out/'protocol.json')==continuation['original_protocol_sha']
    for b in range(3):
        initial=torch.load(trial.study.checkpoint(task,b,2),map_location='cpu',weights_only=False)['model']
        assert base.sha(trial.study.checkpoint(task,b,2))==protocol['parent_sha'][str(b)]
        for arm in trial.ARMS:
            m=trial.model_at(str(task),b,arm,0);assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items())
            for v in (1,2):
                m=trial.model_at(str(task),b,arm,v)
                assert all(torch.equal(value,m.state_dict()[k]) for k,value in initial.items() if k.startswith('allocation.'))
    episodes=[load(p) for p in (out/'episodes').glob('*.json')];train=[r for r in episodes if r['training']];ev=[r for r in episodes if not r['training']]
    assert len(train)==576 and len(ev)==288
    for r in episodes:
        c=r['counts'];assert c['generated']==c['success']+c['expired'];assert sum(x['success'] for x in r['rows'])==c['success'];assert r['rows'][-1]['active']==0
        if r['training']:
            p=out/'experience'/f"{r['arm']}_{r['version']}_{r['block']}_{r['workload']}_train.pkl.gz";assert base.sha(p)==r['experience_sha']
            rows=base.read(p);targets=mc_returns([x['reward'] for x in rows]);assert all(abs(x['returns']-target)<1e-9 for x,target in zip(rows,targets))
            assert abs(sum(x['reward'] for x in rows)-c['success']/20)<1e-8
    measured=[]
    for path in (out/'updates').glob('*.json'):
        d=load(path);rows=[];b=d['block'];arm=d['arm'];v=d['version'];metrics=d['metrics']
        for w in d['workloads']:rows+=base.read(out/'experience'/f'{arm}_{v-1}_{b}_{w}_train.pkl.gz')
        batch=MotionBatch(trial.model_at(str(task),b,arm,v),rows);probs=batch.probabilities();kl=MotionBatch.kl(batch.old,probs)
        assert abs(kl-metrics['measured_full_kl'])<2e-6
        assert base.sha(trial.checkpoint(task,b,arm,v))==d['checkpoint_sha']
        assert metrics['critic_steps']==2*math.ceil(len(rows)/256)
        saved=torch.load(trial.checkpoint(task,b,arm,v),map_location='cpu',weights_only=False)['optimizer_states']
        accepted=metrics['actor_steps'] if arm=='trust' else metrics['actor_steps']['motion']
        if v>1:
            previous=load(out/'updates'/f'{arm}_{v-1}_{b}.json')['metrics']['actor_steps']
            accepted+=previous if arm=='trust' else previous['motion']
        assert max(float(x['step']) for x in saved['motion']['state'].values())==accepted
        assert max(float(x['step']) for x in saved['critic']['state'].values())==v*metrics['critic_steps']
        assert all(group['lr']==.0003 for group in saved['motion']['param_groups'])
        if arm=='trust':
            assert kl<=.020001 and metrics['actor_steps']==len(metrics['accepted_trace'])
            assert metrics['actor_steps']+metrics['actor_skips']==metrics['planned_steps']==2*math.ceil(len(rows)/256)
            assert all(t['full_kl']<=.020001 and t['step_kl']<=.0002001 for t in metrics['accepted_trace'])
        measured.append(dict(block=b,arm=arm,version=v,full_kl=kl,probability_l1=float((batch.old-probs).abs().sum(-1).mean())))
    state_index=[];branches=0;chunks=0
    for b in range(3):
        for w in trial.RANK:
            for t in trial.TIMES:
                snap=trial.state_at(str(task),b,w,t);k=len(snap['pre']['actions']);pre=snap['pre'];post=snap['post']
                assert pre['actions']==post['actions']
                for mode in trial.MODES:
                    with torch.no_grad():p=trial.mode_model(str(task),b,mode).motion(tensor([pre['state']]),tensor([pre['features']]),tensor([pre['mask']],torch.bool),tensor([post['state']]),tensor([post['features']]),True).probs[0,:k].tolist()
                    assert max(abs(x-y) for x,y in zip(p,snap['probabilities'][mode]))<1e-6
                state_index.append(dict(block=b,workload=w,time=t,k=k,initial_tasks=len(snap['engine'].tasks),ues=snap['engine'].ues,probabilities=snap['probabilities'],sha=base.sha(out/'states'/f'{b}_{w}_{t}.pkl.gz')))
                for r in trial.FUTURES:
                    future=trial.remaining_future(w,snap['engine'].ues,t,r);digest=hashlib.sha256(pickle.dumps(future,5)).hexdigest();total=len(snap['engine'].tasks)+sum(len(xs) for at,xs in future.items() if t<at<200)
                    for mode in trial.MODES:
                        indexes=[];q=[]
                        for c in range(math.ceil(k/4)):
                            d=load(out/'labels'/f'{b}_{w}_{t}_{r}_{mode}_{c}.json');chunks+=1;indexes+=d['indexes'];q+=d['values']
                            assert d['future_sha']==digest
                            for value,count in zip(d['values'],d['counts']):assert value==count['success'] and total==count['generated']==count['success']+count['expired']
                        assert indexes==list(range(k));branches+=k
                        expected=next(x for x in analysis['ranking_rows'] if x['block']==b and x['workload']==w and x['time']==t and x['mode']==mode)
                        assert q==expected['values'][trial.FUTURES.index(r)]
    assert branches==analysis['full_remaining_branches'];assert len(list((out/'labels').glob('*.json')))==chunks
    base.write(out/'state_index.json',state_index)
    result=dict(status='PASS',training=576,evaluation=288,ranking_states=48,rank_chunks=chunks,full_remaining_branches=branches,mc_experiences=576,
        measured_updates=measured,independent_review=False,scope='Full raw terminal counts and MC; reconstructed independent futures; prediction and frozen-parameter checks; post-update full-rollout KL replay. No second physical simulation or independent replication.',
        files={p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file() and 'experience' not in p.parts})
    base.write(Path(task)/'output'/base.TASK/'validation'/trial.REV/'server_review.json',result);print('PASS',576,'training',288,'evaluation',branches,'branches',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();review(a.task,a.source)
