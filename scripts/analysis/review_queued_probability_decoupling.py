"""Raw MC, frozen teachers, calibrated probabilities and independent Q audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys


def review(task,source):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_probability_decoupling as trial
    from utils.queued_probability_scale import calibrate
    from utils.queued_service_diagnostics import mc_returns
    from marl_models.queued_motion_trust import MotionBatch
    from marl_models.queued_service_joint import tensor
    import torch
    torch.set_num_threads(1);base=trial.base;out=trial.out(task);load=lambda p:json.loads(p.read_text(encoding='utf-8'));protocol=load(out/'protocol.json');analysis=load(out/'analysis.json')
    for name,digest in protocol['source_sha'].items():assert base.sha(source/name)==digest,name
    for b in range(3):
        assert base.sha(trial.study.checkpoint(task,b,2))==protocol['parent_sha'][str(b)]
        parent=trial.study.model_at(str(task),b,2);c=load(out/'calibration'/f'{b}.json');derived=calibrate(c['rows'],.5)
        assert abs(c['temperature']-derived['temperature'])<1e-12 and c['calibrated'];assert len(c['rows'])==320 and c['future_labels_used'] is False
        for name,digest in c['input_sha'].items():assert base.sha(task/name)==digest,name
        for arm in trial.ARMS:
            for v in (0,1,2):
                m=trial.model_at(str(task),b,arm,v)
                assert all(torch.equal(p,m.allocation.state_dict()[name]) for name,p in parent.allocation.state_dict().items())
                if arm!='raw':assert all(torch.equal(p,m.motion.value_scorer.state_dict()[name]) for name,p in parent.motion.state_dict().items())
                elif v==0:assert all(torch.equal(p,m.state_dict()[name]) for name,p in parent.state_dict().items())
        raw=trial.model_at(str(task),b,'raw_residual',0);norm=trial.model_at(str(task),b,'decoupled',0)
        assert all(torch.equal(p,norm.motion.policy_net.state_dict()[name]) for name,p in raw.motion.policy_net.state_dict().items())
        assert abs(float(norm.motion.policy_temperature)-c['temperature'])<1e-6
    records=[load(p) for p in (out/'episodes').glob('*.json')];train=[r for r in records if r['training']];ev=[r for r in records if not r['training']]
    assert len(train)==864 and len(ev)==480
    for r in records:
        c=r['counts'];assert c['generated']==c['success']+c['expired'];assert sum(x['success'] for x in r['rows'])==c['success'] and r['rows'][-1]['active']==0
        if r['training']:
            p=out/'experience'/f"{r['arm']}_{r['version']}_{r['block']}_{r['workload']}_train.pkl.gz";assert base.sha(p)==r['experience_sha'];rows=base.read(p);target=mc_returns([x['reward'] for x in rows])
            assert all(abs(x['returns']-y)<1e-9 for x,y in zip(rows,target));assert abs(sum(x['reward'] for x in rows)-c['success']/20)<1e-8
    measured=[];updates=[load(p) for p in (out/'updates').glob('*.json')];assert len(updates)==18
    for d in updates:
        b,arm,v=d['block'],d['arm'],d['version'];rows=[];metrics=d['metrics']
        for w in d['workloads']:rows+=base.read(out/'experience'/f'{arm}_{v-1}_{b}_{w}_train.pkl.gz')
        batch=MotionBatch(trial.model_at(str(task),b,arm,v),rows);p=batch.probabilities();kl=MotionBatch.kl(batch.old,p)
        assert abs(kl-metrics['final_full_kl'])<2e-6 and kl<=.020001
        assert metrics['actor_steps']==len(metrics['accepted_trace']);assert metrics['actor_steps']+metrics['actor_skips']==metrics['planned_steps']==2*math.ceil(len(rows)/256)
        assert metrics['critic_steps']==2*math.ceil(len(rows)/256);assert metrics['replay_error']<2e-4
        assert all(x['full_kl']<=.020001 and x['step_kl']<=.0002001 for x in metrics['accepted_trace'])
        pth=trial.checkpoint(task,b,arm,v);assert base.sha(pth)==d['checkpoint_sha'];saved=torch.load(pth,map_location='cpu',weights_only=False)['optimizer_states']
        accepted=metrics['actor_steps']+(load(out/'updates'/f'{arm}_{v-1}_{b}.json')['metrics']['actor_steps'] if v>1 else 0)
        assert max(float(x['step']) for x in saved['motion']['state'].values())==accepted
        if arm!='raw':
            ids=saved['motion']['param_groups'][0]['params'];names=list(trial.model_at(str(task),b,arm,v).motion.named_parameters())
            assert all(ids[i] not in saved['motion']['state'] for i,(name,_) in enumerate(names) if name.startswith('value_scorer.'))
        measured.append(dict(block=b,arm=arm,version=v,full_kl=kl,accepted=metrics['actor_steps']))
    states=[];branches=0;chunks=0
    for b in range(3):
        for w in trial.RANK:
            for t in trial.TIMES:
                snap=trial.state_at(str(task),b,w,t);k=len(snap['pre']['actions']);pre,post=snap['pre'],snap['post'];assert pre['actions']==post['actions']
                for mode in trial.MODES:
                    with torch.no_grad():p=trial.mode_model(str(task),b,mode).motion(tensor([pre['state']]),tensor([pre['features']]),tensor([pre['mask']],torch.bool),tensor([post['state']]),tensor([post['features']]),True).probs[0,:k].tolist()
                    assert max(abs(x-y) for x,y in zip(p,snap['probabilities'][mode]))<1e-6
                states.append(dict(block=b,workload=w,time=t,k=k,initial_tasks=len(snap['engine'].tasks),ues=snap['engine'].ues,probabilities=snap['probabilities']))
                for r in trial.FUTURES:
                    future=trial.remaining_future(w,snap['engine'].ues,t,r);digest=hashlib.sha256(pickle.dumps(future,5)).hexdigest();total=len(snap['engine'].tasks)+sum(len(xs) for at,xs in future.items() if t<at<200);q=[]
                    for c in range(math.ceil(k/4)):
                        d=load(out/'labels'/f'{b}_{w}_{t}_{r}_{c}.json');assert d['future_sha']==digest;assert d['indexes']==list(range(4*c,min(4*c+4,k)));q+=d['values'];chunks+=1
                        for value,count in zip(d['values'],d['counts']):assert value==count['success'] and total==count['generated']==count['success']+count['expired']
                    assert len(q)==k;branches+=k
                    for mode in trial.MODES:
                        row=next(x for x in analysis['ranking_rows'] if x['block']==b and x['workload']==w and x['time']==t and x['mode']==mode);assert q==row['values'][trial.FUTURES.index(r)]
    assert branches==analysis['full_remaining_branches'] and chunks==len(list((out/'labels').glob('*.json')))
    base.write(out/'state_index.json',states)
    result=dict(status='PASS',training=864,evaluation=480,updates=18,ranking_states=24,rank_chunks=chunks,full_remaining_branches=branches,measured_updates=measured,
        independent_review=False,scope='Full raw counts and MC, parent/allocation/teacher freeze, calibration arithmetic and sources, checkpoint full KL and Adam replay, future fingerprints and frozen predictions. No independent physical replication.',
        files={p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file() and 'experience' not in p.parts})
    base.write(task/'output'/base.TASK/'validation'/trial.REV/'server_review.json',result);print('PASS',864,'training',480,'evaluation',branches,'branches',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();review(a.task,a.source)
