"""Full-return value-label, closed-loop and parameter-freeze audit."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import sys


def review(task,source,torch_checks=False):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_long_value_experiment as trial
    from utils.queued_long_value import remaining_future
    from environment.queued_service import safe_path
    base=trial.base;out=trial.out(task);load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=load(out/'protocol.json');analysis=load(out/'analysis.json');label_count=0;branches=0;states=0
    assert not set(protocol['test'])&set(sum(protocol['train'].values(),[])+sum(protocol['validation'].values(),[]))
    @lru_cache(maxsize=128)
    def future_signature(w,t,r):
        from utils.queued_joint_opportunity import geometry
        ues=dict(enumerate(geometry(w).ue_positions));future=remaining_future(w,ues,t,r)
        return hashlib.sha256(pickle.dumps(future,5)).hexdigest(),sum(len(row) for when,row in future.items() if t<when<200)
    for iteration in (1,2):
        for b in range(3):
            for split,field in (('train','train'),('valid','validation')):
                for w in protocol[field][str(iteration)]:
                    for t in protocol['times']:
                        snap=base.read(out/'states'/f'{iteration}_{b}_{w}_{t}.pkl.gz');states+=1
                        assert snap['split']==split and snap['engine'].t==t
                        assert all(safe_path(snap['engine'].positions,u,snap['engine'].p) for u in snap['pre']['actions'])
                        assert snap['pre']['actions']==snap['post']['actions'];k=len(snap['pre']['actions'])
                        for r in range(protocol['replicas']):
                            indexes=[];expected_sha,number=future_signature(w,t,r)
                            for c in range(math.ceil(k/4)):
                                d=load(out/'labels'/f'{iteration}_{b}_{w}_{t}_{r}_{c}.json');label_count+=1
                                assert d['future_sha']==expected_sha
                                assert d['indexes']==list(range(c*4,min((c+1)*4,k)));indexes+=d['indexes']
                                for q,count in zip(d['values'],d['counts']):
                                    assert count['generated']==len(snap['engine'].tasks)+number
                                    assert q==count['success'] and count['success']+count['expired']==count['generated']
                                    assert 200-t<=count['elapsed']<=204-t;branches+=1
                            assert indexes==list(range(k))
            for bank in ('old','expanded'):
                fit=load(out/'fit'/f'{iteration}_{b}_{bank}.json')
                assert fit['allocation_frozen'] and fit['metrics']['epochs']==100
                assert fit['metrics']['steps']==100*math.ceil(fit['train_states']/16)
                for stage in ('initial','final'):
                    rows=fit[stage]['rows']
                    assert len(rows)==16
                    reconstructed=trial.dataset(str(task),b,iteration,'valid',bank)
                    for row,truth in zip(rows,reconstructed):
                        assert row['workload']==truth['workload'] and row['time']==truth['time']
                        assert row['values']==truth['values']
                        assert abs(sum(row['probabilities'])-1)<1e-5
                        expected=sum(a*b for a,b in zip(row['values'],row['probabilities']))-statistics.mean(row['values'])
                        assert abs(expected-row['gain'])<1e-10
                    assert abs(statistics.mean(row['gain'] for row in rows)-fit[stage]['gain'])<1e-10
    records=[load(p) for p in (out/'evaluation').glob('*.json')];assert len(records)==432
    for r in records:
        c=r['counts'];assert c['generated']==c['success']+c['expired']
        assert sum(row['success'] for row in r['rows'])==c['success'] and r['rows'][-1]['active']==0
    for b in range(3):
        for w in protocol['test']:
            rs=[r for r in records if r['block']==b and r['workload']==w]
            assert len(rs)==6 and len({r['fingerprint'] for r in rs})==1
            assert len({r['counts']['generated'] for r in rs})==1
        for mode,summary in analysis['summary'].items():
            rs=[r for r in records if r['block']==b and r['mode']==mode]
            value=sum(r['counts']['success'] for r in rs)/sum(r['counts']['generated'] for r in rs)
            assert abs(value-summary[b]['dsr'])<1e-12
    if torch_checks:
        import torch
        torch.set_num_threads(1)
        for b in range(3):
            zero=torch.load(trial.checkpoint(task,b,0,'expanded'),map_location='cpu',weights_only=False)['model']
            old=torch.load(trial.checkpoint(task,b,0,'old'),map_location='cpu',weights_only=False)['model']
            assert all(torch.equal(v,old[k]) for k,v in zero.items())
            for iteration in (1,2):
                for bank in ('old','expanded'):
                    p=trial.checkpoint(task,b,iteration,bank);saved=torch.load(p,map_location='cpu',weights_only=False)['model']
                    assert all(torch.equal(v,saved[k]) for k,v in zero.items() if not k.startswith('motion.'))
                    assert base.sha(p)==load(out/'fit'/f'{iteration}_{b}_{bank}.json')['checkpoint_sha']
            assert base.sha(trial.parent.checkpoint(task,b,3))==protocol['parent_sha'][str(b)]
    result=dict(status='PASS',states=states,label_chunks=label_count,full_remaining_branches=branches,test_episodes=432,
        torch_parameter_audit=torch_checks,raw_sha={p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file()},
        scope='Reconstructed complete conditional futures, terminal counts, rank and closed-loop arithmetic; Torch parameter/freeze checks on server only. No second physical rollout audit.',independent_review=False)
    p=Path(task)/'output'/base.TASK/'validation'/trial.REV/('server_review.json' if torch_checks else 'local_review.json')
    base.write(p,result);print('PASS',states,'states',label_count,'chunks',branches,'branches',432,'test episodes')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--torch',action='store_true');a=p.parse_args();review(a.task,a.source,a.torch)
