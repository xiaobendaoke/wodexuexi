"""Unfreezing budgets, factual MC, prefix logs and matched DSR audit."""
import argparse
import json
import math
from pathlib import Path
import sys


def review(task,source,experience=False,torch_checks=False):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_long_value_unfreeze as trial
    from utils.queued_service_diagnostics import mc_returns
    base=trial.base;out=trial.out(task);load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    analysis=load(out/'analysis.json');protocol=load(out/'protocol.json')
    records=[load(p) for p in (out/'episodes').glob('*.json')]
    train=[r for r in records if r['training']];ev=[r for r in records if not r['training']]
    assert len(train)==1152 and len(ev)==720
    assert not set(protocol['train'])&set(protocol['test'])
    checked=0
    for r in records:
        c=r['counts'];assert c['generated']==c['success']+c['expired']
        assert sum(x['success'] for x in r['rows'])==c['success'] and r['rows'][-1]['active']==0
        if experience and r['training']:
            name=f"{r['arm']}_{r['version']}_{r['block']}_{r['workload']}_{r['mode']}"
            p=out/'experience'/f'{name}.pkl.gz';assert base.sha(p)==r['experience_sha']
            rows=base.read(p);target=mc_returns([x['reward'] for x in rows])
            assert all(abs(row['returns']-v)<1e-9 for row,v in zip(rows,target))
            assert abs(sum(x['reward'] for x in rows)-c['success']/20)<1e-9
            for row in rows:
                for t in row['allocations']:
                    assert t['mask'][t['action']] and math.isfinite(t['prefix_value'])
                    assert abs(sum(t['probs'])-1)<1e-5
            checked+=1
    updates=[load(p) for p in (out/'updates').glob('*.json')];assert len(updates)==30
    budget={}
    for d in updates:
        metrics=d['metrics'];actor_epochs=4 if d['arm']=='alternating' else 2
        count=math.ceil(d['rows']/256);assert metrics['critic_steps']==2*count and metrics['replay_error']<2e-4
        expected=('motion','allocation') if d['active']=='joint' else (d['active'],)
        for name in expected:assert metrics['actor_steps'][name]+metrics['actor_skips'][name]==actor_epochs*count
        for name in ('motion','allocation'):
            if name not in expected:assert metrics['parameter_changes'][name]==0
        key=f"{d['block']}/{d['arm']}";budget.setdefault(key,dict(episodes=0,critic=0,motion=0,allocation=0))
        budget[key]['episodes']+=d['episodes'];budget[key]['critic']+=metrics['critic_steps']
        for name in expected:budget[key][name]+=metrics['actor_steps'][name]+metrics['actor_skips'][name]
    for k,b in budget.items():assert b['episodes']==96
    for b in range(3):
        for w in protocol['test']:
            matched=[r for r in ev if r['block']==b and r['workload']==w]
            assert len(matched)==10 and len({r['fingerprint'] for r in matched})==1
            assert len({r['counts']['generated'] for r in matched})==1
        for mode,summary in analysis['summary'].items():
            rows=[r for r in ev if r['block']==b and r['mode']==mode]
            assert abs(sum(r['counts']['success'] for r in rows)/sum(r['counts']['generated'] for r in rows)-summary[b]['dsr'])<1e-12
    if torch_checks:
        import torch
        torch.set_num_threads(1)
        for b in range(3):
            parent=torch.load(trial.stage1.checkpoint(task,b,2),map_location='cpu',weights_only=False)['model']
            for arm in trial.ARMS:
                initial=torch.load(trial.checkpoint(task,b,arm,0),map_location='cpu',weights_only=False)['model']
                assert all(torch.equal(v,initial[k]) for k,v in parent.items())
                final=torch.load(trial.checkpoint(task,b,arm,2),map_location='cpu',weights_only=False)['model']
                frozen='allocation.' if arm=='motion' else 'motion.' if arm=='allocation' else None
                if frozen:assert all(torch.equal(v,final[k]) for k,v in initial.items() if k.startswith(frozen))
            assert base.sha(trial.stage1.checkpoint(task,b,2))==protocol['parent_sha'][str(b)]
    result=dict(status='PASS',training=1152,evaluation=720,updates=30,experiences_checked=checked,torch_audit=torch_checks,budget=budget,
        files={p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file() and 'experience' not in p.parts},
        independent_review=False,scope='Terminal counts, MC and prefix log structure, actor/critic budgets and matched DSR; parameter freezes on server. Not an independent replication.')
    p=Path(task)/'output'/base.TASK/'validation'/trial.REV/('server_review.json' if torch_checks else 'local_review.json')
    base.write(p,result);print('PASS',1152,'training',720,'evaluation',30,'updates',checked,'experience')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--experience',action='store_true');p.add_argument('--torch',action='store_true');a=p.parse_args();review(a.task,a.source,a.experience,a.torch)
