"""No-Torch recomputation of motion-PPO episodes, MC and matched test metrics."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_service_motion_ppo as trial
from utils.queued_service_diagnostics import mc_returns


def review(task,experience=False):
    out=trial.out(task);load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=load(out/'protocol.json');analysis=load(out/'analysis.json');assert not set(protocol['train'])&set(protocol['test'])
    records=[load(p) for p in (out/'episodes').glob('*.json')];training=[r for r in records if r['training'] and r['mode']=='ppo']
    testing=[r for r in records if not r['training']]
    assert len(training)==864 and len(testing)==360
    experience_checked=0
    for r in records:
        c=r['counts'];assert c['generated']==c['success']+c['expired']
        assert sum(x['success'] for x in r['rows'])==c['success'] and r['rows'][-1]['active']==0
        if experience and r['training']:
            name=f"{r['block']}_{r['round']}_{r['workload']}_{r['mode']}"
            p=out/'experience'/f'{name}.pkl.gz';assert trial.base.sha(p)==r['experience_sha']
            rows=trial.base.read(p);target=mc_returns([x['reward'] for x in rows])
            for row,value in zip(rows,target):
                assert abs(row['returns']-value)<1e-9
                assert abs(row['motion_advantage']-(value-row['motion_value']))<1e-9
                assert abs(math.log(row['motion_probs'][row['motion_action']])-row['motion_log'])<2e-5
            assert abs(sum(x['reward'] for x in rows)-c['success']/20)<1e-9
            experience_checked+=1
    updates=[]
    for b in range(3):
        for r in range(1,4):
            d=load(out/'updates'/f'{b}_{r}.json');assert d['episodes']==96 and d['metrics']['allocation_change']==0
            assert d['metrics']['optimizer_steps']==2*math.ceil(d['rows']/256)
            assert d['metrics']['actor_steps']+d['metrics']['actor_skips']==d['metrics']['optimizer_steps']
            assert d['metrics']['replay_error']<2e-5
            updates.append(d)
        for workload in protocol['test']:
            paired=[r for r in testing if r['block']==b and r['workload']==workload]
            assert len(paired)==5 and len({r['fingerprint'] for r in paired})==1 and len({r['counts']['generated'] for r in paired})==1
        for mode,summary in analysis['summary'].items():
            rows=[r for r in testing if r['block']==b and r['mode']==mode]
            g=sum(x['counts']['generated'] for x in rows);s=sum(x['counts']['success'] for x in rows)
            assert abs(s/g-summary[b]['dsr'])<1e-12
    interpret=dict(mean_dsr={m:statistics.mean(x['dsr'] for x in rows) for m,rows in analysis['summary'].items()},
                   mean_gain_percentage_points={m:100*statistics.mean(v) for m,v in analysis['paired_gains'].items()},
                   per_workload={},cpu={},optimizer_steps=sum(x['metrics']['optimizer_steps'] for x in updates),
                   actor_steps=sum(x['metrics']['actor_steps'] for x in updates),actor_skips=sum(x['metrics']['actor_skips'] for x in updates))
    for mode,reference in (('ppo','frozen'),('ppo_argmax','frozen_argmax')):
        interpret['per_workload'][mode]=[]
        for b in range(3):
            delta=[]
            for w in protocol['test']:
                a=next(r for r in testing if r['block']==b and r['workload']==w and r['mode']==mode)
                z=next(r for r in testing if r['block']==b and r['workload']==w and r['mode']==reference)
                delta.append((a['counts']['success']-z['counts']['success'])/a['counts']['generated'])
            interpret['per_workload'][mode].append(dict(block=b,positive=sum(x>0 for x in delta),negative=sum(x<0 for x in delta),differences=delta))
    for p in (out/'execution').glob('*.json'):
        d=load(p);assert d['jobs']==d['completed']
        interpret['cpu'][d['stage']]=dict(jobs=d['jobs'],cpu_count=len(d['observed_cpus']),seconds=d['seconds'])
    trial.base.write(out/('interpretation.json' if experience else 'local_interpretation.json'),interpret)
    hashes={p.relative_to(out).as_posix():trial.base.sha(p) for p in out.rglob('*') if p.is_file() and 'experience' not in p.parts}
    result=dict(status='PASS',training_episodes=864,test_episodes=360,experience_checked=experience_checked,
                interpretation=interpret,files=hashes,independent_review=False)
    path=Path(task)/'output'/trial.base.TASK/'validation'/trial.REV/('server_review.json' if experience else 'local_review.json')
    trial.base.write(path,result);print('PASS',864,360,'experience',experience_checked)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,default=REPO_ROOT);p.add_argument('--experience',action='store_true');a=p.parse_args();review(a.task,a.experience)
