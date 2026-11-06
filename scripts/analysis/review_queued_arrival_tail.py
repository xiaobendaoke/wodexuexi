"""Read-only factorial coverage diagnostics on the frozen heldout tail branches."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_arrival_adaptation as experiment
from utils.queued_learning_validation import augment
from utils.queued_arrival_adaptation import with_rates
from utils.queued_score_control import row_scores
from utils.queued_recovery_audit import factorial_effects


def paired_summary(records,key,now):
    selected=[r for r in records if r['time']==now]
    cells=np.zeros((6,4))
    for run in range(6):
        rows=sorted((r for r in selected if r['run']==run),key=lambda r:r['workload'])
        assert len(rows)==4
        cells[run]=[r['effects'][key] for r in rows]
    rng=np.random.default_rng(2026110433);samples=[]
    for _ in range(4000):
        values=[]
        for parent in range(3):
            runs=parent*2+rng.integers(0,2,2)
            values.extend(cells[run,rng.integers(0,4,4)].mean() for run in runs)
        samples.append(np.mean(values))
    return dict(mean=float(cells.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
        scope='Exploratory heldout roots. Fixed parents, repeats and four distinct workloads within repeat.')


def review(root):
    root=Path(root);folder=experiment.out(root);records=[]
    for path in sorted((folder/'tail_roots').glob('*.json')):
        descriptor=experiment.load(path)
        if descriptor['split']!='holdout':continue
        state=experiment.saved(root/descriptor['state_file'])[descriptor['index']]
        row=state['row'];run=row['run'];now=row['time']
        base=augment(row,experiment.old.previous.extra(state['obs'],state['bank'],'pool'))
        anchor,_=experiment.old.distribution(root,'hard_joint',run,base,state['obs'],state['bank'])
        scores=row_scores(with_rates(base,state['rates'],True),experiment.actor(str(root),'fast_rates_joint',run),'joint')
        rng=np.random.default_rng(experiment.t.seed(experiment.REVISION,'tail_anchor',run,row['workload'],now))
        first=tuple(map(int,np.unravel_index(rng.choice(anchor.size,p=anchor.ravel()),anchor.shape)))
        second=tuple(map(int,np.unravel_index(np.argmax(np.where(anchor>0,scores,-np.inf)),scores.shape)))
        mapping=dict(anchor=first,allocation=(second[0],first[1]),motion=(first[0],second[1]),full=second)
        assert sorted(set(mapping.values()))==[tuple(p) for p in descriptor['pairs']]
        branches={name:experiment.load(folder/'tail_branches'/path.stem/f'{i}_{j}.json') for name,(i,j) in mapping.items()}
        values={key:{name:branch[key] for name,branch in branches.items()} for key in ('h3','h20','tail20','final_success')}
        for label,stop in [('coverage20',now+20),('coverage_end',200)]:
            values[label]={name:sum(frame['covered_new'] for frame in branch['trace'] if frame['time']<stop) for name,branch in branches.items()}
        effects={}
        for key,value in values.items():
            effects.update({key+'/'+factor:effect for factor,effect in factorial_effects(value).items()})
        records.append(dict(run=run,workload=row['workload'],time=now,mapping=mapping,values=values,effects=effects,
            motion_changed=first[1]!=second[1],allocation_changed=first[0]!=second[0]))
    assert len(records)==96
    summary={str(now):dict(roots=24,motion_changes=sum(r['motion_changed'] for r in records if r['time']==now),
        effects={key:paired_summary(records,key,now) for key in records[0]['effects']}) for now in experiment.TAIL_TIMES}
    result=dict(records=records,summary=summary,
        scope='Frozen fast_rates greedy proposal against sampled hard anchor, then shared hard suffix. Coverage starts at t+3. Descriptive coverage changes do not establish a unique mediator.')
    experiment.write(folder/'tail_interventions.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,required=True)
    review(parser.parse_args().task_root)
