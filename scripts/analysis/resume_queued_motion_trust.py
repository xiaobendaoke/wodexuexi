"""Resume only independent ranking after the documented collector repair."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_motion_trust_experiment as trial


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--original-source',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();task=a.task;out=trial.out(task);base=trial.base
    import torch,fcntl
    torch.set_num_threads(1);lock=open(out/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((out/'protocol.json').read_text());protected={p.relative_to(task).as_posix():base.sha(p) for d in ('models','episodes','updates','experience') for p in (out/d).rglob('*') if p.is_file()}
    records=[json.loads(p.read_text()) for p in (out/'episodes').glob('*.json')]
    assert sum(r['training'] for r in records)==576 and sum(not r['training'] for r in records)==288
    assert len(list((out/'updates').glob('*.json')))==12 and len(list((out/'labels').glob('*.json')))==0
    for name,digest in protocol['source_sha'].items():assert base.sha(a.original_source/name)==digest,name
    old=(a.original_source/'scripts/analysis/queued_motion_trust_experiment.py').read_text();new=(REPO_ROOT/'scripts/analysis/queued_motion_trust_experiment.py').read_text()
    fix='        e.assign(plan);e.advance(move)\n'
    marker='    e.audit();base.write(out(task)/\'collection\''
    assert old.replace(marker,fix+marker,1)==new,'Collector repair must be the only change to original runner'
    if a.smoke:
        trial.collect((str(task),0,trial.RANK[0]))
        for t in trial.TIMES:assert trial.state_at(str(task),0,trial.RANK[0],t)['engine'].t==t
        result=dict(status='PASS',complete_collection_episode=True,snapshots=4,trained_artifacts_unchanged=all(base.sha(task/name)==digest for name,digest in protected.items()))
        base.write(task/'output'/base.TASK/'validation'/trial.REV/'collection_smoke.json',result);assert result['trained_artifacts_unchanged'];print('COLLECTION SMOKE PASS',flush=True);return
    smoke=json.loads((task/'output'/base.TASK/'validation'/trial.REV/'collection_smoke.json').read_text());assert smoke['status']=='PASS'
    files={p.relative_to(REPO_ROOT).as_posix():base.sha(p) for d in ('environment','marl_models','utils','scripts/analysis','tests') for p in (REPO_ROOT/d).glob('*.py')}
    base.write(out/'continuation.json',dict(reason='Collector omitted advancing the simulator; original process stopped before any value labels. Only collector repaired; completed training and DSR evaluation reused unchanged.',
        original_source=a.original_source.name,source_sha=files,protected=protected,original_protocol_sha=base.sha(out/'protocol.json'),training_rerun=False,evaluation_rerun=False))
    cpus=sorted(os.sched_getaffinity(0));original=base.out;base.out=trial.out
    try:
        jobs=[(str(task),b,w) for b in range(3) for w in trial.RANK if not (out/'collection'/f'{b}_{w}.json').exists()]
        base.parallel(task,'rank_collection_resumed',trial.collect,jobs,cpus)
        jobs=[]
        for t in trial.TIMES:
            for b in range(3):
                for w in trial.RANK:
                    k=len(trial.state_at(str(task),b,w,t)['pre']['actions'])
                    jobs.extend((str(task),b,w,t,r,mode,c) for mode in trial.MODES for r in trial.FUTURES for c in range(math.ceil(k/4)))
        base.parallel(task,'independent_futures',trial.label,jobs,cpus)
    finally:base.out=original
    from scripts.analysis.analyze_queued_motion_trust import analyze
    analyze(task)
    assert all(base.sha(task/name)==digest for name,digest in protected.items())
    assert all(base.sha(REPO_ROOT/name)==digest for name,digest in files.items())
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
