"""Read-only terminal-count, future, and parent checks for coupling probe."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys


def review(task,source):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_long_coupling_probe as probe
    load=lambda p:json.loads(p.read_text(encoding='utf-8'));base=probe.base;out=probe.out(task);protocol=load(out/'protocol.json')
    records=list((out/'branches').glob('*.json'));assert len(records)==protocol['jobs'];branches=0
    for b,w,t,k in protocol['states']:
        snap=probe.study.state_at(task,2,b,w,t)
        assert base.sha(probe.study.checkpoint(task,b,1))==protocol['parent_sha'][str(b)]
        for r in protocol['replicas']:
            future=probe.remaining_future(w,snap['engine'].ues,t,r);digest=hashlib.sha256(pickle.dumps(future,5)).hexdigest()
            total=len(snap['engine'].tasks)+sum(len(xs) for at,xs in future.items() if t<at<200)
            for mode in ('original','semantic'):
                selected=[load(p) for p in records if p.stem.startswith(f'{b}_{w}_{t}_{r}_{mode}_')]
                indexes=[]
                for d in selected:
                    assert d['future_sha']==digest;indexes+=d['indexes']
                    for v,c in zip(d['values'],d['counts']):assert c['success']==v and total==c['generated']==c['success']+c['expired']
                assert sorted(indexes)==list(range(k));branches+=k
    expected=load(out/'analysis.json');written={};old=base.write;base.write=lambda p,d:written.update(d)
    try:probe.summarize(task,protocol['states'])
    finally:base.write=old
    assert written==expected
    result=dict(status='PASS',chunks=len(records),full_remaining_branches=branches,independent_review=False,
        source_sha={name:base.sha(source/name) for name in ('utils/queued_sampling_coupling.py','scripts/analysis/queued_long_coupling_probe.py')},
        files={p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*.json')},
        scope='All terminal counters, reconstructed exogenous future fingerprints, split-half arithmetic and unchanged parent checkpoints; no second physical rollout.')
    base.write(Path(task)/'output'/base.TASK/'validation'/probe.REV/'server_review.json',result);print('PASS',len(records),'chunks',branches,'branches')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();review(a.task,a.source)
