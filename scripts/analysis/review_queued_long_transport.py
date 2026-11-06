"""Local arithmetic and transport audit; heavy raw/parameter checks stay on server."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT


def review(task):
    topic='chapter5_dynamic_service_20261028';validation=task/'output'/topic/'validation'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=load(validation/'queued_long_value_20261102'/'transport_manifest.json')
    for name,digest in manifest.items():assert sha(task/name)==digest,name
    audited={}
    for rev,n in (('queued_long_value_20261102',432),('queued_long_unfreeze_20261102',720)):
        out=task/'results'/topic/rev;analysis=load(out/'analysis.json');protocol=load(out/'protocol.json');server=load(validation/rev/'server_review.json')
        assert server['status']=='PASS';rows=load(out/'episode_index.json');ev=[r for r in rows if not r.get('training',False)]
        assert len(ev)==n;raw_hash=server.get('raw_sha',server.get('files'))
        for r in rows:
            assert r['sha']==raw_hash[r['path']]
            c=r['counts'];assert c['generated']==c['success']+c['expired']
        for mode,summaries in analysis['summary'].items():
            for b,s in enumerate(summaries):
                selected=[r for r in ev if r['mode']==mode and r['block']==b];assert len(selected)==24
                c={k:sum(r['counts'][k] for r in selected) for k in ('success','generated','admitted')}
                assert all(c[k]==s[k] for k in c)
                assert abs(c['success']/c['generated']-s['dsr'])<1e-12
                assert abs(s['admission']*s['conditional']-s['dsr'])<1e-12
        for b in range(3):
            for w in protocol['test']:
                selected=[r for r in ev if r['block']==b and r['workload']==w]
                assert len({r['fingerprint'] for r in selected})==1
                assert len({r['counts']['generated'] for r in selected})==1
        files=['utils/queued_long_value.py','marl_models/queued_long_value.py']
        files+=['scripts/analysis/queued_long_value_unfreeze.py'] if 'unfreeze' in rev else ['scripts/analysis/queued_long_value_experiment.py']
        for name in files:assert sha(task/name)==protocol['source_sha'][name],name
        def aggregate(mode,b,omit=None):
            ds=[r['counts'] for r in ev if r['mode']==mode and r['block']==b and r['workload']!=omit]
            return sum(x['success'] for x in ds)/sum(x['generated'] for x in ds)
        if 'unfreeze' in rev:
            for ref,deltas in analysis['joint_contrasts'].items():
                for b,d in enumerate(deltas):assert abs(aggregate('joint',b)-aggregate(ref,b)-d)<1e-12
            for ref,rs in analysis['robustness'].items():
                for b,r in enumerate(rs):
                    ds=[aggregate('joint',b,w)-aggregate(ref,b,w) for w in protocol['test']]
                    assert abs(min(ds)-r['min_leave_one_out'])<1e-12
            executed={}
            for path in (out/'updates').glob('*.json'):
                d=load(path)
                for name,steps in d['metrics']['actor_steps'].items():
                    key=d['arm']+'/'+name;z=executed.setdefault(key,dict(executed=0,skipped=0))
                    z['executed']+=steps;z['skipped']+=d['metrics']['actor_skips'][name]
            assert all(z['executed']+z['skipped']==456 for z in executed.values())
        else:
            for b,r in enumerate(analysis['robustness']):
                ds=[aggregate('expanded_learned',b,w)-aggregate('initial',b,w) for w in protocol['test']]
                assert abs(min(ds)-r['min_leave_one_out'])<1e-12
        audited[rev]=dict(episodes_indexed=len(rows),test_episodes=n,server_audit=server['status'],source_modules_checked=files)
        if 'unfreeze' in rev:audited[rev]['actual_actor_steps']=executed
    probe=task/'results'/topic/'queued_long_coupling_20261102'
    if (probe/'analysis.json').exists():
        from scripts.analysis.queued_long_coupling_probe import summarize
        expected=load(probe/'analysis.json');states=load(probe/'protocol.json')['states']
        # Same arithmetic, new local file location avoided by a temporary override.
        from scripts.analysis import queued_long_coupling_probe as module
        records=[load(p) for p in (probe/'branches').glob('*.json')]
        for r in records:
            for c in r['counts']:assert c['generated']==c['success']+c['expired']
        written={};old=module.base.write;module.base.write=lambda p,d:written.update(d)
        try:summarize(task,states)
        finally:module.base.write=old
        assert written==expected
        audited['coupling']=dict(chunks=len(records),full_remaining_branches=sum(len(r['indexes']) for r in records))
    result=dict(status='PASS',transport_files=len(manifest),audited=audited,independent_review=False,
        scope='Metadata hashes, all evaluation count indexes, paired DSR and leave-one-out arithmetic. No local Torch, raw experience, full simulator-state or model audit; those are server checks.')
    dest=validation/'queued_long_value_20261102'/'local_review.json';dest.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,default=REPO_ROOT);a=p.parse_args();review(a.task)
