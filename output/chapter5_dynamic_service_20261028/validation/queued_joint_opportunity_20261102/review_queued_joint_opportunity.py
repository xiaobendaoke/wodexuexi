"""Pure-Python raw-matrix/conditional-future audit, without policy inference."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import sys


def review(task,source):
    sys.path.insert(0,str(source))
    from scripts.analysis import queued_joint_opportunity as trial
    from utils.queued_joint_opportunity import conditional_suffix,crossfit
    from environment.queued_service import safe_path,Engine
    base=trial.base;out=trial.out(task);load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=load(out/'protocol.json');analysis=load(out/'analysis.json');rows=[];branches=0;cpu=set();futures_checked=0
    assert len(list((out/'cases').glob('*.json')))==768
    for b in range(3):
        for w in protocol['workloads']:
            for t in protocol['times']:
                snap=base.read(out/'states'/f'{b}_{w}_{t}.pkl.gz');e=snap['engine'];e.audit()
                assert snap['bank'][:snap['old_count']]==snap['adapter'].plans and len(snap['bank'])<=16
                assert all(safe_path(e.positions,u,e.p) for u in snap['bank'])
                expected={x.uid for x in e.tasks.values() if x.source is not None and x.target is None}
                for plan in snap['plans']+snap['diagonal']:
                    assert set(plan)==expected
                    check=Engine.from_observation(e.observe());check.assign(plan)
                records=[]
                for r in range(8):
                    d=load(out/'cases'/f'{b}_{w}_{t}_{r}.json');records.append(d)
                    assert (d['block'],d['workload'],d['time'],d['replica'])==(b,w,t,r)
                    assert d['count']==len(snap['bank']) and d['old_count']==snap['old_count']
                    assert len(d['values'])==4 and all(len(row)==d['count'] for row in d['values'])
                    assert len(d['diagonal'])==d['count'] and d['all_branch_audits_passed']
                    assert all(isinstance(x,int) and 0<=x<=d['at_risk'] for row in d['values'] for x in row)
                    assert all(isinstance(x,int) and 0<=x<=d['at_risk'] for x in d['diagonal'])
                    future=conditional_suffix(w,e.ues,t,r)
                    assert hashlib.sha256(pickle.dumps(future,5)).hexdigest()==d['future_sha']
                    assert [len(row) for row in future]==d['future_counts']
                    assert d['initial_tasks']==len(e.tasks) and d['at_risk']==len(e.tasks)+sum(d['future_counts'])
                    branches+=d['branch_runs'];cpu.update(d['cpu']);futures_checked+=1
                own=trial.diagnostic(records)
                saved=next(row for row in analysis['states'] if (row['block'],row['workload'],row['time'])==(b,w,t))
                for bank in ('old','expanded'):
                    for name,value in own[bank].items():assert abs(value-saved[bank][name])<1e-12
                assert abs(own['motion_aware_vs_queue']-saved['motion_aware_vs_queue'])<1e-12
                # Independent split selector check of the added-candidate metric.
                diffs=[]
                for train,test in ((records[:4],records[4:]),(records[4:],records[:4])):
                    def select(limit):
                        options=[(a,u) for a in range(4) for u in range(limit)]
                        return sorted(options,key=lambda z:(-sum(d['values'][z[0]][z[1]] for d in train),z))[0]
                    a,u=select(snap['old_count']);aa,uu=select(len(snap['bank']))
                    diffs.append(statistics.mean(d['values'][aa][uu]-d['values'][a][u] for d in test))
                assert abs(statistics.mean(diffs)-own['expanded']['capacity_gain'])<1e-12
                rows.append(saved)
    for key,g in analysis['groups'].items():
        b,phase=key.split('/');subset=[r for r in rows if r['block']==int(b) and (phase=='all' or ('before' if r['time']<50 else 'peak' if r['time']<100 else 'after')==phase)]
        assert len(subset)==g['states']
        for bank in ('old','expanded'):
            for name,v in g[bank].items():assert abs(statistics.mean(r[bank][name] for r in subset)-v)<1e-12
    parents_checked=0
    for b,digest in protocol['parent_sha'].items():
        p=trial.policy.checkpoint(task,int(b),3)
        if p.exists():assert base.sha(p)==digest;parents_checked+=1
    hashes={p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*') if p.is_file()}
    result=dict(status='PASS',states=len(rows),future_samples_reconstructed=futures_checked,unique_branch_rollouts=branches,
                observed_cpu=sorted(cpu),parent_checkpoints_checked=parents_checked,raw_sha=hashes,
                scope='Matrices, independent conditional future regeneration, split selection and group arithmetic; physical branch audits executed by server engine. No second Torch inference or simulation audit.',independent_review=False)
    dest=Path(task)/'output'/base.TASK/'validation'/trial.REV/('server_review.json' if parents_checked==3 else 'local_review.json')
    base.write(dest,result);print('PASS',len(rows),'states',futures_checked,'futures',branches,'branches',len(cpu),'CPUs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();review(a.task,a.source)
