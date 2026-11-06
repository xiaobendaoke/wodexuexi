"""Local transport, independent-future counters and ranking arithmetic audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_motion_trust_experiment as trial
from scripts.analysis.analyze_queued_motion_trust import ranking,correlation


def review(task):
    out=trial.out(task);validation=task/'output'/trial.base.TASK/'validation'/trial.REV
    load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=load(validation/'transport_manifest.json')
    for name,digest in manifest.items():assert sha(task/name)==digest,name
    server=load(validation/'server_review.json');assert server['status']=='PASS'
    protocol=load(out/'protocol.json');analysis=load(out/'analysis.json');index=load(out/'episode_index.json')
    sources=load(out/'continuation.json')['source_sha'] if (out/'continuation.json').exists() else protocol['source_sha']
    for name in ('marl_models/queued_motion_trust.py','scripts/analysis/queued_motion_trust_experiment.py','scripts/analysis/analyze_queued_motion_trust.py','tests/test_queued_motion_trust.py'):
        assert sha(task/name)==sources[name]
    assert len(index)==864;ev=[r for r in index if not r['training']];assert len(ev)==288
    for r in index:
        assert r['sha']==server['files'][r['path']];c=r['counts'];assert c['generated']==c['success']+c['expired']
    for mode,summaries in analysis['summary'].items():
        for b,s in enumerate(summaries):
            rows=[r for r in ev if r['mode']==mode and r['block']==b];assert len(rows)==32
            c={key:sum(r['counts'][key] for r in rows) for key in ('generated','success','admitted')}
            assert all(s[key]==value for key,value in c.items());assert abs(s['dsr']-c['success']/c['generated'])<1e-12
    for b in range(3):
        for w in protocol['test']:
            rows=[r for r in ev if r['block']==b and r['workload']==w];assert len({r['fingerprint'] for r in rows})==len({r['counts']['generated'] for r in rows})==1
    branches=0;chunks=0
    for state in load(out/'state_index.json'):
        b,w,t,k=(state[x] for x in ('block','workload','time','k'));ues={int(key):tuple(value) for key,value in state['ues'].items()}
        for r in protocol['futures']:
            future=trial.remaining_future(w,ues,t,r);digest=hashlib.sha256(pickle.dumps(future,5)).hexdigest();total=state['initial_tasks']+sum(len(xs) for at,xs in future.items() if t<at<200)
            for mode in trial.MODES:
                q=[]
                for chunk in range(math.ceil(k/4)):
                    path=out/'labels'/f'{b}_{w}_{t}_{r}_{mode}_{chunk}.json';d=load(path);assert sha(path)==server['files'][path.relative_to(out).as_posix()];assert d['future_sha']==digest
                    assert d['indexes']==list(range(4*chunk,min(4*chunk+4,k)))
                    for value,c in zip(d['values'],d['counts']):assert value==c['success'] and total==c['generated']==c['success']+c['expired']
                    q+=d['values'];chunks+=1
                row=next(x for x in analysis['ranking_rows'] if x['block']==b and x['workload']==w and x['time']==t and x['mode']==mode)
                assert q==row['values'][protocol['futures'].index(r)] and row['probabilities']==state['probabilities'][mode];branches+=k
                gain=sum((p-1/k)*v for p,v in zip(row['probabilities'],q));assert abs(gain-row['gains'][protocol['futures'].index(r)])<1e-8
    assert branches==analysis['full_remaining_branches']==server['full_remaining_branches']
    for mode in trial.MODES:
        for b in range(3):
            rows=[r for r in analysis['ranking_rows'] if r['mode']==mode and r['block']==b]
            recomputed=ranking(rows,list(trial.RANK),trial.base.seed(trial.REV,'bootstrap',b))
            assert recomputed==analysis['ranking'][mode][b]
    probability_path=out/'probability_diagnosis.json'
    if probability_path.exists():
        pdiag=load(probability_path);assert pdiag['analysis_sha']==sha(out/'analysis.json')
        assert pdiag['diagnostic_source_sha']==sha(task/'scripts/analysis/diagnose_queued_motion_trust.py')
        for mode,groups in pdiag['groups'].items():
            for t,g in groups.items():
                rs=[r for r in analysis['ranking_rows'] if r['mode']==mode and r['time']==int(t)];valid=[r for r in rs if len(r['probabilities'])>1]
                assert g['states']==len(rs) and g['multiple_candidates']==len(valid)
                assert g['pmax_ge_0999']==sum(max(r['probabilities'])>=.999 for r in valid)
        for t,g in pdiag['trust_minus_initial'].items():
            pairs=[]
            for r in analysis['ranking_rows']:
                if r['mode']!='trust' or r['time']!=int(t):continue
                p=r['probabilities'];q=next(x['probabilities'] for x in analysis['ranking_rows'] if x['mode']=='initial' and x['block']==r['block'] and x['workload']==r['workload'] and x['time']==r['time'])
                pairs.append((sum(abs(a-b) for a,b in zip(p,q))/2,max(range(len(p)),key=p.__getitem__)!=max(range(len(q)),key=q.__getitem__)))
            assert abs(g['mean_tv']-statistics.mean(x[0] for x in pairs))<1e-12 and g['changed_argmax']==sum(x[1] for x in pairs)
    result=dict(status='PASS',transport_files=len(manifest),episode_indexes=864,test_episodes=288,rank_chunks=chunks,full_remaining_branches=branches,
        independent_review=False,scope='Local raw label counts and independent future reconstruction, frozen prediction indexes, all episode count indexes and ranking bootstrap. Full states, model replay and MC experiences checked on server only.')
    (validation/'local_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,default=REPO_ROOT);a=p.parse_args();review(a.task)
