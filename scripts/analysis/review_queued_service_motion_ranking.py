"""Recompute saved ranking/closed-loop metrics without Torch or training."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT


def review(root):
    task='chapter5_dynamic_service_20261028';rev='queued_motion_ranking_20261101'
    out=root/'results'/task/rev
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=load(out/'protocol.json');analysis=load(out/'analysis.json')
    files={};n=0;fits=[]
    for split in ('train','validation','test'):
        for other in ('train','validation','test'):
            if split!=other:assert not set(protocol[split])&set(protocol[other])
    for block in range(3):
        fit=load(out/'fit'/f'{block}.json');assert fit['allocation_unchanged'] and fit['epochs']==40 and fit['optimizer_steps']==120
        fits.append(fit)
        for stage in ('initial','final'):
            for split in ('train','validation'):
                rows=fit[stage][split]['rows'];assert len(rows)==96
                for row in rows:
                    p,q=row['probabilities'],row['values'];assert abs(sum(p)-1)<1e-5 and len(p)==len(q)
                    index=protocol['times'].index(row['time'])
                    labels=[load(out/'labels'/f"{block}_{row['workload']}_{index}_{r}.json")['values'] for r in range(3)]
                    assert all(abs(a-statistics.mean(xs))<1e-12 for a,xs in zip(q,zip(*labels)))
                    gain=sum(a*b for a,b in zip(p,q))-statistics.mean(q)
                    assert abs(gain-row['gain'])<1e-8
                    assert abs(max(q)-sum(a*b for a,b in zip(p,q))-row['regret'])<1e-8
                assert abs(statistics.mean(r['gain'] for r in rows)-fit[stage][split]['gain'])<1e-8
        for mode,summary in analysis['summary'].items():
            rows=[load(out/'episodes'/f'{block}_{w}_{mode}.json') for w in protocol['test']]
            generated=sum(r['counts']['generated'] for r in rows);success=sum(r['counts']['success'] for r in rows)
            for row in rows:
                c=row['counts'];assert c['generated']==c['success']+c['expired']
                assert sum(x['success'] for x in row['rows'])==c['success']
                assert row['rows'][-1]['active']==0
                assert row['allocation_sha']==protocol['parent_checkpoints'][str(block)]
                n+=1
            assert abs(success/generated-summary[block]['dsr'])<1e-12
    for block in range(3):
        for workload in protocol['test']:
            rows=[load(out/'episodes'/f'{block}_{workload}_{mode}.json') for mode in analysis['summary']]
            assert len({r['fingerprint'] for r in rows})==1
            assert len({r['counts']['generated'] for r in rows})==1
    interpretation=dict(mean_dsr={m:statistics.mean(r['dsr'] for r in rows) for m,rows in analysis['summary'].items()},
                        mean_gain_percentage_points={m:100*statistics.mean(values) for m,values in analysis['paired_dsr_gain'].items()},
                        ranking_validation=[dict(block=b,initial=fits[b]['initial']['validation']['gain'],final=fits[b]['final']['validation']['gain']) for b in range(3)],
                        per_workload={},cpu={},state_snapshots_present=(out/'states').exists())
    for mode in ('learned','learned_argmax','model'):
        interpretation['per_workload'][mode]=[]
        for b in range(3):
            differences=[]
            for w in protocol['test']:
                a=load(out/'episodes'/f'{b}_{w}_{mode}.json')['counts'];z=load(out/'episodes'/f'{b}_{w}_native.json')['counts']
                differences.append((a['success']-z['success'])/a['generated'])
            interpretation['per_workload'][mode].append(dict(block=b,positive=sum(x>0 for x in differences),negative=sum(x<0 for x in differences),zero=sum(x==0 for x in differences),differences=differences))
    for stage in ('collect','labels','fit','evaluation'):
        e=load(out/'execution'/f'{stage}.json')
        interpretation['cpu'][stage]=dict(jobs=e['jobs'],seconds=e['seconds'],observed_cpu_count=len(e['observed_cpus']),allowed_cpu_count=len(e['allowed_cpus']))
    (out/'interpretation.json').write_text(json.dumps(interpretation,indent=2),encoding='utf-8')
    for path in out.rglob('*'):
        if path.is_file():files[path.relative_to(out).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(status='PASS',closed_loop_records=n,files=files,interpretation=interpretation,review='explicit self-review; no independent training confirmation; frozen simulator snapshots remain on server if absent locally')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=Path,default=REPO_ROOT);args=parser.parse_args()
    result=review(args.task)
    path=args.task/'output/chapter5_dynamic_service_20261028/validation/queued_motion_ranking_20261101/local_review.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(result['status'],result['closed_loop_records'])
