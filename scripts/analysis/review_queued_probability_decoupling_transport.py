"""Pure local verification of transported counts, calibration and ranking."""
import argparse,hashlib,json,math,pickle,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from scripts.analysis import queued_probability_decoupling as trial
from scripts.analysis.analyze_queued_probability_decoupling import bootstrap
from utils.queued_probability_scale import calibrate


def review(task):
    out=trial.out(task);validation=task/'output'/trial.base.TASK/'validation'/trial.REV;load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=load(validation/'transport_manifest.json')
    for name,digest in manifest.items():assert sha(task/name)==digest,name
    server=load(validation/'server_review.json');assert server['status']=='PASS';protocol=load(out/'protocol.json');data=load(out/'analysis.json')
    for name in ('utils/queued_probability_scale.py','marl_models/queued_probability_decoupling.py','scripts/analysis/queued_probability_decoupling.py','scripts/analysis/analyze_queued_probability_decoupling.py','tests/test_queued_probability_decoupling.py'):
        assert sha(task/name)==protocol['source_sha'][name]
    index=load(out/'episode_index.json');assert len(index)==1344;ev=[r for r in index if not r['training']];assert len(ev)==480
    for r in index:
        assert r['sha']==server['files'][r['path']];c=r['counts'];assert c['generated']==c['success']+c['expired']
    for mode,summaries in data['summary'].items():
        for b,s in enumerate(summaries):
            rs=[r for r in ev if r['block']==b and r['mode']==mode];assert len(rs)==32
            counts={k:sum(r['counts'][k] for r in rs) for k in ('generated','success','admitted')};assert all(s[k]==v for k,v in counts.items());assert abs(s['dsr']-counts['success']/counts['generated'])<1e-12
    for b in range(3):
        for w in trial.TEST:
            rs=[r for r in ev if r['block']==b and r['workload']==w];assert len(rs)==5 and len({r['fingerprint'] for r in rs})==len({r['counts']['generated'] for r in rs})==1
        c=load(out/'calibration'/f'{b}.json');again=calibrate(c['rows'],.5);assert len(c['rows'])==320 and abs(again['temperature']-c['temperature'])<1e-9 and c['future_labels_used'] is False
    for ref,rows in data['robustness'].items():
        for b,r in enumerate(rows):
            def ratio(mode,omit=None):
                selected=[x['counts'] for x in ev if x['block']==b and x['mode']==mode and x['workload']!=omit]
                return sum(x['success'] for x in selected)/sum(x['generated'] for x in selected)
            assert abs(min(ratio('decoupled',w)-ratio(ref,w) for w in trial.TEST)-r['min_leave_one_out'])<1e-12
    branches=0;chunks=0
    for state in load(out/'state_index.json'):
        b,w,t,k=(state[x] for x in ('block','workload','time','k'));ues={int(key):tuple(value) for key,value in state['ues'].items()}
        for r in trial.FUTURES:
            future=trial.remaining_future(w,ues,t,r);digest=hashlib.sha256(pickle.dumps(future,5)).hexdigest();total=state['initial_tasks']+sum(len(xs) for at,xs in future.items() if t<at<200);q=[]
            for c in range(math.ceil(k/4)):
                path=out/'labels'/f'{b}_{w}_{t}_{r}_{c}.json';d=load(path);assert sha(path)==server['files'][path.relative_to(out).as_posix()] and d['future_sha']==digest
                assert d['indexes']==list(range(4*c,min(4*c+4,k)));q+=d['values'];chunks+=1
                for value,count in zip(d['values'],d['counts']):assert value==count['success'] and total==count['generated']==count['success']+count['expired']
            branches+=k
            for mode in trial.MODES:
                row=next(x for x in data['ranking_rows'] if x['block']==b and x['workload']==w and x['time']==t and x['mode']==mode);assert row['values'][trial.FUTURES.index(r)]==q and row['probabilities']==state['probabilities'][mode]
        for mode in trial.MODES:
            row=next(x for x in data['ranking_rows'] if x['block']==b and x['workload']==w and x['time']==t and x['mode']==mode)
            p=row['probabilities'];reference=state['probabilities']['calibrated_initial']
            for q,gain in zip(row['values'],row['gains']):assert abs(sum((pp-rr)*(v-sum(q)/len(q)) for pp,rr,v in zip(p,reference,q))-gain)<1e-8
    assert branches==data['full_remaining_branches']==server['full_remaining_branches']
    for mode in trial.MODES:
        for b in range(3):assert bootstrap([r for r in data['ranking_rows'] if r['mode']==mode and r['block']==b],trial.base.seed(trial.REV,'bootstrap',b))==data['ranking'][mode][b]
    by_episode={r['path'].removeprefix('episodes/').removesuffix('.json'):r for r in index}
    for path in (out/'proposal_response').glob('*.json'):
        d=load(path);assert d['all_proposals_rolled_back'] and not d['checkpoint_saved']
        for name,digest in d['input_sha'].items():assert by_episode[Path(name).stem.removesuffix('.pkl')]['experience_sha']==digest
    protection=load(validation/'posthoc_protection.json');assert protection['status']=='PASS' and protection['new_model_files']==0
    result=dict(status='PASS',transport_files=len(manifest),episode_indexes=1344,test_episodes=480,rank_chunks=chunks,full_remaining_branches=branches,calibration_states=960,
        independent_review=False,scope='Metadata/source hashes, raw candidate label counters and reconstructed futures, all episode count indexes, calibration and paired/bootstrap arithmetic. Full MC, teacher/actor parameter and Adam replay on server only; no local Torch or second physical simulation.')
    (validation/'local_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,default=REPO_ROOT);a=p.parse_args();review(a.task)
