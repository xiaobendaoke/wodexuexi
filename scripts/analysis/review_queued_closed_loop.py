"""Saved-count, public-score, final-fit and provenance self-review only.

No new scenarios, physical rollout, Torch inference, coefficient selection or
policy training. Large immutable fit shards need not be transported.
"""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_closed_loop_experiment as e
from scripts.analysis.review_queued_candidate_reliability import assert_nested_close
from utils.queued_closed_loop import design,ARMS,BASELINES,relative_scores,pack_rows,batch_improve,packed_statistics,select_method

def review(root,phase='pilot'):
    root=Path(root).resolve();r=e.out(root,phase);v=e.val(root,phase);spec=design(phase)
    transport=v/'transport_manifest.json'
    if transport.exists():
        for name,digest in e.load(transport).items():assert e.t.sha(root/name)==digest,name
    protocol=e.load(r/'protocol.json');assert protocol['specification']==spec
    for name in ['utils/queued_closed_loop.py','scripts/analysis/queued_closed_loop_experiment.py','tests/test_queued_closed_loop.py']:
        assert e.t.sha(root/name)==protocol['source'][name],name
    before=e.load(v/'protection_before.json');after=e.load(v/'protection_after.json')
    assert before==after['hashes'] and after['status']=='PASS'
    model=e.scorer(str(root));seal=e.load(v/'final_models_sealed.json')
    for name,digest in seal['models'].items():assert e.t.sha(root/name)==digest,name
    train=[e.load(p) for p in (r/'episodes').glob('training_*/*.json')];ev=[e.load(p) for p in (r/'episodes').glob('evaluation_*/*.json')]
    assert len(train)==spec['maximum_training_episodes'] and len(ev)==spec['maximum_evaluation_episodes']
    for record in train+ev:
        c=record['counts'];assert record['audit']['status']=='PASS'
        assert c['generated']==c.get('success',0)+c.get('expired',0) and sum(record['successes'])==c.get('success',0)
        assert len(record['successes'])==record['slots'] and record['joule_energy'] is None
    checked_rows=0;updates=[]
    for arm in ARMS:
        for run in spec['runs']:
            for version in range(1,spec['rounds']+1):
                rows=[]
                for w in run['training'][version-1]:rows+=e.cached(str(r/'experience'/arm/f'{run["id"]}_{version-1}_{w}.pkl.gz'))
                assert len(rows)==20*spec['training_episodes_per_round']
                for row in rows:
                    p=np.asarray(row['short']);which=e.teacher_for(arm)
                    q=p-p.mean() if which=='short' else relative_scores(model,row['terminal_features'],p,row['terminal_empty'],'root_centered' if which=='relative' else 'root_shuffled')
                    np.testing.assert_allclose(q,row['q'],atol=1e-11);checked_rows+=1
                old=e.cached(str(e.theta_path(root,phase,arm,run['id'],version-1)));new=e.cached(str(e.theta_path(root,phase,arm,run['id'],version)))
                pack=pack_rows(rows,old['theta'],e.mode(arm));targets,estep=batch_improve(pack['old'],pack['q'],.1);pack['target']=targets
                saved=e.load(r/'updates'/f'{version}_{arm}_{run["id"]}.json');assert_nested_close(estep,saved['estep'],'estep')
                final=packed_statistics(pack,new['theta'],new['theta']);n=final.pop('n');final.pop('gradient');final={k:v/n for k,v in final.items()}
                assert_nested_close(final,{k:z for k,z in saved['final'].items() if k!='n'},'final_fit')
                assert saved['final']['n']==n
                assert new['optimizer']['step']==old['optimizer']['step']+len(saved['accepted'])
                assert len(saved['accepted'])+saved['rejected']==spec['fit_steps']
                for z in saved['accepted']:assert z['kl']<=.02000001 and z['delta']<=.00200001
                assert np.isfinite(new['theta']).all();updates.append(saved)
    original=e.load(r/'analysis.json');computed=e.analyze(root,phase,write_outputs=False)
    assert_nested_close(computed,original,'analysis_recomputed')
    resources=e.load(v/'resources.json');records=resources['records']
    assert len(records)==resources['tasks'] and all(z['ok'] for z in records)
    assert len([z for z in records if z['kind']=='episode'])==len(train)+len(ev)
    assert len({z['key'] for z in records})==len(records)
    assert all(len(z['affinity'])==1 and z['threads']==1 for z in records)
    cpu=sum(z['cpu_seconds'] for z in records);assert abs(cpu-resources['worker_cpu_seconds'])<1e-7
    equivalent=(cpu+resources['coordinator_cpu_seconds'])/resources['wall_seconds'];assert abs(equivalent-resources['equivalent_cpus'])<1e-8
    assert abs(equivalent/len(resources['allowed_cpus'])-resources['utilization'])<1e-10
    active=peak=0
    for _,change in sorted([(x['start'],1) for x in records]+[(x['end'],-1) for x in records]):active+=change;peak=max(peak,active)
    assert active==0 and peak==resources['peak_active']
    frozen=e.load(e.out(root,'formal')/'frozen_config.json')
    assert frozen['pilot_analysis_sha256']==e.t.sha(e.out(root,'pilot')/'analysis.json')
    assert frozen['pilot_selection']==original['selection']
    if phase=='pilot':
        assert not (e.out(root,'formal')/'episodes').exists() and not (e.val(root,'formal')/'execution_authorized.json').exists()
        assert frozen['formal_outcomes_generated'] is frozen['formal_outcomes_read'] is False
        assert e.load(v/'formal_freeze.json')['config_sha256']==e.t.sha(e.out(root,'formal')/'frozen_config.json')
    csv_path=root/'output'/e.TASK/'deliverables'/f'queued_closed_loop_{phase}_episodes.csv'
    cols=['kind','arm','run','parent','workload','replica','version','generated','success','expired','decision_mean','flight_distance_m']
    with csv_path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=cols);writer.writeheader()
        for z in sorted(train+ev,key=lambda z:(z['kind'],z['arm'],z['run'],z['version'],z['workload'],z['replica'])):
            writer.writerow({**{k:z[k] for k in cols if k not in ('generated','success','expired')},**{k:z['counts'].get(k,0) for k in ('generated','success','expired')}})
    result=dict(status='PASS',phase=phase,training_episodes=len(train),evaluation_episodes=len(ev),teacher_rows_checked=checked_rows,
        updates_checked=len(updates),fresh_residual_training_runs=len(spec['runs']),fixed_parent_checkpoints=3,
        formal_test_outcomes_read=phase=='formal',frozen_status=frozen['status'],selection=original['selection'],
        final_actor_statistics_recomputed=True,optimizer_step_accounting=True,public_scores_reconstructed=True,
        full_optimizer_trajectory_replayed=False,physical_replay=False,independent_review=False,
        equivalent_cpus=equivalent,utilization=resources['utilization'],csv_rows=len(train)+len(ev),
        scope='Arithmetic/provenance self-review. No new Torch inference, scenarios, physical rollout, trained-model selection or optimizer training.')
    e.write(v/'local_review.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--phase',choices=('pilot','formal'),default='pilot');a=p.parse_args();review(a.task_root,a.phase)

if __name__=='__main__':main()
