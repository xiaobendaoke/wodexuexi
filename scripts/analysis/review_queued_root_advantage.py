"""Arithmetic/provenance self-review; no local Torch or physical simulation.

Reconstructs the same three small analytic ridge solutions without saving or
selecting a new model. Evaluation uses only the sealed transported coefficients.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_root_advantage_experiment as e
from scripts.analysis.review_queued_candidate_reliability import assert_nested_close
from utils.queued_root_advantage import fit_models,predict,tie_weights


def supervision_controls(root,model,analyses):
    """Post hoc control interpretation; never changes the frozen primary gate."""
    output=dict(status='COMPLETE',post_hoc=True,primary_gate_changed=False,
        scope='Descriptive paired comparisons using sealed scores and existing labels only; no new fit, model selection or physical rollout.',
        model_sha256=e.sha(e.result_dir(root)/'models.json'),splits={})
    for split in ('dev','test'):
        wids=e.WORKLOADS[split];replicas=len(e.REPLICAS[split])
        fs=[f for f in e.load_features(root) if f['split']==split]
        deltas={m:np.zeros((len(wids),3,2,replicas)) for m in ('short_window','root_shuffled')}
        agreements=Counter()
        for f in fs:
            y=e.outcomes(root,f);scores=e.method_scores(model,f)
            selected={m:tie_weights(scores[m]) for m in ('root_centered','short_window','root_shuffled')}
            for m in deltas:
                deltas[m][wids.index(f['w']),f['b'],(75,175).index(f['time'])]=(selected['root_centered']-selected[m])@y
                agreements[m]+=int(np.allclose(selected['root_centered'],selected[m],atol=1e-12,rtol=0))
        comparisons={}
        for m,d in deltas.items():
            stats=e.bootstrap_blocks(d.mean(axis=(1,2)))
            stats.pop('simultaneous98_333') # Outside the three predeclared primary contrasts.
            stats.update(by_lineage=d.mean(axis=(0,2,3)).tolist(),
                by_time={str(t):float(d[:,:,i].mean()) for i,t in enumerate((75,175))},
                equal_selection_weight_roots=agreements[m],roots=len(fs))
            expected=analyses[split]['methods']['root_centered']['mean']-analyses[split]['methods'][m]['mean']
            assert abs(stats['mean']-expected)<1e-10
            if m=='root_shuffled':
                np.testing.assert_allclose(stats['ci95'],analyses[split]['comparisons'][m]['ci95'],atol=1e-10)
            comparisons[m]=stats
        output['splits'][split]=dict(comparisons=comparisons,
            analysis_sha256=e.sha(e.result_dir(root)/f'analysis_{split}.json'))
    e.write(e.result_dir(root)/'supervision_controls.json',output)
    return output


def review(root):
    root=Path(root).resolve();r=e.result_dir(root);v=e.validation_dir(root)
    manifest=e.readj(v/'transport_manifest.json')
    for name,sha in manifest.items():assert e.sha(root/name)==sha,name
    protocol=e.readj(r/'protocol.json')
    for name,sha in protocol['source'].items():assert e.sha(root/name)==sha,name
    fs=e.load_features(root);assert len(fs)==168
    splits=Counter(x['split'] for x in fs);assert splits==dict(old_train=24,train=72,dev=24,test=48),splits
    for f in fs:
        assert len(f['choices']) in (1,2,4) and np.array(f['x']).shape==(len(f['choices']),99)
        assert np.isfinite(np.array(f['x'])).all()
        if f['split']!='old_train':assert f['w'] in e.WORKLOADS[f['split']]
    assert len({(x['b'],x['w'],x['time']) for x in fs})==168
    assert set(e.WORKLOADS['train']).isdisjoint(e.WORKLOADS['dev']) and set(e.WORKLOADS['train']).isdisjoint(e.WORKLOADS['test']) and set(e.WORKLOADS['dev']).isdisjoint(e.WORKLOADS['test'])
    snapshots=e.readj(r/'snapshot_index.json')
    assert snapshots['status']=='PASS' and snapshots['feature_reconstruction'] and snapshots['unique_geometries']==28
    labels=[e.readj(p) for p in (r/'labels').glob('*.json')]
    expected=sum(len(f['choices'])*len(e.REPLICAS[f['split']]) for f in fs if f['split']!='old_train')
    assert len(labels)==expected<=9984
    seen=set();generated=defaultdict(set)
    for x in labels:
        key=tuple(x[k] for k in ('b','w','time','i','j','replica'));assert key not in seen;seen.add(key)
        assert x['w'] in e.WORKLOADS[x['split']] and x['replica'] in e.REPLICAS[x['split']]
        assert x['generated']==x['success']+x['expired'] and x['success']==x['actual_prefix']+x['actual_tail'] and x['audit']['status']=='PASS'
        generated[x['b'],x['w'],x['time'],x['replica']].add(x['generated'])
    assert all(len(x)==1 for x in generated.values())
    sealed=e.readj(v/'fit_frozen.json');read_started=e.readj(v/'evaluation_read_started.json')
    model=e.readj(r/'models.json')
    assert sealed['dev_or_test_labels_read'] is False and model['train_roots']==96 and model['feature_dim']==99 and model['alpha']==1.
    design=e.readj(r/'feature_design.json');assert design['train_roots']==96
    np.testing.assert_allclose(model['mean'],design['mean'],atol=1e-12)
    np.testing.assert_allclose(model['scale'],design['scale'],atol=1e-12)
    assert sealed['epoch']<=read_started['epoch'] and sealed['model_sha256']==read_started['frozen_model_sha256']==e.sha(r/'models.json')
    rows=[];hashes={};byroot={}
    for f in fs:
        if f['split'] not in ('old_train','train'):continue
        y=e.outcomes(root,f).mean(1);key=f"{f['b']}_{f['w']}_{f['time']}"
        rows.append(dict(x=f['x'],p=f['p'],y=y,time=f['time'],key=key));byroot[key]=(f,y)
        hashes[key]=hashlib.sha256(y.tobytes()).hexdigest()
    assert hashes==sealed['training_target_hashes']
    other_keys={str(f['b'])+'_'+str(f['w'])+'_'+str(f['time']) for f in fs if f['split'] in ('dev','test')}
    assert set(hashes).isdisjoint(other_keys)
    reconstructed=fit_models(rows,alpha=1.)
    assert_nested_close(reconstructed,model,'analytic_ridge_reconstruction')
    analyses={}
    for split in ('dev','test'):
        saved=e.readj(r/f'analysis_{split}.json');computed=e.evaluate(root,model,split,write_outputs=False)
        assert_nested_close(json.loads(json.dumps(computed)),saved,'analysis_'+split)
        analyses[split]=saved
    assert e.readj(v/'protection_before.json')==e.readj(v/'protection_after.json')
    resources=e.readj(v/'resources.json');records=resources['records']
    assert len(records)==resources['tasks']==expected+96
    assert Counter(x['kind'] for x in records)==dict(label=expected,collect=72,export_old=24)
    assert all(x['ok'] and len(x['affinity'])==1 and x['threads']==1 for x in records)
    cpu=sum(x['cpu_seconds'] for x in records)
    assert abs(cpu-resources['worker_cpu_seconds'])<1e-6
    equivalent=(cpu+resources['coordinator_cpu_seconds'])/resources['wall_seconds']
    assert abs(equivalent-resources['equivalent_cpus'])<1e-8
    assert abs(equivalent/len(resources['allowed_cpus'])-resources['utilization'])<1e-10
    active=peak=0
    for _,change in sorted([(x['start'],1) for x in records]+[(x['end'],-1) for x in records]):active+=change;peak=max(peak,active)
    assert active==0 and peak==resources['peak_active']
    train_diagnostics={}
    for name in ('root_centered','time_centered_absolute','root_shuffled','original_q','short_window'):
        moments=[];gain=[]
        for key,(f,y) in byroot.items():
            s=e.method_scores(model,f)[name];truth=y-y.mean();pc=s-s.mean()
            moments.append([np.mean((pc-truth)**2),np.mean(truth**2),np.mean(pc**2)])
            gain.append(float(tie_weights(s)@y-y.mean()))
        mse,t2,p2=np.mean(moments,0)
        train_diagnostics[name]=dict(within_root_mse=float(mse),within_root_ev=float(1-mse/max(t2,1e-12)),mean_score_spread=float(np.sqrt(p2)),
            mean_label_spread=float(np.sqrt(t2)),training_selection_gain=float(np.mean(gain)),
            coefficient_norm=float(np.linalg.norm(model['models'][name]['beta'])) if name in model['models'] else None)
    e.write(r/'training_diagnostics.json',dict(status='COMPLETE',models=train_diagnostics,
        scope='Descriptive fit/calibration on already used TRAIN roots; not additional model fitting or a model-selection step.'))
    controls=supervision_controls(root,model,analyses)
    csv_path=root/'output'/e.TASK/'deliverables'/'queued_root_advantage_states.csv'
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    columns=['split','b','workload','time','candidate_count','mean_pair_se']+list(e.METHODS)
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for split in ('dev','test'):
            for row in analyses[split]['state_rows']:
                writer.writerow(dict(split=split,b=row['b'],workload=row['w'],time=row['time'],candidate_count=row['candidate_count'],mean_pair_se=row['mean_pair_se'],**row['gains']))
    result=dict(status='PASS',new_physical_branches=expected,feature_roots=168,train_roots=96,dev_roots=24,test_roots=48,
        common_generated_inputs=True,sealed_before_evaluation=True,train_only_normalization_and_fit=True,
        analytic_coefficients_reconstructed=True,saved_evaluations_recomputed=True,protection_manifests_equal=True,resources_recomputed=True,post_hoc_controls_recomputed=controls['status']=='COMPLETE',
        candidate_counts=dict(Counter(len(x['choices']) for x in fs)),test_gate=analyses['test']['centered_supervision_gate'],
        equivalent_cpus=resources['equivalent_cpus'],utilization=resources['utilization'],
        scope='Explicit arithmetic/provenance self-review; no new trained-model selection, Torch inference or local physical rollout.',independent_review=False)
    e.write(v/'local_review.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);args=p.parse_args();review(args.task_root)

if __name__=='__main__':main()
