"""Saved-count, paired performance and candidate-label calibration review."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT
from utils.queued_taskset_closed_loop import REVISION,ARMS,BASELINES,design,calibration_summary,paired_ratios
from utils.queued_closed_loop import paired_summary

TASK='chapter5_dynamic_service_20261028'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def performance(records):
    spec=design();lookup={(r['arm'],r['run'],r['parent'],r['workload'],r['replica']):r for r in records}
    assert len(lookup)==len(records)==1056
    matrices={};summary={};fingerprints={}
    for arm in ARMS+BASELINES:
        counts={p:{key:np.zeros((3,2,8,2)) for key in ('generated','admitted','success')}
                for p in ('all','pre','peak','post')}
        own=[r for r in records if r['arm']==arm]
        assert len(own)==(96 if arm in ARMS else 48)
        for run in range(6):
            parent=run//2;repeat=run%2;rid=run if arm in ARMS else -1
            for wi,w in enumerate(spec['workloads']):
                for rep in spec['replicas']:
                    r=lookup[arm,rid,parent,w,rep];c=r['counts'];index=(parent,repeat,wi,rep)
                    assert c['generated']==c.get('success',0)+c.get('expired',0) and sum(r['successes'])==c.get('success',0)
                    assert r['audit']['status']=='PASS' and r['public_task_truncation']==0
                    fingerprint=(r['fingerprint'],r['initial_fingerprint'],c['generated'])
                    pair=(w,rep);assert pair not in fingerprints or fingerprints[pair]==fingerprint;fingerprints[pair]=fingerprint
                    for key in counts['all']:counts['all'][key][index]=c.get(key,0)
                    for p in ('pre','peak','post'):
                        for key in counts[p]:counts[p][key][index]=r['cohorts']['arrival'][p][key]
                    for key in counts['all']:
                        assert sum(counts[p][key][index] for p in ('pre','peak','post'))==counts['all'][key][index]
        matrices[arm]=counts;summary[arm]={}
        for p in counts:
            g,a,s=(counts[p][k].sum((2,3)) for k in ('generated','admitted','success'))
            # Baseline matrices repeat the same physical record only to pair
            # it with two residual runs. Report actual request totals here.
            pooled={k:float(sum((r['counts'] if p=='all' else r['cohorts']['arrival'][p]).get(k,0) for r in own))
                    for k in ('generated','admitted','success')}
            summary[arm][p]=dict(mean_dsr=float(np.mean(s/g)),mean_admission=float(np.mean(a/g)),
                mean_conditional=float(np.mean(s/a)),pooled=pooled,
                pooled_dsr=pooled['success']/pooled['generated'],pooled_admission=pooled['admitted']/pooled['generated'],
                pooled_conditional=pooled['success']/pooled['admitted'])
            assert abs(summary[arm][p]['pooled_dsr']-summary[arm][p]['pooled_admission']*summary[arm][p]['pooled_conditional'])<1e-12
        summary[arm]['physical_episodes']=len(own)
        summary[arm]['baseline_repeated_only_for_pairing']=arm in BASELINES
        summary[arm]['decision_mean']=float(np.mean([r['decision_mean'] for r in own]))
        summary[arm]['decision_over_second_fraction']=sum(r['over_second'] for r in own)/sum(r['slots'] for r in own)
        summary[arm]['flight_distance_mean_m']=float(np.mean([r['flight_distance_m'] for r in own]))
        summary[arm]['arrival_quality']={p:dict(
            cpu_wait_per_admitted=float(np.mean([r['cohorts']['arrival'][p]['cpu_wait_per_admitted'] for r in own])),
            completion_latency=float(np.mean([r['cohorts']['arrival'][p]['completion_latency'] for r in own])),
            expiration_stage={k:sum(r['cohorts']['arrival'][p]['expiration_stage'][k] for r in own) for k in ('waiting','upload','transfer','compute')})
            for p in ('pre','peak','post')}
        summary[arm]['admission_quality']={p:dict(admitted=sum(r['cohorts']['admission'][p]['admitted'] for r in own),
            success=sum(r['cohorts']['admission'][p]['success'] for r in own)) for p in ('pre','peak','post')}
    pairs=list(spec['primary_pairs'])+[
        'taskset_mean16_joint--summary_mean16_joint',
        'taskset_mean16_joint--taskset_mean16_decomposed',
        'taskset_single_decomposed--summary_single_decomposed',
        'taskset_mean16_decomposed--summary_mean16_decomposed',
        'taskset_single_joint--original_decomposed',
        'taskset_mean16_joint--original_decomposed',
        'taskset_single_joint--taskset_mean16_joint',
        'taskset_single_joint--initial','taskset_single_joint--plan_short']
    comparisons={}
    for pair in pairs:
        a,b=pair.split('--');comparisons[pair]={}
        for p in ('all','peak'):
            ac,bc=matrices[a][p],matrices[b][p]
            assert np.array_equal(ac['generated'],bc['generated'])
            dsr=paired_summary(ac['success'],bc['success'],ac['generated'],draws=4000,seed=2026110411)
            if pair in spec['primary_pairs']:
                # Same resampling seed/draws; obtain the two-comparison interval
                # directly rather than reuse a four-comparison correction.
                d=ac['success']-bc['success'];g=ac['generated'];rng=np.random.default_rng(2026110411);boot=[]
                for _ in range(4000):
                    ws=rng.integers(0,8,8);rs=rng.integers(0,2,(8,2));values=[]
                    for parent in range(3):
                        ss=rng.integers(0,2,2)
                        values.extend(d[parent][ss][:,ws[:,None],rs].sum((1,2))/g[parent][ss][:,ws[:,None],rs].sum((1,2)))
                    boot.append(float(np.mean(values)))
                dsr['primary_simultaneous97_5']=np.quantile(boot,[.0125,.9875]).tolist()
            # Conditional quality uses policy-dependent admitted denominators;
            # report point effects separately, not the shared-denominator test.
            effect_cond=(ac['success'].sum((2,3))/ac['admitted'].sum((2,3))-
                         bc['success'].sum((2,3))/bc['admitted'].sum((2,3)))
            effect_admit=(ac['admitted'].sum((2,3))/ac['generated'].sum((2,3))-
                          bc['admitted'].sum((2,3))/bc['generated'].sum((2,3)))
            comparisons[pair][p]=dict(dsr=dsr,conditional_difference=float(effect_cond.mean()),
                admission_difference=float(effect_admit.mean()),
                conditional=paired_ratios(ac['success'],ac['admitted'],bc['success'],bc['admitted']),
                admission=paired_ratios(ac['admitted'],ac['generated'],bc['admitted'],bc['generated']))
    return dict(summary=summary,comparisons=comparisons)


def diagnostic_summary(records):
    assert len(records)==792 and all(r['factual_one_step_match'] and r['root_tasks_settled'] and not r['private_future_online_access'] for r in records)
    own={};cross={}
    for arm in ARMS+BASELINES:
        rs=[r for r in records if r['arm']==arm];assert len(rs)==(72 if arm in ARMS else 36)
        own[arm]={p:{reference:calibration_summary([r['own'][reference] for r in rs if p=='all' or (50<=r['time']<100)==(p=='peak')])
                     for reference in ('public','realized','root_admitted')} for p in ('all','peak','nonpeak')}
    learned=[r for r in records if r['run']>=0];assert len(learned)==720
    assert all(set(r['cross'])==set(ARMS+('initial',)) for r in learned)
    for arm in ARMS+('initial',):
        cross[arm]={p:{reference:calibration_summary([r['cross'][arm][reference] for r in learned if p=='all' or (50<=r['time']<100)==(p=='peak')])
                       for reference in ('public','realized','root_admitted')} for p in ('all','peak','nonpeak')}
    contrasts={}
    for pair in design()['primary_pairs']+['taskset_mean16_joint--summary_mean16_joint','taskset_single_joint--original_decomposed']:
        a,b=pair.split('--');contrasts[pair]={}
        for p in ('all','peak','nonpeak'):
            rs=[r for r in learned if p=='all' or (50<=r['time']<100)==(p=='peak')]
            contrasts[pair][p]={reference:dict(
                regret_reduction=float(np.mean([r['cross'][b][reference]['score_regret']-r['cross'][a][reference]['score_regret'] for r in rs])),
                brier_reduction=float(np.mean([r['cross'][b][reference]['brier']-r['cross'][a][reference]['brier'] for r in rs])))
                for reference in ('public','realized','root_admitted')}
    return dict(own_states=own,same_states=cross,paired_same_state_contrasts=contrasts,
        scope='Single realized future plus fixed EDF/hover H=3 continuation, not current-policy Q or a DSR bound. ECE is categorical-label calibration, not request-success calibration. Roots are correlated.')


def review(root):
    root=Path(root);r=root/'results'/TASK/REVISION;v=root/'output'/TASK/'validation'/REVISION
    protocol=load(r/'protocol.json');assert protocol['specification']==design()
    episodes=[load(p) for p in (r/'episodes').glob('*/*.json')]
    diagnostics=[load(p) for p in (r/'diagnostics').glob('*/*.json')]
    result=dict(status='COMPLETE',performance=performance(episodes),diagnostics=diagnostic_summary(diagnostics),
        resources={kind:{k:z for k,z in load(v/(kind+'_complete.json')).items() if k!='records'} for kind in ('episode','diagnostic')},
        primary_family=design()['primary_pairs'],primary_interval='Bonferroni simultaneous 97.5% for two all-episode DSR primary comparisons; peak and mechanisms exploratory.',
        new_optimizer_steps=0,formal_test_access=False,independent_review=False,
        scope='Frozen-model fresh-workload closed-loop verification conditional on three inherited parents; not new end-to-end training repeats.')
    path=r/'analysis.json'
    if path.exists():raise FileExistsError('refuse analysis overwrite')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,default=REPO_ROOT);print(review(p.parse_args().task_root)['status'])
