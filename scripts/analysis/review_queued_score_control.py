"""Counts, paired uncertainty, gate behavior and score/control diagnostics."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis import queued_score_control as experiment
from utils.queued_taskset_closed_loop import calibration_summary


def paired(a, da, b, db, draws=4000):
    a,da,b,db = (np.asarray(x,float) for x in (a,da,b,db))
    assert a.shape == da.shape == b.shape == db.shape == (3,2,8,2)
    assert all(np.isfinite(x).all() for x in (a,da,b,db)) and np.all(da > 0) and np.all(db > 0)
    rng = np.random.default_rng(2026110422)
    ws = rng.integers(0,8,(draws,8));rs = rng.integers(0,2,(draws,8,2));boot = np.zeros(draws)
    effect = a.sum((2,3))/da.sum((2,3))-b.sum((2,3))/db.sum((2,3))
    for parent in range(3):
        ss = rng.integers(0,2,(draws,2))
        sums = [x[parent][ss[:,:,None,None],ws[:,None,:,None],rs[:,None,:,:]].sum((2,3)) for x in (a,da,b,db)]
        boot += (sums[0]/sums[1]-sums[2]/sums[3]).mean(1)/3
    return dict(mean=float(effect.mean()),by_parent=effect.mean(1).tolist(),draws=draws,
        ci95=np.quantile(boot,[.025,.975]).tolist(),simultaneous98_75=np.quantile(boot,[.00625,.99375]).tolist(),
        scope='Fixed inherited parents, repeats within parent and shared workload/action blocks.')


def review(root):
    root = Path(root);folder = experiment.out(root)
    records = [experiment.load(p) for p in sorted((folder/'episodes').glob('*/*.json'))]
    lookup = {(x['arm'],x['run'],x['workload'],x['replica']):x for x in records}
    assert len(lookup) == len(records) == 1824
    matrices = {};performance = {};fingerprints = {}
    for arm in experiment.ARMS+experiment.BASELINES:
        own = [r for r in records if r['arm'] == arm]
        baseline = arm in experiment.BASELINES
        assert len(own) == (48 if baseline else 96)
        arrays = {phase:{k:np.zeros((3,2,8,2)) for k in ('generated','admitted','success')}
                  for phase in ('all','pre','peak','post')}
        for run in range(6):
            for wi,w in enumerate(experiment.WORKLOADS):
                for rep in (0,1):
                    r = lookup[arm,run//2*2 if baseline else run,w,rep]
                    assert r['audit']['status'] == 'PASS'
                    counts = r['counts']
                    assert counts['generated'] == counts.get('success',0)+counts.get('expired',0)
                    assert sum(r['successes']) == counts.get('success',0)
                    pair = (w,rep);fp = (r['fingerprint'],counts['generated'])
                    assert pair not in fingerprints or fingerprints[pair] == fp
                    fingerprints[pair] = fp
                    for phase in arrays:
                        c = counts if phase == 'all' else r['cohorts']['arrival'][phase]
                        for k in arrays[phase]:arrays[phase][k][run//2,run%2,wi,rep] = c.get(k,0)
        arrays['nonpeak'] = {k:arrays['pre'][k]+arrays['post'][k] for k in arrays['pre']}
        matrices[arm] = arrays;summary = {}
        for phase in arrays:
            parts = ('pre','post') if phase == 'nonpeak' else (phase,)
            totals = {k:sum((r['counts'] if p == 'all' else r['cohorts']['arrival'][p]).get(k,0) for r in own for p in parts)
                      for k in ('generated','admitted','success')}
            g,a,s = (totals[k] for k in ('generated','admitted','success'))
            summary[phase] = dict(counts=totals,dsr=s/g,admission=a/g,conditional=s/a)
        summary['controls'] = {}
        for phase in ('all','peak','nonpeak'):
            controls = [c for r in own for c in r['controls'] if phase == 'all' or (50 <= c['time'] < 100) == (phase == 'peak')]
            pressures = [c['pressure'] for c in controls if c['pressure'] is not None]
            summary['controls'][phase] = dict(states=len(controls),active_fraction=float(np.mean([c['alpha'] > 0 for c in controls])),
                mean_tv=float(np.mean([c['tv'] for c in controls])),
                mean_pressure=float(np.mean(pressures)) if pressures else None)
            assert all(c['tv'] <= .25+1e-10 for c in controls)
            if '_gated_' in arm or arm in ('reg_alloc_joint','reg_motion_joint'):
                assert all(c['alpha'] == (.25 if c['pressure'] > 1 else 0.) for c in controls)
        summary['physical_episodes'] = len(own)
        summary['decision_mean'] = float(np.mean([t for r in own for _,t in r['decisions']]))
        summary['decision_over_second'] = float(np.mean([t > 1 for r in own for _,t in r['decisions']]))
        summary['peak_node_quality'] = []
        for node in range(6):
            xs = [r['node_quality']['peak'][node] for r in own]
            summary['peak_node_quality'].append({k:sum(x[k] for x in xs) for k in xs[0] if k != 'node'} | dict(node=node))
        summary['peak_expiration_stage'] = {k:sum(r['cohorts']['arrival']['peak']['expiration_stage'][k] for r in own)
                                             for k in ('waiting','upload','transfer','compute')}
        performance[arm] = summary
    pairs = set()
    for mode in experiment.MODES:
        for name in ('vote','directrank','reg_uniform','reg_gated','rawrank_uniform','rawrank_gated','prob_gated'):
            pairs.add((name+'_'+mode,'hard_'+mode))
        for objective in ('reg','rawrank'):
            pairs.add((objective+'_gated_'+mode,objective+'_uniform_'+mode))
        pairs.add(('reg_gated_'+mode,'prob_gated_'+mode))
        pairs.add(('reg_gated_'+mode,'rawrank_gated_'+mode))
        pairs.add(('reg_gated_'+mode,'vote_'+mode))
    for name in ('hard','vote','directrank','reg_uniform','reg_gated','rawrank_uniform','rawrank_gated','prob_gated'):
        pairs.add((name+'_joint',name+'_decomposed'))
    pairs.update([('reg_gated_joint','reg_alloc_joint'),('reg_gated_joint','reg_motion_joint'),
                  ('reg_alloc_joint','hard_joint'),('reg_motion_joint','hard_joint'),
                  ('reg_gated_joint','plan_short'),('reg_gated_joint','initial')])
    comparisons = {}
    for a,b in sorted(pairs):
        comparison = {}
        for phase in ('all','peak','nonpeak'):
            x,y = matrices[a][phase],matrices[b][phase]
            assert np.array_equal(x['generated'],y['generated'])
            comparison[phase] = {metric:paired(x['success'],x[den],y['success'],y[den])
                for metric,den in (('dsr','generated'),('conditional','admitted'))}
        comparisons[a+'--'+b] = comparison
    primary_specs = [('reg_gated_joint--hard_joint','all','dsr',0.),
                     ('reg_gated_joint--hard_joint','peak','conditional',0.),
                     ('reg_gated_joint--hard_joint','nonpeak','dsr',-.0025),
                     ('reg_gated_joint--reg_gated_decomposed','all','dsr',0.)]
    primary = []
    for name,phase,metric,threshold in primary_specs:
        value = comparisons[name][phase][metric]
        primary.append(dict(pair=name,phase=phase,metric=metric,threshold=threshold,
                            result=value,passed=value['simultaneous98_75'][0] > threshold))
    calibration = defaultdict(list);score_rankings = defaultdict(list);roots = 0
    for file in sorted((folder/'cross_audit').glob('*/*.pkl.gz')):
        for record in experiment.saved(file):
            roots += 1;phase = 'peak' if 50 <= record['time'] < 100 else 'nonpeak'
            for arm,metrics in record['methods'].items():
                for kind in ('public','realized'):calibration[arm+'/'+phase+'/'+kind].append(metrics[kind])
            for arm,metrics in record['score_ranking'].items():
                for kind,values in metrics.items():score_rankings[arm+'/'+phase+'/'+kind].append(values)
    assert roots == 720
    diagnostic = {key:calibration_summary(rows) for key,rows in calibration.items()}
    score_summary = {key:{k:float(np.mean([r[k] for r in rows if r[k] is not None]))
        if any(r[k] is not None for r in rows) else None for k in rows[0]} for key,rows in score_rankings.items()}
    result = dict(status='COMPLETE',performance=performance,comparisons=comparisons,primary=primary,
        diagnostics=diagnostic,score_ranking=score_summary,cross_roots=roots,
        fits={f'{o}_{m}/{r}':experiment.load(folder/'fits'/f'{o}_{m}'/f'{r}.json')
              for o in ('reg','rawrank') for m in experiment.MODES for r in range(6)},
        formal_test_access=False,independent_review=False,
        scope='Fixed inherited parents, fixed scoring and gate choices, new development workloads. Four primary intervals simultaneous98_75.')
    experiment.write(folder/'analysis.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--task-root',type=Path,required=True)
    review(parser.parse_args().task_root)
