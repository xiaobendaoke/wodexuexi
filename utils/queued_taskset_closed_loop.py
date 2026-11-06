"""Closed-loop design, settled cohorts and candidate-label calibration metrics."""
import numpy as np
from utils.queued_learning_validation import projection

REVISION='queued_taskset_closed_loop_20261104'
ARMS=tuple(f'{feature}_{label}_{mode}' for feature in ('taskset','summary')
           for label in ('single','mean16') for mode in ('joint','decomposed'))+('original_joint','original_decomposed')
BASELINES=('initial','plan_short')
TIMES=(30,60,80,100,150,180)


def design():
    return dict(revision=REVISION,arms=list(ARMS),baselines=list(BASELINES),workloads=list(range(943000,943008)),
        replicas=[0,1],runs=list(range(6)),diagnostic_workloads=[943000,943004],diagnostic_replica=0,
        times=list(TIMES),episodes=1056,roots=792,public_forecasts=16,real_horizon=3,
        primary_pairs=['taskset_single_joint--summary_single_joint','taskset_single_joint--taskset_single_decomposed'],
        frozen_models=True,new_optimizer_steps=0,formal_test_access=False)


def period(t):return 'pre' if t<50 else 'peak' if t<100 else 'post'


def cohort_statistics(tasks):
    tasks=list(tasks)
    def describe(xs):
        n=len(xs);admitted=[x for x in xs if x.source is not None];success=[x for x in xs if x.stage=='completed']
        stages={k:sum(x.stage=='expired' and x.expired_stage==k for x in xs) for k in ('waiting','upload','transfer','compute')}
        a=len(admitted);s=len(success)
        return dict(generated=n,admitted=a,success=s,expired=n-s,dsr=s/n if n else None,
            admission=a/n if n else None,conditional=s/a if a else None,expiration_stage=stages,
            completion_latency=sum(x.finished-x.arrival for x in success)/s if s else None,
            cpu_wait_per_admitted=sum(x.time_in['compute_wait'] for x in admitted)/a if a else None)
    return dict(arrival={p:describe([x for x in tasks if period(x.arrival)==p]) for p in ('pre','peak','post')},
        admission={p:describe([x for x in tasks if x.admitted is not None and period(x.admitted)==p]) for p in ('pre','peak','post')})


def summary_extra(row):
    k,m=row['m'].shape[:2]
    raw=np.concatenate([np.broadcast_to(row['a'][:,None,:],(k,m,row['a'].shape[1])),row['m'],
                        np.broadcast_to(row['context'],(k,m,len(row['context'])))],axis=2)
    return projection(raw)


def calibration_record(probability,reference,scores):
    p,ref,q=(np.asarray(x,float) for x in (probability,reference,scores))
    if p.shape!=ref.shape or p.shape!=q.shape or p.ndim!=2 or not p.size:
        raise ValueError('paired finite candidate banks required')
    if not all(np.isfinite(x).all() for x in (p,ref,q)) or np.any(p<0) or np.any(ref<0) or not np.isclose(p.sum(),1.) or not np.isclose(ref.sum(),1.):
        raise ValueError('normalized finite candidate distributions required')
    top=p>=p.max()-1e-12;best=q>=q.max()-1e-12
    ce=-float(np.sum(ref*np.log(p.clip(1e-300))))
    entropy=-float(np.sum(ref*np.log(ref.clip(1e-300))))
    i,j=np.triu_indices(p.size,1);dq=q.ravel()[i]-q.ravel()[j];dp=p.ravel()[i]-p.ravel()[j]
    informative=np.abs(dq)>1e-12
    pair_correct=np.where(np.abs(dp)<=1e-12,.5,(np.sign(dp)==np.sign(dq)).astype(float))
    weights=np.abs(dq[informative])
    return dict(confidence=float(p.max()),top_label_probability=float(ref[top].mean()),
        brier=float(np.sum((p-ref)**2)),nll=ce,kl=ce-entropy,
        score_regret=float(q.max()-np.sum(p*q)),greedy_score_regret=float(q.max()-q[top].mean()),
        bestset_probability=float(p[best].sum()),candidate_count=int(p.size),
        normalized_entropy=-float(np.sum(p*np.log(p.clip(1e-300))))/np.log(p.size) if p.size>1 else 0.,
        pairwise_accuracy=float(pair_correct[informative].mean()) if informative.any() else None,
        weighted_pairwise_accuracy=float(np.sum(pair_correct[informative]*weights)/weights.sum()) if weights.size else None,
        all_scores_tied=bool(best.all()))


def calibration_summary(records):
    if not records:raise ValueError('nonempty calibration observations required')
    keys=('brier','nll','kl','score_regret','greedy_score_regret','bestset_probability','normalized_entropy',
          'pairwise_accuracy','weighted_pairwise_accuracy','confidence','top_label_probability','candidate_count')
    result={key:float(np.mean([r[key] for r in records if r[key] is not None]))
            if any(r[key] is not None for r in records) else None for key in keys}
    bins=[];ece=0.
    for index in range(10):
        selected=[r for r in records if min(9,int(r['confidence']*10))==index]
        if selected:
            confidence=float(np.mean([r['confidence'] for r in selected]));accuracy=float(np.mean([r['top_label_probability'] for r in selected]))
            ece+=len(selected)/len(records)*abs(confidence-accuracy)
        else:confidence=accuracy=None
        bins.append(dict(lower=index/10,upper=(index+1)/10,states=len(selected),confidence=confidence,label_frequency=accuracy))
    result.update(states=len(records),ece=ece,bins=bins,
        all_scores_tied_fraction=float(np.mean([r['all_scores_tied'] for r in records])),
        note='Calibration against the specified optimal-candidate categorical label, not probability of request success. Correlated roots are not independent training repeats.')
    informative=[r for r in records if not r['all_scores_tied']]
    if informative and len(informative)<len(records):
        detail=calibration_summary(informative)
        result['informative']={k:v for k,v in detail.items() if k not in ('informative','note')}
    elif informative:result['informative']={k:v for k,v in result.items() if k!='note'}
    else:result['informative']=None
    return result


def paired_ratios(a,da,b,db,draws=4000,seed=2026110412):
    """Paired policies with their own admitted denominators; no pseudo-repeats."""
    a,da,b,db=(np.asarray(x,float) for x in (a,da,b,db))
    if not(a.shape==da.shape==b.shape==db.shape) or a.ndim!=4 or np.any(da<=0) or np.any(db<=0):
        raise ValueError('positive policy-specific paired denominators required')
    if not all(np.isfinite(x).all() for x in (a,da,b,db)) or np.any(a<0) or np.any(b<0) or np.any(a>da) or np.any(b>db):
        raise ValueError('finite conserved paired counts required')
    B,S,W,R=a.shape;rng=np.random.default_rng(seed);boot=[]
    effect=a.sum((2,3))/da.sum((2,3))-b.sum((2,3))/db.sum((2,3))
    for _ in range(draws):
        ws=rng.integers(0,W,W);rs=rng.integers(0,R,(W,R));values=[]
        for parent in range(B):
            ss=rng.integers(0,S,S)
            sums=[x[parent][ss][:,ws[:,None],rs].sum((1,2)) for x in (a,da,b,db)]
            values.extend(sums[0]/sums[1]-sums[2]/sums[3])
        boot.append(float(np.mean(values)))
    return dict(mean=float(effect.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        by_parent=effect.mean(1).tolist(),draws=draws,
        scope='Policy-specific admitted denominators, fixed-parent paired policy/workload bootstrap; exploratory.')
