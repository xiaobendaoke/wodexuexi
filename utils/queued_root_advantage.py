"""Same-feature ridge controls for root-centered candidate supervision.

Only supplied TRAIN rows affect coefficients or normalization. Predictors use
public forecast endpoint features and a fixed coefficient-one window term.
"""
import numpy as np


def tie_weights(scores,tolerance=1e-12):
    x=np.asarray(scores,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('finite scores required')
    top=np.isclose(x,x.max(),atol=tolerance,rtol=0)
    return top.astype(float)/top.sum()


def pairwise_loss(residual):
    r=np.asarray(residual,dtype=float)
    return float(np.mean((r[:,None]-r[None,:])**2)/2)


def fit_models(rows,alpha=1.):
    if not rows or alpha<=0:raise ValueError('nonempty train rows and positive ridge required')
    x=[np.asarray(r['x'],float) for r in rows];p=[np.asarray(r['p'],float) for r in rows];y=[np.asarray(r['y'],float) for r in rows]
    dim=x[0].shape[1]
    for a,b,c in zip(x,p,y):
        if a.ndim!=2 or a.shape[1]!=dim or b.shape!=(len(a),) or c.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(c).all():raise ValueError('invalid paired root')
    # Every root has unit weight, independently of its candidate count.
    mean=np.mean([a.mean(0) for a in x],0)
    variance=np.mean([np.mean((a-mean)**2,0) for a in x],0)
    scale=np.sqrt(variance);scale[scale<1e-12]=1.
    z=[(a-mean)/scale for a in x];targets=[a-b for a,b in zip(y,p)]
    times=sorted({int(r['time']) for r in rows});time_means={}
    for t in times:
        ids=[i for i,r in enumerate(rows) if r['time']==t]
        time_means[str(t)]=dict(x=np.mean([z[i].mean(0) for i in ids],0).tolist(),y=float(np.mean([targets[i].mean() for i in ids])))
    models={};rng=np.random.default_rng(20261103)
    shuffled=[rng.permutation(a) for a in targets]
    for kind in ('root_centered','time_centered_absolute','root_shuffled'):
        gram=np.eye(dim)*alpha;rhs=np.zeros(dim)
        for i,row in enumerate(rows):
            target=shuffled[i] if kind=='root_shuffled' else targets[i]
            if kind=='time_centered_absolute':
                center=time_means[str(int(row['time']))];a=z[i]-np.array(center['x']);b=target-center['y']
            else:a=z[i]-z[i].mean(0);b=target-target.mean()
            gram+=a.T@a/len(a);rhs+=a.T@b/len(a)
        beta=np.linalg.solve(gram,rhs)
        assert np.isfinite(beta).all()
        models[kind]=dict(beta=beta.tolist(),normal_equation_max_residual=float(np.max(np.abs(gram@beta-rhs))))
    return dict(alpha=float(alpha),feature_dim=dim,train_roots=len(rows),mean=mean.tolist(),scale=scale.tolist(),time_means=time_means,models=models,
                source='Train roots only; all roots equally weighted; no dev/test normalization or coefficient selection.')


def predict(fitted,kind,x,p,now):
    z=(np.asarray(x,float)-np.asarray(fitted['mean']))/np.asarray(fitted['scale']);p=np.asarray(p,float)
    beta=np.asarray(fitted['models'][kind]['beta'])
    if kind=='time_centered_absolute':
        center=fitted['time_means'][str(int(now))];score=p+(z-np.asarray(center['x']))@beta+center['y']
    else:score=p+(z-z.mean(0))@beta
    return score-score.mean()


def bootstrap_blocks(values,draws=4000,seed=2026110301):
    """Workload blocks, then paired replicas within each selected block.

    Input averages fixed inherited lineages/times before resampling. Returned
    intervals concern workloads/conditional futures, never new training seeds.
    """
    x=np.asarray(values,float)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('workload-by-replica values required')
    w,r=x.shape;rng=np.random.default_rng(seed);samples=[]
    for _ in range(draws):
        wi=rng.integers(0,w,w);ri=rng.integers(0,r,(w,r));samples.append(float(x[wi[:,None],ri].mean()))
    tail=.05/(2*3)
    return dict(mean=float(x.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
                simultaneous98_333=np.quantile(samples,[tail,1-tail]).tolist(),draws=draws,workloads=w,replicas=r)
