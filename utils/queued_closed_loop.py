"""Frozen split guards, vectorized finite-bank fitting and paired pilot decisions.

Training runs are independent residual-policy repeats conditional on three FIXED
inherited A2 checkpoints. Parents are never bootstrapped as newly trained models.
"""
import numpy as np

ARMS=('relative_joint','short_joint','relative_decomposed','short_decomposed','shuffled_joint')
BASELINES=('initial','parent','plan_short','plan_relative','plan_old')

def design(phase):
    if phase not in ('pilot','formal'):raise ValueError('pilot or formal required')
    pilot=phase=='pilot';repeat=2 if pilot else 5;rounds=2 if pilot else 4;ntrain=6 if pilot else 24
    start=910000 if pilot else 920000;seeds=202611030100 if pilot else 202611031000
    runs=[dict(id=i,parent=i//repeat,repeat=i%repeat,seed=seeds+i,
        training=[list(range(start+(i*rounds+j)*ntrain,start+(i*rounds+j+1)*ntrain)) for j in range(rounds)]) for i in range(3*repeat)]
    return dict(phase=phase,runs=runs,rounds=rounds,training_episodes_per_round=ntrain,
        evaluation_workloads=list(range(911000,911008)) if pilot else list(range(930000,930040)),
        action_replicas=[0,1],arms=list(ARMS),baselines=list(BASELINES),fit_steps=12,estep_kl=.1,
        cumulative_kl=.02,per_step_kl=.002,minimum_dsr_gain=.0025,
        maximum_training_episodes=len(ARMS)*len(runs)*rounds*ntrain,
        maximum_evaluation_episodes=(len(ARMS)*len(runs)+len(BASELINES)*3)*(8 if pilot else 40)*2,
        independent_unit='Fresh residual-policy training repeat conditional on a fixed inherited parent; paired workloads and action replicates are crossed/nested evaluation measurements.')

def validate_job(spec,kind,run_id,workload):
    if kind=='evaluation':valid=spec['evaluation_workloads']
    elif kind=='training':
        run=next((r for r in spec['runs'] if r['id']==run_id),None)
        if run is None:raise ValueError('unknown training run')
        valid=sum(run['training'],[])
    else:raise ValueError('unsupported data access')
    if workload not in valid:raise ValueError('workload outside frozen phase/split')

def relative_scores(model,x,p,empty,kind):
    x=np.asarray(x,float);p=np.asarray(p,float);empty=np.asarray(empty,bool)
    if x.shape[:-1]!=p.shape or empty.shape!=p.shape:raise ValueError('paired forecast shapes required')
    if empty.all():score=p.copy()
    else:
        z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
        score=p+z@np.asarray(model['models'][kind]['beta'])
    return score-score.mean()

def paired_summary(a,b,generated,draws=4000,seed=2026110303):
    a,b,g=(np.asarray(x,float) for x in (a,b,generated))
    if a.shape!=b.shape or a.shape!=g.shape or a.ndim!=4 or np.any(g<=0) or np.any(a<0) or np.any(b<0) or np.any(a>g) or np.any(b>g):raise ValueError('parent x repeat x workload x action-replica conserved counts required')
    if not all(np.isfinite(x).all() for x in (a,b,g)):raise ValueError('finite counts required')
    B,S,W,R=a.shape;d=a-b;rng=np.random.default_rng(seed)
    effect=d.sum((2,3))/g.sum((2,3));boot=[]
    for _ in range(draws):
        ws=rng.integers(0,W,W);rs=rng.integers(0,R,(W,R));values=[]
        for parent in range(B):
            ss=rng.integers(0,S,S)
            den=g[parent][ss][:,ws[:,None],rs].sum((1,2))
            num=d[parent][ss][:,ws[:,None],rs].sum((1,2))
            values.extend(num/den)
        boot.append(float(np.mean(values)))
    return dict(mean=float(effect.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        simultaneous98_75=np.quantile(boot,[.05/8,1-.05/8]).tolist(),by_parent=effect.mean(1).tolist(),
        fixed_inherited_parents=B,training_repeats_per_fixed_parent=S,workloads=W,action_replicas=R,draws=draws,
        uncertainty='Training repeats resampled within fixed parent; workload blocks and paired action replicas shared across methods. Conditional on inherited checkpoints and the frozen fitted scorer.')

def select_method(comparisons,mechanism_ok):
    eligible=[]
    for m in ('relative_joint','short_joint'):
        c=comparisons[m+'--initial'];decomp=m.replace('joint','decomposed')
        if mechanism_ok and c['mean']>=.0025 and c['ci95'][0]>0 and min(c['by_parent'])>=-.0025 and comparisons[m+'--'+decomp]['mean']>=0:eligible.append(m)
    specific=all(comparisons['relative_joint--'+m]['ci95'][0]>0 for m in ('short_joint','shuffled_joint'))
    selected='relative_joint' if 'relative_joint' in eligible and (specific or 'short_joint' not in eligible) else 'short_joint' if 'short_joint' in eligible else None
    return dict(status='GO' if selected else 'NO_GO',selected_method=selected,eligible=eligible,
        relative_supervision_specificity_screen=bool(specific),
        rule='At least +0.25 DSR percentage points vs initial, descriptive paired 95% lower bound >0, no fixed-parent mean below -0.25 points, and nonnegative point contrast vs matching decomposition. Prefer short when both qualify but correct relative labels do not separate from short and shuffled.',
        scope='Pilot development selection only; no confirmatory claim or formal-test access.')

def _masked_softmax(x,mask,axis):
    x=np.where(mask,x,-np.inf);maximum=np.max(x,axis=axis,keepdims=True)
    maximum=np.where(np.isfinite(maximum),maximum,0.)
    y=np.where(mask,np.exp(x-maximum),0.);den=y.sum(axis=axis,keepdims=True)
    return np.divide(y,den,out=np.zeros_like(y),where=den>0)

def pack_rows(rows,oldtheta,mode='joint'):
    from marl_models.queued_joint_mpo import policy
    if not rows or mode not in ('joint','decomposed'):raise ValueError('nonempty supported actor batch required')
    N=len(rows);K=max(len(r['prior']) for r in rows);M=max(r['m'].shape[1] for r in rows)
    A=rows[0]['a'].shape[1];D=rows[0]['m'].shape[2]
    p=dict(a=np.zeros((N,K,A)),m=np.zeros((N,K,M,D)),prior=np.zeros((N,K)),
        mask=np.zeros((N,K,M),bool),old=np.zeros((N,K,M)),q=np.zeros((N,K,M)))
    for i,r in enumerate(rows):
        k,m=r['m'].shape[:2];p['a'][i,:k]=r['a'];p['prior'][i,:k]=r['prior']
        p['m'][i,:k,:m]=r['m'] if mode=='joint' else np.broadcast_to(r['m'].mean(0,keepdims=True),r['m'].shape)
        p['mask'][i,:k,:m]=True;p['old'][i,:k,:m]=policy(r,oldtheta,mode);p['q'][i,:k,:m]=r['q']
    return p

def _policies(p,theta):
    cut=p['a'].shape[-1]
    pa=_masked_softmax(np.log(p['prior'].clip(1e-300))+np.einsum('nka,a->nk',p['a'],theta[:cut]),p['prior']>0,1)
    pm=_masked_softmax(np.einsum('nkmc,c->nkm',p['m'],theta[cut:]),p['mask'],2)
    return pa,pm,pa[:,:,None]*pm

def _kl(a,b):return np.sum(a*(np.log(a.clip(1e-300))-np.log(b.clip(1e-300))))

def packed_statistics(p,theta,previous):
    pa,pm,current=_policies(p,theta);_,_,prev=_policies(p,previous)
    target=p['target'];old=p['old'];qa=target.sum(2);oa=old.sum(2)
    loss=-float(np.sum(target*np.log(current.clip(1e-300))))
    ga=np.einsum('nka,nk->a',p['a'],pa-qa)
    gm=np.einsum('nkmc,nkm->c',p['m'],qa[:,:,None]*pm-target)
    kl=float(_kl(old,current));akl=float(_kl(oa,pa))
    return dict(n=len(pa),loss=loss,gradient=np.r_[ga,gm],kl=kl,delta=float(_kl(prev,current)),
        qkl=float(_kl(target,current)),expected=float(np.sum((current-old)*p['q'])),akl=akl,mkl=kl-akl)

def batch_improve(old,values,budget=.1):
    old=np.asarray(old,float);values=np.asarray(values,float);mask=old>0
    if old.shape!=values.shape or old.ndim!=3 or budget<0:raise ValueError('paired finite bank batch required')
    if not np.isfinite(values).all() or np.any(old<0) or not np.allclose(old.sum((1,2)),1):raise ValueError('valid probability batch required')
    vmax=np.max(np.where(mask,values,-np.inf),axis=(1,2),keepdims=True)
    vmin=np.min(np.where(mask,values,np.inf),axis=(1,2),keepdims=True)
    scale=float(np.max(vmax-vmin));centered=values-vmax
    def at(eta):
        logits=np.log(old.clip(1e-300))+centered/eta;shape=logits.shape
        q=_masked_softmax(logits.reshape(len(old),-1),mask.reshape(len(old),-1),1).reshape(shape)
        return q,float(_kl(q,old)/len(old))
    if budget==0 or scale<=1e-12:return old.copy(),dict(eta=None,kl=0.,gain=0.)
    lo=scale*1e-12;hi=scale
    while at(hi)[1]>budget:hi*=2
    if at(lo)[1]<=budget:hi=lo
    else:
        for _ in range(60):
            mid=(lo+hi)/2
            if at(mid)[1]>budget:lo=mid
            else:hi=mid
    q,kl=at(hi);gain=float(np.sum((q-old)*values)/len(old))
    if kl>budget+1e-8 or gain< -1e-8:raise AssertionError('global E-step contract')
    return q,dict(eta=hi,kl=kl,gain=gain)
