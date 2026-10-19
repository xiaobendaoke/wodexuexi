"""Public-only cache-hit probability calibration; no simulator inputs here."""
from __future__ import annotations
import math
import sys

FEATURES=('clipped_prior_logit','prior','own_cache','own_cache_visible','prior_times_own_cache',
    'log_size','log_cycles_per_byte','log_target_capacity',
    *('source_'+str(j) for j in range(5)),*('target_'+str(j) for j in range(5)))
MODEL=dict(C=1.,penalty='l2',solver='lbfgs',max_iter=1000,tol=1e-8)
FEATURE_MODES=('clipped','full_logit_context')


def feature_names(mode='clipped'):
    if mode not in FEATURE_MODES:raise ValueError('Unknown public feature encoding')
    if mode=='clipped':return FEATURES
    names=list(FEATURES);names[0]='prior_logit';names[4]='prior_logit_times_own_cache'
    return tuple(names)


def validate_bundle(bundle,*,supported_context_rules=('none',)):
    mode=bundle.get('feature_mode','clipped')
    if tuple(bundle['features'])!=feature_names(mode) or getattr(bundle['model'],'public_cache_feature_mode','clipped')!=mode:
        raise ValueError('Cache model encoding metadata differs from feature schema')
    rule=bundle.get('context_rule','none')
    if rule not in supported_context_rules or getattr(bundle['model'],'public_cache_context_rule','none')!=rule:
        raise ValueError('Cache context rule is unsupported or differs from model metadata')
    return mode


def own_cache(row):
    own=[o for o in row['options'].values() if o['target']==row['source']]
    if not own:return 0.,0.
    values=[float(o['cache_belief']) for o in own]
    if any(not 0<=v<=1 for v in values) or len(set(values))!=1:raise ValueError('Inconsistent public source cache')
    return values[0],1.


def controls(row,action):
    p=float(row['options'][action]['cache_belief']);own,visible=own_cache(row)
    return dict(prior=p,own_proxy=own if visible else p,half_proxy=(p+own)/2 if visible else p)


def cache_features(row,action,*,mode='clipped'):
    option=row['options'][action];target=option['target'];source=row['source'];p=float(option['cache_belief'])
    if target=='mbs' or target==source or target not in range(5) or source not in range(5) or not 0<p<1:raise ValueError('Non-calibratable public cooperative edge')
    if mode not in FEATURE_MODES:raise ValueError('Unknown public feature encoding')
    own,visible=own_cache(row);raw_logit=math.log(p)-math.log1p(-p)
    logit=max(-12.,min(12.,raw_logit))/12 if mode=='clipped' else raw_logit/12
    interaction=p*own if mode=='clipped' else logit*own
    x=[logit,p,own,visible,interaction,math.log1p(float(row['size'])),math.log1p(float(row['cycles'])/max(float(row['size']),1e-12)),
        math.log1p(float(option['capacity'])),*[float(source==j) for j in range(5)],*[float(target==j) for j in range(5)]]
    if len(x)!=18 or not all(math.isfinite(v) for v in x):raise ValueError('Invalid cache features')
    return x


def eligible(part):
    return not part['deterministic'] and part['fallback'] is None and 0<part['p']<1 and part['miss_base']-part['hit_base']>1e-10


class ConstantCacheModel:
    def __init__(self,value):self.value=float(value)
    @property
    def classes_(self):
        import numpy as np
        return np.asarray([0.,1.])
    def predict_proba(self,x):
        import numpy as np
        p=np.full(len(x),self.value,dtype=float);return np.c_[1-p,p]


def fit_model(x,y,weight):
    if sys.platform!='linux':raise RuntimeError('Model fitting runs on the experiment server')
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if len(np.unique(y))==1:
        return ConstantCacheModel((float(np.dot(weight,y))+.5)/(float(np.sum(weight))+1.)),dict(kind='smoothed_single_class_constant')
    scaler=StandardScaler().fit(x,sample_weight=weight)
    model=LogisticRegression(**MODEL).fit(scaler.transform(x),y,sample_weight=weight)
    return Pipeline([('standardize',scaler),('logistic',model)]),dict(kind='weighted_standardized_logistic',iterations=model.n_iter_.tolist())
