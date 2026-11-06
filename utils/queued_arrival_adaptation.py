"""Causal temporal arrival estimates and train-only relative tail values."""
import copy
import math
import numpy as np
from environment.queued_service import Engine, Task
from utils.queued_joint_mpo import arrival_prediction, value_features
from utils.queued_service_controllers import assignments
from utils.queued_root_advantage import fit_models

REVISION='queued_arrival_adaptation_20261104'
FAST_METHODS=('dual','fast2','fast5','window3')


class RateHistory:
    def __init__(self):
        self.last_time=None;self.counts=[];self.fast2=0.;self.fast5=0.

    def update_count(self,time,count):
        if not np.isfinite(count) or count<0 or (self.last_time is not None and time!=self.last_time+1):
            raise ValueError('one finite public count per increasing macro slot required')
        if self.last_time is None:self.fast2=self.fast5=float(count)
        else:
            self.fast2+=(1-math.exp(-1/2))*(count-self.fast2)
            self.fast5+=(1-math.exp(-1/5))*(count-self.fast5)
        self.counts=(self.counts+[float(count)])[-20:];self.last_time=int(time)

    def update(self,obs):
        current=sum(abs(x['arrival']-obs['time'])<1e-9 for x in obs['tasks'])
        self.update_count(int(obs['time']),current)


def slow_rate(obs,history):
    return float(sum(history.weights.get(uid,0.) for uid in obs['ues'])/20)


def rate(method,obs,history,temporal):
    slow=slow_rate(obs,history)
    if method=='slow':return slow
    if not temporal.counts:raise ValueError('causal temporal history required')
    if method=='fast2':return temporal.fast2
    if method=='fast5':return temporal.fast5
    if method=='window3':return float(np.mean(temporal.counts[-3:]))
    if method=='dual':
        threshold=2*math.sqrt(slow+1)
        changed=abs(temporal.counts[-1]-slow)>threshold or abs(temporal.fast2-slow)>threshold
        return temporal.fast2 if changed else slow
    raise ValueError('unknown registered arrival predictor')


def rate_features(obs,history,temporal):
    slow=slow_rate(obs,history);window=float(np.mean(temporal.counts[-3:]))
    return np.array([slow/60,temporal.fast2/60,temporal.fast5/60,window/60,temporal.counts[-1]/60,
        (temporal.fast2-slow)/60,(window-temporal.fast5)/60,len(temporal.counts)/20],float)


def with_rates(row,features,enabled):
    features=np.asarray(features,float)
    if features.shape!=(8,) or not np.isfinite(features).all():raise ValueError('eight causal rates required')
    return dict(row,context=np.r_[row['context'],features if enabled else np.zeros(8)])


def future_arrivals(obs,history,temporal,method,seed):
    if method=='slow':return arrival_prediction(obs,history,seed,3)
    rng=np.random.default_rng(seed);ids=sorted(obs['ues']);weights=np.array([history.weights.get(uid,0.) for uid in ids],float)
    probabilities=weights/weights.sum() if weights.sum() else np.full(len(ids),1/max(len(ids),1))
    estimate=rate(method,obs,history,temporal);output=[]
    for offset in (1,2):
        now=int(obs['time'])+offset;row=[]
        count=int(rng.poisson(estimate)) if now<200 and ids else 0
        for index in range(count):
            ue=ids[int(rng.choice(len(ids),p=probabilities))]
            bits=float(8*rng.integers(1_500_000,3_000_001));cycles=bits*int(rng.integers(500,1001))
            row.append(Task(-(now*10000+index+1),ue,float(now),now+2+float(rng.random()),bits,cycles))
        output.append(row)
    return output


def endpoint_features(obs,history,temporal):
    base=value_features(obs,history)
    # Keep the known finite horizon, remove explicit 50/100-second phase bases.
    result=np.r_[base[:2],base[4:],rate_features(obs,history,temporal)]
    future_fraction=min(1.,max(0.,200-obs['time'])/17)
    for node in range(6):result[2+node*12:4+node*12]*=future_fraction
    result[82:97]*=future_fraction
    result[97:]*=future_fraction
    if result.shape!=(105,) or not np.isfinite(result).all():raise ValueError('finite 105-channel public endpoint required')
    return result


def forecast_endpoint(obs,history,temporal,plan,motion,arrivals):
    engine=Engine.from_observation(obs);copied=copy.deepcopy(history);rates=copy.deepcopy(temporal)
    engine.assign(plan);engine.advance(motion)
    for tasks in arrivals:
        engine.add(tasks);engine.admit();public=engine.observe()
        copied._update_history(public);rates.update(public);engine.assign(assignments(public))
        engine.advance([(0.,0.)]*len(engine.positions))
    engine.audit();public=engine.observe()
    return dict(success=engine.counts.get('success',0),features=endpoint_features(public,copied,rates),
        terminal_empty=not engine.tasks and engine.t>=200)


def queue_features(x):
    result=np.asarray(x,float).copy()
    # Two horizon channels, 80 state channels and 15 spatial demand kernels.
    for node in range(6):result[...,2+node*12:4+node*12]=0.
    result[...,82:97]=0.
    return result


def fit_tail(rows):
    if any(row['split']!='training' for row in rows):raise ValueError('tail fit uses training roots only')
    fitted={}
    for feature in ('full','queue'):
        inputs=[dict(x=queue_features(r['x']) if feature=='queue' else r['x'],p=np.zeros(len(r['y'])),
                     y=r['y'],time=r['time']) for r in rows]
        fitted[feature]=fit_models(inputs,alpha=1.)
    return fitted


def tail_prediction(fitted,kind,x,terminal_empty=None):
    x=np.asarray(x,float);shape=x.shape[:-1];flat=x.reshape(-1,x.shape[-1])
    feature='queue' if kind=='queue' else 'full'
    if feature=='queue':flat=queue_features(flat)
    model=fitted[feature];name='root_shuffled' if kind=='shuffled' else 'root_centered'
    result=(flat/np.asarray(model['scale']))@np.asarray(model['models'][name]['beta'])
    result=result.reshape(shape)
    if terminal_empty is not None:result=np.where(terminal_empty,0.,result)
    return result-result.mean()


def count_losses(estimate,actual):
    estimate=max(float(estimate),1e-12);actual=float(actual)
    deviance=2*(actual*math.log(actual/estimate)-(actual-estimate)) if actual>0 else 2*estimate
    return dict(error=float(estimate-actual),absolute_error=abs(estimate-actual),poisson_deviance=deviance)
