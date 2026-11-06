"""Dimensionless candidate policy mapping; no rewards or future labels."""
import math
import statistics


def probabilities(q,mask,temperature,normalize):
    valid=[i for i,ok in enumerate(mask) if ok]
    if not valid or temperature<=0:raise ValueError('legal candidates and positive temperature required')
    if any(not math.isfinite(q[i]) for i in valid):raise ValueError('nonfinite legal score')
    values=[q[i] for i in valid]
    if normalize:
        mean=statistics.mean(values);values=[x-mean for x in values];scale=math.sqrt(statistics.mean(x*x for x in values))
        values=[x/scale for x in values] if scale else [0.]*len(values)
    values=[x/temperature for x in values];high=max(values);weights=[math.exp(x-high) for x in values];total=sum(weights)
    result=[0.]*len(q)
    for i,w in zip(valid,weights):result[i]=w/total
    return result


def entropy(p,legal_count):
    valid=[x for x in p if x>0]
    return -sum(x*math.log(x) for x in valid)/math.log(legal_count) if legal_count>1 else 0.


def calibrate(rows,target=.5):
    eligible=[r for r in rows if sum(r['mask'])>1 and max(q for q,m in zip(r['q'],r['mask']) if m)-min(q for q,m in zip(r['q'],r['mask']) if m)>1e-9]
    if not eligible:return dict(calibrated=False,reason='no_nonflat_states',temperature=1.,states=0)
    def measure(t):
        return statistics.mean(entropy(probabilities(r['q'],r['mask'],t,True),sum(r['mask'])) for r in eligible)
    low,high=.001,100.
    for _ in range(48):
        mid=(low+high)/2
        if measure(mid)<target:low=mid
        else:high=mid
    temperature=(low+high)/2;achieved=measure(temperature)
    return dict(calibrated=abs(achieved-target)<.005,temperature=temperature,target_entropy=target,achieved_entropy=achieved,states=len(eligible))
