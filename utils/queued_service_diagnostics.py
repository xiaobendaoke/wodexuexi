"""Pure diagnostic statistics and safe action commitments; never a trainer."""
import math
import statistics
from environment.queued_service import safe_path


def commit_sequence(engine, first):
    positions=list(engine.positions);sequence=[];zero=[(0.,0.)]*engine.n
    for _ in range(6):
        move=zero
        for scale in (1.,.5,.25):
            candidate=[(dx*scale,dy*scale) for dx,dy in first]
            if safe_path(positions,candidate,engine.p):move=candidate;break
        sequence.append(move)
        positions=[(a[0]+d[0],a[1]+d[1]) for a,d in zip(positions,move)]
    return sequence


def probability_metrics(p,q):
    if len(p)!=len(q) or not p:raise ValueError('aligned nonempty probabilities required')
    if any(x<0 or not math.isfinite(x) for x in (*p,*q)):raise ValueError('invalid probability')
    if abs(sum(p)-1)>1e-5 or abs(sum(q)-1)>1e-5:raise ValueError('probabilities must sum to one')
    entropy=-sum(x*math.log(max(x,1e-30)) for x in p)
    return dict(tv=sum(abs(a-b) for a,b in zip(p,q))/2,
        entropy_fraction=entropy/math.log(len(p)) if len(p)>1 else 1.,
        argmax_changed=max(range(len(p)),key=p.__getitem__)!=max(range(len(q)),key=q.__getitem__),
        maximum_probability=max(p))


def payoff_metrics(local,coop):
    if len(local)!=len(coop) or not local:raise ValueError('aligned payoff vectors required')
    differences=[b-a for a,b in zip(local,coop)]
    return dict(local_range=max(local)-min(local),coop_range=max(coop)-min(coop),
        interaction_range=max(differences)-min(differences),
        strict_reversal=any((local[i]-local[j])*(coop[i]-coop[j])<0 for i in range(len(local)) for j in range(i)))


def mc_returns(rewards):
    total=0.;returns=[]
    for value in reversed(rewards):total+=value;returns.append(total)
    return list(reversed(returns))


def explained_variance(target,prediction):
    if not target or len(target)!=len(prediction):raise ValueError('aligned nonempty values required')
    variance=statistics.pvariance(target)
    return 1-statistics.pvariance(a-b for a,b in zip(target,prediction))/variance if variance>1e-12 else None
