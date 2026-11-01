"""Pure helpers for public pre/plan summaries and separate branch credit."""
import statistics


def pooled_features(features,mask):
    rows=[x for x,valid in zip(features,mask) if valid]
    if len(features)!=len(mask) or not rows or any(len(x)!=len(rows[0]) for x in rows):
        raise ValueError('aligned nonempty legal candidate bank required')
    cols=list(zip(*rows))
    return [statistics.mean(x) for x in cols]+[min(x) for x in cols]+[max(x) for x in cols]


def phase_keys(rows):
    return [r.get('decision_time',round(200*(1-r['state'][72]))) for r in rows]


def branch_advantages(returns,pre_values,plan_values):
    if not returns or len(returns)!=len(pre_values) or len(returns)!=len(plan_values):
        raise ValueError('aligned complete episode values required')
    return [a-b for a,b in zip(returns,pre_values)],[a-b for a,b in zip(returns,plan_values)]
