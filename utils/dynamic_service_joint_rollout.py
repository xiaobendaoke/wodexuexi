"""Physical-time lambda returns for allocation -> reward -> movement.

No time or discount elapses between requests. Inputs are complete physical
episodes; truncation must not be silently treated as a terminal transition.
"""
import numpy as np


def causal_returns(rows, lam=.99, terminal=True, allocation_only=False):
    if not terminal:
        raise ValueError('complete episodes required; truncated rollouts need an explicit bootstrap')
    if not 0 <= lam <= 1:
        raise ValueError('lambda outside [0,1]')
    result=[None]*len(rows)
    next_value=next_return=0.
    for t in reversed(range(len(rows))):
        item=rows[t]
        prefix=np.asarray(item['prefix_values'],dtype=np.float64)
        future=(1-lam)*next_value+lam*next_return if t+1<len(rows) else 0.
        if allocation_only:future=0.
        target=float(item['reward'])+future
        result[t]=dict(prefix_targets=np.full(len(prefix),target,dtype=np.float32),
                       motion_target=float(future))
        next_value=float(prefix[0]) if len(prefix) else float(item['motion_value'])
        next_return=target
    return result


def slot_surrogate(log, old_log, advantage, clip=.2):
    """Sum conditional-action terms. Caller averages physical slots."""
    import torch
    ratio=(log-old_log).exp()
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).sum()
