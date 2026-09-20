"""Deterministic assignment search; caller must supply the scientific objective.

This module does not silently choose weights or a Chapter 4 formulation.
The evaluator returns a finite scalar cost for minimization.
"""
from itertools import product
import math


def _space(candidates, initial=None):
    candidates = tuple(tuple(sorted(set(c))) for c in candidates)
    if any(not c for c in candidates):
        raise ValueError('Every request needs a feasible destination')
    if initial is not None and (len(initial) != len(candidates) or
            any(a not in c for a, c in zip(initial, candidates))):
        raise ValueError('Invalid initial assignment')
    return candidates


def _cost(evaluator, assignment):
    value = float(evaluator(tuple(assignment)))
    if not math.isfinite(value):
        raise ValueError('Objective must be finite')
    return value


def best_improvement(candidates, initial, evaluator, *, tau=1e-9, max_rounds=1000):
    """Full one-request neighborhood, best true marginal cost, fixed tie-break."""
    if not math.isfinite(tau) or tau < 0 or max_rounds < 0:
        raise ValueError('Invalid search tolerance or budget')
    candidates = _space(candidates, initial)
    current = tuple(initial)
    value = _cost(evaluator, current)
    trace = [value]
    evaluations = 1
    for _ in range(max_rounds):
        best = None
        for request, options in enumerate(candidates):
            for target in options:
                if target == current[request]:
                    continue
                proposed = current[:request] + (target,) + current[request + 1:]
                cost = _cost(evaluator, proposed)
                evaluations += 1
                if cost < value - tau:
                    key = (cost, request, target)
                    if best is None or key < best[0]:
                        best = (key, proposed)
        if best is None:
            return {'assignment': current, 'objective': value, 'trace': trace,
                    'termination': 'no_improving_move', 'local_optimality_verified': True,
                    'evaluations': evaluations}
        verified = _cost(evaluator, best[1])
        evaluations += 1
        if verified != best[0][0] or not verified < value - tau:
            raise ValueError('Evaluator is not deterministic or move is not improving')
        current, value = best[1], verified
        trace.append(value)
    return {'assignment': current, 'objective': value, 'trace': trace,
            'termination': 'iteration_budget', 'local_optimality_verified': False,
            'evaluations': evaluations}


def exact_small(candidates, evaluator, *, max_assignments=100000):
    """Exhaustive reference; reject oversize spaces before evaluating anything."""
    candidates = _space(candidates)
    size = math.prod(len(c) for c in candidates)
    if size > max_assignments:
        raise ValueError(f'Exact space {size} exceeds budget {max_assignments}')
    best = None
    for assignment in product(*candidates):
        key = (_cost(evaluator, assignment), assignment)
        if best is None or key < best:
            best = key
    return {'assignment': best[1], 'objective': best[0],
            'optimality_complete': True, 'evaluations': size}
