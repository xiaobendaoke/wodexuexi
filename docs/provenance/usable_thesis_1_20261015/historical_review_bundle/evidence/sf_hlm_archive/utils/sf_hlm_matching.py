"""Fixed-cardinality assignment under the frozen proposal predictor.

Optimality is conditional on counts and predictor inputs, WITHOUT an energy
side constraint. The online search applies its separate common energy guard.
No simulator, future state, cache truth or random generator is accessed here.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.deadline_repair import assess

MATCHING_CONTRACT = 'sf_hlm_fixed_count_lex_miss_energy_v1'
FEEDBACK_CONTRACT = 'sf_hlm_predecision_six_per_node_v1'
FEEDBACK_NAMES = ('count_fraction', 'miss_fraction', 'mean_margin', 'min_margin',
                  'release_rescues_fraction', 'normalized_assignment_energy')


def problem_from_rows(rows, num_uavs):
    if type(num_uavs) is not int or num_uavs < 1:
        raise ValueError('Positive UAV count required')
    shape = (len(rows), num_uavs + 1)
    result = {k: np.zeros(shape, dtype=np.float64) for k in ('base', 'work', 'energy', 'compute_energy')}
    result['legal'] = np.zeros(shape, dtype=bool)
    result['action_ids'] = np.full(shape, -1, dtype=np.int64)
    result['deadline'] = np.asarray([r['deadline'] for r in rows], dtype=np.float64)
    if not np.isfinite(result['deadline']).all() or (result['deadline'] <= 0).any():
        raise ValueError('Finite positive deadlines required')
    for i, row in enumerate(rows):
        if not row['options']: raise ValueError('No legal execution target')
        for action, option in row['options'].items():
            if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
                raise ValueError('Noninteger action identifier')
            target = option['target']
            if target == 'mbs': j = num_uavs
            elif isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)) or not 0 <= target < num_uavs:
                raise ValueError('Invalid global execution target')
            else: j = int(target)
            if result['legal'][i, j]: raise ValueError('Duplicate execution target')
            for key in ('base', 'work', 'energy', 'compute_energy'):
                value = float(option[key])
                if not math.isfinite(value) or value < 0: raise ValueError('Invalid predictor quantity')
                result[key][i, j] = value
            if j == num_uavs and (option['work'] != 0 or option['compute_energy'] != 0):
                raise ValueError('MBS must retain its nonsharing predictor')
            result['legal'][i, j] = True
            result['action_ids'][i, j] = int(action)
    return result


def validated_counts(counts, n, num_uavs):
    if any(isinstance(v, (bool, np.bool_)) for v in counts): raise ValueError('Boolean count')
    raw = np.asarray(counts)
    if raw.shape != (num_uavs + 1,) or raw.dtype.kind not in 'iu' or raw.dtype.kind == 'b':
        raise ValueError('Counts must be an integer vector, not booleans/floats')
    if (raw < 0).any() or (raw > n).any() or sum(int(v) for v in raw) != n: raise ValueError('Counts must conserve every admitted request')
    return raw.astype(np.int64)


def allocation_counts(rows, assignment, num_uavs):
    if len(rows) != len(assignment): raise ValueError('Missing request assignment')
    counts = np.zeros(num_uavs + 1, dtype=np.int64)
    for row, action in zip(rows, assignment):
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or action not in row['options']:
            raise ValueError('Assignment violates quality mask')
        target = row['options'][action]['target']
        j = num_uavs if target == 'mbs' else target
        if isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) or not 0 <= j <= num_uavs:
            raise ValueError('Invalid execution target')
        counts[int(j)] += 1
    return counts


def fixed_costs(problem, counts):
    n = np.maximum(counts, 1).copy(); n[-1] = 1
    latency = problem['base'] + problem['work'] * n[None, :]
    energy = problem['energy'] + problem['compute_energy'] / n[None, :] ** 2
    misses = latency > problem['deadline'][:, None]
    # Every normalized edge energy <= 1, hence total <= R. M=R+1
    # makes one extra miss strictly worse than any possible energy reduction.
    scale = max(float(energy[problem['legal']].max(initial=0.)), np.finfo(float).tiny)
    cost = (len(latency) + 1) * misses + energy / scale
    cost[~problem['legal']] = np.inf
    return cost, latency, energy


def solve_fixed_counts(rows, counts, num_uavs, *, problem=None):
    counts = validated_counts(counts, len(rows), num_uavs)
    problem = problem_from_rows(rows, num_uavs) if problem is None else problem
    if not rows:
        return dict(assignment=(), counts=counts, predicted=assess([], ()))
    slots = np.repeat(np.arange(num_uavs + 1), counts)
    cost, _, _ = fixed_costs(problem, counts)
    try:
        rr, cc = linear_sum_assignment(cost[:, slots])
    except ValueError as exc:
        if 'infeasible' in str(exc): return None
        raise
    targets = slots[cc]
    if not problem['legal'][rr, targets].all(): return None
    assignment = tuple(int(a) for a in problem['action_ids'][rr, targets])
    if not np.array_equal(allocation_counts(rows, assignment, num_uavs), counts):
        raise RuntimeError('Matching did not conserve target cardinalities')
    return dict(assignment=assignment, counts=counts, predicted=assess(rows, assignment))


def feedback(rows, assignment, num_uavs, *, energy_reference):
    if not math.isfinite(energy_reference) or energy_reference <= 0: raise ValueError('Invalid energy normalization')
    counts = allocation_counts(rows, assignment, num_uavs)
    output = np.zeros((num_uavs + 1, 6), dtype=np.float32)
    if not rows: return output
    groups = [[] for _ in counts]
    for row, action in zip(rows, assignment):
        option = row['options'][action]
        j = num_uavs if option['target'] == 'mbs' else int(option['target'])
        n = 1 if j == num_uavs else int(counts[j])
        latency = option['base'] + option['work'] * n
        margin = 1. - latency / row['deadline']
        rescued = j != num_uavs and option['base'] + option['work'] * max(n - 1, 1) <= row['deadline'] < latency
        energy = option['energy'] + option['compute_energy'] / n ** 2
        groups[j].append((margin, rescued, energy))
    for j, group in enumerate(groups):
        if not group: continue
        margins = [g[0] for g in group]
        output[j] = [counts[j] / len(rows), sum(v < 0 for v in margins) / len(group),
                     np.clip(np.mean(margins), -10, 1), np.clip(min(margins), -10, 1),
                     sum(g[1] for g in group) / len(rows), sum(g[2] for g in group) / energy_reference]
    return output
