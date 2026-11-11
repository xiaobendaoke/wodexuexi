"""Frozen Chapter 4 objective and controlled assignment comparison family."""
import math
import pickle
import time
from functools import lru_cache

import config
from environment import comm_model as comms
from marl_models.offload_policy import OFFLOAD_TARGET_COOPERATIVE, OFFLOAD_TARGET_MBS
from utils.assignment_search import best_improvement, exact_small
from utils.snapshot_evaluator import evaluate

PROTOCOL = {
    'version': 'chapter4_lexicographic_energy_free_v1',
    'ordering': ['deadline_miss_count', 'mean_latency_over_deadline'],
    'comparison': 'strict lexicographic; no scalarization, rounding or tolerance',
    'deadline_epsilon_s': 1e-9,
    'physics': 'BATCH_FINAL_LOAD_EQUAL_SHARING',
    'candidate_set': 'all admitted requests; local, MBS, specific hard-feasible neighbors',
    'request_order': 'ascending UE id; production batch scheduler unchanged',
    'mbs_penalty': 0.,
    'initialization': 'heuristic-control then one own-cost greedy pass; shared by MC-ITR and Iterative-Base',
    'iterative_base': 'best own-request improvement, ignoring effects on other requests; cycle/budget guarded',
    'snapshot_distribution': 'independent counterfactual evaluations along a shared heuristic-control reference history',
}


def request_objective(record):
    latency, deadline = float(record['latency']), float(record['deadline'])
    if not all(math.isfinite(v) for v in (latency, deadline)) or latency < 0 or deadline <= 0:
        raise ValueError('Invalid request cost inputs')
    return (int(latency > deadline), latency / max(deadline, PROTOCOL['deadline_epsilon_s']))


def objective(result):
    costs = [request_objective(r) for r in result['requests']]
    return (sum(c[0] for c in costs), math.fsum(c[1] for c in costs) / max(len(costs), 1))


def heuristic_control(snapshot):
    """Original target-selection rule, explicit targets, canonical batch execution."""
    env = pickle.loads(snapshot.payload)
    ues = {ue.id: ue for ue in env.ues}
    assignment = []
    for request, candidates in zip(snapshot.requests, snapshot.candidates):
        row, _, ue_id, _, _ = request
        uav, ue = env.uavs[row], ues[ue_id]
        rate = comms.calculate_ue_uav_rate(comms.calculate_channel_gain(ue.pos, uav.pos),
                                          max(len(uav.current_covered_ues), 1))
        context, neighbor = uav._build_service_offload_context(ue.current_request, rate)
        target, neighbor = uav._select_service_target_from_context(context, neighbor)
        action = (int(config.OFFLOAD_ACTION_COOP_BASE + neighbor.id) if target == OFFLOAD_TARGET_COOPERATIVE
                  else int(config.OFFLOAD_ACTION_MBS if target == OFFLOAD_TARGET_MBS else config.OFFLOAD_ACTION_LOCAL))
        if action not in candidates:
            raise ValueError('Original heuristic selected a non-hard-feasible target')
        assignment.append(action)
    return tuple(assignment)


def solve_snapshot(snapshot, *, max_rounds=1000, exact_budget=100000):
    """Evaluate every method on one shared immutable full request set."""
    if max_rounds < 0 or exact_budget < 0:
        raise ValueError('Invalid search budgets')
    @lru_cache(maxsize=1024)
    def assessment(a):
        return evaluate(snapshot, a)
    def cost(a):
        return objective(assessment(tuple(a)))
    def own(a, i):
        return request_objective(assessment(tuple(a))['requests'][i])
    def finish(a, **extra):
        a = tuple(a)
        return {'assignment': a, 'objective': cost(a), 'result': assessment(a), **extra}

    output = {}
    started = time.perf_counter()
    initial = heuristic_control(snapshot)
    output['heuristic_control'] = finish(initial)
    output['heuristic_control']['wall_seconds'] = time.perf_counter() - started
    assessment.cache_clear()
    started = time.perf_counter()
    greedy = initial
    for i, candidates in enumerate(snapshot.candidates):
        best = (own(greedy, i), greedy[i], greedy)
        for target in candidates:
            proposed = greedy[:i] + (target,) + greedy[i+1:]
            key = (own(proposed, i), target, proposed)
            if key[0] < best[0]:
                best = key
        greedy = best[2]
    output['greedy_single_pass'] = finish(greedy, initial_assignment=initial)
    output['greedy_single_pass']['wall_seconds'] = time.perf_counter() - started

    assessment.cache_clear()
    started = time.perf_counter()
    current, seen, trace = greedy, {greedy}, [cost(greedy)]
    termination = 'iteration_budget'
    for _ in range(max_rounds):
        best = None
        for i, candidates in enumerate(snapshot.candidates):
            before = own(current, i)
            for target in candidates:
                proposed = current[:i] + (target,) + current[i+1:]
                after = own(proposed, i)
                if after < before:
                    key = (tuple(a-b for a, b in zip(after, before)), i, target)
                    if best is None or key < best[0]:
                        best = (key, proposed)
        if best is None:
            termination = 'no_own_cost_improvement'
            break
        if best[1] in seen:
            termination = 'cycle_guard'
            break
        current = best[1]
        seen.add(current)
        trace.append(cost(current))
    output['iterative_base'] = finish(current, initial_assignment=greedy, trace=trace,
                                      termination=termination, local_optimality_verified=False)
    output['iterative_base']['wall_seconds'] = time.perf_counter() - started

    assessment.cache_clear()
    started = time.perf_counter()
    # Search uses uncached production evaluations, including its post-move check.
    final = best_improvement(snapshot.candidates, greedy,
                             lambda a: objective(evaluate(snapshot, a)), tau=0., max_rounds=max_rounds)
    output['mc_itr'] = finish(final.pop('assignment'), initial_assignment=greedy, **final)
    output['mc_itr']['wall_seconds'] = time.perf_counter() - started
    size = math.prod(map(len, snapshot.candidates))
    assessment.cache_clear()
    started = time.perf_counter()
    if size <= exact_budget:
        exact = exact_small(snapshot.candidates, cost, max_assignments=exact_budget)
        output['exact_small'] = finish(exact.pop('assignment'), **exact)
    else:
        output['exact_small'] = {'optimality_complete': False, 'status': 'SKIPPED_SPACE_LIMIT',
                                 'assignment_space': size, 'budget': exact_budget}
    output['exact_small']['wall_seconds'] = time.perf_counter() - started
    if output['exact_small']['optimality_complete']:
        optimum = output['exact_small']['objective']
        for result in output.values():
            result['matches_exact'] = result['objective'] == optimum
            result['exact_gap_components'] = tuple(a-b for a, b in zip(result['objective'], optimum))
    return output
