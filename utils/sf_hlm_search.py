"""Bounded quota search with a shared greedy energy reference and explicit MDP."""
from __future__ import annotations
import math
import time
import numpy as np
from utils.deadline_repair import assess, ordered_moves
from utils.deadline_repair_online import greedy_assignment
from utils.sf_hlm_matching import allocation_counts, problem_from_rows, solve_fixed_counts, feedback

SEARCH_CONTRACT = 'sf_hlm_quota_transfer_124_stop_common_cap_v1'
REQUEST_DIM = 4
EDGE_DIM = 12
NODE_DIM = 8


def transfer_actions(num_nodes):
    return (None,) + tuple((i, j, amount) for i in range(num_nodes) for j in range(num_nodes)
                         if i != j for amount in (1, 2, 4))


def common_repair(rows, initial, budget=128, wall_seconds=None, started=None):
    """Ordinary migrations with a shared greedy cap and whole-path clock."""
    started = time.perf_counter() if started is None else started
    greedy, _ = greedy_assignment(rows); reference = assess(rows, greedy); cap = reference['energy'] * 1.05
    current = tuple(initial); cost = assess(rows, current)
    if cost['energy'] > cap: current = greedy; cost = reference
    seen = {current}; evaluations = 0
    while evaluations < budget:
        if wall_seconds is not None and time.perf_counter() - started >= wall_seconds: break
        chosen, best = current, cost
        for i, a in ordered_moves(rows, current, 'ordinary')[0]:
            if wall_seconds is not None and time.perf_counter() - started >= wall_seconds: break
            candidate = current[:i] + (a,) + current[i + 1:]
            if candidate in seen: continue
            if evaluations >= budget: break
            seen.add(candidate); evaluations += 1; value = assess(rows, candidate)
            if value['energy'] <= cap and value['objective'] < best['objective']: chosen, best = candidate, value
        if chosen == current: break
        current, cost = chosen, best
    return current, dict(chosen_predicted=cost, reference_predicted=reference, common_energy_cap=cap,
                         base_evaluations=evaluations, matching_calls=0, matching_budget=0)


class QuotaSearch:
    def __init__(self, rows, *, num_uavs=5, initial=None, matching_budget=16, energy_ratio=1.05,
                 base_repair_budget=128, wall_seconds=None):
        if type(matching_budget) is not int or matching_budget < 0 or type(base_repair_budget) is not int or base_repair_budget < 0:
            raise ValueError('Invalid evaluation budget')
        if not math.isfinite(energy_ratio) or energy_ratio < 1: raise ValueError('Invalid common guard')
        if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0): raise ValueError('Invalid wall budget')
        self.started = time.perf_counter(); self.wall_seconds = wall_seconds
        self.rows = rows; self.num_uavs = num_uavs; self.problem = problem_from_rows(rows, num_uavs)
        greedy, self.initial_evaluations = greedy_assignment(rows)
        self.reference = assess(rows, greedy); self.cap = energy_ratio * self.reference['energy']
        if energy_ratio != 1.05: raise ValueError('SF-HLM requires the frozen common 1.05 guard')
        self.assignment, base = common_repair(rows, greedy, base_repair_budget, wall_seconds, self.started)
        self.base_evaluations = base['base_evaluations']; self.best = base['chosen_predicted']
        if initial is not None:
            value = assess(rows, tuple(initial))
            if value['energy'] <= self.cap and value['objective'] < self.best['objective']:
                self.assignment = tuple(initial); self.best = value
        self.current = allocation_counts(rows, self.assignment, num_uavs)
        self.current_predicted = self.best; self.calls = 0; self.budget = matching_budget
        self.actions = transfer_actions(num_uavs + 1); self.done = not rows or matching_budget == 0
        self.history = []; self.seen = set()
        self.reason = 'empty_or_zero_budget' if self.done else None
        if wall_seconds is not None and time.perf_counter() - self.started >= wall_seconds:
            self.done = True; self.reason = 'wall_budget'
        if not self.done:
            candidate = solve_fixed_counts(rows, self.current, num_uavs, problem=self.problem)
            self.calls = 1; self.seen.add(tuple(self.current))
            if candidate is not None and candidate['predicted']['energy'] <= self.cap and candidate['predicted']['objective'][:2] < self.best['objective'][:2]:
                self.assignment = candidate['assignment']; self.best = candidate['predicted']; self.current_predicted = self.best
            if self.calls >= self.budget: self.done = True; self.reason = 'matching_budget'

    def action_mask(self):
        mask = np.zeros(len(self.actions), dtype=bool); mask[0] = True
        if self.done: return mask
        for k, (i, j, amount) in enumerate(self.actions[1:], 1):
            if self.current[i] < amount: continue
            q = self.current.copy(); q[i] -= amount; q[j] += amount
            if tuple(q) in self.seen: continue
            # Necessary, not sufficient: Hall feasibility is checked by matching.
            if (q <= self.problem['legal'].sum(axis=0)).all(): mask[k] = True
        return mask

    def step(self, action):
        if self.done: raise ValueError('Search already terminated')
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < len(self.actions) or not self.action_mask()[action]:
            raise ValueError('Illegal quota action')
        before = self.best; candidate = None
        if action == 0:
            self.done = True; self.reason = 'policy_stop'; reward = 0.
        elif self.wall_seconds is not None and time.perf_counter() - self.started >= self.wall_seconds:
            self.done = True; self.reason = 'wall_budget'; reward = 0.
        else:
            i, j, amount = self.actions[action]; q = self.current.copy(); q[i] -= amount; q[j] += amount
            self.calls += 1; self.seen.add(tuple(q))
            candidate = solve_fixed_counts(self.rows, q, self.num_uavs, problem=self.problem)
            if candidate is not None and candidate['predicted']['energy'] <= self.cap:
                # Search may visit a worse incumbent; execution always uses best.
                self.current = q; self.current_predicted = candidate['predicted']
                if candidate['predicted']['objective'][:2] < self.best['objective'][:2]:
                    self.assignment = candidate['assignment']; self.best = candidate['predicted']
            reward = float(before['misses'] - self.best['misses'])
            reward += .1 * (before['energy'] - self.best['energy']) / max(self.cap, 1e-12) - .001
            if self.calls >= self.budget: self.done = True; self.reason = 'matching_budget'
        self.history.append(dict(action=int(action), reward=reward, done=self.done,
                                 feasible=candidate is not None, best_misses=self.best['misses'], best_energy=self.best['energy']))
        return reward, self.done

    def summary(self):
        return dict(matching_calls=self.calls, matching_budget=self.budget, base_evaluations=self.base_evaluations,
                    initial_evaluations=self.initial_evaluations, common_energy_cap=self.cap,
                    reference_predicted=self.reference, chosen_predicted=self.best,
                    termination=self.reason, search_seconds=time.perf_counter() - self.started)


def graph_state(engine):
    p = engine.problem; rows = engine.rows; r = len(rows); j = engine.num_uavs + 1
    req = np.zeros((r, REQUEST_DIM), dtype=np.float32)
    edges = np.zeros((r, j, EDGE_DIM), dtype=np.float32)
    load = np.maximum(engine.current, 1).copy(); load[-1] = 1
    for i, row in enumerate(rows):
        req[i] = [min(row['deadline'], 10.), math.log1p(row.get('cycles', 0.)) / 30,
                  math.log1p(row.get('size', 0.)) / 20, row.get('source', 0) / max(engine.num_uavs, 1)]
        for target in np.flatnonzero(p['legal'][i]):
            action = int(p['action_ids'][i, target]); o = row['options'][action]
            ratio = (o['base'] + o['work'] * load[target]) / row['deadline']
            tolerance = (row['deadline'] - o['base']) / max(o['work'], 1e-12)
            edges[i, target] = [o['base'] / row['deadline'], o['work'] / row['deadline'],
                math.log1p(o['energy']) / 20, math.log1p(o['compute_energy']) / 30,
                engine.current[target] / max(r, 1), min(ratio, 10.), np.clip(1 - ratio, -10, 1),
                np.clip(tolerance / max(r, 1), -1, 2), float(target == engine.num_uavs),
                o.get('cache_belief', 1. if target == engine.num_uavs else .5),
                math.log1p(o.get('capacity', 0.)) / 30, float(target == row.get('source', -1))]
    best_summary = feedback(rows, engine.assignment, engine.num_uavs, energy_reference=max(engine.cap, 1e-12))
    nodes = np.concatenate([best_summary, engine.current[:, None] / max(r, 1),
                            np.full((j, 1), (engine.budget - engine.calls) / max(engine.budget, 1))], -1).astype(np.float32)
    # PPO replay requires the *sample-time* mask, including visited quotas.
    return dict(requests=req, edges=edges, legal=p['legal'].copy(), nodes=nodes, mask=engine.action_mask())


def run_search(rows, *, num_uavs, kind, rng=None, model=None, exploration=False, budget=16,
               wall_seconds=None, base_repair_budget=128, initial=None):
    engine = QuotaSearch(rows, num_uavs=num_uavs, matching_budget=budget, wall_seconds=wall_seconds,
                         base_repair_budget=base_repair_budget, initial=initial)
    records = []
    if kind == 'fixed' and not engine.done:
        engine.done = True; engine.reason = 'fixed_counts'
    while not engine.done:
        state = graph_state(engine); mask = state['mask']; choices = np.flatnonzero(mask[1:]) + 1
        lp = value = 0.
        if kind == 'learned':
            if model is None: raise ValueError('Learned search needs a checkpoint')
            action, lp, value = model.act(state, exploration=exploration)
        elif not len(choices): action = 0
        elif kind == 'random':
            if rng is None: raise ValueError('Explicit independent search RNG required')
            action = int(rng.choice(choices))
        elif kind in ('heuristic', 'teacher'):
            # Prioritize source nodes whose current count breaches more deadlines.
            lat = engine.problem['base'] + engine.problem['work'] * np.r_[np.maximum(engine.current[:-1], 1), 1][None, :]
            risk = ((lat > engine.problem['deadline'][:, None]) & engine.problem['legal']).sum(0)
            action = min(choices, key=lambda k: (-risk[engine.actions[k][0]],
                         risk[engine.actions[k][1]], engine.actions[k][2], int(k)))
        else: raise ValueError('Unknown search policy')
        reward, done = engine.step(int(action))
        records.append(dict(state=state, action=int(action), log_prob=lp, value=value, reward=reward, done=done))
    return engine.assignment, engine.summary(), records
