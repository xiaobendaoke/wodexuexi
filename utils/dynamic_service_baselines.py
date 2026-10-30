"""Deterministic current-information controls; no tape/future access."""
from collections import deque
from time import perf_counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from environment.dynamic_service import evaluate, prefix_batch, project_positions


def objective(batch, assignment):
    out = evaluate(batch, assignment)
    return len(batch)-out['success'], float(np.minimum(out['latencies']/batch.deadline, 2).sum())


def greedy(batch):
    chosen = []
    for i in range(len(batch)):
        sub = prefix_batch(batch, i+1)
        target = min(np.flatnonzero(batch.legal[i]), key=lambda j: (objective(sub, chosen+[j]), j))
        chosen.append(int(target))
    return np.array(chosen, dtype=np.int64)


def conditional_match(batch, quotas):
    slots = np.repeat(np.arange(6), quotas)
    if len(slots) != len(batch):
        raise ValueError('quota sum must equal request count')
    if not len(batch):
        return np.empty(0, np.int64)
    latency = batch.tx[:, slots] + batch.cycles[:, None] * quotas[slots] / batch.capacity[slots]
    # Each secondary term <=1, so R+1 strictly prioritizes miss count.
    cost = (len(batch)+1)*(latency > batch.deadline[:, None]) + np.minimum(latency/batch.deadline[:, None], 2)/2
    cost = np.where(batch.legal[:, slots], cost, 1e12)
    row, col = linear_sum_assignment(cost)
    if np.any(cost[row, col] >= 1e12):
        return None
    assignment = np.empty(len(batch), np.int64)
    assignment[row] = slots[col]
    return assignment


def matching_search(batch, budget=16):
    if budget < 1:
        raise ValueError('positive matching budget required')
    best = greedy(batch)
    q = np.bincount(best, minlength=6)
    initial = conditional_match(batch, q)
    calls = 1
    if initial is not None and objective(batch, initial) < objective(batch, best):
        best = initial
    seen = {tuple(q)}
    while calls < budget:
        proposals = []
        for i in range(6):
            for j in range(6):
                for amount in (1, 2, 4):
                    if i != j and q[i] >= amount:
                        cand = q.copy()
                        cand[i] -= amount
                        cand[j] += amount
                        if tuple(cand) not in seen:
                            # Deterministic load-pressure ordering, not hidden truth.
                            pressure = np.max(cand/batch.capacity)
                            proposals.append((pressure, tuple(cand)))
        if not proposals:
            break
        _, candidate = min(proposals)
        seen.add(candidate)
        trial = conditional_match(batch, np.array(candidate))
        calls += 1
        if trial is not None and objective(batch, trial) < objective(batch, best):
            best, q = trial, np.array(candidate)
    return best


class HistoryController:
    """Public observation history only. No true hotspot center or scenario id."""
    def __init__(self):
        self.history = deque(maxlen=20)

    def update(self, obs, positions):
        visible = obs['visible'].any(axis=0)
        pos, active = np.zeros((100, 2)), np.zeros(100)
        for u in np.flatnonzero(visible):
            i = int(np.flatnonzero(obs['visible'][:, u])[0])
            pos[u] = obs['ue'][i, u, :2]*700 + positions[i]
            active[u] = obs['ue'][i, u, 2]
        self.history.append((pos, active, visible))

    def forecast(self, horizon=0):
        pos = self.history[-1][0].copy()
        masks = np.stack([x[2] for x in self.history])
        activity = np.stack([x[1] for x in self.history])
        mean = activity.sum(axis=0)/np.maximum(masks.sum(axis=0), 1)
        if len(activity) > 1:
            split = max(1, len(activity)//2)
            trend = activity[split:].mean(axis=0)-activity[:split].mean(axis=0)
        else:
            trend = np.zeros(100)
        for old_pos, _, seen in self.history:
            pos[seen] = old_pos[seen]
        weight = np.clip(mean + horizon*trend/max(len(activity), 1), 0, 1)
        return pos, weight

    def actions(self, positions):
        pos, weight = self.forecast(1)
        if not weight.sum():
            return np.zeros((5, 2))
        centers = positions.copy()
        for _ in range(3):
            groups = np.linalg.norm(pos[:, None]-centers[None], axis=-1).argmin(axis=1)
            for i in range(5):
                w = weight*(groups == i)
                if w.sum():
                    centers[i] = (pos*w[:, None]).sum(axis=0)/w.sum()
        delta = centers-positions
        return delta/np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 15)

    def mpc(self, env, batch, budget=.2):
        start = perf_counter()
        self.last_mpc_candidates = 0
        best = self.actions(env.positions)
        dirs = np.r_[np.zeros((1, 2)), np.c_[np.cos(np.arange(8)*np.pi/4), np.sin(np.arange(8)*np.pi/4)]]
        # Current observed tasks are persistence forecasts, never future tape.
        known = [r for r in env.current_requests if self.history[-1][2][r.ue]]
        def score(action):
            score_value, positions = 0., env.positions.copy()
            for horizon in range(1, 21):
                if perf_counter()-start >= budget:
                    return None
                positions = project_positions(positions, action)
                pred = env.batch(positions=positions, requests=known)
                # One conditional assignment per predicted slot; the final
                # current-slot control still uses the full 16-call search.
                # This keeps 20-step forecasts within a useful soft budget.
                allocation = conditional_match(pred, np.bincount(pred.source, minlength=6))
                if allocation is None:
                    allocation = pred.source
                _, weights = self.forecast(horizon)
                served = evaluate(pred, allocation)['satisfied']
                score_value += float(weights[pred.ue[served]].sum())
            return score_value
        incumbent = score(best)
        if incumbent is None:
            return best
        self.last_mpc_candidates += 1
        for _ in range(2):
            for i in range(5):
                for direction in dirs:
                    trial = best.copy()
                    trial[i] = direction
                    candidate = score(trial)
                    if candidate is None:
                        return best
                    self.last_mpc_candidates += 1
                    if candidate > incumbent:
                        best, incumbent = trial, candidate
        return best
