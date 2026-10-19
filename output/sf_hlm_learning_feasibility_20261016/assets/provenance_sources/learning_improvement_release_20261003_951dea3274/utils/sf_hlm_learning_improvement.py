"""Observed-state interventions and a budgeted multi-step imitation teacher."""
from __future__ import annotations
import copy
import numpy as np
from utils.sf_hlm_v2 import CompleteQuotaSearch, graph


def action_effects(engine):
    """Eight observed features; these are estimates, never executed outcomes.

    amount, resulting source/destination quota, incumbent deadline rescues/losses,
    eligible destination deadline-safe fraction, capacity slack, MBS indicator.
    MBS has no load sharing, exactly as in the frozen matching predictor.
    """
    out = np.zeros((len(engine.actions), 8), np.float32)
    if not engine.rows: return out
    p = engine.problem; n = len(engine.rows); j = engine.num_uavs
    targets = np.asarray([j if row['options'][a]['target'] == 'mbs' else row['options'][a]['target']
                          for row, a in zip(engine.rows, engine.current_assignment)])
    load = np.maximum(engine.current, 1).copy(); load[-1] = 1
    old = p['base'] + p['work'] * load[None, :]
    for k, (source, dest, amount) in enumerate(engine.actions[1:], 1):
        q = engine.current.copy(); q[source] -= amount; q[dest] += amount
        if q[source] < 0: continue
        newload = np.maximum(q, 1); newload[-1] = 1
        new = p['base'] + p['work'] * newload[None, :]
        rescue = ((targets == source) & (old[:, source] > p['deadline']) & (new[:, source] <= p['deadline'])).sum()
        loss = ((targets == dest) & (old[:, dest] <= p['deadline']) & (new[:, dest] > p['deadline'])).sum()
        legal = p['legal'][:, dest]
        safe = (legal & (new[:, dest] <= p['deadline'])).sum()
        out[k] = [amount/n, q[source]/n, q[dest]/n, rescue/n, loss/n,
                  safe/n, (legal.sum()-q[dest])/n, float(dest == j)]
    if not np.isfinite(out).all(): raise ValueError('Nonfinite observed action effects')
    return out


def state_for(engine, effects=False):
    state = graph(engine)
    if effects: state['effects'] = action_effects(engine)
    return state


def run_search(rows, *, num_uavs=5, kind='learned', model=None, rng=None,
               exploration=False, budget=16, base_repair_budget=128, effects=False):
    if kind not in ('learned', 'no_stop', 'hybrid', 'random'): raise ValueError('Unknown intervention')
    engine = CompleteQuotaSearch(rows, num_uavs=num_uavs, matching_budget=budget,
                                 base_repair_budget=base_repair_budget)
    records = []; early_stop = None; random_completion = kind == 'random'
    while not engine.done:
        state = state_for(engine, effects); choices = np.flatnonzero(state['mask'][1:])+1
        lp = value = 0.
        if kind == 'no_stop' and len(choices): state['mask'][0] = False
        if random_completion:
            if rng is None: raise ValueError('Explicit search RNG required')
            action = int(rng.choice(choices)) if len(choices) else 0
        else:
            action, lp, value = model.act(state, exploration=exploration)
            if kind == 'hybrid' and action == 0 and len(choices):
                early_stop = engine.calls; random_completion = True
                if rng is None: raise ValueError('Explicit completion RNG required')
                action = int(rng.choice(choices)); lp = value = 0.
        reward, done = engine.step(int(action))
        records.append(dict(state=state, action=int(action), log_prob=lp, value=value, reward=reward, done=done))
    info = engine.summary(); info['early_stop_at'] = early_stop
    return engine.assignment, info, records


def multistep_teacher(rows, *, num_uavs=5, seed, total_budget=128,
                      beam_width=4, branch_width=8, depth=6, online_budget=16):
    """Branch search including neutral/worse feasible intermediate incumbents.

    Count every matching call, including initialization and unsuccessful branches.
    Return only witnessed improving paths; truncated searches cannot label STOP.
    Offline states use the online budget and actual visited-quota masks. The total
    budget counts branch solves, not the shared ordinary-repair evaluations.
    """
    if total_budget < 1: raise ValueError('Teacher budget must include initialization')
    root = CompleteQuotaSearch(rows, num_uavs=num_uavs, matching_budget=online_budget)
    calls = root.calls; initial = root.best['objective'][:2]; best = initial
    winner = []; frontier = [(root, [])]; seen = set(root.seen); rng = np.random.default_rng(seed)
    expanded = 0; exhausted = False
    for _ in range(min(depth, max(online_budget-1, 0))):
        children = []
        for parent, path in frontier:
            if parent.done: continue
            choices = np.flatnonzero(parent.action_mask()[1:])+1
            # Seeded candidate order avoids the old fixed source/destination bias.
            for action in rng.permutation(choices)[:branch_width]:
                i, j, amount = parent.actions[action]
                q = parent.current.copy(); q[i] -= amount; q[j] += amount
                if tuple(q) in seen: continue
                if calls >= total_budget: exhausted = True; break
                seen.add(tuple(q)); child = copy.copy(parent)
                child.seen = set(parent.seen); child.history = list(parent.history)
                sample = dict(state=state_for(parent, True), action=int(action))
                before = child.calls; child.step(int(action)); calls += child.calls-before; expanded += 1
                # Rejected matching/energy candidates did not reach q.
                if not np.array_equal(child.current, q): continue
                nextpath = path+[sample]
                if child.best['objective'][:2] < best: best = child.best['objective'][:2]; winner = nextpath
                children.append((child, nextpath))
            if exhausted: break
        if not children: break
        children.sort(key=lambda x: (x[0].best['objective'][:2], x[0].current_predicted['objective'][:2],
                                     tuple(s['action'] for s in x[1])))
        frontier = children[:beam_width]
        if exhausted: break
    # Never train a suffix after the first visit to the winning best assignment.
    return winner, dict(matching_calls=calls, expanded=expanded, samples=len(winner),
                        base_evaluations=root.base_evaluations, initial_evaluations=root.initial_evaluations,
                        budget=total_budget, truncated=exhausted, improved=best < initial,
                        initial_objective=list(initial), best_objective=list(best),
                        unresolved_no_improvement=not winner, stop_labels=0)
