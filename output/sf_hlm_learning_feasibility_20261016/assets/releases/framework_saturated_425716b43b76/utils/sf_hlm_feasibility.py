"""Frozen-budget adapters and request-count contracts; no eager numerical imports."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
import sys
import time

METHODS = ('G', 'L16', 'L8', 'R16', 'R8', 'H16', 'H8')
NONLEARNING = ('G', 'R16', 'R8', 'H16', 'H8')
FEATURE_BUDGET = 16


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest()[:4], 'big')


def branch_specs(identity, repeats=5):
    if type(repeats) is not int or repeats < 1: raise ValueError('Invalid repeats')
    out = []
    for method in METHODS:
        kind = {'G': 'greedy', 'L': 'learned', 'R': 'random', 'H': 'heuristic'}[method[0]]
        for repetition in range(repeats if kind == 'random' else 1):
            out.append(dict(method=method, kind=kind, cap=0 if method == 'G' else int(method[1:]),
                            repeat=repetition, seed=seed_for('sf_hlm_feasibility_20261016', identity, repetition)))
    return out


def require_population(observed, expected):
    observed, expected = list(observed), list(expected)
    if len(observed) != len(expected) or len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise ValueError('Missing, duplicated or foreign population')


def fee_count(rows, seconds):
    if not math.isfinite(seconds) or seconds < 0: raise ValueError('Invalid software fee')
    for row in rows:
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in ('latency', 'deadline')):
            raise ValueError('Invalid request time')
    return sum(row['latency'] + seconds <= row['deadline'] for row in rows)


def average_repeats(values):
    if not values: raise ValueError('Missing repetitions')
    return statistics.mean(values)


def mean_fee_count(rows, seconds): return average_repeats([fee_count(rows, t) for t in seconds])


def block_rate(rows):
    generated = success = 0.
    for row in rows:
        if row['generated'] < 0 or not 0 <= row['success'] <= row['generated']:
            raise ValueError('Invalid request accounting')
        generated += row['generated']; success += row['success']
    if not generated: raise ValueError('No generated requests in block')
    return success / generated


def truncated_search(rows, *, num_uavs, kind, model=None, rng=None, cap=16, trace=False):
    """Keep the complete trained input budget; stop externally after cap solves.

    The action selector and record construction mirror the frozen complete search.
    Trace construction is diagnostic only and must be disabled inside formal timers.
    """
    if sys.platform != 'linux': raise RuntimeError('Matching only runs on experiment server')
    if cap not in (8, 16) or kind not in ('learned', 'random', 'heuristic'):
        raise ValueError('Unregistered branch')
    import numpy as np
    from utils.sf_hlm_v2 import CompleteQuotaSearch, graph
    engine = CompleteQuotaSearch(rows, num_uavs=num_uavs, matching_budget=FEATURE_BUDGET,
                                 base_repair_budget=128)
    records = []; prefix = []
    while not engine.done and engine.calls < cap:
        state = graph(engine); lp = value = 0.
        if kind == 'learned': action, lp, value = model.act(state, exploration=False)
        else:
            choices = np.flatnonzero(state['mask'][1:]) + 1
            if not len(choices): action = 0
            elif kind == 'random': action = int(rng.choice(choices))
            else:
                lat = engine.problem['base'] + engine.problem['work'] * np.r_[np.maximum(engine.current[:-1], 1), 1][None, :]
                risk = ((lat > engine.problem['deadline'][:, None]) & engine.problem['legal']).sum(0)
                action = min(choices, key=lambda k: (-risk[engine.actions[k][0]],
                    risk[engine.actions[k][1]], engine.actions[k][2], int(k)))
        if trace:
            state_hash = hashlib.sha256()
            for key in sorted(state):
                array = np.asarray(state[key]); state_hash.update(key.encode())
                state_hash.update(str((array.shape, array.dtype)).encode()); state_hash.update(array.tobytes())
            before = dict(state_sha256=state_hash.hexdigest(), action=int(action), calls=engine.calls)
        reward, done = engine.step(int(action))
        records.append(dict(state=state, action=int(action), log_prob=lp, value=value,
                            reward=reward, done=done))
        if trace:
            prefix.append(dict(**before, current=engine.current.tolist(), assignment=list(engine.assignment),
                               calls_after=engine.calls, best=engine.best))
    if not engine.done:
        engine.done = True; engine.reason = 'external_matching_cap'
    diagnostic = engine.summary()
    diagnostic.update(matching_budget=cap, feature_budget=FEATURE_BUDGET,
                      prefix=prefix if trace else None)
    if engine.calls > cap: raise ValueError('Matching cap exceeded')
    return engine.assignment, diagnostic, records


def complete_policy(env, obs, spec, models, lower, runner):
    """Same deployment timer boundary as frozen v2; private truth is absent."""
    import numpy as np
    import config
    from utils.deadline_repair_online import greedy_assignment
    from utils.sf_hlm_matching import feedback
    from utils.sf_hlm_v2 import validate_metadata
    from marl_models.sf_hlm_v2 import service_view
    rng = np.random.default_rng(spec['seed'])
    started = time.perf_counter()
    _, masks, requests, rows = runner.base.prepare(env); validate_metadata(rows)
    if spec['kind'] == 'greedy':
        initial, _ = greedy_assignment(rows); assignment, diagnostic = runner.base.common_repair(rows, initial)
    else:
        assignment, diagnostic, _ = truncated_search(rows, num_uavs=config.NUM_UAVS,
            kind=spec['kind'], model=lower, rng=rng, cap=spec['cap'])
    summary = feedback(rows, assignment, config.NUM_UAVS, energy_reference=config.OFFLOAD_ENERGY_NORM_REF)
    service_view(env, summary)
    motion = models['upper'].select_actions(obs, exploration=False)
    actions = runner.base.actions_from_assignment(requests, assignment, masks)
    upload = len(json.dumps(dict(rows=rows), default=runner.base.serial, separators=(',', ':')).encode())
    download = len(json.dumps(dict(ue_ids=[q[2] for q in requests], actions=assignment,
                                   feedback=summary.tolist()), default=runner.base.serial,
                              separators=(',', ':')).encode())
    elapsed = time.perf_counter() - started
    return np.clip(motion, -1, 1), actions, dict(request_ids=[q[2] for q in requests],
        assignment=assignment, diagnostic=diagnostic, policy_seconds=elapsed,
        logical_upload_bytes=upload, logical_download_bytes=download), rows
