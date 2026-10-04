"""Deterministic proposal-only repair. No simulator calls during search.

The predictor uses equal CPU sharing and predecision cache beliefs. It is not
an oracle, a physical feasibility guarantee, or a deployed coordinator.
"""
from __future__ import annotations
from collections import Counter
import math


def assess(rows, assignment):
    if len(rows) != len(assignment):
        raise ValueError('Every admitted request needs an assignment')
    options = []
    for row, a in zip(rows, assignment):
        if type(a) is not int or a not in row['options']:
            raise ValueError('Illegal or padded action')
        if not math.isfinite(row['deadline']) or row['deadline'] <= 0:
            raise ValueError('Invalid deadline')
        option = row['options'][a]
        if any(not math.isfinite(option[k]) or option[k] < 0
               for k in ('base', 'work', 'energy', 'compute_energy')):
            raise ValueError('Invalid prediction input')
        options.append(option)
    counts = Counter(o['target'] for o in options if o['target'] != 'mbs')
    latencies, energies = [], []
    for o in options:
        n = counts[o['target']] if o['target'] != 'mbs' else 1
        latencies.append(o['base'] + o['work']*n)
        energies.append(o['energy'] + o['compute_energy']/n**2)
    misses = sum(t > r['deadline'] for t, r in zip(latencies, rows))
    energy = math.fsum(energies)
    ratio = math.fsum(t/r['deadline'] for t, r in zip(latencies, rows))/max(len(rows), 1)
    return {'misses': misses, 'energy': energy, 'latency_ratio': ratio,
            'latencies': latencies, 'objective': (misses, energy, ratio)}


def ordered_moves(rows, current, mode):
    """Deadline order prioritizes releasing a task from sensitive targets.

    This screen is counted separately and timed; no call to an actual-world
    evaluator is made. Destination ordering uses own predicted latency only.
    """
    options = [r['options'][a] for r, a in zip(rows, current)]
    counts = Counter(o['target'] for o in options)
    indices = list(range(len(rows)))
    checks = 0
    if mode == 'deadline':
        rescues = Counter()
        for row, o in zip(rows, options):
            if o['target'] == 'mbs': continue
            n = counts[o['target']]
            checks += 1
            rescues[o['target']] += int(o['base']+o['work']*(n-1) <= row['deadline']
                                        < o['base']+o['work']*n)
        def priority(i):
            o = options[i]; n = counts[o['target']] if o['target'] != 'mbs' else 1
            t = o['base']+o['work']*n
            return (-rescues[o['target']], -int(t > rows[i]['deadline']), -t/rows[i]['deadline'], i)
        indices.sort(key=priority)
    moves = []
    for i in indices:
        targets = [a for a in rows[i]['options'] if a != current[i]]
        if mode == 'deadline':
            def latency(a):
                o = rows[i]['options'][a]
                n = counts[o['target']]+1 if o['target'] != 'mbs' else 1
                return (o['base']+o['work']*n, a)
            targets.sort(key=latency)
            checks += len(targets)
        else: targets.sort()
        moves.extend((i, a) for a in targets)
    return moves, checks


def repair(rows, initial, mode, budget, energy_ratio=1.05):
    if mode not in ('ordinary', 'deadline') or type(budget) is not int or budget < 0:
        raise ValueError('Invalid search mode/budget')
    if not math.isfinite(energy_ratio) or energy_ratio < 1:
        raise ValueError('Invalid energy guard')
    current = tuple(initial); baseline = assess(rows, current); best = baseline
    cap = baseline['energy']*energy_ratio
    seen = {current}; evaluations = screens = accepted = 0
    while evaluations < budget:
        moves, checks = ordered_moves(rows, current, mode); screens += checks
        next_a, next_cost = current, best
        for i, action in moves:
            candidate = current[:i]+(action,)+current[i+1:]
            if candidate in seen: continue
            if evaluations >= budget: break
            seen.add(candidate); value = assess(rows, candidate); evaluations += 1
            if value['energy'] <= cap and value['objective'] < next_cost['objective']:
                next_a, next_cost = candidate, value
        if next_a == current: break
        current, best = next_a, next_cost; accepted += 1
    return {'assignment': current, 'predicted': best, 'baseline_predicted': baseline,
            'evaluations': evaluations, 'screen_operations': screens, 'accepted_moves': accepted,
            'budget': budget, 'energy_ratio': energy_ratio, 'mode': mode}


def snapshot_inputs(snapshot):
    """Explicit proposal-information extension; never read neighbor cache truth.

    Geometry/capacities and cache beliefs match existing mask calculations.
    Combining proposals across sources is NEW coordination information. Copy
    deserialization is only a diagnostic adapter, not an online protocol.
    """
    import pickle
    env = pickle.loads(snapshot.payload)
    _, masks = env.get_offloading_obs_and_masks()
    return proposal_inputs(env, snapshot.requests, snapshot.candidates, masks)


def proposal_inputs(env, requests, candidates, masks):
    """Read only predecision inputs; masks are already computed by the policy.

    Each row uses its source UAV's own cache and neighbor cache beliefs. The
    centralized repair receives rows and proposals, never the live environment.
    """
    import config
    from environment import comm_model as comms
    ues = {ue.id: ue for ue in env.ues}
    uavs = {u.id: u for u in env.uavs}
    def tx(size, rate): return size*float(config.BITS_PER_BYTE)/max(float(rate), float(config.EPSILON))
    rows = []
    for request, hard in zip(requests, candidates):
        owner, slot, ue_id, source, deadline = request
        uav, ue = env.uavs[owner], ues[ue_id]; req = ue.current_request
        cycles = float(config.CPU_CYCLES_PER_BYTE[req.req_id])*req.req_size
        rate = comms.calculate_ue_uav_rate(comms.calculate_channel_gain(ue.pos, uav.pos), max(len(uav.current_covered_ues),1))
        upload = tx(req.req_size, rate); options = {}
        for a in hard:
            if not masks[owner,slot,a]: continue
            if a == 1:
                backhaul = tx(req.req_size, uav._uav_mbs_rate)
                options[a] = {'target':'mbs', 'base':upload+backhaul+cycles/config.MBS_COMPUTING_CAPACITY,
                              'work':0., 'energy':backhaul*config.UAV_BACKHAUL_TX_POWER, 'compute_energy':0.}
                continue
            target = uav if a == 0 else uavs[a-2]
            belief = float(uav.cache[req.req_id]) if a == 0 else float(uav._get_service_neighbor_cache_belief(req.req_id,target))
            rate_mbs = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(target.pos,config.MBS_POS))
            fetch = (1-belief)*tx(config.FILE_SIZES[req.req_id],rate_mbs)
            hop = 0. if a == 0 else tx(req.req_size,comms.calculate_uav_uav_rate(comms.calculate_channel_gain(uav.pos,target.pos)))
            capacity = float(config.UAV_COMPUTING_CAPACITY[target.id])
            options[a] = {'target':target.id, 'base':upload+hop+fetch, 'work':cycles/capacity,
                          'energy':fetch*config.UAV_BACKHAUL_RX_POWER+hop*(config.UAV_COMM_TX_POWER+config.UAV_COMM_RX_POWER),
                          'compute_energy':float(config.K_CPU)*cycles*capacity**2}
        rows.append({'source':source,'ue_id':ue_id,'deadline':deadline,'options':options})
    return rows


def compare_snapshot(snapshot, initial, budgets=(32,128)):
    """Choose all actions BEFORE production verification; never feed truth back."""
    import time
    from utils.snapshot_evaluator import evaluate
    started = time.perf_counter(); rows = snapshot_inputs(snapshot)
    preparation = time.perf_counter()-started
    proposals = {}
    for budget in budgets:
        for mode in ('ordinary','deadline'):
            started = time.perf_counter(); result = repair(rows,initial,mode,budget)
            result['search_seconds'] = time.perf_counter()-started
            result['proposal_path_seconds'] = preparation+result['search_seconds']
            proposals[f'{mode}_{budget}'] = result
    def actual(a):
        result = evaluate(snapshot,a)
        return {'misses':sum(not r['success'] for r in result['requests']),
                'assignment_energy':float(result['metrics']['lower_assignment_energy']),
                'latencies':[r['latency'] for r in result['requests']],
                'mbs_count':sum(r['action']==1 for r in result['requests'])}
    baseline = actual(initial)
    for proposal in proposals.values():
        proposal['actual'] = actual(proposal['assignment'])
        proposal['actual_miss_difference'] = proposal['actual']['misses']-baseline['misses']
    return {'information_contract':'centralized_proposal_pool_with_cache_beliefs_v1',
            'candidate_contract':'unchanged_quality_mask', 'initial':list(initial),
            'baseline_actual':baseline,'baseline_predicted':assess(rows,initial),
            'preparation_seconds':preparation,'comparisons':proposals,
            'limitations':['Counterfactual snapshots, not closed-loop DSR.',
                          'Proposal communication is not yet simulated; timing is diagnostic only.',
                          'Energy guard applies to predictions, not guaranteed actual energy.',
                          'Screen operations/time are additional to candidate evaluation budget.']}
