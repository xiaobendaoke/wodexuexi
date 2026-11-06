"""Safe motion proposals from public, currently unadmitted requests."""
import math

from environment.queued_service import Engine, distance, safe_path

REVISION = 'queued_waiting_rescue_20261104'
MAX_ADDITIONS = 4


def service_bounds(obs):
    """Optimistic necessary service time, excluding queues and relay delay."""
    protocol = obs['protocol']
    radio = Engine(protocol, obs['positions'], obs['ues'])
    best_uplink = radio.rate('uplink', (0., 0., 0.), (0., 0., protocol.altitude))
    fastest_cpu = max(protocol.capacity)
    return {
        task['uid']: task['remaining_upload']/best_uplink + task['remaining_cycles']/fastest_cpu
        for task in obs['tasks'] if task['source'] is None
    }


def rescue_value(obs, displacement, bounds=None):
    """Urgency weighted reachable demand after a proposed first macro step.

    Admission occurs at integer boundaries. Subsequent straight flight and
    best-case service are optimistic filters, not a completion guarantee.
    """
    bounds = service_bounds(obs) if bounds is None else bounds
    positions = [(a[0]+d[0], a[1]+d[1]) for a, d in zip(obs['positions'], displacement)]
    protocol = obs['protocol']
    score = 0.
    for task in sorted(obs['tasks'], key=lambda item: item['uid']):
        if task['source'] is not None:
            continue
        remaining = task['deadline']-obs['time']
        gap = max(0., min(distance(obs['ues'][task['ue']], point) for point in positions)-protocol.coverage)
        extra_steps = max(0, math.ceil(gap/protocol.speed-1e-9))
        admission_step = 1+extra_steps
        if admission_step+bounds[task['uid']] <= remaining+1e-9:
            score += 1/(admission_step*max(remaining, .1))
    return score


def rescue_candidates(obs, original, max_additions=MAX_ADDITIONS):
    """Nest the old bank and append up to four distinct safe rescue motions."""
    if not isinstance(max_additions, int) or not 0 <= max_additions <= MAX_ADDITIONS:
        raise ValueError('rescue budget must be an integer between zero and four')
    protocol = obs['protocol']
    positions = obs['positions']
    bank = [[tuple(delta) for delta in motion] for motion in original]
    if not bank or not all(safe_path(positions, action, protocol) for action in bank):
        raise ValueError('a nonempty safe original motion bank is required')
    bounds = service_bounds(obs)
    waiting = sorted((task for task in obs['tasks'] if task['source'] is None),
                     key=lambda task: task['uid'])
    zero = [(0., 0.)]*len(positions)
    baseline = rescue_value(obs, zero, bounds)
    proposals = []
    joint = list(zero)
    for node, position in enumerate(positions):
        best = None
        for ue in sorted({task['ue'] for task in waiting}):
            point = obs['ues'][ue]
            gap = max(0., distance(position, point)-protocol.coverage)
            earliest = max(1, math.ceil(gap/protocol.speed-1e-9))
            feasible = any(task['ue'] == ue and earliest+bounds[task['uid']] <= task['deadline']-obs['time']+1e-9
                           for task in waiting)
            if not feasible:
                continue
            delta = (point[0]-position[0], point[1]-position[1])
            scale = min(1., protocol.speed/max(math.hypot(*delta), 1e-12))
            for fraction in (1., .5, .25):
                action = list(zero)
                action[node] = tuple(value*scale*fraction for value in delta)
                if not safe_path(positions, action, protocol):
                    continue
                value = rescue_value(obs, action, bounds)
                key = (value, -sum(math.hypot(*move) for move in action), tuple(action))
                if value > baseline+1e-12 and (best is None or key > best[0]):
                    best = (key, action)
        if best is not None:
            proposals.append(best[1])
            joint[node] = best[1][node]
    for fraction in (1., .5, .25):
        action = [(dx*fraction, dy*fraction) for dx, dy in joint]
        if safe_path(positions, action, protocol) and rescue_value(obs, action, bounds) > baseline+1e-12:
            proposals.append(action)
    proposals.sort(key=lambda action: (-rescue_value(obs, action, bounds),
                                      sum(math.hypot(*delta) for delta in action), tuple(action)))
    for action in proposals:
        if len(bank)-len(original) >= max_additions:
            break
        if action not in bank:
            bank.append(action)
    return bank, dict(added=len(bank)-len(original), waiting=len(waiting),
                      baseline_rescue_score=baseline,
                      best_rescue_score=max(rescue_value(obs, action, bounds) for action in bank))
