"""Same-information, queue-aware nonlearning controls; no private tape access."""
import math
from collections import defaultdict

from environment.queued_service import Engine, distance, safe_path

METHODS = ('history_queue', 'history_balance', 'rolling_joint')


def assignments(obs, mode='queue'):
    p = obs['protocol']; positions = obs['positions']; n = len(positions); now = obs['time']
    tasks = obs['tasks']; plan = {}; work = [[] for _ in p.capacity]
    inter_bits = sum(x['remaining_transfer'] for x in tasks if x['stage'] == 'transfer' and x['target'] != n)
    back_bits = sum(x['remaining_transfer'] for x in tasks if x['stage'] == 'transfer' and x['target'] == n)
    source_ues = defaultdict(set)
    for x in tasks:
        if x['source'] is not None and x['stage'] == 'upload': source_ues[x['source']].add(x['ue'])
        if x['target'] is not None: work[x['target']].append((x['deadline'], x['remaining_cycles']))
    # This public clone is a rate calculator only, without access to future input.
    radio = Engine(p, positions, obs['ues'])
    for x in tasks:
        if x['source'] is None or x['target'] is not None: continue
        source = x['source']; ue = obs['ues'][x['ue']]
        upload_rate = radio.rate('uplink', (*ue, 0.), (*positions[source], p.altitude))
        upload = x['remaining_upload']*max(len(source_ues[source]), 1)/upload_rate
        choices = []
        for node in range(n+1):
            if node != n and node != source and distance(positions[source], positions[node]) > p.inter_range: continue
            transfer = 0.
            if node != source:
                target = p.mbs if node == n else (*positions[node], p.altitude)
                kind = 'backhaul' if node == n else 'inter'
                rate = radio.rate(kind, (*positions[source], p.altitude), target)
                backlog = back_bits if node == n else inter_bits
                transfer = (backlog+x['bits'])/rate
            load = sum(c for d, c in work[node] if mode == 'balance' or d <= x['deadline'])
            latency = upload+transfer+(load+x['remaining_cycles'])/p.capacity[node]
            # EDF-aware completion estimate versus all-work load balancing.
            score = ((latency > x['deadline']-now), latency, node) if mode == 'queue' else (latency, node)
            choices.append((score, node))
        node = min(choices)[1]; plan[x['uid']] = node
        work[node].append((x['deadline'], x['remaining_cycles']))
        if node != source:
            if node == n: back_bits += x['bits']
            else: inter_bits += x['bits']
    return plan


class Controller:
    """Fixed finite search. Runtime is measured, not hidden as real-time service."""
    def __init__(self, method, forecast_slots=2, max_candidates=4):
        if method not in METHODS: raise ValueError('unknown queue controller')
        self.method = method; self.weights = {}; self.seen = set()
        self.forecast_slots = forecast_slots; self.max_candidates = max_candidates
        self.forecast_calls = 0

    def _update_history(self, obs):
        for u in self.weights: self.weights[u] *= math.exp(-1/20)
        for x in obs['tasks']:
            if x['uid'] not in self.seen:
                self.weights[x['ue']] = self.weights.get(x['ue'], 0.)+1.
                self.seen.add(x['uid'])

    def _history_score(self, obs, displacement):
        after = [(a[0]+d[0], a[1]+d[1]) for a, d in zip(obs['positions'], displacement)]
        return sum(self.weights.get(u, 0.)*math.exp(-min(distance(xy, a)**2 for a in after)/(2*100**2))
                   for u, xy in obs['ues'].items())

    def candidates(self, obs):
        positions = obs['positions']; p = obs['protocol']; n = len(positions)
        grouped = defaultdict(list)
        for u, xy in obs['ues'].items():
            w = self.weights.get(u, 0.)
            if w > 1e-8:
                owner = min(range(n), key=lambda i: distance(xy, positions[i]))
                grouped[owner].append((xy, w))
        desired = []
        for i, a in enumerate(positions):
            xs = grouped[i]; total = sum(w for _, w in xs)
            target = tuple(sum(xy[k]*w for xy, w in xs)/total for k in range(2)) if total else a
            delta = (target[0]-a[0], target[1]-a[1]); norm = math.hypot(*delta)
            scale = min(1., p.speed/max(norm, 1e-12)); desired.append((delta[0]*scale, delta[1]*scale))
        zero = [(0., 0.)]*n; plans = [zero]
        for fraction in (1., .5):
            joint = [(d[0]*fraction, d[1]*fraction) for d in desired]
            if safe_path(positions, joint, p): plans.append(joint)
        for i in range(n):
            plan = list(zero); plan[i] = desired[i]
            if safe_path(positions, plan, p): plans.append(plan)
        unique = {tuple(plan): plan for plan in plans}
        return sorted(unique.values(), key=lambda d: -self._history_score(obs, d))

    def act(self, obs):
        self._update_history(obs); candidates = self.candidates(obs)
        mode = 'balance' if self.method == 'history_balance' else 'queue'
        basic = assignments(obs, mode)
        if self.method != 'rolling_joint': return basic, candidates[0]
        zero = [(0., 0.)]*len(obs['positions'])
        selected = candidates[:max(self.max_candidates-1, 0)]
        if zero not in selected: selected.append(zero)
        plans = [basic, assignments(obs, 'balance')]
        if plans[0] == plans[1]: plans.pop()
        best = None
        for plan in plans:
            for motion in selected:
                forecast = Engine.from_observation(obs); forecast.assign(plan)
                forecast.advance(motion)
                for _ in range(self.forecast_slots-1):
                    forecast.admit()
                    later = forecast.observe(); forecast.assign(assignments(later, 'queue'))
                    # Hold after the proposed first movement; no future arrivals.
                    forecast.advance(zero)
                pending = sum(z.remaining_cycles/max(forecast.p.capacity[z.target if z.target is not None else forecast.n], 1.)
                              for z in forecast.tasks.values())
                score = (forecast.counts['success'], -forecast.counts['expired'], -pending, self._history_score(obs, motion))
                self.forecast_calls += 1
                if best is None or score > best[0]: best = (score, plan, motion)
        return best[1], best[2]
