"""Literature-scaled, marked Poisson batches for event-level image processing.

This is a new synthetic protocol, not an empirical application trace. Keep the
previous one-arrival-per-UE scenario unchanged for exact scientific replay.
"""
from hashlib import sha256
import bisect
import json
import math
import random

from environment.queued_service import Task
from utils.dynamic_service_scenarios import Scenario

NAMESPACE = 'chapter5/business_calibration_20261031/marked_poisson_v1'
SLA = {'tight': (1., 2.), 'normal': (2., 3.), 'relaxed': (3., 5.)}
PROFILES = ('light', 'steady', 'burst', 'overload')
BASIS = dict(input_bytes=(1500000, 3000000), cycles_per_bit=(500, 1000),
             mean_task_cycles=13.5e9, potential_rate=150., macro_seconds=1.,
             arrival_model='Slotted Poisson batches, spatial draws with replacement')


def seed_for(label, index):
    return int.from_bytes(sha256(f'{NAMESPACE}/{label}/{index}'.encode()).digest()[:8], 'big')


def poisson(mean, rng):
    if not math.isfinite(mean) or mean < 0 or mean > 700:
        raise ValueError('Poisson mean must be finite, between 0 and 700')
    if mean == 0: return 0
    threshold = math.exp(-mean); product = 1.; count = 0
    while product > threshold:
        product *= rng.random(); count += 1
    return count-1


class BusinessScenario:
    def __init__(self, workload, profile, sla='normal', scale=1., horizon=200):
        if profile not in PROFILES or sla not in SLA or not math.isfinite(scale) or not 0 < scale <= 1.25 or horizon < 120:
            raise ValueError('invalid business scenario')
        self.workload = workload; self.profile = profile; self.sla = sla; self.scale = scale; self.horizon = horizon
        self.seed = seed_for('workload', workload)
        geometry = Scenario(seed_for('geometry', workload), 'D'); self.ue_positions = geometry.ue_positions
        self.rates = tuple(scale*(10 if profile == 'light' else 40 if profile == 'steady' else 120 if profile == 'overload'
                                  else 100 if 50 <= t < 100 else 20) for t in range(horizon))
        self.requests = []; uid = 0; duplicated_slots = 0; potential = 0
        arrival_digest = sha256(); full_digest = sha256()
        header = json.dumps(dict(namespace=NAMESPACE, workload=workload, ue=self.ue_positions), separators=(',', ':')).encode()
        arrival_digest.update(header); full_digest.update(header)
        for t, rate in enumerate(self.rates):
            count_rng = random.Random(seed_for(f'potential_count/{workload}', t))
            thin_rng = random.Random(seed_for(f'threshold/{workload}', t))
            ue_rng = random.Random(seed_for(f'ue/{workload}', t))
            attrs = random.Random(seed_for(f'attributes/{workload}', t))
            count = poisson(BASIS['potential_rate'], count_rng); potential += count
            center = geometry.center(t); cumulative = []; total = 0.
            for xy in self.ue_positions:
                total += .2+.8*math.exp(-sum((xy[k]-center[k])**2 for k in range(2))/20000)
                cumulative.append(total)
            rows = []; lo, hi = SLA[sla]
            for _ in range(count):
                gate = thin_rng.random(); ue = bisect.bisect_left(cumulative, ue_rng.random()*total)
                size = attrs.randint(*BASIS['input_bytes']); density = attrs.randint(*BASIS['cycles_per_bit']); mark = attrs.random()
                task_id = uid; uid += 1
                if gate >= rate/BASIS['potential_rate']: continue
                task = Task(task_id, ue, float(t), t+lo+(hi-lo)*mark, 8.*size, float(8*size*density)); rows.append(task)
                fields = [task.uid, task.ue, task.arrival, task.bits, task.cycles]
                arrival_digest.update(json.dumps(fields, separators=(',', ':')).encode())
                full_digest.update(json.dumps(fields+[task.deadline], separators=(',', ':')).encode())
            duplicated_slots += len({x.ue for x in rows}) < len(rows)
            self.requests.append(tuple(rows))
        self.counts = tuple(len(r) for r in self.requests)
        self.arrival_fingerprint = arrival_digest.hexdigest(); self.fingerprint = full_digest.hexdigest()
        self.statistics = dict(generated=sum(self.counts), expected=sum(self.rates), potential_generated=potential,
            duplicate_ue_slots=duplicated_slots, maximum_per_slot=max(self.counts), mean_rate=sum(self.counts)/horizon,
            namespace=NAMESPACE, input_bytes=BASIS['input_bytes'], density_cycles_per_bit=BASIS['cycles_per_bit'])

    def slot(self, t):
        if t < 0: raise IndexError(t)
        return self.requests[t] if t < self.horizon else ()
