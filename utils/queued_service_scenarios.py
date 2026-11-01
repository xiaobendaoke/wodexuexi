"""Exogenous calibration arrivals, separately namespaced from all old trials."""
from hashlib import sha256
import json
import math
import random

from environment.queued_service import Task
from utils.dynamic_service_scenarios import Scenario

NAMESPACE = 'chapter5/queued_protocol_20261031/development-v1'
PROFILES = ('light', 'steady', 'burst', 'overload')


def seed_for(label, index):
    return int.from_bytes(sha256(f'{NAMESPACE}/{label}/{index}'.encode()).digest()[:8], 'big')


class QueueScenario:
    def __init__(self, workload, profile, horizon=200, deadlines=(2., 5.)):
        if profile not in PROFILES or horizon < 120 or not (0 < deadlines[0] <= deadlines[1]):
            raise ValueError('invalid calibration scenario')
        self.workload = workload; self.profile = profile; self.horizon = horizon; self.deadlines = tuple(deadlines)
        self.seed = seed_for('workload', workload)
        old = Scenario(self.seed, 'D')
        self.ue_positions = old.ue_positions
        self.counts = tuple(10 if profile == 'light' else 50 if profile == 'steady' else 100 if profile == 'overload'
                            else (75 if 50 <= t < 100 else 20) for t in range(horizon))
        self.requests = []; uid = 0
        for t, count in enumerate(self.counts):
            # Paired exogenous UE priorities and attributes do not depend on control.
            rng = random.Random(seed_for(f'activation/{workload}', t)); center = old.center(t)
            weights = [.2+.8*math.exp(-sum((p[k]-center[k])**2 for k in range(2))/20000) for p in self.ue_positions]
            keys = [-math.log(max(rng.random(), 1e-300))/w for w in weights]
            active = sorted(range(100), key=lambda i: (keys[i], i))[:count]
            attrs = random.Random(seed_for(f'task/{workload}', t)); rows = []
            for u in active:
                size = attrs.randrange(1000000, 5000001); density = attrs.randrange(1200, 2800)
                deadline = t+attrs.uniform(*deadlines)
                rows.append(Task(uid, u, float(t), deadline, 8.*size, float(size*density))); uid += 1
            self.requests.append(tuple(rows))
        self.fingerprint = sha256(json.dumps(dict(namespace=NAMESPACE, workload=workload,
            ue=self.ue_positions, tasks=[[(x.uid, x.ue, x.arrival, x.deadline, x.bits, x.cycles) for x in r] for r in self.requests]),
            separators=(',', ':')).encode()).hexdigest()

    def slot(self, t):
        return self.requests[t] if t < self.horizon else ()
