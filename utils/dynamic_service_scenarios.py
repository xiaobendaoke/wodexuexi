"""Policy-independent service tapes. No NumPy/Torch or mutable global RNG."""
from dataclasses import dataclass
from hashlib import sha256
import math
import random

NAMESPACE = 'chapter5_dynamic_service_20261028/v1'


def seed_for(split, index):
    return int.from_bytes(sha256(f'{NAMESPACE}/{split}/{index}'.encode()).digest()[:4], 'big')


@dataclass(frozen=True)
class Request:
    uid: int
    ue: int
    size: int
    cycles: int
    deadline: float


class Scenario:
    """Private simulation input; policies receive only the current observations.

    All four cells have exactly the same 50,000 task attribute tuples. A/C
    hold demand at the initial point; B/D move demand (not the UE geometry).
    """
    horizon = 1000
    num_ues = 100

    def __init__(self, seed, kind):
        if kind not in ('A', 'B', 'C', 'D'):
            raise ValueError('kind must be A/B/C/D')
        self.seed, self.kind = int(seed), kind
        geo = random.Random(seed_for(f'geometry/{seed}', 0))
        self.ue_positions = tuple((geo.uniform(0, 700), geo.uniform(0, 700)) for _ in range(100))
        rng = random.Random(seed_for(f'attributes/{seed}', 0))
        densities = [rng.randrange(1200, 2800) for _ in range(25)]
        self.tasks = tuple(self._task(rng, densities) for _ in range(50000))
        self.phase = geo.uniform(0, 1400)
        self.direction = geo.choice((-1, 1))
        burst = random.Random(seed_for(f'bursts/{seed}', 0))
        counts, remaining = [], 500
        while remaining:
            length = min(burst.randint(20, 80), remaining)
            pair = (25, 75) if burst.random() < .5 else (75, 25)
            counts.extend([pair[0]] * length + [pair[1]] * length)
            remaining -= length
        self.counts = tuple(counts if kind in 'CD' else [50] * 1000)
        offsets = [0]
        for count in self.counts:
            offsets.append(offsets[-1] + count)
        self.offsets = tuple(offsets)
        self.events = tuple(i for i in range(1, 1000) if self.counts[i] != self.counts[i - 1])

    @staticmethod
    def _task(rng, densities):
        size = rng.randrange(1000000, 5000001)
        return size, size * rng.choice(densities), rng.uniform(.4, .95)

    def center(self, t):
        distance = (self.phase + (2 * t * self.direction if self.kind in 'BD' else 0)) % 1400
        edge, fraction = int(distance // 350), distance % 350 / 350
        corners = ((175, 175), (525, 175), (525, 525), (175, 525))
        a, b = corners[edge], corners[(edge + 1) % 4]
        return tuple(a[k] + fraction * (b[k] - a[k]) for k in range(2))

    def slot(self, t):
        if not 0 <= t < self.horizon:
            raise IndexError(t)
        rng = random.Random(seed_for(f'activation/{self.seed}', t))
        center = self.center(t)
        weights = [.2 + .8 * math.exp(-sum((p[k] - center[k]) ** 2 for k in range(2)) / 20000)
                   for p in self.ue_positions]
        # Exponential-race sampling is weighted sampling without replacement.
        keys = [-math.log(max(rng.random(), 1e-300)) / w for w in weights]
        active = sorted(range(100), key=lambda i: (keys[i], i))[:self.counts[t]]
        start = self.offsets[t]
        return tuple(Request(start + i, ue, *self.tasks[start + i]) for i, ue in enumerate(active))

    def fingerprint(self):
        return sha256(repr((self.seed, self.kind, self.ue_positions, self.tasks, self.counts)).encode()).hexdigest()
