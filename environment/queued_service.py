"""Energy-free fluid queues with fixed association/routing and hard deadlines.

Only the standard library is used. Geometry is continuous and checked over the
whole path; channel rates use midpoint integration between explicit contact
boundaries. Service completions and deadlines are events inside those intervals.
This is a separate protocol: it never imports or mutates legacy config.
"""
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import copy
import math

EPS = 1e-9
STAGES = ('waiting', 'upload', 'transfer', 'compute')


@dataclass(frozen=True)
class Protocol:
    capacity: tuple = (75e9, 60e9, 70e9, 85e9, 50e9, 200e9)
    altitude: float = 100.
    mbs: tuple = (350., 350., 30.)
    separation: float = 200.
    boundary: tuple = (50., 650.)
    coverage: float = 100.
    sensing: float = 460.
    inter_range: float = 460.
    speed: float = 15.
    dt: float = .1
    uplink_bandwidth: float = 40e6
    inter_bandwidth: float = 20e6
    backhaul_bandwidth: float = .75e6
    power: float = .5
    gain: float = 2.2846 * 1.42e-4
    noise: float = 1e-13
    # Fixed-rate injection is solely for deterministic engine verification.
    uplink_bps: float = None
    inter_bps: float = None
    backhaul_bps: float = None

    def __post_init__(self):
        positive = (*self.capacity, self.dt, self.coverage, self.sensing,
                    self.inter_range, self.speed, self.uplink_bandwidth,
                    self.inter_bandwidth, self.backhaul_bandwidth, self.noise)
        if len(self.capacity) < 2 or any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError('finite positive resources required')
        if self.dt > 1 or self.separation < 0 or self.boundary[0] >= self.boundary[1]:
            raise ValueError('invalid geometry/integration step')
        for value in (self.uplink_bps, self.inter_bps, self.backhaul_bps):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('fixed test rates must be positive')


@dataclass
class Task:
    uid: int
    ue: int
    arrival: float
    deadline: float
    bits: float
    cycles: float
    source: int = None
    target: int = None
    stage: str = 'waiting'
    admitted: float = None
    finished: float = None
    expired_stage: str = None
    remaining_upload: float = None
    remaining_transfer: float = 0.
    remaining_cycles: float = None
    served_upload: float = 0.
    served_transfer: float = 0.
    served_cycles: float = 0.
    cancelled_upload: float = 0.
    cancelled_transfer: float = 0.
    cancelled_cycles: float = 0.
    crossed: bool = False
    service_crossed: bool = False
    last_connected: bool = None
    time_in: dict = field(default_factory=lambda: {k: 0. for k in (*STAGES, 'compute_wait', 'disconnected')})

    def __post_init__(self):
        if self.remaining_upload is None: self.remaining_upload = float(self.bits)
        if self.remaining_cycles is None: self.remaining_cycles = float(self.cycles)


def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def safe_path(positions, displacement, p):
    """Analytic minimum separation of affine paths, not endpoint-only checks."""
    if len(displacement) != len(positions): return False
    lo, hi = p.boundary
    for a, d in zip(positions, displacement):
        if len(d) != 2 or not all(math.isfinite(v) for v in (*a, *d)): return False
        if math.hypot(*d) > p.speed+EPS: return False
        if any(x < lo-EPS or x > hi+EPS for x in (*a, a[0]+d[0], a[1]+d[1])): return False
    for i, a in enumerate(positions):
        for j in range(i):
            b = positions[j]; r = (a[0]-b[0], a[1]-b[1])
            v = (displacement[i][0]-displacement[j][0], displacement[i][1]-displacement[j][1])
            vv = v[0]**2+v[1]**2
            s = max(0., min(1., -(r[0]*v[0]+r[1]*v[1])/vv)) if vv else 0.
            if math.hypot(r[0]+s*v[0], r[1]+s*v[1]) < p.separation-EPS: return False
    return True


def contact_roots(r, v, radius):
    a = v[0]**2+v[1]**2
    if a < 1e-20: return []
    b = 2*(r[0]*v[0]+r[1]*v[1]); c = r[0]**2+r[1]**2-radius**2
    disc = b*b-4*a*c
    if disc < 0: return []
    s = math.sqrt(max(0., disc))
    return [x for x in ((-b-s)/(2*a), (-b+s)/(2*a)) if EPS < x < 1-EPS]


class Engine:
    def __init__(self, protocol, positions, ue_positions, trace=False):
        self.p = protocol
        self.positions = [tuple(map(float, x)) for x in positions]
        self.ues = dict(enumerate(ue_positions)) if not isinstance(ue_positions, dict) else dict(ue_positions)
        self.n = len(self.positions)
        if len(protocol.capacity) != self.n+1 or not safe_path(self.positions, [(0., 0.)]*self.n, protocol):
            raise ValueError('initial geometry/resources invalid')
        self.t = 0.; self.tasks = {}; self.finished = {}; self.counts = Counter()
        self.trace = trace; self.events = []; self.served_cycles = [0.]*(self.n+1)
        self.served_bits = Counter(); self.cpu_busy = [0.]*(self.n+1)
        self.airtime = Counter(); self.expiration_stage = Counter()
        self.inventory_time = Counter(); self.nonempty_time = Counter()
        self._intervals = 0; self._reported_success = 0

    def _event(self, event, task, **extra):
        if self.trace: self.events.append(dict(time=self.t, uid=task.uid, event=event, **extra))

    def add(self, tasks):
        tasks = list(tasks); ids = set()
        for task in tasks:
            if (not isinstance(task.uid, int) or task.uid in ids or task.uid in self.tasks or task.uid in self.finished
                    or task.ue not in self.ues or abs(task.arrival-self.t) > EPS
                    or not all(math.isfinite(x) for x in (task.arrival, task.deadline, task.bits, task.cycles))
                    or task.deadline <= task.arrival or task.bits < 0 or task.cycles < 0
                    or task.source is not None or task.target is not None or task.stage != 'waiting'):
                raise ValueError('invalid or duplicate new task')
            ids.add(task.uid)
        for task in tasks:
            task = copy.deepcopy(task); self.tasks[task.uid] = task
            self.counts['generated'] += 1; self._event('arrival', task)

    def admit(self):
        self._settle(); new = []
        for task in sorted(self.tasks.values(), key=lambda z: (z.deadline, z.uid)):
            if task.source is not None: continue
            nearest = min(range(self.n), key=lambda i: (distance(self.ues[task.ue], self.positions[i]), i))
            if distance(self.ues[task.ue], self.positions[nearest]) <= self.p.coverage+EPS:
                task.source = nearest; task.admitted = self.t; task.stage = 'upload'
                self.counts['admitted'] += 1; new.append(task.uid); self._event('admit', task, source=nearest)
        return new

    def legal_targets(self, task):
        return [i for i in range(self.n+1) if i == self.n or i == task.source
                or distance(self.positions[task.source], self.positions[i]) <= self.p.inter_range+EPS]

    def assign(self, assignments):
        # Validate the entire action before modifying any task.
        for uid, node in assignments.items():
            task = self.tasks.get(uid)
            if task is None or task.source is None or task.target is not None or not isinstance(node, int) or node not in self.legal_targets(task):
                raise ValueError('assignment must be legal, new and immutable')
        for uid, node in assignments.items():
            task = self.tasks[uid]; task.target = node
            task.remaining_transfer = task.bits if node != task.source else 0.
            self._event('assign', task, target=node)
        self._settle()

    def _settle(self):
        for uid, task in list(self.tasks.items()):
            if task.target is not None:
                if task.stage == 'upload' and task.remaining_upload <= max(1e-8, task.bits*1e-12):
                    task.remaining_upload = 0.; task.stage = 'transfer' if task.target != task.source else 'compute'
                    task.last_connected = None; self._event('upload_complete', task)
                if task.stage == 'transfer' and task.remaining_transfer <= max(1e-8, task.bits*1e-12):
                    task.remaining_transfer = 0.; task.stage = 'compute'; task.last_connected = None
                    self._event('transfer_complete', task)
                if task.stage == 'compute' and task.remaining_cycles <= max(1e-8, task.cycles*1e-12):
                    task.remaining_cycles = 0.; task.stage = 'completed'; task.finished = self.t
                    self.counts['success'] += 1; self.finished[uid] = self.tasks.pop(uid); self._event('complete', task)
                    continue
            # Completion at the exact deadline wins over expiration.
            if task.deadline <= self.t+EPS:
                task.expired_stage = task.stage; self.expiration_stage[task.stage] += 1
                task.cancelled_upload = task.remaining_upload; task.remaining_upload = 0.
                task.cancelled_transfer = task.remaining_transfer; task.remaining_transfer = 0.
                task.cancelled_cycles = task.remaining_cycles; task.remaining_cycles = 0.
                task.stage = 'expired'; task.finished = self.t; self.counts['expired'] += 1
                self.finished[uid] = self.tasks.pop(uid); self._event('expire', task, phase=task.expired_stage)

    def rate(self, kind, a, b):
        fixed = getattr(self.p, kind+'_bps')
        if fixed is not None: return fixed
        bandwidth = getattr(self.p, kind+'_bandwidth')
        dsq = sum((x-y)**2 for x, y in zip(a, b))
        return bandwidth*math.log2(1+self.p.power*self.p.gain/(max(dsq, 1e-9)*self.p.noise))

    def _rates(self, positions):
        cpu = {}; uploads = {}; transfers = {}; backhaul = {}; connected = {}
        for task in sorted(self.tasks.values(), key=lambda z: (z.deadline, z.uid)):
            if task.stage == 'compute': cpu.setdefault(task.target, task)
            elif task.stage == 'upload':
                ok = distance(self.ues[task.ue], positions[task.source]) <= self.p.coverage+EPS
                connected[task.uid] = ok
                if ok: uploads.setdefault(task.ue, task)
            elif task.stage == 'transfer':
                ok = task.target == self.n or distance(positions[task.source], positions[task.target]) <= self.p.inter_range+EPS
                connected[task.uid] = ok
                if ok:
                    group = backhaul if task.target == self.n else transfers
                    group.setdefault((task.source, task.target), task)
        rates = {}; source_counts = Counter(task.source for task in uploads.values())
        for task in uploads.values():
            a = (*self.ues[task.ue], 0.); b = (*positions[task.source], self.p.altitude)
            rates[task.uid] = self.rate('uplink', a, b)/source_counts[task.source]
        for kind, group in (('inter', transfers), ('backhaul', backhaul)):
            for task in group.values():
                a = (*positions[task.source], self.p.altitude)
                b = self.p.mbs if task.target == self.n else (*positions[task.target], self.p.altitude)
                rates[task.uid] = self.rate(kind, a, b)/len(group)
        for node, task in cpu.items(): rates[task.uid] = self.p.capacity[node]
        return rates, connected, source_counts, transfers, backhaul, cpu

    def _cuts(self, displacement):
        cuts = {0., 1.}; k = 1
        while k*self.p.dt < 1-EPS: cuts.add(k*self.p.dt); k += 1
        pairs = {(task.ue, task.source) for task in self.tasks.values() if task.stage == 'upload'}
        for ue, source in pairs:
            u = self.ues[ue]; a = self.positions[source]
            cuts.update(contact_roots((a[0]-u[0], a[1]-u[1]), displacement[source], self.p.coverage))
        for i, a in enumerate(self.positions):
            for j in range(i):
                b = self.positions[j]
                cuts.update(contact_roots((a[0]-b[0], a[1]-b[1]),
                                         (displacement[i][0]-displacement[j][0], displacement[i][1]-displacement[j][1]), self.p.inter_range))
        return sorted(cuts)

    def advance(self, displacement):
        if not safe_path(self.positions, displacement, self.p): raise ValueError('unsafe full trajectory')
        if any(z.source is not None and z.target is None for z in self.tasks.values()): raise ValueError('newly admitted tasks require allocation')
        displacement = [tuple(x) for x in displacement]; start = self.t; before = self._reported_success
        for task in self.tasks.values():
            if self.t > task.arrival+EPS and not task.crossed:
                task.crossed = True; self.counts['crossed'] += 1
            if task.admitted is not None and self.t > task.admitted+EPS and not task.service_crossed:
                task.service_crossed = True; self.counts['service_crossed'] += 1
        cuts = self._cuts(displacement)
        for left, right in zip(cuts, cuts[1:]):
            if right-left < 1e-12: continue
            midpoint = (left+right)/2
            positions = [(a[0]+midpoint*d[0], a[1]+midpoint*d[1]) for a, d in zip(self.positions, displacement)]
            end = start+right
            while self.t < end-1e-12:
                self._settle()
                rates, connected, source_counts, transfers, backhaul, cpu = self._rates(positions)
                delta = min([end-self.t]+[z.deadline-self.t for z in self.tasks.values()])
                for uid, rate in rates.items():
                    z = self.tasks[uid]
                    remaining = z.remaining_cycles if z.stage == 'compute' else z.remaining_upload if z.stage == 'upload' else z.remaining_transfer
                    delta = min(delta, remaining/rate)
                if delta <= 0: raise RuntimeError('non-progressing service event')
                counts = Counter(z.stage for z in self.tasks.values())
                for stage in STAGES:
                    self.inventory_time[stage] += counts[stage]*delta
                    self.nonempty_time[stage] += bool(counts[stage])*delta
                for z in self.tasks.values():
                    z.time_in[z.stage] += delta
                    if z.stage == 'compute' and z.uid not in rates: z.time_in['compute_wait'] += delta
                    if z.uid in connected:
                        ok = connected[z.uid]
                        if z.last_connected is True and not ok: self.counts['disconnect_events'] += 1
                        z.last_connected = ok
                        if not ok: z.time_in['disconnected'] += delta
                for source in source_counts: self.airtime['uplink_'+str(source)] += delta
                if transfers: self.airtime['inter'] += delta
                if backhaul: self.airtime['backhaul'] += delta
                for node in cpu: self.cpu_busy[node] += delta
                for uid, rate in rates.items():
                    z = self.tasks[uid]; amount = rate*delta
                    if z.stage == 'compute':
                        amount = min(amount, z.remaining_cycles); z.remaining_cycles -= amount
                        z.served_cycles += amount; self.served_cycles[z.target] += amount
                    elif z.stage == 'upload':
                        amount = min(amount, z.remaining_upload); z.remaining_upload -= amount
                        z.served_upload += amount; self.served_bits['upload'] += amount
                    else:
                        amount = min(amount, z.remaining_transfer); z.remaining_transfer -= amount
                        z.served_transfer += amount; self.served_bits['transfer'] += amount
                self.t += delta; self._intervals += 1; self._settle()
            self.t = end
        self.positions = [(a[0]+d[0], a[1]+d[1]) for a, d in zip(self.positions, displacement)]
        self.t = start+1.; self._settle()
        self._reported_success = self.counts['success']
        return dict(success=self.counts['success']-before, reward=(self.counts['success']-before)/100., **self.snapshot())

    def snapshot(self):
        counts = Counter(z.stage for z in self.tasks.values()); work = Counter()
        for z in self.tasks.values():
            if z.stage == 'compute': work['compute_cycles'] += z.remaining_cycles
            elif z.stage == 'upload': work['upload_bits'] += z.remaining_upload
            elif z.stage == 'transfer': work['transfer_bits'] += z.remaining_transfer
            else: work['waiting_bits'] += z.remaining_upload
        return dict(time=self.t, active=len(self.tasks), stages={s: counts[s] for s in STAGES}, work=dict(work),
                    generated=self.counts['generated'], completed=self.counts['success'], expired=self.counts['expired'], admitted=self.counts['admitted'])

    def observe(self):
        visible = {u for u, xy in self.ues.items() if any(distance(xy, a) <= self.p.sensing+EPS for a in self.positions)}
        known = [z for z in self.tasks.values() if z.source is not None or z.ue in visible]
        known_ues = visible | {z.ue for z in known}
        return dict(time=self.t, positions=list(self.positions), ues={u: self.ues[u] for u in sorted(known_ues)},
                    tasks=[asdict(z) for z in sorted(known, key=lambda z: (z.deadline, z.uid))], protocol=self.p)

    @classmethod
    def from_observation(cls, observation):
        e = cls(observation['protocol'], observation['positions'], observation['ues'])
        e.t = observation['time']; e.tasks = {x['uid']: Task(**copy.deepcopy(x)) for x in observation['tasks']}
        e.counts['generated'] = len(e.tasks); e.counts['admitted'] = sum(z.source is not None for z in e.tasks.values())
        e.counts['crossed'] = sum(z.crossed for z in e.tasks.values())
        # Forecast ledgers start at the observed residual state, not original work.
        e._forecast = True
        return e

    def audit(self):
        assert self.counts['generated'] == len(self.tasks)+len(self.finished)
        assert len(self.finished) == self.counts['success']+self.counts['expired']
        assert not (self.tasks.keys() & self.finished.keys())
        assert self.counts['admitted'] == sum(z.source is not None for z in (*self.tasks.values(), *self.finished.values()))
        for z in (*self.tasks.values(), *self.finished.values()):
            for total, parts in ((z.bits, (z.remaining_upload, z.served_upload, z.cancelled_upload)),
                                 (z.cycles, (z.remaining_cycles, z.served_cycles, z.cancelled_cycles)),
                                 (z.bits if z.target is not None and z.target != z.source else 0.,
                                  (z.remaining_transfer, z.served_transfer, z.cancelled_transfer))):
                assert all(x >= -1e-6 for x in parts)
                assert math.isclose(sum(parts), total, rel_tol=1e-9, abs_tol=1e-5), (z.uid, total, parts)
            if z.stage == 'completed': assert z.finished <= z.deadline+EPS
            if z.stage == 'expired': assert abs(z.finished-z.deadline) <= 1e-7
        if not getattr(self, '_forecast', False):
            for node, served in enumerate(self.served_cycles):
                assert served <= self.p.capacity[node]*self.cpu_busy[node]*(1+1e-9)+1e-5
        assert all(v <= self.t+1e-7 for v in self.airtime.values())
        assert all(v <= self.t+1e-7 for v in self.cpu_busy)
        return dict(status='PASS', tasks=self.counts['generated'], resource_intervals=self._intervals,
                    unique_counting=True, work_conservation=True, path_safety=True)

    def summary(self):
        total = self.counts['generated']; success = self.counts['success']; admitted = self.counts['admitted']
        tasks = list(self.finished.values()); completed = [z for z in tasks if z.stage == 'completed']
        def mean(xs): return sum(xs)/len(xs) if xs else 0.
        assigned = Counter(z.target for z in tasks if z.target is not None)
        result = dict(counts=dict(self.counts), dsr=success/max(total, 1), admission=admitted/max(total, 1),
                      conditional=success/max(admitted, 1), expiration_stage=dict(self.expiration_stage),
                      mean_completion_latency=mean([z.finished-z.arrival for z in completed]),
                      mean_stage_time={s: mean([z.time_in[s] for z in tasks]) for s in (*STAGES, 'compute_wait', 'disconnected')},
                      nonempty_fraction={s: self.nonempty_time[s]/max(self.t, EPS) for s in STAGES},
                      mean_inventory={s: self.inventory_time[s]/max(self.t, EPS) for s in STAGES},
                      crossed_fraction=self.counts['crossed']/max(total, 1),
                      service_crossed_fraction=self.counts['service_crossed']/max(admitted, 1),
                      mbs_fraction=assigned[self.n]/max(admitted, 1), assignment_by_node=dict(assigned),
                      cpu_utilization=[x/max(self.t, EPS) for x in self.cpu_busy],
                      airtime_utilization={k: v/max(self.t, EPS) for k, v in self.airtime.items()}, elapsed=self.t)
        return result
