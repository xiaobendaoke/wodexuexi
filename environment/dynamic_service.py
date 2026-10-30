"""Independent, energy-free service protocol; never changes legacy config."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import config
from environment import comm_model as radio
from environment.env import Env


def project_positions(positions, actions):
    positions = np.asarray(positions, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (5, 2) or not np.isfinite(actions).all():
        raise ValueError('five finite 2-D actions required')
    class Motion(Env):
        def __init__(self):
            self._uavs = []
            for p in positions:
                u = SimpleNamespace(pos=np.r_[p, 100.], collision_violation=False, boundary_violation=False)
                u.update_position = lambda xy, u=u: setattr(u, 'pos', np.r_[xy, 100.])
                self._uavs.append(u)
    motion = Motion()
    motion._apply_actions_to_env(np.clip(actions, -1, 1))
    return np.array([u.pos[:2] for u in motion._uavs], dtype=np.float64)


@dataclass
class Batch:
    ids: np.ndarray
    ue: np.ndarray
    source: np.ndarray
    size: np.ndarray
    cycles: np.ndarray
    deadline: np.ndarray
    tx: np.ndarray
    legal: np.ndarray
    capacity: np.ndarray
    node_positions: np.ndarray

    def __len__(self):
        return len(self.ids)


def evaluate(batch, assignment):
    assignment = np.asarray(assignment)
    if (assignment.shape != (len(batch),) or not np.isfinite(assignment).all()
            or not np.equal(assignment, np.floor(assignment)).all()
            or np.any((assignment < 0) | (assignment >= 6))):
        raise ValueError('complete integer assignment required')
    assignment = assignment.astype(np.int64)
    if not batch.legal[np.arange(len(batch)), assignment].all():
        raise ValueError('assignment contains unreachable node')
    counts = np.bincount(assignment, minlength=6)
    latency = batch.tx[np.arange(len(batch)), assignment] + batch.cycles * counts[assignment] / batch.capacity[assignment]
    success = latency <= batch.deadline
    return dict(success=int(success.sum()), satisfied=success, latencies=latency,
                counts=counts, mbs=int(counts[5]), assignment=assignment)


def prefix_batch(batch, n):
    return Batch(*(getattr(batch, field)[:n] for field in
                   ('ids', 'ue', 'source', 'size', 'cycles', 'deadline', 'tx', 'legal')),
                 batch.capacity, batch.node_positions)


class DynamicServiceEnv:
    def __init__(self, tape, compute_scale=1.):
        if (config.NUM_UAVS != 5 or config.NUM_UES != 100 or config.AREA_WIDTH != 700
                or config.AREA_HEIGHT != 700 or config.UAV_SPEED != 15
                or config.MIN_UAV_SEPARATION != 200 or config.TIME_SLOT_DURATION != 1
                or config.UAV_COVERAGE_RADIUS != 100):
            raise ValueError('legacy geometry config differs from the registered service-only protocol')
        if compute_scale not in (.6, 1., 1.4):
            raise ValueError('unregistered compute scale')
        self.tape, self.compute_scale = tape, compute_scale
        self.ue_positions = np.array(tape.ue_positions)
        # Same safe starting geometry across methods and cells.
        self.positions = np.array([[100., 100.], [600., 100.], [350., 350.], [100., 600.], [600., 600.]])
        self.capacity = np.r_[config.UAV_COMPUTING_CAPACITY, config.MBS_COMPUTING_CAPACITY] * compute_scale
        self.t = 0
        self.feedback = np.zeros((5, 3), np.float32)
        self.generated_per_ue = np.zeros(100, np.int64)
        self.success_per_ue = np.zeros(100, np.int64)

    @property
    def current_requests(self):
        return self.tape.slot(self.t) if self.t < 1000 else ()

    def batch(self, positions=None, requests=None):
        positions = self.positions if positions is None else np.asarray(positions)
        requests = self.current_requests if requests is None else requests
        dist = np.linalg.norm(self.ue_positions[:, None] - positions[None], axis=-1)
        nearest = dist.argmin(axis=1)
        admitted = sorted((r for r in requests if dist[r.ue, nearest[r.ue]] <= 100), key=lambda r: (r.deadline, r.uid))
        sources = np.array([nearest[r.ue] for r in admitted], dtype=np.int64)
        n_assoc = np.bincount(sources, minlength=5)
        node_positions = np.vstack([np.c_[positions, np.full(5, 100.)], config.MBS_POS])
        legal, tx = np.zeros((len(admitted), 6), bool), np.zeros((len(admitted), 6))
        for i, r in enumerate(admitted):
            source = sources[i]
            gain = radio.calculate_channel_gain(np.r_[self.ue_positions[r.ue], 0], node_positions[source])
            upload = 8 * r.size / radio.calculate_ue_uav_rate(gain, int(n_assoc[source]))
            for j in range(6):
                legal[i, j] = j == 5 or j == source or np.linalg.norm(positions[source]-positions[j]) <= 460
                hop = 0.
                if j != source:
                    gain = radio.calculate_channel_gain(node_positions[source], node_positions[j])
                    rate = radio.calculate_uav_mbs_rate(gain) if j == 5 else radio.calculate_uav_uav_rate(gain)
                    hop = 8 * r.size / rate
                tx[i, j] = upload + hop
        return Batch(np.array([r.uid for r in admitted], dtype=np.int64),
                     np.array([r.ue for r in admitted], dtype=np.int64), sources,
                     np.array([r.size for r in admitted], dtype=np.float64),
                     np.array([r.cycles for r in admitted], dtype=np.float64),
                     np.array([r.deadline for r in admitted]), tx, legal, self.capacity.copy(), node_positions)

    def observe(self):
        # Every policy receives these same allowed current records. No center,
        # kind, phase, future requests, or unobserved UE request features.
        request_map = {r.ue: r for r in self.current_requests}
        visible = np.linalg.norm(self.ue_positions[:, None]-self.positions[None], axis=-1) <= 460
        ue = np.zeros((5, 100, 6), np.float32)
        for i in range(5):
            for u in np.flatnonzero(visible[:, i]):
                r = request_map.get(int(u))
                ue[i, u, :2] = (self.ue_positions[u]-self.positions[i]) / 700
                if r:
                    ue[i, u, 2:] = [1, r.size/5e6, r.cycles/14e9, r.deadline]
        nodes = np.c_[self.positions/700, self.capacity[:5]/2e11].astype(np.float32)
        return dict(ue=ue, visible=visible.T.copy(), nodes=nodes, feedback=self.feedback.copy())

    def step(self, actions, assignment, decision_timeout=False):
        if self.t >= 1000:
            raise RuntimeError('episode finished')
        batch = self.batch()
        if decision_timeout:
            actions, assignment = np.zeros((5, 2)), batch.source
        result = evaluate(batch, assignment)
        generated = self.current_requests
        for r in generated:
            self.generated_per_ue[r.ue] += 1
        np.add.at(self.success_per_ue, batch.ue[result['satisfied']], 1)
        self.feedback = np.zeros((5, 3), np.float32)
        for i in range(5):
            source = batch.source == i
            self.feedback[i] = [source.sum()/100, result['satisfied'][source].sum()/max(source.sum(), 1), result['counts'][i]/100]
        # Current service is irrevocably evaluated BEFORE current movement.
        self.positions = project_positions(self.positions, actions)
        self.t += 1
        rates = self.success_per_ue / np.maximum(self.generated_per_ue, 1)
        nonempty = self.generated_per_ue > 0
        fair = float(rates[nonempty].sum()**2 / max(nonempty.sum() * (rates[nonempty]**2).sum(), 1e-30))
        return dict(generated=len(generated), admitted=len(batch), success=result['success'],
                    uncovered=len(generated)-len(batch), deadline_misses=len(batch)-result['success'],
                    decision_timeout=int(decision_timeout), mbs=result['mbs'], reward=result['success']/100,
                    fairness=fair, counts=result['counts'].tolist(), assignment=result['assignment'].tolist())
