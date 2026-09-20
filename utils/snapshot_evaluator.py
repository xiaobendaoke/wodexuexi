"""Pure fixed-assignment evaluation using the production Env.step physics.

Snapshots are trusted in-process objects, never unpickled from external files.
No Chapter 4 optimization objective is selected by this module.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import pickle
import math

import numpy as np
import config
from utils.canonical_experiment import config_snapshot, digest


@dataclass(frozen=True)
class AssignmentSnapshot:
    payload: bytes
    config_hash: str
    requests: tuple
    candidates: tuple
    motion: tuple

    @property
    def snapshot_hash(self):
        return hashlib.sha256(self.payload + digest({'config': self.config_hash, 'motion': self.motion}).encode()).hexdigest()


def capture(env, motion=None):
    """Preserve associations, channels, caches, requests and private RNG state."""
    frozen = deepcopy(env)
    for uav in frozen.uavs:
        uav.cache_snapshot = uav.cache.copy()
    motion = np.zeros((config.NUM_UAVS, 2)) if motion is None else np.asarray(motion, dtype=float)
    if motion.shape != (config.NUM_UAVS, 2) or not np.isfinite(motion).all():
        raise ValueError('Invalid frozen trajectory action')
    frozen.get_offloading_obs_and_masks()
    hard = frozen.last_offload_mask_components['hard']
    requests, candidates = [], []
    for row, uav in enumerate(frozen.uavs):
        service = sorted((ue for ue in uav.current_covered_ues if ue.current_request.is_service),
                         key=lambda ue: float(np.linalg.norm(uav.pos[:2] - ue.pos[:2])))
        for slot, ue in enumerate(service):
            requests.append((row, slot, ue.id, uav.id, float(ue.current_request.deadline)))
            candidates.append(tuple(int(a) for a in np.flatnonzero(hard[row, slot])))
    ordered = sorted(zip(requests, candidates), key=lambda item: item[0][2])
    return AssignmentSnapshot(pickle.dumps(frozen, protocol=5), digest(config_snapshot()),
                              tuple(r for r, _ in ordered), tuple(c for _, c in ordered),
                              tuple(tuple(float(v) for v in row) for row in motion))


def evaluate(snapshot, assignment):
    """Execute exactly one frozen slot on a fresh private environment copy."""
    if digest(config_snapshot()) != snapshot.config_hash:
        raise ValueError('Snapshot configuration changed')
    assignment = tuple(assignment)
    if len(assignment) != len(snapshot.requests):
        raise ValueError('Assignment must cover every admitted service request')
    for action, candidates in zip(assignment, snapshot.candidates):
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or action not in candidates:
            raise ValueError('Assignment contains a non-hard-feasible action')
    env = pickle.loads(snapshot.payload)
    request_energy = {}
    # Instrument production calls on this private copy; do not replace physics.
    for uav in env.uavs:
        original = uav._process_service_request
        def measured(ue, rate, target, neighbor, original=original):
            before = [u.assignment_causal_energy for u in env.uavs]
            original(ue, rate, target, neighbor)
            request_energy[ue.id] = math.fsum(u.assignment_causal_energy - b for u, b in zip(env.uavs, before))
        uav._process_service_request = measured
    actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
    for request, action in zip(snapshot.requests, assignment):
        actions[request[0], request[1]] = action
    # Keep just-executed requests observable; do not generate the next workload.
    env._get_obs = lambda: []
    _, _, metrics = env.step(np.asarray(snapshot.motion), offloading_actions=actions)
    if metrics['service_fallback_count']:
        raise RuntimeError('Frozen assignment fell back to a different target')
    ues = {ue.id: ue for ue in env.ues}
    records = []
    for request, action in zip(snapshot.requests, assignment):
        _, _, ue_id, source, deadline = request
        latency = float(ues[ue_id].latency_current_request)
        records.append({'ue_id': ue_id, 'source': source, 'action': int(action),
                        'target': 'mbs' if action == 1 else int(source if action == 0 else action - 2),
                        'latency': latency, 'deadline': deadline, 'success': latency <= deadline,
                        'assignment_energy': request_energy[ue_id]})
    return {'requests': records, 'metrics': metrics,
            'cache_after': [u.cache.tolist() for u in env.uavs],
            'snapshot_hash': snapshot.snapshot_hash}
