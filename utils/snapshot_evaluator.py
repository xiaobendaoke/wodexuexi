"""Pure fixed-assignment evaluation using the production Env.step physics.

Snapshots are trusted in-process objects, never unpickled from external files.
No Chapter 4 optimization objective is selected by this module.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import pickle

import numpy as np
import config
from utils.canonical_experiment import config_snapshot, digest


@dataclass(frozen=True)
class AssignmentSnapshot:
    payload: bytes
    config_hash: str
    requests: tuple
    candidates: tuple

    @property
    def snapshot_hash(self):
        return hashlib.sha256(self.payload).hexdigest()


def capture(env):
    """Preserve associations, channels, caches, requests and private RNG state."""
    frozen = deepcopy(env)
    frozen.get_offloading_obs_and_masks()
    hard = frozen.last_offload_mask_components['hard']
    requests, candidates = [], []
    for row, uav in enumerate(frozen.uavs):
        service = sorted((ue for ue in uav.current_covered_ues if ue.current_request.is_service),
                         key=lambda ue: float(np.linalg.norm(uav.pos[:2] - ue.pos[:2])))
        for slot, ue in enumerate(service):
            requests.append((row, slot, ue.id, uav.id, float(ue.current_request.deadline)))
            candidates.append(tuple(int(a) for a in np.flatnonzero(hard[row, slot])))
    return AssignmentSnapshot(pickle.dumps(frozen, protocol=5), digest(config_snapshot()),
                              tuple(requests), tuple(candidates))


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
    actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
    for request, action in zip(snapshot.requests, assignment):
        actions[request[0], request[1]] = action
    # Keep just-executed requests observable; do not generate the next workload.
    env._get_obs = lambda: []
    _, _, metrics = env.step(np.zeros((config.NUM_UAVS, 2)), offloading_actions=actions)
    if metrics['service_fallback_count']:
        raise RuntimeError('Frozen assignment fell back to a different target')
    ues = {ue.id: ue for ue in env.ues}
    records = []
    for request, action in zip(snapshot.requests, assignment):
        _, _, ue_id, source, deadline = request
        latency = float(ues[ue_id].latency_current_request)
        records.append({'ue_id': ue_id, 'source': source, 'action': int(action),
                        'target': 'mbs' if action == 1 else int(source if action == 0 else action - 2),
                        'latency': latency, 'deadline': deadline, 'success': latency <= deadline})
    return {'requests': records, 'metrics': metrics,
            'cache_after': [u.cache.tolist() for u in env.uavs],
            'snapshot_hash': snapshot.snapshot_hash}
