"""Remote-only observation and diagnostic extension; production physics untouched."""
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import config
from environment.env import Env as BaseEnv
from environment import comm_model as comms
from environment.uavs import OFFLOAD_TARGET_LOCAL, OFFLOAD_TARGET_COOPERATIVE


def packet_kept(seed, episode, slot, uav, ue, drop):
    key = f"ch3-telemetry-v1:{seed}:{episode}:{slot}:{uav}:{ue}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64 >= drop


def configure(args):
    # Preserve checkpoint shapes under frozen-policy physical interventions.
    config.MAX_ASSOCIATED_UES = 30
    config.OBS_DIM_SINGLE = 325
    config.MAX_OFFLOAD_REQUESTS_PER_UAV = 100
    config.OFFLOAD_OBS_DIM_SINGLE = 5 + 100 * config.OFFLOAD_REQUEST_FEATURE_DIM
    config.UPPER_MLP_ACTOR_WIDTH = args.ch3_actor_width
    config.UPPER_MLP_CRITIC_WIDTH = args.ch3_critic_width
    if args.ch3_uniform:
        config.USE_HOTSPOTS = False
        config.HOTSPOT_UE_PROB = 0.0
    if not (100 <= args.ch3_demand_radius <= 460 and 0 <= args.ch3_drop <= 1 and 0 <= args.ch3_delay <= 5):
        raise ValueError('Invalid telemetry intervention')
    config.CH3_DEMAND_RADIUS = args.ch3_demand_radius
    config.CH3_DELAY = args.ch3_delay
    config.CH3_DROP = args.ch3_drop
    ExperimentEnv.record_requests = args.ch3_record_requests
    ExperimentEnv.output = Path(args.output)


def compatible(contract, snapshot, source_hash, args):
    from utils.canonical_experiment import digest
    if contract['method'] != args.method or contract['source_hash'] != source_hash:
        return False
    if contract['config_hash'] == digest(snapshot):
        return True
    if not args.ch3_boundary or 'config' not in contract:
        return False
    allowed = {'NUM_UES', 'UAV_COMPUTING_CAPACITY', 'USE_HOTSPOTS', 'HOTSPOT_UE_PROB',
               'CH3_DEMAND_RADIUS', 'CH3_DELAY', 'CH3_DROP'}
    old = contract['config']
    return old.keys() == snapshot.keys() and all(old[k] == snapshot[k] for k in old.keys() - allowed)


class ExperimentEnv(BaseEnv):
    record_requests = False
    output = None

    def _get_obs(self):
        original = super()._get_obs()  # Sole generator of requests and associations.
        stamp = (self._episode_index, self._time_step)
        if not hasattr(self, '_telemetry') or self._telemetry_episode != stamp[0]:
            self._telemetry = deque(maxlen=6)
            self._telemetry_episode = stamp[0]
        # Reset creates new UAV instances. Install passive observers on each one.
        if self.record_requests and self._time_step == 0:
            self._install_observers()
        mode = config.UPPER_UE_OBSERVATION_MODE
        self.ch3_slot_counts = []
        if mode == 'covered' or (config.CH3_DEMAND_RADIUS == 460 and config.CH3_DELAY == 0 and config.CH3_DROP == 0):
            # Avoid duplicate Python per-UE work during the 8 million training transitions.
            positions = np.asarray([ue.pos[:2] for ue in self.ues])
            unassigned = np.asarray([not ue.assigned for ue in self.ues])
            for uav in self.uavs:
                own = len(uav.current_covered_ues)
                unseen = int(np.count_nonzero(unassigned & (np.linalg.norm(positions-uav.pos[:2], axis=1)<=460))) if mode != 'covered' else 0
                self.ch3_slot_counts.append(dict(uav=uav.id, covered=own, visible_uncovered=unseen,
                                                saturated=own+unseen >= 30, truncated=max(own+unseen-30, 0)))
            return original
        packet_frame = []
        for uav in self.uavs:
            packets = []
            for ue in self.ues:
                if ue.assigned or np.linalg.norm(uav.pos[:2] - ue.pos[:2]) > config.CH3_DEMAND_RADIUS:
                    continue
                req = ue.current_request
                if not packet_kept(self.scenario_seed, *stamp, uav.id, ue.id, config.CH3_DROP):
                    continue
                features = [float(req.req_type)/2, req.req_size/config.MAX_INPUT_SIZE,
                            req.req_id/config.NUM_FILES,
                            req.deadline/config.SERVICE_DEADLINE_MAX if req.is_service else 0.,
                            req.priority/config.SERVICE_PRIORITY_MAX if req.is_service else 0.,
                            ue.battery_level/config.UE_BATTERY_CAPACITY]
                packets.append((ue.id, ue.pos[:2].copy(), features))
            packet_frame.append(packets)
        if not self._telemetry or self._telemetry[-1][0] != stamp:
            self._telemetry.append((stamp, packet_frame))
        mode = config.UPPER_UE_OBSERVATION_MODE
        if mode not in ('covered', 'covered_plus_uncovered'):
            raise ValueError('Unexpected observation mode in preregistered experiment')
        self.ch3_slot_counts = []
        for uav in self.uavs:
            own = len(uav.current_covered_ues)
            unseen = len(packet_frame[uav.id]) if mode != 'covered' else 0
            self.ch3_slot_counts.append(dict(uav=uav.id, covered=own, visible_uncovered=unseen,
                                            saturated=own+unseen >= 30, truncated=max(own+unseen-30, 0)))
        if mode == 'covered' or (config.CH3_DEMAND_RADIUS == 460 and config.CH3_DELAY == 0 and config.CH3_DROP == 0):
            return original  # Main experiment is byte-for-byte original observation.
        wanted = (stamp[0], stamp[1] - config.CH3_DELAY)
        historic = next((frame for key, frame in self._telemetry if key == wanted), None)
        answer = []
        for uav, obs in zip(self.uavs, original):
            own = sorted(uav.current_covered_ues, key=lambda ue: (float(np.linalg.norm(uav.pos[:2]-ue.pos[:2])), ue.id))
            own_ids = {ue.id for ue in own}
            packets = [] if historic is None else [x for x in historic[uav.id] if x[0] not in own_ids]
            packets.sort(key=lambda x: (float(np.linalg.norm(uav.pos[:2]-x[1])), x[0]))
            # Candidate saturation describes packets actually available at the receiver,
            # not the current sender frame (which can differ under delay).
            self.ch3_slot_counts[uav.id] = dict(uav=uav.id, covered=len(own),
                transmitted_uncovered=len(packet_frame[uav.id]), visible_uncovered=len(packets),
                saturated=len(own)+len(packets)>=30, truncated=max(len(own)+len(packets)-30,0))
            current = obs[85:].reshape(30, 8)
            slots = np.zeros((30, 8), dtype=np.float32)
            count = min(len(own), 30)
            slots[:count] = current[:count]
            for slot, (_, position, data) in enumerate(packets[:30-count], start=count):
                slots[slot] = np.concatenate(((position-uav.pos[:2])/np.array([config.AREA_WIDTH, config.AREA_HEIGHT], dtype=np.float32), data))
            answer.append(np.concatenate((obs[:85], slots.flatten())))
        return answer

    def get_offloading_obs_and_masks(self):
        needed = max(sum(ue.current_request.is_service for ue in u.current_covered_ues) for u in self.uavs)
        if needed > config.MAX_OFFLOAD_REQUESTS_PER_UAV:
            raise RuntimeError(f'INTERFACE_NOT_APPLICABLE: {needed} admitted requests exceed checkpoint capacity')
        return super().get_offloading_obs_and_masks()

    def _install_observers(self):
        for uav in self.uavs:
            original = uav._process_service_request
            def measured(ue, rate, target, neighbor, original=original, uav=uav):
                req = ue.current_request
                execution = uav if target == OFFLOAD_TARGET_LOCAL else neighbor if target == OFFLOAD_TARGET_COOPERATIVE else None
                cycles = float(config.CPU_CYCLES_PER_BYTE[req.req_id]) * req.req_size
                hit = bool(execution.cache_snapshot[req.req_id]) if execution is not None else None
                load = int(execution._current_service_request_count) if execution is not None else None
                cpu = float(config.UAV_COMPUTING_CAPACITY[execution.id]) if execution is not None else float(config.MBS_COMPUTING_CAPACITY)
                compute = cycles / (cpu / max(load, 1) if load is not None else cpu)
                upload = 8. * req.req_size / max(float(rate), config.EPSILON)
                forward = 0.
                if target == OFFLOAD_TARGET_COOPERATIVE:
                    link = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(uav.pos, neighbor.pos))
                    forward = 8. * req.req_size / max(float(link), config.EPSILON)
                elif execution is None:
                    forward = 8. * req.req_size / max(float(uav._uav_mbs_rate), config.EPSILON)
                original(ue, rate, target, neighbor)
                latency = float(ue.latency_current_request)
                self._ch3_processed[ue.id] = dict(source=uav.id, target='mbs' if execution is None else execution.id,
                    associated_users=len(uav.current_covered_ues), cache_hit=hit, final_node_tasks=load,
                    upload_latency=upload, forward_latency=forward, compute_latency=compute,
                    fetch_latency=max(latency-upload-forward-compute, 0.), latency=latency,
                    success=latency <= req.deadline)
            uav._process_service_request = measured

    def step(self, actions, sample_recorder=None, offloading_actions=None):
        if not self.record_requests:
            return super().step(actions, sample_recorder=sample_recorder, offloading_actions=offloading_actions)
        self._ch3_processed = {}
        episode, slot = self._episode_index, self._time_step
        records = []
        for ue in self.ues:
            req = ue.current_request
            if req.is_service:
                records.append(dict(workload_seed=self.scenario_seed, episode=episode, slot=slot, ue=ue.id,
                    request_id=req.req_id, size=req.req_size, deadline=req.deadline, priority=req.priority,
                    cycles=float(config.CPU_CYCLES_PER_BYTE[req.req_id])*req.req_size,
                    admitted=bool(ue.assigned), success=False))
        counts = list(self.ch3_slot_counts)
        result = super().step(actions, sample_recorder=sample_recorder, offloading_actions=offloading_actions)
        metrics = result[2]
        for row in records:
            row.update(self._ch3_processed.get(row['ue'], {}))
        g = len(records); a = sum(r['admitted'] for r in records); s = sum(r['success'] for r in records)
        if (g != metrics['generated_service_count'] or a != metrics['processed_service_count'] or s != metrics['deadline_satisfied_count']):
            raise AssertionError('Diagnostic counts disagree with production transition')
        directory = self.output/'requests'; directory.mkdir(parents=True, exist_ok=True)
        with gzip.open(directory/f'episode_{episode:05d}.jsonl.gz', 'at', encoding='utf-8') as f:
            for row in records:
                f.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
        with (directory/f'slots_{episode:05d}.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(slot=slot, counts=counts))+'\n')
        return result
