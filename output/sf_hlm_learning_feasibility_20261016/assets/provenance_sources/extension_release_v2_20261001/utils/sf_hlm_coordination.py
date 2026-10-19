"""Separate declared control-channel sensitivity model, never ideal-run physics.

One serialized control round through the MBS coordinator; rates supplied from
current geometry. Backhaul TX/RX powers charge UAVs, not the MBS. This is a
simulation assumption, not measurement of a deployed coordination system.
"""
from __future__ import annotations
import math
import numpy as np


def control_cost(rows, rates, tx_power, rx_power, *, additional_delay=0.):
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1 or not np.isfinite(rates).all() or (rates <= 0).any(): raise ValueError('Invalid control rates')
    if any(not math.isfinite(v) or v < 0 for v in (tx_power, rx_power, additional_delay)): raise ValueError('Invalid signaling cost')
    # Fixed binary wire budget: 16-byte header per source, 26 bytes/request,
    # 50 bytes/candidate (IDs + six float64 predictor/cache/capacity values).
    upload = np.full(len(rates), 16, dtype=np.int64)
    for row in rows:
        source = row['source']
        if type(source) is not int or not 0 <= source < len(rates): raise ValueError('Invalid source')
        upload[source] += 26 + 50 * len(row['options'])
    # UE ID + action byte, plus six float32 features for every execution node.
    reply = 16 + 5 * len(rows) + (len(rates) + 1) * 6 * 4
    up_seconds = 8 * upload / rates; down_seconds = 8 * reply / rates
    energy = tx_power * up_seconds + rx_power * down_seconds
    return dict(delay_seconds=float(up_seconds.sum() + down_seconds.sum() + additional_delay),
                uav_energy=energy, upload_bytes=upload, download_bytes=int(reply * len(rates)),
                assumption='dedicated serial lossless control channel, current symmetric backhaul rates')


def execute_controlled_slot(env, motion, actions, cost):
    """Apply delay BEFORE deadline metrics and energy BEFORE reward calculation.

    Hooks are restored even on failure; no next-slot state enters this model.
    """
    delay = cost['delay_seconds']; energy = cost['uav_energy']; originals = []; overrides = []
    for uav in env.uavs:
        overrides.append(uav.__dict__.get('_process_service_request'))
        original = uav._process_service_request; originals.append(original)
        def delayed(ue, rate, target, neighbor, original=original):
            original(ue, rate, target, neighbor); ue.latency_current_request += delay
        uav._process_service_request = delayed
    rewards_override = env.__dict__.get('_get_rewards_and_metrics'); rewards_original = env._get_rewards_and_metrics
    def charged():
        for uav, amount in zip(env.uavs, energy): uav._energy_current_slot += float(amount)
        rewards, metrics = rewards_original()
        metrics['coordination_delay_seconds'] = float(delay); metrics['coordination_uav_energy'] = float(np.sum(energy))
        return rewards, metrics
    env._get_rewards_and_metrics = charged
    try: return env.step(motion, offloading_actions=actions)
    finally:
        for uav, override in zip(env.uavs, overrides):
            if override is None: uav.__dict__.pop('_process_service_request', None)
            else: uav._process_service_request = override
        if rewards_override is None: env.__dict__.pop('_get_rewards_and_metrics', None)
        else: env._get_rewards_and_metrics = rewards_override
