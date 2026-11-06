"""All-public-task set features; fixed size, no truncation or label inputs."""
import numpy as np
from utils.queued_learning_validation import projection


def taskset_features(obs, plan, motion):
    tasks = obs['tasks'];p = obs['protocol'];n = len(tasks)
    if not n:return np.zeros(64), 0
    positions = np.asarray(obs['positions'])+np.asarray(motion)
    planned = [x['target'] if x['target'] is not None else plan.get(x['uid']) for x in tasks]
    queues = {}
    for node in range(len(p.capacity)):
        selected = sorted((x['deadline'], x['remaining_cycles']) for x, target in zip(tasks, planned) if target == node)
        queues[node] = (np.array([z[0] for z in selected]), np.r_[0., np.cumsum([z[1] for z in selected])]/p.capacity[node])
    rows = []
    for x in tasks:
        ue = np.asarray(obs['ues'][x['ue']]);waiting = x['source'] is None
        source = int(np.argmin(np.linalg.norm(positions-ue, axis=1))) if waiting else x['source']
        target = x['target'] if x['target'] is not None else plan.get(x['uid'], source)
        target_position = np.asarray(p.mbs[:2]) if target == len(positions) else positions[target]
        d = float(np.linalg.norm(positions[source]-target_position))
        upload = float(np.linalg.norm(positions[source]-ue));deadlines, work = queues[target]
        queued = work[np.searchsorted(deadlines, x['deadline'], side='right')]
        transfer = x['remaining_transfer'] if x['target'] is not None else x['bits'] if target != source else 0.
        rows.append([(x['deadline']-obs['time'])/3, x['remaining_upload']/24e6, transfer/24e6,
            x['remaining_cycles']/13.5e9, (obs['time']-x['arrival'])/3,
            ('waiting', 'upload', 'transfer', 'compute').index(x['stage'])/3,
            p.capacity[source]/85e9, p.capacity[target]/85e9, upload/p.coverage, d/p.inter_range,
            queued/3, float(upload <= p.coverage), float(target == source or target == len(positions) or d <= p.inter_range),
            float(x['target'] is None), float(target == source), float(waiting)])
    encoded = projection(np.asarray(rows), 32)
    features = np.r_[encoded.mean(0), encoded.max(0)]
    features[0] = n/100.
    if not np.isfinite(features).all():raise ValueError('nonfinite public task-set encoding')
    return features, n
