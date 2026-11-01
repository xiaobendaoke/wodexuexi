"""Public queue features and virtual prefixes; no Torch, future tapes or teachers."""
import copy
import math

from environment.queued_service import Engine, distance
from utils.queued_service_controllers import Controller

STATE_DIM = 80
ALLOCATION_DIM = 30
MOTION_DIM = 44
MAX_MOTIONS = 16


def bounded(xs):
    return [max(-20., min(20., float(x))) for x in xs]


def node_features(obs):
    p = obs['protocol']; now = obs['time']; tasks = obs['tasks']; rows = []
    for node, capacity in enumerate(p.capacity):
        xy = p.mbs if node == 5 else obs['positions'][node]
        assigned = [x for x in tasks if x['target'] == node]
        uploads = [x for x in tasks if x['source'] == node and x['stage'] == 'upload']
        incoming = [x for x in tasks if x['target'] == node and x['target'] != x['source']]
        outgoing = [x for x in tasks if x['source'] == node and x['target'] is not None and x['target'] != node]
        rows.append(bounded([xy[0]/700, xy[1]/700, capacity/200e9,
            sum(x['remaining_cycles'] for x in assigned)/capacity,
            sum(x['remaining_cycles'] for x in assigned if x['stage'] == 'compute')/capacity,
            sum(x['remaining_upload'] for x in uploads)/20e6,
            sum(x['remaining_transfer'] for x in incoming)/20e6,
            sum(x['remaining_transfer'] for x in outgoing)/20e6, len(assigned)/20,
            sum(x['remaining_cycles'] for x in assigned if x['deadline']-now <= 1)/capacity,
            min([x['deadline']-now for x in assigned] or [3.])/3,
            sum(1 for x in assigned if x['stage'] == 'compute')/20]))
    return rows


def state_features(obs, weights):
    tasks = obs['tasks']; now = obs['time']
    nodes = node_features(obs)
    extra = [max(0., 200-now)/200, sum(x['source'] is None for x in tasks)/100,
        sum(x['source'] is not None and x['target'] is None for x in tasks)/40,
        sum(x['remaining_upload']+x['remaining_transfer'] for x in tasks)/1e9,
        sum(x['deadline']-now for x in tasks)/max(len(tasks), 1)/3,
        len(obs['ues'])/100, sum(weights.values())/100, len(tasks)/100]
    result = [v for row in nodes for v in row]+bounded(extra)
    assert len(result) == STATE_DIM
    return result


class VirtualQueue:
    def __init__(self, obs):
        self.obs = copy.deepcopy(obs)
        self.by_id = {x['uid']: x for x in self.obs['tasks']}
        self.radio = Engine(obs['protocol'], obs['positions'], obs['ues'])

    def features(self, uid, weights):
        x = self.by_id[uid]; obs = self.obs; p = obs['protocol']; positions = obs['positions']
        source = x['source']
        if source is None or x['target'] is not None: raise ValueError('requires a new admitted task')
        now = obs['time']; slack = x['deadline']-now; ue = obs['ues'][x['ue']]
        nodes = node_features(obs)
        active_ues = {z['ue'] for z in obs['tasks'] if z['source'] == source and z['stage'] == 'upload'}
        rate = self.radio.rate('uplink', (*ue, 0.), (*positions[source], p.altitude))
        upload = x['remaining_upload']*max(1, len(active_ues))/rate
        task = bounded([x['bits']/20e6, x['remaining_cycles']/13.5e9, slack/3,
            x['remaining_upload']/max(x['bits'], 1), ue[0]/700, ue[1]/700,
            positions[source][0]/700, positions[source][1]/700, nodes[source][3], (now-x['arrival'])/3])
        features = []; mask = []; prior = []
        for node in range(6):
            legal = node in (source, 5) or distance(positions[source], positions[node]) <= p.inter_range
            target = p.mbs if node == 5 else (*positions[node], p.altitude)
            transfer = 0.; transfer_rate = 0.
            if node != source:
                kind = 'backhaul' if node == 5 else 'inter'
                backlog = sum(z['remaining_transfer'] for z in obs['tasks']
                    if z['target'] is not None and z['target'] != z['source'] and (z['target'] == 5) == (node == 5))
                transfer_rate = self.radio.rate(kind, (*positions[source], p.altitude), target)
                transfer = (backlog+x['bits'])/transfer_rate
            cpu = (x['remaining_cycles']+sum(z['remaining_cycles'] for z in obs['tasks']
                if z['target'] == node and z['deadline'] <= x['deadline']))/p.capacity[node]
            latency = upload+transfer+cpu
            link = bounded([node == source, node == 5, distance(positions[source], target)/700,
                upload/3, transfer/3, cpu/3, latency > slack, transfer_rate/1e8])
            features.append(nodes[node]+task+link); mask.append(legal)
            prior.append(max(-20., -2*latency))
        assert all(len(row) == ALLOCATION_DIM for row in features)
        return state_features(obs, weights), features, mask, prior

    def assign(self, uid, node):
        x = self.by_id[uid]; source = x['source']; p = self.obs['protocol']
        if source is None or x['target'] is not None or node not in range(6): raise ValueError('invalid virtual assignment')
        if node not in (source, 5) and distance(self.obs['positions'][source], self.obs['positions'][node]) > p.inter_range:
            raise ValueError('unreachable virtual target')
        x['target'] = node; x['remaining_transfer'] = x['bits'] if node != source else 0.


class LearningAdapter:
    def __init__(self):
        self.history = Controller('history_queue'); self.last_time = None; self.plans = None

    def prepare(self, obs):
        if self.last_time is not None and obs['time'] <= self.last_time: raise ValueError('history updates once per increasing slot')
        self.last_time = obs['time']; self.history._update_history(obs)
        self.plans = self.history.candidates(obs)
        assert 1 <= len(self.plans) <= MAX_MOTIONS

    def motion(self, obs):
        if self.plans is None or obs['time'] != self.last_time: raise ValueError('prepare current observation first')
        p = obs['protocol']; radio = Engine(p, obs['positions'], obs['ues']); weights = self.history.weights
        features = []
        for plan in self.plans:
            after = [(a[0]+d[0], a[1]+d[1]) for a, d in zip(obs['positions'], plan)]
            middle = [(a[0]+d[0]/2, a[1]+d[1]/2) for a, d in zip(obs['positions'], plan)]
            row = []
            for i in range(5):
                uploads = [x for x in obs['tasks'] if x['source'] == i and x['stage'] == 'upload']
                contact = sum(x['remaining_upload'] for x in uploads if distance(obs['ues'][x['ue']], after[i]) <= p.coverage)/20e6
                progress = 0.
                for x in obs['tasks']:
                    source, target = x['source'], x['target']
                    if source is None or target is None or source == target or x['remaining_transfer'] <= 0 or i not in (source, target): continue
                    if target != 5 and distance(middle[source], middle[target]) > p.inter_range: continue
                    end = p.mbs if target == 5 else (*middle[target], p.altitude)
                    kind = 'backhaul' if target == 5 else 'inter'
                    progress += min(1., radio.rate(kind, (*middle[source], p.altitude), end)/x['remaining_transfer'])
                history = sum(weights.get(u, 0.)*math.exp(-distance(xy, after[i])**2/20000) for u, xy in obs['ues'].items())/100
                waiting = sum(x['source'] is None and distance(obs['ues'][x['ue']], after[i]) <= p.coverage for x in obs['tasks'])/100
                row += bounded([after[i][0]/700, after[i][1]/700, plan[i][0]/15, plan[i][1]/15,
                    history, contact, progress/20, waiting])
            row += bounded([self.history._history_score(obs, plan)/100,
                sum(math.hypot(*d) for d in plan)/75,
                sum(x['source'] is None and any(distance(obs['ues'][x['ue']], a) <= p.coverage for a in after) for x in obs['tasks'])/100,
                sum(x['remaining_transfer'] for x in obs['tasks'])/1e9])
            assert len(row) == MOTION_DIM; features.append(row)
        mask = [True]*len(features)+[False]*(MAX_MOTIONS-len(features))
        features += [[0.]*MOTION_DIM for _ in range(MAX_MOTIONS-len(features))]
        return dict(state=state_features(obs, weights), features=features, mask=mask, actions=self.plans)


def finite_gae(rewards, values, gamma=1., lam=.95):
    if not rewards or len(rewards) != len(values): raise ValueError('one complete finite episode required')
    advantages = [0.]*len(rewards); future = 0.; next_value = 0.
    for i in range(len(rewards)-1, -1, -1):
        delta = rewards[i]+gamma*next_value-values[i]
        future = delta+gamma*lam*future; advantages[i] = future; next_value = values[i]
    return advantages, [a+v for a, v in zip(advantages, values)]
