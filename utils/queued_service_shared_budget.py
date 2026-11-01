"""Public one-step service features with the frozen engine's shared airtime."""
from collections import defaultdict
import copy
import math
import time
from environment.queued_service import Engine,distance
from utils.queued_service_learning import LearningAdapter as FrozenAdapter
from utils.queued_service_diagnostics import mc_returns

MOTION_DIM=86


def smooth(value):
    return math.copysign(math.log1p(abs(value)),value)


def evaluate_candidate(obs,action):
    """Predict known tasks for one second, without new arrivals or admissions."""
    e=Engine.from_observation(obs);p=e.p
    defaults={x.uid:x.source for x in e.tasks.values() if x.source is not None and x.target is None}
    e.assign(defaults)
    before={x.uid:(x.served_upload,x.served_transfer) for x in e.tasks.values()}
    initial=copy.deepcopy(list(e.tasks.values()));generated=len(initial)
    e.advance(action);audit=e.audit()
    assert all(v<=1.+1e-7 for v in e.airtime.values()) and all(v<=1.+1e-7 for v in e.cpu_busy)
    upload=[0.]*5;transfer=[0.]*5
    for x in (*e.tasks.values(),*e.finished.values()):
        if x.source is not None:
            upload[x.source]+=x.served_upload-before[x.uid][0]
            transfer[x.source]+=x.served_transfer-before[x.uid][1]
    airtime_work=defaultdict(float);cpu=[[] for _ in range(6)]
    for x in e.tasks.values():
        if x.source is None or x.target is None:continue
        cpu[x.target].append(x)
        if x.remaining_upload>0:
            rate=e.rate('uplink',(*e.ues[x.ue],0.),(*e.positions[x.source],p.altitude))
            airtime_work['upload_'+str(x.source)]+=x.remaining_upload/rate
        if x.remaining_transfer>0:
            target=p.mbs if x.target==5 else (*e.positions[x.target],p.altitude)
            kind='backhaul' if x.target==5 else 'inter'
            airtime_work[kind]+=x.remaining_transfer/e.rate(kind,(*e.positions[x.source],p.altitude),target)
    node_features=[];margins=[]
    for node in range(6):
        assigned=cpu[node];capacity=p.capacity[node]
        node_features+= [smooth(sum(x.remaining_cycles for x in assigned)/capacity),
            smooth(sum(x.remaining_cycles for x in assigned if x.deadline-e.t<=1)/capacity),
            smooth(sum(x.remaining_upload for x in e.tasks.values() if x.source==node)/20e6),
            smooth(sum(x.remaining_transfer for x in assigned)/20e6),e.cpu_busy[node]]
        for x in assigned:
            upload_work=airtime_work['upload_'+str(x.source)] if x.remaining_upload>0 else 0.
            transfer_work=airtime_work['backhaul' if x.target==5 else 'inter'] if x.remaining_transfer>0 else 0.
            edf=sum(y.remaining_cycles for y in assigned if y.deadline<=x.deadline)/capacity
            contact=max(0.,distance(e.ues[x.ue],e.positions[x.source])-p.coverage)/p.speed if x.remaining_upload>0 else 0.
            if x.remaining_transfer>0 and x.target!=5:
                contact+=max(0.,distance(e.positions[x.source],e.positions[x.target])-p.inter_range)/(2*p.speed)
            margins.append(x.deadline-e.t-upload_work-transfer_work-edf-contact)
    upload_margin=[]
    for node in range(5):
        pending=[x for x in initial if x.source==node and x.remaining_upload>0]
        total=sum(x.remaining_upload for x in pending)
        upload_margin.append(sum(x.remaining_upload*(p.coverage-distance(e.ues[x.ue],e.positions[node]))/p.coverage
            for x in pending)/max(total,1.))
    cooperation=[x for x in initial if x.source is not None and x.target is not None and x.source!=x.target and x.target!=5 and x.remaining_transfer>0]
    total=sum(x.remaining_transfer for x in cooperation)
    inter_margin=sum(x.remaining_transfer*(p.inter_range-distance(e.positions[x.source],e.positions[x.target]))/p.inter_range
        for x in cooperation)/max(total,1.)
    global_features=[e.counts['success']/20,e.counts['expired']/20,
        sum(x.source is None and any(distance(e.ues[x.ue],a)<=p.coverage for a in e.positions) for x in e.tasks.values())/100,
        smooth(airtime_work['inter']),smooth(airtime_work['backhaul']),smooth(min(margins) if margins else 0.)]
    return dict(generated=generated,default_source_assignments=len(defaults),upload_by_source=upload,transfer_by_source=transfer,
        airtime=dict(e.airtime),cpu_busy=e.cpu_busy,node_features=node_features,global_features=global_features,
        upload_margin=upload_margin,inter_margin=inter_margin,audit=audit)


class BudgetAdapter(FrozenAdapter):
    def __init__(self):
        super().__init__();self.forecast_calls=0;self.forecast_seconds=0.

    def motion(self,obs):
        bank=super().motion(obs);start=time.perf_counter();rows=[]
        for i,action in enumerate(bank['actions']):
            service=evaluate_candidate(obs,action);self.forecast_calls+=1
            base=list(bank['features'][i])
            for node in range(5):
                base[node*8+5]=service['upload_by_source'][node]/20e6
                base[node*8+6]=service['transfer_by_source'][node]/20e6
            rows.append(base+service['node_features']+service['global_features']+service['upload_margin']+[service['inter_margin']])
        assert all(len(row)==MOTION_DIM and all(math.isfinite(x) for x in row) for row in rows)
        bank['features']=rows+[[0.]*MOTION_DIM for _ in range(16-len(rows))]
        self.forecast_seconds+=time.perf_counter()-start
        return bank


def mc_credit(rewards,values):
    if not rewards or len(rewards)!=len(values):raise ValueError('one complete drained episode required')
    returns=mc_returns(rewards)
    return [v-baseline for v,baseline in zip(returns,values)],returns
