"""Recovery revision: keep off-peak business fixed, vary only the burst peak."""
import hashlib
import json
from utils.queued_service_business import BusinessScenario

NAMESPACE='chapter5/business_recovery_20261031/v1'


class RecoveryScenario:
    def __init__(self,workload,profile,sla='normal',scale=.45,horizon=200):
        if sla!='normal' or (profile=='burst' and scale not in (.45,.6,.75)) or (profile!='burst' and scale!=.75):
            raise ValueError('recovery revision fixes SLA and off-peak rates')
        baseline=BusinessScenario(workload,profile,'normal',.75,horizon)
        self.__dict__.update(baseline.__dict__)
        if profile=='burst':
            peak=BusinessScenario(workload,profile,'normal',scale,horizon)
            self.requests=[peak.slot(t) if 50<=t<100 else baseline.slot(t) for t in range(horizon)]
            self.rates=tuple(100*scale if 50<=t<100 else 15. for t in range(horizon))
        self.scale=scale;self.counts=tuple(len(r) for r in self.requests)
        header=json.dumps(dict(namespace=NAMESPACE,workload=workload,ue=self.ue_positions),separators=(',',':')).encode()
        arrival=hashlib.sha256(header);full=hashlib.sha256(header)
        for row in self.requests:
            for t in row:
                fields=[t.uid,t.ue,t.arrival,t.bits,t.cycles]
                arrival.update(json.dumps(fields,separators=(',',':')).encode())
                full.update(json.dumps(fields+[t.deadline],separators=(',',':')).encode())
        self.arrival_fingerprint=arrival.hexdigest();self.fingerprint=full.hexdigest()
        self.statistics=dict(baseline.statistics,namespace=NAMESPACE,generated=sum(self.counts),expected=sum(self.rates),
            maximum_per_slot=max(self.counts),mean_rate=sum(self.counts)/horizon,
            duplicate_ue_slots=sum(len({x.ue for x in row})<len(row) for row in self.requests),
            rates_definition='Off-peak 15, selected burst peak; common profiles 7.5/30/90')

    def slot(self,t):
        if t<0:raise IndexError(t)
        return self.requests[t] if t<self.horizon else ()
