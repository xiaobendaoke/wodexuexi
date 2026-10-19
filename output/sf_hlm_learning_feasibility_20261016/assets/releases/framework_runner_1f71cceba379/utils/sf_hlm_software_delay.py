"""Process-local, once-per-step delay injection before rewards/next observation.

Only the evaluator sees raw physical outcomes. Policies receive neither these
outcomes nor the elapsed time of the action they have not selected yet.
"""
from __future__ import annotations
import math

_ACTIVE=set()


class SoftwareServiceDelay:
    def __init__(self,env,mode):
        if mode not in ('zero','measured'):raise ValueError('Unknown environment delay mode')
        self.env=env;self.mode=mode;self.cls=type(env.uavs[0]);self.pending=None;self.observed=None

    def __enter__(self):
        if self.cls in _ACTIVE:raise RuntimeError('Nested delay instrumentation')
        self.descriptor=self.cls.__dict__['execute_request_batch']
        if not isinstance(self.descriptor,staticmethod):raise TypeError('Batch executor contract changed')
        original=self.descriptor.__func__;_ACTIVE.add(self.cls)
        def execute(plans):
            if self.pending is None or self.observed is not None:raise RuntimeError('Delay hook must be armed exactly once per step')
            original(plans)
            ids,seconds=self.pending;delay=seconds if self.mode=='measured' else 0.
            ues=sorted((u for u in self.env.ues if u.assigned and u.current_request.is_service),key=lambda u:u.id)
            if [int(u.id) for u in ues]!=ids:raise ValueError('Delay request identities differ from admitted policy requests')
            rows=[]
            for u in ues:
                raw=float(u.latency_current_request);deadline=float(u.current_request.deadline)
                if not math.isfinite(raw) or raw<0 or not math.isfinite(deadline) or deadline<=0:raise ValueError('Invalid raw physical request')
                if delay:u.latency_current_request=raw+delay
                rows.append(dict(ue_id=int(u.id),latency=raw,deadline=deadline))
            self.observed=dict(requests=rows,injected_seconds=delay,software_seconds=seconds,
                success=sum(r['latency']+delay<=r['deadline'] for r in rows),
                raw_success=sum(r['latency']<=r['deadline'] for r in rows),
                shadow_fee_success=sum(r['latency']+seconds<=r['deadline'] for r in rows))
        self.cls.execute_request_batch=staticmethod(execute)
        return self

    def arm(self,request_ids,seconds):
        if self.pending is not None:raise RuntimeError('Previous step was not consumed')
        seconds=float(seconds);ids=sorted(int(i) for i in request_ids)
        if len(set(ids))!=len(ids) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid delay arm')
        self.pending=(ids,seconds);self.observed=None

    def finish(self):
        if self.observed is None:raise RuntimeError('Batch executor was not observed')
        result=self.observed;self.pending=None;self.observed=None
        return result

    def __exit__(self,*exc):
        self.cls.execute_request_batch=self.descriptor;_ACTIVE.remove(self.cls)
