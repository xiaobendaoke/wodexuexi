"""Lazy fixed prefixes of the sealed analytical mixture policy.

Import after bootstrapping the external frozen numerical release. No private
environment or clock value is accepted as a decision input.
"""
import time

from utils.sf_hlm_priced_runtime import PricedPolicy

BUDGETS=(0,2,4,6,10)


class NativeRecordedPolicy(PricedPolicy):
    """Keep the sealed native implementation; retain its existing public pool."""
    def run(self,rows,**kwargs):
        requested=kwargs.pop('trace',False)
        assignment,diagnostic,candidates=super().run(rows,trace=True,**kwargs)
        self.last_trace=dict(budget=10,native=True,candidates=candidates)
        return assignment,diagnostic,candidates if requested else []


class PrefixPolicy(PricedPolicy):
    def __init__(self,runtime,prices,powers,budget):
        if budget not in BUDGETS:raise ValueError('Registered prefixes are 0,2,4,6,10')
        super().__init__(runtime,None,prices,powers);self.budget=budget

    def run(self,rows,*,num_uavs,kind,model=None,rng=None,cap=16,trace=False):
        if num_uavs!=5 or kind!='mixture_price' or cap!=self.budget:
            raise ValueError('Frozen mixture prefix contract')
        if self.budget==0:
            assignment,diagnostic,candidates=super().run(rows,num_uavs=5,kind='native_G',trace=trace)
            self.last_trace=dict(budget=0,native=False,initialization_calls=0,quota_order=[],visited=[],
                computed_matrix_quotas=[],candidates=[],stage_seconds={})
            return assignment,diagnostic,candidates
        import numpy as np
        from utils import sf_hlm_calibrated_matching as cm
        from utils.sf_hlm_action_ranker import features
        from utils.sf_hlm_request_risk import proposal_actions
        from utils.sf_hlm_cache_mixture import CacheMixtureKernel
        from utils.sf_hlm_energy_price import choose
        stages={};start=time.perf_counter()
        g,repair,base=self.initialize(rows);gd=self.rt.assess(rows,g)
        stages['repair_and_binary_initialization']=time.perf_counter()-start
        start=time.perf_counter()
        legal=[int(k) for k in np.flatnonzero(base.action_mask()[1:])+1]
        safe=dict(rows=rows,greedy_assignment=g,initial_assignment=base.assignment,
            initial_counts=base.current.tolist(),energy_cap=base.cap,legal_nonstop_actions=legal)
        ids,x=features(safe,10.,problem=base.problem)
        proposals=proposal_actions(ids,x);quotas=cm.quota_set(base.current.tolist(),proposals)
        stages['frozen_proposal_generation']=time.perf_counter()-start
        start=time.perf_counter();kernel=CacheMixtureKernel(rows,self.powers)
        stages['public_branch_decode']=time.perf_counter()-start
        matrices={};computed=[];stages['visited_matrix_scoring']=0.;stages['conditional_matching']=0.
        def matrix(q):
            key=tuple(int(k) for k in q)
            if key not in matrices:
                started=time.perf_counter()
                matrices[key]=kernel.probabilities([key])[key]
                stages['visited_matrix_scoring']+=time.perf_counter()-started
                computed.append(list(key))
            return matrices[key]
        prob=matrix(base.current);targets=self.rt.kernel(rows).targets(g)
        gscore=float(sum(prob[i,j] for i,j in enumerate(targets)))
        candidates=[dict(action=-1,assignment=g,predicted=gd,expected_success=gscore)]
        calls=base.calls;visited=[]
        for action,counts in quotas[:self.budget-1]:
            prob=matrix(counts);started=time.perf_counter()
            result=cm.solve(rows,counts,10.,'probability',self.model,assess=self.rt.assess,
                problem=base.problem,probability_matrix=prob)
            stages['conditional_matching']+=time.perf_counter()-started
            calls+=int(bool(rows))
            accepted=result is not None and result['predicted']['energy']<=base.cap+1e-7
            visited.append(dict(action=action,quota=list(counts),matching_call=int(bool(rows)),
                feasible=result is not None,accepted=accepted,
                rejection=None if accepted else 'infeasible' if result is None else 'energy_guard'))
            if accepted:candidates.append(dict(action=action,**result))
        start=time.perf_counter();parameters=self.prices['prices']['mixture']
        index=choose([dict(action=c['action'],score=c['expected_success'],energy=c['predicted']['energy']) for c in candidates],
            parameters['price'],parameters['normalizer'])
        chosen=candidates[index];stages['candidate_selection']=time.perf_counter()-start
        diagnostic=dict(chosen_predicted=chosen['predicted'],reference_predicted=base.reference,common_energy_cap=base.cap,
            matching_calls=calls,matching_budget=self.budget,base_evaluations=repair['base_evaluations'],
            selected_action=chosen['action'],nominal_cost_ms=10.,g_anchored_energy=False,price_applied=True,
            expected_success=chosen['expected_success'],hgb_batches=0,mixture_fallback_counts=kernel.fallback_counts)
        if calls>self.budget:raise ValueError('Prefix exceeded total matching budget')
        self.last_trace=dict(budget=self.budget,native=False,initialization_calls=base.calls,
            quota_order=[dict(action=a,quota=list(q)) for a,q in quotas],visited=visited,
            computed_matrix_quotas=computed,candidates=candidates,stage_seconds=stages)
        return tuple(chosen['assignment']),diagnostic,candidates if trace else []
