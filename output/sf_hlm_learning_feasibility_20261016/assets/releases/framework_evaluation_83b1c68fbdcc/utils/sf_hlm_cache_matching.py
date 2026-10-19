"""Conditional matching with public cache probabilities and fixed branch times.

This module receives proposal rows and frozen predictors only. The private
direct components used for label validation are deliberately unavailable.
"""
from __future__ import annotations
import math
from utils.sf_hlm_cache_mixture import CacheMixtureKernel
from utils.sf_hlm_cache_calibration import cache_features,controls,eligible
from utils.sf_hlm_priced_runtime import PricedPolicy

CONTEXT_RULE='all_visible_miss_zero_v1'
COLD_VARIANTS=dict(cold_prior='prior',cold_own='own_proxy',cold_half='half_proxy')
VARIANTS=('prior','own_proxy','half_proxy',*COLD_VARIANTS,'learned')
CACHE_METHODS=dict(mixture='prior',mixture_price='prior',own_proxy_price='own_proxy',
    half_proxy_price='half_proxy',cache_probability_price='learned',cache_rerank_price='learned',
    cold_prior_price='cold_prior',cold_own_proxy_price='cold_own',cold_half_proxy_price='cold_half',
    cache_regime_probability_price='learned')


class CacheWeightKernel(CacheMixtureKernel):
    def __init__(self,rows,powers,variant,model=None,cost_ms=10.):
        if variant not in VARIANTS:raise ValueError('Unregistered cache weight')
        super().__init__(rows,powers,cost_ms);self.variant=variant;self.prediction_batches=0
        model_rule=getattr(model,'public_cache_context_rule','none') if variant=='learned' else 'none'
        if model_rule not in ('none',CONTEXT_RULE):raise ValueError('Unsupported public cache context rule')
        self.public_guard_checked=variant in COLD_VARIANTS or model_rule==CONTEXT_RULE
        self.public_guard_triggered=False
        if self.public_guard_checked:
            from utils.sf_hlm_cache_context import public_context
            self.public_guard_triggered=public_context(rows)['global']['all_visible_miss']
        base_variant=COLD_VARIANTS.get(variant,variant)
        selected=[(i,a) for i,parts in enumerate(self.parts) for a,part in parts.items() if eligible(part)]
        self.eligible_edges=len(selected)
        if self.public_guard_triggered:probabilities=[0.]*len(selected)
        elif base_variant=='learned' and selected:
            import numpy as np
            if model is None:raise ValueError('Frozen cache model absent')
            mode=getattr(model,'public_cache_feature_mode','clipped')
            x=np.asarray([cache_features(rows[i],a,mode=mode) for i,a in selected],dtype=np.float64)
            pp=np.asarray(model.predict_proba(x),dtype=np.float64)
            if pp.shape!=(len(selected),2) or not np.isfinite(pp).all() or (pp<0).any() or (pp>1).any():raise ValueError('Invalid cache probabilities')
            probabilities=pp[:,1];self.prediction_batches=1
        elif base_variant=='prior':return
        else:probabilities=[controls(rows[i],a)[base_variant] for i,a in selected]
        for (i,a),value in zip(selected,probabilities):self.parts[i][a]['p']=float(value)


class CacheMatchingPolicy(PricedPolicy):
    def __init__(self,runtime,hgb_model,cache_model,prices,powers):
        super().__init__(runtime,hgb_model,prices,powers);self.cache_model=cache_model

    def run(self,rows,*,num_uavs,kind,model=None,rng=None,cap=16,trace=False):
        if kind not in CACHE_METHODS:
            return super().run(rows,num_uavs=num_uavs,kind=kind,model=model,rng=rng,cap=cap,trace=trace)
        if num_uavs!=5:raise ValueError('Frozen five-node contract')
        import numpy as np
        from utils import sf_hlm_calibrated_matching as cm
        from utils.sf_hlm_action_ranker import features
        from utils.sf_hlm_request_risk import proposal_actions
        from utils.sf_hlm_energy_price import choose
        variant=CACHE_METHODS[kind];g,repair,base=self.initialize(rows);gd=self.rt.assess(rows,g)
        legal=[int(k)+1 for k in np.flatnonzero(base.action_mask()[1:])]
        safe=dict(rows=rows,greedy_assignment=g,initial_assignment=base.assignment,initial_counts=base.current.tolist(),
            energy_cap=base.cap,legal_nonstop_actions=legal)
        ids,x=features(safe,10.,problem=base.problem);proposals=proposal_actions(ids,x);quotas=cm.quota_set(base.current.tolist(),proposals)
        kernel=CacheWeightKernel(rows,self.powers,variant,self.cache_model);matrices=kernel.probabilities([q for _,q in quotas])
        rerank=kind=='cache_rerank_price'
        matching_matrices=CacheWeightKernel(rows,self.powers,'half_proxy').probabilities([q for _,q in quotas]) if rerank else matrices
        targets=self.rt.kernel(rows).targets(g);matrix=matrices[tuple(int(k) for k in base.current)]
        gscore=float(sum(matrix[i,j] for i,j in enumerate(targets)));calls=base.calls
        candidates=[dict(action=-1,assignment=g,predicted=gd,expected_success=gscore)]
        for action,q in quotas:
            result=cm.solve(rows,q,10.,'probability',assess=self.rt.assess,problem=base.problem,probability_matrix=matching_matrices[tuple(q)])
            calls+=int(bool(rows))
            if result is not None and rerank:
                targets=self.rt.kernel(rows).targets(result['assignment']);matrix=matrices[tuple(q)]
                result['expected_success']=float(sum(matrix[i,j] for i,j in enumerate(targets)))
            if result is not None and result['predicted']['energy']<=base.cap+1e-7:candidates.append(dict(action=action,**result))
        if kind.endswith('_price'):
            parameters=self.prices['prices']['rerank' if rerank else variant]
            chosen=choose([dict(action=c['action'],score=c['expected_success'],energy=c['predicted']['energy']) for c in candidates],parameters['price'],parameters['normalizer'])
        else:chosen=cm.select([c['expected_success'] for c in candidates],[c['predicted']['energy'] for c in candidates])
        result=candidates[chosen]
        diagnostic=dict(chosen_predicted=result['predicted'],reference_predicted=base.reference,common_energy_cap=base.cap,
            matching_calls=calls,matching_budget=10,base_evaluations=repair['base_evaluations'],selected_action=result['action'],
            nominal_cost_ms=10.,price_applied=kind.endswith('_price'),expected_success=result['expected_success'],cache_weight=variant,
            cache_eligible_edges=kernel.eligible_edges,cache_prediction_batches=kernel.prediction_batches,
            mixture_fallback_counts=kernel.fallback_counts,public_inverse_approximation_retained=True)
        if calls>10:raise ValueError('Cache matching budget exceeded')
        if kernel.public_guard_checked:diagnostic.update(public_cache_context_rule=CONTEXT_RULE,public_cache_guard_triggered=kernel.public_guard_triggered)
        if rerank:diagnostic.update(assignment_weight='half_proxy',selection_weight='learned',fixed_half_assignment_pool=True)
        return tuple(result['assignment']),diagnostic,candidates if trace else []
