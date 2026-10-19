"""Per-call equivalent batching for the frozen public-price matching policy."""
from __future__ import annotations
import copy
import math

class LocalEdgeKernel:
    """Original scalar static fields once; identical dynamic fields per quota."""
    def __init__(self,rows,initial_counts,cost_ms,problem):
        import numpy as np
        from utils.sf_hlm_calibrated_matching import edge_features
        self.rows=rows;self.cost_ms=cost_ms;self.delay=cost_ms/1000;self.problem=problem
        self.static,self.index,_,self.initial=edge_features(rows,initial_counts,cost_ms,problem=problem)
        self.groups=[np.flatnonzero(self.index[:,0]==i) for i in range(len(rows))]
        self.ii=self.index[:,0];self.jj=self.index[:,1];self.deadline=problem['deadline'][self.ii]
        self.base=problem['base'][self.ii,self.jj];self.work=problem['work'][self.ii,self.jj]

    def features(self,counts):
        import numpy as np
        from utils.sf_hlm_matching import validated_counts
        q=validated_counts(counts,len(self.rows),5)
        if np.array_equal(q,self.initial):return self.static.copy()
        x=self.static.copy();n=len(self.rows)
        if not n:return x
        load=np.maximum(q,1);load[-1]=1;compute=self.work*load[self.jj];t=self.base+compute
        x[:,8]=compute;x[:,9]=t;x[:,10]=np.clip(1-(t+self.delay)/self.deadline,-10.,1.)
        x[:,11]=np.clip((t+self.delay)/self.deadline,-10.,10.);x[:,18]=q[self.jj]/100;x[:,19]=q[self.jj]/n;x[:,34]=q[5]/n
        for i,group in enumerate(self.groups):
            best=min(group,key=lambda k:t[k]);ratio=max(-10.,min(10.,(t[best]+self.delay)/self.problem['deadline'][i]))
            x[group,36]=ratio;x[group,37]=self.static[best,16]
        return x

    def probabilities(self,quotas,model):
        import numpy as np
        matrices={}
        keys=[];features=[]
        for counts in quotas:
            key=tuple(int(k) for k in counts)
            if key not in keys:keys.append(key);features.append(self.features(counts))
        if not len(self.rows):return {key:np.zeros((0,6),dtype=np.float64) for key in keys}
        values=model.predict_proba(np.concatenate(features))[:,list(model.classes_).index(1.)];offset=0
        for key,x in zip(keys,features):
            matrix=np.zeros_like(self.problem['base']);matrix[self.ii,self.jj]=values[offset:offset+len(x)];offset+=len(x);matrices[key]=matrix
        return matrices

def clone_search(base):
    child=copy.copy(base);child.current=base.current.copy();child.seen=set(base.seen);child.history=copy.deepcopy(base.history)
    child.best=copy.deepcopy(base.best);child.current_predicted=copy.deepcopy(base.current_predicted)
    child.assignment=tuple(base.assignment);child.current_assignment=tuple(base.current_assignment)
    if getattr(base,'_mask_cache',None) is not None:child._mask_cache=base._mask_cache.copy()
    return child

class PricedPolicy:
    def __init__(self,runtime,model,prices,powers=None):self.rt=runtime;self.model=model;self.prices=prices;self.powers=powers

    def initialize(self,rows):
        greedy,_=self.rt.greedy(rows);g,diagnostic=self.rt.common_repair(rows,greedy,128)
        base=self.rt.search_class(rows,num_uavs=5,matching_budget=16,base_repair_budget=0,initial=g)
        return g,diagnostic,base

    def run(self,rows,*,num_uavs,kind,model=None,rng=None,cap=16,trace=False):
        import numpy as np
        from utils import sf_hlm_calibrated_matching as cm
        from utils.sf_hlm_action_ranker import features
        from utils.sf_hlm_request_risk import proposal_actions
        from utils.sf_hlm_energy_price import choose
        if kind in ('random','learned'):return self.rt.truncated_search(rows,num_uavs=num_uavs,kind=kind,model=model,rng=rng,cap=cap,trace=trace)
        if num_uavs!=5:raise ValueError('Frozen five-node deployment only')
        if kind=='native_G':
            greedy,_=self.rt.greedy(rows);g,diag=self.rt.common_repair(rows,greedy,128);return g,diag,[]
        allowed=('old_pool_fee','binary','smooth','gcap_binary','gcap_smooth','binary_price','smooth_price','probability_price','mixture','mixture_price')
        if kind not in allowed:raise ValueError('Unregistered native method')
        g,repair,base=self.initialize(rows);gd=self.rt.assess(rows,g);legal=[int(k) for k in np.flatnonzero(base.action_mask()[1:])+1]
        safe=dict(rows=rows,greedy_assignment=g,initial_assignment=base.assignment,initial_counts=base.current.tolist(),energy_cap=base.cap,legal_nonstop_actions=legal)
        ids,x=features(safe,10.,problem=base.problem);proposed=proposal_actions(ids,x);quotas=cm.quota_set(base.current.tolist(),proposed)
        matching_calls=base.calls;gcap=min(base.cap,gd['energy']*1.05);fallback_counts={}
        if kind=='old_pool_fee':
            candidates=[dict(action=-1,assignment=g,predicted=gd)]
            for action in [0,*sorted(proposed)]:
                branch=clone_search(base)
                if not branch.done:branch.step(action)
                if action>0:matching_calls+=branch.calls-base.calls
                candidates.append(dict(action=action,assignment=branch.assignment,predicted=branch.best))
            unique=[];seen=set()
            for c in candidates:
                key=tuple(c['assignment'])
                if key not in seen:unique.append(c);seen.add(key)
            chosen=max(sorted(unique,key=lambda c:c['action']),key=lambda c:sum(t+.010<=r['deadline'] for t,r in zip(c['predicted']['latencies'],rows)))
        else:
            objective=kind.removeprefix('gcap_').removesuffix('_price');energy_cap=gcap if kind.startswith('gcap_') else base.cap;matrices=None
            if objective in ('probability','mixture'):
                if objective=='mixture':
                    from utils.sf_hlm_cache_mixture import CacheMixtureKernel
                    kernel=CacheMixtureKernel(rows,self.powers);matrices=kernel.probabilities([q for _,q in quotas]);fallback_counts=kernel.fallback_counts
                else:
                    kernel=LocalEdgeKernel(rows,base.current,10.,base.problem);matrices=kernel.probabilities([q for _,q in quotas],self.model)
                key=tuple(int(k) for k in base.current);prob=matrices[key];targets=self.rt.kernel(rows).targets(g)
                gscore=float(sum(prob[i,j] for i,j in enumerate(targets)))
            else:gscore=cm.score_assignment(rows,g,10.,objective,problem=base.problem)
            candidates=[dict(action=-1,assignment=g,predicted=gd,expected_success=gscore)]
            for action,counts in quotas:
                result=cm.solve(rows,counts,10.,'probability' if objective=='mixture' else objective,self.model,assess=self.rt.assess,problem=base.problem,
                    probability_matrix=matrices[tuple(counts)] if matrices is not None else None)
                matching_calls+=int(bool(rows))
                if result is not None and result['predicted']['energy']<=energy_cap+1e-7:candidates.append(dict(action=action,**result))
            if kind.endswith('_price'):
                parameters=self.prices['prices'][objective];simple=[dict(action=c['action'],score=c['expected_success'],energy=c['predicted']['energy']) for c in candidates]
                index=choose(simple,parameters['price'],parameters['normalizer'])
            else:index=cm.select([c['expected_success'] for c in candidates],[c['predicted']['energy'] for c in candidates])
            chosen=candidates[index]
        diagnostic=dict(chosen_predicted=chosen['predicted'],reference_predicted=base.reference,common_energy_cap=base.cap,
            matching_calls=matching_calls,matching_budget=9 if kind=='old_pool_fee' else 10,base_evaluations=repair['base_evaluations'],
            selected_action=chosen['action'],nominal_cost_ms=10.,g_anchored_energy=kind.startswith('gcap_'),price_applied=kind.endswith('_price'),
            expected_success=chosen.get('expected_success'),hgb_batches=int(kind=='probability_price'),mixture_fallback_counts=fallback_counts)
        if matching_calls>diagnostic['matching_budget']:raise ValueError('Native matching limit exceeded')
        return tuple(chosen['assignment']),diagnostic,candidates if trace else []
