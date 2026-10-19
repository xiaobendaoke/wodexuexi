"""Deployment-only caching for immutable observed SF-HLM batches.

The archived implementations remain external read-only inputs. This context
temporarily replaces selected module bindings in a single evaluation process;
it restores them on exit, never updates weights and never reads private truth.
Call-cache construction and clearing belong inside the full deployment timer.
"""
from __future__ import annotations
import math
import sys
import time


class RowKernel:
    def __init__(self,rows,num_uavs=5):
        import numpy as np
        from utils.sf_hlm_matching import problem_from_rows
        from utils.sf_hlm_v2 import validate_metadata
        validate_metadata(rows);self.rows=rows;self.num_uavs=num_uavs
        self.problem=problem_from_rows(rows,num_uavs);self.indices=[];self.greedy_result=None
        self.indices=[{a:int(num_uavs if o['target']=='mbs' else o['target']) for a,o in row['options'].items()} for row in rows]
        self.ii,self.jj=np.nonzero(self.problem['legal']);self.feedback_cache={}
        self.request=None;self.edges=None

    def ensure_graph(self):
        if self.request is not None:return
        import numpy as np
        rows=self.rows;num_uavs=self.num_uavs;n=len(rows);j=num_uavs+1;self.request=np.zeros((n,4),np.float32)
        self.edges=np.zeros((n,j,14),np.float32)
        for i,row in enumerate(rows):
            self.request[i]=[min(row['deadline'],10.),math.log1p(row['cycles'])/30,
                             math.log1p(row['size'])/20,row['source']/max(num_uavs,1)]
            for target in np.flatnonzero(self.problem['legal'][i]):
                option=row['options'][int(self.problem['action_ids'][i,target])]
                # Retain the scalar math/float32 rounding of the archived graph.
                self.edges[i,target,0]=option['base']/row['deadline']
                self.edges[i,target,1]=option['work']/row['deadline']
                self.edges[i,target,2]=math.log1p(option['energy'])/20
                self.edges[i,target,3]=math.log1p(option['compute_energy'])/30
                self.edges[i,target,7]=np.clip((row['deadline']-option['base'])/max(option['work'],1e-12)/max(n,1),-1,2)
                self.edges[i,target,8:12]=[float(target==num_uavs),option['cache_belief'],
                    math.log1p(option['capacity'])/30,float(target==row['source'])]

    def targets(self,assignment):
        import numpy as np
        if len(assignment)!=len(self.rows) or any(type(a) is not int or a not in row['options'] for row,a in zip(self.rows,assignment)):
            raise ValueError('Every admitted request requires a legal Python-integer action')
        return np.asarray([index[a] for index,a in zip(self.indices,assignment)],dtype=np.int64)

    def assess(self,assignment):
        import numpy as np
        targets=self.targets(assignment);n=len(self.rows);idx=np.arange(n)
        counts=np.bincount(targets,minlength=self.num_uavs+1);counts[-1]=1
        load=counts[targets];p=self.problem
        latency=p['base'][idx,targets]+p['work'][idx,targets]*load
        energy=p['energy'][idx,targets]+p['compute_energy'][idx,targets]/load**2
        misses=int(np.count_nonzero(latency>p['deadline']))
        energy_total=math.fsum(energy);ratio=math.fsum(latency/p['deadline'])/max(n,1)
        return dict(misses=misses,energy=energy_total,latency_ratio=ratio,latencies=latency.tolist(),
                    objective=(misses,energy_total,ratio))


class FastRuntime:
    """One worker only; no nested use, training, wall-time budgets or row mutation."""
    def __init__(self):
        self.kernels={};self.patches=[];self.active=False;self.search_class=None
        self.sparse_calls=0;self.sparse_fallbacks=0

    def clear(self):
        self.kernels.clear();self.sparse_calls=0;self.sparse_fallbacks=0

    def kernel(self,rows,num_uavs=5):
        old=self.kernels.get(id(rows))
        if old is not None and old.rows is rows:return old
        kernel=RowKernel(rows,num_uavs);self.kernels[id(rows)]=kernel;return kernel

    def assess(self,rows,assignment):return self.kernel(rows).assess(assignment)

    def greedy(self,rows):
        kernel=self.kernel(rows)
        if kernel.greedy_result is None:kernel.greedy_result=self.original_greedy(rows)
        return kernel.greedy_result

    def feedback(self,rows,assignment,num_uavs,*,energy_reference):
        if not math.isfinite(energy_reference) or energy_reference<=0:raise ValueError('Invalid energy normalization')
        kernel=self.kernel(rows,num_uavs);key=(tuple(assignment),float(energy_reference))
        if key not in kernel.feedback_cache:
            kernel.feedback_cache[key]=self.original_feedback(rows,assignment,num_uavs,energy_reference=energy_reference)
        return kernel.feedback_cache[key].copy()

    def common_repair(self,rows,initial,budget=128,wall_seconds=None,started=None):
        if wall_seconds is not None:raise ValueError('This frozen equivalence experiment uses matching budgets, not wall budgets')
        if type(budget) is not int or budget<0:raise ValueError('Invalid ordinary repair budget')
        from utils.deadline_repair import ordered_moves
        kernel=self.kernel(rows);greedy,_=self.greedy(rows);reference=kernel.assess(greedy);cap=reference['energy']*1.05
        current=tuple(initial);cost=kernel.assess(current)
        if cost['energy']>cap:current=greedy;cost=reference
        seen={current};evaluations=0
        while evaluations<budget:
            chosen,best=current,cost
            for i,action in ordered_moves(rows,current,'ordinary')[0]:
                candidate=current[:i]+(action,)+current[i+1:]
                if candidate in seen:continue
                if evaluations>=budget:break
                seen.add(candidate);evaluations+=1;value=kernel.assess(candidate)
                if value['energy']<=cap and value['objective']<best['objective']:chosen,best=candidate,value
            if chosen==current:break
            current,cost=chosen,best
        return current,dict(chosen_predicted=cost,reference_predicted=reference,common_energy_cap=cap,
                            base_evaluations=evaluations,matching_calls=0,matching_budget=0)

    def graph(self,engine):
        import numpy as np
        kernel=self.kernel(engine.rows,engine.num_uavs);p=kernel.problem;n=max(len(engine.rows),1);j=engine.num_uavs+1
        kernel.ensure_graph()
        ii,jj=kernel.ii,kernel.jj;edges=kernel.edges.copy();load=np.maximum(engine.current,1).copy();load[-1]=1
        ratio=(p['base'][ii,jj]+p['work'][ii,jj]*load[jj])/p['deadline'][ii]
        edges[ii,jj,4]=engine.current[jj]/n;edges[ii,jj,5]=np.minimum(ratio,10.)
        edges[ii,jj,6]=np.clip(1-ratio,-10,1)
        for column,assignment in enumerate((engine.current_assignment,engine.assignment),12):
            targets=kernel.targets(assignment);edges[np.arange(len(engine.rows)),targets,column]=1.
        current=self.feedback(engine.rows,engine.current_assignment,engine.num_uavs,energy_reference=max(engine.cap,1e-12))
        best=self.feedback(engine.rows,engine.assignment,engine.num_uavs,energy_reference=max(engine.cap,1e-12))
        extra=np.tile([(engine.budget-engine.calls)/max(engine.budget,1),engine.current_predicted['misses']/n,engine.best['misses']/n],(j,1))
        nodes=np.concatenate([current,best,engine.current[:,None]/n,extra],axis=1).astype(np.float32)
        return dict(requests=kernel.request.copy(),edges=edges,legal=p['legal'].copy(),nodes=nodes,mask=engine.action_mask())

    def truncated_search(self,rows,*,num_uavs,kind,model=None,rng=None,cap=16,trace=False):
        import numpy as np
        if (cap not in (8,16) and not (cap==10 and kind=='random')) or kind not in ('learned','random','heuristic'):raise ValueError('Unregistered deployment search')
        engine=self.search_class(rows,num_uavs=num_uavs,matching_budget=16,base_repair_budget=128);records=[]
        while not engine.done and engine.calls<cap:
            if kind=='learned':
                state=self.graph(engine);action,_,_=model.act(state,exploration=False)
            else:
                choices=np.flatnonzero(engine.action_mask()[1:])+1
                if not len(choices):action=0
                elif kind=='random':action=int(rng.choice(choices))
                else:
                    p=engine.problem;load=np.r_[np.maximum(engine.current[:-1],1),1]
                    risk=(((p['base']+p['work']*load[None,:])>p['deadline'][:,None])&p['legal']).sum(0)
                    action=min(choices,key=lambda k:(-risk[engine.actions[k][0]],risk[engine.actions[k][1]],engine.actions[k][2],int(k)))
            engine.step(int(action))
            if trace:records.append(dict(action=int(action),state=state if kind=='learned' else None))
        if not engine.done:engine.done=True;engine.reason='external_matching_cap'
        diagnostic=engine.summary();diagnostic.update(matching_budget=cap,feature_budget=16,prefix=None)
        return engine.assignment,diagnostic,records

    def complete_policy(self,core,env,obs,spec,models,lower,runner):
        started=time.perf_counter();self.clear()
        out=core.complete_policy(env,obs,spec,models,lower,runner)
        out[2]['runtime_stats']=dict(sparse_forwards=self.sparse_calls,dense_fallbacks=self.sparse_fallbacks)
        out[2]['policy_seconds']=time.perf_counter()-started
        return out

    def sparse_forward(self,model,states):
        """Same legal-edge message equations, with dense fallback near argmax ties.

        Arithmetic kernels can round differently with smaller matrix batches.
        Real-state parity is therefore verified separately; this is not a claim
        of bitwise equality of hidden tensors. Weights and graph inputs match.
        """
        import torch
        from marl_models.sf_hlm_v2 import collate
        net=model.network;g=collate(states)
        if len(states)!=1:return net(g)
        self.sparse_calls+=1
        legal=g['legal'];valid=legal.any(-1)[...,None].float();weight=legal[...,None].float()
        req=net.request(g['requests'])*valid;node=net.node(g['nodes'])
        bi,ri,ji=torch.nonzero(legal,as_tuple=True);edge=net.edge(g['edges'][bi,ri,ji])
        b,r,j=legal.shape;hidden=net.hidden
        for message,ru,nu in zip(net.messages,net.req_updates,net.node_updates):
            pair=torch.cat([req[bi,ri],node[bi,ji],edge],-1)
            msg=message(pair);dense=msg.new_zeros((b,r,j,hidden));dense[bi,ri,ji]=msg
            req_msg=dense.sum(2)/weight.sum(2).clamp_min(1.)
            node_msg=dense.sum(1)/weight.sum(1).clamp_min(1.)
            req=ru(torch.cat([req,req_msg],-1))*valid;node=nu(torch.cat([node,node_msg],-1))
        pooled_req=req.sum(1)/valid.sum(1).clamp_min(1.);pooled=torch.cat([pooled_req,node.mean(1)],-1)
        choices=torch.nonzero(g['mask'][0,1:],as_tuple=True)[0]
        action_features=torch.cat([node[0,net.sources[choices]],node[0,net.destinations[choices]],
            pooled_req[0].expand(len(choices),-1),net.amounts[choices]],-1)
        logits=pooled.new_full(g['mask'].shape,-1e9);logits[:,0]=net.stop(pooled).squeeze(-1)
        logits[0,choices+1]=net.transfer(action_features).squeeze(-1)
        top=torch.topk(logits,2,dim=-1).values
        if float((top[:,0]-top[:,1]).min())<1e-4:
            self.sparse_fallbacks+=1;return net(g)
        return logits,net.value(pooled).squeeze(-1)

    def enable_sparse(self,model):
        if not self.active:raise RuntimeError('Enable inference within the runtime context')
        def forward(states):return self.sparse_forward(model,states)
        self.patch(model,'forward_states',forward)

    def patch(self,owner,name,value):
        self.patches.append((owner,name,getattr(owner,name)));setattr(owner,name,value)

    def __enter__(self):
        if sys.platform!='linux':raise RuntimeError('Numerical deployment implementation is evaluated only on remote Linux')
        if self.active:raise RuntimeError('Nested runtime context is not supported')
        import numpy as np
        import utils.deadline_repair_online as online
        import utils.sf_hlm_search as search
        import utils.sf_hlm_matching as matching
        import utils.sf_hlm_v2 as v2
        self.original_greedy=online.greedy_assignment;self.original_feedback=matching.feedback
        original_class=v2.CompleteQuotaSearch;runtime=self
        class CachedSearch(original_class):
            def __init__(self,rows,**kwargs):
                if kwargs.get('wall_seconds') is not None:raise ValueError('No wall-budget protocol here')
                if kwargs.get('num_uavs',5)!=5:raise ValueError('Frozen scenario has five UAVs')
                self._mask_cache=None
                super().__init__(rows,**kwargs)
                self._legal_counts=self.problem['legal'].sum(axis=0)
                self._deltas=np.zeros((len(self.actions)-1,self.num_uavs+1),np.int64)
                for k,(i,j,amount) in enumerate(self.actions[1:]):self._deltas[k,i]=-amount;self._deltas[k,j]=amount

            def action_mask(self):
                if self._mask_cache is None:
                    mask=np.zeros(len(self.actions),bool);mask[0]=True
                    if not self.done:
                        q=self.current[None,:]+self._deltas
                        valid=(q>=0).all(1)&(q<=self._legal_counts[None,:]).all(1)
                        for k in np.flatnonzero(valid):
                            if tuple(q[k]) not in self.seen:mask[k+1]=True
                    self._mask_cache=mask
                return self._mask_cache.copy()

            def step(self,action):
                out=super().step(action);self._mask_cache=None;return out
        self.search_class=CachedSearch
        self.patch(search,'assess',self.assess);self.patch(matching,'assess',self.assess)
        self.patch(search,'common_repair',self.common_repair)
        self.patch(search,'greedy_assignment',self.greedy);self.patch(online,'greedy_assignment',self.greedy)
        self.patch(matching,'feedback',self.feedback)
        self.patch(v2,'CompleteQuotaSearch',CachedSearch);self.patch(v2,'graph',self.graph)
        self.active=True;return self

    def __exit__(self,*_):
        for owner,name,old in reversed(self.patches):setattr(owner,name,old)
        self.patches.clear();self.clear();self.active=False
