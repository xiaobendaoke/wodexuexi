"""Strict observed metadata and conditional-match search, separate v2 contract."""
from __future__ import annotations
import math
from types import SimpleNamespace
import time
import numpy as np
from utils.deadline_repair import assess
from utils.sf_hlm_matching import allocation_counts, feedback, solve_fixed_counts
from utils.sf_hlm_search import QuotaSearch, graph_state as legacy_graph

INPUT_CONTRACT = 'observed_complete_current_best_graph_v2'
SEARCH_CONTRACT = 'strict_quota_124_v2'
ARCHIVE_CONTRACT = 'critical_quota_best_recent2_group_pg_v2'
NODE_DIM = 16
EDGE_DIM = 14


def validate_metadata(rows):
    for r in rows:
        if not {'cycles','size','source','ue_id','deadline','options'}.issubset(r): raise ValueError('Missing observed request metadata')
        for key in ('cycles','size'):
            if not math.isfinite(float(r[key])) or r[key] < 0: raise ValueError('Invalid request metadata')
        for o in r['options'].values():
            if not {'cache_belief','capacity'}.issubset(o): raise ValueError('Missing observed edge metadata')
            if not math.isfinite(float(o['capacity'])) or o['capacity']<=0 or not math.isfinite(float(o['cache_belief'])) or not 0<=o['cache_belief']<=1:
                raise ValueError('Invalid observed edge metadata')


def graph(engine):
    validate_metadata(engine.rows)
    state=legacy_graph(engine); r=max(len(engine.rows),1)
    current=feedback(engine.rows,engine.current_assignment,engine.num_uavs,energy_reference=max(engine.cap,1e-12))
    best=feedback(engine.rows,engine.assignment,engine.num_uavs,energy_reference=max(engine.cap,1e-12))
    extra=np.tile([(engine.budget-engine.calls)/max(engine.budget,1),engine.current_predicted['misses']/r,engine.best['misses']/r],(engine.num_uavs+1,1))
    marks=np.zeros((*state['legal'].shape,2),np.float32)
    for column,assignment in enumerate((engine.current_assignment,engine.assignment)):
        for i,(row,action) in enumerate(zip(engine.rows,assignment)):
            target=row['options'][action]['target'];marks[i,engine.num_uavs if target=='mbs' else target,column]=1.
    state['edges']=np.concatenate([state['edges'],marks],axis=-1)
    state['nodes']=np.concatenate([current,best,engine.current[:,None]/r,extra],axis=1).astype(np.float32)
    return state


class CompleteQuotaSearch(QuotaSearch):
    def __init__(self,rows,**kwargs):
        validate_metadata(rows);super().__init__(rows,**kwargs)
        self.current_assignment=self.assignment

    def step(self,action):
        if self.done or isinstance(action,(bool,np.bool_)) or not isinstance(action,(int,np.integer)) or not 0<=action<len(self.actions) or not self.action_mask()[action]:
            raise ValueError('Illegal quota action')
        if action==0:
            self.done=True;self.reason='policy_stop';return 0.,True
        if self.wall_seconds is not None and time.perf_counter()-self.started>=self.wall_seconds:
            self.done=True;self.reason='wall_budget';return 0.,True
        before=self.best;i,j,k=self.actions[action];q=self.current.copy();q[i]-=k;q[j]+=k
        self.calls+=1;self.seen.add(tuple(q));candidate=solve_fixed_counts(self.rows,q,self.num_uavs,problem=self.problem)
        if candidate is not None and candidate['predicted']['energy']<=self.cap:
            self.current=q;self.current_assignment=candidate['assignment'];self.current_predicted=candidate['predicted']
            if candidate['predicted']['objective'][:2]<self.best['objective'][:2]:self.assignment=candidate['assignment'];self.best=candidate['predicted']
        reward=float(before['misses']-self.best['misses'])+.1*(before['energy']-self.best['energy'])/max(self.cap,1e-12)-.001
        if self.calls>=self.budget:self.done=True;self.reason='matching_budget'
        self.history.append(dict(action=int(action),reward=reward,done=self.done,feasible=candidate is not None))
        return reward,self.done


def critical_count(base,work,deadline,total):
    if work==0:return total if base<=deadline else 0
    k=max(0,min(total,int(math.floor((deadline-base)/work))))
    # Correct rounding at exact deadline boundaries using the same comparison as
    # the matching predictor. n=0 is only a sentinel, never a served request.
    while k>0 and base+work*k>deadline:k-=1
    while k<total and base+work*(k+1)<=deadline:k+=1
    return k


class CriticalArchiveSearch:
    def __init__(self,rows,*,num_uavs=5,matching_budget=16,wall_seconds=None,base_repair_budget=128,
                 breakpoints=True,multiple=True,fixed_mbs=False):
        base=CompleteQuotaSearch(rows,num_uavs=num_uavs,matching_budget=matching_budget,wall_seconds=wall_seconds,base_repair_budget=base_repair_budget)
        self.rows=rows;self.num_uavs=num_uavs;self.problem=base.problem;self.cap=base.cap;self.reference=base.reference
        self.started=base.started;self.wall_seconds=wall_seconds;self.budget=matching_budget;self.calls=base.calls
        self.base_evaluations=base.base_evaluations;self.initial_evaluations=base.initial_evaluations
        self.breakpoints=breakpoints;self.multiple=multiple;self.fixed_mbs=fixed_mbs
        self.best=dict(counts=base.current.copy(),assignment=base.assignment,predicted=base.best)
        self.initial=self.best['predicted'];self.recent=[];self.seen=set(base.seen);self.reason=base.reason
        self.done=base.done;self.history=[];self._candidates=None
        self.threshold=np.zeros_like(self.problem['base'],dtype=np.int64)
        for r,row in enumerate(rows):
            for j in np.flatnonzero(self.problem['legal'][r]):
                self.threshold[r,j]=critical_count(self.problem['base'][r,j],self.problem['work'][r,j],row['deadline'],len(rows))

    def parents(self):
        if not self.multiple:return [self.recent[-1] if self.recent else self.best]
        result=[self.best]
        for p in reversed(self.recent):
            if not any(np.array_equal(p['counts'],x['counts']) for x in result):result.append(p)
            if len(result)==3:break
        return result

    def candidates(self):
        if self._candidates is not None:return self._candidates
        result=[];proposed=set();r=len(self.rows)
        for parent,p in enumerate(self.parents()):
            q=p['counts'];targets=np.array([self.num_uavs if row['options'][a]['target']=='mbs' else row['options'][a]['target'] for row,a in zip(self.rows,p['assignment'])])
            for source in range(self.num_uavs+1):
                if q[source]==0:continue
                for dest in range(self.num_uavs+1):
                    if source==dest or (self.fixed_mbs and self.num_uavs in (source,dest)):continue
                    amounts={1,2,4}
                    if self.breakpoints:
                        if source!=self.num_uavs:
                            release=[int(q[source]-self.threshold[i,source]) for i in np.flatnonzero(targets==source)
                                     if 1<=self.threshold[i,source]<q[source]]
                            if release:amounts.add(min(release))
                        if dest!=self.num_uavs:
                            safe=[int(self.threshold[i,dest]-q[dest]) for i in np.flatnonzero(targets==dest) if self.threshold[i,dest]>q[dest]]
                            if safe:amounts.add(min(safe))
                    for amount in sorted(amounts):
                        if not 0<amount<=q[source]:continue
                        new=q.copy();new[source]-=amount;new[dest]+=amount
                        if tuple(new) in self.seen or tuple(new) in proposed or not (new<=self.problem['legal'].sum(0)).all():continue
                        proposed.add(tuple(new))
                        rescue=sum(q[source]>self.threshold[i,source]>=new[source] for i in np.flatnonzero(targets==source)) if source!=self.num_uavs else 0
                        loss=sum(q[dest]<=self.threshold[i,dest]<new[dest] for i in np.flatnonzero(targets==dest)) if dest!=self.num_uavs else 0
                        result.append((parent,source,dest,amount,rescue,loss))
        self._candidates=result;return result

    def state(self):
        parents=[]
        for p in self.parents():
            proxy=SimpleNamespace(rows=self.rows,num_uavs=self.num_uavs,problem=self.problem,cap=self.cap,
                current=p['counts'],current_assignment=p['assignment'],current_predicted=p['predicted'],
                assignment=self.best['assignment'],best=self.best['predicted'],budget=self.budget,calls=self.calls,
                action_mask=lambda:np.ones(1,dtype=bool))
            parents.append(graph(proxy))
        candidates=self.candidates()
        tokens=np.asarray(candidates if candidates else [(-1,0,0,0,0,0)],dtype=np.int64)
        return dict(parents=parents,tokens=tokens,mask=np.ones(len(tokens),bool))

    def step(self,action):
        if self.done:raise ValueError('Search already terminated')
        if self.wall_seconds is not None and time.perf_counter()-self.started>=self.wall_seconds:
            self.done=True;self.reason='wall_budget';return 0.,True
        candidates=self.candidates()
        if not candidates:
            if action!=0:raise ValueError('Invalid forced stop')
            self.done=True;self.reason='no_candidates';return 0.,True
        if isinstance(action,(bool,np.bool_)) or not isinstance(action,(int,np.integer)) or not 0<=action<len(candidates):raise ValueError('Illegal archive action')
        parent,source,dest,amount,*_=candidates[action];q=self.parents()[parent]['counts'].copy();q[source]-=amount;q[dest]+=amount
        before=self.best['predicted'];self.calls+=1;self.seen.add(tuple(q))
        candidate=solve_fixed_counts(self.rows,q,self.num_uavs,problem=self.problem)
        if candidate is not None and candidate['predicted']['energy']<=self.cap:
            self.recent.append(candidate);self.recent=self.recent[-3:]
            if candidate['predicted']['objective'][:2]<self.best['predicted']['objective'][:2]:self.best=candidate
        self._candidates=None
        if self.calls>=self.budget:self.done=True;self.reason='matching_budget'
        after=self.best['predicted'];reward=before['misses']-after['misses']+.1*(before['energy']-after['energy'])/max(self.cap,1e-12)-.001
        self.history.append(dict(action=int(action),reward=float(reward),done=self.done));return float(reward),self.done

    @property
    def assignment(self):return self.best['assignment']

    def summary(self):
        return dict(matching_calls=self.calls,matching_budget=self.budget,base_evaluations=self.base_evaluations,
            initial_evaluations=self.initial_evaluations,common_energy_cap=self.cap,reference_predicted=self.reference,
            chosen_predicted=self.best['predicted'],termination=self.reason,search_seconds=time.perf_counter()-self.started)

    def terminal_return(self):
        best=self.best['predicted'];return self.initial['misses']-best['misses']+.1*(self.initial['energy']-best['energy'])/max(self.cap,1e-12)-.001*self.calls


def search(rows,*,num_uavs=5,kind='learned',family='complete',model=None,rng=None,exploration=False,
           budget=16,wall_seconds=None,base_repair_budget=128,breakpoints=True,multiple=True,fixed_mbs=False):
    engine=CompleteQuotaSearch(rows,num_uavs=num_uavs,matching_budget=budget,wall_seconds=wall_seconds,base_repair_budget=base_repair_budget) if family=='complete' else CriticalArchiveSearch(
        rows,num_uavs=num_uavs,matching_budget=budget,wall_seconds=wall_seconds,base_repair_budget=base_repair_budget,breakpoints=breakpoints,multiple=multiple,fixed_mbs=fixed_mbs)
    records=[]
    while not engine.done:
        state=graph(engine) if family=='complete' else engine.state();lp=value=0.
        if kind=='learned':action,lp,value=model.act(state,exploration=exploration)
        elif family=='complete':
            choices=np.flatnonzero(state['mask'][1:])+1
            if not len(choices):action=0
            elif kind=='random':action=int(rng.choice(choices))
            else:
                lat=engine.problem['base']+engine.problem['work']*np.r_[np.maximum(engine.current[:-1],1),1][None,:]
                risk=((lat>engine.problem['deadline'][:,None])&engine.problem['legal']).sum(0)
                action=min(choices,key=lambda k:(-risk[engine.actions[k][0]],risk[engine.actions[k][1]],engine.actions[k][2],int(k)))
        else:
            candidates=engine.candidates()
            if not candidates:action=0
            elif kind=='random':action=int(rng.integers(len(candidates)))
            else:action=min(range(len(candidates)),key=lambda k:(-(candidates[k][4]-candidates[k][5]),engine.parents()[candidates[k][0]]['predicted']['misses'],candidates[k][3],k))
        reward,done=engine.step(int(action));records.append(dict(state=state,action=int(action),log_prob=lp,value=value,reward=reward,done=done))
    diagnostic=engine.summary()
    if family!='complete':diagnostic['terminal_return']=engine.terminal_return()
    return engine.assignment,diagnostic,records


def teacher(rows,num_uavs,total_budget=128):
    engine=CompleteQuotaSearch(rows,num_uavs=num_uavs,matching_budget=16);calls=0;samples=[]
    while not engine.done and calls+engine.calls<total_budget:
        state=graph(engine);best=engine.current_predicted['objective'][:2];chosen=0
        for action in np.flatnonzero(state['mask'][1:])+1:
            if calls+engine.calls+1>=total_budget:break
            i,j,k=engine.actions[action];q=engine.current.copy();q[i]-=k;q[j]+=k
            c=solve_fixed_counts(rows,q,num_uavs,problem=engine.problem);calls+=1
            if c is not None and c['predicted']['energy']<=engine.cap and c['predicted']['objective'][:2]<best:chosen=int(action);best=c['predicted']['objective'][:2]
        samples.append(dict(state=state,action=chosen));engine.step(chosen)
    return samples,calls+engine.calls
