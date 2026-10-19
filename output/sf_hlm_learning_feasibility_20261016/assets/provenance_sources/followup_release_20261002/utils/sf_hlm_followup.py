"""Common-MBS snapshot diagnostics. Historical search implementations stay frozen."""
import time
import math
import numpy as np
from utils.deadline_repair import assess
from utils.deadline_repair_online import greedy_assignment
from utils.sf_hlm_matching import allocation_counts, problem_from_rows, solve_fixed_counts
from utils.sf_hlm_search import transfer_actions
from utils.sf_hlm_v2 import CompleteQuotaSearch, graph, validate_metadata

CONTRACT = 'raw_greedy_mbs_cardinality_every_search_state_v1'


def common_mbs_initial(rows, *, num_uavs, repair_budget=128, started=None, wall_seconds=None):
    """Ordinary single-request repair restricted to UAV-to-UAV migrations.

    MBS request identities can subsequently change in conditional matching.
    Raw greedy is always a feasible fallback and defines both common guards.
    """
    validate_metadata(rows)
    if type(repair_budget) is not int or repair_budget<0:raise ValueError('Invalid repair budget')
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds<=0):raise ValueError('Invalid wall budget')
    started = time.perf_counter() if started is None else started
    initial, evaluations = greedy_assignment(rows)
    reference = assess(rows, initial)
    cap = 1.05*reference['energy']
    target = int(allocation_counts(rows, initial, num_uavs)[-1])
    chosen = initial; best = reference; calls = 0
    while calls < repair_budget:
        improved = None
        for i,row in enumerate(rows):
            if row['options'][chosen[i]]['target'] == 'mbs':continue
            for action, option in row['options'].items():
                if action == chosen[i] or option['target'] == 'mbs':continue
                if calls >= repair_budget or (wall_seconds is not None and time.perf_counter()-started >= wall_seconds):break
                candidate = list(chosen);candidate[i] = action;candidate = tuple(candidate)
                score = assess(rows, candidate);calls += 1
                if score['energy'] <= cap and score['objective'] < best['objective']:
                    if improved is None or score['objective'] < improved[1]['objective']:
                        improved = candidate,score
            if calls >= repair_budget:break
        if improved is None:break
        chosen,best = improved
    if allocation_counts(rows, chosen, num_uavs)[-1] != target:raise RuntimeError('MBS initial count changed')
    return chosen,dict(contract=CONTRACT, fixed_mbs_count=target, reference_predicted=reference,
        chosen_predicted=best, common_energy_cap=cap, base_evaluations=calls,
        initial_evaluations=evaluations, matching_calls=0, matching_budget=0)


class FixedMBSSearch(CompleteQuotaSearch):
    """The complete PPO action vocabulary with MBS transfers actually masked."""
    def __init__(self, rows, *, num_uavs=5, matching_budget=16, wall_seconds=None, base_repair_budget=128):
        if type(matching_budget) is not int or matching_budget < 0:raise ValueError('Invalid matching budget')
        if type(base_repair_budget) is not int or base_repair_budget < 0:raise ValueError('Invalid repair budget')
        self.started=time.perf_counter();self.rows=rows;self.num_uavs=num_uavs
        self.problem=problem_from_rows(rows,num_uavs);self.wall_seconds=wall_seconds
        self.assignment, d = common_mbs_initial(rows,num_uavs=num_uavs,repair_budget=base_repair_budget,
                                               started=self.started,wall_seconds=wall_seconds)
        self.reference=d['reference_predicted'];self.cap=d['common_energy_cap'];self.fixed_mbs_count=d['fixed_mbs_count']
        self.base_evaluations=d['base_evaluations'];self.initial_evaluations=d['initial_evaluations'];self.best=d['chosen_predicted']
        self.current=allocation_counts(rows,self.assignment,num_uavs)
        self.current_assignment=self.assignment;self.current_predicted=self.best
        self.calls=0;self.budget=matching_budget;self.actions=transfer_actions(num_uavs+1)
        self.history=[];self.seen=set();self.done=not rows or matching_budget==0;self.reason='empty_or_zero_budget' if self.done else None
        if wall_seconds is not None and time.perf_counter()-self.started >= wall_seconds:
            self.done=True;self.reason='wall_budget'
        if not self.done:
            candidate=solve_fixed_counts(rows,self.current,num_uavs,problem=self.problem)
            self.calls=1;self.seen.add(tuple(self.current))
            if candidate is not None and candidate['predicted']['energy'] <= self.cap and candidate['predicted']['objective'][:2] < self.best['objective'][:2]:
                self.assignment=candidate['assignment'];self.current_assignment=self.assignment
                self.best=candidate['predicted'];self.current_predicted=self.best
            if self.calls>=self.budget:self.done=True;self.reason='matching_budget'
        self._assert_count()

    def _assert_count(self):
        for assignment in (self.assignment,self.current_assignment):
            if allocation_counts(self.rows,assignment,self.num_uavs)[-1] != self.fixed_mbs_count:
                raise RuntimeError('Search or fallback changed common MBS count')

    def action_mask(self):
        mask=super().action_mask()
        for k,(source,destination,_) in enumerate(self.actions[1:],1):
            if self.num_uavs in (source,destination):mask[k]=False
        return mask

    def step(self, action):
        value=super().step(action);self._assert_count();return value

    def summary(self):
        self._assert_count()
        return dict(**super().summary(),contract=CONTRACT,fixed_mbs_count=self.fixed_mbs_count,
                    chosen_mbs_count=int(allocation_counts(self.rows,self.assignment,self.num_uavs)[-1]))


def fixed_mbs_search(rows, *, num_uavs, kind, model=None, rng=None, budget=16, wall_seconds=None):
    if kind=='greedy':
        a,d=common_mbs_initial(rows,num_uavs=num_uavs,wall_seconds=wall_seconds)
        return a,d,[]
    engine=FixedMBSSearch(rows,num_uavs=num_uavs,matching_budget=budget,wall_seconds=wall_seconds)
    records=[]
    while not engine.done:
        state=graph(engine);choices=np.flatnonzero(state['mask'][1:])+1;lp=value=0.
        if kind=='learned':
            if model is None:raise ValueError('Learning diagnostic requires a checkpoint')
            action,lp,value=model.act(state,exploration=False)
        elif not len(choices):action=0
        elif kind=='random':
            if rng is None:raise ValueError('Random diagnostic requires an explicit RNG')
            action=int(rng.choice(choices))
        elif kind=='heuristic':
            lat=engine.problem['base']+engine.problem['work']*np.r_[np.maximum(engine.current[:-1],1),1][None,:]
            risk=((lat>engine.problem['deadline'][:,None]) & engine.problem['legal']).sum(0)
            action=int(min(choices,key=lambda k:(-risk[engine.actions[k][0]],risk[engine.actions[k][1]],engine.actions[k][2],int(k))))
        else:raise ValueError('Unknown restricted policy')
        reward,done=engine.step(int(action))
        records.append(dict(state=state,action=int(action),log_prob=lp,value=value,reward=reward,done=done))
    return engine.assignment,engine.summary(),records
