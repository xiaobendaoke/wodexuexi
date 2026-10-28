"""Public two-feature entry gate around the unchanged sealed PricedPolicy."""
from __future__ import annotations
import time

from utils.sf_hlm_priced_runtime import PricedPolicy


def feature_pair(rows, repair, use_quality=True):
    n = len(rows)
    if not use_quality:
        return n, None
    latencies = repair['chosen_predicted']['latencies']
    if len(latencies) != n:
        raise ValueError('G prediction/request identity mismatch')
    q = sum(t + .010 <= r['deadline'] for t, r in zip(latencies, rows))/max(n, 1)
    return n, q


def enter(rule, n, q):
    if rule.get('fallback_full', False):
        return True
    mode = rule['mode']
    if mode == 'A':
        return n >= rule['n0']
    if mode == 'B':
        return q < rule['q0']
    if mode == 'C':
        return n >= rule['n0'] and q < rule['q0']
    raise ValueError('Unregistered gate mode')


class EntryGatePolicy(PricedPolicy):
    def __init__(self, runtime, prices, rule, force=None):
        super().__init__(runtime, None, prices, prices['powers'])
        if force not in (None, 'full', 'skip'):
            raise ValueError('Invalid engineering control')
        self.rule, self.force, self._prepared = dict(rule), force, None
        self.last_trace = None

    def initialize(self, rows):
        if self._prepared is None or self._prepared[0] is not rows:
            raise RuntimeError('Full branch must reuse this call\'s G')
        _, g, repair = self._prepared
        base = self.rt.search_class(rows, num_uavs=5, matching_budget=16,
                                    base_repair_budget=0, initial=g)
        return g, repair, base

    def run(self, rows, *, num_uavs, kind, model=None, rng=None, cap=16, trace=False):
        if num_uavs != 5 or kind != 'mixture_price' or cap != 10:
            raise ValueError('Frozen entry gate contract')
        if self._prepared is not None:
            raise RuntimeError('Entry gate is not reentrant')
        started = time.perf_counter()
        greedy, _ = self.rt.greedy(rows)
        g, repair = self.rt.common_repair(rows, greedy, 128)
        after_g = time.perf_counter()
        n, q = feature_pair(rows, repair, self.rule.get('mode') != 'A')
        proceed = (self.force == 'full') if self.force else enter(self.rule, n, q)
        proceed = bool(rows) and proceed
        after_gate = time.perf_counter()
        if proceed:
            self._prepared = (rows, g, repair)
            try:
                result = super().run(rows, num_uavs=5, kind='mixture_price', model=model,
                                     rng=rng, cap=10, trace=trace)
            finally:
                self._prepared = None
        else:
            result = (g, repair, [])
        self.last_trace = dict(mode=self.rule.get('mode'), n=n, q=q, entered=proceed,
                               g_seconds=after_g-started, gate_seconds=after_gate-after_g,
                               force=self.force, rule=self.rule)
        return result
