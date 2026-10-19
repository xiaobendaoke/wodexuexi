import importlib
import unittest
from itertools import product
import json
from pathlib import Path
import tempfile

import numpy as np
import config
from environment.env import Env
from environment.request_types import Request
from utils.snapshot_evaluator import capture, evaluate
from utils.chapter4 import objective, solve_snapshot, PROTOCOL


def small_snapshot():
    env = Env(scenario_seed=19)
    env.reset(initial_positions=[[100, 100], [310, 100], [520, 100], [100, 500], [520, 500]])
    for ue in env.ues:
        ue.current_request = Request.energy()
        ue.assigned = False
    for uav in env.uavs:
        uav._current_covered_ues = []
        uav.set_neighbors(env.uavs)
    for i in range(2):
        ue = env.ues[i]
        ue.pos[:2] = env.uavs[i].pos[:2] + [10, 0]
        ue.current_request = Request.service(50000, i, 1.0, 1)
        ue.assigned = True
        env.uavs[i]._current_covered_ues = [ue]
    return capture(env)


class TestChapter4(unittest.TestCase):
    def setUp(self):
        importlib.reload(config)

    def tearDown(self):
        importlib.reload(config)

    def test_objective_priority_and_normalization(self):
        def result(latency, energy):
            return {'requests': [{'latency': latency, 'deadline': 1., 'success': latency <= 1.}],
                    'metrics': {'lower_assignment_energy': energy}}
        self.assertLess(objective(result(.99, 1e12)), objective(result(1.01, 0)))
        self.assertLess(objective(result(.2, 1e12)), objective(result(.3, 0)))
        self.assertEqual(objective(result(.2, 25000)), (0, .2, 1.))
        self.assertEqual(PROTOCOL['energy_reference_j'], 25000.)
        with self.assertRaises(ValueError):
            objective(result(float('nan'), 0))

    def test_all_methods_share_snapshot_and_exact_reference(self):
        snapshot = small_snapshot()
        results = solve_snapshot(snapshot, max_rounds=30, exact_budget=1000)
        self.assertEqual(set(results), {'heuristic_control', 'greedy_single_pass', 'iterative_base', 'mc_itr', 'exact_small'})
        exact = results['exact_small']
        self.assertTrue(exact['optimality_complete'])
        brute = min(objective(evaluate(snapshot, a)) for a in product(*snapshot.candidates))
        self.assertEqual(exact['objective'], brute)
        initial = results['greedy_single_pass']['assignment']
        self.assertEqual(results['mc_itr']['initial_assignment'], initial)
        self.assertEqual(results['iterative_base']['initial_assignment'], initial)
        self.assertTrue(all(a > b for a, b in zip(results['mc_itr']['trace'], results['mc_itr']['trace'][1:])))
        self.assertTrue(all(r['objective'] >= exact['objective'] for r in results.values()))
        final = tuple(results['mc_itr']['assignment'])
        for i, candidates in enumerate(snapshot.candidates):
            for target in candidates:
                neighbor = final[:i] + (target,) + final[i+1:]
                self.assertGreaterEqual(objective(evaluate(snapshot, neighbor)), results['mc_itr']['objective'])
        per_request = evaluate(snapshot, final)
        self.assertAlmostEqual(sum(r['assignment_energy'] for r in per_request['requests']),
                               per_request['metrics']['lower_assignment_energy'], places=8)

    def test_exact_does_not_claim_completion_when_budget_is_insufficient(self):
        results = solve_snapshot(small_snapshot(), max_rounds=0, exact_budget=1)
        self.assertFalse(results['exact_small']['optimality_complete'])
        self.assertFalse(results['mc_itr']['local_optimality_verified'])

    def test_frozen_trajectory_is_part_of_snapshot_identity(self):
        env = Env(scenario_seed=19)
        env.reset()
        stationary = capture(env)
        moving = capture(env, np.full((config.NUM_UAVS, 2), .1))
        self.assertNotEqual(stationary.snapshot_hash, moving.snapshot_hash)

    def test_suite_resume_and_result_integrity(self):
        from run_chapter4_suite import plan, run
        with tempfile.TemporaryDirectory() as tmp:
            specification = plan('exact', smoke=True)
            report = run(specification, tmp)
            self.assertEqual(len(report['units']), 5)
            self.assertTrue(all(c['ci95'] is None for c in report['comparisons']))
            before = (Path(tmp) / 'analysis.json').read_bytes()
            run(specification, tmp)
            self.assertEqual(before, (Path(tmp) / 'analysis.json').read_bytes())
            data = next(Path(tmp).glob('exact_small/workload_*/episode_*.json'))
            data.write_text('{}')
            with self.assertRaises((ValueError, KeyError)):
                run(specification, tmp)
