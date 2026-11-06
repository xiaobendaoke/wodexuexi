import copy
import unittest
import numpy as np

from environment.queued_service import Engine, Protocol, Task
from utils.queued_service_controllers import Controller
from utils.queued_quality_amortization import (
    analytical_arrival_outcomes,
    evaluate_amortized_bank,
    expand_safe_plan_bank,
    safe_assignments,
    should_expand_rescue,
)


def make_congested_engine():
    protocol = Protocol()
    positions = [(150., 150.), (400., 150.), (600., 150.), (200., 450.), (500., 450.)]
    ues = {0: (150., 150.), 1: (160., 150.), 2: (500., 450.)}
    eng = Engine(protocol, positions, ues)
    # Add heavy active compute backlog on UAV 0 so UAV 0 is congested (>1.2s of work)
    eng.add([
        Task(1, 0, 0., 2.5, 12e6, 60e9),
        Task(2, 0, 0., 2.8, 12e6, 60e9),
        Task(3, 1, 0., 0.35, 24e6, 24e9),  # Doomed task on UAV 0 (slack 0.35s)
    ])
    eng.admit()
    # Assign tasks 1 and 2 to UAV 0 first
    eng.assign({1: 0, 2: 0})
    return eng


class QualityAmortizationTests(unittest.TestCase):
    def test_safe_assignments_isolates_doomed_task_from_congested_uav(self):
        eng = make_congested_engine()
        obs = eng.observe()
        plan = safe_assignments(obs)
        self.assertIn(3, plan)
        # Task 3 is doomed and UAV 0 has >1.2s of queued work, so it must not be assigned to UAV 0
        self.assertNotEqual(plan[3], 0)
        # Resulting assignment must be legal in Engine
        eng.assign(plan)

    def test_expand_safe_plan_bank_preserves_normalization(self):
        eng = make_congested_engine()
        obs = eng.observe()
        plans = [{3: 0}]
        prior = np.array([1.0])
        new_plans, new_prior = expand_safe_plan_bank(obs, plans, prior)
        self.assertEqual(len(new_plans), 2)
        self.assertAlmostEqual(float(new_prior.sum()), 1.0, places=12)
        self.assertIn(safe_assignments(obs), new_plans)

    def test_should_expand_rescue_dual_gate(self):
        self.assertTrue(should_expand_rescue(0.50, 18.0))
        self.assertTrue(should_expand_rescue(0.85, 25.0))
        self.assertFalse(should_expand_rescue(0.86, 18.0))
        self.assertFalse(should_expand_rescue(0.40, 25.5))

    def test_analytical_arrival_outcomes_and_amortized_bank(self):
        eng = make_congested_engine()
        obs = eng.observe()
        ctrl = Controller('history_queue')
        ctrl._update_history(obs)
        plan = safe_assignments(obs)
        zero = [(0., 0.)] * len(obs['positions'])
        fut_suc, fut_over = analytical_arrival_outcomes(obs, ctrl, plan, zero, 20.0)
        self.assertGreaterEqual(fut_suc, 0.0)
        self.assertGreaterEqual(fut_over, 0.0)
        before = copy.deepcopy(obs)
        scores = evaluate_amortized_bank(
            obs, ctrl, [plan], [zero], pressure=0.6, rate3=15.0, base_neural_scores=np.zeros((1, 1))
        )
        self.assertEqual(scores.shape, (1, 1))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(obs['time'], before['time'])
        self.assertEqual(len(obs['tasks']), len(before['tasks']))


if __name__ == '__main__':
    unittest.main()
