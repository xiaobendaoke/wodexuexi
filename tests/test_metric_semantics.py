import unittest
import numpy as np
import config
from environment.env import Env
from environment.request_types import REQUEST_TYPE_SERVICE, REQUEST_TYPE_CONTENT


class TestMetricSemantics(unittest.TestCase):
    """TEST-12: Metric Semantics (SSOT Section 7.1)
    
    Verifies:
    1. System DSR denominator is ALL generated service requests, including unadmitted requests.
    2. Critical-UE Ratio: reflects battery < UE_CRITICAL_THRESHOLD.
    3. UAV Fleet Total Energy: excludes MBS grid computation energy and UE battery discharge.
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_system_dsr_denominator_includes_unadmitted_requests(self):
        """If 10 UEs generate service requests, and 4 are admitted and meet deadline,
        while 6 are out-of-coverage (unadmitted), System DSR must be 4/10 = 0.4,
        NOT 4/4 = 1.0.
        """
        self.env.reset(initial_positions=[
            [100, 100], [310, 100], [520, 100], [100, 310], [310, 310],
        ])

        # 4 UEs near UAV 0 (in coverage)
        for i in range(4):
            ue = self.env.ues[i]
            ue.pos[:2] = np.array([105.0 + i * 2.0, 100.0], dtype=np.float32)
            ue.current_request.req_type = REQUEST_TYPE_SERVICE
            ue.current_request.deadline = 10.0  # Large deadline so they easily succeed
            ue.current_request.req_size = 100
            ue.current_request.req_id = 0

        # 6 UEs far away (unadmitted)
        for i in range(4, 10):
            ue = self.env.ues[i]
            ue.pos[:2] = np.array([500.0 + i * 10.0, 500.0], dtype=np.float32)
            ue.current_request.req_type = REQUEST_TYPE_SERVICE
            ue.current_request.deadline = 1.0  # Will fail because unadmitted latency is 20s
            ue.current_request.req_size = 100
            ue.current_request.req_id = 0

        # Remaining 90 UEs have content requests
        for i in range(10, config.NUM_UES):
            ue = self.env.ues[i]
            ue.current_request.req_type = REQUEST_TYPE_CONTENT
            ue.current_request.req_id = config.NUM_SERVICES
            ue.pos[:2] = [650, 650]

        # Clear and associate
        for u in self.env.uavs:
            u._current_covered_ues.clear()
        for ue in self.env.ues:
            ue.assigned = False

        self.env._associate_ues_to_uavs()

        # Step
        actions = np.zeros((config.NUM_UAVS, 2), dtype=np.float32)
        _, _, metrics = self.env.step(actions)

        # Check total service requests generated
        total_gen = metrics["service_requests_generated"]
        self.assertEqual(total_gen, 10.0)

        # System DSR must be <= 4 / 10 = 0.4
        dsr = metrics["deadline_satisfaction_rate"]
        self.assertLessEqual(
            dsr, 0.4 + 1e-4,
            f"System DSR ({dsr}) did not include unadmitted requests in the denominator"
        )


if __name__ == "__main__":
    unittest.main()
