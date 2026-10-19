import unittest
import numpy as np
import config
from environment.env import Env
from environment.request_types import REQUEST_TYPE_SERVICE


class TestNoSilentTruncation(unittest.TestCase):
    """TEST-11: No Silent Capacity Truncation (SSOT Section 8)
    
    Verifies that all admitted service requests (up to NUM_UES = 100)
    are 100% visible to and decided by the lower offload policy without
    silent fallback to heuristics:
        SILENT_TRUNCATION_ALLOWED = NO
        MAX_OFFLOAD_REQUESTS_PER_UAV = NUM_UES = 100
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_fifty_admitted_requests_all_decided_by_policy(self):
        """When 50 service requests are covered by UAV 0 and external offload actions
        are provided, all 50 requests must be learned policy decisions.
        Heuristic fallback count must be strictly 0.
        
        Legacy bug: MAX_OFFLOAD_REQUESTS_PER_UAV = 30; requests 31-50 are silently
        truncated and processed via heuristic fallback.
        """
        uav0 = self.env.uavs[0]
        # Attach 50 service UEs to UAV 0
        admitted_ues = []
        for i in range(50):
            ue = self.env.ues[i]
            ue.pos[:2] = uav0.pos[:2] + np.array([5.0 + i * 0.5, 0.0], dtype=np.float32)
            ue.current_request.req_type = REQUEST_TYPE_SERVICE
            ue.current_request.req_id = i % config.NUM_SERVICES
            ue.current_request.req_size = 1000
            admitted_ues.append(ue)

        uav0._current_covered_ues = admitted_ues
        for u in self.env.uavs[1:]:
            u._current_covered_ues = []

        # Check config capacity first: canonical requires >= 50 (specifically 100)
        # Legacy config has MAX_OFFLOAD_REQUESTS_PER_UAV = 30 (EXPECTED_RED)
        self.assertGreaterEqual(
            config.MAX_OFFLOAD_REQUESTS_PER_UAV, 50,
            f"MAX_OFFLOAD_REQUESTS_PER_UAV is {config.MAX_OFFLOAD_REQUESTS_PER_UAV} < 50; legacy silent truncation active"
        )

        # Provide actions for 50 requests
        actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
        _, _, metrics = self.env.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32), offloading_actions=actions)

        # Under canonical semantics:
        # All 50 requests must be learned decisions
        self.assertEqual(
            int(metrics["service_learned_decision_count"]), 50,
            f"Expected 50 learned decisions, got {metrics['service_learned_decision_count']}"
        )
        self.assertEqual(
            int(metrics["service_heuristic_decision_count"]), 0,
            f"Expected 0 heuristic fallbacks, got {metrics['service_heuristic_decision_count']}"
        )


if __name__ == "__main__":
    unittest.main()
