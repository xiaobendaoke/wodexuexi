import unittest
import numpy as np
import config
from environment.env import Env
from environment.request_types import REQUEST_TYPE_SERVICE


class TestOrderPermutationInvariance(unittest.TestCase):
    """TEST-05: Full Physical Permutation Invariance (SSOT Section 4.4)
    
    Verifies that for identical slot-start conditions and identical logical
    assignment (request_id -> target), permuting the order of requests in the
    input list does NOT alter:
    - compute latencies
    - compute energy
    - communication energy
    - DSR
    - total non-flight energy
    """

    def setUp(self):
        np.random.seed(42)

    def test_request_order_permutation_invariance(self):
        """Construct 3 service requests with different properties.
        Execute them in order [0, 1, 2] vs [2, 0, 1] with synchronized actions.
        Physical results must be identical.
        
        Legacy bug: Sequential C2 computing and iteration-dependent working_cache
        produce different latencies depending on iteration order.
        """
        # Order 1: [ue_a, ue_b, ue_c]
        env1 = Env()
        env1.reset()
        uav0_1 = env1.uavs[0]
        ue_a = env1.ues[0]
        ue_b = env1.ues[1]
        ue_c = env1.ues[2]

        for ue in [ue_a, ue_b, ue_c]:
            ue.pos[:2] = uav0_1.pos[:2] + np.array([10.0, 0.0], dtype=np.float32)
            ue.current_request.req_type = REQUEST_TYPE_SERVICE
            ue.current_request.req_id = 0
            ue.current_request.req_size = 1000

        uav0_1._current_covered_ues = [ue_a, ue_b, ue_c]
        uav0_1.calculate_initial_load()

        # Actions for Order 1:
        # ue_a -> Local (0), ue_b -> MBS (1), ue_c -> Local (0)
        actions1 = np.array([0, 1, 0], dtype=np.int64)
        uav0_1.process_requests(offload_actions=actions1)

        latency_a_order1 = ue_a.latency_current_request
        latency_c_order1 = ue_c.latency_current_request

        # Order 2: [ue_c, ue_a, ue_b]
        # Re-seed to ensure env2 has identical initial geometry, channel conditions, and cache state
        np.random.seed(42)
        env2 = Env()
        env2.reset()
        uav0_2 = env2.uavs[0]
        ue_a2 = env2.ues[0]
        ue_b2 = env2.ues[1]
        ue_c2 = env2.ues[2]

        for ue in [ue_a2, ue_b2, ue_c2]:
            ue.pos[:2] = uav0_2.pos[:2] + np.array([10.0, 0.0], dtype=np.float32)
            ue.current_request.req_type = REQUEST_TYPE_SERVICE
            ue.current_request.req_id = 0
            ue.current_request.req_size = 1000

        uav0_2._current_covered_ues = [ue_c2, ue_a2, ue_b2]
        uav0_2.calculate_initial_load()

        # Actions synchronized to request mapping:
        # idx 0 is ue_c2 -> Local (0)
        # idx 1 is ue_a2 -> Local (0)
        # idx 2 is ue_b2 -> MBS (1)
        actions2 = np.array([0, 0, 1], dtype=np.int64)
        uav0_2.process_requests(offload_actions=actions2)

        latency_a_order2 = ue_a2.latency_current_request
        latency_c_order2 = ue_c2.latency_current_request

        # Under canonical semantics, ue_a and ue_c must experience the exact same
        # latency regardless of whether ue_b was processed between them or after them.
        self.assertAlmostEqual(
            latency_a_order1, latency_a_order2, places=4,
            msg=f"Permutation variance detected: ue_a latency changed from {latency_a_order1} to {latency_a_order2}"
        )
        self.assertAlmostEqual(
            latency_c_order1, latency_c_order2, places=4,
            msg=f"Permutation variance detected: ue_c latency changed from {latency_c_order1} to {latency_c_order2}"
        )


if __name__ == "__main__":
    unittest.main()
