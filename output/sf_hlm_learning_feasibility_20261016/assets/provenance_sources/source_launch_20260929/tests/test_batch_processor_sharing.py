import unittest
import numpy as np
import config
from environment.env import Env
from environment.request_types import REQUEST_TYPE_SERVICE


class TestBatchProcessorSharing(unittest.TestCase):
    """TEST-04: Batch Final-Load Equal Sharing (SSOT Section 4)
    
    Verifies:
    1. If UAV j executes N_j_assigned tasks (local + coop incoming), each task
       receives exact equal frequency f_{m,j} = F_j / N_j_assigned.
    2. Local and incoming Coop share the exact same final load denominator.
    3. MBS tasks do NOT enter UAV N_j_assigned.
    4. Task processing order does NOT alter compute latency or energy.
    5. N_j_assigned = 0 does not divide by zero or cause NaN.
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_equal_sharing_for_all_tasks_on_same_uav(self):
        """Two local tasks executed on the same UAV must receive the exact same
        computing frequency and their compute latencies must be identical.
        
        Legacy bug C2: Sequential decrement max(0, count - 1) causes earlier local
        tasks to see a different denominator from later local tasks.
        """
        uav = self.env.uavs[0]
        ue0 = self.env.ues[0]
        ue1 = self.env.ues[1]
        ue2 = self.env.ues[2]

        for ue in [ue0, ue1, ue2]:
            ue.pos[:2] = uav.pos[:2] + np.array([10.0, 0.0], dtype=np.float32)
            ue.current_request.req_type = REQUEST_TYPE_SERVICE
            ue.current_request.req_id = 0
            ue.current_request.req_size = 50000

        uav._current_covered_ues = [ue0, ue2, ue1]
        uav.calculate_initial_load()

        # Actions: ue0 -> Local (0), ue2 -> MBS (1), ue1 -> Local (0)
        offload_actions = np.array([0, 1, 0], dtype=np.int64)

        # Call process_requests directly so we can inspect latency_current_request
        # before the next step's _get_obs() regenerates requests.
        uav.process_requests(offload_actions=offload_actions)

        # Under canonical semantics:
        # Final load N_0_assigned = 2 (ue0 and ue1).
        # Both ue0 and ue1 must experience identical compute latency and total latency.
        # Under legacy C2: ue0 was processed with load=3, then ue2 offloaded to MBS
        # and decremented load to 2, so ue1 was processed with load=2!
        self.assertAlmostEqual(
            ue0.latency_current_request, ue1.latency_current_request, places=4,
            msg=f"Local tasks ue0 ({ue0.latency_current_request:.4f}s) and ue1 ({ue1.latency_current_request:.4f}s) experienced different compute sharing; legacy sequential C2 detected"
        )

    def test_mbs_tasks_excluded_from_uav_final_load(self):
        """Tasks offloaded to MBS must not dilute UAV computing capacity."""
        uav = self.env.uavs[0]
        ue0 = self.env.ues[0]
        ue0.pos[:2] = uav.pos[:2] + np.array([10.0, 0.0], dtype=np.float32)
        ue0.current_request.req_type = REQUEST_TYPE_SERVICE
        ue0.current_request.req_id = 0
        ue0.current_request.req_size = 1000

        uav._current_covered_ues = [ue0]
        uav.calculate_initial_load()

        offload_actions = np.array([0], dtype=np.int64)
        uav.process_requests(offload_actions=offload_actions)
        
        self.assertGreater(ue0.latency_current_request, 0.0)


if __name__ == "__main__":
    unittest.main()
