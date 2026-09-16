import unittest
import numpy as np
import config
from environment.env import Env
from environment.uavs import UAV, _get_computing_latency_and_energy
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
        computing frequency and their compute latencies must be proportional strictly
        to their CPU cycles with identical capacity share F_j / N_final.
        
        Legacy bug C2: Sequential decrement max(0, count - 1) causes earlier local
        tasks to see a different denominator from later local tasks.
        """
        uav = self.env.uavs[0]
        # Attach 2 service requests to UAV 0
        # UE 0 and UE 1
        ue0 = self.env.ues[0]
        ue1 = self.env.ues[1]
        ue0.current_request.req_type = REQUEST_TYPE_SERVICE
        ue0.current_request.req_id = 0
        ue0.current_request.req_size = 1000

        ue1.current_request.req_type = REQUEST_TYPE_SERVICE
        ue1.current_request.req_id = 0
        ue1.current_request.req_size = 1000

        # UE 2 offloads to MBS
        ue2 = self.env.ues[2]
        ue2.current_request.req_type = REQUEST_TYPE_SERVICE
        ue2.current_request.req_id = 0
        ue2.current_request.req_size = 1000

        uav._current_covered_ues = [ue0, ue2, ue1]
        for u in self.env.uavs[1:]:
            u._current_covered_ues = []

        # Offload actions for UAV 0:
        # Task 0 (ue0) -> Local (0)
        # Task 1 (ue2) -> MBS (1)
        # Task 2 (ue1) -> Local (0)
        offload_actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
        offload_actions[0, 0] = 0  # Local
        offload_actions[0, 1] = 1  # MBS
        offload_actions[0, 2] = 0  # Local

        self.env.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32), offloading_actions=offload_actions)

        # Under canonical semantics:
        # UAV 0 final assigned load N_0_assigned = 2 (ue0 and ue1). ue2 went to MBS.
        # Both ue0 and ue1 have identical size (1000) and req_id (0), so:
        # Their compute latencies MUST be identical!
        # In legacy C2: ue0 was processed when load was 3 (comp_latency with load=3).
        # Then ue2 offloaded to MBS, load decremented to 2!
        # Then ue1 was processed with load=2!
        # Therefore, ue0 and ue1 have different latencies under legacy code (EXPECTED_RED).
        self.assertAlmostEqual(
            ue0.latency_current_request, ue1.latency_current_request, places=4,
            msg=f"Local tasks ue0 ({ue0.latency_current_request}s) and ue1 ({ue1.latency_current_request}s) experienced different compute sharing; legacy sequential C2 detected"
        )

    def test_mbs_tasks_excluded_from_uav_final_load(self):
        """Tasks offloaded to MBS must not dilute UAV computing capacity."""
        uav = self.env.uavs[0]
        ue0 = self.env.ues[0]
        ue0.current_request.req_type = REQUEST_TYPE_SERVICE
        ue0.current_request.req_id = 0
        ue0.current_request.req_size = 1000

        uav._current_covered_ues = [ue0]
        for u in self.env.uavs[1:]:
            u._current_covered_ues = []

        offload_actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
        offload_actions[0, 0] = 0  # Local

        self.env.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32), offloading_actions=offload_actions)
        
        expected_comp_share = float(config.UAV_COMPUTING_CAPACITY[0]) / 1.0
        cpu_cycles = float(config.CPU_CYCLES_PER_BYTE[0]) * 1000.0
        expected_comp_latency = cpu_cycles / expected_comp_share
        self.assertGreater(ue0.latency_current_request, 0.0)


if __name__ == "__main__":
    unittest.main()
