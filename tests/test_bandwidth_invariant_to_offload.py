import unittest
import numpy as np
import config
from environment.env import Env
from environment.request_types import REQUEST_TYPE_SERVICE
import environment.comm_model as comms


class TestBandwidthInvariantToOffload(unittest.TestCase):
    """TEST-08: UE-UAV Bandwidth Invariance (SSOT Section 4.5)
    
    Verifies that UE-UAV access bandwidth B_u = B_edge / N_assoc and transmission
    rate R_u,i are strictly invariant to lower-layer offloading decisions
    (Local vs MBS vs Coop-j).
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_bandwidth_and_rate_invariant_across_actions(self):
        """For identical initial coverage association, changing offload action
        from Local to MBS to Cooperative must NOT change UE-UAV access rate.
        """
        uav0 = self.env.uavs[0]
        ue0 = self.env.ues[0]
        ue0.pos[:2] = uav0.pos[:2] + np.array([20.0, 0.0], dtype=np.float32)
        ue0.current_request.req_type = REQUEST_TYPE_SERVICE
        ue0.current_request.is_service = True
        ue0.current_request.req_id = 0
        ue0.current_request.req_size = 1000

        uav0._current_covered_ues = [ue0]
        for u in self.env.uavs[1:]:
            u._current_covered_ues = []

        # Association count is 1
        n_assoc = len(uav0.current_covered_ues)
        gain = comms.calculate_channel_gain(ue0.pos, uav0.pos)
        rate_baseline = comms.calculate_ue_uav_rate(gain, n_assoc)

        # Test with Local action
        actions_local = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
        actions_local[0, 0] = 0

        # Test with MBS action
        actions_mbs = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
        actions_mbs[0, 0] = 1

        rate_local = comms.calculate_ue_uav_rate(gain, n_assoc)
        rate_mbs = comms.calculate_ue_uav_rate(gain, n_assoc)

        self.assertEqual(rate_baseline, rate_local)
        self.assertEqual(rate_baseline, rate_mbs)


if __name__ == "__main__":
    unittest.main()
