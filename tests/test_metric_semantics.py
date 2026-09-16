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
        # Place UAV 0 at (100, 100), others far away
        self.env.uavs[0].pos[:2] = np.array([100.0, 100.0], dtype=np.float32)
        for u in self.env.uavs[1:]:
            u.pos[:2] = np.array([1000.0, 1000.0], dtype=np.float32)

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

    def test_critical_ue_ratio_definition(self):
        """Critical-UE Ratio must equal the proportion of UEs with battery < UE_CRITICAL_THRESHOLD."""
        # Set 15 UEs below threshold
        crit_thresh = config.UE_CRITICAL_THRESHOLD
        for i in range(15):
            self.env.ues[i].battery_level = crit_thresh - 1.0
        for i in range(15, config.NUM_UES):
            self.env.ues[i].battery_level = crit_thresh + 10.0

        _, _, metrics = self.env.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32))

        # Check offline_rate field equals 15/100 = 0.15
        expected_ratio = 15.0 / float(config.NUM_UES)
        self.assertAlmostEqual(
            metrics["offline_rate"], expected_ratio, places=4,
            msg=f"Critical-UE Ratio was {metrics['offline_rate']}, expected {expected_ratio}"
        )

    def test_uav_fleet_total_energy_excludes_mbs_energy(self):
        """UAV Fleet Total Energy is sum of UAV energies, excluding MBS compute power."""
        uav0 = self.env.uavs[0]
        ue0 = self.env.ues[0]
        ue0.pos[:2] = uav0.pos[:2] + np.array([10.0, 0.0], dtype=np.float32)
        ue0.current_request.req_type = REQUEST_TYPE_SERVICE
        ue0.current_request.req_id = 0
        ue0.current_request.req_size = 50000  # Large compute task

        uav0._current_covered_ues = [ue0]
        for u in self.env.uavs[1:]:
            u._current_covered_ues = []

        # Offload to MBS
        offload_actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
        offload_actions[0, 0] = 1  # MBS

        _, _, metrics = self.env.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32), offloading_actions=offload_actions)

        # Total energy metric must strictly equal sum of UAV energies
        sum_uav_energies = sum(u.energy for u in self.env.uavs)
        self.assertAlmostEqual(
            metrics["energy"], sum_uav_energies, places=4,
            msg="metrics['energy'] diverged from UAV fleet total energy"
        )


if __name__ == "__main__":
    unittest.main()
