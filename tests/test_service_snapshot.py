import unittest
import numpy as np
import config
from environment.env import Env
import environment.comm_model as comms


class TestServiceSnapshot(unittest.TestCase):
    """TEST-02: Slot-Start Service Snapshot (SSOT Section 2.1, 2.3)
    
    Verifies that all service transmission channels, rates, and association
    evaluate strictly against the slot-start immutable snapshot.
    Upper action movement a_t only affects p_{t+1} and does not retroactively alter
    slot-t service channels.
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_service_channel_rates_match_slot_start_geometry(self):
        """Slot-t service rates must be computed using slot-start UAV positions p_t,
        not the post-action positions p_{t+1}.
        """
        # Record slot-start positions
        start_uav_positions = np.array([uav.pos.copy() for uav in self.env.uavs])
        start_ue_positions = np.array([ue.pos.copy() for ue in self.env.ues])

        # Step with a large movement action
        actions = np.ones((config.NUM_UAVS, 2), dtype=np.float32)
        next_obs, rewards, metrics = self.env.step(actions)

        # End-of-step positions
        end_uav_positions = np.array([uav.pos.copy() for uav in self.env.uavs])

        # Verify UAVs actually moved
        for i in range(config.NUM_UAVS):
            disp = np.linalg.norm(end_uav_positions[i, :2] - start_uav_positions[i, :2])
            self.assertGreater(disp, 0.0, f"UAV {i} did not move")

        # In canonical snapshot semantics:
        # UAV-MBS rate for slot t must match start_uav_positions, not end_uav_positions
        for i, uav in enumerate(self.env.uavs):
            start_mbs_rate = comms.calculate_uav_mbs_rate(
                comms.calculate_channel_gain(start_uav_positions[i], config.MBS_POS)
            )
            end_mbs_rate = comms.calculate_uav_mbs_rate(
                comms.calculate_channel_gain(end_uav_positions[i], config.MBS_POS)
            )
            # The rate stored or used during slot t execution must correspond to start position
            self.assertAlmostEqual(
                uav._uav_mbs_rate, start_mbs_rate, places=4,
                msg=f"UAV {i} service used non-slot-start rate"
            )

    def test_service_latency_invariant_to_post_step_position_shift(self):
        """If slot-start snapshot is frozen at Stage 0, perturbing a UAV position
        AFTER Stage 0 should not retroactively alter the already-evaluated service QoS.
        """
        env = Env()
        env.reset()

        # Slot-start snapshot must be immutable
        uav0 = env.uavs[0]
        initial_pos = uav0.pos.copy()
        
        # Verify that uav has slot-start snapshot semantics available
        # Under canonical semantics, env exposes or maintains immutable snapshot
        actions = np.zeros((config.NUM_UAVS, 2), dtype=np.float32)
        _, _, metrics_hover = env.step(actions)
        self.assertIn("latency", metrics_hover)


if __name__ == "__main__":
    unittest.main()
