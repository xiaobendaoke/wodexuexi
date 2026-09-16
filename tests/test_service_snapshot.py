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

    def test_service_snapshot_functional_invariance(self):
        """Verify that all transmission rates and service latencies during slot t
        strictly evaluate against the slot-start snapshot geometry p_t, and that
        the UAV moving to p_{t+1} via action a_t does NOT alter slot-t service rates.
        """
        env = Env()
        env.reset()

        # 1. Capture slot-start geometry
        start_uav_pos = np.array([uav.pos.copy() for uav in env.uavs])
        start_ue_pos = np.array([ue.pos.copy() for ue in env.ues])

        # Step with significant movement action
        actions = np.ones((config.NUM_UAVS, 2), dtype=np.float32)
        _, _, metrics = env.step(actions)

        # 2. Check canonical snapshot existence
        snapshot = getattr(env, "slot_snapshot", None) or getattr(env, "_slot_snapshot", None)
        self.assertIsNotNone(snapshot, "Canonical slot_snapshot must exist on env after step")

        # 3. Snapshot positions must match slot-start positions, NOT post-movement positions
        np.testing.assert_allclose(
            snapshot.uav_positions, start_uav_pos,
            err_msg="Snapshot UAV positions did not preserve slot-start positions"
        )

        # 4. Verify rates in snapshot match slot-start geometry, and distinctly differ from post-step geometry
        for i in range(config.NUM_UAVS):
            expected_start_mbs_rate = comms.calculate_uav_mbs_rate(
                comms.calculate_channel_gain(start_uav_pos[i], config.MBS_POS)
            )
            self.assertAlmostEqual(
                float(snapshot.uav_mbs_rates[i]), float(expected_start_mbs_rate), places=4,
                msg=f"Snapshot UAV-MBS rate for UAV {i} does not match slot-start rate"
            )
            # Verify UAV actually moved and end-of-step rate is different
            post_mbs_rate = comms.calculate_uav_mbs_rate(
                comms.calculate_channel_gain(env.uavs[i].pos, config.MBS_POS)
            )
            self.assertNotAlmostEqual(
                float(expected_start_mbs_rate), float(post_mbs_rate), places=2,
                msg=f"UAV {i} did not move enough to differentiate slot-start from post-step rates"
            )

    def test_canonical_slot_snapshot_structure_exists(self):
        """Under canonical semantics, the environment creates and maintains a formal
        immutable slot-start snapshot (SSOT Section 2.1, 2.3).
        """
        snapshot = getattr(self.env, "slot_snapshot", None) or getattr(self.env, "_slot_snapshot", None)
        self.assertIsNotNone(
            snapshot,
            "Canonical slot-start snapshot structure is missing; legacy relies on mutable in-place state"
        )


if __name__ == "__main__":
    unittest.main()
