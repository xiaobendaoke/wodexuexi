import unittest
import numpy as np
import config
from environment.env import Env


class TestCollisionJointFeasibility(unittest.TestCase):
    """TEST-07: Collision Joint Feasibility (SSOT Section 4.6)
    
    Verifies that after collision resolution, final positions simultaneously satisfy:
    1. Maximum speed ball: actual_displacement <= v_max * delta_t
    2. Minimum pairwise separation: ||q_i - q_j|| >= MIN_UAV_SEPARATION (200m)
    3. Boundary constraints: q_i in [d_bound, W - d_bound] x [d_bound, H - d_bound]
    4. Non-negative hover time: time_hovering >= 0
    
    Legacy bug ENV-02: Collision repulsion pushes UAVs apart without re-projecting
    onto the reachability ball, causing dist_moved > max_dist and negative hover time.
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_head_on_collision_joint_feasibility(self):
        """Construct a head-on collision scenario where UAV 0 and UAV 1 fly directly
        towards each other. After collision repair, all 4 constraints must hold jointly.
        """
        # Place UAV 0 and UAV 1 210m apart (just above 200m separation)
        p0 = np.array([300.0, 350.0], dtype=np.float32)
        p1 = np.array([510.0, 350.0], dtype=np.float32)
        self.env.uavs[0].pos[:2] = p0
        self.env.uavs[1].pos[:2] = p1

        # Place other UAVs far away and pairwise separated >= MIN_UAV_SEPARATION
        for i in range(2, config.NUM_UAVS):
            self.env.uavs[i].pos[:2] = np.array([100.0, 100.0 + (i - 2) * 250.0], dtype=np.float32)

        # UAV 0 moves full speed towards UAV 1 (right: +x)
        # UAV 1 moves full speed towards UAV 0 (left: -x)
        actions = np.zeros((config.NUM_UAVS, 2), dtype=np.float32)
        actions[0] = [1.0, 0.0]
        actions[1] = [-1.0, 0.0]

        max_dist = config.UAV_SPEED * config.TIME_SLOT_DURATION
        min_sep = config.MIN_UAV_SEPARATION
        min_bound = config.UAV_COVERAGE_RADIUS / 2.0

        # Step environment
        self.env.step(actions)

        # 1. Check speed ball constraint for all UAVs
        for uav in self.env.uavs:
            self.assertLessEqual(
                uav._dist_moved, max_dist + 1e-4,
                f"UAV {uav.id} displacement ({uav._dist_moved:.2f}m) exceeded max speed ball ({max_dist:.2f}m); legacy bug ENV-02"
            )

        # 2. Check non-negative hover time
        for uav in self.env.uavs:
            time_moving = uav._dist_moved / config.UAV_SPEED
            time_hovering = config.TIME_SLOT_DURATION - time_moving
            self.assertGreaterEqual(
                time_hovering, -1e-6,
                f"UAV {uav.id} hover time is negative ({time_hovering:.4f}s); subtracted hover energy"
            )

        # 3. Check pairwise separation
        for i in range(config.NUM_UAVS):
            for j in range(i + 1, config.NUM_UAVS):
                dist = np.linalg.norm(self.env.uavs[i].pos[:2] - self.env.uavs[j].pos[:2])
                self.assertGreaterEqual(
                    dist, min_sep - 1e-3,
                    f"UAV {i} and {j} violated minimum separation ({dist:.2f}m < {min_sep}m)"
                )

        # 4. Check boundary constraints
        for uav in self.env.uavs:
            x, y = uav.pos[0], uav.pos[1]
            self.assertTrue(
                min_bound - 1e-3 <= x <= config.AREA_WIDTH - min_bound + 1e-3,
                f"UAV {uav.id} x={x} violated boundary"
            )
            self.assertTrue(
                min_bound - 1e-3 <= y <= config.AREA_HEIGHT - min_bound + 1e-3,
                f"UAV {uav.id} y={y} violated boundary"
            )

    def test_reset_positions_are_pairwise_separated(self):
        """Initial positions generated on reset must all satisfy minimum separation."""
        for seed in [1, 42, 100, 2026]:
            np.random.seed(seed)
            env = Env()
            env.reset()
            for i in range(config.NUM_UAVS):
                for j in range(i + 1, config.NUM_UAVS):
                    dist = np.linalg.norm(env.uavs[i].pos[:2] - env.uavs[j].pos[:2])
                    self.assertGreaterEqual(
                        dist, config.MIN_UAV_SEPARATION - 1.0,
                        f"Seed {seed}: Initial UAV {i} and {j} violated separation ({dist:.2f}m)"
                    )


if __name__ == "__main__":
    unittest.main()
