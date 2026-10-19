"""A fallback can create new collisions with a previously checked pair."""
import unittest
import numpy as np
import config
from environment.env import Env


class TestCollisionFallback(unittest.TestCase):
    def test_hover_fallback_propagates_to_new_conflicts(self):
        np.random.seed(42)
        env = Env()
        current = np.array([[100, 100], [300, 100], [500, 100], [100, 500], [500, 500]], dtype=float)
        proposed = current.copy()
        proposed[:3, 0] += [15, 15, -15]
        # (0,1) is initially safe. Reverting the colliding pair (1,2)
        # creates a new collision (0,1), which must also be repaired.
        repaired = env._finalize_joint_positions(current, proposed)
        np.testing.assert_array_equal(repaired, current)
        self.assertTrue(all(env.uavs[i].collision_violation for i in range(3)))

    def test_reset_rejects_colliding_supplied_positions(self):
        np.random.seed(42)
        with self.assertRaises(ValueError):
            Env().reset(initial_positions=[[100, 100]] * config.NUM_UAVS)

    def test_invalid_movement_is_rejected_before_state_changes(self):
        np.random.seed(42)
        env = Env()
        env.reset()
        before = np.array([u.pos.copy() for u in env.uavs])
        actions = np.zeros((config.NUM_UAVS, 2))
        actions[0, 0] = np.nan
        with self.assertRaises(ValueError):
            env.step(actions)
        self.assertEqual(env._time_step, 0)
        np.testing.assert_array_equal(before, [u.pos for u in env.uavs])

    def test_valid_stored_positions_survive_next_step(self):
        env = Env()
        positions = np.array([
            [290.7232243706553, 293.62931818878013],
            [470.0763942917365, 205.13049989012836],
            [100., 100.], [100., 500.], [500., 500.],
        ], dtype=np.float64)
        env._validate_positions(positions)
        for uav, pos in zip(env.uavs, positions):
            uav.update_position(pos)
        env._apply_actions_to_env(np.zeros((config.NUM_UAVS, 2)))
        env._validate_positions(np.array([u.pos[:2] for u in env.uavs]))

    def test_validation_uses_same_precision_as_fallback(self):
        positions = np.array([
            [334.0891418457031, 223.015869140625],
            [492.77557373046875, 101.28578186035156],
            [100., 100.], [100., 500.], [500., 500.],
        ], dtype=np.float32)
        # float32 norm rounds this invalid distance up across the tolerance.
        with self.assertRaises(ValueError):
            Env._validate_positions(positions)

    def test_reset_preserves_valid_supplied_geometry(self):
        positions = np.array([
            [290.7232243706553, 293.62931818878013],
            [470.0763942917365, 205.13049989012836],
            [100., 100.], [100., 500.], [500., 500.],
        ])
        env = Env()
        env.reset(initial_positions=positions)
        np.testing.assert_array_equal([u.pos[:2] for u in env.uavs], positions)
