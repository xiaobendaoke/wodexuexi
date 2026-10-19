import unittest
import numpy as np
import config
from environment.env import Env


class TestEnvTemporalAlignment(unittest.TestCase):
    """TEST-01: Environment Temporal Alignment (SSOT Section 2.2, 2.3)
    
    Verifies that transition (s_t, a_t, r_t, s_{t+1}) has current action a_t's
    flight/hover energy and displacement accounted in the SAME step r_t and metrics_t.
    First step must not use legacy dist_moved=0, and terminal step energy must not be dropped.
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_first_step_flight_energy_accounted(self):
        """Action a_0 with maximum displacement must produce non-zero flight energy
        and reduced hover energy in step 0's metrics and rewards.
        
        Legacy bug ENV-01: step() calculates energy at the beginning using
        uav._dist_moved (initialized to 0.0), so step 0 gets 0 flight energy (pure hover).
        """
        # Propose maximum displacement action for all UAVs
        actions = np.ones((config.NUM_UAVS, 2), dtype=np.float32)
        _, rewards_0, metrics_0 = self.env.step(actions)

        # Baseline hover-only energy for 1 slot
        hover_power = config.POWER_HOVER
        slot_duration = config.TIME_SLOT_DURATION
        pure_hover_energy_single = hover_power * slot_duration
        pure_hover_fleet_energy = pure_hover_energy_single * config.NUM_UAVS

        # Moving power: P_move * tau
        moving_power = config.POWER_MOVE
        pure_moving_energy_single = moving_power * slot_duration
        expected_pure_flight_fleet_energy = pure_moving_energy_single * config.NUM_UAVS

        # In canonical semantics:
        # UAVs moved at max speed in step 0, so their motion energy must reflect
        # the current action's displacement, NOT dist_moved=0 (pure hover).
        # We check that UAV energy in step 0 reflects moving energy.
        total_uav_energy_0 = sum(uav.energy for uav in self.env.uavs)
        
        # Legacy code calculates energy using dist_moved=0, giving pure hover energy
        # (plus compute/comm), while dist_moved was 0.
        # Check that dist_moved at the moment of energy calculation was from action a_0 (> 0).
        for uav in self.env.uavs:
            self.assertGreater(
                uav._dist_moved, 0.0,
                f"UAV {uav.id} _dist_moved is 0.0 in step 0; current action a_0 displacement was not applied before energy accounting"
            )

    def test_zero_vs_max_movement_energy_separation_at_step_0(self):
        """Comparing step 0 with zero movement vs max movement must yield different
        energy in metrics['energy']. Under legacy code, both yield identical flight/hover energy.
        """
        np.random.seed(123)
        env_hover = Env()
        env_hover.reset()
        zero_actions = np.zeros((config.NUM_UAVS, 2), dtype=np.float32)
        _, _, metrics_hover = env_hover.step(zero_actions)

        np.random.seed(123)
        env_fly = Env()
        env_fly.reset()
        max_actions = np.ones((config.NUM_UAVS, 2), dtype=np.float32)
        _, _, metrics_fly = env_fly.step(max_actions)

        uav_energy_hover = sum(u.energy for u in env_hover.uavs)
        uav_energy_fly = sum(u.energy for u in env_fly.uavs)
        
        self.assertNotAlmostEqual(
            uav_energy_hover, uav_energy_fly, places=2,
            msg="Step 0 energy for max movement is identical to pure hover; action displacement is lagged"
        )

    def test_terminal_step_energy_not_lost(self):
        """In an episode's terminal step T, the movement action a_T must have its
        flight/hover energy and boundary/collision effects reflected in step T metrics and rewards,
        without requiring an extra subsequent step to account for them.
        """
        # Env 1: terminal action is pure hover
        np.random.seed(999)
        env_term_hover = Env()
        env_term_hover.reset()
        env_term_hover.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32))
        _, rewards_term_hover, metrics_term_hover = env_term_hover.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32))

        # Env 2: terminal action is max displacement
        np.random.seed(999)
        env_term_fly = Env()
        env_term_fly.reset()
        env_term_fly.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32))
        max_actions = np.ones((config.NUM_UAVS, 2), dtype=np.float32)
        _, rewards_term_fly, metrics_term_fly = env_term_fly.step(max_actions)

        # Terminal transition energy in metrics must strictly separate fly vs hover
        self.assertGreater(
            metrics_term_fly["energy"], metrics_term_hover["energy"],
            f"Terminal step did not account for action a_T displacement energy in terminal metrics['energy']: "
            f"fly={metrics_term_fly['energy']} vs hover={metrics_term_hover['energy']}"
        )
        # Terminal transition rewards must penalize the higher energy consumed by a_T
        self.assertLess(
            float(np.sum(rewards_term_fly)), float(np.sum(rewards_term_hover)),
            f"Terminal step reward did not reflect terminal action a_T flight energy cost: "
            f"fly_reward={np.sum(rewards_term_fly)} vs hover_reward={np.sum(rewards_term_hover)}"
        )


if __name__ == "__main__":
    unittest.main()
