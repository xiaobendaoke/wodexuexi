"""Observation refinements keep physical admission fixed and expose their intended fields."""

import unittest

import numpy as np

import config
from environment.env import Env
from environment.request_types import Request


class UpperObservationRefinementTest(unittest.TestCase):
    def setUp(self):
        self.env = Env(scenario_seed=1907)
        self.env.reset()
        positions = [(50, 50), (650, 50), (50, 650), (650, 650), (350, 350)]
        for uav, xy in zip(self.env.uavs, positions):
            uav.pos[:2] = xy
            uav.current_covered_ues.clear()
        for ue in self.env.ues:
            ue.pos[:2] = (0, 0)
            ue.current_request = Request.content(26)
            ue.generate_request = lambda: None
        self.env.ues[0].pos[:2] = (400, 350)
        for index in range(1, 35):
            self.env.ues[index].pos[:2] = (465 + 5 * index, 350)
        self.env.ues[34].current_request = Request.service(1024, 2, 1.0, 2)
        self.original = config.UPPER_UE_OBSERVATION_MODE

    def tearDown(self):
        config.UPPER_UE_OBSERVATION_MODE = self.original

    def rows(self, mode):
        config.UPPER_UE_OBSERVATION_MODE = mode
        for uav in self.env.uavs:
            uav.current_covered_ues.clear()
        start = config.SELF_OBS_DIM + config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM
        return self.env._get_obs()[4][start:].reshape(-1, config.UE_OBS_DIM)

    def test_service_first_retains_service_when_slots_are_full(self):
        distance = self.rows('covered_plus_uncovered')
        service_first = self.rows('covered_plus_uncovered_service_first')
        self.assertEqual(len(self.env.uavs[4].current_covered_ues), 1)
        self.assertFalse(np.any(np.isclose(distance[:, 4], 2 / config.NUM_FILES)))
        self.assertTrue(np.any(np.isclose(service_first[:, 4], 2 / config.NUM_FILES)))
        np.testing.assert_array_equal(distance[0], service_first[0])

    def test_position_only_hides_uncovered_request_and_battery(self):
        full = self.rows('covered_plus_uncovered')
        position_only = self.rows('covered_plus_uncovered_position_only')
        np.testing.assert_array_equal(full[0], position_only[0])
        np.testing.assert_array_equal(full[1:, :2], position_only[1:, :2])
        self.assertTrue(np.any(full[1:, 2:] != 0))
        self.assertTrue(np.all(position_only[1:, 2:] == 0))
        self.assertFalse(self.env.ues[34].assigned)


if __name__ == '__main__':
    unittest.main()
