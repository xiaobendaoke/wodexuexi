import copy
import unittest
from dataclasses import replace

from environment.queued_service import Protocol, safe_path
from utils.queued_waiting_rescue import rescue_candidates, rescue_value


def observation(distance=110., deadline=2.8, source=None):
    protocol = replace(Protocol(), capacity=(75e9, 200e9), separation=0.)
    return dict(time=0., positions=[(100., 100.)], ues={0:(100.+distance, 100.)},
                protocol=protocol, tasks=[dict(uid=1, ue=0, source=source,
                    target=None, deadline=deadline, remaining_upload=12e6,
                    remaining_cycles=6e9, stage='waiting')])


class WaitingRescueTests(unittest.TestCase):
    def test_reaches_service_circle_instead_of_ue(self):
        obs = observation()
        bank, metadata = rescue_candidates(obs, [[(0., 0.)]])
        self.assertGreater(metadata['added'], 0)
        self.assertTrue(any(move[0][0] > 10 for move in bank[1:]))

    def test_discrete_admission_and_service_time(self):
        obs = observation(deadline=.9)
        bank, metadata = rescue_candidates(obs, [[(0., 0.)]])
        self.assertEqual(metadata['added'], 0)
        self.assertEqual(bank, [[(0., 0.)]])

    def test_unreachable_task_does_not_create_motion(self):
        obs = observation(distance=200.)
        _, metadata = rescue_candidates(obs, [[(0., 0.)]])
        self.assertEqual(metadata['added'], 0)

    def test_only_unadmitted_tasks_drive_rescue(self):
        obs = observation(source=0)
        _, metadata = rescue_candidates(obs, [[(0., 0.)]])
        self.assertEqual(metadata['added'], 0)

    def test_nested_safe_bank_and_no_mutation(self):
        obs = observation()
        original = [[(0., 0.)], [(0., 5.)]]
        before = copy.deepcopy((obs, original))
        bank, metadata = rescue_candidates(obs, original)
        self.assertEqual(bank[:len(original)], original)
        self.assertLessEqual(metadata['added'], 4)
        self.assertTrue(all(safe_path(obs['positions'], move, obs['protocol']) for move in bank))
        self.assertEqual((obs, original), before)

    def test_public_task_order_invariance(self):
        obs = observation()
        other = dict(obs['tasks'][0], uid=2, deadline=2.2)
        obs['tasks'].append(other)
        bank, _ = rescue_candidates(obs, [[(0., 0.)]])
        obs['tasks'].reverse()
        reordered, _ = rescue_candidates(obs, [[(0., 0.)]])
        self.assertEqual(bank, reordered)

    def test_boundary_and_collision_filter(self):
        obs = observation()
        obs['positions'] = [(640., 100.)]
        obs['ues'][0] = (750., 100.)
        bank, _ = rescue_candidates(obs, [[(0., 0.)]])
        self.assertTrue(all(safe_path(obs['positions'], move, obs['protocol']) for move in bank))

    def test_current_requests_without_future_fields(self):
        obs = observation()
        first = rescue_value(obs, [(15., 0.)])
        obs['future'] = 'forbidden data'
        self.assertEqual(first, rescue_value(obs, [(15., 0.)]))
        self.assertGreater(first, rescue_value(obs, [(0., 0.)]))


if __name__ == '__main__':
    unittest.main()
