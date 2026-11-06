import copy
from types import SimpleNamespace
import unittest
import numpy as np
from utils.queued_taskset_validation import taskset_features


class TasksetTests(unittest.TestCase):
    def obs(self):
        p = SimpleNamespace(capacity=(60e9, 80e9, 200e9), mbs=(350., 350., 30.), coverage=100., inter_range=460.)
        tasks = [dict(uid=i, ue=0, source=0, target=1, deadline=2+i*.01, arrival=0., bits=16e6,
            remaining_upload=8e6, remaining_transfer=16e6, remaining_cycles=10e9, stage='upload') for i in range(80)]
        return dict(tasks=tasks, protocol=p, positions=[(100., 100.), (300., 100.)], ues={0:(110.,100.)}, time=1.)

    def test_no_truncation_and_permutation_invariant(self):
        obs = self.obs();move=[(0.,0.)]*2
        f, count = taskset_features(obs, {}, move)
        self.assertEqual(count, 80);self.assertEqual(f.shape, (64,));self.assertEqual(f[0], .8)
        obs['tasks'].reverse();g, _ = taskset_features(obs, {}, move)
        np.testing.assert_allclose(f, g, atol=1e-12)

    def test_late_task_information_changes_output_and_empty_boundary(self):
        obs = self.obs();move=[(0.,0.)]*2
        a, _ = taskset_features(obs, {}, move)
        obs['tasks'][-1]['remaining_cycles'] *= 8
        b, _ = taskset_features(obs, {}, move)
        self.assertGreater(np.linalg.norm(a-b), 1e-6)
        obs['tasks']=[];c, count = taskset_features(obs, {}, move)
        self.assertEqual(count, 0);np.testing.assert_array_equal(c, np.zeros(64))


if __name__=='__main__':unittest.main()
