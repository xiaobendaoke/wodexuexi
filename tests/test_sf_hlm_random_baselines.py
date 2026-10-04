"""Random policy legality, independent streams and scientific receipts (remote)."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paths import REPO_ROOT

ENTRY = Path(os.environ.get('SF_HLM_RANDOM_BASELINES_ENTRY',
    REPO_ROOT / 'scripts/analysis/sf_hlm_random_baselines.py'))
spec = importlib.util.spec_from_file_location('random_baseline_runner', ENTRY)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RandomBaselineTests(unittest.TestCase):
    def test_registration_and_timing_barrier(self):
        tasks = m.registered_tasks(range(10), range(5), 6)
        self.assertEqual(len(tasks), 1200)
        self.assertEqual(len(set(tasks)), 1200)
        quality = {t for t in tasks if t[0] == 'quality'}
        self.assertFalse(any(t[0] == 'timing' for t in m.eligible(tasks, set(), set())))
        self.assertEqual(len(m.eligible(tasks, quality, set())), 600)

    def test_seed_is_independent_of_phase_and_concurrency(self):
        self.assertEqual(m.random_seeds('parent', 1, 2, 3), m.random_seeds('parent', 1, 2, 3))
        self.assertNotEqual(*m.random_seeds('parent', 1, 2, 3))
        self.assertNotEqual(m.random_seeds('parent', 1, 2, 3), m.random_seeds('parent', 2, 2, 3))

    def test_random_draws_only_real_requests_from_the_same_quality_mask(self):
        import numpy as np
        masks = np.zeros((2, 4, 7), np.float32)
        masks[0, 1, [0, 3, 6]] = 1
        masks[1, 0, [1]] = 1
        requests = ((0, 1, 12), (1, 0, 20))
        actions, assignment, legal = m.sample_offload(requests, masks, np.random.default_rng(4))
        self.assertIn(assignment[0], (0, 3, 6))
        self.assertEqual(assignment[1], 1)
        self.assertEqual(actions[0, 0], 0)
        self.assertEqual(actions[0, 2], 0)
        self.assertEqual(legal, [[0, 3, 6], [1]])
        with self.assertRaises(ValueError):m.sample_offload(((0, 0, 99),), masks, np.random.default_rng(4))

    def test_sampling_is_uniform_and_not_a_greedy_repair(self):
        import numpy as np
        masks = np.ones((1, 1, 3))
        rng = np.random.default_rng(917)
        counts = np.zeros(3)
        for _ in range(9000):
            _, assignment, _ = m.sample_offload(((0, 0, 12),), masks, rng)
            counts[assignment[0]] += 1
        self.assertTrue(np.all(np.abs(counts-3000) < 200))

    def test_full_random_never_calls_the_trained_upper_or_a_solver(self):
        import numpy as np
        import config
        ue = SimpleNamespace(id=12, pos=np.array([1., 0.]), current_request=SimpleNamespace(is_service=True))
        uav = SimpleNamespace(pos=np.array([0., 0.]), current_covered_ues=[ue])
        env = SimpleNamespace(uavs=[uav], get_offloading_obs_and_masks=lambda:(None, np.ones((1, 1, 3))))
        upper = SimpleNamespace(select_actions=lambda *a, **kw: (_ for _ in ()).throw(AssertionError('trained upper used')))
        with patch.object(config, 'NUM_UAVS', 1), patch.object(config, 'ACTION_DIM', 2):
            motion, _, info = m.random_policy(env, None, 'full_random', {'upper':upper},
                np.random.default_rng(9), np.random.default_rng(10))
        self.assertTrue(np.all(np.abs(motion) <= 1))
        self.assertEqual(info['diagnostic']['matching_calls'], 0)
        self.assertFalse(info['diagnostic']['guard_applied'])

    def test_lower_random_uses_frozen_upper_and_identical_offload_stream(self):
        import numpy as np
        import config
        ue = SimpleNamespace(id=12, pos=np.array([1., 0.]), current_request=SimpleNamespace(is_service=True))
        env = SimpleNamespace(uavs=[SimpleNamespace(pos=np.zeros(2), current_covered_ues=[ue])],
            get_offloading_obs_and_masks=lambda:(None, np.ones((1, 1, 3))))
        upper = SimpleNamespace(select_actions=lambda *a, **kw:np.array([[.25, -.5]], np.float32))
        with patch.object(config, 'NUM_UAVS', 1), patch.object(config, 'ACTION_DIM', 2):
            full = m.random_policy(env, None, 'full_random', {'upper':upper}, np.random.default_rng(9), np.random.default_rng(10))
            lower = m.random_policy(env, None, 'upper_random_offload', {'upper':upper}, np.random.default_rng(9), np.random.default_rng(10))
        np.testing.assert_array_equal(lower[0], [[.25, -.5]])
        np.testing.assert_array_equal(full[1], lower[1])

    def test_real_request_order_matches_owner_slot_and_global_ids(self):
        import numpy as np
        def ue(uid,x):return SimpleNamespace(id=uid,pos=np.array([x, 0.]),current_request=SimpleNamespace(is_service=True))
        env = SimpleNamespace(uavs=[SimpleNamespace(pos=np.zeros(2),current_covered_ues=[ue(30, 2),ue(10, 1)]),
            SimpleNamespace(pos=np.zeros(2),current_covered_ues=[ue(20, 1)])])
        self.assertEqual(m.request_index(env, np.zeros((2, 2, 7))), ((0, 0, 10), (1, 0, 20), (0, 1, 30)))

    def test_receipts_reject_partial_and_corrupted_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(results_root=Path(tmp))
            (Path(tmp)/'protocol.json').write_text('{}')
            p = dict(steps=8,parent_protocol_sha256='parent',source_hash='source')
            task = ('quality','full_random',1,2,0)
            path = m.folder(args,task);path.mkdir(parents=True)
            (path/'episode.json').write_text('{}')
            d = dict(status='PASS',contract=m.task_contract(args,p,task),steps=8,
                files={'episode.json':m.sha(path/'episode.json')})
            (path/'result.json').write_text(json.dumps(d))
            self.assertTrue(m.verify(args,p,task))
            d['steps']=7;(path/'result.json').write_text(json.dumps(d))
            with self.assertRaises(ValueError):m.verify(args,p,task)
            d['steps']=8;(path/'result.json').write_text(json.dumps(d))
            (path/'episode.json').write_text('changed')
            with self.assertRaises(ValueError):m.verify(args,p,task)


if __name__ == '__main__': unittest.main()
