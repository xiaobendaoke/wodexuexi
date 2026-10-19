"""Pipeline contracts locally; numerical regression only on the Linux server."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ENTRY = Path(os.environ.get('SF_HLM_IMPROVEMENT_ENTRY',
    Path(__file__).resolve().parents[1] / 'scripts/analysis/sf_hlm_learning_improvement.py'))
spec = importlib.util.spec_from_file_location('learning_improvement_runner', ENTRY)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PipelineContracts(unittest.TestCase):
    def test_registered_population_and_order(self):
        tasks = runner.registered_tasks([1, 2], [11, 12], 2, [0, 4])
        self.assertEqual(len(tasks), len(set(tasks)))
        self.assertEqual(sum(t[0] == 'train' for t in tasks), 6)
        self.assertEqual(sum(t[0] == 'evaluate' for t in tasks), 72)
        self.assertEqual(sum(t[0] == 'capture' for t in tasks), 8)
        self.assertEqual(sum(t[0] == 'diagnostic' for t in tasks), 16)
        self.assertEqual(sum(t[0] == 'timing' for t in tasks), 16)
        self.assertEqual(runner.TRAIN_VARIANTS, ('baseline', 'multistep', 'effects'))

    def test_comparison_families_are_fixed_before_results(self):
        self.assertIn(('multistep_ppo', 'baseline_ppo'), runner.PRIMARY_PAIRS)
        self.assertIn(('effects_ppo', 'multistep_ppo'), runner.PRIMARY_PAIRS)
        self.assertIn(('effects_ppo', 'random'), runner.PRIMARY_PAIRS)
        self.assertIn(('multistep_ppo', 'multistep_bc'), runner.MECHANISM_PAIRS)
        self.assertEqual(len(runner.METHODS), 9)

    def test_resume_rejects_partial_or_changed_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(results_root=Path(tmp))
            task = ('evaluate', 'random', 1, 11, 0, -1)
            p = {'steps': 8, 'files': {}, 'snapshot_steps': [0, 4]}
            runner.atomic_json(args.results_root / 'protocol.json', p)
            path = runner.task_folder(args, task)
            path.mkdir(parents=True)
            (path / 'episode.json').write_text('{}')
            runner.receipt(args, p, task, steps=8)
            self.assertTrue(runner.verify_task(args, p, task))
            (path / 'episode.json').write_text('changed')
            with self.assertRaises(ValueError): runner.verify_task(args, p, task)
            runner.receipt(args, p, task, steps=7)
            with self.assertRaises(ValueError): runner.verify_task(args, p, task)

    def test_semantic_task_identity_does_not_depend_on_parallelism(self):
        task = ('train', 'multistep', 1, 0, -1, -1)
        self.assertEqual(runner.task_id(task), runner.task_id(tuple(task)))
        self.assertNotEqual(runner.task_id(task), runner.task_id(('train', 'effects', 1, 0, -1, -1)))

    def test_live_worker_requires_exact_results_identity(self):
        task = ('train', 'multistep', 1, 0, -1, -1)
        args = argparse.Namespace(results_root=Path('/study/current'), release_root=Path('/release'))
        command = runner.worker_command(args, task, paths=False)
        self.assertEqual(runner.decode_worker(command, args), task)
        changed = list(command)
        changed[changed.index('--results-root') + 1] = '/study/other'
        self.assertIsNone(runner.decode_worker(changed, args))

    def test_summary_requires_complete_registered_population(self):
        with self.assertRaises(ValueError): runner.require_population([], [('train', 'baseline', 1, 0, -1, -1)])


@unittest.skipUnless(sys.platform == 'linux' and os.environ.get('SF_HLM_IMPROVEMENT_SOURCE'),
                     'Numerical tests require frozen source and the remote Linux runtime')
class NumericalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = runner.test_runtime_args()
        runner.initialize_numerical(args)
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        from utils.sf_hlm_learning_improvement import run_search, multistep_teacher, action_effects
        from marl_models.sf_hlm_action_effect import ActionEffectPPO
        from utils.sf_hlm_v2 import CompleteQuotaSearch, graph
        cls.search = staticmethod(run_search)
        cls.teacher = staticmethod(multistep_teacher)
        cls.effects = staticmethod(action_effects)
        cls.model_cls, cls.engine_cls, cls.graph = ActionEffectPPO, CompleteQuotaSearch, staticmethod(graph)

    @staticmethod
    def rows(n=8):
        return [dict(deadline=1., source=i % 2, ue_id=i, size=1000, cycles=200000,
            options={0: dict(target=i % 2, base=.1, work=.22, energy=2., compute_energy=2., cache_belief=.8, capacity=1e9),
                     1: dict(target='mbs', base=1.1, work=0., energy=1., compute_energy=0., cache_belief=1., capacity=2e10),
                     2+(1-i % 2): dict(target=1-i % 2, base=.2, work=.3, energy=2., compute_energy=2., cache_belief=.3, capacity=1e9)})
                for i in range(n)]

    def test_teacher_counts_every_branch_and_never_invents_stop(self):
        samples, info = self.teacher(self.rows(), num_uavs=2, seed=17, total_budget=24)
        self.assertLessEqual(info['matching_calls'], 24)
        self.assertTrue(all(s['action'] != 0 for s in samples))
        second, again = self.teacher(self.rows(), num_uavs=2, seed=17, total_budget=24)
        self.assertEqual([s['action'] for s in samples], [s['action'] for s in second])
        self.assertEqual(info, again)

    def test_stop_interventions_preserve_calls_guard_and_requests(self):
        class Stop:
            def act(self, state, exploration=False):
                choices = self_outer.np.flatnonzero(state['mask'])
                return int(choices[0]), 0., 0.
        self_outer = self
        for kind in ('learned', 'no_stop', 'hybrid', 'random'):
            assignment, info, _ = self.search(self.rows(), num_uavs=2, kind=kind, model=Stop(),
                rng=self.np.random.default_rng(7), budget=8)
            self.assertEqual(len(assignment), 8)
            self.assertLessEqual(info['matching_calls'], 8)
            self.assertLessEqual(info['chosen_predicted']['energy'], info['common_energy_cap']+1e-8)
            if kind == 'learned': self.assertEqual(info['matching_calls'], 1)
            if kind == 'hybrid': self.assertEqual(info['early_stop_at'], 1)

    def test_effects_are_observed_finite_and_mbs_nonsharing(self):
        engine = self.engine_cls(self.rows(), num_uavs=2, matching_budget=8)
        features = self.effects(engine)
        self.assertEqual(features.shape, (len(engine.actions), 8))
        self.assertTrue(self.np.isfinite(features).all())
        self.assertTrue(self.np.all(features[0] == 0))

    def test_on_policy_probabilities_and_checkpoint_roundtrip(self):
        engine = self.engine_cls(self.rows(), num_uavs=2, matching_budget=8)
        state = self.graph(engine); state['effects'] = self.effects(engine)
        self.torch.manual_seed(5); model = self.model_cls(2)
        action, lp, value = model.act(state, exploration=True)
        got, _, _ = model.evaluate([state], self.torch.tensor([action]))
        self.assertAlmostEqual(lp, float(got[0]), places=5)
        loss = model.update([dict(state=state, action=action, log_prob=lp, value=value, reward=1., done=True)], epochs=1)
        self.assertAlmostEqual(loss['initial_ratio_mean'], 1., places=5)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp, {'study': 'test'})
            restored = self.model_cls(2); restored.load(tmp, {'study': 'test'})
            for k, v in model.network.state_dict().items():
                self.assertTrue(self.torch.equal(v, restored.network.state_dict()[k]))
            with self.assertRaises(ValueError): restored.load(tmp, {'study': 'different'})

    def test_new_scorer_starts_with_identical_base_policy(self):
        from marl_models.sf_hlm_v2 import CompleteQuotaPPO
        engine = self.engine_cls(self.rows(), num_uavs=2, matching_budget=8)
        state = self.graph(engine); state['effects'] = self.effects(engine)
        self.torch.manual_seed(19); base = CompleteQuotaPPO(2)
        self.torch.manual_seed(19); candidate = self.model_cls(2)
        a, va = base.forward_states([state]); b, vb = candidate.forward_states([state])
        self.assertTrue(self.torch.equal(a, b)); self.assertTrue(self.torch.equal(va, vb))

    def test_empty_batch_is_a_valid_complete_result(self):
        samples, info = self.teacher([], num_uavs=2, seed=1, total_budget=8)
        self.assertEqual(samples, []); self.assertEqual(info['matching_calls'], 0)
        assignment, d, records = self.search([], num_uavs=2, kind='random', rng=self.np.random.default_rng(1))
        self.assertEqual(assignment, ()); self.assertEqual(records, [])


if __name__ == '__main__': unittest.main()
