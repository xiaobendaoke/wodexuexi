"""Remote tests for fixed-count matching and hierarchical search contracts."""
import itertools
import os
from pathlib import Path
import sys
import unittest
import tempfile
import json
import shutil
import subprocess
import platform

if platform.system() != 'Linux': raise unittest.SkipTest('SF-HLM simulation and Torch tests run only on the remote Linux server')

if os.environ.get('SF_HLM_SOURCE'):
    sys.path.insert(0, os.environ['SF_HLM_SOURCE'])
    import importlib.util
    extension = Path(os.environ['SF_HLM_EXTENSION'])
    for name, rel in [('utils.deadline_repair', 'utils/deadline_repair.py'),
                      ('utils.deadline_repair_online', 'utils/deadline_repair_online.py'),
                      ('utils.node_offload_observation', 'utils/node_offload_observation.py'),
                      ('utils.sf_hlm_matching', 'utils/sf_hlm_matching.py'),
                      ('utils.sf_hlm_search', 'utils/sf_hlm_search.py'),
                      ('utils.sf_hlm_coordination', 'utils/sf_hlm_coordination.py'),
                      ('marl_models.sf_hlm', 'marl_models/sf_hlm.py')]:
        spec = importlib.util.spec_from_file_location(name, extension / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

import numpy as np
import torch
from utils.deadline_repair import assess
from utils.sf_hlm_matching import solve_fixed_counts, allocation_counts, feedback, problem_from_rows
from utils.sf_hlm_search import QuotaSearch, transfer_actions, graph_state
from marl_models.sf_hlm import QuotaPPO, FeedbackMAPPO


def fixture():
    return [dict(deadline=1., source=i % 2, ue_id=i, options={
        0: dict(target=i % 2, base=.1, work=.35, energy=2., compute_energy=6.),
        1: dict(target='mbs', base=.9, work=0., energy=3., compute_energy=0.),
        2 + (1 - i % 2): dict(target=1 - i % 2, base=.2, work=.3, energy=2.5, compute_energy=5.)
    }) for i in range(4)]


class MatchingTests(unittest.TestCase):
    def test_fixed_count_matches_exhaustive_lexicographic_objective(self):
        rows = fixture()
        for assignment in itertools.product(*(list(r['options']) for r in rows)):
            counts = allocation_counts(rows, assignment, 2)
            result = solve_fixed_counts(rows, counts, 2)
            legal = (a for a in itertools.product(*(list(r['options']) for r in rows))
                     if np.array_equal(allocation_counts(rows, a, 2), counts))
            best = min((assess(rows, a)['misses'], assess(rows, a)['energy']) for a in legal)
            self.assertEqual(result['predicted']['misses'], best[0])
            self.assertAlmostEqual(result['predicted']['energy'], best[1])
            np.testing.assert_array_equal(allocation_counts(rows, result['assignment'], 2), counts)

    def test_empty_infeasible_and_invalid_counts(self):
        self.assertEqual(solve_fixed_counts([], [0, 0, 0], 2)['assignment'], ())
        row = dict(deadline=1., options={0: dict(target=0, base=.1, work=.2, energy=0., compute_energy=0.)})
        self.assertIsNone(solve_fixed_counts([row], [0, 1, 0], 2))
        for counts in ([0, 0, 0], [-1, 1, 1], [.5, .5, 0], [True, 0, 0]):
            with self.assertRaises(ValueError): solve_fixed_counts([row], counts, 2)
        with self.assertRaises(ValueError): allocation_counts([row], [1], 2)

    def test_mbs_does_not_share_cpu_and_full_capacity(self):
        rows = [fixture()[0]] * 100
        result = solve_fixed_counts(rows, [0, 0, 100], 2)
        self.assertEqual(result['predicted']['misses'], 0)
        self.assertTrue(all(t == .9 for t in result['predicted']['latencies']))
        self.assertTrue(all(a == 1 for a in result['assignment']))

    def test_masks_global_targets_and_feedback(self):
        rows = fixture(); p = problem_from_rows(rows, 2)
        self.assertEqual(p['action_ids'][1, 0], 2)
        a = (0, 0, 0, 0)
        f = feedback(rows, a, 2, energy_reference=10.)
        self.assertEqual(f.shape, (3, 6)); self.assertTrue(np.isfinite(f).all())
        self.assertEqual(f[2, 4], 0.)
        np.testing.assert_array_equal(feedback([], (), 2, energy_reference=10.), np.zeros((3, 6)))

    def test_energy_cap_rejects_candidates_without_losing_baseline(self):
        rows = fixture(); initial = (0, 0, 0, 0)
        engine = QuotaSearch(rows, num_uavs=2, initial=initial, matching_budget=4)
        initial_energy = assess(rows, initial)['energy']
        for action in range(len(transfer_actions(3))):
            if engine.done: break
            if engine.action_mask()[action]: engine.step(action)
        self.assertLessEqual(engine.best['energy'], 1.05 * initial_energy + 1e-9)
        self.assertLessEqual(engine.calls, 4)
        self.assertLessEqual(engine.best['objective'], assess(rows, initial)['objective'])


class LearningTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(13)

    def state(self):
        engine = QuotaSearch(fixture(), num_uavs=2, matching_budget=4)
        return engine, graph_state(engine)

    def test_sample_probability_recompute_and_finite_ppo(self):
        engine, state = self.state(); model = QuotaPPO(num_uavs=2)
        action, lp, value = model.act(state, exploration=True)
        new_lp, entropy, new_value = model.evaluate([state], torch.tensor([action]))
        self.assertAlmostEqual(lp, float(new_lp[0]), places=5)
        batch = [dict(state=state, action=action, log_prob=lp, value=value, reward=1., done=True)]
        loss = model.update(batch, epochs=1)
        self.assertTrue(all(np.isfinite(v) for v in loss.values()))
        self.assertAlmostEqual(loss['initial_ratio_mean'], 1., places=5)

    def test_padding_request_permutation_and_empty_graph(self):
        engine, state = self.state(); model = QuotaPPO(num_uavs=2)
        reordered = {k: v.copy() for k, v in state.items()}
        for key in ('requests', 'edges', 'legal'): reordered[key] = reordered[key][::-1].copy()
        logits, values = model.forward_states([state, reordered])
        self.assertTrue(torch.allclose(logits[0], logits[1], atol=1e-6))
        self.assertTrue(torch.allclose(values[0], values[1], atol=1e-6))
        empty = graph_state(QuotaSearch([], num_uavs=2, matching_budget=4))
        self.assertEqual(model.act(empty)[0], 0)
        alone = model.forward_states([state])[0][0]
        mixed = model.forward_states([state, empty])[0][0]
        self.assertTrue(torch.allclose(alone, mixed, atol=1e-6))

    def test_teacher_loss_is_separate_and_checkpoint_contract(self):
        engine, state = self.state(); model = QuotaPPO(num_uavs=2)
        with tempfile.TemporaryDirectory() as folder:
            model.save(folder, contract={'test': 1})
            restored = QuotaPPO(num_uavs=2); restored.load(folder, contract={'test': 1})
            self.assertEqual(model.act(state)[0], restored.act(state)[0])
            with self.assertRaises(ValueError): restored.load(folder, contract={'test': 2})
        loss = model.pretrain([dict(state=state, action=0)], epochs=1)
        self.assertIn('supervised_loss', loss)
        self.assertNotIn('ppo_loss', loss)

    def test_live_observation_purity_and_quality_mask(self):
        import pickle
        from environment.env import Env
        from utils.deadline_repair_online import prepare
        env = Env(scenario_seed=19); env.reset(); _, masks = env.get_offloading_obs_and_masks()
        requests, rows = prepare(env, masks); before = pickle.dumps(env, protocol=5)
        engine = QuotaSearch(rows, num_uavs=len(env.uavs), matching_budget=3)
        state = graph_state(engine); model = QuotaPPO(num_uavs=len(env.uavs))
        while not engine.done: engine.step(model.act(graph_state(engine))[0])
        for request, action in zip(requests, engine.assignment): self.assertTrue(masks[request[0], request[1], action])
        self.assertEqual(before, pickle.dumps(env, protocol=5))

    def test_feedback_upper_probability_update_and_disabled_feedback(self):
        import config
        from run_canonical_experiment import configure_method
        configure_method('lower_mlp_upper_uncovered_obs')
        model = FeedbackMAPPO(config.OBS_DIM_SINGLE, config.NUM_UAVS, enabled=True)
        base = np.zeros((config.NUM_UAVS, config.OBS_DIM_SINGLE), np.float32)
        summary = np.zeros((config.NUM_UAVS + 1, 6), np.float32)
        action, lp, value = model.act(base, summary, exploration=True)
        new_lp, _, _ = model.evaluate(torch.tensor(base)[None], torch.tensor(summary)[None], torch.tensor(action)[None])
        np.testing.assert_allclose(lp, new_lp.detach().numpy()[0], atol=1e-6)
        loss = model.update([dict(obs=base, feedback=summary, action=action, log_prob=lp,
                                  value=value, reward=np.ones(config.NUM_UAVS), done=True)], epochs=1)
        self.assertTrue(all(np.isfinite(v) for v in loss.values()))
        disabled = FeedbackMAPPO(config.OBS_DIM_SINGLE, config.NUM_UAVS, enabled=False)
        np.testing.assert_array_equal(disabled.act(base, summary)[0], disabled.act(base, summary + 10.)[0])
        with tempfile.TemporaryDirectory() as folder:
            model.save(folder, {'phase': 'test'}); restored = FeedbackMAPPO(config.OBS_DIM_SINGLE, config.NUM_UAVS, enabled=True)
            restored.load(folder, {'phase': 'test'})
            np.testing.assert_array_equal(model.act(base, summary)[0], restored.act(base, summary)[0])
            with self.assertRaises(ValueError): disabled.load(folder, {'phase': 'test'})

    def test_wrong_probabilities_are_rejected(self):
        _, state = self.state(); model = QuotaPPO(num_uavs=2); a, lp, value = model.act(state, exploration=True)
        with self.assertRaises(ValueError): model.update([dict(state=state, action=a, log_prob=lp + .1, value=value, reward=0., done=True)])

    def test_current_assignment_is_independent_of_current_motion(self):
        import copy
        import config
        from environment.env import Env
        from utils.deadline_repair_online import prepare, greedy_assignment
        env = Env(scenario_seed=27); env.reset(); _, masks = env.get_offloading_obs_and_masks(); requests, rows = prepare(env, masks)
        assignment, _ = greedy_assignment(rows); actions = np.zeros(masks.shape[:2], np.int64)
        for r, a in zip(requests, assignment): actions[r[0], r[1]] = a
        other = copy.deepcopy(env)
        _, _, still = env.step(np.zeros((config.NUM_UAVS, 2)), offloading_actions=actions)
        _, _, moving = other.step(np.ones((config.NUM_UAVS, 2)), offloading_actions=actions)
        for key in ('generated_service_count', 'admitted_service_count', 'deadline_satisfied_count', 'lower_assignment_energy', 'lower_service_latency'):
            self.assertAlmostEqual(still[key], moving[key], places=9)

    def test_charged_control_is_separate_and_hooks_restore(self):
        import copy
        import config
        from environment.env import Env
        from utils.sf_hlm_coordination import control_cost, execute_controlled_slot
        from utils.deadline_repair_online import prepare, greedy_assignment
        env = Env(scenario_seed=29); env.reset(); _, masks = env.get_offloading_obs_and_masks(); requests, rows = prepare(env, masks)
        assignment, _ = greedy_assignment(rows); actions = np.zeros(masks.shape[:2], np.int64)
        for r, a in zip(requests, assignment): actions[r[0], r[1]] = a
        charged_env = copy.deepcopy(env); zero_env = copy.deepcopy(env); motion = np.zeros((config.NUM_UAVS, 2))
        cost = control_cost(rows, np.full(config.NUM_UAVS, 1e6), .5, .1, additional_delay=100.)
        _, _, normal = env.step(motion, offloading_actions=actions)
        _, _, zero = execute_controlled_slot(zero_env, motion, actions, dict(delay_seconds=0., uav_energy=np.zeros(config.NUM_UAVS)))
        self.assertAlmostEqual(normal['energy'], zero['energy'], places=9)
        self.assertEqual(normal['deadline_satisfied_count'], zero['deadline_satisfied_count'])
        originals = [u._process_service_request for u in charged_env.uavs]; metric_method = charged_env._get_rewards_and_metrics
        _, _, charged = execute_controlled_slot(charged_env, motion, actions, cost)
        self.assertAlmostEqual(charged['energy'] - normal['energy'], float(sum(cost['uav_energy'])), places=8)
        self.assertEqual(charged['deadline_satisfied_count'], 0)
        self.assertEqual(originals, [u._process_service_request for u in charged_env.uavs]); self.assertEqual(metric_method, charged_env._get_rewards_and_metrics)
        with self.assertRaises(ValueError): control_cost(rows, [0.] * config.NUM_UAVS, .5, .1)

    def test_teacher_budget_and_request_graph_separation(self):
        import importlib.util
        root = Path(os.environ.get('SF_HLM_EXTENSION', Path(__file__).resolve().parents[1]))
        spec = importlib.util.spec_from_file_location('sf_hlm_runner_test', root / 'scripts/analysis/sf_hlm_experiment.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        sample, calls = runner.teacher_samples(fixture(), 2, total_budget=8)
        self.assertLessEqual(calls, 8)
        for row in sample:
            self.assertTrue(row['state']['mask'][row['action']]); self.assertNotIn('log_prob', row)
        import config
        from utils.logger import refresh_derived_config_fields
        old_count = config.NUM_UES; old_dim = config.OBS_DIM_SINGLE; old_capacity = config.MAX_ASSOCIATED_UES
        try:
            runner.apply_physical_variant(dict(ue_count=60, compute_scale=1.))
            self.assertEqual(config.OBS_DIM_SINGLE, old_dim); self.assertEqual(config.MAX_ASSOCIATED_UES, old_capacity)
            self.assertEqual(config.MAX_OFFLOAD_REQUESTS_PER_UAV, 60)
        finally: config.NUM_UES = old_count; refresh_derived_config_fields()


@unittest.skipUnless(os.environ.get('SF_HLM_SMOKE'), 'Requires a completed verification-only pipeline')
class RecoveryTests(unittest.TestCase):
    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor): self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left: self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.assert_state_equal(a, b)
        else: self.assertEqual(left, right)

    def test_full_training_recovery_matches_uninterrupted_run(self):
        original = Path(os.environ['SF_HLM_SMOKE']); protocol = json.loads((original / 'protocol.json').read_text())
        self.assertTrue(protocol['smoke']); self.assertGreaterEqual(protocol['system_episodes'], 6)
        source = Path(os.environ['SF_HLM_SOURCE']); extension = Path(os.environ['SF_HLM_EXTENSION'])
        suite = Path(os.environ['SF_HLM_SUITE'])
        with tempfile.TemporaryDirectory(dir=original.parent) as folder:
            clone = Path(folder) / 'recovery_copy'; shutil.copytree(original, clone)
            jobs = [j for j in protocol['jobs'] if j[0] in ('lower_training', 'system_training')]
            for phase, method, block, workload in jobs:
                with self.subTest(phase=phase, method=method, block=block):
                    rel = Path('lower_training') / f'block_{block}' if phase == 'lower_training' else Path(phase) / method / f'block_{block}_workload_{workload}'
                    baseline = original / rel; out = clone / rel
                    expected = json.loads((baseline / 'result.json').read_text()); (out / 'result.json').unlink()
                    checkpoint = 'recovery/iteration_00000' if phase == 'lower_training' else 'recovery/episode_00002'
                    (out / 'latest.json').write_text(json.dumps(dict(checkpoint=checkpoint)))
                    command = [sys.executable, str(extension / 'scripts/analysis/sf_hlm_experiment.py'), 'worker', '--source-root', str(source),
                               '--suite-root', str(suite), '--extension-root', str(extension), '--output', str(clone), '--phase', phase,
                               '--method', method, '--block', str(block), '--workload', str(workload), '--seed', str(protocol['lower_seeds'][str(block)])]
                    completed = subprocess.run(command, cwd=source, capture_output=True, text=True)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    got = json.loads((out / 'result.json').read_text())
                    self.assertEqual(expected['episodes'], got['episodes'])
                    if phase == 'lower_training': self.assertEqual(expected['losses'], got['losses'])
                    files = list((baseline / 'checkpoint').rglob('*.pt')) + list((baseline / 'checkpoint').rglob('*.pth'))
                    self.assertTrue(files)
                    for file in files:
                        restored = out / 'checkpoint' / file.relative_to(baseline / 'checkpoint')
                        self.assert_state_equal(torch.load(file, weights_only=False, map_location='cpu'), torch.load(restored, weights_only=False, map_location='cpu'))


if __name__ == '__main__': unittest.main()
