"""Small numerical contracts only: no Torch, scenarios or physical rollouts."""
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np

from utils.queued_planner_distillation import (
    specification, validate_workload, validate_episode, planner_target, imitation_metrics, check_rows,
)
from marl_models.queued_planner_distillation import initialize, policy, loss_gradient, fit


def row(seed=1, workload=940000, split='training'):
    rng = np.random.default_rng(seed)
    return dict(a=rng.normal(size=(2, 3)), m=rng.normal(size=(2, 3, 2)),
                context=rng.normal(size=4), prior=np.array([.3, .7]),
                q=np.array([[0., 2., 2.], [1., 0., 0.]]),
                workload=workload, split=split)


class DistillationTests(unittest.TestCase):
    def test_development_analysis_uses_unique_baselines_and_paired_counts(self):
        from scripts.analysis import queued_planner_distillation as e
        spec = specification()
        scores = dict(mlp_joint=60, mlp_decomposed=55, linear_joint=50, initial=40, plan_short=70)
        with TemporaryDirectory() as folder:
            first = None
            for arm, score in scores.items():
                runs = spec['runs'] if arm in e.ARMS else [dict(id=-1, parent=b) for b in range(3)]
                for run in runs:
                    for workload in spec['evaluation_workloads']:
                        for replica in spec['action_replicas']:
                            record = dict(arm=arm, run=run['id'], parent=run['parent'], workload=workload,
                                          replica=replica, version=2 if arm in e.ARMS else 0, split='evaluation',
                                          audit={'status': 'PASS'}, counts={'generated': 100, 'success': score, 'expired': 100-score},
                                          successes=[score], fingerprint=str(workload), initial_fingerprint='shared',
                                          decision_mean=.1, flight_distance_m=1.)
                            p = e.out(folder)/'episodes'/'evaluation'/arm/f'{run["id"]}_{run["parent"]}_{workload}_{replica}.json'
                            e.t.write(p, record)
                            if first is None:
                                first = p
            def contrast(a, b, g):
                return dict(mean=float(np.mean((a-b).sum((2, 3))/g.sum((2, 3)))))
            with patch.object(e, 'paired_summary', side_effect=contrast):
                result = e.analyze(folder)
                self.assertEqual(result['summary']['initial']['physical_evaluation_episodes'], 48)
                self.assertEqual(result['summary']['mlp_joint']['physical_evaluation_episodes'], 96)
                self.assertAlmostEqual(result['comparisons']['mlp_joint--initial']['mean'], .2)
                self.assertFalse(result['formal_test_access'])
                record = e.load(first)
                record['initial_fingerprint'] = 'tampered'
                e.t.write(first, record)
                with self.assertRaises(ValueError):
                    e.analyze(folder)

    def test_server_guard_and_existing_episode_fail_before_parent_loading(self):
        from scripts.analysis import queued_planner_distillation as e
        args = ('mlp_joint', 0, 0, 940000, 0, 0, 'training')
        with patch.object(e.os, 'name', 'nt'), patch.object(e.t, 'parent') as parent:
            with self.assertRaises(RuntimeError):
                e.episode('unused', *args)
            parent.assert_not_called()
        with TemporaryDirectory() as folder:
            p = e.out(folder)/'episodes'/'training'/'mlp_joint'/'0_0_940000_0_0.json'
            p.parent.mkdir(parents=True)
            p.write_text('{}')
            with patch.object(e, 'require_server'), patch.object(e.t, 'parent') as parent:
                with self.assertRaises(FileExistsError):
                    e.episode(folder, *args)
                parent.assert_not_called()

    def test_fit_seals_before_holdout_and_passes_training_only(self):
        from scripts.analysis import queued_planner_distillation as e
        spec = specification()
        samples = {}
        for split in ('training', 'holdout'):
            samples[split] = []
            for workload in spec['runs'][0][split][0]:
                for now in spec['label_times']:
                    r = row(workload=workload, split=split)
                    r.update(time=now, parent=0, run=0, version=0)
                    samples[split].append(r)
        captured = []
        with TemporaryDirectory() as folder:
            def summary(rows, model, mode):
                self.assertTrue(e.model_path(folder, 'mlp_joint', 0, 1).is_file())
                captured.append(rows)
                return dict(states=len(rows))
            with patch.object(e, 'require_server'), \
                    patch.object(e, 'saved_rows', side_effect=lambda root, arm, run, split, version: samples[split]), \
                    patch.object(e, 'fit', side_effect=lambda rows, old, *a, **kw: (old, dict(states=len(rows)))) as fitting, \
                    patch.object(e, 'summarize_imitation', side_effect=summary):
                result = e.fit_job(folder, 'mlp_joint', 0, 1)
                self.assertIs(fitting.call_args.args[0], samples['training'])
                self.assertEqual(len(captured), 4)
                self.assertTrue((Path(folder)/result['model']).exists())
                with self.assertRaises(FileExistsError):
                    e.fit_job(folder, 'mlp_joint', 0, 1)
                self.assertEqual(fitting.call_count, 1)

    def test_public_row_explicitly_excludes_teacher_and_metadata(self):
        from scripts.analysis import queued_planner_distillation as e
        bank = row()
        bank['terminal_features'] = np.ones((2, 3, 99))
        with patch.object(e, 'value_features', return_value=np.ones(4)):
            public = e.public_row(bank, {}, object())
        self.assertEqual(set(public), {'a', 'm', 'prior', 'context'})

    def test_round_parent_final_checkpoint_and_baseline_uniqueness_guards(self):
        s = specification()
        validate_episode(s, 'mlp_joint', 0, 0, 940000, 0, 0, 'training')
        validate_episode(s, 'initial', -1, 0, 911000, 1, 0, 'evaluation')
        for args in [('mlp_joint', 0, 1, 940000, 0, 0, 'training'),
                     ('mlp_joint', 0, 0, 940000, 0, 1, 'training'),
                     ('mlp_joint', 0, 0, 940000, 1, 0, 'training'),
                     ('mlp_joint', 0, 0, 911000, 0, 1, 'evaluation'),
                     ('initial', 0, 0, 911000, 0, 0, 'evaluation')]:
            with self.assertRaises(ValueError):
                validate_episode(s, *args)

    def test_context_conditions_nonlinear_rankings(self):
        r = row()
        model, _ = fit([r], initialize(r, hidden=4), epochs=4)
        other = copy.deepcopy(r)
        other['context'] += 3.
        self.assertGreater(np.linalg.norm(policy(r, model)-policy(other, model)), 1e-7)

    def test_reservations_disjoint_and_budget_exact(self):
        spec = specification()
        train = {w for r in spec['runs'] for group in r['training'] for w in group}
        hold = {w for r in spec['runs'] for group in r['holdout'] for w in group}
        self.assertEqual(len(train), 72)
        self.assertEqual(len(hold), 24)
        self.assertFalse(train & hold)
        self.assertFalse((train | hold) & set(range(920000, 921440)))
        self.assertFalse((train | hold) & set(range(930000, 930040)))
        self.assertEqual(spec['maximum_physical_episodes'], 672)
        for split, workload in [('training', 930000), ('evaluation', 930000), ('holdout', 940000)]:
            with self.assertRaises(ValueError):
                validate_workload(spec, split, 0, workload)

    def test_planner_ties_follow_common_prior_not_current_actor(self):
        prior = np.array([.2, .8])
        scores = np.array([[3., 0.], [3., 3.]])
        target = planner_target(scores, prior)
        np.testing.assert_allclose(target, [[1/9, 0.], [4/9, 4/9]])
        np.testing.assert_allclose(planner_target(np.ones((2, 2)), prior), [[.1, .1], [.4, .4]])
        np.testing.assert_allclose(planner_target(scores + 100, prior), target)
        for bad in [np.array([[np.nan, 0.], [0., 1.]]), np.empty((2, 0))]:
            with self.assertRaises(ValueError):
                planner_target(bad, prior)
        with self.assertRaises(ValueError):
            planner_target(scores, [0., 0.])

    def test_zero_heads_reproduce_common_initialization(self):
        r = row()
        for hidden in [0, 5]:
            model = initialize(r, hidden=hidden, seed=2)
            np.testing.assert_allclose(policy(r, model), r['prior'][:, None] * np.ones((2, 3))/3)
            self.assertEqual(model['context_dim'], 4)

    def test_analytic_gradient_all_parameters_both_modes(self):
        r = row()
        target = planner_target(r['q'], r['prior'])
        for hidden in [0, 3]:
            model = initialize(r, hidden=hidden, seed=2)
            rng = np.random.default_rng(12)
            for branch in model['parameters'].values():
                for p in branch.values():
                    p += rng.normal(size=p.shape)*.1
            for mode in ['joint', 'decomposed']:
                loss, grad = loss_gradient(r, model, target, mode)
                self.assertTrue(np.isfinite(loss))
                for name, branch in model['parameters'].items():
                    for key, p in branch.items():
                        for index in np.ndindex(p.shape):
                            old = p[index]
                            p[index] = old + 1e-5
                            plus = loss_gradient(r, model, target, mode)[0]
                            p[index] = old - 1e-5
                            minus = loss_gradient(r, model, target, mode)[0]
                            p[index] = old
                            self.assertAlmostEqual((plus-minus)/2e-5, grad[name][key][index], places=7)

    def test_candidate_permutations_are_equivariant(self):
        r = row()
        model, _ = fit([r], initialize(r, hidden=4), epochs=2, batch_size=1)
        p = policy(r, model)
        perm = copy.deepcopy(r)
        perm['a'] = r['a'][::-1]
        perm['m'] = r['m'][::-1, ::-1]
        perm['prior'] = r['prior'][::-1]
        np.testing.assert_allclose(policy(perm, model), p[::-1, ::-1])

    def test_decomposed_motion_is_independent_of_plan(self):
        r = row()
        model, _ = fit([r], initialize(r, hidden=4), mode='decomposed', epochs=3)
        p = policy(r, model, 'decomposed')
        np.testing.assert_allclose(p[0]/p[0].sum(), p[1]/p[1].sum())

    def test_inference_does_not_use_teacher_or_training_metadata(self):
        r = row()
        model = initialize(r, hidden=4)
        minimal = {k: r[k] for k in ('a', 'm', 'prior', 'context')}
        np.testing.assert_array_equal(policy(r, model), policy(minimal, model))
        r['q'][:] = np.nan
        r['workload'] = 930000
        np.testing.assert_array_equal(policy(r, model), policy(minimal, model))

    def test_fit_refuses_holdout_and_entire_workload_overlap(self):
        train = row()
        held = row(workload=941000, split='holdout')
        check_rows([train], [held])
        with self.assertRaises(ValueError):
            fit([held], initialize(held))
        with self.assertRaises(ValueError):
            check_rows([train], [row(workload=940000, split='holdout')])
        with self.assertRaises(ValueError):
            check_rows([train], [row(workload=941000, split='training')])

    def test_fit_updates_both_heads_and_hidden_weights_without_mutation(self):
        r = row()
        old = initialize(r, hidden=4)
        before = copy.deepcopy(old)
        new, report = fit([r], old, epochs=8, batch_size=1, learning_rate=.02)
        self.assertLess(loss_gradient(r, new, planner_target(r['q'], r['prior']))[0],
                        loss_gradient(r, old, planner_target(r['q'], r['prior']))[0])
        self.assertEqual(report['optimizer_steps'], 8)
        for branch in old['parameters']:
            for key in old['parameters'][branch]:
                np.testing.assert_array_equal(old['parameters'][branch][key], before['parameters'][branch][key])
                self.assertGreater(np.linalg.norm(new['parameters'][branch][key]-old['parameters'][branch][key]), 0.)

    def test_teacher_regret_and_tied_states_are_retained(self):
        q = np.array([[1., 4.], [4., 2.]])
        p = np.full((2, 2), .25)
        z = imitation_metrics(q, p, [.5, .5])
        self.assertAlmostEqual(z['score_regret'], 1.25)
        self.assertAlmostEqual(z['optimal_probability'], .5)
        best = planner_target(q, [.5, .5])
        self.assertAlmostEqual(imitation_metrics(q, best, [.5, .5])['score_regret'], 0.)
        tied = imitation_metrics(np.ones((2, 2)), p, [.5, .5])
        self.assertTrue(tied['all_tied'])
        self.assertEqual(tied['score_regret'], 0.)
        self.assertIsNone(tied['retained_planning_gain'])

    def test_invalid_actor_inputs_and_nonfinite_updates_rejected(self):
        r = row()
        model = initialize(r)
        r['context'][0] = np.inf
        with self.assertRaises(ValueError):
            policy(r, model)
        r = row()
        with self.assertRaises(ValueError):
            fit([r], initialize(r), learning_rate=float('nan'))


if __name__ == '__main__':
    unittest.main()
