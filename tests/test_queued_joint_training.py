"""Unit tests for upper-lower joint neural training, finite-difference gradients, and zero-simulation inference."""
import copy
import unittest
from unittest.mock import patch
import numpy as np

from environment.queued_service import Engine, Protocol, Task
from marl_models.queued_joint_training import (
    MODES,
    actor_forward,
    actor_loss_and_gradient,
    actor_policy,
    compose_decomposed_actor,
    fit_joint_actor,
    initialize_joint_actor,
)
from utils.queued_arrival_adaptation import RateHistory
from utils.queued_joint_training import (
    NEURAL_EVAL_ARMS,
    advantage_weighted_target,
    compute_offline_teacher_targets,
    extract_actor_row,
    neural_actor_step,
)
from utils.queued_quality_amortization import expand_safe_plan_bank, safe_assignments
from utils.queued_service_controllers import assignments
from utils.queued_service_learning import LearningAdapter
from utils.queued_waiting_rescue import rescue_candidates


def _sample_state(seed=7):
    rng = np.random.default_rng(seed)
    positions = [(150.0, 150.0), (400.0, 150.0), (620.0, 150.0), (200.0, 450.0), (500.0, 450.0)]
    ues = {i: (float(rng.uniform(80, 620)), float(rng.uniform(80, 620))) for i in range(30)}
    engine = Engine(Protocol(), positions, ues)
    adapter = LearningAdapter()
    temporal = RateHistory()
    for t in range(4):
        tasks = [
            Task(t * 20 + k, int(rng.integers(0, 30)), float(t), float(t) + 2.2 + float(rng.uniform(0, 0.8)),
                 float(8 * rng.integers(1_500_000, 2_500_000)),
                 float(8 * rng.integers(1_500_000, 2_500_000) * rng.integers(550, 900)))
            for k in range(12)
        ]
        engine.add(tasks)
        engine.admit()
        obs = engine.observe()
        adapter.prepare(obs)
        temporal.update(obs)
        if t < 3:
            engine.assign(assignments(obs))
            engine.advance([(0.0, 0.0)] * 5)

    obs = engine.observe()
    base_plans = [assignments(obs), safe_assignments(obs)]
    plans, prior = expand_safe_plan_bank(obs, base_plans, [0.6, 0.4])
    base_actions = [
        [(0.0, 0.0)] * 5,
        [(10.0, 0.0)] * 5,
        [(-10.0, 0.0)] * 5,
        [(0.0, 10.0)] * 5,
    ]
    actions, meta = rescue_candidates(obs, base_actions)
    mp = np.full((len(plans), len(actions)), 1.0 / len(actions), dtype=np.float64)
    base_neural = rng.normal(0.0, 0.2, size=(len(base_plans), len(base_actions)))
    bank = dict(
        plans=plans,
        actions=actions,
        prior=prior,
        motion_prior=mp,
        original_count=len(base_actions),
        original_plan_count=len(base_plans),
        safe_plan=safe_assignments(obs),
        pressure=0.75 if seed % 2 == 0 else 1.25,
        rate3=20.0 if seed % 2 == 0 else 45.0,
        base_neural=base_neural,
        metadata=meta,
    )
    row = extract_actor_row(obs, bank, adapter, temporal)
    targets = compute_offline_teacher_targets(obs, bank, adapter)
    return obs, bank, row, targets, adapter, temporal


class TestQueuedJointTraining(unittest.TestCase):
    def test_epoch_zero_reproduces_baseline_prior(self):
        _, _, row, _, _, _ = _sample_state(11)
        model = initialize_joint_actor(row, hidden=16, seed=3)
        expected = row['prior'][:, None] * row['motion_prior']
        expected /= expected.sum()
        for mode in MODES:
            p = actor_policy(row, model, mode=mode)
            np.testing.assert_allclose(p, expected, atol=1e-12, rtol=1e-12)

    def test_finite_difference_gradients_all_modes_and_tensors(self):
        _, _, row, targets, _, _ = _sample_state(19)
        eps = 1e-6
        for mode in MODES:
            model = initialize_joint_actor(row, hidden=8, seed=17)
            rng = np.random.default_rng(101)
            for branch in ('allocation', 'motion'):
                for k in ('w', 'W', 'b', 'v'):
                    model['parameters'][branch][k] = rng.normal(
                        0.0, 0.25, size=model['parameters'][branch][k].shape
                    )

            target, _ = targets[mode]
            loss, grad = actor_loss_and_gradient(row, model, target, mode=mode, l2_reg=1e-4)
            self.assertTrue(np.isfinite(loss))

            for branch in ('allocation', 'motion'):
                if mode == 'upper_only' and branch == 'allocation':
                    for k in ('w', 'W', 'b', 'v'):
                        self.assertEqual(float(np.linalg.norm(grad[branch][k])), 0.0)
                    continue
                if mode == 'lower_only' and branch == 'motion':
                    for k in ('w', 'W', 'b', 'v'):
                        self.assertEqual(float(np.linalg.norm(grad[branch][k])), 0.0)
                    continue

                for k in ('w', 'W', 'b', 'v'):
                    param = model['parameters'][branch][k]
                    analytic = grad[branch][k]
                    flat_param = param.reshape(-1)
                    flat_analytic = analytic.reshape(-1)
                    indices = np.linspace(0, len(flat_param) - 1, num=min(12, len(flat_param)), dtype=int)
                    for idx in indices:
                        orig = float(flat_param[idx])
                        flat_param[idx] = orig + eps
                        loss_plus, _ = actor_loss_and_gradient(row, model, target, mode=mode, l2_reg=1e-4)
                        flat_param[idx] = orig - eps
                        loss_minus, _ = actor_loss_and_gradient(row, model, target, mode=mode, l2_reg=1e-4)
                        flat_param[idx] = orig
                        fd = (loss_plus - loss_minus) / (2.0 * eps)
                        self.assertAlmostEqual(
                            float(flat_analytic[idx]),
                            float(fd),
                            places=6,
                            msg=f"Gradient mismatch in mode={mode}, branch={branch}, param={k}, idx={idx}",
                        )

            if mode == 'joint':
                pa0, pm0, za0, zm0, xa0, *_ = actor_forward(row, model, mode='joint')
                model_perturbed = copy.deepcopy(model)
                model_perturbed['parameters']['motion']['w'] += 0.5
                pa1, pm1, za1, zm1, xa1, *_ = actor_forward(row, model_perturbed, mode='joint')
                da = row['a'].shape[1]
                c_upper0 = xa0[:, da:da + 4]
                c_upper1 = xa1[:, da:da + 4]
                self.assertGreater(
                    float(np.max(np.abs(c_upper1 - c_upper0))),
                    1e-4,
                    "Expected 4D upper-policy expectation vector C_upper to change when motion parameters change.",
                )
                self.assertGreater(
                    float(np.max(np.abs(pa1 - pa0))),
                    1e-5,
                    "Expected lower allocation policy pa to respond to upper motion distribution changes in joint mode.",
                )

    def test_zero_engine_calls_during_online_inference(self):
        obs, bank, row, _, adapter, temporal = _sample_state(23)
        base_model = initialize_joint_actor(row, hidden=12, seed=29)
        rng = np.random.default_rng(31)
        for branch in ('allocation', 'motion'):
            for k in ('w', 'W', 'b', 'v'):
                base_model['parameters'][branch][k] = rng.normal(
                    0.0, 0.1, size=base_model['parameters'][branch][k].shape
                )

        models = {
            'uncoordinated_init': initialize_joint_actor(row, hidden=12, seed=29),
            'lower_only_trained': copy.deepcopy(base_model),
            'upper_only_trained': copy.deepcopy(base_model),
            'decomposed_composed': compose_decomposed_actor(base_model, base_model),
            'decomposed_simultaneous': copy.deepcopy(base_model),
            'joint_trained': copy.deepcopy(base_model),
        }

        class _DummyPrototype:
            REVISION = 'test'

            class t:
                @staticmethod
                def seed(*args):
                    return 12345

            @staticmethod
            def evaluate_bank(obs, history, plans, motions, arrivals):
                return np.zeros((len(plans), len(motions)), dtype=np.float64)

        orig_plans = [p for p in bank['plans'][:bank['original_plan_count']]]
        orig_actions = [a for a in bank['actions'][:bank['original_count']]]
        orig_prior = np.array([0.6, 0.4], dtype=np.float64)
        base_prob = orig_prior[:, None] * np.full((len(orig_plans), len(orig_actions)), 1.0 / len(orig_actions))
        precomputed_base = dict(
            bank=dict(plans=orig_plans, actions=orig_actions, prior=orig_prior),
            base_prob=base_prob,
            base_neural=bank['base_neural'],
        )

        with patch.object(
            Engine,
            'from_observation',
            side_effect=AssertionError("Engine.from_observation must never be called during online neural inference!"),
        ):
            row_online = extract_actor_row(obs, bank, adapter, temporal)
            self.assertEqual(row_online['a'].shape[1], 14)
            self.assertEqual(row_online['m'].shape[2], 16)
            for arm in NEURAL_EVAL_ARMS:
                step = neural_actor_step(
                    None,
                    _DummyPrototype,
                    None,
                    None,
                    obs,
                    adapter,
                    temporal,
                    arm,
                    models[arm],
                    0,
                    953000,
                    0,
                    3,
                    precomputed_base=precomputed_base,
                )
                self.assertEqual(step['deterministic_engine_steps'], 0)
                self.assertEqual(step['forecast_calls'], 0)
                self.assertEqual(len(step['motion']), 5)
                self.assertIsInstance(step['plan'], dict)

        for ref_arm in ('quality_amortized', 'old_plan', 'joint_neural_mpc'):
            step_ref = neural_actor_step(
                None,
                _DummyPrototype,
                None,
                None,
                obs,
                adapter,
                temporal,
                ref_arm,
                models['joint_trained'],
                0,
                953000,
                0,
                3,
                precomputed_base=precomputed_base,
            )
            self.assertEqual(len(step_ref['motion']), 5)
            self.assertIsInstance(step_ref['plan'], dict)
            if ref_arm == 'quality_amortized':
                self.assertGreater(step_ref['deterministic_engine_steps'], 0)
                self.assertEqual(step_ref['forecast_calls'], 0)
            else:
                self.assertEqual(step_ref['deterministic_engine_steps'], 0)
                self.assertGreater(step_ref['forecast_calls'], 0)

    def test_compose_decomposed_actor_and_advantage_weighted_target(self):
        _, _, row, _, _, _ = _sample_state(37)
        lower_m = initialize_joint_actor(row, hidden=10, seed=41)
        upper_m = initialize_joint_actor(row, hidden=10, seed=43)
        rng = np.random.default_rng(47)
        for k in ('w', 'W', 'b', 'v'):
            lower_m['parameters']['allocation'][k] = rng.normal(
                0.0, 0.2, size=lower_m['parameters']['allocation'][k].shape
            )
            upper_m['parameters']['motion'][k] = rng.normal(
                0.0, 0.2, size=upper_m['parameters']['motion'][k].shape
            )

        composed = compose_decomposed_actor(lower_m, upper_m)
        self.assertEqual(composed['composed_from'], ('lower_only_trained', 'upper_only_trained'))
        pa_low, pm_low, *_ = actor_forward(row, lower_m, mode='lower_only')
        pa_up, pm_up, *_ = actor_forward(row, upper_m, mode='upper_only')
        pa_comp, pm_comp, *_ = actor_forward(row, composed, mode='decomposed')
        np.testing.assert_allclose(pa_comp, pa_low, atol=1e-12)
        np.testing.assert_allclose(pm_comp, pm_up, atol=1e-12)

        values = np.array([[0.20, 0.35, 0.10], [0.50, 0.48, 0.15]], dtype=np.float64)
        prior = np.array([0.4, 0.6], dtype=np.float64)
        mp = np.full_like(values, 1.0 / values.shape[1])
        target = advantage_weighted_target(values, prior=prior, motion_prior=mp, temperature=0.12)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=12)
        self.assertTrue(np.all(target > 0.0))
        self.assertEqual(np.unravel_index(int(np.argmax(target)), target.shape), (1, 0))

    def test_fit_joint_actor_monotonically_reduces_kl_and_regret(self):
        states = [_sample_state(s) for s in (10, 11, 12, 13)]
        rows = [s[2] for s in states]
        joint_targets = [s[3]['joint'][0] for s in states]
        joint_values = [s[3]['joint'][1] for s in states]

        init_model = initialize_joint_actor(rows[0], hidden=24, seed=42)
        trained, info, ckpts = fit_joint_actor(
            rows[:3], joint_targets[:3], joint_values[:3],
            rows[3:], joint_targets[3:], joint_values[3:],
            init_model, mode='joint', epochs=32, batch_size=2,
            learning_rate=0.01, seed=42, checkpoint_epochs=(0, 4, 16, 32),
        )
        curve = info['curve']
        self.assertEqual([c['epoch'] for c in curve], [0, 4, 16, 32])
        self.assertLess(curve[-1]['train']['kl'], curve[0]['train']['kl'] * 0.5)
        self.assertLessEqual(curve[-1]['train']['top1_regret'], curve[0]['train']['top1_regret'] + 1e-9)
        self.assertIn(0, ckpts)
        self.assertIn(32, ckpts)

    def test_report_table_and_lindley_counterexample_audit(self):
        import json
        from pathlib import Path
        from scripts.analysis.review_queued_joint_training import (
            format_markdown_comparison_table,
            lindley_counterexample_audit,
        )

        ce = lindley_counterexample_audit()
        self.assertEqual(ce['engine_success'], 2)
        self.assertEqual(ce['engine_expired'], 0)
        self.assertEqual(ce['engine_audit'], 'PASS')
        self.assertEqual(ce['static_lindley_predicted_success'], 1.0)
        self.assertEqual(ce['static_lindley_predicted_overflow'], 1.0)

        root = Path(__file__).resolve().parents[1]
        conf_path = root / 'results/chapter5_dynamic_service_20261028/queued_joint_training_20261105/confirmation_analysis.json'
        md_path = root / 'output/chapter5_dynamic_service_20261028/deliverables/queued_joint_training.md'
        if conf_path.exists() and md_path.exists():
            data = json.loads(conf_path.read_text(encoding='utf-8'))
            expected_table = format_markdown_comparison_table(data)
            md_text = md_path.read_text(encoding='utf-8')
            self.assertIn(expected_table, md_text)


if __name__ == '__main__':
    unittest.main()
