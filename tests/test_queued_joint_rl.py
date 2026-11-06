"""Unit tests for true environment-reward Joint RL (PPO-GAE), finite-difference gradients, and EDF counterexample fix."""
import copy
import unittest
from unittest.mock import patch
import numpy as np

from environment.queued_service import Engine, Protocol, Task
from marl_models.queued_joint_rl import (
    PLAN_DEPENDENT_MOTION_CHANNELS,
    RL_MODES,
    compose_decomposed_rl_actor,
    compute_episode_gae_and_returns,
    critic_forward,
    critic_loss_and_gradient,
    fit_clean_distilled_actor,
    initialize_rl_actor_critic,
    ppo_actor_loss_and_gradient,
    ppo_update_actor_critic,
    prepare_rl_branch_inputs,
    rl_actor_forward,
    rl_actor_policy,
    warmstart_from_distilled_actor,
)
from marl_models.queued_joint_training import initialize_joint_actor
from utils.queued_arrival_adaptation import RateHistory
from utils.queued_joint_rl import (
    _release_time_tandem_queue_eval,
    extract_clean_actor_row,
    quantify_tandem_approximation_error,
    rl_actor_step,
    state_potential,
)
from utils.queued_joint_training import _lindley_plan_motion_eval
from utils.queued_quality_amortization import expand_safe_plan_bank, safe_assignments
from utils.queued_service_controllers import assignments
from utils.queued_service_learning import LearningAdapter
from utils.queued_waiting_rescue import rescue_candidates


def _sample_clean_state(seed=7):
    rng = np.random.default_rng(seed)
    positions = [(150.0, 150.0), (400.0, 150.0), (620.0, 150.0), (200.0, 450.0), (500.0, 450.0)]
    ues = {i: (float(rng.uniform(80, 620)), float(rng.uniform(80, 620))) for i in range(30)}
    engine = Engine(Protocol(), positions, ues)
    adapter = LearningAdapter()
    temporal = RateHistory()
    for t in range(4):
        tasks = [
            Task(
                t * 20 + k,
                int(rng.integers(0, 30)),
                float(t),
                float(t) + 2.2 + float(rng.uniform(0, 0.8)),
                float(8 * rng.integers(1_500_000, 2_500_000)),
                float(8 * rng.integers(1_500_000, 2_500_000) * rng.integers(550, 900)),
            )
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
        pressure=1.35 if seed % 2 == 0 else 0.70,
        rate3=48.0 if seed % 2 == 0 else 18.0,
        base_neural=base_neural,
        metadata=meta,
    )
    row = extract_clean_actor_row(obs, bank, adapter, temporal)
    return obs, bank, row, adapter, temporal


class TestQueuedJointRL(unittest.TestCase):
    def test_release_time_tandem_queue_fixes_cross_stage_counterexample(self):
        """Verify that `_release_time_tandem_queue_eval` resolves the 1-UAV, 2-task cross-stage counterexample."""
        proto = Protocol(capacity=(100e9, 200e9), uplink_bps=10e6, inter_bps=10e6, backhaul_bps=10e6)
        engine = Engine(proto, [(200.0, 200.0)], {0: (200.0, 200.0), 1: (200.0, 200.0)})
        # Task 1: earlier deadline (1.4s), needs 0.7s upload + 0.5s CPU
        # Task 2: later deadline (1.5s), 0 upload (already ready for CPU at t=0), needs 0.6s CPU
        t1 = Task(1, 0, 0.0, 1.4, 7e6, 50e9)
        t2 = Task(2, 1, 0.0, 1.5, 0.0, 60e9)
        engine.add([t1, t2])
        engine.admit()
        engine.assign({1: 0, 2: 0})
        obs = engine.observe()
        admitted = sorted(
            [t for t in obs['tasks'] if t['source'] is not None],
            key=lambda z: (z['deadline'], z['uid']),
        )

        old_comp, old_over = _lindley_plan_motion_eval(obs, admitted, [], {1: 0, 2: 0}, [(0.0, 0.0)])
        new_comp, new_over = _release_time_tandem_queue_eval(obs, admitted, [], {1: 0, 2: 0}, [(0.0, 0.0)])

        for _ in range(2):
            engine.advance([(0.0, 0.0)])
        audit = engine.audit()

        self.assertEqual(audit['status'], 'PASS')
        self.assertEqual(int(engine.counts['success']), 2)
        self.assertEqual(int(engine.counts['expired']), 0)
        # Old static Lindley had the false-expiration counterexample (1 success, 1 overflow)
        self.assertEqual((old_comp, old_over), (1.0, 1.0))
        # New release-time work-conserving EDF matches Engine (2 successes, 0 overflow)!
        self.assertEqual((new_comp, new_over), (2.0, 0.0))

    def test_unconfounded_feature_masking_and_isolation_of_c_upper(self):
        """Verify that `decomposed` preserves global congestion/rescue channels and `joint_no_expectation` isolates C_upper."""
        _, _, row, _, _ = _sample_clean_state(14)  # Even seed -> congested (pressure=1.35, rate3=48.0)
        k_plans, m_actions = row['motion_prior'].shape
        nonuniform_pm = np.linspace(1.0, 3.0, m_actions, dtype=np.float64)[None, :]
        nonuniform_pm = np.broadcast_to(nonuniform_pm / nonuniform_pm.sum(), (k_plans, m_actions))

        xa_joint, xm_joint, *_ = prepare_rl_branch_inputs(row, mode='joint', pm_override=nonuniform_pm)
        xa_no_exp, xm_no_exp, *_ = prepare_rl_branch_inputs(row, mode='joint_no_expectation', pm_override=nonuniform_pm)
        xa_dec, xm_dec, *_ = prepare_rl_branch_inputs(row, mode='decomposed', pm_override=nonuniform_pm)

        # 1. `joint` and `joint_no_expectation` have 100% identical motion inputs `xm`!
        np.testing.assert_allclose(xm_joint, xm_no_exp, atol=1e-12)
        # Their ONLY difference in `xa` is the 4D C_upper slice `xa[:, 14:18]`
        da = row['a'].shape[1]
        np.testing.assert_allclose(xa_no_exp[:, da:da + 4], 0.0, atol=1e-12)
        self.assertGreater(float(np.linalg.norm(xa_joint[:, da:da + 4])), 1e-4)

        # 2. `decomposed` retains global congestion (channel 9) and rescue interaction (channel 10)
        self.assertGreater(float(np.max(np.abs(xm_dec[:, :, 9]))), 0.5)
        self.assertGreater(float(np.max(np.abs(xm_dec[:, :, 10]))), 0.5)
        # And in `decomposed`, `xm` has identical lower-plan-conditioned channels across plans `k`:
        for ch in PLAN_DEPENDENT_MOTION_CHANNELS:
            for k in range(1, xm_dec.shape[0]):
                np.testing.assert_allclose(xm_dec[k, :, ch], xm_dec[0, :, ch], atol=1e-12)
        for ch in range(16):
            if ch not in PLAN_DEPENDENT_MOTION_CHANNELS:
                np.testing.assert_allclose(xm_dec[:, :, ch], xm_joint[:, :, ch], atol=1e-12)

    def test_true_decomposed_zero_plan_dependence(self):
        """Verify `decomposed` upper motion probabilities p_m(m|k) are strictly invariant across lower plans k (< 1e-14)."""
        _, _, row, _, _ = _sample_clean_state(14)
        rng = np.random.default_rng(73)
        # Make `motion_prior[k, m]` strongly dependent on both k and m
        raw_mp = rng.uniform(0.05, 1.0, size=row['motion_prior'].shape)
        row['motion_prior'] = raw_mp / raw_mp.sum(axis=1, keepdims=True)
        row['m'][:, :, 15] = np.clip((np.log(row['motion_prior']) + 2.5) / 2.5, -2.0, 2.0)

        model = initialize_rl_actor_critic(row, hidden=12, critic_hidden=12, seed=79)
        for branch in ('allocation', 'motion'):
            for k in ('w', 'W', 'b', 'v'):
                model['parameters'][branch][k] = rng.normal(0.0, 0.35, size=model['parameters'][branch][k].shape)

        for mode in ('decomposed', 'upper_only', 'lower_only'):
            pa, pm, *_ = rl_actor_forward(row, model, mode=mode)
            joint = rl_actor_policy(row, model, mode=mode)
            max_diff = float(np.max(np.abs(pm - pm[:1, :])))
            self.assertLess(max_diff, 1e-14, msg=f"Plan dependence leaked in mode={mode}: max_diff={max_diff}")
            np.testing.assert_allclose(joint, np.outer(pa, pm[0]), atol=1e-14)

        # Also verify after PPO updates and in composed actor (`lower_only` + `upper_only`)
        target_pair = (0, 0)
        old_lp = float(np.log(rl_actor_policy(row, model, mode='decomposed')[target_pair]))
        fake_ep = [dict(row=row, action_pair=target_pair, old_logp=old_lp, reward=2.5) for _ in range(4)]
        updated_dec, _ = ppo_update_actor_critic([fake_ep], model, mode='decomposed', ppo_epochs=3, critic_epochs=3, seed=83)
        updated_low, _ = ppo_update_actor_critic([fake_ep], model, mode='lower_only', ppo_epochs=3, critic_epochs=3, seed=89)
        updated_up, _ = ppo_update_actor_critic([fake_ep], model, mode='upper_only', ppo_epochs=3, critic_epochs=3, seed=97)
        composed = compose_decomposed_rl_actor(updated_low, updated_up)

        for check_model in (updated_dec, composed):
            pa, pm, *_ = rl_actor_forward(row, check_model, mode='decomposed')
            joint = rl_actor_policy(row, check_model, mode='decomposed')
            self.assertLess(float(np.max(np.abs(pm - pm[:1, :]))), 1e-14)
            np.testing.assert_allclose(joint, np.outer(pa, pm[0]), atol=1e-14)

    def test_drain_steps_recorded_and_reward_conservation(self):
        """Verify post-200s drain transitions are recorded with their own (row, action_pair, old_logp, reward) and conserve counts."""
        positions = [(150.0, 150.0), (400.0, 150.0), (620.0, 150.0), (200.0, 450.0), (500.0, 450.0)]
        ues = {i: (150.0 + (i % 5) * 2.0, 150.0 + (i // 5) * 2.0) for i in range(10)}
        engine = Engine(Protocol(), positions, ues)
        adapter = LearningAdapter()
        temporal = RateHistory()
        # Advance engine directly to t=199 with tasks that require 2 steps (t=199 and drain step t=200) to finish
        engine.t = 199.0
        t_active = Task(1, 0, 199.0, 202.5, 2e6, 20e9)
        t_drain = Task(2, 1, 199.0, 202.5, 18e6, 150e9)
        engine.add([t_active, t_drain])

        _, _, sample_row, _, _ = _sample_clean_state(11)
        model = initialize_rl_actor_critic(sample_row, hidden=8, critic_hidden=8, seed=101)
        steps = []
        rng = np.random.default_rng(103)

        while engine.t < 200 or (engine.tasks and engine.t < 205):
            now = int(round(engine.t))
            engine.admit()
            obs = engine.observe()
            adapter.prepare(obs)
            temporal.update(obs)
            base_plans = [assignments(obs), safe_assignments(obs)]
            plans, prior = expand_safe_plan_bank(obs, base_plans, [0.6, 0.4])
            base_actions = [[(0.0, 0.0)] * 5, [(5.0, 0.0)] * 5]
            actions, meta = rescue_candidates(obs, base_actions)
            mp = np.full((len(plans), len(actions)), 1.0 / len(actions), dtype=np.float64)
            bank = dict(
                plans=plans, actions=actions, prior=prior, motion_prior=mp,
                original_count=len(base_actions), original_plan_count=len(base_plans),
                safe_plan=safe_assignments(obs), pressure=0.8, rate3=20.0,
                base_neural=np.zeros((len(base_plans), len(base_actions))), metadata=meta,
            )
            row = extract_clean_actor_row(obs, bank, adapter, temporal)
            p_mat = rl_actor_policy(row, model, mode='joint')
            flat_idx = int(rng.choice(p_mat.size, p=p_mat.reshape(-1)))
            k_idx, m_idx = divmod(flat_idx, p_mat.shape[1])
            prev_suc = int(engine.counts['success'])
            prev_exp = int(engine.counts['expired'])
            phi_before = state_potential(engine)
            engine.assign(plans[k_idx])
            engine.advance(actions[m_idx])
            d_suc = int(engine.counts['success']) - prev_suc
            d_exp = int(engine.counts['expired']) - prev_exp
            is_terminal = not (engine.t < 200 or (engine.tasks and engine.t < 205))
            phi_after = 0.0 if is_terminal else state_potential(engine)
            reward = float(d_suc) - 0.65 * float(d_exp) + (0.85 * phi_after - phi_before)
            steps.append(dict(
                time=now, is_drain=bool(now >= 200), row=row,
                action_pair=(int(k_idx), int(m_idx)), old_logp=float(np.log(p_mat[k_idx, m_idx])),
                reward=float(reward), d_success=d_suc, d_expired=d_exp,
            ))

        drain_steps = [s for s in steps if s['is_drain']]
        self.assertGreaterEqual(len(drain_steps), 1)
        self.assertEqual(sum(s['d_success'] for s in steps), int(engine.counts['success']))
        self.assertEqual(sum(s['d_expired'] for s in steps), int(engine.counts['expired']))
        trs, gae_stats = compute_episode_gae_and_returns([steps], model, gamma=0.85, gae_lambda=0.90)
        self.assertEqual(gae_stats['transitions'], len(steps))
        self.assertEqual(len(trs), len(steps))

    def test_ppo_actor_and_critic_finite_difference_gradients_all_modes(self):
        """Verify exact analytical gradients for Clipped Joint PPO and Critic across all 5 RL modes."""
        _, _, row, _, _ = _sample_clean_state(19)
        eps = 1e-6
        ref_p = row['prior'][:, None] * row['motion_prior']
        ref_p /= ref_p.sum()

        for mode in RL_MODES:
            model = initialize_rl_actor_critic(row, hidden=8, critic_hidden=8, seed=23)
            rng = np.random.default_rng(107)
            for branch in ('allocation', 'motion'):
                for k in ('w', 'W', 'b', 'v'):
                    model['parameters'][branch][k] = rng.normal(
                        0.0, 0.25, size=model['parameters'][branch][k].shape
                    )
            for k in ('w', 'W', 'b', 'v', 'c0'):
                model['critic'][k] = rng.normal(0.0, 0.25, size=model['critic'][k].shape)

            p_init = rl_actor_policy(row, model, mode=mode)
            action_pair = (1 % p_init.shape[0], min(2, p_init.shape[1] - 1))
            # Test unclipped active region
            old_logp = float(np.log(p_init[action_pair]) - 0.03)
            advantage = 1.35

            loss, grad = ppo_actor_loss_and_gradient(
                row,
                model,
                action_pair,
                old_logp,
                advantage,
                mode=mode,
                clip_eps=0.20,
                ent_coef=0.01,
                kl_coef=0.05,
                ref_policy=ref_p,
                l2_reg=1e-4,
            )
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
                        loss_plus, _ = ppo_actor_loss_and_gradient(
                            row, model, action_pair, old_logp, advantage, mode=mode,
                            clip_eps=0.20, ent_coef=0.01, kl_coef=0.05, ref_policy=ref_p, l2_reg=1e-4,
                        )
                        flat_param[idx] = orig - eps
                        loss_minus, _ = ppo_actor_loss_and_gradient(
                            row, model, action_pair, old_logp, advantage, mode=mode,
                            clip_eps=0.20, ent_coef=0.01, kl_coef=0.05, ref_policy=ref_p, l2_reg=1e-4,
                        )
                        flat_param[idx] = orig
                        fd = (loss_plus - loss_minus) / (2.0 * eps)
                        self.assertAlmostEqual(
                            float(flat_analytic[idx]),
                            float(fd),
                            places=6,
                            msg=f"PPO actor gradient mismatch in mode={mode}, branch={branch}, param={k}, idx={idx}",
                        )

            # Verify Critic analytical gradients against finite differences
            c_loss, c_grad = critic_loss_and_gradient(row, model, target_return=4.25, huber_delta=5.0, l2_reg=1e-4)
            self.assertTrue(np.isfinite(c_loss))
            for k in ('w', 'W', 'b', 'v', 'c0'):
                param = model['critic'][k]
                analytic = c_grad[k]
                flat_param = param.reshape(-1)
                flat_analytic = analytic.reshape(-1)
                indices = np.linspace(0, len(flat_param) - 1, num=min(10, len(flat_param)), dtype=int)
                for idx in indices:
                    orig = float(flat_param[idx])
                    flat_param[idx] = orig + eps
                    lp, _ = critic_loss_and_gradient(row, model, target_return=4.25, huber_delta=5.0, l2_reg=1e-4)
                    flat_param[idx] = orig - eps
                    lm, _ = critic_loss_and_gradient(row, model, target_return=4.25, huber_delta=5.0, l2_reg=1e-4)
                    flat_param[idx] = orig
                    fd = (lp - lm) / (2.0 * eps)
                    self.assertAlmostEqual(
                        float(flat_analytic[idx]),
                        float(fd),
                        places=6,
                        msg=f"Critic gradient mismatch on param={k}, idx={idx}",
                    )

            # Check cross-layer coupling difference between 'joint' and 'joint_no_expectation'
            if mode in ('joint', 'joint_no_expectation'):
                pa0, *_ = rl_actor_forward(row, model, mode=mode)
                perturbed = copy.deepcopy(model)
                perturbed['parameters']['motion']['w'] += 0.4
                pa1, *_ = rl_actor_forward(row, perturbed, mode=mode)
                if mode == 'joint':
                    self.assertGreater(float(np.max(np.abs(pa1 - pa0))), 1e-5)
                else:
                    np.testing.assert_allclose(pa1, pa0, atol=1e-12)

    def test_ppo_update_improves_action_probability_and_trains_critic(self):
        """Verify on-policy PPO-GAE update increases high-return action probability without any teacher."""
        obs, bank, row, adapter, temporal = _sample_clean_state(29)
        distilled_init = initialize_joint_actor(row, hidden=16, seed=31)
        model = warmstart_from_distilled_actor(distilled_init, row, critic_hidden=16, seed=37)

        p_before = rl_actor_policy(row, model, mode='joint')
        target_pair = (int(p_before.shape[0] - 1), int(p_before.shape[1] - 1))
        other_pair = (0, 0)
        old_lp_target = float(np.log(p_before[target_pair]))
        old_lp_other = float(np.log(p_before[other_pair]))

        # Construct synthetic trajectory where `target_pair` receives high environment reward (+4.0)
        # and `other_pair` receives low environment reward (-2.0)
        fake_episode = [
            dict(
                row=row,
                action_pair=target_pair if step_i % 2 == 0 else other_pair,
                old_logp=old_lp_target if step_i % 2 == 0 else old_lp_other,
                ref_policy=p_before,
                reward=4.0 if step_i % 2 == 0 else -2.0,
            )
            for step_i in range(8)
        ]

        updated, stats = ppo_update_actor_critic(
            [fake_episode],
            model,
            mode='joint',
            ppo_epochs=6,
            critic_epochs=10,
            batch_size=4,
            actor_lr=0.01,
            critic_lr=0.02,
            kl_coef=0.01,
            gamma=0.5,
            seed=41,
        )
        p_after = rl_actor_policy(row, updated, mode='joint')
        v_after, *_ = critic_forward(row, updated)

        self.assertEqual(stats['transitions'], 8)
        self.assertGreater(float(p_after[target_pair]), float(p_before[target_pair]))
        self.assertLess(float(p_after[other_pair]), float(p_before[other_pair]))
        self.assertGreater(abs(v_after), 0.05)

        # Also check composed RL actor assembly and approximation error quantifier
        comp = compose_decomposed_rl_actor(updated, updated)
        self.assertEqual(comp['composed_from'], ('lower_only_rl', 'upper_only_rl'))
        err_stats = quantify_tandem_approximation_error(obs, bank['plans'][:2], bank['actions'][:2])
        self.assertEqual(err_stats['pairs'], 4)
        self.assertGreaterEqual(err_stats['mae'], 0.0)

        # Verify zero Engine calls during online RL actor inference
        class _DummyProto:
            REVISION = 'test_rl'

            class t:
                @staticmethod
                def seed(*args):
                    return 99

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
            side_effect=AssertionError("Engine.from_observation must never be called during online RL neural step!"),
        ):
            step = rl_actor_step(
                None, _DummyProto, None, None, obs, adapter, temporal,
                'joint_rl', updated, 0, 954000, 0, 3, precomputed_base=precomputed_base,
            )
            self.assertEqual(step['deterministic_engine_steps'], 0)
            self.assertEqual(step['forecast_calls'], 0)

        # Verify LOO-CRN group baseline in `compute_episode_gae_and_returns` and `fit_clean_distilled_actor`
        ep_a = [
            dict(time=0, workload=940000, row=row, action_pair=target_pair, old_logp=old_lp_target, reward=5.0),
            dict(time=1, workload=940000, row=row, action_pair=other_pair, old_logp=old_lp_other, reward=-1.0),
        ]
        ep_b = [
            dict(time=0, workload=940000, row=row, action_pair=other_pair, old_logp=old_lp_other, reward=1.0),
            dict(time=1, workload=940000, row=row, action_pair=target_pair, old_logp=old_lp_target, reward=3.0),
        ]
        trs, gae_stats = compute_episode_gae_and_returns([ep_a, ep_b], updated, gamma=0.85, gae_lambda=0.90)
        self.assertEqual(gae_stats['transitions'], 4)
        self.assertNotAlmostEqual(trs[0]['raw_gae_advantage'], trs[0]['advantage'])

        target_dist = np.full_like(p_before, 1.0 / p_before.size)
        target_dist[target_pair] += 0.5
        target_dist /= target_dist.sum()
        distilled, d_info = fit_clean_distilled_actor(
            [row, row], [target_dist, target_dist], [row], [target_dist],
            initialize_rl_actor_critic(row, hidden=8, critic_hidden=8, seed=53),
            mode='joint', epochs=6, batch_size=2, learning_rate=0.01, seed=59,
        )
        self.assertTrue(np.isfinite(d_info['best_val_kl']))
        self.assertGreater(float(rl_actor_policy(row, distilled, mode='joint')[target_pair]), float(p_before[target_pair]))

    def test_freeze_manifest_and_report_tables_exact_match(self):
        """Verify freeze_manifest.json SHA256/timestamps, rollout/decomposed audits, and exact Markdown tables in `queued_joint_rl.md`."""
        import json
        from pathlib import Path
        from scripts.analysis.review_queued_joint_rl import (
            format_rl_comparison_table,
            format_rl_selection_table,
            format_rl_summary_table,
            verify_freeze_manifest,
        )

        root = Path(__file__).resolve().parents[1]
        folder = root / 'results/chapter5_dynamic_service_20261028/queued_joint_rl_v2_20261106'
        report_path = root / 'output/chapter5_dynamic_service_20261028/deliverables/queued_joint_rl.md'
        if not (folder / 'confirmation_analysis.json').exists() or not report_path.exists():
            return

        freeze_audit = verify_freeze_manifest(root, folder)
        self.assertTrue(freeze_audit['verified'], msg=f"Freeze manifest audit failed: {freeze_audit}")
        self.assertEqual(freeze_audit['checked_confirmation_episodes'], 816)

        rollout_audit = json.loads((folder / 'rollout_replay_audit.json').read_text(encoding='utf-8'))
        self.assertTrue(rollout_audit['verified'])
        self.assertEqual(rollout_audit['checked_episodes'], 1152)

        decomp_audit = json.loads((folder / 'decomposed_independence_audit.json').read_text(encoding='utf-8'))
        self.assertTrue(decomp_audit['verified'])

        conf = json.loads((folder / 'confirmation_analysis.json').read_text(encoding='utf-8'))
        curves = json.loads((folder / 'rl_training_curves.json').read_text(encoding='utf-8'))
        report_md = report_path.read_text(encoding='utf-8')
        self.assertIn(format_rl_summary_table(conf), report_md)
        self.assertIn(format_rl_comparison_table(conf), report_md)
        self.assertIn(format_rl_selection_table(curves), report_md)
        self.assertIn('[32, 44]', report_md)
        self.assertIn('34', report_md)
        self.assertNotIn('250,224', report_md)
        self.assertNotIn('old_plan', report_md)


if __name__ == '__main__':
    unittest.main()
