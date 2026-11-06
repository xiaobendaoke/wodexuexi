"""Pure numerical and public-observation contract tests; no local training."""
import copy
import unittest
import numpy as np


class ImprovementTests(unittest.TestCase):
    def test_support_ties_scaling_and_kl(self):
        from utils.queued_joint_mpo import improve, divergence
        p = np.array([[.15, .35], [.5, 0.]])
        q, info = improve([p], [np.array([[1., 3.], [2., 999.]])], .07)
        self.assertEqual(q[0][1, 1], 0.)
        self.assertLessEqual(divergence(q[0], p), .070000001)
        self.assertGreater(info['gain'], 0.)
        scaled, _ = improve([p], [np.array([[107., 121.], [114., 9999.]])], .07)
        np.testing.assert_allclose(q, scaled, atol=1e-10)
        tied, _ = improve([p], [np.ones((2, 2))], .07)
        np.testing.assert_allclose(tied[0], p)

    def test_dedup_does_not_multiply_actor_mass(self):
        from utils.queued_joint_mpo import unique_plans, restricted_prior
        plans = [{1: 2}, {1: 2}, {1: 3}]
        self.assertEqual(unique_plans(plans), [{1: 2}, {1: 3}])
        p, mass = restricted_prior([np.log(.1), np.log(.2)], exploration=0)
        np.testing.assert_allclose(p, [1/3, 2/3]); self.assertAlmostEqual(mass, .3)

    def test_chain_rule_and_weighted_projection_gradient(self):
        from utils.queued_joint_mpo import divergence
        from marl_models.queued_joint_mpo import policy, loss_gradient
        rng = np.random.default_rng(3)
        row = dict(a=rng.normal(size=(3, 4)), m=rng.normal(size=(3, 2, 5)), prior=np.array([.2, .3, .5]))
        theta = np.zeros(9); p = policy(row, theta)
        q = np.array([[.05, .05], [.1, .2], [.5, .1]])
        qa, pa = q.sum(1), p.sum(1)
        self.assertAlmostEqual(divergence(q, p), divergence(qa, pa)+sum(qa[i]*divergence(q[i]/qa[i], p[i]/pa[i]) for i in range(3)))
        loss, gradient = loss_gradient(row, theta, q)
        for j in range(9):
            plus = theta.copy(); minus = theta.copy(); plus[j] += 1e-5; minus[j] -= 1e-5
            fd = (loss_gradient(row, plus, q)[0]-loss_gradient(row, minus, q)[0])/2e-5
            self.assertAlmostEqual(fd, gradient[j], places=7)

    def test_rejected_adam_and_parameters_are_identical(self):
        from marl_models.queued_joint_mpo import transactional_step
        theta = np.array([.1, .2]); state = dict(m=np.zeros(2), v=np.zeros(2), step=0)
        old = copy.deepcopy(state)
        new, updated, info = transactional_step(theta, state, np.ones(2), lambda x: False)
        np.testing.assert_array_equal(new, theta)
        for key in old: np.testing.assert_array_equal(updated[key], old[key])
        self.assertFalse(info['accepted'])

    def test_forecast_interface_has_no_private_future(self):
        import inspect
        from utils.queued_joint_mpo import forecast_window
        self.assertEqual(list(inspect.signature(forecast_window).parameters), ['obs', 'history', 'plan', 'motion', 'arrivals', 'horizon'])

    def test_flat_and_decomposed_gradients_and_sharding(self):
        from marl_models.queued_joint_mpo import policy, loss_gradient
        rng = np.random.default_rng(6)
        rows = [dict(a=rng.normal(size=(2, 4)), m=rng.normal(size=(2, 3, 5)), prior=np.array([.4,.6])) for _ in range(5)]
        theta = rng.normal(size=9)*.03
        target = np.array([[.02,.03,.05],[.1,.3,.5]])
        for mode in ('flat','decomposed'):
            loss, grad = loss_gradient(rows[0], theta, target, mode)
            for j in range(9):
                plus=theta.copy(); minus=theta.copy(); plus[j]+=1e-5; minus[j]-=1e-5
                fd=(loss_gradient(rows[0],plus,target,mode)[0]-loss_gradient(rows[0],minus,target,mode)[0])/2e-5
                self.assertAlmostEqual(fd,grad[j],places=7)
            all_grad=np.mean([loss_gradient(r,theta,target,mode)[1] for r in rows],axis=0)
            shards=[rows[:2],rows[2:]]
            combined=sum(np.sum([loss_gradient(r,theta,target,mode)[1] for r in s],axis=0) for s in shards)/len(rows)
            np.testing.assert_allclose(combined,all_grad,atol=1e-12)
        p=policy(rows[0],theta,'decomposed')
        np.testing.assert_allclose(p[0]/p[0].sum(),p[1]/p[1].sum())

    def test_clipped_surrogate_gradient(self):
        from marl_models.queued_joint_mpo import policy, ppo_gradient
        rng=np.random.default_rng(9)
        row=dict(a=rng.normal(size=(2,4)),m=rng.normal(size=(2,3,5)),prior=np.array([.4,.6]))
        old=policy(row,np.zeros(9)); theta=rng.normal(size=9)*.02; advantage=rng.normal(size=(2,3))
        _,gradient=ppo_gradient(row,theta,old,advantage)
        for j in range(9):
            plus=theta.copy();minus=theta.copy();plus[j]+=1e-5;minus[j]-=1e-5
            fd=(ppo_gradient(row,plus,old,advantage)[0]-ppo_gradient(row,minus,old,advantage)[0])/2e-5
            self.assertAlmostEqual(fd,gradient[j],places=7)


if __name__ == '__main__': unittest.main()
