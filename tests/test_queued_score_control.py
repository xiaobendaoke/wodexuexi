import unittest
from types import SimpleNamespace
import numpy as np
from marl_models.queued_planner_distillation import initialize
from utils.queued_score_control import score_data, score_gradient, controlled_policy, congestion_pressure


class ScoreControlTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(51)
        self.row=dict(a=rng.normal(size=(2,3)), m=rng.normal(size=(2,3,4)), context=np.array([.3]),
                      prior=np.array([.4,.6]), q=np.arange(6.).reshape(2,3),time=60)
        self.forecasts=np.tile(self.row['q'],(16,1,1))

    def test_score_gradients(self):
        for mode in ('joint','decomposed'):
            data=score_data([self.row],[self.forecasts],mode)
            model=initialize(self.row,hidden=3,seed=7)
            model['parameters']['allocation']['v'][:]=[.1,.2,-.2]
            model['parameters']['motion']['v'][:]=[.3,-.1,.2]
            for objective in ('reg','rawrank'):
                _,grad=score_gradient(data,model,objective)
                for branch in model['parameters']:
                    for key,values in model['parameters'][branch].items():
                        index=tuple(0 for _ in values.shape);old=values[index];eps=1e-5
                        values[index]=old+eps;a=score_gradient(data,model,objective)[0]
                        values[index]=old-eps;b=score_gradient(data,model,objective)[0]
                        values[index]=old
                        self.assertAlmostEqual(grad[branch][key][index],(a-b)/(2*eps),places=6)

    def test_root_shift_invariance(self):
        shifted=dict(self.row,q=self.row['q']+100)
        a=score_data([self.row],[self.forecasts],'joint')
        b=score_data([shifted],[self.forecasts+100],'joint')
        model=initialize(self.row,hidden=3)
        for objective in ('reg','rawrank'):
            x,g=score_gradient(a,model,objective);y,h=score_gradient(b,model,objective)
            self.assertAlmostEqual(x,y)
            for branch in g:
                for key in g[branch]:np.testing.assert_allclose(g[branch][key],h[branch][key],atol=1e-12)

    def test_inactive_gate_exact_anchor(self):
        p=np.array([[.1,.2],[.3,.4]])
        for pressure in (0.,.9,1.):
            q,record=controlled_policy(p,np.array([[4.,3.],[2.,1.]]),pressure,'gated')
            np.testing.assert_array_equal(q,p);self.assertEqual(record['tv'],0.)

    def test_mixture_bound_permutation_and_flat_scores(self):
        p=np.array([[.1,.2],[.3,.4]]);scores=np.array([[4.,3.],[2.,1.]])
        q,r=controlled_policy(p,scores,2.,'gated');self.assertLessEqual(r['tv'],.25)
        moved,_=controlled_policy(p[::-1,::-1],scores[::-1,::-1],2.,'gated')
        np.testing.assert_allclose(moved,q[::-1,::-1])
        flat,_=controlled_policy(p,np.ones((2,2)),2.,'gated');np.testing.assert_allclose(flat,p)

    def test_branch_replacement_contracts(self):
        p=np.array([[.1,.2],[.3,.4]]);scores=np.array([[4.,3.],[2.,1.]])
        for branch in ('allocation','motion'):
            q,r=controlled_policy(p,scores,2.,'gated',branch)
            self.assertLessEqual(r['tv'],.25)
            if branch=='motion':np.testing.assert_allclose(q.sum(1),p.sum(1))
            else:np.testing.assert_allclose(q/q.sum(1,keepdims=True),p/p.sum(1,keepdims=True))

    def test_pressure_ignores_future_fields_and_waiting_tasks(self):
        task=dict(uid=1,source=0,target=0,deadline=2.,remaining_cycles=3.)
        obs=dict(time=0.,tasks=[task],protocol=SimpleNamespace(capacity=(1.,)*6),future='unused')
        self.assertEqual(congestion_pressure(obs,[{}],[1.]),1.5)
        obs['future']='different';obs['tasks'].append(dict(task,uid=2,source=None,remaining_cycles=100.))
        self.assertEqual(congestion_pressure(obs,[{}],[1.]),1.5)

    def test_zero_anchor_support_remains_finite(self):
        p=np.array([[0.,.2],[.3,.5]])
        q,record=controlled_policy(p,np.array([[100.,3.],[2.,1.]]),2.,'gated')
        self.assertEqual(q[0,0],0.)
        self.assertTrue(np.isfinite(q).all())
        self.assertLessEqual(record['tv'],.25)


if __name__=='__main__':unittest.main()
