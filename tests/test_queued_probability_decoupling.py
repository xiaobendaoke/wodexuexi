"""Value units, masks, calibration, and detached value versus policy gradients."""
import copy
import os
import unittest


class MappingTest(unittest.TestCase):
    def test_units_and_offset_do_not_change_normalized_policy(self):
        from utils.queued_probability_scale import probabilities
        q=[-2.,1.,5.,999.];mask=[True,True,True,False]
        a=probabilities(q,mask,.7,True);b=probabilities([7*x+23 for x in q],mask,.7,True)
        self.assertLess(max(abs(x-y) for x,y in zip(a,b)),1e-12);self.assertEqual(a[-1],0.)
        self.assertEqual(max(range(4),key=a.__getitem__),2)

    def test_flat_scores_are_uniform_and_calibration_hits_target(self):
        from utils.queued_probability_scale import probabilities,calibrate,entropy
        self.assertEqual(probabilities([3.,3.,0.],[True,True,False],1.,True),[.5,.5,0.])
        rows=[dict(q=[-20.,0.,20.,0.],mask=[True,True,True,False]) for _ in range(12)]
        result=calibrate(rows,.5);self.assertTrue(result['calibrated'])
        self.assertAlmostEqual(entropy(probabilities(rows[0]['q'],rows[0]['mask'],result['temperature'],True),3),.5,places=6)


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS')=='1','server-only Torch')
class ModelTest(unittest.TestCase):
    def test_zero_residual_mapping_and_teacher_stays_frozen(self):
        import torch
        from tests.test_queued_long_unfreeze import UnfreezeTest
        from marl_models.queued_probability_decoupling import DecoupledMotion
        from marl_models.queued_motion_trust import update_motion,MotionBatch
        fixture=UnfreezeTest();m=fixture.setup_model();raw=copy.deepcopy(m)
        m.motion=DecoupledMotion(m.motion,False,.5);rows=fixture.rows(raw)
        p=MotionBatch(m,rows).probabilities();old=MotionBatch(raw,rows).probabilities()
        self.assertTrue(torch.equal(p,old))
        teacher=copy.deepcopy(m.motion.value_scorer.state_dict());net=copy.deepcopy(m.motion.policy_net.state_dict())
        d=update_motion(m,rows,actor_epochs=2,critic_epochs=1)
        self.assertLessEqual(d['final_full_kl'],.020001)
        self.assertTrue(all(torch.equal(v,m.motion.value_scorer.state_dict()[k]) for k,v in teacher.items()))
        self.assertTrue(all(p.grad is None for p in m.motion.value_scorer.parameters()))
        self.assertTrue(any(not torch.equal(v,m.motion.policy_net.state_dict()[k]) for k,v in net.items()))

    def test_torch_mapping_agrees_with_pure_calibration(self):
        import torch
        from marl_models.queued_probability_decoupling import normalized_scores
        from utils.queued_probability_scale import probabilities
        q=torch.tensor([[-2.,1.,5.,999.]]);mask=torch.tensor([[True,True,True,False]])
        logits=normalized_scores(q,mask)/.7
        p=torch.softmax(logits.masked_fill(~mask,-1e9),-1)[0].tolist()
        self.assertLess(max(abs(x-y) for x,y in zip(p,probabilities(q[0].tolist(),mask[0].tolist(),.7,True))),1e-6)


if __name__=='__main__':unittest.main()
