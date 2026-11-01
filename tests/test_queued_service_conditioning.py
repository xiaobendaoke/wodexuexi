"""Opportunity-value and phase-conditioned value checks; Torch only on server."""
import os
import unittest
from utils.queued_service_conditioning import conditioning_value,value_fit_by_phase


class ConditioningStatistics(unittest.TestCase):
    def test_opposite_best_actions_have_positive_information_value(self):
        result=conditioning_value([[10,9],[9,10]])
        self.assertEqual(result['conditional_best'],10)
        self.assertEqual(result['common_best'],9.5)
        self.assertEqual(result['information_gain'],.5)

    def test_interaction_can_exist_with_zero_information_gain(self):
        result=conditioning_value([[10,9],[8,5]])
        self.assertGreater(result['interaction_variance'],0)
        self.assertEqual(result['information_gain'],0)
        self.assertTrue(result['common_optimum_exists'])

    def test_variance_decomposition_is_exact(self):
        r=conditioning_value([[5,8,4],[9,5,2],[3,1,7]])
        self.assertAlmostEqual(r['total_variance'],r['routing_variance']+r['motion_variance']+r['interaction_variance'])

    def test_duplicate_sampled_plans_keep_sampling_weight(self):
        r=conditioning_value([[10,9],[10,9],[9,10]])
        self.assertAlmostEqual(r['information_gain'],1/3)

    def test_one_candidate_has_no_conditioning_opportunity(self):
        self.assertEqual(conditioning_value([[2],[5]])['information_gain'],0)

    def test_time_only_prediction_can_have_high_global_ev_and_zero_within(self):
        r=value_fit_by_phase([1,3,101,103],[2,2,102,102],['a','a','b','b'])
        self.assertGreater(r['overall_ev'],.99)
        self.assertEqual(r['within_phase_ev'],0)

    def test_invalid_matrix_rejected(self):
        with self.assertRaises(ValueError):conditioning_value([[1,2],[3]])


@unittest.skipUnless(os.environ.get('QUEUED_TORCH_TESTS')=='1','server-only Torch validation')
class ConditioningTorch(unittest.TestCase):
    def test_native_wrapper_replays_and_swap_preserves_current_allocation(self):
        import torch
        from marl_models.queued_service_shared_budget import BudgetPolicy
        from utils.queued_service_shared_budget import BudgetAdapter
        from tests.test_queued_service_learning import observation
        from scripts.analysis.diagnose_queued_service_conditioning import choose
        for method in ('joint','decomposed'):
            torch.manual_seed(17001);model=BudgetPolicy(method);obs=observation()
            expected,plan,motion=model.act(obs,BudgetAdapter(),42)
            row,p,m=choose(model,obs,BudgetAdapter(),42,'native',torch)
            self.assertEqual(p,plan);self.assertEqual(m,motion)
            _,swapped,_=choose(model,obs,BudgetAdapter(),42,'swap',torch)
            self.assertEqual(swapped,plan);self.assertEqual(model.method,method)

    def test_candidate_normalization_removes_positive_affine_differences(self):
        import torch
        from marl_models.queued_service_shared_budget import normalized_candidates
        x=torch.tensor([[[1.,4.],[2.,8.],[4.,12.]]]);mask=torch.ones((1,3),dtype=torch.bool)
        y=x*torch.tensor([[[2.,3.]]])+torch.tensor([[[5.,9.]]])
        self.assertTrue(torch.allclose(normalized_candidates(x,mask),normalized_candidates(y,mask),atol=1e-6))


if __name__=='__main__':unittest.main()
