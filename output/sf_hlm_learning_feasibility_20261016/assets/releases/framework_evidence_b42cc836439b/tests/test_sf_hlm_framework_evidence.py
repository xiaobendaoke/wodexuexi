"""Scientific reporting boundaries on synthetic complete and incomplete data."""
import copy
import unittest


def fixture(*,smoke=False,unfavorable_binary=False):
    from scripts.analysis.sf_hlm_framework_confirmation import design
    from scripts.analysis.sf_hlm_framework_evaluation import runner_design
    from scripts.analysis.sf_hlm_framework_statistics import summary
    p=runner_design(dict(smoke=smoke,**design(smoke)),smoke);episodes=[];seconds=[.01]*p['steps']
    for task in p['tasks']:
        method,block,workload,episode,mode=task;success=501+(block%3) if method=='mixture_price' else 400 if method=='G' else 610 if method=='binary_price' and unfavorable_binary else 450
        totals=dict(generated=1000,admitted=800,success=success,potential=1000,fleet_energy=1000.,mbs=100.,fairness=p['steps']*.5,raw_success=success+1,shadow_fee_success=success)
        episodes.append(dict(task=task,totals=totals,seconds=seconds,exogenous_hash=str([block,workload,episode])))
    return p,summary(p,episodes)


class FrameworkEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.positive=fixture();cls.negative_p,cls.negative=fixture(unfavorable_binary=True);cls.sp,cls.smoke=fixture(smoke=True)

    def test_completed_positive_framework_still_does_not_prove_learning(self):
        from scripts.analysis.sf_hlm_framework_evidence import check_summary,render,METHODS
        result=check_summary(self.p,self.positive);self.assertTrue(result['confirmation_claim_permitted']);text=render(self.p,self.positive,result)
        for method in METHODS:self.assertIn('| '+method+' |',text)
        for value in ('不证明PPO独有增量','不由本报告自动判定完成','不是另一次零费用闭环','每个训练块各15,000次','连续队列、真实网络'):self.assertIn(value,text)

    def test_failed_binary_comparison_retained_and_reported_without_promotion(self):
        from scripts.analysis.sf_hlm_framework_evidence import check_summary,render
        result=check_summary(self.negative_p,self.negative);self.assertFalse(result['confirmation_claim_permitted']);text=render(self.negative_p,self.negative,result)
        self.assertIn('未通过预定框架晋级条件',text);self.assertIn('| binary_price |',text);self.assertGreater(self.negative['primary_family']['G']['mean'],0);self.assertLess(self.negative['primary_family']['binary_price']['mean'],0)

    def test_missing_method_or_block_cannot_emit_confirmation(self):
        from scripts.analysis.sf_hlm_framework_evidence import check_summary
        data=copy.deepcopy(self.positive);data['groups'].pop('mixture')
        with self.assertRaises(ValueError):check_summary(self.p,data)
        data=copy.deepcopy(self.positive);data['blocks'].pop(str(self.p['blocks'][0]))
        with self.assertRaises(ValueError):check_summary(self.p,data)

    def test_mislabeled_scope_and_learning_claim_are_rejected(self):
        from scripts.analysis.sf_hlm_framework_evidence import check_summary
        data=copy.deepcopy(self.positive);data['decision']['unique_learning_increment_proven']=True
        with self.assertRaises(ValueError):check_summary(self.p,data)
        data=copy.deepcopy(self.positive);data['quota_ppo_models_evaluated']=20
        with self.assertRaises(ValueError):check_summary(self.p,data)

    def test_independent_counts_and_holm_checks_detect_corruption(self):
        from scripts.analysis.sf_hlm_framework_evidence import check_summary
        data=copy.deepcopy(self.positive);data['blocks'][str(self.p['blocks'][0])]['mixture_price']['success']+=1
        with self.assertRaises(ValueError):check_summary(self.p,data)
        data=copy.deepcopy(self.positive);data['primary_family']['binary_price']['holm_p']=1.
        with self.assertRaises(ValueError):check_summary(self.p,data)
        data=copy.deepcopy(self.positive);data['decision']['energy_mean_ratio_pass']=False
        with self.assertRaises(ValueError):check_summary(self.p,data)

    def test_smoke_does_not_emit_method_effects_or_inference(self):
        from scripts.analysis.sf_hlm_framework_evidence import check_summary,render
        result=check_summary(self.sp,self.smoke);text=render(self.sp,self.smoke,result);self.assertEqual(result['classification'],'COMPATIBILITY_ONLY');self.assertFalse(result['confirmation_claim_permitted']);self.assertIn('不报告方法性能差异',text);self.assertNotIn('Holm p',text)
        data=copy.deepcopy(self.smoke);data['decision']['confirmation_gate_evaluated']=True
        with self.assertRaises(ValueError):check_summary(self.sp,data)

    def test_wrong_timing_population_cannot_be_recast_as_complete(self):
        from scripts.analysis.sf_hlm_framework_evidence import check_summary
        data=copy.deepcopy(self.positive);data['blocks'][str(self.p['blocks'][0])]['G']['timing_samples']-=1
        with self.assertRaises(ValueError):check_summary(self.p,data)
        protocol=copy.deepcopy(self.p);protocol['steps']=999
        with self.assertRaises(ValueError):check_summary(protocol,self.positive)

    def test_only_registered_evaluator_entries_are_loaded(self):
        from pathlib import Path
        from scripts.analysis.sf_hlm_framework_evidence import entry_path
        self.assertEqual(entry_path(dict(release='/tmp/report_fixture')),Path('/tmp/report_fixture/scripts/analysis/sf_hlm_framework_evaluation.py'))
        self.assertEqual(entry_path(dict(frozen_release='/tmp/report_fixture',entry_script='scripts/analysis/sf_hlm_framework_saturated.py')),Path('/tmp/report_fixture/scripts/analysis/sf_hlm_framework_saturated.py'))
        with self.assertRaises(ValueError):entry_path(dict(release='/tmp/report_fixture',entry_script='../arbitrary.py'))


if __name__=='__main__':unittest.main()
