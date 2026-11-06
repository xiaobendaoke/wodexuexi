"""Pure cohort accounting and soft-label calibration contracts."""
from types import SimpleNamespace
import unittest
import numpy as np
from utils.queued_taskset_closed_loop import (
    design, cohort_statistics, calibration_record, calibration_summary, summary_extra, paired_ratios,
)
from utils.queued_learning_validation import projection


class ClosedLoopTasksetTests(unittest.TestCase):
    def test_policy_specific_denominators_in_quality_comparison(self):
        g=np.full((3,2,4,2),100.)
        z=paired_ratios(g*.4,g*.5,g*.3,g*.6,draws=40)
        self.assertAlmostEqual(z['mean'],.3)
        np.testing.assert_allclose(z['ci95'],[.3,.3])

    def test_calibration_separately_reports_informative_banks(self):
        a=calibration_record(np.full((1,2),.5),np.full((1,2),.5),np.ones((1,2)))
        b=calibration_record(np.array([[.9,.1]]),np.array([[0.,1.]]),np.array([[0.,1.]]))
        z=calibration_summary([a,b])
        self.assertEqual(z['informative']['states'],1)
        self.assertAlmostEqual(z['informative']['ece'],.9)
        self.assertEqual(z['all_scores_tied_fraction'],.5)

    def test_design_budget_and_formal_separation(self):
        s=design()
        self.assertEqual(len(s['arms']),10);self.assertEqual(s['episodes'],1056)
        self.assertEqual(s['roots'],792);self.assertEqual(len(s['workloads']),8)
        self.assertFalse(set(s['workloads']) & set(range(930000,930040)))
        self.assertTrue(set(s['diagnostic_workloads']) <= set(s['workloads']))

    def test_soft_label_calibration_exact_distribution(self):
        reference=np.array([[.1,.2],[.3,.4]])
        z=calibration_record(reference,reference,np.array([[0.,1.],[2.,3.]]))
        self.assertAlmostEqual(z['brier'],0.)
        self.assertAlmostEqual(z['kl'],0.)
        self.assertAlmostEqual(calibration_summary([z])['ece'],0.)

    def test_tie_aware_reference_does_not_confuse_bestset_with_class_accuracy(self):
        p=np.full((2,2),.25);q=np.ones((2,2))
        z=calibration_record(p,p,q)
        self.assertEqual(z['bestset_probability'],1.)
        self.assertEqual(z['top_label_probability'],.25)
        self.assertAlmostEqual(calibration_summary([z])['ece'],0.)
        self.assertEqual(z['score_regret'],0.)

    def test_confident_wrong_action_has_large_calibration_error(self):
        p=np.array([[.9,.1]]);reference=np.array([[0.,1.]])
        z=calibration_record(p,reference,np.array([[0.,3.]]))
        self.assertAlmostEqual(z['confidence'],.9)
        self.assertAlmostEqual(calibration_summary([z])['ece'],.9)
        self.assertGreater(z['score_regret'],2.)

    def test_arrival_cohorts_settle_same_requests(self):
        def task(arrival,admitted,stage):
            return SimpleNamespace(arrival=arrival,admitted=admitted,source=None if admitted is None else 0,
                stage=stage,finished=arrival+2,expired_stage=None if stage=='completed' else 'waiting' if admitted is None else 'compute',
                time_in={'compute_wait':.5})
        tasks=[task(49.,50.,'completed'),task(50.,50.,'completed'),task(60.,61.,'expired'),task(70.,None,'expired'),task(150.,150.,'completed')]
        z=cohort_statistics(tasks)
        self.assertEqual(z['arrival']['peak']['generated'],3)
        self.assertEqual(z['arrival']['peak']['admitted'],2)
        self.assertEqual(z['arrival']['peak']['success'],1)
        self.assertAlmostEqual(z['arrival']['peak']['dsr'],z['arrival']['peak']['admission']*z['arrival']['peak']['conditional'])
        self.assertEqual(z['admission']['peak']['admitted'],3)
        self.assertEqual(z['arrival']['pre']['success'],1)

    def test_summary_control_projection_matches_existing_encoding(self):
        r=dict(a=np.arange(6).reshape(2,3)/10,m=np.arange(12).reshape(2,3,2)/10,context=np.arange(4)/10)
        raw=np.array([[np.r_[r['a'][i],r['m'][i,j],r['context']] for j in range(3)] for i in range(2)])
        np.testing.assert_array_equal(summary_extra(r),projection(raw))


if __name__=='__main__':unittest.main()
