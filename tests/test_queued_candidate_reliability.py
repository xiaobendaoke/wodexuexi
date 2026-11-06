"""Pure numerical tests for paired candidate reliability; no Torch/physics runs."""
import unittest
import numpy as np
from scripts.analysis.queued_candidate_reliability import paired_statistics, error_components, crossfit_selection


class CandidateReliabilityTests(unittest.TestCase):
    def test_common_future_noise_cancels_in_paired_difference(self):
        noise=np.array([-100.,50.,80.,-30.])
        y=np.array([1.,3.,2.,4.])[:,None]+noise
        d=paired_statistics(y,np.array([1.,3.,2.,4.]))
        self.assertAlmostEqual(max(x['se'] for x in d['pairs']),0.)
        self.assertGreater(min(x['unpaired_se'] for x in d['pairs']),20.)
        self.assertAlmostEqual(d['prediction_gain'],1.5)

    def test_replica_constant_shift_preserves_all_contrasts(self):
        y=np.random.default_rng(9).normal(size=(4,32))
        p=np.array([1.,2.,0.,3.])
        a=paired_statistics(y,p)
        b=paired_statistics(y+np.arange(32)[None,:]*100,p+99)
        np.testing.assert_allclose([x['mean'] for x in a['pairs']],[x['mean'] for x in b['pairs']],atol=1e-12)
        np.testing.assert_allclose([x['se'] for x in a['pairs']],[x['se'] for x in b['pairs']],atol=1e-12)
        self.assertAlmostEqual(a['prediction_gain'],b['prediction_gain'])

    def test_decomposition_is_exact_before_and_after_centering(self):
        rng=np.random.default_rng(4)
        values=[rng.normal(size=(4,8)) for _ in range(3)]
        pred=[rng.normal(size=4) for _ in range(2)]
        d=error_components(*values,*pred)
        np.testing.assert_allclose(sum(d['components']),d['error'],atol=1e-12)
        c=lambda z:z-z.mean(axis=0,keepdims=True)
        np.testing.assert_allclose(sum(c(x) for x in d['components']),c(d['error']),atol=1e-12)

    def test_crossfit_evaluates_on_disjoint_replicas(self):
        y=np.array([[1.,-1.],[-1.,1.]])
        d=crossfit_selection(y,1,seed=5,draws=8)
        self.assertAlmostEqual(d['heldout_selection_gain_mean'],-1.)
        self.assertAlmostEqual(d['half_choice_agreement'],0.)

    def test_crossfit_preserves_exact_ties(self):
        y=np.ones((4,8))
        d=paired_statistics(y,np.arange(4.))
        self.assertIsNone(d['pair_concordance'])
        self.assertAlmostEqual(d['prediction_gain'],0.)
        self.assertAlmostEqual(crossfit_selection(y,4)['heldout_selection_gain_mean'],0.)

    def test_rejects_unpaired_or_nonfinite_inputs(self):
        with self.assertRaises(ValueError):paired_statistics(np.ones((4,1)),np.ones(4))
        with self.assertRaises(ValueError):paired_statistics(np.full((4,4),np.nan),np.ones(4))
        with self.assertRaises(ValueError):crossfit_selection(np.ones((4,4)),3)


if __name__=='__main__':unittest.main()
