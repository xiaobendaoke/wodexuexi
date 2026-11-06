import unittest
import numpy as np
from scripts.analysis.review_queued_candidate_reliability import signal_moments,assert_nested_close

class SignalMomentTests(unittest.TestCase):
    def test_exact_signal_ignores_shared_replica_noise(self):
        y=np.array([1.,3.,2.,4.])[:,None]+np.arange(8)[None,:]*10
        d=signal_moments(y)
        self.assertAlmostEqual(d['estimated_mean_noise'],0.)
        self.assertAlmostEqual(d['unbiased_signal_second_moment'],1.25)
    def test_negative_estimate_is_preserved(self):
        d=signal_moments(np.array([[1.,-1.],[-1.,1.]]))
        self.assertLess(d['unbiased_signal_second_moment'],0.)
    def test_cross_platform_numeric_tolerance_is_bounded(self):
        assert_nested_close({'x':[1.+1e-11]}, {'x':[1.]})
        with self.assertRaises(AssertionError):assert_nested_close({'x':[1.01]}, {'x':[1.]})
    def test_invalid_samples_fail(self):
        with self.assertRaises(ValueError):signal_moments(np.ones((2,1)))

if __name__=='__main__':unittest.main()
