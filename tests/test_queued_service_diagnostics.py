"""Deterministic checks for frozen-policy diagnostics, without local Torch."""
import unittest
from environment.queued_service import Engine, Protocol
from scripts.analysis.queued_service_validation import START
from utils.queued_service_diagnostics import commit_sequence, probability_metrics, payoff_metrics, mc_returns, explained_variance


class Diagnostics(unittest.TestCase):
    def test_hover_remains_hover_in_committed_sequence(self):
        e=Engine(Protocol(),START,[(100.,100.)]);zero=[(0.,0.)]*5
        self.assertEqual(commit_sequence(e,zero),[zero]*6)

    def test_sequence_remains_safe(self):
        from environment.queued_service import safe_path
        e=Engine(Protocol(),START,[(100.,100.)]);first=[(15.,0.)]+[(0.,0.)]*4
        positions=e.positions
        for move in commit_sequence(e,first):
            self.assertTrue(safe_path(positions,move,e.p))
            positions=[(a[0]+d[0],a[1]+d[1]) for a,d in zip(positions,move)]

    def test_probability_tv_and_single_action(self):
        d=probability_metrics([.25,.75],[.75,.25])
        self.assertAlmostEqual(d['tv'],.5);self.assertTrue(d['argmax_changed'])
        self.assertEqual(probability_metrics([1.],[1.])['entropy_fraction'],1.)

    def test_payoff_reversal_and_zero_matrix(self):
        r=payoff_metrics([1,2],[3,1])
        self.assertTrue(r['strict_reversal']);self.assertEqual(r['interaction_range'],3)
        self.assertEqual(payoff_metrics([1,1],[2,2])['interaction_range'],0)
        self.assertFalse(payoff_metrics([1],[2])['strict_reversal'])

    def test_mc_includes_future_but_stops_at_terminal(self):
        self.assertEqual(mc_returns([0.,0.,1.]),[1.,1.,1.])
        self.assertEqual(mc_returns([2.]),[2.])

    def test_explained_variance_preserves_constant_target_boundary(self):
        self.assertEqual(explained_variance([1.,2.,3.],[1.,2.,3.]),1.)
        self.assertEqual(explained_variance([1.,2.,3.],[0.,0.,0.]),0.)
        self.assertIsNone(explained_variance([1.,1.],[0.,0.]))


if __name__=='__main__':unittest.main()
