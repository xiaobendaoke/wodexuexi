import copy
import unittest
from types import SimpleNamespace
import numpy as np
from utils.queued_arrival_adaptation import RateHistory,rate,rate_features,future_arrivals,queue_features,fit_tail,tail_prediction,count_losses
from utils.queued_joint_mpo import arrival_prediction


class ArrivalAdaptationTests(unittest.TestCase):
    def test_causal_update_and_fast_drop(self):
        rates=RateHistory()
        for now in range(20):rates.update_count(now,60)
        rates.update_count(20,15)
        self.assertLess(rates.fast2,rates.fast5)
        old=copy.deepcopy(rates)
        with self.assertRaises(ValueError):rates.update_count(20,15)
        self.assertEqual(rates.counts,old.counts)

    def test_only_current_public_arrivals_count(self):
        rates=RateHistory();rates.update(dict(time=10,tasks=[dict(arrival=10),dict(arrival=9)],future=100))
        self.assertEqual(rates.counts,[1.])

    def test_slow_forecast_exact_compatibility(self):
        obs=dict(time=60,ues={0:(0.,0.),1:(1.,1.)});history=SimpleNamespace(weights={0:100.,1:200.});rates=RateHistory();rates.update_count(60,12)
        a=future_arrivals(obs,history,rates,'slow',3);b=arrival_prediction(obs,history,3,3)
        self.assertEqual(a,b)
        features=rate_features(obs,history,rates);self.assertEqual(features.shape,(8,))

    def test_future_cutoff_and_no_true_future_input(self):
        obs=dict(time=199,ues={0:(0.,0.)},future='unread');history=SimpleNamespace(weights={0:1000.});rates=RateHistory();rates.update_count(199,20)
        self.assertEqual(future_arrivals(obs,history,rates,'fast2',3),[[],[]])

    def test_queue_control_removes_explicit_geometry(self):
        x=np.ones((2,105));out=queue_features(x)
        for node in range(6):np.testing.assert_array_equal(out[:,2+node*12:4+node*12],0.)
        np.testing.assert_array_equal(out[:,82:97],0.)
        np.testing.assert_array_equal(out[:,97:],1.)

    def test_tail_train_only_and_centering(self):
        rng=np.random.default_rng(7);rows=[]
        for index in range(8):
            x=rng.normal(size=(4,105));rows.append(dict(x=x,y=x[:,5]*2,time=100,split='training'))
        fitted=fit_tail(rows);pred=tail_prediction(fitted,'full',rows[0]['x'])
        self.assertAlmostEqual(pred.mean(),0.)
        shifted=[dict(r,y=r['y']+50) for r in rows];other=fit_tail(shifted)
        np.testing.assert_allclose(pred,tail_prediction(other,'full',rows[0]['x']),atol=1e-10)
        with self.assertRaises(ValueError):fit_tail([dict(rows[0],split='holdout')])

    def test_count_loss_zero(self):
        self.assertEqual(count_losses(15,15)['poisson_deviance'],0.)


if __name__=='__main__':unittest.main()
