import unittest
from types import SimpleNamespace
import numpy as np
from utils.queued_recovery_audit import action_factors, telemetry, remainder_cohorts, factorial_effects, gain_layers


class RecoveryAuditTests(unittest.TestCase):
    def test_first_draw_matches_original_sampling(self):
        a=np.array([[.1,.2],[.3,.4]]);b=a[::-1,::-1].copy();seed=91
        draws,unique=action_factors(a,b,[seed,92,93,94])
        expected=tuple(np.unravel_index(np.random.default_rng(seed).choice(a.size,p=a.ravel()),a.shape))
        self.assertEqual(draws[0]['anchor'],expected)
        for draw in draws:
            self.assertEqual(draw['allocation'],(draw['full'][0],draw['anchor'][1]))
            self.assertEqual(draw['motion'],(draw['anchor'][0],draw['full'][1]))
        self.assertEqual(unique,sorted(set(pair for draw in draws for pair in draw.values())))

    def test_identical_distributions_preserve_ties_without_new_repeats(self):
        a=np.array([[1.,0.],[0.,0.]])
        draws,unique=action_factors(a,a,[1,2,3,4])
        self.assertEqual(unique,[(0,0)])
        self.assertTrue(all(len(set(draw.values()))==1 for draw in draws))

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):action_factors(np.array([[1.]]),np.array([[np.nan]]),[1])

    def test_coverage_and_queue_snapshot(self):
        task=SimpleNamespace(stage='compute',target=0,remaining_cycles=2.)
        engine=SimpleNamespace(positions=[(0.,0.)],ues={0:(99.,0.),1:(101.,0.)},p=SimpleNamespace(coverage=100.,capacity=(2.,)),tasks={1:task},t=3.)
        view=telemetry(engine)
        self.assertEqual(view['covered_ues'],[0]);self.assertEqual(view['node_work_seconds'],[1.])

    def test_root_and_future_denominators(self):
        def task(uid,arrival,source,stage):return SimpleNamespace(uid=uid,arrival=arrival,source=source,stage=stage)
        result=remainder_cohorts([task(1,99.,0,'completed'),task(2,101.,None,'expired'),task(3,103.,1,'completed')],{1},100)
        self.assertEqual(result['root']['generated'],1)
        self.assertEqual(result['future']['generated'],2)
        self.assertEqual(result['future_late']['generated'],1)
        for counts in result.values():self.assertEqual(counts['generated'],counts['success']+counts['expired'])

    def test_exact_factor_and_error_identities(self):
        effects=factorial_effects(dict(anchor=4,allocation=6,motion=7,full=12))
        self.assertEqual(effects['total'],effects['allocation']+effects['motion']+effects['interaction'])
        layers=gain_layers(.4,.1,-.1,-.2)
        self.assertAlmostEqual(layers['predicted']-layers['realized'],sum(layers[k] for k in ('approximation_error','prediction_error','visibility_error')))


if __name__=='__main__':unittest.main()
