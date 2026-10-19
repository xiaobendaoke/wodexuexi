import unittest


def synthetic():
    from scripts.analysis.sf_hlm_framework_evaluation import METHODS
    from scripts.analysis.sf_hlm_framework_confirmation import design
    d=design();blocks=list(range(20));tasks=[[m,b,123,0,'measured'] for m in METHODS for b in blocks];p=dict(methods=list(METHODS),blocks=blocks,primary='mixture_price',primary_comparisons=d['primary_comparisons'],gates=d['gates'],tasks=tasks,steps=1,workloads=[123],episodes_per_workload=1,runner_smoke=False);episodes=[]
    for method,block,w,e,mode in tasks:
        count=50+(block%3) if method=='mixture_price' else 40 if method=='G' else 45
        totals=dict(generated=100,admitted=80,success=count,potential=100,fleet_energy=100.,mbs=10.,fairness=.5,raw_success=count+1,shadow_fee_success=count)
        episodes.append(dict(task=[method,block,w,e,mode],totals=totals,seconds=[.01],exogenous_hash='external_'+str(block)))
    return p,episodes


class FrameworkStatistics(unittest.TestCase):
    def test_exact_signflip_and_constant_cases(self):
        from scripts.analysis.sf_hlm_framework_statistics import effect
        self.assertEqual(effect([1.]*4,exact=True)['exact_signflip_p'],.125)
        self.assertEqual(effect([0.]*4,exact=True)['exact_signflip_p'],1.)
        self.assertEqual(effect([0.]*4)['t_p_two_sided'],1.)
        self.assertIsNone(effect([1.]*4)['t_statistic'])

    def test_holm_known_family_and_no_reordering_loss(self):
        from scripts.analysis.sf_hlm_framework_statistics import holm
        self.assertEqual(holm({'a':.01,'b':.04}),{'a':.02,'b':.04})
        self.assertEqual(holm({'b':.04,'a':.03}),{'a':.06,'b':.06})

    def test_complete_registered_population_and_positive_gate(self):
        from scripts.analysis.sf_hlm_framework_statistics import summary
        p,rows=synthetic();s=summary(p,rows);self.assertTrue(s['decision']['framework_confirmation_signal']);self.assertEqual(s['primary_family']['G']['n'],20);self.assertFalse(s['decision']['unique_learning_increment_proven'])
        with self.assertRaises(ValueError):summary(p,rows[:-1])
        with self.assertRaises(ValueError):summary(p,rows+[rows[0]])

    def test_binary_failure_cannot_be_hidden_by_gain_over_g(self):
        from scripts.analysis.sf_hlm_framework_statistics import summary
        p,rows=synthetic()
        for row in rows:
            if row['task'][0]=='binary_price':row['totals'].update(success=60,raw_success=61,shadow_fee_success=60)
        s=summary(p,rows);self.assertFalse(s['decision']['framework_confirmation_signal']);self.assertFalse(s['decision']['mean_thresholds_pass'])

    def test_time_and_energy_are_independent_hard_conditions(self):
        from scripts.analysis.sf_hlm_framework_statistics import summary
        p,rows=synthetic()
        for row in rows:
            if row['task'][0]=='mixture_price':row['totals']['fleet_energy']=106.
        self.assertFalse(summary(p,rows)['decision']['energy_mean_ratio_pass'])
        for row in rows:
            if row['task'][0]=='mixture_price':row['totals']['fleet_energy']=100.;row['seconds']=[1.]
        self.assertFalse(summary(p,rows)['decision']['all_block_p95_pass'])

    def test_positive_block_requirement_is_not_replaced_by_large_mean(self):
        from scripts.analysis.sf_hlm_framework_statistics import summary
        p,rows=synthetic()
        for row in rows:
            if row['task'][0]=='G':
                count=0 if row['task'][1]<15 else 75;row['totals'].update(success=count,raw_success=count+1,shadow_fee_success=count)
        s=summary(p,rows);self.assertTrue(s['decision']['mean_thresholds_pass']);self.assertFalse(s['decision']['positive_block_thresholds_pass']);self.assertFalse(s['decision']['framework_confirmation_signal'])

    def test_p95_is_computed_from_calls_and_counts_are_pooled(self):
        from scripts.analysis.sf_hlm_framework_statistics import summary
        p,rows=synthetic();p['runner_smoke']=True;p['blocks']=[0];p['tasks']=[t for t in p['tasks'] if t[1]==0];rows=[r for r in rows if r['task'][1]==0];p['steps']=20
        for row in rows:row['seconds']=[.01]*19+[.5];row['totals']['fairness']=10.
        s=summary(p,rows);self.assertFalse(s['decision']['confirmation_gate_evaluated']);self.assertAlmostEqual(s['blocks']['0']['G']['p95_seconds'],.0345)

    def test_request_counts_are_pooled_before_block_rates(self):
        from scripts.analysis.sf_hlm_framework_statistics import summary
        p,_=synthetic();p.update(runner_smoke=True,blocks=[0],episodes_per_workload=2);p['tasks']=[[m,0,123,e,'measured'] for m in p['methods'] for e in range(2)];rows=[]
        for task in p['tasks']:
            e=task[3];generated=10 if e==0 else 90;admitted=10 if e==0 else 80;success=9;z=dict(generated=generated,admitted=admitted,success=success,potential=generated,fleet_energy=100.,mbs=0.,fairness=.5,raw_success=success,shadow_fee_success=success)
            rows.append(dict(task=task,totals=z,seconds=[.01],exogenous_hash='episode_'+str(e)))
        s=summary(p,rows);self.assertAlmostEqual(s['blocks']['0']['G']['dsr'],.18);self.assertAlmostEqual(s['blocks']['0']['G']['admission_rate']*s['blocks']['0']['G']['accepted_quality'],.18)


if __name__=='__main__':unittest.main()
