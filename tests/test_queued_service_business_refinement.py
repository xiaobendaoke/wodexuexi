"""Checks for the bounded recovery revision; previous failures remain intact."""
import unittest
from utils.queued_service_business_refinement import RecoveryScenario


class RecoveryWorkload(unittest.TestCase):
    def test_only_the_peak_changes(self):
        a=RecoveryScenario(12,'burst','normal',.45)
        b=RecoveryScenario(12,'burst','normal',.6)
        self.assertEqual(a.rates[:50],(15.,)*50)
        self.assertEqual(a.rates[50:100],(45.,)*50)
        self.assertEqual(b.rates[50:100],(60.,)*50)
        for t in (0,49,100,150):self.assertEqual(a.slot(t),b.slot(t))
        for t in (50,75,99):
            high={x.uid:x for x in b.slot(t)}
            for x in a.slot(t):self.assertEqual(x,high[x.uid])

    def test_common_profiles_keep_their_rates(self):
        for profile,rate in (('light',7.5),('steady',30.),('overload',90.)):
            s=RecoveryScenario(12,profile,'normal',.75)
            self.assertEqual(s.rates,(rate,)*200)

    def test_data_units_and_unique_ids(self):
        s=RecoveryScenario(13,'burst','normal',.45);xs=[x for row in s.requests for x in row]
        self.assertEqual(len(xs),len({x.uid for x in xs}))
        self.assertTrue(all(500<=x.cycles/x.bits<=1000 and 2<=x.deadline-x.arrival<=3 for x in xs))

    def test_explicit_business_boundary(self):
        for args in ((12,'burst','tight',.45),(12,'burst','normal',.8)):
            with self.assertRaises(ValueError):RecoveryScenario(*args)

    def test_matrix_uses_new_workloads_and_no_common_profile_duplicates(self):
        from scripts.analysis.queued_service_business_refinement import specs
        scan=specs('screen');hold=specs('holdout',45)
        from scripts.analysis.queued_service_business import key
        self.assertEqual(len(scan),432);self.assertEqual(len({key(x) for x in scan}),432)
        self.assertEqual({x['workload'] for x in scan},set(range(12,36)))
        self.assertEqual(len(hold),144);self.assertEqual({x['workload'] for x in hold},set(range(36,48)))
        self.assertEqual(len(specs('precision',45)),48)


if __name__=='__main__':unittest.main()
