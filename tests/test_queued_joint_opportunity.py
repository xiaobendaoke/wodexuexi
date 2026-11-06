"""Conditional arrivals, safe candidate nesting and held-out opportunity checks."""
import copy
import unittest


class OpportunityTest(unittest.TestCase):
    def test_bank_nests_original_and_never_mutates_observation(self):
        from tests.test_queued_service_learning import observation
        from utils.queued_service_learning import LearningAdapter
        from utils.queued_joint_opportunity import expanded_bank,motion_plan
        from environment.queued_service import Engine,safe_path
        obs=observation();before=copy.deepcopy(obs);a=LearningAdapter();a.prepare(obs)
        bank=expanded_bank(obs,a.plans)
        self.assertEqual(bank[:len(a.plans)],a.plans);self.assertLessEqual(len(bank),16)
        for move in bank:
            self.assertTrue(safe_path(obs['positions'],move,obs['protocol']))
            e=Engine.from_observation(obs);e.assign(motion_plan(obs,move))
        self.assertEqual(obs,before)

    def test_suffix_changes_arrivals_but_fixes_geometry_and_business(self):
        from utils.queued_joint_opportunity import geometry,conditional_suffix
        from utils.queued_service_business_refinement import RecoveryScenario
        geo=geometry(7200);ues=dict(enumerate(geo.ue_positions))
        self.assertEqual(tuple(ues.values()),RecoveryScenario(7200,'burst','normal',.6).ue_positions)
        a=conditional_suffix(7200,ues,60,0);b=conditional_suffix(7200,ues,60,0);c=conditional_suffix(7200,ues,60,1)
        self.assertEqual(a,b);self.assertNotEqual(a,c)
        for offset,row in enumerate(a,1):
            for t in row:
                self.assertEqual(t.arrival,60+offset);self.assertTrue(2<=t.deadline-t.arrival<=3)
                self.assertTrue(12e6<=t.bits<=24e6);self.assertTrue(500<=t.cycles/t.bits<=1000)
                self.assertIn(t.ue,ues)

    def test_crossfit_evaluates_selection_on_other_futures(self):
        from utils.queued_joint_opportunity import crossfit
        stable=[[[3,0],[0,3]],[[3,0],[0,3]]]
        d=crossfit(stable,1)
        self.assertEqual(d['motion_conditioning'],1.5)
        noise=[[[4,0],[0,0]],[[0,4],[0,0]]]
        d=crossfit(noise,1)
        self.assertLessEqual(d['capacity_gain'],0)
