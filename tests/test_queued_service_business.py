"""Business arrival/units/paired-SLA checks; standard library only."""
import statistics
import unittest

from utils.queued_service_business import BusinessScenario, poisson, BASIS, SLA, seed_for


class BusinessWorkload(unittest.TestCase):
    def test_cycle_unit_and_size_bounds(self):
        tape = BusinessScenario(0, 'burst', 'normal', 1.)
        tasks = [x for t in range(tape.horizon) for x in tape.slot(t)]
        self.assertTrue(all(1.5e6 <= x.bits/8 <= 3e6 for x in tasks))
        self.assertTrue(all(500 <= x.cycles/x.bits <= 1000 for x in tasks))
        self.assertTrue(all(2 <= x.deadline-x.arrival <= 3 for x in tasks))
        self.assertEqual(len(tasks), len({x.uid for x in tasks}))

    def test_same_requests_and_marks_across_slas(self):
        a = BusinessScenario(0, 'burst', 'tight', 1.)
        b = BusinessScenario(0, 'burst', 'normal', 1.)
        c = BusinessScenario(0, 'burst', 'relaxed', 1.)
        for t in (0, 50, 99, 150):
            rows = [x.slot(t) for x in (a, b, c)]
            self.assertEqual([x.uid for x in rows[0]], [x.uid for x in rows[1]])
            for x, y, z in zip(*rows):
                self.assertEqual((x.ue, x.bits, x.cycles), (y.ue, y.bits, y.cycles))
                self.assertAlmostEqual(x.deadline-t-1, y.deadline-t-2)
                self.assertAlmostEqual((z.deadline-t-3)/2, y.deadline-t-2)
        self.assertEqual(a.arrival_fingerprint, b.arrival_fingerprint)
        self.assertNotEqual(a.fingerprint, b.fingerprint)

    def test_rate_thinning_preserves_potential_task_attributes(self):
        a = BusinessScenario(1, 'steady', 'normal', .75)
        b = BusinessScenario(1, 'steady', 'normal', 1.25)
        for t in (0, 25, 99):
            high = {x.uid: x for x in b.slot(t)}
            for x in a.slot(t): self.assertEqual(x, high[x.uid])

    def test_same_ue_can_generate_multiple_and_hotspot_not_capped(self):
        tape = BusinessScenario(0, 'overload', 'normal', 1.25)
        rows = [tape.slot(t) for t in range(10)]
        self.assertTrue(any(len(r) > 100 for r in rows))
        self.assertTrue(any(len({x.ue for x in r}) < len(r) for r in rows))
        self.assertGreater(tape.statistics['duplicate_ue_slots'], 0)

    def test_poisson_sampler_mean_variance_and_validation(self):
        import random
        rng = random.Random(7); xs = [poisson(40, rng) for _ in range(10000)]
        self.assertAlmostEqual(statistics.mean(xs), 40, delta=.35)
        self.assertAlmostEqual(statistics.pvariance(xs), 40, delta=2.)
        self.assertEqual(poisson(0, rng), 0)
        with self.assertRaises(ValueError): poisson(-1, rng)
        with self.assertRaises(ValueError): poisson(float('nan'), rng)

    def test_batch_replay_does_not_mutate_the_tape(self):
        from environment.queued_service import Engine, Protocol
        from scripts.analysis.queued_service_validation import START
        tape = BusinessScenario(2, 'light', 'normal', 1.)
        before = [(x.uid, x.stage, x.source) for x in tape.slot(0)]
        e = Engine(Protocol(), START, tape.ue_positions); e.add(tape.slot(0)); new = e.admit()
        e.assign({u: e.tasks[u].source for u in new}); e.advance([(0., 0.)]*5); e.audit()
        self.assertEqual(before, [(x.uid, x.stage, x.source) for x in tape.slot(0)])

    def test_explicit_validation_and_new_namespace(self):
        self.assertEqual(BASIS['mean_task_cycles'], 13.5e9)
        self.assertNotEqual(seed_for('workload', 0), seed_for('workload', 6))
        for args in ((0, 'missing', 'normal', 1.), (0, 'burst', 'missing', 1.), (0, 'burst', 'normal', 0)):
            with self.assertRaises(ValueError): BusinessScenario(*args)

    def test_selection_does_not_choose_the_highest_dsr_setting(self):
        import tempfile
        from pathlib import Path
        from scripts.analysis.queued_service_business import select
        from utils.queued_service_controllers import METHODS
        rows = [dict(profile='burst', sla=sla, scale=scale, method=m, gate=True,
                     criteria={'physical_test': True}, dsr=.1 if sla=='normal' and scale==1. else .75)
                for sla in SLA for scale in (.75,1.,1.25) for m in METHODS]
        with tempfile.TemporaryDirectory() as directory:
            chosen = select(dict(summary=rows), Path(directory))
            self.assertEqual((chosen['sla'],chosen['scale']), ('normal',1.))
            self.assertEqual(chosen, select(dict(summary=rows), Path(directory)))

    def test_frozen_matrix_has_complete_balanced_cells(self):
        from scripts.analysis.queued_service_business import specs, key
        scan = specs('screen')
        self.assertEqual(len(scan),648); self.assertEqual(len({key(s) for s in scan}),648)
        self.assertEqual(len(specs('holdout','normal',1.)),72)
        self.assertEqual(len(specs('precision','normal',1.)),48)


if __name__ == '__main__': unittest.main()
