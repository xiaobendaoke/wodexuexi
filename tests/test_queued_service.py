"""Deterministic, standard-library checks for the cross-slot service engine."""
import unittest

from environment.queued_service import Engine, Protocol, Task, safe_path


def protocol(**kwargs):
    base = dict(capacity=(10., 10., 20.), altitude=0., mbs=(300., 300., 0.),
                separation=0., boundary=(0., 700.), coverage=100., sensing=460.,
                speed=100., dt=.1, uplink_bps=10., inter_bps=10., backhaul_bps=10.)
    base.update(kwargs)
    return Protocol(**base)


def engine(**kwargs):
    return Engine(protocol(**kwargs), [(100., 100.), (300., 100.)], [(100., 100.)], trace=True)


class QueueContract(unittest.TestCase):
    def test_transmission_then_compute_within_slot(self):
        e = engine()
        e.add([Task(1, 0, 0., 3., 5., 5.)]); e.admit(); e.assign({1: 0})
        e.advance([(0., 0.), (0., 0.)])
        self.assertEqual(e.finished[1].stage, 'completed')
        self.assertAlmostEqual(e.finished[1].finished, 1.)
        events = [x['event'] for x in e.events if x.get('uid') == 1]
        self.assertLess(events.index('upload_complete'), events.index('complete'))

    def test_compute_cannot_start_before_complete_upload(self):
        e = engine(); e.add([Task(1, 0, 0., 4., 15., 10.)]); e.admit(); e.assign({1: 0})
        e.advance([(0., 0.), (0., 0.)])
        self.assertEqual(e.tasks[1].remaining_cycles, 10.)
        self.assertAlmostEqual(e.tasks[1].remaining_upload, 5.)
        self.assertGreater(e.tasks[1].remaining_upload, 0.)

    def test_shared_cpu_and_exact_deadline(self):
        e = engine(); e.add([Task(i, 0, 0., 1., 0., 10.) for i in (1, 2)])
        e.admit(); e.assign({1: 0, 2: 0}); e.advance([(0., 0.), (0., 0.)])
        self.assertEqual(e.counts['success'], 1)
        self.assertEqual(e.counts['expired'], 1)
        self.assertAlmostEqual(e.served_cycles[0], 10.)
        self.assertFalse(e.tasks)

    def test_upload_airtime_shared_and_same_ue_edf(self):
        e = Engine(protocol(), [(100., 100.), (300., 100.)], [(100., 100.), (100., 100.)])
        e.add([Task(i, i-1, 0., 3., 10., 0.) for i in (1, 2)]); e.admit(); e.assign({1: 0, 2: 0})
        e.advance([(0., 0.), (0., 0.)])
        self.assertAlmostEqual(sum(t.remaining_upload for t in e.tasks.values()), 10.)
        self.assertAlmostEqual(e.served_bits['upload'], 10.)
        e.audit()

    def test_transfer_before_remote_compute(self):
        e = engine(); e.add([Task(1, 0, 0., 5., 5., 5.)]); e.admit(); e.assign({1: 1})
        e.advance([(0., 0.), (0., 0.)])
        self.assertEqual(e.counts['success'], 0)
        self.assertEqual(e.tasks[1].remaining_cycles, 5.)
        e.advance([(0., 0.), (0., 0.)])
        self.assertAlmostEqual(e.finished[1].finished, 1.5)

    def test_disconnect_pauses_retains_then_resumes(self):
        e = engine(inter_range=250.)
        e.add([Task(1, 0, 0., 5., 10., 0.)]); e.admit(); e.assign({1: 1})
        e.advance([(0., 0.), (0., 0.)])
        e.advance([(0., 0.), (100., 0.)])
        remaining = e.tasks[1].remaining_transfer
        self.assertAlmostEqual(remaining, 5.)
        e.advance([(0., 0.), (0., 0.)])
        self.assertAlmostEqual(e.tasks[1].remaining_transfer, remaining)
        e.advance([(0., 0.), (-100., 0.)]); self.assertEqual(e.counts['success'], 1)

    def test_expiration_removes_all_work_once(self):
        e = engine(); e.add([Task(1, 0, 0., .4, 20., 30.)]); e.admit(); e.assign({1: 1})
        e.advance([(0., 0.), (0., 0.)]); e.advance([(0., 0.), (0., 0.)])
        self.assertEqual(e.counts['expired'], 1); self.assertEqual(e.counts['generated'], 1)
        self.assertFalse(e.tasks); self.assertEqual(e.finished[1].remaining_upload, 0.)
        self.assertEqual(e.finished[1].remaining_transfer, 0.); self.assertEqual(e.finished[1].remaining_cycles, 0.)

    def test_uncovered_waits_and_can_be_admitted_later(self):
        e = Engine(protocol(), [(0., 100.), (300., 100.)], [(150., 100.)])
        e.add([Task(1, 0, 0., 5., 1., 1.)]); self.assertEqual(e.admit(), [])
        e.advance([(100., 0.), (0., 0.)]); self.assertEqual(e.admit(), [1])
        e.assign({1: 0}); e.advance([(0., 0.), (0., 0.)]); self.assertEqual(e.counts['success'], 1)

    def test_unique_ids_and_assignment_immutability(self):
        e = engine(); e.add([Task(1, 0, 0., 3., 10., 1.)]); e.admit(); e.assign({1: 0})
        with self.assertRaises(ValueError): e.add([Task(1, 0, 0., 3., 10., 1.)])
        with self.assertRaises(ValueError): e.assign({1: 1})

    def test_edf_preempts_old_work(self):
        e = engine(); e.add([Task(1, 0, 0., 5., 0., 30.)]); e.admit(); e.assign({1: 0})
        e.advance([(0., 0.), (0., 0.)]); e.add([Task(2, 0, 1., 2., 0., 10.)]); e.admit(); e.assign({2: 0})
        e.advance([(0., 0.), (0., 0.)]); self.assertEqual(e.finished[2].stage, 'completed')
        self.assertAlmostEqual(e.tasks[1].remaining_cycles, 20.)

    def test_safe_endpoints_can_have_unsafe_intermediate_path(self):
        self.assertFalse(safe_path([(100., 100.), (300., 100.)], [(200., 0.), (-200., 0.)],
                                   protocol(speed=300., separation=100.)))

    def test_speed_boundary_and_nonfinite_rejected(self):
        e = engine()
        for d in ([(101., 0.), (0., 0.)], [(-101., 0.), (0., 0.)], [(float('nan'), 0.), (0., 0.)]):
            with self.assertRaises(ValueError): e.advance(d)
        self.assertEqual(e.t, 0.)

    def test_mid_slot_contact_does_not_transfer_outside_range(self):
        e = Engine(protocol(dt=1.), [(0., 100.), (300., 100.)], [(100., 100.)])
        e.add([Task(1, 0, 0., 3., 20., 0.)]); e.admit(); e.assign({1: 0})
        e.advance([(0., 100.), (0., 0.)])
        self.assertAlmostEqual(e.served_bits['upload'], 0., places=7)

    def test_tail_drain_and_waiting_expiration(self):
        e = engine(); e.add([Task(1, 0, 0., 2.5, 25., 10.)]); e.admit(); e.assign({1: 0})
        for _ in range(3): e.advance([(0., 0.), (0., 0.)])
        self.assertEqual(e.counts['expired'], 1); self.assertFalse(e.tasks); e.audit()

    def test_invisible_waiting_tasks_are_not_in_policy_observation(self):
        e = Engine(protocol(sensing=10.), [(100., 100.), (300., 100.)], [(600., 600.)])
        e.add([Task(1, 0, 0., 3., 10., 1.)])
        self.assertFalse(e.observe()['tasks']); self.assertFalse(e.observe()['ues'])

    def test_assignment_cannot_target_out_of_range_collaborator(self):
        e = engine(inter_range=100.); e.add([Task(1, 0, 0., 3., 10., 10.)]); e.admit()
        with self.assertRaises(ValueError): e.assign({1: 1})
        self.assertIsNone(e.tasks[1].target)

    def test_fractional_deadline_not_rounded_to_grid(self):
        e = engine(dt=.1); e.add([Task(1, 0, 0., .255, 0., 2.55)]); e.admit(); e.assign({1: 0})
        e.advance([(0., 0.), (0., 0.)]); self.assertEqual(e.counts['success'], 1)
        self.assertAlmostEqual(e.finished[1].finished, .255)

    def test_empty_slot_valid_and_reward_is_new_completions(self):
        e = engine(); self.assertEqual(e.advance([(0., 0.), (0., 0.)])['success'], 0)
        e.add([Task(1, 0, 1., 3., 0., 1.)]); e.admit(); e.assign({1: 0})
        self.assertEqual(e.advance([(0., 0.), (0., 0.)])['success'], 1)
        self.assertEqual(e.advance([(0., 0.), (0., 0.)])['success'], 0)

    def test_global_inter_and_backhaul_pools_are_shared(self):
        for target in (1, 2):
            e = Engine(protocol(), [(100., 100.), (300., 100.)], [(100., 100.), (300., 100.)])
            e.add([Task(1, 0, 0., 5., 10., 0.), Task(2, 1, 0., 5., 10., 0.)]); e.admit()
            e.assign({1: target, 2: 0 if target == 1 else 2})
            e.advance([(0., 0.), (0., 0.)]); e.advance([(0., 0.), (0., 0.)])
            self.assertAlmostEqual(e.served_bits['transfer'], 10.)
            self.assertEqual(e.counts['success'], 0); e.audit()

    def test_same_ue_has_one_upload_at_a_time(self):
        e = engine(); e.add([Task(1, 0, 0., 4., 10., 0.), Task(2, 0, 0., 3., 10., 0.)])
        e.admit(); e.assign({1: 0, 2: 0}); e.advance([(0., 0.), (0., 0.)])
        self.assertEqual(e.finished[2].stage, 'completed'); self.assertEqual(e.tasks[1].remaining_upload, 10.)

    def test_hidden_future_not_needed_by_controller(self):
        from utils.queued_service_controllers import Controller
        e = engine(); e.add([Task(1, 0, 0., 3., 5., 5.)]); e.admit()
        obs = e.observe(); c = Controller('rolling_joint'); route, motion = c.act(obs)
        self.assertEqual(obs, e.observe()); self.assertIn(1, route)
        e.assign(route); e.advance(motion); e.audit()

    def test_rejected_batch_add_is_atomic(self):
        e = engine()
        with self.assertRaises(ValueError): e.add([Task(1, 0, 0., 3., 1., 1.), Task(1, 0, 0., 3., 1., 1.)])
        self.assertEqual(e.counts['generated'], 0); self.assertFalse(e.tasks)

    def test_zero_work_completion_is_in_current_reward_once(self):
        e = engine(); e.add([Task(1, 0, 0., 3., 0., 0.)]); e.admit(); e.assign({1: 0})
        self.assertEqual(e.advance([(0., 0.), (0., 0.)])['success'], 1)
        self.assertEqual(e.advance([(0., 0.), (0., 0.)])['success'], 0)


if __name__ == '__main__':
    unittest.main()
