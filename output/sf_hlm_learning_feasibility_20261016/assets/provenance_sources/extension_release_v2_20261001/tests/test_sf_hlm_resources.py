"""Non-simulation tests for scheduling identity and duplicate protection."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys

spec=importlib.util.spec_from_file_location('resources',Path(__file__).resolve().parents[1]/'utils/sf_hlm_resources.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class ResourceTests(unittest.TestCase):
    def test_task_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'task.lock'
            with r.task_lock(p):
                with self.assertRaises(RuntimeError):
                    with r.task_lock(p): pass
            with r.task_lock(p):pass

    def test_seed_does_not_depend_on_dispatch_order(self):
        ids=[('protocol',9,2,i,3) for i in range(32)]
        forward={x:r.trajectory_seed(*x) for x in ids}
        reverse={x:r.trajectory_seed(*x) for x in reversed(ids)}
        self.assertEqual(forward,reverse);self.assertEqual(len(set(forward.values())),32)

    @unittest.skipUnless(sys.platform=='linux','Linux affinity and process identity')
    def test_live_owner_survives_supervisor_lock_loss(self):
        with tempfile.TemporaryDirectory() as d:
            cpu=next(iter(os.sched_getaffinity(0)));r.atomic_json(Path(d)/'profile.json',dict(cpus=[cpu],slots=1,physical_cores=1,reserved=[]))
            child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            try:
                lease=r.CpuLease(d,owner=child.pid,pin=False).acquire()
                os.close(lease.fd);lease.fd=None # supervisor crash without clearing the live owner
                with self.assertRaises(TimeoutError):r.CpuLease(d,pin=False,timeout=.05).acquire()
            finally:child.terminate();child.wait()
            with r.CpuLease(d,pin=False,timeout=.1):pass

    @unittest.skipUnless(sys.platform=='linux','Linux topology')
    def test_reserve_whole_physical_cores_and_exclusive_slots(self):
        info=r.topology();self.assertFalse(set(info['cpus'])&set(info['reserved']))
        self.assertEqual(len(set(info['cpus'])),info['maximum_slots'])
        with tempfile.TemporaryDirectory() as d:
            r.atomic_json(Path(d)/'profile.json',dict(info,slots=2))
            with r.CpuLease(d,pin=False) as first,r.CpuLease(d,pin=False) as second:
                self.assertNotEqual(first.cpu,second.cpu)
                with self.assertRaises(TimeoutError):r.CpuLease(d,pin=False,timeout=.05).acquire()

    @unittest.skipUnless(sys.platform=='linux','Linux leases')
    def test_task_type_limit_is_shared_with_nested_workers(self):
        with tempfile.TemporaryDirectory() as d:
            info=r.topology();r.atomic_json(Path(d)/'profile.json',dict(info,slots=4,concurrency_by_kind={'rollout_complete':1}))
            with r.CpuLease(d,pin=False,kind='rollout_complete') as parent:
                with self.assertRaises(TimeoutError):r.CpuLease(d,pin=False,kind='rollout_complete',timeout=.05).acquire()
                with parent.suspended():
                    with r.CpuLease(d,pin=False,kind='rollout_complete',timeout=.1):pass


if __name__=='__main__':unittest.main()
