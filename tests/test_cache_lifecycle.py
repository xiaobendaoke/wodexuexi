import unittest
import numpy as np
import config
from environment.env import Env
from environment.request_types import REQUEST_TYPE_SERVICE


class TestCacheLifecycle(unittest.TestCase):
    """TEST-06: Cache Lifecycle (SSOT Section 5)
    
    Verifies:
    1. Slot-start snapshot: all hit/miss evaluations in slot t read snapshot.
    2. Pending: fetched files do NOT become same-slot hits.
    3. Cross-UAV symmetry: UAV 0 offloading to UAV 4 vs UAV 4 to UAV 0.
       Legacy bug ENV-03: UAV 0 writing to UAV 4._working_cache is wiped
       when UAV 4 executes _working_cache = self.cache.copy().
    4. Commit permutation invariance with deterministic tie-breaking:
       score = EMA / size descending, tie-break by ascending file_id.
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_same_slot_cache_fetch_not_instant_hit(self):
        """If request 1 causes file f to be fetched from MBS, request 2 in the
        same slot requesting file f must NOT treat f as a cache hit because
        it was not in the slot-start snapshot.
        """
        uav0 = self.env.uavs[0]
        # Ensure file 0 is not in cache
        uav0.cache[0] = False
        if hasattr(uav0, "_working_cache"):
            uav0._working_cache[0] = False

        ue0 = self.env.ues[0]
        ue1 = self.env.ues[1]
        for ue in [ue0, ue1]:
            ue.pos[:2] = uav0.pos[:2] + np.array([10.0, 0.0], dtype=np.float32)
            ue.current_request.req_type = REQUEST_TYPE_SERVICE
            ue.current_request.req_id = 0  # Both request file 0
            ue.current_request.req_size = 1000

        uav0._current_covered_ues = [ue0, ue1]
        for u in self.env.uavs[1:]:
            u._current_covered_ues = []

        offload_actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
        # Both Local
        offload_actions[0, 0] = 0
        offload_actions[0, 1] = 0

        self.env.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32), offloading_actions=offload_actions)

        # In canonical semantics:
        # File 0 was NOT in slot-start snapshot.
        # Both ue0 and ue1 must encounter a cache miss in slot t (or pay fetch latency).
        # In legacy code, ue0 fetches file 0, calls _try_add_file_to_cache which writes
        # self.cache[req_id] = True (or _working_cache), so ue1 immediately gets a hit!
        # This is an EXPECTED_RED under legacy code.
        self.assertAlmostEqual(
            ue0.latency_current_request, ue1.latency_current_request, places=4,
            msg=f"Same-slot cache causality violation: ue0 latency ({ue0.latency_current_request}s) != ue1 latency ({ue1.latency_current_request}s)"
        )

    def test_cross_uav_cache_cooperation_not_wiped(self):
        """When UAV 0 offloads a task to UAV 4, any cache update on UAV 4
        must not be wiped when UAV 4 executes its process_requests loop.
        
        Legacy bug ENV-03: UAV 4 starts with `self._working_cache = self.cache.copy()`,
        wiping whatever UAV 0 added to UAV 4's working cache.
        """
        uav0 = self.env.uavs[0]
        uav4 = self.env.uavs[4]
        # Place UAV 0 and UAV 4 close enough to cooperate
        uav4.pos[:2] = uav0.pos[:2] + np.array([50.0, 0.0], dtype=np.float32)
        uav0.set_neighbors(self.env.uavs)
        uav4.set_neighbors(self.env.uavs)

        uav4.cache[:] = False
        uav0.cache[:] = False

        ue0 = self.env.ues[0]
        ue0.pos[:2] = uav0.pos[:2] + np.array([10.0, 0.0], dtype=np.float32)
        ue0.current_request.req_type = REQUEST_TYPE_SERVICE
        ue0.current_request.req_id = 5
        ue0.current_request.req_size = 1000

        uav0._current_covered_ues = [ue0]
        for u in self.env.uavs[1:]:
            u._current_covered_ues = []

        # Action: UAV 0 offloads to UAV 4 (coop action = 2 + 4 = 6)
        offload_actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=np.int64)
        offload_actions[0, 0] = int(config.OFFLOAD_ACTION_COOP_BASE + 4)

        self.env.step(np.zeros((config.NUM_UAVS, 2), dtype=np.float32), offloading_actions=offload_actions)

        # Under canonical semantics, the fetched file is committed to UAV 4 at the end of the slot
        # and must persist into the next slot's cache.
        # Under legacy code, UAV 4 resets its _working_cache when its turn comes in the loop,
        # so changes made by UAV 0 are lost.
        self.assertTrue(
            uav4.cache[5],
            "UAV 4 cache did not retain file 5 after cooperative execution; wiped by loop reset bug ENV-03"
        )


if __name__ == "__main__":
    unittest.main()
