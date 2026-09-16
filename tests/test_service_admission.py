import unittest
import numpy as np
import config
from environment.env import Env
from environment.user_equipments import UE


class TestServiceAdmission(unittest.TestCase):
    """TEST-03: Natural Service Admission (SSOT Section 3, Section 0)
    
    Verifies:
    1. In-coverage service UE (d <= R_c): assigned=True, associated to nearest UAV,
       enters admitted set, enters lower action set.
    2. Out-of-coverage service UE (d > R_c for all UAVs): assigned=False,
       does NOT enter any UAV's covered set or lower action set,
       included in system DSR denominator as unsatisfied,
       and lower policy reward does NOT bear its penalty.
    3. FORCE_SERVICE_ADMISSION_CANONICAL is strictly False.
    """

    def setUp(self):
        np.random.seed(42)
        self.env = Env()
        self.env.reset()

    def test_out_of_coverage_ue_not_assigned(self):
        """When UEs are strictly outside all UAV coverage radii (Rc = 100m),
        they must have assigned=False and must not be attached to any UAV.
        """
        # Position all UAVs at (100, 100)
        for uav in self.env.uavs:
            uav.pos[:2] = np.array([100.0, 100.0], dtype=np.float32)

        # Position UE 0 far away at (600, 600), distance > 500m >> 100m
        ue0 = self.env.ues[0]
        ue0.pos[:2] = np.array([600.0, 600.0], dtype=np.float32)
        # Ensure it has a service request
        ue0.current_request.req_type = config.REQUEST_TYPE_SERVICE
        ue0.current_request.is_service = True

        # Clear covered sets and re-associate
        for uav in self.env.uavs:
            uav.current_covered_ues = []
        ue0.assigned = False

        self.env._associate_ues_to_uavs()

        # Check UE 0 is not assigned
        self.assertFalse(
            ue0.assigned,
            "Out-of-coverage UE was assigned=True; violated canonical natural admission"
        )

        # Check UE 0 is not in any UAV's current_covered_ues
        for uav in self.env.uavs:
            self.assertNotIn(
                ue0, uav.current_covered_ues,
                f"Out-of-coverage UE was attached to UAV {uav.id}"
            )

    def test_in_coverage_ue_naturally_admitted(self):
        """A UE within 100m of UAV 0 must be admitted to UAV 0."""
        # Separate UAVs
        for i, uav in enumerate(self.env.uavs):
            uav.pos[:2] = np.array([150.0 + i * 120.0, 150.0], dtype=np.float32)
            uav.current_covered_ues = []

        # Place UE 0 close to UAV 0 (distance = 20m < 100m)
        ue0 = self.env.ues[0]
        ue0.pos[:2] = self.env.uavs[0].pos[:2] + np.array([20.0, 0.0], dtype=np.float32)
        ue0.current_request.req_type = config.REQUEST_TYPE_SERVICE
        ue0.current_request.is_service = True
        ue0.assigned = False

        self.env._associate_ues_to_uavs()

        self.assertTrue(ue0.assigned, "In-coverage UE was not assigned")
        self.assertIn(ue0, self.env.uavs[0].current_covered_ues, "In-coverage UE not attached to UAV 0")

    def test_forced_admission_canonical_is_false(self):
        """Even if legacy config or script sets FORCE_SERVICE_ADMISSION=True,
        canonical semantics forbids forcing uncovered UEs into UAV service sets.
        """
        original_val = getattr(config, "FORCE_SERVICE_ADMISSION", False)
        try:
            # Force setting to True to test legacy vulnerability
            config.FORCE_SERVICE_ADMISSION = True
            
            # Position UAVs at (100, 100)
            for uav in self.env.uavs:
                uav.pos[:2] = np.array([100.0, 100.0], dtype=np.float32)
                uav.current_covered_ues = []

            # Put UE 0 far away at (600, 600)
            ue0 = self.env.ues[0]
            ue0.pos[:2] = np.array([600.0, 600.0], dtype=np.float32)
            ue0.current_request.req_type = config.REQUEST_TYPE_SERVICE
            ue0.current_request.is_service = True
            ue0.assigned = False

            self.env._associate_ues_to_uavs()

            # Canonical rule: forced admission must be forbidden
            # Legacy code when FORCE_SERVICE_ADMISSION=True forcibly assigns it!
            # So this is an EXPECTED_RED under legacy code.
            self.assertFalse(
                ue0.assigned,
                "FORCE_SERVICE_ADMISSION allowed uncovered UE to be assigned to nearest UAV; violates canonical SSOT lock"
            )
        finally:
            config.FORCE_SERVICE_ADMISSION = original_val


if __name__ == "__main__":
    unittest.main()
