"""The reported service latency must exclude unserved-UE penalties."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_canonical_suite import read_unit


class TestCanonicalAnalysisMetrics(unittest.TestCase):
    def test_admission_and_admitted_latency_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            attempt = job / "attempt_0000"
            episode = attempt / "episodes" / "00000.json"
            episode.parent.mkdir(parents=True)
            (attempt / "manifest.json").write_text(json.dumps({"source": {"source_hash": "test"}}))
            episode.write_text(json.dumps({"steps": [{"metrics": {
                "generated_service_count": 4,
                "deadline_satisfied_count": 1,
                "admitted_service_count": 2,
                "uav_fleet_total_energy": 10,
                "lower_assignment_energy": 6,
                "lower_service_latency": 3,
                "latency": 43,
                "offline_rate": 0,
                "fairness": 1,
                "service_offloads_mbs": 1,
            }}], "exogenous_trace_hash": "trace"}))
            hashes = {str(path.relative_to(job)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in attempt.rglob("*") if path.is_file()}
            (job / "completed.json").write_text(json.dumps({"status": "PASS", "attempt": attempt.name, "hashes": hashes}))
            metrics, traces = read_unit(job, 1, "test")
            self.assertEqual(metrics["DSR"], 0.25)
            self.assertEqual(metrics["Admission fraction"], 0.5)
            self.assertEqual(metrics["Latency per admitted service request"], 1.5)
            self.assertEqual(metrics["Penalized system latency per generated service request"], 10.75)
            self.assertEqual(metrics["Non-flight assignment energy"], 6)
            self.assertEqual(traces, ["trace"])


if __name__ == "__main__":
    unittest.main()
