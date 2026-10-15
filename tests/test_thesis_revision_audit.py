"""Regression checks for request scoring and the independent-block estimand."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/analysis/thesis_revision_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RevisionAuditTest(unittest.TestCase):
    def test_deadline_equality_and_monotone_delay(self):
        requests = [{"latency": 1., "deadline": 1.}, {"latency": .8, "deadline": 1.},
                    {"latency": 1.2, "deadline": 1.}]
        self.assertEqual(audit.score_requests(requests, 0), 2)
        self.assertEqual(audit.score_requests(requests, .1), 1)
        self.assertEqual(audit.score_requests(requests, .3), 0)

    def test_empty_and_invalid_requests(self):
        self.assertEqual(audit.score_requests([], 0), 0)
        for delay in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                audit.score_requests([], delay)
        with self.assertRaises(ValueError):
            audit.score_requests([{"latency": float("nan"), "deadline": 1}], 0)

    def test_unadmitted_requests_stay_in_denominator(self):
        episodes = [{"method": "learned", "block": 1, "generated": 100,
                     "admitted": 10, "successes": {"0": 8}}]
        self.assertEqual(audit.block_rates(episodes, "0")["learned", 1], .08)

    def test_pool_counts_before_equal_block_weighting(self):
        episodes = [dict(method="learned", block=1, generated=1, successes={"0": 1}),
                    dict(method="learned", block=1, generated=99, successes={"0": 0}),
                    dict(method="learned", block=2, generated=10, successes={"0": 10})]
        values = list(audit.block_rates(episodes, "0").values())
        self.assertEqual(values, [.01, 1.])
        self.assertEqual(audit.interval(values)["mean"], .505)
        self.assertNotAlmostEqual(audit.interval(values)["mean"], 11 / 110)

    def test_complete_design_and_partial_rejection(self):
        p = dict(confirmation_blocks=list(range(1060001, 1060011)),
                 confirmation_workloads=list(range(1070001, 1070006)))
        tasks = audit.expected_tasks(p)
        self.assertEqual(len(tasks), 900)
        self.assertEqual(len(set(tasks)), 900)
        self.assertEqual(len(audit.expected_tasks(p, True)), 1)
        with tempfile.TemporaryDirectory() as tmp:
            audit.write(Path(tmp) / "protocol.json", dict(smoke=True, tasks=tasks[:1]))
            with self.assertRaisesRegex(ValueError, "partial"):
                audit.analyze(Namespace(output_root=Path(tmp)))


if __name__ == "__main__":
    unittest.main()
