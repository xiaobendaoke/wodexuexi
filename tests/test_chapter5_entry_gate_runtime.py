"""Real numerical integration tests for non-forced gate predicate boundaries."""
import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import chapter5_stage_b as b


class GateRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'FAST' not in globals():
            raise unittest.SkipTest('Use this file\'s CLI to bootstrap frozen server inputs')

    def test_nonforced_rules_reuse_native_output_on_both_sides(self):
        from utils.sf_hlm_priced_runtime import PricedPolicy
        from chapter5_entry_gate_policy import EntryGatePolicy, feature_pair
        from chapter5_entry_gate_verify import close
        with FAST.FastRuntime() as rt:
            native = PricedPolicy(rt, None, PRICES, PRICES['powers'])
            g = native.run(ROWS, num_uavs=5, kind='native_G', trace=True)
            n, q = feature_pair(ROWS, g[1])
            rt.clear()
            full = native.run(ROWS, num_uavs=5, kind='mixture_price', cap=10, trace=True)
            cases = [('A', n, 1.01, True), ('A', n+1, 1.01, False),
                     ('B', 0, q+1e-6, True), ('B', 0, q, False),
                     ('C', n, q+1e-6, True), ('C', n+1, q+1e-6, False)]
            for mode, n0, q0, expected in cases:
                with self.subTest(mode=mode, entered=expected):
                    rt.clear()
                    gate = EntryGatePolicy(rt, PRICES, dict(mode=mode, n0=n0, q0=q0))
                    with patch.object(rt, 'common_repair', wraps=rt.common_repair) as repair:
                        result = gate.run(ROWS, num_uavs=5, kind='mixture_price', cap=10, trace=True)
                    self.assertEqual(repair.call_count, 1)
                    self.assertEqual(gate.last_trace['entered'], expected)
                    close(b.normal(result), b.normal(full if expected else g))

    def test_invalid_contract_is_rejected_before_work(self):
        from chapter5_entry_gate_policy import EntryGatePolicy
        with FAST.FastRuntime() as rt:
            gate = EntryGatePolicy(rt, PRICES, dict(mode='A', n0=20))
            with patch.object(rt, 'greedy', side_effect=AssertionError('Invalid request started work')):
                with self.assertRaises(ValueError):
                    gate.run(ROWS, num_uavs=5, kind='mixture_price', cap=16)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--development', type=Path, required=True)
    p.add_argument('--c0', type=Path, required=True)
    a = p.parse_args()
    _, _, FAST, _, PRICES = b.boot(a, 620001)
    state = b.read(a.c0/'protocol.json')['states'][0]
    ROWS = b.read(Path(state['folder'])/'public.json')['rows']
    for row in ROWS:
        row['options'] = {int(k): v for k, v in row['options'].items()}
    unittest.main(argv=[sys.argv[0]])
