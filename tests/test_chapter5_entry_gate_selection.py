"""Check group isolation and conservative fallback in threshold development."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from chapter5_entry_gate_select import choose


class SelectionTests(unittest.TestCase):
    def dataset(self):
        data = dict(block=[], generated=[], admitted=[], Sf=[], Sg=[], Tf=[], Tg=[],
                    Ef=[], Eg=[], Mf=[], Mg=[], n=[], q=[])
        for block in range(5):
            for large in (False, True):
                row = dict(block=block, generated=100, admitted=30, Sf=10+large, Sg=10,
                           Tf=4, Tg=1, Ef=10, Eg=10, Mf=1, Mg=1,
                           n=30 if large else 10, q=.2 if large else .9)
                for k, v in row.items():
                    data[k].append(v)
        return {k: np.asarray(v, dtype=float) for k, v in data.items()}

    def test_heldout_outcomes_cannot_change_training_threshold(self):
        a = self.dataset()
        b = copy.deepcopy(a)
        b['Sg'][b['block'] == 4] = 0
        b['Tg'][b['block'] == 4] = 1000
        for mode in ('A', 'B', 'C'):
            first, _ = choose(a, mode, [0, 1, 2, 3])
            second, _ = choose(b, mode, [0, 1, 2, 3])
            self.assertFalse(first['fallback_full'])
            self.assertEqual(first, second)
            json.dumps(choose(a, mode, [0, 1, 2, 3]), allow_nan=False)

    def test_no_safe_speedup_returns_full(self):
        a = self.dataset()
        a['Tg'][:] = 4
        for mode in ('A', 'B', 'C'):
            rule, grid = choose(a, mode, [0, 1, 2, 3])
            self.assertTrue(rule['fallback_full'])
            self.assertFalse(any(z['eligible'] for z in grid))


if __name__ == '__main__':
    unittest.main()
