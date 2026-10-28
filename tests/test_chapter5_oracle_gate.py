"""Independent enumeration checks for the offline routing optimizer."""
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import chapter5_oracle_gate as gate


def method(success, seconds, energy=10.):
    return dict(requests=[dict(latency=.1 if k < success else 2., deadline=1.) for k in range(3)],
                success=success, seconds=[seconds]*3, mean_seconds=seconds, energy=energy, mbs=0)


def state(block, generated, f_success, g_success, f_time=.02, g_time=.01, g_energy=10.):
    return dict(identity=[block], weight=25, generated=generated, admitted=3,
                methods={'full': method(f_success, f_time), 'G': method(g_success, g_time, g_energy)})


class OracleTests(unittest.TestCase):
    def test_block_denominators_are_preserved(self):
        states = [state(1, 10, 2, 1), state(2, 100, 2, 1)]
        result = gate.aggregate(states, 'full', [1, 0])
        self.assertAlmostEqual(result['dsr_difference_pp'], -5.)
        self.assertNotAlmostEqual(result['pooled_dsr'], result['dsr'])

    def test_optimizer_matches_exhaustive_feasible_routes(self):
        states = [state(1, 100, 2, 1, g_energy=30.), state(1, 100, 1, 2, g_energy=8.),
                  state(2, 50, 3, 1), state(2, 50, 2, 2, g_energy=8.)]
        feasible = []
        for route in itertools.product((0., 1.), repeat=len(states)):
            m = gate.aggregate(states, 'full', route)
            if m['dsr_difference_pp'] >= -.5-1e-10 and m['energy_ratio'] <= 1.05+1e-10 and m['energy_ratio_G'] <= 1.05+1e-10:
                feasible.append(m['time_ratio'])
        binary, _ = gate.solve(states, 'full', .5)
        relaxed, _ = gate.solve(states, 'full', .5, relaxed=True)
        self.assertTrue(binary['optimal'])
        self.assertAlmostEqual(binary['objective_time_ratio'], min(feasible), places=8)
        self.assertLessEqual(relaxed['objective_time_ratio'], binary['objective_time_ratio']+1e-9)

    def test_overhead_rechecks_each_deadline(self):
        z = dict(requests=[dict(latency=.5, deadline=1.)], seconds=[.5, .49, .51])
        self.assertAlmostEqual(gate.paid_success(z), 2/3)
        self.assertAlmostEqual(gate.paid_success(z, .001), 1/3)


if __name__ == '__main__':
    unittest.main()
