"""Statistical edge cases and verification-data exclusion (remote dependencies)."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.analysis.batch_offload_analysis import analyze, effect
from scripts.analysis.batch_offload_protocol import make_plan


class AnalysisTests(unittest.TestCase):
    def test_constant_effects_are_finite(self):
        for difference, p in ((0.,1.),(.02,0.),(-.02,0.)):
            result=effect(np.full((5,5),difference),np.zeros((5,5)))
            self.assertEqual(result['two_sided_p'],p)
            self.assertAlmostEqual(result['mean_difference'],difference)
            json.dumps(result,allow_nan=False)

    def test_training_seed_is_the_unit(self):
        base=np.zeros((5,5))
        values=np.tile(np.array([.01,.02,-.01,.03,.04])[:,None],(1,5))
        result=effect(values,base)
        self.assertEqual(len(result['training_seed_differences']),5)
        self.assertEqual(result['positive_training_seeds'],4)
        doubled=effect(np.tile(values,(1,2)),np.zeros((5,10)))
        self.assertEqual(result['training_seed_t_ci95'],doubled['training_seed_t_ci95'])

    def test_smoke_never_promotes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'suite.json').write_text(json.dumps({'plan':make_plan('smoke')}))
            with self.assertRaisesRegex(ValueError,'Smoke'):
                analyze(root)
