"""Capture server regression evidence without making scientific claims."""
import argparse
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.dynamic_service_experiment import atomic_json, configure_runtime, source_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure_runtime()
    suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_dynamic_service_scenarios', 'tests.test_dynamic_service_runtime'])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    evidence = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                    output=stream.getvalue(), source=source_contract(), scientific_claims=False,
                    review='explicit self-review; not independent agent review')
    atomic_json(args.output, evidence)
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
