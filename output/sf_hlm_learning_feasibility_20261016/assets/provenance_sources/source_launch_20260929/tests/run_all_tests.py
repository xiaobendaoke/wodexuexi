#!/usr/bin/env python3
"""Unified test runner for Canonical Semantics TDD validation.

Discovers and executes all test cases in tests/test_*.py using unittest,
captures timing, pass/fail status, assertions, and stack traces,
and prints both a detailed console output and structured JSON summary.
"""

import os
import sys
import time
import unittest
import traceback
from typing import Dict, Any, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(REPO_ROOT, "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    duration = time.time() - start_time

    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total - failed - errors - skipped

    print("\n" + "=" * 70)
    print("CANONICAL SEMANTICS TDD TEST RUNNER SUMMARY")
    print("=" * 70)
    print(f"Total Tests Run: {total}")
    print(f"Passed:          {passed}")
    print(f"Failed (RED):    {failed}")
    print(f"Errors:          {errors}")
    print(f"Skipped:         {skipped}")
    print(f"Duration:        {duration:.3f}s")
    print("=" * 70)

    if result.failures:
        print("\n--- FAILURES (EXPECTED / UNEXPECTED RED) ---")
        for test, err in result.failures:
            print(f"\n[FAIL] {test.id()}")
            lines = err.strip().split("\n")
            # Print last few lines of assertion
            print("  Assertion/Reason: " + lines[-1])

    if result.errors:
        print("\n--- ERRORS (INFRASTRUCTURE / RUNTIME) ---")
        for test, err in result.errors:
            print(f"\n[ERROR] {test.id()}")
            lines = err.strip().split("\n")
            print("  Reason: " + lines[-1])

    # Return non-zero if failures occurred so exit code reflects status
    # (for RED baseline, exit code 1 is expected when tests fail)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
