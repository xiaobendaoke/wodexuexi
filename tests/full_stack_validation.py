"""Remote bounded validation and source-bound attestation; no formal runs."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if platform.system() != "Linux":
    raise SystemExit("Remote Linux execution required")

from run_canonical_experiment import METHODS, parse_args, train
from utils.canonical_experiment import source_manifest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = {"source_hash": source_manifest()["source_hash"], "formal_training_started": False}
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with contextlib.redirect_stdout(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report["tests"] = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped)}
    report["test_log"] = stream.getvalue()
    report["pilots"] = []
    report["status"] = "FAIL"
    try:
        if result.wasSuccessful():
            with tempfile.TemporaryDirectory(prefix="canonical-validation-") as tmp:
                for method in METHODS:
                    for seed in (7, 11):
                        output = Path(tmp) / f"{method}-{seed}"
                        settings = parse_args(["--method", method, "--training-seed", str(seed), "--workload-seed", "19",
                            "--episodes", "1", "--steps", "8", "--rollout", "4", "--ppo-epochs", "1", "--batch-size", "2",
                            "--purpose", "verification", "--output", str(output)])
                        with contextlib.redirect_stdout(stream):
                            status = train(settings)
                        raw = json.loads((output / "episodes/00000.json").read_text())
                        report["pilots"].append({"method": method, "training_seed": seed, "status": status["status"],
                            "scenario_hash": raw["scenario_hash"], "exogenous_trace_hash": raw["exogenous_trace_hash"],
                            "steps": len(raw["steps"]), "final_losses": raw["final_losses"]})
                assert len({p["scenario_hash"] for p in report["pilots"]}) == 1
                assert len({p["exogenous_trace_hash"] for p in report["pilots"]}) == 1
        if result.wasSuccessful():
            report["status"] = "PASS"
    except Exception:
        report["pilot_error"] = traceback.format_exc()
    report["validation_log"] = stream.getvalue()
    write_json(Path(args.report), report)
    print(json.dumps({k: v for k, v in report.items() if k != "test_log"}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
