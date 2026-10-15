"""Frozen SF-HLM request-slack audit, without training or delayed state updates.

inventory/analyze use only the standard library. collect is Linux-only and
replays archived lower assignments with the frozen upper policy. It reads the
archived launch manifest instead of embedding machine-specific source paths.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time
import types

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("learned", "random", "greedy")
DELAYS_MS = (0, 20, 50, 100, 150)
COMPONENTS = ("service_compute_energy", "flight_energy", "hover_energy",
              "service_comm_energy", "service_fetch_or_backhaul_energy",
              "content_related_energy", "wpt_energy")
T95 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105,
       5: 2.570581836, 6: 2.446911851, 7: 2.364624252, 8: 2.306004135,
       9: 2.262157162854099}
_RUNTIME = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def score_requests(requests, delay_seconds):
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("Delay must be finite and nonnegative")
    for row in requests:
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in ("latency", "deadline")):
            raise ValueError("Invalid actual request time")
    return sum(row["latency"] + delay_seconds <= row["deadline"] for row in requests)


def interval(values):
    n = len(values)
    if n < 2 or n > 10:
        raise ValueError("Paired interval requires 2 to 10 independent blocks")
    mean = statistics.mean(values)
    radius = T95[n - 1] * statistics.stdev(values) / math.sqrt(n)
    return {"mean": mean, "ci95": [mean - radius, mean + radius], "values": values, "n": n}


def block_rates(episodes, score_key):
    totals = {}
    for ep in episodes:
        key = (ep["method"], ep["block"])
        unit = totals.setdefault(key, {"generated": 0, "success": 0})
        unit["generated"] += ep["generated"]
        unit["success"] += ep["successes"][score_key]
    return {key: v["success"] / max(v["generated"], 1) for key, v in totals.items()}


def expected_tasks(protocol, smoke=False):
    tasks = [(m, b, w, e) for m in METHODS for b in protocol["confirmation_blocks"]
             for w in protocol["confirmation_workloads"] for e in range(6)]
    return tasks[:1] if smoke else tasks


def archived_episode(root, phase, task):
    method, block, workload, episode = task
    return Path(root) / phase / method / f"block_{block}_workload_{workload}" / f"episode_{episode:05d}" / "episode.json"


def inventory(args):
    protocol = read(args.archive_root / "protocol.json")
    tasks = expected_tasks(protocol)
    missing = []
    for task in tasks:
        for phase in ("confirmation_eval", "confirmation_timing"):
            p = archived_episode(args.archive_root, phase, task)
            if not p.is_file():
                missing.append(str(p))
    sample = archived_episode(args.archive_root, "confirmation_eval", tasks[0])
    row = read(sample) if sample.exists() else None
    request_detail = bool(row and "requests" in row["steps"][0]
                          and all("latency" in q and "deadline" in q for q in row["steps"][0]["requests"]))
    result = dict(status="AVAILABLE" if not missing else "MISSING_INPUT", expected_episodes=len(tasks),
                  missing_files=missing, sample_step_fields=list(row["steps"][0]) if row else [],
                  actual_request_detail_in_sample=request_detail,
                  protocol_sha256=sha(args.archive_root / "protocol.json"),
                  replay_required=not request_detail, no_training=True)
    write(args.output_root / "inventory.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "missing_files"}, ensure_ascii=False))
    if missing:
        raise FileNotFoundError(f"{len(missing)} required original files unavailable")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def launch_args(args, block, method):
    manifest = args.launch_manifest or args.experiment_root / "output/sf_hlm_v2_20261001/validation/followup_confirmation_launch.json"
    command = read(manifest)["command"]
    paths = {}
    for i, value in enumerate(command[:-1]):
        if value.startswith("--") and not command[i + 1].startswith("--"):
            key = value[2:].replace("-", "_")
            if key in ("source_root", "suite_root", "extension_root", "output", "legacy_results",
                       "resources", "adaptation_root", "followup_root", "results_root"):
                paths[key] = Path(command[i + 1])
    paths["results_root"] = args.archive_root
    required = ("source_root", "suite_root", "extension_root", "output", "legacy_results",
                "resources", "adaptation_root", "followup_root", "results_root")
    if set(paths) != set(required):
        raise ValueError("Launch manifest does not describe all frozen source roots")
    return argparse.Namespace(**paths, block=block, method=method, seed=block, family="complete",
                              phase="confirmation_eval", workload=0, episode=0)


def runtime(args, block, method):
    global _RUNTIME
    original = launch_args(args, block, method)
    if _RUNTIME is None:
        h = load_module("revision_frozen_followup", original.followup_root / "scripts/analysis/sf_hlm_followup.py")
        r = h.runner(original)
        old, _ = h.initialize(original, r)
        _RUNTIME = h, r, old
    h, r, old = _RUNTIME
    protocol = read(args.archive_root / "protocol.json")
    if sha(original.output / "protocol.json") != protocol["original_protocol_sha256"]:
        raise ValueError("Parent frozen protocol changed")
    for key, root in (("files", original.followup_root), ("extension_files", original.extension_root),
                      ("adaptation_files", original.adaptation_root)):
        for name, checksum in protocol[key].items():
            if sha(root / name) != checksum:
                raise ValueError("Archived source changed: " + name)
    return original, h, r, old, protocol


def replay_group(payload):
    args, method, block, tasks = payload
    import numpy as np
    original, h, r, old, protocol = runtime(args, block, method)
    destination = args.output_root / "groups" / method / str(block)
    contract = {"parent_protocol_sha256": sha(args.archive_root / "protocol.json"),
                "audit_script_sha256": sha(__file__), "method": method, "block": block,
                "tasks": [list(t) for t in tasks], "delays_ms": list(DELAYS_MS),
                "smoke": args.smoke, "state_delay_applied": False}
    manifest_path = destination / "result.json"
    if manifest_path.exists():
        receipt = read(manifest_path)
        if receipt["contract"] != contract:
            raise ValueError("Existing audit contract differs; do not overwrite")
        for name, checksum in receipt["files"].items():
            if sha(destination / name) != checksum:
                raise ValueError("Saved audit data changed")
        return str(manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    with r.res.CpuLease(original.resources, task=["thesis_revision", method, block], kind="environment"):
        models, source, checkpoint, _ = h.basis(original, r, old, protocol)
        lower, quota_hash = h.quota(original, r, old, protocol) if method == "learned" else (None, None)
        fingerprint = r.base.policy_fingerprint([models["upper"], models["lower"], lower, None])
        import config
        from utils.canonical_experiment import digest, exogenous_frame
        summaries = []
        input_hashes = {}
        temporary = destination / "requests.jsonl.gz.tmp"
        with gzip.open(temporary, "wt", encoding="utf8") as stream:
            for task in tasks:
                _, _, workload, episode = task
                qp = archived_episode(args.archive_root, "confirmation_eval", task)
                tp = archived_episode(args.archive_root, "confirmation_timing", task)
                qreceipt = h.verified(qp.parent)
                h.verified(tp.parent)
                quality, timing = read(qp), read(tp)
                if quality["summary"] != timing["summary"] or quality["exogenous_trace_hash"] != timing["exogenous_trace_hash"]:
                    raise ValueError("Isolated timing does not match quality trajectory")
                qc = qreceipt["contract"]
                if (qc["basis_checkpoint_sha256"] != sha(checkpoint / "contract.json")
                        or qc["quota_checkpoint_sha256"] != quota_hash
                        or qc["config_hash"] != source["config_hash"]
                        or qreceipt["model_fingerprint"] != fingerprint):
                    raise ValueError("Frozen checkpoint, fingerprint or config differs")
                if len(quality["steps"]) != 1000 or len(timing["steps"]) != 1000:
                    raise ValueError("Incomplete original episode")
                input_hashes[str(qp)] = sha(qp)
                input_hashes[str(tp)] = sha(tp)
                env, obs = r.episode_env(workload, episode)
                observed = {}
                previous_metrics = env._get_rewards_and_metrics

                def capture_metrics():
                    rewards, metrics = previous_metrics()
                    observed["requests"] = [{"ue_id": ue.id, "latency": float(ue.latency_current_request),
                                             "deadline": float(ue.current_request.deadline)}
                                            for ue in env.ues if ue.current_request.is_service and ue.assigned]
                    return rewards, metrics

                env._get_rewards_and_metrics = capture_metrics
                total = dict(method=method, block=block, workload=workload, episode=episode,
                             generated=0, admitted=0, path_over_one_second=0, energy=0.0,
                             components={k: 0.0 for k in COMPONENTS},
                             successes={**{str(ms): 0 for ms in DELAYS_MS}, "software": 0})
                trace = hashlib.sha256()
                action_hash = hashlib.sha256()
                for step, (qstep, tstep) in enumerate(zip(quality["steps"], timing["steps"])):
                    if qstep["step"] != step or tstep["step"] != step or qstep["assignment"] != tstep["assignment"]:
                        raise ValueError("Timing request/action identity changed")
                    trace.update(digest(exogenous_frame(env)).encode())
                    _, masks, requests, _ = r.base.prepare(env)
                    if [v[2] for v in requests] != qstep["request_ids"] or qstep["request_ids"] != tstep["request_ids"]:
                        raise ValueError("Admitted request order differs")
                    assignment = qstep["assignment"]
                    actions = r.base.actions_from_assignment(requests, assignment, masks)
                    motion = np.clip(models["upper"].select_actions(obs, exploration=False), -1, 1)
                    obs, _, metrics = env.step(motion, offloading_actions=actions)
                    obs = np.asarray(obs, np.float32)
                    for key, value in qstep["metrics"].items():
                        if not math.isclose(float(metrics[key]), float(value), rel_tol=1e-10, abs_tol=1e-10):
                            raise ValueError(f"Replay differs at {task}, step {step}, metric {key}")
                    rows = observed["requests"]
                    if len(rows) != int(metrics["admitted_service_count"]):
                        raise ValueError("Captured request count differs")
                    if score_requests(rows, 0) != int(metrics["deadline_satisfied_count"]):
                        raise ValueError("Zero-delay request score differs")
                    elapsed = float(tstep["policy_seconds"])
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("Invalid isolated per-call timing")
                    total["generated"] += int(metrics["generated_service_count"])
                    total["admitted"] += len(rows)
                    total["path_over_one_second"] += sum(v["latency"] > 1 for v in rows)
                    total["energy"] += float(metrics["energy"])
                    for key in COMPONENTS:
                        total["components"][key] += float(metrics[key])
                    if not math.isclose(sum(float(metrics[k]) for k in COMPONENTS), float(metrics["energy"]), rel_tol=1e-10, abs_tol=1e-8):
                        raise ValueError("Energy components do not reconcile")
                    for ms in DELAYS_MS:
                        total["successes"][str(ms)] += score_requests(rows, ms / 1000)
                    total["successes"]["software"] += score_requests(rows, elapsed)
                    action_hash.update(digest(assignment).encode())
                    stream.write(json.dumps(dict(method=method, block=block, workload=workload, episode=episode,
                                                 step=step, generated=int(metrics["generated_service_count"]),
                                                 software_seconds=elapsed, requests=rows), separators=(",", ":")) + "\n")
                if trace.hexdigest() != quality["exogenous_trace_hash"]:
                    raise ValueError("Exogenous stream replay differs")
                if total["generated"] != quality["summary"]["generated"] or total["successes"]["0"] != quality["summary"]["success"]:
                    raise ValueError("Episode counts differ")
                if r.base.policy_fingerprint([models["upper"], models["lower"], lower, None]) != fingerprint:
                    raise ValueError("Replay mutated model parameters")
                total.update(exogenous_trace_hash=trace.hexdigest(), replayed_assignment_hash=action_hash.hexdigest(),
                             steps=1000, checkpoint_fingerprint=fingerprint)
                summaries.append(total)
        temporary.replace(destination / "requests.jsonl.gz")
        write(destination / "episodes.json", summaries)
        receipt = dict(status="PASS_SMOKE" if args.smoke else "PASS", contract=contract, input_hashes=input_hashes,
                       files={name: sha(destination / name) for name in ("requests.jsonl.gz", "episodes.json")},
                       collection_mode="archived_assignment_replay_with_frozen_upper",
                       actual_search_rerun=False, state_delay_applied=False)
        write(manifest_path, receipt)
    return str(manifest_path)


def collect(args):
    if sys.platform != "linux":
        raise RuntimeError("Frozen simulation collection must run on remote Linux")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    protocol = read(args.archive_root / "protocol.json")
    tasks = expected_tasks(protocol, args.smoke)
    grouped = {}
    for task in tasks:
        grouped.setdefault(task[:2], []).append(task)
    contract = dict(parent_protocol_sha256=sha(args.archive_root / "protocol.json"),
                    audit_script_sha256=sha(__file__), tasks=[list(t) for t in tasks],
                    smoke=args.smoke, no_training=True, state_delay_applied=False)
    target = args.output_root / "protocol.json"
    if target.exists() and read(target) != contract:
        raise ValueError("Output protocol differs; do not overwrite another run")
    write(target, contract)
    payloads = [(args, method, block, ts) for (method, block), ts in grouped.items()]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(replay_group, p) for p in payloads]
        done = []
        for future in as_completed(futures):
            done.append(future.result())
            write(args.output_root / "status.json", dict(status="RUNNING", completed_groups=len(done),
                                                       expected_groups=len(grouped), completed_episodes=sum(len(read(Path(p).parent / "episodes.json")) for p in done),
                                                       expected_episodes=len(tasks), started_unix=started))
    write(args.output_root / "status.json", dict(status="PASS_SMOKE" if args.smoke else "PASS",
                                               completed_groups=len(done), completed_episodes=len(tasks),
                                               expected_episodes=len(tasks), elapsed_seconds=time.time() - started))


def analyze(args):
    protocol = read(args.output_root / "protocol.json")
    if protocol["smoke"] or len(protocol["tasks"]) != 900:
        raise ValueError("Smoke or partial collection cannot support thesis results")
    episodes, receipts = [], []
    for method in METHODS:
        for block in range(1060001, 1060011):
            folder = args.output_root / "groups" / method / str(block)
            receipt = read(folder / "result.json")
            if receipt["status"] != "PASS" or receipt["contract"]["parent_protocol_sha256"] != protocol["parent_protocol_sha256"]:
                raise ValueError("Incomplete or incompatible collection")
            for name, checksum in receipt["files"].items():
                if sha(folder / name) != checksum:
                    raise ValueError("Collection file checksum changed")
            episodes.extend(read(folder / "episodes.json"))
            receipts.append({"path": str(folder / "result.json"), "sha256": sha(folder / "result.json")})
    identities = [(v["method"], v["block"], v["workload"], v["episode"]) for v in episodes]
    if len(set(identities)) != len(identities) or set(identities) != {tuple(t) for t in protocol["tasks"]}:
        raise ValueError("Missing or duplicated episode identity")
    blocks = list(range(1060001, 1060011))
    scenarios = {}
    for key in [*(str(ms) for ms in DELAYS_MS), "software"]:
        rates = block_rates(episodes, key)
        groups = {m: interval([rates[m, b] for b in blocks]) for m in METHODS}
        comparisons = {"learned-" + m: interval([rates["learned", b] - rates[m, b] for b in blocks])
                       for m in ("random", "greedy")}
        scenarios[key] = dict(groups=groups, comparisons=comparisons)
    costs = {}
    for method in METHODS:
        by_block = []
        for block in blocks:
            eps = [v for v in episodes if v["method"] == method and v["block"] == block]
            steps = sum(v["steps"] for v in eps)
            energy = sum(v["energy"] for v in eps)
            components = {k: sum(v["components"][k] for v in eps) / steps for k in COMPONENTS}
            admitted = sum(v["admitted"] for v in eps)
            by_block.append(dict(block=block, energy_j_step=energy / steps, components=components,
                                 shares={k: components[k] / max(energy / steps, 1e-15) for k in COMPONENTS},
                                 path_over_one_second=sum(v["path_over_one_second"] for v in eps) / max(admitted, 1)))
        costs[method] = dict(blocks=by_block, energy_j_step=statistics.mean(v["energy_j_step"] for v in by_block),
                            components={k: statistics.mean(v["components"][k] for v in by_block) for k in COMPONENTS},
                            shares={k: statistics.mean(v["shares"][k] for v in by_block) for k in COMPONENTS},
                            path_over_one_second=statistics.mean(v["path_over_one_second"] for v in by_block))
    for method in METHODS:
        means = [scenarios[str(ms)]["groups"][method]["mean"] for ms in DELAYS_MS]
        if any(a < b for a, b in zip(means, means[1:])):
            raise ValueError("Success rate increases with added delay")
    result = dict(status="PASS", episodes=900, independent_blocks=10, evaluation_steps=900000,
                  scenarios=scenarios, energy=costs, receipts=receipts,
                  protocol_sha256=sha(args.output_root / "protocol.json"),
                  aggregation="sum request counts within each block, then equal block means",
                  interpretation="exploratory post-hoc scoring on original trajectories; no delayed state transition",
                  confidence_intervals="pointwise paired t, no new significance claim", state_delay_applied=False)
    write(args.output_root / "analysis/summary.json", result)
    print(json.dumps({k: result[k] for k in ("status", "episodes", "independent_blocks", "evaluation_steps")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "collect", "analyze"))
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--launch-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/thesis_revision_20261014")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command in ("inventory", "collect") and args.archive_root is None:
        parser.error("--archive-root required")
    if args.command == "collect" and args.experiment_root is None:
        parser.error("--experiment-root required")
    if args.workers < 1:
        parser.error("--workers must be positive")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
