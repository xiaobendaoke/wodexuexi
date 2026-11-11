"""Run and analyze energy-free Chapter 3 and Chapter 4 experiments on the remote server.

Stages:
  - canonical_baseline: 5 training seeds x 10 workloads comparing hierarchical vs lower_no_attention
  - bottleneck: 5 training seeds x 5 workloads for 4 lower architectures + 50 frozen snapshots diagnostic
  - ch3_pilot: 5 training seeds x 5 workloads comparing lower_no_attention vs lower_mlp_upper_uncovered_obs
  - ch3_confirm: 10 training seeds x 5 workloads confirmation comparing lower_no_attention vs lower_mlp_upper_uncovered_obs
  - ch4_bridge: 10 training seeds x 5 workloads x 8 frozen snapshots (400 snapshots) bridge comparison
  - export_figure_data: copy all energy-free JSON/CSV artifacts to docs/figures/data/chapter34_20261004/
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from analyze_canonical_suite import paired_effect
from environment.env import Env
from run_canonical_experiment import configure_method, parse_args as parse_canonical_args, policy_step, train as run_canonical_unit
from utils.assignment_search import best_improvement
from utils.canonical_experiment import config_snapshot, digest, source_manifest, write_json
from utils.chapter4 import objective, request_objective, solve_snapshot
from utils.logger import refresh_derived_config_fields
from utils.snapshot_evaluator import capture, evaluate

T95_DF9 = 2.262157162854099

CANONICAL_TRAIN_SEEDS = [42, 84, 126, 168, 210]
CANONICAL_WORKLOAD_SEEDS = [42, 84, 126, 168, 210, 252, 294, 336, 378, 420]

PILOT_TRAIN_SEEDS = [2767, 3191, 3623, 4081, 4597]
PILOT_WORKLOAD_SEEDS = [1433, 1471, 1511, 1553, 1601]
BOTTLENECK_SNAPSHOT_WORKLOADS = [1433, 1511]
BOTTLENECK_SNAPSHOT_STEPS = [100, 300, 500, 700, 900]

CONFIRM_TRAIN_SEEDS = [7013, 7283, 7541, 7823, 8089, 8353, 8627, 8887, 9151, 9419]
CONFIRM_WORKLOAD_SEEDS = [2203, 2243, 2287, 2333, 2377]
BRIDGE_WORKLOAD_SEEDS = [3001, 3037, 3061, 3097, 3121]
BRIDGE_SNAPSHOT_EPISODES = [0, 1]
BRIDGE_SNAPSHOT_STEPS = [100, 300, 500, 700]


def _run_canonical_job(spec: dict) -> str:
    output = Path(spec["output"])
    done_path = output / "last_invocation.json"
    if done_path.exists():
        data = json.loads(done_path.read_text())
        if data.get("status") == "PASS":
            return str(output)
        shutil.rmtree(output, ignore_errors=True)
    elif output.exists():
        shutil.rmtree(output, ignore_errors=True)

    argv = [
        "--method", spec["method"],
        "--training-seed", str(spec["training_seed"]),
        "--workload-seed", str(spec["workload_seed"]),
        "--episodes", str(spec["episodes"]),
        "--steps", str(spec["steps"]),
        "--rollout", str(spec["rollout"]),
        "--ppo-epochs", str(spec["ppo_epochs"]),
        "--batch-size", str(spec["batch_size"]),
        "--threads", str(spec["threads"]),
        "--purpose", "pilot",
        "--checkpoint-every", str(spec["episodes"]),
        "--output", str(output),
    ]
    if spec.get("evaluate_checkpoint"):
        argv.extend(["--evaluate-checkpoint", str(spec["evaluate_checkpoint"])])
    run_canonical_unit(parse_canonical_args(argv))
    return str(output)


def _read_eval_metrics(directory: Path, episodes: int) -> tuple[dict[str, float], list[str], dict[str, float]]:
    files = sorted((directory / "episodes").glob("*.json"))
    if len(files) != episodes:
        raise ValueError(f"Expected {episodes} episodes in {directory}, found {len(files)}")
    raw = [json.loads(p.read_text()) for p in files]
    rows = [r["metrics"] for ep in raw for r in ep["steps"]]
    generated = float(sum(r["generated_service_count"] for r in rows))
    success = float(sum(r["deadline_satisfied_count"] for r in rows))
    admitted = float(sum(r["admitted_service_count"] for r in rows))
    uncovered = float(sum(r["unadmitted_service_count"] for r in rows))
    metrics = {
        "DSR": success / generated if generated else 0.0,
        "Admission fraction": admitted / generated if generated else 0.0,
        "Deadline satisfaction among admitted": success / admitted if admitted else 0.0,
        "Penalized system latency per generated service request": sum(r["latency"] for r in rows) / generated if generated else 0.0,
        "Latency per admitted service request": sum(r["lower_service_latency"] for r in rows) / admitted if admitted else 0.0,
        "Fairness": float(np.mean([r["fairness"] for r in rows])),
        "MBS offload fraction": sum(r["service_offloads_mbs"] for r in rows) / admitted if admitted else 0.0,
    }
    counts = {
        "generated": generated,
        "success": success,
        "admitted": admitted,
        "uncovered": uncovered,
    }
    traces = [r["exogenous_trace_hash"] for r in raw]
    return metrics, traces, counts


def _execute_cohort(
    root: Path,
    methods: list[str],
    train_seeds: list[int],
    workload_seeds: list[int],
    *,
    train_episodes: int,
    eval_episodes: int,
    steps: int,
    rollout: int,
    ppo_epochs: int,
    batch_size: int,
    workers: int,
    threads: int,
    train_workload_seed: int = 2026,
    shared_train_root: Path | None = None,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    train_jobs = []
    for method in methods:
        for seed in train_seeds:
            base = (shared_train_root if shared_train_root is not None else root) / "main" / method / f"seed_{seed}" / "train"
            train_jobs.append({
                "method": method,
                "training_seed": seed,
                "workload_seed": train_workload_seed,
                "episodes": train_episodes,
                "steps": steps,
                "rollout": rollout,
                "ppo_epochs": ppo_epochs,
                "batch_size": batch_size,
                "threads": threads,
                "output": str(base),
            })

    if workers == 1:
        for job in train_jobs:
            _run_canonical_job(job)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            list(pool.map(_run_canonical_job, train_jobs))

    eval_jobs = []
    for method in methods:
        for seed in train_seeds:
            train_base = (shared_train_root if shared_train_root is not None else root) / "main" / method / f"seed_{seed}" / "train"
            ckpt = train_base / f"checkpoints/episode_{train_episodes - 1:05d}"
            for wseed in workload_seeds:
                eval_dir = root / "main" / method / f"seed_{seed}" / f"workload_{wseed}"
                eval_jobs.append({
                    "method": method,
                    "training_seed": seed,
                    "workload_seed": wseed,
                    "episodes": eval_episodes,
                    "steps": steps,
                    "rollout": rollout,
                    "ppo_epochs": ppo_epochs,
                    "batch_size": batch_size,
                    "threads": threads,
                    "evaluate_checkpoint": str(ckpt),
                    "output": str(eval_dir),
                })

    if workers == 1:
        for job in eval_jobs:
            _run_canonical_job(job)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            list(pool.map(_run_canonical_job, eval_jobs))

    units = []
    curves = []
    data_by_method: dict[str, list[list[dict[str, float]]]] = {}
    counts_by_method: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
    expected_traces: dict[int, list[str]] = {}

    for method in methods:
        data_by_method[method] = []
        for seed in train_seeds:
            train_base = (shared_train_root if shared_train_root is not None else root) / "main" / method / f"seed_{seed}" / "train"
            for ep_path in sorted((train_base / "episodes").glob("*.json")):
                ep = json.loads(ep_path.read_text())
                curves.append({
                    "variant": "main",
                    "method": method,
                    "training_seed": seed,
                    "episode": ep["episode"],
                    "training_DSR": ep["request_weighted_dsr"],
                    "mean_training_reward": float(np.mean([r["team_reward"] for r in ep["steps"]])),
                })
            row = []
            for wseed in workload_seeds:
                eval_dir = root / "main" / method / f"seed_{seed}" / f"workload_{wseed}"
                metrics, trace, counts = _read_eval_metrics(eval_dir, eval_episodes)
                if wseed in expected_traces and trace != expected_traces[wseed]:
                    raise ValueError(f"Unpaired exogenous evaluation scenarios at workload {wseed}")
                expected_traces[wseed] = trace
                row.append(metrics)
                counts_by_method[method].append(counts)
                units.append({
                    "variant": "main",
                    "method": method,
                    "training_seed": seed,
                    "workload_seed": wseed,
                    **metrics,
                })
            data_by_method[method].append(row)

    for name, rows in [("evaluation_units.csv", units), ("training_curves.csv", curves)]:
        with (root / name).open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    return {
        "units": units,
        "curves": curves,
        "data_by_method": data_by_method,
        "counts_by_method": counts_by_method,
    }


def _bottleneck_snapshot_worker(job: dict) -> dict:
    import torch
    torch.set_num_threads(job["threads"])
    importlib.reload(config)
    refresh_derived_config_fields()
    models = configure_method("lower_no_attention")
    ckpt = Path(job["checkpoint"])
    for name, model in models.items():
        model.load(str(ckpt / name))

    env = Env(scenario_seed=job["workload_seed"])
    obs = np.asarray(env.reset(), dtype=np.float32)
    for model in models.values():
        model.reset()

    sample_steps = set(job["steps"])
    max_step = max(sample_steps)
    records = []
    for step in range(max_step + 1):
        motion, offload, _ = policy_step(models, env, obs, exploration=False)
        motion_clipped = np.clip(motion, -1.0, 1.0)
        if step in sample_steps:
            snapshot = capture(env, motion_clipped)
            env.get_offloading_obs_and_masks()
            quality_mask = env.last_offload_mask_components["quality"]
            hard_mask = env.last_offload_mask_components["hard"]
            learned_assignment = tuple(int(offload[r[0], r[1]]) for r in snapshot.requests)
            quality_candidates = []
            for req, hard_cands in zip(snapshot.requests, snapshot.candidates):
                row, slot = req[0], req[1]
                q_cands = tuple(sorted(set(int(a) for a in np.flatnonzero(quality_mask[row, slot])) | {int(offload[row, slot])}))
                quality_candidates.append(q_cands if q_cands else hard_cands)
            quality_candidates = tuple(quality_candidates)

            learned_eval = evaluate(snapshot, learned_assignment)
            q_res = best_improvement(
                quality_candidates,
                learned_assignment,
                lambda a: objective(evaluate(snapshot, a)),
                tau=0.0,
                max_rounds=2,
            )
            q_eval = evaluate(snapshot, q_res["assignment"])
            h_res = best_improvement(
                snapshot.candidates,
                learned_assignment,
                lambda a: objective(evaluate(snapshot, a)),
                tau=0.0,
                max_rounds=2,
            )
            h_eval = evaluate(snapshot, h_res["assignment"])

            learned_succ = sum(1 for r in learned_eval["requests"] if r["success"])
            q_succ = sum(1 for r in q_eval["requests"] if r["success"])
            h_succ = sum(1 for r in h_eval["requests"] if r["success"])
            admitted = len(snapshot.requests)
            generated = sum(1 for ue in env.ues if ue.current_request.is_service)
            records.append({
                "step": step,
                "generated": float(generated),
                "admitted": int(admitted),
                "uncovered": float(generated - admitted),
                "learned_success": float(learned_succ),
                "learned_admitted_deadline_misses": float(admitted - learned_succ),
                "quality_search_success": float(q_succ),
                "hard_search_success": float(h_succ),
                "quality_improved": bool(q_succ > learned_succ or tuple(q_res["objective"]) < tuple(objective(learned_eval))),
                "quality_termination": q_res["termination"],
                "quality_evaluations": int(q_res["evaluations"]),
                "learned_mbs_offloads": float(sum(1 for r in learned_eval["requests"] if r["target"] == "mbs")),
                "quality_search_mbs_offloads": float(sum(1 for r in q_eval["requests"] if r["target"] == "mbs")),
                "learned_mean_norm_latency": float(objective(learned_eval)[1]),
                "quality_search_mean_norm_latency": float(objective(q_eval)[1]),
            })
        next_obs, _, _ = env.step(motion_clipped, offloading_actions=offload)
        obs = np.asarray(next_obs, dtype=np.float32)

    return {
        "training_seed": job["training_seed"],
        "workload_seed": job["workload_seed"],
        "snapshots": len(records),
        "records": records,
    }


def run_canonical_baseline(args) -> dict:
    root = ROOT / "results" / "ch3_canonical_energy_free"
    train_seeds = CANONICAL_TRAIN_SEEDS[:2] if args.smoke else CANONICAL_TRAIN_SEEDS
    workload_seeds = CANONICAL_WORKLOAD_SEEDS[:2] if args.smoke else CANONICAL_WORKLOAD_SEEDS
    train_episodes = 2 if args.smoke else 200
    eval_episodes = 1 if args.smoke else 6
    steps = 8 if args.smoke else 1000
    rollout = 8 if args.smoke else 1000
    ppo_epochs = 1 if args.smoke else 8
    batch_size = 4 if args.smoke else 256

    methods = ["hierarchical", "lower_no_attention"]
    cohort = _execute_cohort(
        root,
        methods,
        train_seeds,
        workload_seeds,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        steps=steps,
        rollout=rollout,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        workers=args.workers,
        threads=args.threads,
    )
    comparisons = []
    for metric in cohort["data_by_method"]["hierarchical"][0][0]:
        target = np.array([[u[metric] for u in row] for row in cohort["data_by_method"]["hierarchical"]])
        base = np.array([[u[metric] for u in row] for row in cohort["data_by_method"]["lower_no_attention"]])
        comparisons.append({
            "variant": "main",
            "reference": "hierarchical",
            "comparison": "lower_no_attention",
            "metric": metric,
            "hierarchical_mean": float(target.mean()),
            "lower_no_attention_mean": float(base.mean()),
            "sign": "hierarchical minus lower_no_attention",
            **paired_effect(target - base),
        })
    report = {
        "status": "PASS",
        "source_hash": source_manifest()["source_hash"],
        "units": cohort["units"],
        "comparisons": comparisons,
    }
    write_json(root / "analysis.json", report)
    return report


def run_bottleneck_and_pilot(args) -> dict:
    root = ROOT / "results" / "ch3_bottleneck_and_pilot_energy_free"
    train_seeds = PILOT_TRAIN_SEEDS[:2] if args.smoke else PILOT_TRAIN_SEEDS
    workload_seeds = PILOT_WORKLOAD_SEEDS[:2] if args.smoke else PILOT_WORKLOAD_SEEDS
    train_episodes = 2 if args.smoke else 200
    eval_episodes = 1 if args.smoke else 6
    steps = 8 if args.smoke else 1000
    rollout = 8 if args.smoke else 1000
    ppo_epochs = 1 if args.smoke else 8
    batch_size = 4 if args.smoke else 256

    methods = [
        "lower_no_attention",
        "lower_mlp_upper_uncovered_obs",
    ]
    cohort = _execute_cohort(
        root,
        methods,
        train_seeds,
        workload_seeds,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        steps=steps,
        rollout=rollout,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        workers=args.workers,
        threads=args.threads,
    )

    # 1. Compute failure decomposition per method
    arch_methods = {}
    for m in methods:
        unit_rows = [u for u in cohort["units"] if u["method"] == m]
        succ = float(np.mean([u["DSR"] for u in unit_rows]))
        adm = float(np.mean([u["Admission fraction"] for u in unit_rows]))
        unc = 1.0 - adm
        adm_miss = adm - succ
        arch_methods[m] = {
            "units": len(unit_rows),
            "success_fraction": succ,
            "admission_fraction": adm,
            "uncovered_fraction": unc,
            "admitted_deadline_miss_fraction": adm_miss,
            "fraction_of_failures_uncovered": unc / max(1.0 - succ, 1e-12),
        }

    # 2. Run 50 frozen snapshots diagnostic on lower_no_attention
    snap_workloads = workload_seeds[:2] if args.smoke else BOTTLENECK_SNAPSHOT_WORKLOADS
    snap_steps = [0, 4] if args.smoke else BOTTLENECK_SNAPSHOT_STEPS
    snap_jobs = []
    for seed in train_seeds:
        ckpt = root / "main" / "lower_no_attention" / f"seed_{seed}" / "train" / f"checkpoints/episode_{train_episodes - 1:05d}"
        for wseed in snap_workloads:
            snap_jobs.append({
                "training_seed": seed,
                "workload_seed": wseed,
                "checkpoint": str(ckpt),
                "steps": snap_steps,
                "threads": args.threads,
            })

    if args.workers == 1:
        snap_results = [_bottleneck_snapshot_worker(j) for j in snap_jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            snap_results = list(pool.map(_bottleneck_snapshot_worker, snap_jobs))

    all_records = [r for grp in snap_results for r in grp["records"]]
    term_counts: dict[str, int] = {}
    for r in all_records:
        t = r["quality_termination"]
        term_counts[t] = term_counts.get(t, 0) + 1

    gen_tot = float(sum(r["generated"] for r in all_records))
    adm_tot = int(sum(r["admitted"] for r in all_records))
    unc_tot = float(sum(r["uncovered"] for r in all_records))
    l_succ = float(sum(r["learned_success"] for r in all_records))
    q_succ = float(sum(r["quality_search_success"] for r in all_records))
    h_succ = float(sum(r["hard_search_success"] for r in all_records))

    paired_groups = []
    for grp in snap_results:
        recs = grp["records"]
        paired_groups.append({
            "training_seed": grp["training_seed"],
            "workload_seed": grp["workload_seed"],
            "snapshots": grp["snapshots"],
            "generated": float(sum(r["generated"] for r in recs)),
            "learned_success": float(sum(r["learned_success"] for r in recs)),
            "quality_search_success": float(sum(r["quality_search_success"] for r in recs)),
            "hard_search_success": float(sum(r["hard_search_success"] for r in recs)),
        })

    fe = arch_methods["lower_no_attention"]
    src_manifest = source_manifest()
    bottleneck_review = {
        "status": "PASS",
        "scope": "Exploratory, fixed-trajectory, one-slot counterfactual on energy-free trained checkpoints.",
        "source_commit": src_manifest["git_sha"],
        "source_hash": src_manifest["source_hash"],
        "full_evaluation": {
            "unit_count": fe["units"],
            "training_seeds": train_seeds,
            "workload_seeds": workload_seeds,
            "success_fraction": fe["success_fraction"],
            "admission_fraction": fe["admission_fraction"],
            "uncovered_fraction": fe["uncovered_fraction"],
            "admitted_deadline_miss_fraction": fe["admitted_deadline_miss_fraction"],
            "fraction_of_failures_uncovered": fe["fraction_of_failures_uncovered"],
            "interpretation": "Descriptive decomposition of measured generated requests; it does not establish the recoverable improvement from changing trajectory.",
        },
        "snapshot_protocol": {
            "training_seeds": train_seeds,
            "workload_seeds": snap_workloads,
            "episode": 0,
            "steps": snap_steps,
            "snapshots": len(all_records),
            "candidate_sets": ["original quality mask", "hard physical mask"],
            "objective": "frozen Chapter 4 strict lexicographic (energy-free); deadline misses first, normalized latency second",
            "search": "best-improvement from learned assignment, at most two single-request changes; not exact or converged",
        },
        "snapshot_totals": {
            "generated": gen_tot,
            "admitted": adm_tot,
            "uncovered": unc_tot,
            "learned_success": l_succ,
            "learned_admitted_deadline_misses": float(adm_tot - l_succ),
            "quality_search_success": q_succ,
            "quality_search_gain": float(q_succ - l_succ),
            "quality_search_gain_percentage_points": float(100.0 * (q_succ - l_succ) / max(gen_tot, 1.0)),
            "hard_search_success": h_succ,
            "hard_search_gain": float(h_succ - l_succ),
            "quality_search_improved_snapshots": int(sum(1 for r in all_records if r["quality_improved"])),
            "quality_search_termination": term_counts,
            "quality_search_evaluations": int(sum(r["quality_evaluations"] for r in all_records)),
            "learned_mbs_offloads": float(sum(r["learned_mbs_offloads"] for r in all_records)),
            "quality_search_mbs_offloads": float(sum(r["quality_search_mbs_offloads"] for r in all_records)),
        },
        "paired_snapshot_groups": paired_groups,
        "architecture_pilot_crosscheck": {
            "source": str(root / "analysis.json"),
            "source_hash": src_manifest["source_hash"],
            "methods": arch_methods,
        },
    }
    write_json(root / "bottleneck_review.json", bottleneck_review)

    # 3. Pilot comparison: lower_mlp_upper_uncovered_obs vs lower_no_attention
    pilot_comparisons = {}
    for metric in cohort["data_by_method"]["lower_no_attention"][0][0]:
        cand = np.array([[u[metric] for u in row] for row in cohort["data_by_method"]["lower_mlp_upper_uncovered_obs"]])
        base = np.array([[u[metric] for u in row] for row in cohort["data_by_method"]["lower_no_attention"]])
        pilot_comparisons[metric] = {
            "candidate_mean": float(cand.mean()),
            "baseline_mean": float(base.mean()),
            **paired_effect(cand - base),
        }

    pilot_report = {
        "status": "PASS",
        "source_hash": src_manifest["source_hash"],
        "pilot_uncovered_obs_vs_strong_baseline": pilot_comparisons,
    }
    write_json(root / "pilot_analysis.json", pilot_report)
    return {"bottleneck": bottleneck_review, "pilot": pilot_report}


def run_ch3_confirm(args) -> dict:
    root = ROOT / "results" / "ch3_confirm_energy_free"
    train_seeds = CONFIRM_TRAIN_SEEDS[:2] if args.smoke else CONFIRM_TRAIN_SEEDS
    workload_seeds = CONFIRM_WORKLOAD_SEEDS[:2] if args.smoke else CONFIRM_WORKLOAD_SEEDS
    train_episodes = 2 if args.smoke else 200
    eval_episodes = 1 if args.smoke else 6
    steps = 8 if args.smoke else 1000
    rollout = 8 if args.smoke else 1000
    ppo_epochs = 1 if args.smoke else 8
    batch_size = 4 if args.smoke else 256

    methods = ["lower_no_attention", "lower_mlp_upper_uncovered_obs"]
    cohort = _execute_cohort(
        root,
        methods,
        train_seeds,
        workload_seeds,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        steps=steps,
        rollout=rollout,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        workers=args.workers,
        threads=args.threads,
    )

    shutil.copy2(root / "evaluation_units.csv", root / "confirmation_units.csv")
    shutil.copy2(root / "training_curves.csv", root / "confirmation_training.csv")

    metrics_summary = {}
    for metric in cohort["data_by_method"]["lower_no_attention"][0][0]:
        base = np.array([[u[metric] for u in row] for row in cohort["data_by_method"]["lower_no_attention"]])
        cand = np.array([[u[metric] for u in row] for row in cohort["data_by_method"]["lower_mlp_upper_uncovered_obs"]])
        diff_matrix = cand - base
        seed_diffs = diff_matrix.mean(axis=1)
        mean_diff = float(seed_diffs.mean())
        sd_diff = float(seed_diffs.std(ddof=1)) if len(seed_diffs) > 1 else 0.0
        t_val = T95_DF9 if len(seed_diffs) == 10 else 2.0
        ci_half = t_val * sd_diff / math.sqrt(len(seed_diffs)) if len(seed_diffs) > 1 else 0.0
        metrics_summary[metric] = {
            "baseline_mean": float(base.mean()),
            "candidate_mean": float(cand.mean()),
            "mean_difference": mean_diff,
            "training_seed_t_ci95": [mean_diff - ci_half, mean_diff + ci_half],
            "seed_mean_differences": seed_diffs.tolist(),
            "positive_seeds": int(np.sum(seed_diffs > 0)),
            "positive_units": int(np.sum(diff_matrix > 0)),
            "total_units": int(diff_matrix.size),
            **paired_effect(diff_matrix),
        }

    pooled = {}
    for m in methods:
        gen = sum(c["generated"] for c in cohort["counts_by_method"][m])
        succ = sum(c["success"] for c in cohort["counts_by_method"][m])
        adm = sum(c["admitted"] for c in cohort["counts_by_method"][m])
        unc = sum(c["uncovered"] for c in cohort["counts_by_method"][m])
        pooled[m] = {
            "generated_service_count": int(gen),
            "deadline_satisfied_count": int(succ),
            "admitted_service_count": int(adm),
            "uncovered_service_count": int(unc),
            "pooled_dsr": float(succ / gen) if gen else 0.0,
        }

    report = {
        "status": "PASS",
        "source_hash": source_manifest()["source_hash"],
        "training_seeds": train_seeds,
        "workload_seeds": workload_seeds,
        "metrics": metrics_summary,
        "pooled_totals": pooled,
    }
    write_json(root / "confirmation_paired_comparison.json", report)
    return report


def _bridge_worker(job: dict) -> dict:
    import torch
    torch.set_num_threads(job["threads"])
    importlib.reload(config)
    refresh_derived_config_fields()
    models = configure_method("lower_mlp_upper_uncovered_obs")
    ckpt = Path(job["checkpoint"])
    for name, model in models.items():
        model.load(str(ckpt / name))

    env = Env(scenario_seed=job["workload_seed"])
    snap_episodes = job["episodes"]
    snap_steps = set(job["steps"])
    max_step = max(snap_steps)

    per_method_rows: dict[str, list[dict[str, float]]] = {
        "online_lower": [],
        "heuristic_control": [],
        "greedy_single_pass": [],
        "iterative_base": [],
        "mc_itr": [],
    }

    for ep in range(max(snap_episodes) + 1):
        obs = np.asarray(env.reset(), dtype=np.float32)
        for model in models.values():
            model.reset()
        if ep not in snap_episodes:
            continue
        for step in range(max_step + 1):
            t0 = time.perf_counter()
            motion, offload, _ = policy_step(models, env, obs, exploration=False)
            online_wall = time.perf_counter() - t0
            motion_clipped = np.clip(motion, -1.0, 1.0)
            if step in snap_steps:
                snapshot = capture(env, motion_clipped)
                learned_assignment = tuple(int(offload[r[0], r[1]]) for r in snapshot.requests)
                online_eval = evaluate(snapshot, learned_assignment)
                online_obj = objective(online_eval)
                per_method_rows["online_lower"].append({
                    "deadline_misses": float(online_obj[0]),
                    "mean_normalized_latency": float(online_obj[1]),
                    "wall_seconds": float(online_wall),
                })
                solved = solve_snapshot(snapshot, max_rounds=job["max_rounds"], exact_budget=0)
                for m in ("heuristic_control", "greedy_single_pass", "iterative_base", "mc_itr"):
                    obj = solved[m]["objective"]
                    entry = {
                        "deadline_misses": float(obj[0]),
                        "mean_normalized_latency": float(obj[1]),
                        "wall_seconds": float(solved[m]["wall_seconds"]),
                    }
                    if m == "mc_itr":
                        entry["full_path_wall_seconds"] = float(
                            solved["heuristic_control"]["wall_seconds"]
                            + solved["greedy_single_pass"]["wall_seconds"]
                            + solved["mc_itr"]["wall_seconds"]
                        )
                    per_method_rows[m].append(entry)
            next_obs, _, _ = env.step(motion_clipped, offloading_actions=offload)
            obs = np.asarray(next_obs, dtype=np.float32)

    unit_out: dict[str, object] = {
        "training_seed": job["training_seed"],
        "workload_seed": job["workload_seed"],
        "snapshots": len(per_method_rows["online_lower"]),
    }
    for m, rows in per_method_rows.items():
        agg = {
            "deadline_misses": float(np.mean([r["deadline_misses"] for r in rows])),
            "mean_normalized_latency": float(np.mean([r["mean_normalized_latency"] for r in rows])),
            "wall_seconds": float(np.mean([r["wall_seconds"] for r in rows])),
        }
        if m == "mc_itr":
            agg["full_path_wall_seconds"] = float(np.mean([r["full_path_wall_seconds"] for r in rows]))
        unit_out[m] = agg
    return unit_out


def run_ch4_bridge(args) -> dict:
    confirm_root = ROOT / "results" / "ch3_confirm_energy_free"
    out_root = ROOT / "results" / "ch4_bridge_energy_free"
    out_root.mkdir(parents=True, exist_ok=True)

    train_seeds = CONFIRM_TRAIN_SEEDS[:2] if args.smoke else CONFIRM_TRAIN_SEEDS
    workload_seeds = BRIDGE_WORKLOAD_SEEDS[:2] if args.smoke else BRIDGE_WORKLOAD_SEEDS
    train_episodes = 2 if args.smoke else 200
    snap_episodes = [0] if args.smoke else BRIDGE_SNAPSHOT_EPISODES
    snap_steps = [0, 4] if args.smoke else BRIDGE_SNAPSHOT_STEPS
    max_rounds = 20 if args.smoke else 1000

    jobs = []
    for seed in train_seeds:
        ckpt = confirm_root / "main" / "lower_mlp_upper_uncovered_obs" / f"seed_{seed}" / "train" / f"checkpoints/episode_{train_episodes - 1:05d}"
        for wseed in workload_seeds:
            jobs.append({
                "training_seed": seed,
                "workload_seed": wseed,
                "checkpoint": str(ckpt),
                "episodes": snap_episodes,
                "steps": snap_steps,
                "max_rounds": max_rounds,
                "threads": args.threads,
            })

    if args.workers == 1:
        units = [_bridge_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            units = list(pool.map(_bridge_worker, jobs))

    units.sort(key=lambda u: (int(u["training_seed"]), int(u["workload_seed"])))
    seed_diffs = []
    for seed in train_seeds:
        seed_units = [u for u in units if int(u["training_seed"]) == seed]
        diffs = [float(u["mc_itr"]["deadline_misses"]) - float(u["online_lower"]["deadline_misses"]) for u in seed_units]
        seed_diffs.append(float(np.mean(diffs)))

    mean_diff = float(np.mean(seed_diffs))
    sd_diff = float(np.std(seed_diffs, ddof=1)) if len(seed_diffs) > 1 else 0.0
    t_val = T95_DF9 if len(seed_diffs) == 10 else 2.0
    ci_half = t_val * sd_diff / math.sqrt(len(seed_diffs)) if len(seed_diffs) > 1 else 0.0

    report = {
        "status": "PASS",
        "study": "chapter4_chapter5_bridge_energy_free",
        "source_hash": source_manifest()["source_hash"],
        "statistical_unit": "training seed after averaging paired workload means",
        "training_seeds": train_seeds,
        "workload_seeds": workload_seeds,
        "evaluation_jobs": len(units),
        "snapshots": sum(int(u["snapshots"]) for u in units),
        "units": units,
        "mc_itr_minus_online_lower_deadline_misses": {
            "mean_difference": mean_diff,
            "training_seed_t_ci95": [mean_diff - ci_half, mean_diff + ci_half],
            "seed_mean_differences": seed_diffs,
        },
        "interpretation_boundary": "Frozen counterfactuals on candidate-policy trajectories, not online integration or long-run DSR.",
    }
    write_json(out_root / "bridge_analysis.json", report)
    return report


def run_ch4_suite(args) -> dict:
    from run_chapter4_suite import plan as ch4_plan, run as ch4_run
    spec = ch4_plan("all", smoke=args.smoke)
    spec["purpose"] = "pilot"
    out_root = ROOT / "results" / "chapter4_energy_free"
    return ch4_run(spec, out_root, workers=args.workers, threads=args.threads)


def export_figure_data() -> None:
    target = ROOT / "docs" / "figures" / "data" / "chapter34_20261004"
    target.mkdir(parents=True, exist_ok=True)
    copies = [
        (ROOT / "results" / "ch3_bottleneck_and_pilot_energy_free" / "bottleneck_review.json", target / "bottleneck_review.json"),
        (ROOT / "results" / "ch3_bottleneck_and_pilot_energy_free" / "pilot_analysis.json", target / "pilot_analysis.json"),
        (ROOT / "results" / "ch3_canonical_energy_free" / "analysis.json", target / "canonical_baseline_analysis.json"),
        (ROOT / "results" / "ch3_confirm_energy_free" / "confirmation_units.csv", target / "confirmation_units.csv"),
        (ROOT / "results" / "ch3_confirm_energy_free" / "confirmation_training.csv", target / "confirmation_training.csv"),
        (ROOT / "results" / "ch3_confirm_energy_free" / "confirmation_paired_comparison.json", target / "confirmation_paired_comparison.json"),
        (ROOT / "results" / "chapter4_energy_free" / "analysis.json", target / "chapter4_analysis.json"),
        (ROOT / "results" / "chapter4_energy_free" / "evaluation_units.csv", target / "chapter4_units.csv"),
        (ROOT / "results" / "ch4_bridge_energy_free" / "bridge_analysis.json", target / "bridge_analysis.json"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("Run Chapter 3/4 experiments on the remote Linux server")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["canonical_baseline", "bottleneck_and_pilot", "ch3_confirm", "ch4_suite", "ch4_bridge", "export_figure_data", "all"],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.stage in ("canonical_baseline", "all"):
        run_canonical_baseline(args)
    if args.stage in ("bottleneck_and_pilot", "all"):
        run_bottleneck_and_pilot(args)
    if args.stage in ("ch3_confirm", "all"):
        run_ch3_confirm(args)
    if args.stage in ("ch4_suite", "all"):
        run_ch4_suite(args)
    if args.stage in ("ch4_bridge", "all"):
        run_ch4_bridge(args)
    if args.stage in ("export_figure_data", "all"):
        export_figure_data()


if __name__ == "__main__":
    main()
