"""Supplement two frozen energy-free confirmation episodes with executed spatial states.

Run on the experiment server against the frozen canonical_energy_free_v1 confirmation
checkpoints and archived evaluation logs. Original source and results remain read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import platform
from pathlib import Path
import sys


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-seed", type=int, default=7541)
    p.add_argument("--workload-seed", type=int, default=2203)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--snapshot-step", type=int, default=499)
    args = p.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("Scientific replay runs on the experiment server")
    if args.output.exists():
        raise FileExistsError("Use a new output directory; do not overwrite evidence")
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import torch
    import config
    import run_canonical_experiment as runner
    from environment.env import Env
    from utils.canonical_experiment import config_snapshot, digest, exogenous_frame, source_manifest
    from utils.logger import refresh_derived_config_fields

    source_hash = source_manifest()["source_hash"]
    args.output.mkdir(parents=True)
    methods = ("lower_no_attention", "lower_mlp_upper_uncovered_obs")
    arrays: dict[str, object] = {}
    summaries: dict[str, dict[str, object]] = {}
    step_rows: list[dict[str, object]] = []
    shared_positions = shared_initial = shared_hotspots = None

    for method in methods:
        root = args.results / "main" / method / f"seed_{args.training_seed}"
        eval_root = root / f"workload_{args.workload_seed}"
        last_inv = read(eval_root / "last_invocation.json")
        if last_inv.get("status") != "PASS":
            raise ValueError(f"Evaluation unit did not pass: {eval_root}")
        episode_path = eval_root / "episodes" / f"{args.episode:05d}.json"
        archived = read(episode_path)
        manifest = read(eval_root / "manifest.json")
        settings = manifest["settings"]
        assert settings["training_seed"] == args.training_seed and settings["workload_seed"] == args.workload_seed
        assert manifest["config"].get("ENV_SEMANTICS_VERSION") == "canonical_energy_free_v1"

        train_root = root / "train"
        train_inv = read(train_root / "last_invocation.json")
        if train_inv.get("status") != "PASS":
            raise ValueError(f"Training unit did not pass: {train_root}")
        checkpoint = train_root / read(train_root / "latest.json")["checkpoint"]
        contract = read(checkpoint / "contract.json")
        for relative, expected in contract["files"].items():
            if sha(checkpoint / relative) != expected:
                raise ValueError("Checkpoint receipt mismatch: " + relative)

        torch.set_num_threads(int(settings["threads"]))
        importlib.reload(config)
        config.NUM_UES = settings["ue_count"]
        config.UAV_COMPUTING_CAPACITY = config.UAV_COMPUTING_CAPACITY * settings["compute_scale"]
        refresh_derived_config_fields()
        np.random.seed(args.training_seed)
        torch.manual_seed(args.training_seed)
        config.PPO_EPOCHS = settings["ppo_epochs"]
        config.PPO_BATCH_SIZE = settings["batch_size"]
        config.PPO_ROLLOUT_LENGTH = settings["rollout"]
        models = runner.configure_method(method)
        snapshot = config_snapshot()
        if snapshot != manifest["config"] or digest(snapshot) != contract["config_hash"]:
            raise ValueError("Replay configuration differs")
        assert contract["source_hash"] == source_hash and contract["method"] == method
        for name, model in models.items():
            model.load(str(checkpoint / name))
            model.reset()

        env = Env(scenario_seed=args.workload_seed)
        for _ in range(args.episode):
            env.reset()
        obs = np.asarray(env.reset(), dtype=np.float32)
        initial = np.asarray([u.pos[:2].copy() for u in env.uavs])
        hotspots = np.asarray(env._population["hotspot_centers"])[:, :2]
        positions = [initial]
        users, services, admissions = [], [], []
        counts = dict(generated=0, admitted=0, successful=0, hotspot_generated=0, hotspot_admitted=0)
        trace = hashlib.sha256()
        max_error, exact_metrics, metric_count = 0.0, 0, 0
        min_sep, max_speed = math.inf, 0.0

        for step in range(1000):
            trace.update(digest(exogenous_frame(env)).encode())
            before = np.asarray([u.pos[:2].copy() for u in env.uavs])
            xy = np.asarray([u.pos[:2].copy() for u in env.ues])
            service = np.asarray([bool(u.current_request.is_service) for u in env.ues])
            admitted = np.asarray([bool(u.assigned) for u in env.ues]) & service
            covered = (
                np.linalg.norm(xy[:, None, :] - before[None, :, :], axis=2)
                <= config.UAV_COVERAGE_RADIUS
            ).any(axis=1)
            if not np.array_equal(admitted, service & covered):
                raise ValueError("Admission does not match slot-start physical geometry")
            hot = (
                np.linalg.norm(xy[:, None, :] - hotspots[None, :, :], axis=2)
                <= config.HOTSPOT_RADIUS
            ).any(axis=1)
            motion, offload, _ = runner.policy_step(models, env, obs, exploration=False)
            next_obs, rewards, metrics = env.step(np.clip(motion, -1, 1), offloading_actions=offload)
            saved = archived["steps"][step]["metrics"]
            if set(metrics) != set(saved):
                raise ValueError("Metric key mismatch")
            for key, value in metrics.items():
                error = abs(value - saved[key])
                max_error = max(max_error, error)
                metric_count += 1
                exact_metrics += value == saved[key]
                if not math.isclose(value, saved[key], rel_tol=1e-11, abs_tol=1e-7):
                    raise ValueError(f"Metric replay mismatch {method}:{step}:{key}: {value} != {saved[key]}")
            if (
                int(service.sum()) != metrics["generated_service_count"]
                or int(admitted.sum()) != metrics["admitted_service_count"]
            ):
                raise ValueError("Per-request admission accounting mismatch")
            after = np.asarray([u.pos[:2].copy() for u in env.uavs])
            dist = np.linalg.norm(after - before, axis=1)
            pair_dist = np.linalg.norm(after[:, None, :] - after[None, :, :], axis=2)
            min_sep = min(min_sep, float(pair_dist[np.triu_indices(5, 1)].min()))
            max_speed = max(max_speed, float(dist.max()))
            if after.min() < 50 - 1e-3 or after.max() > 650 + 1e-3 or dist.max() > 15 + 1e-3 or min_sep < 200 - 1e-3:
                raise ValueError("Executed trajectory fails physical constraints")
            positions.append(after)
            users.append(xy)
            services.append(service)
            admissions.append(admitted)
            values = dict(
                generated=int(service.sum()),
                admitted=int(admitted.sum()),
                successful=int(metrics["deadline_satisfied_count"]),
                hotspot_generated=int((hot & service).sum()),
                hotspot_admitted=int((hot & admitted).sum()),
            )
            for key, value in values.items():
                counts[key] += value
            step_rows.append(
                dict(
                    method=method,
                    step=step,
                    **values,
                    fleet_path_m=float(dist.sum()),
                )
            )
            obs = np.asarray(next_obs, dtype=np.float32)
            if step % 250 == 0:
                print(method, step, "verified", flush=True)

        if trace.hexdigest() != archived["exogenous_trace_hash"]:
            raise ValueError("External workload trace differs")
        if (
            counts["generated"] != archived["generated_service_count"]
            or counts["successful"] != archived["deadline_satisfied_count"]
        ):
            raise ValueError("Episode totals differ")
        for relative, expected in contract["files"].items():
            if sha(checkpoint / relative) != expected:
                raise ValueError("Checkpoint changed during replay")

        users_arr = np.asarray(users)
        if shared_positions is None:
            shared_positions, shared_initial, shared_hotspots = users_arr, initial, hotspots
            arrays["ue_positions"] = users_arr
            arrays["hotspot_centers"] = hotspots
        elif not (
            np.array_equal(users_arr, shared_positions)
            and np.array_equal(initial, shared_initial)
            and np.array_equal(hotspots, shared_hotspots)
        ):
            raise ValueError("Paired exogenous positions/initial deployment differ")

        positions_arr = np.asarray(positions)
        path = np.linalg.norm(np.diff(positions_arr, axis=0), axis=2).sum(axis=0)
        arrays[method + "_uav_positions"] = positions_arr
        arrays[method + "_service"] = np.asarray(services)
        arrays[method + "_admitted"] = np.asarray(admissions)
        summaries[method] = dict(
            **counts,
            global_admission=counts["admitted"] / counts["generated"],
            hotspot_admission=counts["hotspot_admitted"] / counts["hotspot_generated"],
            conditional_satisfaction=counts["successful"] / counts["admitted"],
            dsr=counts["successful"] / counts["generated"],
            mean_uav_path_m=float(path.mean()),
            per_uav_path_m=path.tolist(),
            archived_episode_sha256=sha(episode_path),
            checkpoint_contract_sha256=sha(checkpoint / "contract.json"),
            checkpoint_files=contract["files"],
            exogenous_trace_hash=trace.hexdigest(),
            metrics_checked=metric_count,
            metrics_exactly_equal=int(exact_metrics),
            max_metric_absolute_error=max_error,
            min_pair_separation_m=min_sep,
            max_slot_displacement_m=max_speed,
            verified_steps=1000,
        )

    if source_manifest()["source_hash"] != source_hash:
        raise ValueError("Original source changed during replay")
    np.savez_compressed(args.output / "paired_replay.npz", **arrays)
    with (args.output / "step_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0]))
        writer.writeheader()
        writer.writerows(step_rows)
    result = dict(
        status="PASS",
        source_hash=source_hash,
        protocol="canonical_energy_free_v1_upper_uncovered_observation_confirmation",
        training_seed=args.training_seed,
        workload_seed=args.workload_seed,
        episode=args.episode,
        snapshot_step_zero_based=args.snapshot_step,
        displayed_slot=args.snapshot_step + 1,
        selection=(
            "Median-effect confirmation training seed (7541, 5th of 10 seeds sorted by paired DSR gain, "
            "closest to the 10-seed mean +2.85 pp), first confirmation workload (2203), first evaluation episode (0), "
            "midpoint slot 500 (0-based step 499)."
        ),
        paired_initial_and_external_positions_identical=True,
        methods=summaries,
        coverage_radius_m=100,
        hotspot_radius_m=100,
        source_data_sha256=sha(args.output / "paired_replay.npz"),
        independent_replicates=1,
        inference="Fixed illustrative pair only; no new population inference.",
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        replay_script_sha256=sha(Path(__file__)),
    )
    (args.output / "replay_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
