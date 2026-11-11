"""Isolated canonical training/evaluation with episode-boundary resume.

All execution belongs on the remote Linux experiment server. Existing runners
remain available for historical compatibility; new formal outputs use this one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import importlib

import numpy as np
import torch
import config
from environment.env import Env
from marl_models.buffer_and_helpers import AttentionRolloutBuffer, DiscreteOffloadRolloutBuffer, JointMAPPOBuffer
from marl_models.utils import get_model
from run_hierarchical_mappo_experiment import apply_lower_ablation, lower_rewards_from_metrics, update_on_policy_model
from utils.canonical_experiment import config_snapshot, digest, exogenous_frame, scenario_manifest, source_manifest, write_json

METHODS = ("upper_attention", "upper_vanilla", "hierarchical", "no_attention", "no_quality_mask", "no_penalty", "joint", "vanilla_hierarchical", "lower_no_attention", "lower_actor_attention_only", "lower_critic_attention_only", "lower_mlp_upper_dsr_w5", "lower_mlp_upper_uncovered_obs", "ippo",
           "greedy", "local", "mbs", "random", "uniform")
NONLEARNED = {"greedy", "local", "mbs", "random", "uniform"}
REWARD_PILOT_OVERRIDES = {
    "lower_mlp_upper_dsr_w5": {"REWARD_W_DSR": 5.0},
    "lower_mlp_upper_uncovered_obs": {"UPPER_UE_OBSERVATION_MODE": "covered_plus_sensed_uncovered"},
}


def configure_method(method):
    config.UPPER_UE_OBSERVATION_MODE = "covered_only"
    for key, value in REWARD_PILOT_OVERRIDES.get(method, {}).items():
        setattr(config, key, value)
    ablation = {"no_attention": "no_attention", "lower_no_attention": "no_attention",
                "lower_mlp_upper_dsr_w5": "no_attention", "lower_mlp_upper_uncovered_obs": "no_attention",
                "lower_actor_attention_only": "actor_only", "lower_critic_attention_only": "critic_only",
                "no_quality_mask": "no_mask", "no_penalty": "no_lagrange"}.get(method, "full")
    lower_name = apply_lower_ablation(ablation)
    if method in NONLEARNED:
        name = {"random": "random_baseline", "uniform": "uniform_baseline"}.get(method, "uncoordinated_greedy")
        return {"baseline": get_model(name)}
    if method == "joint":
        return {"joint": get_model("joint_mappo")}
    upper_name = "vanilla_mappo" if method in ("upper_vanilla", "no_attention", "vanilla_hierarchical") else "attention_mappo"
    if method == "ippo":
        upper_name = "ippo_baseline"
    models = {"upper": get_model(upper_name)}
    if not method.startswith("upper_"):
        models["lower"] = get_model(lower_name)
    return models


def buffers_for(models, length):
    args = dict(num_agents=config.NUM_UAVS, buffer_size=length, device=next(iter(models.values())).device)
    if "joint" in models:
        return {"joint": JointMAPPOBuffer(joint_obs_dim=config.OBS_DIM_SINGLE + config.OFFLOAD_OBS_DIM_SINGLE,
                    trajectory_action_dim=config.ACTION_DIM, max_requests=config.MAX_OFFLOAD_REQUESTS_PER_UAV,
                    num_offload_actions=config.OFFLOAD_NUM_ACTIONS, **args)}
    buffers = {"upper": AttentionRolloutBuffer(obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, **args)}
    if "lower" in models:
        buffers["lower"] = DiscreteOffloadRolloutBuffer(obs_dim=config.OFFLOAD_OBS_DIM_SINGLE,
                    max_requests=config.MAX_OFFLOAD_REQUESTS_PER_UAV, num_actions=config.OFFLOAD_NUM_ACTIONS, **args)
    return buffers


def policy_step(models, env, obs, exploration):
    lower_obs = masks = None
    if "baseline" in models:
        model = models["baseline"]
        motion = model.select_actions(obs, exploration=False)
        offload = None
        if hasattr(model, "get_offload_action"):
            lower_obs, masks = env.get_offloading_obs_and_masks()
            offload = model.get_offload_action(lower_obs, masks)
        return motion, offload, {}
    if "lower" in models or "joint" in models:
        lower_obs, masks = env.get_offloading_obs_and_masks()
    if "joint" in models:
        combined = np.concatenate((obs, lower_obs), axis=-1)
        motion, offload, lp, values = models["joint"].get_action_and_value(combined, masks=masks, exploration=exploration)
        return motion, offload, {"joint": (combined, motion, offload, masks, lp, values)}
    if exploration:
        motion, lp, values = models["upper"].get_action_and_value(obs, obs.flatten())
    else:
        motion = models["upper"].select_actions(obs, exploration=False)
        lp = values = np.zeros(config.NUM_UAVS, dtype=np.float32)
    entries = {"upper": (obs.flatten(), obs, motion, lp, values)}
    offload = None
    if "lower" in models:
        offload, lp, values = models["lower"].get_action_and_value(lower_obs, masks=masks, exploration=exploration)
        entries["lower"] = (lower_obs, offload, masks, lp, values)
    return motion, offload, entries


def train(args):
    if platform.system() != "Linux":
        raise RuntimeError("Run canonical experiments on the remote Linux server")
    if min(args.episodes, args.steps, args.rollout, args.ppo_epochs, args.batch_size, args.threads, args.checkpoint_every) < 1:
        raise ValueError("Experiment sizes must be positive")
    if min(args.training_seed, args.workload_seed) < 0:
        raise ValueError("Seeds must be nonnegative")
    importlib.reload(config)
    config.NUM_UES = args.ue_count
    config.UAV_COMPUTING_CAPACITY = config.UAV_COMPUTING_CAPACITY * args.compute_scale
    if args.ue_count < 1 or not np.isfinite(args.compute_scale) or args.compute_scale <= 0:
        raise ValueError("Invalid sensitivity settings")
    from utils.logger import refresh_derived_config_fields
    refresh_derived_config_fields()
    torch.set_num_threads(args.threads)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)
    config.PPO_EPOCHS = args.ppo_epochs
    config.PPO_BATCH_SIZE = args.batch_size
    config.PPO_ROLLOUT_LENGTH = args.rollout
    models = configure_method(args.method)
    snapshot, source = config_snapshot(), source_manifest()
    scenario = scenario_manifest(args.workload_seed)
    settings = {k: getattr(args, k) for k in ("method", "training_seed", "workload_seed", "steps", "rollout", "ppo_epochs", "batch_size", "threads", "purpose", "ue_count", "compute_scale", "checkpoint_every")}
    if args.purpose == "formal":
        from utils.formal_protocol import verify_release
        lock = verify_release(args.source_lock, args.validation_report, source)
        # Release directories have no .git; retain the validated origin commit.
        source.update({key: lock.get(key) for key in ('git_sha', 'branch', 'git_clean')})
    output = Path(args.output).resolve()
    evaluation = args.evaluate_checkpoint is not None or args.method in NONLEARNED
    if args.resume and evaluation:
        raise ValueError("Evaluation and resume cannot be combined")
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["settings"] != settings or manifest["config"] != snapshot or manifest["source"]["source_hash"] != source["source_hash"]:
            raise ValueError("Resume source, settings or configuration changed")
        checkpoint = output / json.loads((output / "latest.json").read_text())["checkpoint"]
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Refusing to overwrite an existing experiment directory")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"settings": settings, "config": snapshot, "source": source, "scenario": scenario,
                    "purpose": args.purpose, "evaluation": evaluation, "created_unix": time.time(),
                    "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                                "device": str(next(iter(models.values())).device)}}
        write_json(output / "manifest.json", manifest)
        checkpoint = Path(args.evaluate_checkpoint).resolve() if args.evaluate_checkpoint else None
    start_episode = 0
    rho_dsr = rho_mbs = 0.0
    if checkpoint is not None:
        contract = json.loads((checkpoint / "contract.json").read_text())
        if (contract["method"] != args.method or contract["config_hash"] != digest(snapshot)
                or contract["source_hash"] != source["source_hash"]):
            raise ValueError("Checkpoint contract is incompatible")
        for relative, expected in contract["files"].items():
            if hashlib.sha256((checkpoint / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Checkpoint checksum mismatch")
        for name, model in models.items():
            model.load(str(checkpoint / name))
        state = torch.load(checkpoint / "training_state.pt", weights_only=True, map_location="cpu")
        rho_dsr, rho_mbs = state["rho_dsr"], state["rho_mbs"]
        if args.resume:
            start_episode = state["next_episode"]
            torch.set_rng_state(state["torch_rng"])
            if torch.cuda.is_available() and state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
    if start_episode >= args.episodes:
        raise ValueError("Requested total episodes already completed")
    env = Env(scenario_seed=args.workload_seed)
    # Layout RNG is only consumed at resets; UE workload/mobility streams are
    # separately keyed by episode and UE, so skipped episodes need no simulation.
    for _ in range(start_episode):
        env.reset()
    completed = []
    for episode in range(start_episode, args.episodes):
        obs = np.asarray(env.reset(), dtype=np.float32)
        for model in models.values():
            model.reset()
        rows, losses = [], {}
        trace = hashlib.sha256()
        for offset in range(0, args.steps, args.rollout):
            length = min(args.rollout, args.steps - offset)
            buffers = buffers_for(models, length) if not evaluation else {}
            violations = []
            for index in range(length):
                trace.update(digest(exogenous_frame(env)).encode())
                motion, offload, entries = policy_step(models, env, obs, exploration=not evaluation)
                if args.method in {"local", "mbs"}:
                    offload = np.full((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV),
                                      0 if args.method == "local" else 1, dtype=np.int64)
                next_obs, rewards, metrics = env.step(np.clip(motion, -1, 1), offloading_actions=offload)
                if not np.isfinite(np.asarray(next_obs)).all() or not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError("Non-finite environment output")
                if offload is not None and args.method not in NONLEARNED and (metrics["service_fallback_count"] or metrics["service_heuristic_decision_count"]):
                    raise RuntimeError("Canonical learned actions unexpectedly fell back")
                lower_rewards, diag = lower_rewards_from_metrics(metrics, rewards, rho_dsr, rho_mbs)
                violations.append(diag)
                done = offset + index + 1 == args.steps
                if not evaluation:
                    for name, entry in entries.items():
                        buffers[name].add(*entry[:-1], lower_rewards if name == "lower" else rewards, done, entry[-1])
                rows.append({"step": offset + index, "metrics": metrics, "team_reward": float(np.mean(rewards)),
                             "lower_reward": lower_rewards[0], "rho_dsr": rho_dsr, "rho_mbs": rho_mbs})
                obs = np.asarray(next_obs, dtype=np.float32)
            if not evaluation:
                _, _, bootstrap = policy_step(models, env, obs, exploration=True)
                for name, buffer in buffers.items():
                    values = np.zeros(config.NUM_UAVS, dtype=np.float32) if done else bootstrap[name][-1]
                    losses[name] = update_on_policy_model(models[name], buffer, values)
                    if not all(np.isfinite(v) for v in losses[name].values()):
                        raise FloatingPointError("Non-finite policy loss")
                if "lower" in models and config.OFFLOAD_CONSTRAINT_MODE != "none":
                    rho_dsr = float(np.clip(rho_dsr + config.OFFLOAD_LAGRANGE_LR * np.mean([d["dsr_violation"] for d in violations]), 0, config.OFFLOAD_LAGRANGE_MAX))
                    rho_mbs = float(np.clip(rho_mbs + config.OFFLOAD_LAGRANGE_LR * np.mean([d["mbs_load_violation"] for d in violations]), 0, config.OFFLOAD_LAGRANGE_MAX))
            trace_hex = trace.hexdigest()
        generated = sum(r["metrics"]["generated_service_count"] for r in rows)
        successful = sum(r["metrics"]["deadline_satisfied_count"] for r in rows)
        result = {"episode": episode, "scenario_hash": scenario["scenario_hash"], "exogenous_trace_hash": trace_hex,
                  "request_weighted_dsr": successful / generated if generated else 0.,
                  "generated_service_count": generated, "deadline_satisfied_count": successful,
                  "steps": rows, "final_losses": losses}
        write_json(output / "episodes" / f"{episode:05d}.json", result)
        if not evaluation:
            target = output / "checkpoints" / f"episode_{episode:05d}"
            for name, model in models.items():
                (target / name).mkdir(parents=True, exist_ok=True)
                model.save(str(target / name))
            torch.save({"next_episode": episode + 1, "rho_dsr": rho_dsr, "rho_mbs": rho_mbs,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, target / "training_state.pt")
            files = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob("*.p*") if p.is_file()}
            write_json(target / "contract.json", {"method": args.method, "config_hash": digest(snapshot),
                        "source_hash": source["source_hash"], "files": files})
            write_json(output / "latest.json", {"checkpoint": str(target.relative_to(output)), "completed_episodes": episode + 1})
            previous = output / "checkpoints" / f"episode_{episode - 1:05d}"
            if episode > 0 and episode % args.checkpoint_every != 0 and previous.exists():
                import shutil
                shutil.rmtree(previous)

        completed.append({k: v for k, v in result.items() if k != "steps"})
    write_json(output / "last_invocation.json", {"status": "PASS", "completed": completed, "evaluation": evaluation})
    return {"status": "PASS", "output": str(output), "completed_episodes": len(completed), "evaluation": evaluation}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--workload-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True, help="Total episodes, including previously completed episodes on resume")
    parser.add_argument("--steps", type=int, default=config.STEPS_PER_EPISODE)
    parser.add_argument("--rollout", type=int, default=config.PPO_ROLLOUT_LENGTH)
    parser.add_argument("--ppo-epochs", type=int, default=config.PPO_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.PPO_BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--purpose", choices=("verification", "pilot", "formal"), default="pilot")
    parser.add_argument("--validation-report")
    parser.add_argument("--source-lock")
    parser.add_argument("--ue-count", type=int, default=100)
    parser.add_argument("--compute-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-checkpoint")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
