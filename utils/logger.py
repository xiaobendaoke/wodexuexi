import config as default_config
import json
import os
import numpy as np


def refresh_derived_config_fields() -> None:
    """Recompute derived config fields after loading a config file.

    Older saved configs may carry stale observation dimensions from before the
    request-structure change. Refreshing here keeps runtime dimensions aligned
    with the current environment implementation while preserving base settings.
    """
    default_config.NUM_FILES = default_config.NUM_SERVICES + default_config.NUM_CONTENTS
    default_config.AVG_FILE_SIZE = float(np.mean(default_config.FILE_SIZES))
    default_config.MAX_UAV_NEIGHBORS = max(0, default_config.NUM_UAVS - 1)
    default_config.MAX_ASSOCIATED_UES = min(30, default_config.NUM_UES // default_config.NUM_UAVS + 10)
    default_config.SELF_OBS_DIM = 2 + default_config.NUM_FILES
    default_config.REQUEST_OBS_DIM = 5
    default_config.UE_OBS_DIM = 2 + default_config.REQUEST_OBS_DIM + 1
    default_config.OBS_DIM_SINGLE = (
        default_config.SELF_OBS_DIM
        + (default_config.MAX_UAV_NEIGHBORS * default_config.NEIGHBOR_OBS_DIM)
        + (default_config.MAX_ASSOCIATED_UES * default_config.UE_OBS_DIM)
    )


class Log:
    def __init__(self) -> None:
        self.rewards: list[float] = []
        self.latencies: list[float] = []
        self.energies: list[float] = []
        self.fairness_scores: list[float] = []
        self.offline_rates: list[float] = []
        self.deadline_satisfaction_rates: list[float] = []
        self.offloading_ratio_local: list[float] = []
        self.offloading_ratio_cooperative: list[float] = []
        self.offloading_ratio_mbs: list[float] = []
        self.mbs_load_ratios: list[float] = []
        self.service_learned_decision_counts: list[float] = []
        self.service_heuristic_decision_counts: list[float] = []
        self.service_fallback_counts: list[float] = []
        self.service_predict_exception_fallback_counts: list[float] = []
        self.service_offload_policy_requested: list[str] = []
        self.service_offload_policy_loaded: list[bool] = []
        self.service_offload_policy_checkpoint_path: list[str | None] = []
        self.service_offload_policy_feature_family: list[str | None] = []
        # Training losses (optional, may be empty for baselines)
        self.actor_losses: list[float | None] = []
        self.critic_losses: list[float | None] = []
        self.entropy_losses: list[float | None] = []
        self.alpha_losses: list[float | None] = []

    def append(
        self,
        reward: float,
        latency: float,
        energy: float,
        fairness: float,
        offline_rate: float,
        deadline_satisfaction_rate: float = 0.0,
        offloading_ratio_local: float = 0.0,
        offloading_ratio_cooperative: float = 0.0,
        offloading_ratio_mbs: float = 0.0,
        mbs_load_ratio: float = 0.0,
        *,
        service_learned_decision_count: float = 0.0,
        service_heuristic_decision_count: float = 0.0,
        service_fallback_count: float = 0.0,
        service_predict_exception_fallback_count: float = 0.0,
        service_offload_policy_requested: str = "heuristic",
        service_offload_policy_loaded: bool = False,
        service_offload_policy_checkpoint_path: str | None = None,
        service_offload_policy_feature_family: str | None = None,
        actor_loss: float | None = None,
        critic_loss: float | None = None,
        entropy_loss: float | None = None,
        alpha_loss: float | None = None,
    ) -> None:
        """Append metrics for one logging interval. Loss fields are optional and can be None.

        Parameters are averaged/aggregated externally by training loop before being passed here.
        The new request-level metrics are currently episode means of per-step values,
        not strict count-weighted episode totals.
        """
        self.rewards.append(reward)
        self.latencies.append(latency)
        self.energies.append(energy)
        self.fairness_scores.append(fairness)
        self.offline_rates.append(offline_rate)
        self.deadline_satisfaction_rates.append(deadline_satisfaction_rate)
        self.offloading_ratio_local.append(offloading_ratio_local)
        self.offloading_ratio_cooperative.append(offloading_ratio_cooperative)
        self.offloading_ratio_mbs.append(offloading_ratio_mbs)
        self.mbs_load_ratios.append(mbs_load_ratio)
        self.service_learned_decision_counts.append(service_learned_decision_count)
        self.service_heuristic_decision_counts.append(service_heuristic_decision_count)
        self.service_fallback_counts.append(service_fallback_count)
        self.service_predict_exception_fallback_counts.append(service_predict_exception_fallback_count)
        self.service_offload_policy_requested.append(service_offload_policy_requested)
        self.service_offload_policy_loaded.append(service_offload_policy_loaded)
        self.service_offload_policy_checkpoint_path.append(service_offload_policy_checkpoint_path)
        self.service_offload_policy_feature_family.append(service_offload_policy_feature_family)

        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.entropy_losses.append(entropy_loss)
        self.alpha_losses.append(alpha_loss)


class Logger:
    def __init__(self, log_dir: str, timestamp: str) -> None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.timestamp: str = timestamp
        self.log_dir: str = log_dir
        self.log_file_path: str = os.path.join(self.log_dir, f"logs_{timestamp}.txt")
        self.json_file_path: str = os.path.join(self.log_dir, f"log_data_{timestamp}.json")
        self.config_file_path: str = os.path.join(self.log_dir, f"config_{timestamp}.json")

    def log_configs(self) -> None:
        config_dict: dict = {key: getattr(default_config, key) for key in dir(default_config) if key.isupper() and not key.startswith("__") and not callable(getattr(default_config, key))}

        # Custom serializer for numpy arrays
        def numpy_encoder(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(self.config_file_path, "w") as f:
            json.dump(config_dict, f, indent=4, default=numpy_encoder)
        print(f"Configs saved to {self.config_file_path}")

    def log_metrics(
        self,
        progress_step: int,
        log: Log,
        log_freq: int,
        elapsed_time: float,
        losses: dict | None = None,
    ) -> None:
        """Log aggregated metrics to text and JSON files.

        `losses` is optional and may contain keys like 'actor', 'critic', 'entropy', 'alpha'. If present,
        they will be included in the saved logs. The training loops should pass averaged loss values for the
        logging interval (matching `log_freq`) when available.
        """
        rewards_slice: np.ndarray = np.array(log.rewards[-log_freq:])
        latencies_slice: np.ndarray = np.array(log.latencies[-log_freq:])
        energies_slice: np.ndarray = np.array(log.energies[-log_freq:])
        fairness_slice: np.ndarray = np.array(log.fairness_scores[-log_freq:])
        offline_slice: np.ndarray = np.array(log.offline_rates[-log_freq:])
        deadline_slice: np.ndarray = np.array(log.deadline_satisfaction_rates[-log_freq:])
        offload_local_slice: np.ndarray = np.array(log.offloading_ratio_local[-log_freq:])
        offload_coop_slice: np.ndarray = np.array(log.offloading_ratio_cooperative[-log_freq:])
        offload_mbs_slice: np.ndarray = np.array(log.offloading_ratio_mbs[-log_freq:])
        mbs_load_slice: np.ndarray = np.array(log.mbs_load_ratios[-log_freq:])
        learned_decision_slice: np.ndarray = np.array(log.service_learned_decision_counts[-log_freq:])
        heuristic_decision_slice: np.ndarray = np.array(log.service_heuristic_decision_counts[-log_freq:])
        fallback_slice: np.ndarray = np.array(log.service_fallback_counts[-log_freq:])
        predict_exception_fallback_slice: np.ndarray = np.array(log.service_predict_exception_fallback_counts[-log_freq:])
        policy_requested_slice: list[str] = log.service_offload_policy_requested[-log_freq:]
        policy_loaded_slice: list[bool] = log.service_offload_policy_loaded[-log_freq:]
        checkpoint_path_slice: list[str | None] = log.service_offload_policy_checkpoint_path[-log_freq:]
        feature_family_slice: list[str | None] = log.service_offload_policy_feature_family[-log_freq:]

        reward_avg: float = float(np.mean(rewards_slice))
        latency_avg: float = float(np.mean(latencies_slice))
        energy_avg: float = float(np.mean(energies_slice))
        fairness_avg: float = float(np.mean(fairness_slice))
        offline_avg: float = float(np.mean(offline_slice))
        deadline_avg: float = float(np.mean(deadline_slice))
        offload_local_avg: float = float(np.mean(offload_local_slice))
        offload_coop_avg: float = float(np.mean(offload_coop_slice))
        offload_mbs_avg: float = float(np.mean(offload_mbs_slice))
        mbs_load_avg: float = float(np.mean(mbs_load_slice))
        learned_decision_avg: float = float(np.mean(learned_decision_slice))
        heuristic_decision_avg: float = float(np.mean(heuristic_decision_slice))
        fallback_avg: float = float(np.mean(fallback_slice))
        predict_exception_fallback_avg: float = float(np.mean(predict_exception_fallback_slice))

        def _resolve_constant_or_mixed(values: list[object], fallback: object) -> object:
            if not values:
                return fallback
            first_value = values[0]
            for value in values[1:]:
                if value != first_value:
                    return "mixed"
            return first_value

        policy_requested_value = _resolve_constant_or_mixed(policy_requested_slice, "heuristic")
        policy_loaded_value = _resolve_constant_or_mixed(policy_loaded_slice, False)
        checkpoint_path_value = _resolve_constant_or_mixed(checkpoint_path_slice, None)
        feature_family_value = _resolve_constant_or_mixed(feature_family_slice, None)

        # Prepare loss averages from the Log object if available; prefer explicit `losses` dict when provided
        def _safe_mean(lst: list) -> float | None:
            if not lst:
                return None
            vals = [x for x in lst[-log_freq:] if x is not None]
            if not vals:
                return None
            return float(np.mean(np.array(vals)))

        actor_avg = None
        critic_avg = None
        entropy_avg = None
        alpha_avg = None

        if losses is not None:
            actor_loss_val = losses.get("actor")
            actor_avg = float(actor_loss_val) if actor_loss_val is not None else None
            critic_loss_val = losses.get("critic")
            critic_avg = float(critic_loss_val) if critic_loss_val is not None else None
            entropy_loss_val = losses.get("entropy")
            entropy_avg = float(entropy_loss_val) if entropy_loss_val is not None else None
            alpha_loss_val = losses.get("alpha")
            alpha_avg = float(alpha_loss_val) if alpha_loss_val is not None else None
        else:
            actor_avg = _safe_mean(log.actor_losses)
            critic_avg = _safe_mean(log.critic_losses)
            entropy_avg = _safe_mean(log.entropy_losses)
            alpha_avg = _safe_mean(log.alpha_losses)

        # Build human readable log message
        loss_parts: list[str] = []
        if actor_avg is not None:
            loss_parts.append(f"Actor Loss: {actor_avg:.6f}")
        if critic_avg is not None:
            loss_parts.append(f"Critic Loss: {critic_avg:.6f}")
        if entropy_avg is not None:
            loss_parts.append(f"Entropy Loss: {entropy_avg:.6f}")
        if alpha_avg is not None:
            loss_parts.append(f"Alpha Loss: {alpha_avg:.6f}")

        loss_str = " | ".join(loss_parts) + " | " if loss_parts else ""

        log_msg: str = (
            f"🔄 Episode {progress_step} | "
            f"Total Reward: {reward_avg:.3f} | "
            f"Total Latency: {latency_avg:.3f} | "
            f"Total Energy: {energy_avg:.3f} | "
            f"Final Fairness: {fairness_avg:.3f} | "
            f"Offline Rate: {offline_avg:.3f} | "
            f"Mean Deadline Sat: {deadline_avg:.3f} | "
            f"Mean Offload L/C/M: {offload_local_avg:.3f}/{offload_coop_avg:.3f}/{offload_mbs_avg:.3f} | "
            f"Mean MBS Load: {mbs_load_avg:.3f} | "
            f"Offload Policy: req={policy_requested_value} loaded={policy_loaded_value} family={feature_family_value} | "
            f"Mean Audit Learned/Heuristic/Fallback/Exception: "
            f"{learned_decision_avg:.3f}/{heuristic_decision_avg:.3f}/{fallback_avg:.3f}/{predict_exception_fallback_avg:.3f} | "
            + loss_str
            + f"Elapsed Time: {elapsed_time:.2f}s\n"
        )

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(log_msg)

        # Prepare JSON entry, only include keys that are not None
        data_entry: dict = {
            "episode": progress_step,
            "reward": reward_avg,
            "latency": latency_avg,
            "energy": energy_avg,
            "fairness": fairness_avg,
            "offline_rate": offline_avg,
            "deadline_satisfaction_rate": deadline_avg,
            "offloading_ratio_local": offload_local_avg,
            "offloading_ratio_cooperative": offload_coop_avg,
            "offloading_ratio_mbs": offload_mbs_avg,
            "mbs_load_ratio": mbs_load_avg,
            "service_learned_decision_count": learned_decision_avg,
            "service_heuristic_decision_count": heuristic_decision_avg,
            "service_fallback_count": fallback_avg,
            "service_predict_exception_fallback_count": predict_exception_fallback_avg,
            "service_offload_policy_requested": policy_requested_value,
            "service_offload_policy_loaded": policy_loaded_value,
            "service_offload_policy_checkpoint_path": checkpoint_path_value,
            "service_offload_policy_feature_family": feature_family_value,
            "time": elapsed_time,
        }
        if actor_avg is not None:
            data_entry["actor_loss"] = actor_avg
        if critic_avg is not None:
            data_entry["critic_loss"] = critic_avg
        if entropy_avg is not None:
            data_entry["entropy_loss"] = entropy_avg
        if alpha_avg is not None:
            data_entry["alpha_loss"] = alpha_avg

        json_data: list[dict] = []
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, "r") as jf:
                try:
                    json_data = json.load(jf)
                except json.JSONDecodeError:
                    json_data = []

        json_data.append(data_entry)
        with open(self.json_file_path, "w") as f:
            json.dump(json_data, f, indent=4)


def load_configs(config_path: str) -> None:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    for key, value in config_dict.items():
        # Convert lists back to numpy arrays where appropriate
        if isinstance(getattr(default_config, key, None), np.ndarray):
            setattr(default_config, key, np.array(value))
        else:
            setattr(default_config, key, value)

    refresh_derived_config_fields()
    print(f"Configs loaded from {config_path}")
