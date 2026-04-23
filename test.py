from marl_models.base_model import MARLModel
from environment.env import Env
from utils.logger import Logger, Log

# from utils.plot_snapshots import plot_snapshot

# from utils.plot_snapshots import update_trajectories, reset_trajectories  # trajectory tracking, comment if not needed
import config
import numpy as np
import time


def _get_episode_runtime_audit(env: Env) -> dict[str, object]:
    """Read the latest episode-level offloading audit snapshot from the environment."""

    audit: dict[str, object] = env.last_runtime_audit or {}
    return {
        "service_learned_decision_count": float(audit.get("episode_service_learned_decision_count", 0.0)),
        "service_heuristic_decision_count": float(audit.get("episode_service_heuristic_decision_count", 0.0)),
        "service_fallback_count": float(audit.get("episode_service_fallback_count", 0.0)),
        "service_predict_exception_fallback_count": float(
            audit.get("episode_service_predict_exception_fallback_count", 0.0)
        ),
        "service_offload_policy_requested": str(audit.get("service_offload_policy_requested", "heuristic")),
        "service_offload_policy_loaded": bool(audit.get("service_offload_policy_loaded", False)),
        "service_offload_policy_checkpoint_path": audit.get("service_offload_policy_checkpoint_path"),
        "service_offload_policy_feature_family": audit.get("service_offload_policy_feature_family"),
    }


def test_model(env: Env, model: MARLModel, logger: Logger, num_episodes: int) -> None:
    start_time: float = time.time()
    episode_log: Log = Log()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        model.reset()
        episode_reward: float = 0.0
        episode_latency: float = 0.0
        episode_energy: float = 0.0
        episode_fairness: float = 0.0
        episode_offline_rate: float = 0.0
        episode_deadline_satisfaction_sum: float = 0.0
        episode_offload_local_sum: float = 0.0
        episode_offload_cooperative_sum: float = 0.0
        episode_offload_mbs_sum: float = 0.0
        episode_mbs_load_sum: float = 0.0
        # reset_trajectories(env)  # tracking code, comment if not needed
        # plot_snapshot(env, episode, 0, logger.log_dir, logger.timestamp, True)

        for step in range(1, config.STEPS_PER_EPISODE + 1):
            # if step % config.TEST_IMG_FREQ == 0:
            # plot_snapshot(env, episode, step, logger.log_dir, logger.timestamp)

            obs_arr: np.ndarray = np.asarray(obs, dtype=np.float32)
            actions: np.ndarray = model.select_actions(obs_arr, exploration=False)
            next_obs, rewards, metrics = env.step(actions)
            # update_trajectories(env)  # tracking code, comment if not needed
            done: bool = step >= config.STEPS_PER_EPISODE
            obs = next_obs

            total_latency = metrics["latency"]
            total_energy = metrics["energy"]
            jfi = metrics["fairness"]
            offline_rate = metrics["offline_rate"]
            episode_reward += np.sum(rewards)
            episode_latency += total_latency
            episode_energy += total_energy
            episode_fairness = jfi
            episode_offline_rate = offline_rate
            episode_deadline_satisfaction_sum += metrics["deadline_satisfaction_rate"]
            episode_offload_local_sum += metrics["offloading_ratio_local"]
            episode_offload_cooperative_sum += metrics["offloading_ratio_cooperative"]
            episode_offload_mbs_sum += metrics["offloading_ratio_mbs"]
            episode_mbs_load_sum += metrics["mbs_load_ratio"]
            if done:
                break

        # Request-level metrics are currently logged as means of per-step ratios.
        episode_length = max(float(config.STEPS_PER_EPISODE), 1.0)
        runtime_audit = _get_episode_runtime_audit(env)
        episode_log.append(
            episode_reward,
            episode_latency,
            episode_energy,
            episode_fairness,
            episode_offline_rate,
            deadline_satisfaction_rate=episode_deadline_satisfaction_sum / episode_length,
            offloading_ratio_local=episode_offload_local_sum / episode_length,
            offloading_ratio_cooperative=episode_offload_cooperative_sum / episode_length,
            offloading_ratio_mbs=episode_offload_mbs_sum / episode_length,
            mbs_load_ratio=episode_mbs_load_sum / episode_length,
            service_learned_decision_count=float(runtime_audit["service_learned_decision_count"]),
            service_heuristic_decision_count=float(runtime_audit["service_heuristic_decision_count"]),
            service_fallback_count=float(runtime_audit["service_fallback_count"]),
            service_predict_exception_fallback_count=float(runtime_audit["service_predict_exception_fallback_count"]),
            service_offload_policy_requested=str(runtime_audit["service_offload_policy_requested"]),
            service_offload_policy_loaded=bool(runtime_audit["service_offload_policy_loaded"]),
            service_offload_policy_checkpoint_path=runtime_audit["service_offload_policy_checkpoint_path"],
            service_offload_policy_feature_family=runtime_audit["service_offload_policy_feature_family"],
        )
        if episode % config.TEST_LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.TEST_LOG_FREQ, elapsed_time)
