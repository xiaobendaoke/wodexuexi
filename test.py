from marl_models.base_model import MARLModel
from environment.env import Env
from utils.logger import Logger, Log

# from utils.plot_snapshots import plot_snapshot

# from utils.plot_snapshots import update_trajectories, reset_trajectories  # trajectory tracking, comment if not needed
import config
import numpy as np
import time


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
        )
        if episode % config.TEST_LOG_FREQ == 0:
            elapsed_time: float = time.time() - start_time
            logger.log_metrics(episode, episode_log, config.TEST_LOG_FREQ, elapsed_time)
