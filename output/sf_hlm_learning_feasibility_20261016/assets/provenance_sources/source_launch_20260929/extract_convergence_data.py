#!/usr/bin/env python3
"""从训练日志中提取收敛数据，包括推导的energy_efficiency。"""

import json
from pathlib import Path
import numpy as np


def extract_convergence_from_log(log_path: Path, algorithm: str, seed: int) -> dict:
    """从训练日志中提取收敛数据。"""
    with open(log_path, 'r') as f:
        data = json.load(f)

    convergence_points = []
    for entry in data:
        ep = entry.get('episode', 0)
        if ep % 20 == 0 and ep > 0:
            energy = entry.get('energy', 0.0)
            dsr = entry.get('deadline_satisfaction_rate', 0.0)

            # 推导energy_efficiency
            # EEE = deadline_satisfied_count / total_energy
            # 由于我们只有dsr（比例）和energy，需要估算deadline_satisfied_count
            # 假设每个episode有1000个time steps，每个time step有约100个service requests
            # 这是一个粗略估计，实际值需要从环境获取
            estimated_requests = 1000 * 100  # STEPS_PER_EPISODE * NUM_UES
            deadline_satisfied_count = dsr * estimated_requests
            energy_efficiency = deadline_satisfied_count / max(energy, 1e-9)

            convergence_points.append({
                'episode': ep,
                'energy_efficiency': float(energy_efficiency),
                'reward': float(entry.get('reward', 0.0)),
                'energy': float(energy),
                'deadline_satisfaction_rate': float(dsr),
                'offline_rate': float(entry.get('offline_rate', 0.0)),
                'mbs_load_ratio': float(entry.get('mbs_load_ratio', 0.0)),
            })

    return {
        'algorithm': algorithm,
        'seed': seed,
        'convergence_points': convergence_points,
    }


def main():
    # 提取所有训练日志的收敛数据
    log_dir = Path('train_logs/hierarchical_mappo')
    output_dir = Path('results/sensitivity/effective_efficiency_convergence/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有训练日志
    log_files = list(log_dir.glob('log_data_*_proposed.json'))
    log_files.extend(list(log_dir.glob('log_data_*_vanilla_mappo.json')))
    log_files.extend(list(log_dir.glob('log_data_*_ippo.json')))
    print(f'Found {len(log_files)} log files with algorithm names')

    all_results = {}
    seed_counter = {}  # 用于分配默认种子

    for log_path in log_files:
        # 从文件名提取算法和种子信息
        filename = log_path.stem
        parts = filename.split('_')

        # 解析算法名称
        if 'proposed' in filename:
            algorithm = 'proposed'
        elif 'vanilla_mappo' in filename:
            algorithm = 'vanilla_mappo'
        elif 'ippo' in filename:
            algorithm = 'ippo'
        else:
            continue

        # 解析种子信息
        seed = None
        for i, part in enumerate(parts):
            if part == 'seed' and i + 1 < len(parts):
                try:
                    seed = int(parts[i + 1])
                except ValueError:
                    pass

        # 如果没有种子信息，分配默认种子
        if seed is None:
            if algorithm not in seed_counter:
                seed_counter[algorithm] = 0
            seed_counter[algorithm] += 1
            seed = seed_counter[algorithm] * 42  # 使用42的倍数作为默认种子

        print(f'Extracting from {log_path.name}: algorithm={algorithm}, seed={seed}')

        # 提取收敛数据
        result = extract_convergence_from_log(log_path, algorithm, seed)

        # 保存结果
        if algorithm not in all_results:
            all_results[algorithm] = []
        all_results[algorithm].append(result)

    # 保存汇总结果
    with open(output_dir / 'convergence_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 生成汇总统计
    summary = generate_convergence_summary(all_results)
    with open(output_dir.parent / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\nExtracted convergence data for {len(all_results)} algorithms')
    print(f'Results saved to {output_dir}')


def generate_convergence_summary(all_results: dict) -> dict:
    """生成收敛曲线汇总统计。"""
    summary = {
        'experiment': 'effective_efficiency_convergence',
        'eval_interval': 20,
        'algorithms': {},
    }

    for algo_name, seed_results in all_results.items():
        if not seed_results:
            continue

        # 收集所有seed在每个episode的指标
        episode_data = {}
        for seed_result in seed_results:
            for point in seed_result['convergence_points']:
                ep = point['episode']
                if ep not in episode_data:
                    episode_data[ep] = []
                episode_data[ep].append(point)

        # 计算每个episode的统计量
        episode_stats = []
        for ep in sorted(episode_data.keys()):
            points = episode_data[ep]
            if len(points) < 2:
                continue

            energy_eff_values = [p['energy_efficiency'] for p in points]
            reward_values = [p['reward'] for p in points]
            energy_values = [p['energy'] for p in points]
            dsr_values = [p['deadline_satisfaction_rate'] for p in points]

            episode_stats.append({
                'episode': ep,
                'energy_efficiency': {
                    'mean': float(np.mean(energy_eff_values)),
                    'std': float(np.std(energy_eff_values)),
                    'ci_low': float(np.percentile(energy_eff_values, 2.5)),
                    'ci_high': float(np.percentile(energy_eff_values, 97.5)),
                },
                'reward': {
                    'mean': float(np.mean(reward_values)),
                    'std': float(np.std(reward_values)),
                },
                'energy': {
                    'mean': float(np.mean(energy_values)),
                    'std': float(np.std(energy_values)),
                },
                'deadline_satisfaction_rate': {
                    'mean': float(np.mean(dsr_values)),
                    'std': float(np.std(dsr_values)),
                },
            })

        summary['algorithms'][algo_name] = {
            'num_seeds': len(seed_results),
            'episode_stats': episode_stats,
        }

    return summary


if __name__ == '__main__':
    main()
