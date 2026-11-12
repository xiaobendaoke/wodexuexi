"""Deterministic exact-enumeration verification on multi-request competitive snapshots.

Replays the exact Chapter 4 formal main-setting reference history (100 UEs, scale 1.0,
10 formal WORKLOAD_SEEDS x 6 episodes x 10 equidistant snapshot steps = 600 snapshots),
asserts bit-exact snapshot SHA-256 hashes and seed-level admitted-request / empty-fraction
parity against archived docs/figures/data/chapter34_20261004/chapter4_analysis.json, and runs
complete global exact enumeration on every multi-request snapshot with K >= 2 and |A| <= 50,000.

Every snapshot's global optimum assignment and reference methods are cross-checked for exact
IEEE-754 floating-point equality against the untouched production evaluator
utils.snapshot_evaluator.evaluate().
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
from itertools import product
import json
import math
import multiprocessing
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT  # noqa: E402

import config  # noqa: E402
from environment import comm_model as comms  # noqa: E402
from environment.env import Env  # noqa: E402
from marl_models.offload_policy import OFFLOAD_TARGET_COOPERATIVE, OFFLOAD_TARGET_MBS  # noqa: E402
from utils.canonical_experiment import write_json  # noqa: E402
from utils.chapter4 import PROTOCOL, objective, solve_snapshot  # noqa: E402
from utils.formal_protocol import WORKLOAD_SEEDS  # noqa: E402
from utils.logger import refresh_derived_config_fields  # noqa: E402
from utils.snapshot_evaluator import capture, evaluate  # noqa: E402

ARCHIVED_CH4 = REPO_ROOT / "docs" / "figures" / "data" / "chapter34_20261004" / "chapter4_analysis.json"
ARCHIVED_TRACE = REPO_ROOT / "docs" / "figures" / "data" / "chapter34_20261004" / "mc_itr_trace_example.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "figures" / "data" / "chapter34_20261004" / "exact_competitive_verification.json"

METHODS = ("heuristic_control", "greedy_single_pass", "iterative_base", "mc_itr", "exact_small")


def _open_loop_motion(step: int) -> np.ndarray:
    phase = 2.0 * np.pi * (step / 1000.0 + np.arange(config.NUM_UAVS) / config.NUM_UAVS)
    return 0.4 * np.column_stack((np.cos(phase), np.sin(phase)))


def _fast_reference_offload_matrix(env: Env) -> np.ndarray:
    """Compute Heuristic-Control actions directly on live env without pickling."""
    for uav in env.uavs:
        uav._current_service_request_count = 0
        uav.calculate_initial_load()
        uav._uav_mbs_rate = float(
            comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(uav.pos, config.MBS_POS))
        )
    actions = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=int)
    for row, uav in enumerate(env.uavs):
        service = sorted(
            (ue for ue in uav.current_covered_ues if ue.current_request.is_service),
            key=lambda ue: float(np.linalg.norm(uav.pos[:2] - ue.pos[:2])),
        )
        for slot, ue in enumerate(service):
            rate = comms.calculate_ue_uav_rate(
                comms.calculate_channel_gain(ue.pos, uav.pos),
                max(len(uav.current_covered_ues), 1),
            )
            context, neighbor = uav._build_service_offload_context(ue.current_request, rate)
            target, neighbor = uav._select_service_target_from_context(context, neighbor)
            action = (
                int(config.OFFLOAD_ACTION_COOP_BASE + neighbor.id)
                if target == OFFLOAD_TARGET_COOPERATIVE
                else int(config.OFFLOAD_ACTION_MBS if target == OFFLOAD_TARGET_MBS else config.OFFLOAD_ACTION_LOCAL)
            )
            actions[row, slot] = action
    return actions


def _offload_matrix(snapshot, assignment) -> np.ndarray:
    result = np.zeros((config.NUM_UAVS, config.MAX_OFFLOAD_REQUESTS_PER_UAV), dtype=int)
    for request, target in zip(snapshot.requests, assignment):
        result[request[0], request[1]] = target
    return result


def _tx_bytes(size_bytes: float, rate_bps: float) -> float:
    return float(size_bytes) * float(config.BITS_PER_BYTE) / max(float(rate_bps), float(config.EPSILON))


def _exact_enumerate_verified(snapshot, solved_methods: dict, max_assignments: int) -> dict:
    """Bit-exact global enumeration verified against production evaluate()."""
    env = pickle.loads(snapshot.payload)
    for uav in env.uavs:
        uav._uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(uav.pos, config.MBS_POS))
    ues = {ue.id: ue for ue in env.ues}
    k = len(snapshot.requests)
    n_uavs = config.NUM_UAVS
    eps_d = PROTOCOL["deadline_epsilon_s"]

    # Precompute exact IEEE-754 request cost tuple (miss_int, lat_over_deadline)
    # indexed by [m][action][target_uav_load]
    target_uav_table = [[-1] * (n_uavs + 2) for _ in range(k)]
    cost_table = [[[None] * (k + 1) for _ in range(n_uavs + 2)] for _ in range(k)]

    for m, (req_tuple, cands) in enumerate(zip(snapshot.requests, snapshot.candidates)):
        row, _, ue_id, source_id, deadline = req_tuple
        uav = env.uavs[row]
        ue = ues[ue_id]
        cur = ue.current_request
        req_size = cur.req_size
        req_id = cur.req_id
        cpu_cycles = float(config.CPU_CYCLES_PER_BYTE[req_id]) * float(req_size)
        file_size = int(config.FILE_SIZES[req_id])
        ue_uav_rate = comms.calculate_ue_uav_rate(
            comms.calculate_channel_gain(ue.pos, uav.pos),
            max(len(uav.current_covered_ues), 1),
        )
        up_lat = _tx_bytes(req_size, ue_uav_rate)
        denom_d = max(float(deadline), eps_d)

        for a in cands:
            if a == 0:
                target_uav_table[m][a] = source_id
                read_cache = getattr(uav, "cache_snapshot", uav.cache)
                fetch_lat = 0.0 if read_cache[req_id] else _tx_bytes(file_size, uav._uav_mbs_rate)
                cap = config.UAV_COMPUTING_CAPACITY[source_id]
                for load in range(1, k + 1):
                    comp_lat = cpu_cycles / (cap / load)
                    lat = float(up_lat + fetch_lat + comp_lat)
                    cost_table[m][a][load] = (int(lat > deadline), lat / denom_d)
            elif a == 1:
                target_uav_table[m][a] = -1
                mbs_up = _tx_bytes(req_size, uav._uav_mbs_rate)
                mbs_cmp = cpu_cycles / float(config.MBS_COMPUTING_CAPACITY)
                lat = float(up_lat + mbs_up + mbs_cmp)
                entry = (int(lat > deadline), lat / denom_d)
                for load in range(0, k + 1):
                    cost_table[m][a][load] = entry
            else:
                tgt_id = int(a - 2)
                target_uav_table[m][a] = tgt_id
                tgt_uav = env.uavs[tgt_id]
                inter_rate = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(uav.pos, tgt_uav.pos))
                tgt_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(tgt_uav.pos, config.MBS_POS))
                inter_lat = _tx_bytes(req_size, inter_rate)
                tgt_cache = getattr(tgt_uav, "cache_snapshot", tgt_uav.cache)
                fetch_lat = 0.0 if tgt_cache[req_id] else _tx_bytes(file_size, tgt_mbs_rate)
                cap = config.UAV_COMPUTING_CAPACITY[tgt_id]
                for load in range(1, k + 1):
                    comp_lat = cpu_cycles / (cap / load)
                    lat = float(up_lat + inter_lat + fetch_lat + comp_lat)
                    cost_table[m][a][load] = (int(lat > deadline), lat / denom_d)

    def fast_obj(assignment: tuple[int, ...]) -> tuple[int, float]:
        loads = [0] * n_uavs
        for m, a in enumerate(assignment):
            t_u = target_uav_table[m][a]
            if t_u >= 0:
                loads[t_u] += 1
        miss_sum = 0
        ratios = [0.0] * k
        for m, a in enumerate(assignment):
            t_u = target_uav_table[m][a]
            ld = loads[t_u] if t_u >= 0 else 0
            c_miss, c_rat = cost_table[m][a][ld]
            miss_sum += c_miss
            ratios[m] = c_rat
        return (miss_sum, math.fsum(ratios) / k)

    # Cross-check bit-exact agreement on all 4 solved assignments before enumeration
    for mname in ("heuristic_control", "greedy_single_pass", "iterative_base", "mc_itr"):
        a_ref = tuple(solved_methods[mname]["assignment"])
        expected_obj = tuple(solved_methods[mname]["objective"])
        got_obj = fast_obj(a_ref)
        if got_obj != expected_obj:
            raise AssertionError(f"Fast-table mismatch on {mname}: {got_obj} vs {expected_obj}")

    space = math.prod(map(len, snapshot.candidates))
    if space > max_assignments:
        raise ValueError(f"Assignment space {space} exceeds budget {max_assignments}")

    # Vectorized C-speed enumeration over Cartesian product
    grids = np.array(list(product(*snapshot.candidates)), dtype=np.int64)  # (N, K)
    n_cand = grids.shape[0]
    tgt_arr = np.asarray(target_uav_table, dtype=np.int64)  # (K, 7)
    miss_arr = np.zeros((k, n_uavs + 2, k + 1), dtype=np.int64)
    rat_arr = np.zeros((k, n_uavs + 2, k + 1), dtype=np.float64)
    for m in range(k):
        for a in snapshot.candidates[m]:
            for ld in range(0, k + 1):
                if cost_table[m][a][ld] is not None:
                    miss_arr[m, a, ld] = cost_table[m][a][ld][0]
                    rat_arr[m, a, ld] = cost_table[m][a][ld][1]

    req_idx = np.arange(k)[None, :]  # (1, K)
    chosen_uav = tgt_arr[req_idx, grids]  # (N, K), -1 for MBS
    loads_mat = np.zeros((n_cand, n_uavs), dtype=np.int64)
    for u_id in range(n_uavs):
        loads_mat[:, u_id] = np.sum(chosen_uav == u_id, axis=1)

    eff_load = np.where(chosen_uav >= 0, loads_mat[np.arange(n_cand)[:, None], np.maximum(chosen_uav, 0)], 0)
    cand_misses = np.sum(miss_arr[req_idx, grids, eff_load], axis=1)
    min_miss = int(np.min(cand_misses))
    tied_indices = np.flatnonzero(cand_misses == min_miss)

    # Exact math.fsum tie-breaking over all min-miss candidates (matching objective() bit-for-bit)
    best_key = None
    for idx in tied_indices:
        a_tup = tuple(int(v) for v in grids[idx])
        obj_tup = fast_obj(a_tup)
        key = (obj_tup, a_tup)
        if best_key is None or key < best_key:
            best_key = key

    best_obj, best_assignment = best_key
    # Verify winning optimum directly via untouched production evaluate(snapshot, best_assignment)
    prod_eval = evaluate(snapshot, best_assignment)
    prod_obj = objective(prod_eval)
    if prod_obj != best_obj:
        raise AssertionError(f"Production evaluate() mismatch at optimum: {prod_obj} vs {best_obj}")

    return {
        "assignment": best_assignment,
        "objective": prod_obj,
        "optimality_complete": True,
        "evaluations": int(space),
    }


def evaluate_seed_competitive_snapshots(job: dict) -> dict:
    """Replay one formal workload seed and solve all tractable K>=2 snapshots exactly."""
    import torch

    torch.set_num_threads(1)
    importlib.reload(config)
    config.NUM_UES = int(job["ues"])
    refresh_derived_config_fields()

    seed = int(job["seed"])
    exact_budget = int(job["exact_budget"])
    max_rounds = int(job["max_rounds"])
    snapshot_steps = set(job["snapshot_steps"])

    env = Env(scenario_seed=seed)
    admitted_counts: list[int] = []
    empty_flags: list[float] = []
    competitive_records: list[dict] = []
    anchor_hashes: dict[str, str] = {}

    for episode in range(int(job["episodes"])):
        env.reset()
        for step in range(int(job["steps"])):
            motion = _open_loop_motion(step)
            if step in snapshot_steps:
                snapshot = capture(env, motion)
                k = len(snapshot.requests)
                space = math.prod(map(len, snapshot.candidates)) if k > 0 else 1
                admitted_counts.append(k)
                empty_flags.append(float(k == 0))
                if episode == 0 and step in (100, 200):
                    anchor_hashes[f"ep{episode}_step{step}"] = snapshot.snapshot_hash

                if k >= 2 and space <= exact_budget:
                    solved = solve_snapshot(snapshot, max_rounds=max_rounds, exact_budget=0)
                    solved["exact_small"] = _exact_enumerate_verified(snapshot, solved, exact_budget)
                    optimum = tuple(solved["exact_small"]["objective"])
                    source_uavs = [int(r[0]) for r in snapshot.requests]
                    has_same_source = len(set(source_uavs)) < len(source_uavs)
                    method_objs = {}
                    method_matches = {}
                    for m in METHODS:
                        obj = [float(solved[m]["objective"][0]), float(solved[m]["objective"][1])]
                        method_objs[m] = obj
                        method_matches[m] = bool(tuple(solved[m]["objective"]) == optimum)
                        if tuple(solved[m]["objective"]) < optimum:
                            raise AssertionError(f"Method {m} beat exact optimum on seed={seed} ep={episode} step={step}")
                    competitive_records.append(
                        {
                            "workload_seed": seed,
                            "episode": episode,
                            "step": step,
                            "snapshot_hash": snapshot.snapshot_hash,
                            "num_requests": k,
                            "assignment_space": int(space),
                            "has_same_source_uav": has_same_source,
                            "objectives": method_objs,
                            "matches_exact": method_matches,
                        }
                    )
                    ref_actions = _offload_matrix(snapshot, solved["heuristic_control"]["assignment"])
                else:
                    ref_actions = _fast_reference_offload_matrix(env)
            else:
                ref_actions = _fast_reference_offload_matrix(env)
            env.step(motion, offloading_actions=ref_actions)

    return {
        "workload_seed": seed,
        "total_snapshots": len(admitted_counts),
        "mean_admitted_requests": float(np.mean(admitted_counts)),
        "empty_fraction": float(np.mean(empty_flags)),
        "k_histogram": {str(v): int(admitted_counts.count(v)) for v in sorted(set(admitted_counts))},
        "anchor_hashes": anchor_hashes,
        "competitive_tractable_count": len(competitive_records),
        "competitive_records": competitive_records,
    }


def verify_against_archived(seed_outputs: list[dict], variant_name: str) -> None:
    """Verify exact replay match against archived chapter4_analysis.json & trace hashes."""
    archived = json.loads(ARCHIVED_CH4.read_text(encoding="utf-8"))
    ref_units = {
        int(u["workload_seed"]): u["mean"]
        for u in archived["units"]
        if u["variant"] == variant_name and u["method"] == "mc_itr"
    }
    for item in seed_outputs:
        seed = int(item["workload_seed"])
        expected_k = float(ref_units[seed][4])
        expected_empty = float(ref_units[seed][5])
        if not math.isclose(item["mean_admitted_requests"], expected_k, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Admitted request mismatch at seed {seed}: {item['mean_admitted_requests']} vs {expected_k}")
        if not math.isclose(item["empty_fraction"], expected_empty, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Empty fraction mismatch at seed {seed}: {item['empty_fraction']} vs {expected_empty}")

    if variant_name == "main" and ARCHIVED_TRACE.exists():
        trace_doc = json.loads(ARCHIVED_TRACE.read_text(encoding="utf-8"))
        by_seed = {int(s["workload_seed"]): s["anchor_hashes"] for s in seed_outputs}
        s126_hash = trace_doc["snapshots"]["lexicographic_tradeoff_s126_t200"]["snapshot_hash"]
        s168_hash = trace_doc["snapshots"]["self_interested_worsening_s168_t100"]["snapshot_hash"]
        if by_seed[126].get("ep0_step200") != s126_hash:
            raise ValueError("SHA-256 mismatch on seed 126 step 200")
        if by_seed[168].get("ep0_step100") != s168_hash:
            raise ValueError("SHA-256 mismatch on seed 168 step 100")


def run_verification(
    *,
    ues: int = 100,
    variant_name: str = "main",
    exact_budget: int = 50000,
    workers: int = 5,
    output: Path = DEFAULT_OUTPUT,
) -> dict:
    t0 = time.perf_counter()
    jobs = [
        {
            "ues": ues,
            "seed": int(s),
            "episodes": 6,
            "steps": 1000,
            "snapshot_steps": list(range(0, 1000, 100)),
            "max_rounds": 1000,
            "exact_budget": exact_budget,
        }
        for s in WORKLOAD_SEEDS
    ]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        seed_outputs = list(pool.map(evaluate_seed_competitive_snapshots, jobs))

    seed_outputs.sort(key=lambda x: int(x["workload_seed"]))
    verify_against_archived(seed_outputs, variant_name)

    all_records = [r for s in seed_outputs for r in s["competitive_records"]]
    n_comp = len(all_records)
    if n_comp == 0:
        raise RuntimeError("No competitive tractable snapshots found")

    k_dist: dict[str, int] = {}
    for r in all_records:
        key = str(r["num_requests"])
        k_dist[key] = k_dist.get(key, 0) + 1

    pooled_summary = {}
    for m in METHODS:
        misses = [r["objectives"][m][0] for r in all_records]
        norm_lat = [r["objectives"][m][1] for r in all_records]
        matches = [r["matches_exact"][m] for r in all_records]
        miss_matches = [
            r["objectives"][m][0] == r["objectives"]["exact_small"][0] for r in all_records
        ]
        pooled_summary[m] = {
            "mean_deadline_misses": float(np.mean(misses)),
            "mean_normalized_latency": float(np.mean(norm_lat)),
            "exact_lex_hit_count": int(sum(matches)),
            "exact_lex_hit_rate": float(np.mean(matches)),
            "exact_f1_hit_count": int(sum(miss_matches)),
            "exact_f1_hit_rate": float(np.mean(miss_matches)),
        }

    seed_level = []
    for s in seed_outputs:
        recs = s["competitive_records"]
        if not recs:
            continue
        entry = {"workload_seed": s["workload_seed"], "count": len(recs)}
        for m in METHODS:
            entry[m] = [
                float(np.mean([r["objectives"][m][0] for r in recs])),
                float(np.mean([r["objectives"][m][1] for r in recs])),
                float(np.mean([r["matches_exact"][m] for r in recs])),
            ]
        seed_level.append(entry)

    elapsed = time.perf_counter() - t0
    report = {
        "status": "PASS",
        "protocol_version": PROTOCOL["version"],
        "source_variant": variant_name,
        "ues": ues,
        "exact_budget": exact_budget,
        "verified_against_archived_analysis": True,
        "production_evaluator_verified_per_snapshot": True,
        "total_grid_snapshots": sum(s["total_snapshots"] for s in seed_outputs),
        "competitive_tractable_snapshots": n_comp,
        "same_source_uav_snapshots": int(sum(1 for r in all_records if r["has_same_source_uav"])),
        "k_distribution": {k: k_dist[k] for k in sorted(k_dist, key=int)},
        "max_assignment_space_enumerated": int(max(r["assignment_space"] for r in all_records)),
        "mean_assignment_space_enumerated": float(np.mean([r["assignment_space"] for r in all_records])),
        "pooled_summary": pooled_summary,
        "seed_level_summary": seed_level,
        "wall_seconds": elapsed,
        "records": all_records,
    }
    write_json(output, report)
    print(
        f"[PASS] Verified {n_comp} multi-request competitive snapshots "
        f"(K in {sorted(map(int, k_dist))}, budget <= {exact_budget}) in {elapsed:.1f}s -> {output}"
    )
    for m in METHODS:
        ps = pooled_summary[m]
        print(
            f"  {m:20s}: F1={ps['mean_deadline_misses']:.4f}, F2={ps['mean_normalized_latency']:.5f}, "
            f"F1_opt_hit={ps['exact_f1_hit_count']}/{n_comp} ({100*ps['exact_f1_hit_rate']:.1f}%), "
            f"Lex_opt_hit={ps['exact_lex_hit_count']}/{n_comp} ({100*ps['exact_lex_hit_rate']:.1f}%)"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ues", type=int, default=100)
    parser.add_argument("--variant", type=str, default="main")
    parser.add_argument("--exact-budget", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_verification(
        ues=args.ues,
        variant_name=args.variant,
        exact_budget=args.exact_budget,
        workers=args.workers,
        output=args.output,
    )


if __name__ == "__main__":
    main()
