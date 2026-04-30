"""Generate a paper-facing complexity and deployment analysis report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.nn as nn

import config
from marl_models.offload_policy import OFFLOAD_NUM_CLASSES, load_offload_policy_checkpoint
from paths import results_path


def count_linear_macs(model: nn.Module) -> int:
    macs = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            macs += int(module.in_features) * int(module.out_features)
    return macs


def load_checkpoint_stats(checkpoint_path: str | None) -> dict[str, object]:
    if checkpoint_path is None:
        return {
            "checkpoint": None,
            "parameter_count": None,
            "linear_macs_per_request": None,
            "feature_family": None,
            "input_dim": None,
        }
    model, metadata = load_offload_policy_checkpoint(checkpoint_path, device="cpu")
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    input_dim = int(metadata.get("input_dim", 0))
    return {
        "checkpoint": str(checkpoint_path),
        "parameter_count": int(parameter_count),
        "linear_macs_per_request": int(count_linear_macs(model)),
        "feature_family": str(metadata.get("feature_family", "")),
        "input_dim": input_dim,
    }


def build_complexity_payload(checkpoint_path: str | None, request_counts: list[int]) -> dict[str, object]:
    uavs = int(config.NUM_UAVS)
    upper_action_dim = int(config.NUM_UAVS * config.ACTION_DIM)
    lower_classes = int(OFFLOAD_NUM_CLASSES)
    checkpoint_stats = load_checkpoint_stats(checkpoint_path)
    examples = []
    for requests in request_counts:
        examples.append(
            {
                "requests_per_slot": int(requests),
                "two_level_outputs": int(upper_action_dim + lower_classes * int(requests)),
                "request_assignment_combinations_three_way": int(lower_classes ** int(requests)),
                "explicit_uav_or_mbs_combinations": int((uavs + 1) ** int(requests)),
            }
        )

    return {
        "config": {
            "num_uavs": uavs,
            "num_ues": int(config.NUM_UES),
            "action_dim_per_uav": int(config.ACTION_DIM),
            "upper_action_dim": upper_action_dim,
            "lower_classes_per_request": lower_classes,
            "obs_dim_single": int(config.OBS_DIM_SINGLE),
            "max_uav_neighbors": int(config.MAX_UAV_NEIGHBORS),
            "max_associated_ues": int(config.MAX_ASSOCIATED_UES),
        },
        "checkpoint_stats": checkpoint_stats,
        "scaling_examples": examples,
    }


def write_markdown_report(payload: dict[str, object], output_path: str | Path) -> None:
    cfg = payload["config"]
    checkpoint_stats = payload["checkpoint_stats"]
    examples = payload["scaling_examples"]

    lines: list[str] = [
        "# Complexity and Deployment Analysis",
        "",
        "## Core claim",
        "",
        (
            "The two-level design keeps the upper-layer UAV control action fixed at "
            f"`2U = {cfg['upper_action_dim']}` continuous outputs for `U = {cfg['num_uavs']}`. "
            "Request-level offloading is handled by a separate three-class policy over "
            "`local`, `cooperative`, and `MBS`, so the upper-layer MARL action space does not grow with the number of service requests in a slot."
        ),
        "",
        "## Default configuration",
        "",
        f"- UAVs: `{cfg['num_uavs']}`",
        f"- UEs: `{cfg['num_ues']}`",
        f"- Per-UAV movement action dimension: `{cfg['action_dim_per_uav']}`",
        f"- Upper-layer joint movement action dimension: `{cfg['upper_action_dim']}`",
        f"- Lower-layer request classes: `{cfg['lower_classes_per_request']}`",
        f"- Single-UAV observation dimension: `{cfg['obs_dim_single']}`",
        f"- Max UAV neighbors in observation: `{cfg['max_uav_neighbors']}`",
        f"- Max associated UEs in observation: `{cfg['max_associated_ues']}`",
        "",
        "## Surrogate inference cost",
        "",
    ]

    if checkpoint_stats["checkpoint"] is None:
        lines.append("No checkpoint was provided, so model parameter and MAC counts were not computed.")
    else:
        lines.extend(
            [
                f"- Checkpoint: `{checkpoint_stats['checkpoint']}`",
                f"- Feature family: `{checkpoint_stats['feature_family']}`",
                f"- Input dimension: `{checkpoint_stats['input_dim']}`",
                f"- Parameters: `{checkpoint_stats['parameter_count']}`",
                f"- Approximate linear MACs per request: `{checkpoint_stats['linear_macs_per_request']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Action-space comparison",
            "",
            "| Service requests in one slot | Two-level output count | Three-way request assignments | Explicit UAV/MBS assignments |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for example in examples:
        lines.append(
            "| "
            f"{example['requests_per_slot']} | "
            f"{example['two_level_outputs']} | "
            f"{example['request_assignment_combinations_three_way']} | "
            f"{example['explicit_uav_or_mbs_combinations']} |"
        )

    lines.extend(
        [
            "",
            "## Paper wording",
            "",
            (
                "A safe statement is: the method is not a globally optimal monolithic joint optimizer. "
                "Its contribution is a deployable decomposition that fixes the upper-layer continuous action interface "
                "and moves request-dependent discrete choices into a lightweight online classifier plus safety reranking."
            ),
        ]
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complexity/deployment analysis for the two-level framework.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional surrogate checkpoint for parameter/MAC counts.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(results_path("reports", "complexity_deployment_analysis.md")),
        help="Output Markdown report.",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=str(results_path("reports", "complexity_deployment_analysis.json")),
        help="Output JSON payload.",
    )
    parser.add_argument("--request_counts", type=int, nargs="+", default=[1, 5, 10, 20], help="Request-count examples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_complexity_payload(args.checkpoint, [int(value) for value in args.request_counts])
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
