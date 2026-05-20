#!/usr/bin/env python3
"""Apply DSR-aware config overrides to config.py."""
import sys

profile = sys.argv[1] if len(sys.argv) > 1 else "moderate"

overrides = {
    "moderate": {
        "REWARD_W_DSR": 2.0,
        "REWARD_W_ENERGY": 0.3,
        "OFFLOAD_REWARD_DEADLINE_WEIGHT": 5.0,
        "OFFLOAD_REWARD_SUCCESS_BONUS": 1.0,
        "OFFLOAD_REWARD_LATENCY_WEIGHT": 3.0,
        "OFFLOAD_REWARD_ENERGY_WEIGHT": 0.8,
        "OFFLOAD_REWARD_MBS_WEIGHT": 0.25,
        "OFFLOAD_DSR_TARGET": 0.26,
        "OFFLOAD_MBS_LOAD_CEILING": 0.06,
        "OFFLOAD_LAGRANGE_LR": 0.3,
        "OFFLOAD_LAGRANGE_MAX": 30.0,
    },
    "strong": {
        "REWARD_W_DSR": 3.0,
        "REWARD_W_ENERGY": 0.2,
        "OFFLOAD_REWARD_DEADLINE_WEIGHT": 7.0,
        "OFFLOAD_REWARD_SUCCESS_BONUS": 1.5,
        "OFFLOAD_REWARD_LATENCY_WEIGHT": 2.5,
        "OFFLOAD_REWARD_ENERGY_WEIGHT": 0.5,
        "OFFLOAD_REWARD_MBS_WEIGHT": 0.20,
        "OFFLOAD_DSR_TARGET": 0.28,
        "OFFLOAD_MBS_LOAD_CEILING": 0.08,
        "OFFLOAD_LAGRANGE_LR": 0.5,
        "OFFLOAD_LAGRANGE_MAX": 50.0,
    },
}

if profile not in overrides:
    print(f"Unknown profile: {profile}")
    sys.exit(1)

import re

with open("config.py", "r") as f:
    content = f.read()

for key, val in overrides[profile].items():
    # Match: KEY: type = old_value
    pattern = rf"({key}:\s*(\w+)\s*=\s*)[\d.]+(.*)"
    replacement = rf"\g<1>{val}\g<3>"
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        print(f"  {key}: -> {val}")
        content = new_content
    else:
        print(f"  WARNING: {key} not matched")

with open("config.py", "w") as f:
    f.write(content)

print(f"Applied {profile} profile successfully.")
