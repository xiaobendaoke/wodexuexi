"""Source/configuration provenance for isolated, reproducible experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import config

ROOT = Path(__file__).resolve().parents[1]


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def config_snapshot() -> dict:
    return {key: json_value(value) for key, value in vars(config).items()
            if key.isupper() and isinstance(value, (str, int, float, bool, list, tuple, np.ndarray, type(None)))}


def source_manifest() -> dict:
    paths = list(ROOT.glob("*.py"))
    for folder in ("environment", "marl_models", "utils", "tests", "scripts"):
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend(ROOT.glob("run*.sh"))
    paths.extend(p for p in (ROOT / "CANONICAL_SEMANTICS_SPEC.md", ROOT / "CHAPTER4_PROTOCOL.md", ROOT / "AGENTS.md") if p.exists())
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"git_sha": head, "branch": git("branch", "--show-current"),
            "git_clean": head is not None and status == "", "files": files, "source_hash": digest(files)}


def scenario_manifest(seed: int) -> dict:
    snapshot = config_snapshot()
    # Scientific exogenous state only: ablations and policy hyperparameters
    # must not change the scenario identifier.
    prefixes = ("UAV_", "UE_", "AREA_", "HOTSPOT", "NUM_", "MBS_", "SERVICE_")
    names = {"TIME_SLOT_DURATION", "USE_HOTSPOTS", "ZIPF_BETA", "CPU_CYCLES_PER_BYTE", "FILE_SIZES",
             "MIN_INPUT_SIZE", "MAX_INPUT_SIZE", "MIN_UAV_SEPARATION", "ENV_SEMANTICS_VERSION"}
    physical = {k: v for k, v in snapshot.items()
                if (k in names or k.startswith(prefixes)) and not k.startswith("SERVICE_OFFLOAD_")}
    hardware = {k: snapshot[k] for k in ("UAV_COMPUTING_CAPACITY", "UAV_STORAGE_CAPACITY", "CPU_CYCLES_PER_BYTE", "FILE_SIZES")}
    result = {"scenario_seed": int(seed), "rng_contract": "per_ue_request_mobility_v1", "physical_config": physical,
              "static_parameter_hash": digest(hardware)}
    result["scenario_hash"] = digest(result)
    return result


def exogenous_frame(env) -> dict:
    return {"ue_positions": [ue.pos.tolist() for ue in env.ues],
            "latent_requests": [ue.latent_request for ue in env.ues]}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
