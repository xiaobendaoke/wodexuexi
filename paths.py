from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results"


def results_path(*parts: str) -> Path:
    return RESULTS_ROOT.joinpath(*parts)
