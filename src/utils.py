"""Common utilities: run dirs, config IO, seeding."""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "reports" / "runs"


def git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def new_run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M")


def make_run_dir(run_id: str | None = None) -> Path:
    rid = run_id or new_run_id()
    p = RUNS_DIR / rid
    (p / "samples").mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())
