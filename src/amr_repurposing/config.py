"""Central paths and constants for the whole project.

Every module imports directories from here instead of recomputing ``parents[n]``,
so moving code between packages never breaks a path. Locations can be overridden
with the ``AMR_DATA_DIR`` / ``AMR_MODELS_DIR`` / ``AMR_FIGURES_DIR`` env vars
(useful for CI or a scratch run).
"""
from __future__ import annotations

import os
from pathlib import Path

# src/amr_repurposing/config.py -> parents[2] is the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dir(env: str, default: Path) -> Path:
    return Path(os.environ[env]).expanduser() if env in os.environ else default


DATA_DIR = _dir("AMR_DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR = _dir("AMR_MODELS_DIR", PROJECT_ROOT / "models")
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = _dir("AMR_FIGURES_DIR", REPORTS_DIR / "figures")

# data sub-locations
REFERENCE_DIR = DATA_DIR / "reference"
TARGETS_DIR = DATA_DIR / "targets"
# model sub-locations
TARGET_MODELS_DIR = MODELS_DIR / "target_models"

# reproducibility
SEED = 42
ECFP_N_BITS = 2048
ECFP_RADIUS = 2

for _d in (DATA_DIR, MODELS_DIR, FIGURES_DIR, REFERENCE_DIR, TARGETS_DIR, TARGET_MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_device():
    """MPS (Apple Silicon) > CUDA > CPU. Imported lazily to avoid a hard torch dep."""
    import torch
    return torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")
