"""
Path config with clear precedence:

1) Environment vars: ORBIT_ANALYSIS_ROOT, ORBIT_ANALYSIS_DATA_DIR, ORBIT_ANALYSIS_OUTPUT_DIR, ORBIT_ANALYSIS_SCRATCH_DIR
2) Local untracked ./config_local.py with DATA_DIR/OUTPUT_DIR/SCRATCH_DIR
3) Defaults under the repository root

Why: centralize paths, retire hard-coded absolute paths gradually.
"""
from __future__ import annotations
import importlib
import os
from pathlib import Path
from typing import Optional

def _env_path(var: str) -> Optional[Path]:
    v = os.getenv(var)
    return Path(v).expanduser() if v else None

# repo root unless overridden
PROJECT_ROOT = _env_path("ORBIT_ANALYSIS_ROOT") or Path(__file__).resolve().parents[2]

# defaults
_default_data = PROJECT_ROOT / "data"
_default_output = PROJECT_ROOT / "output"
_default_scratch = PROJECT_ROOT / "scratch"

# optional local overrides from ./config_local.py
_data = _output = _scratch = None
try:
    cfg = importlib.import_module("config_local")  # at repo root
    _data = Path(getattr(cfg, "DATA_DIR")) if hasattr(cfg, "DATA_DIR") else None
    _output = Path(getattr(cfg, "OUTPUT_DIR")) if hasattr(cfg, "OUTPUT_DIR") else None
    _scratch = Path(getattr(cfg, "SCRATCH_DIR")) if hasattr(cfg, "SCRATCH_DIR") else None
except Exception:
    # swallow: fall back to env/defaults
    pass

# env > local > defaults
DATA_DIR   = _env_path("ORBIT_ANALYSIS_DATA_DIR")   or _data    or _default_data
OUTPUT_DIR = _env_path("ORBIT_ANALYSIS_OUTPUT_DIR") or _output  or _default_output
SCRATCH_DIR= _env_path("ORBIT_ANALYSIS_SCRATCH_DIR")or _scratch or _default_scratch

def ensure_dirs() -> None:
    """Create common dirs if missing (idempotent)."""
    for p in (DATA_DIR, OUTPUT_DIR, SCRATCH_DIR):
        p.mkdir(parents=True, exist_ok=True)
