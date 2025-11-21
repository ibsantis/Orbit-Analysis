# Forwarder: load the real package implementation from src/orbit_analysis
import sys, importlib.util
from pathlib import Path

_src_init = Path(__file__).resolve().parent / "src" / "orbit_analysis" / "__init__.py"
if not _src_init.exists():
    raise ImportError(f"Cannot find {_src_init}")

spec = importlib.util.spec_from_file_location("orbit_analysis", _src_init)
real = importlib.util.module_from_spec(spec)
sys.modules[__name__] = real  # replace this package with the real one
spec.loader.exec_module(real)
