# legacy import shim
import importlib as _il, sys as _sys
_impl = _il.import_module("orbit_analysis.satellite_io")
_sys.modules[__name__] = _impl
