# Compatibility stub: expose the real module under the legacy name.
# Why: lets existing code `import model_io` without changes.
import importlib as _il, sys as _sys
_impl = _il.import_module("orbit_analysis.model_io")
_sys.modules[__name__] = _impl
