from .registry import algo_registry
import importlib, pkgutil

IGNORE = {
    "__init__", 
    "registry", 
    "absorbing_state", 
    "algo_utils", 
    "trainer_base", 
    "uniform_state", 
    }

for _, name, _ in pkgutil.iter_modules(__path__):
    if name not in IGNORE:
        importlib.import_module(f".{name}", __name__)
