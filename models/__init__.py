
from .registry import model_registry
import importlib, pkgutil

IGNORE = {
    "__init__", 
    "registry", 
    "unit_test_attention", 
    "common",
    "cosmos",
    }

for _, name, _ in pkgutil.iter_modules(__path__):
    if name not in IGNORE:
        importlib.import_module(f".{name}", __name__)