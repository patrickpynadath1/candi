
from .registry import models as model_registry
import importlib, pkgutil

IGNORE = {
    "__init__", 
    "registry", 
    "unit_test_attention", 
    "common",
    "sphere_checkpoint",
    }

for _, name, _ in pkgutil.iter_modules(__path__):
    if name not in IGNORE:
        importlib.import_module(f".{name}", __name__)
