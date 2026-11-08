"""Shared test helpers: load a module from a file, run Python in a subprocess."""

import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def load_module(path):
    """Import the Python file at ``path`` as a module named after its stem."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_python(*args, env=None, timeout=None, cwd=ROOT):
    """Run ``python *args`` with this interpreter, assert it succeeds, return stdout."""
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return result.stdout
