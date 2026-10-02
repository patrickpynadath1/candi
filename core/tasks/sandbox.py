"""Fail-closed, filesystem/network-isolated Python execution for task metrics.

Restricted syntax/builtins reduce the supported language; Bubblewrap provides
isolation. There is deliberately no direct-execution fallback.
"""

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess


@dataclass(frozen=True)
class ExecutionResult:
    result: object = None
    stdout: str = ""
    parsed: bool = False
    executed: bool = False
    timed_out: bool = False
    sandbox_error: bool = False


def validate_code(code):
    if not code.strip():
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)
        ):
            return False
        if isinstance(node, ast.Import) and any(a.name != "math" for a in node.names):
            return False
        if isinstance(node, ast.ImportFrom) and (
            node.module != "math"
            or node.level
            or any(a.name.startswith("_") for a in node.names)
        ):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False
    return True


_RUNNER = r"""
import contextlib, io, json, math, resource, sys
memory = 256 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
class Output(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 65536:
            raise ValueError("output limit")
        return super().write(text)
def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "math":
        return math
    raise ImportError(name)
allowed = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len,
           "range": range, "enumerate": enumerate, "int": int, "float": float,
           "str": str, "bool": bool, "pow": pow, "round": round,
           "print": print, "__import__": limited_import}
try:
    request = json.load(sys.stdin)
    namespace = {"__builtins__": allowed, "math": math}
    output = Output()
    with contextlib.redirect_stdout(output):
        exec(compile(request["code"], "<generated>", "exec"), namespace, namespace)
        function = namespace.get(request["entrypoint"])
        result = function() if callable(function) else None
    if isinstance(result, int) and result.bit_length() > 4096:
        result = None
    encoded = json.dumps({"result": result, "stdout": output.getvalue()}, default=str)
    if len(encoded) > 1048576:
        raise ValueError("result limit")
    print(encoded)
except BaseException as error:
    print(json.dumps({"error": type(error).__name__}))
"""


def bubblewrap_command():
    executable = shutil.which("bwrap")
    if executable is None:
        raise RuntimeError("Bubblewrap is required for code evaluation")
    command = [
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
    ]
    for path in ("/usr", "/lib", "/lib64"):
        if Path(path).exists():
            command.extend(["--ro-bind", path, path])
    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/tmp",
            "/usr/bin/env",
            "-i",
            "PATH=/usr/bin:/bin",
            "/usr/bin/python3",
            "-I",
            "-c",
            _RUNNER,
        ]
    )
    return command


def execute_python(code, *, entrypoint, timeout=3.0):
    """Execute one program in a fresh sandbox with a bounded host-side wall time."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if not validate_code(code):
        return ExecutionResult()
    try:
        completed = subprocess.run(
            bubblewrap_command(),
            input=json.dumps({"code": code, "entrypoint": entrypoint}),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(parsed=True, timed_out=True)
    except (OSError, RuntimeError):
        return ExecutionResult(parsed=True, sandbox_error=True)
    if completed.returncode in {
        -signal.SIGKILL,
        -signal.SIGXCPU,
        128 + signal.SIGKILL,
        128 + signal.SIGXCPU,
    }:
        return ExecutionResult(parsed=True, timed_out=True)
    if completed.returncode:
        return ExecutionResult(parsed=True, sandbox_error=True)
    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("invalid protocol")
    except (json.JSONDecodeError, ValueError):
        return ExecutionResult(parsed=True, sandbox_error=True)
    if "error" in payload:
        return ExecutionResult(parsed=True)
    return ExecutionResult(
        result=payload.get("result"),
        stdout=payload.get("stdout", ""),
        parsed=True,
        executed=True,
    )
