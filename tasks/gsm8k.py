"""GSM8K prompt dataset and answer scoring for TinyGSM evaluation."""

import ast
from dataclasses import dataclass
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import datasets

from .tinygsm import _token_ids


def get_gsm8k_dataset(config, tokenizer):
    data = config.data
    path = getattr(data, "data_path", None)
    if path:
        records = json.loads(Path(path).read_text())
    else:
        raw = datasets.load_dataset(
            "gsm8k", "main", split="test", cache_dir=data.cache_dir
        )
        records = [
            {"prompt": row["question"], "response_ground_truth": row["answer"]}
            for row in raw
        ]
    bos = tokenizer.bos_token_id
    separator = _token_ids(tokenizer, data.separator)

    def tokenize(record):
        prompt = [bos] + _token_ids(tokenizer, record["prompt"].strip()) + separator
        return {**record, "input_ids": prompt}

    return (
        datasets.Dataset.from_list(records)
        .map(tokenize, desc="Tokenizing GSM8K prompts")
        .with_format("torch", columns=["input_ids"], output_all_columns=True)
    )


def to_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            number = float(value)
        else:
            match = re.search(r"[-+]?\$?\d[\d,]*\.?\d*", str(value))
            if match is None:
                return None
            number = float(match.group().replace("$", "").replace(",", ""))
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(round(number)) if abs(number - round(number)) < 1e-6 else number


def gold_answer(answer):
    match = re.search(r"####\s*(.+)", answer)
    return to_number(match.group(1) if match else answer)


def numeric_answer(response):
    patterns = [
        r"(?:final answer|answer|result)\s*(?:is|=|:)\s*([-+]?\$?\d[\d,]*\.?\d*)",
        r"####\s*([-+]?\$?\d[\d,]*\.?\d*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        if matches:
            return to_number(matches[-1])
    matches = re.findall(r"[-+]?\$?\d[\d,]*\.?\d*", response)
    return to_number(matches[-1]) if matches else None


def numbers_equal(prediction, target):
    if prediction is None or target is None:
        return False
    return abs(float(prediction) - float(target)) <= 1e-3


def extract_code(response):
    fenced = re.search(
        r"```(?:python)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE
    )
    code = fenced.group(1) if fenced else response
    start = code.find("def ")
    code = code[start:].strip() if start >= 0 else ""
    if not code:
        return ""
    try:
        ast.parse(code)
        return code
    except SyntaxError as error:
        if error.lineno is None or error.lineno <= 1:
            return code
        error_line = error.lineno
    candidate = "\n".join(code.splitlines()[: error_line - 1]).strip()
    try:
        ast.parse(candidate)
        return candidate
    except SyntaxError:
        return code


_CODE_RUNNER = r"""
import contextlib, io, json, math, os, resource, select, signal, sys, time

def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "math":
        return math
    raise ImportError(name)
allowed = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len,
           "range": range, "enumerate": enumerate, "int": int, "float": float,
           "str": str, "bool": bool, "pow": pow, "round": round,
           "print": print, "__import__": limited_import}

def run_one(code, write_fd):
    memory = 256 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    namespace = {"__builtins__": allowed, "math": math}
    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, "<generated>", "exec"), namespace, namespace)
            function = namespace.get("simple_math_problem")
            result = function() if callable(function) else None
        # Avoid Python's huge-integer conversion limit and bound protocol output.
        if isinstance(result, int) and result.bit_length() > 4096:
            result = None
        stdout = output.getvalue()[:65536]
        payload = {"result": result, "stdout": stdout}
        encoded = json.dumps(payload, default=str).encode()[:1048576]
    except BaseException as error:
        encoded = json.dumps({"error": type(error).__name__}).encode()
    os.write(write_fd, encoded)

for line in sys.stdin:
    try:
        request = json.loads(line)
        code = request["code"]
        timeout = float(request.get("timeout", 3.0))
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            run_one(code, write_fd)
            os._exit(0)
        os.close(write_fd)
        ready, _, _ = select.select([read_fd], [], [], timeout)
        if ready:
            encoded = os.read(read_fd, 1048577)
            os.waitpid(pid, 0)
            response = json.loads(encoded) if encoded else {"error": "EmptyResult"}
        else:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            response = {"timed_out": True}
        os.close(read_fd)
    except BaseException as error:
        response = {"sandbox_error": type(error).__name__}
    sys.stdout.write(json.dumps(response, default=str) + "\n")
    sys.stdout.flush()
"""


@dataclass(frozen=True)
class CodeEvaluation:
    result: int | float | None
    parsed: bool
    executed: bool
    timed_out: bool = False
    sandbox_error: bool = False


def _resource_limits():
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        memory = 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    except (ImportError, OSError, ValueError):
        pass


def _bubblewrap_command():
    executable = shutil.which("bwrap")
    if executable is None:
        raise RuntimeError("Bubblewrap is required for GSM8K code evaluation")
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
            _CODE_RUNNER,
        ]
    )
    return command


def _validated_code(response):
    code = extract_code(response)
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    forbidden = (ast.ClassDef, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            return None
        if isinstance(node, ast.Import):
            if any(alias.name != "math" for alias in node.names):
                return None
        if isinstance(node, ast.ImportFrom):
            if (
                node.module != "math"
                or node.level != 0
                or any(alias.name.startswith("_") for alias in node.names)
            ):
                return None
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return None
    return code


def evaluate_codes(responses, timeout=3.0):
    """Evaluate many responses through one Bubblewrap worker.

    Each response still runs in a fresh fork with independent resource limits.
    """
    codes = [_validated_code(response) for response in responses]
    evaluations = [None] * len(codes)
    try:
        worker = subprocess.Popen(
            _bubblewrap_command(),
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
    except OSError:
        return [
            CodeEvaluation(None, parsed=code is not None, executed=False, sandbox_error=code is not None)
            for code in codes
        ]
    try:
        for index, code in enumerate(codes):
            if code is None:
                evaluations[index] = CodeEvaluation(None, parsed=False, executed=False)
                continue
            worker.stdin.write(json.dumps({"code": code, "timeout": timeout}) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                evaluations[index] = CodeEvaluation(
                    None, parsed=True, executed=False, sandbox_error=True
                )
                continue
            if payload.get("timed_out"):
                evaluations[index] = CodeEvaluation(
                    None, parsed=True, executed=False, timed_out=True
                )
            elif "sandbox_error" in payload:
                evaluations[index] = CodeEvaluation(
                    None, parsed=True, executed=False, sandbox_error=True
                )
            elif "error" in payload:
                evaluations[index] = CodeEvaluation(None, parsed=True, executed=False)
            else:
                result = to_number(payload.get("result"))
                if result is None:
                    result = numeric_answer(payload.get("stdout", ""))
                evaluations[index] = CodeEvaluation(result, parsed=True, executed=True)
    except (BrokenPipeError, OSError):
        for index, code in enumerate(codes):
            if evaluations[index] is None:
                evaluations[index] = CodeEvaluation(
                    None, parsed=code is not None, executed=False, sandbox_error=code is not None
                )
    finally:
        if worker.stdin:
            worker.stdin.close()
        try:
            worker.wait(timeout=1)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
    return evaluations


def evaluate_code(response, timeout=3.0):
    return evaluate_codes([response], timeout)[0]


def code_answer(response, timeout=3.0):
    return evaluate_code(response, timeout).result


def score_response(response, answer, method="numeric", timeout=3.0):
    if method == "numeric":
        prediction = numeric_answer(response)
    elif method == "code":
        prediction = code_answer(response, timeout)
    else:
        raise ValueError(f"unknown GSM8K scoring method: {method}")
    return numbers_equal(prediction, gold_answer(answer))
