"""GSM8K prompt dataset and answer scoring for TinyGSM evaluation."""

import ast
import json
import math
import re
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
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(r"[-+]?\$?\d[\d,]*\.?\d*", str(value))
        if match is None:
            return None
        number = float(match.group().replace("$", "").replace(",", ""))
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
    return code[start:].strip() if start >= 0 else ""


_CODE_RUNNER = r"""
import contextlib, io, json, math, sys
code = sys.stdin.read()
def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "math":
        return math
    raise ImportError(name)
allowed = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len,
           "range": range, "enumerate": enumerate, "int": int, "float": float,
           "round": round, "print": print, "__import__": limited_import}
namespace = {"__builtins__": allowed, "math": math}
try:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<generated>", "exec"), namespace, namespace)
        function = namespace.get("simple_math_problem")
        result = function() if callable(function) else None
    print(json.dumps({"result": result, "stdout": output.getvalue()}, default=str))
except BaseException as error:
    print(json.dumps({"error": type(error).__name__}))
"""


def _resource_limits():
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        memory = 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    except (ImportError, OSError, ValueError):
        pass


def code_answer(response, timeout=3.0):
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
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if names != ["math"]:
                return None
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return None
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _CODE_RUNNER],
            input=code,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            preexec_fn=_resource_limits if sys.platform != "win32" else None,
        )
        payload = json.loads(completed.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None
    result = to_number(payload.get("result"))
    return result if result is not None else numeric_answer(payload.get("stdout", ""))


def score_response(response, answer, method="numeric", timeout=3.0):
    if method == "numeric":
        prediction = numeric_answer(response)
    elif method == "code":
        prediction = code_answer(response, timeout)
    else:
        raise ValueError(f"unknown GSM8K scoring method: {method}")
    return numbers_equal(prediction, gold_answer(answer))
