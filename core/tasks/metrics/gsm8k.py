"""GSM8K answer extraction; execution is delegated to the reusable sandbox."""

import ast
import math
import re


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
