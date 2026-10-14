"""Read and flatten aggregate metrics, excluding sample records."""

import math


def scalar_metrics(value, prefix=""):
    if isinstance(value, dict):
        for name, child in value.items():
            if name != "records":
                yield from scalar_metrics(child, f"{prefix}/{name}" if prefix else name)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            label = child.get("min_length", index) if isinstance(child, dict) else index
            yield from scalar_metrics(child, f"{prefix}/{label}")
    elif isinstance(value, (int, float)):
        yield prefix, value


def metric_value(result, path):
    for key in path.split("/"):
        if isinstance(result, list):
            result = next(row for row in result if str(row["min_length"]) == key)
        else:
            result = result[key]
    value = float(result)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite metric: {path}")
    return value
