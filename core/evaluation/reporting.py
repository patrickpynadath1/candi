"""Flatten aggregate metrics for experiment loggers, excluding sample records."""


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
