"""Nested-tuning case directories as plain records.

A case directory (``nested_tuning`` output, one recipe x task x training seed) holds
``protocol.json`` (config and evaluation spec), ``training/candidates/*/inference/
<policy>/selected.json`` (each sampler policy's best validation trial) and
``final/`` (the selected policy's test score and per-seed records).
"""

from dataclasses import dataclass, field
import json
from pathlib import Path


# The MI210 runs named the renoising samplers renoising / renoising_greedy; the
# post and the worked examples call them renoise_sample / renoise_greedy.
POLICY_ALIASES = {"renoising": "renoise_sample", "renoising_greedy": "renoise_greedy"}


def short_parameters(parameters):
    """{'algorithm.predictor.temperature': 0.8277} -> {'temperature': 0.828}."""
    return {
        key.rsplit(".", 1)[-1]: round(value, 3) if isinstance(value, float) else value
        for key, value in parameters.items()
    }


@dataclass
class Policy:
    """One sampler policy's selected validation trial."""

    name: str
    validation: float
    seeds: list
    parameters: dict


@dataclass
class Case:
    name: str
    path: Path
    task: str
    recipe: str
    config: dict
    policies: dict = field(default_factory=dict)  # name -> Policy (validation)
    selected: str | None = None  # policy evaluated on the test split
    test: float | None = None
    test_seeds: list = field(default_factory=list)
    final_path: Path | None = None

    def records(self):
        """{test seed: records} of the selected policy's final evaluation."""
        if self.final_path is None:
            return {}
        return {
            int(path.parent.name.removeprefix("seed-")): json.loads(path.read_text())["records"]
            for path in sorted(self.final_path.glob("seed-*/results.json"))
        }

    def winner(self):
        """(policy, tied policies): best validation score, ties to the first listed."""
        if not self.policies:
            return None, []
        best = max(policy.validation for policy in self.policies.values())
        tied = [name for name, policy in self.policies.items() if policy.validation == best]
        return tied[0], tied


def _read(path):
    return json.loads(Path(path).read_text())


def load_case(path):
    """One case directory as a ``Case``; missing parts (unfinished runs) stay empty."""
    path = Path(path)
    protocol = _read(path / "protocol.json")
    config = protocol.get("config", {})
    benchmark = config.get("benchmark", {})
    case = Case(
        name=protocol.get("spec", {}).get("name", path.name),
        path=path,
        task=benchmark.get("task_id", ""),
        recipe=benchmark.get("algorithm_name", ""),
        config=config,
    )
    order = [policy["name"] for policy in protocol.get("spec", {}).get("inference", [])]
    found = {}
    for selected in path.glob("training/candidates/*/inference/*/selected.json"):
        data = _read(selected)
        name = selected.parent.name
        found[name] = Policy(name, data["score"], data.get("seed_scores", []), data.get("parameters", {}))
    for name in order + sorted(set(found) - set(order)):
        if name in found:
            case.policies[name] = found[name]
    result = path / "final" / "result.json"
    if result.exists():
        data = _read(result)
        case.selected = data.get("selection", {}).get("policy", {}).get("name")
        case.test, case.test_seeds = data.get("score"), data.get("seed_scores", [])
        case.final_path = path / "final"
    return case


def load_cases(root):
    """Every case under ``root`` (searched recursively for protocol.json)."""
    return [load_case(protocol.parent) for protocol in sorted(Path(root).rglob("protocol.json"))]


def merge_cases(cases):
    """Combine cases of the same name (one checkpoint tuned in several runs).

    Policies are united in order; the test result comes from the run that
    evaluated the overall validation winner.
    """
    merged = {}
    for case in cases:
        if case.name not in merged:
            merged[case.name] = case
            continue
        target = merged[case.name]
        for name, policy in case.policies.items():
            target.policies.setdefault(name, policy)
        winner, _ = target.winner()
        if case.selected == winner and case.final_path is not None:
            target.selected, target.test, target.test_seeds = case.selected, case.test, case.test_seeds
            target.final_path = case.final_path
    return list(merged.values())


def bleu_parts(records, *, continuation="continuation", reference="reference"):
    """Corpus BLEU of generated continuations: with and without the brevity penalty.

    Returns ``{bleu, bleu_nobp, bp, len_ratio}``; ``bleu_nobp`` is BLEU / BP, the
    n-gram precision term alone. Needs the ``scoring`` extra (sacrebleu).
    """
    from sacrebleu.metrics import BLEU

    hypotheses = [record[continuation] for record in records]
    references = [record[reference] for record in records]
    score = BLEU().corpus_score(hypotheses, [references])
    return {
        "bleu": score.score,
        "bleu_nobp": score.score / score.bp if score.bp > 0 else 0.0,
        "bp": score.bp,
        "len_ratio": score.sys_len / score.ref_len if score.ref_len else 0.0,
    }


def bleu_summary(record_lists, digits=4):
    """BLEU / BP over test seeds: mean, per seed, mean BP and mean length ratio, rounded.

    ``record_lists`` holds one list of generated records per test seed, in seed order.
    """
    parts = [bleu_parts(records) for records in record_lists]

    def mean(key):
        return round(sum(part[key] for part in parts) / len(parts), digits)

    return {
        "score_nobp": mean("bleu_nobp"),
        "seeds_nobp": [round(part["bleu_nobp"], digits) for part in parts],
        "bp": mean("bp"),
        "len_ratio": mean("len_ratio"),
    }
