"""Cheap diagnostics plus inspectable records for optional external story judges."""

from collections import Counter
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class StoryRecord:
    id: str
    index: int
    prompt: str
    continuation: str
    reference: str
    generated_text: str
    token_count: int
    word_count: int
    eos_present: bool
    invalid_special_count: int
    invalid_utf8: bool
    repeated_trigram_fraction: float


@dataclass(frozen=True)
class StoryAggregate:
    num_total: int
    termination_rate: float | None
    nonempty_rate: float | None
    average_length: float | None
    average_word_count: float | None
    duplicate_rate: float | None
    distinct_1: float | None
    distinct_2: float | None
    repeated_trigram_fraction: float | None
    invalid_special_rate: float | None
    invalid_utf8_rate: float | None
    corpus_bleu: float | None
    bleu_signature: str | None
    data_manifest: dict
    records: list[StoryRecord]


class TinyStoriesMetrics:
    supports_variable_length = True

    def __init__(self, tokenizer, calculate_bleu=False, data_manifest=None):
        self.tokenizer, self.records = tokenizer, []
        self.data_manifest = data_manifest or {}
        self.bleu = None
        if calculate_bleu:
            from sacrebleu.metrics import BLEU

            self.bleu = BLEU()

    def update(self, samples, examples, indices, *, lengths=None):
        for row, (sample, example, index) in enumerate(zip(samples, examples, indices)):
            sample = sample.tolist() if hasattr(sample, "tolist") else list(sample)
            if lengths is not None:
                sample = sample[: int(lengths[row])]
            tail = sample[example["prompt_length"] :]
            eos = self.tokenizer.eos_token_id in tail
            if eos:
                tail = tail[: tail.index(self.tokenizer.eos_token_id)]
            invalid_utf8 = False
            try:
                bytes(t - 4 for t in tail if 4 <= t < 260).decode("utf-8")
            except UnicodeDecodeError:
                invalid_utf8 = True
            text = self.tokenizer.decode(tail)
            words = re.findall(r"\b\w+\b", text.lower())
            trigrams = list(zip(words, words[1:], words[2:]))
            repeated = sum(n - 1 for n in Counter(trigrams).values())
            self.records.append(
                StoryRecord(
                    example["id"],
                    int(index),
                    example["prompt"],
                    text,
                    example["reference"],
                    example["prompt"] + text,
                    len(tail),
                    len(words),
                    eos,
                    sum(not 4 <= token < 260 for token in tail),
                    invalid_utf8,
                    repeated / len(trigrams) if trigrams else 0.0,
                )
            )

    def compute(self):
        records, n = list(self.records), len(self.records)

        def mean(field):
            return sum(getattr(r, field) for r in records) / n if n else None

        words = [re.findall(r"\b\w+\b", r.continuation.lower()) for r in records]

        def distinct(order):
            grams = [g for ws in words for g in zip(*(ws[i:] for i in range(order)))]
            return len(set(grams)) / len(grams) if grams else None

        total = sum(r.token_count for r in records)
        bleu, signature = None, None
        if self.bleu is not None and n:
            bleu = self.bleu.corpus_score(
                [r.continuation for r in records], [[r.reference for r in records]]
            ).score
            signature = str(self.bleu.get_signature())
        return StoryAggregate(
            n,
            mean("eos_present"),
            sum(bool(r.continuation.strip()) for r in records) / n if n else None,
            mean("token_count"),
            mean("word_count"),
            1 - len({r.continuation for r in records}) / n if n else None,
            distinct(1),
            distinct(2),
            mean("repeated_trigram_fraction"),
            sum(r.invalid_special_count for r in records) / total if total else None,
            mean("invalid_utf8"),
            bleu,
            signature,
            self.data_manifest,
            records,
        )
