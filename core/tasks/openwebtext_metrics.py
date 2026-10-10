"""Typed OpenWebText generated sample metrics."""

from dataclasses import dataclass
from .metrics.generative import token_entropy, corpus_perplexity


@dataclass(frozen=True)
class OpenWebTextRecord:
    index: int
    text: str
    entropy: float
    negative_log_likelihood: float
    scored_tokens: int


@dataclass(frozen=True)
class OpenWebTextAggregate:
    num_total: int
    generative_perplexity: float | None
    entropy: float | None
    negative_log_likelihood: float
    scored_tokens: int
    reference_model: str
    records: list[OpenWebTextRecord]


class OpenWebTextMetrics:
    supports_variable_length = True

    def __init__(self, tokenizer, scorer, length):
        self.tokenizer, self.scorer, self.length = tokenizer, scorer, length
        self.records = []

    def update(self, samples, examples, indices, *, lengths=None):
        if lengths is not None:
            samples = [sample[:n] for sample, n in zip(samples, lengths)]
        texts = self.tokenizer.batch_decode(samples)
        scores = self.scorer.score(texts, self.length)
        for index, sample, text, score in zip(indices, samples, texts, scores):
            self.records.append(
                OpenWebTextRecord(
                    index,
                    text,
                    token_entropy(sample),
                    score.negative_log_likelihood,
                    score.scored_tokens,
                )
            )

    def compute(self):
        nll = sum(record.negative_log_likelihood for record in self.records)
        count = sum(record.scored_tokens for record in self.records)
        return OpenWebTextAggregate(
            len(self.records),
            corpus_perplexity(nll, count),
            sum(record.entropy for record in self.records) / len(self.records)
            if self.records
            else None,
            nll,
            count,
            self.scorer.model_name,
            list(self.records),
        )
