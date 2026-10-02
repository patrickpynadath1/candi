"""Typed generated/held-out Text8 word-count comparisons."""

from dataclasses import dataclass

from .metrics.text import (
    WordCounts,
    WordAggregate,
    FilteredWordCounts,
    FilteredWordAggregate,
    WordAccumulator,
    complete_words,
)


@dataclass(frozen=True)
class Text8Sample:
    text: str
    character_count: int
    special_token_count: int
    words: WordCounts
    by_min_length: list[FilteredWordCounts]


@dataclass(frozen=True)
class Text8Record:
    index: int
    generated: Text8Sample
    held_out: Text8Sample


@dataclass(frozen=True)
class Text8WordAggregate:
    words: WordAggregate
    by_min_length: list[FilteredWordAggregate]


@dataclass(frozen=True)
class Text8Aggregate:
    num_total: int
    valid_word_rate: float | None
    generated: Text8WordAggregate
    held_out: Text8WordAggregate
    reference_split: str
    vocabulary_source: str
    vocabulary_sha256: str
    vocabulary_size: int
    records: list[Text8Record]


class SampleAccumulator:
    def __init__(self, vocabulary, min_lengths, tokenizer):
        self.tokenizer = tokenizer
        self.words = WordAccumulator(vocabulary)
        self.filtered = [WordAccumulator(vocabulary, length) for length in min_lengths]

    def update(self, tokens):
        segments = self.tokenizer.word_segments(tokens)
        words = [word for segment in segments for word in complete_words(segment)]
        return Text8Sample(
            text=self.tokenizer.decode(tokens),
            character_count=sum(len(segment) for segment in segments),
            special_token_count=sum(not 8 <= int(token) < 35 for token in tokens),
            words=self.words.update(words),
            by_min_length=[
                FilteredWordCounts(acc.min_length, acc.update(words))
                for acc in self.filtered
            ],
        )

    def compute(self):
        return Text8WordAggregate(
            self.words.compute(),
            [
                FilteredWordAggregate(acc.min_length, acc.compute())
                for acc in self.filtered
            ],
        )


class Text8Metrics:
    def __init__(
        self, tokenizer, vocabulary, min_lengths, reference_split, vocabulary_sha256
    ):
        self.generated = SampleAccumulator(vocabulary, min_lengths, tokenizer)
        self.held_out = SampleAccumulator(vocabulary, min_lengths, tokenizer)
        self.reference_split = reference_split
        self.vocabulary_sha256 = vocabulary_sha256
        self.vocabulary_size = len(vocabulary)
        self.records: list[Text8Record] = []

    def update(self, samples, examples, indices):
        for sample, example, index in zip(samples, examples, indices):
            if len(sample) != len(example["input_ids"]):
                raise ValueError(
                    "generated and held-out token windows must have equal length"
                )
            self.records.append(
                Text8Record(
                    index,
                    self.generated.update(sample),
                    self.held_out.update(example["input_ids"]),
                )
            )

    def compute(self) -> Text8Aggregate:
        generated = self.generated.compute()
        return Text8Aggregate(
            len(self.records),
            generated.words.rates.valid,
            generated,
            self.held_out.compute(),
            self.reference_split,
            "train",
            self.vocabulary_sha256,
            self.vocabulary_size,
            list(self.records),
        )
