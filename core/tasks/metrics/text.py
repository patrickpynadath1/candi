"""Word counts against a fixed reference vocabulary, with explicit length filters."""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class WordCounts:
    total: int
    valid: int
    unique: int
    unique_valid: int


@dataclass(frozen=True)
class WordMeans:
    total: float
    valid: float
    unique: float
    unique_valid: float


@dataclass(frozen=True)
class WordRates:
    valid: float | None
    unique: float | None
    unique_valid: float | None


@dataclass(frozen=True)
class FilteredWordCounts:
    min_length: int
    counts: WordCounts


@dataclass(frozen=True)
class WordAggregate:
    counts: WordCounts
    mean_per_sample: WordMeans
    rates: WordRates


@dataclass(frozen=True)
class FilteredWordAggregate:
    min_length: int
    aggregate: WordAggregate


def complete_words(text):
    """Discard potentially truncated words at either end of a character window."""
    start = 0 if text.startswith(" ") else text.find(" ")
    end = len(text) if text.endswith(" ") else text.rfind(" ")
    if start < 0 or end <= start:
        return []
    return text[start:end].split()


def count_words(words, vocabulary, min_length=1):
    counts = Counter(word for word in words if len(word) >= min_length)
    valid = counts.keys() & vocabulary
    return WordCounts(
        total=sum(counts.values()),
        valid=sum(counts[word] for word in valid),
        unique=len(counts),
        unique_valid=len(valid),
    )


class WordAccumulator:
    def __init__(self, vocabulary, min_length=1):
        self.vocabulary, self.min_length = vocabulary, min_length
        self.counts = Counter()
        self.samples = 0
        self.unique_sum = self.unique_valid_sum = 0

    def update(self, words):
        words = [word for word in words if len(word) >= self.min_length]
        sample = count_words(words, self.vocabulary)
        self.counts.update(words)
        self.samples += 1
        self.unique_sum += sample.unique
        self.unique_valid_sum += sample.unique_valid
        return sample

    def compute(self):
        if not self.samples:
            raise ValueError("cannot aggregate zero samples")
        valid_words = self.counts.keys() & self.vocabulary
        counts = WordCounts(
            sum(self.counts.values()),
            sum(self.counts[word] for word in valid_words),
            len(self.counts),
            len(valid_words),
        )
        return WordAggregate(
            counts,
            WordMeans(
                counts.total / self.samples,
                counts.valid / self.samples,
                self.unique_sum / self.samples,
                self.unique_valid_sum / self.samples,
            ),
            WordRates(
                counts.valid / counts.total if counts.total else None,
                counts.unique / counts.total if counts.total else None,
                counts.unique_valid / counts.unique if counts.unique else None,
            ),
        )
