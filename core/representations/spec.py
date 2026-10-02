"""Small construction-time descriptions of representation geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RepresentationSpec:
    kind: str
    dimension: int
    vocab_size: int
    unit_normalized: bool


@dataclass(frozen=True)
class RepresentationRequirements:
    kinds: tuple[str, ...] | None = None
    dimension: int | None = None
    vocab_size: int | None = None
    unit_normalized: bool = False

    def validate(self, spec: RepresentationSpec, schedule_name: str):
        problems = []
        if self.kinds is not None and spec.kind not in self.kinds:
            problems.append(
                f"representation kind must be one of {self.kinds}, got {spec.kind}"
            )
        if self.dimension is not None and self.dimension != spec.dimension:
            problems.append(
                f"continuous dimension must be {self.dimension}, got {spec.dimension}"
            )
        if self.vocab_size is not None and self.vocab_size != spec.vocab_size:
            problems.append(
                f"vocabulary size must be {self.vocab_size}, got {spec.vocab_size}"
            )
        if self.unit_normalized and not spec.unit_normalized:
            problems.append("representation must enforce unit-normalized embeddings")
        if problems:
            raise ValueError(f"{schedule_name} is incompatible: " + "; ".join(problems))
