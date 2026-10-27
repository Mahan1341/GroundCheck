"""Training-only majority sanity baseline; no evidence-based detection yet."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MajorityBaseline:
    label: int
    train_size: int
    train_positive_count: int

    @classmethod
    def fit(cls, labels: list[int]) -> "MajorityBaseline":
        if not labels or any(type(y) is not int or y not in (0, 1) for y in labels):
            raise ValueError("fit requires nonempty binary integer labels")
        positives = sum(labels)
        return cls(int(positives > len(labels) / 2), len(labels), positives)

    def predict(self, size: int) -> list[int]:
        if size < 0:
            raise ValueError("size must be nonnegative")
        return [self.label] * size
