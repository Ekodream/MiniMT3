from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReadabilityRating:
    stem: str
    rhythm: int
    hand_split: int
    clutter: int
    comment: str = ""

    @property
    def mean_score(self) -> float:
        return (self.rhythm + self.hand_split + self.clutter) / 3.0
