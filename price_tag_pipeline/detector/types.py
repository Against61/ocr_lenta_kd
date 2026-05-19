from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    red_bbox: tuple[int, int, int, int]
    score: float
    white_ratio: float
    upper_white_ratio: float
    red_ratio: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

