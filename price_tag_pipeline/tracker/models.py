from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    frame: int
    time_sec: float
    bbox: tuple[int, int, int, int]
    red_bbox: tuple[int, int, int, int]
    score: float
    white_ratio: float
    upper_white_ratio: float
    red_ratio: float
    sharpness: float
    crop_quality: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0


@dataclass
class CropCandidate:
    observation: Observation
    crop: np.ndarray
    crop_bbox_full: tuple[int, int, int, int]
    deskew_angle: float = 0.0
    deskew_confidence: float = 0.0
    deskew_applied: bool = False
    deskew_method: str = "none"
    quality_label: int | None = None
    quality_score: float | None = None
    quality_passed: bool | None = None


@dataclass
class Track:
    track_id: int
    observations: list[Observation] = field(default_factory=list)
    missed: int = 0
    counted_frame: int | None = None
    counted_time_sec: float | None = None
    best_observation: Observation | None = None
    best_crop: np.ndarray | None = None
    best_crop_bbox_full: tuple[int, int, int, int] | None = None
    crop_candidates: list[CropCandidate] = field(default_factory=list)
    selected_crop_rank: int = 0
    selected_quality_label: int | None = None
    selected_quality_score: float | None = None
    selected_quality_passed: bool | None = None
    selected_deskew_angle: float = 0.0
    selected_deskew_confidence: float = 0.0
    selected_deskew_applied: bool = False
    selected_deskew_method: str = "none"
    quality_attempts: int = 0
    rejected_quality_candidates: int = 0

    @property
    def last_observation(self) -> Observation:
        return self.observations[-1]

    @property
    def center(self) -> tuple[float, float]:
        return self.last_observation.center

    def add(
        self,
        observation: Observation,
        crop_full: np.ndarray,
        crop_bbox_full: tuple[int, int, int, int],
        crossing_x: float,
        count_mode: str,
        count_first_at_line: bool,
        line_entry_margin: float,
        max_crop_candidates: int = 8,
    ) -> None:
        previous = self.observations[-1] if self.observations else None
        self.observations.append(observation)
        self.missed = 0

        if count_mode == "line" and self.counted_frame is None:
            x, _, w, _ = observation.bbox
            curr_x = observation.center[0]
            if previous is None:
                intersects_line = x <= crossing_x <= x + w
                first_seen_near_line = curr_x <= crossing_x + line_entry_margin
                should_count = count_first_at_line and (intersects_line or first_seen_near_line)
            else:
                prev_x = previous.center[0]
                should_count = prev_x < crossing_x <= curr_x

            if should_count:
                self.counted_frame = observation.frame
                self.counted_time_sec = observation.time_sec

        candidate = CropCandidate(
            observation=observation,
            crop=crop_full.copy(),
            crop_bbox_full=crop_bbox_full,
        )
        self.crop_candidates.append(candidate)
        self.crop_candidates.sort(key=lambda item: item.observation.crop_quality, reverse=True)
        if max_crop_candidates > 0:
            del self.crop_candidates[max_crop_candidates:]

        if self.crop_candidates:
            self.select_crop_candidate(self.crop_candidates[0], rank=1)

    def select_crop_candidate(self, candidate: CropCandidate, rank: int) -> None:
        self.best_observation = candidate.observation
        self.best_crop = candidate.crop
        self.best_crop_bbox_full = candidate.crop_bbox_full
        self.selected_crop_rank = rank
        self.selected_quality_label = candidate.quality_label
        self.selected_quality_score = candidate.quality_score
        self.selected_quality_passed = candidate.quality_passed
        self.selected_deskew_angle = candidate.deskew_angle
        self.selected_deskew_confidence = candidate.deskew_confidence
        self.selected_deskew_applied = candidate.deskew_applied
        self.selected_deskew_method = candidate.deskew_method
