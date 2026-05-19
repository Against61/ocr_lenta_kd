from __future__ import annotations

from typing import Protocol

from price_tag_pipeline.quality.classifier import CropQualityResult
from price_tag_pipeline.tracker.models import CropCandidate, Track


class CropQualityClassifier(Protocol):
    enabled: bool
    model_path: str

    def predict(self, crop) -> CropQualityResult:
        ...


def select_track_crops(
    tracks: list[Track],
    classifier: CropQualityClassifier,
) -> dict[str, int]:
    summary = {
        "tracks_evaluated": 0,
        "tracks_with_passed_crop": 0,
        "tracks_without_passed_crop": 0,
        "candidate_predictions": 0,
        "fallback_best_quality": 0,
    }

    for track in tracks:
        candidates = sorted(track.crop_candidates, key=lambda item: item.observation.crop_quality, reverse=True)
        if not candidates:
            continue

        if not classifier.enabled:
            track.select_crop_candidate(candidates[0], rank=1)
            continue

        summary["tracks_evaluated"] += 1
        selected: tuple[int, CropCandidate] | None = None
        best_failed: tuple[float, int, CropCandidate] | None = None

        for rank, candidate in enumerate(candidates, start=1):
            result = classifier.predict(candidate.crop)
            candidate.quality_label = result.label
            candidate.quality_score = result.score
            candidate.quality_passed = result.passed
            summary["candidate_predictions"] += 1

            if result.passed:
                selected = (rank, candidate)
                break

            failed_key = (result.score, -rank, candidate)
            if best_failed is None or failed_key[:2] > best_failed[:2]:
                best_failed = failed_key

        if selected is not None:
            rank, candidate = selected
            track.quality_attempts = rank
            track.rejected_quality_candidates = rank - 1
            track.select_crop_candidate(candidate, rank=rank)
            summary["tracks_with_passed_crop"] += 1
            continue

        if best_failed is not None:
            _, negative_rank, candidate = best_failed
            rank = -negative_rank
            track.quality_attempts = len(candidates)
            track.rejected_quality_candidates = len(candidates)
            track.select_crop_candidate(candidate, rank=rank)
            summary["tracks_without_passed_crop"] += 1
            summary["fallback_best_quality"] += 1

    return summary
