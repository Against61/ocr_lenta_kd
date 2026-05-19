from __future__ import annotations

import cv2
import numpy as np

from price_tag_pipeline.tracker.models import Track
from price_tag_pipeline.tracker.tracker import track_is_valid


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return (
        int((53 * track_id + 70) % 255),
        int((97 * track_id + 120) % 255),
        int((193 * track_id + 40) % 255),
    )


def draw_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    crossing_x: float,
    count_mode: str,
    counted_ids: set[int],
    visible_counted: int,
    min_track_frames: int,
    min_delta_x: float,
    min_track_white_ratio: float,
    min_track_aspect: float,
    max_track_height: int,
) -> np.ndarray:
    overlay = frame.copy()
    if count_mode == "line":
        cv2.line(overlay, (int(crossing_x), 0), (int(crossing_x), frame.shape[0]), (255, 255, 255), 2)
        status_text = f"counted: {len(counted_ids)}"
    else:
        status_text = f"visible: {visible_counted} unique: {len(counted_ids)}"

    cv2.putText(overlay, status_text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(overlay, status_text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 1, cv2.LINE_AA)

    for track in tracks:
        if not track.observations:
            continue
        obs = track.last_observation
        x, y, w, h = obs.bbox
        color = color_for_track(track.track_id)
        valid = track_is_valid(
            track,
            min_track_frames,
            min_delta_x,
            min_track_white_ratio,
            min_track_aspect,
            max_track_height,
        )
        thickness = 3 if track.track_id in counted_ids else 2
        if not valid:
            color = (110, 110, 110)
            thickness = 1
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
        label = f"#{track.track_id}"
        if track.track_id in counted_ids:
            label += " C"
        cv2.putText(overlay, label, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        points = [(int(item.center[0]), int(item.center[1])) for item in track.observations[-36:]]
        for p1, p2 in zip(points, points[1:]):
            cv2.line(overlay, p1, p2, color, 2)
    return overlay

