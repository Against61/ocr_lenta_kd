from __future__ import annotations

import numpy as np

from price_tag_pipeline.core.geometry import scale_box
from price_tag_pipeline.core.image import crop, laplacian_sharpness
from price_tag_pipeline.detector.types import Detection
from price_tag_pipeline.tracker.models import Observation, Track


def build_observation(
    detection: Detection,
    frame_idx: int,
    fps: float,
    frame_small: np.ndarray,
) -> Observation:
    patch = crop(frame_small, detection.bbox)
    sharp = laplacian_sharpness(patch)
    cx, _ = detection.center
    quality_x = frame_small.shape[1] * 0.5
    center_bonus = max(0.0, 1.0 - abs(cx - quality_x) / max(1.0, quality_x))
    x, y, bw, bh = detection.bbox
    area_bonus = min(1.0, (bw * bh) / 4500.0)
    quality = detection.score * 2.0 + min(2.0, sharp / 1200.0) + center_bonus + area_bonus
    return Observation(
        frame=frame_idx,
        time_sec=frame_idx / fps if fps else 0.0,
        bbox=detection.bbox,
        red_bbox=detection.red_bbox,
        score=detection.score,
        white_ratio=detection.white_ratio,
        upper_white_ratio=detection.upper_white_ratio,
        red_ratio=detection.red_ratio,
        sharpness=sharp,
        crop_quality=quality,
    )


def filter_right_edge_detections(
    detections: list[Detection],
    frame_width: int,
    right_edge_ignore_ratio: float,
) -> list[Detection]:
    if right_edge_ignore_ratio >= 1.0:
        return detections
    right_edge_x = frame_width * right_edge_ignore_ratio
    return [det for det in detections if det.bbox[0] + det.bbox[2] < right_edge_x]


def assign_detections(
    active_tracks: list[Track],
    detections: list[Detection],
    frame_idx: int,
    fps: float,
    frame_small: np.ndarray,
    frame_full: np.ndarray,
    scale: float,
    crossing_x: float,
    next_track_id: int,
    max_link_distance: float,
    max_missed: int,
    max_backward_step: float,
    max_y_jump: float,
    right_edge_retire_ratio: float,
    right_edge_ignore_ratio: float,
    count_mode: str,
    count_first_at_line: bool,
    line_entry_margin: float,
    max_crop_candidates: int,
) -> tuple[int, list[Track]]:
    finished: list[Track] = []
    right_edge_x = frame_small.shape[1] * right_edge_retire_ratio
    retained_tracks: list[Track] = []
    for track in active_tracks:
        x, _, w, _ = track.last_observation.bbox
        center_x = x + w / 2.0
        if len(track.observations) >= 2 and (x + w >= right_edge_x or center_x >= right_edge_x):
            finished.append(track)
        else:
            retained_tracks.append(track)
    active_tracks[:] = retained_tracks

    detections = filter_right_edge_detections(detections, frame_small.shape[1], right_edge_ignore_ratio)
    unmatched_tracks = set(range(len(active_tracks)))
    unmatched_detections = set(range(len(detections)))
    pairs: list[tuple[float, int, int]] = []

    for track_idx, track in enumerate(active_tracks):
        tx, ty = track.center
        _, _, track_w, track_h = track.last_observation.bbox
        for det_idx, detection in enumerate(detections):
            dx, dy = detection.center
            step_x = dx - tx
            step_y = dy - ty
            if step_x < -max_backward_step:
                continue
            if abs(step_y) > max(max_y_jump, track_h * 0.85):
                continue
            dist = float(np.hypot(step_x, step_y))
            if dist <= max_link_distance:
                area_prev = track.last_observation.bbox[2] * track.last_observation.bbox[3]
                area_curr = detection.bbox[2] * detection.bbox[3]
                area_penalty = abs(np.log((area_curr + 1.0) / (area_prev + 1.0))) * 8.0
                backward_penalty = max(0.0, -step_x) * 6.0
                vertical_penalty = abs(step_y) * 0.35
                width_penalty = abs(detection.bbox[2] - track_w) * 0.08
                pairs.append((dist + area_penalty + backward_penalty + vertical_penalty + width_penalty, track_idx, det_idx))

    for _, track_idx, det_idx in sorted(pairs, key=lambda item: item[0]):
        if track_idx not in unmatched_tracks or det_idx not in unmatched_detections:
            continue
        detection = detections[det_idx]
        observation = build_observation(detection, frame_idx, fps, frame_small)
        full_box = scale_box(detection.bbox, scale, frame_full.shape[1], frame_full.shape[0])
        track = active_tracks[track_idx]
        track.add(
            observation,
            crop(frame_full, full_box),
            full_box,
            crossing_x,
            count_mode,
            count_first_at_line,
            line_entry_margin,
            max_crop_candidates,
        )
        unmatched_tracks.remove(track_idx)
        unmatched_detections.remove(det_idx)

    for track_idx in sorted(unmatched_tracks, reverse=True):
        active_tracks[track_idx].missed += 1
        if active_tracks[track_idx].missed > max_missed:
            finished.append(active_tracks.pop(track_idx))

    for det_idx in unmatched_detections:
        detection = detections[det_idx]
        observation = build_observation(detection, frame_idx, fps, frame_small)
        full_box = scale_box(detection.bbox, scale, frame_full.shape[1], frame_full.shape[0])
        track = Track(track_id=next_track_id)
        track.add(
            observation,
            crop(frame_full, full_box),
            full_box,
            crossing_x,
            count_mode,
            count_first_at_line,
            line_entry_margin,
            max_crop_candidates,
        )
        active_tracks.append(track)
        next_track_id += 1

    return next_track_id, finished


def track_is_valid(
    track: Track,
    min_track_frames: int,
    min_delta_x: float,
    min_track_white_ratio: float,
    min_track_aspect: float,
    max_track_height: int,
) -> bool:
    if len(track.observations) < min_track_frames:
        return False
    first_x = track.observations[0].center[0]
    last_x = track.observations[-1].center[0]
    if (last_x - first_x) < min_delta_x:
        return False

    best = track.best_observation or track.last_observation
    _, _, bw, bh = best.bbox
    if max_track_height > 0 and bh > max_track_height:
        return False
    if (bw / float(bh)) < min_track_aspect:
        return False

    white_values = [obs.white_ratio for obs in track.observations]
    upper_white_values = [obs.upper_white_ratio for obs in track.observations]
    red_values = [obs.red_ratio for obs in track.observations]
    if float(np.median(white_values)) < min_track_white_ratio:
        return False
    if float(np.median(upper_white_values)) < min_track_white_ratio:
        return False
    if float(np.median(red_values)) > 0.58:
        return False
    return True


def track_is_counted(
    track: Track,
    min_track_frames: int,
    min_delta_x: float,
    min_track_white_ratio: float,
    min_track_aspect: float,
    max_track_height: int,
) -> bool:
    return track.counted_frame is not None and track_is_valid(
        track,
        min_track_frames,
        min_delta_x,
        min_track_white_ratio,
        min_track_aspect,
        max_track_height,
    )


def mark_seen_counted(track: Track) -> None:
    if track.counted_frame is not None or not track.observations:
        return
    first = track.observations[0]
    track.counted_frame = first.frame
    track.counted_time_sec = first.time_sec
