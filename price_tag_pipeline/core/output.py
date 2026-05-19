from __future__ import annotations

import csv
from pathlib import Path

from price_tag_pipeline.tracker.models import Track
from price_tag_pipeline.tracker.tracker import track_is_counted, track_is_valid


def write_track_history(path: Path, tracks: list[Track]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "frame",
                "time_sec",
                "x",
                "y",
                "w",
                "h",
                "red_x",
                "red_y",
                "red_w",
                "red_h",
                "score",
                "white_ratio",
                "upper_white_ratio",
                "red_ratio",
                "sharpness",
                "crop_quality",
            ],
        )
        writer.writeheader()
        for track in sorted(tracks, key=lambda item: item.track_id):
            for obs in track.observations:
                x, y, w, h = obs.bbox
                rx, ry, rw, rh = obs.red_bbox
                writer.writerow(
                    {
                        "track_id": track.track_id,
                        "frame": obs.frame,
                        "time_sec": round(obs.time_sec, 3),
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "red_x": rx,
                        "red_y": ry,
                        "red_w": rw,
                        "red_h": rh,
                        "score": round(obs.score, 4),
                        "white_ratio": round(obs.white_ratio, 4),
                        "upper_white_ratio": round(obs.upper_white_ratio, 4),
                        "red_ratio": round(obs.red_ratio, 4),
                        "sharpness": round(obs.sharpness, 2),
                        "crop_quality": round(obs.crop_quality, 4),
                    }
                )


def write_tracks_csv(
    path: Path,
    tracks: list[Track],
    crop_dir: Path,
    min_track_frames: int,
    min_delta_x: float,
    min_track_white_ratio: float,
    min_track_aspect: float,
    max_track_height: int,
    counted_only: bool,
    prefiltered: bool = False,
    saved_crop_paths: set[Path] | None = None,
    source_width: int = 0,
    source_height: int = 0,
    rotation: str = "none",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for track in sorted(tracks, key=lambda item: item.track_id):
        if not prefiltered and counted_only and not track_is_counted(
            track,
            min_track_frames,
            min_delta_x,
            min_track_white_ratio,
            min_track_aspect,
            max_track_height,
        ):
            continue
        if not prefiltered and not counted_only and not track_is_valid(
            track,
            min_track_frames,
            min_delta_x,
            min_track_white_ratio,
            min_track_aspect,
            max_track_height,
        ):
            continue
        first = track.observations[0]
        last = track.observations[-1]
        best = track.best_observation or last
        x, y, w, h = best.bbox
        full_box = track.best_crop_bbox_full or (0, 0, 0, 0)
        source_box = source_box_from_normalized_box(full_box, source_width, source_height, rotation)
        crop_path = crop_dir / f"track_{track.track_id:04d}_frame_{best.frame:04d}.jpg"
        crop_saved = saved_crop_paths is None or crop_path in saved_crop_paths
        rows.append(
            {
                "track_id": track.track_id,
                "counted": int(
                    track_is_counted(
                        track,
                        min_track_frames,
                        min_delta_x,
                        min_track_white_ratio,
                        min_track_aspect,
                        max_track_height,
                    )
                ),
                "counted_frame": track.counted_frame if track.counted_frame is not None else "",
                "counted_time_sec": round(track.counted_time_sec, 3) if track.counted_time_sec is not None else "",
                "first_frame": first.frame,
                "first_time_sec": round(first.time_sec, 3),
                "last_frame": last.frame,
                "last_time_sec": round(last.time_sec, 3),
                "frames_seen": len(track.observations),
                "first_x": round(first.center[0], 2),
                "last_x": round(last.center[0], 2),
                "delta_x": round(last.center[0] - first.center[0], 2),
                "first_y": round(first.center[1], 2),
                "last_y": round(last.center[1], 2),
                "delta_y": round(last.center[1] - first.center[1], 2),
                "best_frame": best.frame,
                "best_time_sec": round(best.time_sec, 3),
                "best_x": x,
                "best_y": y,
                "best_w": w,
                "best_h": h,
                "best_full_x": full_box[0],
                "best_full_y": full_box[1],
                "best_full_w": full_box[2],
                "best_full_h": full_box[3],
                "source_x_min": source_box[0],
                "source_y_min": source_box[1],
                "source_x_max": source_box[0] + source_box[2],
                "source_y_max": source_box[1] + source_box[3],
                "best_score": round(best.score, 4),
                "best_white_ratio": round(best.white_ratio, 4),
                "best_upper_white_ratio": round(best.upper_white_ratio, 4),
                "best_red_ratio": round(best.red_ratio, 4),
                "best_sharpness": round(best.sharpness, 2),
                "selected_crop_rank": track.selected_crop_rank,
                "deskew_applied": int(track.selected_deskew_applied),
                "deskew_angle": round(track.selected_deskew_angle, 3),
                "deskew_confidence": round(track.selected_deskew_confidence, 4),
                "deskew_method": track.selected_deskew_method,
                "quality_label": track.selected_quality_label if track.selected_quality_label is not None else "",
                "quality_score": round(track.selected_quality_score, 4)
                if track.selected_quality_score is not None
                else "",
                "quality_passed": int(track.selected_quality_passed)
                if track.selected_quality_passed is not None
                else "",
                "quality_attempts": track.quality_attempts,
                "rejected_quality_candidates": track.rejected_quality_candidates,
                "crop_candidates": len(track.crop_candidates),
                "crop_saved": int(crop_saved),
                "crop_path": str(crop_path) if crop_saved else "",
            }
        )

    fieldnames = list(rows[0].keys()) if rows else [
        "track_id",
        "counted",
        "counted_frame",
        "counted_time_sec",
        "first_frame",
        "first_time_sec",
        "last_frame",
        "last_time_sec",
        "frames_seen",
        "first_x",
        "last_x",
        "delta_x",
        "first_y",
        "last_y",
        "delta_y",
        "best_frame",
        "best_time_sec",
        "best_x",
        "best_y",
        "best_w",
        "best_h",
        "best_full_x",
        "best_full_y",
        "best_full_w",
        "best_full_h",
        "source_x_min",
        "source_y_min",
        "source_x_max",
        "source_y_max",
        "best_score",
        "best_white_ratio",
        "best_upper_white_ratio",
        "best_red_ratio",
        "best_sharpness",
        "selected_crop_rank",
        "deskew_applied",
        "deskew_angle",
        "deskew_confidence",
        "deskew_method",
        "quality_label",
        "quality_score",
        "quality_passed",
        "quality_attempts",
        "rejected_quality_candidates",
        "crop_candidates",
        "crop_saved",
        "crop_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def source_box_from_normalized_box(
    box: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
    rotation: str,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    if source_width <= 0 or source_height <= 0:
        return box

    if rotation == "ccw":
        x1 = source_width - (y + h)
        y1 = x
        x2 = source_width - y
        y2 = x + w
    elif rotation == "cw":
        x1 = y
        y1 = source_height - (x + w)
        x2 = y + h
        y2 = source_height - x
    else:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

    x1 = max(0, min(source_width - 1, int(round(x1))))
    y1 = max(0, min(source_height - 1, int(round(y1))))
    x2 = max(0, min(source_width, int(round(x2))))
    y2 = max(0, min(source_height, int(round(y2))))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)
