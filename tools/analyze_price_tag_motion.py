#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Track:
    track_id: int
    last_frame: int
    last_center: tuple[float, float]
    boxes: list[tuple[int, int, int, int, int, float]] = field(default_factory=list)
    missed: int = 0

    def add(self, frame_idx: int, box: tuple[int, int, int, int], score: float) -> None:
        x, y, w, h = box
        self.last_frame = frame_idx
        self.last_center = (x + w / 2.0, y + h / 2.0)
        self.boxes.append((frame_idx, x, y, w, h, score))
        self.missed = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--max-link-distance", type=float, default=80.0)
    parser.add_argument("--max-missed", type=int, default=6)
    parser.add_argument("--rotate-cw", action="store_true")
    parser.add_argument("--rotate-ccw", action="store_true")
    return parser.parse_args()


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def normalize_orientation(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.rotate_cw and args.rotate_ccw:
        raise ValueError("Use only one of --rotate-cw or --rotate-ccw")
    if args.rotate_cw:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if args.rotate_ccw:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def price_red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 35, 105], dtype=np.uint8)
    upper_red_1 = np.array([16, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([168, 35, 105], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_red_label_parts(frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    mask = price_red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[tuple[tuple[int, int, int, int], float]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 110 or area > 8000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 9 or h < 9:
            continue
        extent = area / float(w * h)
        aspect = w / float(h)
        if extent < 0.22:
            continue
        if aspect < 0.18 or aspect > 5.5:
            continue

        roi = mask[y : y + h, x : x + w]
        red_fill = float(np.count_nonzero(roi)) / float(w * h)
        score = min(1.0, 0.5 * extent + 0.5 * red_fill)
        detections.append(((x, y, w, h), score))

    return detections


def greedy_update_tracks(
    tracks: list[Track],
    detections: list[tuple[tuple[int, int, int, int], float]],
    frame_idx: int,
    next_track_id: int,
    max_link_distance: float,
    max_missed: int,
) -> int:
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))
    pairs: list[tuple[float, int, int]] = []

    for track_idx, track in enumerate(tracks):
        for det_idx, (box, _) in enumerate(detections):
            x, y, w, h = box
            cx = x + w / 2.0
            cy = y + h / 2.0
            dist = float(np.hypot(cx - track.last_center[0], cy - track.last_center[1]))
            if dist <= max_link_distance:
                pairs.append((dist, track_idx, det_idx))

    for _, track_idx, det_idx in sorted(pairs, key=lambda item: item[0]):
        if track_idx not in unmatched_tracks or det_idx not in unmatched_detections:
            continue
        box, score = detections[det_idx]
        tracks[track_idx].add(frame_idx, box, score)
        unmatched_tracks.remove(track_idx)
        unmatched_detections.remove(det_idx)

    for track_idx in unmatched_tracks:
        tracks[track_idx].missed += 1

    tracks[:] = [track for track in tracks if track.missed <= max_missed]

    for det_idx in unmatched_detections:
        box, score = detections[det_idx]
        x, y, w, h = box
        track = Track(
            track_id=next_track_id,
            last_frame=frame_idx,
            last_center=(x + w / 2.0, y + h / 2.0),
        )
        track.add(frame_idx, box, score)
        tracks.append(track)
        next_track_id += 1

    return next_track_id


def draw_overlay(frame: np.ndarray, tracks: list[Track], min_len: int = 4) -> np.ndarray:
    overlay = frame.copy()
    for track in tracks:
        if len(track.boxes) < min_len:
            continue
        frame_idx, x, y, w, h, _ = track.boxes[-1]
        color = (
            int((37 * track.track_id) % 255),
            int((97 * track.track_id) % 255),
            int((171 * track.track_id) % 255),
        )
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            overlay,
            f"#{track.track_id}",
            (x, max(14, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        pts = []
        for _, bx, by, bw, bh, _ in track.boxes[-20:]:
            pts.append((int(bx + bw / 2), int(by + bh / 2)))
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(overlay, p1, p2, color, 2)
    return overlay


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def median_feature_flow(prev_gray: np.ndarray, gray: np.ndarray) -> tuple[float, float, int]:
    points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=900, qualityLevel=0.01, minDistance=8)
    if points is None:
        return 0.0, 0.0, 0

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
    if next_points is None or status is None:
        return 0.0, 0.0, 0

    status = status.reshape(-1).astype(bool)
    p0 = points.reshape(-1, 2)[status]
    p1 = next_points.reshape(-1, 2)[status]
    if len(p0) < 20:
        return 0.0, 0.0, int(len(p0))

    flow = p1 - p0
    median = np.median(flow, axis=0)
    deviation = np.linalg.norm(flow - median, axis=1)
    mad = np.median(np.abs(deviation - np.median(deviation))) + 1e-6
    keep = deviation < np.median(deviation) + 3.0 * mad
    filtered = flow[keep]
    if len(filtered) < 20:
        filtered = flow

    median = np.median(filtered, axis=0)
    return float(median[0]), float(median[1]), int(len(filtered))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    active_tracks: list[Track] = []
    finished_tracks: list[Track] = []
    next_track_id = 1
    prev_gray = None
    motion_rows = []
    flow_rows = []
    detection_rows = []
    overlay_frames = {}
    requested_overlays = {0, 30, 60, 90, 120, 150, 180, 210, 240, 270, frame_count - 1}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.sample_step != 0:
            frame_idx += 1
            continue

        frame = normalize_orientation(frame, args)
        frame = resize_frame(frame, args.width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0
        flow_x = 0.0
        flow_y = 0.0
        flow_points = 0
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = float(np.mean(diff))
            flow_x, flow_y, flow_points = median_feature_flow(prev_gray, gray)
        prev_gray = gray

        detections = detect_red_label_parts(frame)
        before_ids = {track.track_id for track in active_tracks}
        next_track_id = greedy_update_tracks(
            active_tracks,
            detections,
            frame_idx,
            next_track_id,
            args.max_link_distance,
            args.max_missed,
        )
        after_ids = {track.track_id for track in active_tracks}
        lost_ids = before_ids - after_ids
        if lost_ids:
            # Finished tracks remain available in per-track CSV output.
            pass

        for box, score in detections:
            x, y, w, h = box
            detection_rows.append(
                {
                    "frame": frame_idx,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "score": round(score, 4),
                }
            )

        motion_rows.append(
            {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3) if fps else "",
                "motion_mean_absdiff": round(motion, 4),
                "red_detections": len(detections),
                "sharpness": round(laplacian_sharpness(gray), 2),
            }
        )
        flow_rows.append(
            {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3) if fps else "",
                "median_flow_x": round(flow_x, 4),
                "median_flow_y": round(flow_y, 4),
                "tracked_points": flow_points,
            }
        )

        if frame_idx in requested_overlays:
            overlay_frames[frame_idx] = draw_overlay(frame, active_tracks)

        finished_tracks.extend([track for track in active_tracks if track.missed > args.max_missed])
        frame_idx += 1

    cap.release()
    all_tracks = active_tracks + finished_tracks
    # Deduplicate by id because tracks that never expired live only in active_tracks.
    by_id = {track.track_id: track for track in all_tracks}
    all_tracks = list(by_id.values())

    with (args.out_dir / "motion_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "time_sec", "motion_mean_absdiff", "red_detections", "sharpness"],
        )
        writer.writeheader()
        writer.writerows(motion_rows)

    with (args.out_dir / "red_detections.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "x", "y", "w", "h", "score"])
        writer.writeheader()
        writer.writerows(detection_rows)

    with (args.out_dir / "global_flow.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time_sec", "median_flow_x", "median_flow_y", "tracked_points"])
        writer.writeheader()
        writer.writerows(flow_rows)

    track_rows = []
    for track in sorted(all_tracks, key=lambda item: item.track_id):
        if not track.boxes:
            continue
        first = track.boxes[0]
        last = track.boxes[-1]
        xs = [x + w / 2.0 for _, x, y, w, h, _ in track.boxes]
        ys = [y + h / 2.0 for _, x, y, w, h, _ in track.boxes]
        scores = [score for *_, score in track.boxes]
        track_rows.append(
            {
                "track_id": track.track_id,
                "frames_seen": len(track.boxes),
                "first_frame": first[0],
                "last_frame": last[0],
                "first_x": round(xs[0], 2),
                "last_x": round(xs[-1], 2),
                "delta_x": round(xs[-1] - xs[0], 2),
                "first_y": round(ys[0], 2),
                "last_y": round(ys[-1], 2),
                "delta_y": round(ys[-1] - ys[0], 2),
                "mean_score": round(float(np.mean(scores)), 4),
            }
        )

    with (args.out_dir / "red_tracks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "frames_seen",
                "first_frame",
                "last_frame",
                "first_x",
                "last_x",
                "delta_x",
                "first_y",
                "last_y",
                "delta_y",
                "mean_score",
            ],
        )
        writer.writeheader()
        writer.writerows(track_rows)

    for frame_idx, overlay in overlay_frames.items():
        cv2.imwrite(str(args.out_dir / f"overlay_{frame_idx:04d}.jpg"), overlay)

    persistent_tracks = [row for row in track_rows if row["frames_seen"] >= 8]
    moving_tracks = [row for row in persistent_tracks if abs(row["delta_x"]) >= 30 or abs(row["delta_y"]) >= 30]
    report = {
        "video": str(args.video),
        "fps": fps,
        "frame_count": frame_count,
        "analyzed_width": args.width,
        "rotate_cw": args.rotate_cw,
        "rotate_ccw": args.rotate_ccw,
        "frames_analyzed": len(motion_rows),
        "mean_motion": round(float(np.mean([row["motion_mean_absdiff"] for row in motion_rows[1:]])), 4)
        if len(motion_rows) > 1
        else 0.0,
        "max_motion": max([row["motion_mean_absdiff"] for row in motion_rows], default=0.0),
        "mean_red_detections_per_frame": round(
            float(np.mean([row["red_detections"] for row in motion_rows])), 2
        )
        if motion_rows
        else 0.0,
        "all_tracks": len(track_rows),
        "persistent_tracks_min_8_frames": len(persistent_tracks),
        "moving_persistent_tracks": len(moving_tracks),
        "mean_feature_flow_when_moving": {
            "x": round(
                float(
                    np.mean(
                        [
                            row["median_flow_x"]
                            for row, motion_row in zip(flow_rows, motion_rows)
                            if motion_row["motion_mean_absdiff"] > 2.0
                        ]
                    )
                ),
                4,
            )
            if any(row["motion_mean_absdiff"] > 2.0 for row in motion_rows)
            else 0.0,
            "y": round(
                float(
                    np.mean(
                        [
                            row["median_flow_y"]
                            for row, motion_row in zip(flow_rows, motion_rows)
                            if motion_row["motion_mean_absdiff"] > 2.0
                        ]
                    )
                ),
                4,
            )
            if any(row["motion_mean_absdiff"] > 2.0 for row in motion_rows)
            else 0.0,
        },
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
