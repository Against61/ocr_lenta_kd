from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from price_tag_pipeline.core.output import write_track_history, write_tracks_csv
from price_tag_pipeline.tracker.models import Observation, Track


def make_track(track_id: int = 1) -> Track:
    track = Track(track_id=track_id)
    track.observations = [
        Observation(
            frame=0,
            time_sec=0.0,
            bbox=(10, 5, 30, 20),
            red_bbox=(12, 8, 10, 8),
            score=0.9,
            white_ratio=0.5,
            upper_white_ratio=0.5,
            red_ratio=0.2,
            sharpness=120.0,
            crop_quality=4.0,
        ),
        Observation(
            frame=1,
            time_sec=0.1,
            bbox=(20, 5, 30, 20),
            red_bbox=(22, 8, 10, 8),
            score=0.95,
            white_ratio=0.55,
            upper_white_ratio=0.55,
            red_ratio=0.22,
            sharpness=140.0,
            crop_quality=4.5,
        ),
    ]
    track.best_observation = track.observations[-1]
    track.best_crop_bbox_full = (40, 10, 60, 40)
    track.counted_frame = 0
    track.counted_time_sec = 0.0
    track.selected_crop_rank = 1
    return track


class OutputTests(unittest.TestCase):
    def test_write_track_history_includes_all_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "history.csv"
            write_track_history(csv_path, [make_track()])
            content = csv_path.read_text(encoding="utf-8")

        self.assertIn("track_id,frame", content)
        self.assertIn("1,0,0.0", content)
        self.assertIn("1,1,0.1", content)

    def test_write_tracks_csv_writes_header_and_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            crop_dir = out_dir / "crops"
            crop_dir.mkdir()
            csv_path = out_dir / "tracks.csv"

            rows = write_tracks_csv(
                csv_path,
                [make_track()],
                crop_dir,
                min_track_frames=1,
                min_delta_x=0.0,
                min_track_white_ratio=0.1,
                min_track_aspect=0.5,
                max_track_height=100,
                counted_only=False,
            )
            content = csv_path.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 1)
        self.assertIn("track_id,counted", content)
        self.assertEqual(rows[0]["crop_saved"], 1)
        self.assertTrue(str(rows[0]["crop_path"]).endswith("crops/track_0001_frame_0001.jpg"))
        self.assertIn("source_x_min", content)


if __name__ == "__main__":
    unittest.main()
