from __future__ import annotations

import unittest

import numpy as np

from price_tag_pipeline.tracker.models import Observation, Track
from price_tag_pipeline.tracker.tracker import mark_seen_counted, track_is_counted, track_is_valid


def make_observation(frame: int, x: int) -> Observation:
    return Observation(
        frame=frame,
        time_sec=frame / 10.0,
        bbox=(x, 5, 30, 20),
        red_bbox=(x + 2, 8, 10, 8),
        score=0.9,
        white_ratio=0.5,
        upper_white_ratio=0.5,
        red_ratio=0.2,
        sharpness=100.0,
        crop_quality=4.0 + frame,
    )


class TrackerTests(unittest.TestCase):
    def test_line_mode_counts_crossing(self) -> None:
        track = Track(track_id=7)
        crop = np.zeros((20, 30, 3), dtype=np.uint8)
        track.add(
            make_observation(0, 0),
            crop,
            (0, 0, 30, 20),
            crossing_x=20.0,
            count_mode="line",
            count_first_at_line=False,
            line_entry_margin=0.0,
        )
        track.add(
            make_observation(1, 20),
            crop,
            (20, 0, 30, 20),
            crossing_x=20.0,
            count_mode="line",
            count_first_at_line=False,
            line_entry_margin=0.0,
        )

        self.assertEqual(track.counted_frame, 1)

    def test_seen_mode_track_becomes_counted_when_marked(self) -> None:
        track = Track(track_id=1)
        track.observations = [make_observation(0, 0), make_observation(1, 15)]
        track.best_observation = track.observations[-1]

        self.assertTrue(track_is_valid(track, 2, 10.0, 0.1, 0.5, 100))
        mark_seen_counted(track)

        self.assertTrue(track_is_counted(track, 2, 10.0, 0.1, 0.5, 100))
        self.assertEqual(track.counted_frame, 0)


if __name__ == "__main__":
    unittest.main()
