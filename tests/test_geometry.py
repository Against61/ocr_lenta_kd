from __future__ import annotations

import unittest

import numpy as np

from price_tag_pipeline.core.geometry import box_iou, nms_xyxy, scale_box


class GeometryTests(unittest.TestCase):
    def test_scale_box_rescales_and_clamps(self) -> None:
        self.assertEqual(scale_box((20, 10, 40, 20), 0.5, 100, 80), (40, 20, 60, 40))

    def test_box_iou_returns_overlap_ratio(self) -> None:
        iou = box_iou((0, 0, 10, 10), (5, 5, 10, 10))
        self.assertAlmostEqual(iou, 25 / 175)

    def test_nms_keeps_best_non_overlapping_boxes(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [20.0, 20.0, 30.0, 30.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        kept = nms_xyxy(boxes, scores, 0.5)

        self.assertEqual(kept, [0, 2])


if __name__ == "__main__":
    unittest.main()
