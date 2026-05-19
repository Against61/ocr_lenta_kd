from __future__ import annotations

import unittest

import numpy as np

from price_tag_pipeline.codes.detector import decode_yolo_ncnn_output


class CodeDetectorTests(unittest.TestCase):
    def test_decode_yolo_ncnn_output_maps_box_to_original_crop(self) -> None:
        raw = np.array(
            [
                [320.0, 320.0, 160.0, 80.0, 0.91, 0.05, 0.01],
                [320.0, 320.0, 120.0, 60.0, 0.12, 0.88, 0.02],
            ],
            dtype=np.float32,
        )

        regions = decode_yolo_ncnn_output(
            raw,
            image_shape=(320, 640),
            input_size=640,
            conf_threshold=0.25,
            iou_threshold=0.45,
        )

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].bbox, (240, 140, 160, 40))
        self.assertEqual(regions[0].class_id, 0)
        self.assertEqual(regions[0].role, "qr")
        self.assertAlmostEqual(regions[0].score, 0.91, places=5)
        self.assertEqual(regions[1].class_id, 1)
        self.assertEqual(regions[1].role, "barcode")

    def test_decode_yolo_ncnn_output_applies_same_class_nms(self) -> None:
        raw = np.array(
            [
                [320.0, 320.0, 160.0, 80.0, 0.91, 0.05],
                [322.0, 322.0, 160.0, 80.0, 0.89, 0.04],
                [80.0, 80.0, 40.0, 40.0, 0.80, 0.10],
            ],
            dtype=np.float32,
        )

        regions = decode_yolo_ncnn_output(
            raw,
            image_shape=(640, 640),
            input_size=640,
            conf_threshold=0.25,
            iou_threshold=0.45,
        )

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].bbox, (240, 280, 160, 80))
        self.assertEqual(regions[1].bbox, (60, 60, 40, 40))


if __name__ == "__main__":
    unittest.main()
