from __future__ import annotations

import argparse

from price_tag_pipeline.detector.heuristic import HeuristicPriceTagDetector
from price_tag_pipeline.detector.onnx_yolo import YoloOnnxDetector


def build_detector(args: argparse.Namespace):
    if args.detector == "onnx":
        return YoloOnnxDetector(
            model_path=args.onnx_model,
            input_size=args.onnx_input_size,
            conf_threshold=args.onnx_conf_threshold,
            iou_threshold=args.onnx_iou_threshold,
        )
    return HeuristicPriceTagDetector()
