#!/usr/bin/env python3
"""
Model Evaluation Script
========================
Evaluates the detector on an annotated dataset and reports
mAP, precision, recall, F1, and latency statistics.

Expects annotations in YOLO format (.txt files alongside images):
    <class_id> <cx> <cy> <w> <h>   (all normalised to [0, 1])

Usage:
    python scripts/evaluate_model.py \
        --images data/val/images \
        --labels data/val/labels \
        --model  yolov8n.pt \
        --conf   0.25 \
        --output results/eval_report.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.detector import YOLODetector, Detection
from src.metrics.evaluator import (
    DetectionEvaluator,
    GroundTruth,
    Prediction,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate YOLO detector on annotated dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", required=True, help="Directory of validation images.")
    p.add_argument("--labels", required=True, help="Directory of YOLO label files.")
    p.add_argument("--model",  default="yolov8n.pt")
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--output", default=None, help="Save JSON report to this path.")
    p.add_argument("--max-images", type=int, default=None)
    return p.parse_args()


def load_yolo_labels(
    label_path: Path, img_w: int, img_h: int
) -> list[GroundTruth]:
    """Parse a YOLO .txt label file into GroundTruth objects."""
    if not label_path.exists():
        return []

    gts = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id    = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h

            gts.append(GroundTruth(
                bbox=np.array([x1, y1, x2, y2]),
                class_id=cls_id,
            ))
    return gts


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        logger.error(f"No images found in {images_dir}")
        sys.exit(1)

    logger.info(f"Evaluating on {len(image_paths)} images...")

    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=0.45,
        device=args.device,
        image_size=args.imgsz,
    )
    detector.warmup()

    evaluator = DetectionEvaluator(
        class_names=detector.class_names,
        iou_threshold=args.iou,
    )

    for i, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Could not read {img_path}")
            continue

        h, w = frame.shape[:2]
        label_path = labels_dir / (img_path.stem + ".txt")
        ground_truths = load_yolo_labels(label_path, w, h)

        result = detector.detect(frame)

        predictions = [
            Prediction(
                bbox=det.bbox,
                class_id=det.class_id,
                confidence=det.confidence,
            )
            for det in result.detections
        ]

        evaluator.add(ground_truths, predictions, result.inference_time_ms)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(image_paths)} images...")

    report = evaluator.evaluate()
    report.print_summary()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Evaluation report saved → {out_path}")


if __name__ == "__main__":
    main()
