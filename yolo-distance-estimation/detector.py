"""
YOLO-based Object Detection Pipeline
=====================================
Real-time object detection using YOLOv8 with support for
multi-class detection, confidence filtering, and NMS tuning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Represents a single object detection result."""
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixel coords
    class_id: int
    class_name: str
    confidence: float
    center: tuple[float, float] = field(init=False)
    width: float = field(init=False)
    height: float = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.width = float(x2 - x1)
        self.height = float(y2 - y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1e-6)

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "center": self.center,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


@dataclass
class InferenceResult:
    """Full result for a single frame inference."""
    detections: list[Detection]
    frame: np.ndarray
    inference_time_ms: float
    frame_id: int = 0

    @property
    def fps(self) -> float:
        return 1000.0 / max(self.inference_time_ms, 1e-6)

    @property
    def num_detections(self) -> int:
        return len(self.detections)


class YOLODetector:
    """
    Real-time object detector using YOLOv8.

    Wraps Ultralytics YOLOv8 with a clean interface for
    single-frame and video-stream inference. Supports GPU
    acceleration, half-precision (FP16), and custom class filtering.

    Args:
        model_path: Path to YOLOv8 weights (.pt file) or model name
                    (e.g. 'yolov8n.pt', 'yolov8s.pt').
        confidence_threshold: Minimum detection confidence [0, 1].
        iou_threshold: IoU threshold for Non-Maximum Suppression.
        target_classes: Optional list of class IDs to keep.
                        None means keep all classes.
        device: Inference device — 'cpu', 'cuda', or 'mps'.
        image_size: Input resolution for the model.
        half_precision: Use FP16 inference (GPU only).
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_classes: Optional[list[int]] = None,
        device: str = "cpu",
        image_size: int = 640,
        half_precision: bool = False,
    ) -> None:
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            )

        self.model_path = Path(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.device = device
        self.image_size = image_size
        self.half_precision = half_precision

        self._frame_count = 0
        self._total_inference_time = 0.0

        logger.info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(str(model_path))
        self.class_names: dict[int, str] = self.model.names

        logger.info(
            f"Model loaded | Device: {device} | Classes: {len(self.class_names)} "
            f"| FP16: {half_precision}"
        )

    def detect(self, frame: np.ndarray) -> InferenceResult:
        """
        Run detection on a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            InferenceResult with all detections and timing info.
        """
        self._frame_count += 1
        t_start = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            imgsz=self.image_size,
            half=self.half_precision,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - t_start) * 1000
        self._total_inference_time += inference_ms

        detections = self._parse_results(results[0])

        return InferenceResult(
            detections=detections,
            frame=frame.copy(),
            inference_time_ms=inference_ms,
            frame_id=self._frame_count,
        )

    def detect_batch(self, frames: list[np.ndarray]) -> list[InferenceResult]:
        """Run batched inference across multiple frames."""
        t_start = time.perf_counter()

        results = self.model.predict(
            source=frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            imgsz=self.image_size,
            half=self.half_precision,
            verbose=False,
        )

        total_ms = (time.perf_counter() - t_start) * 1000
        per_frame_ms = total_ms / max(len(frames), 1)

        outputs = []
        for i, (result, frame) in enumerate(zip(results, frames)):
            self._frame_count += 1
            self._total_inference_time += per_frame_ms
            outputs.append(InferenceResult(
                detections=self._parse_results(result),
                frame=frame.copy(),
                inference_time_ms=per_frame_ms,
                frame_id=self._frame_count,
            ))

        return outputs

    def _parse_results(self, result) -> list[Detection]:
        """Parse raw YOLO result into Detection objects."""
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
            detections.append(Detection(
                bbox=bbox,
                class_id=int(cls_id),
                class_name=self.class_names.get(int(cls_id), f"class_{cls_id}"),
                confidence=float(conf),
            ))

        return detections

    @property
    def average_fps(self) -> float:
        """Average FPS over all processed frames."""
        if self._frame_count == 0:
            return 0.0
        avg_ms = self._total_inference_time / self._frame_count
        return 1000.0 / max(avg_ms, 1e-6)

    @property
    def stats(self) -> dict:
        """Cumulative inference statistics."""
        return {
            "frames_processed": self._frame_count,
            "total_inference_ms": round(self._total_inference_time, 2),
            "average_inference_ms": round(
                self._total_inference_time / max(self._frame_count, 1), 2
            ),
            "average_fps": round(self.average_fps, 2),
        }

    def reset_stats(self) -> None:
        """Reset frame counter and timing accumulators."""
        self._frame_count = 0
        self._total_inference_time = 0.0

    def warmup(self, n_iterations: int = 3) -> None:
        """
        Run inference on dummy frames to warm up the model.
        Eliminates first-frame latency spikes.
        """
        logger.info(f"Warming up model with {n_iterations} dummy iterations...")
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        for _ in range(n_iterations):
            self.model.predict(
                source=dummy, verbose=False, device=self.device
            )
        self.reset_stats()
        logger.info("Warmup complete.")

    def __repr__(self) -> str:
        return (
            f"YOLODetector(model={self.model_path.name}, "
            f"conf={self.conf_threshold}, iou={self.iou_threshold}, "
            f"device={self.device})"
        )
