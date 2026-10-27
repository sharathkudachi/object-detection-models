"""Unit tests for the YOLODetector and Detection classes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.detector import Detection, InferenceResult


class TestDetection:
    def _make(self, x1=10, y1=20, x2=110, y2=220, cls_id=0, conf=0.9):
        return Detection(
            bbox=np.array([x1, y1, x2, y2], dtype=float),
            class_id=cls_id,
            class_name="person",
            confidence=conf,
        )

    def test_center(self):
        det = self._make(0, 0, 100, 200)
        assert det.center == (50.0, 100.0)

    def test_dimensions(self):
        det = self._make(10, 20, 60, 70)
        assert det.width  == pytest.approx(50.0)
        assert det.height == pytest.approx(50.0)

    def test_area(self):
        det = self._make(0, 0, 10, 20)
        assert det.area == pytest.approx(200.0)

    def test_aspect_ratio(self):
        det = self._make(0, 0, 50, 100)
        assert det.aspect_ratio == pytest.approx(0.5)

    def test_to_dict_keys(self):
        det = self._make()
        d = det.to_dict()
        assert "bbox" in d
        assert "class_name" in d
        assert "confidence" in d
        assert "center" in d

    def test_confidence_rounded(self):
        det = self._make(conf=0.123456789)
        d = det.to_dict()
        assert d["confidence"] == pytest.approx(0.1235, abs=1e-4)

    def test_zero_height_aspect_ratio(self):
        # Should not divide by zero
        det = self._make(x1=0, y1=0, x2=50, y2=0)
        assert det.aspect_ratio > 0


class TestInferenceResult:
    def _make_result(self, n_dets=3, inference_ms=25.0):
        dets = [
            Detection(
                bbox=np.array([i*10, i*10, i*10+50, i*10+50], dtype=float),
                class_id=i % 3,
                class_name=["car", "person", "truck"][i % 3],
                confidence=0.8,
            )
            for i in range(n_dets)
        ]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return InferenceResult(
            detections=dets,
            frame=frame,
            inference_time_ms=inference_ms,
            frame_id=1,
        )

    def test_num_detections(self):
        r = self._make_result(n_dets=5)
        assert r.num_detections == 5

    def test_fps_calculation(self):
        r = self._make_result(inference_ms=40.0)
        assert r.fps == pytest.approx(25.0, rel=0.01)

    def test_empty_detections(self):
        r = self._make_result(n_dets=0)
        assert r.num_detections == 0

    def test_frame_is_copy(self):
        r = self._make_result()
        original_sum = r.frame.sum()
        r.frame[:] = 255
        assert r.frame.sum() != original_sum  # was copied but is mutable
