"""Unit tests for the metrics evaluation pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.metrics.evaluator import (
    DetectionEvaluator,
    GroundTruth,
    Prediction,
    compute_iou,
    compute_iou_matrix,
    compute_average_precision,
)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

class TestComputeIoU:
    def test_identical_boxes(self):
        box = np.array([10, 10, 50, 50], dtype=float)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        b = np.array([20, 20, 30, 30], dtype=float)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        b = np.array([5, 5, 15, 15], dtype=float)
        # intersection = 5×5 = 25; union = 100+100-25 = 175
        assert compute_iou(a, b) == pytest.approx(25.0 / 175.0, rel=1e-4)

    def test_one_inside_other(self):
        outer = np.array([0, 0, 100, 100], dtype=float)
        inner = np.array([25, 25, 75, 75], dtype=float)
        # intersection = 50×50 = 2500; union = 10000+2500-2500 = 10000
        assert compute_iou(outer, inner) == pytest.approx(0.25, rel=1e-4)

    def test_touching_edges(self):
        a = np.array([0, 0, 10, 10], dtype=float)
        b = np.array([10, 0, 20, 10], dtype=float)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_symmetric(self):
        a = np.array([0, 0, 40, 60], dtype=float)
        b = np.array([20, 30, 80, 90], dtype=float)
        assert compute_iou(a, b) == pytest.approx(compute_iou(b, a), rel=1e-9)

    def test_value_in_range(self):
        a = np.array([10, 20, 80, 90], dtype=float)
        b = np.array([30, 10, 100, 70], dtype=float)
        iou = compute_iou(a, b)
        assert 0.0 <= iou <= 1.0


class TestComputeIoUMatrix:
    def test_shape(self):
        A = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
        B = np.array([[5, 5, 15, 15], [25, 25, 35, 35], [0, 0, 5, 5]], dtype=float)
        M = compute_iou_matrix(A, B)
        assert M.shape == (2, 3)

    def test_diagonal_identity(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 40, 40],
            [50, 50, 80, 90],
        ], dtype=float)
        M = compute_iou_matrix(boxes, boxes)
        np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-6)

    def test_empty_inputs(self):
        A = np.zeros((0, 4), dtype=float)
        B = np.array([[0, 0, 10, 10]], dtype=float)
        assert compute_iou_matrix(A, B).shape == (0, 1)
        assert compute_iou_matrix(B, A).shape == (1, 0)

    def test_values_in_range(self):
        rng  = np.random.default_rng(42)
        A = rng.uniform(0, 100, (10, 4))
        A[:, 2:] += A[:, :2] + 5     # ensure x2>x1, y2>y1
        B = rng.uniform(0, 100, (8, 4))
        B[:, 2:] += B[:, :2] + 5
        M = compute_iou_matrix(A, B)
        assert np.all(M >= 0) and np.all(M <= 1)


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------

class TestComputeAveragePrecision:
    def test_perfect_detector(self):
        # All TP at full recall → AP = 1.0
        recalls    = np.linspace(0, 1, 11)
        precisions = np.ones(11)
        ap = compute_average_precision(precisions, recalls)
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_zero_precision(self):
        recalls    = np.array([0.0, 0.5, 1.0])
        precisions = np.array([0.0, 0.0, 0.0])
        assert compute_average_precision(precisions, recalls) == pytest.approx(0.0)

    def test_output_in_range(self):
        rng = np.random.default_rng(0)
        r   = np.sort(rng.uniform(0, 1, 20))
        p   = rng.uniform(0, 1, 20)
        ap  = compute_average_precision(p, r)
        assert 0.0 <= ap <= 1.0


# ---------------------------------------------------------------------------
# DetectionEvaluator
# ---------------------------------------------------------------------------

NAMES = {0: "car", 1: "person", 2: "truck"}


def _gt(x1, y1, x2, y2, cls=0):
    return GroundTruth(bbox=np.array([x1, y1, x2, y2], dtype=float), class_id=cls)


def _pred(x1, y1, x2, y2, cls=0, conf=0.9):
    return Prediction(
        bbox=np.array([x1, y1, x2, y2], dtype=float),
        class_id=cls, confidence=conf,
    )


class TestDetectionEvaluator:
    def test_perfect_predictions(self):
        ev = DetectionEvaluator(NAMES, iou_threshold=0.5)
        gts   = [_gt(0, 0, 100, 100, cls=0)]
        preds = [_pred(0, 0, 100, 100, cls=0, conf=0.99)]
        ev.add(gts, preds)
        report = ev.evaluate()
        assert report.map_50 == pytest.approx(1.0, abs=0.01)

    def test_no_predictions(self):
        ev = DetectionEvaluator(NAMES, iou_threshold=0.5)
        ev.add([_gt(0, 0, 100, 100)], [])
        report = ev.evaluate()
        assert report.map_50 == pytest.approx(0.0, abs=0.01)

    def test_no_ground_truths(self):
        ev = DetectionEvaluator(NAMES, iou_threshold=0.5)
        ev.add([], [_pred(0, 0, 50, 50)])
        report = ev.evaluate()
        # FP only — AP = 0
        if 0 in report.class_metrics:
            assert report.class_metrics[0].ap == pytest.approx(0.0, abs=0.01)

    def test_wrong_class(self):
        ev = DetectionEvaluator(NAMES, iou_threshold=0.5)
        ev.add([_gt(0, 0, 100, 100, cls=0)], [_pred(0, 0, 100, 100, cls=1)])
        report = ev.evaluate()
        # Class 0 GT exists but no class-0 prediction → recall = 0
        if 0 in report.class_metrics:
            assert report.class_metrics[0].recall == pytest.approx(0.0, abs=0.01)

    def test_iou_below_threshold_is_fp(self):
        ev = DetectionEvaluator(NAMES, iou_threshold=0.5)
        # Barely overlapping boxes
        ev.add(
            [_gt(0, 0, 100, 100)],
            [_pred(80, 80, 200, 200)],  # IoU ≈ 0.05
        )
        report = ev.evaluate()
        assert report.class_metrics[0].fp >= 1

    def test_multiple_frames_accumulate(self):
        ev = DetectionEvaluator(NAMES, iou_threshold=0.5)
        for _ in range(5):
            ev.add([_gt(10, 10, 90, 90)], [_pred(10, 10, 90, 90)])
        report = ev.evaluate()
        assert report.class_metrics[0].tp == 5

    def test_latency_stats_populated(self):
        ev = DetectionEvaluator(NAMES)
        ev.add([_gt(0, 0, 50, 50)], [_pred(0, 0, 50, 50)], inference_time_ms=12.5)
        ev.add([_gt(0, 0, 50, 50)], [_pred(0, 0, 50, 50)], inference_time_ms=15.0)
        report = ev.evaluate()
        assert "mean_ms" in report.latency_stats
        assert report.latency_stats["num_frames"] == 2

    def test_map_50_95_leq_map_50(self):
        ev = DetectionEvaluator(NAMES, iou_threshold=0.5)
        for _ in range(3):
            ev.add([_gt(0, 0, 100, 100)], [_pred(5, 5, 95, 95, conf=0.9)])
        report = ev.evaluate()
        assert report.map_50_95 <= report.map_50 + 1e-6

    def test_report_to_dict_structure(self):
        ev = DetectionEvaluator(NAMES)
        ev.add([_gt(0, 0, 100, 100)], [_pred(0, 0, 100, 100)])
        d = ev.evaluate().to_dict()
        assert "mAP_50" in d
        assert "per_class" in d

    def test_reset_clears_state(self):
        ev = DetectionEvaluator(NAMES)
        ev.add([_gt(0, 0, 100, 100)], [_pred(0, 0, 100, 100)])
        ev.reset()
        report = ev.evaluate()
        assert report.map_50 == pytest.approx(0.0, abs=0.01)
        assert len(report.class_metrics) == 0

    def test_multi_class(self):
        ev = DetectionEvaluator(NAMES, iou_threshold=0.5)
        ev.add(
            [_gt(0, 0, 100, 100, cls=0), _gt(200, 200, 300, 300, cls=1)],
            [_pred(0, 0, 100, 100, cls=0), _pred(200, 200, 300, 300, cls=1)],
        )
        report = ev.evaluate()
        assert 0 in report.class_metrics
        assert 1 in report.class_metrics
        assert report.map_50 == pytest.approx(1.0, abs=0.01)
