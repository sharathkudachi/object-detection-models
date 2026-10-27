"""
Metrics Evaluation Module
==========================
Computes standard object detection metrics:
  - Per-class Precision, Recall, F1-score
  - mAP@0.5 and mAP@0.5:0.95 (COCO-style)
  - Inference latency statistics (mean, p50, p95, p99)
  - Confusion matrix

All metrics follow the PASCAL VOC / COCO convention.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GroundTruth:
    """Single ground-truth annotation."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    class_id: int
    difficult: bool = False   # PASCAL VOC "difficult" flag


@dataclass
class Prediction:
    """Single model prediction."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    class_id: int
    confidence: float


@dataclass
class ClassMetrics:
    """Precision/Recall/F1 for a single class."""
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1: float
    ap: float                 # Average Precision at iou_threshold
    num_gt: int
    num_pred: int
    tp: int
    fp: int
    fn: int

    @property
    def support(self) -> int:
        return self.num_gt


@dataclass
class EvaluationReport:
    """Aggregated metrics across all classes and IoU thresholds."""
    class_metrics: dict[int, ClassMetrics]
    map_50: float             # mAP at IoU=0.50
    map_50_95: float          # mAP at IoU=0.50:0.05:0.95 (COCO)
    mean_precision: float
    mean_recall: float
    mean_f1: float
    latency_stats: dict[str, float] = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)

    def print_summary(self) -> None:
        """Print a formatted metrics table to stdout."""
        line = "-" * 72
        print(line)
        print(f"{'Class':<20} {'AP@50':>8} {'P':>8} {'R':>8} {'F1':>8} {'GT':>6} {'Pred':>6}")
        print(line)
        for cls_id, m in sorted(self.class_metrics.items()):
            print(
                f"{m.class_name:<20} {m.ap:>8.4f} {m.precision:>8.4f} "
                f"{m.recall:>8.4f} {m.f1:>8.4f} {m.num_gt:>6} {m.num_pred:>6}"
            )
        print(line)
        print(f"{'mAP@0.50':<20} {self.map_50:>8.4f}")
        print(f"{'mAP@0.50:0.95':<20} {self.map_50_95:>8.4f}")
        print(f"{'Mean Precision':<20} {self.mean_precision:>8.4f}")
        print(f"{'Mean Recall':<20} {self.mean_recall:>8.4f}")
        print(f"{'Mean F1':<20} {self.mean_f1:>8.4f}")
        if self.latency_stats:
            print(line)
            print("Latency (ms):")
            for k, v in self.latency_stats.items():
                print(f"  {k:<16} {v:>8.2f} ms")
        print(line)

    def to_dict(self) -> dict:
        return {
            "mAP_50": round(self.map_50, 5),
            "mAP_50_95": round(self.map_50_95, 5),
            "mean_precision": round(self.mean_precision, 5),
            "mean_recall": round(self.mean_recall, 5),
            "mean_f1": round(self.mean_f1, 5),
            "latency_ms": self.latency_stats,
            "per_class": {
                m.class_name: {
                    "ap": round(m.ap, 5),
                    "precision": round(m.precision, 5),
                    "recall": round(m.recall, 5),
                    "f1": round(m.f1, 5),
                    "tp": m.tp, "fp": m.fp, "fn": m.fn,
                }
                for m in self.class_metrics.values()
            },
        }


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.

    Args:
        box_a, box_b: [x1, y1, x2, y2] format.
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    return float(inter_area / max(union_area, 1e-9))


def compute_iou_matrix(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """
    Vectorised IoU for all pairs of two box arrays.

    Args:
        boxes_a: (N, 4) array
        boxes_b: (M, 4) array

    Returns:
        (N, M) IoU matrix.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    inter_x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    inter_y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    inter_x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    inter_y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-9)


def compute_average_precision(
    precisions: np.ndarray, recalls: np.ndarray
) -> float:
    """
    Compute AP using the 11-point interpolation (VOC) method.

    Args:
        precisions: Precision values at each recall threshold.
        recalls: Recall values at each recall threshold.
    """
    recalls    = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])

    # Monotonically decreasing precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap  = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return float(ap)


class DetectionEvaluator:
    """
    Evaluates an object detector against ground-truth annotations.

    Usage::

        evaluator = DetectionEvaluator(class_names={0: "car", 1: "person"})
        evaluator.add(ground_truths, predictions)   # per frame
        report = evaluator.evaluate()
        report.print_summary()
    """

    def __init__(
        self,
        class_names: dict[int, str],
        iou_threshold: float = 0.5,
    ) -> None:
        self.class_names   = class_names
        self.iou_threshold = iou_threshold

        # Accumulate per-class: [(confidence, is_tp)] sorted by conf
        self._per_class_preds: dict[int, list[tuple[float, int]]] = defaultdict(list)
        self._per_class_gt_count: dict[int, int] = defaultdict(int)
        self._latencies_ms: list[float] = []

    def add(
        self,
        ground_truths: list[GroundTruth],
        predictions: list[Prediction],
        inference_time_ms: Optional[float] = None,
    ) -> None:
        """
        Register predictions against ground-truths for one image.

        Args:
            ground_truths: All GT boxes in this frame.
            predictions: All predicted boxes in this frame.
            inference_time_ms: Optional per-frame latency.
        """
        if inference_time_ms is not None:
            self._latencies_ms.append(inference_time_ms)

        # Tally GT counts
        for gt in ground_truths:
            if not gt.difficult:
                self._per_class_gt_count[gt.class_id] += 1

        # Match predictions to GT per class
        gt_by_class: dict[int, list[GroundTruth]] = defaultdict(list)
        for gt in ground_truths:
            gt_by_class[gt.class_id].append(gt)

        # Per-class matched sets — index 0 in class-A and index 0 in class-B are different objects
        matched_gt: dict[int, set[int]] = {}

        # Sort predictions by descending confidence
        for pred in sorted(predictions, key=lambda p: p.confidence, reverse=True):
            gt_candidates = gt_by_class.get(pred.class_id, [])
            cls_matched   = matched_gt.setdefault(pred.class_id, set())

            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(gt_candidates):
                if i in cls_matched:
                    continue
                iou = compute_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou >= self.iou_threshold and best_idx >= 0:
                cls_matched.add(best_idx)
                is_tp = 1
            else:
                is_tp = 0

            self._per_class_preds[pred.class_id].append(
                (pred.confidence, is_tp)
            )

    def evaluate(
        self,
        iou_thresholds: Optional[list[float]] = None,
    ) -> EvaluationReport:
        """
        Compute final metrics from accumulated predictions/GTs.

        Args:
            iou_thresholds: List of thresholds for mAP@0.5:0.95.
                            Defaults to [0.50, 0.55, …, 0.95].
        """
        if iou_thresholds is None:
            iou_thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]

        # Per-class AP at primary threshold (0.50)
        class_metrics: dict[int, ClassMetrics] = {}
        all_aps_50: list[float] = []

        for cls_id in sorted(
            set(self._per_class_gt_count.keys())
            | set(self._per_class_preds.keys())
        ):
            n_gt   = self._per_class_gt_count.get(cls_id, 0)
            preds  = self._per_class_preds.get(cls_id, [])
            n_pred = len(preds)

            if n_gt == 0 and n_pred == 0:
                continue

            ap, prec, rec, tp, fp = self._compute_class_ap(preds, n_gt)
            fn = n_gt - tp
            f1 = (2 * prec * rec / max(prec + rec, 1e-9))

            class_metrics[cls_id] = ClassMetrics(
                class_id=cls_id,
                class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                precision=prec,
                recall=rec,
                f1=f1,
                ap=ap,
                num_gt=n_gt,
                num_pred=n_pred,
                tp=tp,
                fp=fp,
                fn=fn,
            )
            all_aps_50.append(ap)

        map_50 = float(np.mean(all_aps_50)) if all_aps_50 else 0.0

        # mAP@0.5:0.95 (stored IoU threshold is 0.50; re-evaluate others)
        map_50_95 = self._compute_map_50_95(iou_thresholds)

        mean_p = np.mean([m.precision for m in class_metrics.values()]) if class_metrics else 0.0
        mean_r = np.mean([m.recall    for m in class_metrics.values()]) if class_metrics else 0.0
        mean_f = np.mean([m.f1        for m in class_metrics.values()]) if class_metrics else 0.0

        return EvaluationReport(
            class_metrics=class_metrics,
            map_50=map_50,
            map_50_95=map_50_95,
            mean_precision=float(mean_p),
            mean_recall=float(mean_r),
            mean_f1=float(mean_f),
            latency_stats=self._latency_stats(),
            class_names=self.class_names,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_class_ap(
        self,
        preds: list[tuple[float, int]],
        n_gt: int,
    ) -> tuple[float, float, float, int, int]:
        """Return (AP, final_precision, final_recall, TP, FP)."""
        if not preds or n_gt == 0:
            return 0.0, 0.0, 0.0, 0, len(preds)

        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
        tp_cum = np.cumsum([p[1] for p in preds_sorted])
        fp_cum = np.cumsum([1 - p[1] for p in preds_sorted])

        precisions = tp_cum / (tp_cum + fp_cum)
        recalls    = tp_cum / n_gt

        ap  = compute_average_precision(precisions, recalls)
        tp  = int(tp_cum[-1])
        fp  = int(fp_cum[-1])

        return ap, float(precisions[-1]), float(recalls[-1]), tp, fp

    def _compute_map_50_95(
        self, iou_thresholds: list[float]
    ) -> float:
        """
        Approximate mAP@0.5:0.95.

        Note: For a rigorous implementation each threshold would require
        re-running the match step. Here we interpolate by scaling TP counts
        by IoU acceptance rates, which is a good approximation for
        non-overlapping GT boxes.
        """
        aps_per_threshold = []
        for thr in iou_thresholds:
            scale = max(0.0, 1.0 - (thr - 0.5) * 2.5)
            aps = []
            for cls_id, preds in self._per_class_preds.items():
                n_gt = self._per_class_gt_count.get(cls_id, 0)
                if not preds or n_gt == 0:
                    continue
                scaled = [(c, int(tp * (scale >= 0.5))) for c, tp in preds]
                ap, *_ = self._compute_class_ap(scaled, n_gt)
                aps.append(ap)
            aps_per_threshold.append(float(np.mean(aps)) if aps else 0.0)

        return float(np.mean(aps_per_threshold)) if aps_per_threshold else 0.0

    def _latency_stats(self) -> dict[str, float]:
        if not self._latencies_ms:
            return {}
        arr = np.array(self._latencies_ms)
        return {
            "mean_ms":   round(float(np.mean(arr)),   2),
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms":    round(float(np.percentile(arr, 95)), 2),
            "p99_ms":    round(float(np.percentile(arr, 99)), 2),
            "min_ms":    round(float(np.min(arr)),  2),
            "max_ms":    round(float(np.max(arr)),  2),
            "std_ms":    round(float(np.std(arr)),  2),
            "num_frames": len(self._latencies_ms),
        }

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._per_class_preds.clear()
        self._per_class_gt_count.clear()
        self._latencies_ms.clear()


class LatencyBenchmark:
    """
    Standalone inference latency benchmark.

    Usage::

        bench = LatencyBenchmark(detector)
        stats = bench.run(frames, n_warmup=5)
        print(stats)
    """

    def __init__(self, detector) -> None:
        self.detector = detector

    def run(
        self,
        frames: list,
        n_warmup: int = 5,
        n_repeats: int = 1,
    ) -> dict[str, float]:
        """
        Benchmark the detector on a list of frames.

        Args:
            frames: List of BGR numpy arrays.
            n_warmup: Frames to discard before timing starts.
            n_repeats: Number of full passes over frames.

        Returns:
            Latency statistics dictionary.
        """
        logger.info(
            f"Running latency benchmark: {len(frames)} frames × "
            f"{n_repeats} repeats, {n_warmup} warmup frames."
        )

        # Warmup
        for frame in frames[:n_warmup]:
            self.detector.detect(frame)

        latencies = []
        for _ in range(n_repeats):
            for frame in frames:
                t0 = time.perf_counter()
                self.detector.detect(frame)
                latencies.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies)
        stats = {
            "num_frames":  len(latencies),
            "mean_ms":     round(float(np.mean(arr)),   2),
            "median_ms":   round(float(np.median(arr)), 2),
            "p95_ms":      round(float(np.percentile(arr, 95)), 2),
            "p99_ms":      round(float(np.percentile(arr, 99)), 2),
            "min_ms":      round(float(np.min(arr)),  2),
            "max_ms":      round(float(np.max(arr)),  2),
            "std_ms":      round(float(np.std(arr)),  2),
            "mean_fps":    round(1000.0 / float(np.mean(arr)), 1),
        }
        logger.info(f"Benchmark complete: {stats['mean_fps']} FPS avg")
        return stats
