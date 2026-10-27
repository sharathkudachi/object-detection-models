"""
Visualization Utilities
========================
Drawing helpers for rendering detections, distances, overlays,
and performance HUDs onto frames.
"""

from __future__ import annotations

import colorsys
from typing import Optional

import cv2
import numpy as np

from src.detector import Detection, InferenceResult
from src.distance_estimator import DistanceEstimate


# ---------------------------------------------------------------------------
# Colour palette — deterministic per class_id
# ---------------------------------------------------------------------------

def _class_color(class_id: int) -> tuple[int, int, int]:
    """Return a vivid BGR colour unique to this class_id."""
    hue = (class_id * 0.618033988749895) % 1.0   # Golden ratio spacing
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


# ---------------------------------------------------------------------------
# Core drawing functions
# ---------------------------------------------------------------------------

def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    color: Optional[tuple[int, int, int]] = None,
    thickness: int = 2,
    font_scale: float = 0.55,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw a single bounding box with label onto a frame (in-place)."""
    x1, y1, x2, y2 = detection.bbox.astype(int)
    clr = color or _class_color(detection.class_id)

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), clr, thickness)

    # Label background
    label = (
        f"{detection.class_name} {detection.confidence:.0%}"
        if show_confidence else detection.class_name
    )
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    label_y = max(y1, th + 4)
    cv2.rectangle(
        frame,
        (x1, label_y - th - 4),
        (x1 + tw + 4, label_y + baseline),
        clr, -1
    )
    cv2.putText(
        frame, label,
        (x1 + 2, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_distance(
    frame: np.ndarray,
    estimate: DistanceEstimate,
    color: Optional[tuple[int, int, int]] = None,
    font_scale: float = 0.50,
) -> np.ndarray:
    """Overlay distance text below the detection's bounding box (in-place)."""
    det = estimate.detection
    x1, y1, x2, y2 = det.bbox.astype(int)
    clr = color or _class_color(det.class_id)
    cx = (x1 + x2) // 2

    dist_label = f"{estimate.distance_m:.2f} m"
    (tw, th), _ = cv2.getTextSize(
        dist_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )

    # Background pill
    pad = 4
    bx1, by1 = cx - tw // 2 - pad, y2 + 2
    bx2, by2 = cx + tw // 2 + pad, y2 + th + pad + 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), clr, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame, dist_label,
        (cx - tw // 2, y2 + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_all_detections(
    frame: np.ndarray,
    result: InferenceResult,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw all detections in a result onto the frame."""
    out = frame.copy()
    for det in result.detections:
        draw_detection(out, det, show_confidence=show_confidence)
    return out


def draw_distance_estimates(
    frame: np.ndarray,
    estimates: list[DistanceEstimate],
    show_detection: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw all detection boxes with distance overlays."""
    out = frame.copy()
    for est in estimates:
        if show_detection:
            draw_detection(
                out, est.detection,
                show_confidence=show_confidence,
            )
        draw_distance(out, est)
    return out


def draw_fps_hud(
    frame: np.ndarray,
    fps: float,
    inference_ms: float,
    num_detections: int,
    frame_id: int,
    extra: Optional[dict[str, str]] = None,
) -> np.ndarray:
    """
    Render a semi-transparent HUD in the top-left corner.

    Shows FPS, inference latency, detection count, and any extra fields.
    """
    lines = [
        f"FPS: {fps:.1f}",
        f"Latency: {inference_ms:.1f} ms",
        f"Objects: {num_detections}",
        f"Frame: {frame_id}",
    ]
    if extra:
        lines += [f"{k}: {v}" for k, v in extra.items()]

    font      = cv2.FONT_HERSHEY_SIMPLEX
    fscale    = 0.55
    thickness = 1
    padding   = 8
    line_h    = 22

    panel_w = 200
    panel_h = len(lines) * line_h + padding * 2

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0), (panel_w, panel_h),
        (0, 0, 0), -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = padding + (i + 1) * line_h - 4
        cv2.putText(
            frame, line, (padding, y),
            font, fscale, (0, 255, 120), thickness, cv2.LINE_AA,
        )

    return frame


def draw_distance_zones(
    frame: np.ndarray,
    estimates: list[DistanceEstimate],
    near_thresh_m: float = 2.0,
    mid_thresh_m: float  = 5.0,
) -> np.ndarray:
    """
    Colour-code bounding boxes by proximity zone:
      🔴 Red   — near  (< near_thresh_m)
      🟡 Yellow — mid   (near_thresh_m .. mid_thresh_m)
      🟢 Green  — far   (> mid_thresh_m)
    """
    out = frame.copy()
    for est in estimates:
        d = est.distance_m
        if d < near_thresh_m:
            color = (0, 0, 220)    # Red
        elif d < mid_thresh_m:
            color = (0, 210, 255)  # Yellow
        else:
            color = (0, 200, 80)   # Green

        draw_detection(out, est.detection, color=color)
        draw_distance(out, est, color=color)

    return out


def draw_crosshair(
    frame: np.ndarray,
    center: Optional[tuple[int, int]] = None,
    color: tuple[int, int, int] = (200, 200, 200),
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw a centre crosshair (useful for alignment reference)."""
    h, w = frame.shape[:2]
    cx, cy = center or (w // 2, h // 2)

    overlay = frame.copy()
    cv2.line(overlay, (0, cy), (w, cy), color, 1)
    cv2.line(overlay, (cx, 0), (cx, h), color, 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def make_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    label_left: str = "Left",
    label_right: str = "Right",
) -> np.ndarray:
    """Stack two frames horizontally with labels."""
    h = max(left.shape[0], right.shape[0])

    def _pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] < h:
            pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
            return np.vstack([img, pad])
        return img

    left  = _pad(left.copy())
    right = _pad(right.copy())

    for img, lbl in [(left, label_left), (right, label_right)]:
        cv2.putText(
            img, lbl, (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    divider = np.full((h, 2, 3), 100, dtype=np.uint8)
    return np.hstack([left, divider, right])
