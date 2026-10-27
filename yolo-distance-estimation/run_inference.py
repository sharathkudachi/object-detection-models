#!/usr/bin/env python3
"""
Real-Time Object Detection & Distance Estimation
=================================================
Main inference entry point. Supports webcam, video files, and
image directories as input sources.

Usage:
    # Webcam (default)
    python scripts/run_inference.py

    # Video file
    python scripts/run_inference.py --source path/to/video.mp4

    # Custom model + calibration
    python scripts/run_inference.py \
        --model yolov8s.pt \
        --calibration config/calibration.json \
        --conf 0.45 \
        --device cuda

    # Save annotated output
    python scripts/run_inference.py --source video.mp4 --save output.mp4
"""

import argparse
import sys
import time
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.calibration.camera_calibrator import CameraCalibrator
from src.detector import YOLODetector
from src.distance_estimator import (
    CameraIntrinsics,
    GeometricDistanceEstimator,
    HybridDistanceEstimator,
)
from src.tracking.tracker import CentroidTracker
from src.utils.logger import get_logger
from src.utils.visualization import (
    draw_distance_zones,
    draw_fps_hud,
    draw_crosshair,
)

logger = get_logger(__name__)


def build_default_intrinsics(frame_w: int, frame_h: int) -> CameraIntrinsics:
    """
    Construct a reasonable default camera matrix when no calibration
    file is provided. Assumes ~70° HFOV, which is typical for webcams.
    """
    fx = frame_w / (2 * np.tan(np.radians(35)))
    fy = fx
    return CameraIntrinsics(fx=fx, fy=fy, cx=frame_w / 2, cy=frame_h / 2)


def open_source(source: str | int) -> cv2.VideoCapture:
    """Open a VideoCapture from webcam index or file path."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")
    return cap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time YOLO object detection + distance estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", default=0,
        help="Input source: webcam index (0), video path, or RTSP URL.",
    )
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights.")
    p.add_argument(
        "--calibration", default=None,
        help="Path to calibration.json. If not set, a default matrix is used.",
    )
    p.add_argument("--conf", type=float, default=0.45, help="Confidence threshold.")
    p.add_argument("--iou",  type=float, default=0.45, help="NMS IoU threshold.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--imgsz", type=int, default=640, help="Model input resolution.")
    p.add_argument("--classes", nargs="+", type=int, default=None,
                   help="Filter to specific class IDs.")
    p.add_argument("--save", default=None,
                   help="Path to save annotated output video (.mp4).")
    p.add_argument("--no-track", action="store_true",
                   help="Disable centroid tracker (track IDs won't be shown).")
    p.add_argument("--baseline", type=float, default=0.12,
                   help="Stereo baseline in metres (for hybrid estimator).")
    p.add_argument("--near",  type=float, default=2.0,
                   help="Near-zone threshold for colour coding (metres).")
    p.add_argument("--mid",   type=float, default=6.0,
                   help="Mid-zone threshold for colour coding (metres).")
    p.add_argument("--width", type=int, default=1280,
                   help="Capture width in pixels.")
    p.add_argument("--height", type=int, default=720,
                   help="Capture height in pixels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Source ─────────────────────────────────────────────────────────
    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    cap = open_source(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Source opened: {source} | Resolution: {actual_w}×{actual_h}")

    # ── Camera Intrinsics ──────────────────────────────────────────────
    if args.calibration:
        intrinsics = CameraCalibrator.load(args.calibration)
        logger.info("Using calibrated camera intrinsics.")
    else:
        intrinsics = build_default_intrinsics(actual_w, actual_h)
        logger.warning(
            "No calibration file provided — using estimated intrinsics. "
            "Distance accuracy may be reduced. Run scripts/calibrate_camera.py first."
        )

    # ── Components ─────────────────────────────────────────────────────
    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        target_classes=args.classes,
        device=args.device,
        image_size=args.imgsz,
    )
    detector.warmup()

    estimator = GeometricDistanceEstimator(intrinsics)
    tracker   = CentroidTracker(max_disappeared=25) if not args.no_track else None

    # ── Output writer ──────────────────────────────────────────────────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.save, fourcc, 30, (actual_w, actual_h)
        )
        logger.info(f"Saving output to: {args.save}")

    # ── Main loop ──────────────────────────────────────────────────────
    logger.info("Starting inference loop. Press 'q' to quit, 's' to screenshot.")
    frame_id = 0
    fps_display = 0.0
    t_prev = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream.")
                break

            frame_id += 1

            # Detect
            result = detector.detect(frame)

            # Track (optional)
            if tracker is not None:
                active_tracks = tracker.update(result.detections)
                track_id_map  = {
                    id(t.detection): tid
                    for tid, t in active_tracks.items()
                }
            else:
                track_id_map = {}

            # Estimate distances
            estimates = estimator.estimate_batch(result.detections)

            # Visualise
            vis = draw_crosshair(frame.copy(), alpha=0.25)
            vis = draw_distance_zones(
                vis, estimates,
                near_thresh_m=args.near,
                mid_thresh_m=args.mid,
            )

            # Overlay track IDs
            if tracker and track_id_map:
                for tid, track in active_tracks.items():
                    cx, cy = track.centroid.astype(int)
                    cv2.putText(
                        vis, f"ID:{tid}",
                        (cx + 5, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 0), 1, cv2.LINE_AA,
                    )

            # FPS counter (smoothed)
            t_now   = time.perf_counter()
            fps_raw = 1.0 / max(t_now - t_prev, 1e-9)
            fps_display = 0.9 * fps_display + 0.1 * fps_raw
            t_prev  = t_now

            draw_fps_hud(
                vis,
                fps=fps_display,
                inference_ms=result.inference_time_ms,
                num_detections=result.num_detections,
                frame_id=frame_id,
                extra={"Tracker": f"{tracker.num_active_tracks} tracks"} if tracker else None,
            )

            cv2.imshow("YOLO | Distance Estimation", vis)

            if writer:
                writer.write(vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested.")
                break
            elif key == ord("s"):
                fname = f"screenshot_{frame_id:06d}.jpg"
                cv2.imwrite(fname, vis)
                logger.info(f"Screenshot saved: {fname}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(f"Session stats: {detector.stats}")


if __name__ == "__main__":
    main()
