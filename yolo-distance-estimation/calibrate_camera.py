#!/usr/bin/env python3
"""
Camera Calibration Script
==========================
Collect calibration frames from webcam or directory and save intrinsics.

Usage:
    # Interactive webcam capture (press SPACE to capture, Q to quit & calibrate)
    python scripts/calibrate_camera.py --mode webcam

    # Calibrate from existing images
    python scripts/calibrate_camera.py --mode images \
        --images data/calibration_images \
        --pattern 9x6 --square-size 0.025

    # Output saved to config/calibration.json by default
    python scripts/calibrate_camera.py --output config/my_camera.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.calibration.camera_calibrator import CameraCalibrator, CalibrationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera calibration tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["webcam", "images"], default="images")
    p.add_argument("--images", default="data/calibration_images",
                   help="Directory of calibration images (mode=images).")
    p.add_argument("--pattern", default="9x6",
                   help="Chessboard inner corners, e.g. '9x6'.")
    p.add_argument("--square-size", type=float, default=0.025,
                   help="Physical square side length in metres.")
    p.add_argument("--output", default="config/calibration.json",
                   help="Output path for calibration JSON.")
    p.add_argument("--source", type=int, default=0, help="Webcam index.")
    p.add_argument("--min-frames", type=int, default=15,
                   help="Minimum valid frames required for calibration.")
    return p.parse_args()


def webcam_mode(
    calibrator: CameraCalibrator,
    source: int,
    min_frames: int,
    output: str,
) -> None:
    """Interactive webcam-based calibration capture."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {source}")

    logger.info(
        "Webcam mode: Press SPACE to capture a frame, Q to finish & calibrate."
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preview = calibrator.draw_corners(frame) or frame.copy()

        status = (
            f"Captured: {calibrator.num_frames} | "
            f"Press SPACE to capture, Q to calibrate"
        )
        cv2.putText(preview, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration Capture", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            added = calibrator.add_frame(frame)
            msg   = "✓ Corners found!" if added else "✗ No corners — try again."
            logger.info(f"Frame {calibrator.num_frames}: {msg}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if calibrator.num_frames < min_frames:
        logger.error(
            f"Only {calibrator.num_frames} valid frames — need {min_frames}."
        )
        sys.exit(1)

    intrinsics = calibrator.calibrate(min_frames=min_frames)
    calibrator.save(output)
    logger.info(f"fx={intrinsics.fx:.2f}  fy={intrinsics.fy:.2f}  "
                f"cx={intrinsics.cx:.2f}  cy={intrinsics.cy:.2f}")


def images_mode(
    calibrator: CameraCalibrator,
    images_dir: str,
    min_frames: int,
    output: str,
) -> None:
    """Calibrate from a directory of pre-captured images."""
    n = calibrator.add_frames_from_directory(images_dir, max_frames=80)
    if n < min_frames:
        raise CalibrationError(
            f"Only {n} valid frames found in {images_dir!r}. "
            f"Need at least {min_frames}."
        )

    intrinsics = calibrator.calibrate(min_frames=min_frames)
    calibrator.save(output)

    rpe = calibrator.calibration_data["mean_reprojection_error_px"]
    logger.info(f"Reprojection error: {rpe:.4f} px  (< 0.5 is excellent)")
    logger.info(
        f"fx={intrinsics.fx:.2f}  fy={intrinsics.fy:.2f}  "
        f"cx={intrinsics.cx:.2f}  cy={intrinsics.cy:.2f}"
    )


def main() -> None:
    args = parse_args()

    cols, rows = map(int, args.pattern.lower().split("x"))
    calibrator = CameraCalibrator(
        pattern_size=(cols, rows),
        square_size_m=args.square_size,
    )

    if args.mode == "webcam":
        webcam_mode(calibrator, args.source, args.min_frames, args.output)
    else:
        images_mode(calibrator, args.images, args.min_frames, args.output)


if __name__ == "__main__":
    main()
