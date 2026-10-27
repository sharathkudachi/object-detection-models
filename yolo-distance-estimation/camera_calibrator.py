"""
Camera Calibration Pipeline
=============================
Full chessboard-based monocular camera calibration using OpenCV.

Workflow:
  1. Collect frames containing a printed chessboard pattern.
  2. Detect corners sub-pixel accurately with cornerSubPix.
  3. Run calibrateCamera to estimate K and distortion coefficients.
  4. Compute and report reprojection error.
  5. Persist results to a JSON file for downstream use.

Usage (script):
    python scripts/calibrate_camera.py --images data/calibration_images \
        --pattern 9x6 --square-size 0.025

Usage (API):
    calibrator = CameraCalibrator(pattern_size=(9, 6), square_size_m=0.025)
    calibrator.add_frames_from_directory("data/calibration_images")
    intrinsics = calibrator.calibrate()
    calibrator.save("config/calibration.json")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.distance_estimator import CameraIntrinsics
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Sub-pixel corner refinement termination criteria
_SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


class CalibrationError(RuntimeError):
    """Raised when calibration cannot proceed."""


class CameraCalibrator:
    """
    Chessboard-based camera calibrator.

    Args:
        pattern_size: Inner corner count (cols, rows). For a 10×7
                      board the inner pattern is (9, 6).
        square_size_m: Physical size of each square in metres.
        flags: OpenCV calibrateCamera flags. Defaults are suitable for
               most lenses; add cv2.CALIB_RATIONAL_MODEL for fish-eye.
    """

    def __init__(
        self,
        pattern_size: tuple[int, int] = (9, 6),
        square_size_m: float = 0.025,
        flags: int = cv2.CALIB_FIX_ASPECT_RATIO,
    ) -> None:
        self.pattern_size   = pattern_size
        self.square_size_m  = square_size_m
        self.flags          = flags

        # 3D world-space coordinates of the chessboard corners
        self._obj_pts_template = self._make_obj_pts()

        self._obj_points: list[np.ndarray] = []   # World space
        self._img_points: list[np.ndarray] = []   # Image space
        self._image_size: Optional[tuple[int, int]] = None
        self._calibration: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Process one frame and extract chessboard corners.

        Args:
            frame: BGR or grayscale image.

        Returns:
            True if corners were found, False otherwise.
        """
        gray = self._to_gray(frame)
        h, w = gray.shape

        if self._image_size is None:
            self._image_size = (w, h)
        elif self._image_size != (w, h):
            logger.warning(
                f"Frame size mismatch: expected {self._image_size}, "
                f"got {(w, h)}. Skipping."
            )
            return False

        found, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FAST_CHECK
            ),
        )

        if not found:
            return False

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA
        )

        self._obj_points.append(self._obj_pts_template)
        self._img_points.append(corners_refined)

        logger.debug(f"Corners found. Total frames: {self.num_frames}")
        return True

    def add_frames_from_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        max_frames: int = 60,
    ) -> int:
        """
        Scan a directory and add all valid calibration frames.

        Args:
            directory: Path containing calibration images.
            extensions: Accepted image file extensions.
            max_frames: Cap on how many frames to process.

        Returns:
            Number of frames successfully added.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        image_paths = sorted(
            p for p in directory.iterdir()
            if p.suffix.lower() in extensions
        )[:max_frames]

        if not image_paths:
            raise CalibrationError(
                f"No images found in {directory} with extensions {extensions}"
            )

        logger.info(f"Processing {len(image_paths)} calibration images...")
        success = 0
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                logger.warning(f"Could not read {path}")
                continue
            if self.add_frame(frame):
                success += 1
            else:
                logger.debug(f"No corners found: {path.name}")

        logger.info(f"Accepted {success}/{len(image_paths)} frames.")
        return success

    def calibrate(self, min_frames: int = 10) -> CameraIntrinsics:
        """
        Run calibration and return camera intrinsics.

        Args:
            min_frames: Minimum frames needed for a valid calibration.

        Returns:
            CameraIntrinsics with K and distortion coefficients.

        Raises:
            CalibrationError: If fewer than min_frames are available.
        """
        if self.num_frames < min_frames:
            raise CalibrationError(
                f"Need at least {min_frames} frames, "
                f"but only {self.num_frames} detected corners successfully."
            )

        logger.info(
            f"Calibrating with {self.num_frames} frames "
            f"({self.pattern_size[0]}x{self.pattern_size[1]} pattern, "
            f"{self.square_size_m*100:.1f} cm squares)..."
        )
        t0 = time.perf_counter()

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._obj_points,
            self._img_points,
            self._image_size,
            None,
            None,
            flags=self.flags,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        rpe = self._reprojection_error(K, dist, rvecs, tvecs)

        logger.info(
            f"Calibration done in {elapsed:.0f} ms | "
            f"RMS reprojection error: {ret:.4f} px | "
            f"Mean point error: {rpe:.4f} px"
        )

        self._calibration = {
            "rms_error": float(ret),
            "mean_reprojection_error_px": float(rpe),
            "image_size": list(self._image_size),
            "num_frames": self.num_frames,
            "pattern_size": list(self.pattern_size),
            "square_size_m": self.square_size_m,
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist.flatten().tolist(),
        }

        return CameraIntrinsics.from_matrix(K, dist.flatten())

    def save(self, path: str | Path) -> None:
        """Persist calibration results to JSON."""
        if self._calibration is None:
            raise CalibrationError("Call calibrate() before save().")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._calibration, f, indent=2)

        logger.info(f"Calibration saved → {path}")

    @staticmethod
    def load(path: str | Path) -> CameraIntrinsics:
        """
        Load previously saved calibration from JSON.

        Returns:
            CameraIntrinsics restored from file.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        K = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"], dtype=np.float64)

        logger.info(
            f"Loaded calibration from {path} | "
            f"RMS error: {data.get('rms_error', 'N/A'):.4f} px"
        )
        return CameraIntrinsics.from_matrix(K, dist)

    def draw_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Draw detected corners on the frame (for visual verification).

        Returns:
            Annotated frame, or None if no corners found in this frame.
        """
        gray = self._to_gray(frame)
        found, corners = cv2.findChessboardCorners(
            gray, self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            return None

        out = frame.copy()
        cv2.drawChessboardCorners(out, self.pattern_size, corners, found)
        return out

    def undistort(self, frame: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """Apply undistortion to a frame using calibrated parameters."""
        h, w = frame.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, K, dist, None, new_K)
        x, y, rw, rh = roi
        return undistorted[y:y+rh, x:x+rw]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return len(self._obj_points)

    @property
    def calibration_data(self) -> Optional[dict]:
        return self._calibration

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_obj_pts(self) -> np.ndarray:
        """Generate 3D object-space corner coordinates for the pattern."""
        cols, rows = self.pattern_size
        pts = np.zeros((cols * rows, 3), dtype=np.float32)
        pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        pts *= self.square_size_m
        return pts

    def _reprojection_error(
        self,
        K: np.ndarray,
        dist: np.ndarray,
        rvecs: list,
        tvecs: list,
    ) -> float:
        """Compute mean per-point reprojection error across all frames."""
        errors = []
        for obj_pts, img_pts, rvec, tvec in zip(
            self._obj_points, self._img_points, rvecs, tvecs
        ):
            projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            error = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
            errors.append(error)
        return float(np.mean(errors))

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
