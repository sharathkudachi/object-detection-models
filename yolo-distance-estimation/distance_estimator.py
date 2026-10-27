"""
Distance Estimation Module
===========================
Estimates the real-world distance from camera to detected objects
using two complementary methods:

  1. Geometric (monocular) — uses known object dimensions + focal length.
  2. Stereo — uses disparity from a rectified stereo pair.

Both methods rely on camera intrinsics from the calibration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from src.detector import Detection
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Known real-world object heights (metres) for monocular estimation
# ---------------------------------------------------------------------------
KNOWN_OBJECT_HEIGHTS: dict[str, float] = {
    "person":      1.70,
    "car":         1.50,
    "truck":       3.50,
    "bus":         3.20,
    "bicycle":     1.00,
    "motorbike":   1.10,
    "chair":       0.85,
    "bottle":      0.25,
    "laptop":      0.03,
    "cell phone":  0.14,
    "dog":         0.50,
    "cat":         0.30,
    "traffic light": 0.80,
    "stop sign":   0.75,
}


class EstimationMethod(Enum):
    GEOMETRIC = auto()
    STEREO    = auto()
    HYBRID    = auto()


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters.

    Attributes:
        fx: Focal length in pixels (x-axis).
        fy: Focal length in pixels (y-axis).
        cx: Principal point x.
        cy: Principal point y.
        matrix: 3x3 camera matrix K.
        dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3).
    """
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray = None

    def __post_init__(self):
        self.matrix = np.array([
            [self.fx,    0.0, self.cx],
            [   0.0, self.fy, self.cy],
            [   0.0,    0.0,    1.0],
        ], dtype=np.float64)

        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5, dtype=np.float64)

    @classmethod
    def from_matrix(
        cls,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> "CameraIntrinsics":
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            dist_coeffs=dist_coeffs,
        )

    @property
    def fov_x_deg(self) -> float:
        """Horizontal field of view in degrees (requires image width)."""
        raise NotImplementedError("Provide image_width to compute FOV.")


@dataclass
class DistanceEstimate:
    """Output of the distance estimator for one detection."""
    detection: Detection
    distance_m: float
    method: EstimationMethod
    confidence: float           # Estimation confidence [0, 1]

    @property
    def distance_cm(self) -> float:
        return self.distance_m * 100.0

    @property
    def distance_ft(self) -> float:
        return self.distance_m * 3.28084

    def to_dict(self) -> dict:
        return {
            **self.detection.to_dict(),
            "distance_m":  round(self.distance_m, 3),
            "distance_ft": round(self.distance_ft, 3),
            "method":      self.method.name,
            "est_confidence": round(self.confidence, 3),
        }


class GeometricDistanceEstimator:
    """
    Monocular distance estimator using the pinhole camera model.

    Formula:
        distance = (real_height * focal_length_y) / pixel_height

    This gives the depth Z along the optical axis assuming the object
    is upright and its base is on the ground plane.

    Args:
        intrinsics: Calibrated camera intrinsic parameters.
        known_heights: Mapping from class_name → real-world height (m).
        fallback_height: Used when class is not in known_heights.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        known_heights: dict[str, float] = None,
        fallback_height: float = 1.0,
    ) -> None:
        self.intrinsics = intrinsics
        self.known_heights = known_heights or KNOWN_OBJECT_HEIGHTS
        self.fallback_height = fallback_height

    def estimate(self, detection: Detection) -> DistanceEstimate:
        """
        Estimate distance for a single detection.

        Confidence is penalised when:
         - The class is unknown (using fallback height).
         - The bounding box aspect ratio is unusual.
         - The pixel height is very small (far objects).
        """
        real_h = self.known_heights.get(
            detection.class_name.lower(), None
        )
        is_known = real_h is not None
        if not is_known:
            real_h = self.fallback_height
            logger.debug(
                f"Unknown class '{detection.class_name}', "
                f"using fallback height {self.fallback_height} m"
            )

        pixel_h = max(detection.height, 1.0)
        distance_m = (real_h * self.intrinsics.fy) / pixel_h

        # Confidence heuristics
        conf = detection.confidence
        if not is_known:
            conf *= 0.60
        if pixel_h < 20:
            conf *= 0.70     # Very small bbox → noisy estimate
        if detection.aspect_ratio > 5 or detection.aspect_ratio < 0.1:
            conf *= 0.80     # Unusual shape → height may be truncated

        return DistanceEstimate(
            detection=detection,
            distance_m=max(distance_m, 0.01),
            method=EstimationMethod.GEOMETRIC,
            confidence=min(conf, 1.0),
        )

    def estimate_batch(
        self, detections: list[Detection]
    ) -> list[DistanceEstimate]:
        return [self.estimate(d) for d in detections]


class StereoDistanceEstimator:
    """
    Stereo-based distance estimator using a pre-computed disparity map.

    Uses the standard stereo formula:
        Z = (fx * baseline) / disparity

    Args:
        intrinsics: Camera intrinsics for the left camera.
        baseline_m: Physical separation between left and right cameras (m).
        min_disparity: Disparity values below this are considered invalid.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        baseline_m: float = 0.12,
        min_disparity: float = 1.0,
    ) -> None:
        self.intrinsics = intrinsics
        self.baseline = baseline_m
        self.min_disparity = min_disparity

    def estimate(
        self,
        detection: Detection,
        disparity_map: np.ndarray,
    ) -> DistanceEstimate:
        """
        Estimate distance using the disparity map.

        Samples a central ROI of the bounding box and uses the
        median disparity to suppress noisy pixels.
        """
        x1, y1, x2, y2 = detection.bbox.astype(int)
        h, w = disparity_map.shape[:2]

        # Clamp to frame
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)

        # Use central 60% of bbox to avoid edge noise
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rw = max((x2 - x1) // 3, 4)
        rh = max((y2 - y1) // 3, 4)

        roi = disparity_map[
            max(cy - rh, 0): cy + rh,
            max(cx - rw, 0): cx + rw,
        ]

        valid = roi[roi >= self.min_disparity]
        if valid.size == 0:
            logger.warning(
                f"No valid disparity in ROI for {detection.class_name}. "
                "Returning max-range distance."
            )
            return DistanceEstimate(
                detection=detection,
                distance_m=999.0,
                method=EstimationMethod.STEREO,
                confidence=0.0,
            )

        median_disp = float(np.median(valid))
        distance_m = (self.intrinsics.fx * self.baseline) / median_disp
        est_conf = min(float(valid.size) / roi.size, 1.0) * detection.confidence

        return DistanceEstimate(
            detection=detection,
            distance_m=max(distance_m, 0.01),
            method=EstimationMethod.STEREO,
            confidence=est_conf,
        )


class HybridDistanceEstimator:
    """
    Fuses geometric and stereo estimates with a confidence-weighted average.

    Prefers stereo when valid disparity is available and falls back
    to the geometric estimator for objects outside the stereo range
    or when a disparity map is not provided.

    Args:
        intrinsics: Camera intrinsics.
        baseline_m: Stereo baseline (metres).
        known_heights: Real-world object heights for monocular fallback.
        stereo_weight: Weight given to stereo estimate in fusion.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        baseline_m: float = 0.12,
        known_heights: dict[str, float] = None,
        stereo_weight: float = 0.7,
    ) -> None:
        self.geometric = GeometricDistanceEstimator(intrinsics, known_heights)
        self.stereo    = StereoDistanceEstimator(intrinsics, baseline_m)
        self.stereo_weight = stereo_weight
        self.geo_weight    = 1.0 - stereo_weight

    def estimate(
        self,
        detection: Detection,
        disparity_map: Optional[np.ndarray] = None,
    ) -> DistanceEstimate:
        geo_est = self.geometric.estimate(detection)

        if disparity_map is None:
            geo_est.method = EstimationMethod.GEOMETRIC
            return geo_est

        ste_est = self.stereo.estimate(detection, disparity_map)

        if ste_est.confidence < 0.1:
            return geo_est

        # Confidence-weighted fusion
        total_w = (
            self.stereo_weight * ste_est.confidence
            + self.geo_weight  * geo_est.confidence
        )
        if total_w < 1e-6:
            fused_dist = (geo_est.distance_m + ste_est.distance_m) / 2
        else:
            fused_dist = (
                self.stereo_weight * ste_est.confidence * ste_est.distance_m
                + self.geo_weight  * geo_est.confidence * geo_est.distance_m
            ) / total_w

        fused_conf = max(geo_est.confidence, ste_est.confidence)

        return DistanceEstimate(
            detection=detection,
            distance_m=fused_dist,
            method=EstimationMethod.HYBRID,
            confidence=fused_conf,
        )

    def estimate_batch(
        self,
        detections: list[Detection],
        disparity_map: Optional[np.ndarray] = None,
    ) -> list[DistanceEstimate]:
        return [self.estimate(d, disparity_map) for d in detections]
