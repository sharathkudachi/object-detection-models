"""Unit tests for distance estimation modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.detector import Detection
from src.distance_estimator import (
    CameraIntrinsics,
    GeometricDistanceEstimator,
    StereoDistanceEstimator,
    HybridDistanceEstimator,
    EstimationMethod,
    KNOWN_OBJECT_HEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def intrinsics():
    """Typical 1080p camera intrinsics."""
    return CameraIntrinsics(fx=1200.0, fy=1200.0, cx=960.0, cy=540.0)


@pytest.fixture
def person_detection():
    """A 'person' detection whose pixel height == 200px at a known distance."""
    return Detection(
        bbox=np.array([310.0, 340.0, 410.0, 540.0]),
        class_id=0,
        class_name="person",
        confidence=0.92,
    )


@pytest.fixture
def geo_estimator(intrinsics):
    return GeometricDistanceEstimator(intrinsics)


# ---------------------------------------------------------------------------
# CameraIntrinsics
# ---------------------------------------------------------------------------

class TestCameraIntrinsics:
    def test_matrix_shape(self, intrinsics):
        assert intrinsics.matrix.shape == (3, 3)

    def test_matrix_values(self, intrinsics):
        K = intrinsics.matrix
        assert K[0, 0] == pytest.approx(1200.0)
        assert K[1, 1] == pytest.approx(1200.0)
        assert K[0, 2] == pytest.approx(960.0)
        assert K[1, 2] == pytest.approx(540.0)

    def test_default_dist_coeffs(self, intrinsics):
        assert intrinsics.dist_coeffs is not None
        assert len(intrinsics.dist_coeffs) == 5
        assert np.all(intrinsics.dist_coeffs == 0)

    def test_from_matrix(self, intrinsics):
        restored = CameraIntrinsics.from_matrix(intrinsics.matrix)
        assert restored.fx == pytest.approx(intrinsics.fx)
        assert restored.fy == pytest.approx(intrinsics.fy)

    def test_roundtrip(self):
        K = np.array([
            [800.0,   0.0, 320.0],
            [  0.0, 810.0, 240.0],
            [  0.0,   0.0,   1.0],
        ])
        intr = CameraIntrinsics.from_matrix(K)
        assert intr.fx == pytest.approx(800.0)
        assert intr.cy == pytest.approx(240.0)


# ---------------------------------------------------------------------------
# GeometricDistanceEstimator
# ---------------------------------------------------------------------------

class TestGeometricDistanceEstimator:
    def test_known_class_formula(self, geo_estimator, intrinsics):
        """distance = real_height * fy / pixel_height"""
        pixel_h = 200.0
        real_h  = KNOWN_OBJECT_HEIGHTS["person"]          # 1.70 m
        expected = (real_h * intrinsics.fy) / pixel_h     # 10.20 m

        det = Detection(
            bbox=np.array([100.0, 100.0, 200.0, 100.0 + pixel_h]),
            class_id=0,
            class_name="person",
            confidence=0.9,
        )
        result = geo_estimator.estimate(det)
        assert result.distance_m == pytest.approx(expected, rel=1e-3)

    def test_returns_geometric_method(self, geo_estimator, person_detection):
        result = geo_estimator.estimate(person_detection)
        assert result.method == EstimationMethod.GEOMETRIC

    def test_unknown_class_uses_fallback(self, intrinsics):
        estimator = GeometricDistanceEstimator(intrinsics, fallback_height=1.0)
        det = Detection(
            bbox=np.array([0.0, 0.0, 50.0, 100.0]),
            class_id=99,
            class_name="alien_spacecraft",
            confidence=0.7,
        )
        result = estimator.estimate(det)
        expected = (1.0 * intrinsics.fy) / 100.0
        assert result.distance_m == pytest.approx(expected, rel=1e-3)

    def test_unknown_class_lowers_confidence(self, geo_estimator, intrinsics):
        det = Detection(
            bbox=np.array([0.0, 0.0, 100.0, 200.0]),
            class_id=99,
            class_name="unknown_thing",
            confidence=1.0,
        )
        result = geo_estimator.estimate(det)
        assert result.confidence < 1.0

    def test_distance_positive(self, geo_estimator, person_detection):
        result = geo_estimator.estimate(person_detection)
        assert result.distance_m > 0.0

    def test_farther_object_less_pixel_height(self, intrinsics):
        """Smaller pixel height → greater estimated distance."""
        def make_det(pixel_h: float) -> Detection:
            return Detection(
                bbox=np.array([100.0, 100.0, 200.0, 100.0 + pixel_h]),
                class_id=0, class_name="person", confidence=0.9,
            )

        estimator = GeometricDistanceEstimator(intrinsics)
        near = estimator.estimate(make_det(400.0))
        far  = estimator.estimate(make_det(100.0))
        assert far.distance_m > near.distance_m

    def test_batch_length(self, geo_estimator, person_detection):
        dets   = [person_detection] * 5
        results = geo_estimator.estimate_batch(dets)
        assert len(results) == 5

    def test_distance_to_dict(self, geo_estimator, person_detection):
        result = geo_estimator.estimate(person_detection)
        d = result.to_dict()
        assert "distance_m" in d
        assert "distance_ft" in d
        assert "method" in d

    def test_feet_conversion(self, geo_estimator, person_detection):
        result = geo_estimator.estimate(person_detection)
        assert result.distance_ft == pytest.approx(result.distance_m * 3.28084, rel=1e-4)

    def test_very_small_bbox_lowers_confidence(self, intrinsics):
        estimator = GeometricDistanceEstimator(intrinsics)
        det = Detection(
            bbox=np.array([100.0, 100.0, 120.0, 110.0]),   # tiny bbox, height=10
            class_id=0, class_name="person", confidence=0.9,
        )
        result = estimator.estimate(det)
        # height < 20 applies a 0.70 factor → confidence ≤ 0.9 * 0.70 = 0.63
        assert result.confidence <= 0.9 * 0.70


# ---------------------------------------------------------------------------
# StereoDistanceEstimator
# ---------------------------------------------------------------------------

class TestStereoDistanceEstimator:
    @pytest.fixture
    def stereo(self, intrinsics):
        return StereoDistanceEstimator(intrinsics, baseline_m=0.12, min_disparity=1.0)

    def _disparity_for_distance(self, intrinsics, baseline, distance):
        return (intrinsics.fx * baseline) / distance

    def test_formula(self, stereo, intrinsics):
        """Z = fx * baseline / disparity."""
        target_distance = 5.0   # metres
        disp_val = self._disparity_for_distance(intrinsics, 0.12, target_distance)

        # Build a disparity map where the bbox ROI is filled with disp_val
        disp_map = np.full((480, 640), disp_val, dtype=np.float32)
        det = Detection(
            bbox=np.array([200.0, 150.0, 400.0, 350.0]),
            class_id=2, class_name="car", confidence=0.85,
        )
        result = stereo.estimate(det, disp_map)
        assert result.distance_m == pytest.approx(target_distance, rel=0.01)

    def test_no_valid_disparity_returns_max(self, stereo):
        """Zero disparity map → max range returned."""
        disp_map = np.zeros((480, 640), dtype=np.float32)
        det = Detection(
            bbox=np.array([100.0, 100.0, 200.0, 200.0]),
            class_id=0, class_name="person", confidence=0.8,
        )
        result = stereo.estimate(det, disp_map)
        assert result.distance_m >= 999.0
        assert result.confidence == 0.0

    def test_method_is_stereo(self, stereo, intrinsics):
        disp_val = self._disparity_for_distance(intrinsics, 0.12, 3.0)
        disp_map = np.full((480, 640), disp_val, dtype=np.float32)
        det = Detection(
            bbox=np.array([100.0, 100.0, 300.0, 300.0]),
            class_id=0, class_name="person", confidence=0.9,
        )
        result = stereo.estimate(det, disp_map)
        assert result.method == EstimationMethod.STEREO

    def test_bbox_clamped_to_frame(self, stereo):
        """Bboxes partially outside frame should not raise."""
        disp_map = np.full((480, 640), 50.0, dtype=np.float32)
        det = Detection(
            bbox=np.array([-10.0, -10.0, 700.0, 500.0]),   # out of bounds
            class_id=0, class_name="car", confidence=0.7,
        )
        result = stereo.estimate(det, disp_map)
        assert result.distance_m > 0.0


# ---------------------------------------------------------------------------
# HybridDistanceEstimator
# ---------------------------------------------------------------------------

class TestHybridDistanceEstimator:
    @pytest.fixture
    def hybrid(self, intrinsics):
        return HybridDistanceEstimator(intrinsics, baseline_m=0.12)

    def test_falls_back_to_geometric_without_disparity(self, hybrid, person_detection):
        result = hybrid.estimate(person_detection, disparity_map=None)
        assert result.method == EstimationMethod.GEOMETRIC

    def test_uses_hybrid_with_disparity(self, hybrid, intrinsics, person_detection):
        disp_val = (intrinsics.fx * 0.12) / 5.0
        disp_map = np.full((480, 640), disp_val, dtype=np.float32)
        result   = hybrid.estimate(person_detection, disp_map)
        assert result.method == EstimationMethod.HYBRID

    def test_fused_distance_between_geo_and_stereo(self, hybrid, intrinsics, person_detection):
        """Fused estimate should be between the two component estimates."""
        geo_dist = (KNOWN_OBJECT_HEIGHTS["person"] * intrinsics.fy) / person_detection.height

        stereo_target = 8.0
        disp_val = (intrinsics.fx * 0.12) / stereo_target
        disp_map = np.full((480, 640), disp_val, dtype=np.float32)

        result = hybrid.estimate(person_detection, disp_map)
        lo = min(geo_dist, stereo_target)
        hi = max(geo_dist, stereo_target)
        assert lo * 0.9 <= result.distance_m <= hi * 1.1

    def test_batch_no_disparity(self, hybrid, person_detection):
        results = hybrid.estimate_batch([person_detection] * 4)
        assert len(results) == 4
        assert all(r.method == EstimationMethod.GEOMETRIC for r in results)
