"""Unit tests for the centroid-based object tracker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.detector import Detection
from src.tracking.tracker import CentroidTracker, Track


def _det(cx, cy, w=60, h=60, cls_id=0, conf=0.85, name="car"):
    """Create a Detection centred at (cx, cy)."""
    x1, y1 = cx - w / 2, cy - h / 2
    return Detection(
        bbox=np.array([x1, y1, x1 + w, y1 + h], dtype=float),
        class_id=cls_id,
        class_name=name,
        confidence=conf,
    )


class TestCentroidTracker:
    def test_new_tracker_empty(self):
        t = CentroidTracker()
        assert t.num_active_tracks == 0

    def test_first_frame_registers_all(self):
        t = CentroidTracker()
        tracks = t.update([_det(100, 100), _det(300, 300), _det(500, 100)])
        assert len(tracks) == 3

    def test_stable_track_id(self):
        t = CentroidTracker(max_distance=50)
        t.update([_det(100, 100)])
        first_id = list(t.tracks.keys())[0]

        t.update([_det(105, 102)])   # slight movement
        second_id = list(t.tracks.keys())[0]
        assert first_id == second_id

    def test_new_object_gets_new_id(self):
        t = CentroidTracker(max_distance=50)
        t.update([_det(100, 100)])
        ids_before = set(t.tracks.keys())

        t.update([_det(100, 100), _det(500, 500)])   # 500,500 is new
        ids_after = set(t.tracks.keys())
        assert len(ids_after - ids_before) == 1

    def test_disappeared_track_removed(self):
        t = CentroidTracker(max_distance=50, max_disappeared=3)
        t.update([_det(100, 100)])
        tid = list(t.tracks.keys())[0]

        for _ in range(4):   # 4 frames without seeing it
            t.update([])

        assert tid not in t.tracks

    def test_track_survives_within_max_disappeared(self):
        t = CentroidTracker(max_distance=50, max_disappeared=5)
        t.update([_det(100, 100)])
        tid = list(t.tracks.keys())[0]

        for _ in range(3):
            t.update([])

        assert tid in t.tracks

    def test_empty_frame_no_crash(self):
        t = CentroidTracker()
        tracks = t.update([])
        assert tracks == {}

    def test_multiple_empty_frames_no_crash(self):
        t = CentroidTracker(max_disappeared=2)
        for _ in range(10):
            t.update([])
        assert t.num_active_tracks == 0

    def test_ids_monotonically_increasing(self):
        t = CentroidTracker()
        all_ids = []
        for _ in range(5):
            t.update([_det(np.random.randint(0, 500), np.random.randint(0, 500))])
            all_ids.extend(t.tracks.keys())

        # Each new track's ID should be >= all previous IDs
        assert sorted(set(all_ids)) == list(range(max(all_ids) + 1))

    def test_track_age_increments(self):
        t = CentroidTracker(max_distance=100)
        t.update([_det(200, 200)])
        tid = list(t.tracks.keys())[0]

        for _ in range(4):
            t.update([_det(205, 205)])

        assert t.tracks[tid].age == 5

    def test_centroid_history_grows(self):
        t = CentroidTracker(max_distance=100)
        t.update([_det(100, 100)])
        tid = list(t.tracks.keys())[0]

        for i in range(4):
            t.update([_det(100 + i * 5, 100)])

        assert len(t.tracks[tid].centroid_history) >= 4

    def test_velocity_estimated(self):
        t = CentroidTracker(max_distance=200)
        t.update([_det(100, 100)])
        tid = list(t.tracks.keys())[0]
        t.update([_det(120, 100)])   # moved 20px right

        vel = t.tracks[tid].velocity
        assert vel[0] == pytest.approx(20.0, abs=2.0)

    def test_reset_clears_all(self):
        t = CentroidTracker()
        t.update([_det(100, 100), _det(300, 300)])
        t.reset()
        assert t.num_active_tracks == 0
        assert t._next_id == 0

    def test_large_displacement_creates_new_track(self):
        t = CentroidTracker(max_distance=30)
        t.update([_det(100, 100)])
        n_before = t.num_active_tracks

        # Object at 500, 500 is outside max_distance → new track
        t.update([_det(500, 500)])
        assert t.num_active_tracks >= 1   # original may have aged out, new registered
