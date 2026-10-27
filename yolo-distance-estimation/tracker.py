"""
Centroid-Based Object Tracker
================================
Lightweight multi-object tracker using Euclidean centroid distance.
Maintains track IDs across frames without requiring re-identification.

For production use, swap this for DeepSORT or ByteTrack.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

from src.detector import Detection
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    """A tracked object with history."""
    track_id: int
    detection: Detection
    centroid: np.ndarray
    age: int = 1                        # Frames since first seen
    frames_since_seen: int = 0
    centroid_history: list = field(default_factory=list)

    def update(self, detection: Detection) -> None:
        self.detection = detection
        self.centroid  = np.array(detection.center)
        self.centroid_history.append(self.centroid.copy())
        self.age += 1
        self.frames_since_seen = 0

    @property
    def velocity(self) -> np.ndarray:
        """Estimated velocity vector (pixels/frame)."""
        if len(self.centroid_history) < 2:
            return np.zeros(2)
        return self.centroid_history[-1] - self.centroid_history[-2]


class CentroidTracker:
    """
    Associates detections to existing tracks by nearest centroid.

    Args:
        max_disappeared: Frames a track can go unmatched before removal.
        max_distance: Maximum pixel distance for a valid association.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 80.0,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

        self._next_id: int = 0
        self.tracks: OrderedDict[int, Track] = OrderedDict()

    def update(self, detections: list[Detection]) -> dict[int, Track]:
        """
        Update tracker with current frame detections.

        Args:
            detections: Current frame detections.

        Returns:
            Active tracks keyed by track_id.
        """
        # -- No detections: age out all tracks
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid].frames_since_seen += 1
                if self.tracks[tid].frames_since_seen > self.max_disappeared:
                    del self.tracks[tid]
            return dict(self.tracks)

        new_centroids = np.array([d.center for d in detections])

        # -- No existing tracks: register all
        if not self.tracks:
            for det in detections:
                self._register(det)
            return dict(self.tracks)

        # -- Match existing tracks to detections
        existing_ids      = list(self.tracks.keys())
        existing_centroids = np.array([
            self.tracks[tid].centroid for tid in existing_ids
        ])

        dist_matrix = cdist(existing_centroids, new_centroids)
        row_ind = dist_matrix.min(axis=1).argsort()
        col_ind = dist_matrix.argmin(axis=1)

        matched_rows: set[int] = set()
        matched_cols: set[int] = set()

        for row in row_ind:
            col = col_ind[row]
            if row in matched_rows or col in matched_cols:
                continue
            if dist_matrix[row, col] > self.max_distance:
                continue

            tid = existing_ids[row]
            self.tracks[tid].update(detections[col])
            matched_rows.add(row)
            matched_cols.add(col)

        # Age unmatched tracks
        for i, tid in enumerate(existing_ids):
            if i not in matched_rows:
                self.tracks[tid].frames_since_seen += 1
                if self.tracks[tid].frames_since_seen > self.max_disappeared:
                    del self.tracks[tid]

        # Register unmatched detections as new tracks
        for j, det in enumerate(detections):
            if j not in matched_cols:
                self._register(det)

        return dict(self.tracks)

    def _register(self, detection: Detection) -> None:
        self.tracks[self._next_id] = Track(
            track_id=self._next_id,
            detection=detection,
            centroid=np.array(detection.center),
            centroid_history=[np.array(detection.center)],
        )
        self._next_id += 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0

    @property
    def num_active_tracks(self) -> int:
        return len(self.tracks)
