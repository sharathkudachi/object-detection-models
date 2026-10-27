import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.config import settings, ROOT
from backend.app.detection.yolo import ModelRegistry, YoloDetector
from backend.app.analysis.motion import estimate_roi_motion, direction
from backend.app.analysis.convoy import ConvoyAnalyzer
from backend.app.interfaces import Det
from backend.app.pipeline import run, _create_video_writer, _close_writer, _write_frame


class MockReq:
    def __init__(self, model="best1.pt", mode="military+optical_flow", profile="balanced", conf=0.35, iou=0.5, stride=2, imgsz=480):
        self.model = model
        self.mode = mode
        self.profile = profile
        self.conf = conf
        self.iou = iou
        self.stride = stride
        self.imgsz = imgsz
        self.link_dist_frac = 0.2
        self.min_members = 3
        self.min_persist_frames = 5
        self.motion_threshold = 2.0


def test_model_registry_caching():
    """Verify ModelRegistry reuses instances without reloading from disk."""
    m1 = ModelRegistry.get_or_load("best1.pt", "cpu")
    m2 = ModelRegistry.get_or_load("best1.pt", "cpu")
    assert m1 is m2, "ModelRegistry did not return cached model instance"
    assert hasattr(m1, "names")
    assert len(m1.names) == 10, f"Expected 10 military classes, got {len(m1.names)}"
    assert m1.names[0] == "camouflage_soldier"
    assert m1.names[2] == "military_tank"


def test_military_class_mapping():
    """Verify all 10 trained military asset classes are preserved."""
    expected_classes = [
        "camouflage_soldier", "weapon", "military_tank", "military_truck",
        "military_vehicle", "soldier", "civilian_vehicle", "military_artillery",
        "military_aircraft", "military_warship"
    ]
    det = YoloDetector(model="best1.pt", device="cpu", conf=0.35)
    det.load()
    for idx, name in enumerate(expected_classes):
        assert det.names.get(idx) == name, f"Class {idx} mismatch: expected {name}, got {det.names.get(idx)}"


def test_tracker_reset_between_jobs():
    """Verify ByteTrack tracker state can be reset cleanly between video jobs."""
    det = YoloDetector(model="best1.pt", device="cpu")
    det.load()
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    _ = det.predict(dummy)
    det.reset_tracker()
    assert True, "Tracker reset executed without errors"


def test_optical_flow_downscaled_accuracy():
    """Verify downscaled optical flow accurately distinguishes moving vs static."""
    # Synthetic stationary frame
    f1 = np.ones((200, 200), dtype=np.uint8) * 128
    f2 = np.ones((200, 200), dtype=np.uint8) * 128
    bbox = [10, 10, 150, 150]

    status, mag = estimate_roi_motion(f1, f2, bbox, max_dim=80, motion_threshold=2.0)
    assert status == "Static"
    assert mag < 1.0

    # Synthetic moving frame (textured noise frame shifted by 8 pixels)
    rng = np.random.RandomState(42)
    tex1 = rng.randint(0, 255, (200, 200), dtype=np.uint8)
    tex2 = np.roll(tex1, shift=8, axis=1)

    status_mov, mag_mov = estimate_roi_motion(tex1, tex2, bbox, max_dim=80, motion_threshold=2.0)
    assert status_mov == "Moving"
    assert mag_mov > 2.0


def test_video_writer_and_h264_encoding():
    """Verify H.264 video writer creates valid, playable MP4 file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "test_out.mp4"
        writer_dict = _create_video_writer(out_path, 320, 240, 25.0)
        
        for i in range(10):
            frame = np.full((240, 320, 3), (i * 20, 100, 200), dtype=np.uint8)
            _write_frame(writer_dict, frame)
        _close_writer(writer_dict)

        assert out_path.exists()
        assert out_path.stat().st_size > 500, "Output video is empty"

        # Verify OpenCV can read back the generated MP4
        cap = cv2.VideoCapture(str(out_path))
        assert cap.isOpened()
        ret, read_frame = cap.read()
        assert ret
        assert read_frame.shape == (240, 320, 3)
        cap.release()


def test_pipeline_execution_profiles():
    """Test full pipeline run with Balanced profile on a short test video segment."""
    video_source = ROOT / "test_video.mp4"
    if not video_source.exists():
        video_source = ROOT.parent / "ai-military-surveillance-system" / "test_video.mp4"

    assert video_source.exists(), "test_video.mp4 not found for pipeline test"

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a fast 10-frame clip from test_video.mp4 to keep unit test runtime < 3s
        short_video = Path(tmp_dir) / "short_test.mp4"
        cap = cv2.VideoCapture(str(video_source))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(short_video), fourcc, fps, (w, h))
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()

        json_out = Path(tmp_dir) / "test_results.json"
        video_out = Path(tmp_dir) / "test_annotated.mp4"

        job = {"id": "test_job_1", "status": "queued", "cancel": False}
        req = MockReq(model="best1.pt", mode="military+optical_flow", profile="balanced", stride=2, imgsz=480)

        # Run pipeline
        run(job, req, short_video, json_out, video_out)

        assert job["status"] == "completed"
        assert json_out.exists()
        assert video_out.exists()
        assert video_out.stat().st_size > 1000

        # Validate JSON metadata
        data = json.loads(json_out.read_text())
        assert "meta" in data
        assert "frames" in data
        assert "perf" in data
        assert data["meta"]["profile"] == "balanced"
        assert data["perf"]["avg_fps"] > 0
        assert "timing_breakdown" in data["perf"]

