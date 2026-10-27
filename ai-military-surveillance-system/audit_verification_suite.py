import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

from backend.app.analysis.convoy import ConvoyAnalyzer
from backend.app.analysis.motion import estimate_roi_motion, direction
from backend.app.config import ROOT, settings
from backend.app.detection.yolo import ModelRegistry, YoloDetector, resolve_device
from backend.app.pipeline import run, _create_video_writer, _close_writer, _write_frame


class MockReq:
    def __init__(self, mode="military+optical_flow", profile="balanced", stride=2, imgsz=480, model="best1.pt"):
        self.model = model
        self.mode = mode
        self.profile = profile
        self.conf = 0.35
        self.iou = 0.5
        self.stride = stride
        self.imgsz = imgsz
        self.link_dist_frac = 0.2
        self.min_members = 3
        self.min_persist_frames = 5
        self.motion_threshold = 2.0


def audit_profiles_workload():
    """Verify that processing profile genuinely changes the backend workload."""
    print("\n--- AUDIT 1: Profile Workload Verification ---")
    video_source = ROOT / "test_video.mp4"
    if not video_source.exists():
        video_source = ROOT.parent / "ai-military-surveillance-system" / "test_video.mp4"

    results = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a small 30-frame clip for quick, exact frame counting
        clip_path = Path(tmp_dir) / "test_30.mp4"
        cap = cv2.VideoCapture(str(video_source))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()

        for prof, expected_stride in [("high_accuracy", 1), ("balanced", 2), ("fast", 3)]:
            json_out = Path(tmp_dir) / f"{prof}.json"
            vid_out = Path(tmp_dir) / f"{prof}.mp4"
            job = {"id": f"test_{prof}", "status": "queued", "cancel": False}
            req = MockReq(profile=prof, stride=expected_stride, imgsz=480)

            run(job, req, clip_path, json_out, vid_out)
            data = json.loads(json_out.read_text())

            analyzed = data["perf"]["frames_analyzed"]
            total_read = data["perf"]["frames_read"]
            actual_stride = data["perf"]["stride"]

            # Expected analyzed count: ceil(30 / stride)
            expected_analyzed = (30 + expected_stride - 1) // expected_stride
            results[prof] = {
                "stride": actual_stride,
                "frames_read": total_read,
                "frames_analyzed": analyzed,
                "expected_analyzed": expected_analyzed,
                "workload_ratio": round(analyzed / total_read, 3),
                "verified": (analyzed == expected_analyzed) and (actual_stride == expected_stride)
            }
            print(f"Profile: {prof:14} | Stride: {actual_stride} | Analyzed: {analyzed}/{total_read} frames ({results[prof]['workload_ratio']*100:.1f}%) | Match: {results[prof]['verified']}")

    return results


def audit_gpu_acceleration():
    """Verify GPU acceleration logic and error handling when CUDA requested."""
    print("\n--- AUDIT 2: GPU Acceleration & Device Resolution ---")
    cuda_avail = torch.cuda.is_available()
    print(f"Hardware CUDA Available: {cuda_avail}")
    if cuda_avail:
        dev_name = torch.cuda.get_device_name(0)
        print(f"Device Name: {dev_name}")

    # Test pref="cpu"
    dev_cpu = resolve_device("cpu")
    print(f"resolve_device('cpu') -> '{dev_cpu}' (Expected: 'cpu')")
    assert dev_cpu == "cpu"

    # Test pref="auto"
    dev_auto = resolve_device("auto")
    expected_auto = "cuda:0" if cuda_avail else "cpu"
    print(f"resolve_device('auto') -> '{dev_auto}' (Expected: '{expected_auto}')")
    assert dev_auto == expected_auto

    # Test pref="cuda"
    cuda_error_caught = False
    if not cuda_avail:
        try:
            resolve_device("cuda")
        except RuntimeError as e:
            cuda_error_caught = True
            print(f"resolve_device('cuda') properly raised RuntimeError when CUDA unavailable: {e}")
    else:
        dev_cuda = resolve_device("cuda")
        print(f"resolve_device('cuda') -> '{dev_cuda}' (Expected: 'cuda:0')")
        assert dev_cuda == "cuda:0"

    return {
        "cuda_available": cuda_avail,
        "dev_cpu": dev_cpu,
        "dev_auto": dev_auto,
        "cuda_error_handled": cuda_error_caught if not cuda_avail else True
    }


def audit_model_caching():
    """Confirm application does not reload model unnecessarily."""
    print("\n--- AUDIT 3: Model Caching Verification ---")
    t0 = time.perf_counter()
    m1 = ModelRegistry.get_or_load("best1.pt", "cpu")
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    m2 = ModelRegistry.get_or_load("best1.pt", "cpu")
    t_second = time.perf_counter() - t0

    is_identical = (m1 is m2)
    print(f"First load time:  {t_first:.4f} s")
    print(f"Second load time: {t_second:.6f} s (Speedup: {t_first / max(t_second, 1e-6):.1f}x)")
    print(f"Identical Instance (m1 is m2): {is_identical}")

    assert is_identical, "ModelRegistry failed to cache model instance!"
    return {
        "first_load_s": t_first,
        "second_load_s": t_second,
        "identical_instance": is_identical
    }


def audit_video_fidelity():
    """Verify output-video duration, timestamps, frame dimensions, and playback speed."""
    print("\n--- AUDIT 4: Video Fidelity & Temporal Integrity ---")
    video_source = ROOT / "test_video.mp4"
    if not video_source.exists():
        video_source = ROOT.parent / "ai-military-surveillance-system" / "test_video.mp4"

    cap_in = cv2.VideoCapture(str(video_source))
    in_w = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap_in.get(cv2.CAP_PROP_FPS) or 25.0
    cap_in.release()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a 50-frame clip with known duration: 50 frames @ 25 FPS = 2.000s
        clip_path = Path(tmp_dir) / "test_50.mp4"
        cap = cv2.VideoCapture(str(video_source))
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), in_fps, (in_w, in_h))
        for _ in range(50):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()

        # Check input clip duration
        cap_clip = cv2.VideoCapture(str(clip_path))
        clip_frames = int(cap_clip.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_fps = cap_clip.get(cv2.CAP_PROP_FPS)
        clip_duration = clip_frames / clip_fps
        cap_clip.release()

        # Test across balanced profile (stride=2) where half frames are interpolated
        json_out = Path(tmp_dir) / "fidelity.json"
        vid_out = Path(tmp_dir) / "fidelity.mp4"
        job = {"id": "fidelity_test", "status": "queued", "cancel": False}
        req = MockReq(profile="balanced", stride=2, imgsz=480)

        run(job, req, clip_path, json_out, vid_out)

        # Inspect generated output video
        cap_out = cv2.VideoCapture(str(vid_out))
        out_w = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = cap_out.get(cv2.CAP_PROP_FPS)
        out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
        out_duration = out_frames / out_fps if out_fps > 0 else 0.0
        cap_out.release()

        # Inspect JSON timestamps
        data = json.loads(json_out.read_text())
        timestamps = [f["t"] for f in data["frames"]]
        is_monotonic = all(x < y for x, y in zip(timestamps, timestamps[1:]))

        print(f"Input Video:  {in_w}x{in_h} @ {clip_fps:.2f} FPS | Frames: {clip_frames} | Duration: {clip_duration:.3f} s")
        print(f"Output Video: {out_w}x{out_h} @ {out_fps:.2f} FPS | Frames: {out_frames} | Duration: {out_duration:.3f} s")
        print(f"Dimensions Match: {out_w == in_w and out_h == in_h}")
        print(f"Frame Count Match: {out_frames == clip_frames}")
        print(f"FPS Match: {abs(out_fps - clip_fps) < 0.1}")
        print(f"Duration Match: {abs(out_duration - clip_duration) < 0.05}")
        print(f"Timestamps Monotonic: {is_monotonic}")

        assert out_w == in_w and out_h == in_h, "Output dimensions mismatch!"
        assert out_frames == clip_frames, f"Frame count mismatch: {out_frames} vs {clip_frames}"
        assert abs(out_duration - clip_duration) < 0.05, "Duration mismatch!"
        assert is_monotonic, "Timestamps are not strictly monotonic!"

        return {
            "in_w": in_w, "out_w": out_w,
            "in_h": in_h, "out_h": out_h,
            "in_frames": clip_frames, "out_frames": out_frames,
            "in_duration": clip_duration, "out_duration": out_duration,
            "timestamps_monotonic": is_monotonic
        }


def audit_analysis_modes():
    """Test all existing analysis modes."""
    print("\n--- AUDIT 5: Analysis Modes Verification ---")
    video_source = ROOT / "test_video.mp4"
    if not video_source.exists():
        video_source = ROOT.parent / "ai-military-surveillance-system" / "test_video.mp4"

    modes = [
        "military+optical_flow",
        "detection+segmentation",
        "detection"
    ]
    results = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 10-frame clip for fast mode verification
        clip_path = Path(tmp_dir) / "test_10.mp4"
        cap = cv2.VideoCapture(str(video_source))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()

        for mode in modes:
            json_out = Path(tmp_dir) / f"mode_{mode.replace('+', '_')}.json"
            vid_out = Path(tmp_dir) / f"mode_{mode.replace('+', '_')}.mp4"
            job = {"id": f"test_{mode}", "status": "queued", "cancel": False}
            req = MockReq(mode=mode, profile="high_accuracy", stride=1, imgsz=480)

            run(job, req, clip_path, json_out, vid_out)
            data = json.loads(json_out.read_text())

            has_optflow = "optical_flow" in mode or "military" in mode
            first_frame_dets = data["frames"][0]["detections"] if data["frames"] else []
            if first_frame_dets:
                motion_status_sample = first_frame_dets[0].get("motion_status")
            else:
                motion_status_sample = "No detections"

            optflow_time = data["perf"]["timing_breakdown"]["optical_flow_s"]
            results[mode] = {
                "status": job["status"],
                "has_optflow": has_optflow,
                "optflow_time_s": optflow_time,
                "motion_status_sample": motion_status_sample,
                "output_video_exists": vid_out.exists(),
                "output_video_size": vid_out.stat().st_size
            }
            print(f"Mode: {mode:24} | Status: {job['status']} | OptFlow Time: {optflow_time:.4f}s | Sample Motion: {motion_status_sample} | OutVid: {vid_out.stat().st_size} bytes")

    return results


def audit_memory_and_resources():
    """Check for memory leaks across consecutive jobs."""
    print("\n--- AUDIT 6: Memory & Resource Leak Verification ---")
    video_source = ROOT / "test_video.mp4"
    if not video_source.exists():
        video_source = ROOT.parent / "ai-military-surveillance-system" / "test_video.mp4"

    process = psutil.Process(os.getpid())
    mem_initial = process.memory_info().rss / (1024 * 1024)

    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_path = Path(tmp_dir) / "test_20.mp4"
        cap = cv2.VideoCapture(str(video_source))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(20):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()

        mem_readings = []
        # Run 5 consecutive jobs back to back
        for i in range(5):
            json_out = Path(tmp_dir) / f"job_{i}.json"
            vid_out = Path(tmp_dir) / f"job_{i}.mp4"
            job = {"id": f"mem_job_{i}", "status": "queued", "cancel": False}
            req = MockReq(profile="fast", stride=3, imgsz=480)
            run(job, req, clip_path, json_out, vid_out)
            rss = process.memory_info().rss / (1024 * 1024)
            mem_readings.append(rss)

        mem_final = process.memory_info().rss / (1024 * 1024)
        mem_growth = mem_final - mem_readings[0]

        print(f"Initial Memory RSS:        {mem_initial:.1f} MB")
        print(f"After Job 1 Memory RSS:    {mem_readings[0]:.1f} MB")
        print(f"After Job 5 Memory RSS:    {mem_final:.1f} MB")
        print(f"Memory Growth (Job 1->5):  {mem_growth:+.2f} MB")

        # Memory growth between job 1 and job 5 should be negligible (< 30MB)
        assert mem_growth < 30.0, f"Excessive memory growth detected: {mem_growth:.1f} MB"
        return {
            "initial_mb": mem_initial,
            "job1_mb": mem_readings[0],
            "job5_mb": mem_final,
            "growth_mb": mem_growth
        }


if __name__ == "__main__":
    print("=" * 65)
    print("AI MILITARY SURVEILLANCE SYSTEM - COMPREHENSIVE AUDIT SUITE")
    print("=" * 65)

    r_prof = audit_profiles_workload()
    r_gpu = audit_gpu_acceleration()
    r_cache = audit_model_caching()
    r_fid = audit_video_fidelity()
    r_modes = audit_analysis_modes()
    r_mem = audit_memory_and_resources()

    print("\n" + "=" * 65)
    print("ALL 6 AUDIT MODULES COMPLETED SUCCESSFULLY")
    print("=" * 65)
