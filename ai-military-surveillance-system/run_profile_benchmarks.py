import time
import json
import tempfile
from pathlib import Path
import cv2

from backend.app.config import ROOT
from backend.app.pipeline import run

class MockReq:
    def __init__(self, profile="balanced", stride=2, imgsz=480):
        self.model = "best1.pt"
        self.mode = "military+optical_flow"
        self.profile = profile
        self.conf = 0.35
        self.iou = 0.5
        self.stride = stride
        self.imgsz = imgsz
        self.link_dist_frac = 0.2
        self.min_members = 3
        self.min_persist_frames = 5
        self.motion_threshold = 2.0

def benchmark_profile(profile_name, stride, imgsz, max_frames=100):
    video_source = ROOT / "test_video.mp4"
    if not video_source.exists():
        video_source = ROOT.parent / "ai-military-surveillance-system" / "test_video.mp4"

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create 100-frame test clip
        clip_path = Path(tmp_dir) / "test_100.mp4"
        cap = cv2.VideoCapture(str(video_source))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()

        json_out = Path(tmp_dir) / f"{profile_name}.json"
        video_out = Path(tmp_dir) / f"{profile_name}.mp4"

        job = {"id": f"bench_{profile_name}", "status": "queued", "cancel": False}
        req = MockReq(profile=profile_name, stride=stride, imgsz=imgsz)

        t0 = time.perf_counter()
        run(job, req, clip_path, json_out, video_out)
        t_total = time.perf_counter() - t0

        data = json.loads(json_out.read_text())
        perf = data["perf"]
        tb = perf["timing_breakdown"]

        return {
            "profile": profile_name,
            "stride": stride,
            "imgsz": imgsz,
            "total_time_s": round(t_total, 3),
            "fps": round(max_frames / t_total, 2),
            "analyzed_frames": perf["frames_analyzed"],
            "yolo_time_s": tb["yolo_inference_s"],
            "yolo_ms_per_frame": tb["yolo_ms_per_analyzed_frame"],
            "optflow_time_s": tb["optical_flow_s"],
            "optflow_ms_per_frame": tb["optflow_ms_per_analyzed_frame"],
            "encoding_time_s": tb["encoding_s"],
            "decoding_time_s": tb["decoding_s"],
            "drawing_time_s": tb["drawing_s"]
        }

if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING BENCHMARKS ACROSS OPTIMIZED PROFILES (100 FRAMES)")
    print("=" * 65)

    results = []
    # 1. High Accuracy
    print("Benchmarking High Accuracy (stride=1, imgsz=640)...")
    res_high = benchmark_profile("high_accuracy", stride=1, imgsz=640, max_frames=100)
    results.append(res_high)
    print(f"High Accuracy: {res_high['total_time_s']}s | {res_high['fps']} FPS | YOLO: {res_high['yolo_time_s']}s | OptFlow: {res_high['optflow_time_s']}s")

    # 2. Balanced
    print("Benchmarking Balanced (stride=2, imgsz=480)...")
    res_bal = benchmark_profile("balanced", stride=2, imgsz=480, max_frames=100)
    results.append(res_bal)
    print(f"Balanced: {res_bal['total_time_s']}s | {res_bal['fps']} FPS | YOLO: {res_bal['yolo_time_s']}s | OptFlow: {res_bal['optflow_time_s']}s")

    # 3. Fast
    print("Benchmarking Fast (stride=3, imgsz=480)...")
    res_fast = benchmark_profile("fast", stride=3, imgsz=480, max_frames=100)
    results.append(res_fast)
    print(f"Fast: {res_fast['total_time_s']}s | {res_fast['fps']} FPS | YOLO: {res_fast['yolo_time_s']}s | OptFlow: {res_fast['optflow_time_s']}s")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 65)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 65)
    baseline_time = 54.420
    baseline_fps = 1.84
    print(f"{'Profile':<15} | {'Stride':<6} | {'imgsz':<5} | {'Time (s)':<8} | {'FPS':<6} | {'Speedup':<8} | {'YOLO (s)':<8} | {'Flow (s)':<8}")
    print("-" * 75)
    print(f"{'Baseline':<15} | {'1':<6} | {'640':<5} | {baseline_time:<8.2f} | {baseline_fps:<6.2f} | {'1.00x':<8} | {'45.13':<8} | {'7.96':<8}")
    for r in results:
        speedup = f"{baseline_time / r['total_time_s']:.2f}x"
        print(f"{r['profile']:<15} | {r['stride']:<6} | {r['imgsz']:<5} | {r['total_time_s']:<8.2f} | {r['fps']:<6.2f} | {speedup:<8} | {r['yolo_time_s']:<8.2f} | {r['optflow_time_s']:<8.2f}")
    print("=" * 65)
