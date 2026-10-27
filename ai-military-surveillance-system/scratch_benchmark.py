import time
import os
import psutil
import cv2
import numpy as np
from pathlib import Path
from dataclasses import asdict

# Import existing pipeline components
from backend.app.config import settings
from backend.app.detection.yolo import YoloDetector
from backend.app.analysis.convoy import ConvoyAnalyzer
from backend.app.pipeline import _create_video_writer, _write_frame, _close_writer

def run_benchmark(video_path="test_video.mp4", model_name="best1.pt", max_frames=100, stride=1, mode="military+optical_flow"):
    print("=" * 65)
    print("AI MILITARY SURVEILLANCE SYSTEM - PERFORMANCE PROFILER")
    print("=" * 65)

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # 1. Measure Model Initialization
    t_model_start = time.perf_counter()
    det = YoloDetector(
        model=model_name,
        device=settings.device,
        conf=0.35,
        iou=0.5,
        segment=(mode != "detection")
    )
    det.load()
    t_model_init = time.perf_counter() - t_model_start
    print(f"[STAGE 1] Model Initialization: {t_model_init:.3f} s (Model: {model_name}, Device: {det.device})")

    # 2. Measure Video Opening
    t_vid_start = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_vid_open = time.perf_counter() - t_vid_start
    print(f"[STAGE 2] Video Opening: {t_vid_open:.4f} s (Resolution: {w}x{h}, FPS: {fps}, Total: {total_frames} frames)")

    # Output writer setup
    out_dir = Path("storage/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / "baseline_out.mp4"
    writer_dict = _create_video_writer(out_video_path, w, h, fps)

    ana = ConvoyAnalyzer(link_dist=0.2 * w, min_members=3, min_persist=15, fps=fps)

    # Profiling accumulators
    t_decode_total = 0.0
    t_yolo_total = 0.0
    t_optflow_total = 0.0
    t_tracking_total = 0.0
    t_drawing_total = 0.0
    t_encoding_total = 0.0

    frames_read = 0
    frames_processed = 0
    prev_gray = None
    track_hist = {}

    benchmark_limit = min(max_frames, total_frames) if max_frames else total_frames
    print(f"[STAGE 3] Running frame processing loop for {benchmark_limit} frames (Stride={stride})...")

    t_loop_start = time.perf_counter()

    while frames_read < benchmark_limit:
        # Frame decoding
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t_decode_total += time.perf_counter() - t0
        if not ok:
            break

        frames_read += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = []
        groups = []

        if (frames_read - 1) % stride == 0:
            frames_processed += 1

            # YOLO inference + internal tracker
            t0 = time.perf_counter()
            dets = det.predict(frame)
            t_yolo_total += time.perf_counter() - t0

            # Optical Flow
            t0 = time.perf_counter()
            moving_count = 0
            static_count = 0
            for d in dets:
                x1, y1, x2, y2 = map(int, d.bbox)
                x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
                avg_motion = 0.0
                status = "Static"
                if prev_gray is not None:
                    roi_prev = prev_gray[y1:y2, x1:x2]
                    roi_curr = curr_gray[y1:y2, x1:x2]
                    if roi_prev.size > 0 and roi_curr.size > 0 and roi_prev.shape == roi_curr.shape:
                        try:
                            flow = cv2.calcOpticalFlowFarneback(
                                roi_prev, roi_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
                            )
                            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            avg_motion = float(np.mean(mag))
                            status = "Moving" if avg_motion > 2.0 else "Static"
                        except Exception:
                            status = "Static"
                if status == "Moving":
                    moving_count += 1
                else:
                    static_count += 1
                d.motion_status = status
                d.motion_magnitude = round(avg_motion, 2)
                track_hist.setdefault(d.track_id, []).append((int(d.centroid[0]), int(d.centroid[1])))
                if len(track_hist[d.track_id]) > 30:
                    track_hist[d.track_id].pop(0)
            t_optflow_total += time.perf_counter() - t0

            # Convoy graph tracking
            t0 = time.perf_counter()
            groups = ana.update(frames_read - 1, dets)
            t_tracking_total += time.perf_counter() - t0

        # Annotation & Drawing
        t0 = time.perf_counter()
        annotated = frame.copy()
        for tid, pts in track_hist.items():
            if len(pts) > 1:
                for k in range(1, len(pts)):
                    cv2.line(annotated, pts[k - 1], pts[k], (245, 158, 11), 2)

        for d in dets:
            x1, y1, x2, y2 = map(int, d.bbox)
            is_moving = getattr(d, "motion_status", "Static") == "Moving"
            box_color = (0, 0, 245) if is_moving else (34, 197, 94)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            label = f"{d.class_name} {d.conf:.2f} | {d.motion_status} #{d.track_id}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - lh - 8)), (x1 + lw + 8, y1), box_color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        hud_text = f"Frame: {frames_read}/{benchmark_limit} | Detected: {len(dets)}"
        cv2.putText(annotated, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)
        t_drawing_total += time.perf_counter() - t0

        # Encoding
        t0 = time.perf_counter()
        _write_frame(writer_dict, annotated)
        t_encoding_total += time.perf_counter() - t0

        prev_gray = curr_gray

    cap.release()
    _close_writer(writer_dict)

    t_loop_total = time.perf_counter() - t_loop_start
    mem_after = process.memory_info().rss / (1024 * 1024)

    print("\n" + "=" * 65)
    print("DETAILED PERFORMANCE BREAKDOWN")
    print("=" * 65)
    print(f"Total loop time:           {t_loop_total:.3f} s")
    print(f"Average Processing FPS:    {frames_read / t_loop_total:.2f} FPS")
    print(f"Frames read:               {frames_read}")
    print(f"Frames analyzed:           {frames_processed}")
    print(f"Memory RSS before/after:   {mem_before:.1f} MB -> {mem_after:.1f} MB (Delta: {mem_after - mem_before:.1f} MB)")
    print("-" * 65)
    print(f"1. Frame Decoding:         {t_decode_total:.3f} s ({t_decode_total/t_loop_total*100:.1f}%) | {t_decode_total/frames_read*1000:.2f} ms/frame")
    print(f"2. YOLO Inference:         {t_yolo_total:.3f} s ({t_yolo_total/t_loop_total*100:.1f}%) | {t_yolo_total/frames_processed*1000:.2f} ms/frame")
    print(f"3. Optical Flow:           {t_optflow_total:.3f} s ({t_optflow_total/t_loop_total*100:.1f}%) | {t_optflow_total/frames_processed*1000:.2f} ms/frame")
    print(f"4. Convoy/Tracking Update: {t_tracking_total:.3f} s ({t_tracking_total/t_loop_total*100:.1f}%) | {t_tracking_total/frames_processed*1000:.2f} ms/frame")
    print(f"5. Annotation / Drawing:   {t_drawing_total:.3f} s ({t_drawing_total/t_loop_total*100:.1f}%) | {t_drawing_total/frames_read*1000:.2f} ms/frame")
    print(f"6. Video Encoding:         {t_encoding_total:.3f} s ({t_encoding_total/t_loop_total*100:.1f}%) | {t_encoding_total/frames_read*1000:.2f} ms/frame")
    print("=" * 65)

    return {
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "total_time": t_loop_total,
        "fps": frames_read / t_loop_total,
        "t_decode": t_decode_total,
        "t_yolo": t_yolo_total,
        "t_optflow": t_optflow_total,
        "t_tracking": t_tracking_total,
        "t_drawing": t_drawing_total,
        "t_encoding": t_encoding_total,
        "mem_delta_mb": mem_after - mem_before
    }

if __name__ == "__main__":
    run_benchmark(max_frames=100, stride=1)
