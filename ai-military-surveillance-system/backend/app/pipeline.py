import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .analysis.convoy import ConvoyAnalyzer
from .analysis.motion import estimate_roi_motion
from .config import settings
from .detection.yolo import YoloDetector


def _create_video_writer(output_path, width, height, fps):
    """Attempt to create an H.264 video writer via imageio_ffmpeg, fallback to cv2.VideoWriter."""
    try:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(
            str(output_path),
            size=(width, height),
            fps=fps,
            codec="libx264",
            pix_fmt_in="bgr24",
            macro_block_size=1,
            output_params=["-crf", "22", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        )
        writer.send(None)
        return {"type": "ffmpeg", "writer": writer}
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        return {"type": "cv2", "writer": cv_writer}


def _write_frame(writer_dict, frame):
    if writer_dict["type"] == "ffmpeg":
        writer_dict["writer"].send(frame.tobytes())
    else:
        writer_dict["writer"].write(frame)


def _close_writer(writer_dict):
    try:
        if writer_dict["type"] == "ffmpeg":
            writer_dict["writer"].close()
        else:
            writer_dict["writer"].release()
    except Exception:
        pass


def run(job, req, src, json_out_path, video_out_path):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    # Resolve stride from profile or direct parameter
    profile = getattr(req, "profile", "high_accuracy").lower()
    if profile == "fast":
        stride = max(3, getattr(req, "stride", 3))
    elif profile == "balanced":
        stride = max(2, getattr(req, "stride", 2))
    else:  # high_accuracy
        stride = getattr(req, "stride", 1)

    imgsz = getattr(req, "imgsz", 640)
    enable_optflow = "optical_flow" in req.mode or "military" in req.mode
    enable_seg = req.mode != "detection"

    job["status"] = "loading_model"
    job["stage"] = "Initializing Model"

    det = YoloDetector(
        model=req.model,
        device=settings.device,
        conf=req.conf,
        iou=req.iou,
        segment=enable_seg,
        imgsz=imgsz
    )
    det.load()

    ana = ConvoyAnalyzer(
        link_dist=req.link_dist_frac * w,
        min_members=req.min_members,
        min_persist=req.min_persist_frames,
        fps=fps
    )

    writer_dict = _create_video_writer(video_out_path, w, h, fps)

    job["status"] = "processing"
    job["stage"] = "Analyzing Video"
    frames = []
    fid = 0
    t0_job = time.perf_counter()
    prev_gray = None

    track_hist = {}
    last_dets = []
    last_groups = []

    # Telemetry accumulators
    t_decode_accum = 0.0
    t_yolo_accum = 0.0
    t_optflow_accum = 0.0
    t_draw_accum = 0.0
    t_encode_accum = 0.0
    analyzed_frames_count = 0

    try:
        while not job.get("cancel", False):
            t_start_frame = time.perf_counter()
            ok, frame = cap.read()
            t_decode_accum += time.perf_counter() - t_start_frame
            if not ok:
                break

            is_inference_frame = (fid % stride == 0)
            moving_count = 0
            static_count = 0

            if is_inference_frame:
                analyzed_frames_count += 1
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 1. YOLO Inference
                t_infer_start = time.perf_counter()
                dets = det.predict(frame)
                t_yolo_accum += time.perf_counter() - t_infer_start

                # 2. Optimized Optical Flow (Conditional on Mode)
                t_flow_start = time.perf_counter()
                if enable_optflow and prev_gray is not None:
                    m_thresh = getattr(req, "motion_threshold", 2.0)
                    for d in dets:
                        status, avg_m = estimate_roi_motion(
                            prev_gray, curr_gray, d.bbox,
                            max_dim=80, motion_threshold=m_thresh
                        )
                        d.motion_status = status
                        d.motion_magnitude = avg_m
                        if status == "Moving":
                            moving_count += 1
                        else:
                            static_count += 1
                else:
                    for d in dets:
                        d.motion_status = "N/A" if not enable_optflow else "Static"
                        static_count += 1
                t_optflow_accum += time.perf_counter() - t_flow_start

                # 3. Update trajectories (bounded to 30 past points)
                for d in dets:
                    track_hist.setdefault(d.track_id, []).append((int(d.centroid[0]), int(d.centroid[1])))
                    if len(track_hist[d.track_id]) > 30:
                        track_hist[d.track_id].pop(0)

                # 4. Convoy grouping
                groups = ana.update(fid, dets)
                last_dets = dets
                last_groups = groups

                frames.append({
                    "frame_id": fid,
                    "t": round(fid / fps, 3),
                    "detections": [asdict(d) for d in dets],
                    "groups": groups,
                    "moving_count": moving_count,
                    "static_count": static_count,
                    "interpolated": False
                })
                prev_gray = curr_gray
            else:
                # Interpolated frame: reuse last detections for smooth display
                dets = last_dets
                groups = last_groups

            # 5. Annotation & Drawing
            t_draw_start = time.perf_counter()
            annotated = frame.copy()

            # Draw segmentation masks if present
            if enable_seg:
                for d in dets:
                    if d.mask and len(d.mask) > 2:
                        try:
                            poly_pts = np.array(d.mask, dtype=np.int32)
                            overlay = annotated.copy()
                            color = (0, 0, 240) if d.motion_status == "Moving" else (34, 197, 94)
                            cv2.fillPoly(overlay, [poly_pts], color)
                            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
                        except Exception:
                            pass

            # Draw trajectory trails
            for tid, pts in track_hist.items():
                if len(pts) > 1:
                    for k in range(1, len(pts)):
                        cv2.line(annotated, pts[k - 1], pts[k], (245, 158, 11), 2)

            # Draw bounding boxes & tactical labels
            for d in dets:
                x1, y1, x2, y2 = map(int, d.bbox)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                is_moving = (d.motion_status == "Moving")
                box_color = (0, 0, 245) if is_moving else (34, 197, 94)
                if not enable_optflow:
                    box_color = (245, 158, 11)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                cx, cy = int(d.centroid[0]), int(d.centroid[1])
                cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1)

                label_suffix = f" | {d.motion_status}" if enable_optflow else ""
                label = f"{d.class_name} {d.conf:.2f}{label_suffix} #{d.track_id}"
                font_scale = 0.50
                thickness = 1
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                label_y1 = max(0, y1 - lh - 8)
                cv2.rectangle(annotated, (x1, label_y1), (x1 + lw + 8, y1), box_color, -1)
                cv2.putText(
                    annotated, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
                )

            # Top Tactical Surveillance HUD
            sampling_tag = "" if is_inference_frame else " [INTERPOLATED]"
            hud_text = f"Frame: {fid}/{total} | Objects: {len(dets)} | Profile: {profile.upper()} ({stride}x){sampling_tag}"
            hud_w = min(w - 20, 800)
            cv2.rectangle(annotated, (10, 10), (10 + hud_w, 42), (10, 15, 25), -1)
            cv2.rectangle(annotated, (10, 10), (10 + hud_w, 42), (40, 60, 90), 1)
            cv2.putText(annotated, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)
            t_draw_accum += time.perf_counter() - t_draw_start

            # 6. Video Encoding
            t_encode_start = time.perf_counter()
            _write_frame(writer_dict, annotated)
            t_encode_accum += time.perf_counter() - t_encode_start

            fid += 1

            # Update live telemetry every 5 frames
            if fid % 5 == 0 or fid == total:
                elapsed = time.perf_counter() - t0_job
                current_fps = fid / elapsed if elapsed > 0 else 0.0
                remaining_frames = total - fid
                eta_s = int(remaining_frames / current_fps) if current_fps > 0 else 0
                job["progress"] = min(fid / total, 1.0)
                job["fps"] = round(current_fps, 1)
                job["frame"] = fid
                job["total_frames"] = total
                job["elapsed_s"] = round(elapsed, 1)
                job["eta_s"] = eta_s

    finally:
        cap.release()
        _close_writer(writer_dict)

    total_time = time.perf_counter() - t0_job
    avg_fps = fid / total_time if total_time > 0 else 0.0

    perf_report = {
        "total_time_s": round(total_time, 3),
        "avg_fps": round(avg_fps, 2),
        "frames_read": fid,
        "frames_analyzed": analyzed_frames_count,
        "stride": stride,
        "profile": profile,
        "imgsz": imgsz,
        "device": det.device,
        "timing_breakdown": {
            "yolo_inference_s": round(t_yolo_accum, 3),
            "yolo_ms_per_analyzed_frame": round((t_yolo_accum / analyzed_frames_count * 1000), 1) if analyzed_frames_count else 0,
            "optical_flow_s": round(t_optflow_accum, 3),
            "optflow_ms_per_analyzed_frame": round((t_optflow_accum / analyzed_frames_count * 1000), 1) if analyzed_frames_count else 0,
            "drawing_s": round(t_draw_accum, 3),
            "encoding_s": round(t_encode_accum, 3),
            "decoding_s": round(t_decode_accum, 3)
        }
    }

    job["perf_summary"] = perf_report

    meta = {
        "width": w,
        "height": h,
        "fps": fps,
        "total_frames": total,
        "processed_frames": fid,
        "stride": stride,
        "profile": profile,
        "model": req.model,
        "mode": req.mode,
        "device": det.device,
        "conf": req.conf,
        "iou": req.iou,
        "perf_report": perf_report,
        "has_output_video": True
    }
    json_out_path.write_text(json.dumps({"meta": meta, "frames": frames, "groups": ana.summary(), "perf": perf_report}))

    job["status"] = "cancelled" if job.get("cancel") else "completed"
    job["stage"] = "Completed" if job["status"] == "completed" else "Cancelled"
    job["progress"] = 1.0

