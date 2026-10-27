import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .analysis.convoy import ConvoyAnalyzer
from .config import settings
from .detection.yolo import YoloDetector


def _create_video_writer(output_path, width, height, fps):
    """Attempt to create an H.264 video writer via imageio_ffmpeg, fallback to cv2.VideoWriter."""
    try:
        import imageio_ffmpeg
        # Macro_block_size=None or 1 preserves exact dimensions without cropping/padding
        writer = imageio_ffmpeg.write_frames(
            str(output_path),
            size=(width, height),
            fps=fps,
            codec="libx264",
            pix_fmt_in="bgr24",
            macro_block_size=1,
            output_params=["-crf", "22", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        )
        writer.send(None)  # Prime the generator
        return {"type": "ffmpeg", "writer": writer}
    except Exception as e:
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

    job["status"] = "loading_model"
    det = YoloDetector(
        model=req.model,
        device=settings.device,
        conf=req.conf,
        iou=req.iou,
        segment=(req.mode != "detection")
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
    frames = []
    fid = 0
    t0 = time.time()
    prev_gray = None

    # Track histories for trajectory trail drawing on video
    track_hist = {}

    try:
        while not job.get("cancel", False):
            ok, frame = cap.read()
            if not ok:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = []
            groups = []

            # Process detection on requested stride frames
            if fid % req.stride == 0:
                dets = det.predict(frame)

                # Optical flow motion classification on object ROIs
                moving_count = 0
                static_count = 0

                for d in dets:
                    x1, y1, x2, y2 = map(int, d.bbox)
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))

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
                                if avg_motion > getattr(req, "motion_threshold", 2.0):
                                    status = "Moving"
                                else:
                                    status = "Static"
                            except Exception:
                                status = "Static"

                    if status == "Moving":
                        moving_count += 1
                    else:
                        static_count += 1

                    d.motion_status = status
                    d.motion_magnitude = round(avg_motion, 2)

                    # Update track history for trajectory trails
                    track_hist.setdefault(d.track_id, []).append((int(d.centroid[0]), int(d.centroid[1])))
                    if len(track_hist[d.track_id]) > 30:
                        track_hist[d.track_id].pop(0)

                groups = ana.update(fid, dets)
                frames.append({
                    "frame_id": fid,
                    "t": round(fid / fps, 3),
                    "detections": [asdict(d) for d in dets],
                    "groups": groups,
                    "moving_count": moving_count,
                    "static_count": static_count
                })
            else:
                moving_count = 0
                static_count = 0

            # --- RENDER ANNOTATED FRAME FOR OUTPUT VIDEO ---
            annotated = frame.copy()

            # 1. Draw segmentation masks if present
            for d in dets:
                if d.mask and len(d.mask) > 2:
                    try:
                        poly_pts = np.array(d.mask, dtype=np.int32)
                        overlay = annotated.copy()
                        # Color: Red if Moving, Green if Static
                        color = (0, 0, 240) if d.motion_status == "Moving" else (34, 197, 94)
                        cv2.fillPoly(overlay, [poly_pts], color)
                        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
                    except Exception:
                        pass

            # 2. Draw trajectory trails
            for tid, pts in track_hist.items():
                if len(pts) > 1:
                    for k in range(1, len(pts)):
                        cv2.line(annotated, pts[k - 1], pts[k], (245, 158, 11), 2)

            # 3. Draw bounding boxes, centroids, and HUD labels
            for d in dets:
                x1, y1, x2, y2 = map(int, d.bbox)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                # Color: Red for Moving, Green for Static
                is_moving = d.motion_status == "Moving"
                box_color = (0, 0, 245) if is_moving else (34, 197, 94)

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                # Centroid dot
                cx, cy = int(d.centroid[0]), int(d.centroid[1])
                cv2.circle(annotated, (cx, cy), 4, (255, 255, 255), -1)

                # Label text and background pill
                label = f"{d.class_name} {d.conf:.2f} | {d.motion_status} #{d.track_id}"
                font_scale = 0.52
                thickness = 1
                (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                label_y1 = max(0, y1 - lh - 8)
                label_y2 = y1
                cv2.rectangle(annotated, (x1, label_y1), (x1 + lw + 8, label_y2), box_color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

            # 4. Top Surveillance HUD Overlay Bar
            hud_text = f"Frame: {fid}/{total} | Objects: {len(dets)} (Moving: {moving_count}, Static: {static_count}) | Model: {Path(req.model).name}"
            hud_w = min(w - 20, 800)
            cv2.rectangle(annotated, (10, 10), (10 + hud_w, 42), (10, 15, 25), -1)
            cv2.rectangle(annotated, (10, 10), (10 + hud_w, 42), (40, 60, 90), 1)
            cv2.putText(annotated, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

            # Write to output video stream
            _write_frame(writer_dict, annotated)

            prev_gray = curr_gray
            fid += 1

            # Update job progress telemetry
            elapsed = time.time() - t0
            job["progress"] = min(fid / total, 1.0)
            job["fps"] = round(fid / elapsed, 1) if elapsed > 0 else 0.0
            job["frame"] = fid
            job["total_frames"] = total

    finally:
        cap.release()
        _close_writer(writer_dict)

    # Write telemetry and analysis metadata JSON
    meta = {
        "width": w,
        "height": h,
        "fps": fps,
        "total_frames": total,
        "stride": req.stride,
        "model": req.model,
        "mode": req.mode,
        "device": det.device,
        "conf": req.conf,
        "iou": req.iou,
        "link_dist_px": req.link_dist_frac * w,
        "min_members": req.min_members,
        "min_persist_frames": req.min_persist_frames,
        "has_output_video": True
    }
    json_out_path.write_text(json.dumps({"meta": meta, "frames": frames, "groups": ana.summary()}))
