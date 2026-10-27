"""
Real-Time Object Detection + Intelligent Distance Estimation
=============================================================
Uses THREE models:
  1. YOLOv8      — detects 80 COCO classes out of the box
  2. MiDaS DNN   — dense neural depth map (no hardcoded heights)
  3. Centroid Tracker — stable IDs across frames

Install:
    pip install ultralytics opencv-python

Usage:
    python run_detection.py                   # webcam
    python run_detection.py --source video.mp4
    python run_detection.py --save out.mp4
    python run_detection.py --model yolov8m.pt  # more accurate

Controls:
    Q = quit
    S = screenshot
    D = toggle depth map overlay
"""

import argparse
import colorsys
import time
import urllib.request
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import gc

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run:  pip install ultralytics opencv-python")

try:
    from scipy.spatial.distance import cdist
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ── MiDaS depth model ─────────────────────────────────────────────────────────
MIDAS_URL  = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
MIDAS_PATH = Path("midas_small.onnx")


def _download_midas():
    if MIDAS_PATH.exists():
        return
    print("[INFO] Downloading MiDaS (~50 MB) — one time only ...")
    urllib.request.urlretrieve(MIDAS_URL, MIDAS_PATH)
    print("[INFO] MiDaS ready.")


class MiDaSEstimator:
    def __init__(self):
        _download_midas()
        self.net = cv2.dnn.readNetFromONNX(str(MIDAS_PATH))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._scale_history: deque = deque(maxlen=40)
        self._scale: Optional[float] = None

    def infer(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (256, 256),
            (123.675, 116.28, 103.53), swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        out   = self.net.forward()[0, 0]
        depth = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = depth.min(), depth.max()
        if hi - lo > 1e-6:
            depth = (depth - lo) / (hi - lo)
        return depth.astype(np.float32)

    def roi_median(self, dmap: np.ndarray, x1, y1, x2, y2) -> float:
        H, W = dmap.shape
        mx1 = max(int(x1 + (x2 - x1) * 0.25), 0)
        mx2 = min(int(x2 - (x2 - x1) * 0.25), W - 1)
        my1 = max(int(y1 + (y2 - y1) * 0.25), 0)
        my2 = min(int(y2 - (y2 - y1) * 0.25), H - 1)
        roi = dmap[my1:my2, mx1:mx2]
        return float(np.median(roi)) if roi.size > 0 else 0.0

    def calibrate(self, depth_val: float, metric_m: float):
        if depth_val > 0.05:
            self._scale_history.append(metric_m * depth_val)
            self._scale = float(np.median(self._scale_history))

    def to_metric(self, depth_val: float) -> Optional[float]:
        if self._scale and depth_val > 0.01:
            return self._scale / depth_val
        return None

    @property
    def calibrated(self) -> bool:
        return self._scale is not None


# ── Calibration anchors ────────────────────────────────────────────────────────
ANCHORS = {
    "person": 1.70, "man": 1.75, "woman": 1.65,
    "car": 1.50, "bus": 3.20, "truck": 3.50,
}


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class Detection:
    bbox: np.ndarray
    class_id: int
    class_name: str
    confidence: float

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.width  = float(x2 - x1)
        self.height = float(y2 - y1)


@dataclass
class Track:
    track_id: int
    detection: Detection
    centroid: np.ndarray
    age: int = 1
    frames_since_seen: int = 0
    dist_history: deque = field(default_factory=lambda: deque(maxlen=12))

    def update(self, det: Detection):
        self.detection = det
        self.centroid  = np.array(det.center)
        self.age += 1
        self.frames_since_seen = 0

    def smooth(self, raw: float) -> float:
        self.dist_history.append(raw)
        w = np.exp(np.linspace(-1, 0, len(self.dist_history)))
        w /= w.sum()
        return float(np.dot(w, list(self.dist_history)))


# ── Centroid Tracker ───────────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_gone=30, max_dist=90.0):
        self.max_gone = max_gone
        self.max_dist = max_dist
        self._nid     = 0
        self.tracks: OrderedDict[int, Track] = OrderedDict()

    @property
    def num_active_tracks(self) -> int:
        return len(self.tracks)

    def update(self, dets: list) -> dict:
        if not dets:
            for tid in list(self.tracks):
                self.tracks[tid].frames_since_seen += 1
                if self.tracks[tid].frames_since_seen > self.max_gone:
                    del self.tracks[tid]
            return dict(self.tracks)

        nc = np.array([d.center for d in dets])
        if not self.tracks:
            for d in dets:
                self._reg(d)
            return dict(self.tracks)

        eids = list(self.tracks.keys())
        ec   = np.array([self.tracks[t].centroid for t in eids])
        D    = (cdist(ec, nc) if SCIPY_OK
                else np.linalg.norm(ec[:, None] - nc[None, :], axis=2))

        ro = D.min(axis=1).argsort()
        co = D.argmin(axis=1)
        ur, uc = set(), set()
        for r in ro:
            c = co[r]
            if r in ur or c in uc or D[r, c] > self.max_dist:
                continue
            self.tracks[eids[r]].update(dets[c])
            ur.add(r); uc.add(c)

        for i, tid in enumerate(eids):
            if i not in ur:
                self.tracks[tid].frames_since_seen += 1
                if self.tracks[tid].frames_since_seen > self.max_gone:
                    del self.tracks[tid]

        for j, d in enumerate(dets):
            if j not in uc:
                self._reg(d)

        return dict(self.tracks)

    def _reg(self, det):
        self.tracks[self._nid] = Track(
            track_id=self._nid,
            detection=det,
            centroid=np.array(det.center),
        )
        self._nid += 1


# ── Colours ────────────────────────────────────────────────────────────────────
_PALETTE: dict = {}

def obj_color(name: str) -> tuple:
    if name not in _PALETTE:
        h = (hash(name) * 0.618033) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        _PALETTE[name] = (int(b * 255), int(g * 255), int(r * 255))
    return _PALETTE[name]

def zone_color(d: float, near: float, mid: float) -> tuple:
    if d < near: return (30,  30, 220)
    if d < mid:  return (0,  190, 255)
    return              (30, 200,  60)


# ── Drawing ────────────────────────────────────────────────────────────────────
def draw_box(frame, det: Detection, color: tuple,
             dist_str: str, track_id: Optional[int]):
    x1, y1, x2, y2 = det.bbox.astype(int)
    cx = (x1 + x2) // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    tid   = f"#{track_id}  " if track_id is not None else ""
    label = f"{tid}{det.class_name}  {det.confidence:.0%}"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ly = max(y1, th + 4)
    cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + bl), color, -1)
    cv2.putText(frame, label, (x1 + 3, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    (dw, dh), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    p  = 5
    ov = frame.copy()
    cv2.rectangle(ov, (cx - dw // 2 - p, y2 + 2),
                      (cx + dw // 2 + p, y2 + dh + p + 4), color, -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, dist_str, (cx - dw // 2, y2 + dh + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, fps, det_ms, dep_ms, n_obj, n_trk, model_name, calibrated):
    cal   = "YES" if calibrated else "stand near camera..."
    lines = [
        f"Model:   {model_name}",
        f"FPS:     {fps:>6.1f}",
        f"Detect:  {det_ms:>5.1f} ms",
        f"Depth:   {dep_ms:>5.1f} ms",
        f"Objects: {n_obj:>3}",
        f"Tracks:  {n_trk:>3}",
        f"Calib:   {cal}",
    ]
    lh, pad = 21, 8
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (230, len(lines) * lh + pad * 2), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    for i, ln in enumerate(lines):
        col = (180, 180, 180) if i == 0 else (
            (0, 255, 100) if (i != 6 or calibrated) else (0, 165, 255)
        )
        cv2.putText(frame, ln, (pad, pad + (i + 1) * lh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)


def draw_legend(frame, near, mid, calibrated):
    items = [
        (f"< {near}m  CLOSE", (30,  30, 220)),
        (f"< {mid}m   MID",   (0,  190, 255)),
        (f"> {mid}m   FAR",   (30, 200,  60)),
    ]
    H, W = frame.shape[:2]
    x0, y0 = W - 192, H - 88
    ov = frame.copy()
    cv2.rectangle(ov, (x0 - 8, y0 - 20), (W - 4, H - 4), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    note = "metric m" if calibrated else "relative ~"
    cv2.putText(frame, f"Depth: {note}", (x0 - 2, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 100) if calibrated else (0, 165, 255), 1, cv2.LINE_AA)
    for i, (txt, clr) in enumerate(items):
        y = y0 + i * 26
        cv2.circle(frame, (x0 + 7, y - 5), 7, clr, -1)
        cv2.putText(frame, txt, (x0 + 22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


# ── Args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source",  default=0,
                   help="0=webcam, or path to video file")
    p.add_argument("--model",   default="yolov8s.pt",
                   help="yolov8n (fast) / yolov8s / yolov8m / yolov8l (accurate)")
    p.add_argument("--conf",    type=float, default=0.35,
                   help="Confidence threshold — lower detects more objects")
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--device",  default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--save",    default=None, help="Save output video, e.g. out.mp4")
    p.add_argument("--width",   type=int,   default=1280)
    p.add_argument("--height",  type=int,   default=720)
    p.add_argument("--near",    type=float, default=2.0)
    p.add_argument("--mid",     type=float, default=6.0)
    p.add_argument("--no-track", action="store_true")
    p.add_argument("--classes", nargs="+", type=int, default=None,
                   help="Filter class IDs e.g. --classes 0 2 5")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source!r}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fy = (W / 2) / np.tan(np.radians(39))   # ~78° FOV

    print(f"[INFO] Resolution : {W}x{H}")
    print(f"[INFO] Loading YOLO: {args.model}")
    model       = YOLO(args.model)
    class_names = model.names
    model_short = args.model.replace(".pt", "")

    print("[INFO] Loading MiDaS depth model ...")
    depth_model = MiDaSEstimator()

    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model.predict(source=dummy, verbose=False)
    print("[INFO] Ready!")
    print("[INFO] Controls:  Q=quit  S=screenshot  D=toggle depth map\n")

    tracker    = None if args.no_track else CentroidTracker()
    det_to_tid: dict = {}
    det_to_trk: dict = {}
    show_depth  = False
    fps_smooth  = 0.0
    t_prev      = time.perf_counter()
    frame_id    = 0

    writer = None
    if args.save:
        writer = cv2.VideoWriter(
            args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H)
        )
        print(f"[INFO] Saving to: {args.save}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            # ── YOLO inference ────────────────────────────────────────────────
            t0 = time.perf_counter()
            with torch.inference_mode():
                results = model.predict(
                    source=frame, conf=args.conf, iou=args.iou,
                    classes=args.classes, device=args.device,
                    imgsz=args.imgsz, verbose=False,
                )
            det_ms = (time.perf_counter() - t0) * 1000

            detections = []
            r = results[0]
            if r.boxes is not None and len(r.boxes):
                for bbox, conf, cid in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(int),
                ):
                    detections.append(Detection(
                        bbox=bbox, class_id=int(cid),
                        class_name=class_names.get(int(cid), f"obj_{cid}"),
                        confidence=float(conf),
                    ))
            del results
            if frame_id % 150 == 0:
                gc.collect()

            # ── MiDaS depth ───────────────────────────────────────────────────
            td = time.perf_counter()
            dmap   = depth_model.infer(frame)
            dep_ms = (time.perf_counter() - td) * 1000

            # ── Auto-calibrate from anchor objects ────────────────────────────
            for det in detections:
                anchor_h = ANCHORS.get(det.class_name.lower())
                if not anchor_h:
                    continue
                x1, y1, x2, y2 = det.bbox.astype(int)
                dv  = depth_model.roi_median(dmap, x1, y1, x2, y2)
                geo = (anchor_h * fy) / max(det.height, 1.0)
                depth_model.calibrate(dv, geo)

            # ── Tracking ──────────────────────────────────────────────────────
            det_to_tid.clear(); det_to_trk.clear()
            if tracker:
                active = tracker.update(detections)
                for tid, trk in active.items():
                    det_to_tid[id(trk.detection)] = tid
                    det_to_trk[id(trk.detection)] = trk

            # ── Visualise ─────────────────────────────────────────────────────
            vis = frame.copy()
            if show_depth:
                cmap = cv2.applyColorMap(
                    (dmap * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
                )
                cv2.addWeighted(cmap, 0.35, vis, 0.65, 0, vis)

            cv2.line(vis, (W // 2, 0),     (W // 2, H),     (140, 140, 140), 1)
            cv2.line(vis, (0,     H // 2), (W,      H // 2), (140, 140, 140), 1)

            for det in detections:
                x1, y1, x2, y2 = det.bbox.astype(int)
                dv     = depth_model.roi_median(dmap, x1, y1, x2, y2)
                metric = depth_model.to_metric(dv)

                if metric is not None:
                    trk    = det_to_trk.get(id(det))
                    dist_m = trk.smooth(metric) if trk else metric
                    dstr   = f"{dist_m:.2f} m"
                    color  = zone_color(dist_m, args.near, args.mid)
                else:
                    dstr  = f"~{dv:.2f}"
                    color = obj_color(det.class_name)

                draw_box(vis, det, color, dstr, det_to_tid.get(id(det)))

            # ── HUD & legend ──────────────────────────────────────────────────
            t_now      = time.perf_counter()
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(t_now - t_prev, 1e-9)
            t_prev     = t_now
            n_trk      = tracker.num_active_tracks if tracker else len(detections)

            draw_hud(vis, fps_smooth, det_ms, dep_ms,
                     len(detections), n_trk, model_short, depth_model.calibrated)
            draw_legend(vis, args.near, args.mid, depth_model.calibrated)

            cv2.imshow("YOLO + MiDaS Distance Estimation  (Q=quit  S=screenshot  D=depth)", vis)
            if writer:
                writer.write(vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fn = f"screenshot_{frame_id:06d}.jpg"
                cv2.imwrite(fn, vis)
                print(f"[INFO] Saved {fn}")
            elif key == ord("d"):
                show_depth = not show_depth
                print(f"[INFO] Depth overlay: {'ON' if show_depth else 'OFF'}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Done — {frame_id} frames processed.")


if __name__ == "__main__":
    main()
