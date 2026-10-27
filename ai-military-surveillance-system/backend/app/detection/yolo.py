from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from ..config import settings, ROOT
from ..interfaces import BaseDetector, Det


def resolve_device(pref: str) -> str:
    import torch
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if pref == "cuda":
        raise RuntimeError("CUDA requested but not available")
    return "cpu"


class YoloDetector(BaseDetector):
    """Detection + optional instance masks + persistent IDs (ByteTrack) via ultralytics."""

    def __init__(self, model, device="auto", conf=0.35, iou=0.5, segment=True, tracker="bytetrack.yaml", classes=None):
        self.model_raw = model
        self.conf = conf
        self.iou = iou
        self.segment = segment
        self.tracker = tracker
        self.device = resolve_device(device)
        self.classes = classes
        self.model = None
        self.names = {}
        self.is_segmentation = False

    def load(self):
        p = Path(self.model_raw)
        if not p.is_absolute() and not p.exists():
            search_dirs = [
                ROOT,
                ROOT / "models",
                ROOT.parent / "full",
                ROOT.parent / "ai-military-surveillance-system",
                ROOT.parent / "Yolo-Weights"
            ]
            for sd in search_dirs:
                cand = sd / self.model_raw
                if cand.exists():
                    p = cand
                    break

        self.model = YOLO(str(p))
        self.names = self.model.names or {}
        # Check if model has segmentation capability
        task = getattr(self.model, "task", None) or ""
        self.is_segmentation = "segment" in str(task).lower() or "seg" in str(p).lower()

    def predict(self, frame):
        do_seg = self.segment and self.is_segmentation
        # For custom military model (best1.pt), detect all classes unless specific list provided
        is_military_model = "best" in str(self.model_raw).lower()
        classes_filter = None if is_military_model else (self.classes if self.classes else None)

        try:
            r = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
                iou=self.iou,
                classes=classes_filter,
                device=self.device,
                verbose=False
            )[0]
        except Exception:
            # Fallback to standard predict if tracker throws
            r = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                classes=classes_filter,
                device=self.device,
                verbose=False
            )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        # If ByteTrack hasn't assigned IDs yet or tracking was unassigned, provide stable frame-level IDs
        if r.boxes.id is None:
            ids = [i + 1 for i in range(len(r.boxes))]
        else:
            ids = r.boxes.id.int().tolist()

        cls = r.boxes.cls.int().tolist()
        confs = r.boxes.conf.tolist()
        xyxy = r.boxes.xyxy.tolist()

        polys = r.masks.xy if (do_seg and r.masks is not None and len(r.masks.xy) == len(ids)) else [None] * len(ids)

        out = []
        for tid, c, cf, b, p in zip(ids, cls, confs, xyxy, polys):
            has = p is not None and len(p) > 2
            c_name = self.names.get(c, f"class_{c}")
            out.append(Det(
                track_id=tid,
                class_id=c,
                class_name=c_name,
                conf=round(float(cf), 3),
                bbox=[round(float(v), 1) for v in b],
                centroid=((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0),
                area=float((b[2] - b[0]) * (b[3] - b[1])),
                mask=np.round(p, 1).tolist() if has else None,
                mask_area=float(cv2.contourArea(p.astype(np.float32))) if has else 0.0,
                motion_status="Static",
                motion_magnitude=0.0
            ))
        return out
