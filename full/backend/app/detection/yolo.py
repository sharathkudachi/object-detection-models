import threading
from pathlib import Path
from typing import Dict, Optional, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ..config import settings, ROOT
from ..interfaces import BaseDetector, Det


def resolve_device(pref: str) -> str:
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if pref == "cuda":
        raise RuntimeError("CUDA requested but not available")
    return "cpu"


class ModelRegistry:
    """Thread-safe persistent model cache preventing repeated weight loading from disk."""
    _lock = threading.Lock()
    _models: Dict[str, YOLO] = {}

    @classmethod
    def get_or_load(cls, model_path: str, device: str = "cpu") -> YOLO:
        key = f"{model_path}:{device}"
        with cls._lock:
            if key in cls._models:
                return cls._models[key]

            # Resolve path if relative
            p = Path(model_path)
            if not p.is_absolute() and not p.exists():
                search_dirs = [
                    ROOT,
                    ROOT / "models",
                    ROOT.parent / "full",
                    ROOT.parent / "ai-military-surveillance-system",
                    ROOT.parent / "Yolo-Weights"
                ]
                for sd in search_dirs:
                    cand = sd / model_path
                    if cand.exists():
                        p = cand
                        break

            # Optimize PyTorch CPU execution threads if running on CPU
            if device == "cpu":
                torch.set_num_threads(min(8, torch.get_num_threads() or 4))

            loaded_model = YOLO(str(p))

            # Warm-up pass to initialize PyTorch graphs and predictor
            try:
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                with torch.inference_mode():
                    _ = loaded_model.predict(dummy, imgsz=320, verbose=False, device=device)
            except Exception:
                pass

            cls._models[key] = loaded_model
            return loaded_model

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._models.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class YoloDetector(BaseDetector):
    """Detection + optional instance masks + persistent IDs (ByteTrack) with persistent caching."""

    def __init__(
        self,
        model: str,
        device: str = "auto",
        conf: float = 0.35,
        iou: float = 0.5,
        segment: bool = True,
        tracker: str = "bytetrack.yaml",
        classes: Optional[list] = None,
        imgsz: int = 640
    ):
        self.model_raw = model
        self.conf = conf
        self.iou = iou
        self.segment = segment
        self.tracker = tracker
        self.device = resolve_device(device)
        self.classes = classes
        self.imgsz = imgsz
        self.model = None
        self.names = {}
        self.is_segmentation = False
        self.half = (self.device.startswith("cuda"))

    def load(self):
        """Fetch model from persistent ModelRegistry."""
        self.model = ModelRegistry.get_or_load(self.model_raw, self.device)
        self.names = self.model.names or {}
        task = getattr(self.model, "task", None) or ""
        self.is_segmentation = "segment" in str(task).lower() or "seg" in str(self.model_raw).lower()
        self.reset_tracker()

    def reset_tracker(self):
        """Reset internal ByteTrack tracking state cleanly between video jobs."""
        if hasattr(self.model, "predictor") and self.model.predictor:
            trackers = getattr(self.model.predictor, "trackers", None)
            if trackers and isinstance(trackers, list):
                for t in trackers:
                    if hasattr(t, "reset"):
                        try:
                            t.reset()
                        except Exception:
                            pass

    def predict(self, frame: np.ndarray) -> List[Det]:
        do_seg = self.segment and self.is_segmentation
        is_military_model = "best" in str(self.model_raw).lower()
        classes_filter = None if is_military_model else (self.classes if self.classes else None)

        with torch.inference_mode():
            try:
                r = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    classes=classes_filter,
                    device=self.device,
                    verbose=False
                )[0]
            except Exception:
                r = self.model.predict(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    classes=classes_filter,
                    device=self.device,
                    verbose=False
                )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

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
