import base64
import logging
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO

from . import pipeline
from .config import ROOT, settings
from .detection.yolo import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("surveillance")

app = FastAPI(
    title="AI Military Surveillance & Convoy Analysis",
    version="2.0.0",
    description="High-performance AI Military Surveillance & Convoy Analysis System"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bounded worker pool to prevent resource starvation (Max 2 concurrent video processing workers)
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="VideoWorker")
JOBS: dict = {}
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

LIVE_MODEL = None


class StartReq(BaseModel):
    upload_id: str
    model: str = settings.models[0] if settings.models else "best1.pt"
    mode: str = Field("military+optical_flow", description="Analysis mode: military+optical_flow, detection+segmentation, detection")
    profile: str = Field("high_accuracy", description="Processing profile: high_accuracy (1x), balanced (2x), fast (3x)")
    imgsz: int = Field(640, ge=320, le=1280, description="Inference image size: 480 (fast), 640 (default), 800 (high detail)")
    conf: float = Field(settings.conf, ge=0.05, le=0.95)
    iou: float = Field(settings.iou, ge=0.1, le=0.95)
    stride: int = Field(1, ge=1, le=10)
    link_dist_frac: float = Field(0.2, gt=0, le=1.0)
    min_members: int = Field(3, ge=2, le=20)
    min_persist_frames: int = Field(15, ge=1)
    motion_threshold: float = Field(2.0, ge=0.1, le=20.0)


class SampleSelectReq(BaseModel):
    filename: str


class LiveFrameReq(BaseModel):
    image_base64: str
    model: Optional[str] = "best1.pt"
    conf: Optional[float] = 0.35


def _sanitize_id(identifier: str) -> str:
    """Validate and sanitize ID to prevent directory traversal attacks."""
    if not identifier or not re.match(r"^[a-zA-Z0-9_-]+$", identifier):
        raise HTTPException(400, "Invalid identifier format")
    return identifier


def _find_upload(uid: str) -> Path:
    clean_uid = _sanitize_id(uid)
    for p in settings.upload_dir.glob(f"{clean_uid}.*"):
        return p
    raise HTTPException(404, "Upload not found")


@app.get("/api/v1/system")
def system():
    import torch
    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    return {
        "device": device_name,
        "cuda": cuda,
        "models": settings.models,
        "default_model": settings.models[0] if settings.models else "best1.pt",
        "cpu_threads": torch.get_num_threads(),
        "profiles": ["high_accuracy", "balanced", "fast"],
        "modes": ["military+optical_flow", "detection+segmentation", "detection"]
    }


@app.get("/api/v1/samples")
def list_samples():
    """List sample videos available in workspace and parent folders for instant 1-click testing."""
    samples = []
    seen = set()
    search_dirs = [
        ROOT,
        ROOT.parent / "ai-military-surveillance-system",
        ROOT.parent / "full" / "storage" / "uploads"
    ]
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for f in d.glob("*.mp4"):
                if f.name not in seen and not f.name.startswith("output_") and not f.name.startswith("test_h264") and not f.name.startswith("annotated_"):
                    seen.add(f.name)
                    size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                    samples.append({
                        "name": f.name,
                        "path": str(f.resolve()),
                        "size_mb": size_mb
                    })
    return {"samples": samples}


@app.post("/api/v1/sample/select")
def select_sample(req: SampleSelectReq):
    """Load an existing sample video into the uploads directory with validation."""
    clean_name = Path(req.filename).name
    target_file = None
    search_dirs = [
        ROOT,
        ROOT.parent / "ai-military-surveillance-system",
        ROOT.parent / "full" / "storage" / "uploads"
    ]
    for d in search_dirs:
        cand = d / clean_name
        if cand.exists() and cand.is_file():
            target_file = cand
            break

    if not target_file:
        raise HTTPException(404, f"Sample video '{clean_name}' not found")

    ext = target_file.suffix.lower()
    uid = uuid.uuid4().hex
    dest = settings.upload_dir / f"{uid}{ext}"
    shutil.copyfile(target_file, dest)

    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "Unable to read sample video")

    meta = {
        "upload_id": uid,
        "filename": clean_name,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return meta


@app.post("/api/v1/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    uid = uuid.uuid4().hex
    dest = settings.upload_dir / f"{uid}{ext}"
    size = 0
    too_big = False

    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > settings.max_upload_mb << 20:
                too_big = True
                break
            f.write(chunk)

    if too_big:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"File exceeds maximum upload size of {settings.max_upload_mb} MB")

    # Validate that uploaded file is an actual readable video
    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "Uploaded file is not a valid or readable video")

    meta = {
        "upload_id": uid,
        "filename": file.filename,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    log.info("Uploaded video %s (%s bytes, %s frames)", uid, size, meta["frames"])
    return meta


@app.get("/api/v1/video/{uid}")
def video(uid: str):
    """Stream input video with range request support."""
    return FileResponse(_find_upload(uid), media_type="video/mp4")


def _work(aid: str, req: StartReq, src: Path):
    job = JOBS[aid]
    json_path = settings.result_dir / f"{aid}.json"
    video_path = settings.output_dir / f"{aid}.mp4"

    try:
        pipeline.run(job, req, src, json_path, video_path)
        job["status"] = "cancelled" if job.get("cancel") else "completed"
        job["stage"] = "Completed" if job["status"] == "completed" else "Cancelled"
        job["output_video"] = f"/api/v1/analysis/{aid}/video"
        job["has_output_video"] = video_path.exists()
    except Exception as e:
        log.exception("Analysis job %s failed: %s", aid, e)
        job.update(status="failed", stage="Failed", error=str(e))


@app.post("/api/v1/analysis/start")
def start(req: StartReq):
    src = _find_upload(req.upload_id)
    aid = uuid.uuid4().hex[:12]
    JOBS[aid] = {
        "id": aid,
        "status": "queued",
        "stage": "Queued in Worker Pool",
        "progress": 0.0,
        "fps": 0.0,
        "frame": 0,
        "total_frames": 0,
        "elapsed_s": 0.0,
        "eta_s": 0,
        "profile": req.profile,
        "model": req.model,
        "error": None,
        "cancel": False,
        "has_output_video": False,
        "output_video": None,
        "perf_summary": None
    }
    JOB_EXECUTOR.submit(_work, aid, req, src)
    return JOBS[aid]


def _job(aid: str):
    clean_aid = _sanitize_id(aid)
    if clean_aid not in JOBS:
        raise HTTPException(404, "Analysis job not found")
    return JOBS[clean_aid]


@app.get("/api/v1/analysis/{aid}")
def status(aid: str):
    return _job(aid)


@app.post("/api/v1/analysis/{aid}/stop")
def stop(aid: str):
    job = _job(aid)
    job["cancel"] = True
    job["stage"] = "Cancelling..."
    return {"ok": True, "message": "Cancellation signal sent"}


@app.get("/api/v1/analysis/{aid}/result")
def result(aid: str):
    job = _job(aid)
    if job["status"] not in ("completed", "cancelled"):
        raise HTTPException(409, f"Analysis results not ready (status: {job['status']})")
    clean_aid = _sanitize_id(aid)
    result_path = settings.result_dir / f"{clean_aid}.json"
    if not result_path.exists():
        raise HTTPException(404, "Result file not found")
    return FileResponse(result_path, media_type="application/json")


@app.get("/api/v1/analysis/{aid}/video")
def get_output_video(aid: str):
    """Stream generated annotated output video with HTTP Range header support."""
    clean_aid = _sanitize_id(aid)
    out_file = settings.output_dir / f"{clean_aid}.mp4"
    if not out_file.exists():
        raise HTTPException(404, "Annotated video not found or processing incomplete")
    return FileResponse(
        out_file,
        media_type="video/mp4",
        filename=f"annotated_{clean_aid}.mp4"
    )


@app.get("/api/v1/analysis/{aid}/download")
def download_output_video(aid: str):
    """Download the generated annotated output MP4 video."""
    clean_aid = _sanitize_id(aid)
    out_file = settings.output_dir / f"{clean_aid}.mp4"
    if not out_file.exists():
        raise HTTPException(404, "Output video file not found")
    return FileResponse(
        out_file,
        media_type="application/octet-stream",
        filename=f"annotated_surveillance_{clean_aid}.mp4"
    )


MODELS_INFO_CACHE = None

@app.get("/api/v1/models/info")
def models_info():
    """Return model classes, tasks, and details (cached)."""
    global MODELS_INFO_CACHE
    if MODELS_INFO_CACHE is not None:
        return {"models": MODELS_INFO_CACHE}

    info = []
    for m in settings.models:
        model_path = Path(m)
        if not model_path.is_absolute() and not model_path.exists():
            for sd in [ROOT, ROOT / "models", ROOT.parent / "full", ROOT.parent / "ai-military-surveillance-system"]:
                cand = sd / m
                if cand.exists():
                    model_path = cand
                    break

        size_mb = round(model_path.stat().st_size / (1024 * 1024), 1) if model_path.exists() else 0.0
        classes_map = {}
        is_seg = "seg" in m.lower()

        if "best" in m.lower():
            classes_map = {
                0: "camouflage_soldier", 1: "weapon", 2: "military_tank", 3: "military_truck",
                4: "military_vehicle", 5: "soldier", 6: "civilian_vehicle", 7: "military_artillery",
                8: "military_aircraft", 9: "military_warship"
            }
        else:
            try:
                temp_yolo = ModelRegistry.get_or_load(str(model_path), "cpu")
                classes_map = temp_yolo.names or {}
                task = getattr(temp_yolo, "task", None) or ""
                is_seg = "segment" in str(task).lower() or "seg" in str(model_path).lower()
            except Exception:
                pass

        info.append({
            "name": m,
            "size_mb": size_mb,
            "is_segmentation": is_seg,
            "is_military": "best" in m.lower(),
            "classes": classes_map,
            "classes_count": len(classes_map)
        })
    MODELS_INFO_CACHE = info
    return {"models": info}


@app.post("/api/v1/live/process_frame")
def process_live_frame(req: LiveFrameReq):
    """Process a single frame from webcam feed for real-time live surveillance."""
    global LIVE_MODEL
    try:
        img_data = base64.b64decode(req.image_base64.split(",")[-1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Invalid frame data")

        m_path = req.model or "best1.pt"
        live_model = ModelRegistry.get_or_load(m_path, settings.device)

        res = live_model.predict(frame, conf=req.conf or 0.35, imgsz=480, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                c = int(b.cls[0])
                cf = float(b.conf[0])
                c_name = live_model.names.get(c, str(c))
                xy = b.xyxy[0].tolist()
                dets.append({
                    "class_name": c_name,
                    "conf": round(cf, 2),
                    "bbox": [round(v, 1) for v in xy]
                })

        return {"detections": dets}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/v1/jobs/cleanup")
def cleanup_storage(max_age_hours: int = 24):
    """Clean up old temporary uploads and outputs older than max_age_hours."""
    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    cleaned = 0
    for target_dir in [settings.upload_dir, settings.output_dir, settings.result_dir]:
        for f in target_dir.glob("*.*"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
                    cleaned += 1
            except Exception:
                pass
    return {"cleaned_files": cleaned}


# Mount frontend static UI
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="ui")
