import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(ROOT / ".env")
except ImportError:
    ROOT = Path(__file__).resolve().parents[2]


def _p(k, d):
    return Path(os.getenv(k, ROOT / d))


def discover_models():
    """Discover available .pt models dynamically across workspace and project directories."""
    found = []
    seen = set()
    search_dirs = [
        ROOT,
        ROOT / "models",
        ROOT.parent / "ai-military-surveillance-system",
        ROOT.parent / "full",
        ROOT.parent / "Yolo-Weights"
    ]
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for pt in d.glob("*.pt"):
                if pt.name not in seen:
                    seen.add(pt.name)
                    found.append(pt.name)

    # Prioritize military surveillance model first, then segmentation models
    priority_order = ["best1.pt", "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8n.pt", "yolov8l.pt"]
    ordered = [m for m in priority_order if m in seen]
    for m in found:
        if m not in ordered:
            ordered.append(m)

    if not ordered:
        ordered = ["best1.pt", "yolov8n-seg.pt", "yolov8n.pt"]
    return ordered


class Settings:
    upload_dir = _p("UPLOAD_DIR", "storage/uploads")
    result_dir = _p("RESULT_DIR", "storage/results")
    output_dir = _p("OUTPUT_DIR", "storage/outputs")
    models = discover_models()
    device = os.getenv("DEVICE", "auto")  # auto | cpu | cuda
    conf = float(os.getenv("CONFIDENCE_THRESHOLD", 0.35))
    iou = float(os.getenv("IOU_THRESHOLD", 0.50))
    max_upload_mb = int(os.getenv("MAX_UPLOAD_SIZE", 500))
    # COCO ids: 2 car, 3 motorcycle, 5 bus, 7 truck (used only if explicit vehicle filtering is desired)
    classes_str = os.getenv("VEHICLE_CLASSES", "")
    classes = [int(c) for c in classes_str.split(",") if c.strip().isdigit()] if classes_str else None


settings = Settings()
for _d in (settings.upload_dir, settings.result_dir, settings.output_dir):
    _d.mkdir(parents=True, exist_ok=True)
