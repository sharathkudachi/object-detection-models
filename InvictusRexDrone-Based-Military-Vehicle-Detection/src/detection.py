import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# === PATH RESOLUTION ===
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BASE_DIR / "YOLOv8m Results & Metrics" / "weights" / "best.pt"
DEFAULT_VIDEO = BASE_DIR / "Testing Videos" / "Test3.mp4"

def resolve_source(source_str: str, base_dir: Path):
    if str(source_str).isdigit():
        return int(source_str)
    p = Path(source_str)
    if p.exists():
        return str(p.resolve())
    candidate = base_dir / p
    if candidate.exists():
        return str(candidate.resolve())
    tv_dir = base_dir / "Testing Videos"
    if tv_dir.exists():
        candidate_tv = tv_dir / p
        if candidate_tv.exists():
            return str(candidate_tv.resolve())
        for f in tv_dir.iterdir():
            if f.name.lower() == p.name.lower() or f.stem.lower() == p.stem.lower():
                return str(f.resolve())
    for f in base_dir.iterdir():
        if f.name.lower() == p.name.lower() or f.stem.lower() == p.stem.lower():
            return str(f.resolve())
    return str(p)

parser = argparse.ArgumentParser(description="Drone Military Vehicle Detection")
parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to YOLO weights (.pt)")
parser.add_argument("--source", type=str, default=str(DEFAULT_VIDEO), help="Path to video file, webcam index, or image")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
parser.add_argument("--no-show", action="store_true", help="Disable GUI window display")
parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process")
args, _ = parser.parse_known_args()

MODEL_PATH = args.model
VIDEO_PATH = resolve_source(args.source, BASE_DIR)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === OUTPUT CONFIGURATION ===
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "output_video.mp4"
TEMP_OUTPUT_PATH = OUTPUT_DIR / "temp_output_video.mp4"

if TEMP_OUTPUT_PATH.exists():
    try:
        TEMP_OUTPUT_PATH.unlink()
    except Exception:
        pass

# === LOAD MODEL ===
print(f'Using device: {DEVICE}')
print(f'Loading model: {MODEL_PATH}')
model = YOLO(MODEL_PATH).to(DEVICE)

# === LOAD VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_PATH}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if fps <= 0 or np.isnan(fps):
    fps = 25.0

print(f"Video input: {VIDEO_PATH}")
print(f"Resolution: {width}x{height} @ {fps:.1f} FPS (Total frames: {total_frames if total_frames > 0 else 'Stream'})")

# === VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(TEMP_OUTPUT_PATH), fourcc, fps, (width, height))
if not writer.isOpened():
    print("Error: Could not initialize VideoWriter with mp4v codec.")
    cap.release()
    exit(1)

gui_available = not args.no_show
processed_frames = 0
success = False

# === MAIN LOOP ===
try:
    while True:
        if args.max_frames and processed_frames >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame, device=DEVICE, conf=args.conf, verbose=False)

        # Draw results
        annotated_frame = results[0].plot()

        # Write to temporary output video
        writer.write(annotated_frame)
        processed_frames += 1

        if processed_frames % 50 == 0 or processed_frames == total_frames:
            progress = f"[{processed_frames}/{total_frames}]" if total_frames > 0 else f"[{processed_frames}]"
            print(f"Processed {progress} frames...")

        # Show frame if display is active
        if gui_available:
            try:
                cv2.imshow("YOLOv8 Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user ('q' pressed).")
                    break
            except cv2.error:
                gui_available = False

    if processed_frames > 0:
        success = True

except Exception as e:
    print(f"\n[ERROR] Error occurred during processing: {e}")
    success = False
finally:
    # === CLEANUP ===
    cap.release()
    writer.release()
    if gui_available:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

# === ATOMIC OUTPUT REPLACEMENT ===
if success and TEMP_OUTPUT_PATH.exists() and TEMP_OUTPUT_PATH.stat().st_size > 0:
    if OUTPUT_VIDEO_PATH.exists():
        try:
            OUTPUT_VIDEO_PATH.unlink()
        except Exception:
            pass
    TEMP_OUTPUT_PATH.replace(OUTPUT_VIDEO_PATH)
    print("\n" + "="*60)
    print("[SUCCESS] Processed video saved successfully!")
    print(f"Frames Processed : {processed_frames}")
    print(f"Full Output Path : {OUTPUT_VIDEO_PATH.resolve()}")
    print(f"File Size        : {OUTPUT_VIDEO_PATH.stat().st_size:,} bytes")
    print("="*60 + "\n")
else:
    if TEMP_OUTPUT_PATH.exists():
        try:
            TEMP_OUTPUT_PATH.unlink()
        except Exception:
            pass
    print("\n[FAILED] Video processing failed. Previous output was not modified.")
    exit(1)
