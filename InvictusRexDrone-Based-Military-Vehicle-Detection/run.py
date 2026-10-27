import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import torch
import numpy as np
from ultralytics import YOLO

MODELS = {
    "v8m": PROJECT_ROOT / "YOLOv8m Results & Metrics" / "weights" / "best.pt",
    "v8s": PROJECT_ROOT / "YOLOv8s Results & Metrics" / "weights" / "best.pt",
    "v8n": PROJECT_ROOT / "YOLOv8n Results & Metrics" / "weights" / "best.pt",
    "v11m": PROJECT_ROOT / "YOLOv11m Results & Metrics" / "weights" / "best.pt",
    "v11s": PROJECT_ROOT / "YOLOv11s Results & Metrics" / "weights" / "best.pt",
    "v11n": PROJECT_ROOT / "YOLOv11n Results & Metrics" / "weights" / "best.pt",
}

DEFAULT_MODEL = MODELS["v8m"]
DEFAULT_SOURCE = PROJECT_ROOT / "Testing Videos" / "Test3.mp4"

def resolve_source(source_str: str, project_root: Path):
    if str(source_str).isdigit():
        return int(source_str)
    
    p = Path(source_str)
    if p.exists():
        return str(p.resolve())
    
    # Try relative to project root
    candidate = project_root / p
    if candidate.exists():
        return str(candidate.resolve())
    
    # Try inside "Testing Videos"
    candidate_tv = project_root / "Testing Videos" / p
    if candidate_tv.exists():
        return str(candidate_tv.resolve())
    
    # Case-insensitive search in "Testing Videos"
    tv_dir = project_root / "Testing Videos"
    if tv_dir.exists():
        for f in tv_dir.iterdir():
            if f.name.lower() == p.name.lower() or f.stem.lower() == p.stem.lower():
                return str(f.resolve())
            
    # Case-insensitive search in project root
    for f in project_root.iterdir():
        if f.name.lower() == p.name.lower() or f.stem.lower() == p.stem.lower():
            return str(f.resolve())

    return str(p)

def process_video(model_path, source, conf=0.25, show=True, max_frames=None):
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / "output_video.mp4"
    temp_output_path = output_dir / "temp_output_video.mp4"

    # Clean up any leftover temporary file
    if temp_output_path.exists():
        try:
            temp_output_path.unlink()
        except Exception:
            pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model: {model_path.name} (device: {device})")
    print(f"Input source: {source}")
    model = YOLO(str(model_path)).to(device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    print(f"Video Info: {width}x{height} @ {fps:.1f} FPS, Total Frames: {total_frames if total_frames > 0 else 'Stream'}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("[ERROR] Failed to initialize VideoWriter with mp4v codec.")
        cap.release()
        return False

    processed_count = 0
    success = False
    gui_available = show

    print("\nProcessing frames...")
    try:
        while True:
            if max_frames and processed_count >= max_frames:
                print(f"Reached requested frame limit: {max_frames}")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference
            results = model(frame, device=device, conf=conf, verbose=False)
            annotated_frame = results[0].plot()

            # Write annotated frame to temporary output
            writer.write(annotated_frame)
            processed_count += 1

            if processed_count % 50 == 0 or processed_count == total_frames:
                progress = f"[{processed_count}/{total_frames}]" if total_frames > 0 else f"[{processed_count} frames]"
                print(f"Processed {progress} frames...")

            if gui_available:
                try:
                    cv2.imshow("Military Vehicle Detection (Drone View)", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing stopped by user ('q' pressed).")
                        break
                except cv2.error:
                    gui_available = False

        if processed_count > 0:
            success = True

    except Exception as e:
        print(f"\n[ERROR] An error occurred during video processing: {e}")
        success = False
    finally:
        cap.release()
        writer.release()
        if gui_available:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if success and temp_output_path.exists() and temp_output_path.stat().st_size > 0:
        # Atomic replace: replace previous output_video.mp4 only on success
        if final_output_path.exists():
            try:
                final_output_path.unlink()
            except Exception:
                pass
        temp_output_path.replace(final_output_path)
        print("\n" + "="*60)
        print("[SUCCESS] Processed video saved successfully!")
        print(f"Total Frames Processed : {processed_count}")
        print(f"Full Output Path       : {final_output_path.resolve()}")
        print(f"Output File Size       : {final_output_path.stat().st_size:,} bytes")
        print("="*60 + "\n")
        return True
    else:
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except Exception:
                pass
        print("\n[FAILED] Video processing failed or no frames were produced. Previous output was preserved.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Drone-Based Military Vehicle Detection")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help=f"Model variant (e.g. v8m, v8s, v8n, v11m, v11s, v11n) or path to .pt file. Default: v8m",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help="Path to video file (e.g. 'Testing Videos/Test1.mp4' or 'test1.mp4'), webcam index ('0'), or image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0 to 1.0). Default: 0.25",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run without displaying GUI window (useful for headless / background processing)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process",
    )
    args = parser.parse_args()

    # Resolve model alias if provided
    model_input = args.model.lower().strip()
    if model_input in MODELS:
        model_path = MODELS[model_input]
    else:
        model_path = Path(args.model)

    if not model_path.exists():
        print(f"Error: Model weights not found at: {model_path}")
        sys.exit(1)

    source = resolve_source(args.source, PROJECT_ROOT)

    show_window = not args.no_show
    success = process_video(
        model_path=model_path,
        source=source,
        conf=args.conf,
        show=show_window,
        max_frames=args.max_frames,
    )
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
