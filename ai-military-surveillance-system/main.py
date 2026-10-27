import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time

# Determine video path (CLI argument or default to test_video2.mp4)
if len(sys.argv) > 1:
    joined_arg = " ".join(sys.argv[1:])
    if os.path.exists(joined_arg):
        video_path = joined_arg
    elif sys.argv[1].isdigit():
        video_path = int(sys.argv[1])
    else:
        video_path = sys.argv[1]
elif os.path.exists("test_video2.mp4"):
    video_path = "test_video2.mp4"
else:
    video_path = "test_video.mp4"

print(f"Loading video source: {video_path}")

# Load model
model = YOLO("best1.pt")

cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()

if not ret:
    print(f"Error: Could not open video file or camera '{video_path}'")
    exit(1)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set dynamic output path based on input source
base_name = f"webcam_{video_path}" if isinstance(video_path, int) else os.path.splitext(os.path.basename(video_path))[0]
output_path = f"output_{base_name}.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")
print(f"Output will be saved to: {output_path}")
print("Press 'q' in the display window to stop and save early.\n")

frame_idx = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model(frame, verbose=False)

    detections_count = 0
    moving_count = 0
    static_count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Clamp coordinates to frame dimensions
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            detections_count += 1

            # ROI for optical flow
            roi_prev = prev_gray[y1:y2, x1:x2]
            roi_curr = gray[y1:y2, x1:x2]

            status = "Static"
            color = (0, 255, 0)

            if roi_prev.size > 0 and roi_curr.size > 0:
                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        roi_prev,
                        roi_curr,
                        None,
                        0.5,
                        3,
                        15,
                        3,
                        5,
                        1.2,
                        0
                    )

                    mag, _ = cv2.cartToPolar(
                        flow[..., 0],
                        flow[..., 1]
                    )

                    avg_motion = np.mean(mag)

                    if avg_motion > 2:
                        status = "Moving"
                        color = (0, 0, 255)
                        moving_count += 1
                    else:
                        static_count += 1
                except:
                    static_count += 1
            else:
                static_count += 1

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{class_name} {conf:.2f} | {status}"
            cv2.putText(
                frame,
                text,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Add HUD Overlay (Surveillance Info)
    hud_text = f"Frame: {frame_idx}/{total_frames} | Detected: {detections_count} (Moving: {moving_count}, Static: {static_count})"
    cv2.rectangle(frame, (10, 10), (min(width - 10, 680), 45), (0, 0, 0), -1)
    cv2.putText(frame, hud_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("YOLO + Optical Flow", frame)

    prev_gray = gray.copy()

    # Progress telemetry in terminal every 10 frames
    if frame_idx % 10 == 0 or frame_idx == 1:
        elapsed = time.time() - start_time
        current_fps = frame_idx / elapsed if elapsed > 0 else 0
        pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        eta_seconds = (total_frames - frame_idx) / current_fps if current_fps > 0 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        print(f"\r[Progress: {frame_idx}/{total_frames} ({pct:.1f}%)] FPS: {current_fps:.1f} | Detections: {detections_count} | ETA: {eta_str}", end="", flush=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nStopping processing (user pressed 'q')...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

total_elapsed = time.time() - start_time
print(f"\nProcessing complete! Processed {frame_idx} frames in {total_elapsed:.1f}s.")
print(f"Video saved as: {output_path}")