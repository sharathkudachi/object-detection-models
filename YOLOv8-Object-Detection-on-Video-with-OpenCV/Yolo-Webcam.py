from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os

import sys

# Load YOLO model
weights_path = "../Yolo-Weights/yolov8l.pt"
if not os.path.exists(weights_path) and os.path.exists("Yolo-Weights/yolov8l.pt"):
    weights_path = "Yolo-Weights/yolov8l.pt"
elif not os.path.exists(weights_path) and os.path.exists("yolov8l.pt"):
    weights_path = "yolov8l.pt"
model = YOLO(weights_path)

# Video capture
video_source = sys.argv[1] if len(sys.argv) > 1 else "../Videos/motorbikes.mp4"
if not os.path.exists(video_source) and os.path.exists(os.path.join("Videos", os.path.basename(video_source))):
    video_source = os.path.join("Videos", os.path.basename(video_source))
cap = cv2.VideoCapture(video_source)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Loaded video: {video_source} (FPS: {fps}, Total Frames: {total_frames})")


# Output folder for saving frames
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)
for f in os.listdir(output_folder):
    if f.endswith(".jpg"):
        try:
            os.remove(os.path.join(output_folder, f))
        except Exception:
            pass

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    if not success:
        break  # Break out of the loop if there are no more frames to read

    # Perform object detection with YOLO
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Save frame with YOLO detections applied
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    output_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(output_path, img)

    if frame_idx % 25 == 0 or frame_idx == total_frames:
        print(f"Processed frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Detection stopped by user.")
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()

# Recreate video from saved frames
output_video_path = "output_video.mp4"
files = [f for f in sorted(os.listdir(output_folder)) if f.endswith(".jpg")]
if files:
    first_img = cv2.imread(os.path.join(output_folder, files[0]))
    height, width, layers = first_img.shape
    size = (width, height)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), size)
    print(f"Creating output video from {len(files)} frames at {int(fps)} FPS...")
    for filename in files:
        img_path = os.path.join(output_folder, filename)
        img = cv2.imread(img_path)
        out.write(img)
    out.release()
    print("Video created successfully.")
else:
    print("No frames were processed.")
