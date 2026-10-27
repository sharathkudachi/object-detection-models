# Object Detection Models & Systems

A comprehensive collection of object detection, military surveillance, distance estimation, and AI vehicle convoy analysis models and applications based on YOLO architectures (YOLOv8 & YOLOv11) and computer vision pipelines.

---

## 📁 Repository Structure

| Folder / Project | Description | Key Tech Stack |
| :--- | :--- | :--- |
| **[`ai-military-surveillance-system`](./ai-military-surveillance-system)** | Military surveillance system with optical flow motion tracking and YOLO-based threat detection | Python, YOLO, OpenCV, Optical Flow |
| **[`full`](./full)** | Full-stack AI Vehicle Convoy Analysis Platform with REST API and interactive web interface | FastAPI, React, Vite, YOLOv8-seg |
| **[`InvictusRexDrone-Based-Military-Vehicle-Detection`](./InvictusRexDrone-Based-Military-Vehicle-Detection)** | Drone-based military vehicle detection benchmarks across YOLOv8 & YOLOv11 (Nano, Small, Medium) with evaluation metrics and Jupyter notebooks | PyTorch, Ultralytics YOLOv8/v11, Git LFS |
| **[`yolo-distance-estimation`](./yolo-distance-estimation)** | Real-time object distance and depth estimation combining YOLO bounding boxes with MiDaS monocular depth maps | YOLO, MiDaS, ONNX Runtime, OpenCV |
| **[`YOLOv8-Object-Detection-on-Video-with-OpenCV`](./YOLOv8-Object-Detection-on-Video-with-OpenCV)** | Real-time video stream and webcam object detection pipeline with bounding box rendering | YOLOv8, OpenCV, CVZone |
| **[`Yolo-Weights`](./Yolo-Weights)** | Pre-trained model weights (e.g., `yolov8l.pt`) shared across pipelines | Ultralytics Weights |

---

## 🚀 Projects Overview

### 1. [AI Military Surveillance System](./ai-military-surveillance-system)
- Automated surveillance pipeline analyzing motion and classifying targets.
- Optical flow motion analysis combined with custom-trained military object detection weights (`best1.pt`).
- Includes evaluation curves (`confusion_matrix.png`, `BoxPR_curve.png`) and demo project videos.

### 2. [AI Vehicle Convoy Analysis Platform](./full)
- Full-stack production-grade architecture for convoy surveillance and video analysis.
- **Backend:** FastAPI pipeline orchestrating instance segmentation (`yolov8n-seg.pt`), motion tracking, and convoy grouping.
- **Frontend:** Modern interactive dashboard built with React and Vite.

### 3. [Drone-Based Military Vehicle Detection](./InvictusRexDrone-Based-Military-Vehicle-Detection)
- Comprehensive comparative study between YOLOv8 and YOLOv11 architectures on aerial drone imagery.
- Features models across various parameter sizes:
  - YOLOv11 Nano, Small, and Medium (`mvd-v11n.ipynb`, `mvd-v11s.ipynb`, `mvd-v11m.ipynb`)
  - YOLOv8 Nano, Small, and Medium (`mvd-v8n.ipynb`, `mvd-v8s.ipynb`, `mvd-v8m.ipynb`)
- Includes complete evaluation metrics, PR curves, and detection outputs for each variant.
- Dataset zip managed via **Git LFS**.

### 4. [YOLO Distance Estimation](./yolo-distance-estimation)
- Monocular distance estimation integrating YOLO 2D bounding boxes with depth maps produced by MiDaS (`midas_small.onnx`).
- Includes camera calibration scripts (`calibrate_camera.py`, `camera_calibrator.py`) and evaluation tracking tools.

### 5. [YOLOv8 Video & Webcam Detection](./YOLOv8-Object-Detection-on-Video-with-OpenCV)
- Lightweight real-time OpenCV detection with CVZone overlays.
- Processes webcam streams or sample test video clips (`motorbikes.mp4`, `cars.mp4`, `bikes.mp4`, `people.mp4`).

---

## 📦 Prerequisites & Git LFS

This repository includes large assets (such as datasets) tracked with [Git Large File Storage (Git LFS)](https://git-lfs.com).

When cloning the repository, ensure Git LFS is installed and initialized:

```bash
git lfs install
git clone https://github.com/sharathkudachi/object-detection-models.git
cd object-detection-models
git lfs pull
```

---

## 🛠️ General Setup & Usage

To run individual projects, navigate to the specific project directory and install the corresponding requirements:

```bash
# Example: Setting up YOLO Distance Estimation
cd yolo-distance-estimation
pip install -r requirements.txt
python run_detection.py

# Example: Running the Convoy Analysis Platform Backend
cd full/backend
pip install -r ../requirements.txt
uvicorn app.main:app --reload
```

---

## 📄 License
Individual sub-projects retain their respective licenses and attribution. Refer to the project folders for specific details.
