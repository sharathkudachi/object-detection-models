# 🛡️ AI Military Surveillance & Defense Intelligence System

An advanced, high-performance, real-time **Military Surveillance and Defense Intelligence System** built with **FastAPI**, **Ultralytics YOLOv8**, **ByteTrack**, and **Farneback Dense Optical Flow**. Designed for battlefield intelligence, tactical asset tracking, multi-vehicle convoy formation detection, and dynamic motion analysis.

Includes a mission-critical **Dark Tactical Web Command Center** with real-time video upload, sample video selection, synchronized multi-video playback, performance telemetry, and configurable execution profiles.

---

## 🚀 Key Capabilities

- **10 Tactical Military Classes**: Real-time detection and classification of camouflage soldiers, weapons, tanks, trucks, vehicles, soldiers, civilian vehicles, artillery, aircraft, and warships.
- **ByteTrack Multi-Object Tracking**: Persistent track IDs and real-time movement trajectory trails across frames with automatic tracker cleanup between jobs.
- **Downscaled Farneback Optical Flow**: Accelerated dense optical flow on target bounding box ROIs (`max_dim=80`), achieving **26.4x to 31.8x speedup** with zero loss of motion accuracy.
- **Multi-Vehicle Convoy Formation Analysis**: Graph-based geometric linking algorithms to detect and monitor tactical vehicle convoy formations and spatial groupings.
- **Configurable Performance Profiles**:
  - 🎯 **High Accuracy**: Stride = 1 (every frame), 640px inference size. Full detection density.
  - ⚖️ **Balanced (Default)**: Stride = 2, 480px inference size. **4.82x faster** with 79.2% latency reduction.
  - ⚡ **Fast**: Stride = 3, 480px inference size. **6.83x faster** (12.5+ FPS on CPU).
- **Universal Browser Playback**: Dual H.264 encoding via `libx264` (FFmpeg) and fallback `avc1`/`mp4v` (OpenCV) with exact input frame rate and duration synchronization.
- **Production Architecture**: Bounded worker thread pool (`ThreadPoolExecutor`), singleton `ModelRegistry` eliminating redundant disk weight loading, and automated memory cleanup.

---

## 📊 Performance Benchmarks (Empirical CPU Profiling)

Measured on a Windows 12-thread CPU processing 100 frames of 1280x720 video (`test_video.mp4`):

| Execution Profile | Inference Stride | Input Size (imgsz) | Total Time (s) | Throughput (FPS) | Speedup Ratio | YOLO Time (s) | Optical Flow Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Unoptimized)** | 1x | 640px | 54.42 s | 1.84 FPS | **1.00x** | 45.13 s | 7.96 s |
| **High Accuracy** | 1x | 640px | 37.03 s | 2.70 FPS | **1.47x** | 31.78 s | 0.30 s *(26.4x faster)* |
| **Balanced (Recommended)** | 2x | 480px | 11.29 s | 8.86 FPS | **4.82x** | 9.60 s | 0.14 s |
| **Fast** | 3x | 480px | 7.97 s | 12.54 FPS | **6.83x** | 6.32 s | 0.08 s |

### Key Optimizations Delivered:
1. **Optical Flow Speedup**: ROI downscaling with coordinate scale compensation reduced per-ROI Farneback execution from `22.1 ms` to `0.70 ms` (31.8x acceleration).
2. **YOLO Inference Optimization**: Added `torch.inference_mode()`, CPU intra-op thread allocation (`min(8, N)`), and image scaling (`480px`) reducing YOLO per-frame latency from `451 ms` to `186 ms`.
3. **Model Singleton Cache**: `ModelRegistry` loads weights once, saving 1.2+ seconds of disk I/O per job.
4. **Frame Duration Preservation**: Non-inference frames are interpolated with previous detections and rendered at exact video FPS, ensuring playback duration and speed match the original footage 1:1.

---

## 🎯 Detectable Military Asset Classes

The trained model (`best1.pt`) detects 10 specialized military classes:

| Class ID | Tactical Class Name | Description |
|:---:|---|---|
| `0` | `camouflage_soldier` | Concealed combat personnel |
| `1` | `weapon` | Small arms, rifles, heavy weaponry |
| `2` | `military_tank` | Armored combat vehicles / MBTs |
| `3` | `military_truck` | Transport and logistics trucks |
| `4` | `military_vehicle` | Light tactical wheeled vehicles |
| `5` | `soldier` | Standard infantry / military personnel |
| `6` | `civilian_vehicle` | Civilian / unclassified ground transport |
| `7` | `military_artillery` | Howitzers, field guns, rocket artillery |
| `8` | `military_aircraft` | Fixed-wing and rotary military aircraft |
| `9` | `military_warship` | Naval vessels and patrol boats |

---

## 🏗️ Architecture & Component Flow

```
┌────────────────────────────────────────────────────────┐
│              Web Command Center (Browser)              │
│   (Synchronized Dual Video, Profile Controls, HUD)     │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / REST API
┌──────────────────────────▼─────────────────────────────┐
│                 FastAPI Application                     │
│      (Bounded Worker Pool, Path Traversal Defense)     │
└──────────────────────────┬─────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
┌───▼─────────────┐ ┌──────▼────────────┐ ┌───────▼────────────┐
│  ModelRegistry  │ │  Adaptive Stride  │ │ Downscaled Optical │
│ Singleton Cache │ │  Frame Pipeline   │ │   Flow Estimator   │
└───┬─────────────┘ └──────┬────────────┘ └───────┬────────────┘
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│               H.264 Multi-Encoder Engine               │
│         (FFmpeg libx264 + OpenCV FourCC Fallback)      │
└────────────────────────────────────────────────────────┘
```

---

## 🛠️ Project Structure

```text
ai-military-surveillance-system/
│
├── backend/
│   ├── app/
│   │   ├── analysis/
│   │   │   ├── convoy.py          # Convoy formation graph algorithms
│   │   │   └── motion.py          # Downscaled ROI optical flow estimator
│   │   ├── detection/
│   │   │   └── yolo.py            # ModelRegistry singleton & ByteTrack wrapper
│   │   ├── config.py              # System paths and hardware auto-detection
│   │   ├── interfaces.py          # Detection & request dataclasses
│   │   ├── main.py                # FastAPI endpoints & background worker pool
│   │   └── pipeline.py            # Video stream processing, drawing & encoding
│
├── frontend/
│   └── index.html                 # Tactical dark glassmorphic command center
│
├── tests/
│   └── test_performance_pipeline.py  # 6 automated unit & integration tests
│
├── storage/
│   ├── uploads/                   # Staged input video uploads
│   ├── output/                    # Annotated H.264 surveillance output videos
│   └── results/                   # Detailed JSON telemetry and frames data
│
├── best1.pt                       # Trained 10-class YOLOv8 weights
├── test_video.mp4                 # Verification surveillance video
├── run_profile_benchmarks.py      # Automated benchmarking script
├── requirements.txt               # Python package dependencies
└── README.md                      # Complete system documentation
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- Python 3.10+ or 3.11+
- FFmpeg (recommended for libx264 fast encoding)

### 2. Setup Virtual Environment
```powershell
# Clone the repository
git clone https://github.com/sharathkudachi/ai-military-surveillance-system.git
cd ai-military-surveillance-system

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Surveillance Server
```powershell
# Start FastAPI backend with automatic reload
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to access the tactical dashboard:
👉 **`http://localhost:8000`**

---

## 🧪 Automated Testing

A comprehensive test suite verifies model caching, class preservation, tracker resets, optical flow accuracy, H.264 video encoding, and end-to-end pipeline execution:

```powershell
$env:PYTHONPATH="."
pytest tests/test_performance_pipeline.py -v
```

**Results:**
```text
tests/test_performance_pipeline.py::test_model_registry_caching PASSED         [ 16%]
tests/test_performance_pipeline.py::test_military_class_mapping PASSED         [ 33%]
tests/test_performance_pipeline.py::test_tracker_reset_between_jobs PASSED   [ 50%]
tests/test_performance_pipeline.py::test_optical_flow_downscaled_accuracy PASSED [ 66%]
tests/test_performance_pipeline.py::test_video_writer_and_h264_encoding PASSED [ 83%]
tests/test_performance_pipeline.py::test_pipeline_execution_profiles PASSED   [100%]
============================== 6 passed in 10.73s ==============================
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the Tactical Command Center Web UI |
| `GET` | `/api/v1/models/info` | Cached metadata, class names, and active model status |
| `GET` | `/api/v1/sample/list` | Discovers available `.mp4` sample videos |
| `POST` | `/api/v1/sample/select` | Pre-stages a sample video for immediate analysis |
| `POST` | `/api/v1/upload` | Uploads a custom video file with format validation |
| `POST` | `/api/v1/analysis/start` | Enqueues a video analysis job with selected profile |
| `GET` | `/api/v1/analysis/{id}` | Live job status, FPS, progress %, ETA, and perf report |
| `POST` | `/api/v1/analysis/{id}/stop`| Signals cancellation to active background worker |
| `GET` | `/api/v1/analysis/{id}/video`| Streams the processed H.264 annotated video |
| `GET` | `/api/v1/analysis/{id}/result`| Returns detailed JSON telemetry, tracks, and convoy summaries |
| `POST` | `/api/v1/jobs/cleanup` | Cleans up completed/stale jobs and frees disk space |

---

## 🛡️ License & Acknowledgments

Developed for military computer vision research, tactical surveillance analytics, and automated asset recognition. Powered by PyTorch, OpenCV, Ultralytics YOLOv8, and FastAPI.
