# YOLO Object Detection on Video

This project implements YOLOv8 (You Only Look Once) object detection on a video using Python and OpenCV. YOLO is a state-of-the-art, real-time object detection system that achieves high accuracy and fast processing times.

![Object Detection Demo](demo.gif)

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your_username/yolo-object-detection.git
    ```

2. Install the required libraries:

    ```
    pip install -r requirements.txt
    ```

3. Download the YOLO weights file (e.g., `yolov8l.pt`) and place it in the `Yolo-Weights` directory.

## Usage

1. Replace `bikes.mp4` in the `Videos` directory with your desired video file.

2. Run the Python script:

    ```
    python yolo_detection.py
    ```

3. The script will perform object detection on the video frames using YOLO and save the output frames with bounding boxes in the `output_frames` directory.

4. Once the processing is complete, the script will create a new video (`output_video.mp4`) with the object detection results.

## Output Video

Here is the output video with object detection applied:

https://github.com/JoyKarmoker/YOLOv8-Object-Detection-on-Video-with-OpenCV/assets/48152122/f11bcab0-8c67-4186-8eeb-57859a91e595


## Customization

- You can customize the classes to be detected by modifying the `classNames` list in the Python script.
- Adjust the confidence threshold and other parameters for object detection according to your requirements.

## Credits

- [YOLOv8](https://github.com/ultralytics) by Ultralytics for the YOLO implementation.
- [OpenCV](https://opencv.org/) for image and video processing.
- [cvzone](https://github.com/cvzone/cvzone) for drawing bounding boxes and text on images.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
