# Project Documentation — HackZen 2.0 Day 3

## Project Title
Real-Time Object Detection Using Python, OpenCV, and YOLO11n

## Abstract
This project implements an object-detection pipeline that processes a webcam stream, image, or video and highlights objects recognized by a pretrained YOLO11n model. Bounding boxes, class labels, and confidence scores are shown to make the predictions understandable. OpenCV handles video capture, visualization, and saving annotated results.

## Problem Statement
Object identification in visual media can be repetitive and slow when done manually. The project explores how a pretrained Computer Vision model can automate detection of common object categories in image and video inputs.

## Objectives
1. Build a runnable object-detection application.
2. Support webcam, image, and video sources.
3. Visualize detected classes with bounding boxes and confidence scores.
4. Save annotated results for review.
5. Record practical observations and document limitations.

## Proposed Solution
Use Ultralytics YOLO11n pretrained weights for inference. OpenCV manages input/output. A confidence threshold filters detections below the selected level. The application draws detections and reports the count per frame.

## System Requirements
- Python 3.10 or a compatible Python version supported by the installed Ultralytics/PyTorch packages.
- Windows, Linux, or macOS.
- Webcam for live mode, optional.
- Internet connection for first-time model-weight download.
- CPU-compatible execution; a supported GPU can improve throughput.

## Technical Approach
1. Parse the input source and confidence threshold.
2. Load the pretrained YOLO11n model.
3. Read an image or capture frames from a camera/video.
4. Run object detection on each image/frame.
5. Obtain class IDs, bounding boxes, and confidence scores.
6. Render predictions, display them, and save output.
7. Release resources on completion.

## Model and Libraries
- **YOLO11n:** pretrained, compact object-detection model.
- **Ultralytics:** model loading and prediction API.
- **OpenCV:** video capture, display, and writing annotated output.
- **NumPy:** numerical dependency.

## Results
Enter results only after running the program. Include:
- Device/processor used.
- Input type and sample duration or image count.
- Number of frames processed.
- Processing time and estimated FPS, if measured.
- Example objects detected and notable missed/incorrect detections.
- Screenshots of successful output.

Do not claim a measured accuracy or FPS until you have measured it. This repository does not include ground-truth annotations, so formal precision, recall, and mAP are not computed by default.

## Limitations
The pretrained model supports a fixed set of classes. Detection can degrade with small objects, occlusion, motion blur, unusual viewpoints, and poor lighting. Confidence scores are not calibrated probabilities. Processing speed depends on hardware. The starter implementation does not perform persistent multi-object tracking, custom training, or formal dataset evaluation.

## Ethical and Privacy Considerations
Only process media you are authorized to use. Obtain consent before recording people, protect identifiable footage, and do not use this prototype as the sole basis for safety-critical or high-impact decisions.

## Future Work
- Add object tracking and persistent IDs.
- Add a graphical interface.
- Benchmark CPU and GPU performance.
- Evaluate with a labelled dataset.
- Fine-tune for a specific domain.
- Export to ONNX and optimize inference.

## Reproducibility
Install dependencies from `requirements.txt`, then follow the commands in `README.md`. Record your Python version, hardware, input file, confidence threshold, and observed results in the final submission.
