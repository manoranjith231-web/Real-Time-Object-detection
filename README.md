# HackZen 2.0 Day 3: Real-Time Object Detection

A beginner-friendly Computer Vision project that detects everyday objects in a webcam stream, image, or video using a pretrained YOLO11n model and OpenCV.

## 1. Problem Statement

Manually identifying and locating objects in images or video takes time. This project demonstrates an automated Computer Vision pipeline that identifies supported object classes, draws bounding boxes, and displays confidence scores.

## 2. Proposed Solution

The application uses the pretrained YOLO11n object-detection model. OpenCV reads webcam/video frames and displays the annotated output. The program also saves the processed result in the `outputs/` directory.

## 3. Features

- Detects multiple objects in one frame.
- Shows object labels, bounding boxes, and confidence scores.
- Supports a webcam, image file, or video file.
- Shows the number of detected objects per frame.
- Saves annotated images and videos.
- Lets the user adjust the confidence threshold.

## 4. Technologies and Model

| Component | Purpose |
|---|---|
| Python | Main programming language |
| Ultralytics YOLO11n | Pretrained object detector |
| OpenCV | Camera/video input, display, and output writing |
| NumPy | Numerical dependency used by the computer-vision stack |

The `yolo11n.pt` weights are downloaded automatically by Ultralytics the first time the model is loaded, if they are not already available. An internet connection is therefore needed on first run. This project uses pretrained weights; it does not train a new model.

## 5. Project Structure

```text
hackzen-day3-object-detection/
├── detect.py
├── requirements.txt
├── README.md
├── .gitignore
├── images/
│   └── .gitkeep
├── videos/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

Add your own test image to `images/` and your own test video to `videos/`. Generated output files go into `outputs/`. Do not commit large model weights or private recordings.

## 6. Installation (Windows PowerShell)

### Step 1: Open the project folder

```powershell
cd path\to\hackzen-day3-object-detection
```

### Step 2: Create a virtual environment

```powershell
py -m venv .venv
```

### Step 3: Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this only for the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Step 4: Update pip and install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the project

First run can download model weights, so allow time and keep internet connected.

**Webcam:**

```powershell
python detect.py --source 0
```

**Image:**

```powershell
python detect.py --source images/sample.jpg
```

**Video:**

```powershell
python detect.py --source videos/sample.mp4
```

**Change confidence threshold:**

```powershell
python detect.py --source 0 --confidence 0.50
```

Press **Q** in the webcam/video window to stop. For an image, press any key while the image window is active.

## 7. How the Code Works

1. The program reads command-line options, including the input source and confidence threshold.
2. It loads the pretrained YOLO11n model.
3. OpenCV opens the selected webcam or video, or the model reads the image file.
4. YOLO predicts the objects in each image/frame.
5. The model returns object classes, bounding-box coordinates, and confidence scores.
6. `result.plot()` draws the boxes and labels.
7. OpenCV displays the annotated frame and writes the result to the output file.
8. The program releases the camera/video writer and closes its windows.

## 8. Technical Approach / Architecture

```text
Webcam / Image / Video
          |
          v
 OpenCV Input Handling
          |
          v
 YOLO11n Object Detection
          |
          v
 Classes + Bounding Boxes + Confidence
          |
          v
 Annotate Frame and Count Objects
          |
          v
 Display Result + Save Output
```

## 9. Results and Evaluation

Record your actual results after running the program. Do not report estimated FPS or accuracy as measured results.

| Test | Expected observation | Actual result |
|---|---|---|
| Webcam | Boxes and labels appear for visible supported objects | Fill after testing |
| Image | Annotated image saved in `outputs/` | Fill after testing |
| Video | Annotated video saved in `outputs/` | Fill after testing |
| Confidence threshold | Higher threshold usually reduces lower-confidence detections | Fill after testing |
| Performance | Depends on hardware, image size, and input source | Record measured FPS if available |

For a simple speed estimate, time a fixed video clip and calculate:

`FPS = number of processed frames / processing time in seconds`

This is processing throughput, not necessarily the source video's frame rate. Detection quality can be evaluated on a labelled test set using precision, recall, and mAP, but this starter project does not include a labelled evaluation dataset, so it does not claim a measured accuracy score.

## 10. Limitations

- It detects only the classes supported by the pretrained model.
- Small, blurred, hidden, or poorly lit objects may be missed.
- Confidence scores are not guaranteed probabilities of correctness.
- Performance depends on the CPU/GPU, camera resolution, and scene complexity.
- The webcam may not be available if another application is using it.
- First-time model download requires internet access.
- It does not track identities across frames; each frame is detected independently.
- No custom training or domain-specific validation is included.
- Video output codec support may vary by system.

## 11. Future Improvements

- Add object tracking with persistent track IDs.
- Measure FPS and inference latency on different hardware.
- Add a user interface to select images and videos.
- Evaluate precision, recall, and mAP on a labelled dataset.
- Fine-tune the detector for a specific application.
- Export the model to ONNX and benchmark deployment performance.

## 12. Privacy and Responsible Use

Use your own test images/videos or media you have permission to use. Ask for consent before recording people, avoid publishing identifiable footage without permission, and do not use detections as the sole basis for high-impact decisions.

## 13. Demo Video Plan

1. Introduce the project and its real-world purpose.
2. Show the project files and installed libraries.
3. Run `python detect.py --source 0` or use a test video if a webcam is unavailable.
4. Hold up a few everyday objects and show the labels and boxes.
5. Demonstrate an image/video run and show the saved output.
6. Explain the YOLO model and OpenCV pipeline.
7. Mention observed results, limitations, and future improvements.

## 14. Submission Checklist

- [ ] Push the source code and README to a public or reviewer-accessible GitHub repository.
- [ ] Replace all “Fill after testing” cells with genuine observations.
- [ ] Add your own screenshots to `screenshots/` if desired.
- [ ] Record and upload the demo video.
- [ ] Verify the repository and video links open in a private/incognito browser.
- [ ] Submit the GitHub URL, documentation, and demo video link in the HackZen 2.0 form.

## 15. Credits

This project uses the Ultralytics YOLO implementation and pretrained model. Review the applicable Ultralytics licensing terms before redistribution or commercial use. See the official documentation: https://docs.ultralytics.com/
