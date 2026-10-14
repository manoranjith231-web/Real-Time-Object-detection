# Demo Video Script (about 2–3 minutes)

## 1. Introduction (15–20 seconds)
"Hello everyone. This is my HackZen 2.0 Day 3 Computer Vision project, Real-Time Object Detection using Python, OpenCV, and YOLO11n. The goal is to identify common objects in images and video and highlight them automatically."

## 2. Problem and solution (20 seconds)
"Manually finding objects in visual media can take time. My solution uses a pretrained YOLO object-detection model. It draws bounding boxes, displays object names and confidence scores, and counts detected objects in each frame."

## 3. Show the code and structure (20–30 seconds)
"Here is the project structure. The main program is detect.py, requirements.txt lists the dependencies, and README.md explains installation and usage. OpenCV handles the camera or video input, while YOLO performs object detection."

## 4. Live demonstration (45–60 seconds)
"Now I will run the application. The first run may download the pretrained model. The program is processing the input and drawing boxes around objects it recognizes. The label shows the predicted class, and the confidence score indicates the model's confidence for that detection. The object count is shown on the frame."

Run:
`python detect.py --source 0`

Hold up a bottle, book, phone, or other everyday objects. If the webcam is unavailable, use a video:
`python detect.py --source videos/sample.mp4`

## 5. Show saved results (15–20 seconds)
"The annotated output is saved in the outputs folder, so the result can be reviewed after processing."

## 6. Limitations and conclusion (20–30 seconds)
"The model can miss small, blurred, hidden, or poorly lit objects, and performance depends on the computer. This version detects each frame independently and does not track identities across frames. In future, I would add tracking and evaluate the model on a labelled dataset. Thank you."

## Recording checklist
- Test the commands before recording.
- Close unrelated windows and notifications.
- Make sure labels are readable in the recording.
- Show the actual output; do not edit in fake detections or unmeasured performance figures.
- Upload the video and test the share permissions before submitting.
