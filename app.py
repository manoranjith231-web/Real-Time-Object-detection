import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Object Detection", page_icon="🎯", layout="wide")
st.title("🎯 Real-Time Object Detection Dashboard")
st.write("Detect objects in photos, videos, and live camera feeds.")

with st.sidebar:
    st.header("Detection Settings")
    model_name = st.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], index=1)
    confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    st.caption("The first run may download the selected model.")

@st.cache_resource
def load_model(name):
    return YOLO(name)

try:
    model = load_model(model_name)
except Exception as exc:
    st.error(f"Could not load the YOLO model: {exc}")
    st.stop()

mode = st.radio(
    "Choose input",
    ["Photos", "Videos", "Live webcam detection", "Webcam snapshot", "Android phone camera"],
    horizontal=True,
)

def detect_image(image_bgr):
    result = model.predict(image_bgr, conf=confidence, verbose=False)
    return result[0].plot()

if mode == "Photos":
    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        raw = np.frombuffer(uploaded.getvalue(), np.uint8)
        image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Could not read that image. Try another file.")
        else:
            annotated = detect_image(image_bgr)
            left, right = st.columns(2)
            with left:
                st.subheader("Original photo")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with right:
                st.subheader("Detected objects")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            ok, encoded = cv2.imencode(".jpg", annotated)
            if ok:
                st.download_button("Download detected photo", encoded.tobytes(), "detected_photo.jpg", "image/jpeg")

elif mode == "Videos":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        st.video(uploaded_video.getvalue())
        if st.button("Process video", type="primary"):
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            input_path = None
            output_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_file:
                    input_file.write(uploaded_video.getvalue())
                    input_path = input_file.name
                output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                output_path = output_file.name
                output_file.close()

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    st.error("Could not open the uploaded video.")
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0 or fps > 120:
                        fps = 25.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress = st.progress(0)
                    status = st.empty()
                    frame_count = 0
                    preview = st.empty()
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        annotated = detect_image(frame)
                        writer.write(annotated)
                        frame_count += 1
                        if frame_count % 10 == 0:
                            progress.progress(min(frame_count / max(total, 1), 1.0))
                            status.write(f"Processed {frame_count} frames...")
                            preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Latest processed frame", use_container_width=True)
                    cap.release()
                    writer.release()
                    progress.progress(1.0)
                    status.success(f"Finished processing {frame_count} frames.")
                    with open(output_path, "rb") as result_file:
                        st.download_button("Download detected video", result_file.read(), "detected_video.mp4", "video/mp4")
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
            finally:
                if input_path and os.path.exists(input_path):
                    os.remove(input_path)

elif mode == "Live webcam detection":
    st.subheader("Live webcam object detection")
    st.write("Click **START** and allow camera access in your browser. Objects will be detected continuously.")
    st.warning("If the browser asks for camera permission, choose Allow. This mode uses your computer's browser webcam.")

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
        import av

        current_confidence = confidence
        current_model = model

        class YOLOVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.lock = __import__("threading").Lock()

            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")
                with self.lock:
                    results = current_model.predict(image, conf=current_confidence, verbose=False)
                    annotated = results[0].plot()
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_streamer(
            key="live-yolo-webcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    except ImportError:
        st.error("Live webcam packages are not installed yet. In the VS Code terminal, run: python -m pip install streamlit-webrtc av")
        st.info("After installation, stop and restart Streamlit, then choose Live webcam detection again.")
    except Exception as exc:
        st.error(f"Could not start the live webcam: {exc}")
        st.info("Check browser camera permission and ensure no other app is using the camera.")

elif mode == "Webcam snapshot":
    st.info("This option takes one photo from your browser webcam.")
    picture = st.camera_input("Take a webcam photo")
    if picture:
        raw = np.frombuffer(picture.getvalue(), np.uint8)
        image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            annotated = detect_image(image_bgr)
            left, right = st.columns(2)
            with left:
                st.subheader("Webcam photo")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with right:
                st.subheader("Detected objects")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            ok, encoded = cv2.imencode(".jpg", annotated)
            if ok:
                st.download_button("Download detected photo", encoded.tobytes(), "webcam_detection.jpg", "image/jpeg")

elif mode == "Android phone camera":
    st.info("Your phone and computer must be on the same Wi-Fi. Start a camera-stream server app on Android, then enter its video-stream URL.")
    phone_url = st.text_input("Phone camera stream URL", placeholder="http://192.168.1.5:8080/video")
    st.caption("This option captures and detects one frame from the phone stream each time you click the button.")
    if st.button("Capture phone frame", type="primary"):
        if not phone_url.strip().lower().startswith(("http://", "https://", "rtsp://")):
            st.error("Enter a valid URL beginning with http://, https://, or rtsp://.")
        else:
            cap = cv2.VideoCapture(phone_url.strip())
            if not cap.isOpened():
                st.error("Could not connect. Check the URL, Wi-Fi, and that the phone camera server is running.")
            else:
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    st.error("Connected, but no frame was received. Check the stream URL.")
                else:
                    annotated = detect_image(frame)
                    left, right = st.columns(2)
                    with left:
                        st.subheader("Phone camera frame")
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with right:
                        st.subheader("Detected objects")
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    ok, encoded = cv2.imencode(".jpg", annotated)
                    if ok:
                        st.download_button("Download detected frame", encoded.tobytes(), "phone_detection.jpg", "image/jpeg")

st.divider()
st.caption("YOLO detects classes learned by its pretrained model. It may not recognize every object or always assign the correct label.")
