
import argparse
import os

import cv2
from ultralytics import YOLO


def detect_objects(source, confidence, output):
    # Load the pretrained YOLO11 small model
    print("Loading YOLO11 model...")
    model = YOLO("yolo11s.pt")

    # Convert webcam number to integer
    if str(source).isdigit():
        source = int(source)

    # Create output folder if needed
    if output:
        folder = os.path.dirname(output)
        if folder:
            os.makedirs(folder, exist_ok=True)

    # Check whether source is a webcam or video
    is_video = isinstance(source, int) or (
        isinstance(source, str)
        and source.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )
    )

    if is_video:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Error: Cannot open webcam or video.")
            return

        writer = None

        if output:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 20.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = cv2.VideoWriter(
                output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            if not writer.isOpened():
                print("Error: Cannot create output video.")
                cap.release()
                return

        print("Detection started. Press Q to quit.")

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Predict objects with a confidence threshold
            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=640,
                iou=0.45,
                verbose=False
            )

            result = results[0]

            # Display detected names and confidence scores
            for box in result.boxes:
                class_id = int(box.cls[0])
                score = float(box.conf[0])
                name = model.names[class_id]

                print(
                    f"Object: {name}, "
                    f"Confidence: {score:.2f}"
                )

            # Draw bounding boxes and labels
            annotated_frame = result.plot()

            cv2.imshow(
                "YOLO11 Object Detection",
                annotated_frame
            )

            if writer is not None:
                writer.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

        if writer is not None:
            writer.release()
            print("Video saved to:", output)

        cv2.destroyAllWindows()

    else:
        # Detect objects in an image
        results = model.predict(
            source=source,
            conf=confidence,
            imgsz=640,
            iou=0.45,
            verbose=False
        )

        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            name = model.names[class_id]

            print(
                f"Object: {name}, "
                f"Confidence: {score:.2f}"
            )

        annotated_image = result.plot()

        if output:
            cv2.imwrite(output, annotated_image)
            print("Image saved to:", output)

        cv2.imshow("YOLO11 Object Detection", annotated_image)
        print("Press any key to close the image.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Detection completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO11 Real-Time Object Detection"
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, image path, or video path"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum confidence threshold (0 to 1)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image or video path"
    )

    args = parser.parse_args()

    if not 0 <= args.confidence <= 1:
        parser.error("Confidence must be between 0 and 1.")

    detect_objects(
        args.source,
        args.confidence,
        args.output
    )
