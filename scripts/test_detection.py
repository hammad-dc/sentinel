"""
Sentinel — Module B Prototype: Person & Vehicle Detection
Takes a surveillance video, detects people and vehicles,
and generates an annotated output video with timestamps.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

# 1. Target surveillance classes (COCO dataset IDs)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Detection confidence threshold (0.0 to 1.0)
# Start at 0.35: surveillance cameras have small objects, so 0.5 is often too strict.
CONFIDENCE_THRESHOLD = 0.20


def run_surveillance_detection(
    input_video_path: str,
    output_video_path: str = "output_detected.mp4",
    sample_stride: int = 6,
) -> None:
    """
    Runs object detection on a video and writes out an annotated video.

    Args:
        input_video_path: Path to the raw surveillance video (VIRAT clip).
        output_video_path: Path where the annotated video will be saved.
        sample_stride: Process every Nth frame (1 = every frame, 2 = every 2nd frame).
    """
    input_path = Path(input_video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input video: {input_path}")

    # Load pretrained YOLOv8 Nano weights
    print("[INFO] Loading detection model...")
    model = YOLO("yolov8s.pt")

    # Open video capture
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {input_path}")

    # Extract source video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"[INFO] Video loaded: {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames}"
    )

    # Initialize video writer to save annotated output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / sample_stride
    out = cv2.VideoWriter(output_video_path, fourcc, out_fps, (width, height))

    frame_index = 0
    processed_count = 0

    print("[INFO] Running detection on VIRAT footage. Press Ctrl+C to abort early...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Video finished

            # Frame sampling: skip frames if stride > 1
            if frame_index % sample_stride != 0:
                frame_index += 1
                continue

            # Read presentation timestamp directly from media container (in ms)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Run inference on single frame
            # classes=list(TARGET_CLASSES.keys()) tells YOLO to ignore cups, dogs, chairs, etc.
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=list(TARGET_CLASSES.keys()),
                imgsz=9600,  # THIS: prevents shrinking small people
                verbose=False,
            )[0]

            detections_in_frame = []

            # Extract coordinates and draw bounding boxes
            for box in results.boxes:
                # Bounding box in pixel coordinates: [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = TARGET_CLASSES.get(cls_id, "unknown")

                detections_in_frame.append(
                    {
                        "class": class_name,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

                # Visual overlay: Color distinction (Blue for Vehicles, Green for People)
                color = (0, 255, 0) if class_name == "person" else (255, 120, 0)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label text: "car: 0.89"
                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Draw timestamp watermark at top-left
            timestamp_str = f"Time: {timestamp_ms / 1000.0:.2f}s | Frame: {frame_index}"
            cv2.putText(
                frame,
                timestamp_str,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # Write annotated frame
            out.write(frame)

            # Log sample detections every 30 processed frames
            if processed_count % 30 == 0:
                print(
                    f"[Frame {frame_index} | {timestamp_ms}ms] Detections: {detections_in_frame}"
                )

            frame_index += 1
            processed_count += 1

    finally:
        cap.release()
        out.release()
        print(
            f"\n[DONE] Processing complete. Output saved to: {Path(output_video_path).resolve()}"
        )


if __name__ == "__main__":
    # Point this to one of your downloaded VIRAT video files
    # Example: "data/VIRAT_S_000001.mp4" or just "sample.mp4"
    INPUT_FILE = "test_main2.mp4"

    # Set sample_stride=1 to process every frame, or 2 to run 2x faster
    run_surveillance_detection(
        input_video_path=INPUT_FILE,
        output_video_path="output_detected2.mp4",
        sample_stride=1,
    )
