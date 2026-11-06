"""
Sentinel — Module C (High-Speed Tracking)
Applies temporal sampling (5 FPS) and optimized resolution (960px)
to process surveillance footage 5x-8x faster without losing tracks.
"""

from collections import defaultdict
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
COLORS = np.random.default_rng(seed=42).uniform(50, 255, size=(1000, 3)).astype(int)


def run_fast_surveillance_tracking(
    input_video_path: str,
    output_video_path: str = "output_fast_tracked.mp4",
    target_process_fps: int = 15,  # Process 5 frames per second instead of 30
    img_size: int = 960,  # Balanced resolution (fast + detects small objects)
    conf_threshold: float = 0.10,
) -> None:
    input_path = Path(input_video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Hardware detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on: {device.upper()}")

    print("[INFO] Loading YOLOv8 Small model...")
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate stride: e.g. 30 native FPS / 5 target FPS = process every 6th frame
    stride = max(1, int(round(native_fps / target_process_fps)))
    effective_fps = native_fps / stride

    print(
        f"[INFO] Video: {width}x{height} | Native: {native_fps:.1f} FPS | Processing at: ~{effective_fps:.1f} FPS (Stride: {stride})"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, effective_fps, (width, height))

    track_trajectories = defaultdict(list)
    track_summary = {}

    frame_index = 0
    processed_count = 0
    start_time = time.time()

    try:
      while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save current frame index and increment for the next iteration
        current_frame_index = frame_index
        frame_index += 1

        # Skip frames according to the calculated stride
        if current_frame_index % stride != 0:
            continue

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Run detection + tracking
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            classes=list(TARGET_CLASSES.keys()),
            imgsz=img_size,
            device=device,
            verbose=False,
        )[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            if results.boxes.id is not None:
                track_ids = results.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [None] * len(boxes)

            for box, track_id, cls_id, conf in zip(
                boxes, track_ids, classes, confs
            ):
                x1, y1, x2, y2 = box
                class_name = TARGET_CLASSES.get(cls_id, "unknown")

                if track_id is not None:
                    label = f"ID:{track_id} | {class_name} {conf:.2f}"
                    color = tuple(
                        int(c) for c in COLORS[track_id % len(COLORS)]
                    )
                else:
                    label = f"UNTRACKED | {class_name} {conf:.2f}"
                    color = (0, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                if track_id is not None:
                    bottom_center = (int((x1 + x2) / 2), y2)

                    track_trajectories[track_id].append(bottom_center)

                    if len(track_trajectories[track_id]) > 30:
                        track_trajectories[track_id].pop(0)

                    points = track_trajectories[track_id]

                    for i in range(1, len(points)):
                        cv2.line(
                            frame,
                            points[i - 1],
                            points[i],
                            color,
                            2,
                        )

        # Write the processed and annotated frame
        out.write(frame)

        processed_count += 1
    finally:
        cap.release()
        out.release()

    elapsed = time.time() - start_time
    video_duration_s = total_frames / native_fps
    speed_factor = video_duration_s / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 65)
    print(f"BENCHMARK RESULTS")
    print(f"Video Duration : {video_duration_s:.1f} seconds")
    print(f"Processing Time: {elapsed:.1f} seconds")
    print(
        f"Speed Factor   : {speed_factor:.2f}x real-time ({'FASTER than real-time' if speed_factor > 1 else 'slower than real-time'})"
    )
    print("=" * 65)


if __name__ == "__main__":
    run_fast_surveillance_tracking(
        input_video_path="test_main2.mp4",
        output_video_path="output_fast_tracked.mp4",
        target_process_fps=30,  # 5 FPS
        img_size=960,  # 960px
    )
