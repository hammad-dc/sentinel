import json
import hashlib
import subprocess
from pathlib import Path
from pydantic import BaseModel


class VideoMetadata(BaseModel):
    """Container for validated media metadata."""
    duration_ms: int
    fps: float
    width: int
    height: int
    time_base: str


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Computes SHA-256 hash by reading in binary chunks.
    Avoids loading full gigabyte-scale videos into RAM.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def probe_video(file_path: Path) -> VideoMetadata:
    """
    Inspects media file via ffprobe and returns container & stream properties.
    Raises RuntimeError if ffprobe fails or streams are unreadable.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Command breakdown:
    # -v quiet: Suppress banner and noise
    # -print_format json: Output raw machine-readable JSON
    # -show_format: Get container-level metadata (duration, size)
    # -show_streams: Get codec, dimensions, and frame rates per stream
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    
    # Locate the primary video stream (ignoring audio/subtitle streams)
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), 
        None
    )

    if not video_stream:
        raise ValueError("No valid video stream found in file.")

    # 1. Parse Frame Rate (FFprobe returns frame rate as a fraction like '30/1' or '2997/100')
    fps_raw = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, fps_raw.split("/"))
        fps = float(num / den) if den != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    # 2. Parse Duration (Prefer stream-level duration, fallback to format container)
    duration_str = video_stream.get("duration") or data.get("format", {}).get("duration", "0")
    duration_ms = int(float(duration_str) * 1000)

    # 3. Extract Dimensions and Time Base
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    time_base = video_stream.get("time_base", "unknown")

    return VideoMetadata(
        duration_ms=duration_ms,
        fps=round(fps, 2),
        width=width,
        height=height,
        time_base=time_base
    )