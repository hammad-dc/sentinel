from pathlib import Path
from app.services.video_service import compute_sha256, probe_video

video_path = Path("test_big1.mpg")

# Test 1: Calculate SHA-256
print("\n--- SHA-256 TEST ---")

file_hash = compute_sha256(video_path)
print("SHA-256:", file_hash)

# Test 2: Extract video metadata
print("\n--- VIDEO METADATA TEST ---")

metadata = probe_video(video_path)

print("Duration:", metadata.duration_ms, "ms")
print("FPS:", metadata.fps)
print("Width:", metadata.width)
print("Height:", metadata.height)
print("Time Base:", metadata.time_base)
