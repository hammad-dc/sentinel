import uuid
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import get_db
from app.core.config import settings
from app.models.media import Video, Camera
from app.services.video_service import compute_sha256, probe_video

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    camera_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Receives surveillance video, streams it to disk, computes SHA-256,
    extracts metadata via ffprobe, and persists the record to the database.
    """
    parsed_camera_id = None

    if camera_id:
        try:
          parsed_camera_id = uuid.UUID(camera_id)
        except ValueError:
          raise HTTPException(
              status_code=status.HTTP_400_BAD_REQUEST,
              detail="Invalid camera ID format."
          )

        camera = db.query(Camera).filter(
          Camera.id == parsed_camera_id
        ).first()

        if camera is None:
          raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found."
          )

    # Reject non-video extensions early
    allowed_extensions = {".mp4", ".mkv", ".avi", ".mov"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format {suffix}. Allowed: {allowed_extensions}"
        )

    video_id = uuid.uuid4()
    saved_filename = f"{video_id}{suffix}"
    destination_path = settings.ORIGINALS_DIR / saved_filename

    # Stream file to storage in 64 KB chunks to prevent high memory usage
    try:
        with open(destination_path, "wb") as buffer:
            while chunk := await file.read(65536):
                buffer.write(chunk)
    except Exception as e:
        if destination_path.exists():
            destination_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write file: {str(e)}"
        )

    # Inspect media and calculate cryptographic signature
    try:
        sha256_hash = compute_sha256(destination_path)
        metadata = probe_video(destination_path)
    except Exception as e:
        # If extraction fails, remove file so unreadable files don't linger
        if destination_path.exists():
            destination_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid or corrupted video file: {str(e)}"
        )

    # Save to PostgreSQL
    video_record = Video(
        id=video_id,
        camera_id=parsed_camera_id,
        filename=file.filename,
        storage_path=str(destination_path),
        sha256=sha256_hash,
        duration_ms=metadata.duration_ms,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        time_base=metadata.time_base,
        status="UPLOADED"
    )

    db.add(video_record)

    try:
      db.commit()
    except SQLAlchemyError:
      db.rollback()

      if destination_path.exists():
        destination_path.unlink()

      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to save video information to database."
      )

    db.refresh(video_record)

    return {
        "message": "Video successfully ingested and verified.",
        "video": {
            "id": str(video_record.id),
            "filename": video_record.filename,
            "duration_ms": video_record.duration_ms,
            "fps": video_record.fps,
            "resolution": f"{video_record.width}x{video_record.height}",
            "sha256": video_record.sha256,
            "status": video_record.status
        }
    }