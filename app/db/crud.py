"""CRUD operations for database models"""

from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional, List
from datetime import datetime
from app.models.video import Video
from app.models.detection import Detection
from app.models.processing import ProcessingStatus, ProcessingStatusEnum


# ==================== Video CRUD ====================


def create_video(
    db: Session,
    video_id: str,
    filename: str,
    original_path: str,
    json_file_path: str,
    detection_type: str,
    speed_kmh: int,
    chainage_id: Optional[str] = None,
    lane_id: Optional[str] = None,
) -> Video:
    """Create a new video record"""
    video = Video(
        id=video_id,
        filename=filename,
        original_path=original_path,
        json_file_path=json_file_path,
        detection_type=detection_type,
        speed_kmh=speed_kmh,
        chainage_id=chainage_id,
        lane_id=lane_id,
        uploaded_at=datetime.utcnow(),
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def get_video(db: Session, video_id: str) -> Optional[Video]:
    """Get a video by ID"""
    return db.query(Video).filter(Video.id == video_id).first()


def list_videos(db: Session, skip: int = 0, limit: int = 100) -> List[Video]:
    """Get all videos ordered by upload time (newest first)"""
    return (
        db.query(Video)
        .order_by(desc(Video.uploaded_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_video_processed_path(
    db: Session, video_id: str, processed_path: str
) -> Optional[Video]:
    """Update the processed video path"""
    video = get_video(db, video_id)
    if video:
        video.processed_video_path = processed_path
        db.commit()
        db.refresh(video)
    return video


def delete_video(db: Session, video_id: str) -> bool:
    """Delete a video and all related records (cascades to detections and status)"""
    video = get_video(db, video_id)
    if video:
        db.delete(video)
        db.commit()
        return True
    return False


# ==================== Detection CRUD ====================


def create_detection(
    db: Session,
    video_id: str,
    frame_number: int,
    timestamp_ms: int,
    confidence: float,
    detection_type: str,
    class_name: str,
    bounding_box: dict,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    project_id: Optional[str] = None,
    package_id: Optional[str] = None,
    chainage_id: Optional[str] = None,
    lane_id: Optional[str] = None,
) -> Detection:
    """Create a new detection record"""
    detection = Detection(
        video_id=video_id,
        frame_number=frame_number,
        timestamp_ms=timestamp_ms,
        latitude=latitude,
        longitude=longitude,
        confidence=confidence,
        detection_type=detection_type,
        class_name=class_name,
        bounding_box="",  # Will be set by set_bounding_box
        project_id=project_id,
        package_id=package_id,
        chainage_id=chainage_id,
        lane_id=lane_id,
    )
    detection.set_bounding_box(bounding_box)
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def create_detections_bulk(db: Session, detections: List[Detection]) -> None:
    """Bulk insert detections for better performance"""
    db.add_all(detections)
    db.commit()


def get_detections_by_video(
    db: Session, video_id: str, skip: int = 0, limit: int = 1000
) -> List[Detection]:
    """Get all detections for a video"""
    return (
        db.query(Detection)
        .filter(Detection.video_id == video_id)
        .order_by(Detection.frame_number)
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_detection_count(db: Session, video_id: str) -> int:
    """Get total number of detections for a video"""
    return db.query(Detection).filter(Detection.video_id == video_id).count()


# ==================== Processing Status CRUD ====================


def create_processing_status(
    db: Session,
    video_id: str,
    status: ProcessingStatusEnum = ProcessingStatusEnum.PENDING,
) -> ProcessingStatus:
    """Create a new processing status record"""
    processing_status = ProcessingStatus(video_id=video_id, status=status, progress=0)
    db.add(processing_status)
    db.commit()
    db.refresh(processing_status)
    return processing_status


def get_processing_status(db: Session, video_id: str) -> Optional[ProcessingStatus]:
    """Get processing status for a video"""
    return (
        db.query(ProcessingStatus).filter(ProcessingStatus.video_id == video_id).first()
    )


def update_processing_status(
    db: Session,
    video_id: str,
    status: Optional[ProcessingStatusEnum] = None,
    progress: Optional[int] = None,
    current_frame: Optional[int] = None,
    total_frames: Optional[int] = None,
    error_message: Optional[str] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> Optional[ProcessingStatus]:
    """Update processing status for a video"""
    proc_status = get_processing_status(db, video_id)
    if proc_status:
        if status is not None:
            proc_status.status = status
        if progress is not None:
            proc_status.progress = progress
        if current_frame is not None:
            proc_status.current_frame = current_frame
        if total_frames is not None:
            proc_status.total_frames = total_frames
        if error_message is not None:
            proc_status.error_message = error_message
        if started_at is not None:
            proc_status.started_at = started_at
        if completed_at is not None:
            proc_status.completed_at = completed_at

        db.commit()
        db.refresh(proc_status)
    return proc_status


def get_all_processing_statuses(db: Session) -> List[ProcessingStatus]:
    """Get all processing statuses"""
    return db.query(ProcessingStatus).all()
