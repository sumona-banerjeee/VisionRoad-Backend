"""
YoloDetector — Pure YOLO detection with optional verification callback.

The detector focuses solely on YOLO inference, tracking, and deduplication.
An optional `verify_fn` callback can be provided to add post-detection
verification (e.g., VL verification). The callback is invoked asynchronously
during frame processing and its results are used to confirm, override,
or reject detections.

This module is the slim orchestrator — the heavy lifting lives in:
  - config.py          : constants & env vars
  - tracker_state.py   : tracking data structures & state management
  - verify_handler.py  : async verification futures lifecycle
  - frame_processor.py : per-frame detection loop
  - result_builder.py  : final results dict, GPS enrichment, DB save
  - perf_reporter.py   : perf timing accumulation & report formatting
"""

import cv2
import json
import asyncio
import time
import logging
import torch
import os
import tempfile

from app.detectors.base.base_detector import BaseDetector
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.core.logging_config import PerfTimer

from app.detectors.yolo.config import (
    MODEL_PATH,
    TRACKER,
    CONF_THRESHOLD,
    FRAME_SKIP,
    YOLO_IMGSZ,
    USE_HALF,
    ROAD_DAMAGE_CLASSES,
)
from app.detectors.yolo.tracker_state import TrackerState
from app.detectors.yolo import verify_handler
from app.detectors.yolo import frame_processor
from app.detectors.yolo import result_builder
from app.detectors.yolo import perf_reporter

# Re-export for backward compatibility (app/__init__.py imports this)
from app.detectors.yolo.verify_handler import get_verify_executor  # noqa: F401

logger = logging.getLogger(__name__)


class YoloDetector(BaseDetector):
    """
    Pure YOLO detector with optional verification callback.

    Args:
        verify_fn: Optional callable(frame, bbox, predicted_class) -> dict | None.
                   If provided, it is called asynchronously to verify each detection.
        detection_mode: String label for the detection mode (e.g. "yolo", "yolo_vl", "sam3").
    """

    def __init__(self, verify_fn=None, detection_mode="yolo"):
        """Initialize detector with YOLO model and optional verification callback."""
        super().__init__(model_path=MODEL_PATH)
        self.verify_fn = verify_fn
        self.detection_mode = detection_mode

        mode_label = (
            f"YOLO+verify ({self.detection_mode})" if self.verify_fn else "YOLO-only"
        )
        logger.info(f"YoloDetector ready — mode: {mode_label}")

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        has_verify = self.verify_fn is not None
        cap = None
        _ndjson_fd = None
        _ndjson_path = None
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id, {"type": "status", "status": "processing", "progress": 0}
                ),
                loop,
            )

            gps_points = self._load_gps_data(json_path)
            process_start = time.time()
            perf_timings = perf_reporter.create_perf_timings()

            # ── Open video ───────────────────────────────────────────────
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = total_frames / fps

            logger.info(
                f"Processing [{self.detection_mode}] for {video_id}: {total_frames} frames @ {fps:.1f} FPS"
            )
            logger.info(
                f"Performance settings: FRAME_SKIP={FRAME_SKIP}, YOLO_IMGSZ={YOLO_IMGSZ}, "
                f"FP16={'ON' if USE_HALF else 'OFF'}, VERIFY={'ON' if has_verify else 'OFF'}"
            )

            # ── Adaptive parameters ──────────────────────────────────────
            DETECTION_TIME_WINDOW = video_duration * 0.25
            TIME_THRESHOLD = video_duration * 0.30
            HIGH_CONFIDENCE_THRESHOLD = 0.75
            LOW_CONFIDENCE_MIN_FRAMES = 2
            MIN_DISTANCE_THRESHOLD = 120

            ROI_LEFT = 0
            ROI_RIGHT = width
            ROI_TOP = int(height * 0.05)
            ROI_BOTTOM = int(height * 0.95)

            # ── State ────────────────────────────────────────────────────
            state = TrackerState(
                time_threshold=TIME_THRESHOLD,
                has_verify=has_verify,
            )

            # ── NDJSON streaming ─────────────────────────────────────────
            _ndjson_fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".ndjson",
                prefix=f"frames_{video_id}_",
                dir=str(RESULTS_DIR),
                delete=False,
            )
            _ndjson_path = _ndjson_fd.name
            _frames_written = 0
            logger.info(f"NDJSON streaming enabled — temp file: {_ndjson_path}")
            total_detections_count = 0
            frame_count = 0
            last_progress = 0

            # ── Main loop ────────────────────────────────────────────────
            while cap.isOpened():
                _t0_read = time.perf_counter()
                ret, frame = cap.read()
                _read_elapsed = time.perf_counter() - _t0_read
                if not ret:
                    break
                perf_reporter.record(perf_timings, "frame_decode", _read_elapsed)

                frame_count += 1
                if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                    continue

                current_time = frame_count / fps

                with PerfTimer("YOLO inference", video_id) as _t_yolo:
                    results = self.model.track(
                        frame,
                        persist=True,
                        conf=CONF_THRESHOLD,
                        tracker=TRACKER,
                        verbose=False,
                        device=self.device,
                        imgsz=YOLO_IMGSZ,
                        half=USE_HALF,
                    )
                perf_reporter.record(perf_timings, "yolo_inference", _t_yolo.elapsed)

                # ── Per-frame processing ─────────────────────────────────
                frame_data, det_delta = frame_processor.process_frame(
                    results,
                    self.model,
                    frame,
                    frame_count,
                    current_time,
                    state,
                    roi_left=ROI_LEFT,
                    roi_right=ROI_RIGHT,
                    roi_top=ROI_TOP,
                    roi_bottom=ROI_BOTTOM,
                    detection_time_window=DETECTION_TIME_WINDOW,
                    high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
                    low_confidence_min_frames=LOW_CONFIDENCE_MIN_FRAMES,
                    min_distance_threshold=MIN_DISTANCE_THRESHOLD,
                    has_verify=has_verify,
                    verify_fn=self.verify_fn,
                    perf_timings=perf_timings,
                )
                total_detections_count += det_delta

                if frame_data["detections"]:
                    _ndjson_fd.write(json.dumps(frame_data) + "\n")
                    _frames_written += 1
                    if _frames_written % 50 == 0:
                        logger.info(
                            f" NDJSON streamed {_frames_written} frames to disk so far"
                        )

                # ── Progress ─────────────────────────────────────────────
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id,
                        progress,
                        loop,
                        {
                            "unique_pothole": len(state.counted_ids["pothole"]),
                            "unique_defected_sign_board": len(
                                state.counted_ids["defected_sign_board"]
                            ),
                            "unique_road_crack": len(state.counted_ids["road_crack"]),
                            "unique_damaged_road_marking": len(
                                state.counted_ids["damaged_road_marking"]
                            ),
                            "unique_good_sign_board": len(
                                state.counted_ids["good_sign_board"]
                            ),
                            "total_road_damage": sum(
                                len(state.counted_ids[c]) for c in ROAD_DAMAGE_CLASSES
                            ),
                            "total_detections": total_detections_count,
                        },
                    )
                    last_progress = progress
                    state.evict_stale_trackers(current_time)

            yolo_end = time.time()

            # ── Drain remaining verify futures ───────────────────────────
            if has_verify:
                verify_handler.drain_pending(state, perf_timings)
            vl_drain_end = time.time()

            # ── Build results ────────────────────────────────────────────
            video_info = {
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration": round(video_duration, 2),
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
            }

            results_dict, all_detections_flat = result_builder.build_results_dict(
                video_id=video_id,
                video_path=video_path,
                detection_mode=self.detection_mode,
                tracker_state=state,
                gps_points=gps_points,
                perf_timings=perf_timings,
                video_info=video_info,
                total_detections_count=total_detections_count,
                frames_written=_frames_written,
                frame_count=frame_count,
                ndjson_fd=_ndjson_fd,
                ndjson_path=_ndjson_path,
                has_verify=has_verify,
            )

            detection_results[video_id] = results_dict
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results_dict, f, indent=2)

            result_builder.save_to_db(video_id, all_detections_flat, perf_timings)
            process_end = time.time()

            # ── Perf report ──────────────────────────────────────────────
            total_time = process_end - process_start
            frames_processed = (
                frame_count // FRAME_SKIP if FRAME_SKIP > 1 else frame_count
            )
            perf_reporter.generate_report(
                perf_timings=perf_timings,
                video_id=video_id,
                detection_mode=self.detection_mode,
                total_time=total_time,
                yolo_time=yolo_end - process_start,
                drain_time=vl_drain_end - yolo_end,
                video_duration=video_duration,
                total_frames=total_frames,
                fps=fps,
                frames_processed=frames_processed,
                frame_skip=FRAME_SKIP,
                detections_saved=len(all_detections_flat),
                total_verified=state.verify_stats["total_verified"],
            )

            processing_status[video_id] = {"status": "completed", "progress": 100}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id,
                    {
                        "type": "complete",
                        "status": "completed",
                        "summary": results_dict["summary"],
                        "rejection_stats": results_dict["rejection_stats"],
                    },
                ),
                loop,
            )
            return results_dict

        except Exception as e:
            logger.error(f"Error processing {video_id}: {e}")
            processing_status[video_id] = {"status": "error", "message": str(e)}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(video_id, {"type": "error", "message": str(e)}),
                loop,
            )
            raise
        finally:
            if cap:
                cap.release()
            try:
                if _ndjson_fd is not None and not _ndjson_fd.closed:
                    _ndjson_fd.close()
                if _ndjson_path is not None and os.path.exists(_ndjson_path):
                    os.remove(_ndjson_path)
                    logger.info(f"NDJSON temp file cleaned up: {_ndjson_path}")
            except OSError:
                pass
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
