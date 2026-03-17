"""
YoloDetector — Pure YOLO detection with optional verification callback.

The detector focuses solely on YOLO inference, tracking, and deduplication.
An optional `verify_fn` callback can be provided to add post-detection
verification (e.g., VL verification). The callback is invoked asynchronously
during frame processing and its results are used to confirm, override,
or reject detections.

This module is the slim orchestrator — all heavy logic lives in sibling modules:
  config.py, filters.py, tracker_state.py, verification.py,
  frame_io.py, result_builder.py, db_writer.py
"""

import cv2
import json
import asyncio
import time
import logging
import torch
import os

from app.detectors.base.base_detector import BaseDetector
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.core.logging_config import PerfTimer

# ── Sibling module imports ────────────────────────────────────────────────────
from app.detectors.yolo.config import (
    MODEL_PATH, TRACKER, CONF_THRESHOLD, FRAME_SKIP, YOLO_IMGSZ,
    ROAD_DAMAGE_CLASSES, ALL_CLASSES,
    VL_TIMEOUT, get_verify_executor, get_adaptive_params,
    _async_verify_executor,
)
from app.detectors.yolo.filters import is_inside_roi, is_duplicate_location
from app.detectors.yolo.tracker_state import TrackerState
from app.detectors.yolo.verification import (
    submit_verification, process_completed_futures, drain_pending,
)
from app.detectors.yolo.frame_io import (
    create_ndjson_file, filter_and_write_frames, read_ndjson_frames,
    cleanup_ndjson,
)
from app.detectors.yolo.result_builder import (
    build_class_list, build_result_dict, format_perf_report,
)
from app.detectors.yolo.db_writer import save_detections_to_db

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

    # ── Backward-compatible static methods (used by subclasses) ───────────────

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        from app.detectors.yolo.filters import calculate_distance
        return calculate_distance(p1, p2)

    @staticmethod
    def calculate_ios(box1, box2):
        """Calculate Intersection over Smaller Box (IoS)."""
        from app.detectors.yolo.filters import calculate_ios
        return calculate_ios(box1, box2)

    def is_duplicate_location(
        self, cx, cy, bbox, class_name, current_time,
        spatial_locations, time_threshold, min_distance_threshold,
    ):
        """Check if this location/class was already counted recently."""
        return is_duplicate_location(
            cx, cy, bbox, class_name, current_time,
            spatial_locations, time_threshold, min_distance_threshold,
        )

    # ── Main processing loop ──────────────────────────────────────────────────

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

            # Performance accumulators
            perf_timings = {
                "frame_decode": {"total": 0.0, "count": 0},
                "yolo_inference": {"total": 0.0, "count": 0},
                "gps_coord": {"total": 0.0, "count": 0},
                "verification": {"total": 0.0, "count": 0},
                "db_gps_match": {"total": 0.0, "count": 0},
                "db_bulk_write": {"total": 0.0, "count": 0},
            }

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = total_frames / fps

            logger.info(
                f"Processing [{self.detection_mode}] for {video_id}: "
                f"{total_frames} frames @ {fps:.1f} FPS"
            )
            logger.info(
                f"Performance settings: FRAME_SKIP={FRAME_SKIP}, YOLO_IMGSZ={YOLO_IMGSZ}, "
                f"VERIFY={'ON' if has_verify else 'OFF'}"
            )

            # Adaptive parameters from config
            params = get_adaptive_params(video_duration, width, height)
            roi = params["roi"]

            # Tracking state
            state = TrackerState(has_verify=has_verify)

            # NDJSON for frame streaming
            _ndjson_fd, _ndjson_path = create_ndjson_file(video_id)
            _pending_frames = []

            frame_count = 0
            last_progress = 0

            # ── Frame loop ────────────────────────────────────────────────────
            while cap.isOpened():
                _t0_read = time.perf_counter()
                ret, frame = cap.read()
                _read_elapsed = time.perf_counter() - _t0_read
                if not ret:
                    break
                perf_timings["frame_decode"]["total"] += _read_elapsed
                perf_timings["frame_decode"]["count"] += 1

                frame_count += 1

                # Frame skipping — road defects persist across frames
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
                    )
                perf_timings["yolo_inference"]["total"] += _t_yolo.elapsed
                perf_timings["yolo_inference"]["count"] += 1

                frame_data = {"frame_id": frame_count, "detections": []}

                # Process any completed verify futures from previous frames
                if has_verify:
                    process_completed_futures(state, perf_timings)

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for tid, cid, box, conf in zip(
                        track_ids, class_ids, boxes, confidences
                    ):
                        tid, cid = int(tid), int(cid)
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        class_name = str(self.model.names[cid])

                        # ── Filter: ROI ───────────────────────────────────────
                        if not is_inside_roi(cx, cy, roi):
                            state.rejection_stats["roi_outside"] += 1
                            continue

                        # ── Filter: Class lock ────────────────────────────────
                        if tid in state.tracker_class_lock:
                            if state.tracker_class_lock[tid] != class_name:
                                state.rejection_stats["class_mismatch"] += 1
                                continue
                        else:
                            state.tracker_class_lock[tid] = class_name

                        # ── Tracker history ───────────────────────────────────
                        state.tracker_history[tid].append(current_time)
                        state._tid_last_seen[tid] = current_time
                        recent = [
                            t
                            for t in state.tracker_history[tid]
                            if current_time - t <= params["detection_time_window"]
                        ]
                        min_needed = (
                            1
                            if conf >= params["high_confidence_threshold"]
                            else params["low_confidence_min_frames"]
                        )

                        # ── Confirmation gate ─────────────────────────────────
                        if (
                            len(recent) >= min_needed
                            and tid not in state.confirmed
                            and tid not in state.rejected_tids
                        ):
                            is_dup, _ = self.is_duplicate_location(
                                cx, cy, (x1, y1, x2, y2), class_name,
                                current_time, state.spatial_locations,
                                params["time_threshold"],
                                params["min_distance_threshold"],
                            )
                            if not is_dup:
                                state.confirm_detection(
                                    tid, class_name, frame_count, current_time,
                                    conf, x1, y1, x2, y2, cx, cy,
                                )
                                # Submit async verification
                                if has_verify:
                                    submit_verification(
                                        _async_verify_executor,
                                        self.verify_fn,
                                        tid, frame, (x1, y1, x2, y2),
                                        class_name, state,
                                    )
                            else:
                                state.rejection_stats["spatial_duplicate"] += 1
                        elif tid not in state.confirmed:
                            state.rejection_stats["multi_frame_pending"].add(tid)

                        # ── Append to frame data if confirmed ─────────────────
                        if tid in state.confirmed:
                            counts = state.get_counts_snapshot()
                            frame_data["detections"].append(
                                {
                                    "frame_id": frame_count,
                                    "detection_id": tid,
                                    "type": class_name,
                                    "confidence": round(float(conf), 3),
                                    "count": counts,
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "center": {"x": cx, "y": cy},
                                    "area": (x2 - x1) * (y2 - y1),
                                }
                            )

                # Buffer frame data (written after post-drain filter)
                if frame_data["detections"]:
                    _pending_frames.append(frame_data)

                # Progress reporting
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    counts = state.get_counts_snapshot()
                    self._send_progress(
                        video_id, progress, loop,
                        {
                            **{f"unique_{k}": v for k, v in counts.items()},
                            "total_road_damage": sum(
                                counts.get(c, 0) for c in ROAD_DAMAGE_CLASSES
                            ),
                            "total_detections": sum(
                                len(f["detections"]) for f in _pending_frames
                            ),
                        },
                    )
                    last_progress = progress
                    state.evict_stale(current_time, params["time_threshold"])

            yolo_end = time.time()

            # ── Drain remaining verify futures ────────────────────────────────
            if has_verify:
                drain_pending(state, perf_timings, VL_TIMEOUT)

            vl_drain_end = time.time()

            # ── Post-drain NDJSON filter ──────────────────────────────────────
            confirmed_ids = set(state.confirmed.keys())
            frames_written, total_detections_count = filter_and_write_frames(
                _pending_frames, confirmed_ids, _ndjson_fd,
            )
            del _pending_frames  # release memory

            # ── Build final results ───────────────────────────────────────────
            class_names = [
                "defected_sign_board", "pothole", "road_crack",
                "damaged_road_marking", "good_sign_board",
            ]
            class_lists = {
                cn: build_class_list(state.confirmed, cn, gps_points, perf_timings)
                for cn in class_names
            }

            video_info = {
                "total_frames": frame_count,
                "total_frames_raw": total_frames,
                "fps": fps,
                "duration": video_duration,
                "width": width,
                "height": height,
            }

            frames = read_ndjson_frames(_ndjson_fd, _ndjson_path, frames_written)

            results = build_result_dict(
                video_id=video_id,
                video_path=video_path,
                detection_mode=self.detection_mode,
                video_info=video_info,
                state=state,
                class_lists=class_lists,
                frames=frames,
                total_detections_count=total_detections_count,
                frames_with_detections=frames_written,
                has_verify=has_verify,
            )

            # Save results to disk
            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            # Save to DB
            all_detections_flat = []
            for cn in class_names:
                all_detections_flat.extend(class_lists[cn])
            save_detections_to_db(video_id, all_detections_flat, perf_timings)
            process_end = time.time()

            # ── Perf report ───────────────────────────────────────────────────
            format_perf_report(
                video_id=video_id,
                detection_mode=self.detection_mode,
                perf_timings=perf_timings,
                video_info=video_info,
                verify_stats=state.verify_stats,
                total_time=process_end - process_start,
                yolo_time=yolo_end - process_start,
                drain_time=vl_drain_end - yolo_end,
                num_detections=len(all_detections_flat),
            )

            processing_status[video_id] = {"status": "completed", "progress": 100}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id,
                    {
                        "type": "complete",
                        "status": "completed",
                        "summary": results["summary"],
                        "rejection_stats": results["rejection_stats"],
                    },
                ),
                loop,
            )
            return results

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
            cleanup_ndjson(_ndjson_fd, _ndjson_path)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
