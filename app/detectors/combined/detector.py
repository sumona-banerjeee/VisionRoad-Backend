"""
CombinedDetector — Single-pass dual-model detector.

Runs two YOLO models over the same video in one pass:
  - yoloe_trained_vl  (best-11m.pt)   → all classes EXCEPT drain_issue
  - yolo              (all-with-drain.pt) → drain_issue ONLY

Both models write into shared tracking structures to produce one
merged JSON result containing all road-damage classes.

Tracker ID isolation:
  YOLOE model IDs are stored as-is (1, 2, 3, ...).
  YOLO drain model IDs are offset by DRAIN_TID_OFFSET (100_000) so they
  never collide with YOLOE IDs in the same confirmed{} dict.
"""

import cv2
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
import time
import logging
import torch
import os
import tempfile
from datetime import datetime
from collections import defaultdict, deque

from ultralytics import YOLO
from app.detectors.yoloe_trained_vl.detector import YoloeTrainedVlDetector
from app.detectors.yolo.detector import CONF_THRESHOLD as YOLO_CONF_THRESHOLD
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.core.logging_config import PerfTimer, perf_logger
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.models.video import Video
from app.db.crud_hierarchy import find_chainage_by_gps
from app.detectors.yolo.pole_tilt import analyze_pole_tilt
from app.helpers.yoloe_vl_trained_helper import (
    YOLOE_TRAINED_CONF,
    YOLOE_TRAINED_MODEL_PATH,
    process_with_trained_vl,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# YOLO drain model (re-uses the same path/conf as yolo/detector.py)
YOLO_DRAIN_MODEL_PATH = r"models\all-with-drain.pt"

# Only this class is accepted from the YOLO drain model
DRAIN_CLASS = "drain_issue"

# Offset applied to YOLO drain tracker IDs to prevent collision with YOLOE IDs
DRAIN_TID_OFFSET = 100_000

TRACKER = "botsort.yaml"
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
MAX_VERIFY_CONCURRENT = int(os.getenv("MAX_VL_CONCURRENT", "4"))

# Road damage classes for the combined output (no culvert, no drain excluded)
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
    "drain_issue",
}
EXCLUDED_CLASSES = {"good_sign_board"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES

_async_verify_executor = ThreadPoolExecutor(
    max_workers=MAX_VERIFY_CONCURRENT, thread_name_prefix="combined_verify"
)


def get_verify_executor() -> ThreadPoolExecutor:
    return _async_verify_executor


# ── Detector ─────────────────────────────────────────────────────────────────

class CombinedDetector(YoloeTrainedVlDetector):
    """
    Dual-model detector: yoloe_trained_vl for all classes + yolo for drain_issue.

    Inherits __init__ from YoloeTrainedVlDetector (which loads best-11m.pt as
    self.model and sets verify_fn=process_with_trained_vl).
    Additionally loads all-with-drain.pt as self.drain_model.
    """

    def __init__(self):
        # Loads best-11m.pt → self.model, sets verify_fn, detection_mode
        super().__init__()
        self.detection_mode = "combined"
        self._load_drain_model()

    def _load_drain_model(self):
        """Load the YOLO drain model as self.drain_model."""
        try:
            from app.detectors.base.base_detector import _yolo_load_ctx
            logger.info(f"CombinedDetector: loading drain model {YOLO_DRAIN_MODEL_PATH}")
            with _yolo_load_ctx():
                self.drain_model = YOLO(YOLO_DRAIN_MODEL_PATH)
            # Warm up
            import numpy as np
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.drain_model.predict(dummy, verbose=False, device=self.device)
            logger.info("CombinedDetector: drain model ready")
        except Exception as e:
            logger.error(f"Failed to load drain model: {e}")
            raise

    # ── Spatial dedup: disabled (VL handles it, same as yoloe_trained_vl) ──
    def is_duplicate_location(self, *args, **kwargs):
        return False, None

    # ── Main processing loop ──────────────────────────────────────────────────

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        has_verify = self.verify_fn is not None
        cap = None

        # ── Reset tracker state for both models before processing ─────────────
        # persist=True carries Kalman-filter state between .track() calls.
        # Without a reset, IDs and trajectories from a PREVIOUS video bleed
        # into this run, producing wrong class labels and huge track IDs.
        for _m in (self.model, self.drain_model):
            if hasattr(_m, "predictor") and _m.predictor is not None:
                if hasattr(_m.predictor, "trackers"):
                    _m.predictor.trackers = None
                if hasattr(_m.predictor, "tracker"):
                    _m.predictor.tracker = None

        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id, {"type": "status", "status": "processing", "progress": 0}
                ),
                loop,
            )

            gps_points = self._load_gps_data(json_path)
            process_start = time.time()

            perf_timings = {
                "frame_decode":  {"total": 0.0, "count": 0},
                "yolo_inference": {"total": 0.0, "count": 0},
                "drain_inference": {"total": 0.0, "count": 0},
                "gps_coord":     {"total": 0.0, "count": 0},
                "verification":  {"total": 0.0, "count": 0},
                "db_gps_match":  {"total": 0.0, "count": 0},
                "db_bulk_write": {"total": 0.0, "count": 0},
            }

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = total_frames / fps

            logger.info(
                f"Processing [combined] for {video_id}: {total_frames} frames @ {fps:.1f} FPS"
            )

            # Adaptive thresholds
            DETECTION_TIME_WINDOW   = video_duration * 0.25
            TIME_THRESHOLD          = video_duration * 0.30
            HIGH_CONFIDENCE_THRESHOLD = 0.75
            LOW_CONFIDENCE_MIN_FRAMES = 2
            MIN_DISTANCE_THRESHOLD  = 120

            # ── Shared tracking structures (both models write here) ──────────
            tracker_history   = defaultdict(lambda: deque(maxlen=50))
            confirmed         = {}
            counted_ids       = {cls: set() for cls in ALL_CLASSES}
            spatial_locations = defaultdict(deque)
            tracker_class_lock = {}
            pending_verify    = {}
            verify_cache      = {}
            rejected_tids     = set()
            _tid_last_seen    = {}

            rejection_stats = {
                "multi_frame_pending": set(),
                "spatial_duplicate":   0,
                "roi_outside":         0,
                "class_mismatch":      0,
                "vl_mismatch":         0,
                "vl_errors":           0,
            }
            verify_stats = {
                "total_verified":   0,
                "verified_success": 0,
                "verified_failed":  0,
                "skipped":          0,
                "vl_overrides":     0,
            }

            # ── Inner helpers ────────────────────────────────────────────────

            def _confirm_detection(
                tid, class_name, frame_count, current_time, conf,
                x1, y1, x2, y2, cx, cy,
                vl_verified=False, vl_confidence=None, vl_category=None,
            ):
                confirmed[tid] = {
                    "detection_id": tid,
                    "type": class_name,
                    "first_detected_frame": frame_count,
                    "first_detected_time": round(current_time, 2),
                    "confidence": round(float(conf), 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "vl_verified": vl_verified if has_verify else None,
                    "vl_confidence": vl_confidence,
                    "vl_category": vl_category,
                }
                if class_name in counted_ids:
                    counted_ids[class_name].add(tid)
                spatial_locations[class_name].append(
                    {"center": (cx, cy), "time": current_time, "bbox": (x1, y1, x2, y2)}
                )
                rejection_stats["multi_frame_pending"].discard(tid)

            def _process_completed_verify_futures():
                done_tids = []
                for tid, pending in pending_verify.items():
                    future = pending["future"]
                    if not future.done():
                        continue
                    done_tids.append(tid)
                    try:
                        vl_result = future.result(timeout=0)
                    except Exception as e:
                        logger.warning(f"Verify async error for tid={tid}: {e}")
                        rejection_stats["vl_errors"] += 1
                        continue

                    if not vl_result:
                        rejection_stats["vl_errors"] += 1
                        continue

                    verify_stats["total_verified"] += 1
                    verify_cache[tid] = vl_result
                    _vl_elapsed = vl_result.pop("_vl_elapsed_s", 0.0)
                    perf_timings["verification"]["total"] += _vl_elapsed
                    perf_timings["verification"]["count"] += 1

                    vl_category   = vl_result.get("category")
                    vl_confidence = vl_result.get("confidence")
                    belongs       = vl_result.get("belongs_to_category", False)
                    yolo_class    = pending["class_name"]

                    if vl_category == yolo_class and belongs:
                        verify_stats["verified_success"] += 1
                        if tid in confirmed:
                            confirmed[tid]["vl_verified"]    = True
                            confirmed[tid]["vl_confidence"]  = vl_confidence
                            confirmed[tid]["vl_category"]    = vl_category
                    elif (
                        vl_category
                        and vl_category != "null"
                        and vl_category in ALL_CLASSES
                        and belongs
                        and vl_confidence in ("high", "medium")
                    ):
                        logger.info(
                            f"[combined] Verify override tid={tid}: "
                            f"YOLO={yolo_class} → VL={vl_category} (conf={vl_confidence})"
                        )
                        verify_stats["verified_success"] += 1
                        verify_stats["vl_overrides"] += 1
                        if tid in confirmed:
                            old_class = confirmed[tid]["type"]
                            if old_class in counted_ids:
                                counted_ids[old_class].discard(tid)
                            if vl_category in counted_ids:
                                counted_ids[vl_category].add(tid)
                            confirmed[tid]["type"]          = vl_category
                            confirmed[tid]["vl_verified"]   = True
                            confirmed[tid]["vl_confidence"] = vl_confidence
                            confirmed[tid]["vl_category"]   = vl_category
                            tracker_class_lock[tid]         = vl_category
                    else:
                        verify_stats["verified_failed"] += 1
                        rejection_stats["vl_mismatch"] += 1
                        logger.info(
                            f"[combined] Verify rejected tid={tid}: "
                            f"YOLO={yolo_class}, VL={vl_category} "
                            f"(conf={vl_confidence}, belongs={belongs})"
                        )
                        if tid in confirmed:
                            old_class = confirmed[tid]["type"]
                            if old_class in counted_ids:
                                counted_ids[old_class].discard(tid)
                            del confirmed[tid]
                        rejected_tids.add(tid)
                        verify_cache[tid] = vl_result

                for tid in done_tids:
                    del pending_verify[tid]

            def _evict_stale_trackers():
                stale = [
                    tid for tid, last_t in _tid_last_seen.items()
                    if current_time - last_t > TIME_THRESHOLD
                    and tid not in confirmed
                    and tid not in pending_verify
                ]
                for tid in stale:
                    has_cached = tid in verify_cache
                    verify_cache.pop(tid, None)
                    if has_cached:
                        rejected_tids.discard(tid)
                    tracker_class_lock.pop(tid, None)
                    tracker_history.pop(tid, None)
                    del _tid_last_seen[tid]
                if stale:
                    logger.info(
                        f"[combined] Evicted {len(stale)} stale tracker IDs"
                    )

            def _handle_detection(
                tid, class_name, conf, box,
                frame_count, current_time, frame,
                submit_vl=True,
            ):
                """
                Common confirmation + VL-submit logic used for both model outputs.
                Returns the detection entry dict if tid is already/now confirmed.
                """
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if tid in tracker_class_lock:
                    if tracker_class_lock[tid] != class_name:
                        rejection_stats["class_mismatch"] += 1
                        return None
                else:
                    tracker_class_lock[tid] = class_name

                tracker_history[tid].append(current_time)
                _tid_last_seen[tid] = current_time

                recent = [
                    t for t in tracker_history[tid]
                    if current_time - t <= DETECTION_TIME_WINDOW
                ]
                min_needed = (
                    1 if conf >= HIGH_CONFIDENCE_THRESHOLD
                    else LOW_CONFIDENCE_MIN_FRAMES
                )

                if (
                    len(recent) >= min_needed
                    and tid not in confirmed
                    and tid not in rejected_tids
                ):
                    # Spatial dedup always returns False (overridden)
                    is_dup, _ = self.is_duplicate_location(
                        cx, cy, (x1, y1, x2, y2), class_name,
                        current_time, spatial_locations,
                        TIME_THRESHOLD, MIN_DISTANCE_THRESHOLD,
                    )
                    if not is_dup:
                        _confirm_detection(
                            tid, class_name, frame_count, current_time,
                            conf, x1, y1, x2, y2, cx, cy,
                        )

                        # Pole tilt for sign boards
                        if class_name == "defected_sign_board":
                            tilt_angle, pole_status = analyze_pole_tilt(
                                frame, (x1, y1, x2, y2)
                            )
                            confirmed[tid]["pole_tilt_angle"] = round(tilt_angle, 1)
                            confirmed[tid]["pole_status"]     = pole_status

                        # Submit VL verification (only for yoloe classes, not drain)
                        if (
                            submit_vl
                            and has_verify
                            and tid not in verify_cache
                            and tid not in pending_verify
                            and len(pending_verify) < MAX_VERIFY_CONCURRENT
                        ):
                            frame_copy = frame.copy()
                            future = _async_verify_executor.submit(
                                self.verify_fn, frame_copy, (x1, y1, x2, y2), class_name
                            )
                            pending_verify[tid] = {
                                "future": future,
                                "class_name": class_name,
                            }
                    else:
                        rejection_stats["spatial_duplicate"] += 1
                elif tid not in confirmed:
                    rejection_stats["multi_frame_pending"].add(tid)

                return None  # frame-level entry built by caller if tid in confirmed

            # ── NDJSON temp file ─────────────────────────────────────────────
            _ndjson_fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".ndjson",
                prefix=f"frames_{video_id}_",
                dir=str(RESULTS_DIR),
                delete=False,
            )
            _ndjson_path = _ndjson_fd.name
            _pending_frames   = []
            _frames_written   = 0
            total_detections_count = 0
            frame_count       = 0
            last_progress     = 0

            logger.info(f"[combined] NDJSON temp file: {_ndjson_path}")

            # ── Main frame loop ───────────────────────────────────────────────
            while cap.isOpened():
                _t0_read = time.perf_counter()
                ret, frame = cap.read()
                _read_elapsed = time.perf_counter() - _t0_read
                if not ret:
                    break
                perf_timings["frame_decode"]["total"] += _read_elapsed
                perf_timings["frame_decode"]["count"] += 1

                frame_count += 1
                if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                    continue

                current_time = frame_count / fps

                # Process completed VL futures from previous frames
                if has_verify:
                    _process_completed_verify_futures()

                frame_data = {"frame_id": frame_count, "detections": []}

                # ── Model 1: YOLOE (all classes except drain_issue) ───────────
                with PerfTimer("YOLO inference", video_id) as _t_yoloe:
                    yoloe_results = self.model.track(
                        frame,
                        persist=True,
                        conf=self.conf_threshold,
                        tracker=TRACKER,
                        verbose=False,
                        device=self.device,
                        imgsz=YOLO_IMGSZ,
                    )
                perf_timings["yolo_inference"]["total"] += _t_yoloe.elapsed
                perf_timings["yolo_inference"]["count"] += 1

                if yoloe_results[0].boxes.id is not None:
                    track_ids   = yoloe_results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids   = yoloe_results[0].boxes.cls.cpu().numpy().astype(int)
                    boxes       = yoloe_results[0].boxes.xyxy.cpu().numpy()
                    confidences = yoloe_results[0].boxes.conf.cpu().numpy()

                    for tid, cid, box, conf in zip(track_ids, class_ids, boxes, confidences):
                        tid, cid    = int(tid), int(cid)
                        class_name  = str(self.model.names[cid])

                        # Skip drain_issue — handled by drain model below
                        if class_name == DRAIN_CLASS:
                            continue

                        _handle_detection(
                            tid, class_name, conf, box,
                            frame_count, current_time, frame,
                            submit_vl=True,
                        )

                        if tid in confirmed:
                            x1, y1, x2, y2 = map(int, box)
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            det_entry = {
                                "frame_id":    frame_count,
                                "detection_id": tid,
                                "type":        class_name,
                                "confidence":  round(float(conf), 3),
                                "count": {
                                    "defected_sign_board":   len(counted_ids["defected_sign_board"]),
                                    "pothole":               len(counted_ids["pothole"]),
                                    "road_crack":            len(counted_ids["road_crack"]),
                                    "damaged_road_marking":  len(counted_ids["damaged_road_marking"]),
                                    "good_sign_board":       len(counted_ids["good_sign_board"]),
                                    "drain_issue":           len(counted_ids["drain_issue"]),
                                },
                                "bbox":   {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "center": {"x": cx, "y": cy},
                                "area":   (x2 - x1) * (y2 - y1),
                            }
                            if "pole_tilt_angle" in confirmed[tid]:
                                det_entry["pole_tilt_angle"] = confirmed[tid]["pole_tilt_angle"]
                                det_entry["pole_status"]     = confirmed[tid]["pole_status"]
                            frame_data["detections"].append(det_entry)

                # ── Model 2: YOLO (drain_issue ONLY) ─────────────────────────
                _t0_drain = time.perf_counter()
                drain_results = self.drain_model.track(
                    frame,
                    persist=True,
                    conf=YOLO_CONF_THRESHOLD,
                    tracker=TRACKER,
                    verbose=False,
                    device=self.device,
                    imgsz=YOLO_IMGSZ,
                )
                perf_timings["drain_inference"]["total"] += time.perf_counter() - _t0_drain
                perf_timings["drain_inference"]["count"] += 1

                if drain_results[0].boxes.id is not None:
                    d_track_ids   = drain_results[0].boxes.id.cpu().numpy().astype(int)
                    d_class_ids   = drain_results[0].boxes.cls.cpu().numpy().astype(int)
                    d_boxes       = drain_results[0].boxes.xyxy.cpu().numpy()
                    d_confidences = drain_results[0].boxes.conf.cpu().numpy()

                    for tid, cid, box, conf in zip(d_track_ids, d_class_ids, d_boxes, d_confidences):
                        tid, cid   = int(tid), int(cid)
                        class_name = str(self.drain_model.names[cid])

                        # Accept drain_issue ONLY from this model
                        if class_name != DRAIN_CLASS:
                            continue

                        # Offset ID to isolate from YOLOE ID space
                        drain_tid = tid + DRAIN_TID_OFFSET

                        _handle_detection(
                            drain_tid, class_name, conf, box,
                            frame_count, current_time, frame,
                            submit_vl=False,  # No VL for drain (not in VL training set)
                        )

                        if drain_tid in confirmed:
                            x1, y1, x2, y2 = map(int, box)
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            frame_data["detections"].append({
                                "frame_id":    frame_count,
                                "detection_id": drain_tid,
                                "type":        DRAIN_CLASS,
                                "confidence":  round(float(conf), 3),
                                "count": {
                                    "defected_sign_board":   len(counted_ids["defected_sign_board"]),
                                    "pothole":               len(counted_ids["pothole"]),
                                    "road_crack":            len(counted_ids["road_crack"]),
                                    "damaged_road_marking":  len(counted_ids["damaged_road_marking"]),
                                    "good_sign_board":       len(counted_ids["good_sign_board"]),
                                    "drain_issue":           len(counted_ids["drain_issue"]),
                                },
                                "bbox":   {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "center": {"x": cx, "y": cy},
                                "area":   (x2 - x1) * (y2 - y1),
                            })

                # Buffer frame data
                if frame_data["detections"]:
                    _pending_frames.append(frame_data)

                # Progress updates
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id, progress, loop,
                        {
                            "unique_pothole":              len(counted_ids["pothole"]),
                            "unique_defected_sign_board":  len(counted_ids["defected_sign_board"]),
                            "unique_road_crack":           len(counted_ids["road_crack"]),
                            "unique_damaged_road_marking": len(counted_ids["damaged_road_marking"]),
                            "unique_good_sign_board":      len(counted_ids["good_sign_board"]),
                            "unique_drain_issue":          len(counted_ids["drain_issue"]),
                            "total_road_damage":           sum(
                                len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES
                            ),
                            "total_detections": sum(
                                len(f["detections"]) for f in _pending_frames
                            ),
                        },
                    )
                    last_progress = progress
                    _evict_stale_trackers()

            yolo_end = time.time()

            # ── Drain remaining VL futures ───────────────────────────────────
            if has_verify and pending_verify:
                logger.info(f"[combined] Draining {len(pending_verify)} pending verify futures...")
                VL_TIMEOUT = int(os.getenv("VL_TIMEOUT_SECONDS", "30"))
                futures_wait(
                    [p["future"] for p in pending_verify.values()],
                    timeout=VL_TIMEOUT,
                )
                _process_completed_verify_futures()
                if pending_verify:
                    logger.warning(
                        f"[combined] {len(pending_verify)} verify futures timed out — cancelling"
                    )
                    for tid in list(pending_verify.keys()):
                        pending_verify[tid]["future"].cancel()
                        rejection_stats["vl_errors"] += 1
                        del pending_verify[tid]

            vl_drain_end = time.time()

            # ── Post-drain NDJSON filter ─────────────────────────────────────
            confirmed_ids = set(confirmed.keys())
            for _fdata in _pending_frames:
                _surviving = [
                    d for d in _fdata["detections"]
                    if d["detection_id"] in confirmed_ids
                ]
                if _surviving:
                    # Backfill true VL-corrected class names
                    for d in _surviving:
                        true_type = confirmed.get(d["detection_id"], {}).get("type")
                        if true_type:
                            d["type"] = true_type
                    _fdata["detections"] = _surviving
                    _ndjson_fd.write(json.dumps(_fdata) + "\n")
                    _frames_written += 1
                    total_detections_count += len(_surviving)
            del _pending_frames
            logger.info(
                f"[combined] Post-drain filter — {_frames_written} frames, "
                f"{total_detections_count} detections written"
            )

            # ── Build per-class lists ────────────────────────────────────────
            defected_sign_board_list   = self._get_class_list(confirmed, "defected_sign_board",   gps_points, perf_timings)
            pothole_list               = self._get_class_list(confirmed, "pothole",               gps_points, perf_timings)
            road_crack_list            = self._get_class_list(confirmed, "road_crack",            gps_points, perf_timings)
            damaged_road_marking_list  = self._get_class_list(confirmed, "damaged_road_marking",  gps_points, perf_timings)
            good_sign_board_list       = self._get_class_list(confirmed, "good_sign_board",       gps_points, perf_timings)
            drain_issue_list           = self._get_class_list(confirmed, "drain_issue",           gps_points, perf_timings)

            frames_with_detections = _frames_written
            detection_rate = (
                round((frames_with_detections / frame_count) * 100, 2)
                if frame_count > 0 else 0
            )

            results = {
                "video_id":        video_id,
                "video_path":      video_path,
                "detection_mode":  self.detection_mode,
                "processed_at":    datetime.now().isoformat(),
                "video_info": {
                    "total_frames": total_frames,
                    "fps":          round(fps, 2),
                    "duration":     round(video_duration, 2),
                    "width":        width,
                    "height":       height,
                    "resolution":   f"{width}x{height}",
                },
                "summary": {
                    "total_frames":                frame_count,
                    "unique_defected_sign_board":  len(counted_ids["defected_sign_board"]),
                    "unique_pothole":              len(counted_ids["pothole"]),
                    "unique_road_crack":           len(counted_ids["road_crack"]),
                    "unique_damaged_road_marking": len(counted_ids["damaged_road_marking"]),
                    "unique_good_sign_board":      len(counted_ids["good_sign_board"]),
                    "unique_drain_issue":          len(counted_ids["drain_issue"]),
                    "total_road_damage":           sum(
                        len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES
                    ),
                    "total_detections":            total_detections_count,
                    "frames_with_detections":      frames_with_detections,
                    "detection_rate":              detection_rate,
                },
                "rejection_stats": {
                    "multi_frame_pending": len(rejection_stats["multi_frame_pending"]),
                    "spatial_duplicate":   rejection_stats["spatial_duplicate"],
                    "class_mismatch":      rejection_stats["class_mismatch"],
                    "roi_outside":         rejection_stats["roi_outside"],
                    "vl_mismatch":         rejection_stats["vl_mismatch"],
                    "vl_errors":           rejection_stats["vl_errors"],
                },
                "vl_stats": (
                    {
                        "enabled":          has_verify,
                        "total_verified":   verify_stats["total_verified"],
                        "verified_success": verify_stats["verified_success"],
                        "verified_failed":  verify_stats["verified_failed"],
                        "vl_overrides":     verify_stats["vl_overrides"],
                        "cache_hits":       verify_stats["skipped"],
                    }
                    if has_verify else None
                ),
                "defected_sign_board_list":   defected_sign_board_list,
                "pothole_list":               pothole_list,
                "road_crack_list":            road_crack_list,
                "damaged_road_marking_list":  damaged_road_marking_list,
                "good_sign_board_list":       good_sign_board_list,
                "drain_issue_list":           drain_issue_list,
                "frames": self._read_ndjson_frames(_ndjson_fd, _ndjson_path, _frames_written),
            }

            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            all_detections_flat = (
                defected_sign_board_list
                + pothole_list
                + road_crack_list
                + damaged_road_marking_list
                + good_sign_board_list
                + drain_issue_list
            )
            self._save_to_db(video_id, all_detections_flat, perf_timings)
            process_end = time.time()

            # ── Perf report ──────────────────────────────────────────────────
            total_time   = process_end - process_start
            yolo_time    = yolo_end    - process_start
            drain_time   = vl_drain_end - yolo_end
            frames_processed = frame_count // FRAME_SKIP if FRAME_SKIP > 1 else frame_count

            def _fmt(key):
                d = perf_timings[key]
                t, c = d["total"], d["count"]
                return t, c, (t / c * 1000) if c > 0 else 0.0

            def _flag(t):
                return " ⚠️" if total_time > 0 and (t / total_time) > 0.20 else ""

            fd_t,  fd_c,  fd_avg  = _fmt("frame_decode")
            yi_t,  yi_c,  yi_avg  = _fmt("yolo_inference")
            di_t,  di_c,  di_avg  = _fmt("drain_inference")
            gc_t,  gc_c,  gc_avg  = _fmt("gps_coord")
            vl_t,  vl_c,  vl_avg  = _fmt("verification")
            dg_t,  dg_c,  dg_avg  = _fmt("db_gps_match")
            db_t,  db_c,  db_avg  = _fmt("db_bulk_write")

            report_lines = [
                f"{'=' * 78}",
                f"  PERF REPORT — [{video_id}]  mode=combined",
                f"{'=' * 78}",
                f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'Frame decode (I/O)':<30} {fd_t:>10.3f} {fd_c:>7d} {fd_avg:>15.2f}{_flag(fd_t)}",
                f"  {'YOLOE inference':<30} {yi_t:>10.3f} {yi_c:>7d} {yi_avg:>15.2f}{_flag(yi_t)}",
                f"  {'YOLO drain inference':<30} {di_t:>10.3f} {di_c:>7d} {di_avg:>15.2f}{_flag(di_t)}",
                f"  {'GPS coord lookup':<30} {gc_t:>10.3f} {gc_c:>7d} {gc_avg:>15.2f}{_flag(gc_t)}",
                f"  {'Verification (async)':<30} {vl_t:>10.3f} {vl_c:>7d} {vl_avg:>15.2f}{_flag(vl_t)}",
                f"  {'Verify drain (post-loop)':<30} {drain_time:>10.3f} {'N/A':>7} {'N/A':>15}",
                f"  {'DB GPS matching':<30} {dg_t:>10.3f} {dg_c:>7d} {dg_avg:>15.2f}{_flag(dg_t)}",
                f"  {'DB bulk write':<30} {db_t:>10.3f} {db_c:>7d} {db_avg:>15.2f}{_flag(db_t)}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'TOTAL pipeline time':<30} {total_time:>10.3f}s",
                f"  {'Video duration':<30} {video_duration:>10.1f}s  ({total_frames} frames @ {fps:.0f} FPS)",
                f"  {'Frames processed':<30} {frames_processed:>10d}  (FRAME_SKIP={FRAME_SKIP})",
                f"  {'Detections saved':<30} {len(all_detections_flat):>10d}",
                f"  {'Verifications done':<30} {verify_stats['total_verified']:>10d}",
                f"{'=' * 78}",
            ]
            perf_logger.info("\n" + "\n".join(report_lines))

            processing_status[video_id] = {"status": "completed", "progress": 100}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id,
                    {
                        "type":             "complete",
                        "status":           "completed",
                        "summary":          results["summary"],
                        "rejection_stats":  results["rejection_stats"],
                    },
                ),
                loop,
            )
            return results

        except Exception as e:
            logger.error(f"[combined] Error processing {video_id}: {e}")
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
                if "_ndjson_fd" in dir() and not _ndjson_fd.closed:
                    _ndjson_fd.close()
                if "_ndjson_path" in dir() and os.path.exists(_ndjson_path):
                    os.remove(_ndjson_path)
                    logger.info(f"[combined] NDJSON temp file cleaned up: {_ndjson_path}")
            except OSError:
                pass
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
