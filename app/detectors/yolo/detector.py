"""
YoloDetector — Pure YOLO detection with optional verification callback.

The detector focuses solely on YOLO inference, tracking, and deduplication.
An optional `verify_fn` callback can be provided to add post-detection
verification (e.g., VL verification). The callback is invoked asynchronously
during frame processing and its results are used to confirm, override,
or reject detections.
"""

import cv2
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time
import logging
import torch
import os
import tempfile
from datetime import datetime
from collections import defaultdict, deque

from app.detectors.base.base_detector import BaseDetector
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.core.logging_config import PerfTimer, perf_logger
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.db.crud_hierarchy import find_chainage_by_gps
from app.detectors.yolo.pole_tilt import analyze_pole_tilt

logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = r"models\best-11m.pt"
TRACKER = "botsort.yaml"
CONF_THRESHOLD = 0.50

# Performance tuning
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))  # Process every Nth frame (1=no skip)
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))  # YOLO inference resolution
# Always run in FP32 — passing half=True can cause Ultralytics 8.3+ to internally
# use bfloat16, which numpy cannot convert (tracker crashes on .cpu().numpy()).

# Road damage classes
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}
EXCLUDED_CLASSES = {"good_sign_board"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES

# Async verification config
MAX_VERIFY_CONCURRENT = int(os.getenv("MAX_VL_CONCURRENT", "4"))
_async_verify_executor = ThreadPoolExecutor(
    max_workers=MAX_VERIFY_CONCURRENT, thread_name_prefix="verify_async"
)


def get_verify_executor() -> ThreadPoolExecutor:
    """Return the async-verify executor (for lifespan shutdown)."""
    return _async_verify_executor


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

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    @staticmethod
    def calculate_ios(box1, box2):
        """Calculate Intersection over Smaller Box (IoS)"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        smaller_area = min(area1, area2)
        if smaller_area == 0:
            return 0.0

        return intersection_area / smaller_area

    def is_duplicate_location(
        self,
        cx,
        cy,
        bbox,
        class_name,
        current_time,
        spatial_locations,
        time_threshold,
        min_distance_threshold,
    ):
        """
        Check if this location/class was already counted recently.

        spatial_locations is a defaultdict(deque) keyed by class_name.
        Entries are appended in time order so expired ones are pruned
        from the front in O(1) before scanning — keeping the active
        window small regardless of video length.
        """
        bucket = spatial_locations[class_name]

        # Prune entries outside the time window (deque is time-ordered)
        while bucket and (current_time - bucket[0]["time"]) > time_threshold:
            bucket.popleft()

        # Only same-class entries remain — no class check needed in the loop
        for existing in bucket:
            distance = self.calculate_distance((cx, cy), existing["center"])
            if distance < min_distance_threshold:
                time_gap = current_time - existing["time"]
                return True, f"{distance:.1f}px from existing, {time_gap:.2f}s ago"

            if bbox is not None and "bbox" in existing:
                ios = self.calculate_ios(bbox, existing["bbox"])
                if ios > 0.5: # adjust threshold
                    time_gap = current_time - existing["time"]
                    return True, f"Overlap IoS {ios:.2f} with existing, {time_gap:.2f}s ago"

        return False, None

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        has_verify = self.verify_fn is not None
        cap = None
        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id, {"type": "status", "status": "processing", "progress": 0}
                ),
                loop,
            )

            gps_points = self._load_gps_data(json_path)
            process_start = time.time()

            # ── Performance accumulators ─────────────────────────────────────
            # Each entry: {"total": float seconds, "count": int}
            perf_timings = {
                "frame_decode": {"total": 0.0, "count": 0},
                "yolo_inference": {"total": 0.0, "count": 0},
                "gps_coord": {"total": 0.0, "count": 0},
                "verification": {"total": 0.0, "count": 0},
                "db_gps_match": {"total": 0.0, "count": 0},
                "db_bulk_write": {"total": 0.0, "count": 0},
            }
            # ────────────────────────────────────────────────────────────────

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
                f"VERIFY={'ON' if has_verify else 'OFF'}"
            )

            # Adaptive parameters
            DETECTION_TIME_WINDOW = video_duration * 0.25
            TIME_THRESHOLD = video_duration * 0.30
            HIGH_CONFIDENCE_THRESHOLD = 0.75
            LOW_CONFIDENCE_MIN_FRAMES = 2
            MIN_DISTANCE_THRESHOLD = 120

            ROI_LEFT = 0
            ROI_RIGHT = width
            ROI_TOP = int(height * 0.05)
            ROI_BOTTOM = int(height * 0.95)

            # Tracking structures
            tracker_history = defaultdict(lambda: deque(maxlen=50))
            confirmed = {}
            counted_ids = {cls: set() for cls in ALL_CLASSES}
            # Per-class time-windowed deque — O(1) pruning, O(R) scan (R = recent same-class entries)
            spatial_locations = defaultdict(deque)
            tracker_class_lock = {}
            # Pending verification futures: tid -> {future, detection_info}
            pending_verify = {}
            verify_cache = (
                {}
            )  # Cache verify results by detection_id to avoid redundant calls
            rejected_tids = (
                set()
            )  # Track verify-rejected tids to prevent re-confirmation
            _tid_last_seen = {}  # tid -> last current_time the tid was observed

            def _confirm_detection(
                tid,
                class_name,
                frame_count,
                current_time,
                conf,
                x1,
                y1,
                x2,
                y2,
                cx,
                cy,
                vl_verified=False,
                vl_confidence=None,
                vl_category=None,
            ):
                """Helper to confirm a detection and update all tracking structures."""
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
                    {
                        "center": (cx, cy),
                        "time": current_time,
                        "bbox": (x1, y1, x2, y2),
                    }
                )
                rejection_stats["multi_frame_pending"].discard(tid)

            def _process_completed_verify_futures():
                """Check pending verify futures and apply results retroactively."""
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
                        # Already confirmed with YOLO — leave as-is
                        continue

                    if not vl_result:
                        rejection_stats["vl_errors"] += 1
                        continue

                    verify_stats["total_verified"] += 1
                    verify_cache[tid] = vl_result

                    # Accumulate verify elapsed time into perf_timings
                    _vl_elapsed = vl_result.pop("_vl_elapsed_s", 0.0)
                    perf_timings["verification"]["total"] += _vl_elapsed
                    perf_timings["verification"]["count"] += 1

                    vl_category = vl_result.get("category")
                    vl_confidence = vl_result.get("confidence")
                    belongs = vl_result.get("belongs_to_category", False)
                    yolo_class = pending["class_name"]

                    if vl_category == yolo_class and belongs:
                        # Tier 1: Verify agrees — mark as verified
                        verify_stats["verified_success"] += 1
                        if tid in confirmed:
                            confirmed[tid]["vl_verified"] = True
                            confirmed[tid]["vl_confidence"] = vl_confidence
                            confirmed[tid]["vl_category"] = vl_category
                    elif (
                        vl_category
                        and vl_category != "null"
                        and vl_category in ALL_CLASSES
                        and belongs
                        and vl_confidence in ("high", "medium")
                    ):
                        # Tier 2: Verify disagrees but has valid class — override
                        logger.info(
                            f"Verify async override tid={tid}: YOLO={yolo_class} → VL={vl_category} "
                            f"(conf={vl_confidence})"
                        )
                        verify_stats["verified_success"] += 1
                        verify_stats["vl_overrides"] += 1
                        if tid in confirmed:
                            # Move from old class to new class in counted_ids
                            old_class = confirmed[tid]["type"]
                            if old_class in counted_ids:
                                counted_ids[old_class].discard(tid)
                            if vl_category in counted_ids:
                                counted_ids[vl_category].add(tid)
                            confirmed[tid]["type"] = vl_category
                            confirmed[tid]["vl_verified"] = True
                            confirmed[tid]["vl_confidence"] = vl_confidence
                            confirmed[tid]["vl_category"] = vl_category
                            tracker_class_lock[tid] = vl_category
                    else:
                        # Tier 3: Verify rejects — remove the confirmed detection
                        verify_stats["verified_failed"] += 1
                        rejection_stats["vl_mismatch"] += 1
                        logger.info(
                            f"Verify async rejected tid={tid}: YOLO={yolo_class}, "
                            f"VL={vl_category} (conf={vl_confidence}, belongs={belongs})"
                        )
                        if tid in confirmed:
                            old_class = confirmed[tid]["type"]
                            if old_class in counted_ids:
                                counted_ids[old_class].discard(tid)
                            del confirmed[tid]
                        rejected_tids.add(tid)  # Prevent re-confirmation
                        verify_cache[tid] = vl_result  # Prevent re-submission

                for tid in done_tids:
                    del pending_verify[tid]

            def _evict_stale_trackers():
                """Remove stale entries from verify_cache, rejected_tids,
                tracker_class_lock, and tracker_history whose tid hasn't been
                seen for more than TIME_THRESHOLD seconds."""
                stale_tids = [
                    tid
                    for tid, last_t in _tid_last_seen.items()
                    if current_time - last_t > TIME_THRESHOLD
                    and tid not in confirmed
                    and tid not in pending_verify
                ]
                for tid in stale_tids:
                    # check verify_cache BEFORE popping.
                    # Only evict from rejected_tids when we have a cached verify
                    # result (i.e. VL/SAM3 gave a definitive answer). If the
                    # rejection came from a timeout/None pass-through path this
                    # tid won't be in rejected_tids anyway, but we still want to
                    # preserve any genuine rejection flag so YOLO can't silently
                    # reuse the same numeric tracker ID on a new object.
                    _has_cached_result = tid in verify_cache
                    verify_cache.pop(tid, None)
                    if _has_cached_result:
                        rejected_tids.discard(tid)
                    tracker_class_lock.pop(tid, None)
                    tracker_history.pop(tid, None)
                    del _tid_last_seen[tid]
                if stale_tids:
                    logger.info(
                        f"Evicted {len(stale_tids)} stale tracker IDs | "
                        f"verify_cache={len(verify_cache)} rejected={len(rejected_tids)} "
                        f"tracker_lock={len(tracker_class_lock)} last_seen={len(_tid_last_seen)}"
                    )

            rejection_stats = {
                "multi_frame_pending": set(),
                "spatial_duplicate": 0,
                "roi_outside": 0,
                "class_mismatch": 0,
                "vl_mismatch": 0,
                "vl_errors": 0,
            }

            verify_stats = {
                "total_verified": 0,
                "verified_success": 0,
                "verified_failed": 0,
                "skipped": 0,
                "vl_overrides": 0,
            }

            # Stream frame data to NDJSON temp file instead of unbounded list
            _ndjson_fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".ndjson",
                prefix=f"frames_{video_id}_",
                dir=str(RESULTS_DIR),
                delete=False,
            )
            _ndjson_path = _ndjson_fd.name
            # buffer frame data in memory during the loop;
            # written to NDJSON only after post-drain filter removes rejected tids.
            _pending_frames = []
            _frames_written = 0
            logger.info(f"NDJSON (deferred-write) temp file: {_ndjson_path}")
            total_detections_count = 0
            frame_count = 0
            last_progress = 0

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
                    _process_completed_verify_futures()

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

                        if not (
                            ROI_LEFT < cx < ROI_RIGHT and ROI_TOP < cy < ROI_BOTTOM
                        ):
                            rejection_stats["roi_outside"] += 1
                            continue

                        if tid in tracker_class_lock:
                            if tracker_class_lock[tid] != class_name:
                                rejection_stats["class_mismatch"] += 1
                                continue
                        else:
                            tracker_class_lock[tid] = class_name

                        tracker_history[tid].append(current_time)
                        _tid_last_seen[tid] = current_time
                        recent = [
                            t
                            for t in tracker_history[tid]
                            if current_time - t <= DETECTION_TIME_WINDOW
                        ]
                        min_needed = (
                            1
                            if conf >= HIGH_CONFIDENCE_THRESHOLD
                            else LOW_CONFIDENCE_MIN_FRAMES
                        )

                        if (
                            len(recent) >= min_needed
                            and tid not in confirmed
                            and tid not in rejected_tids
                        ):
                            is_dup, _ = self.is_duplicate_location(
                                cx,
                                cy,
                                (x1, y1, x2, y2),
                                class_name,
                                current_time,
                                spatial_locations,
                                TIME_THRESHOLD,
                                MIN_DISTANCE_THRESHOLD,
                            )
                            if not is_dup:
                                # Optimistic accept — confirm now, let verify callback check async
                                _confirm_detection(
                                    tid,
                                    class_name,
                                    frame_count,
                                    current_time,
                                    conf,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    cx,
                                    cy,
                                )

                                # ── Pole tilt for confirmed signboards ─────
                                if class_name == "defected_sign_board":
                                    tilt_angle, pole_status = analyze_pole_tilt(
                                        frame, (x1, y1, x2, y2)
                                    )
                                    confirmed[tid]["pole_tilt_angle"] = round(tilt_angle, 1)
                                    confirmed[tid]["pole_status"] = pole_status

                                # Submit async verification if callback is provided
                                if (
                                    has_verify
                                    and tid not in verify_cache
                                    and tid not in pending_verify
                                    and len(pending_verify) < MAX_VERIFY_CONCURRENT
                                ):
                                    frame_copy = frame.copy()
                                    bbox_copy = (x1, y1, x2, y2)
                                    class_copy = class_name
                                    future = _async_verify_executor.submit(
                                        self.verify_fn,
                                        frame_copy,
                                        bbox_copy,
                                        class_copy,
                                    )
                                    pending_verify[tid] = {
                                        "future": future,
                                        "class_name": class_copy,
                                    }
                            else:
                                rejection_stats["spatial_duplicate"] += 1
                        elif tid not in confirmed:
                            rejection_stats["multi_frame_pending"].add(tid)

                        if tid in confirmed:
                            # Bug 2 fix: no per-frame count increment here;
                            # total_detections_count is recomputed after post-drain filter.
                            det_entry = {
                                "frame_id": frame_count,
                                "detection_id": tid,
                                "type": class_name,
                                "confidence": round(float(conf), 3),
                                "count": {
                                    "defected_sign_board": len(
                                        counted_ids["defected_sign_board"]
                                    ),
                                    "pothole": len(counted_ids["pothole"]),
                                    "road_crack": len(counted_ids["road_crack"]),
                                    "damaged_road_marking": len(
                                        counted_ids["damaged_road_marking"]
                                    ),
                                    "good_sign_board": len(
                                        counted_ids["good_sign_board"]
                                    ),
                                },
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "center": {"x": cx, "y": cy},
                                "area": (x2 - x1) * (y2 - y1),
                            }
                            # Include pole tilt data for signboards
                            if "pole_tilt_angle" in confirmed[tid]:
                                det_entry["pole_tilt_angle"] = confirmed[tid]["pole_tilt_angle"]
                                det_entry["pole_status"] = confirmed[tid]["pole_status"]
                            frame_data["detections"].append(det_entry)

                # Bug 1 fix: buffer instead of writing directly to NDJSON.
                # Actual write happens after the VL/SAM3 drain (see post-drain block).
                if frame_data["detections"]:
                    _pending_frames.append(frame_data)

                # Progress
                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id,
                        progress,
                        loop,
                        {
                            "unique_pothole": len(counted_ids["pothole"]),
                            "unique_defected_sign_board": len(
                                counted_ids["defected_sign_board"]
                            ),
                            "unique_road_crack": len(counted_ids["road_crack"]),
                            "unique_damaged_road_marking": len(
                                counted_ids["damaged_road_marking"]
                            ),
                            "unique_good_sign_board": len(
                                counted_ids["good_sign_board"]
                            ),
                            "total_road_damage": sum(
                                len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES
                            ),
                            # Note: total_detections_count is recomputed after
                            # the post-drain filter; use pending frame count here
                            # as an in-progress approximation.
                            "total_detections": sum(
                                len(f["detections"]) for f in _pending_frames
                            ),
                        },
                    )
                    last_progress = progress
                    # Periodically evict stale tracker entries to bound memory
                    _evict_stale_trackers()

            yolo_end = time.time()

            # --- Drain remaining verify futures after video processing ---
            if has_verify and pending_verify:
                logger.info(f"Draining {len(pending_verify)} pending verify futures...")
                from concurrent.futures import wait

                remaining_futures = [p["future"] for p in pending_verify.values()]
                VL_TIMEOUT = int(os.getenv("VL_TIMEOUT_SECONDS", "30"))
                done, not_done = wait(remaining_futures, timeout=VL_TIMEOUT)
                # Process completed ones
                _process_completed_verify_futures()
                # Cancel any that didn't finish in time
                if pending_verify:
                    logger.warning(
                        f"{len(pending_verify)} verify futures timed out — cancelling"
                    )
                    for tid in list(pending_verify.keys()):
                        pending_verify[tid]["future"].cancel()
                        rejection_stats["vl_errors"] += 1
                        del pending_verify[tid]

            vl_drain_end = time.time()

            # ── Post-drain NDJSON filter ────────────────
            # Now that confirmed{} is final (all VL/SAM3 results applied),
            # filter buffered frame data to remove any rejected tids, then
            # write surviving frames to NDJSON and recompute the true count.
            confirmed_ids = set(confirmed.keys())
            _frames_written = 0
            total_detections_count = 0
            for _fdata in _pending_frames:
                _surviving = [
                    d for d in _fdata["detections"]
                    if d["detection_id"] in confirmed_ids
                ]
                if _surviving:
                    _fdata["detections"] = _surviving
                    _ndjson_fd.write(json.dumps(_fdata) + "\n")
                    _frames_written += 1
                    total_detections_count += len(_surviving)
            del _pending_frames  # release memory
            logger.info(
                f"NDJSON post-drain filter — {_frames_written} frames, "
                f"{total_detections_count} detections written "
                f"(rejected tids filtered from frame data)"
            )
            # ────────────────────────────────────────────────────────────────

            # Build final results — time GPS coord lookup here
            defected_sign_board_list = self._get_class_list(
                confirmed, "defected_sign_board", gps_points, perf_timings
            )
            pothole_list = self._get_class_list(
                confirmed, "pothole", gps_points, perf_timings
            )
            road_crack_list = self._get_class_list(
                confirmed, "road_crack", gps_points, perf_timings
            )
            damaged_road_marking_list = self._get_class_list(
                confirmed, "damaged_road_marking", gps_points, perf_timings
            )
            good_sign_board_list = self._get_class_list(
                confirmed, "good_sign_board", gps_points, perf_timings
            )

            frames_with_detections = _frames_written
            detection_rate = (
                round((frames_with_detections / frame_count) * 100, 2)
                if frame_count > 0
                else 0
            )

            results = {
                "video_id": video_id,
                "video_path": video_path,
                "detection_mode": self.detection_mode,
                "processed_at": datetime.now().isoformat(),
                "video_info": {
                    "total_frames": total_frames,
                    "fps": round(fps, 2),
                    "duration": round(video_duration, 2),
                    "width": width,
                    "height": height,
                    "resolution": f"{width}x{height}",
                },
                "summary": {
                    "total_frames": frame_count,
                    "unique_defected_sign_board": len(
                        counted_ids["defected_sign_board"]
                    ),
                    "unique_pothole": len(counted_ids["pothole"]),
                    "unique_road_crack": len(counted_ids["road_crack"]),
                    "unique_damaged_road_marking": len(
                        counted_ids["damaged_road_marking"]
                    ),
                    "unique_good_sign_board": len(counted_ids["good_sign_board"]),
                    "total_road_damage": sum(
                        len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES
                    ),
                    "total_detections": total_detections_count,
                    "frames_with_detections": frames_with_detections,
                    "detection_rate": detection_rate,
                },
                "rejection_stats": {
                    "multi_frame_pending": len(rejection_stats["multi_frame_pending"]),
                    "spatial_duplicate": rejection_stats["spatial_duplicate"],
                    "class_mismatch": rejection_stats["class_mismatch"],
                    "roi_outside": rejection_stats["roi_outside"],
                    "vl_mismatch": rejection_stats["vl_mismatch"],
                    "vl_errors": rejection_stats["vl_errors"],
                },
                "vl_stats": (
                    {
                        "enabled": has_verify,
                        "total_verified": verify_stats["total_verified"],
                        "verified_success": verify_stats["verified_success"],
                        "verified_failed": verify_stats["verified_failed"],
                        "vl_overrides": verify_stats["vl_overrides"],
                        "cache_hits": verify_stats["skipped"],
                    }
                    if has_verify
                    else None
                ),
                "defected_sign_board_list": defected_sign_board_list,
                "pothole_list": pothole_list,
                "road_crack_list": road_crack_list,
                "damaged_road_marking_list": damaged_road_marking_list,
                "good_sign_board_list": good_sign_board_list,
                # Read back streamed frames from NDJSON temp file
                "frames": self._read_ndjson_frames(
                    _ndjson_fd, _ndjson_path, _frames_written
                ),
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
            )
            self._save_to_db(video_id, all_detections_flat, perf_timings)
            process_end = time.time()

            # ── Structured PERF REPORT ────────────────────────────────────────
            total_time = process_end - process_start
            yolo_time = yolo_end - process_start
            drain_time = vl_drain_end - yolo_end
            frames_processed = (
                frame_count // FRAME_SKIP if FRAME_SKIP > 1 else frame_count
            )

            def _fmt(stage_key):
                """Format a perf_timings entry as (total_s, count, avg_ms)."""
                d = perf_timings[stage_key]
                total = d["total"]
                count = d["count"]
                avg_ms = (total / count * 1000) if count > 0 else 0.0
                return total, count, avg_ms

            fd_t, fd_c, fd_avg = _fmt("frame_decode")
            yi_t, yi_c, yi_avg = _fmt("yolo_inference")
            gc_t, gc_c, gc_avg = _fmt("gps_coord")
            vl_t, vl_c, vl_avg = _fmt("verification")
            dg_t, dg_c, dg_avg = _fmt("db_gps_match")
            db_t, db_c, db_avg = _fmt("db_bulk_write")

            # Highlight potential bottlenecks (> 20% of total time)
            def _flag(t):
                return " ⚠️" if total_time > 0 and (t / total_time) > 0.20 else ""

            report_lines = [
                f"{'=' * 78}",
                f"  PERF REPORT — [{video_id}]  mode={self.detection_mode}",
                f"{'=' * 78}",
                f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'Frame decode (I/O)':<30} {fd_t:>10.3f} {fd_c:>7d} {fd_avg:>15.2f}{_flag(fd_t)}",
                f"  {'YOLO inference':<30} {yi_t:>10.3f} {yi_c:>7d} {yi_avg:>15.2f}{_flag(yi_t)}",
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
            report_str = "\n".join(report_lines)

            # Write to the dedicated perf log only — detection.log stays clean
            perf_logger.info(f"\n{report_str}")

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
            # Clean up NDJSON temp file
            try:
                if "_ndjson_fd" in dir() and not _ndjson_fd.closed:
                    _ndjson_fd.close()
                if "_ndjson_path" in dir() and os.path.exists(_ndjson_path):
                    os.remove(_ndjson_path)
                    logger.info(f"NDJSON temp file cleaned up: {_ndjson_path}")
            except OSError:
                pass
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    @staticmethod
    def _read_ndjson_frames(
        ndjson_fd, ndjson_path: str, expected_count: int = 0
    ) -> list:
        """Close the NDJSON temp file and read all frame lines back as a list."""
        if not ndjson_fd.closed:
            ndjson_fd.close()
        frames = []
        try:
            with open(ndjson_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        frames.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read back NDJSON frames: {e}")
        logger.info(
            f"NDJSON read-back complete — {len(frames)} frames loaded "
            f"(expected {expected_count}) from {ndjson_path}"
        )
        return frames

    def _get_class_list(self, confirmed, class_name, gps_points, perf_timings=None):
        lst = []
        # Pre-build timestamp list once — find_nearest_gps reuses it (O(log N) per call)
        gps_timestamps = (
            [p.get("timestamp", 0) for p in gps_points] if gps_points else []
        )
        for tid, info in confirmed.items():
            if info["type"] == class_name:
                det_data = {
                    "detection_id": tid,
                    "type": info["type"],
                    "first_detected_frame": info["first_detected_frame"],
                    "first_detected_time": info["first_detected_time"],
                    "confidence": info["confidence"],
                    "bbox": info.get("bbox", {}),
                    "vl_verified": info.get("vl_verified"),
                    "vl_confidence": info.get("vl_confidence"),
                    "vl_category": info.get("vl_category"),
                }
                # Include pole tilt data for signboard detections
                if "pole_tilt_angle" in info:
                    det_data["pole_tilt_angle"] = info["pole_tilt_angle"]
                    det_data["pole_status"] = info["pole_status"]
                if gps_points:
                    _t0 = time.perf_counter()
                    gps_coords = self.find_nearest_gps(
                        info["first_detected_time"], gps_points, gps_timestamps
                    )
                    _gps_elapsed = time.perf_counter() - _t0
                    if perf_timings is not None:
                        perf_timings["gps_coord"]["total"] += _gps_elapsed
                        perf_timings["gps_coord"]["count"] += 1
                    det_data.update(gps_coords)
                lst.append(det_data)
        return sorted(lst, key=lambda x: x["first_detected_frame"])

    def _save_to_db(self, video_id, all_detections, perf_timings=None):
        db = SessionLocal()
        try:
            db_detections = []

            for det in all_detections:
                lat, lng = det.get("lat"), det.get("lng")
                project_id, package_id, chainage_id = None, None, None
                if lat and lng:
                    # ⏱ Time the GPS-based chainage DB lookup
                    _t0 = time.perf_counter()
                    chainage = find_chainage_by_gps(db, lat, lng)
                    _gps_db_elapsed = time.perf_counter() - _t0
                    if perf_timings is not None:
                        perf_timings["db_gps_match"]["total"] += _gps_db_elapsed
                        perf_timings["db_gps_match"]["count"] += 1
                    if chainage:
                        chainage_id = chainage.id
                        package_id = chainage.package_id
                        if chainage.package:
                            project_id = chainage.package.project_id

                db_det = Detection(
                    video_id=video_id,
                    frame_number=det["first_detected_frame"],
                    timestamp_ms=int(det["first_detected_time"] * 1000),
                    confidence=det["confidence"],
                    detection_type=det["type"],
                    class_name=det["type"],
                    latitude=lat,
                    longitude=lng,
                    project_id=project_id,
                    package_id=package_id,
                    chainage_id=chainage_id,
                )
                db_det.set_bounding_box(det.get("bbox", {}))
                db_detections.append(db_det)

            if db_detections:
                # ⏱ Time the bulk insert
                _t0 = time.perf_counter()
                crud.create_detections_bulk(db, db_detections)
                _bulk_elapsed = time.perf_counter() - _t0
                if perf_timings is not None:
                    perf_timings["db_bulk_write"]["total"] += _bulk_elapsed
                    perf_timings["db_bulk_write"]["count"] += 1
                logger.info(
                    f"Saved {len(db_detections)} detections to database for {video_id}"
                )
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()
