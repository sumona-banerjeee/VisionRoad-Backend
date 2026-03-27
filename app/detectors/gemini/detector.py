"""
GeminiVideoDetector — Gemini 1.5 based video road-inspection detection.

Instead of frame-by-frame YOLO inference, this detector:
  1. Uploads the whole video to the Gemini Files API.
  2. Sends a single generate_content() call with a structured prompt.
  3. Parses the JSON response (timestamps, labels, bounding boxes).
  4. Extracts frame snapshots at each detection timestamp.
  5. Maps timestamps to GPS coords and saves results in the standard format.

Does NOT inherit BaseDetector (which is YOLO-specific).
"""

import cv2
import json
import re
import os
import time
import asyncio
import logging
import bisect
import yaml
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from google import genai

from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.models.video import Video
from app.db.crud_hierarchy import find_chainage_by_gps

logger = logging.getLogger(__name__)
from app.core.logging_config import perf_logger

# Shared background executor
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini_proc")

# ── Configuration ──────────────────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-robotics-er-1.5-preview")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# The categories the model should detect
DETECTION_CATEGORIES = [
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
    "good_sign_board",
    "drain_issue",
    "defective_culvert",
    "good_culvert",
]

# Qualitative → numeric confidence mapping
CONFIDENCE_MAP = {
    "high": 0.95,
    "medium": 0.75,
    "low": 0.50,
}

# Snapshot output directory
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Prompt (loaded from prompts.yaml next to this file) ───────────────────────
_PROMPT_FILE = Path(__file__).parent / "prompts.yaml"
try:
    with _PROMPT_FILE.open(encoding="utf-8") as _f:
        DETECTION_PROMPT: str = yaml.safe_load(_f)["detection_prompt"]
    logger.debug(f"Loaded detection prompt from {_PROMPT_FILE}")
except Exception as _e:
    logger.error(f"Failed to load prompts.yaml: {_e}. Using fallback prompt.")
    DETECTION_PROMPT = (
        "Detect road defects in this dashcam video. Return ONLY a JSON array."
    )


def get_gemini_executor() -> ThreadPoolExecutor:
    """Return the Gemini processing executor (for lifespan shutdown)."""
    return _executor


class GeminiVideoDetector:
    """
    Video-level road inspection detector using Gemini 1.5.

    Implements the same interface as BaseDetector subclasses:
      - process_video(video_id, video_path, json_path, speed_kmh)
      - _process_video_blocking(video_id, video_path, json_path, speed, loop)
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it in your .env file."
            )
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.detection_mode = "gemini_video"
        logger.info(f"GeminiVideoDetector ready — model: {GEMINI_MODEL}")

    # ── Public interface (matches BaseDetector) ────────────────────────────────

    async def process_video(
        self, video_id: str, video_path: str, json_path: str, speed_kmh: int
    ):
        processing_status[video_id] = {"status": "processing", "progress": 0}
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor,
            self._process_video_blocking,
            video_id,
            video_path,
            json_path,
            speed_kmh,
            loop,
        )

    # ── Core processing ────────────────────────────────────────────────────────

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        try:
            self._send_ws(
                loop,
                video_id,
                {"type": "status", "status": "processing", "progress": 0},
            )

            process_start = time.time()
            # ── Perf accumulators ─────────────────────────────────────────────
            _t_upload       = 0.0   # file upload + Gemini processing poll
            _t_gemini_api   = 0.0   # generate_content call
            _t_window_scan  = 0.0   # _find_best_frame_in_window total
            _t_gps          = 0.0   # GPS nearest-point lookups
            _t_db           = 0.0   # DB save
            _n_window_scan  = 0     # number of detections processed
            _n_gps          = 0
            gps_points = self._load_gps_data(json_path)

            # ── 1. Get video metadata via OpenCV ──────────────────────────────
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = total_frames / fps
            cap.release()

            logger.info(
                f"GeminiVideoDetector [{video_id}]: {total_frames} frames "
                f"@ {fps:.1f} FPS, {width}x{height}, duration={video_duration:.1f}s"
            )

            # ── 2. Upload video to Gemini Files API ───────────────────────────
            self._send_progress(loop, video_id, 10, "Uploading video to Gemini...")

            logger.info(f"Uploading video to Gemini Files API: {video_path}")
            _t0 = time.perf_counter()
            uploaded_file = self.client.files.upload(file=video_path)
            logger.info(
                f"File uploaded: name={uploaded_file.name}, state={uploaded_file.state}"
            )

            # Poll until active
            self._send_progress(
                loop, video_id, 20, "Waiting for Gemini to process file..."
            )
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
                logger.debug(f"File state: {uploaded_file.state}")
            _t_upload = time.perf_counter() - _t0

            if uploaded_file.state.name == "FAILED":
                raise Exception(f"Gemini file processing failed: {uploaded_file.state}")

            logger.info(f"File ready: {uploaded_file.name}")

            # ── 3. Call Gemini generate_content (with retry for rate limits) ──
            self._send_progress(loop, video_id, 40, "Analyzing video with Gemini...")

            logger.info("Sending generate_content request to Gemini...")

            MAX_RETRIES = 3
            response = None
            _t0 = time.perf_counter()
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = self.client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[uploaded_file, DETECTION_PROMPT],
                    )
                    break  # Success — exit retry loop
                except Exception as api_err:
                    err_str = str(api_err)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        # Parse retry delay from error if available
                        retry_match = re.search(
                            r"retry\w*\s*in\s*([\d.]+)", err_str, re.IGNORECASE
                        )
                        wait_secs = float(retry_match.group(1)) if retry_match else 60.0
                        wait_secs = max(wait_secs, 30.0)  # At least 30 seconds

                        if attempt < MAX_RETRIES:
                            logger.warning(
                                f"Rate limited (attempt {attempt}/{MAX_RETRIES}), "
                                f"retrying in {wait_secs:.0f}s..."
                            )
                            self._send_progress(
                                loop,
                                video_id,
                                45,
                                f"Rate limited — retrying in {int(wait_secs)}s "
                                f"(attempt {attempt}/{MAX_RETRIES})...",
                            )
                            time.sleep(wait_secs)
                        else:
                            logger.error(
                                f"Rate limited on final attempt ({attempt}/{MAX_RETRIES})"
                            )
                            raise
                    else:
                        raise  # Non-rate-limit error — don't retry

            _t_gemini_api = time.perf_counter() - _t0
            raw_text = response.text
            logger.info(f"Gemini response received ({len(raw_text)} chars)")
            logger.debug(f"Raw response: {raw_text[:500]}")

            self._send_progress(loop, video_id, 60, "Parsing detections...")

            # ── 4. Parse JSON from response ───────────────────────────────────
            detections = self._parse_response(raw_text)
            logger.info(f"Parsed {len(detections)} detections from Gemini response")

            # ── 5. Process detections ─────────────────────────────────────────
            self._send_progress(loop, video_id, 70, "Extracting frame snapshots...")

            confirmed = {}
            counted_ids = {cat: set() for cat in DETECTION_CATEGORIES}
            det_id = 0

            for det in detections:
                label = det.get("label", "").strip()
                if label not in DETECTION_CATEGORIES:
                    logger.warning(f"Skipping unknown label: {label}")
                    continue

                det_id += 1
                timestamp_str = det.get("timestamp", "00:00")
                timestamp_sec = self._parse_timestamp(timestamp_str)

                # Clamp to video duration
                timestamp_sec = min(timestamp_sec, video_duration)

                # Rescale bounding box
                box_2d = det.get("box_2d", [0, 0, 1000, 1000])
                bbox = self._rescale_bbox(box_2d, width, height)

                # Confidence
                conf_str = det.get("confidence", "medium").lower()
                confidence = CONFIDENCE_MAP.get(conf_str, 0.75)

                description = det.get("description", "")

                # Scan a ±1 s window to find the frame where YOLOE is most
                # confident the target object is visible.  Gemini timestamps
                # are whole-second resolution so the actual peak frame can be
                # up to ~1 s away from the reported timestamp.
                snapshot_path = None
                _t0 = time.perf_counter()
                best_frame_bgr, best_bbox, best_ts = self._find_best_frame_in_window(
                    video_path, timestamp_sec, bbox, label, video_duration
                )
                _t_window_scan += time.perf_counter() - _t0
                _n_window_scan += 1

                if best_frame_bgr is not None:
                    bbox = best_bbox
                    timestamp_sec = best_ts

                # Frame number derived from the winning timestamp
                frame_number = max(1, int(timestamp_sec * fps))

                # GPS lookup
                _t0 = time.perf_counter()
                gps_coords = self._find_nearest_gps(timestamp_sec, gps_points)
                _t_gps += time.perf_counter() - _t0
                _n_gps += 1

                confirmed[det_id] = {
                    "detection_id": det_id,
                    "type": label,
                    "first_detected_frame": frame_number,
                    "first_detected_time": round(timestamp_sec, 2),
                    "confidence": confidence,
                    "bbox": bbox,
                    "snapshot_path": snapshot_path,
                    "description": description,
                    "lat": gps_coords.get("lat"),
                    "lng": gps_coords.get("lng"),
                }

                if label in counted_ids:
                    counted_ids[label].add(det_id)

            self._send_progress(loop, video_id, 85, "Building results...")

            # ── 6. Build per-class lists ──────────────────────────────────────
            class_lists = {}
            for cat in DETECTION_CATEGORIES:
                class_lists[cat] = sorted(
                    [info for info in confirmed.values() if info["type"] == cat],
                    key=lambda x: x["first_detected_time"],
                )

            total_detections = len(confirmed)
            total_road_damage = sum(
                len(counted_ids[c])
                for c in [
                    "defected_sign_board",
                    "pothole",
                    "road_crack",
                    "damaged_road_marking",
                    "drain_issue",
                ]
            )

            # ── 7. Assemble result JSON ───────────────────────────────────────

            # Group into frames array
            frames_dict = {}
            for info in confirmed.values():
                f_id = info["first_detected_frame"]
                if f_id not in frames_dict:
                    frames_dict[f_id] = {"frame_id": f_id, "detections": []}

                bbox = info.get("bbox", {"x1": 0, "y1": 0, "x2": 0, "y2": 0})
                det_entry = {
                    "frame_id": f_id,
                    "detection_id": info["detection_id"],
                    "type": info["type"],
                    "confidence": info["confidence"],
                    "count": {
                        c: len(counted_ids.get(c, set())) for c in DETECTION_CATEGORIES
                    },
                    "bbox": bbox,
                    "center": {
                        "x": (bbox["x1"] + bbox["x2"]) // 2,
                        "y": (bbox["y1"] + bbox["y2"]) // 2,
                    },
                    "area": max(0, bbox["x2"] - bbox["x1"])
                    * max(0, bbox["y2"] - bbox["y1"]),
                }
                frames_dict[f_id]["detections"].append(det_entry)

            frames_list = [frames_dict[k] for k in sorted(frames_dict.keys())]

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
                    "total_frames": total_frames,
                    "unique_defected_sign_board": len(
                        counted_ids["defected_sign_board"]
                    ),
                    "unique_pothole": len(counted_ids["pothole"]),
                    "unique_road_crack": len(counted_ids["road_crack"]),
                    "unique_damaged_road_marking": len(
                        counted_ids["damaged_road_marking"]
                    ),
                    "unique_good_sign_board": len(counted_ids["good_sign_board"]),
                    "unique_drain_issue": len(counted_ids["drain_issue"]),
                    "unique_defective_culvert": len(
                        counted_ids.get("defective_culvert", set())
                    ),
                    "unique_good_culvert": len(counted_ids.get("good_culvert", set())),
                    "total_road_damage": total_road_damage,
                    "total_detections": total_detections,
                },
                "defected_sign_board_list": class_lists["defected_sign_board"],
                "pothole_list": class_lists["pothole"],
                "road_crack_list": class_lists["road_crack"],
                "damaged_road_marking_list": class_lists["damaged_road_marking"],
                "good_sign_board_list": class_lists["good_sign_board"],
                "drain_issue_list": class_lists["drain_issue"],
                "defective_culvert_list": class_lists["defective_culvert"],
                "good_culvert_list": class_lists["good_culvert"],
                "frames": frames_list,
            }

            # Save to disk
            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved: {total_detections} detections for {video_id}")

            # ── 8. Save to DB ─────────────────────────────────────────────────
            self._send_progress(loop, video_id, 90, "Saving to database...")
            all_detections_flat = list(confirmed.values())
            _t0 = time.perf_counter()
            self._save_to_db(video_id, all_detections_flat)
            _t_db = time.perf_counter() - _t0

            process_end = time.time()
            elapsed = process_end - process_start
            logger.info(
                f"GeminiVideoDetector completed for {video_id} in {elapsed:.1f}s — "
                f"{total_detections} detections"
            )

            # ── Perf report → perf.log ────────────────────────────────────────
            _avg = lambda t, n: (t / n * 1000) if n else 0.0
            _warn = lambda t: " ⚠️" if t > 60 else ""
            perf_logger.info(
                "\n"
                f"{'=' * 78}\n"
                f"  PERF REPORT — [{video_id}]  mode=gemini_video\n"
                f"{'=' * 78}\n"
                f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}\n"
                f"  {'-' * 30} {'-' * 10} {'-' * 7} {'-' * 15}\n"
                f"  {'File upload + poll':<30} {_t_upload:>10.3f} {'N/A':>7} {'N/A':>15}{_warn(_t_upload)}\n"
                f"  {'Gemini API inference':<30} {_t_gemini_api:>10.3f} {'N/A':>7} {'N/A':>15}{_warn(_t_gemini_api)}\n"
                f"  {'Window scan (YOLOE)':<30} {_t_window_scan:>10.3f} {_n_window_scan:>7} {_avg(_t_window_scan, _n_window_scan):>15.2f}{_warn(_t_window_scan)}\n"
                f"  {'GPS coord lookup':<30} {_t_gps:>10.3f} {_n_gps:>7} {_avg(_t_gps, _n_gps):>15.2f}\n"
                f"  {'DB bulk write':<30} {_t_db:>10.3f} {'N/A':>7} {'N/A':>15}\n"
                f"  {'-' * 30} {'-' * 10} {'-' * 7} {'-' * 15}\n"
                f"  TOTAL pipeline time               {elapsed:.3f}s\n"
                f"  Video duration                    {video_duration:.1f}s  ({total_frames} frames @ {fps:.0f} FPS)\n"
                f"  Detections saved                  {total_detections}\n"
                f"{'=' * 78}"
            )

            # ── 9. Complete ───────────────────────────────────────────────────
            processing_status[video_id] = {"status": "completed", "progress": 100}
            self._send_ws(
                loop,
                video_id,
                {
                    "type": "complete",
                    "status": "completed",
                    "summary": results["summary"],
                },
            )
            return results

        except Exception as e:
            logger.error(
                f"GeminiVideoDetector error for {video_id}: {e}", exc_info=True
            )
            processing_status[video_id] = {"status": "error", "message": str(e)}
            self._send_ws(loop, video_id, {"type": "error", "message": str(e)})
            raise

        finally:
            # Clean up the uploaded file from Gemini
            try:
                if "uploaded_file" in dir() and uploaded_file:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.info(f"Cleaned up Gemini file: {uploaded_file.name}")
            except Exception:
                pass

    # ── Helper methods ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw_text: str) -> list:
        """Extract a JSON array from the Gemini response text."""
        # Try to find JSON array in the response
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            logger.warning("No JSON array found in Gemini response")
            return []

        try:
            data = json.loads(match.group())
            if not isinstance(data, list):
                logger.warning("Gemini response JSON is not a list")
                return []
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response JSON: {e}")
            return []

    @staticmethod
    def _parse_timestamp(ts: str) -> float:
        """Convert MM:SS or HH:MM:SS or MM:SS.mmm to seconds as float."""
        parts = ts.strip().split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            elif len(parts) == 3:
                return (
                    float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
                )
        except (ValueError, IndexError):
            pass
        return 0.0

    @staticmethod
    def _rescale_bbox(box_2d: list, width: int, height: int) -> dict:
        """
        Rescale Gemini's normalized [ymin, xmin, ymax, xmax] (0-1000)
        to pixel coordinates {x1, y1, x2, y2}, then expand by 2× (center-
        preserving) so YOLOE has a generous search region around the object.
        The result is clamped to the actual frame dimensions.
        """
        if not box_2d or len(box_2d) != 4:
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

        ymin, xmin, ymax, xmax = box_2d
        px1 = int(xmin / 1000 * width)
        py1 = int(ymin / 1000 * height)
        px2 = int(xmax / 1000 * width)
        py2 = int(ymax / 1000 * height)

        # Expand 2× around the center: output spans cx ± box_w and cy ± box_h,
        # which doubles each side relative to the original box dimensions.
        cx = (px1 + px2) / 2
        cy = (py1 + py2) / 2
        box_w = px2 - px1  # full original box width  → 2× expansion delta
        box_h = py2 - py1  # full original box height → 2× expansion delta

        return {
            "x1": max(0, int(cx - box_w)),
            "y1": max(0, int(cy - box_h)),
            "x2": min(width, int(cx + box_w)),
            "y2": min(height, int(cy + box_h)),
        }

    @staticmethod
    def _extract_snapshot(
        video_path: str, timestamp_sec: float, video_id: str, det_id: int
    ) -> str | None:
        """DEPRECATED: Now handled inline with _get_frame_at_time for refinement."""
        pass

    @staticmethod
    def _get_frame_at_time(video_path: str, timestamp_sec: float):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        except Exception as e:
            logger.warning(
                f"Failed to explicitly extract frame at {timestamp_sec}s: {e}"
            )
            return None

    # ── Gemini class → YOLOE prompt keyword mapping ─────────────────────────────
    # Keywords are substrings matched against YOLOE's free-text prompt labels.
    # Expanded from real log analysis — add any new labels seen in the ↳ lines.
    GEMINI_TO_YOLOE_KEYWORDS = {
        "defected_sign_board": [
            "signboard",
            "circular traffic sign",
            "triangular",
            "prohibitory",
            "round",
            # observed in logs: model uses 'circular' standalone, 'erased', 'convex', etc.
            "circular",
            "no parking",
            "erased",
            "convex",
            "rectangular road",
            "faded triangular",
            "faded rectangular",
            "damaged triangle",
        ],
        "good_sign_board": [
            "signboard",
            "circular traffic sign",
            "triangular",
            "prohibitory",
            "round",
            "circular",
            "no parking",
            "rectangular road",
        ],
        # pothole: keep tight — 'aggregate'/'raveling' labels are road surface degradation NOT potholes
        "pothole": [
            "pothole",
            "pothole with",
            "deep pothole",
            "shallow pothole",
            "multiple pothole",
        ],
        "road_crack": ["crack", "pavement crack"],
        # disabled: YOLOE has no lane/crosswalk-marking prompts
        "damaged_road_marking": [],
        # Classes below have no YOLOE prompts — window scan returns Gemini-timestamp frame
        "drain_issue": [],
        "defective_culvert": [],
        "good_culvert": [],
    }

    # Max fraction of frame area a YOLOE box may cover before being rejected
    # (prevents whole-frame false-positives from being used as a bbox)
    _YOLOE_MAX_BOX_AREA_FRACTION = 0.70

    @staticmethod
    def _find_best_frame_in_window(

        video_path: str,
        timestamp_sec: float,
        gemini_bbox: dict,
        target_class: str,
        video_duration: float,
        window_sec: float = 1.0,
        step_sec: float = 0.25,
    ) -> tuple:
        """
        Scan frames in [timestamp_sec - window_sec, timestamp_sec + window_sec]
        at step_sec intervals and return the (frame, refined_bbox, best_ts) where
        YOLOE yielded the highest-confidence match for target_class.

        For classes with no YOLOE prompts, falls back immediately to the single
        frame at timestamp_sec (no extra I/O cost).

        Returns (frame_bgr, bbox_dict, actual_timestamp_sec).
        If no suitable frame is found, returns (None, gemini_bbox, timestamp_sec).
        """
        match_keywords = GeminiVideoDetector.GEMINI_TO_YOLOE_KEYWORDS.get(
            target_class, []
        )

        # No YOLOE prompts for this class — just return the single frame
        if not match_keywords:
            frame = GeminiVideoDetector._get_frame_at_time(video_path, timestamp_sec)
            return frame, gemini_bbox, timestamp_sec

        offsets = []
        t = -window_sec
        while t <= window_sec + 1e-9:
            offsets.append(round(t, 4))
            t += step_sec

        best_frame = None
        best_bbox = gemini_bbox
        best_ts = timestamp_sec
        best_conf = -1.0

        # Open the video once and reuse
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, gemini_bbox, timestamp_sec

            from app.helpers.yoloe_helper import load_yoloe_model
            model = load_yoloe_model()

            for offset in offsets:
                probe_ts = timestamp_sec + offset
                if probe_ts < 0 or probe_ts > video_duration:
                    continue

                cap.set(cv2.CAP_PROP_POS_MSEC, probe_ts * 1000)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                results = model.predict(frame, conf=0.10, verbose=False)
                r = results[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                names = r.names
                fh, fw = frame.shape[:2]
                is_sign = "sign" in target_class
                min_conf = 0.40 if is_sign else 0.10

                for box, cls_id, conf in zip(boxes, class_ids, confs):
                    prompt_name = names.get(cls_id, "").lower()
                    if not any(kw in prompt_name for kw in match_keywords):
                        continue
                    if conf < min_conf:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    area_frac = ((x2 - x1) * (y2 - y1)) / (fw * fh)
                    if area_frac > GeminiVideoDetector._YOLOE_MAX_BOX_AREA_FRACTION:
                        continue

                    if conf > best_conf:
                        best_conf = conf
                        best_frame = frame.copy()
                        best_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        best_ts = probe_ts
                        logger.info(
                            f"[Window Scan] offset={offset:+.2f}s, ts={probe_ts:.2f}s "
                            f"→ '{prompt_name}' conf={conf:.2f} box=({x1},{y1})-({x2},{y2})"
                        )

            cap.release()
        except Exception as e:
            logger.error(f"[Window Scan] Error: {e}", exc_info=True)
            return None, gemini_bbox, timestamp_sec

        if best_frame is not None:
            logger.info(
                f"[Window Scan] ✅ Best frame for '{target_class}' at ts={best_ts:.2f}s "
                f"(offset={best_ts - timestamp_sec:+.2f}s, conf={best_conf:.2f})"
            )
        else:
            # No YOLOE hit anywhere in the window — return the Gemini-timestamp frame
            logger.warning(
                f"[Window Scan] No YOLOE match in ±{window_sec}s window for "
                f"'{target_class}'. Using Gemini timestamp frame."
            )
            best_frame = GeminiVideoDetector._get_frame_at_time(video_path, timestamp_sec)

        return best_frame, best_bbox, best_ts


    


    @staticmethod
    def _find_nearest_gps(detection_time: float, gps_points: list) -> dict:
        """Return the GPS point whose timestamp is closest to detection_time."""
        if not gps_points:
            return {"lat": None, "lng": None}

        timestamps = [p.get("timestamp", 0) for p in gps_points]
        idx = bisect.bisect_left(timestamps, detection_time)

        if idx == 0:
            nearest = gps_points[0]
        elif idx >= len(gps_points):
            nearest = gps_points[-1]
        else:
            before, after = gps_points[idx - 1], gps_points[idx]
            nearest = (
                before
                if abs(before.get("timestamp", 0) - detection_time)
                <= abs(after.get("timestamp", 0) - detection_time)
                else after
            )

        return {"lat": nearest.get("lat"), "lng": nearest.get("lng")}

    @staticmethod
    def _load_gps_data(json_path: str) -> list:
        """Load GPS points from the uploaded JSON file."""
        if not json_path or not Path(json_path).exists():
            return []
        try:
            with open(json_path, "r") as f:
                gps_data = json.load(f)
                gps_points = gps_data.get("gpsPoints", [])
                logger.info(f"Loaded {len(gps_points)} GPS points from {json_path}")
                return gps_points
        except Exception as e:
            logger.warning(f"Failed to load GPS data: {e}")
            return []

    def _save_to_db(self, video_id: str, all_detections: list):
        """Bulk-insert detections into the database."""
        db = SessionLocal()
        try:
            video = db.query(Video).filter(Video.id == video_id).first()
            video_chainage_id = video.chainage_id if video else None
            video_package_id = None
            video_project_id = None
            video_direction = None

            if video_chainage_id and video.chainage:
                video_package_id = video.chainage.package_id
                video_direction = video.chainage.direction
                if video.chainage.package:
                    video_project_id = video.chainage.package.project_id

            db_detections = []
            for det in all_detections:
                lat, lng = det.get("lat"), det.get("lng")
                project_id = video_project_id
                package_id = video_package_id
                chainage_id = video_chainage_id

                if lat and lng:
                    chainage = find_chainage_by_gps(
                        db,
                        lat,
                        lng,
                        package_id=video_package_id,
                        direction=video_direction,
                    )
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
                crud.create_detections_bulk(db, db_detections)
                logger.info(
                    f"Saved {len(db_detections)} Gemini detections to DB for {video_id}"
                )
        except Exception as e:
            logger.error(f"GeminiVideoDetector DB save error: {e}")
        finally:
            db.close()

    def _send_progress(self, loop, video_id: str, progress: int, message: str = ""):
        """Update in-memory status and send WebSocket progress message."""
        processing_status[video_id]["progress"] = progress
        processing_status[video_id]["status"] = "processing"
        if message:
            processing_status[video_id]["message"] = message
        self._send_ws(
            loop,
            video_id,
            {
                "type": "progress",
                "progress": progress,
                "message": message,
            },
        )

    @staticmethod
    def _send_ws(loop, video_id: str, payload: dict):
        """Send a WebSocket message via the event loop."""
        asyncio.run_coroutine_threadsafe(
            manager.send_message(video_id, payload),
            loop,
        )
