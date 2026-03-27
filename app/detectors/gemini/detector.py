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
from google.genai import types

from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.models.video import Video
from app.db.crud_hierarchy import find_chainage_by_gps

logger = logging.getLogger(__name__)

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

            if uploaded_file.state.name == "FAILED":
                raise Exception(f"Gemini file processing failed: {uploaded_file.state}")

            logger.info(f"File ready: {uploaded_file.name}")

            # ── 3. Call Gemini generate_content (with retry for rate limits) ──
            self._send_progress(loop, video_id, 40, "Analyzing video with Gemini...")

            logger.info("Sending generate_content request to Gemini...")

            MAX_RETRIES = 3
            response = None
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

                # Frame number
                frame_number = max(1, int(timestamp_sec * fps))

                # Extract frame and refine bbox with YOLOE for tight pixel-level coordinates
                snapshot_path = None
                frame_bgr = self._get_frame_at_time(video_path, timestamp_sec)
                if frame_bgr is not None:
                    bbox = self._refine_bbox_with_yoloe(frame_bgr, bbox, label)

                # GPS lookup
                gps_coords = self._find_nearest_gps(timestamp_sec, gps_points)

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
            self._save_to_db(video_id, all_detections_flat)

            process_end = time.time()
            elapsed = process_end - process_start
            logger.info(
                f"GeminiVideoDetector completed for {video_id} in {elapsed:.1f}s — "
                f"{total_detections} detections"
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

        # Expand 2× around the center (half-width/height doubled)
        cx = (px1 + px2) / 2
        cy = (py1 + py2) / 2
        half_w = px2 - px1  # original half-width × 2 = full original width
        half_h = py2 - py1  # original half-height × 2 = full original height

        return {
            "x1": max(0, int(cx - half_w)),
            "y1": max(0, int(cy - half_h)),
            "x2": min(width, int(cx + half_w)),
            "y2": min(height, int(cy + half_h)),
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
        # disabled: YOLOE has no lane/crosswalk-marking prompts — all misses show
        # 'aggregate', 'raveling', 'asphalt' labels which are NOT road markings.
        "damaged_road_marking": [],
        # Classes below have no YOLOE prompts — refinement will be skipped
        "drain_issue": [],
        "defective_culvert": [],
        "good_culvert": [],
    }

    # Max fraction of frame area a YOLOE box may cover before being rejected
    # (prevents whole-frame false-positives from being used as a bbox)
    _YOLOE_MAX_BOX_AREA_FRACTION = 0.70

    @staticmethod
    def _refine_bbox_with_yoloe(
        frame_bgr, gemini_bbox: dict, target_class: str
    ) -> dict:
        """
        Use the loaded YOLOEv11m-Seg open-vocabulary model on a single frame
        to snap the bounding box to the perfect pixel boundaries of the object.

        Strategy:
          1. Check if YOLOE has prompts that can locate this Gemini class
          2. Run single-frame inference
          3. For each YOLOE detection, check if its prompt contains any keyword
             associated with the target Gemini class (lenient matching)
          4. Pick the closest matching box to Gemini's center
        """
        from app.helpers.yoloe_helper import load_yoloe_model

        # Check if this Gemini class can be refined by YOLOE at all
        match_keywords = GeminiVideoDetector.GEMINI_TO_YOLOE_KEYWORDS.get(
            target_class, []
        )
        if not match_keywords:
            logger.info(
                f"[YOLOE Refine] Skipping refinement for '{target_class}' — no YOLOE prompts available."
            )
            return gemini_bbox

        logger.info(
            f"[YOLOE Refine] Start: target_class='{target_class}', "
            f"gemini_box=({gemini_bbox['x1']},{gemini_bbox['y1']})-({gemini_bbox['x2']},{gemini_bbox['y2']}), "
            f"match_keywords={match_keywords}"
        )

        try:
            model = load_yoloe_model()
            results = model.predict(frame_bgr, conf=0.10, verbose=False)

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                logger.warning(
                    "[YOLOE Refine] No detections found by YOLOE in this frame."
                )
                return gemini_bbox

            boxes = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            names = r.names

            logger.info(f"[YOLOE Refine] YOLOE found {len(boxes)} detections in frame.")

            # Gemini bbox center (from the 2.0x expanded box)
            gx_center = (gemini_bbox["x1"] + gemini_bbox["x2"]) / 2
            gy_center = (gemini_bbox["y1"] + gemini_bbox["y2"]) / 2

            best_box = None
            best_conf = 0.0
            min_dist = float("inf")
            best_prompt = ""

            for box, cls_id, conf in zip(boxes, class_ids, confs):
                prompt_name = names.get(cls_id, "").lower()

                # Lenient matching: does this YOLOE prompt contain ANY keyword
                # associated with the target Gemini category?
                prompt_matches = any(kw in prompt_name for kw in match_keywords)

                if not prompt_matches:
                    continue

                # Apply confidence threshold: stricter for sign boards
                is_sign = "sign" in target_class
                min_conf = 0.40 if is_sign else 0.10
                if conf < min_conf:
                    logger.info(
                        f"[YOLOE Refine] Skipping '{prompt_name}' — "
                        f"conf={conf:.2f} < threshold={min_conf}"
                    )
                    continue

                x1, y1, x2, y2 = map(int, box)

                # FIX: Reject whole-frame boxes (YOLOE texture match on the whole image)
                fh, fw = frame_bgr.shape[:2]
                box_area_frac = ((x2 - x1) * (y2 - y1)) / (fw * fh)
                max_frac = GeminiVideoDetector._YOLOE_MAX_BOX_AREA_FRACTION
                if box_area_frac > max_frac:
                    logger.info(
                        f"[YOLOE Refine] Skipping '{prompt_name}' — "
                        f"box covers {box_area_frac:.0%} of frame (> {max_frac:.0%} limit)"
                    )
                    continue

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dist = (cx - gx_center) ** 2 + (cy - gy_center) ** 2

                logger.info(
                    f"[YOLOE Refine] Candidate: prompt='{prompt_name}', "
                    f"conf={conf:.2f}, box=({x1},{y1})-({x2},{y2}), "
                    f"area={box_area_frac:.0%}, dist={dist:.0f}"
                )

                if dist < min_dist:
                    min_dist = dist
                    best_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    best_conf = conf
                    best_prompt = prompt_name

            # Accept if within ~632px radius (400,000 sq px)
            if best_box and min_dist < 400000:
                logger.info(
                    f"[YOLOE Refine] ✅ SUCCESS: Replaced Gemini box with YOLOE box. "
                    f"prompt='{best_prompt}', conf={best_conf:.2f}, dist={min_dist:.0f}"
                )
                return best_box
            elif best_box:
                logger.warning(
                    f"[YOLOE Refine] ⚠ Match found but too far away "
                    f"(dist={min_dist:.0f} > 400000). Keeping Gemini box."
                )
            else:
                logger.warning(
                    f"[YOLOE Refine] ✗ No YOLOE detection matched keywords "
                    f"{match_keywords} for '{target_class}'. Keeping Gemini box."
                )
                # Log what YOLOE actually found — use this to expand keywords
                all_labels = [names.get(int(c), "") for c in class_ids]
                logger.warning(
                    f"[YOLOE Refine]   ↳ YOLOE labels in this frame: {all_labels} "
                    f"(confs: {[round(float(c),2) for c in confs]})"
                )

        except Exception as e:
            logger.error(f"[YOLOE Refine] Error: {e}", exc_info=True)

        return gemini_bbox

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
