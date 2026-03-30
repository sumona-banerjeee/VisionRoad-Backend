"""
GeminiVideoDetector — Gemini 1.5 based video road-inspection detection with YOLOE-seg hybrid refinement.

Instead of frame-by-frame YOLO inference, this detector:
  1. Uploads the whole video to the Gemini Files API.
  2. Sends a single generate_content() call with a structured prompt.
  3. Parses the JSON response (timestamps, labels, bounding boxes).
  4. Extracts the sharpest frame snapshot around each detection timestamp.
  5. Runs YOLOE segmentation on that frame to get precise masks and boxes.
  6. Maps timestamps to GPS coords and saves results in the standard format.

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
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.core.logging_config import perf_logger
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.models.video import Video
from app.db.crud_hierarchy import find_chainage_by_gps
from app.helpers.yoloe_helper import load_yoloe_model, process_frame_with_yoloe

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

# ── Prompt System Instruction ──────────────────────────────────────────────────
DETECTION_SYSTEM_INSTRUCTION = """You are a road maintenance inspector AI with expertise in computer vision.
Your task is to analyze dashcam video footage and detect all occurrences of road defects and relevant infrastructure.

Target categories and their visual characteristics:
- pothole: Circular or irregular depressions/holes on the road surface.
- road_crack: Longitudinal or transverse separations in the pavement surface.
- damaged_road_marking: Faded, peeling, or missing white/yellow lane markings.
- defected_sign_board: Tilting, bent, faded, or graffiti-covered road signs.
- good_sign_board: Clear, upright, and legible road signs.
- drain_issue: Blocked, broken, or overflowing road-side drainage systems.
- defective_culvert: Structurally compromised small bridge/tunnel under the road.
- good_culvert: Intact and functional small bridge/tunnel under the road.

Your goal is to provide high-precision bounding boxes that TIGHTLY enclose the visible portion of the defect or object.
"""

DETECTION_TASK_PROMPT = """Analyze the entire dashcam video and identify ALL occurrences of the target categories.

For EACH object, determine its full appearance duration.
1. Find the exact MIDDLE timestamp of its appearance (exactly halfway between start and end).
2. Write a brief 'spatial_reasoning' step describing exactly where the defect is located in the image frame (e.g., bottom-left, center, top-right).
3. For that MIDDLE frame, identify the tightest possible bounding box based on your spatial reasoning. Output as [ymin, xmin, ymax, xmax] normalized to (0-1000).
4. Assign a confidence rating (high, medium, low).
5. Provide a very brief description of the defect or object.

Rules:
- Timestamps MUST be in MM:SS format.
- Normalized coordinates MUST be integers [0-1000].
- Bounding boxes should be conservative yet complete, covering the entire visible extent of the anomaly.
- Return ONLY a JSON array of objects.
- Ensure 'spatial_reasoning' appears BEFORE 'box_2d' in the JSON object to enable Chain of Thought.

Output Format Example:
[
  {
    "timestamp": "01:23",
    "label": "pothole",
    "description": "Medium-sized pothole in center lane",
    "spatial_reasoning": "The pothole is directly in front of the camera, appearing in the lower-middle section of the frame.",
    "box_2d": [750, 420, 880, 580],
    "confidence": "high"
  }
]
"""


def get_gemini_executor() -> ThreadPoolExecutor:
    """Return the Gemini processing executor (for lifespan shutdown)."""
    return _executor


class GeminiVideoDetector:
    """
    Video-level road inspection detector using Gemini 1.5 + YOLOE-seg refinement.
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it in your .env file."
            )
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.yoloe_model = None
        self.detection_mode = "gemini_yoloe_hybrid"
        logger.info(
            f"GeminiVideoDetector ready — model: {GEMINI_MODEL} + YOLOE"
        )

    def _ensure_yoloe_model(self):
        """Ensure the YOLOE model is loaded."""
        if self.yoloe_model is None:
            self.yoloe_model = load_yoloe_model()

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
            self._ensure_yoloe_model()
            self._send_ws(loop, video_id, {
                "type": "status", "status": "processing", "progress": 0
            })

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
            logger.info(f"File uploaded: name={uploaded_file.name}, state={uploaded_file.state}")

            # Poll until active
            self._send_progress(loop, video_id, 20, "Waiting for Gemini to process file...")
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
                logger.debug(f"File state: {uploaded_file.state}")

            if uploaded_file.state.name == "FAILED":
                raise Exception(
                    f"Gemini file processing failed: {uploaded_file.state}"
                )

            logger.info(f"File ready: {uploaded_file.name}")

            # ── 3. Call Gemini generate_content ─────────────────────────────
            self._send_progress(loop, video_id, 40, "Analyzing video with Gemini...")

            logger.info("Sending generate_content request to Gemini (with JSON mode)...")

            MAX_RETRIES = 3
            response = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = self.client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[uploaded_file, DETECTION_TASK_PROMPT],
                        config=types.GenerateContentConfig(
                            system_instruction=DETECTION_SYSTEM_INSTRUCTION,
                            response_mime_type="application/json",
                            temperature=0.0,
                        )
                    )
                    break 
                except Exception as api_err:
                    err_str = str(api_err)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        retry_match = re.search(r'retry\w*\s*in\s*([\d.]+)', err_str, re.IGNORECASE)
                        wait_secs = float(retry_match.group(1)) if retry_match else 60.0
                        wait_secs = max(wait_secs, 30.0)

                        if attempt < MAX_RETRIES:
                            logger.warning(
                                f"Rate limited (attempt {attempt}/{MAX_RETRIES}), "
                                f"retrying in {wait_secs:.0f}s..."
                            )
                            self._send_progress(
                                loop, video_id, 45,
                                f"Rate limited — retrying in {int(wait_secs)}s "
                                f"(attempt {attempt}/{MAX_RETRIES})..."
                            )
                            time.sleep(wait_secs)
                        else:
                            raise
                    else:
                        raise

            if not response:
                raise Exception("Failed to get response from Gemini after all retry attempts.")

            raw_text = response.text or ""
            logger.info(f"Gemini response received ({len(raw_text)} chars)")
            
            self._send_progress(loop, video_id, 60, "Parsing detections...")

            # ── 4. Parse JSON from response ───────────────────────────────────
            detections = self._parse_response(raw_text)
            logger.info(f"Parsed {len(detections)} detections from Gemini response")

            # ── 5. Process detections ─────────────────────────────────────────
            self._send_progress(loop, video_id, 70, "Refining with YOLOE segmentation...")

            confirmed = {}
            counted_ids = {cat: set() for cat in DETECTION_CATEGORIES}
            det_id = 0

            for det in detections:
                label = det.get("label", "").strip()
                if label not in DETECTION_CATEGORIES:
                    continue

                det_id += 1
                timestamp_str = det.get("timestamp", "00:00")
                timestamp_sec = self._parse_timestamp(timestamp_str)
                timestamp_sec = min(timestamp_sec, video_duration)

                # Gemini's box
                box_2d = det.get("box_2d", [0, 0, 1000, 1000])
                bbox = self._rescale_bbox(box_2d, width, height)

                # Extract best sharp frame
                snapshot_result = self._extract_best_snapshot(
                    video_path, timestamp_sec, video_id, det_id
                )
                
                snapshot_path = None
                frame = None
                if snapshot_result:
                    snapshot_path, frame = snapshot_result

                # YOLOE Refinement
                mask = None
                if frame is not None:
                    yoloe_dets = process_frame_with_yoloe(self.yoloe_model, frame)
                    # Find best IoU match
                    best_match = None
                    max_iou = 0.3 # threshold for matching
                    
                    for y_det in yoloe_dets:
                        iou = self._calculate_iou(bbox, y_det["bbox"])
                        if iou > max_iou:
                            max_iou = iou
                            best_match = y_det
                    
                    if best_match:
                        logger.info(f"YOLOE match for {label} (IoU={max_iou:.2f})")
                        x1, y1, x2, y2 = best_match["bbox"]
                        bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        mask = best_match.get("mask")

                confidence = CONFIDENCE_MAP.get(det.get("confidence", "medium").lower(), 0.75)
                frame_number = max(1, int(timestamp_sec * float(fps)))
                gps_coords = self._find_nearest_gps(timestamp_sec, gps_points)

                confirmed[det_id] = {
                    "detection_id": det_id,
                    "type": label,
                    "first_detected_frame": frame_number,
                    "first_detected_time": float(f"{timestamp_sec:.2f}"),
                    "confidence": confidence,
                    "bbox": bbox,
                    "snapshot_path": snapshot_path,
                    "description": det.get("description", ""),
                    "spatial_reasoning": det.get("spatial_reasoning", ""),
                    "lat": gps_coords.get("lat"),
                    "lng": gps_coords.get("lng"),
                    "mask": mask,
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
                for c in ["defected_sign_board", "pothole", "road_crack", "damaged_road_marking", "drain_issue"]
            )

            # ── 7. Assemble result JSON ───────────────────────────────────────
            # Build frames array for playback
            frames_dict = {}
            for info in confirmed.values():
                f_id = info["first_detected_frame"]
                if f_id not in frames_dict:
                    frames_dict[f_id] = {"frame_id": f_id, "detections": []}
                
                b = info["bbox"]
                det_entry = {
                    "frame_id": f_id,
                    "detection_id": info["detection_id"],
                    "type": info["type"],
                    "confidence": info["confidence"],
                    "bbox": b,
                    "center": {"x": (b["x1"] + b["x2"]) // 2, "y": (b["y1"] + b["y2"]) // 2},
                    "mask": info.get("mask"),
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
                    "fps": float(f"{fps:.2f}"),
                    "duration": float(f"{video_duration:.2f}"),
                    "width": width,
                    "height": height,
                },
                "summary": {
                    "total_road_damage": total_road_damage,
                    "total_detections": total_detections,
                    "unique_pothole": len(counted_ids["pothole"]),
                    "unique_defected_sign_board": len(counted_ids["defected_sign_board"]),
                },
                "frames": frames_list,
                "defected_sign_board_list": class_lists["defected_sign_board"],
                "pothole_list": class_lists["pothole"],
                "road_crack_list": class_lists["road_crack"],
                "damaged_road_marking_list": class_lists["damaged_road_marking"],
                "good_sign_board_list": class_lists["good_sign_board"],
                "drain_issue_list": class_lists["drain_issue"],
                "defective_culvert_list": class_lists["defective_culvert"],
                "good_culvert_list": class_lists["good_culvert"],
            }

            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            # ── 8. Save to DB ─────────────────────────────────────────────────
            self._send_progress(loop, video_id, 90, "Saving to database...")
            self._save_to_db(video_id, list(confirmed.values()))

            process_end = time.time()
            logger.info(f"GeminiVideoDetector completed in {process_end - process_start:.1f}s")

            processing_status[video_id] = {"status": "completed", "progress": 100}
            self._send_ws(loop, video_id, {"type": "complete", "status": "completed", "summary": results["summary"]})
            return results

        except Exception as e:
            logger.error(f"GeminiVideoDetector error: {e}", exc_info=True)
            processing_status[video_id] = {"status": "error", "message": str(e)}
            self._send_ws(loop, video_id, {"type": "error", "message": str(e)})
            raise
        finally:
            try:
                if "uploaded_file" in locals() and uploaded_file:
                    self.client.files.delete(name=uploaded_file.name)
            except: pass

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw_text: str) -> list:
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if not match: return []
        try:
            return json.loads(match.group())
        except: return []

    @staticmethod
    def _parse_timestamp(ts: str) -> float:
        parts = ts.strip().split(":")
        try:
            if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
            if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except: pass
        return 0.0

    @staticmethod
    def _rescale_bbox(box_2d: list, width: int, height: int) -> dict:
        if not box_2d or len(box_2d) != 4: return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        ymin, xmin, ymax, xmax = map(float, box_2d)
        return {
            "x1": int(max(0, xmin / 1000 * width)),
            "y1": int(max(0, ymin / 1000 * height)),
            "x2": int(min(width, xmax / 1000 * width)),
            "y2": int(min(height, ymax / 1000 * height)),
        }

    @staticmethod
    def _calculate_iou(box1: dict, box2_tuple: tuple) -> float:
        x1, y1, x2, y2 = box1["x1"], box1["y1"], box1["x2"], box1["y2"]
        bx1, by1, bx2, by2 = box2_tuple
        xi1, yi1, xi2, yi2 = max(x1, bx1), max(y1, by1), min(x2, bx2), min(y2, by2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (bx2 - bx1) * (by2 - by1)
        return inter / (area1 + area2 - inter + 1e-9)

    @staticmethod
    def _extract_best_snapshot(video_path, timestamp_sec, video_id, det_id):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return None
            best_frame, best_score = None, -1
            for offset in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp_sec + offset) * 1000)
                ret, frame = cap.read()
                if not ret: continue
                score = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                if score > best_score:
                    best_score, best_frame = score, frame
            cap.release()
            if best_frame is not None:
                path = SNAPSHOT_DIR / f"{video_id}_det_{det_id}.jpg"
                cv2.imwrite(str(path), best_frame)
                return str(path), best_frame
        except: pass
        return None

    @staticmethod
    def _find_nearest_gps(t, points):
        if not points: return {"lat": None, "lng": None}
        times = [p.get("timestamp", 0) for p in points]
        idx = bisect.bisect_left(times, t)
        if idx == 0: n = points[0]
        elif idx >= len(points): n = points[-1]
        else:
            b, a = points[idx-1], points[idx]
            n = b if abs(b.get("timestamp",0)-t) <= abs(a.get("timestamp",0)-t) else a
        return {"lat": n.get("lat"), "lng": n.get("lng")}

    @staticmethod
    def _load_gps_data(path):
        if not Path(path).exists(): return []
        try:
            with open(path, "r") as f: return json.load(f).get("gpsPoints", [])
        except: return []

    def _save_to_db(self, video_id, dets):
        db = SessionLocal()
        try:
            video = db.query(Video).filter(Video.id == video_id).first()
            vc_id = video.chainage_id if video else None
            vp_id, vpr_id, vdir = None, None, None
            if vc_id and video.chainage:
                vp_id, vdir = video.chainage.package_id, video.chainage.direction
                if video.chainage.package: vpr_id = video.chainage.package.project_id

            db_dets = []
            for d in dets:
                lat, lng = d.get("lat"), d.get("lng")
                c_id, p_id, pr_id = vc_id, vp_id, vpr_id
                if lat and lng:
                    ch = find_chainage_by_gps(db, lat, lng, package_id=vp_id, direction=vdir)
                    if ch:
                        c_id, p_id = ch.id, ch.package_id
                        if ch.package: pr_id = ch.package.project_id

                db_det = Detection(
                    video_id=video_id, frame_number=d["first_detected_frame"],
                    timestamp_ms=int(d["first_detected_time"] * 1000),
                    confidence=d["confidence"], detection_type=d["type"], class_name=d["type"],
                    latitude=lat, longitude=lng, project_id=pr_id, package_id=p_id, chainage_id=c_id,
                )
                db_det.set_bounding_box(d.get("bbox", {}))
                db_det.set_segmentation_mask(d.get("mask"))
                db_dets.append(db_det)
            if db_dets:
                crud.create_detections_bulk(db, db_dets)
        except Exception as e: logger.error(f"DB save error: {e}")
        finally: db.close()

    def _send_progress(self, loop, video_id, progress, message=""):
        processing_status[video_id].update({"progress": progress, "status": "processing", "message": message})
        self._send_ws(loop, video_id, {"type": "progress", "progress": progress, "message": message})

    @staticmethod
    def _send_ws(loop, video_id, payload):
        asyncio.run_coroutine_threadsafe(manager.send_message(video_id, payload), loop)
