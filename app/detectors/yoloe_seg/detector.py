"""
YoloeSegDetector — Dual-model detector for comprehensive road feature detection.

Model 1: yoloe-26x-seg-pf.pt (prompt-free, 1200+ built-in classes) →
         pothole, manhole_cover, defected_sign_board, street_light_pole, water_puddle

Model 2: yolo12s_RDD2022_best.pt (trained) → road_crack, pothole

KEY FIX vs previous version:
  - Uses -pf (prompt-free) checkpoint instead of -seg.pt + set_classes()
  - map_yoloe_seg_class() does case-insensitive lookup for -pf model output names
  - No other logic changes — tracking, ROI, dedup all still absent by design.
"""

import cv2
import json
import asyncio
import time
import logging
import torch
import os
import tempfile
from datetime import datetime

import numpy as np

from app.detectors.base.base_detector import BaseDetector, executor
from app.ws.websocket_manager import manager
from app.core.config import processing_status, detection_results, RESULTS_DIR
from app.core.logging_config import PerfTimer, perf_logger
from app.db.database import SessionLocal
from app.db import crud
from app.models.detection import Detection
from app.models.video import Video
from app.db.crud_hierarchy import find_chainage_by_gps
from app.helpers.yoloe_seg_helper import (
    load_yoloe_seg_model,
    load_rdd_crack_model,
    map_yoloe_seg_class,
    map_rdd_crack_class,
    YOLOE_SEG_CONF,
    RDD_CRACK_CONF,
    ALL_CLASSES,
)

logger = logging.getLogger(__name__)

FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
YOLOE_IMGSZ = int(os.getenv("YOLOE_IMGSZ", "640"))
RDD_IMGSZ = int(os.getenv("RDD_IMGSZ", "640"))


class YoloeSegDetector(BaseDetector):
    """
    Dual-model detector:
      - YOLOE-26x-seg-pf for prompt-free road feature detection (matches notebook behavior)
      - RDD2022 for road crack detection
    No tracking, no ROI, no dedup — pure frame-by-frame detection.
    """

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.yoloe_model = None
        self.rdd_model = None
        self.detection_mode = "yoloe_seg"
        logger.info("YoloeSegDetector created (dual-model, prompt-free)")

    def _load_models(self):
        self.yoloe_model = load_yoloe_seg_model()
        self.rdd_model = load_rdd_crack_model()

    def _ensure_models(self):
        if self.yoloe_model is None or self.rdd_model is None:
            self._load_models()

    async def process_video(
        self, video_id: str, video_path: str, json_path: str, speed_kmh: int
    ):
        processing_status[video_id] = {"status": "processing", "progress": 0}
        self._ensure_models()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            self._process_video_blocking,
            video_id, video_path, json_path, speed_kmh, loop,
        )

    def _process_video_blocking(
        self, video_id: str, video_path: str, json_path: str, speed: int, loop
    ):
        cap = None

        try:
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id,
                    {"type": "status", "status": "processing", "progress": 0},
                ),
                loop,
            )

            gps_points = self._load_gps_data(json_path)
            process_start = time.time()

            perf_timings = {
                "frame_decode":    {"total": 0.0, "count": 0},
                "yoloe_inference": {"total": 0.0, "count": 0},
                "rdd_inference":   {"total": 0.0, "count": 0},
                "gps_coord":       {"total": 0.0, "count": 0},
                "db_gps_match":    {"total": 0.0, "count": 0},
                "db_bulk_write":   {"total": 0.0, "count": 0},
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
                f"Processing [yoloe_seg_pf dual-model] for {video_id}: "
                f"{total_frames} frames @ {fps:.1f} FPS, {width}x{height}"
            )

            detection_id_counter = 0
            all_confirmed = []

            _ndjson_fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".ndjson",
                prefix=f"yoloe_seg_{video_id}_",
                dir=str(RESULTS_DIR),
                delete=False,
            )
            _ndjson_path = _ndjson_fd.name
            _frames_written = 0
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
                if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                    continue

                current_time = frame_count / fps
                frame_data = {"frame_id": frame_count, "detections": []}

                # ── Model 1: YOLOE-26x-seg-pf (prompt-free) ────────────────
                _t0_yoloe = time.perf_counter()
                yoloe_results = self.yoloe_model.predict(
                    frame,
                    conf=YOLOE_SEG_CONF,
                    verbose=False,
                    device=self.device,
                    imgsz=YOLOE_IMGSZ,
                )
                _yoloe_elapsed = time.perf_counter() - _t0_yoloe
                perf_timings["yoloe_inference"]["total"] += _yoloe_elapsed
                perf_timings["yoloe_inference"]["count"] += 1

                if yoloe_results[0].boxes is not None and len(yoloe_results[0].boxes) > 0:
                    class_ids    = yoloe_results[0].boxes.cls.cpu().numpy().astype(int)
                    boxes        = yoloe_results[0].boxes.xyxy.cpu().numpy()
                    confidences  = yoloe_results[0].boxes.conf.cpu().numpy()
                    names        = yoloe_results[0].names

                    for cls_id, box, conf in zip(class_ids, boxes, confidences):
                        cls_id = int(cls_id)
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        raw_class_name = names.get(cls_id, f"class_{cls_id}")

                        # map_yoloe_seg_class does a case-insensitive lookup
                        # against YOLOE_PF_CLASS_MAP — returns None for irrelevant classes
                        backend_class = map_yoloe_seg_class(raw_class_name)
                        if backend_class is None:
                            continue

                        detection_id_counter += 1
                        det_data = {
                            "detection_id":         detection_id_counter,
                            "type":                 backend_class,
                            "raw_class":            raw_class_name,
                            "model_source":         "yoloe_seg_pf",
                            "first_detected_frame": frame_count,
                            "first_detected_time":  round(current_time, 2),
                            "confidence":           round(float(conf), 3),
                            "bbox":                 {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        }
                        all_confirmed.append(det_data)
                        total_detections_count += 1

                        frame_data["detections"].append({
                            "frame_id":     frame_count,
                            "detection_id": detection_id_counter,
                            "type":         backend_class,
                            "raw_class":    raw_class_name,
                            "model_source": "yoloe_seg_pf",
                            "confidence":   round(float(conf), 3),
                            "bbox":         {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "center":       {"x": cx, "y": cy},
                        })

                # ── Model 2: RDD2022 (road crack + pothole) ─────────────────
                _t0_rdd = time.perf_counter()
                rdd_results = self.rdd_model.predict(
                    frame,
                    conf=RDD_CRACK_CONF,
                    verbose=False,
                    device=self.device,
                    imgsz=RDD_IMGSZ,
                )
                _rdd_elapsed = time.perf_counter() - _t0_rdd
                perf_timings["rdd_inference"]["total"] += _rdd_elapsed
                perf_timings["rdd_inference"]["count"] += 1

                if rdd_results[0].boxes is not None and len(rdd_results[0].boxes) > 0:
                    class_ids   = rdd_results[0].boxes.cls.cpu().numpy().astype(int)
                    boxes       = rdd_results[0].boxes.xyxy.cpu().numpy()
                    confidences = rdd_results[0].boxes.conf.cpu().numpy()
                    names       = rdd_results[0].names

                    for cls_id, box, conf in zip(class_ids, boxes, confidences):
                        cls_id = int(cls_id)
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        rdd_class_name = names.get(cls_id, f"class_{cls_id}")
                        backend_class  = map_rdd_crack_class(rdd_class_name)
                        if backend_class is None:
                            continue  # skip Repair class

                        detection_id_counter += 1
                        det_data = {
                            "detection_id":         detection_id_counter,
                            "type":                 backend_class,
                            "raw_class":            rdd_class_name,
                            "model_source":         "rdd2022",
                            "first_detected_frame": frame_count,
                            "first_detected_time":  round(current_time, 2),
                            "confidence":           round(float(conf), 3),
                            "bbox":                 {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        }
                        all_confirmed.append(det_data)
                        total_detections_count += 1

                        frame_data["detections"].append({
                            "frame_id":     frame_count,
                            "detection_id": detection_id_counter,
                            "type":         backend_class,
                            "raw_class":    rdd_class_name,
                            "model_source": "rdd2022",
                            "confidence":   round(float(conf), 3),
                            "bbox":         {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "center":       {"x": cx, "y": cy},
                        })

                if frame_data["detections"]:
                    _ndjson_fd.write(json.dumps(frame_data) + "\n")
                    _frames_written += 1

                progress = int((frame_count / total_frames) * 100)
                if progress - last_progress >= 5:
                    self._send_progress(
                        video_id, progress, loop,
                        {"total_detections": total_detections_count},
                    )
                    last_progress = progress

            process_end = time.time()

            # ── Build per-class lists ────────────────────────────────────────
            final_lists = {}
            for cls in ALL_CLASSES:
                cls_list = self._get_class_list(
                    all_confirmed, cls, gps_points, perf_timings
                )
                final_lists[f"{cls}_list"] = cls_list

            all_detections_flat = []
            for cls in ALL_CLASSES:
                all_detections_flat.extend(final_lists[f"{cls}_list"])

            frames_with_detections = _frames_written
            detection_rate = (
                round((frames_with_detections / frame_count) * 100, 2)
                if frame_count > 0 else 0
            )

            frames = self._read_ndjson_frames(_ndjson_fd, _ndjson_path, _frames_written)

            class_counts = {}
            for cls in ALL_CLASSES:
                class_counts[f"unique_{cls}"] = len(final_lists.get(f"{cls}_list", []))

            results = {
                "video_id":       video_id,
                "video_path":     video_path,
                "detection_mode": self.detection_mode,
                "processed_at":   datetime.now().isoformat(),
                "video_info": {
                    "total_frames": total_frames,
                    "fps":          round(fps, 2),
                    "duration":     round(video_duration, 2),
                    "width":        width,
                    "height":       height,
                },
                "summary": {
                    "total_frames":     frame_count,
                    "total_detections": total_detections_count,
                    "detection_rate":   detection_rate,
                    **class_counts,
                },
                "frames": frames,
            }
            results.update(final_lists)

            detection_results[video_id] = results
            with open(RESULTS_DIR / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

            self._save_to_db(video_id, all_detections_flat, perf_timings)

            # ── Perf report ──────────────────────────────────────────────────
            total_time = time.time() - process_start
            frames_processed = (
                frame_count // FRAME_SKIP if FRAME_SKIP > 1 else frame_count
            )

            def _fmt(key):
                d = perf_timings[key]
                t, c = d["total"], d["count"]
                return t, c, (t / c * 1000) if c > 0 else 0.0

            def _flag(t):
                return " ⚠️" if total_time > 0 and (t / total_time) > 0.20 else ""

            fd_t,  fd_c,  fd_avg  = _fmt("frame_decode")
            yi_t,  yi_c,  yi_avg  = _fmt("yoloe_inference")
            ri_t,  ri_c,  ri_avg  = _fmt("rdd_inference")
            gc_t,  gc_c,  gc_avg  = _fmt("gps_coord")
            dg_t,  dg_c,  dg_avg  = _fmt("db_gps_match")
            db_t,  db_c,  db_avg  = _fmt("db_bulk_write")

            report_lines = [
                f"{'=' * 78}",
                f"  PERF REPORT — [{video_id}]  mode=yoloe_seg_pf (dual-model, prompt-free)",
                f"{'=' * 78}",
                f"  {'Stage':<30} {'Total (s)':>10} {'Count':>7} {'Avg/call (ms)':>15}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'Frame decode (I/O)':<30} {fd_t:>10.3f} {fd_c:>7d} {fd_avg:>15.2f}{_flag(fd_t)}",
                f"  {'YOLOE-pf inference':<30} {yi_t:>10.3f} {yi_c:>7d} {yi_avg:>15.2f}{_flag(yi_t)}",
                f"  {'RDD2022 crack inference':<30} {ri_t:>10.3f} {ri_c:>7d} {ri_avg:>15.2f}{_flag(ri_t)}",
                f"  {'GPS coord lookup':<30} {gc_t:>10.3f} {gc_c:>7d} {gc_avg:>15.2f}{_flag(gc_t)}",
                f"  {'DB GPS matching':<30} {dg_t:>10.3f} {dg_c:>7d} {dg_avg:>15.2f}{_flag(dg_t)}",
                f"  {'DB bulk write':<30} {db_t:>10.3f} {db_c:>7d} {db_avg:>15.2f}{_flag(db_t)}",
                f"  {'-'*30} {'-'*10} {'-'*7} {'-'*15}",
                f"  {'TOTAL pipeline time':<30} {total_time:>10.3f}s",
                f"  {'Video duration':<30} {video_duration:>10.1f}s  ({total_frames} frames @ {fps:.0f} FPS)",
                f"  {'Frames processed':<30} {frames_processed:>10d}  (FRAME_SKIP={FRAME_SKIP})",
                f"  {'Detections saved':<30} {len(all_detections_flat):>10d}",
                f"{'=' * 78}",
            ]
            perf_logger.info(f"\n" + "\n".join(report_lines))
            logger.info(f"\n" + "\n".join(report_lines))

            processing_status[video_id] = {"status": "completed", "progress": 100}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id,
                    {
                        "type":    "complete",
                        "status":  "completed",
                        "summary": results["summary"],
                    },
                ),
                loop,
            )
            return results

        except Exception as e:
            logger.error(f"Error processing {video_id}: {e}")
            processing_status[video_id] = {"status": "error", "message": str(e)}
            asyncio.run_coroutine_threadsafe(
                manager.send_message(
                    video_id, {"type": "error", "message": str(e)}
                ),
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
            except OSError:
                pass
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    @staticmethod
    def _read_ndjson_frames(ndjson_fd, ndjson_path: str, expected_count: int = 0) -> list:
        if not ndjson_fd.closed:
            ndjson_fd.close()
        frames = []
        try:
            with open(ndjson_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        frames.append(json.loads(line))
        except Exception:
            pass
        return frames

    def _get_class_list(self, all_detections, class_name, gps_points, perf_timings=None):
        """Build a sorted list of detections for a specific class."""
        lst = []
        for det in all_detections:
            if det["type"] == class_name:
                det_data = {
                    "detection_id":         det["detection_id"],
                    "type":                 det["type"],
                    "model_source":         det.get("model_source", "unknown"),
                    "first_detected_frame": det["first_detected_frame"],
                    "first_detected_time":  det["first_detected_time"],
                    "confidence":           det["confidence"],
                    "bbox":                 det.get("bbox", {}),
                }
                if gps_points:
                    _t0 = time.perf_counter()
                    gps_coords = self.find_nearest_gps(det["first_detected_time"], gps_points)
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
            video = db.query(Video).filter(Video.id == video_id).first()
            video_chainage_id  = video.chainage_id if video else None
            video_package_id   = None
            video_project_id   = None
            video_direction    = None

            if video_chainage_id and video.chainage:
                video_package_id = video.chainage.package_id
                video_direction  = video.chainage.direction
                if video.chainage.package:
                    video_project_id = video.chainage.package.project_id

            db_detections = []
            for det in all_detections:
                lat, lng = det.get("lat"), det.get("lng")
                project_id  = video_project_id
                package_id  = video_package_id
                chainage_id = video_chainage_id

                if lat and lng:
                    _t0 = time.perf_counter()
                    chainage = find_chainage_by_gps(
                        db, lat, lng,
                        package_id=video_package_id,
                        direction=video_direction,
                    )
                    _gps_db_elapsed = time.perf_counter() - _t0
                    if perf_timings is not None:
                        perf_timings["db_gps_match"]["total"] += _gps_db_elapsed
                        perf_timings["db_gps_match"]["count"] += 1
                    if chainage:
                        chainage_id = chainage.id
                        package_id  = chainage.package_id
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
                _t0 = time.perf_counter()
                crud.create_detections_bulk(db, db_detections)
                _bulk_elapsed = time.perf_counter() - _t0
                if perf_timings is not None:
                    perf_timings["db_bulk_write"]["total"] += _bulk_elapsed
                    perf_timings["db_bulk_write"]["count"] += 1
                logger.info(f"Saved {len(db_detections)} detections to DB for {video_id}")
        except Exception as e:
            logger.error(f"Database save error: {e}")
        finally:
            db.close()