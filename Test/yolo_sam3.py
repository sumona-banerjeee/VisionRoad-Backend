"""
YOLO + SAM3 Verification — Standalone Test Script
===================================================
Pipeline:
  1. YOLO tracks every frame (botsort) and collects detection candidates
  2. Once a candidate crosses the confidence / multi-frame threshold it is
     "optimistically confirmed" (same logic as the backend YOLO+VL detector)
  3. Instead of sending a crop to the cloud VL API, we crop the region and
     run a SAM3 text-prompted segmentation locally on GPU.
  4. If SAM3 finds a mask with score >= SAM3_VERIFY_THRESHOLD the detection
     is marked verified.  If it does NOT find a mask the detection is
     REJECTED and removed.
  5. A final JSON report + annotated video are written to Test/output/.

Usage:
    python Test/yolo_sam3.py
    python Test/yolo_sam3.py --video Test/video/pothole.mp4

Requirements (already in your venv from SAM3 work):
    ultralytics, transformers, torch, supervision, huggingface_hub,
    python-dotenv, opencv-python, Pillow, numpy
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import argparse
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from datetime import datetime

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from ultralytics import YOLO

load_dotenv()

# ==============================================================================
# CONFIG  —  edit these or pass via env / CLI
# ==============================================================================
YOLO_MODEL_PATH = r"models\final-v1.pt"
DEFAULT_INPUT_VIDEO = r"Test\video\pothole.mp4"
OUTPUT_VIDEO = r"Test\output\yolo_sam3_output.mp4"
OUTPUT_JSON = r"Test\output\yolo_sam3_report.json"

# YOLO settings (mirrors backend defaults)
YOLO_CONF_THRESHOLD = 0.50
YOLO_IMGSZ = 640
TRACKER = "botsort.yaml"
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))

# SAM3 settings
SAM3_MODEL_ID = "facebook/sam3"
SAM3_VERIFY_THRESHOLD = float(os.getenv("SAM3_VERIFY_THRESHOLD", "0.55"))
SAM3_MIN_BBOX_SIZE = int(os.getenv("SAM3_MIN_BBOX_SIZE", "30"))
SAM3_CROP_MAX_SIZE = 512  # resize crop before SAM3 inference
HF_TOKEN = os.getenv("HF_TOKEN", "")
COMPILE_MODEL = False if os.name == "nt" else True  # torch.compile off on Windows

# Detection / tracking thresholds (mirrors backend)
HIGH_CONF_THRESHOLD = 0.75
LOW_CONF_MIN_FRAMES = 2
MIN_DISTANCE_THRESHOLD = 120  # px between spatial duplicates
# Time-window fractions are computed from video duration at runtime

# Draw settings
BBOX_COLORS = {
    "pothole": (0, 0, 255),
    "road_crack": (0, 165, 255),
    "defected_sign_board": (0, 255, 255),
    "good_sign_board": (0, 255, 0),
    "damaged_road_marking": (255, 0, 255),
}

# ==============================================================================
# YOLO CLASS → SAM3 TEXT PROMPT  (one or more prompts per YOLO class)
# ==============================================================================
YOLO_TO_SAM3_PROMPTS = {
    "pothole": ["pothole", "road puddle"],
    "road_crack": ["road crack", "pavement crack"],
    "defected_sign_board": ["broken traffic signboard", "discolored traffic signboard"],
    "good_sign_board": ["traffic signboard"],
    "damaged_road_marking": ["damaged road marking", "faded road marking"],
}

# ==============================================================================
# ROAD DAMAGE CLASSES (mirrors backend)
# ==============================================================================
ROAD_DAMAGE_CLASSES = {
    "defected_sign_board",
    "pothole",
    "road_crack",
    "damaged_road_marking",
}
EXCLUDED_CLASSES = {"good_sign_board"}
ALL_CLASSES = ROAD_DAMAGE_CLASSES | EXCLUDED_CLASSES

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yolo_sam3")


# ==============================================================================
# SAM3 VERIFIER CLASS  —  loaded once, called per-detection crop
# ==============================================================================
class Sam3Verifier:
    """
    Wraps facebook/sam3 for crop-level verification.

    Call verify(crop_bgr, yolo_class) → dict
        {
          "sam3_agrees":   bool,
          "sam3_score":    float,   # best mask score across all prompts
          "sam3_prompt":   str,     # which prompt matched (or "none")
          "sam3_elapsed":  float,   # seconds
        }
    Thread-safe after __init__ (stateless per call).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.info(f"⏳ Loading SAM3 ({SAM3_MODEL_ID}) onto {device} …")
        t0 = time.time()

        if HF_TOKEN and "PASTE" not in HF_TOKEN:
            try:
                login(token=HF_TOKEN, add_to_git_credential=False)
            except Exception as e:
                logger.warning(f"HF login warning: {e}")

        self.model = Sam3Model.from_pretrained(
            SAM3_MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)
        self.model.eval()

        if COMPILE_MODEL:
            logger.info("torch.compile — pass: reduce-overhead")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
        logger.info(f"✅ SAM3 loaded in {time.time() - t0:.1f}s")

    @torch.inference_mode()
    def verify(self, crop_bgr: np.ndarray, yolo_class: str) -> dict:
        """Run SAM3 on the crop and return verification result."""
        t0 = time.time()

        prompts = YOLO_TO_SAM3_PROMPTS.get(yolo_class, [yolo_class])

        # Convert BGR crop → PIL RGB
        h, w = crop_bgr.shape[:2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        # Resize so the largest edge ≤ SAM3_CROP_MAX_SIZE
        pil_img.thumbnail((SAM3_CROP_MAX_SIZE, SAM3_CROP_MAX_SIZE))

        best_score = 0.0
        best_prompt = "none"

        for prompt_text in prompts:
            try:
                inputs = self.processor(
                    images=[pil_img],
                    text=[prompt_text],
                    return_tensors="pt",
                )
                inputs = {
                    k: (
                        v.to(self.device, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in inputs.items()
                }
                # Cast pixel_values to bfloat16 to match model dtype
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(**inputs)

                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=SAM3_VERIFY_THRESHOLD
                    * 0.8,  # looser threshold at post-process
                    target_sizes=[(pil_img.height, pil_img.width)],
                )

                if results and len(results[0]["masks"]) > 0:
                    scores = results[0]["scores"].float().cpu().numpy()
                    top_score = float(scores.max())
                    if top_score > best_score:
                        best_score = top_score
                        best_prompt = prompt_text

                del outputs, inputs
                torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f"SAM3 verify error for prompt '{prompt_text}': {e}")

        agrees = best_score >= SAM3_VERIFY_THRESHOLD
        elapsed = time.time() - t0

        icon = "✓" if agrees else "✗"
        logger.info(
            f"SAM3 {icon} [{elapsed:.2f}s] yolo={yolo_class!r} "
            f"best_prompt={best_prompt!r} score={best_score:.3f} "
            f"threshold={SAM3_VERIFY_THRESHOLD}"
        )
        return {
            "sam3_agrees": agrees,
            "sam3_score": round(best_score, 4),
            "sam3_prompt": best_prompt,
            "sam3_elapsed": round(elapsed, 3),
        }


# ==============================================================================
# HELPERS
# ==============================================================================
def calc_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def is_spatial_duplicate(cx, cy, cls, now, spatial_locations, time_thresh, dist_thresh):
    for loc in spatial_locations:
        if (
            loc["class"] == cls
            and calc_distance((cx, cy), loc["center"]) < dist_thresh
            and (now - loc["time"]) < time_thresh
        ):
            return True
    return False


def draw_detection(frame, x1, y1, x2, y2, label, color, verified=None):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Verification badge
    if verified is True:
        badge, badge_color = "SAM3✓", (0, 200, 0)
    elif verified is False:
        badge, badge_color = "SAM3✗", (0, 0, 200)
    else:
        badge, badge_color = "PENDING", (200, 200, 0)
    full_label = f"{label} [{badge}]"
    cv2.putText(
        frame,
        full_label,
        (x1, max(y1 - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        badge_color,
        2,
    )


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run(video_path: str):
    # ── Validate paths ───────────────────────────────────────────────────────
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    if not os.path.exists(YOLO_MODEL_PATH):
        logger.error(f"YOLO model not found: {YOLO_MODEL_PATH}")
        sys.exit(1)
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available — SAM3 will run on CPU (very slow)")

    # ── Load models ──────────────────────────────────────────────────────────
    logger.info(f"Loading YOLO from {YOLO_MODEL_PATH} …")
    yolo = YOLO(YOLO_MODEL_PATH)
    # NOTE: do NOT call yolo.model.half() here — Ultralytics runs Conv+BN fusion
    # (fuse()) on the first track() call and it requires FP32 weights at that point.
    # FP16 inference is enabled by passing half=True directly to yolo.track() below.
    logger.info("YOLO ready.")

    verifier = Sam3Verifier(device=device)

    # ── Video setup ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    logger.info(
        f"Video: {width}x{height} | {total_frames} frames | "
        f"{fps:.1f} FPS | {duration:.1f}s | FRAME_SKIP={FRAME_SKIP}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Adaptive thresholds (mirror backend)
    TIME_WINDOW = duration * 0.25
    TIME_THRESHOLD = duration * 0.30
    ROI_TOP = int(height * 0.05)
    ROI_BOTTOM = int(height * 0.95)

    # ── State ─────────────────────────────────────────────────────────────────
    tracker_history = defaultdict(lambda: deque(maxlen=50))
    confirmed = {}  # tid → detection record
    rejected_tids = set()
    counted_ids = {cls: set() for cls in ALL_CLASSES}
    spatial_locations = []
    tracker_class_lock = {}

    # Async SAM3 verification — same pattern as backend VL async
    sam3_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sam3_verify")
    pending_sam3 = {}  # tid → {"future": Future, "class_name": str}
    sam3_cache = {}  # tid → result (prevent re-submission)

    # Stats
    stats = {
        "sam3_verified": 0,
        "sam3_rejected": 0,
        "sam3_errors": 0,
        "spatial_dup": 0,
        "roi_outside": 0,
        "class_mismatch": 0,
        "total_sam3_time": 0.0,
    }

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
        sam3_verified=None,
        sam3_score=None,
        sam3_prompt=None,
    ):
        confirmed[tid] = {
            "detection_id": tid,
            "type": class_name,
            "first_detected_frame": frame_count,
            "first_detected_time": round(current_time, 2),
            "confidence": round(float(conf), 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center": {"x": cx, "y": cy},
            "sam3_verified": sam3_verified,
            "sam3_score": sam3_score,
            "sam3_prompt": sam3_prompt,
        }
        if class_name in counted_ids:
            counted_ids[class_name].add(tid)
        spatial_locations.append(
            {"center": (cx, cy), "time": current_time, "class": class_name}
        )

    def _process_completed_sam3_futures():
        """Poll pending SAM3 futures; apply verify/reject retroactively."""
        done_tids = []
        for tid, meta in pending_sam3.items():
            future = meta["future"]
            if not future.done():
                continue
            done_tids.append(tid)

            try:
                result = future.result(timeout=0)
            except Exception as e:
                logger.warning(f"SAM3 future error tid={tid}: {e}")
                stats["sam3_errors"] += 1
                # Leave detection as-is (no SAM3 data)
                continue

            if not result:
                stats["sam3_errors"] += 1
                continue

            sam3_cache[tid] = result
            stats["total_sam3_time"] += result.get("sam3_elapsed", 0.0)

            if result["sam3_agrees"]:
                stats["sam3_verified"] += 1
                if tid in confirmed:
                    confirmed[tid]["sam3_verified"] = True
                    confirmed[tid]["sam3_score"] = result["sam3_score"]
                    confirmed[tid]["sam3_prompt"] = result["sam3_prompt"]
            else:
                # SAM3 disagrees — reject the detection
                stats["sam3_rejected"] += 1
                logger.info(
                    f"SAM3 REJECT tid={tid} class={meta['class_name']!r} "
                    f"score={result['sam3_score']:.3f}"
                )
                if tid in confirmed:
                    old_cls = confirmed[tid]["type"]
                    counted_ids.get(old_cls, set()).discard(tid)
                    del confirmed[tid]
                rejected_tids.add(tid)

        for tid in done_tids:
            del pending_sam3[tid]

    # ── Frame loop ────────────────────────────────────────────────────────────
    frame_count = 0
    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
            out_vid.write(frame)  # write original (unprocessed) frame
            continue

        current_time = frame_count / fps

        # YOLO tracking
        results = yolo.track(
            frame,
            persist=True,
            conf=YOLO_CONF_THRESHOLD,
            tracker=TRACKER,
            verbose=False,
            device=device,
            imgsz=YOLO_IMGSZ,
            half=(device.startswith("cuda")),
        )

        # Poll pending SAM3 futures
        _process_completed_sam3_futures()

        draw_frame = frame.copy()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for tid, cid, box, conf in zip(track_ids, class_ids, boxes, confidences):
                tid, cid = int(tid), int(cid)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = str(yolo.names[cid])

                # ROI filter
                if not (0 < cx < width and ROI_TOP < cy < ROI_BOTTOM):
                    stats["roi_outside"] += 1
                    continue

                # Class lock (prevent flip-flopping)
                if tid in tracker_class_lock:
                    if tracker_class_lock[tid] != class_name:
                        stats["class_mismatch"] += 1
                        continue
                else:
                    tracker_class_lock[tid] = class_name

                # Track history
                tracker_history[tid].append(current_time)
                recent = [
                    t for t in tracker_history[tid] if current_time - t <= TIME_WINDOW
                ]
                min_needed = 1 if conf >= HIGH_CONF_THRESHOLD else LOW_CONF_MIN_FRAMES

                # Candidate graduation
                if (
                    len(recent) >= min_needed
                    and tid not in confirmed
                    and tid not in rejected_tids
                ):
                    if not is_spatial_duplicate(
                        cx,
                        cy,
                        class_name,
                        current_time,
                        spatial_locations,
                        TIME_THRESHOLD,
                        MIN_DISTANCE_THRESHOLD,
                    ):
                        # Optimistically confirm
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

                        # Submit async SAM3 verification (if bbox is large enough)
                        bbox_w, bbox_h = x2 - x1, y2 - y1
                        if (
                            bbox_w >= SAM3_MIN_BBOX_SIZE
                            and bbox_h >= SAM3_MIN_BBOX_SIZE
                            and tid not in sam3_cache
                            and tid not in pending_sam3
                        ):
                            crop = frame[y1:y2, x1:x2].copy()
                            cls_copy = class_name
                            future = sam3_executor.submit(
                                verifier.verify, crop, cls_copy
                            )
                            pending_sam3[tid] = {
                                "future": future,
                                "class_name": cls_copy,
                            }
                            logger.info(
                                f"SAM3 submitted tid={tid} class={class_name!r} "
                                f"bbox={bbox_w}x{bbox_h}px"
                            )
                        else:
                            # Bbox too small — auto-accept without verification
                            logger.debug(
                                f"SAM3 skipped (small bbox {bbox_w}x{bbox_h}) tid={tid}"
                            )
                    else:
                        stats["spatial_dup"] += 1

                # Draw if confirmed
                if tid in confirmed:
                    color = BBOX_COLORS.get(class_name, (200, 200, 200))
                    sam3v = confirmed[tid].get("sam3_verified")
                    label = f"[{tid}] {class_name} {conf:.2f}"
                    draw_detection(draw_frame, x1, y1, x2, y2, label, color, sam3v)

        # Progress
        pct = int((frame_count / total_frames) * 100)
        if frame_count % 30 == 0 or frame_count == total_frames:
            counts_str = " | ".join(
                f"{cls}:{len(counted_ids[cls])}"
                for cls in ALL_CLASSES
                if len(counted_ids[cls]) > 0
            )
            sys.stdout.write(
                f"\r[{pct:3d}%] Frame {frame_count}/{total_frames} | "
                f"confirmed={len(confirmed)} | pending_sam3={len(pending_sam3)} | {counts_str}  "
            )
            sys.stdout.flush()

        out_vid.write(draw_frame)

    print()  # newline after progress

    # ── Drain remaining SAM3 futures ──────────────────────────────────────────
    if pending_sam3:
        logger.info(f"Draining {len(pending_sam3)} pending SAM3 futures …")
        remaining = [m["future"] for m in pending_sam3.values()]
        futures_wait(remaining, timeout=60)
        _process_completed_sam3_futures()
        # Cancel anything still pending
        for tid in list(pending_sam3.keys()):
            pending_sam3[tid]["future"].cancel()
            stats["sam3_errors"] += 1
        pending_sam3.clear()

    sam3_executor.shutdown(wait=False)
    cap.release()
    out_vid.release()

    # ── Build final report ────────────────────────────────────────────────────
    t_elapsed = time.time() - t_start
    total_damage = sum(len(counted_ids[c]) for c in ROAD_DAMAGE_CLASSES)

    detection_list = list(confirmed.values())

    report = {
        "video_id": str(uuid.uuid4()),
        "video_path": os.path.abspath(video_path),
        "detection_mode": "yolo_sam3",
        "processed_at": datetime.now().isoformat(),
        "video_info": {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration": round(duration, 2),
            "width": width,
            "height": height,
        },
        "summary": {
            "total_confirmed": len(confirmed),
            "total_road_damage": total_damage,
            "unique_pothole": len(counted_ids["pothole"]),
            "unique_road_crack": len(counted_ids["road_crack"]),
            "unique_defected_sign_board": len(counted_ids["defected_sign_board"]),
            "unique_damaged_road_marking": len(counted_ids["damaged_road_marking"]),
            "unique_good_sign_board": len(counted_ids["good_sign_board"]),
        },
        "sam3_stats": {
            "verified": stats["sam3_verified"],
            "rejected": stats["sam3_rejected"],
            "errors": stats["sam3_errors"],
            "total_verify_time_s": round(stats["total_sam3_time"], 2),
            "avg_verify_time_ms": round(
                (
                    stats["total_sam3_time"]
                    / max(stats["sam3_verified"] + stats["sam3_rejected"], 1)
                )
                * 1000,
                1,
            ),
        },
        "rejection_stats": {
            "spatial_duplicate": stats["spatial_dup"],
            "roi_outside": stats["roi_outside"],
            "class_mismatch": stats["class_mismatch"],
            "sam3_rejected": stats["sam3_rejected"],
        },
        "detections": detection_list,
        "perf": {
            "wall_time_s": round(t_elapsed, 2),
            "frames_processed": frame_count,
            "avg_ms_per_frame": round((t_elapsed / max(frame_count, 1)) * 1000, 1),
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Summary print ─────────────────────────────────────────────────────────
    W = 62
    print(f"\n{'=' * W}")
    print(f"  ✅  YOLO + SAM3 Verification — Results Summary")
    print(f"{'=' * W}")
    print(f"  Detection Mode   : yolo_sam3")
    print(f"  Video            : {os.path.basename(video_path)}")
    print(f"  Frames processed : {frame_count} / {total_frames}")
    print(f"{'─' * W}")
    print(f"  {'Class':<35} {'Count':>6}")
    print(f"  {'─' * (W - 4)}")
    for cls in sorted(ALL_CLASSES):
        cnt = len(counted_ids[cls])
        if cnt > 0:
            print(f"  {cls:<35} {cnt:>6}")
    print(f"  {'─' * (W - 4)}")
    print(f"  {'TOTAL ROAD DAMAGE':<35} {total_damage:>6}")
    print(f"\n{'─' * W}")
    print(f"  SAM3 Verification Results:")
    print(f"    ✓ Verified   : {stats['sam3_verified']}")
    print(f"    ✗ Rejected   : {stats['sam3_rejected']}")
    print(f"    ⚠ Errors     : {stats['sam3_errors']}")
    if stats["sam3_verified"] + stats["sam3_rejected"] > 0:
        avg_ms = report["sam3_stats"]["avg_verify_time_ms"]
        print(f"    ⏱ Avg verify : {avg_ms:.1f} ms/detection")
    print(f"{'─' * W}")
    print(f"  ⏱  Wall time  : {t_elapsed:.1f}s")
    print(f"  📹  Video     → {OUTPUT_VIDEO}")
    print(f"  📄  Report    → {OUTPUT_JSON}")
    print(f"{'=' * W}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO + SAM3 verification pipeline (standalone test)"
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_INPUT_VIDEO,
        help=f"Path to input video (default: {DEFAULT_INPUT_VIDEO})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SAM3_VERIFY_THRESHOLD,
        help=f"SAM3 score threshold for verification (default: {SAM3_VERIFY_THRESHOLD})",
    )
    args = parser.parse_args()

    # Allow CLI override of threshold
    SAM3_VERIFY_THRESHOLD = args.threshold

    run(args.video)
