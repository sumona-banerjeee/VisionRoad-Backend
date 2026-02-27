import torch
from transformers import Sam3Processor, Sam3Model
import cv2
import numpy as np
from PIL import Image
import os
import sys
import time
import json
import uuid
import math
from collections import deque, Counter
from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 🛑 HF TOKEN
# ==============================================================================
HF_TOKEN = str(os.getenv("HF_TOKEN"))

# CONFIG
INPUT_VIDEO = r"Test\video\drone-vid-2.mp4"
OUTPUT_VIDEO = r"Test\output\output_video_v1.mp4"
OUTPUT_JSON = r"Test\output\output_report_v1.json"

# FPS: read from video; clamp to [10, 60] to avoid the OpenCV 120fps codec bug
FPS_CLAMP_MIN = 10
FPS_CLAMP_MAX = 60

PROMPTS = [
    "broken traffic signboard",
    "discolored traffic signboard",
    "pothole",
    "puddle",
    "road crack",
    "manhole cover",
    "damaged road marking",
]
MANHOLE_IDX = PROMPTS.index("manhole cover")

INFERENCE_SIZE = 1024
BOX_THRESHOLD = 0.6
SHOW_PREVIEW = False
BATCH_SIZE = 4

COMPILE_MODEL = False if os.name == "nt" else True

COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 128),
]

# ==============================================================================
# TEMPORAL TRACKER — stabilises flickering labels across frames
# ==============================================================================
TRACK_DIST_THRESH = 80  # px — max centroid distance to match same object
TRACK_HISTORY_LEN = 5  # frames of label history kept per track
TRACK_EVICT_FRAMES = 10  # frames of absence before track is removed


class TemporalTracker:
    """
    Lightweight centroid tracker.
    Matches new detections to existing tracks by Euclidean distance.
    Keeps a rolling window of labels per track and returns majority-vote label.
    """

    def __init__(self):
        self.tracks = {}  # track_id -> dict
        self.next_id = 0
        self.current_frame = 0

    def update(self, detections):
        """
        detections: list of dicts with keys 'label', 'cx', 'cy', 'score', 'bbox', 'area'
        Returns same list with 'stable_label' and 'track_id' added.
        """
        self.current_frame += 1
        matched_track_ids = set()

        for det in detections:
            cx, cy = det["cx"], det["cy"]
            best_id, best_dist = None, float("inf")

            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                dist = math.hypot(cx - track["cx"], cy - track["cy"])
                if dist < TRACK_DIST_THRESH and dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                # update existing track
                track = self.tracks[best_id]
                track["cx"] = cx
                track["cy"] = cy
                track["label_history"].append(det["label"])
                track["last_seen"] = self.current_frame
                stable = Counter(track["label_history"]).most_common(1)[0][0]
                track["stable_label"] = stable
                det["stable_label"] = stable
                det["track_id"] = best_id
                matched_track_ids.add(best_id)
            else:
                # new track
                tid = self.next_id
                self.next_id += 1
                history = deque([det["label"]], maxlen=TRACK_HISTORY_LEN)
                self.tracks[tid] = {
                    "cx": cx,
                    "cy": cy,
                    "label_history": history,
                    "stable_label": det["label"],
                    "last_seen": self.current_frame,
                }
                det["stable_label"] = det["label"]
                det["track_id"] = tid

        # evict stale tracks
        stale = [
            tid
            for tid, t in self.tracks.items()
            if self.current_frame - t["last_seen"] > TRACK_EVICT_FRAMES
        ]
        for tid in stale:
            del self.tracks[tid]

        return detections


# ==============================================================================
# PROMPT → JSON key mapping  
# ==============================================================================
PROMPT_TO_KEY = {
    "broken traffic signboard": "broken_traffic_signboard",
    "discolored traffic signboard": "discolored_traffic_signboard",
    "pothole": "pothole",
    "puddle": "puddle",
    "road crack": "road_crack",
    "manhole cover": "manhole_cover",
    "damaged road marking": "damaged_road_marking",
}


def main():
    if "PASTE" in HF_TOKEN:
        print("❌ Please paste your token in the script.")
        return
    try:
        login(token=HF_TOKEN)
    except Exception:
        pass

    device = "cuda"
    if not torch.cuda.is_available():
        print("❌ CUDA not detected.")
        return

    print(f"✅ Hardware Detected: {torch.cuda.get_device_name(0)}")

    # RTX optimisations — unchanged from v0
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    print("⏳ Loading SAM 3 into VRAM (with TF32 Acceleration)...")
    try:
        model = Sam3Model.from_pretrained(
            "facebook/sam3",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)

        processor = Sam3Processor.from_pretrained("facebook/sam3")

        if COMPILE_MODEL:
            model = torch.compile(model, mode="reduce-overhead")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if not os.path.exists(INPUT_VIDEO):
        print("❌ Input video not found.")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- FIX 1: Smart FPS detection ----------------------------------------
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = raw_fps if FPS_CLAMP_MIN <= raw_fps <= FPS_CLAMP_MAX else 30.0
    duration = total / output_fps if output_fps > 0 else 0.0
    # -------------------------------------------------------------------------

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, output_fps, (width, height))
    print(
        f"🎬 {width}x{height} | FPS: {output_fps:.1f} (raw={raw_fps:.1f}) | "
        f"Frames: {total} | Duration: {duration:.1f}s | Batch: {BATCH_SIZE}"
    )

    # ---- FIX 2: Tracking + counting state -----------------------------------
    tracker = TemporalTracker()
    defect_counts = {PROMPT_TO_KEY[p]: 0 for p in PROMPTS}  # running unique count
    detection_id_counter = 0

    # Per-class lists and frame log
    class_lists = {PROMPT_TO_KEY[p]: [] for p in PROMPTS}
    frames_log = []  # only frames WITH detections
    cumulative_counts = {PROMPT_TO_KEY[p]: 0 for p in PROMPTS}
    # -------------------------------------------------------------------------

    frame_count = 0
    frames_buffer = []
    original_frames = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                image_pil.thumbnail((INFERENCE_SIZE, INFERENCE_SIZE))

                frames_buffer.append(image_pil)
                original_frames.append(frame)

            if len(frames_buffer) == BATCH_SIZE or (not ret and len(frames_buffer) > 0):

                t0_cpu = time.time()
                batch_images, batch_texts = [], []
                for img in frames_buffer:
                    batch_images.extend([img] * len(PROMPTS))
                    batch_texts.extend(PROMPTS)

                inputs = processor(
                    images=batch_images, text=batch_texts, return_tensors="pt"
                )
                inputs = inputs.to(device)
                if inputs["pixel_values"].dtype != torch.bfloat16:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                t1_cpu = time.time()
                cpu_prep_time = t1_cpu - t0_cpu

                # --- GPU INFERENCE ---
                t0_gpu = time.time()
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    outputs = model(**inputs)

                torch.cuda.synchronize()
                t1_gpu = time.time()
                gpu_time = t1_gpu - t0_gpu

                # --- POST-PROCESS & DRAW ---
                t0_post = time.time()
                target_sizes = [(height, width)] * len(batch_texts)
                results = processor.post_process_instance_segmentation(
                    outputs, threshold=BOX_THRESHOLD, target_sizes=target_sizes
                )

                for b_idx in range(len(frames_buffer)):
                    global_frame_id = frame_count + b_idx + 1
                    frame_results = results[
                        b_idx * len(PROMPTS) : (b_idx + 1) * len(PROMPTS)
                    ]
                    base_frame = original_frames[b_idx]

                    # --- manhole mask (unchanged logic) ---
                    manhole_res = frame_results[MANHOLE_IDX]
                    manhole_masks = manhole_res["masks"].cpu().numpy()
                    manhole_scores = manhole_res["scores"].float().cpu().numpy()

                    unified_manhole_mask = np.zeros((height, width), dtype=bool)
                    for m, s in zip(manhole_masks, manhole_scores):
                        if s >= BOX_THRESHOLD:
                            unified_manhole_mask |= m.astype(bool)

                    # --- collect raw detections for this frame ---
                    raw_detections = []

                    for i, result in enumerate(frame_results):
                        if i == MANHOLE_IDX:
                            continue

                        prompt_name = PROMPTS[i]
                        masks = result["masks"].cpu().numpy()
                        scores = result["scores"].float().cpu().numpy()

                        for mask, score in zip(masks, scores):
                            if score < BOX_THRESHOLD:
                                continue

                            mask_bool = mask.astype(bool)

                            # pothole-manhole suppression (unchanged)
                            if prompt_name == "pothole":
                                overlap_area = (mask_bool & unified_manhole_mask).sum()
                                total_area = mask_bool.sum()
                                if total_area > 0 and (overlap_area / total_area) > 0.4:
                                    continue
                                mask_bool = mask_bool & (~unified_manhole_mask)

                            if not mask_bool.any():
                                continue

                            # compute centroid & bbox from mask
                            ys, xs = np.where(mask_bool)
                            cx = int(xs.mean())
                            cy = int(ys.mean())
                            x1 = int(xs.min())
                            y1 = int(ys.min())
                            x2 = int(xs.max())
                            y2 = int(ys.max())
                            area = int(mask_bool.sum())

                            raw_detections.append(
                                {
                                    "label": prompt_name,
                                    "cx": cx,
                                    "cy": cy,
                                    "score": float(score),
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "area": area,
                                    "mask_bool": mask_bool,
                                }
                            )

                    # ---- FIX 2: temporal smoothing -------------------------
                    stable_detections = tracker.update(raw_detections)
                    # --------------------------------------------------------

                    frame_json_detections = []

                    for det in stable_detections:
                        stable_label = det["stable_label"]
                        key = PROMPT_TO_KEY[stable_label]
                        score = det["score"]
                        bbox = det["bbox"]
                        mask_bool = det["mask_bool"]

                        # ---- FIX 3: class-wise counting --------------------
                        detection_id_counter += 1
                        defect_counts[key] += 1
                        cumulative_counts[key] = defect_counts[key]
                        # -----------------------------------------------------

                        # add to per-class list 
                        class_lists[key].append(
                            {
                                "detection_id": detection_id_counter,
                                "type": key,
                                "first_detected_frame": global_frame_id,
                                "first_detected_time": round(
                                    global_frame_id / output_fps, 2
                                ),
                                "confidence": round(score, 3),
                                "bbox": bbox,
                                "center": {"x": det["cx"], "y": det["cy"]},
                                "area": det["area"],
                                "track_id": det.get("track_id", -1),
                            }
                        )

                        frame_json_detections.append(
                            {
                                "frame_id": global_frame_id,
                                "detection_id": detection_id_counter,
                                "type": key,
                                "stable_label": stable_label,
                                "confidence": round(score, 3),
                                "count": dict(cumulative_counts),
                                "bbox": bbox,
                                "center": {"x": det["cx"], "y": det["cy"]},
                                "area": det["area"],
                            }
                        )

                        # --- DRAW  ---
                        color = COLORS[PROMPTS.index(stable_label) % len(COLORS)]
                        mask_uint8 = (mask_bool * 255).astype(np.uint8)

                        colored_layer = np.zeros_like(base_frame, dtype=np.uint8)
                        colored_layer[:] = color
                        base_frame = cv2.addWeighted(
                            base_frame,
                            1.0,
                            cv2.bitwise_and(
                                colored_layer, colored_layer, mask=mask_uint8
                            ),
                            0.5,
                            0,
                        )

                        contours, _ = cv2.findContours(
                            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(base_frame, contours, -1, color, 2)

                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            extTop = tuple(c[c[:, :, 1].argmin()][0])
                            label_text = f"{stable_label} {score:.2f}"
                            cv2.putText(
                                base_frame,
                                label_text,
                                (extTop[0], extTop[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2,
                            )

                    if frame_json_detections:
                        frames_log.append(
                            {
                                "frame_id": global_frame_id,
                                "detections": frame_json_detections,
                            }
                        )

                    # also draw manhole detections (unchanged, no counting — it's an exclusion class)
                    for m, s in zip(manhole_masks, manhole_scores):
                        if s < BOX_THRESHOLD:
                            continue
                        mask_bool = m.astype(bool)
                        mask_uint8 = (mask_bool * 255).astype(np.uint8)
                        color = COLORS[MANHOLE_IDX % len(COLORS)]
                        colored_layer = np.zeros_like(base_frame, dtype=np.uint8)
                        colored_layer[:] = color
                        base_frame = cv2.addWeighted(
                            base_frame,
                            1.0,
                            cv2.bitwise_and(
                                colored_layer, colored_layer, mask=mask_uint8
                            ),
                            0.5,
                            0,
                        )
                        contours, _ = cv2.findContours(
                            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(base_frame, contours, -1, color, 2)
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            extTop = tuple(c[c[:, :, 1].argmin()][0])
                            cv2.putText(
                                base_frame,
                                f"manhole cover {s:.2f}",
                                (extTop[0], extTop[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2,
                            )

                    out.write(base_frame)

                frame_count += len(frames_buffer)

                t1_post = time.time()
                post_time = t1_post - t0_post

                sys.stdout.write(
                    f"\rFrame: {frame_count}/{total} | "
                    f"GPU: {gpu_time:.3f}s | "
                    f"CPU Prep: {cpu_prep_time:.3f}s | "
                    f"Draw: {post_time:.3f}s | "
                    f"Counts: { {k: v for k, v in defect_counts.items() if v > 0} }  "
                )
                sys.stdout.flush()

                frames_buffer.clear()
                original_frames.clear()

            if not ret:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # ---- FIX 4: JSON output -------
    total_defects = sum(defect_counts.values())
    frames_with_detections = len(frames_log)
    detection_rate = (
        round((frames_with_detections / frame_count) * 100, 2)
        if frame_count > 0
        else 0.0
    )

    video_id = str(uuid.uuid4())
    report = {
        "video_id": video_id,
        "video_path": os.path.abspath(INPUT_VIDEO),
        "detection_type": "sam3-road-damage",
        "processed_at": datetime.now().isoformat(),
        "video_info": {
            "total_frames": frame_count,
            "fps": output_fps,
            "duration": round(frame_count / output_fps, 2) if output_fps > 0 else 0,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
        },
        "summary": {
            "total_frames": frame_count,
            "total_defects": total_defects,
            "frames_with_detections": frames_with_detections,
            "detection_rate": detection_rate,
            **{f"unique_{k}": v for k, v in defect_counts.items()},
        },
        # per-class detection lists
        **{f"{k}_list": class_lists[k] for k in class_lists},
        # per-frame log (only frames with detections)
        "frames": frames_log,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # -------------------------------------------------------------------------

    # final summary table
    print(f"\n\n{'='*55}")
    print(f"  ✅  SAM 3 v1 — Detection Summary")
    print(f"{'='*55}")
    print(f"  {'Class':<35} {'Count':>6}")
    print(f"  {'-'*41}")
    for k, v in defect_counts.items():
        print(f"  {k:<35} {v:>6}")
    print(f"  {'-'*41}")
    print(f"  {'TOTAL':<35} {total_defects:>6}")
    print(f"{'='*55}")
    print(f"  📹  Video  → {OUTPUT_VIDEO}")
    print(f"  📄  Report → {OUTPUT_JSON}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
