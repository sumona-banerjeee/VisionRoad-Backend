"""
SAM 3 Road Damage Detection — Optimized v3
===========================================
Key optimizations over v2:
  1. Threaded video I/O      — read/write frames in background threads (no I/O stalls)
  2. Async CPU prefetch       — tokenize next batch on CPU while GPU runs current batch
  3. CUDA pinned memory       — faster CPU→GPU tensor transfer
  4. Pre-allocated draw buf   — reuse a single colored_layer instead of malloc per mask
  5. Cached bboxes            — store bbox in meta_list to avoid np.where() in IoU loop
  6. Pre-computed color map   — avoid PROMPTS.index() in hot loop
  7. Reduced frame copies     — avoid .copy() when no detections
  8. CUDA non_blocking xfer   — overlap transfer with compute
"""

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
from datetime import datetime
from huggingface_hub import login
from dotenv import load_dotenv
import supervision as sv
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# ==============================================================================
# CONFIG
# ==============================================================================
HF_TOKEN = str(os.getenv("HF_TOKEN"))
INPUT_VIDEO = r"Test\video\drone-vid-2.mp4"
OUTPUT_VIDEO = r"Test\output\output_video_v3.mp4"
OUTPUT_JSON = r"Test\output\output_report_v3.json"

FPS_CLAMP_MIN = 10
FPS_CLAMP_MAX = 60

PROMPTS = [
    "broken traffic signboard",
    "discolored traffic signboard",
    "pothole OR puddle",
    "road crack",
    "manhole cover",
    "damaged road marking",
]
MANHOLE_IDX = PROMPTS.index("manhole cover")
NUM_PROMPTS = len(PROMPTS)

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

# Pre-computed: prompt index → color (avoids PROMPTS.index() in hot loop)
PROMPT_COLOR = {i: COLORS[i % len(COLORS)] for i in range(NUM_PROMPTS)}

PROMPT_TO_KEY = {
    "broken traffic signboard": "broken_traffic_signboard",
    "discolored traffic signboard": "discolored_traffic_signboard",
    "pothole OR puddle": "pothole_or_puddle",
    "road crack": "road_crack",
    "manhole cover": "manhole_cover",
    "damaged road marking": "damaged_road_marking",
}

# Pre-computed: prompt index → key (avoids dict lookup in hot loop)
PROMPT_IDX_TO_KEY = {i: PROMPT_TO_KEY[p] for i, p in enumerate(PROMPTS)}


# ==============================================================================
# OPT 1: THREADED VIDEO READER — reads frames ahead in a background thread
# ==============================================================================
class ThreadedVideoReader:
    def __init__(self, path, queue_size=32):
        self.cap = cv2.VideoCapture(path)
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.queue.put((False, None))
                    return
                self.queue.put((True, frame))
            else:
                time.sleep(0.001)  # back-pressure

    def read(self):
        return self.queue.get()

    def release(self):
        self.stopped = True
        self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)


# ==============================================================================
# OPT 2: THREADED VIDEO WRITER — writes frames in a background thread
# ==============================================================================
class ThreadedVideoWriter:
    def __init__(self, path, fourcc, fps, size):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.queue = Queue(maxsize=64)
        self.stopped = False
        self.thread = Thread(target=self._writer, daemon=True)
        self.thread.start()

    def _writer(self):
        while True:
            frame = self.queue.get()
            if frame is None:  # poison pill
                break
            self.writer.write(frame)
        self.writer.release()

    def write(self, frame):
        self.queue.put(frame)

    def release(self):
        self.queue.put(None)  # poison pill
        self.thread.join()


# ==============================================================================
# OPT 5: CACHED BBOX — build detections with pre-computed bboxes in meta
# ==============================================================================
def build_sv_detections(frame_results, height, width, manhole_mask):
    """
    Turns SAM3 frame_results → sv.Detections (excluding manhole class).
    Returns detections and meta list with PRE-COMPUTED bboxes for IoU matching.
    meta_list entries: (prompt_name, mask_bool, score, (x1, y1, x2, y2))
    """
    xyxy_list, score_list, class_list, mask_list, meta_list = [], [], [], [], []

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

            if prompt_name == "pothole OR puddle":
                total_area = mask_bool.sum()
                overlap_area = (mask_bool & manhole_mask).sum()
                if total_area > 0 and (overlap_area / total_area) > 0.4:
                    continue
                mask_bool = mask_bool & (~manhole_mask)

            if not mask_bool.any():
                continue

            ys, xs = np.where(mask_bool)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())

            xyxy_list.append([x1, y1, x2, y2])
            score_list.append(float(score))
            class_list.append(i)
            mask_list.append(mask_bool)
            # OPT: store bbox so IoU loop doesn't need np.where() again
            meta_list.append((prompt_name, mask_bool, float(score), (x1, y1, x2, y2)))

    if not xyxy_list:
        return None, []

    detections = sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        confidence=np.array(score_list, dtype=np.float32),
        class_id=np.array(class_list, dtype=int),
        mask=np.stack(mask_list).astype(bool),
    )
    return detections, meta_list


# ==============================================================================
# HELPER — draw a single mask overlay + contour + label onto base_frame
# ==============================================================================
def draw_mask_overlay(base_frame, mask_bool, color, label_text, colored_layer):
    """Draws mask overlay, contour, and label. Reuses pre-allocated colored_layer."""
    mask_uint8 = (mask_bool * 255).astype(np.uint8)

    colored_layer[:] = color
    masked_color = cv2.bitwise_and(colored_layer, colored_layer, mask=mask_uint8)
    cv2.addWeighted(base_frame, 1.0, masked_color, 0.5, 0, dst=base_frame)  # in-place

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(base_frame, contours, -1, color, 2)

    if contours:
        c = max(contours, key=cv2.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        cv2.putText(
            base_frame,
            label_text,
            (extTop[0], extTop[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


# ==============================================================================
# HELPER — IoU matching with cached bboxes (no np.where needed)
# ==============================================================================
def match_mask_by_iou(bbox, meta_list, prompt_name):
    """Find best matching mask from meta_list using cached bboxes."""
    best_mask = None
    best_iou = 0.0

    bx1, by1, bx2, by2 = bbox
    area_a = (bx2 - bx1) * (by2 - by1)

    for pn, mb, sc, (mx1, my1, mx2, my2) in meta_list:
        if pn != prompt_name:
            continue

        # IoU between tracked bbox and cached mask bbox
        ix1 = max(bx1, mx1)
        iy1 = max(by1, my1)
        ix2 = min(bx2, mx2)
        iy2 = min(by2, my2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_b = (mx2 - mx1) * (my2 - my1)
        union = area_a + area_b - inter
        iou = inter / union if union > 0 else 0.0

        if iou > best_iou:
            best_iou = iou
            best_mask = mb

    return best_mask


# ==============================================================================
# OPT 3: ASYNC PREFETCH — tokenize batch N+1 on CPU while GPU runs batch N
# ==============================================================================
def prepare_inputs(frames, processor):
    """CPU-only: build processor inputs from a list of PIL images."""
    batch_images, batch_texts = [], []
    for img in frames:
        batch_images.extend([img] * NUM_PROMPTS)
        batch_texts.extend(PROMPTS)

    inputs = processor(images=batch_images, text=batch_texts, return_tensors="pt")
    # Pin memory for faster .to(cuda, non_blocking=True)
    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].pin_memory()
    return inputs


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if "PASTE" in HF_TOKEN:
        print("❌ Please paste your HF token.")
        return
    try:
        login(token=HF_TOKEN)
    except Exception:
        pass

    device = "cuda"
    if not torch.cuda.is_available():
        print("❌ CUDA not detected.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ Hardware Detected: {gpu_name}")

    # Hardware unlocks
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    print("⏳ Loading SAM 3 into VRAM...")
    t_load_start = time.time()
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
    t_model_load = time.time() - t_load_start
    print(f"✅ Model loaded in {t_model_load:.1f}s")

    if not os.path.exists(INPUT_VIDEO):
        print("❌ Input video not found.")
        return

    # --- Video properties (read from regular cap, then use threaded reader)
    probe = cv2.VideoCapture(INPUT_VIDEO)
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_fps = probe.get(cv2.CAP_PROP_FPS)
    probe.release()

    output_fps = raw_fps if FPS_CLAMP_MIN <= raw_fps <= FPS_CLAMP_MAX else 30.0
    duration = round(total / output_fps, 2) if output_fps > 0 else 0.0
    print(
        f"🎬 {width}x{height} | FPS: {output_fps:.1f} (raw={raw_fps:.1f}) | "
        f"Frames: {total} | Duration: {duration}s | Batch: {BATCH_SIZE}"
    )

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # OPT 1: Threaded reader + writer
    cap = ThreadedVideoReader(INPUT_VIDEO, queue_size=BATCH_SIZE * 4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = ThreadedVideoWriter(OUTPUT_VIDEO, fourcc, output_fps, (width, height))

    # ByteTrack — one tracker per class
    trackers = {p: sv.ByteTrack() for p in PROMPTS if p != "manhole cover"}

    # Counting state
    seen_tracks = set()
    defect_counts = {PROMPT_TO_KEY[p]: 0 for p in PROMPTS}
    class_lists = {PROMPT_TO_KEY[p]: [] for p in PROMPTS}
    cumulative = {PROMPT_TO_KEY[p]: 0 for p in PROMPTS}
    frames_log = []
    frame_count = 0
    det_id_ctr = 0

    # Timing accumulators
    t_total_start = time.time()
    acc_cpu_prep = 0.0
    acc_gpu_infer = 0.0
    acc_post_draw = 0.0

    frames_buffer = []  # PIL images for processor
    original_frames = []  # raw BGR frames for drawing

    # OPT 4: Pre-allocate drawing buffer (reused every frame)
    colored_layer = np.zeros((height, width, 3), dtype=np.uint8)

    # OPT 3: Thread pool for async CPU preprocessing
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    pending_inputs = None  # Future for the next batch's preprocessed inputs

    try:
        ret = True
        while True:
            # ---- COLLECT A BATCH OF FRAMES ----
            while len(frames_buffer) < BATCH_SIZE and ret:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                image_pil.thumbnail((INFERENCE_SIZE, INFERENCE_SIZE))
                frames_buffer.append(image_pil)
                original_frames.append(frame)

            if len(frames_buffer) == 0:
                break

            # ---- CPU PREP (may be pre-fetched from previous iteration) ----
            t0_cpu = time.time()
            if pending_inputs is not None:
                inputs = pending_inputs.result()  # wait for prefetch
                pending_inputs = None  # consume it
            else:
                inputs = prepare_inputs(frames_buffer, processor)

            # OPT: non-blocking transfer with pinned memory
            inputs = {
                k: (
                    v.to(device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in inputs.items()
            }
            if inputs["pixel_values"].dtype != torch.bfloat16:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            current_batch_size = len(frames_buffer)
            cpu_prep_time = time.time() - t0_cpu
            acc_cpu_prep += cpu_prep_time

            # ---- OPT 3: START PREFETCH FOR NEXT BATCH (overlaps with GPU) ----
            next_frames_buffer = []
            next_original_frames = []
            next_ret = ret

            if next_ret:
                while len(next_frames_buffer) < BATCH_SIZE:
                    next_ret, next_frame = cap.read()
                    if not next_ret:
                        break
                    nf_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
                    nf_pil = Image.fromarray(nf_rgb)
                    nf_pil.thumbnail((INFERENCE_SIZE, INFERENCE_SIZE))
                    next_frames_buffer.append(nf_pil)
                    next_original_frames.append(next_frame)

                if next_frames_buffer:
                    pending_inputs = prefetch_executor.submit(
                        prepare_inputs, next_frames_buffer, processor
                    )
                else:
                    pending_inputs = None
            else:
                pending_inputs = None

                # ---- GPU INFERENCE ----
                t0_gpu = time.time()
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    outputs = model(**inputs)
                torch.cuda.synchronize()
                gpu_time = time.time() - t0_gpu
                acc_gpu_infer += gpu_time

                # ---- POST-PROCESS ----
                t0_post = time.time()
                # Use actual thumbnail dims from the PIL images (not original video dims),
                # and use the snapshot count — frames_buffer may be swapped by prefetch.
                thumb_h, thumb_w = frames_buffer[0].height, frames_buffer[0].width
                target_sizes = [(thumb_h, thumb_w)] * (current_batch_size * NUM_PROMPTS)
                results = processor.post_process_instance_segmentation(
                    outputs, threshold=BOX_THRESHOLD, target_sizes=target_sizes
                )

                # Free GPU memory early
                del outputs, inputs
                torch.cuda.empty_cache()

                for b_idx in range(current_batch_size):
                    global_frame_id = frame_count + b_idx + 1
                    frame_results = results[
                        b_idx * NUM_PROMPTS : (b_idx + 1) * NUM_PROMPTS
                    ]

                    # --- build manhole exclusion mask ---
                    manhole_res = frame_results[MANHOLE_IDX]
                    manhole_masks = manhole_res["masks"].cpu().numpy()
                    manhole_scores = manhole_res["scores"].float().cpu().numpy()
                    unified_manhole_mask = np.zeros((height, width), dtype=bool)
                    for m, s in zip(manhole_masks, manhole_scores):
                        if s >= BOX_THRESHOLD:
                            unified_manhole_mask |= m.astype(bool)

                    # --- convert SAM output → sv.Detections ---
                    sv_dets, meta_list = build_sv_detections(
                        frame_results, height, width, unified_manhole_mask
                    )

                    frame_json_dets = []
                    has_detections = sv_dets is not None and len(sv_dets) > 0

                    # OPT 7: only copy frame if we need to draw on it
                    base_frame = (
                        original_frames[b_idx].copy()
                        if has_detections
                        else original_frames[b_idx]
                    )

                    if has_detections:
                        tracked_entries = []

                        for p_idx, prompt_name in enumerate(PROMPTS):
                            if prompt_name == "manhole cover":
                                continue

                            class_mask = sv_dets.class_id == p_idx
                            if not class_mask.any():
                                continue

                            class_dets = sv_dets[class_mask]
                            tracked = trackers[prompt_name].update_with_detections(
                                class_dets
                            )

                            if tracked.tracker_id is None:
                                continue

                            for t_idx in range(len(tracked)):
                                tid = int(tracked.tracker_id[t_idx])
                                bbox = tracked.xyxy[t_idx]
                                score = (
                                    float(tracked.confidence[t_idx])
                                    if tracked.confidence is not None
                                    else BOX_THRESHOLD
                                )
                                # OPT 5: use cached bboxes for IoU matching
                                best_mask = match_mask_by_iou(
                                    bbox, meta_list, prompt_name
                                )
                                tracked_entries.append(
                                    (prompt_name, tid, bbox, score, best_mask, p_idx)
                                )

                        # --- count + draw ---
                        for (
                            prompt_name,
                            tid,
                            bbox,
                            score,
                            mask_bool,
                            p_idx,
                        ) in tracked_entries:
                            key = PROMPT_IDX_TO_KEY[p_idx]
                            color = PROMPT_COLOR[p_idx]  # OPT 6: pre-computed
                            track_key = (prompt_name, tid)

                            if track_key not in seen_tracks:
                                seen_tracks.add(track_key)
                                det_id_ctr += 1
                                defect_counts[key] += 1
                                cumulative[key] = defect_counts[key]

                                x1, y1, x2, y2 = (
                                    int(bbox[0]),
                                    int(bbox[1]),
                                    int(bbox[2]),
                                    int(bbox[3]),
                                )
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                                class_lists[key].append(
                                    {
                                        "detection_id": det_id_ctr,
                                        "type": key,
                                        "first_detected_frame": global_frame_id,
                                        "first_detected_time": round(
                                            global_frame_id / output_fps, 2
                                        ),
                                        "confidence": round(score, 3),
                                        "bbox": {
                                            "x1": x1,
                                            "y1": y1,
                                            "x2": x2,
                                            "y2": y2,
                                        },
                                        "center": {"x": cx, "y": cy},
                                        "area": (x2 - x1) * (y2 - y1),
                                        "track_id": tid,
                                    }
                                )

                            x1, y1, x2, y2 = (
                                int(bbox[0]),
                                int(bbox[1]),
                                int(bbox[2]),
                                int(bbox[3]),
                            )
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            frame_json_dets.append(
                                {
                                    "frame_id": global_frame_id,
                                    "track_id": tid,
                                    "type": key,
                                    "confidence": round(score, 3),
                                    "count": dict(cumulative),
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "center": {"x": cx, "y": cy},
                                }
                            )

                            label = f"[{tid}] {prompt_name} {score:.2f}"
                            if mask_bool is not None and mask_bool.any():
                                draw_mask_overlay(
                                    base_frame, mask_bool, color, label, colored_layer
                                )
                            else:
                                cv2.rectangle(base_frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(
                                    base_frame,
                                    label,
                                    (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 255),
                                    2,
                                )

                    # --- draw manhole (not tracked/counted) ---
                    need_copy_for_manhole = not has_detections
                    for m, s in zip(manhole_masks, manhole_scores):
                        if s < BOX_THRESHOLD:
                            continue
                        if need_copy_for_manhole:
                            base_frame = base_frame.copy()
                            need_copy_for_manhole = False
                        mb = m.astype(bool)
                        color = PROMPT_COLOR[MANHOLE_IDX]
                        draw_mask_overlay(
                            base_frame, mb, color, f"manhole {s:.2f}", colored_layer
                        )

                    if frame_json_dets:
                        frames_log.append(
                            {
                                "frame_id": global_frame_id,
                                "detections": frame_json_dets,
                            }
                        )

                    # OPT 2: non-blocking write via threaded writer
                    out.write(base_frame)

                frame_count += current_batch_size
                post_time = time.time() - t0_post
                acc_post_draw += post_time

                active_counts = {k: v for k, v in defect_counts.items() if v > 0}
                sys.stdout.write(
                    f"\rFrame: {frame_count}/{total} | "
                    f"GPU: {gpu_time:.3f}s | Prep: {cpu_prep_time:.3f}s | "
                    f"Draw: {post_time:.3f}s | {active_counts}  "
                )
                sys.stdout.flush()

                # ---- SWAP IN PREFETCHED FRAMES ----
                frames_buffer = next_frames_buffer
                original_frames = next_original_frames
                ret = next_ret

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        out.release()  # flushes write queue
        cv2.destroyAllWindows()
        prefetch_executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------
    total_defects = sum(defect_counts.values())
    frames_with_detections = len(frames_log)
    detection_rate = (
        round((frames_with_detections / frame_count) * 100, 2)
        if frame_count > 0
        else 0.0
    )

    report = {
        "video_id": str(uuid.uuid4()),
        "video_path": os.path.abspath(INPUT_VIDEO),
        "detection_type": "sam3-road-damage-v3-optimized",
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
        **{f"{k}_list": class_lists[k] for k in class_lists},
        "frames": frames_log,
    }

    t_json_start = time.time()
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    t_json_write = time.time() - t_json_start

    t_total_elapsed = time.time() - t_total_start
    acc_other = max(
        0,
        t_total_elapsed
        - t_model_load
        - acc_cpu_prep
        - acc_gpu_infer
        - acc_post_draw
        - t_json_write,
    )

    W = 60
    print(f"\n\n{'=' * W}")
    print(f"  ✅  SAM 3 v3 — Optimized Detection Summary (ByteTrack)")
    print(f"{'=' * W}")
    print(f"  {'Class':<40} {'Count':>6}")
    print(f"  {'-' * (W - 4)}")
    for k, v in defect_counts.items():
        print(f"  {k:<40} {v:>6}")
    print(f"  {'-' * (W - 4)}")
    print(f"  {'TOTAL UNIQUE OBJECTS':<40} {total_defects:>6}")
    print(f"\n{'=' * W}")
    print(f"  ⏱️   Timing Breakdown")
    print(f"{'=' * W}")
    print(f"  {'Stage':<40} {'Time':>8}")
    print(f"  {'-' * (W - 4)}")
    print(f"  {'Model Load (SAM3 + processor)':<40} {t_model_load:>7.1f}s")
    print(f"  {'CPU Prep (tokenise + .to(cuda))':<40} {acc_cpu_prep:>7.1f}s")
    print(f"  {'GPU Inference (SAM3 forward)':<40} {acc_gpu_infer:>7.1f}s")
    print(f"  {'Post-process + Track + Draw':<40} {acc_post_draw:>7.1f}s")
    print(f"  {'JSON Write':<40} {t_json_write:>7.2f}s")
    print(f"  {'Other (I/O, loop overhead)':<40} {acc_other:>7.1f}s")
    print(f"  {'-' * (W - 4)}")
    print(f"  {'TOTAL WALL TIME':<40} {t_total_elapsed:>7.1f}s")
    per_frame = t_total_elapsed / frame_count if frame_count else 0
    print(f"  {'Per Frame (avg)':<40} {per_frame:>7.3f}s")
    print(
        f"  {'Effective FPS processed':<40} {1 / per_frame if per_frame else 0:>7.2f}"
    )
    print(f"{'=' * W}")
    print(f"  📹  Video  → {OUTPUT_VIDEO}")
    print(f"  📄  Report → {OUTPUT_JSON}")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
