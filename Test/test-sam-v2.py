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

load_dotenv()

# ==============================================================================
# CONFIG
# ==============================================================================
HF_TOKEN = str(os.getenv("HF_TOKEN"))
INPUT_VIDEO = r"Test\video\drone-vid-2.mp4"
OUTPUT_VIDEO = r"Test\output\output_video_v2.mp4"
OUTPUT_JSON = r"Test\output\output_report_v2.json"

FPS_CLAMP_MIN = 10
FPS_CLAMP_MAX = 60

PROMPTS = [
    "broken traffic signboard",
    "discolored traffic signboard",
    "pothole and puddle",
    # "puddle",
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

# Prompt → JSON key (YOLO-compatible naming)
PROMPT_TO_KEY = {
    "broken traffic signboard": "broken_traffic_signboard",
    "discolored traffic signboard": "discolored_traffic_signboard",
    "pothole and puddle": "pothole_and_puddle",
    # "puddle": "puddle",
    "road crack": "road_crack",
    "manhole cover": "manhole_cover",
    "damaged road marking": "damaged_road_marking",
}


# ==============================================================================
# HELPER — build supervision Detections from raw SAM masks for one frame
# ==============================================================================
def build_sv_detections(frame_results, height, width, manhole_mask):
    """
    Turns SAM3 frame_results into a sv.Detections object (excluding manhole class).
    Returns detections and a parallel list of (prompt_name, mask_bool, score) tuples.
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

            # pothole-manhole suppression (unchanged core logic)
            if prompt_name == "pothole and puddle":
                total_area = mask_bool.sum()
                overlap_area = (mask_bool & manhole_mask).sum()
                if total_area > 0 and (overlap_area / total_area) > 0.4:
                    continue
                mask_bool = mask_bool & (~manhole_mask)

            if not mask_bool.any():
                continue

            ys, xs = np.where(mask_bool)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

            xyxy_list.append([x1, y1, x2, y2])
            score_list.append(float(score))
            class_list.append(i)  # use prompt index as class_id
            mask_list.append(mask_bool)
            meta_list.append((prompt_name, mask_bool, float(score)))

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

    print(f"✅ Hardware Detected: {torch.cuda.get_device_name(0)}")

    # RTX optimisations — unchanged
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    print("⏳ Loading SAM 3 into VRAM...")
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

    # Smart FPS — clamp to avoid OpenCV codec bug
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = raw_fps if FPS_CLAMP_MIN <= raw_fps <= FPS_CLAMP_MAX else 30.0
    duration = round(total / output_fps, 2) if output_fps > 0 else 0.0
    print(
        f"🎬 {width}x{height} | FPS: {output_fps:.1f} (raw={raw_fps:.1f}) | "
        f"Frames: {total} | Duration: {duration}s | Batch: {BATCH_SIZE}"
    )

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, output_fps, (width, height))

    # ------------------------------------------------------------------
    # ByteTrack — one tracker per class so tracks don't bleed across classes
    # ------------------------------------------------------------------
    trackers = {p: sv.ByteTrack() for p in PROMPTS if p != "manhole cover"}

    # Counting state — keyed by (prompt_name, tracker_id)
    seen_tracks = set()  # (prompt_name, track_id) — to count each object once
    defect_counts = {PROMPT_TO_KEY[p]: 0 for p in PROMPTS}
    class_lists = {PROMPT_TO_KEY[p]: [] for p in PROMPTS}
    cumulative = {PROMPT_TO_KEY[p]: 0 for p in PROMPTS}
    frames_log = []
    frame_count = 0
    det_id_ctr = 0  # used only for first-seen entries in class_lists

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

                # --- CPU PREP ---
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
                cpu_prep_time = time.time() - t0_cpu

                # --- GPU INFERENCE (unchanged core) ---
                t0_gpu = time.time()
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    outputs = model(**inputs)
                torch.cuda.synchronize()
                gpu_time = time.time() - t0_gpu

                # --- POST-PROCESS ---
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
                    base_frame = original_frames[b_idx].copy()

                    # --- build manhole exclusion mask (unchanged logic) ---
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

                    if sv_dets is not None and len(sv_dets) > 0:
                        # Run per-class ByteTrack update
                        # We need to split detections by class, track, then re-merge
                        tracked_entries = (
                            []
                        )  # list of (prompt_name, track_id, bbox, score, mask_bool)

                        for p_idx, prompt_name in enumerate(PROMPTS):
                            if prompt_name == "manhole cover":
                                continue

                            # filter detections to this class only
                            class_mask = sv_dets.class_id == p_idx
                            if not class_mask.any():
                                continue

                            class_dets = sv_dets[class_mask]
                            # update ByteTrack for this class
                            tracked = trackers[prompt_name].update_with_detections(
                                class_dets
                            )

                            if tracked.tracker_id is None:
                                continue

                            # rebuild mask_bool per tracked detection via IoU match
                            # (ByteTrack may drop/reorder detections)
                            for t_idx in range(len(tracked)):
                                tid = int(tracked.tracker_id[t_idx])
                                bbox = tracked.xyxy[t_idx]  # [x1,y1,x2,y2]
                                score = (
                                    float(tracked.confidence[t_idx])
                                    if tracked.confidence is not None
                                    else BOX_THRESHOLD
                                )

                                # find best matching meta entry by bbox IoU
                                best_mask = None
                                best_iou = 0.0
                                for pn, mb, sc in meta_list:
                                    if pn != prompt_name:
                                        continue
                                    ys, xs = np.where(mb)
                                    if len(xs) == 0:
                                        continue
                                    mx1, my1, mx2, my2 = (
                                        xs.min(),
                                        ys.min(),
                                        xs.max(),
                                        ys.max(),
                                    )
                                    # compute IoU between tracked bbox and mask bbox
                                    ix1 = max(bbox[0], mx1)
                                    iy1 = max(bbox[1], my1)
                                    ix2 = min(bbox[2], mx2)
                                    iy2 = min(bbox[3], my2)
                                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                                    area_a = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                    area_b = (mx2 - mx1) * (my2 - my1)
                                    union = area_a + area_b - inter
                                    iou = inter / union if union > 0 else 0.0
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_mask = mb

                                tracked_entries.append(
                                    (prompt_name, tid, bbox, score, best_mask)
                                )

                        # --- count + draw ---
                        for prompt_name, tid, bbox, score, mask_bool in tracked_entries:
                            key = PROMPT_TO_KEY[prompt_name]
                            color = COLORS[PROMPTS.index(prompt_name) % len(COLORS)]

                            track_key = (prompt_name, tid)

                            # ✅ Count only the FIRST time we see this track
                            if track_key not in seen_tracks:
                                seen_tracks.add(track_key)
                                det_id_ctr += 1
                                defect_counts[key] += 1
                                cumulative[key] = defect_counts[key]

                                x1, y1, x2, y2 = [int(v) for v in bbox]
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                area = int((x2 - x1) * (y2 - y1))

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
                                        "area": area,
                                        "track_id": tid,
                                    }
                                )

                            # Per-frame log entry (every frame the track is visible)
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
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

                            # --- DRAW mask + label (unchanged appearance) ---
                            if mask_bool is not None and mask_bool.any():
                                mask_uint8 = (mask_bool * 255).astype(np.uint8)
                                colored_layer = np.zeros_like(
                                    base_frame, dtype=np.uint8
                                )
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
                                    mask_uint8,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE,
                                )
                                cv2.drawContours(base_frame, contours, -1, color, 2)
                                if contours:
                                    c = max(contours, key=cv2.contourArea)
                                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                                    label = f"[{tid}] {prompt_name} {score:.2f}"
                                    cv2.putText(
                                        base_frame,
                                        label,
                                        (extTop[0], extTop[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (255, 255, 255),
                                        2,
                                    )
                            else:
                                # fallback: draw bbox only
                                x1, y1, x2, y2 = [int(v) for v in bbox]
                                cv2.rectangle(base_frame, (x1, y1), (x2, y2), color, 2)
                                label = f"[{tid}] {prompt_name} {score:.2f}"
                                cv2.putText(
                                    base_frame,
                                    label,
                                    (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (255, 255, 255),
                                    2,
                                )

                    # --- draw manhole (unchanged, not tracked/counted) ---
                    for m, s in zip(manhole_masks, manhole_scores):
                        if s < BOX_THRESHOLD:
                            continue
                        mb = m.astype(bool)
                        mask_uint8 = (mb * 255).astype(np.uint8)
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
                                f"manhole {s:.2f}",
                                (extTop[0], extTop[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (255, 255, 255),
                                2,
                            )

                    if frame_json_dets:
                        frames_log.append(
                            {
                                "frame_id": global_frame_id,
                                "detections": frame_json_dets,
                            }
                        )

                    out.write(base_frame)

                frame_count += len(frames_buffer)
                post_time = time.time() - t0_post

                active_counts = {k: v for k, v in defect_counts.items() if v > 0}
                sys.stdout.write(
                    f"\rFrame: {frame_count}/{total} | "
                    f"GPU: {gpu_time:.3f}s | Prep: {cpu_prep_time:.3f}s | "
                    f"Draw: {post_time:.3f}s | {active_counts}  "
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

    # ------------------------------------------------------------------
    # JSON report — YOLO-compatible format
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
        "detection_type": "sam3-road-damage-v2",
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

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    print(f"\n\n{'='*55}")
    print(f"  ✅  SAM 3 v2 — Unique Object Count (ByteTrack)")
    print(f"{'='*55}")
    print(f"  {'Class':<35} {'Count':>6}")
    print(f"  {'-'*41}")
    for k, v in defect_counts.items():
        print(f"  {k:<35} {v:>6}")
    print(f"  {'-'*41}")
    print(f"  {'TOTAL UNIQUE OBJECTS':<35} {total_defects:>6}")
    print(f"{'='*55}")
    print(f"  📹  Video  → {OUTPUT_VIDEO}")
    print(f"  📄  Report → {OUTPUT_JSON}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
