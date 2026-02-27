import torch
from transformers import Sam3Processor, Sam3Model
import cv2
import numpy as np
from PIL import Image
import os
import sys
import time
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
 
# ==============================================================================
# 🛑 PASTE YOUR TOKEN BELOW
# ==============================================================================
HF_TOKEN = str(os.getenv("HF_TOKEN"))
 
# CONFIG
INPUT_VIDEO = r"Test\video\drone-vid-2.mp4"
OUTPUT_VIDEO = r"Test\output\output_video.mp4"
 
# --- FIX: FORCE FPS ---
# OpenCV falsely reads your video as 120fps. We force it to 24 here.
OUTPUT_FPS = 24  
 
PROMPTS =[
    "broken traffic signboard", "discolored traffic signboard",
    "pothole", "puddle", "road crack", "manhole cover", "damaged road marking"
]
MANHOLE_IDX = PROMPTS.index("manhole cover")  
 
INFERENCE_SIZE = 1024  
BOX_THRESHOLD = 0.6    
SHOW_PREVIEW = False
BATCH_SIZE = 8        
 
COMPILE_MODEL = False if os.name == 'nt' else True
 
COLORS =[
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 128)
]
 
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
   
    # ==============================================================================
    # 🚀 RTX 5090 HARDWARE UNLOCKS
    # ==============================================================================
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
            attn_implementation="sdpa"
        ).to(device)
       
        processor = Sam3Processor.from_pretrained("facebook/sam3")
       
        if COMPILE_MODEL:
            model = torch.compile(model, mode="reduce-overhead")
           
    except Exception as e:
        print(f"Error loading model: {e}")
        return
 
    if not os.path.exists(INPUT_VIDEO):
        print("❌ input.mp4 not found.")
        return
 
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, OUTPUT_FPS, (width, height))
    print(f"🎬 Processing {width}x{height} - Forced to {OUTPUT_FPS} FPS - Batch Size: {BATCH_SIZE}")
 
    frame_count = 0
    frames_buffer =[]
    original_frames =[]
 
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
                batch_images = []
                batch_texts =[]
               
                for img in frames_buffer:
                    batch_images.extend([img] * len(PROMPTS))
                    batch_texts.extend(PROMPTS)
 
                inputs = processor(
                    images=batch_images,
                    text=batch_texts,
                    return_tensors="pt"
                )
 
                inputs = inputs.to(device)
                if inputs["pixel_values"].dtype != torch.bfloat16:
                     inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
               
                t1_cpu = time.time()
                cpu_prep_time = t1_cpu - t0_cpu
 
                # --- GPU INFERENCE ---
                t0_gpu = time.time()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)
               
                torch.cuda.synchronize()
                t1_gpu = time.time()
                gpu_time = t1_gpu - t0_gpu
 
                # --- POST-PROCESSING & DRAWING ---
                t0_post = time.time()
                target_sizes = [(height, width)] * len(batch_texts)
                results = processor.post_process_instance_segmentation(
                    outputs, threshold=BOX_THRESHOLD, target_sizes=target_sizes
                )
 
                for b_idx in range(len(frames_buffer)):
                    frame_results = results[b_idx * len(PROMPTS) : (b_idx + 1) * len(PROMPTS)]
                    base_frame = original_frames[b_idx]
                   
                    manhole_res = frame_results[MANHOLE_IDX]
                    manhole_masks = manhole_res['masks'].cpu().numpy()
                    manhole_scores = manhole_res['scores'].float().cpu().numpy()
                   
                    unified_manhole_mask = np.zeros((height, width), dtype=bool)
                    for m, s in zip(manhole_masks, manhole_scores):
                        if s >= BOX_THRESHOLD:
                            unified_manhole_mask |= m.astype(bool)
 
                    for i, result in enumerate(frame_results):
                        if i == MANHOLE_IDX: continue
                       
                        prompt_name = PROMPTS[i]
                        color = COLORS[i % len(COLORS)]
                       
                        masks = result['masks'].cpu().numpy()
                        scores = result['scores'].float().cpu().numpy()
                       
                        for mask, score in zip(masks, scores):
                            if score < BOX_THRESHOLD: continue
                           
                            mask_bool = mask.astype(bool)
                           
                            if prompt_name == "pothole":
                                overlap_area = (mask_bool & unified_manhole_mask).sum()
                                total_area = mask_bool.sum()
                               
                                if total_area > 0 and (overlap_area / total_area) > 0.4:
                                    continue
                               
                                mask_bool = mask_bool & (~unified_manhole_mask)
                           
                            if not mask_bool.any(): continue
                           
                            mask_uint8 = (mask_bool * 255).astype(np.uint8)
                           
                            colored_layer = np.zeros_like(base_frame, dtype=np.uint8)
                            colored_layer[:] = color
                            base_frame = cv2.addWeighted(base_frame, 1.0, cv2.bitwise_and(colored_layer, colored_layer, mask=mask_uint8), 0.5, 0)
                           
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(base_frame, contours, -1, color, 2)
                           
                            if len(contours) > 0:
                                c = max(contours, key=cv2.contourArea)
                                extTop = tuple(c[c[:, :, 1].argmin()][0])
                                label_text = f"{prompt_name} {score:.2f}"
                                cv2.putText(base_frame, label_text, (extTop[0], extTop[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
 
                    out.write(base_frame)
                    frame_count += 1
 
                t1_post = time.time()
                post_time = t1_post - t0_post
               
                sys.stdout.write(
                    f"\rFrame: {frame_count}/{total} | "
                    f"Device: {inputs['pixel_values'].device} | "
                    f"CPU Prep: {cpu_prep_time:.3f}s | "
                    f"GPU Run: {gpu_time:.3f}s | "
                    f"OpenCV Draw: {post_time:.3f}s  "
                )
                sys.stdout.flush()
 
                frames_buffer.clear()
                original_frames.clear()
 
            if not ret: break
 
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Done! Saved to {OUTPUT_VIDEO}")
 
if __name__ == "__main__":
    main()