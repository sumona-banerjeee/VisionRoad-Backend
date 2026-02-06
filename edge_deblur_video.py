import cv2
import numpy as np
import sys
import os
import time

def apply_edge_deblur(frame):
    """
    Applies edge-specific sharpening to a frame.
    Sharpened at edges, original in the middle using a Gaussian blurred mask.
    """
    h, w = frame.shape[:2]
    
    # 1. Create a mask that is white at the edges and black in the middle
    mask = np.zeros((h, w), dtype=np.float32)
    edge_width = int(w * 0.25) # Focus on outer 25% of each side
    mask[:, :edge_width] = 1.0
    mask[:, -edge_width:] = 1.0
    
    # Blur the mask so the transition isn't a sharp line
    # Note: Using half of edge_width or min dimension to ensure kernel size is reasonable
    kernel_size = 101
    if h < kernel_size or w < kernel_size:
        kernel_size = min(h, w) // 2 * 2 + 1 # Ensure odd
        if kernel_size < 3: kernel_size = 3

    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    # 2. Create a sharpened version of the frame
    # Standard sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(frame, -1, kernel)
    
    # 3. Blend: Sharpened at edges, Original in middle
    # Convert mask to 3 channels for broadcasting
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # Float conversion for precision during blending
    frame_float = frame.astype(np.float32)
    sharpened_float = sharpened.astype(np.float32)
    
    result = (frame_float * (1.0 - mask_3ch) + sharpened_float * mask_3ch)
    return np.clip(result, 0, 255).astype(np.uint8)

def process_video(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_deblurred{ext}"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_path}'")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Starting processing: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    start_time = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply deblur effect
            processed_frame = apply_edge_deblur(frame)
            
            # Write to output
            out.write(processed_frame)
            
            frame_idx += 1
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames) * 100
                elapsed = time.time() - start_time
                remaining = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
                print(f"\rProgress: {progress:.1f}% ({frame_idx}/{total_frames}) | Elapsed: {elapsed:.1f}s | Est. Remaining: {remaining:.1f}s", end="")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    finally:
        cap.release()
        out.release()
        print(f"\n\nProcessing complete!")
        print(f"Total time: {time.time() - start_time:.1f} seconds")
        print(f"Output file: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python edge_deblur_video.py <input_video_path> [output_video_path]")
        # Check for default test file if no args provided
        test_video = "motion_blurred_video.mp4"
        if os.path.exists(test_video):
            print(f"No input provided, but found '{test_video}'. Processing it...")
            process_video(test_video)
        else:
            sys.exit(1)
    else:
        input_vid = sys.argv[1]
        output_vid = sys.argv[2] if len(sys.argv) > 2 else None
        process_video(input_vid, output_vid)
