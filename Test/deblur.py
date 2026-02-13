import cv2
import numpy as np
import sys
import os
import time

# Default deblur/sharpening strength (0.0 = no effect, 1.0 = full sharpening)
DEFAULT_STRENGTH = 0.5

def apply_deblur(frame, strength=DEFAULT_STRENGTH):
    """
    Applies motion-blur removal / sharpening to the entire frame.
    
    Args:
        frame: Input BGR frame (numpy array).
        strength: Float between 0.0 and 1.0 controlling how aggressively 
                  the sharpening is applied. 0.0 = original, 1.0 = maximum sharpening.
    Returns:
        Deblurred/sharpened frame.
    """
    strength = np.clip(strength, 0.0, 1.0)

    # 1. Sharpening via unsharp mask (good for motion blur)
    #    Blur the frame, then amplify the difference between original and blurred
    gaussian_blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(frame, 1.0 + strength, gaussian_blur, -strength, 0)

    # 2. Additional detail recovery with Laplacian-based sharpening for stronger settings
    if strength > 0.3:
        kernel = np.array([
            [0, -1,  0],
            [-1,  5, -1],
            [0, -1,  0]
        ], dtype=np.float32)
        detail_sharpened = cv2.filter2D(sharpened, -1, kernel)
        # Blend based on how far above 0.3 the strength is (0.0 to 1.0 range)
        detail_blend = (strength - 0.3) / 0.7
        sharpened = cv2.addWeighted(sharpened, 1.0 - detail_blend * 0.5, 
                                     detail_sharpened, detail_blend * 0.5, 0)

    # 3. Slight denoising to reduce any artifacts from sharpening
    if strength > 0.5:
        sharpened = cv2.fastNlMeansDenoisingColored(sharpened, None, h=3, hForColorComponents=3,
                                                      templateWindowSize=7, searchWindowSize=21)

    return sharpened


def process_video(input_path, output_path=None, strength=DEFAULT_STRENGTH):
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n{'='*60}")
    print(f"VIDEO DEBLUR PROCESSING")
    print(f"{'='*60}")
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Strength: {strength:.2f} (0.0=none, 1.0=max)")
    print(f"Video:    {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"{'='*60}\n")

    start_time = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply full-frame deblur
            processed_frame = apply_deblur(frame, strength=strength)

            out.write(processed_frame)

            frame_idx += 1
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames) * 100
                elapsed = time.time() - start_time
                remaining = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
                print(f"\rProgress: {progress:.1f}% ({frame_idx}/{total_frames}) | "
                      f"Elapsed: {elapsed:.1f}s | Est. Remaining: {remaining:.1f}s", end="")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    finally:
        cap.release()
        out.release()
        total_time = time.time() - start_time
        print(f"\n\n{'='*60}")
        print(f"COMPLETE")
        print(f"{'='*60}")
        print(f"Frames processed: {frame_idx}/{total_frames}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Output saved to: {os.path.abspath(output_path)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python edge_deblur_video.py <input_video> [output_video] [strength]")
        print(f"  strength: 0.0 to 1.0 (default: {DEFAULT_STRENGTH})")
        test_video = "motion_blurred_video.mp4"
        if os.path.exists(test_video):
            print(f"\nNo input provided, but found '{test_video}'. Processing it...")
            process_video(test_video)
        else:
            sys.exit(1)
    else:
        input_vid = sys.argv[1]
        output_vid = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].replace('.', '', 1).isdigit() else None
        
        # Parse strength from last argument
        strength = DEFAULT_STRENGTH
        for arg in sys.argv[2:]:
            try:
                val = float(arg)
                if 0.0 <= val <= 1.0:
                    strength = val
                    break
            except ValueError:
                continue
        
        process_video(input_vid, output_vid, strength=strength)
 