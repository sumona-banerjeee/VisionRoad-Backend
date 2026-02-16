
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO)

def test_instantiation():
    print("Testing service instantiation...")
    try:
        from app.services.video_processor import VideoProcessor
        from app.services.signboard_detector import SignBoardDetector
        from app.services.pot_sign_detector import PotSignDetector
        
        print("Importing services successful.")
        
        # We won't actually load models here to save time/memory, 
        # but we can check if they inherit from BaseDetector
        from app.services.base_detector import BaseDetector
        
        print(f"VideoProcessor is subclass of BaseDetector: {issubclass(VideoProcessor, BaseDetector)}")
        print(f"SignBoardDetector is subclass of BaseDetector: {issubclass(SignBoardDetector, BaseDetector)}")
        print(f"PotSignDetector is subclass of BaseDetector: {issubclass(PotSignDetector, BaseDetector)}")
        
        # Check for common methods
        services = [VideoProcessor, SignBoardDetector, PotSignDetector]
        for service in services:
            for method in ['process_video', 'get_status', 'get_results']:
                if not hasattr(service, method):
                    print(f"Error: {service.__name__} missing method {method}")
                    return False
        
        print("All services have required methods.")
        return True
    except Exception as e:
        print(f"Error during instantiation test: {e}")
        return False

if __name__ == "__main__":
    if test_instantiation():
        print("Verification SUCCESSFUL")
    else:
        print("Verification FAILED")
        sys.exit(1)
