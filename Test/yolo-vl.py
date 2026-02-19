import ollama
import base64
import json
import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Set up Ollama client for cloud access
def setup_ollama_client():
    """Configure Ollama client with API key from environment"""
    api_key = os.getenv('OLLAMA_API_KEY')
    
    if not api_key:
        raise ValueError("OLLAMA_API_KEY environment variable not set. "
                        "Set it with: export OLLAMA_API_KEY=your_api_key")
    
    # Create client with cloud endpoint
    client = ollama.Client(
        host='https://ollama.com',
        headers={'Authorization': f'Bearer {api_key}'}
    )
    
    return client


def detect_and_classify_video_frame(frame_path, confidence_threshold=0.5):
    """
    Detect objects using YOLO, then classify each with vision model
    """
    # Setup Ollama client with API key
    ollama_client = setup_ollama_client()
    
    # Load YOLO model for generic object detection
    yolo_model = YOLO(r'models\final-v1.pt')  
    
    # Read frame
    frame = cv2.imread(frame_path)
    
    # Detect objects
    results = yolo_model(frame, conf=confidence_threshold)
    
    classifications = []
    
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection
 
        # Crop detected region
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped = frame[y1:y2, x1:x2]
        
        # Save temporarily
        temp_path = r"Test\output\temp_detection.jpg"
        cv2.imwrite(temp_path, cropped)
        
        # Encode and classify
        with open(temp_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = """Classify into ONE: defected_sign_board, good_sign_board, pothole, road_crack, damaged_road_marking, or null.

JSON: {"category": "name", "confidence": "high/medium/low", "belongs_to_category": true/false}"""
        
        try:
            response = ollama_client.chat(
                model='qwen3-vl:235b-instruct-cloud',
                format='json',
                messages=[{
                    'role': 'user',
                    'content': prompt + '\n\nRespond ONLY with valid JSON.',
                    'images': [base64_image]
                }],
                options={'temperature': 0.1}
            )
            
            classification = json.loads(response['message']['content'])
            classification['bbox'] = [x1, y1, x2, y2]
            classification['yolo_confidence'] = float(conf)
            
            classifications.append(classification)
            
        except Exception as e:
            print(f"Error classifying detection: {e}")
            continue
    
    return classifications


# Main execution
if __name__ == "__main__":
    # Process video frame
    results = detect_and_classify_video_frame(
        r"Test\Screenshot 2026-02-17 122339.png"
    )
    
    for idx, result in enumerate(results):
        print(f"\nObject {idx + 1}:")
        print(json.dumps(result, indent=2))
