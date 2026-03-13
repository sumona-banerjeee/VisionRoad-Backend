import os
import sys

# Add backend root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.helpers.yoloe_vl_trained_helper import (
    load_yoloe_trained_model,
    process_with_trained_vl,
    VL_SKIP_CLASSES,
)

def test_yoloe_trained_vl():
    print("1. Loading trained YOLOE model...")
    model = load_yoloe_trained_model()
    print("Model loaded successfully!")
    print(f"Model names: {model.names}")

    print("\n2. Testing process_with_trained_vl branching...")
    
    # Mock frame and bbox
    dummy_frame = None  # None is okay if it skips VL, will crash if it hits VL (which is good for this test)
    dummy_bbox = (0, 0, 100, 100)

    # Test a skipped class (should return immediately without calling actual VL)
    test_class1 = "defected_sign_board"
    print(f"\nTesting class: '{test_class1}' (in VL_SKIP_CLASSES: {test_class1 in VL_SKIP_CLASSES})")
    result1 = process_with_trained_vl(dummy_frame, dummy_bbox, test_class1)
    print(f"Result: {result1}")
    assert result1["category"] == test_class1
    assert result1["_vl_elapsed_s"] == 0.0

    # Test an API class (should attempt to call VL, which might fail or timeout due to dummy frame/no API key, but we'll catch it)
    test_class2 = "pothole"
    print(f"\nTesting class: '{test_class2}' (in VL_SKIP_CLASSES: {test_class2 in VL_SKIP_CLASSES})")
    try:
        from unittest.mock import patch
        # Patch the actual VL call so we don't need real API keys during this synthetic test
        with patch('app.helpers.yoloe_vl_trained_helper.process_with_vl') as mock_vl:
            mock_vl.return_value = {"mocked": True, "category": "pothole"}
            result2 = process_with_trained_vl(dummy_frame, dummy_bbox, test_class2)
            print(f"Result: {result2}")
            assert result2["mocked"] is True
            assert mock_vl.called
    except ImportError:
        print("Mocking not available, skipping actual VL call testing.")

    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_yoloe_trained_vl()
