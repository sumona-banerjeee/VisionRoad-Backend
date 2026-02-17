import ollama
import base64
import json

# Function to encode image to Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to local image
image_path = r"Test\dam-sign.jpg"
base64_image = encode_image_to_base64(image_path)


# Detailed prompt for classification
prompt = """Analyze this image carefully and classify it into ONE of these categories:

1. defected_sign_board - Damaged, broken, bent, faded, or deteriorated sign boards
2. good_sign_board - Sign boards in good, readable, and intact condition
3. pothole - Holes, depressions, or cavities in road surface
4. road_crack - Cracks, fissures, or splits in road pavement
5. damaged_road_marking - Faded, worn out, or broken road lines/markings

If the image does NOT match any of these categories, set category to null.

Respond with this exact JSON structure:
{
  "category": "category_name_or_null",
  "confidence": "high_or_medium_or_low",
  "belongs_to_category": true_or_false
}"""


response = ollama.chat(
    model='qwen3-vl:235b-instruct-cloud',
    format='json',  # Simple JSON mode
    messages=[
        {
            'role': 'user',
            'content': prompt + '\n\nRespond ONLY with valid JSON.',
            'images': [base64_image]
        }
    ],
    options={
        'temperature': 0.1,  # Lower temperature for consistent output
    }
)

# Parse and display result
try:
    result = json.loads(response['message']['content'])
    print(json.dumps(result, indent=2))
    
    # Display summary
    if result.get('belongs_to_category', False):
        print(f"\n✓ Category: {result['category']}")
        print(f"  Confidence: {result['confidence']}")
    else:
        print(f"\n✗ No matching category found")
        
except json.JSONDecodeError as e:
    print(f"❌ Failed to parse JSON response")
    print(f"Error: {e}")
    print(f"\nRaw response:\n{response['message']['content']}")
