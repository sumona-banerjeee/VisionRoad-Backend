import google.generativeai as genai
import os
import json
import re

# 🔑 Set your API key (replace with your new one if you regenerate)
genai.configure(api_key="AIzaSyC28HeIKmhAj1J71Qazh4Rpk9MT9ch6Esw")

# 📦 Load model
model = genai.GenerativeModel("gemini-robotics-er-1.5-preview")

# 📂 Folder containing images
IMAGE_FOLDER = r"Test\GEMINI_IMAGES"

# 🧾 Get first 10 images
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))][:10]

# 🧠 Prompt (❗ removed image_name — we handle it ourselves)
prompt = """
You are a road inspection AI.

Classify each image into exactly ONE of these categories:
- defective signboard
- pothole
- faded road marking
- good signboard
- road crack
- defective culvert
- good culvert
- bad drainage issue

You will receive multiple images.

Return ONLY a JSON array.

Each item must be:
{
  "image_index": number,
  "label": "one of the categories"
}

Do not explain anything.
"""

# 🖼️ Prepare contents
contents = [prompt]

for img_name in image_files:
    path = os.path.join(IMAGE_FOLDER, img_name)

    with open(path, "rb") as img_file:
        contents.append({
            "mime_type": "image/jpeg",  # works for png too in most cases
            "data": img_file.read()
        })

# 🚀 Send request
response = model.generate_content(contents)

# 🧾 Raw output (optional debug)
print("RAW RESPONSE:\n", response.text)

# 🔍 Extract JSON safely
match = re.search(r'\[.*\]', response.text, re.DOTALL)

if not match:
    print("❌ No valid JSON found")
    exit()

data = json.loads(match.group())

# ✅ Attach REAL filenames
for item in data:
    idx = item["image_index"] - 1

    if 0 <= idx < len(image_files):
        item["image_name"] = image_files[idx]
    else:
        item["image_name"] = "UNKNOWN"

# 📤 Final clean output
print("\n✅ FINAL RESULT:\n")
print(json.dumps(data, indent=2))