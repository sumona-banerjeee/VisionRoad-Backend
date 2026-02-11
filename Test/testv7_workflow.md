# TestV7 Detection Workflow - Complete Guide

## Overview
The v7 detection system uses **adaptive parameters**, **multi-frame confirmation**, and **spatial-temporal deduplication** to accurately count unique objects (potholes and signboards) in videos or images.

---

## 🎯 Core Philosophy

**Goal**: Count each unique real-world object exactly once, even if it appears in multiple frames.

**Key Challenges**:
- Same pothole/sign appears in 20+ consecutive frames → Must deduplicate
- Camera moves slowly → Multiple detections of same object at slightly different positions
- Low confidence detections → Need confirmation across frames to avoid false positives
- Variable video lengths → Parameters must adapt (1-second video vs 10-minute video)

---

## 📊 Detection Pipeline (Step-by-Step)

### Stage 1: Input Processing
```
Input File → Detect Type (Image/Video) → Load with appropriate handler
```

**For Videos**: VideoCapture, read frame-by-frame  
**For Images**: Single frame, processed once

---

### Stage 2: Adaptive Parameter Calculation

Parameters scale based on **video duration** to work for both short clips and long videos.

#### Key Parameters:

| Parameter | Formula | Purpose |
|-----------|---------|---------|
| `DETECTION_TIME_WINDOW` | `video_duration × 0.25` | How long to track recent detections (25% of video) |
| `TIME_THRESHOLD` | `video_duration × 0.30` | Max time gap for spatial deduplication (30% of video) |
| `MIN_DISTANCE_THRESHOLD` | `100 pixels` | Spatial distance to consider "same object" |

**Example**:
- 4-second video → Detection window = 1.0s, Time threshold = 1.2s
- 40-second video → Detection window = 10.0s, Time threshold = 12.0s

---

### Stage 3: Frame-by-Frame Detection

For each frame:

```python
1. Run YOLO tracking with model.track()
   ↓
2. Get tracked objects with IDs
   ↓
3. For each detection:
   - Check if inside ROI
   - Update temporal tracker
   - Apply multi-frame confirmation
   - Check for spatial duplicates
   - Add to confirmed list if new & valid
```

---

### Stage 4: Multi-Frame Confirmation

**Purpose**: Avoid false positives from fleeting detections.

#### Confidence-Based Logic:

```python
if confidence >= 0.75:  # HIGH_CONFIDENCE_THRESHOLD
    MIN_FRAMES_NEEDED = 1  # Trust immediately
else:
    MIN_FRAMES_NEEDED = 2  # Need 2+ frames to confirm
```

**How it works**:
- Each track ID maintains a history of detection times
- Count recent detections within `DETECTION_TIME_WINDOW`
- Only confirm if `recent_detections >= MIN_FRAMES_NEEDED`

**Example**:
```
Frame 1: ID 5, conf=0.80 → High confidence, needs 1 frame → ✅ CONFIRMED
Frame 1: ID 6, conf=0.50 → Low confidence, needs 2 frames → ⏳ PENDING
Frame 2: ID 6, conf=0.52 → Now has 2 frames → ✅ CONFIRMED
```

---

### Stage 5: Spatial Deduplication

**Purpose**: Don't count the same object twice if detected nearby in time.

#### Algorithm:

```python
For new detection at (cx, cy):
    For each previously confirmed object:
        distance = sqrt((cx - prev_cx)² + (cy - prev_cy)²)
        time_gap = current_time - prev_time
        
        if (same_class AND distance < 100px AND time_gap < TIME_THRESHOLD):
            → REJECT as duplicate
```

**Key Insight**: As camera moves, same pothole shifts position slightly. If detected within 100px and within time window, it's likely the same object.

---

### Stage 6: Confirmation & Categorization

Once an object passes all checks:

```python
1. Create detection record with:
   - detection_id (track ID)
   - type (pothole/signboard)
   - bbox, confidence, frame, time
   
2. Add to appropriate list:
   - pothole → pothole_list
   - signboard → signboard_list
   
3. Track in spatial_locations for future deduplication

4. Mark in confirmed dictionary
```

---

### Stage 7: Output Generation

#### JSON Structure:
```json
{
  "detection_id": "uuid",
  "input_type": "video" or "image",
  "summary": {
    "unique_potholes": 10,
    "unique_signboards": 5,
    "total_detections": 450
  },
  "pothole_list": [...],
  "signboard_list": [...],
  "frames": [...]
}
```

#### Video Output:
- Annotated frames with bounding boxes
- ROI overlay (green rectangle)
- Real-time counter display

---

## 🔧 Parameter Tuning Guide

### 1. Confidence Threshold (`CONF_THRESHOLD`)

**Default**: `0.30`

**What it does**: Minimum confidence for YOLO to report a detection.

**Tuning**:
- **Too Low (0.15-0.25)**: Many false positives, but catches all objects
- **Sweet Spot (0.30-0.40)**: Balanced
- **Too High (0.50+)**: Misses real objects, but very accurate

**Recommendation**: Start at 0.30, increase if too many false detections.

---

### 2. High Confidence Threshold (`HIGH_CONFIDENCE_THRESHOLD`)

**Default**: `0.75`

**What it does**: Detections above this get 1-frame confirmation instead of 2.

**Tuning**:
- **Lower (0.60-0.70)**: More aggressive, confirms objects faster
- **Higher (0.80-0.90)**: Very conservative, most detections need 2 frames

**When to adjust**: If you see high-confidence objects being missed, lower this.

---

### 3. Spatial Distance Threshold (`MIN_DISTANCE_THRESHOLD`)

**Default**: `100 pixels`

**What it does**: How close (in pixels) two detections must be to consider them "same object".

**Tuning**:
- **Too Small (30-50px)**: Same object counted multiple times
- **Too Large (200-300px)**: Different objects incorrectly merged
- **Good Range**: 80-150px depending on object size

**Consider**:
- Large potholes → Larger threshold (150px)
- Small signs → Smaller threshold (70px)
- Wide-angle camera → Larger threshold
- Zoomed camera → Smaller threshold

---

### 4. Time Window Percentages

#### `DETECTION_TIME_WINDOW_PERCENT`
**Default**: `0.25` (25% of video)

**What it does**: How far back to look for recent detections when confirming multi-frame.

**Tuning**:
- **Shorter videos (1-5s)**: Keep at 25%
- **Longer videos (60s+)**: Can reduce to 10-15% for efficiency

#### `TIME_THRESHOLD_PERCENT`
**Default**: `0.30` (30% of video)

**What it does**: Max time gap for spatial deduplication.

**Tuning**:
- **Slow-moving camera**: Increase to 40-50% (objects stay in frame longer)
- **Fast-moving camera**: Decrease to 20-25% (objects leave frame quickly)

---

### 5. ROI (Region of Interest)

**Current Settings**:
```python
ROI_TOP = int(height * 0.05)     # 5% from top
ROI_BOTTOM = int(height * 0.95)  # 95% from top
ROI_LEFT = int(width * 0.0)      # Full width
ROI_RIGHT = int(width * 1.0)
```

**Purpose**: Ignore detections outside the useful area (sky, vehicle hood, etc.).

**Tuning for Different Scenarios**:

| Scenario | ROI Settings |
|----------|--------------|
| **Signboards only** | TOP=0.05, BOTTOM=0.50 (upper half) |
| **Potholes only** | TOP=0.40, BOTTOM=0.95 (lower half) |
| **Both** | TOP=0.05, BOTTOM=0.95 (current) |
| **Dashboard cam** | TOP=0.15, BOTTOM=0.85 (exclude hood) |

---

## 🐛 Debugging Features

### Rejection Statistics

The system tracks why detections were rejected:

```python
rejection_stats = {
    "multi_frame_pending": {...},  # Waiting for confirmation
    "spatial_duplicate": 120,      # Too close to existing
    "roi_outside": 45              # Outside ROI
}
```

### Debug Output

Every detection prints:
```
🔍 Frame  10: ID  3 - pothole  conf=0.82, frames=5/1
✅ Frame  10 (0.33s): CONFIRMED ID 3 - pothole at (320, 450) [conf=0.82, frames=5]
❌ Frame  15 (0.50s): DUPLICATE ID 7 - pothole
   Reason: 85.3px from existing, 0.17s ago
```

**Symbols**:
- 🔍 = Detection found
- ✅ = Confirmed as new unique object
- ❌ = Rejected as duplicate
- ⏳ = Pending multi-frame confirmation
- ⊗ = Outside ROI

---

## 💡 Common Scenarios & Solutions

### Scenario 1: Same pothole counted 3 times

**Diagnosis**: Spatial deduplication not working.

**Solutions**:
1. Increase `MIN_DISTANCE_THRESHOLD` to 150-200px
2. Increase `TIME_THRESHOLD_PERCENT` to 0.40-0.50

---

### Scenario 2: Missing real potholes

**Diagnosis**: Too strict confirmation or confidence threshold.

**Solutions**:
1. Lower `CONF_THRESHOLD` from 0.30 to 0.25
2. Lower `HIGH_CONFIDENCE_THRESHOLD` to 0.65
3. Check ROI settings - are potholes outside the ROI?

---

### Scenario 3: Too many false positives

**Diagnosis**: Accepting low-confidence detections too easily.

**Solutions**:
1. Increase `CONF_THRESHOLD` to 0.35-0.40
2. Increase `LOW_CONFIDENCE_MIN_FRAMES` from 2 to 3-4
3. Tighten ROI to exclude problematic areas

---

### Scenario 4: Objects detected but not confirmed

**Diagnosis**: Check debug logs for rejection reason.

**Look for**:
- ⏳ Pending → Increase video length or lower frame requirements
- ❌ Duplicate → Adjust spatial/temporal thresholds
- ⊗ Outside ROI → Adjust ROI boundaries

---

## 🎬 Workflow Summary

```
INPUT FILE
    ↓
Detect Type (Image/Video)
    ↓
Calculate Adaptive Parameters
    ↓
FOR EACH FRAME:
    YOLO Detection + Tracking
        ↓
    ROI Filter
        ↓
    Temporal Tracking (update history)
        ↓
    Multi-Frame Confirmation Check
        ↓
    Spatial Deduplication Check
        ↓
    IF PASSES ALL CHECKS:
        Add to confirmed list
        Categorize (pothole/signboard)
        ↓
OUTPUT:
    - Annotated video/image
    - JSON with unique counts & lists
    - Console report
```

---

## 📝 Quick Reference: Parameter Effects

| Want to... | Adjust... | Direction |
|------------|-----------|-----------|
| Reduce duplicates | `MIN_DISTANCE_THRESHOLD` | ↑ Increase |
| Catch more objects | `CONF_THRESHOLD` | ↓ Decrease |
| Faster confirmation | `HIGH_CONFIDENCE_THRESHOLD` | ↓ Decrease |
| Be more selective | `LOW_CONFIDENCE_MIN_FRAMES` | ↑ Increase |
| Handle slow camera | `TIME_THRESHOLD_PERCENT` | ↑ Increase |
| Focus on specific area | `ROI_TOP/BOTTOM` | Adjust boundaries |

---

## 🚀 Best Practices

1. **Start with defaults** and run on sample video
2. **Check rejection stats** to see what's failing
3. **Adjust one parameter at a time** and re-test
4. **Use debug output** to understand decision-making
5. **Validate** by manually counting objects in a few frames

---

## Example: Tuning for Highway vs City

### Highway (Fast-moving, far objects)
```python
CONF_THRESHOLD = 0.35              # Higher - less noise
MIN_DISTANCE_THRESHOLD = 150       # Larger - objects appear smaller
TIME_THRESHOLD_PERCENT = 0.20      # Shorter - objects leave frame quickly
ROI_TOP = 0.10                     # Exclude more sky
```

### City (Slow-moving, close objects)
```python
CONF_THRESHOLD = 0.25              # Lower - catch everything
MIN_DISTANCE_THRESHOLD = 80        # Smaller - objects larger/closer
TIME_THRESHOLD_PERCENT = 0.40      # Longer - objects stay in frame
ROI_TOP = 0.05                     # Include more vertical space
```
