# Pothole-RoadSign Detection Application: Bottleneck Analysis & Fixes

This report identifies the primary performance bottlenecks in the current backend architecture and provides targeted solutions to improve processing speed and resource efficiency.

---

## 🚀 Overview of Performance Bottlenecks

### 1. Synchronous Frame Processing

The current video processing logic follows a strict linear sequence for every frame:

- **Problem**: Decoding, preprocessing, inference, and results logging occur one after another in a single thread. This causes "bubbles" where the GPU sits idle while the CPU decodes frames, and vice versa.
- **Impact**: Hardware utilization is sub-optimal, leading to slower-than-real-time processing on high-resolution videos (e.g., 4K or 60FPS).

### 2. PyTorch vs. TensorRT Inference

The application predominantly uses `.pt` files for inference via the `ultralytics` YOLO library.

- **Problem**: PyTorch models are not natively optimized for NVIDIA's Tensor Cores.
- **Impact**: Inference latency is higher than necessary. Switching to TensorRT engines (`.engine`) can provide a **2x to 5x speedup** on compatible hardware.

### 3. Inefficient GPS mapping ($O(N)$ Linear Search)

When a pothole is detected, the system maps it to a database-defined "Location" using GPS coordinates.

- **Problem**: `location_mapper.py` pulls _all_ locations from the database into memory and iterates through them in a Python `for` loop.
- **Impact**: As the project grows to include thousands of road segments, this mapping step will eventually become the dominant bottleneck, blocking the processing pipeline.

### 4. Sequential Database I/O

Detections are currently pushed to the database individually or in a single large block at the end.

- **Problem**: High I/O overhead from individual inserts or blocking the completion of processing while waiting for a massive single commit.
- **Impact**: Increased total "Wall Clock" time for the end user.

---

## 🛠️ Proposed Fixes & Optimizations

### Fix 1: Implement a Parallel Processing Pipeline

**Strategy**: Transition to a Producer-Consumer architecture using multi-threading or multi-processing.

- **Action**:
  1. Dedicated **Decoding Thread** to keep a buffer of frames ready.
  2. Dedicated **Inference Thread** (GPU) to run model predictions continuously.
  3. Dedicated **Post-processing Thread** for mapping and DB updates.
- **Benefit**: Overlaps I/O and CPU tasks with GPU tasks, maximizing throughput.

### Fix 2: Migrate to TensorRT (TensorRT Export)

**Strategy**: Utilize NVIDIA's TensorRT engine for all production models.

- **Action**: Export existing `.pt` models to `.engine` format using `model.export(format='engine')`.
- **Benefit**: Drastic reduction in inference latency and memory footprint.

### Fix 3: Optimized Spatial Queries

**Strategy**: Leverage database indexing for geographic lookups.

- **Action**: Replace Python-based linear search with a SQL query using bounding box constraints (e.g., `WHERE lat BETWEEN min_lat AND max_lat`).
- **Benefit**: Reduces search complexity from $O(N)$ (linear) to $O(log N)$ (logarithmic) or better, ensuring performance remains stable as the database grows.

### Fix 4: Batched Database Inserts

**Strategy**: Use SQL bulk operations.

- **Action**: Accumulate detections and use SQLAlchemy's `bulk_save_objects` or `bulk_insert_mappings` every ~100 frames.
- **Benefit**: Dramatically reduces the number of database transactions and network round trips.

---

## 📈 Expected Results

| Metric              | Current State            | Optimized State (Estimate)   |
| :------------------ | :----------------------- | :--------------------------- |
| **Inference Speed** | ~15-30 FPS (PyTorch)     | ~60-120 FPS (TensorRT)       |
| **GPS Mapping**     | $O(N)$ Slows with growth | $O(1)$ constant/indexed time |
| **GPU Utilization** | 30% - 50% (Spiky)        | 80% - 95% (Sustained)        |
| **End-to-End Time** | > Video Duration         | < 50% Video Duration         |
