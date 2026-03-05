# VisionRoad Backend — Additional Bottlenecks (v2)

> Findings beyond the original `codebase-drawbacks.md`. Detection pipeline issues and SQLite are excluded — covered there already.

---

## 🔴 Critical

### 1. `shutil.copyfileobj` Blocks the Async Event Loop on Upload
**File:** `app/services/upload_service.py`

```python
with open(video_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)   # synchronous — freezes the event loop
```

Runs synchronously inside an `async` function. For a 500 MB video this stalls the entire FastAPI event loop for several seconds — no other requests are served during the copy.

**Fix:** Use `aiofiles` + async chunked writes, or offload to `loop.run_in_executor`.

---

### 2. Global `torch.load` Monkeypatch Disables Pickle Safety
**File:** `main.py`

```python
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False   # disables safety for ALL torch.load calls
    return _original_load(*args, **kwargs)

torch.load = _patched_load
```

`weights_only=False` disables PyTorch 2.6+'s pickle guard for every `torch.load` call in the entire process (including SAM3 and any third-party lib). A malicious `.pt` file could execute arbitrary Python code on load.

**Fix:** Use `torch.serialization.add_safe_globals([...])` or a scoped override only where needed.

---

## 🟠 Significant Bottlenecks

### 3. Second Untracked Executor Pool Leaks in `BaseDetector`
**File:** `app/detectors/base/base_detector.py`

```python
executor = ThreadPoolExecutor(max_workers=4)  # module-level, never shut down
```

A separate pool from the verify executor already flagged in v1. It caps all video processing to 4 concurrent slots with no backpressure signalling, and leaks threads on shutdown. Two independently leaking pools now exist.

**Fix:** Manage both via FastAPI's `lifespan` context and expose queue depth metrics.

---

### 4. `find_nearest_gps` Is an O(N) Linear Scan Per Detection
**File:** `app/detectors/base/base_detector.py`

```python
nearest_point = min(gps_points, key=lambda p: abs(p.get("timestamp", 0) - detection_time))
```

`min()` scans every GPS point for each confirmed detection. With 5,000 GPS points and 30 detections → ~150,000 comparisons. GPS data is already time-ordered so this should be a binary search.

**Fix:** Pre-extract timestamps into a list and use `bisect.bisect_left` — O(log N).

---

### 5. `torch.cuda.empty_cache()` Called Inside the Verify Inner Loop
**File:** `app/helpers/sam3_helper.py`

```python
for prompt_text in prompts:     # 1–2 iterations per detection
    ...
    del outputs, inputs
    torch.cuda.empty_cache()    # GPU sync point — expensive in a tight loop
```

`empty_cache()` is a CUDA synchronization barrier. Calling it every prompt iteration adds tens to hundreds of milliseconds per verification call unnecessarily — PyTorch's allocator handles fragmentation on its own.

**Fix:** Remove from the inner loop. Call once at the very end of `verify()` if at all.

---

### 6. `results_log["frames"]` Grows Unboundedly in RAM
**File:** `app/detectors/yolo/detector.py`

```python
results_log = {"frames": []}
...
results_log["frames"].append(frame_data)   # grows for entire video duration
```

All detection-frame dicts accumulate in memory and are only written to disk at the very end. A long video with frequent detections can hold thousands of dicts in RAM before flushing.

**Fix:** Stream-write frames as NDJSON (one JSON line per frame) during processing, or cap to a rolling buffer.

---

## 🟡 Maintainability / Minor

### 7. `verify_cache` and `rejected_tids` Never Pruned Mid-Video
**File:** `app/detectors/yolo/detector.py`

```python
verify_cache = {}
rejected_tids = set()
```

Both grow monotonically with tracker IDs for the entire video. For very long videos (30+ min) tracker IDs can number in the thousands. Stale entries (IDs not seen for 5+ min) are never evicted.

**Fix:** Evict entries when their tracker ID hasn't appeared for more than `TIME_THRESHOLD` seconds.

---

## Summary

| Priority | Issue | File | Impact |
|---|---|---|---|
| 🔴 Critical | Sync file copy blocks async event loop | `upload_service.py` | All requests freeze during upload |
| 🔴 Critical | Global `torch.load` pickle safety bypass | `main.py` | Arbitrary code exec via `.pt` file |
| 🟠 High | Second leaked executor pool | `base_detector.py` | Resource leak, silent concurrency cap |
| 🟠 High | O(N) GPS nearest-point scan | `base_detector.py` | Should be binary search |
| 🟠 High | `empty_cache()` in SAM3 inner loop | `sam3_helper.py` | GPU stall per verification call |
| 🟠 Medium | Unbounded `results_log` RAM growth | `detector.py` | OOM risk on long videos |
| 🟡 Low | `verify_cache`/`rejected_tids` never pruned | `detector.py` | Memory creep on long videos |
