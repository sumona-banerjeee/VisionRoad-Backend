# VisionRoad Backend — Drawbacks & Bottlenecks

> Analysis of the most significant issues to address in future versions.

---

## 🔴 Critical Issues

### 1. SQLite as the Database
**File:** `app/core/config.py`

```python
DATABASE_URL: str = "sqlite:///./visionroad.db"
```

SQLite has a **single-writer lock**. When multiple videos are processed concurrently (each doing a `_save_to_db` at the end), writes will serialize or crash. The `check_same_thread=False` flag is only a band-aid.

**Fix:** Switch to **PostgreSQL** for any real multi-user deployment.

---

### 2. In-Memory State Lost on Restart
**File:** `app/core/config.py`

```python
processing_status: Dict[str, dict] = {}
detection_results: Dict[str, dict] = {}
```

If the server crashes or restarts mid-processing, all in-flight video statuses and cached results are gone. A client polling `/status/{video_id}` gets a 404 even if the data exists in the DB.

**Fix:** Persist state in the database (or Redis for speed).

---

### 3. `_process_video_blocking` is a Monolith
**File:** `app/services/pot_sign_detector.py`

The entire frame loop, tracking logic, VL submission, VL draining, result building, and DB saving are all nested inside one giant `try` block (~600 lines). This makes it nearly impossible to unit test, hard to debug, and prone to subtle state bugs through closures over mutable dicts (`confirmed`, `pending_vl`, `rejected_tids`).

**Fix:** Break into clearly separated, testable methods or pipeline stages.

---

## 🟠 Significant Bottlenecks

### 4. Module-Level Thread Pools Never Shut Down
**File:** `app/services/pot_sign_detector.py`

```python
_vl_timeout_executor = ThreadPoolExecutor(max_workers=4, ...)
_async_vl_executor   = ThreadPoolExecutor(max_workers=MAX_VL_CONCURRENT, ...)
```

These are never explicitly shut down. On process exit they leak threads and can prevent clean shutdown.

**Fix:** Manage via FastAPI's `lifespan` context manager.

---

### 5. `is_duplicate_location` is O(N) Linear Scan
**File:** `app/services/pot_sign_detector.py`

```python
for existing in spatial_locations:  # grows unboundedly
```

`spatial_locations` grows with every confirmed detection and is never pruned within the frame loop. For long videos this becomes slow.

**Fix:** Use a spatial index (e.g. `scipy.spatial.cKDTree`) or prune entries older than `TIME_THRESHOLD`.

---

### 6. `find_location_by_gps` Does a Full Table Scan in Python
**File:** `app/services/location_mapper.py`

```python
locations = query.all()
for location in locations:
    if location.contains_point(lat, lng):
```

Loads every road segment from the DB and filters in Python for every detection. With 1,000 segments and 50 detections, that's 50,000 Python-level comparisons.

**Fix:** Filter in SQL using a bounding-box range query (`WHERE min_lat <= :lat AND :lat <= max_lat AND ...`), or use PostGIS.

---

### 7. No Task Queue — `asyncio.create_task` Is the "Queue"
**File:** `app/services/upload_service.py`

```python
asyncio.create_task(processor.process_video(...))
```

If 10 videos are uploaded at once, 10 GPU-heavy tasks start immediately with no queuing, rate limiting, or priority.

**Fix:** Use **Celery + Redis** (or at minimum an `asyncio.Queue` with a bounded worker pool).

---

### 8. Only One WebSocket Connection Per `video_id`
**File:** `app/ws/websocket_manager.py`

```python
self.active_connections: Dict[str, WebSocket] = {}
```

If two browser tabs watch the same video processing, the second connection silently overwrites the first.

**Fix:** Change to `Dict[str, List[WebSocket]]` to support multiple subscribers per job.

---

## 🟡 Maintainability & Design Issues

### 9. No Authentication / Authorization
**File:** `app/routes/upload_process_routes.py`

Any anonymous user can upload a video, trigger GPU-intensive processing, and read any other video's results by guessing a UUID.

**Fix:** Add API key or JWT-based authentication.

---

### 10. No File Cleanup / Disk Management
**Dirs:** `uploads/`, `results/`

Uploaded videos and JSON result files are never deleted. Over time the disk fills up. There are no size limits on uploads, and `shutil.copyfileobj` blocks the async event loop.

**Fix:** Add a TTL-based cleanup job and use `aiofiles` for async file I/O on upload.

---

### 11. Duplicated Logic Across Three Detector Services
**Files:** `video_processor.py`, `signboard_detector.py`, `pot_sign_detector.py`

GPS loading, progress reporting, `find_nearest_gps`, spatial deduplication, and DB saving are repeated across all three detectors. `BaseDetector` only partially addresses this.

**Fix:** Move shared logic fully into `BaseDetector` using a proper Template Method pattern.

---

### 12. `tracker_history` Filtering Runs on Every Frame per Object
**File:** `app/services/pot_sign_detector.py`

```python
recent = [t for t in tracker_history[tid] if current_time - t <= DETECTION_TIME_WINDOW]
```

This list comprehension runs for every tracked object on every processed frame. Minor but cumulative.

**Fix:** Replace with a simple first-seen timestamp + frame counter per tracker ID.

---

## Summary

| Priority | Issue | Impact |
|---|---|---|
| 🔴 Critical | SQLite for production | Data corruption under concurrency |
| 🔴 Critical | In-memory state | Data loss on restart |
| 🔴 Critical | Monolithic `_process_video_blocking` | Untestable, fragile |
| 🟠 High | No task queue | GPU OOM on concurrent uploads |
| 🟠 High | Linear spatial dedup scan | Slow on long videos |
| 🟠 High | Full table scan in GPS mapper | Per-detection DB overhead |
| 🟠 High | Thread pools never shut down | Resource leak |
| 🟠 High | Single WebSocket per video | Dropped connections |
| 🟡 Medium | No auth | Open API |
| 🟡 Medium | No file cleanup | Disk exhaustion |
| 🟡 Medium | Duplicated detector logic | Maintenance burden |
| 🟡 Medium | Per-frame tracker history scan | Minor CPU overhead |

### Recommended v2 Priority Order

1. **Switch to PostgreSQL** — foundational for all concurrency fixes
2. **Add a task queue (Celery)** — prevents GPU overload
3. **Persist processing status in DB** — survive restarts
4. **Add authentication** — before any public deployment
