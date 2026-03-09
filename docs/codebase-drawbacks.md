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

## 🟠 Newly Identified Issues

### 13. `tracker_history` List Comprehension Still Runs on Every Detected Object
**File:** `app/detectors/yolo/detector.py` (line 443)

```python
recent = [t for t in tracker_history[tid] if current_time - t <= DETECTION_TIME_WINDOW]
```

Although `tracker_history` is a `deque(maxlen=50)`, this full list comprehension still runs for **every tracked object on every processed frame**. The deque's maxlen prevents unbounded growth, but the scan is still O(50) per detection per frame, not O(1).

**Fix:** Store only the first-seen timestamp and a frame counter per `tid`. Drop the deque entirely for confirmation logic.

---

### 14. `_save_to_db` Opens a New DB Session Every Time
**File:** `app/detectors/yolo/detector.py` (line 842)

```python
def _save_to_db(self, video_id, all_detections, perf_timings=None):
    db = SessionLocal()
```

`_save_to_db` creates a fresh `SessionLocal()` instead of accepting the DB session that is already open in the upload path. This creates a second connection, bypasses any session-level transaction state, and is fragile — if an exception occurs before `db.close()`, the connection leaks.

**Fix:** Pass the existing `db: Session` as a parameter, or use a `with SessionLocal() as db:` context manager.

---

### 15. `location.package.project_id` Causes N+1 Lazy-Load in `_save_to_db`
**File:** `app/detectors/yolo/detector.py` (line 858)

```python
if location.package:
    project_id = location.package.project_id
```

For every detection that has a location match, SQLAlchemy lazy-loads the `Package` row separately (`SELECT ... FROM packages WHERE id = ?`). With 50 detections, that's 50 extra round-trips to the DB.

**Fix:** Use `joinedload(Location.package)` when calling `find_location_by_gps`, or store `project_id` directly on the `Location` model (denormalize).

---

### 16. Summary Routes Load All Detections Into Python Memory
**File:** `app/routes/summary_routes.py` (lines 56, 138, 201)

```python
detections = query.all()   # all rows for a project / package / location
```

`get_project_summary` and `get_package_summary` load every detection row into Python, then build the response dict by looping in Python. For a project with thousands of detections, this is a large memory allocation and a slow response.

**Fix:** Use `GROUP BY` + `COUNT` aggregation at the SQL level for the counts. Only fetch individual detection rows when the client explicitly requests them (e.g., paginated detail endpoint).

---

### 17. `get_location_summary` Makes Two Separate DB Queries for the Same Scope
**File:** `app/routes/summary_routes.py` (lines 195–209)

```python
detections = query.all()                          # query 1 — all rows
detection_counts = db.query(...).group_by(...).all()  # query 2 — same rows, aggregated
```

Two separate SQL queries are made for the same `location_id` scope: one fetches every row, the other counts them by type. The count query is redundant since the same information can be derived from the first result set.

**Fix:** Remove the second query. Compute `by_type` counts from `detections` in Python using `collections.Counter`, or consolidate into a single SQL query with both the rows and aggregates.

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
| 🟠 High | tracker_history list comp per detected object | Wasted CPU every frame |
| 🟠 High | `_save_to_db` opens new DB session | Connection leak risk |
| 🟠 High | N+1 lazy-load of `location.package` in DB save | 50+ hidden queries per video |
| 🟡 Medium | Summary routes: all detections loaded into Python | High memory on large datasets |
| 🟡 Medium | `get_location_summary` double-queries same scope | Redundant DB round-trip |

### Recommended v2 Priority Order

1. **Switch to PostgreSQL** — foundational for all concurrency fixes
2. **Add a task queue (Celery)** — prevents GPU overload
3. **Persist processing status in DB** — survive restarts
4. **Add authentication** — before any public deployment
