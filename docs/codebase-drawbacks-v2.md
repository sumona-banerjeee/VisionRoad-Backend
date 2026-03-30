# VisionRoad Backend — Additional Bottlenecks (v2)

> Findings beyond the original `codebase-drawbacks.md`. Detection pipeline issues and SQLite are excluded — covered there already.
>
> **Branch:** `fix/async-safety-perf-hardening` — all 7 issues resolved and tested (2026-03-05).

---

## 🔴 Critical

### 1. ~~`shutil.copyfileobj` Blocks the Async Event Loop on Upload~~ Fixed

**File:** `app/services/upload_service.py`

Replaced with `_save_upload_async()` — streams in 1 MB chunks via `anyio.open_file`, yielding the event loop between writes. No new dependencies (`anyio` is already a FastAPI transitive dep).

---

### 2. ~~Global `torch.load` Monkeypatch Disables Pickle Safety~~ Fixed

**Files:** `main.py`, `app/detectors/base/base_detector.py`

Global patch removed from `main.py`. Replaced with `_yolo_load_ctx()` context manager in `base_detector.py` that scopes `weights_only=False` only to the `YOLO(model_path)` call, then restores PyTorch's safe default immediately after.

---

## 🟠 Significant Bottlenecks

### 3. ~~Second Untracked Executor Pool Leaks in `BaseDetector`~~ Fixed

**Files:** `app/detectors/base/base_detector.py`, `app/detectors/yolo/detector.py`, `app/__init__.py`

Both executors (`video_proc` and `verify_async`) now expose `get_executor()` / `get_verify_executor()` getters and are shut down via `shutdown(wait=True, cancel_futures=False)` in the FastAPI lifespan teardown. Confirmed working in test run.

---

### 4. ~~`find_nearest_gps` Is an O(N) Linear Scan Per Detection~~ Fixed

**Files:** `app/detectors/base/base_detector.py`, `app/detectors/yolo/detector.py`

Replaced `min()` with `bisect.bisect_left` (O(log N)). `_get_class_list` pre-builds the timestamp list once before its loop and passes it as `_timestamps` on each call — O(M) setup + O(N log M) lookups vs. previous O(N×M).

---

### 5. ~~`torch.cuda.empty_cache()` Called Inside the Verify Inner Loop~~ Fixed

**File:** `app/helpers/sam3_helper.py`

Removed from the per-prompt loop. Now called once at the end of `verify()` only if CUDA is available, with a single log line confirming the single clear.

---

### 6. ~~`results_log["frames"]` Grows Unboundedly in RAM~~ Fixed

**File:** `app/detectors/yolo/detector.py`

Replaced the in-memory list with NDJSON streaming to a temp file in `RESULTS_DIR`. Frames are written line-by-line during processing and read back once at the end via `_read_ndjson_frames()`. Temp file is always cleaned up in the `finally` block. Confirmed working in test run (`[FIX-6] NDJSON temp file cleaned up`).

---

### 7. ~~`verify_cache` and `rejected_tids` Never Pruned Mid-Video~~ Fixed

**File:** `app/detectors/yolo/detector.py`

Added `_tid_last_seen` dict (updated every frame per tracker ID) and `_evict_stale_trackers()` which runs every progress tick. Evicts entries for tracker IDs not seen for more than `TIME_THRESHOLD` seconds that are not confirmed or pending.

---

## Summary

| Priority    | Issue                                       | Status   |
| ----------- | ------------------------------------------- | -------- |
| 🔴 Critical | Sync file copy blocks async event loop      | Resolved |
| 🔴 Critical | Global `torch.load` pickle safety bypass    | Resolved |
| 🟠 High     | Second leaked executor pool                 | Resolved |
| 🟠 High     | O(N) GPS nearest-point scan                 | Resolved |
| 🟠 High     | `empty_cache()` in SAM3 inner loop          | Resolved |
| 🟠 Medium   | Unbounded `results_log` RAM growth          | Resolved |
| 🟡 Low      | `verify_cache`/`rejected_tids` never pruned | Resolved |
